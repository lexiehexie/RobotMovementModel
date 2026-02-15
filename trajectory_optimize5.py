"""
trajectory_optimize_astar_playback.py

Lazy A* planner in discretized JOINT space with Cartesian tube pruning.

Idea:
- State: 5 joint angles (degrees) discretized by STEP_DEG (default 4.0)
- Neighbors: +/- STEP_DEG on one joint (10 neighbors for 5 joints)
- Valid state: inside servo limits AND FK(state) is within DIST_THRESHOLD of the
  Cartesian straight line segment between FK(q_start) and FK(q_end).
- Cost: 1 per move (or weighted by joint delta)
- Heuristic: joint-space Manhattan distance to q_end / STEP_DEG (admissible)

Output:
- Writes optimal_path.jsonl and optimal_path.csv compatible with main_with_playback.py
- Returns a result dict with P, P_target, speed so it works with plot_utils.plot_dashboard

Notes:
- This is a first baseline. With a very small threshold like 0.1, the pruned tube
  will likely be disconnected at 4° resolution; if no path is found, increase threshold
  or decrease STEP_DEG.

Dependencies: numpy
"""

from __future__ import annotations

import heapq
import time
import json
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import fkmodel


# -----------------------------
# Robot model
# -----------------------------

def create_default_arm() -> fkmodel.Manipulator6DOF:
    lengths = fkmodel.LinkLengths(
        base_height=80.0,
        l2=105.0,
        l3=100.0,
        l4=60.0,
        tool=40.0,
        finger=55.0,
    )
    servos = {
        1: fkmodel.ServoSpec(min_deg=-90, max_deg=+90, steps=181),
        2: fkmodel.ServoSpec(min_deg=-90, max_deg=+90, steps=181),
        3: fkmodel.ServoSpec(min_deg=-80, max_deg=+80, steps=161),
        4: fkmodel.ServoSpec(min_deg=-90, max_deg=+90, steps=181),
        5: fkmodel.ServoSpec(min_deg=-90, max_deg=+90, steps=181),
        6: fkmodel.ServoSpec(min_deg=0, max_deg=90, steps=91),
    }
    return fkmodel.Manipulator6DOF(
        lengths=lengths,
        servos=servos,
        claw_max_opening=70.0,
        claw_min_opening=5.0,
    )


# -----------------------------
# Geometry helpers
# -----------------------------

def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point p to segment a-b."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def fk_end(arm: fkmodel.Manipulator6DOF, q_deg: np.ndarray, joints: Tuple[int, ...]) -> np.ndarray:
    for k, j in enumerate(joints):
        arm.set_servo_deg(j, float(q_deg[k]))
    fk = arm.forward_kinematics()
    x, y, z = fk["end_effector_mid"]
    return np.array([x, y, z], dtype=float)


# -----------------------------
# Discretization
# -----------------------------

def snap_to_grid(q_deg: np.ndarray, step_deg: float) -> np.ndarray:
    step_deg = float(step_deg)
    return np.round(np.asarray(q_deg, dtype=float) / step_deg) * step_deg


def state_key(q_deg: np.ndarray) -> Tuple[int, ...]:
    # store as integer "ticks" of 0.01 degree to avoid float key issues
    return tuple(int(round(float(x) * 100)) for x in q_deg)


# -----------------------------
# Playback export
# -----------------------------

def _deg_to_us_linear(spec: fkmodel.ServoSpec, deg: float, us_min: float = 500.0, us_max: float = 2500.0) -> int:
    deg = spec.clamp_deg(float(deg))
    if abs(spec.max_deg - spec.min_deg) < 1e-12:
        return int(round((us_min + us_max) / 2.0))
    alpha = (deg - spec.min_deg) / (spec.max_deg - spec.min_deg)
    us = us_min + alpha * (us_max - us_min)
    return int(round(max(us_min, min(us_max, us))))


def resample_joint_path_uniform_ms(Q: np.ndarray, total_time_s: float, step_ms: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.asarray(Q, dtype=float)
    N, D = Q.shape
    total_time_s = float(total_time_s)
    step_ms = int(step_ms)

    if N < 2:
        return np.array([0], dtype=int), Q[:1].copy()

    t_wp = np.linspace(0.0, total_time_s, N)
    times_ms = np.arange(0, int(round(total_time_s * 1000.0)) + 1, step_ms, dtype=int)
    t = times_ms.astype(float) / 1000.0

    Q_ms = np.zeros((len(t), D), dtype=float)
    for k in range(D):
        Q_ms[:, k] = np.interp(t, t_wp, Q[:, k])
    return times_ms, Q_ms


def save_for_main_with_playback(
    arm: fkmodel.Manipulator6DOF,
    Q_deg_waypoints: np.ndarray,
    joints: Tuple[int, ...],
    total_time_s: float,
    out_base: str = "optimal_path",
    step_ms: int = 20,
) -> Tuple[str, str]:
    times_ms, Q_ms = resample_joint_path_uniform_ms(Q_deg_waypoints, total_time_s, step_ms=step_ms)
    specs = arm.servo_specs()

    jsonl_path = f"{out_base}.jsonl"
    csv_path = f"{out_base}.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, t_ms in enumerate(times_ms):
            rec = {"time_ms": int(t_ms)}
            rec["joints_us"] = {str(j): _deg_to_us_linear(specs[j], float(Q_ms[i, joints.index(j)])) for j in joints}
            rec["joints_deg"] = {str(j): float(Q_ms[i, joints.index(j)]) for j in joints}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    header = ["time_ms"] + [f"j{j}_us" for j in range(1, 7)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t_ms in enumerate(times_ms):
            row = [int(t_ms)]
            for j in range(1, 7):
                if j in joints:
                    us = _deg_to_us_linear(specs[j], float(Q_ms[i, joints.index(j)]))
                else:
                    us = _deg_to_us_linear(specs[j], float(arm.get_servo_deg(j)))
                row.append(us)
            w.writerow(row)

    return jsonl_path, csv_path


# -----------------------------
# A* planner
# -----------------------------

@dataclass
class AStarConfig:
    joints: Tuple[int, ...] = (1, 2, 3, 4, 5)
    step_deg: float = 4.0
    dist_threshold: float = 10.0  # tube radius in same units as fkmodel (likely mm)

    total_time: float = 3.0
    step_ms: int = 20

    max_expansions: int = 2_000_000
    cache_fk: bool = True

    progress_every: int = 25_000  # print every N expansions


def astar_plan(
    arm: fkmodel.Manipulator6DOF,
    q_start: Dict[int, float],
    q_goal: Dict[int, float],
    cfg: AStarConfig,
) -> Dict[str, Any]:
    joints = cfg.joints
    D = len(joints)
    specs = arm.servo_specs()

    q0 = np.array([float(q_start[j]) for j in joints], dtype=float)
    qg = np.array([float(q_goal[j]) for j in joints], dtype=float)

    q0 = snap_to_grid(q0, cfg.step_deg)
    qg = snap_to_grid(qg, cfg.step_deg)

    # Cartesian line endpoints
    p0 = fk_end(arm, q0, joints)
    p1 = fk_end(arm, qg, joints)

    # admissible heuristic: L1 distance in steps (each move changes one joint by step_deg)
    def h(q: np.ndarray) -> float:
        return float(np.sum(np.abs(q - qg)) / cfg.step_deg)

    # validity test (limits + tube)
    fk_cache: Dict[Tuple[int, ...], np.ndarray] = {}

    def valid(q: np.ndarray) -> bool:
        for k, j in enumerate(joints):
            if q[k] < specs[j].min_deg - 1e-9 or q[k] > specs[j].max_deg + 1e-9:
                return False
        key = state_key(q)
        if cfg.cache_fk and key in fk_cache:
            p = fk_cache[key]
        else:
            p = fk_end(arm, q, joints)
            if cfg.cache_fk:
                fk_cache[key] = p
        return point_segment_distance(p, p0, p1) <= cfg.dist_threshold + 1e-12

    # early reject
    if not valid(q0):
        raise RuntimeError("Start state is outside the tube (increase threshold or adjust start).")
    if not valid(qg):
        raise RuntimeError("Goal state is outside the tube (increase threshold or adjust goal).")

    start_key = state_key(q0)
    goal_key = state_key(qg)

    open_heap: List[Tuple[float, float, Tuple[int, ...]]] = []
    heapq.heappush(open_heap, (h(q0), 0.0, start_key))

    came_from: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    g_score: Dict[Tuple[int, ...], float] = {start_key: 0.0}

    # store actual q arrays for keys we touch
    key_to_q: Dict[Tuple[int, ...], np.ndarray] = {start_key: q0}

    expansions = 0
    t_start = time.time()

    while open_heap:
        fcur, gcur, kcur = heapq.heappop(open_heap)

        if kcur == goal_key:
            break

        expansions += 1
        if expansions > cfg.max_expansions:
            raise RuntimeError(f"A* exceeded max_expansions={cfg.max_expansions}. Try larger threshold or smaller step_deg.")

        if cfg.progress_every and (expansions % cfg.progress_every == 0):
            # Rough progress info (A* has no true progress percentage)
            elapsed = time.time() - t_start
            rate = expansions / elapsed if elapsed > 1e-9 else float('inf')
            open_size = len(open_heap)
            best_f = open_heap[0][0] if open_heap else float('nan')
            rem = h(qcur)  # joint-space lower bound (steps)
            print(f"[A*] expanded={expansions:,} open={open_size:,} touched={len(key_to_q):,} "
                  f"fk_cache={len(fk_cache):,} best_f={best_f:.2f} rem_steps~{rem:.1f} "
                  f"rate={rate:,.0f}/s elapsed={elapsed:,.1f}s")

        qcur = key_to_q[kcur]

        # neighbors: +/- step on one joint
        for d in range(D):
            for sgn in (-1.0, +1.0):
                qn = qcur.copy()
                qn[d] += sgn * cfg.step_deg
                qn = snap_to_grid(qn, cfg.step_deg)
                kn = state_key(qn)

                if kn not in key_to_q:
                    key_to_q[kn] = qn

                if not valid(qn):
                    continue

                tentative = g_score[kcur] + 1.0  # uniform cost per move
                old = g_score.get(kn)
                if old is None or tentative < old:
                    came_from[kn] = kcur
                    g_score[kn] = tentative
                    heapq.heappush(open_heap, (tentative + h(qn), tentative, kn))

    if goal_key not in g_score:
        raise RuntimeError("No path found within the tube. Increase dist_threshold or reduce step_deg.")

    # reconstruct path in joint space
    path_keys = [goal_key]
    while path_keys[-1] != start_key:
        path_keys.append(came_from[path_keys[-1]])
    path_keys.reverse()

    Q = np.array([key_to_q[k] for k in path_keys], dtype=float)

    # Cartesian path
    P = np.zeros((len(Q), 3), dtype=float)
    for i in range(len(Q)):
        key = state_key(Q[i])
        if cfg.cache_fk and key in fk_cache:
            P[i] = fk_cache[key]
        else:
            P[i] = fk_end(arm, Q[i], joints)

    # Target points for plotting: sample along line with same number of points
    T = np.linspace(0.0, 1.0, len(Q))
    P_target = (1 - T)[:, None] * p0[None, :] + T[:, None] * p1[None, :]

    dt = cfg.total_time / max(1, (len(Q) - 1))
    dP = P[1:] - P[:-1]
    speed = np.linalg.norm(dP, axis=1) / dt

    return {
        "Q_deg": Q,
        "P": P,
        "P_target": P_target,
        "speed": speed,
        "dt": dt,
        "joints": joints,
        "expansions": expansions,
        "states_touched": len(key_to_q),
        "fk_cached": len(fk_cache),
        "tube_threshold": cfg.dist_threshold,
        "step_deg": cfg.step_deg,
    }


# -----------------------------
# Demo
# -----------------------------

if __name__ == "__main__":
    arm = create_default_arm()

    # Example
    q_start = {1: 0.0, 2: 70.0, 3: 0.0, 4: 0.0, 5: 0.0}
    q_end   = {1: 0.0, 2: -70.0, 3: 0.0, 4: 0.0, 5: 0.0}

    cfg = AStarConfig(
        joints=(1, 2, 3, 4, 5),
        step_deg=8.0,
        dist_threshold=100.0,   # sensible starting tube radius; adjust as needed
        total_time=3.0,
        step_ms=20,
        max_expansions=2_000_000,
        cache_fk=True,
    )

    res = astar_plan(arm, q_start, q_end, cfg)

    print("A* expansions:", res["expansions"])
    print("States touched:", res["states_touched"])
    print("FK cached:", res["fk_cached"])
    print("Mean tracking error:", float(np.mean(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Max tracking error:", float(np.max(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Speed std:", float(np.std(res["speed"])))

    jsonl_path, csv_path = save_for_main_with_playback(
        arm=arm,
        Q_deg_waypoints=res["Q_deg"],
        joints=tuple(res["joints"]),
        total_time_s=cfg.total_time,
        out_base="optimal_path",
        step_ms=cfg.step_ms,
    )
    print("Saved for playback:", jsonl_path, csv_path)

    # Optional plotting (shared utility)
    try:
        from plot_utils import plot_dashboard
        plot_dashboard(res, title=f"A* (step={cfg.step_deg}°, tube={cfg.dist_threshold})")
    except Exception as e:
        print("Plot skipped:", e)
