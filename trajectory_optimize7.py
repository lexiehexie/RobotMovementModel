#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_optimize6_dijkstra.py

Dijkstra baseline for joint-grid planning of TCP motion.

Why try this:
- Removes heuristic influence entirely.
- Expands nodes in order of true accumulated path cost g.
- Useful as a baseline to understand whether 'bad behavior' comes from the
  heuristic or from the cost function itself.

Important caveat:
- Dijkstra can be MUCH slower and more memory-hungry than A*.
- If the cost function itself prefers a wide arc, Dijkstra will still choose it.
  It only removes heuristic bias; it does not change the objective.

Memory/Progress:
- Prints progress every N expansions.
- Tracks RSS memory (MB) if psutil is available, otherwise prints 'n/a'.
- Stops if max_expansions or max_memory_mb is exceeded, then returns the best
  partial path found so far (closest to Cartesian goal).

Exports:
- optimal_path.jsonl and optimal_path.csv compatible with main_with_playback.py
- dashboard-compatible result dict (P, P_target, speed)

This version uses 4 joints by default: (1,2,3,4).
"""

from __future__ import annotations

import heapq
import time
import json
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import fkmodel

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


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


def point_segment_distance_and_param(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a)), 0.0, a
    t = float(np.dot(p - a, ab) / denom)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)), t, proj


def fk_end(arm: fkmodel.Manipulator6DOF, q_deg: np.ndarray, joints: Tuple[int, ...]) -> np.ndarray:
    for k, j in enumerate(joints):
        arm.set_servo_deg(j, float(q_deg[k]))
    fk = arm.forward_kinematics()
    x, y, z = fk["end_effector_mid"]
    return np.array([x, y, z], dtype=float)


def snap_to_grid(q_deg: np.ndarray, step_deg: float) -> np.ndarray:
    return np.round(np.asarray(q_deg, dtype=float) / float(step_deg)) * float(step_deg)


def state_key(q_deg: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(round(float(x) * 100)) for x in q_deg)


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


@dataclass
class DijkstraConfig:
    joints: Tuple[int, ...] = (1, 2, 3, 4)
    step_deg: float = 1.0
    goal_pos_eps: float = 10.0
    line_cost_weight: float = 0.001
    line_soft_radius: float = 0.0
    cart_step_cost_weight: float = 1.0
    total_time: float = 3.0
    step_ms: int = 20
    max_expansions: int = 500_000
    max_memory_mb: int = 1800
    cache_fk: bool = True
    progress_every: int = 25_000


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def dijkstra_plan(
    arm: fkmodel.Manipulator6DOF,
    q_start: Dict[int, float],
    q_goal: Dict[int, float],
    cfg: DijkstraConfig,
) -> Dict[str, Any]:
    joints = cfg.joints
    D = len(joints)
    specs = arm.servo_specs()

    q0 = np.array([float(q_start[j]) for j in joints], dtype=float)
    qg = np.array([float(q_goal[j]) for j in joints], dtype=float)

    q0 = snap_to_grid(q0, cfg.step_deg)
    qg = snap_to_grid(qg, cfg.step_deg)

    fk_cache: Dict[Tuple[int, ...], np.ndarray] = {}

    def fk_cached(q: np.ndarray) -> np.ndarray:
        k = state_key(q)
        p = fk_cache.get(k)
        if p is None:
            p = fk_end(arm, q, joints)
            if cfg.cache_fk:
                fk_cache[k] = p
        return p

    def valid(q: np.ndarray) -> bool:
        for k, j in enumerate(joints):
            if q[k] < specs[j].min_deg - 1e-9 or q[k] > specs[j].max_deg + 1e-9:
                return False
        return True

    if not valid(q0):
        raise RuntimeError("Start state violates joint limits.")
    if not valid(qg):
        raise RuntimeError("Goal state violates joint limits.")

    p0 = fk_cached(q0)
    p1 = fk_cached(qg)

    def estimate_dp_per_step(q_ref: np.ndarray) -> float:
        p_ref = fk_cached(q_ref)
        best = 0.0
        for d in range(D):
            dq = np.zeros(D, dtype=float)
            dq[d] = cfg.step_deg
            p_plus = fk_cached(snap_to_grid(q_ref + dq, cfg.step_deg))
            p_minus = fk_cached(snap_to_grid(q_ref - dq, cfg.step_deg))
            best = max(best, float(np.linalg.norm(p_plus - p_ref)), float(np.linalg.norm(p_minus - p_ref)))
        return best

    dp0 = estimate_dp_per_step(q0)
    dpg = estimate_dp_per_step(qg)
    dp_per_step = max(1e-6, dp0, dpg)
    print(f"[Dijkstra] dp_per_step≈{dp_per_step:.3f} (Cartesian units per one ±step_deg move)")

    start_key = state_key(q0)
    key_to_q: Dict[Tuple[int, ...], np.ndarray] = {start_key: q0}
    open_heap: List[Tuple[float, Tuple[int, ...]]] = []
    heapq.heappush(open_heap, (0.0, start_key))

    g_score: Dict[Tuple[int, ...], float] = {start_key: 0.0}
    came_from: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    best_goal_key = start_key
    best_goal_dist = float(np.linalg.norm(p1 - p0))

    expansions = 0
    t_start = time.time()

    while open_heap:
        gcur, kcur = heapq.heappop(open_heap)
        if gcur != g_score.get(kcur, None):
            continue

        qcur = key_to_q[kcur]
        pcur = fk_cached(qcur)

        dist_goal = float(np.linalg.norm(p1 - pcur))
        if dist_goal < best_goal_dist:
            best_goal_dist = dist_goal
            best_goal_key = kcur

        if dist_goal <= cfg.goal_pos_eps:
            best_goal_key = kcur
            best_goal_dist = dist_goal
            break

        expansions += 1

        if expansions >= cfg.max_expansions:
            print(f"[Dijkstra] max_expansions reached; using best_goal_dist={best_goal_dist:.3f}")
            break

        rss_mb = _rss_mb()
        if rss_mb == rss_mb and rss_mb >= float(cfg.max_memory_mb):
            print(f"[Dijkstra] max_memory_mb reached ({rss_mb:.1f} MB); using best_goal_dist={best_goal_dist:.3f}")
            break

        if cfg.progress_every and expansions % cfg.progress_every == 0:
            elapsed = time.time() - t_start
            rate = expansions / elapsed if elapsed > 1e-9 else float('inf')
            open_size = len(open_heap)
            rss_txt = f"{rss_mb:.1f} MB" if rss_mb == rss_mb else "n/a"
            print(f"[Dijkstra] expanded={expansions:,} open={open_size:,} touched={len(key_to_q):,} "
                  f"fk_cache={len(fk_cache):,} best_goal_dist={best_goal_dist:.2f} "
                  f"g_current={gcur:.2f} rss={rss_txt} rate={rate:,.0f}/s elapsed={elapsed:,.1f}s")

        base_g = g_score[kcur]
        for d in range(D):
            for sgn in (-1.0, +1.0):
                qn = qcur.copy()
                qn[d] += sgn * cfg.step_deg
                qn = snap_to_grid(qn, cfg.step_deg)
                if not valid(qn):
                    continue

                kn = state_key(qn)
                if kn not in key_to_q:
                    key_to_q[kn] = qn

                pn = fk_cached(qn)
                d_line, _, _ = point_segment_distance_and_param(pn, p0, p1)
                excess = max(0.0, d_line - cfg.line_soft_radius)

                dp = float(np.linalg.norm(pn - pcur))
                dp_steps = dp / dp_per_step

                tentative_g = (
                    base_g
                    + 1.0
                    + cfg.line_cost_weight * (excess * excess)
                    + cfg.cart_step_cost_weight * dp_steps
                )

                old = g_score.get(kn)
                if old is None or tentative_g < old:
                    came_from[kn] = kcur
                    g_score[kn] = tentative_g
                    heapq.heappush(open_heap, (tentative_g, kn))

    path_keys = [best_goal_key]
    while path_keys[-1] != start_key:
        cur = path_keys[-1]
        parent = came_from.get(cur)
        if parent is None:
            print(f"[Dijkstra] Warning: broken parent chain at {cur}; returning partial path.")
            break
        path_keys.append(parent)
    path_keys.reverse()

    Q = np.array([key_to_q[k] for k in path_keys], dtype=float)

    P = np.zeros((len(Q), 3), dtype=float)
    P_target = np.zeros((len(Q), 3), dtype=float)
    for i in range(len(Q)):
        P[i] = fk_cached(Q[i])
        _, _, proj = point_segment_distance_and_param(P[i], p0, p1)
        P_target[i] = proj

    dt = cfg.total_time / max(1, (len(Q) - 1))
    speed = np.linalg.norm(P[1:] - P[:-1], axis=1) / dt if len(P) > 1 else np.array([], dtype=float)

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
        "goal_pos_eps": cfg.goal_pos_eps,
        "best_goal_dist": best_goal_dist,
        "goal_reached": best_goal_dist <= cfg.goal_pos_eps + 1e-12,
        "step_deg": cfg.step_deg,
        "dp_per_step": dp_per_step,
    }


if __name__ == "__main__":
    arm = create_default_arm()

    q_start = {1: 0.0, 2: 70.0, 3: 0.0, 4: 0.0, 5: 0.0}
    q_end   = {1: 0.0, 2: -70.0, 3: 0.0, 4: 0.0, 5: 0.0}

    cfg = DijkstraConfig(
        joints=(1, 2, 3, 4),
        step_deg=3.0,
        goal_pos_eps=10.0,
        line_cost_weight=0.001,
        line_soft_radius=0.0,
        cart_step_cost_weight=1.0,
        total_time=3.0,
        step_ms=20,
        max_expansions=300_000,
        max_memory_mb=1800,
        cache_fk=True,
        progress_every=25_000,
    )

    res = dijkstra_plan(arm, q_start, q_end, cfg)

    print("Dijkstra expansions:", res["expansions"])
    print("States touched:", res["states_touched"])
    print("FK cached:", res["fk_cached"])
    print("best_goal_dist:", float(res["best_goal_dist"]))
    print("Mean tracking error:", float(np.mean(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Max tracking error:", float(np.max(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Speed std:", float(np.std(res["speed"])) if len(res["speed"]) else float("nan"))

    jsonl_path, csv_path = save_for_main_with_playback(
        arm=arm,
        Q_deg_waypoints=res["Q_deg"],
        joints=tuple(res["joints"]),
        total_time_s=cfg.total_time,
        out_base="optimal_path",
        step_ms=cfg.step_ms,
    )
    print("Saved for playback:", jsonl_path, csv_path)

    try:
        from plot_utils import plot_dashboard
        plot_dashboard(res, title=f"Dijkstra (step={cfg.step_deg}°)")
    except Exception as e:
        print("Plot skipped:", e)
