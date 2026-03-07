#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import heapq
import json
import csv
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Set

import numpy as np
import fkmodel

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


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
class BidirectionalDijkstraConfig:
    joints: Tuple[int, ...] = (1, 2, 3, 4)
    step_deg: float = 3.0
    total_time: float = 3.0
    step_ms: int = 20
    max_expansions: int = 1_000_000
    max_memory_mb: int = 4000
    cache_fk: bool = True
    progress_every: int = 25_000
    frontier_gap_sample_limit: int = 300   # approximate d_fb using sampled recent states from each frontier
    meet_pos_eps: float = 15.0


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _reconstruct_forward(parent_f: Dict[Tuple[int, ...], Tuple[int, ...]], start_key: Tuple[int, ...], target_key: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    path = [target_key]
    while path[-1] != start_key:
        par = parent_f.get(path[-1])
        if par is None:
            break
        path.append(par)
    path.reverse()
    return path


def _reconstruct_backward(parent_b: Dict[Tuple[int, ...], Tuple[int, ...]], goal_key: Tuple[int, ...], start_from: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    path = [start_from]
    cur = start_from
    while cur != goal_key:
        nxt = parent_b.get(cur)
        if nxt is None:
            break
        path.append(nxt)
        cur = nxt
    return path


def _compute_frontier_gap(
    sample_f_keys: List[Tuple[int, ...]],
    sample_b_keys: List[Tuple[int, ...]],
    fk_cache: Dict[Tuple[int, ...], np.ndarray],
) -> Tuple[float, Tuple[int, ...] | None, Tuple[int, ...] | None]:
    if not sample_f_keys or not sample_b_keys:
        return float("inf"), None, None

    # Build compact point arrays only for keys actually present in cache
    f_keys = []
    f_pts = []
    for k in sample_f_keys:
        p = fk_cache.get(k)
        if p is not None:
            f_keys.append(k)
            f_pts.append(p)

    b_keys = []
    b_pts = []
    for k in sample_b_keys:
        p = fk_cache.get(k)
        if p is not None:
            b_keys.append(k)
            b_pts.append(p)

    if not f_pts or not b_pts:
        return float("inf"), None, None

    F = np.asarray(f_pts, dtype=float)
    B = np.asarray(b_pts, dtype=float)

    # Fast path with SciPy KD-tree
    if cKDTree is not None:
        tree = cKDTree(B)
        dist, idx = tree.query(F, k=1)
        j = int(np.argmin(dist))
        best = float(dist[j])
        bj = int(idx[j])
        return best, f_keys[j], b_keys[bj]

    # Fallback: sampled O(n*m) pairwise scan
    best = float("inf")
    best_fk = None
    best_bk = None
    for i, pf in enumerate(F):
        for j, pb in enumerate(B):
            d = float(np.linalg.norm(pf - pb))
            if d < best:
                best = d
                best_fk = f_keys[i]
                best_bk = b_keys[j]
    return best, best_fk, best_bk


def bidirectional_dijkstra_cartesian_shortest(
    arm: fkmodel.Manipulator6DOF,
    q_start: Dict[int, float],
    q_goal: Dict[int, float],
    cfg: BidirectionalDijkstraConfig,
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

    def iter_neighbors(qcur: np.ndarray):
        pcur = fk_cached(qcur)
        for d in range(D):
            for sgn in (-1.0, +1.0):
                qn = qcur.copy()
                qn[d] += sgn * cfg.step_deg
                qn = snap_to_grid(qn, cfg.step_deg)
                if not valid(qn):
                    continue
                pn = fk_cached(qn)
                yield qn, pn, float(np.linalg.norm(pn - pcur))

    if not valid(q0):
        raise RuntimeError("Start state violates joint limits.")
    if not valid(qg):
        raise RuntimeError("Goal state violates joint limits.")

    p0 = fk_cached(q0)
    p1 = fk_cached(qg)

    k0 = state_key(q0)
    kg = state_key(qg)

    key_to_q: Dict[Tuple[int, ...], np.ndarray] = {k0: q0, kg: qg}

    open_f: List[Tuple[float, Tuple[int, ...]]] = [(0.0, k0)]
    open_b: List[Tuple[float, Tuple[int, ...]]] = [(0.0, kg)]
    g_f: Dict[Tuple[int, ...], float] = {k0: 0.0}
    g_b: Dict[Tuple[int, ...], float] = {kg: 0.0}
    parent_f: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    parent_b: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    closed_f: Set[Tuple[int, ...]] = set()
    closed_b: Set[Tuple[int, ...]] = set()

    best_meet_key = None
    best_total_cost = float("inf")
    eps_meet_forward_key = None
    eps_meet_backward_key = None

    best_forward_goal_key = k0
    best_forward_goal_dist = float(np.linalg.norm(p1 - p0))

    best_backward_start_key = kg
    best_backward_start_dist = float(np.linalg.norm(p0 - p1))

    recent_forward_keys: List[Tuple[int, ...]] = [k0]
    recent_backward_keys: List[Tuple[int, ...]] = [kg]
    best_frontier_gap = float(np.linalg.norm(p1 - p0))
    print(f"[BiDijkstra] start step={cfg.step_deg} progress_every={cfg.progress_every} frontier_gap_every=5000 sample_limit={cfg.frontier_gap_sample_limit} gap_method={'cKDTree' if cKDTree is not None else 'pairwise'} meet_pos_eps={cfg.meet_pos_eps}")
    best_frontier_gap_fkey = k0
    best_frontier_gap_bkey = kg

    expansions = 0
    t0 = time.time()

    def heap_min_valid(heap, gmap):
        while heap and heap[0][0] != gmap.get(heap[0][1], None):
            heapq.heappop(heap)
        return heap[0][0] if heap else float("inf")

    while open_f and open_b:
        min_f = heap_min_valid(open_f, g_f)
        min_b = heap_min_valid(open_b, g_b)

        if best_meet_key is not None and (min_f + min_b) >= best_total_cost:
            print(f"[BiDijkstra] stop criterion met: {min_f:.3f} + {min_b:.3f} >= {best_total_cost:.3f}")
            break

        forward = min_f <= min_b

        if forward:
            gcur, kcur = heapq.heappop(open_f)
            if gcur != g_f.get(kcur, None) or kcur in closed_f:
                continue
            closed_f.add(kcur)

            qcur = key_to_q[kcur]
            pcur = fk_cached(qcur)

            dist_goal = float(np.linalg.norm(p1 - pcur))
            if dist_goal < best_forward_goal_dist:
                best_forward_goal_dist = dist_goal
                best_forward_goal_key = kcur

            recent_forward_keys.append(kcur)
            if len(recent_forward_keys) > cfg.frontier_gap_sample_limit * 4:
                recent_forward_keys = recent_forward_keys[-cfg.frontier_gap_sample_limit * 4:]

            if kcur in g_b:
                total = gcur + g_b[kcur]
                if total < best_total_cost:
                    best_total_cost = total
                    best_meet_key = kcur

            for qn, _, edge_cost in iter_neighbors(qcur):
                kn = state_key(qn)
                if kn not in key_to_q:
                    key_to_q[kn] = qn
                tentative = gcur + edge_cost
                if tentative < g_f.get(kn, float("inf")):
                    g_f[kn] = tentative
                    parent_f[kn] = kcur
                    heapq.heappush(open_f, (tentative, kn))
                    if kn in g_b:
                        total = tentative + g_b[kn]
                        if total < best_total_cost:
                            best_total_cost = total
                            best_meet_key = kn
        else:
            gcur, kcur = heapq.heappop(open_b)
            if gcur != g_b.get(kcur, None) or kcur in closed_b:
                continue
            closed_b.add(kcur)

            qcur = key_to_q[kcur]
            pcur = fk_cached(qcur)

            dist_start = float(np.linalg.norm(p0 - pcur))
            if dist_start < best_backward_start_dist:
                best_backward_start_dist = dist_start
                best_backward_start_key = kcur

            recent_backward_keys.append(kcur)
            if len(recent_backward_keys) > cfg.frontier_gap_sample_limit * 4:
                recent_backward_keys = recent_backward_keys[-cfg.frontier_gap_sample_limit * 4:]

            if kcur in g_f:
                total = gcur + g_f[kcur]
                if total < best_total_cost:
                    best_total_cost = total
                    best_meet_key = kcur

            for qn, _, edge_cost in iter_neighbors(qcur):
                kn = state_key(qn)
                if kn not in key_to_q:
                    key_to_q[kn] = qn
                tentative = gcur + edge_cost
                if tentative < g_b.get(kn, float("inf")):
                    g_b[kn] = tentative
                    parent_b[kn] = kcur
                    heapq.heappush(open_b, (tentative, kn))
                    if kn in g_f:
                        total = tentative + g_f[kn]
                        if total < best_total_cost:
                            best_total_cost = total
                            best_meet_key = kn

        if expansions % 5000 == 0:
            sf = recent_forward_keys if len(recent_forward_keys) <= cfg.frontier_gap_sample_limit else recent_forward_keys[-cfg.frontier_gap_sample_limit:]
            sb = recent_backward_keys if len(recent_backward_keys) <= cfg.frontier_gap_sample_limit else recent_backward_keys[-cfg.frontier_gap_sample_limit:]
            gap, gfk, gbk = _compute_frontier_gap(sf, sb, fk_cache)
            if gap < best_frontier_gap:
                best_frontier_gap = gap
                best_frontier_gap_fkey = gfk if gfk is not None else best_frontier_gap_fkey
                best_frontier_gap_bkey = gbk if gbk is not None else best_frontier_gap_bkey

        expansions += 1
        if expansions >= cfg.max_expansions:
            print(f"[BiDijkstra] max_expansions reached; d_f={best_forward_goal_dist:.3f}, d_b={best_backward_start_dist:.3f}, d_fb={best_frontier_gap:.3f}")
            break

        rss = _rss_mb()
        if rss == rss and rss >= float(cfg.max_memory_mb):
            print(f"[BiDijkstra] max_memory_mb reached ({rss:.1f} MB); d_f={best_forward_goal_dist:.3f}, d_b={best_backward_start_dist:.3f}, d_fb={best_frontier_gap:.3f}")
            break

        if cfg.progress_every and expansions % cfg.progress_every == 0:
            elapsed = time.time() - t0
            rate = expansions / elapsed if elapsed > 1e-9 else float("inf")
            rss_txt = f"{rss:.1f} MB" if rss == rss else "n/a"
            btc = best_total_cost if best_meet_key is not None else float("nan")
            print(
                f"[BiDijkstra] expanded={expansions:,} "
                f"open_f={len(open_f):,} open_b={len(open_b):,} "
                f"seen_f={len(g_f):,} seen_b={len(g_b):,} fk_cache={len(fk_cache):,} "
                f"d_f={best_forward_goal_dist:.2f} d_b={best_backward_start_dist:.2f} d_fb={best_frontier_gap:.2f} "
                f"best_total_cost={btc:.2f} rss={rss_txt} rate={rate:,.0f}/s elapsed={elapsed:,.1f}s"
            )

    if best_meet_key is not None:
        if best_meet_key == ("EPSILON",):
            left = _reconstruct_forward(parent_f, k0, eps_meet_forward_key)
            right = _reconstruct_backward(parent_b, kg, eps_meet_backward_key)
            path_keys = left + right
        else:
            left = _reconstruct_forward(parent_f, k0, best_meet_key)
            right = _reconstruct_backward(parent_b, kg, best_meet_key)
            path_keys = left + right[1:]
        P_forward_fail = None
        P_backward_fail = None
        Q_forward_fail = None
        Q_backward_fail = None
    else:
        # Return best forward partial path. Also expose both forward/backward partial paths for plotting/diagnostics.
        path_keys = _reconstruct_forward(parent_f, k0, best_forward_goal_key)

        forward_fail_keys = _reconstruct_forward(parent_f, k0, best_frontier_gap_fkey)
        backward_fail_keys = _reconstruct_backward(parent_b, kg, best_frontier_gap_bkey)

        Q_forward_fail = np.array([key_to_q[k] for k in forward_fail_keys], dtype=float) if forward_fail_keys else None
        Q_backward_fail = np.array([key_to_q[k] for k in backward_fail_keys], dtype=float) if backward_fail_keys else None

        P_forward_fail = np.array([fk_cached(key_to_q[k]) for k in forward_fail_keys], dtype=float) if forward_fail_keys else None
        P_backward_fail = np.array([fk_cached(key_to_q[k]) for k in backward_fail_keys], dtype=float) if backward_fail_keys else None
        if P_backward_fail is not None:
            P_backward_fail = P_backward_fail[::-1].copy()
        if Q_backward_fail is not None:
            Q_backward_fail = Q_backward_fail[::-1].copy()

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
        "goal_reached": bool(best_meet_key is not None),
        "step_deg": cfg.step_deg,
        "best_total_cost": best_total_cost if best_meet_key is not None else float("nan"),
        "epsilon_meet_used": bool(best_meet_key == ("EPSILON",)),
        "meet_pos_eps": cfg.meet_pos_eps,
        "best_forward_goal_dist": best_forward_goal_dist,
        "best_backward_start_dist": best_backward_start_dist,
        "best_frontier_gap": best_frontier_gap,
        "P_forward_fail": P_forward_fail,
        "P_backward_fail": P_backward_fail,
        "Q_forward_fail": Q_forward_fail,
        "Q_backward_fail": Q_backward_fail,
        "p_start": p0,
        "p_goal": p1,
    }


def plot_bidirectional_result(res: Dict[str, Any], title: str = "Bidirectional Dijkstra") -> None:
    import matplotlib.pyplot as plt

    P = np.asarray(res["P"], dtype=float)
    P_target = np.asarray(res["P_target"], dtype=float)
    speed = np.asarray(res["speed"], dtype=float) if res.get("speed") is not None else np.array([], dtype=float)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2])
    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    axs = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1])

    fig.suptitle(title, fontsize=16)

    ax3d.plot(P_target[:, 0], P_target[:, 1], P_target[:, 2], label="target")
    if res.get("goal_reached", False):
        ax3d.plot(P[:, 0], P[:, 1], P[:, 2], label="achieved")
    else:
        Pf = res.get("P_forward_fail")
        Pb = res.get("P_backward_fail")
        if Pf is not None and len(Pf) > 0:
            ax3d.plot(Pf[:, 0], Pf[:, 1], Pf[:, 2], label="forward partial")
        if Pb is not None and len(Pb) > 0:
            ax3d.plot(Pb[:, 0], Pb[:, 1], Pb[:, 2], label="backward partial")
    ax3d.set_title("3D path")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()

    if len(speed) > 0:
        axs.plot(np.arange(len(speed)), speed)
        axs.set_title("Speed")
        axs.set_xlabel("segment")
        axs.set_ylabel("units/s")
    else:
        axs.text(0.5, 0.5, "speed not provided", ha="center", va="center")
        axs.set_axis_off()

    err = np.linalg.norm(P - P_target, axis=1) if len(P) > 0 else np.array([], dtype=float)
    if len(err) > 0:
        axe.plot(np.arange(len(err)), err)
    axe.set_title("Tracking error")
    axe.set_xlabel("sample")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    arm = create_default_arm()
    q_start = {1: 0.0, 2: 70.0, 3: 10.0, 4: 0.0, 5: 0.0}
    q_end   = {1: 0.0, 2: -70.0, 3: 0.0, 4: 0.0, 5: 0.0}

    cfg = BidirectionalDijkstraConfig(
        joints=(1, 2, 3, 4),
        step_deg=4.0,
        total_time=3.0,
        step_ms=20,
        max_expansions=2_000_000,
        max_memory_mb=8000,
        cache_fk=True,
        progress_every=25_000,
        frontier_gap_sample_limit=10000,
    )

    res = bidirectional_dijkstra_cartesian_shortest(arm, q_start, q_end, cfg)
    print("BiDijkstra expansions:", res["expansions"])
    print("States touched:", res["states_touched"])
    print("FK cached:", res["fk_cached"])
    print("best_forward_goal_dist:", float(res["best_forward_goal_dist"]))
    print("best_backward_start_dist:", float(res["best_backward_start_dist"]))
    print("best_frontier_gap:", float(res["best_frontier_gap"]))
    print("best_total_cost:", float(res["best_total_cost"]) if res["best_total_cost"] == res["best_total_cost"] else float("nan"))
    print("epsilon_meet_used:", bool(res["epsilon_meet_used"]), "meet_pos_eps:", float(res["meet_pos_eps"]))
    print("Mean tracking error:", float(np.mean(np.linalg.norm(res["P"] - res["P_target"], axis=1))) if len(res["P"]) else float("nan"))
    print("Max tracking error:", float(np.max(np.linalg.norm(res["P"] - res["P_target"], axis=1))) if len(res["P"]) else float("nan"))
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

    plot_bidirectional_result(res, title=f"Bidirectional Dijkstra Cartesian shortest (step={cfg.step_deg}°)")
