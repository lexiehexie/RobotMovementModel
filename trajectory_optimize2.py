"""
Stable IK trajectory follower (playback-compatible export)

This version fixes the "teleport / branch jump / huge speed spikes" seen in the
previous Jacobian IK by adding:

1) Trust region across the path:
   Each waypoint is constrained to stay close to the previous waypoint in joint
   space (limits per-step joint change). This enforces continuity.

2) Step rejection + backtracking line search:
   For each IK iteration we only accept an update if it actually reduces the
   Cartesian position error. Otherwise we shrink the step (alpha *= 0.5).

3) Adaptive damping:
   If updates are frequently rejected (near singularities), damping increases.
   If updates are accepted easily, damping decreases.

4) Forward + backward passes:
   We solve the path forward from start, then backward from end, then merge and
   re-fit, improving global consistency.

5) Smooth + re-IK:
   Light smoothing in joint space followed by re-IK "projection" back onto the
   target Cartesian samples.

Outputs:
  optimal_path.jsonl and optimal_path.csv in the same schema used by
  main_with_playback.py.

Dependencies: numpy (matplotlib optional).
"""

from __future__ import annotations
from plot_utils import plot_dashboard

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List

import numpy as np
import fkmodel

Vec3 = Tuple[float, float, float]
TrajectoryFn = Callable[[float], Vec3]


# -----------------------------
# Default robot (matches GUI)
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
# Trajectories
# -----------------------------

def line_trajectory(p0: np.ndarray, p1: np.ndarray) -> TrajectoryFn:
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)

    def traj(t01: float) -> Vec3:
        t01 = float(t01)
        p = (1 - t01) * p0 + t01 * p1
        return (float(p[0]), float(p[1]), float(p[2]))
    return traj


def bezier_quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
    s = float(s)
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


def make_floor_arc_trajectory(
    a_xy: Tuple[float, float],
    b_xy: Tuple[float, float],
    apex_z: float,
    apex_xy: Tuple[float, float] | None = None,
) -> TrajectoryFn:
    ax, ay = a_xy
    bx, by = b_xy
    if apex_xy is None:
        cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
    else:
        cx, cy = apex_xy

    p0 = np.array([ax, ay, 0.0], dtype=float)
    p1 = np.array([cx, cy, float(apex_z)], dtype=float)
    p2 = np.array([bx, by, 0.0], dtype=float)

    def traj(t01: float) -> Vec3:
        p = bezier_quadratic(p0, p1, p2, float(t01))
        return (float(p[0]), float(p[1]), float(p[2]))
    return traj


# -----------------------------
# FK / limits / Jacobian
# -----------------------------

def _fk_end(arm: fkmodel.Manipulator6DOF, q_deg: np.ndarray, joints: Tuple[int, ...]) -> np.ndarray:
    for k, j in enumerate(joints):
        arm.set_servo_deg(j, float(q_deg[k]))
    fk = arm.forward_kinematics()
    x, y, z = fk["end_effector_mid"]
    return np.array([x, y, z], dtype=float)


def _project_to_limits(arm: fkmodel.Manipulator6DOF, q: np.ndarray, joints: Tuple[int, ...]) -> np.ndarray:
    out = q.copy()
    specs = arm.servo_specs()
    for k, j in enumerate(joints):
        out[k] = specs[j].clamp_deg(float(out[k]))
    return out


def _joint_mid_and_range(arm: fkmodel.Manipulator6DOF, joints: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    specs = arm.servo_specs()
    mid = []
    rng = []
    for j in joints:
        s = specs[j]
        mid.append((s.min_deg + s.max_deg) / 2.0)
        rng.append(max(1e-6, (s.max_deg - s.min_deg) / 2.0))
    return np.array(mid, dtype=float), np.array(rng, dtype=float)


def numerical_jacobian_pos(
    arm: fkmodel.Manipulator6DOF,
    q_deg: np.ndarray,
    joints: Tuple[int, ...],
    eps_deg: float = 0.5,
) -> np.ndarray:
    q_deg = np.array(q_deg, dtype=float)
    p0 = _fk_end(arm, q_deg, joints)
    D = len(joints)
    J = np.zeros((3, D), dtype=float)
    for k in range(D):
        q1 = q_deg.copy()
        q1[k] += eps_deg
        q1 = _project_to_limits(arm, q1, joints)
        p1 = _fk_end(arm, q1, joints)
        J[:, k] = (p1 - p0) / eps_deg
    return J


# -----------------------------
# Arc-length sampling (constant speed parameterization)
# -----------------------------

def sample_by_arclength(traj: TrajectoryFn, n: int, dense: int = 500) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, dense)
    P = np.array([traj(float(t)) for t in ts], dtype=float)
    seg = P[1:] - P[:-1]
    ds = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if total < 1e-9:
        return np.repeat(P[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 3), dtype=float)
    j = 0
    for i, si in enumerate(targets):
        while j + 1 < len(s) and s[j + 1] < si:
            j += 1
        if j + 1 >= len(s):
            out[i] = P[-1]
        else:
            s0, s1 = s[j], s[j + 1]
            a = 0.0 if (s1 - s0) < 1e-12 else (si - s0) / (s1 - s0)
            out[i] = (1 - a) * P[j] + a * P[j + 1]
    return out


# -----------------------------
# Stable DLS IK
# -----------------------------

@dataclass
class IKStableConfig:
    joints: Tuple[int, ...] = (1, 2, 3, 4, 5)
    n_steps: int = 60
    total_time: float = 3.0

    # IK loop
    ik_iters: int = 80
    tol_mm: float = 0.5

    # Jacobian
    jac_eps_deg: float = 0.5

    # Damping (adaptive)
    damping_init: float = 8.0
    damping_min: float = 1.0
    damping_max: float = 80.0
    damping_up: float = 2.0       # multiply if rejected
    damping_down: float = 0.85    # multiply if accepted

    # Step control
    max_step_deg: float = 2.0     # per-IK-iteration clamp
    line_search_steps: int = 8    # 1,1/2,1/4,...

    # Trust region across waypoints (continuity)
    max_waypoint_delta_deg: float = 3.0  # clamp |q_i - q_{i-1}| each joint

    # Nullspace limit avoidance
    w_null: float = 0.6

    # Path refinement
    smooth_passes: int = 2


def _dls(J: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    A = J @ J.T + (lam * lam) * np.eye(3)
    y = np.linalg.solve(A, e)
    return J.T @ y


def _pseudoinv_dls(J: np.ndarray, lam: float) -> np.ndarray:
    # J+ = J^T (J J^T + λ^2 I)^-1
    A = J @ J.T + (lam * lam) * np.eye(3)
    return J.T @ np.linalg.solve(A, np.eye(3))


def _clamp_waypoint_delta(q: np.ndarray, q_prev: np.ndarray, max_delta: float) -> np.ndarray:
    q = q.copy()
    dq = q - q_prev
    dq = np.clip(dq, -max_delta, +max_delta)
    return q_prev + dq


def ik_to_point_stable(
    arm: fkmodel.Manipulator6DOF,
    q_init: np.ndarray,
    q_prev: np.ndarray,
    p_target: np.ndarray,
    cfg: IKStableConfig,
    q_mid: np.ndarray,
    q_rng: np.ndarray,
) -> np.ndarray:
    joints = cfg.joints
    q = _project_to_limits(arm, q_init, joints)
    q = _clamp_waypoint_delta(q, q_prev, cfg.max_waypoint_delta_deg)
    q = _project_to_limits(arm, q, joints)

    lam = float(cfg.damping_init)

    p = _fk_end(arm, q, joints)
    err = float(np.linalg.norm(p_target - p))

    for _ in range(cfg.ik_iters):
        if err <= cfg.tol_mm:
            break

        J = numerical_jacobian_pos(arm, q, joints, eps_deg=cfg.jac_eps_deg)

        e = p_target - p
        dq = _dls(J, e, lam)

        # Nullspace: avoid joint limits (prefer mid-range)
        Jplus = _pseudoinv_dls(J, lam)
        N = np.eye(len(joints)) - Jplus @ J
        grad_limits = (q - q_mid) / (q_rng * q_rng)
        dq = dq - cfg.w_null * (N @ grad_limits)

        # Clamp per-iteration update
        m = float(np.max(np.abs(dq)))
        if m > cfg.max_step_deg:
            dq *= (cfg.max_step_deg / m)

        # Backtracking line search: accept only if improves error
        accepted = False
        alpha = 1.0
        q_best = q
        p_best = p
        err_best = err

        for _ls in range(cfg.line_search_steps):
            q_try = q + alpha * dq
            q_try = _project_to_limits(arm, q_try, joints)
            q_try = _clamp_waypoint_delta(q_try, q_prev, cfg.max_waypoint_delta_deg)
            q_try = _project_to_limits(arm, q_try, joints)

            p_try = _fk_end(arm, q_try, joints)
            err_try = float(np.linalg.norm(p_target - p_try))

            if err_try < err_best - 1e-6:
                accepted = True
                q_best, p_best, err_best = q_try, p_try, err_try
                break

            alpha *= 0.5

        if accepted:
            q, p, err = q_best, p_best, err_best
            lam = max(cfg.damping_min, lam * cfg.damping_down)
        else:
            # If we cannot find an improving step, increase damping and try again.
            lam = min(cfg.damping_max, lam * cfg.damping_up)
            # If damping is already huge, we are stuck -> stop.
            if lam >= cfg.damping_max - 1e-9:
                break

    return q


def smooth_joint_path(Q: np.ndarray, passes: int = 1) -> np.ndarray:
    Qs = Q.copy()
    for _ in range(passes):
        Qn = Qs.copy()
        Qn[1:-1] = 0.25 * Qs[:-2] + 0.5 * Qs[1:-1] + 0.25 * Qs[2:]
        Qn[0] = Qs[0]
        Qn[-1] = Qs[-1]
        Qs = Qn
    return Qs


def plan_path_stable_ik(
    arm: fkmodel.Manipulator6DOF,
    q_start_deg: Dict[int, float],
    q_end_deg: Dict[int, float],
    traj: TrajectoryFn,
    cfg: IKStableConfig = IKStableConfig(),
) -> Dict[str, object]:
    joints = cfg.joints
    N = int(cfg.n_steps)
    D = len(joints)
    dt = float(cfg.total_time) / float(N - 1)

    q0 = np.array([float(q_start_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)
    qT = np.array([float(q_end_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)
    q0 = _project_to_limits(arm, q0, joints)
    qT = _project_to_limits(arm, qT, joints)

    P_target = sample_by_arclength(traj, N, dense=700)

    q_mid, q_rng = _joint_mid_and_range(arm, joints)

    # Initial joint path guess
    Q = np.zeros((N, D), dtype=float)
    for i in range(N):
        s = i / (N - 1)
        Q[i] = (1 - s) * q0 + s * qT
    Q[0] = q0
    Q[-1] = qT

    # Forward pass (use previous waypoint as warm start)
    for i in range(1, N - 1):
        Q[i] = ik_to_point_stable(arm, Q[i], Q[i - 1], P_target[i], cfg, q_mid, q_rng)

    # Backward pass to propagate end constraint influence
    Qb = Q.copy()
    for i in range(N - 2, 0, -1):
        Qb[i] = ik_to_point_stable(arm, Qb[i], Qb[i + 1], P_target[i], cfg, q_mid, q_rng)

    # Merge (average) + enforce limits/continuity
    Qm = Q.copy()
    Qm[1:-1] = 0.5 * (Q[1:-1] + Qb[1:-1])
    for i in range(1, N - 1):
        Qm[i] = _project_to_limits(arm, Qm[i], joints)
        Qm[i] = _clamp_waypoint_delta(Qm[i], Qm[i - 1], cfg.max_waypoint_delta_deg)
        Qm[i] = _project_to_limits(arm, Qm[i], joints)

    Q = Qm

    # Smooth + re-IK projection passes
    for _ in range(int(cfg.smooth_passes)):
        Q = smooth_joint_path(Q, passes=1)
        Q[0] = q0
        Q[-1] = qT
        for i in range(1, N - 1):
            Q[i] = ik_to_point_stable(arm, Q[i], Q[i - 1], P_target[i], cfg, q_mid, q_rng)

    # Achieved Cartesian path
    P = np.zeros((N, 3), dtype=float)
    for i in range(N):
        P[i] = _fk_end(arm, Q[i], joints)

    dP = P[1:] - P[:-1]
    speed = np.linalg.norm(dP, axis=1) / dt

    return {
        "Q_deg": Q,
        "P": P,
        "P_target": P_target,
        "speed": speed,
        "dt": dt,
        "joints": joints,
    }


# -----------------------------
# Playback export (same schema)
# -----------------------------

import json
import csv


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
# Demo (your problematic start/end)
# -----------------------------

if __name__ == "__main__":
    arm = create_default_arm()

    q_start = {1: 0.0, 2: 70.0, 3: 0.0, 4: 0.0, 5: 0.0}
    q_end   = {1: 0.0, 2: -70.0, 3: 0.0, 4: 0.0, 5: 0.0}

    joints = (1, 2, 3, 4, 5)

    # Track a straight line between FK(start) and FK(end)
    q0 = np.array([q_start[j] for j in joints], dtype=float)
    qT = np.array([q_end[j] for j in joints], dtype=float)
    p0 = _fk_end(arm, q0, joints)
    p1 = _fk_end(arm, qT, joints)
    traj = line_trajectory(p0, p1)

    cfg = IKStableConfig(
        joints=joints,
        n_steps=140,
        total_time=3.0,
        ik_iters=100,
        tol_mm=0.8,
        damping_init=12.0,
        max_step_deg=2.0,
        max_waypoint_delta_deg=1.5,
        w_null=0.8,
        smooth_passes=2,
    )

    res = plan_path_stable_ik(arm, q_start, q_end, traj, cfg)

    print("Mean tracking error:", float(np.mean(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Max tracking error:", float(np.max(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Speed std:", float(np.std(res["speed"])))

    jsonl_path, csv_path = save_for_main_with_playback(
        arm=arm,
        Q_deg_waypoints=res["Q_deg"],
        joints=tuple(res["joints"]),
        total_time_s=cfg.total_time,
        out_base="optimal_path",
        step_ms=20,
    )
    print("Saved for playback:", jsonl_path, csv_path)

    plot_dashboard(res, title="Stable IK", show=True)
