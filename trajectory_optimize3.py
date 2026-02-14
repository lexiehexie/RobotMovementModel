"""
Batch Levenberg–Marquardt trajectory optimizer (playback-compatible)

Why this is better than per-point IK
-----------------------------------
Per-point (sequential) IK can get stuck or drift when the trajectory passes near
singularities or when multiple IK branches exist. This implementation optimizes
ALL intermediate waypoints at once, with a strong tracking term and an explicit
smoothness term that enforces continuity.

We minimize a least-squares objective:

  sum_i  w_pos^2 * || FK(q_i) - p_target_i ||^2
+ sum_i  w_smooth^2 * || q_{i+1} - q_i ||^2
+ sum_i  w_limits^2 * || (q_i - q_mid)/q_rng ||^2   (soft limit avoidance)

Endpoints are fixed (q_0 = q_start, q_{N-1} = q_end).

We solve using Levenberg–Marquardt (damped Gauss–Newton) with backtracking line search.

Outputs optimal_path.jsonl and optimal_path.csv in the same schema as main_with_playback.py.

Dependencies: numpy (matplotlib optional)
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


def sample_by_arclength(traj: TrajectoryFn, n: int, dense: int = 800) -> np.ndarray:
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
# Batch LM optimizer
# -----------------------------

@dataclass
class LMConfig:
    joints: Tuple[int, ...] = (1, 2, 3, 4, 5)
    n_steps: int = 120
    total_time: float = 3.0

    w_pos: float = 1.0          # tracking weight
    w_smooth: float = 0.20      # joint smoothness (continuity)
    w_limits: float = 0.02      # soft limit avoidance

    jac_eps_deg: float = 0.5

    iters: int = 40
    damping_init: float = 30.0
    damping_min: float = 1e-3
    damping_max: float = 1e6

    line_search_steps: int = 8
    step_scale_max: float = 1.0

    clip_dx: float = 10.0       # clamp max absolute joint update per LM step (deg)


def _pack(Q: np.ndarray) -> np.ndarray:
    return Q[1:-1].reshape(-1)


def _unpack(x: np.ndarray, q0: np.ndarray, qT: np.ndarray, N: int, D: int) -> np.ndarray:
    Q = np.zeros((N, D), dtype=float)
    Q[0] = q0
    Q[-1] = qT
    Q[1:-1] = x.reshape(N - 2, D)
    return Q


def build_residual_and_jacobian(
    arm: fkmodel.Manipulator6DOF,
    Q: np.ndarray,
    P_target: np.ndarray,
    cfg: LMConfig,
    q_mid: np.ndarray,
    q_rng: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Residual vector r and Jacobian J for least squares:
      r = [ w_pos*(p_i - p*_i) for i=0..N-1 ,
            w_smooth*(q_{i+1}-q_i) for i=0..N-2 ,
            w_limits*((q_i-qmid)/qrng) for i=0..N-1 ]
    Variables are only intermediate waypoints (1..N-2).
    """
    joints = cfg.joints
    N, D = Q.shape
    nvar = (N - 2) * D

    # residual sizes
    m_pos = 3 * N
    m_smooth = D * (N - 1)
    m_lim = D * N
    m = m_pos + m_smooth + m_lim

    r = np.zeros((m,), dtype=float)
    J = np.zeros((m, nvar), dtype=float)

    def var_index(i: int, k: int) -> int:
        # i in [1..N-2]
        return (i - 1) * D + k

    row = 0

    # --- position residuals ---
    w = cfg.w_pos
    for i in range(N):
        qi = _project_to_limits(arm, Q[i], joints)
        pi = _fk_end(arm, qi, joints)
        ei = pi - P_target[i]
        r[row:row + 3] = w * ei

        if 1 <= i <= N - 2:
            Ji = numerical_jacobian_pos(arm, qi, joints, eps_deg=cfg.jac_eps_deg)  # 3xD
            # scaled
            Ji = w * Ji
            for k in range(D):
                J[row:row + 3, var_index(i, k)] = Ji[:, k]

        row += 3

    # --- smoothness residuals ---
    w = cfg.w_smooth
    for i in range(N - 1):
        di = Q[i + 1] - Q[i]
        r[row:row + D] = w * di

        # Jacobian contribution depends on whether i or i+1 are variables
        for k in range(D):
            if 1 <= i <= N - 2:
                J[row + k, var_index(i, k)] += -w
            if 1 <= (i + 1) <= N - 2:
                J[row + k, var_index(i + 1, k)] += +w
        row += D

    # --- soft joint-limit avoidance ---
    w = cfg.w_limits
    for i in range(N):
        gi = (Q[i] - q_mid) / q_rng
        r[row:row + D] = w * gi
        if 1 <= i <= N - 2:
            for k in range(D):
                J[row + k, var_index(i, k)] = w / q_rng[k]
        row += D

    return r, J


def solve_batch_lm(
    arm: fkmodel.Manipulator6DOF,
    q_start_deg: Dict[int, float],
    q_end_deg: Dict[int, float],
    traj: TrajectoryFn,
    cfg: LMConfig = LMConfig(),
) -> Dict[str, object]:
    joints = cfg.joints
    N = int(cfg.n_steps)
    D = len(joints)
    dt = float(cfg.total_time) / float(N - 1)

    q0 = np.array([float(q_start_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)
    qT = np.array([float(q_end_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)
    q0 = _project_to_limits(arm, q0, joints)
    qT = _project_to_limits(arm, qT, joints)

    P_target = sample_by_arclength(traj, N, dense=1000)

    q_mid, q_rng = _joint_mid_and_range(arm, joints)

    # initial Q: linear interpolation
    Q = np.zeros((N, D), dtype=float)
    for i in range(N):
        s = i / (N - 1)
        Q[i] = (1 - s) * q0 + s * qT
    Q[0], Q[-1] = q0, qT
    for i in range(1, N - 1):
        Q[i] = _project_to_limits(arm, Q[i], joints)

    x = _pack(Q)

    lam = float(cfg.damping_init)
    loss_hist: List[float] = []

    def loss_from_Q(Qcur: np.ndarray) -> float:
        r, _ = build_residual_and_jacobian(arm, Qcur, P_target, cfg, q_mid, q_rng)
        return float(r @ r)

    base_loss = loss_from_Q(Q)
    loss_hist.append(base_loss)

    for _it in range(cfg.iters):
        Q = _unpack(x, q0, qT, N, D)
        for i in range(1, N - 1):
            Q[i] = _project_to_limits(arm, Q[i], joints)

        r, J = build_residual_and_jacobian(arm, Q, P_target, cfg, q_mid, q_rng)
        loss = float(r @ r)
        loss_hist.append(loss)

        # LM normal equations: (J^T J + lam I) dx = -J^T r
        JTJ = J.T @ J
        g = J.T @ r
        A = JTJ + lam * np.eye(JTJ.shape[0])
        try:
            dx = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            lam = min(cfg.damping_max, lam * 10.0)
            continue

        # clamp dx
        dx = np.clip(dx, -cfg.clip_dx, +cfg.clip_dx)

        # backtracking line search on x
        accepted = False
        alpha = float(cfg.step_scale_max)
        x_best = x
        best = loss

        for _ls in range(cfg.line_search_steps):
            x_try = x + alpha * dx
            Q_try = _unpack(x_try, q0, qT, N, D)
            for i in range(1, N - 1):
                Q_try[i] = _project_to_limits(arm, Q_try[i], joints)

            l_try = loss_from_Q(Q_try)
            if l_try < best - 1e-9:
                accepted = True
                x_best = x_try
                best = l_try
                break
            alpha *= 0.5

        if accepted:
            x = x_best
            # decrease damping when good progress
            lam = max(cfg.damping_min, lam * 0.7)
        else:
            # increase damping when stuck
            lam = min(cfg.damping_max, lam * 3.0)

        # early exit if tiny improvement
        if len(loss_hist) >= 3 and abs(loss_hist[-1] - loss_hist[-2]) < 1e-6:
            break

    # final reconstruction
    Q = _unpack(x, q0, qT, N, D)
    for i in range(1, N - 1):
        Q[i] = _project_to_limits(arm, Q[i], joints)

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
        "loss_history": loss_hist,
        "lm_damping_final": lam,
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
# Demo (your scenario)
# -----------------------------

if __name__ == "__main__":
    arm = create_default_arm()

    q_start = {1: 0.0, 2: 70.0, 3: 0.0, 4: 0.0, 5: 0.0}
    q_end   = {1: 0.0, 2: -70.0, 3: 0.0, 4: 0.0, 5: 0.0}

    joints = (1, 2, 3, 4, 5)

    q0 = np.array([q_start[j] for j in joints], dtype=float)
    qT = np.array([q_end[j] for j in joints], dtype=float)
    p0 = _fk_end(arm, q0, joints)
    p1 = _fk_end(arm, qT, joints)

    traj = line_trajectory(p0, p1)

    cfg = LMConfig(
        joints=joints,
        n_steps=200,
        total_time=3.0,
        w_pos=3.0,
        w_smooth=0.20,
        w_limits=0.02,
        iters=35,
        damping_init=60.0,
        clip_dx=6.0,
    )

    res = solve_batch_lm(arm, q_start, q_end, traj, cfg)

    Q = res["Q_deg"]
    dQ = np.abs(Q[1:] - Q[:-1])
    idx = np.argmax(dQ.max(axis=1))
    print("Worst joint step at segment:", idx, "max dQ:", dQ[idx].max())
    print("dQ per joint:", dQ[idx])
    print("Q[idx]:", Q[idx])
    print("Q[idx+1]:", Q[idx + 1])

    print("Mean tracking error:", float(np.mean(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Max tracking error:", float(np.max(np.linalg.norm(res["P"] - res["P_target"], axis=1))))
    print("Speed std:", float(np.std(res["speed"])))
    print("Final damping:", float(res["lm_damping_final"]))

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
