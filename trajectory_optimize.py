"""
Trajectory-following path optimization for the 6-DOF FK model.

What this does
--------------
Given:
- a target Cartesian-space trajectory p_target(t) in (x,y,z)
- start and end joint angles (deg)
it finds a sequence of joint angles q[0..N-1] that makes the end-effector midpoint
track the trajectory while also moving with near-constant Cartesian speed.

Optimization targets
--------------------
1) Tracking error:   sum_t || p(q_t) - p_target(t) ||^2
2) Constant speed:   sum_t ( ||p_{t+1}-p_t||/dt - v_des )^2
3) Low velocity flux:sum_t || (p_{t+1}-p_t) - (p_t-p_{t-1}) ||^2   (Cartesian accel^2)
(Optional) joint smoothness: sum_t ||q_{t+1}-q_t||^2

No SciPy required. Uses simple finite-difference gradients + projected gradient descent.

Dependencies: only numpy (and optionally matplotlib for plots).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

import fkmodel  # uses Manipulator6DOF forward kinematics (user-provided)

Vec3 = Tuple[float, float, float]
TrajectoryFn = Callable[[float], Vec3]


# -----------------------------
# Default robot (matches GUI)
# -----------------------------

def create_default_arm() -> fkmodel.Manipulator6DOF:
    """Create a Manipulator6DOF with the same geometry/limits as the GUI example."""
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
        2: fkmodel.ServoSpec(min_deg=-60, max_deg=+60, steps=121),
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
# Test trajectory: 3D arc
# -----------------------------

def bezier_quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
    """Quadratic Bezier in R^3."""
    s = float(s)
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


def make_floor_arc_trajectory(
    a_xy: Tuple[float, float],
    b_xy: Tuple[float, float],
    apex_z: float,
    apex_xy: Tuple[float, float] | None = None,
) -> TrajectoryFn:
    """
    Arc between two floor points A and B (z=0), with one 'upper' apex control point.

    - If apex_xy is None, apex XY is chosen as the midpoint between A and B.
    - apex_z controls how high the arc rises.
    """
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
# Optimization
# -----------------------------

@dataclass
class OptimizeConfig:
    n_steps: int = 35               # number of discrete points along the path
    total_time: float = 3.0         # seconds (used only for dt in speed terms)
    joints_to_opt: Tuple[int, ...] = (1, 2, 3, 4, 5)  # ignore J6 (claw opening)

    # Weights
    w_track: float = 1.0
    w_speed: float = 0.5
    w_flux: float = 0.2
    w_joint_smooth: float = 0.02

    # Optimizer
    iters: int = 140
    lr: float = 0.12               # learning rate in degrees (scaled internally)
    fd_eps_deg: float = 0.5        # finite-diff epsilon (deg)
    clip_grad: float = 200.0       # to avoid blowups


def _fk_end(arm: fkmodel.Manipulator6DOF, q_deg: np.ndarray, joints: Tuple[int, ...]) -> np.ndarray:
    """Set joints in arm and return end-effector midpoint as np.array([x,y,z])."""
    for k, j in enumerate(joints):
        arm.set_servo_deg(j, float(q_deg[k]))
    fk = arm.forward_kinematics()
    x, y, z = fk["end_effector_mid"]
    return np.array([x, y, z], dtype=float)


def _project_to_limits(arm: fkmodel.Manipulator6DOF, q: np.ndarray, joints: Tuple[int, ...]) -> np.ndarray:
    """Clamp each joint angle to its ServoSpec bounds."""
    out = q.copy()
    for k, j in enumerate(joints):
        spec = arm.servo_specs()[j]
        out[k] = spec.clamp_deg(float(out[k]))
    return out


def _pack_vars(Q: np.ndarray) -> np.ndarray:
    """Flatten intermediate waypoints (excluding endpoints)."""
    # Q shape: (N, D)
    return Q[1:-1].reshape(-1)


def _unpack_vars(x: np.ndarray, q0: np.ndarray, qT: np.ndarray, N: int, D: int) -> np.ndarray:
    """Reconstruct full Q with fixed endpoints."""
    Q = np.zeros((N, D), dtype=float)
    Q[0] = q0
    Q[-1] = qT
    Q[1:-1] = x.reshape(N - 2, D)
    return Q


def optimize_trajectory_following(
    arm: fkmodel.Manipulator6DOF,
    q_start_deg: Dict[int, float],
    q_end_deg: Dict[int, float],
    target_traj: TrajectoryFn,
    cfg: OptimizeConfig = OptimizeConfig(),
) -> Dict[str, object]:
    """
    Returns dict with:
      - Q_deg: (N,D) optimized joint angles for joints cfg.joints_to_opt
      - P: (N,3) achieved end-effector path
      - P_target: (N,3) target samples
      - loss_history: list[float]
      - joints: tuple of optimized joint indices
    """
    joints = cfg.joints_to_opt
    D = len(joints)
    N = int(cfg.n_steps)
    if N < 3:
        raise ValueError("n_steps must be >= 3")

    dt = float(cfg.total_time) / float(N - 1)

    q0 = np.array([float(q_start_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)
    qT = np.array([float(q_end_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)

    # Initial guess: linear interpolation in joint space.
    Q = np.zeros((N, D), dtype=float)
    for i in range(N):
        s = i / (N - 1)
        Q[i] = (1 - s) * q0 + s * qT

    # Project to limits (endpoints too)
    for i in range(N):
        Q[i] = _project_to_limits(arm, Q[i], joints)

    # Pre-sample target trajectory uniformly in [0,1]
    S = np.linspace(0.0, 1.0, N)
    P_target = np.array([target_traj(float(s)) for s in S], dtype=float)

    # Desired average Cartesian speed along the target
    seg = P_target[1:] - P_target[:-1]
    target_len = float(np.sum(np.linalg.norm(seg, axis=1)))
    v_des = target_len / float(cfg.total_time)

    loss_hist: List[float] = []

    def loss_from_vars(x: np.ndarray) -> float:
        Qx = _unpack_vars(x, q0, qT, N, D)

        # FK positions
        P = np.zeros((N, 3), dtype=float)
        for i in range(N):
            qi = _project_to_limits(arm, Qx[i], joints)
            P[i] = _fk_end(arm, qi, joints)

        # 1) tracking
        track = np.sum(np.sum((P - P_target) ** 2, axis=1))

        # 2) constant speed
        dP = P[1:] - P[:-1]
        speeds = np.linalg.norm(dP, axis=1) / dt
        speed = np.sum((speeds - v_des) ** 2)

        # 3) velocity flux (acceleration)
        v = dP / dt
        a = v[1:] - v[:-1]
        flux = np.sum(np.sum(a ** 2, axis=1))

        # 4) joint smoothness (discourage wild joint motions)
        dQ = Qx[1:] - Qx[:-1]
        js = np.sum(np.sum(dQ ** 2, axis=1))

        return cfg.w_track * track + cfg.w_speed * speed + cfg.w_flux * flux + cfg.w_joint_smooth * js

    # Variables: all intermediate joint angles
    x = _pack_vars(Q)

    # Project intermediate points to joint limits at init
    Q = _unpack_vars(x, q0, qT, N, D)
    for i in range(1, N - 1):
        Q[i] = _project_to_limits(arm, Q[i], joints)
    x = _pack_vars(Q)

    # Finite-difference gradient descent
    eps = float(cfg.fd_eps_deg)
    lr = float(cfg.lr)

    for it in range(int(cfg.iters)):
        base = loss_from_vars(x)
        loss_hist.append(base)

        g = np.zeros_like(x)
        # Central differences
        for k in range(x.size):
            xk = x[k]
            x[k] = xk + eps
            f1 = loss_from_vars(x)
            x[k] = xk - eps
            f2 = loss_from_vars(x)
            x[k] = xk
            g[k] = (f1 - f2) / (2.0 * eps)

        # Gradient clipping
        gn = float(np.linalg.norm(g))
        if gn > cfg.clip_grad:
            g *= (cfg.clip_grad / gn)

        # Step (note: lr is in "deg" units effectively)
        x = x - lr * g

        # Project back to joint limits
        Q = _unpack_vars(x, q0, qT, N, D)
        for i in range(1, N - 1):
            Q[i] = _project_to_limits(arm, Q[i], joints)
        x = _pack_vars(Q)

        # Simple lr schedule (optional): if we got worse, shrink lr a bit.
        if it >= 2 and loss_hist[-1] > loss_hist[-2]:
            lr *= 0.95

    # Final reconstruction
    Q = _unpack_vars(x, q0, qT, N, D)
    P = np.zeros((N, 3), dtype=float)
    for i in range(N):
        qi = _project_to_limits(arm, Q[i], joints)
        P[i] = _fk_end(arm, qi, joints)

    return {
        "Q_deg": Q,
        "P": P,
        "P_target": P_target,
        "loss_history": loss_hist,
        "joints": joints,
        "dt": dt,
        "v_des": v_des,
    }


# -----------------------------
# Optional plotting helper
# -----------------------------

def plot_result(result: Dict[str, object]) -> None:
    """
    Requires matplotlib. Plots:
      - 3D path: achieved vs target
      - loss curve
      - speed over time
    """
    import matplotlib.pyplot as plt

    P = np.asarray(result["P"])
    T = np.asarray(result["P_target"])
    dt = float(result["dt"])

    # 3D path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(T[:, 0], T[:, 1], T[:, 2])
    ax.plot(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(["target", "achieved"])

    # loss curve
    plt.figure()
    plt.plot(result["loss_history"])
    plt.xlabel("iter")
    plt.ylabel("loss")

    # speed curve
    dP = P[1:] - P[:-1]
    speed = np.linalg.norm(dP, axis=1) / dt
    plt.figure()
    plt.plot(speed)
    plt.axhline(float(result["v_des"]), linestyle="--")
    plt.xlabel("segment")
    plt.ylabel("speed (units/s)")

    plt.show()


# -----------------------------
# Demo
# -----------------------------

if __name__ == "__main__":
    arm = create_default_arm()

    # Example: start/end joint angles (degrees). Fill any subset; unspecified -> current arm values.
    q_start = {1: 0.0, 2: 90.0, 3: 0.0, 4: 0.0, 5: 0.0}
    q_end = {1: 0.0, 2: -90.0, 3: 0.0, 4: 0.0, 5: 0.0}

    # Target Cartesian arc on the "floor" with an apex
    traj = make_floor_arc_trajectory(
        a_xy=(120.0, -40.0),
        b_xy=(140.0, +60.0),
        apex_z=80.0,
        apex_xy=(160.0, 10.0),
    )

    cfg = OptimizeConfig(
        n_steps=35,
        total_time=3.0,
        w_track=1.0,
        w_speed=0.6,
        w_flux=0.25,
        w_joint_smooth=0.02,
        iters=120,
        lr=0.10,
        fd_eps_deg=0.5,
    )

    res = optimize_trajectory_following(arm, q_start, q_end, traj, cfg)
    print("Final loss:", res["loss_history"][-1])

    #Uncomment if you have matplotlib:
    plot_result(res)
