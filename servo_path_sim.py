"""
servo_path_sim.py

Path planning + servo-command simulation on top of fkmodel.py (6-DOF manipulator FK).

Scenario
--------
Python program sends joint targets over COM to an Arduino which uses the Arduino Servo library
(typical 50 Hz / 20 ms refresh) to drive hobby servos (e.g. MG996R).

This module provides:
- A target Cartesian trajectory p_target(t) (x,y,z).
- A planner that optimizes a sequence of *commanded* joint angles (deg) to:
    (1) track the target trajectory in Cartesian space
    (2) keep approximately constant Cartesian speed, measured over a sliding window
    (3) minimize velocity "flux" (Cartesian acceleration)
    (4) keep joints smooth (optional)

- A command quantizer that produces replayable per-frame commands in microseconds suitable
  for Servo.writeMicroseconds() (with deadband modeling).
- A simple servo dynamics simulator (first-order lag + rate limit) that converts commands
  into actual joint angles, then runs FK to produce the achieved path.

No SciPy required. Uses finite-difference gradients + projected gradient descent for the
continuous plan. Then quantizes and simulates to verify.

Files produced:
- JSON Lines: one frame per line with time_ms and per-joint microseconds (+ optional degrees)
- CSV: time_ms, j1_us..j5_us,(optional j6_us)

Note: fkmodel.py already includes ServoSpec (min/max and optional "steps"). We still output
microseconds because that's what Arduino Servo library consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import numpy as np

import fkmodel

Vec3 = Tuple[float, float, float]
TrajectoryFn = Callable[[float], Vec3]


# -----------------------------
# Robot factory (matches your GUI defaults)
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
# Target trajectory: quadratic Bezier "floor arc"
# -----------------------------

def bezier_quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
    s = float(s)
    return (1.0 - s) ** 2 * p0 + 2.0 * (1.0 - s) * s * p1 + s ** 2 * p2


def make_floor_arc_trajectory(
    a_xy: Tuple[float, float],
    b_xy: Tuple[float, float],
    apex_z: float,
    apex_xy: Optional[Tuple[float, float]] = None,
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


def sample_trajectory(traj: TrajectoryFn, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, int(n))
    return np.array([traj(float(si)) for si in s], dtype=float)


def polyline_length(P: np.ndarray) -> float:
    d = P[1:] - P[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


# -----------------------------
# Command model (deg <-> us + deadband)
# -----------------------------

@dataclass
class JointPulseMap:
    """
    Linear map between degrees and pulse width (microseconds).

    By default we use the common 500..2500 us range. Many hobby servos respond roughly
    around 1000..2000 us; calibration is recommended eventually.

    deadband_us:
        If abs(new_us - prev_us) < deadband_us -> keep prev_us (models effective deadband).
    """
    us_min: int = 500
    us_max: int = 2500
    deg_min: float = -90.0
    deg_max: float = +90.0
    deadband_us: int = 6

    def deg_to_us_float(self, deg: float) -> float:
        deg = float(deg)
        # clamp to mapping range
        deg_c = max(self.deg_min, min(self.deg_max, deg))
        alpha = (deg_c - self.deg_min) / (self.deg_max - self.deg_min)
        return self.us_min + alpha * (self.us_max - self.us_min)

    def us_to_deg_float(self, us: float) -> float:
        us = float(us)
        us_c = max(self.us_min, min(self.us_max, us))
        alpha = (us_c - self.us_min) / (self.us_max - self.us_min)
        return self.deg_min + alpha * (self.deg_max - self.deg_min)

    def quantize_us(self, us_float: float) -> int:
        us_i = int(round(us_float))
        return int(max(self.us_min, min(self.us_max, us_i)))

    def apply_deadband(self, prev_us: int, new_us: int) -> int:
        if abs(int(new_us) - int(prev_us)) < int(self.deadband_us):
            return int(prev_us)
        return int(new_us)


@dataclass
class ServoCommandModel:
    """
    Per-joint pulse mapping. Joints 1..5 are planned, J6 optional.

    You can override any joint mapping to match your real calibration.
    """
    joint_map: Dict[int, JointPulseMap] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.joint_map:
            # Default maps:
            # J2 has a smaller physical range in fkmodel specs, but pulse map can still cover full.
            self.joint_map = {
                1: JointPulseMap(deg_min=-90, deg_max=+90),
                2: JointPulseMap(deg_min=-60, deg_max=+60),
                3: JointPulseMap(deg_min=-80, deg_max=+80),
                4: JointPulseMap(deg_min=-90, deg_max=+90),
                5: JointPulseMap(deg_min=-90, deg_max=+90),
                6: JointPulseMap(deg_min=0, deg_max=90),
            }

    def deg_to_us(self, joint: int, deg: float) -> int:
        m = self.joint_map[joint]
        return m.quantize_us(m.deg_to_us_float(deg))

    def us_to_deg(self, joint: int, us: float) -> float:
        return self.joint_map[joint].us_to_deg_float(us)


def quantize_command_sequence_us(
    model: ServoCommandModel,
    Q_deg: np.ndarray,
    joints: Sequence[int],
    initial_us: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """
    Convert Q_deg [N x D] into integer microseconds [N x D] with deadband applied sequentially.
    """
    N, D = Q_deg.shape
    out = np.zeros((N, D), dtype=int)

    prev: Dict[int, int] = {}
    if initial_us:
        prev.update({int(k): int(v) for k, v in initial_us.items()})

    for i in range(N):
        for k, j in enumerate(joints):
            new_us = model.deg_to_us(int(j), float(Q_deg[i, k]))
            if j in prev:
                new_us = model.joint_map[int(j)].apply_deadband(prev[int(j)], new_us)
            out[i, k] = int(new_us)
            prev[int(j)] = int(new_us)
    return out


# -----------------------------
# Servo dynamics simulation
# -----------------------------

@dataclass
class JointDynamics:
    """
    Simple servo model for joint angle evolution under discrete commands.
    """
    tau_s: float = 0.10        # first-order lag time constant
    omega_max_dps: float = 250.0  # max speed (deg/s)


@dataclass
class ServoDynamicsModel:
    dt_cmd: float = 0.02
    joint_dyn: Dict[int, JointDynamics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.joint_dyn:
            self.joint_dyn = {j: JointDynamics() for j in range(1, 7)}


def simulate_servo_response_deg(
    cmd_deg: np.ndarray,
    joints: Sequence[int],
    dyn: ServoDynamicsModel,
    theta0_deg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate actual joint angles given commanded degrees.

    cmd_deg: [N x D]
    returns theta_actual_deg: [N x D]
    """
    N, D = cmd_deg.shape
    dt = float(dyn.dt_cmd)

    theta = np.zeros((N, D), dtype=float)
    if theta0_deg is not None:
        if theta0_deg.shape != (D,):
            raise ValueError("theta0_deg must have shape (D,)")
        theta[0] = theta0_deg
    else:
        theta[0] = cmd_deg[0]

    for i in range(1, N):
        for k, j in enumerate(joints):
            jd = dyn.joint_dyn[int(j)]
            tau = max(1e-4, float(jd.tau_s))
            omega = float(jd.omega_max_dps)

            # first-order lag velocity command:
            v = (cmd_deg[i, k] - theta[i - 1, k]) / tau
            # rate limit:
            v = max(-omega, min(omega, v))
            theta[i, k] = theta[i - 1, k] + v * dt
    return theta


def simulate_from_commands_us(
    arm: fkmodel.Manipulator6DOF,
    cmd_us: np.ndarray,
    joints: Sequence[int],
    cmd_model: ServoCommandModel,
    dyn: ServoDynamicsModel,
    theta0_deg: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert microseconds -> commanded degrees -> simulate servo -> run FK.

    Returns:
      cmd_deg         [N x D]
      theta_actual_deg[N x D]
      P               [N x 3]
    """
    N, D = cmd_us.shape
    cmd_deg = np.zeros((N, D), dtype=float)
    for i in range(N):
        for k, j in enumerate(joints):
            cmd_deg[i, k] = cmd_model.us_to_deg(int(j), float(cmd_us[i, k]))

    theta_actual = simulate_servo_response_deg(cmd_deg, joints, dyn, theta0_deg=theta0_deg)

    P = np.zeros((N, 3), dtype=float)
    for i in range(N):
        for k, j in enumerate(joints):
            arm.set_servo_deg(int(j), float(theta_actual[i, k]))
        fk = arm.forward_kinematics()
        x, y, z = fk["end_effector_mid"]
        P[i] = (float(x), float(y), float(z))

    return {"cmd_deg": cmd_deg, "theta_actual_deg": theta_actual, "P": P}


# -----------------------------
# Planner (continuous optimization + verification)
# -----------------------------

@dataclass
class PlannerConfig:
    # Time discretization
    total_time_s: float = 3.0
    n_frames: int = 151               # at 50 Hz, 3 seconds -> 151 frames (including endpoints)

    # Joints to plan
    joints: Tuple[int, ...] = (1, 2, 3, 4, 5)

    # Objective weights
    w_track: float = 1.0
    w_speed_window: float = 0.5
    w_flux: float = 0.15
    w_joint_smooth: float = 0.01

    # Speed window
    speed_window_s: float = 0.5       # default 0.5 sec
    # Optimizer
    iters: int = 80
    lr: float = 0.08
    fd_eps_deg: float = 0.5
    clip_grad: float = 200.0


def _project_joint_limits(arm: fkmodel.Manipulator6DOF, q_deg: np.ndarray, joints: Sequence[int]) -> np.ndarray:
    out = q_deg.copy()
    specs = arm.servo_specs()
    for k, j in enumerate(joints):
        out[k] = specs[int(j)].clamp_deg(float(out[k]))
    return out


def _fk_path_from_Q(
    arm: fkmodel.Manipulator6DOF,
    Q_deg: np.ndarray,
    joints: Sequence[int],
) -> np.ndarray:
    N, D = Q_deg.shape
    P = np.zeros((N, 3), dtype=float)
    for i in range(N):
        qi = _project_joint_limits(arm, Q_deg[i], joints)
        for k, j in enumerate(joints):
            arm.set_servo_deg(int(j), float(qi[k]))
        fk = arm.forward_kinematics()
        x, y, z = fk["end_effector_mid"]
        P[i] = (float(x), float(y), float(z))
    return P


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    win = int(win)
    k = np.ones(win, dtype=float) / float(win)
    # "same" convolution with reflect padding
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    y = np.convolve(xpad, k, mode="valid")
    return y[: x.shape[0]]


def plan_path(
    arm: fkmodel.Manipulator6DOF,
    q_start_deg: Dict[int, float],
    q_end_deg: Dict[int, float],
    target_traj: TrajectoryFn,
    cfg: PlannerConfig = PlannerConfig(),
    cmd_model: ServoCommandModel = ServoCommandModel(),
    dyn_model: ServoDynamicsModel = ServoDynamicsModel(),
) -> Dict[str, object]:
    """
    Plan and simulate.

    Returns:
      commands_us:        [N x D] int
      commands_deg:       [N x D] float (after us->deg mapping)
      theta_actual_deg:   [N x D] float
      P:                  [N x 3] achieved
      P_target:           [N x 3] target samples
      Q_cont_deg:         [N x D] continuous planned deg (pre-quantization)
      loss_history:       list[float]
      speed:              [N-1] float
      speed_windowed:     [N-1] float
      dt_cmd:             float
      v_des:              float
    """
    joints = cfg.joints
    D = len(joints)

    dt = float(dyn_model.dt_cmd)
    N = int(cfg.n_frames)

    if abs(cfg.total_time_s - dt * (N - 1)) > 1e-6:
        # keep internal consistency: dt_cmd dictates total time implied by N
        total_time = dt * (N - 1)
    else:
        total_time = float(cfg.total_time_s)

    # sample target at N points (uniform parameter)
    P_target = sample_trajectory(target_traj, N)
    v_des = polyline_length(P_target) / max(1e-6, total_time)

    # initial joint endpoints
    q0 = np.array([float(q_start_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)
    qT = np.array([float(q_end_deg.get(j, arm.get_servo_deg(j))) for j in joints], dtype=float)

    # initial guess: joint linear interpolation
    Q = np.zeros((N, D), dtype=float)
    for i in range(N):
        s = i / (N - 1)
        Q[i] = (1 - s) * q0 + s * qT
        Q[i] = _project_joint_limits(arm, Q[i], joints)

    # optimize intermediate points only
    def pack(Qall: np.ndarray) -> np.ndarray:
        return Qall[1:-1].reshape(-1)

    def unpack(x: np.ndarray) -> np.ndarray:
        Qall = np.zeros((N, D), dtype=float)
        Qall[0] = q0
        Qall[-1] = qT
        Qall[1:-1] = x.reshape(N - 2, D)
        # clamp
        for i in range(1, N - 1):
            Qall[i] = _project_joint_limits(arm, Qall[i], joints)
        return Qall

    win = max(1, int(round(cfg.speed_window_s / dt)))
    loss_hist: List[float] = []

    def loss_from_vars(x: np.ndarray) -> float:
        Qall = unpack(x)
        P = _fk_path_from_Q(arm, Qall, joints)

        # tracking
        track = float(np.sum(np.sum((P - P_target) ** 2, axis=1)))

        # per-segment speed
        dP = P[1:] - P[:-1]
        speed = np.linalg.norm(dP, axis=1) / dt
        speed_w = _moving_average(speed, win)

        # constant windowed speed around v_des
        speed_term = float(np.sum((speed_w - v_des) ** 2))

        # flux / accel (second difference in position)
        # P'' approx: P[i+1] - 2P[i] + P[i-1]
        acc = P[2:] - 2.0 * P[1:-1] + P[:-2]
        flux = float(np.sum(np.sum((acc / (dt * dt)) ** 2, axis=1)))

        # joint smoothness
        dQ = Qall[1:] - Qall[:-1]
        js = float(np.sum(np.sum(dQ ** 2, axis=1)))

        return (
            cfg.w_track * track
            + cfg.w_speed_window * speed_term
            + cfg.w_flux * flux
            + cfg.w_joint_smooth * js
        )

    x = pack(Q)

    # Finite-difference gradient descent on the continuous plan
    eps = float(cfg.fd_eps_deg)
    lr = float(cfg.lr)

    for it in range(int(cfg.iters)):
        base = loss_from_vars(x)
        loss_hist.append(base)

        g = np.zeros_like(x)
        for k in range(x.size):
            xk = x[k]
            x[k] = xk + eps
            f1 = loss_from_vars(x)
            x[k] = xk - eps
            f2 = loss_from_vars(x)
            x[k] = xk
            g[k] = (f1 - f2) / (2.0 * eps)

        gn = float(np.linalg.norm(g))
        if gn > cfg.clip_grad:
            g *= (cfg.clip_grad / gn)

        x_new = x - lr * g

        # accept & small lr adaptation
        new_loss = loss_from_vars(x_new)
        if new_loss <= base:
            x = x_new
            lr *= 1.01
        else:
            lr *= 0.90

    Q_cont = unpack(x)

    # --- Quantize to microseconds (replayable) ---
    cmd_us = quantize_command_sequence_us(cmd_model, Q_cont, joints)

    # --- Simulate real servo response (with dynamics) ---
    sim = simulate_from_commands_us(arm, cmd_us, joints, cmd_model, dyn_model, theta0_deg=q0)

    P = sim["P"]
    dP = P[1:] - P[:-1]
    speed = np.linalg.norm(dP, axis=1) / dt
    speed_w = _moving_average(speed, win)

    return {
        "commands_us": cmd_us,
        "commands_deg": sim["cmd_deg"],
        "theta_actual_deg": sim["theta_actual_deg"],
        "P": P,
        "P_target": P_target,
        "Q_cont_deg": Q_cont,
        "loss_history": loss_hist,
        "speed": speed,
        "speed_windowed": speed_w,
        "dt_cmd": dt,
        "total_time_s": total_time,
        "v_des": v_des,
        "joints": joints,
        "planner_cfg": asdict(cfg),
    }


# -----------------------------
# Export helpers
# -----------------------------

def export_commands_jsonl(
    path: str,
    commands_us: np.ndarray,
    joints: Sequence[int],
    dt_cmd: float,
    commands_deg: Optional[np.ndarray] = None,
) -> None:
    """
    Write JSON Lines (one frame per line).
    Fields:
      time_ms: int
      joints_us: { "1": 1500, ... }
      joints_deg: { "1": 10.0, ... }   (optional)
    """
    N, D = commands_us.shape
    with open(path, "w", encoding="utf-8") as f:
        for i in range(N):
            t_ms = int(round(1000.0 * dt_cmd * i))
            rec = {
                "time_ms": t_ms,
                "joints_us": {str(int(joints[k])): int(commands_us[i, k]) for k in range(D)},
            }
            if commands_deg is not None:
                rec["joints_deg"] = {str(int(joints[k])): float(commands_deg[i, k]) for k in range(D)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_commands_csv(
    path: str,
    commands_us: np.ndarray,
    joints: Sequence[int],
    dt_cmd: float,
) -> None:
    """
    CSV columns: time_ms, j1_us, j2_us, ...
    """
    N, D = commands_us.shape
    header = ["time_ms"] + [f"j{int(j)}_us" for j in joints]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(N):
            t_ms = int(round(1000.0 * dt_cmd * i))
            row = [str(t_ms)] + [str(int(commands_us[i, k])) for k in range(D)]
            f.write(",".join(row) + "\n")


def replay_simulation(
    arm: fkmodel.Manipulator6DOF,
    commands_us: np.ndarray,
    joints: Sequence[int],
    cmd_model: ServoCommandModel,
    dyn_model: ServoDynamicsModel,
    theta0_deg: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper to re-simulate (useful after re-loading commands from file).
    """
    return simulate_from_commands_us(arm, commands_us, joints, cmd_model, dyn_model, theta0_deg=theta0_deg)


# -----------------------------
# Optional plotting
# -----------------------------

def plot_diagnostics(res: Dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    P = np.asarray(res["P"])
    T = np.asarray(res["P_target"])
    speed = np.asarray(res["speed"])
    speed_w = np.asarray(res["speed_windowed"])
    v_des = float(res["v_des"])
    loss = np.asarray(res["loss_history"])

    # 3D path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(T[:, 0], T[:, 1], T[:, 2])
    ax.plot(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(["target", "achieved"])

    # loss
    plt.figure()
    plt.plot(loss)
    plt.xlabel("iter")
    plt.ylabel("loss")

    # speed
    plt.figure()
    plt.plot(speed, label="speed")
    plt.plot(speed_w, label="windowed speed")
    plt.axhline(v_des, linestyle="--", label="v_des")
    plt.xlabel("segment")
    plt.ylabel("speed (units/s)")
    plt.legend()

    plt.show()


# -----------------------------
# Demo
# -----------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


if __name__ == "__main__":
    arm = create_default_arm()

    # Example start/end (deg) for joints 1..5
    q_start = {1: 0.0, 2: 90.0, 3: 0.0, 4: 0.0, 5: 0.0}
    q_end = {1: 0.0, 2: -90.0, 3: 0.0, 4: 0.0, 5: 0.0}

    # Target arc on the floor with one apex
    traj = make_floor_arc_trajectory(
        a_xy=(120.0, -40.0),
        b_xy=(140.0, +60.0),
        apex_z=80.0,
        apex_xy=(160.0, 10.0),
    )

    # Models
    cmd_model = ServoCommandModel()
    dyn_model = ServoDynamicsModel(
        dt_cmd=0.02,  # 50 Hz
        joint_dyn={j: JointDynamics(tau_s=0.10, omega_max_dps=250.0) for j in range(1, 7)},
    )

    cfg = PlannerConfig(
        total_time_s=3.0,
        n_frames=int(round(3.0 / dyn_model.dt_cmd)) + 1,
        w_track=1.0,
        w_speed_window=0.6,
        w_flux=0.15,
        w_joint_smooth=0.01,
        speed_window_s=0.5,
        iters=60,
        lr=0.08,
        fd_eps_deg=0.5,
    )

    res = plan_path(arm, q_start, q_end, traj, cfg, cmd_model, dyn_model)

    # Export replay files
    export_commands_jsonl(
        "optimal_path.jsonl",
        res["commands_us"],
        res["joints"],
        float(res["dt_cmd"]),
        commands_deg=res["commands_deg"],
    )
    export_commands_csv(
        "optimal_path.csv",
        res["commands_us"],
        res["joints"],
        float(res["dt_cmd"]),
    )

    # Print quick diagnostics
    P = np.asarray(res["P"])
    T = np.asarray(res["P_target"])
    err = P - T
    rms_err = _rms(err)

    speed = np.asarray(res["speed"])
    speed_w = np.asarray(res["speed_windowed"])
    print("Saved: optimal_path.jsonl and optimal_path.csv")
    print(f"RMS tracking error: {rms_err:.3f} (same units as FK)")
    print(f"Speed mean: {float(np.mean(speed)):.3f}, std: {float(np.std(speed)):.3f}")
    print(f"Windowed speed std: {float(np.std(speed_w)):.3f} (target v_des={float(res['v_des']):.3f})")

    # Uncomment if you have matplotlib installed:
    plot_diagnostics(res)
