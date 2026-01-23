# fkmodel.py
"""
Forward-kinematics model of a 6DOF manipulator (no external deps).

Coordinate system:
- Origin at base center.
- +X "forward" when J1=0.
- +Y left, +Z up (right-handed).

User requirements:
- Default starting angles are 0° for all servos.
- At that pose, the robot should be upright (arm aligned with +Z).
- Joint layout:
    J1: base yaw (Z)
    J2: bend/pitch (Y)
    J3: bend/pitch (Y)
    J4: bend/pitch (Y)   <-- changed per user
    J5: claw/tool roll (X) <-- changed per user (only rotates claw/tool)
    J6: claw open/close (maps angle to opening distance)

Implementation notes:
- Links translate along local +X.
- We use per-joint internal offsets:
    effective_angle = servo_angle + offset
- With our rot_y definition, rot_y(-90) maps +X -> +Z, so we apply:
    offset[J2] = -90°
  and keep offsets for J3/J4/J5 = 0 so zero pose stays straight.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, radians
from typing import Dict, List, Tuple, Optional

Vec3 = Tuple[float, float, float]
Mat4 = List[List[float]]


# ----------------------------
# Small 4x4 transform helpers
# ----------------------------

def mat4_identity() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                a[i][0] * b[0][j] +
                a[i][1] * b[1][j] +
                a[i][2] * b[2][j] +
                a[i][3] * b[3][j]
            )
    return out


def mat4_translate(x: float, y: float, z: float) -> Mat4:
    m = mat4_identity()
    m[0][3] = x
    m[1][3] = y
    m[2][3] = z
    return m


def mat4_rot_x(deg: float) -> Mat4:
    a = radians(deg)
    c, s = cos(a), sin(a)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c,   -s,  0.0],
        [0.0, s,    c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_rot_y(deg: float) -> Mat4:
    a = radians(deg)
    c, s = cos(a), sin(a)
    return [
        [c,   0.0, s,   0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0, c,   0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_rot_z(deg: float) -> Mat4:
    a = radians(deg)
    c, s = cos(a), sin(a)
    return [
        [c,   -s,  0.0, 0.0],
        [s,    c,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(t: Mat4, p: Vec3) -> Vec3:
    x, y, z = p
    return (
        t[0][0] * x + t[0][1] * y + t[0][2] * z + t[0][3],
        t[1][0] * x + t[1][1] * y + t[1][2] * z + t[1][3],
        t[2][0] * x + t[2][1] * y + t[2][2] * z + t[2][3],
    )


# ----------------------------
# Specs + model
# ----------------------------

@dataclass(frozen=True)
class ServoSpec:
    min_deg: float
    max_deg: float
    steps: int

    def clamp_deg(self, deg: float) -> float:
        return max(self.min_deg, min(self.max_deg, deg))

    def step_to_deg(self, step: int) -> float:
        if self.steps < 2:
            return self.clamp_deg(0.0)
        step = max(0, min(self.steps - 1, step))
        frac = step / (self.steps - 1)
        return self.min_deg + frac * (self.max_deg - self.min_deg)

    def deg_to_step(self, deg: float) -> int:
        if self.steps < 2:
            return 0
        deg = self.clamp_deg(deg)
        if self.max_deg == self.min_deg:
            return 0
        frac = (deg - self.min_deg) / (self.max_deg - self.min_deg)
        step = int(round(frac * (self.steps - 1)))
        return max(0, min(self.steps - 1, step))


@dataclass(frozen=True)
class LinkLengths:
    base_height: float
    l2: float
    l3: float
    l4: float
    tool: float
    finger: float


class Manipulator6DOF:
    def __init__(
        self,
        lengths: LinkLengths,
        servos: Dict[int, ServoSpec],
        claw_max_opening: float,
        claw_min_opening: float,
        angle_offsets_deg: Optional[Dict[int, float]] = None,
        default_angles_deg: Optional[Dict[int, float]] = None,
    ):
        for i in range(1, 7):
            if i not in servos:
                raise ValueError(f"Missing servo spec for joint {i}")

        self.lengths = lengths
        self.servos = servos
        self.claw_max_opening = float(claw_max_opening)
        self.claw_min_opening = float(claw_min_opening)

        # Only J2 needs offset to make zeros pose upright.
        default_offsets = {1: 0.0, 2: -90.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
        self.angle_offsets_deg: Dict[int, float] = dict(default_offsets)
        if angle_offsets_deg:
            self.angle_offsets_deg.update(angle_offsets_deg)

        defaults = {i: 0.0 for i in range(1, 7)}
        if default_angles_deg:
            defaults.update(default_angles_deg)

        self._deg: Dict[int, float] = {}
        for i in range(1, 7):
            self._deg[i] = self.servos[i].clamp_deg(defaults[i])

    # ---- Convenience for GUI ----

    def servo_specs(self) -> Dict[int, ServoSpec]:
        return self.servos

    def _eff(self, joint: int) -> float:
        return self._deg[joint] + self.angle_offsets_deg.get(joint, 0.0)

    # ---- State setters/getters ----

    def set_servo_deg(self, joint: int, deg: float) -> None:
        self._deg[joint] = self.servos[joint].clamp_deg(deg)

    def set_servo_steps(self, joint: int, step: int) -> None:
        self._deg[joint] = self.servos[joint].step_to_deg(step)

    def get_servo_deg(self, joint: int) -> float:
        return self._deg[joint]

    def get_servo_step(self, joint: int) -> int:
        return self.servos[joint].deg_to_step(self._deg[joint])

    def set_all_zero(self) -> None:
        for j in range(1, 7):
            self._deg[j] = self.servos[j].clamp_deg(0.0)

    # ---- Claw mapping ----

    def claw_opening(self) -> float:
        s6 = self.servos[6]
        a = self._deg[6]
        if s6.max_deg == s6.min_deg:
            return (self.claw_max_opening + self.claw_min_opening) / 2.0
        frac = (a - s6.min_deg) / (s6.max_deg - s6.min_deg)
        return self.claw_min_opening + frac * (self.claw_max_opening - self.claw_min_opening)

    # ---- Forward kinematics ----

    def forward_kinematics(self) -> Dict[str, object]:
        """
        Joint layout used here:
          J1: rot Z
          base height (fixed)
          J2: rot Y + translate l2
          J3: rot Y + translate l3
          J4: rot Y + translate l4      (bend)
          J5: rot X + translate tool    (claw/tool roll)
          J6: opening only

        Returns:
          joint_positions: [p0..p6] where p6 is end-effector midpoint
          p_gripper_base: p5 (origin of tool frame / finger base)
          finger_tips: left/right
          claw_opening
        """
        L = self.lengths

        a1 = self._eff(1)
        a2 = self._eff(2)
        a3 = self._eff(3)
        a4 = self._eff(4)
        a5 = self._eff(5)

        T = mat4_identity()
        joint_positions: List[Vec3] = [(0.0, 0.0, 0.0)]  # p0

        # J1 yaw
        T = mat4_mul(T, mat4_rot_z(a1))

        # fixed base height
        T = mat4_mul(T, mat4_translate(0.0, 0.0, L.base_height))
        p1 = transform_point(T, (0.0, 0.0, 0.0))
        joint_positions.append(p1)

        # J2 bend
        T = mat4_mul(T, mat4_rot_y(a2))
        T = mat4_mul(T, mat4_translate(L.l2, 0.0, 0.0))
        p2 = transform_point(T, (0.0, 0.0, 0.0))
        joint_positions.append(p2)

        # J3 bend
        T = mat4_mul(T, mat4_rot_y(a3))
        T = mat4_mul(T, mat4_translate(L.l3, 0.0, 0.0))
        p3 = transform_point(T, (0.0, 0.0, 0.0))
        joint_positions.append(p3)

        # J4 bend (CHANGED: was roll, now pitch like J3)
        T = mat4_mul(T, mat4_rot_y(a4))
        T = mat4_mul(T, mat4_translate(L.l4, 0.0, 0.0))
        p4 = transform_point(T, (0.0, 0.0, 0.0))
        joint_positions.append(p4)

        # J5 roll of claw/tool only (CHANGED: was pitch, now roll)
        T = mat4_mul(T, mat4_rot_x(a5))
        T_tool = mat4_mul(T, mat4_translate(L.tool, 0.0, 0.0))
        p5 = transform_point(T_tool, (0.0, 0.0, 0.0))
        joint_positions.append(p5)

        # Fingers in tool frame
        opening = self.claw_opening()
        half = opening / 2.0
        left_tip_local = (L.finger, +half, 0.0)
        right_tip_local = (L.finger, -half, 0.0)

        left_tip = transform_point(T_tool, left_tip_local)
        right_tip = transform_point(T_tool, right_tip_local)

        end_mid = (
            (left_tip[0] + right_tip[0]) / 2.0,
            (left_tip[1] + right_tip[1]) / 2.0,
            (left_tip[2] + right_tip[2]) / 2.0,
        )
        joint_positions.append(end_mid)

        return {
            "joint_positions": joint_positions,
            "p_gripper_base": p5,
            "end_effector_mid": end_mid,
            "finger_tips": {"left": left_tip, "right": right_tip},
            "claw_opening": opening,
        }
