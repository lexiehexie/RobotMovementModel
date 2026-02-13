# gui_fk.py
"""
Tkinter GUI for fkmodel.py
- Fixed camera (user adjustable)
- Fixed axis scale (no auto-rescale)
- Z-limits shifted up so there is little space below the base
- First immovable segment (base_height) not drawn as a cylinder

Run:
  python3 gui_fk.py

Deps:
  pip install matplotlib numpy
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
import json
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

import fkmodel

Vec3 = Tuple[float, float, float]


def vec(a: Vec3) -> np.ndarray:
    return np.array([a[0], a[1], a[2]], dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def orthonormal_basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = unit(axis)
    if abs(axis[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(axis, tmp))
    v = unit(np.cross(axis, u))
    return u, v


def cylinder_between(p0: Vec3, p1: Vec3, radius: float, n_theta: int = 18, n_z: int = 2):
    a = vec(p0)
    b = vec(p1)
    axis = b - a
    L = float(np.linalg.norm(axis))
    if L < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
        L = 1e-6

    w = unit(axis)
    u, v = orthonormal_basis_from_axis(w)

    theta = np.linspace(0, 2 * math.pi, n_theta)
    z = np.linspace(0, L, n_z)

    Theta, Z = np.meshgrid(theta, z)
    Xc = radius * np.cos(Theta)
    Yc = radius * np.sin(Theta)

    P = (
        a[None, None, :]
        + (w[None, None, :] * Z[:, :, None])
        + (u[None, None, :] * Xc[:, :, None])
        + (v[None, None, :] * Yc[:, :, None])
    )
    return P[:, :, 0], P[:, :, 1], P[:, :, 2]


def sphere(center: Vec3, radius: float, n_u: int = 18, n_v: int = 12):
    c = vec(center)
    u = np.linspace(0, 2 * math.pi, n_u)
    v = np.linspace(0, math.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = c[0] + radius * np.cos(U) * np.sin(V)
    Y = c[1] + radius * np.sin(U) * np.sin(V)
    Z = c[2] + radius * np.cos(V)
    return X, Y, Z


class RobotViewer3D:
    def __init__(self, parent: tk.Widget):
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Fixed camera
        self.elev = 25.0
        self.azim = -60.0

        # Fixed scale parameters
        self.world_radius_xy = 200.0     # half-size for X/Y
        self.world_zmax = 200.0          # upper Z bound
        self.world_floor = 20.0          # how much below Z=0 we show

        self._apply_view()
        self._apply_fixed_limits()

    def _apply_view(self):
        self.ax.view_init(elev=self.elev, azim=self.azim)

    def set_world(self, r_xy: float, zmax: float, floor: float):
        self.world_radius_xy = float(max(1.0, r_xy))
        self.world_zmax = float(max(1.0, zmax))
        self.world_floor = float(max(0.0, floor))

    def _apply_fixed_limits(self):
        r = self.world_radius_xy
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_zlim(-self.world_floor, self.world_zmax)

    def clear(self):
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._apply_view()
        self._apply_fixed_limits()

    def draw_robot(
        self,
        joint_positions: List[Vec3],
        gripper_base: Vec3,
        left_tip: Vec3,
        right_tip: Vec3,
        link_radius: float,
        joint_radius: float,
        tip_radius: float,
        draw_base_height_segment: bool = False,
    ):
        self.clear()

        p0, p1, p2, p3, p4, p5, p6 = joint_positions

        segments = []
        if draw_base_height_segment:
            segments.append((p0, p1))
        segments += [(p1, p2), (p2, p3), (p3, p4), (p4, p5)]

        for a, b in segments:
            X, Y, Z = cylinder_between(a, b, radius=link_radius)
            self.ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, shade=True)

        for tip in (left_tip, right_tip):
            X, Y, Z = cylinder_between(gripper_base, tip, radius=max(link_radius * 0.65, 0.5))
            self.ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, shade=True)

        for p in (p0, p1, p2, p3, p4, p5):
            Xs, Ys, Zs = sphere(p, radius=joint_radius)
            self.ax.plot_surface(Xs, Ys, Zs, linewidth=0, antialiased=True, shade=True)

        for p in (left_tip, right_tip, p6):
            Xs, Ys, Zs = sphere(p, radius=tip_radius)
            self.ax.plot_surface(Xs, Ys, Zs, linewidth=0, antialiased=True, shade=True)

        # Re-apply fixed view/limits (matplotlib can tweak after plot_surface)
        self._apply_view()
        self._apply_fixed_limits()
        self.canvas.draw_idle()


class FKGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6DOF FK Viewer (fixed camera + fixed scale, shifted Z)")
        self.geometry("1250x720")

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

        self.arm = fkmodel.Manipulator6DOF(
            lengths=lengths,
            servos=servos,
            claw_max_opening=70.0,
            claw_min_opening=5.0,
        )

        avg_len = (lengths.l2 + lengths.l3 + lengths.l4 + lengths.tool) / 4.0
        self.link_radius = max(2.0, 0.06 * avg_len)
        self.joint_radius = self.link_radius * 1.2
        self.tip_radius = self.link_radius * 1.0

        # --- Fixed world extents ---
        reach = (lengths.l2 + lengths.l3 + lengths.l4 + lengths.tool + lengths.finger)
        margin = 20.0
        xy_r = reach + self.arm.claw_max_opening / 2.0 + margin + 3.0 * self.joint_radius
        zmax = lengths.base_height + reach + margin + 3.0 * self.joint_radius

        # little space below base (Z=0)
        floor = max(10.0, 0.10 * zmax)

        self.base_xy_r = float(xy_r)
        self.base_zmax = float(zmax)
        self.base_floor = float(floor)

        # ---- Layout ----
        root = ttk.PanedWindow(self, orient="horizontal")
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root, padding=10)
        right = ttk.Frame(root, padding=5)
        root.add(left, weight=0)
        root.add(right, weight=1)

        self.viewer = RobotViewer3D(right)
        self.viewer.set_world(self.base_xy_r, self.base_zmax, self.base_floor)

        ttk.Label(left, text="Servo angles (degrees)").pack(anchor="w")

        self.vars: Dict[int, tk.DoubleVar] = {}
        self.labels: Dict[int, ttk.Label] = {}
        self._ignore_slider = False  # used during playback to avoid slider callbacks

        meaning = {1: "yaw", 2: "bend", 3: "bend", 4: "bend", 5: "claw roll", 6: "open"}

        for j in range(1, 7):
            spec = self.arm.servo_specs()[j]
            v = tk.DoubleVar(value=self.arm.get_servo_deg(j))
            self.vars[j] = v

            frame = ttk.Frame(left)
            frame.pack(fill="x", pady=6)

            ttk.Label(frame, text=f"J{j} ({meaning[j]})  [{spec.min_deg:.0f} .. {spec.max_deg:.0f}]").pack(anchor="w")

            row = ttk.Frame(frame)
            row.pack(fill="x")

            s = ttk.Scale(
                row,
                from_=spec.min_deg,
                to=spec.max_deg,
                orient="horizontal",
                variable=v,
                command=lambda _val, jj=j: self.on_slider(jj),
            )
            s.pack(side="left", fill="x", expand=True)

            val_lbl = ttk.Label(row, text="")
            val_lbl.pack(side="right", padx=(8, 0))
            self.labels[j] = val_lbl

        # Camera controls
        sep = ttk.Separator(left, orient="horizontal")
        sep.pack(fill="x", pady=10)
        ttk.Label(left, text="Camera (fixed during motion)").pack(anchor="w")

        self.elev_var = tk.DoubleVar(value=self.viewer.elev)
        self.azim_var = tk.DoubleVar(value=self.viewer.azim)

        cam1 = ttk.Frame(left)
        cam1.pack(fill="x", pady=4)
        ttk.Label(cam1, text="Elev").pack(side="left")
        ttk.Scale(
            cam1, from_=-10, to=80, orient="horizontal", variable=self.elev_var,
            command=lambda _v: self.on_camera_change()
        ).pack(side="left", fill="x", expand=True)

        cam2 = ttk.Frame(left)
        cam2.pack(fill="x", pady=4)
        ttk.Label(cam2, text="Azim").pack(side="left")
        ttk.Scale(
            cam2, from_=-180, to=180, orient="horizontal", variable=self.azim_var,
            command=lambda _v: self.on_camera_change()
        ).pack(side="left", fill="x", expand=True)

        # Scale controls (still fixed; user adjusts constants)
        sep2 = ttk.Separator(left, orient="horizontal")
        sep2.pack(fill="x", pady=10)
        ttk.Label(left, text="Scale (fixed) + floor").pack(anchor="w")

        self.scale_var = tk.DoubleVar(value=1.0)  # multiplier
        self.floor_var = tk.DoubleVar(value=self.base_floor)

        srow = ttk.Frame(left)
        srow.pack(fill="x", pady=4)
        ttk.Label(srow, text="Scale").pack(side="left")
        ttk.Scale(
            srow, from_=0.7, to=1.8, orient="horizontal", variable=self.scale_var,
            command=lambda _v: self.on_scale_change()
        ).pack(side="left", fill="x", expand=True)

        frow = ttk.Frame(left)
        frow.pack(fill="x", pady=4)
        ttk.Label(frow, text="Floor").pack(side="left")
        ttk.Scale(
            frow, from_=0.0, to=max(30.0, self.base_floor * 2.0), orient="horizontal",
            variable=self.floor_var, command=lambda _v: self.on_scale_change()
        ).pack(side="left", fill="x", expand=True)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_row, text="Set all to 0°", command=self.set_all_zero).pack(fill="x")

        # ---- Playback controls (load CSV/JSONL and animate) ----
        sep_play = ttk.Separator(left, orient="horizontal")
        sep_play.pack(fill="x", pady=10)
        ttk.Label(left, text="Playback (CSV/JSONL from planner)").pack(anchor="w")

        self.path_file_var = tk.StringVar(value="")
        pf_row = ttk.Frame(left)
        pf_row.pack(fill="x", pady=4)
        ttk.Entry(pf_row, textvariable=self.path_file_var).pack(side="left", fill="x", expand=True)
        ttk.Button(pf_row, text="Load…", command=self.load_path_file).pack(side="right", padx=(6, 0))

        ctl_row = ttk.Frame(left)
        ctl_row.pack(fill="x", pady=4)
        ttk.Button(ctl_row, text="Play", command=self.play_loaded_path).pack(side="left", fill="x", expand=True)
        ttk.Button(ctl_row, text="Stop", command=self.stop_playback).pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.play_speed_var = tk.DoubleVar(value=1.0)
        sp_row = ttk.Frame(left)
        sp_row.pack(fill="x", pady=4)
        ttk.Label(sp_row, text="Speed").pack(side="left")
        ttk.Scale(
            sp_row, from_=0.2, to=2.0, orient="horizontal", variable=self.play_speed_var
        ).pack(side="left", fill="x", expand=True)

        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Loop", variable=self.loop_var).pack(anchor="w")

        self._loaded_frames: List[Dict[int, float]] = []
        self._loaded_times_ms: List[int] = []
        self._play_idx = 0
        self._play_after_id = None

        self.info = ttk.Label(left, text="", justify="left")
        self.info.pack(fill="x", pady=(10, 0))

        self._pending_redraw = False
        self.refresh_ui_and_redraw()

    def set_all_zero(self):
        self.arm.set_all_zero()
        for j in range(1, 7):
            self.vars[j].set(self.arm.get_servo_deg(j))
        self.refresh_ui_and_redraw()

    def on_slider(self, joint: int):
        if self._ignore_slider:
            return
        self.arm.set_servo_deg(joint, float(self.vars[joint].get()))
        if not self._pending_redraw:
            self._pending_redraw = True
            self.after(10, self.refresh_ui_and_redraw)

    def on_camera_change(self):
        self.viewer.elev = float(self.elev_var.get())
        self.viewer.azim = float(self.azim_var.get())
        self.refresh_ui_and_redraw()

    def on_scale_change(self):
        m = float(self.scale_var.get())
        floor = float(self.floor_var.get())
        self.viewer.set_world(self.base_xy_r * m, self.base_zmax * m, floor)
        self.refresh_ui_and_redraw()

    def refresh_ui_and_redraw(self):
        self._pending_redraw = False

        offs = self.arm.angle_offsets_deg
        for j in range(1, 7):
            deg = self.arm.get_servo_deg(j)
            step = self.arm.get_servo_step(j)
            steps = self.arm.servo_specs()[j].steps
            eff = deg + offs.get(j, 0.0)
            self.labels[j].config(text=f"{deg:7.2f}° (eff {eff:7.2f}°)  step {step}/{steps-1}")

        fk = self.arm.forward_kinematics()
        joints: List[Vec3] = fk["joint_positions"]
        p5: Vec3 = fk["p_gripper_base"]
        tips = fk["finger_tips"]
        left_tip: Vec3 = tips["left"]
        right_tip: Vec3 = tips["right"]
        mid: Vec3 = fk["end_effector_mid"]
        opening: float = fk["claw_opening"]

        self.info.config(
            text=(
                f"End-effector midpoint:\n"
                f"  X={mid[0]:.2f}  Y={mid[1]:.2f}  Z={mid[2]:.2f}\n"
                f"Claw opening: {opening:.2f}\n"
                f"Camera: elev={self.viewer.elev:.1f}, azim={self.viewer.azim:.1f}\n"
                f"Scale: xy_r={self.viewer.world_radius_xy:.1f}, zmax={self.viewer.world_zmax:.1f}, floor={self.viewer.world_floor:.1f}\n"
                f"Offsets: {self.arm.angle_offsets_deg}"
            )
        )

        self.viewer.draw_robot(
            joint_positions=joints,
            gripper_base=p5,
            left_tip=left_tip,
            right_tip=right_tip,
            link_radius=self.link_radius,
            joint_radius=self.joint_radius,
            tip_radius=self.tip_radius,
            draw_base_height_segment=False,
        )

    # ---------------- Playback: load + animate ----------------

    def load_path_file(self):
        path = filedialog.askopenfilename(
            title="Open path file",
            filetypes=[("Planner files", "*.jsonl *.csv"), ("JSON Lines", "*.jsonl"), ("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            frames, times = self._read_path_file(path)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load file:\n{path}\n\n{e}")
            return

        if not frames:
            messagebox.showwarning("Empty file", "Loaded file contains no frames.")
            return

        self.path_file_var.set(path)
        self._loaded_frames = frames
        self._loaded_times_ms = times
        self._play_idx = 0

        messagebox.showinfo(
            "Loaded",
            f"Loaded {len(frames)} frames.\n"
            f"Duration ~{(times[-1] - times[0]) / 1000.0:.2f}s\n"
            f"File: {os.path.basename(path)}",
        )

    def _read_path_file(self, path: str) -> Tuple[List[Dict[int, float]], List[int]]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return self._read_csv(path)
        if ext == ".jsonl":
            return self._read_jsonl(path)
        raise ValueError("Unsupported file type (use .csv or .jsonl)")

    def _us_to_deg(self, joint: int, us: float) -> float:
        """Simple linear map 500..2500 us -> [min_deg..max_deg] for this joint."""
        spec = self.arm.servo_specs()[joint]
        us_min, us_max = 500.0, 2500.0
        u = max(us_min, min(us_max, float(us)))
        alpha = (u - us_min) / (us_max - us_min)
        return float(spec.min_deg + alpha * (spec.max_deg - spec.min_deg))

    def _read_csv(self, path: str) -> Tuple[List[Dict[int, float]], List[int]]:
        frames: List[Dict[int, float]] = []
        times: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], []
            # Expect time_ms plus j{n}_us columns.
            for row in reader:
                if not row:
                    continue
                t_ms = int(float(row.get("time_ms", "0") or 0))
                frame: Dict[int, float] = {}
                for j in range(1, 7):
                    key = f"j{j}_us"
                    if key in row and row[key] not in (None, ""):
                        deg = self._us_to_deg(j, float(row[key]))
                        frame[j] = deg
                if frame:
                    frames.append(frame)
                    times.append(t_ms)
        # If time missing/constant, synthesize 20ms steps
        if frames and (len(times) != len(frames) or (len(times) > 1 and times[-1] == times[0])):
            times = [i * 20 for i in range(len(frames))]
        return frames, times

    def _read_jsonl(self, path: str) -> Tuple[List[Dict[int, float]], List[int]]:
        frames: List[Dict[int, float]] = []
        times: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t_ms = int(rec.get("time_ms", 0))
                frame: Dict[int, float] = {}

                # Prefer joints_deg if present (more direct)
                jd = rec.get("joints_deg")
                if isinstance(jd, dict) and jd:
                    for k, v in jd.items():
                        try:
                            j = int(k)
                            frame[j] = float(v)
                        except Exception:
                            pass
                else:
                    ju = rec.get("joints_us")
                    if isinstance(ju, dict) and ju:
                        for k, v in ju.items():
                            try:
                                j = int(k)
                                frame[j] = self._us_to_deg(j, float(v))
                            except Exception:
                                pass

                if frame:
                    frames.append(frame)
                    times.append(t_ms)

        if frames and (len(times) != len(frames) or (len(times) > 1 and times[-1] == times[0])):
            times = [i * 20 for i in range(len(frames))]
        return frames, times

    def play_loaded_path(self):
        if not self._loaded_frames:
            messagebox.showwarning("No path loaded", "Load a .csv or .jsonl path file first.")
            return
        self.stop_playback()
        self._play_idx = 0
        self._playback_step()

    def stop_playback(self):
        if self._play_after_id is not None:
            try:
                self.after_cancel(self._play_after_id)
            except Exception:
                pass
        self._play_after_id = None

    def _playback_step(self):
        if not self._loaded_frames:
            self.stop_playback()
            return

        if self._play_idx >= len(self._loaded_frames):
            if bool(self.loop_var.get()):
                self._play_idx = 0
            else:
                self.stop_playback()
                return

        frame = self._loaded_frames[self._play_idx]

        # Apply frame to arm + sliders without triggering callbacks
        self._ignore_slider = True
        try:
            for j, deg in frame.items():
                if 1 <= int(j) <= 6:
                    self.arm.set_servo_deg(int(j), float(deg))
                    self.vars[int(j)].set(float(deg))
        finally:
            self._ignore_slider = False

        self.refresh_ui_and_redraw()

        # Schedule next
        speed = max(0.05, float(self.play_speed_var.get()))
        if self._play_idx + 1 < len(self._loaded_times_ms):
            dt_ms = self._loaded_times_ms[self._play_idx + 1] - self._loaded_times_ms[self._play_idx]
            dt_ms = max(1, int(round(dt_ms / speed)))
        else:
            dt_ms = max(1, int(round(20 / speed)))

        self._play_idx += 1
        self._play_after_id = self.after(dt_ms, self._playback_step)



if __name__ == "__main__":
    app = FKGuiApp()
    app.mainloop()
