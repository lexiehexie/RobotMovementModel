"""
plot_utils.py

Reusable plotting helpers for trajectory optimization experiments.

Usage:
    from plot_utils import plot_dashboard

    res = {...}  # dict with keys: P, P_target, speed (optional)
    plot_dashboard(res, title="Batch LM", show=True, save_path=None)

The function intentionally does NOT require optimization loss, so it works with
different optimizers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def plot_dashboard(
    res: Dict[str, Any],
    *,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    One-window dashboard:
      - Large 3D plot (target vs achieved) at the top (spans two columns)
      - Speed at bottom-left (if available)
      - Tracking error at bottom-right

    Required in res:
      - "P": (N,3) achieved Cartesian points
      - "P_target": (N,3) target Cartesian points

    Optional in res:
      - "speed": (N-1,) speed per segment (units/s)
      - "dt": scalar time step (not required)
    """
    import matplotlib.pyplot as plt  # local import to keep module lightweight

    P = np.asarray(res["P"], dtype=float)
    T = np.asarray(res["P_target"], dtype=float)

    if P.shape != T.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("res['P'] and res['P_target'] must both have shape (N, 3)")

    err = np.linalg.norm(P - T, axis=1)

    speed = None
    if "speed" in res and res["speed"] is not None:
        speed = np.asarray(res["speed"], dtype=float)

    fig = plt.figure(figsize=(14, 10))
    if title:
        fig.suptitle(title)

    # ----- Big 3D plot (top, full width) -----
    ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, projection="3d")
    ax.plot(T[:, 0], T[:, 1], T[:, 2])
    ax.plot(P[:, 0], P[:, 1], P[:, 2])
    ax.set_title("3D path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(["target", "achieved"])

    # ----- Speed (bottom-left) -----
    ax = plt.subplot2grid((2, 2), (1, 0))
    if speed is not None and speed.size > 0:
        ax.plot(speed)
        ax.set_title("Speed")
        ax.set_xlabel("segment")
        ax.set_ylabel("units/s")
    else:
        ax.text(0.5, 0.5, "speed not provided", ha="center", va="center")
        ax.set_axis_off()

    # ----- Tracking error (bottom-right) -----
    ax = plt.subplot2grid((2, 2), (1, 1))
    ax.plot(err)
    ax.set_title("Tracking error")
    ax.set_xlabel("sample")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
