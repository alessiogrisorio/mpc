"""Multimetric plots and fast video export, also usable without acados.

Offline: python animation.py results/RUN/hj.npz --video --brt-dir ../brt
Legacy timestamp_hj.npz / timestamp_simulation.npz pairs are supported.
Only optional BRT contours require loading the BRT files.
"""

from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil
import subprocess
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from hj_diagnostics import HJDiagnostics

EGO_LENGTH, EGO_WIDTH = 4.68, 2.20
HUMAN_LENGTH, HUMAN_WIDTH = 4.28, 1.80
ROAD_Y_MIN, ROAD_Y_MAX = -2.0, 6.0

# Video settings are independent of the simulation frequency.
VIDEO_FPS = 25
VIDEO_DPI = 90
VIDEO_PRESET = "ultrafast"
VIDEO_CRF = 22
CONTOUR_HZ = 5
CONTOUR_GRID = (50, 40)

COST_GROUPS = {
    "Longitudinal": (0,),
    "Velocity": (3,),
    "Lateral": (1,),
    "Collision": (10, 11, 12, 13),
    "Overtake": (14,),
    "Road left": (7,),
    "Road right": (8,),
    "Other": (2, 4, 5, 6, 9),
    "HJ slack": (),
}


class OcpCostDiagnostics:
    """Decompose the solved acados NONLINEAR_LS objective by residual.

    acados gives the total through get_cost(), but does not expose the
    contribution of each individual residual. This evaluator uses the *same*
    CasADi model expressions and the current solver W/yref and slacks.
    """

    def __init__(self, model, solver, N, dt_ocp):
        from casadi import Function

        self.N = int(N)
        self.dt_ocp = float(dt_ocp)
        self.stage = Function(
            "diagnostic_stage_residual",
            [model.x, model.u, model.p],
            [model.cost_y_expr],
        ).map(self.N)
        self.terminal = Function(
            "diagnostic_terminal_residual",
            [model.x, model.p],
            [model.cost_y_expr_e],
        )
        self.stage_W = np.asarray(solver.cost_get(0, "W"), dtype=float)
        self.terminal_W = np.asarray(solver.cost_get(self.N, "W"), dtype=float)
        self.zl = np.asarray(solver.cost_get(0, "zl"), dtype=float).ravel()
        self.zu = np.asarray(solver.cost_get(0, "zu"), dtype=float).ravel()
        self.Zl = np.asarray(solver.cost_get(0, "Zl"), dtype=float).ravel()
        self.Zu = np.asarray(solver.cost_get(0, "Zu"), dtype=float).ravel()
        if not np.allclose(self.stage_W, np.diag(np.diag(self.stage_W))):
            raise ValueError("Cost diagnostics expects diagonal W")
        if not np.allclose(self.terminal_W, np.diag(np.diag(self.terminal_W))):
            raise ValueError("Cost diagnostics expects diagonal W_e")

    def evaluate(self, solver):
        N = self.N
        x = np.asarray(solver.get_flat("x"), dtype=float).reshape(N + 1, -1)
        u = np.asarray(solver.get_flat("u"), dtype=float).reshape(N, -1)
        p = np.column_stack([solver.get(j, "p") for j in range(N + 1)])
        yref = np.column_stack([solver.cost_get(j, "yref") for j in range(N)])
        yref_e = np.asarray(solver.cost_get(N, "yref"), dtype=float).ravel()

        y = np.asarray(self.stage(x[:-1].T, u.T, p[:, :N]), dtype=float)
        e = y - yref
        stage_parts = 0.5 * (e**2) * np.diag(self.stage_W)[:, None]
        stage_parts *= self.dt_ocp

        ye = np.asarray(self.terminal(x[-1], p[:, -1]), dtype=float).ravel()
        ee = ye - yref_e
        terminal_parts = 0.5 * np.diag(self.terminal_W) * ee**2

        residual_parts = stage_parts.sum(axis=1)
        residual_parts[:len(terminal_parts)] += terminal_parts

        sl = np.asarray([solver.get(j, "sl") for j in range(N)], dtype=float)
        su = np.asarray([solver.get(j, "su") for j in range(N)], dtype=float)
        slack_cost = self.dt_ocp * np.sum(
            sl * self.zl + 0.5 * sl**2 * self.Zl
            + su * self.zu + 0.5 * su**2 * self.Zu
        )

        components = {
            name: float(np.sum(residual_parts[list(indices)])) if indices else float(slack_cost)
            for name, indices in COST_GROUPS.items()
        }
        total = float(solver.get_cost())
        return total, components


def metric_colors(metrics):
    known = {"euclidean": "tab:blue", "dce": "tab:orange",
             "ttc": "tab:green", "eggert": "tab:red",
             "rss": "tab:purple", "btn": "tab:brown", "stn": "tab:pink"}
    return [known.get(name.lower(), plt.get_cmap("tab20")(k % 20))
            for k, name in enumerate(metrics)]


def _style(axes, xlabel=True):
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        if xlabel:
            ax.set_xlabel("Time [s]")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")


def _shade(ax, t, mask, color="gold", label=None):
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    end_time = t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0.02)
    for k, (start, end) in enumerate(zip(starts, ends)):
        ax.axvspan(t[start], t[end] if end < len(t) else end_time,
                   color=color, alpha=0.15, linewidth=0,
                   label=label if k == 0 else None)


def _metric_series(ax, t, values, metrics, colors, **kwargs):
    for k, (name, color) in enumerate(zip(metrics, colors)):
        ax.plot(t, values[:, k], color=color, label=name, **kwargs)


def plot_control_plane(diagnostics, index, stage=0, ax=None, *, softened=False):
    """Draw HJ half-planes plus their joint admissible control polygon."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    d = diagnostics.arrays()
    lo, hi = diagnostics.u_lower, diagnostics.u_upper
    colors = metric_colors(diagnostics.metrics)
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 100),
                         np.linspace(lo[1], hi[1], 100))
    for k, (name, color) in enumerate(zip(diagnostics.metrics, colors)):
        if not d["active"][index, stage, k]:
            continue
        M = d["M"][index, stage, k]
        b = d["b"][index, stage, k]
        if softened:
            b += d["slack"][index, stage, k]
        field = M[0] * xx + M[1] * yy + b
        scale = max(float(np.max(np.abs(field))), np.finfo(float).tiny)
        field = field / scale
        if field.min() < 0:
            ax.contourf(xx, yy, field, levels=[-2, 0], colors=[color], alpha=0.09)
        if field.min() < 0 < field.max():
            ax.contour(xx, yy, field, levels=[0], colors=[color], linewidths=1.5)
            ax.plot([], [], color=color, label=name)
        elif field.max() < 0:
            ax.plot([], [], color=color, label=f"{name}: no feasible command")
    polygon = diagnostics.control_polygon(index, stage, softened=softened)
    if len(polygon):
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor="tab:green", alpha=0.2,
                edgecolor="black", linewidth=1.2, label="Joint feasible set")
    else:
        ax.text(0.5, 0.9, "Empty joint feasible set", transform=ax.transAxes,
                ha="center", color="red")
    u = d["u"][index, stage]
    ax.plot(u[0], u[1], "k*", markersize=11, label="MPC command")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="Steering rate [rad/s]", ylabel="Acceleration [m/s²]",
           title=f"t = {d['time'][index]:.2f} s | stage {stage} | "
                 + ("with recorded slack" if softened else "without slack"))
    _style([ax], xlabel=False)
    return ax.figure


def save_plots(simulation, diagnostics, path, plane_times=None):
    """Export all static diagnostics into one multipage PDF, without windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = diagnostics.arrays()
    c = diagnostics.contributions()
    summary = diagnostics.summary()
    limits = diagnostics.compute_limits()
    feasibility = diagnostics.feasibility()
    names = diagnostics.metrics
    colors = metric_colors(names)
    t = d["time"]
    x = np.asarray(simulation["simX"])
    u = np.asarray(simulation["simU"])
    tx = np.arange(len(x)) * float(simulation["dt"])
    if plane_times is None:
        active_indices = np.flatnonzero(np.any(d["active"][:, 0], axis=1))
        candidates = ([int(np.argmin(x[:-1, 3]))] +
                      (active_indices[[0, len(active_indices) // 2, -1]].tolist()
                       if len(active_indices) else [0]))
        plane_indices = list(dict.fromkeys(candidates))
    else:
        plane_indices = list(dict.fromkeys(
            int(np.argmin(np.abs(t - value))) for value in plane_times))

    def write(pdf, fig):
        pdf.savefig(fig)
        plt.close(fig)

    with plt.rc_context({"font.size": 9, "axes.titlesize": 11}), PdfPages(path) as pdf:
        pdf.infodict()["Title"] = "Multimetric HJ diagnostics"
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
        ax = axes[0, 0]
        ax.plot(x[:, 0], x[:, 1], label="Ego")
        for y in (ROAD_Y_MIN, ROAD_Y_MAX):
            ax.axhline(y, color="black")
        ax.axhline(2, color="gray", linestyle="--")
        ax.set(xlabel="X [m]", ylabel="Y [m]", title="Global trajectory")
        axes[0, 1].plot(tx, x[:, 3], label="Ego")
        axes[0, 1].axhline(float(simulation["v_ref"]), color="black",
                           linestyle="--", label="Reference")
        axes[0, 1].set(ylabel="m/s", title="Velocity")
        axes[1, 0].step(t, u[:, 0], where="post")
        axes[1, 0].set(ylabel="rad/s", title="Steering rate")
        axes[1, 1].step(t, u[:, 1], where="post")
        axes[1, 1].set(ylabel="m/s²", title="Acceleration")
        axes[2, 0].plot(tx, np.rad2deg(x[:, 2]))
        axes[2, 0].set(ylabel="deg", title="Heading")
        axes[2, 1].plot(tx, np.rad2deg(x[:, 4]))
        axes[2, 1].set(ylabel="deg", title="Steering angle")
        _style(axes)
        axes[0, 0].set_xlabel("X [m]")
        write(pdf, fig)

        fig, axes = plt.subplots(len(names), 3, figsize=(13, max(4, 2.5 * len(names))),
                                 squeeze=False, sharex=True, layout="constrained")
        for k, (name, color) in enumerate(zip(names, colors)):
            axes[k, 0].plot(t, d["V"][:, k], color=color)
            axes[k, 0].axhline(0, color="black", linestyle="--", linewidth=.8)
            axes[k, 0].set(title=f"{name}: BRT", ylabel="V")
            axes[k, 1].plot(t, c["residual"][:, 0, k], color=color, label="Stage 0")
            axes[k, 1].plot(t, summary["min_residual"][:, k], color=color,
                            linestyle="--", label="Minimum over HJ stages")
            axes[k, 1].axhline(0, color="black", linestyle=":")
            axes[k, 1].set_title("HJ residual")
            axes[k, 2].plot(t, c["slack"][:, 0, k], color=color, label="Stage 0")
            axes[k, 2].plot(t, summary["max_slack"][:, k], color=color,
                            linestyle="--", label="Maximum over HJ stages")
            axes[k, 2].set_title("Lower slack")
            for ax in axes[k]:
                _shade(ax, t, d["active"][:, 0, k])
        _style(axes)
        write(pdf, fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
        for axis, key in enumerate(("steering_rate", "acceleration")):
            ax = axes[axis]
            bound = limits[key]
            for k, (name, color) in enumerate(zip(names, colors)):
                low = bound["metric_lower"][:, 0, k]
                high = bound["metric_upper"][:, 0, k]
                ax.plot(t, np.where(np.isfinite(low), low, np.nan),
                        color=color, linestyle="--", label=f"{name}: lower")
                ax.plot(t, np.where(np.isfinite(high), high, np.nan),
                        color=color, label=f"{name}: upper")
            ax.fill_between(t, bound["lower"][:, 0], bound["upper"][:, 0],
                            where=bound["feasible"][:, 0], color="gray", alpha=.18,
                            label="Joint conditional interval")
            ax.step(t, u[:, axis], where="post", color="black", linewidth=1.7,
                    label="Applied command")
            span = diagnostics.u_upper[axis] - diagnostics.u_lower[axis]
            ax.set_ylim(diagnostics.u_lower[axis] - .1 * span,
                        diagnostics.u_upper[axis] + .1 * span)
            ax.set_ylabel("rad/s" if axis == 0 else "m/s²")
            ax.set_title(f"{key.replace('_', ' ').title()} limits | other command fixed to MPC | no slack")
        _style(axes)
        for ax in axes:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
        write(pdf, fig)

        for k, (name, color) in enumerate(zip(names, colors)):
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
            for key, label in (("steering", "Steering"), ("acceleration", "Acceleration"),
                               ("drift_and_worst_human", "Drift + worst human"),
                               ("relaxation", "alpha V")):
                axes[0].plot(t, c[key][:, 0, k], label=label)
            axes[0].plot(t, c["residual"][:, 0, k], color="black", linewidth=1.8, label="Sum / residual")
            axes[0].axhline(0, color="gray", linewidth=.8)
            axes[0].set_title(f"{name}: HJ contributions, stage 0")
            axes[1].plot(t, d["M"][:, 0, k, 0], color=color, label="M1: steering coefficient")
            axes[1].plot(t, d["M"][:, 0, k, 1], color=color, linestyle="--", label="M2: acceleration coefficient")
            # Disabled coefficients are placeholders, not BRT gradients.
            for line in axes[1].lines:
                line.set_ydata(np.where(d["active"][:, 0, k], line.get_ydata(), np.nan))
            axes[1].set_title("Control coefficients (different physical units)")
            _style(axes)
            write(pdf, fig)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
        for j in range(diagnostics.stages):
            axes[0].step(t, feasibility["constant_speed_feasible"][:, j].astype(int),
                         where="post", label=f"Stage {j}")
            axes[1].step(t, feasibility["braking_required"][:, j].astype(int),
                         where="post", label=f"Stage {j}")
            axes[2].step(t, feasibility["conflict"][:, j].astype(int),
                         where="post", label=f"Stage {j}")
        for ax, title in zip(axes, ("Constant speed feasible (a = 0)",
                                   "Braking required (feasible set exists, no a >= 0)",
                                   "HJ/control bounds conflict (empty feasible set)")):
            ax.set(title=title, yticks=[0, 1], ylim=(-.1, 1.1))
            _shade(ax, t, np.any(d["active"][:, 0], axis=1), label="Any HJ imposed")
        fig.suptitle("Instantaneous HJ + control bounds, without slack; road/state constraints excluded")
        _style(axes)
        write(pdf, fig)

        for index in plane_indices:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
            plot_control_plane(diagnostics, index, ax=axes[0])
            plot_control_plane(diagnostics, index, ax=axes[1], softened=True)
            write(pdf, fig)

        if "simOcpCost" in simulation:
            fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
            ax.plot(t, simulation["simOcpCost"], color="black", linewidth=1.8, label="Total OCP cost")
            for name, values in simulation.get("simCostComponents", {}).items():
                ax.plot(t, values, label=name)
            ax.set_title("Cost over the entire MPC prediction horizon")
            _style([ax])
            write(pdf, fig)
    return path


class _Contours:
    def __init__(self, brts, metrics, colors, ax, grid):
        self.items = []
        for name, color in zip(metrics, colors):
            if name not in brts:
                continue
            brt = brts[name]
            x = np.linspace(brt.lower[0], brt.upper[0], grid[0])
            y = np.linspace(brt.lower[1], brt.upper[1], grid[1])
            xx, yy = np.meshgrid(x, y)
            artist = LineCollection([], colors=[color], linewidths=1, zorder=2)
            ax.add_collection(artist)
            self.items.append((brt, x, y, xx, yy, artist))
        self.lines = [[] for _ in self.items]

    def update(self, state, v_h, recompute):
        if recompute and self.items:
            import contourpy
            for k, (brt, x, y, xx, yy, artist) in enumerate(self.items):
                fixed = np.array([v_h, state[4], state[3]])
                if (np.any(fixed < brt.lower[3:]) or np.any(fixed > brt.upper[3:])):
                    self.lines[k] = []
                    continue
                points = np.empty((xx.size, 6))
                points[:, 0], points[:, 1] = xx.ravel(), yy.ravel()
                points[:, 2] = brt.theta_min + (-state[2] - brt.theta_min) % brt.theta_period
                points[:, 3:] = fixed
                field = np.asarray(brt.value_interpolator(points)).reshape(xx.shape)
                self.lines[k] = (contourpy.contour_generator(x=x, y=y, z=field).lines(0.)
                                 if np.isfinite(field).all() and field.min() <= 0 <= field.max() else [])
        c, s = np.cos(state[2]), np.sin(state[2])
        rotation = np.array([[c, -s], [s, c]])
        for lines, item in zip(self.lines, self.items):
            item[-1].set_segments([line @ rotation.T + [0, state[1]] for line in lines])


def _rectangle(length, width):
    return np.array([[-length/2, -width/2], [length/2, -width/2],
                     [length/2, width/2], [-length/2, width/2]])


def animate_simulation(
    simulation, diagnostics, path="simulation.mp4", *, brts=None,
    fps=VIDEO_FPS, dpi=VIDEO_DPI, preset=VIDEO_PRESET, crf=VIDEO_CRF,
    contour_hz=CONTOUR_HZ, contour_grid=CONTOUR_GRID, threads=2,
):
    """Agg cached-background drawing piped directly to ffmpeg (no PNG files).

    The static panels and labels are drawn once. Only moving artists and
    cursors are redrawn per frame. Camera follows ego longitudinal position;
    road Y coordinates stay global. FPS does not affect MPC integration.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found: install it before exporting the video")
    fps, dpi = float(fps), int(dpi)
    if not np.isfinite(fps) or fps <= 0 or dpi < 1:
        raise ValueError("fps and dpi must be positive")
    if contour_hz < 0 or min(contour_grid) < 2:
        raise ValueError("Invalid contour refresh rate or grid")
    d = diagnostics.arrays()
    c = diagnostics.contributions()
    summary = diagnostics.summary()
    names, colors = diagnostics.metrics, metric_colors(diagnostics.metrics)
    x, u = np.asarray(simulation["simX"]), np.asarray(simulation["simU"])
    dt = float(simulation["dt"])
    duration = (len(x) - 1) * dt
    # Each frame represents a playback interval. No extra terminal hold/frame.
    frame_times = np.arange(max(1, int(np.ceil(duration * fps)))) / fps
    indices = np.minimum(np.rint(frame_times / dt).astype(int), len(x) - 1)
    t = d["time"]
    tx = np.arange(len(x)) * dt
    pixels = (max(2, int(13.6 * dpi) // 2 * 2),
              max(2, int(7.6 * dpi) // 2 * 2))
    fig = Figure(figsize=(pixels[0] / dpi, pixels[1] / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(4, 2, width_ratios=[1.6, 1],
                            left=.065, right=.985, bottom=.07, top=.94,
                            hspace=.65, wspace=.25)
    ax = fig.add_subplot(grid[:, 0])
    panels = [fig.add_subplot(grid[j, 1]) for j in range(4)]
    v_h = float(simulation["v_H"])
    human_x = float(simulation["X_H_initial"]) + v_h * tx
    human_y = float(simulation["Y_H_initial"])
    right = max(32., min(65., float(human_x[0] - x[0, 0]) + 8.))
    ax.set(xlim=(-12, right), ylim=(ROAD_Y_MIN - 1, ROAD_Y_MAX + 1),
           xlabel="X - X_E [m]", ylabel="Global Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=.25)
    for y in (ROAD_Y_MIN, ROAD_Y_MAX):
        ax.axhline(y, color="black", linewidth=1.5)
    ax.axhline(2, color="gray", linestyle="--")
    ax.axhline(0, color="gray", linestyle=":")
    ego = Polygon(_rectangle(EGO_LENGTH, EGO_WIDTH), color="tab:blue", alpha=.7, zorder=5)
    human = Polygon(_rectangle(HUMAN_LENGTH, HUMAN_WIDTH), color="tab:orange", alpha=.7, zorder=5)
    ax.add_patch(ego)
    ax.add_patch(human)
    ego_radius = np.hypot(EGO_LENGTH / 4, EGO_WIDTH / 2)
    human_radius = np.hypot(HUMAN_LENGTH / 4, HUMAN_WIDTH / 2)
    circles = [Circle((0, 0), radius, fill=False, color=color,
                      linestyle="--", linewidth=.8, alpha=.8, zorder=4)
               for radius, color in ((ego_radius, "tab:blue"),) * 2
               + ((human_radius, "tab:orange"),) * 2]
    for circle in circles:
        ax.add_patch(circle)
    heading, = ax.plot([], [], color="black", linewidth=1, zorder=6)
    trajectory, = ax.plot([], [], color="tab:blue", linewidth=1)
    prediction, = ax.plot([], [], color="tab:green", linestyle="--", linewidth=1.5)
    target, = ax.plot([], [], "D", color="tab:green", markersize=4)
    text = ax.text(.01, 1.04, "", transform=ax.transAxes, fontsize=10)
    visible_brts = (brts or {}) if contour_hz > 0 else {}
    contours = _Contours(visible_brts, names, colors, ax, contour_grid)
    for name, color in zip(names, colors):
        if name in visible_brts:
            ax.plot([], [], color=color, label=f"{name}: V=0")
    if contours.items:
        ax.legend(fontsize=7, loc="upper right")
    panels[0].plot(tx, x[:, 3], color="tab:blue")
    panels[0].axhline(float(simulation["v_ref"]), color="black", linestyle="--")
    panels[0].set(title="Velocity", ylabel="m/s")
    panels[1].step(t, u[:, 1], where="post", color="tab:blue")
    panels[1].axhline(0, color="gray", linewidth=.7)
    panels[1].set(title="Acceleration", ylabel="m/s²")
    _metric_series(panels[2], t, d["V"], names, colors)
    panels[2].axhline(0, color="black", linestyle="--", linewidth=.8)
    panels[2].set(title="BRT values (metric-specific units)", ylabel="V")
    _metric_series(panels[3], t, summary["min_residual"], names, colors)
    panels[3].axhline(0, color="black", linestyle="--", linewidth=.8)
    panels[3].set(title="Minimum residual over HJ stages", ylabel="Residual")
    for panel in panels[2:]:
        _shade(panel, t, np.any(d["active"][:, 0], axis=1))
    for panel in panels:
        panel.set_xlim(0, duration)
        panel.tick_params(labelsize=8)
    _style(panels)
    cursors = [panel.axvline(0, color="red", linewidth=1) for panel in panels]
    dynamic = [ego, human, *circles, heading, trajectory, prediction, target, text, *cursors,
               *[item[-1] for item in contours.items]]
    for artist in dynamic:
        artist.set_animated(True)
    # Fix the layout and cache the complete static figure exactly once.
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    width, height = canvas.get_width_height()
    if width % 2 or height % 2:
        raise ValueError("Video pixel dimensions must be even; use an even dpi")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(suffix=".mp4", dir=path.parent)
    import os
    os.close(handle)
    temporary_path = Path(temporary_name)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-threads", str(threads), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(temporary_path)]
    last_contour_time = -np.inf
    started = time.perf_counter()
    try:
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=log)
            try:
                for playback_time, index in zip(frame_times, indices):
                    state = x[index]
                    cos, sin = np.cos(state[2]), np.sin(state[2])
                    rotation = np.array([[cos, -sin], [sin, cos]])
                    ego.set_xy(_rectangle(EGO_LENGTH, EGO_WIDTH) @ rotation.T + [0, state[1]])
                    human.set_xy(_rectangle(HUMAN_LENGTH, HUMAN_WIDTH) +
                                 [human_x[index] - state[0], human_y])
                    for sign, circle in zip((1, -1), circles[:2]):
                        circle.center = (sign * EGO_LENGTH / 4 * cos,
                                         state[1] + sign * EGO_LENGTH / 4 * sin)
                    for sign, circle in zip((1, -1), circles[2:]):
                        circle.center = (human_x[index] - state[0] + sign * HUMAN_LENGTH / 4,
                                         human_y)
                    heading.set_data([0, EGO_LENGTH / 2 * cos],
                                     [state[1], state[1] + EGO_LENGTH / 2 * sin])
                    trajectory.set_data(x[:index+1, 0] - state[0], x[:index+1, 1])
                    pred = simulation.get("simX_pred")
                    if pred is not None and index < len(pred):
                        p = pred[index]
                        prediction.set_data(p[:, 0] - state[0], p[:, 1])
                        target.set_data([p[-1, 0] - state[0]], [p[-1, 1]])
                    else:
                        prediction.set_data([], [])
                        target.set_data([], [])
                    refresh = contour_hz > 0 and playback_time - last_contour_time >= 1 / contour_hz - 1e-10
                    if refresh:
                        last_contour_time = playback_time
                    contours.update(state, v_h, refresh)
                    active_names = [name for k, name in enumerate(names)
                                    if d["active"][min(index, len(t)-1), 0, k]]
                    text.set_text(f"t = {index*dt:.2f} s | v_E = {state[3]:.2f} m/s | v_H = {v_h:.2f} m/s\n"
                                  + "HJ: " + (", ".join(active_names) or "outside BRT grids"))
                    for cursor in cursors:
                        cursor.set_xdata([playback_time, playback_time])
                    canvas.restore_region(background)
                    for artist in sorted(dynamic, key=lambda a: a.get_zorder()):
                        artist.axes.draw_artist(artist)
                    process.stdin.write(canvas.buffer_rgba())
            except BaseException:
                process.kill()
                process.wait()
                log.seek(0)
                message = log.read().decode(errors="replace")
                if message:
                    print(message)
                raise
            finally:
                process.stdin.close()
            if process.wait() != 0:
                log.seek(0)
                raise RuntimeError("ffmpeg export failed: " + log.read().decode(errors="replace"))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
        fig.clear()
    print(f"Video: {len(indices)} frames, {fps:g} fps, "
          f"{time.perf_counter()-started:.1f} s export time")
    return path


def _validate(simulation, diagnostics):
    d = diagnostics.arrays()
    count = len(d["time"])
    if np.asarray(simulation["simX"]).shape != (count + 1, 5):
        raise ValueError("Simulation states and diagnostics do not match")
    if np.asarray(simulation["simU"]).shape != (count, 2):
        raise ValueError("Simulation controls and diagnostics do not match")
    if not np.isclose(float(simulation["dt"]), diagnostics.dt):
        raise ValueError("Simulation and diagnostic dt differ")
    if not np.allclose(d["time"], np.arange(count) * diagnostics.dt):
        raise ValueError("Expected uniformly sampled records starting at zero")
    if not np.allclose(d["state"], simulation["simX"][:-1], equal_nan=False):
        raise ValueError("Simulation and diagnostic ego states differ")
    if not np.allclose(d["u"][:, 0], simulation["simU"]):
        raise ValueError("Applied commands and stage-0 diagnostic commands differ")
    residual = diagnostics.contributions()["residual"]
    if "simHJResidual" in simulation and not np.allclose(
        residual[:, 0], simulation["simHJResidual"], rtol=1e-8, atol=1e-10, equal_nan=True,
    ):
        raise ValueError("Main and diagnostic HJ residuals differ")
    return d


def export_results(simulation, diagnostics, *, output_dir=None, make_plots=True,
                   make_animation=True, brts=None, video_options=None):
    """Validate, save data, then write grafici.pdf and simulation.mp4 together."""
    d = _validate(simulation, diagnostics)
    c = diagnostics.contributions()
    if output_dir is None:
        run = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + "_".join(diagnostics.metrics)
        output_dir = Path(__file__).resolve().parent / "results" / run
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diagnostics.save(output_dir / "hj.npz")
    payload = {name: np.asarray(simulation[name]) for name in (
        "simX", "simU", "dt", "v_ref", "X_H_initial", "Y_H_initial", "v_H",
        "simX_pred", "simOcpCost", "solve_time",
    ) if simulation.get(name) is not None}
    payload["metrics"] = np.asarray(diagnostics.metrics)
    for name, values in simulation.get("simCostComponents", {}).items():
        payload[f"cost__{name}"] = values
    np.savez_compressed(output_dir / "simulation.npz", **payload)
    print("Residui HJ registrati correttamente: True")
    for k, name in enumerate(diagnostics.metrics):
        count = np.count_nonzero(d["active"][:, :, k])
        violations = np.count_nonzero(c["soft_violated"][:, :, k])
        print(f"{name}: {count} condizioni attive, {violations} violazioni oltre tolleranza")
    cost_parts = simulation.get("simCostComponents")
    if cost_parts and simulation.get("simOcpCost") is not None:
        error = np.max(np.abs(sum(cost_parts.values()) - simulation["simOcpCost"]))
        print(f"Maximum cost decomposition error: {error:.6e}")
    if simulation.get("solve_time") is not None:
        elapsed = np.asarray(simulation["solve_time"])
        print(f"OCP solve time: mean {1e3*elapsed.mean():.2f} ms, max {1e3*elapsed.max():.2f} ms")
    files = dict(diagnostics_path=diagnostics_path,
                 simulation_path=output_dir / "simulation.npz")
    if make_plots:
        files["plots_path"] = save_plots(simulation, diagnostics, output_dir / "grafici.pdf")
    if make_animation:
        files["animation_path"] = animate_simulation(
            simulation, diagnostics, output_dir / "simulation.mp4",
            brts=brts, **(video_options or {}),
        )
    print(f"Risultati: {output_dir}")
    return files


def load_results(diagnostics_path, simulation_path=None):
    diagnostics_path = Path(diagnostics_path)
    if simulation_path is None:
        simulation_path = (diagnostics_path.with_name(
            diagnostics_path.name.removesuffix("_hj.npz") + "_simulation.npz")
            if diagnostics_path.name.endswith("_hj.npz") else
            diagnostics_path.with_name("simulation.npz"))
    diagnostics = HJDiagnostics.load(diagnostics_path)
    with np.load(simulation_path, allow_pickle=False) as archive:
        simulation = {key: archive[key].copy() for key in archive.files}
    simulation["simCostComponents"] = {
        key.removeprefix("cost__"): value
        for key, value in simulation.items() if key.startswith("cost__")
    }
    _validate(simulation, diagnostics)
    return simulation, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--fps", type=float, default=VIDEO_FPS)
    parser.add_argument("--dpi", type=int, default=VIDEO_DPI)
    parser.add_argument("--contour-hz", type=float, default=CONTOUR_HZ)
    parser.add_argument("--brt-dir", type=Path)
    args = parser.parse_args()
    simulation, diagnostics = load_results(args.diagnostics, args.simulation)
    folder = args.output_dir or args.diagnostics.parent
    folder.mkdir(parents=True, exist_ok=True)
    if not args.no_pdf:
        save_plots(simulation, diagnostics, folder / "grafici.pdf")
        print(f"PDF: {folder / 'grafici.pdf'}")
    if args.video:
        brts = {}
        if args.brt_dir:
            from brt_utils import BRTinterpolator
            brts = {name: BRTinterpolator(args.brt_dir / f"{name}.npz")
                    for name in diagnostics.metrics}
        animate_simulation(simulation, diagnostics, folder / "simulation.mp4",
                           brts=brts, fps=args.fps, dpi=args.dpi,
                           contour_hz=args.contour_hz)


if __name__ == "__main__":
    main()



