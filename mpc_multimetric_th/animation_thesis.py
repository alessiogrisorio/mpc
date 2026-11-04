"""Thesis graphics: grafici.pdf and animation.pdf.

Keep the original animation.py and hj_diagnostics.py alongside this file.

Offline regeneration:
    python animation_thesis.py results/RUN/hj.npz
"""

from pathlib import Path
import argparse
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from animation import (
    COST_GROUPS,
    OcpCostDiagnostics,
    export_results as _save_original_results,
    load_results,
    metric_colors,
)


# ============================================================
# SETTINGS
# ============================================================

FONT_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 16
TIME_SIZE = 16
COLORBAR_SIZE = 16

FIGSIZE = (9, 5)
STATES_FIGSIZE = (9, 13)
DOUBLE_FIGSIZE = (10, 8)

# Additional width accommodates legends outside the plotting area.
LIMITS_FIGSIZE = (13, 8)
PLANE_FIGSIZE = (12, 6.5)
FRAME_FIGSIZE = (12, 4.6)
HJ_TIME_PADDING = 1.0

LINE_WIDTH = 1.8
COMMAND_WIDTH = 2.0
BOUND_WIDTH = 2.0
MARKER_SIZE = 28

FRAME_FPS = 2.0
VIEW_WIDTH = 30.0
VIEW_Y_MIN = -2.0
VIEW_Y_MAX = 6.0

SHOW_PREDICTION = True
SHOW_BRT_CONTOURS = False
CONTOUR_GRID = (70, 50)


# None: first, middle and last sample with an active HJ constraint.
# Example: PLANE_TIMES = [2.0, 4.0, 6.0]
PLANE_TIMES = None

SHOW_RELAXED_BOUNDARIES = True
SLACK_DISPLAY_TOL = 1e-7

EGO_LENGTH, EGO_WIDTH = 4.68, 2.20
HUMAN_LENGTH, HUMAN_WIDTH = 4.28, 1.80

ROAD_Y_MIN, ROAD_Y_MAX = -2.0, 6.0
LANE_DIVIDER = 2.0

METRIC_LABELS = {
    "euclidean": "Euclidean",
    "dce": "DCE",
    "ttc": "TTC",
    "rss": "RSS",
    "btn": "BTN",
    "stn": "STN",
    "long_lat": "min(BTN, STN)",
    "eggert": "Eggert",
    "time_eggert": "Time Eggert",
    "mixed": "Mixed",
}

STYLE = {
    "font.family": "serif",
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FONT_SIZE,
    "lines.linewidth": LINE_WIDTH,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.8",
    "pdf.fonttype": 42,
}


# ============================================================
# COMMON HELPERS
# ============================================================

def metric_label(name):
    return METRIC_LABELS.get(name, name)


def scalar(simulation, key):
    return float(np.asarray(simulation[key]))


def finish(ax, *, time_axis=True, legend=True, outside=False):
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)

    if time_axis:
        ax.set_xlabel("Time [s]")

    handles, labels = ax.get_legend_handles_labels()
    if legend and handles:
        options = {
            "frameon": True,
            "facecolor": "white",
            "framealpha": 1.0,
            "fontsize": LEGEND_SIZE,
        }
        if outside:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
                **options,
            )
        else:
            ax.legend(loc="best", **options)


def set_time_domain(ax, t):
    # Same exact time interval for BRT and control-limit plots.
    if len(t) > 1:
        ax.set_xlim(float(t[0]), float(t[-1]))
    else:
        ax.set_xlim(float(t[0]), float(t[0]) + 0.02)

def set_hj_time_domain(ax, diagnostics):
    """Common time window for all BRT and conditional control plots."""
    d = diagnostics.arrays()
    t = np.asarray(d["time"])

    # Stage 0: constraints associated with the applied command.
    active = np.any(d["active"][:, 0, :], axis=1)
    indices = np.flatnonzero(active)

    if len(indices) == 0:
        set_time_domain(ax, t)
        return

    first = int(indices[0])
    last = int(indices[-1])

    # First sample after the final active interval.
    switch_off = (
        t[last + 1]
        if last + 1 < len(t)
        else t[last] + diagnostics.dt
    )

    start = max(t[0], t[first] - HJ_TIME_PADDING)
    stop = min(t[-1], switch_off + HJ_TIME_PADDING)

    if stop <= start:
        set_time_domain(ax, t)
    else:
        ax.set_xlim(float(start), float(stop))


def write_page(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def road(ax):
    for y in (ROAD_Y_MIN, ROAD_Y_MAX):
        ax.axhline(y, color="black", linewidth=1.2)

    ax.axhline(
        LANE_DIVIDER,
        color="0.45",
        linestyle="--",
        linewidth=0.9,
    )


def vehicle(ax, state, length, width, color, label=None):
    X, Y, psi = state[:3]

    corners = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2],
    ])

    c, s = np.cos(psi), np.sin(psi)
    rotation = np.array([[c, -s], [s, c]])
    vertices = corners @ rotation.T + [X, Y]

    ax.add_patch(Polygon(
        vertices,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.75,
        label=label,
        zorder=5,
    ))


def human_state(simulation, time):
    return np.array([
        scalar(simulation, "X_H_initial")
        + scalar(simulation, "v_H") * time,
        scalar(simulation, "Y_H_initial"),
        0.0,
    ])


def header_figure(figsize, height=0.85):
    """Compact header above the plot; dimensions are in inches."""
    fig = plt.figure(figsize=figsize)

    fw, fh = fig.get_size_inches()

    left = 0.85
    right = 0.20
    bottom = 0.65
    top = 0.12
    gap = 0.05

    width = fw - left - right
    plot_height = fh - top - height - gap - bottom

    if plot_height <= 0:
        raise ValueError("Figure too short for the requested header")

    header = fig.add_axes([
        left / fw,
        (fh - top - height) / fh,
        width / fw,
        height / fh,
    ])
    header.set_axis_off()

    ax = fig.add_axes([
        left / fw,
        bottom / fh,
        width / fw,
        plot_height / fh,
    ])

    # With equal aspect, keep the plot close to the header.
    ax.set_anchor("N")

    return fig, header, ax


def header_legend(header, ax, *, ncol=2):
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        header.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0,
            ncol=ncol,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            fontsize=LEGEND_SIZE,
            columnspacing=1.2,
            handletextpad=0.5,
            labelspacing=0.35,
            borderpad=0.4,
        )


# ============================================================
# CONTROL LIMITS
# ============================================================

def draw_limits(axes, t, u, diagnostics, limits, indices, colors):
    """Stage-0 bounds conditional on the other MPC control; no slack."""
    for axis, key, ylabel in (
        (0, "steering_rate", r"$\dot{\delta}_E$ [rad/s]"),
        (1, "acceleration", r"$a_E$ [m/s$^2$]"),
    ):
        ax = axes[axis]
        bound = limits[key]

        ax.step(
            t,
            u[:, axis],
            where="post",
            color="black",
            linewidth=COMMAND_WIDTH,
            label="MPC command",
            zorder=3,
        )

        lower_physical = diagnostics.u_lower[axis]
        upper_physical = diagnostics.u_upper[axis]

        for n, value in enumerate((lower_physical, upper_physical)):
            ax.axhline(
                value,
                color="0.55",
                linewidth=0.9,
                label="Physical bounds" if n == 0 else None,
            )

        violated = np.zeros(len(t), dtype=bool)

        for k in indices:
            name = metric_label(diagnostics.metrics[k])
            lower = bound["metric_lower"][:, 0, k]
            upper = bound["metric_upper"][:, 0, k]

            for values, pattern, side in (
                (lower, "--", "lower"),
                (upper, ":", "upper"),
            ):
                visible = np.where(
                    np.isfinite(values), values, np.nan
                )

                # Skip bounds that do not exist for this metric.
                if not np.any(np.isfinite(visible)):
                    continue

                ax.step(
                    t,
                    visible,
                    where="post",
                    color=colors[k],
                    linestyle=pattern,
                    linewidth=BOUND_WIDTH,
                    label=f"{name}: {side}",
                    zorder=5,
                )

            violated |= (
                (u[:, axis] < lower - diagnostics.tolerance)
                | (u[:, axis] > upper + diagnostics.tolerance)
            )

        if np.any(violated):
            ax.scatter(
                t[violated],
                u[violated, axis],
                color="red",
                s=MARKER_SIZE / 2,
                label="HJ bound violated",
                zorder=6,
            )

        span = upper_physical - lower_physical
        ax.set_ylim(
            lower_physical - 0.1 * span,
            upper_physical + 0.1 * span,
        )

        ax.set_ylabel(ylabel)
        set_hj_time_domain(ax, diagnostics)
        finish(ax, outside=True)


# ============================================================
# COMMAND PLANE
# ============================================================

def command_plane(diagnostics, index, colors):
    d = diagnostics.arrays()

    fig, header, ax = header_figure(
        PLANE_FIGSIZE, height=1.25
    )

    lo = diagnostics.u_lower
    hi = diagnostics.u_upper

    ax.add_patch(Polygon(
        [
            [lo[0], lo[1]],
            [hi[0], lo[1]],
            [hi[0], hi[1]],
            [lo[0], hi[1]],
        ],
        fill=False,
        edgecolor="0.5",
        linewidth=1.0,
        label="Physical bounds",
    ))

    polygon = diagnostics.control_polygon(
        index, 0, softened=False
    )

    empty = len(polygon) == 0

    if len(polygon) >= 3:
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            color="tab:green",
            alpha=0.18,
            label="Joint HJ feasible set",
            zorder=1,
        )
    elif len(polygon):
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="tab:green",
            marker="o",
            label="Joint HJ feasible set",
        )

    xx, yy = np.meshgrid(
        np.linspace(lo[0], hi[0], 100),
        np.linspace(lo[1], hi[1], 100),
    )

    for k, name in enumerate(diagnostics.metrics):
        if not d["active"][index, 0, k]:
            continue

        M = d["M"][index, 0, k]
        b = d["b"][index, 0, k]
        field = M[0] * xx + M[1] * yy + b

        if field.min() < 0 < field.max():
            ax.contour(
                xx, yy, field,
                levels=[0],
                colors=[colors[k]],
                linewidths=BOUND_WIDTH,
            )
            ax.plot(
                [], [],
                color=colors[k],
                label=metric_label(name),
            )
        elif field.max() < 0:
            ax.plot(
                [], [],
                color=colors[k],
                label=f"{metric_label(name)}: infeasible",
            )

        slack = d["slack"][index, 0, k]

        if (
            SHOW_RELAXED_BOUNDARIES
            and slack > SLACK_DISPLAY_TOL
        ):
            relaxed = field + slack

            if relaxed.min() < 0 < relaxed.max():
                ax.contour(
                    xx, yy, relaxed,
                    levels=[0],
                    colors=[colors[k]],
                    linestyles="--",
                    linewidths=BOUND_WIDTH,
                )
                ax.plot(
                    [], [],
                    color=colors[k],
                    linestyle="--",
                    label=f"{metric_label(name)}: with slack",
                )

    command = d["u"][index, 0]

    ax.scatter(
        command[0],
        command[1],
        color="black",
        s=MARKER_SIZE * 2,
        label="MPC command",
        zorder=8,
    )

    ax.set(
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        xlabel=r"$\dot{\delta}_E$ [rad/s]",
        ylabel=r"$a_E$ [m/s$^2$]",
    )

    message = f"t = {d['time'][index]:.2f} s"

    if empty:
        message += "\nEmpty original\nHJ feasible set"

    header.text(
        0.0,
        0.5,
        message,
        transform=header.transAxes,
        va="center",
        ha="left",
        fontsize=TIME_SIZE,
        linespacing=1.2,
    )

    header_legend(header, ax, ncol=2)
    finish(ax, time_axis=False, legend=False)

    ax.set_xlim(
        diagnostics.u_lower[0],
        diagnostics.u_upper[0],
    )
    ax.set_ylim(
        diagnostics.u_lower[1],
        diagnostics.u_upper[1],
    )

    return fig


# ============================================================
# STATIC PLOTS: grafici.pdf
# ============================================================

def save_plots(simulation, diagnostics, path, plane_times=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    d = diagnostics.arrays()
    c = diagnostics.contributions()
    summary = diagnostics.summary()
    limits = diagnostics.compute_limits(softened=False)

    x = np.asarray(simulation["simX"])
    u = np.asarray(simulation["simU"])

    t = d["time"]
    tx = np.arange(len(x)) * scalar(simulation, "dt")

    colors = metric_colors(diagnostics.metrics)
    all_indices = list(range(len(diagnostics.metrics)))

    with plt.rc_context(STYLE), PdfPages(path) as pdf:

        # 1. Colored trajectory: line and colorbar only.
        fig, ax = plt.subplots(
            figsize=FIGSIZE,
            layout="constrained",
        )

        points = x[:, :2]
        segments = np.stack(
            [points[:-1], points[1:]], axis=1
        )

        collection = LineCollection(
            segments,
            cmap="viridis",
            norm=Normalize(vmin=tx[0], vmax=tx[-1]),
            linewidths=2.8,
            zorder=3,
        )

        collection.set_array(
            (tx[:-1] + tx[1:]) / 2
        )

        ax.add_collection(collection)
        ax.update_datalim(points)
        ax.autoscale_view()
        ax.margins(x=0.03, y=0.12)

        # Independent axis scales: Y is visually expanded.
        ax.set_aspect("auto")
        ax.set(xlabel="X [m]", ylabel="Y [m]")

        bar = fig.colorbar(
            collection, ax=ax, pad=0.02
        )
        bar.set_label(
            "Time [s]",
            fontsize=COLORBAR_SIZE,
        )
        bar.ax.tick_params(labelsize=TICK_SIZE)

        finish(
            ax, time_axis=False, legend=False
        )
        write_page(pdf, fig)

        # 2. Ego states.
        fig, axes = plt.subplots(
            5, 1,
            sharex=True,
            figsize=STATES_FIGSIZE,
            layout="constrained",
        )

        labels = [
            r"$X_E$ [m]",
            r"$Y_E$ [m]",
            r"$\theta_E$ [rad]",
            r"$v_E$ [m/s]",
            r"$\delta_E$ [rad]",
        ]

        for j, ax in enumerate(axes):
            ax.plot(
                tx, x[:, j],
                color="tab:blue",
                label="Ego",
            )
            ax.set_ylabel(labels[j])

            if j == 3:
                ax.axhline(
                    scalar(simulation, "v_ref"),
                    color="black",
                    linestyle="--",
                    label="Reference",
                )

            set_time_domain(ax, tx)
            finish(ax)

        write_page(pdf, fig)

        # 3. Applied commands.
        fig, axes = plt.subplots(
            2, 1,
            sharex=True,
            figsize=DOUBLE_FIGSIZE,
            layout="constrained",
        )

        for axis, ax in enumerate(axes):
            ax.step(
                t,
                u[:, axis],
                where="post",
                color="black",
                label="MPC command",
            )

            for n, value in enumerate((
                diagnostics.u_lower[axis],
                diagnostics.u_upper[axis],
            )):
                ax.axhline(
                    value,
                    color="0.55",
                    linestyle="--",
                    label="Physical bounds" if n == 0 else None,
                )

            ax.set_ylabel(
                r"$\dot{\delta}_E$ [rad/s]"
                if axis == 0
                else r"$a_E$ [m/s$^2$]"
            )

            set_time_domain(ax, t)
            finish(ax)

        write_page(pdf, fig)

        # 4. Combined BRT and one page per metric.
        # The two pages are intentionally identical for one metric.
        groups = [all_indices] + [
            [k] for k in all_indices
        ]

        for indices in groups:
            fig, ax = plt.subplots(
                figsize=FIGSIZE,
                layout="constrained",
            )

            for k in indices:
                ax.plot(
                    t,
                    d["V"][:, k],
                    color=colors[k],
                    label=metric_label(
                        diagnostics.metrics[k]
                    ),
                )

            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=0.9,
            )

            ax.set_ylabel("V")
            set_time_domain(ax, t)
            finish(ax)
            write_page(pdf, fig)

        # 5. Combined and individual control limits.
        for indices in groups:
            fig, axes = plt.subplots(
                2, 1,
                sharex=True,
                figsize=LIMITS_FIGSIZE,
                layout="constrained",
            )

            draw_limits(
                axes, t, u,
                diagnostics, limits,
                indices, colors,
            )

            write_page(pdf, fig)

        # 6. Cost over the entire optimized prediction horizon.
        if simulation.get("simOcpCost") is not None:
            fig, axes = plt.subplots(
                2, 1,
                sharex=True,
                figsize=LIMITS_FIGSIZE,
                layout="constrained",
            )

            axes[0].plot(
                t,
                simulation["simOcpCost"],
                color="black",
                label="Total OCP cost",
            )
            axes[0].set_ylabel("OCP cost")

            for name, values in simulation.get(
                "simCostComponents", {}
            ).items():
                axes[1].plot(
                    t, values, label=name
                )

            axes[1].set_ylabel("Cost components")

            for ax in axes:
                set_time_domain(ax, t)
                finish(ax, outside=True)

            write_page(pdf, fig)

        # 7. Residual and slack, per metric.
        for k, name in enumerate(diagnostics.metrics):
            entries = (
                (
                    c["residual"][:, 0, k],
                    summary["min_residual"][:, k],
                    "HJ residual",
                    "Minimum over HJ stages",
                ),
                (
                    c["slack"][:, 0, k],
                    summary["max_slack"][:, k],
                    "HJ slack",
                    "Maximum over HJ stages",
                ),
            )

            for values, extreme, ylabel, extreme_label in entries:
                fig, ax = plt.subplots(
                    figsize=FIGSIZE,
                    layout="constrained",
                )

                ax.plot(
                    t, values,
                    color=colors[k],
                    label=f"{metric_label(name)}: stage 0",
                )

                ax.plot(
                    t, extreme,
                    color=colors[k],
                    linestyle="--",
                    label=extreme_label,
                )

                ax.axhline(
                    0, color="0.4", linewidth=0.8
                )
                ax.set_ylabel(ylabel)

                set_time_domain(ax, t)
                finish(ax)
                write_page(pdf, fig)

        # 8. Command planes.
        requested = (
            PLANE_TIMES
            if plane_times is None
            else plane_times
        )

        if requested is None:
            active = np.flatnonzero(
                np.any(d["active"][:, 0], axis=1)
            )

            indices = (
                active[[0, len(active) // 2, -1]]
                if len(active)
                else [0]
            )
        else:
            indices = [
                int(np.argmin(np.abs(t - value)))
                for value in requested
            ]

        for index in dict.fromkeys(map(int, indices)):
            write_page(
                pdf,
                command_plane(
                    diagnostics, index, colors
                ),
            )

    return path


# ============================================================
# BRT CONTOURS IN GLOBAL COORDINATES
# ============================================================

def draw_brt_contours(
    ax, brts, state, human_speed, colors, names
):
    nx, ny = CONTOUR_GRID

    for k, name in enumerate(names):
        if name not in brts:
            continue

        brt = brts[name]
        fixed = np.array([
            human_speed, state[4], state[3]
        ])

        if (
            np.any(fixed < brt.lower[3:])
            or np.any(fixed > brt.upper[3:])
        ):
            continue

        xr = np.linspace(
            brt.lower[0], brt.upper[0], nx
        )
        yr = np.linspace(
            brt.lower[1], brt.upper[1], ny
        )

        xx, yy = np.meshgrid(xr, yr)
        points = np.empty((xx.size, 6))

        points[:, :2] = np.column_stack([
            xx.ravel(), yy.ravel()
        ])

        points[:, 2] = (
            brt.theta_min
            + (-state[2] - brt.theta_min)
            % brt.theta_period
        )
        points[:, 3:] = fixed

        values = np.asarray(
            brt.value_interpolator(points)
        ).reshape(xx.shape)

        if not (
            np.isfinite(values).all()
            and values.min() < 0 < values.max()
        ):
            continue

        c, s = np.cos(state[2]), np.sin(state[2])
        X = state[0] + c * xx - s * yy
        Y = state[1] + s * xx + c * yy

        ax.contour(
            X, Y, values,
            levels=[0],
            colors=[colors[k]],
            linewidths=1.2,
        )

        ax.plot(
            [], [],
            color=colors[k],
            label=f"{metric_label(name)}: V = 0",
        )


# ============================================================
# ANIMATION FRAMES: animation.pdf
# ============================================================

def save_animation_pdf(
    simulation,
    diagnostics,
    path,
    *,
    brts=None,
    fps=None,
):
    fps = FRAME_FPS if fps is None else float(fps)

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            "FRAME_FPS must be finite and positive"
        )

    if (
        not np.isfinite([
            VIEW_WIDTH, VIEW_Y_MIN, VIEW_Y_MAX
        ]).all()
        or VIEW_WIDTH <= 0
        or VIEW_Y_MIN >= VIEW_Y_MAX
    ):
        raise ValueError("Invalid camera dimensions")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(simulation["simX"])
    dt = scalar(simulation, "dt")

    tx = np.arange(len(x)) * dt
    duration = tx[-1]

    if duration <= 0:
        raise ValueError(
            "Simulation must have positive duration"
        )

    frame_times = np.arange(
        0.0, duration, 1.0 / fps
    )
    frame_times = np.r_[frame_times, duration]

    colors = metric_colors(diagnostics.metrics)
    predictions = simulation.get("simX_pred")
    d = diagnostics.arrays()

    unwrapped_heading = np.unwrap(x[:, 2])

    with plt.rc_context(STYLE), PdfPages(path) as pdf:
        for frame_time in frame_times:

            state = np.array([
                np.interp(
                    frame_time,
                    tx,
                    unwrapped_heading if j == 2 else x[:, j],
                )
                for j in range(x.shape[1])
            ])

            index = min(
                int(np.searchsorted(
                    tx, frame_time, side="right"
                ) - 1),
                len(x) - 1,
            )

            # Active metrics refer to the most recent recorded
            # MPC solution, held until the following control step.
            diagnostic_index = min(
                max(
                    int(np.searchsorted(
                        d["time"],
                        frame_time,
                        side="right",
                    ) - 1),
                    0,
                ),
                len(d["time"]) - 1,
            )

            active_names = [
                name
                for k, name in enumerate(diagnostics.metrics)
                if d["active"][diagnostic_index, 0, k]
            ]

            metrics_text = (
                "metrics: " + ", ".join(active_names)
                if active_names
                else "metrics: none"
            )

            metric_lines = textwrap.wrap(
                metrics_text,
                width=34,
                break_long_words=False,
                break_on_hyphens=False,
            )

            header_height = (
                0.85 + 0.24 * max(0, len(metric_lines) - 2)
            )

            # Grow the page only when the metric list requires more space.
            frame_size = (
                FRAME_FIGSIZE[0],
                FRAME_FIGSIZE[1] + header_height - 0.85,
            )

            fig, header, ax = header_figure(
                frame_size,
                height=header_height,
            )

            fig, header, ax = header_figure(
                FRAME_FIGSIZE,
                height=header_height,
            )

            road(ax)

            past = x[:index + 1, :2]

            ax.plot(
                np.r_[past[:, 0], state[0]],
                np.r_[past[:, 1], state[1]],
                color="tab:blue",
                linewidth=1.2,
                label="Ego trajectory",
            )

            vehicle(
                ax, state,
                EGO_LENGTH, EGO_WIDTH,
                "tab:blue", "Ego",
            )

            vehicle(
                ax,
                human_state(simulation, frame_time),
                HUMAN_LENGTH, HUMAN_WIDTH,
                "tab:orange", "Human",
            )

            if (
                SHOW_PREDICTION
                and predictions is not None
                and index < len(predictions)
            ):
                prediction = np.asarray(
                    predictions[index]
                )

                ax.plot(
                    prediction[:, 0],
                    prediction[:, 1],
                    color="tab:green",
                    linestyle="--",
                    label="MPC prediction",
                )

            if SHOW_BRT_CONTOURS and brts:
                draw_brt_contours(
                    ax, brts, state,
                    scalar(simulation, "v_H"),
                    colors,
                    diagnostics.metrics,
                )

            # Absolute coordinates; ego remains horizontally centred.
            ax.set(
                xlim=(
                    state[0] - VIEW_WIDTH / 2,
                    state[0] + VIEW_WIDTH / 2,
                ),
                ylim=(VIEW_Y_MIN, VIEW_Y_MAX),
                xlabel="X [m]",
                ylabel="Y [m]",
            )

            ax.set_aspect("equal", adjustable="box")
            ax.ticklabel_format(
                axis="x", style="plain", useOffset=False
            )

            header.text(
                0.0,
                0.5,
                f"t = {frame_time:.2f} s\n" + "\n".join(metric_lines),
                transform=header.transAxes,
                va="center",
                ha="left",
                fontsize=FONT_SIZE,
                linespacing=1.2,
            )

            header_legend(header, ax, ncol=2)

            write_page(pdf, fig)

    return path


# ============================================================
# MAIN COMPATIBILITY
# ============================================================

def export_results(
    simulation,
    diagnostics,
    *,
    output_dir=None,
    make_plots=True,
    make_animation=True,
    brts=None,
    video_options=None,
):
    # Original validation and NPZ persistence.
    # Disable the original PDF/MP4 generation.
    files = _save_original_results(
        simulation,
        diagnostics,
        output_dir=output_dir,
        make_plots=False,
        make_animation=False,
    )

    folder = Path(files["simulation_path"]).parent

    if make_plots:
        files["plots_path"] = save_plots(
            simulation,
            diagnostics,
            folder / "grafici.pdf",
        )
        print(f"Plots: {files['plots_path']}")

    if make_animation:
        files["animation_path"] = save_animation_pdf(
            simulation,
            diagnostics,
            folder / "animation.pdf",
            brts=brts,
            fps=(video_options or {}).get("fps"),
        )
        print(f"Frames: {files['animation_path']}")

    return files


# ============================================================
# OFFLINE PDF REGENERATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "diagnostics", type=Path
    )
    parser.add_argument(
        "--simulation", type=Path
    )
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument(
        "--brt-dir", type=Path
    )
    parser.add_argument(
        "--fps", type=float, default=FRAME_FPS
    )
    parser.add_argument(
        "--no-animation", action="store_true"
    )
    parser.add_argument(
        "--no-plots", action="store_true"
    )

    args = parser.parse_args()

    simulation, diagnostics = load_results(
        args.diagnostics,
        args.simulation,
    )

    folder = (
        args.output_dir
        or args.diagnostics.parent
    )
    folder.mkdir(parents=True, exist_ok=True)

    brts = {}

    if (
        SHOW_BRT_CONTOURS
        and args.brt_dir is not None
        and not args.no_animation
    ):
        from brt_utils import BRTinterpolator

        brts = {
            name: BRTinterpolator(
                args.brt_dir / f"{name}.npz"
            )
            for name in diagnostics.metrics
        }

    if not args.no_plots:
        print(save_plots(
            simulation,
            diagnostics,
            folder / "grafici.pdf",
        ))

    if not args.no_animation:
        print(save_animation_pdf(
            simulation,
            diagnostics,
            folder / "animation.pdf",
            brts=brts,
            fps=args.fps,
        ))


if __name__ == "__main__":
    main()