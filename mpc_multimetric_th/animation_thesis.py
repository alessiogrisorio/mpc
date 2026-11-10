"""Thesis exports: grafici.pdf and animation.pdf.

Keep the original animation.py and hj_diagnostics.py alongside this file.

Offline:
    python animation_thesis.py results/RUN/hj.npz
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
)

from animation import (
    COST_GROUPS,
    OcpCostDiagnostics,
    load_results,
    metric_colors,
    export_results as _save_original_results,
)


# ============================================================
# SETTINGS
# ============================================================

FONT_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 24
TIME_SIZE = 24
COLORBAR_SIZE = 24

FIGSIZE = (9, 5)
STATES_FIGSIZE = (9, 13)
DOUBLE_FIGSIZE = (10, 8)
LIMITS_FIGSIZE = (13, 8)
PLANE_FIGSIZE = (13, 6.5)

LINE_WIDTH = 2.5
COMMAND_WIDTH = 2.0
BOUND_WIDTH = 2.0
MARKER_SIZE = 28

HJ_TIME_PADDING = 1.0

# Animation PDF.
FRAME_TEXT_SCALE = 2.0
FRAME_LINE_SCALE = 2.0

FRAME_FPS = 2.0
FRAME_WIDTH = 20.0       # Page width [inches].

VIEW_BEHIND = 10.0      # Space behind ego [m].
VIEW_AHEAD = 20.0       # Space ahead of ego [m].

VIEW_Y_MIN = -2.0
VIEW_Y_MAX = 6.0

SHOW_PREDICTION = True
SHOW_BRT_CONTOURS = False
CONTOUR_GRID = (70, 50)

# None: first, middle and last HJ-active sample.
# Example: PLANE_TIMES = [2.0, 4.0, 6.0]
# PLANE_TIMES = [6.0, 8.0, 10.0]
PLANE_TIMES = None

SHOW_RELAXED_BOUNDARIES = True
SLACK_DISPLAY_TOL = 1e-7

# Produce both the original and green-area visualizations.
MAKE_LINE_LIMIT_PLOTS = True
MAKE_GREEN_LIMIT_PLOTS = True

HJ_COLOR = "tab:green"
HJ_ALPHA = 0.18
PHYSICAL_COLOR = "tab:red"

EGO_LENGTH, EGO_WIDTH = 4.68, 2.20
HUMAN_LENGTH, HUMAN_WIDTH = 4.28, 1.80

ROAD_Y_MIN = -2.0
ROAD_Y_MAX = 6.0
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
    "time_eggert": "TBR",
    "mixed": "Mixed",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
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

    if legend and ax.get_legend_handles_labels()[0]:
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
    upper = float(t[-1]) if len(t) > 1 else float(t[0]) + 0.02
    ax.set_xlim(float(t[0]), upper)


def set_hj_time_domain(ax, diagnostics):
    """Common HJ time window across all metrics."""
    d = diagnostics.arrays()
    t = d["time"]

    indices = np.flatnonzero(
        np.any(d["active"][:, 0], axis=1)
    )

    if not len(indices):
        set_time_domain(ax, t)
        return

    first = int(indices[0])
    last = int(indices[-1])

    switch_off = (
        t[last + 1]
        if last + 1 < len(t)
        else t[last] + diagnostics.dt
    )

    lower = max(t[0], t[first] - HJ_TIME_PADDING)
    upper = min(t[-1], switch_off + HJ_TIME_PADDING)

    if upper > lower:
        ax.set_xlim(float(lower), float(upper))
    else:
        set_time_domain(ax, t)


def write_page(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def road(ax):
    for y in (ROAD_Y_MIN, ROAD_Y_MAX):
        ax.axhline(
            y,
            color="black",
            linewidth=4.0 * FRAME_LINE_SCALE,
        )

    ax.axhline(
        LANE_DIVIDER,
        color="0.45",
        linestyle="--",
        linewidth=2.5 * FRAME_LINE_SCALE,
    )


def vehicle(ax, state, length, width, color):
    X, Y, psi = state[:3]

    corners = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2],
    ])

    c, s = np.cos(psi), np.sin(psi)
    rotation_transpose = np.array([
        [c, s],
        [-s, c],
    ])

    vertices = corners @ rotation_transpose + [X, Y]

    ax.add_patch(Polygon(
        vertices,
        facecolor=color,
        edgecolor="black",
        linewidth=0.8 * FRAME_LINE_SCALE,
        alpha=0.75,
        zorder=5,
    ))


def human_state(simulation, time):
    return [
        scalar(simulation, "X_H_initial")
        + scalar(simulation, "v_H") * time,
        scalar(simulation, "Y_H_initial"),
        0.0,
    ]


def physical_bounds(ax, diagnostics, axis, *, label=True):
    values = (
        diagnostics.u_lower[axis],
        diagnostics.u_upper[axis],
    )

    for n, value in enumerate(values):
        ax.axhline(
            value,
            color=PHYSICAL_COLOR,
            linestyle="--",
            linewidth=1.1,
            zorder=1,
            label="Physical bounds" if label and n == 0 else None,
        )


def control_axis(ax, diagnostics, axis):
    lower = diagnostics.u_lower[axis]
    upper = diagnostics.u_upper[axis]
    span = upper - lower

    ax.set_ylim(
        lower - 0.1 * span,
        upper + 0.1 * span,
    )

    ax.set_ylabel(
        r"$\dot{\delta}_E$ [rad/s]"
        if axis == 0
        else r"$a_E$ [m/s$^2$]"
    )

    set_hj_time_domain(ax, diagnostics)


# ============================================================
# ORIGINAL CONTROL-LIMIT VISUALIZATION
# ============================================================

def draw_limits(axes, t, u, diagnostics, limits, indices, colors):
    """Conditional control limits at stage 0, without slack."""
    for axis, key in enumerate(("steering_rate", "acceleration")):
        ax = axes[axis]
        bound = limits[key]

        ax.step(
            t,
            u[:, axis],
            where="post",
            color="black",
            linewidth=COMMAND_WIDTH,
            label="MPC command",
            zorder=4,
        )

        physical_bounds(ax, diagnostics, axis)

        violated = np.zeros(len(t), dtype=bool)

        for k in indices:
            lower = bound["metric_lower"][:, 0, k]
            upper = bound["metric_upper"][:, 0, k]

            for values, pattern, side in (
                (lower, "--", "lower"),
                (upper, ":", "upper"),
            ):
                if not np.any(np.isfinite(values)):
                    continue

                ax.step(
                    t,
                    np.where(np.isfinite(values), values, np.nan),
                    where="post",
                    color=colors[k],
                    linestyle=pattern,
                    linewidth=BOUND_WIDTH,
                    label=(
                        f"{metric_label(diagnostics.metrics[k])}: {side}"
                    ),
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

        control_axis(ax, diagnostics, axis)
        finish(ax, outside=True)


# ============================================================
# GREEN ADMISSIBLE AREA
# ============================================================

def allowed_interval(diagnostics, limits, indices, axis):
    """Intersection of selected HJ intervals and physical bounds.

    The other control is fixed to its MPC value.
    Recorded slack is not included.
    """
    d = diagnostics.arrays()
    key = ("steering_rate", "acceleration")[axis]

    lower = np.maximum(
        diagnostics.u_lower[axis],
        np.max(
            limits[key]["metric_lower"][:, 0, indices],
            axis=-1,
        ),
    )

    upper = np.minimum(
        diagnostics.u_upper[axis],
        np.min(
            limits[key]["metric_upper"][:, 0, indices],
            axis=-1,
        ),
    )

    # A near-zero coefficient produces no divided bound.
    # Explicitly check whether the remaining inequality is impossible.
    M = d["M"][:, 0][:, indices, :]
    b = d["b"][:, 0][:, indices]
    active = d["active"][:, 0][:, indices]

    scale = (
        diagnostics.u_upper - diagnostics.u_lower
    ) / 2.0

    magnitude = np.max(
        np.abs(M) * scale,
        axis=-1,
    )

    zero = (
        np.abs(M[..., axis]) * scale[axis]
        <= diagnostics.coefficient_tolerance * magnitude
    )

    constant = (
        M[..., 1 - axis]
        * d["u"][:, 0, None, 1 - axis]
        + b
    )

    impossible = np.any(
        active
        & zero
        & (
            constant
            < -diagnostics.tolerance
            * np.maximum(np.abs(constant), magnitude)
        ),
        axis=-1,
    )

    feasible = (lower <= upper) & ~impossible

    return lower, upper, feasible


def draw_green_limits(axes, t, u, diagnostics, limits, indices):
    label = (
        "HJ multi constraint"
        if len(indices) > 1
        else "HJ " + metric_label(diagnostics.metrics[indices[0]])
    )

    for axis, ax in enumerate(axes):
        lower, upper, feasible = allowed_interval(
            diagnostics,
            limits,
            indices,
            axis,
        )

        # One area and one joint envelope, also for multiple metrics.
        ax.fill_between(
            t,
            lower,
            upper,
            where=feasible,
            step="post",
            color=HJ_COLOR,
            alpha=HJ_ALPHA,
            linewidth=0,
            zorder=2,
        )

        # Draw green only where the boundary is imposed by HJ.
        # Boundaries coinciding with physical limits remain red dashed.
        boundaries = (
            (
                lower,
                lower
                > diagnostics.u_lower[axis]
                + diagnostics.tolerance,
            ),
            (
                upper,
                upper
                < diagnostics.u_upper[axis]
                - diagnostics.tolerance,
            ),
        )

        for values, imposed in boundaries:
            ax.step(
                t,
                np.where(
                    feasible & imposed,
                    values,
                    np.nan,
                ),
                where="post",
                color=HJ_COLOR,
                linewidth=BOUND_WIDTH,
                zorder=3,
            )

        physical_bounds(
            ax,
            diagnostics,
            axis,
            label=False,
        )

        ax.step(
            t,
            u[:, axis],
            where="post",
            color="black",
            linewidth=COMMAND_WIDTH,
            zorder=5,
        )

        control_axis(ax, diagnostics, axis)
        finish(ax, legend=False)

        ax.legend(
            handles=[
                Line2D(
                    [], [],
                    color="black",
                    linewidth=COMMAND_WIDTH,
                    label="MPC command",
                ),
                Line2D(
                    [], [],
                    color=PHYSICAL_COLOR,
                    linestyle="--",
                    linewidth=1.1,
                    label="Physical bounds",
                ),
                Patch(
                    facecolor=HJ_COLOR,
                    edgecolor=HJ_COLOR,
                    alpha=HJ_ALPHA,
                    label=label,
                ),
            ],
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            fontsize=LEGEND_SIZE,
        )


# ============================================================
# COMMAND PLANE
# ============================================================

def header_figure(figsize, height=1.25, left=1.3):
    fig = plt.figure(figsize=figsize)
    fw, fh = fig.get_size_inches()

    right = 0.20
    bottom = 1.1
    top = 0.12
    gap = 0.05

    width = fw - left - right
    plot_height = fh - top - height - gap - bottom

    if plot_height <= 0:
        raise ValueError("Figure too short for header")

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

    return fig, header, ax


def command_plane(diagnostics, index, colors):
    d = diagnostics.arrays()

    fig, header, ax = header_figure(PLANE_FIGSIZE, height=1.25, left=1.3)

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
        edgecolor=PHYSICAL_COLOR,
        linestyle="--",
        linewidth=1.1,
        label="Physical bounds",
        zorder=2,
    ))

    polygon = diagnostics.control_polygon(
        index,
        0,
        softened=False,
    )

    if len(polygon) >= 3:
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            color=HJ_COLOR,
            alpha=HJ_ALPHA,
            label="Joint HJ feasible set",
            zorder=1,
        )

    elif len(polygon):
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color=HJ_COLOR,
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
                xx,
                yy,
                field,
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
        relaxed = field + slack

        if (
            SHOW_RELAXED_BOUNDARIES
            and slack > SLACK_DISPLAY_TOL
            and relaxed.min() < 0 < relaxed.max()
        ):
            ax.contour(
                xx,
                yy,
                relaxed,
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
        *command,
        color="black",
        s=MARKER_SIZE * 2,
        label="MPC command",
        zorder=8,
        clip_on=False,
    )

    ax.set(
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        xlabel=r"$\dot{\delta}_E$ [rad/s]",
        ylabel=r"$a_E$ [m/s$^2$]",
    )

    message = rf"$t = {d['time'][index]:.2f}\,\mathrm{{s}}$"

    if not len(polygon):
        message += "\nEmpty original\nHJ feasible set"

    header.text(
        0.0,
        0.5,
        message,
        va="center",
        fontsize=TIME_SIZE,
    )

    header.legend(
        *ax.get_legend_handles_labels(),
        loc="center right",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fontsize=LEGEND_SIZE,
    )

    finish(
        ax,
        time_axis=False,
        legend=False,
    )

    return fig


# ============================================================
# STATIC PLOTS
# ============================================================

def save_plots(simulation, diagnostics, path, plane_times=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    d = diagnostics.arrays()
    c = diagnostics.contributions()
    summary = diagnostics.summary()

    limits = diagnostics.compute_limits(
        softened=False
    )

    x = np.asarray(simulation["simX"])
    u = np.asarray(simulation["simU"])

    t = d["time"]
    tx = np.arange(len(x)) * scalar(simulation, "dt")

    colors = metric_colors(diagnostics.metrics)

    groups = [
        list(range(len(diagnostics.metrics)))
    ] + [
        [k] for k in range(len(diagnostics.metrics))
    ]

    with plt.rc_context(STYLE), PdfPages(path) as pdf:

        # 1. Time-colored trajectory.
        fig, ax = plt.subplots(
            figsize=FIGSIZE,
            layout="constrained",
        )

        points = x[:, :2]
        segments = np.stack(
            [points[:-1], points[1:]],
            axis=1,
        )

        line = LineCollection(
            segments,
            cmap="viridis",
            norm=Normalize(tx[0], tx[-1]),
            linewidths=2.8,
        )
        line.set_array(
            (tx[:-1] + tx[1:]) / 2.0
        )

        ax.add_collection(line)
        ax.update_datalim(points)
        ax.autoscale_view()
        ax.margins(x=0.03, y=0.12)

        ax.set(
            xlabel=r"$X$ [m]",
            ylabel=r"$Y$ [m]",
            aspect="auto",
        )

        bar = fig.colorbar(
            line,
            ax=ax,
            pad=0.02,
        )
        bar.set_label(
            "Time [s]",
            fontsize=COLORBAR_SIZE,
        )
        bar.ax.tick_params(
            labelsize=TICK_SIZE
        )

        finish(
            ax,
            time_axis=False,
            legend=False,
        )
        write_page(pdf, fig)

        # 2. Ego states.
        fig, axes = plt.subplots(
            5,
            1,
            sharex=True,
            figsize=STATES_FIGSIZE,
            layout="constrained",
        )

        labels = [
            r"$X_E$ [m]",
            r"$Y_E$ [m]",
            r"$**\t**heta_E$ [rad]",
            r"$v_E$ [m/s]",
            r"$\delta_E$ [rad]",
        ]

        for j, ax in enumerate(axes):
            ax.plot(
                tx,
                x[:, j],
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

        # 3. Applied controls.
        fig, axes = plt.subplots(
            2,
            1,
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
                zorder=4,
            )

            physical_bounds(
                ax,
                diagnostics,
                axis,
            )

            ax.set_ylabel(
                r"$\dot{\delta}_E$ [rad/s]"
                if axis == 0
                else r"$a_E$ [m/s$^2$]"
            )

            set_time_domain(ax, t)
            finish(ax)

        write_page(pdf, fig)

        # 4. BRT: combined and individual.
        # Two figures are intentionally retained for one metric.
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

            ax.set_ylabel(r"$V$")
            set_hj_time_domain(ax, diagnostics)
            finish(ax)
            write_page(pdf, fig)

        # 5. Control limits:
        # first line-based figures, then green-area figures.
        for green, enabled in (
            (False, MAKE_LINE_LIMIT_PLOTS),
            (True, MAKE_GREEN_LIMIT_PLOTS),
        ):
            if not enabled:
                continue

            for indices in groups:
                fig, axes = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=LIMITS_FIGSIZE,
                    layout="constrained",
                )

                if green:
                    draw_green_limits(
                        axes,
                        t,
                        u,
                        diagnostics,
                        limits,
                        indices,
                    )
                else:
                    draw_limits(
                        axes,
                        t,
                        u,
                        diagnostics,
                        limits,
                        indices,
                        colors,
                    )

                write_page(pdf, fig)

        # 6. Cost over the optimized prediction horizon.
        if simulation.get("simOcpCost") is not None:
            fig, axes = plt.subplots(
                2,
                1,
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
                    t,
                    values,
                    label=name,
                )

            axes[1].set_ylabel("Cost components")

            for ax in axes:
                set_time_domain(ax, t)
                finish(ax, outside=True)

            write_page(pdf, fig)

        # 7. Residuals and slack.
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

            for values, extreme, label, extreme_label in entries:
                fig, ax = plt.subplots(
                    figsize=FIGSIZE,
                    layout="constrained",
                )

                ax.plot(
                    t,
                    values,
                    color=colors[k],
                    label=f"{metric_label(name)}: stage 0",
                )

                ax.plot(
                    t,
                    extreme,
                    color=colors[k],
                    linestyle="--",
                    label=extreme_label,
                )

                ax.axhline(
                    0,
                    color="0.4",
                    linewidth=0.8,
                )

                ax.set_ylabel(label)
                set_time_domain(ax, t)
                finish(ax)
                write_page(pdf, fig)

        # 8. Selected command planes.
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
                    diagnostics,
                    index,
                    colors,
                ),
            )

    return path


# ============================================================
# OPTIONAL GLOBAL BRT CONTOURS
# ============================================================

def draw_brt_contours(
    ax,
    brts,
    state,
    human_speed,
    colors,
    names,
):
    nx, ny = CONTOUR_GRID

    for k, name in enumerate(names):
        if name not in brts:
            continue

        brt = brts[name]

        fixed = np.array([
            human_speed,
            state[4],
            state[3],
        ])

        if (
            np.any(fixed < brt.lower[3:])
            or np.any(fixed > brt.upper[3:])
        ):
            continue

        xx, yy = np.meshgrid(
            np.linspace(brt.lower[0], brt.upper[0], nx),
            np.linspace(brt.lower[1], brt.upper[1], ny),
        )

        points = np.empty((xx.size, 6))

        points[:, :2] = np.column_stack([
            xx.ravel(),
            yy.ravel(),
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

        if (
            np.isfinite(values).all()
            and values.min() < 0 < values.max()
        ):
            c, s = np.cos(state[2]), np.sin(state[2])

            ax.contour(
                state[0] + c * xx - s * yy,
                state[1] + s * xx + c * yy,
                values,
                levels=[0],
                colors=[colors[k]],
                linewidths=1.2 * FRAME_LINE_SCALE,
            )


# ============================================================
# COMPACT ANIMATION FRAMES
# ============================================================

def frame_figure():
    left = 1.2
    right = 0.15
    bottom = 0.85
    top = 0.10

    header_height = 0.40 * FRAME_TEXT_SCALE
    gap = 0.04

    width = FRAME_WIDTH - left - right

    road_height = (
        width
        * (VIEW_Y_MAX - VIEW_Y_MIN)
        / (VIEW_BEHIND + VIEW_AHEAD)
    )

    height = (
        bottom
        + road_height
        + gap
        + header_height
        + top
    )

    fig = plt.figure(
        figsize=(FRAME_WIDTH, height)
    )

    ax = fig.add_axes([
        left / FRAME_WIDTH,
        bottom / height,
        width / FRAME_WIDTH,
        road_height / height,
    ])

    header = fig.add_axes([
        left / FRAME_WIDTH,
        (bottom + road_height + gap) / height,
        width / FRAME_WIDTH,
        header_height / height,
    ])
    header.set_axis_off()

    return fig, header, ax


def frame_header(header, time, prediction_visible):
    entries = [
        TextArea(
            rf"$t = {time:.2f}\,\mathrm{{s}}$",
            textprops={
                "fontsize": TIME_SIZE * FRAME_TEXT_SCALE,
            },
        )
    ]

    if prediction_visible:
        sample = DrawingArea(
            32,
            12,
            0,
            0,
        )

        sample.add_artist(Line2D(
            [0, 32],
            [6, 6],
            color="tab:green",
            linestyle="--",
            linewidth=LINE_WIDTH * FRAME_LINE_SCALE,
        ))

        prediction_key = HPacker(
            children=[
                sample,
                TextArea(
                    "MPC prediction",
                    textprops={
                        "fontsize": LEGEND_SIZE * FRAME_TEXT_SCALE,
                    },
                ),
            ],
            align="center",
            pad=0,
            sep=6,
        )

        entries.append(prediction_key)

    row = HPacker(
        children=entries,
        align="center",
        pad=0,
        sep=24,
    )

    header.add_artist(AnchoredOffsetbox(
        loc="center left",
        child=row,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(0.0, 0.5),
        bbox_transform=header.transAxes,
    ))


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
        raise ValueError("FPS must be finite and positive")

    settings = [
        VIEW_BEHIND,
        VIEW_AHEAD,
        VIEW_Y_MIN,
        VIEW_Y_MAX,
        FRAME_WIDTH,
    ]

    if (
        not np.isfinite(settings).all()
        or VIEW_BEHIND < 0
        or VIEW_AHEAD < 0
        or VIEW_BEHIND + VIEW_AHEAD <= 0
        or VIEW_Y_MIN >= VIEW_Y_MAX
        or FRAME_WIDTH <= 1
    ):
        raise ValueError("Invalid camera dimensions")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(simulation["simX"])

    tx = (
        np.arange(len(x))
        * scalar(simulation, "dt")
    )

    if tx[-1] <= 0:
        raise ValueError(
            "Simulation must have positive duration"
        )

    frame_times = np.r_[
        np.arange(0, tx[-1], 1.0 / fps),
        tx[-1],
    ]

    heading = np.unwrap(x[:, 2])
    predictions = simulation.get("simX_pred")
    colors = metric_colors(diagnostics.metrics)

    frame_style = dict(STYLE)
    for key in (
        "font.size", "axes.labelsize", "axes.titlesize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
        "figure.titlesize",
    ):
        frame_style[key] *= FRAME_TEXT_SCALE

    with plt.rc_context(frame_style), PdfPages(path) as pdf:
        for time in frame_times:
            state = np.array([
                np.interp(
                    time,
                    tx,
                    heading if j == 2 else x[:, j],
                )
                for j in range(x.shape[1])
            ])

            index = min(
                int(np.searchsorted(
                    tx,
                    time,
                    side="right",
                ) - 1),
                len(x) - 1,
            )

            fig, header, ax = frame_figure()

            road(ax)

            ax.plot(
                np.r_[x[:index + 1, 0], state[0]],
                np.r_[x[:index + 1, 1], state[1]],
                color="tab:blue",
                linewidth=4.0 * FRAME_LINE_SCALE,
            )

            vehicle(
                ax,
                state,
                EGO_LENGTH,
                EGO_WIDTH,
                "tab:blue",
            )

            vehicle(
                ax,
                human_state(simulation, time),
                HUMAN_LENGTH,
                HUMAN_WIDTH,
                "tab:orange",
            )

            prediction_visible = (
                SHOW_PREDICTION
                and predictions is not None
                and len(predictions) > 0
            )

            if prediction_visible:
                prediction_index = min(
                    index,
                    len(predictions) - 1,
                )

                prediction = np.asarray(
                    predictions[prediction_index]
                )

                ax.plot(
                    prediction[:, 0],
                    prediction[:, 1],
                    color="tab:green",
                    linestyle="--",
                    linewidth=LINE_WIDTH * FRAME_LINE_SCALE,
                )

            if SHOW_BRT_CONTOURS and brts:
                draw_brt_contours(
                    ax,
                    brts,
                    state,
                    scalar(simulation, "v_H"),
                    colors,
                    diagnostics.metrics,
                )

            ax.set(
                xlim=(
                    state[0] - VIEW_BEHIND,
                    state[0] + VIEW_AHEAD,
                ),
                ylim=(
                    VIEW_Y_MIN,
                    VIEW_Y_MAX,
                ),
                xlabel=r"$X$ [m]",
                ylabel=r"$Y$ [m]",
                aspect="equal",
            )

            # No surrounding box: only the actual road edges remain.
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.grid(False)
            ax.minorticks_off()

            # Keep axis numbers and labels; remove tick marks.
            ax.tick_params(
                axis="both",
                which="both",
                length=0,
                top=False,
                right=False,
                pad=7,
            )

            ax.ticklabel_format(
                axis="x",
                style="plain",
                useOffset=False,
            )

            frame_header(
                header,
                time,
                prediction_visible,
            )

            # Keep enlarged labels inside the PDF page.
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)

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
# OFFLINE REGENERATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--brt-dir", type=Path)

    parser.add_argument(
        "--fps",
        type=float,
        default=FRAME_FPS,
    )

    parser.add_argument(
        "--no-animation",
        action="store_true",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
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