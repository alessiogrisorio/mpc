from pathlib import Path

import contourpy
import matplotlib.pyplot as plt
import numpy as np
from casadi import Function
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon

from tuning_config import V_MARGIN

from acados_settings import (
    EGO_LENGTH,
    EGO_WIDTH,
    HUMAN_LENGTH,
    HUMAN_WIDTH,
    ROAD_Y_MIN,
    ROAD_Y_MAX,
)


# Contributions of model.cost_y_expr, defined in acados_settings.py.
# Grouping is only for plotting; no MPC cost is changed here.
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


def hj_prediction_diagnostics(solver, M_HJ, b_HJ, stages):
    """Smallest HJ residual and largest lower slack on the constrained stages."""
    values = []
    slacks = []
    for j in range(stages):
        values.append(float(np.dot(M_HJ, solver.get(j, "u")) + b_HJ))
        s = np.asarray(solver.get(j, "sl"), dtype=float)
        slacks.append(float(s[0]) if s.size else 0.0)
    return float(np.min(values)), float(np.max(slacks))


class _BRTZeroContour:
    """Extract V(-3)=0 in ego coordinates, then rotate/translate to world."""

    def __init__(self, brt, nx=85, ny=65):
        self.brt = brt
        self.x_axis = np.linspace(brt.lower[0], brt.upper[0], nx)
        self.y_axis = np.linspace(brt.lower[1], brt.upper[1], ny)
        xx, yy = np.meshgrid(self.x_axis, self.y_axis)
        self.relative_xy = np.column_stack((xx.ravel(), yy.ravel()))
        self.relative_lines = []

    def compute(self, psi, v_h, delta_e, v_e):
        brt = self.brt
        fixed = np.array((v_h, delta_e, v_e), dtype=float)
        if not np.all(np.isfinite(fixed)):
            self.relative_lines = []
            return
        if np.any(fixed < brt.lower[3:]) or np.any(fixed > brt.upper[3:]):
            self.relative_lines = []
            return

        theta_rel = brt.theta_min + (-psi - brt.theta_min) % brt.theta_period
        n = len(self.relative_xy)
        points = np.empty((n, 6), dtype=float)
        points[:, :2] = self.relative_xy
        points[:, 2] = theta_rel
        points[:, 3] = v_h
        points[:, 4] = delta_e
        points[:, 5] = v_e
        field = np.asarray(brt.value_interpolator(points), dtype=float)
        field = field.reshape(len(self.y_axis), len(self.x_axis))
        if not np.all(np.isfinite(field)) or not (field.min() <= 0 <= field.max()):
            self.relative_lines = []
            return

        generator = contourpy.contour_generator(
            x=self.x_axis, y=self.y_axis, z=field
        )
        self.relative_lines = generator.lines(0.0)

    def global_lines(self, x, y, psi):
        c, s = np.cos(psi), np.sin(psi)
        rotation = np.array(((c, -s), (s, c)))
        position = np.array((x, y))
        return [line @ rotation.T + position for line in self.relative_lines]


def _shade_hj_active(ax, times, active):
    active = np.asarray(active, dtype=bool)
    transitions = np.diff(np.r_[False, active, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    for j, (start, end) in enumerate(zip(starts, ends)):
        ax.axvspan(
            times[start], times[min(end, len(times) - 1)],
            color="#fff2b2", alpha=0.7, linewidth=0,
            label="HJ imposed" if j == 0 else None, zorder=0,
        )


def plot_results(dt, simX, simU, v_ref, v_H, simV, simHJActive):
    """Keep the existing standalone MPC diagnostics available."""
    t_x = np.arange(len(simX)) * dt
    t_u = np.arange(len(simU)) * dt
    fig = plt.figure(figsize=(12, 12), layout="constrained")
    grid = fig.add_gridspec(4, 2)
    ax_brt = fig.add_subplot(grid[0, :])
    axes = np.array([
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])],
        [fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1])],
    ])

    _shade_hj_active(ax_brt, t_x, simHJActive)
    ax_brt.plot(t_u, simV, label="BRT")
    ax_brt.axhline(0.0, color="black", linestyle="--", label="V = 0")
    ax_brt.axhline(V_MARGIN, color="tab:red", linestyle="--", label=f"V = {V_MARGIN:g}")
    ax_brt.set(xlim=(t_x[0], t_x[-1]), xlabel="Time [s]", ylabel="V", title="BRT value function")
    ax_brt.legend()

    axes[0, 0].plot(simX[:, 0], simX[:, 1], label="Ego")
    axes[0, 0].axhline(ROAD_Y_MIN, color="black", label="Road boundaries")
    axes[0, 0].axhline(ROAD_Y_MAX, color="black")
    axes[0, 0].axhline(2.0, color="black", linestyle="--", label="Lane divider")
    axes[0, 0].axhline(0.0, color="gray", linestyle=":", label="Reference")
    axes[0, 0].set(xlabel="X [m]", ylabel="Y [m]", title="Trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(t_x, simX[:, 3], label="Ego")
    axes[0, 1].axhline(v_ref, color="black", linestyle="--", label="Reference")
    axes[0, 1].axhline(v_H, color="gray", linestyle=":", label="Human")
    axes[0, 1].set(xlabel="Time [s]", ylabel="Speed [m/s]")
    axes[0, 1].legend()
    axes[1, 0].plot(t_x, np.rad2deg(simX[:, 2]))
    axes[1, 0].set(xlabel="Time [s]", ylabel="Heading [deg]")
    axes[1, 1].plot(t_x, np.rad2deg(simX[:, 4]))
    axes[1, 1].set(xlabel="Time [s]", ylabel="Steering angle [deg]")
    axes[2, 0].step(t_u, np.rad2deg(simU[:, 0]), where="post")
    axes[2, 0].set(xlabel="Time [s]", ylabel="Steering rate [deg/s]")
    axes[2, 1].step(t_u, simU[:, 1], where="post")
    axes[2, 1].set(xlabel="Time [s]", ylabel="Acceleration [m/s²]")
    for ax in fig.axes:
        ax.grid(True)
        ax.set_axisbelow(True)
    return fig, axes


def animate_simulation(
    t,
    simX,
    X_H_initial,
    Y_H_initial,
    v_H,
    save=False,
    filename="simulation.mp4",
    dpi=80,
    simX_pred=None,
    *,
    simU=None,
    v_ref=None,
    simV=None,
    simHJActive=None,
    simHJResidual=None,
    simHJSlack=None,
    simHJSlackUpper=None,
    simHJMinResidual=None,
    simHJMaxSlack=None,
    simOcpCost=None,
    simCostComponents=None,
    brt=None,
    v_margin=V_MARGIN,
    contour_update_every=5,
):
    """Animate the original vehicle view next to diagnostics of the solved MPC."""
    t = np.asarray(t, dtype=float)
    simX = np.asarray(simX, dtype=float)
    nsteps = len(t) - 1
    t_u = t[:-1]

    fig = plt.figure(figsize=(18, 9.6))
    gs = fig.add_gridspec(
        4, 3, left=0.045, right=0.98, bottom=0.07, top=0.955,
        width_ratios=(2.35, 1.0, 1.0), hspace=0.56, wspace=0.46,
    )
    ax = fig.add_subplot(gs[:, 0])
    ax_velocity = fig.add_subplot(gs[0, 1])
    ax_accel = fig.add_subplot(gs[0, 2])
    ax_xy = fig.add_subplot(gs[1, 1])
    ax_brt = fig.add_subplot(gs[1, 2])
    ax_hj = fig.add_subplot(gs[2, 1:])
    ax_cost = fig.add_subplot(gs[3, 1:])

    # --- Main vehicle view: same vehicles, circles, camera and MPC prediction ---
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axhline(ROAD_Y_MIN, color="black", linewidth=2, label="Road boundaries")
    ax.axhline(ROAD_Y_MAX, color="black", linewidth=2)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.5, label="Lane divider")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.5, label="Reference")
    ax.set_ylim(ROAD_Y_MIN - 1.0, ROAD_Y_MAX + 1.0)

    ego_corners = np.array([
        [-EGO_LENGTH / 2, -EGO_WIDTH / 2],
        [ EGO_LENGTH / 2, -EGO_WIDTH / 2],
        [ EGO_LENGTH / 2,  EGO_WIDTH / 2],
        [-EGO_LENGTH / 2,  EGO_WIDTH / 2],
    ])
    human_corners = np.array([
        [-HUMAN_LENGTH / 2, -HUMAN_WIDTH / 2],
        [ HUMAN_LENGTH / 2, -HUMAN_WIDTH / 2],
        [ HUMAN_LENGTH / 2,  HUMAN_WIDTH / 2],
        [-HUMAN_LENGTH / 2,  HUMAN_WIDTH / 2],
    ])
    ego_vehicle = Polygon(
        ego_corners, closed=True, facecolor="tab:blue", edgecolor="black",
        alpha=0.7, label="Ego",
    )
    human_vehicle = Polygon(
        human_corners, closed=True, facecolor="tab:orange", edgecolor="black",
        alpha=0.7, label="Human",
    )
    ax.add_patch(ego_vehicle)
    ax.add_patch(human_vehicle)

    rho_e = np.hypot(EGO_LENGTH / 4.0, EGO_WIDTH / 2.0)
    rho_h = np.hypot(HUMAN_LENGTH / 4.0, HUMAN_WIDTH / 2.0)
    ego_front_circle = Circle(
        (0, 0), radius=rho_e, fill=False, edgecolor="tab:blue",
        linestyle="--", linewidth=1.5, label="Ego circles",
    )
    ego_rear_circle = Circle(
        (0, 0), radius=rho_e, fill=False, edgecolor="tab:blue",
        linestyle="--", linewidth=1.5,
    )
    human_front_circle = Circle(
        (0, 0), radius=rho_h, fill=False, edgecolor="tab:orange",
        linestyle="--", linewidth=1.5, label="Human circles",
    )
    human_rear_circle = Circle(
        (0, 0), radius=rho_h, fill=False, edgecolor="tab:orange",
        linestyle="--", linewidth=1.5,
    )
    for circle in (ego_front_circle, ego_rear_circle, human_front_circle, human_rear_circle):
        ax.add_patch(circle)

    trajectory, = ax.plot([], [], color="tab:blue", linewidth=1.5, label="Trajectory")
    predicted_trajectory, = ax.plot(
        [], [], color="tab:green", linestyle="--", linewidth=2, label="MPC prediction"
    )
    predicted_terminal, = ax.plot(
        [], [], color="tab:green", marker="D", linestyle="None",
        markersize=6, label="Prediction horizon",
    )
    ego_center, = ax.plot([], [], "ko", markersize=4)
    human_center, = ax.plot([], [], "ko", markersize=4)
    heading, = ax.plot([], [], color="black", linewidth=2)

    brt_contour_artist = LineCollection(
        [], colors="tab:red", linewidths=0.9, label="BRT V = 0", zorder=3
    )
    ax.add_collection(brt_contour_artist)
    contour = _BRTZeroContour(brt) if brt is not None else None
    contour_update_every = max(1, int(contour_update_every))
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=7.5)

    # --- Diagnostics ---
    ax_velocity.plot(t, simX[:, 3], label="Ego")
    if v_ref is not None:
        ax_velocity.axhline(v_ref, linestyle="--", color="black", label="Reference")
    ax_velocity.set(title="Velocity", xlabel="Time [s]", ylabel="m/s", xlim=(t[0], t[-1]))
    ax_velocity.legend(fontsize=7, loc="best")

    if simU is not None:
        ax_accel.step(t_u, np.asarray(simU)[:, 1], where="post", label="Ego")
    ax_accel.axhline(0.0, color="gray", linewidth=0.7)
    ax_accel.set(title="Acceleration", xlabel="Time [s]", ylabel="m/s²", xlim=(t[0], t[-1]))

    # Trajectory colored by absolute simulation time, with a fixed colorbar.
    segments = np.stack([simX[:-1, :2], simX[1:, :2]], axis=1)
    tmiddle = 0.5 * (t[:-1] + t[1:])
    norm = Normalize(vmin=t[0], vmax=t[-1])
    path = LineCollection([], cmap="viridis", norm=norm, linewidths=2.8)
    ax_xy.add_collection(path)
    marker_xy, = ax_xy.plot([], [], "ko", markersize=4, zorder=4)
    ax_xy.axhline(ROAD_Y_MIN, color="gray", linewidth=0.8)
    ax_xy.axhline(ROAD_Y_MAX, color="gray", linewidth=0.8)
    ax_xy.axhline(2.0, color="gray", linestyle="--", linewidth=0.8)
    xr = np.ptp(simX[:, 0])
    yr = np.ptp(simX[:, 1])
    ax_xy.set_xlim(np.min(simX[:, 0]) - max(2, xr * 0.06), np.max(simX[:, 0]) + max(2, xr * 0.06))
    ax_xy.set_ylim(min(ROAD_Y_MIN - 0.5, np.min(simX[:, 1]) - max(1, yr * 0.1)),
                   max(ROAD_Y_MAX + 0.5, np.max(simX[:, 1]) + max(1, yr * 0.1)))
    ax_xy.set(title="Ego trajectory by time", xlabel="X [m]", ylabel="Y [m]")
    cbar = fig.colorbar(path, ax=ax_xy, fraction=0.055, pad=0.03)
    cbar.set_label("Time [s]", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if simHJActive is None:
        active = np.zeros(nsteps, dtype=bool)
    else:
        active = np.asarray(simHJActive, dtype=bool)

    _shade_hj_active(ax_brt, t, active)
    if simV is not None:
        ax_brt.plot(t_u, simV, label="BRT value")
    ax_brt.axhline(v_margin, linestyle="--", color="tab:red", label=f"V = {v_margin:g}")
    ax_brt.axhline(0, linestyle=":", color="black", linewidth=0.9, label="V = 0")
    ax_brt.set(title="BRT V(-3)", xlabel="Time [s]", ylabel="V", xlim=(t[0], t[-1]))
    ax_brt.legend(fontsize=7, loc="best")

    _shade_hj_active(ax_hj, t, active)
    if simHJResidual is not None:
        ax_hj.plot(t_u, simHJResidual, label="HJ residual (stage 0)", linewidth=1.4)
    if simHJMinResidual is not None:
        ax_hj.plot(t_u, simHJMinResidual, label="Min residual (HJ stages)", linewidth=1.1)
    if simHJSlack is not None:
        ax_hj.plot(t_u, simHJSlack, label="Lower slack (stage 0)", linewidth=1.1)
        ax_hj.plot(t_u, -np.asarray(simHJSlack), linestyle=":", color="tab:red", label="Allowed limit -slack")
    if simHJMaxSlack is not None:
        ax_hj.plot(t_u, simHJMaxSlack, label="Max slack (HJ stages)", linewidth=1.1)
    ax_hj.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_hj.set(title="HJ constraint: residual and slack", xlabel="Time [s]", ylabel="Residual / slack", xlim=(t[0], t[-1]))
    ax_hj.legend(fontsize=7, loc="upper right", ncol=2)

    if simOcpCost is not None:
        ax_cost.plot(t_u, simOcpCost, label="TOTAL (acados)", color="black", linewidth=2.0)
    if simCostComponents is not None:
        for name in COST_GROUPS:
            series = simCostComponents.get(name)
            if series is not None:
                ax_cost.plot(t_u, series, linewidth=1.0, label=name)
    ax_cost.set(title="MPC cost over entire prediction horizon", xlabel="Time [s]", ylabel="Cost", xlim=(t[0], t[-1]))
    ax_cost.legend(fontsize=7, loc="upper right", ncol=3)

    cursor_axes = (ax_velocity, ax_accel, ax_brt, ax_hj, ax_cost)
    cursors = [a.axvline(t[0], color="tab:red", alpha=0.7, linewidth=1.1) for a in cursor_axes]
    for a in (ax_velocity, ax_accel, ax_xy, ax_brt, ax_hj, ax_cost):
        a.grid(True, alpha=0.3)
        a.set_axisbelow(True)
        a.tick_params(axis="both", labelsize=8)
        a.title.set_fontsize(10)

    def update(i):
        X_E, Y_E, psi_E, v_E, delta_E = simX[i]
        X_H = X_H_initial + v_H * t[i]
        Y_H = Y_H_initial
        c, s = np.cos(psi_E), np.sin(psi_E)
        rotation = np.array([[c, -s], [s, c]])
        ego_vehicle.set_xy(ego_corners @ rotation.T + np.array([X_E, Y_E]))
        human_vehicle.set_xy(human_corners + np.array([X_H, Y_H]))

        ego_front_x = X_E + EGO_LENGTH / 4.0 * c
        ego_front_y = Y_E + EGO_LENGTH / 4.0 * s
        ego_rear_x = X_E - EGO_LENGTH / 4.0 * c
        ego_rear_y = Y_E - EGO_LENGTH / 4.0 * s
        ego_front_circle.center = (ego_front_x, ego_front_y)
        ego_rear_circle.center = (ego_rear_x, ego_rear_y)
        human_front_x = X_H + HUMAN_LENGTH / 4.0
        human_rear_x = X_H - HUMAN_LENGTH / 4.0
        human_front_circle.center = (human_front_x, Y_H)
        human_rear_circle.center = (human_rear_x, Y_H)

        ego_center.set_data([X_E], [Y_E])
        human_center.set_data([X_H], [Y_H])
        heading.set_data(
            [X_E, X_E + EGO_LENGTH / 2.0 * c],
            [Y_E, Y_E + EGO_LENGTH / 2.0 * s],
        )
        trajectory.set_data(simX[:i + 1, 0], simX[:i + 1, 1])

        if simX_pred is not None and i < len(simX_pred):
            prediction = simX_pred[i]
            predicted_trajectory.set_data(prediction[:, 0], prediction[:, 1])
            predicted_terminal.set_data([prediction[-1, 0]], [prediction[-1, 1]])
        else:
            predicted_trajectory.set_data([], [])
            predicted_terminal.set_data([], [])

        if contour is not None:
            if i == 0 or i % contour_update_every == 0:
                contour.compute(psi_E, v_H, delta_E, v_E)
            brt_contour_artist.set_segments(contour.global_lines(X_E, Y_E, psi_E))

        x_min = min(X_E, X_H)
        x_max = max(X_E, X_H)
        if simX_pred is not None and i < len(simX_pred):
            x_min = min(x_min, float(np.min(simX_pred[i, :, 0])))
            x_max = max(x_max, float(np.max(simX_pred[i, :, 0])))
        ax.set_xlim(x_min - 12.0, x_max + 12.0)

        ego_circles = np.array(((ego_front_x, ego_front_y), (ego_rear_x, ego_rear_y)))
        human_circles = np.array(((human_front_x, Y_H), (human_rear_x, Y_H)))
        min_center_distance = float(np.linalg.norm(
            ego_circles[:, None, :] - human_circles[None, :, :], axis=-1
        ).min())
        title.set_text(
            f"t = {t[i]:.2f} s   |   "
            f"v_E = {v_E:.2f} m/s   |   "
            f"v_H = {v_H:.2f} m/s   |   "
            f"delta = {np.rad2deg(delta_E):.1f}°\n"
            f"Minimum circle distance = {min_center_distance:.2f} m   |   "
            f"Required = {rho_e + rho_h:.2f} m"
        )

        path.set_segments(segments[:i])
        path.set_array(tmiddle[:i])
        marker_xy.set_data([X_E], [Y_E])
        for line in cursors:
            line.set_xdata([t[i], t[i]])
        return ()

    update(0)
    frames = range(len(t))
    movie = FuncAnimation(
        fig, update, frames=frames,
        interval=1000 * (t[1] - t[0]), repeat=False, blit=False,
    )

    if save:
        video_path = Path(__file__).resolve().parent / filename
        movie.save(str(video_path), writer="ffmpeg", fps=1.0 / (t[1] - t[0]), dpi=dpi)
        print(f"Animation saved to: {video_path}")

    return movie
