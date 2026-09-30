import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle


def animate_simulation(
    t, simX, X_H_initial, v_H,
    length=4.68, width=2.2,
    human_length=4.28, human_width=1.80,
):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    # Reference path and traveled trajectory
    ax.axhline(
        0.0, color="black", linestyle="--", label="Reference"
    )
    trajectory, = ax.plot(
        [], [], color="tab:blue", label="Trajectory"
    )

    # Ego outline relative to its center
    corners = np.array([
        [-length / 2, -width / 2],
        [ length / 2, -width / 2],
        [ length / 2,  width / 2],
        [-length / 2,  width / 2],
    ])

    vehicle = Polygon(
        corners,
        closed=True,
        facecolor="tab:blue",
        edgecolor="black",
        alpha=0.7,
        label="Ego",
    )
    ax.add_patch(vehicle)

    heading, = ax.plot([], [], color="black", linewidth=2)
    ego_center, = ax.plot([], [], "ko", markersize=4)

    # Human outline: fixed heading, moving along Y = 0
    human_corners = np.array([
        [-human_length / 2, -human_width / 2],
        [ human_length / 2, -human_width / 2],
        [ human_length / 2,  human_width / 2],
        [-human_length / 2,  human_width / 2],
    ])

    human_vehicle = Polygon(
        human_corners,
        closed=True,
        facecolor="tab:orange",
        edgecolor="black",
        alpha=0.7,
        label="Human",
    )
    ax.add_patch(human_vehicle)

    # Exclusion circle for the ego center
    safety_distance = (
        0.5 * np.hypot(length, width)
        + 0.5 * np.hypot(human_length, human_width)
    )

    safety_circle = Circle(
        (X_H_initial, 0.0),
        radius=safety_distance,
        fill=False,
        edgecolor="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Exclusion circle",
    )
    ax.add_patch(safety_circle)

    title = ax.set_title("")
    ax.legend(loc="upper right")

    # Keep the lateral motion and exclusion circle visible
    ax.set_ylim(
        min(-safety_distance, simX[:, 1].min() - width / 2) - 2.0,
        max(safety_distance, simX[:, 1].max() + width / 2) + 2.0,
    )

    def update(i):

        X, Y, psi, v, delta = simX[i]

        # Rotate and translate ego
        rotation = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)],
        ])

        vehicle.set_xy(
            corners @ rotation.T + np.array([X, Y])
        )

        heading.set_data(
            [X, X + length / 2 * np.cos(psi)],
            [Y, Y + length / 2 * np.sin(psi)],
        )
        ego_center.set_data([X], [Y])

        trajectory.set_data(
            simX[:i + 1, 0], simX[:i + 1, 1]
        )

        # Update human and its exclusion circle
        X_H = X_H_initial + v_H * t[i]

        human_vehicle.set_xy(
            human_corners + np.array([X_H, 0.0])
        )
        safety_circle.center = (X_H, 0.0)

        # Follow ego, keeping human visible
        ax.set_xlim(
            min(X - 10.0, X_H - safety_distance - 2.0),
            max(X + 20.0, X_H + safety_distance + 2.0),
        )

        distance = np.hypot(X - X_H, Y)

        title.set_text(
            f"t = {t[i]:.2f} s   |   "
            f"v = {v:.2f} m/s   |   "
            f"delta = {np.rad2deg(delta):.1f}°\n"
            f"Distance = {distance:.2f} m   |   "
            f"Minimum = {safety_distance:.2f} m"
        )

        return (
            vehicle, heading, ego_center, trajectory,
            human_vehicle, safety_circle, title,
        )

    update(0)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=1000 * (t[1] - t[0]),
        repeat=False,
        blit=False,
    )

    return animation