from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

from acados_settings import (
    EGO_LENGTH,
    EGO_WIDTH,
    HUMAN_LENGTH,
    HUMAN_WIDTH,
    ROAD_Y_MAX,
    ROAD_Y_MIN,
)


def plot_results(t_x, t_u, simX, simU, v_ref, v_H):

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    # Trajectory
    axes[0, 0].plot(
        simX[:, 0],
        simX[:, 1],
        label="Ego",
    )

    axes[0, 0].axhline(
        ROAD_Y_MIN,
        color="black",
        label="Road boundaries",
    )

    axes[0, 0].axhline(
        ROAD_Y_MAX,
        color="black",
    )

    axes[0, 0].axhline(
        2.0,
        color="black",
        linestyle="--",
        label="Lane divider",
    )

    axes[0, 0].axhline(
        0.0,
        color="gray",
        linestyle=":",
        label="Reference",
    )

    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    axes[0, 0].set_title("Trajectory")
    axes[0, 0].legend()

    # Speed
    axes[0, 1].plot(
        t_x,
        simX[:, 3],
        label="Ego",
    )

    axes[0, 1].axhline(
        v_ref,
        color="black",
        linestyle="--",
        label="Reference",
    )

    axes[0, 1].axhline(
        v_H,
        color="gray",
        linestyle=":",
        label="Human",
    )

    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Speed [m/s]")
    axes[0, 1].legend()

    # Heading
    axes[1, 0].plot(
        t_x,
        np.rad2deg(simX[:, 2]),
    )

    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Heading [deg]")

    # Steering angle
    axes[1, 1].plot(
        t_x,
        np.rad2deg(simX[:, 4]),
    )

    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Steering angle [deg]")

    # Steering rate
    axes[2, 0].step(
        t_u,
        np.rad2deg(simU[:, 0]),
        where="post",
    )

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_ylabel("Steering rate [deg/s]")

    # Acceleration
    axes[2, 1].step(
        t_u,
        simU[:, 1],
        where="post",
    )

    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_ylabel("Acceleration [m/s²]")

    for ax in axes.flat:
        ax.grid(True)

    fig.tight_layout()

    return fig, axes


def animate_simulation(
    t,
    simX,
    X_H_initial,
    Y_H_initial,
    v_H,
    save=False,
    filename="simulation.mp4",
    dpi=100,
):

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    # Road
    ax.axhline(
        ROAD_Y_MIN,
        color="black",
        linewidth=2,
        label="Road boundaries",
    )

    ax.axhline(
        ROAD_Y_MAX,
        color="black",
        linewidth=2,
    )

    ax.axhline(
        2.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Lane divider",
    )

    ax.axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Reference",
    )

    ax.set_ylim(
        ROAD_Y_MIN - 1.0,
        ROAD_Y_MAX + 1.0,
    )

    # Ego rectangle
    ego_corners = np.array([
        [-EGO_LENGTH / 2.0, -EGO_WIDTH / 2.0],
        [ EGO_LENGTH / 2.0, -EGO_WIDTH / 2.0],
        [ EGO_LENGTH / 2.0,  EGO_WIDTH / 2.0],
        [-EGO_LENGTH / 2.0,  EGO_WIDTH / 2.0],
    ])

    ego_vehicle = Polygon(
        ego_corners,
        closed=True,
        facecolor="tab:blue",
        edgecolor="black",
        alpha=0.7,
        label="Ego",
    )

    ax.add_patch(ego_vehicle)

    # Human rectangle
    human_corners = np.array([
        [-HUMAN_LENGTH / 2.0, -HUMAN_WIDTH / 2.0],
        [ HUMAN_LENGTH / 2.0, -HUMAN_WIDTH / 2.0],
        [ HUMAN_LENGTH / 2.0,  HUMAN_WIDTH / 2.0],
        [-HUMAN_LENGTH / 2.0,  HUMAN_WIDTH / 2.0],
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

    # Double-circle radii
    rho_E = np.hypot(
        EGO_LENGTH / 4.0,
        EGO_WIDTH / 2.0,
    )

    rho_H = np.hypot(
        HUMAN_LENGTH / 4.0,
        HUMAN_WIDTH / 2.0,
    )

    # Ego circles
    ego_front_circle = Circle(
        (0.0, 0.0),
        radius=rho_E,
        fill=False,
        edgecolor="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label="Ego circles",
    )

    ego_rear_circle = Circle(
        (0.0, 0.0),
        radius=rho_E,
        fill=False,
        edgecolor="tab:blue",
        linestyle="--",
        linewidth=1.5,
    )

    ax.add_patch(ego_front_circle)
    ax.add_patch(ego_rear_circle)

    # Human circles
    human_front_circle = Circle(
        (0.0, 0.0),
        radius=rho_H,
        fill=False,
        edgecolor="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="Human circles",
    )

    human_rear_circle = Circle(
        (0.0, 0.0),
        radius=rho_H,
        fill=False,
        edgecolor="tab:orange",
        linestyle="--",
        linewidth=1.5,
    )

    ax.add_patch(human_front_circle)
    ax.add_patch(human_rear_circle)

    # Ego trajectory
    trajectory, = ax.plot(
        [],
        [],
        color="tab:blue",
        linewidth=1.5,
        label="Trajectory",
    )

    # Vehicle centers
    ego_center, = ax.plot(
        [],
        [],
        "ko",
        markersize=4,
    )

    human_center, = ax.plot(
        [],
        [],
        "ko",
        markersize=4,
    )

    # Ego heading
    heading, = ax.plot(
        [],
        [],
        color="black",
        linewidth=2,
    )

    title = ax.set_title("")

    ax.legend(loc="upper right")


    def update(i):

        X_E, Y_E, psi_E, v_E, delta_E = simX[i]

        # Human position
        X_H = X_H_initial + v_H * t[i]
        Y_H = Y_H_initial

        # Ego rotation matrix
        rotation = np.array([
            [np.cos(psi_E), -np.sin(psi_E)],
            [np.sin(psi_E),  np.cos(psi_E)],
        ])

        # Ego rectangle
        ego_vehicle.set_xy(
            ego_corners @ rotation.T
            + np.array([X_E, Y_E])
        )

        # Human rectangle
        human_vehicle.set_xy(
            human_corners
            + np.array([X_H, Y_H])
        )

        # Ego double-circle centers
        ego_front_x = (
            X_E
            + EGO_LENGTH / 4.0 * np.cos(psi_E)
        )

        ego_front_y = (
            Y_E
            + EGO_LENGTH / 4.0 * np.sin(psi_E)
        )

        ego_rear_x = (
            X_E
            - EGO_LENGTH / 4.0 * np.cos(psi_E)
        )

        ego_rear_y = (
            Y_E
            - EGO_LENGTH / 4.0 * np.sin(psi_E)
        )

        ego_front_circle.center = (
            ego_front_x,
            ego_front_y,
        )

        ego_rear_circle.center = (
            ego_rear_x,
            ego_rear_y,
        )

        # Human double-circle centers
        human_front_x = (
            X_H
            + HUMAN_LENGTH / 4.0
        )

        human_rear_x = (
            X_H
            - HUMAN_LENGTH / 4.0
        )

        human_front_circle.center = (
            human_front_x,
            Y_H,
        )

        human_rear_circle.center = (
            human_rear_x,
            Y_H,
        )

        # Centers
        ego_center.set_data(
            [X_E],
            [Y_E],
        )

        human_center.set_data(
            [X_H],
            [Y_H],
        )

        # Heading
        heading.set_data(
            [
                X_E,
                X_E + EGO_LENGTH / 2.0 * np.cos(psi_E),
            ],
            [
                Y_E,
                Y_E + EGO_LENGTH / 2.0 * np.sin(psi_E),
            ],
        )

        # Trajectory
        trajectory.set_data(
            simX[:i + 1, 0],
            simX[:i + 1, 1],
        )

        # Camera
        ax.set_xlim(
            min(X_E, X_H) - 12.0,
            max(X_E, X_H) + 20.0,
        )

        # Minimum distance between double circles
        ego_circle_centers = np.array([
            [ego_front_x, ego_front_y],
            [ego_rear_x, ego_rear_y],
        ])

        human_circle_centers = np.array([
            [human_front_x, Y_H],
            [human_rear_x, Y_H],
        ])

        min_center_distance = np.inf

        for ego_circle_center in ego_circle_centers:
            for human_circle_center in human_circle_centers:

                distance = np.linalg.norm(
                    ego_circle_center
                    - human_circle_center
                )

                min_center_distance = min(
                    min_center_distance,
                    distance,
                )

        safety_distance = rho_E + rho_H

        title.set_text(
            f"t = {t[i]:.2f} s   |   "
            f"v_E = {v_E:.2f} m/s   |   "
            f"v_H = {v_H:.2f} m/s   |   "
            f"delta = {np.rad2deg(delta_E):.1f}°\n"
            f"Minimum circle distance = "
            f"{min_center_distance:.2f} m   |   "
            f"Required = {safety_distance:.2f} m"
        )

        return (
            ego_vehicle,
            human_vehicle,
            ego_front_circle,
            ego_rear_circle,
            human_front_circle,
            human_rear_circle,
            trajectory,
            ego_center,
            human_center,
            heading,
            title,
        )


    update(0)

    frame_step = 2
    frames = range(0, len(t), frame_step)
    animation = FuncAnimation(
        fig,
        update,
        frames= frames,
        interval=1000.0 * (t[1] - t[0]) * frame_step,
        repeat=False,
        blit=False,
    )

    if save:

        video_path = (
            Path(__file__).resolve().parent
            / filename
        )

        animation.save(
            str(video_path),
            writer="ffmpeg",
            fps=1.0 / (t[1] - t[0]),
            dpi=dpi,
        )

        print(f"Animation saved to: {video_path}")

    return animation