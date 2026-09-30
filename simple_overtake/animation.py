import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

def animate_simulation(
    t, simX, X_H_initial, v_H,
    length=4.68, width=2.2, human_length=4.28, human_width=1.8,
):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    # Reference path and traveled trajectory
    ax.axhline(0.0, color="black", linestyle="--", label="Reference")
    trajectory, = ax.plot([], [], color="tab:blue", label="Ego trajectory")

    # Vehicle outline relative to its center
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

    human_corners = np.array([
        [-human_length / 2, -human_width / 2],
        [ human_length / 2, -human_width / 2],
        [ human_length / 2,  human_width / 2],
        [-human_length / 2,  human_width / 2],
    ])
    human = Polygon(
        human_corners, closed=True, facecolor="tab:orange",
        edgecolor="black", alpha=0.7, label="Human",
    )
    ax.add_patch(human)
    human_heading, = ax.plot([], [], color="black", linewidth=2)

    heading, = ax.plot([], [], color="black", linewidth=2)
    title = ax.set_title("")
    ax.legend(loc="upper right")

    # Keep the reference and the full lateral motion visible
    ax.set_ylim(
        min(0.0, simX[:, 1].min()) - 5.0,
        max(0.0, simX[:, 1].max()) + 5.0,
    )

    def update(i):

        X, Y, psi, v, delta = simX[i]
        X_H = X_H_initial + v_H * t[i]
        human.set_xy(human_corners + np.array([X_H, 0.0]))
        human_heading.set_data([X_H, X_H + human_length / 2], [0.0, 0.0])

        # Rotate the rectangle and translate it to the vehicle position
        rotation = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)],
        ])

        vehicle.set_xy(corners @ rotation.T + np.array([X, Y]))

        # Line from the center toward the front of the vehicle
        heading.set_data(
            [X, X + length / 2 * np.cos(psi)],
            [Y, Y + length / 2 * np.sin(psi)],
        )

        trajectory.set_data(simX[:i + 1, 0], simX[:i + 1, 1])

        ax.set_xlim(
            min(X - 10.0, X_H - human_length / 2 - 5.0),
            max(X + 20.0, X_H + human_length / 2 + 5.0),
        )

        title.set_text(
            f"t = {t[i]:.2f} s   |   "
            f"v = {v:.2f} m/s   |   "
            f"delta = {np.rad2deg(delta):.1f}°"
        )

        return vehicle, human, heading, human_heading, trajectory, title

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
