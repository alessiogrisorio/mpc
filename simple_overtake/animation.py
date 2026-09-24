import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

def animate_simulation(t, simX, length=4.68, width=2.2):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)

    # Reference path and traveled trajectory
    ax.axhline(0.0, color="black", linestyle="--", label="Reference")
    trajectory, = ax.plot([], [], color="tab:blue", label="Trajectory")

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
    )
    ax.add_patch(vehicle)

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

        ax.set_xlim(X - 10.0, X + 20.0)

        title.set_text(
            f"t = {t[i]:.2f} s   |   "
            f"v = {v:.2f} m/s   |   "
            f"delta = {np.rad2deg(delta):.1f}°"
        )

        return vehicle, heading, trajectory, title

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