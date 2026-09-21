import time

import matplotlib.pyplot as plt
import numpy as np

from acados_template import AcadosSim, AcadosSimSolver
from acados_settings import acados_settings
from animation import animate_simulation
from pathlib import Path

# Vehicle parameters
lf = 1.2
lr = 1.5

# MPC and simulation settings
Tf = 1.0
N = 50
dt = Tf / N

Tsim = 10.0
Nsim = int(round(Tsim / dt))

v_ref = 5.0

# Initial state: [X, Y, psi, v, delta]
x0 = np.array([0.0, 2.0, 0.0, 2.0, 0.0])

# MPC solver
model, acados_solver = acados_settings(
    Tf, N, lf, lr, x0, v_ref
)

nx = model.x.rows()     # 5
nu = model.u.rows()     # 2

# Vehicle simulator
sim = AcadosSim()
sim.model = model

sim.solver_options.T = dt
sim.solver_options.integrator_type = "ERK"
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 3

acados_integrator = AcadosSimSolver(sim)

# Inizializzazione
for j in range(N + 1):
    acados_solver.set(j, "x", x0)

for j in range(N):
    acados_solver.set(j, "u", np.zeros(nu))

simX = np.zeros((Nsim + 1, nx))     # 501 x 5
simU = np.zeros((Nsim, nu))         # 500 x 5
solve_time = np.zeros(Nsim)         # 500 x 1

# Simulation
for i in range(Nsim):

    x_current = simX[i, :]

    acados_solver.set(0, "lbx", x_current)
    acados_solver.set(0, "ubx", x_current)

    # Solve ocp
    start = time.perf_counter()
    status = acados_solver.solve()
    solve_time[i] = time.perf_counter() - start

    if status != 0:
        acados_solver.print_statistics()
        raise RuntimeError(
            f"OCP solver failed at step {i}, t = {i * dt:.3f} s, status = {status}"
            )

    # First optimal input
    u0 = acados_solver.get(0, "u")
    simU[i, :] = u0

    # Vehicle sim
    acados_integrator.set("x", x_current)
    acados_integrator.set("u", u0)

    status = acados_integrator.solve()

    if status != 0:
        raise RuntimeError(f"Integrator failed at step {i}, status = {status}")

    simX[i + 1, :] = acados_integrator.get("x")


# Print results
print(f"\n\nMean OCP solve time: {1e3 * solve_time.mean():.3f} ms")
print(f"Maximum OCP solve time: {1e3 * solve_time.max():.3f} ms")

# Plot results
t_x = np.arange(Nsim + 1) * dt
t_u = np.arange(Nsim) * dt

fig, axes = plt.subplots(3, 2, figsize=(12, 9))

axes[0, 0].plot(simX[:, 0], simX[:, 1], label="Ego")
axes[0, 0].axhline(0.0, color="black", linestyle="--", label="Reference")
axes[0, 0].set_xlabel("X [m]")
axes[0, 0].set_ylabel("Y [m]")
axes[0, 0].set_title("Trajectory")
axes[0, 0].legend()

axes[0, 1].plot(t_x, simX[:, 3], label="Ego")
axes[0, 1].axhline(
    v_ref, color="black", linestyle="--", label="Reference"
)
axes[0, 1].set_ylabel("Speed [m/s]")
axes[0, 1].legend()

axes[1, 0].plot(t_x, np.rad2deg(simX[:, 2]))
axes[1, 0].set_ylabel("Heading [deg]")

axes[1, 1].plot(t_x, np.rad2deg(simX[:, 4]))
axes[1, 1].set_ylabel("Steering angle [deg]")

axes[2, 0].step(
    t_u, np.rad2deg(simU[:, 0]), where="post"
)
axes[2, 0].set_ylabel("Steering rate [deg/s]")

axes[2, 1].step(t_u, simU[:, 1], where="post")
axes[2, 1].set_ylabel("Acceleration [m/s²]")

for ax in axes.flat:
    ax.grid(True)

for ax in [axes[0, 1], *axes[1, :], *axes[2, :]]:
    ax.set_xlabel("Time [s]")

fig.tight_layout()

animation = animate_simulation(t_x, simX)
video_path = Path(__file__).resolve().parent / "simulation.mp4"
animation.save(
    str(video_path),
    writer="ffmpeg",
    fps=1.0 / (t_x[1] - t_x[0]),
    dpi=150,
)

plt.show()