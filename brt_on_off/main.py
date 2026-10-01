import time

import numpy as np

from acados_template import AcadosSim, AcadosSimSolver
from acados_settings import acados_settings, EGO_LENGTH, HUMAN_LENGTH, safety_r
from animation import animate_simulation, plot_results
from pathlib import Path
from brt_utils import BRTinterpolator, hj_coefficients
import matplotlib.pyplot as plt

# BRT parameters
brt_type = "euclidean.npz"
brt_path = Path(__file__).resolve().parents[1] / "brt" / brt_type
brt = BRTinterpolator(brt_path)

# Vehicle parameters
lf = 1.2
lr = 1.5

# MPC and simulation settings
Tf = 3.0
f_mpc = 50
dt_ctrl = 1.0 / f_mpc
N = int(round(Tf / dt_ctrl))
dt_ocp = Tf / N

# Simulation settings
Tsim = 20.0
f_sim = 50
dt = 1.0 / f_sim
Nsim = int(round(Tsim / dt))

# Ego: [X, Y, psi, v, delta]
v_ref = 5.0
x0 = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
pass_margin = EGO_LENGTH / 4.0 + HUMAN_LENGTH / 4.0 + safety_r

# Human
X_H_initial = 12.0
Y_H_initial = 0.0
v_H = 2.0
human_pos = np.array([X_H_initial, Y_H_initial])

# MPC solver
model, acados_solver = acados_settings(
    Tf, N, lf, lr, x0, v_ref, human_pos
)
nx = model.x.rows()     # 5
nu = model.u.rows()     # 2

# Vehicle simulator
return_on = 0.0
sim = AcadosSim()
sim.model = model
sim.parameter_values = np.array([X_H_initial, Y_H_initial, return_on])
sim.solver_options.T = dt
sim.solver_options.integrator_type = "ERK"
sim.solver_options.num_stages = 4
sim.solver_options.num_steps = 3
acados_integrator = AcadosSimSolver(sim)


# Inizializzazione
x_guess = x0.copy()
u_guess = np.zeros(2)
acados_solver.set(0, "x", x_guess)

for j in range(N):
    acados_solver.set(j, "u", u_guess)
    acados_integrator.set("x", x_guess)
    acados_integrator.set("u", u_guess)
    status_sim = acados_integrator.solve()
    if status_sim != 0:
        raise RuntimeError(f"Initial guess integration failed: {status_sim}")
    x_guess = acados_integrator.get("x")
    acados_solver.set(j + 1, "x", x_guess)

# Simulation data
simX = np.zeros((Nsim + 1, nx))     # 501 x 5
simU = np.zeros((Nsim, nu))         # 500 x 2
solve_time = np.zeros(Nsim)         # 500 x 1
simX[0, :] = x0
simV = np.full(Nsim, np.nan)                # 500 x 1
simGradV = np.full((Nsim, 6), np.nan)       # 500 x 6
simInsideGrid = np.zeros(Nsim, dtype=bool)  # 500 x 1

# Simulation
for i in range(Nsim):

    x_current = simX[i, :]

    # Current ego state
    acados_solver.set(0, "lbx", x_current)
    acados_solver.set(0, "ubx", x_current)

    # Aggiorno posizione human
    X_H_current = X_H_initial + v_H * i * dt
    X_E, Y_E, PSI_E, V_E, DELTA_E = x_current
    relative_state = np.array([
        np.cos(PSI_E) * (X_H_current - X_E) + np.sin(PSI_E) * (Y_H_initial - Y_E),
        -np.sin(PSI_E) * (X_H_current - X_E) + np.cos(PSI_E) * (Y_H_initial - Y_E),
        -PSI_E,
        v_H,
        DELTA_E,
        V_E,
    ])

    # Valutazione BRT
    simV[i], simGradV[i], simInsideGrid[i] = brt.evaluate(relative_state)
    if simInsideGrid[i]:
        M_HJ, b_HJ = hj_coefficients(relative_state, simGradV[i], brt.dynamics_parameters)
    else:
        M_HJ = np.zeros(2)
        b_HJ = 1.0

    # Activete return to right lane
    if x_current[0] > X_H_current + pass_margin:
        return_on = 1.0

    # Human prediction over MPC horizon
    for j in range(N + 1):
        human_prediction = np.array([X_H_current + v_H * j * dt_ocp, Y_H_initial, return_on])
        acados_solver.set(j, "p", human_prediction)

    # Solve ocp
    start = time.perf_counter()
    status = acados_solver.solve()
    solve_time[i] = time.perf_counter() - start
    if status not in (0, 2):
        acados_solver.print_statistics()
        raise RuntimeError(
            f"OCP solver failed at step {i}, t = {i * dt:.3f} s, status = {status}"
            )

    # First optimal input
    u0 = acados_solver.get(0, "u")
    simU[i, :] = u0

    # Ego simulator
    acados_integrator.set("x", x_current)
    acados_integrator.set("u", u0)
    status = acados_integrator.solve()

    if status not in (0, 2):
        raise RuntimeError(f"Integrator failed at step {i}, status = {status}")
    simX[i + 1, :] = acados_integrator.get("x")


# Print results
print(f"\n\nMean OCP solve time: {1e3 * solve_time.mean():.3f} ms")
print(f"Maximum OCP solve time: {1e3 * solve_time.max():.3f} ms")

t_x = np.arange(Nsim + 1) * dt
t_u = np.arange(Nsim) * dt

fig_brt, ax_brt = plt.subplots(figsize=(10, 4))

ax_brt.plot(t_u, simV, label="BRT")
ax_brt.axhline(0.0, color="black", linestyle="--", label="V = 0")

ax_brt.set_xlabel("Time [s]")
ax_brt.set_ylabel("V")
ax_brt.grid(True)
ax_brt.legend()
fig_brt.tight_layout()

print(
    f"Stati dentro la griglia BRT: "
    f"{simInsideGrid.sum()}/{Nsim}"
)

plot_results(
    t_x,
    t_u,
    simX,
    simU,
    v_ref,
    v_H,
)

animation = animate_simulation(
    t_x,
    simX,
    X_H_initial,
    Y_H_initial,
    v_H,
    save=True,
)
plt.show()