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
V_thr = 1
HJ_STAGES = 3

def run_simulation(
    weights=None,
    scenario=None,
    make_plots=True,
    make_animation=True,
    raise_on_fail=True,
):
    if scenario is None:
        scenario = {}

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
    Tsim = scenario.get("Tsim", 20.0)
    f_sim = 50
    dt = 1.0 / f_sim
    Nsim = int(round(Tsim / dt))

    # Ego: [X, Y, psi, v, delta]
    v_ref = scenario.get("v_ref", 5.0)
    x0 = np.array(scenario.get("x0", [0.0, 0.0, 0.0, 5.0, 0.0]), dtype=float)
    pass_margin = EGO_LENGTH / 4.0 + HUMAN_LENGTH / 4.0 + safety_r

    # Human
    X_H_initial = scenario.get("X_H_initial", 20.0)
    Y_H_initial = scenario.get("Y_H_initial", 0.0)
    v_H = scenario.get("v_H", 2.0)
    human_pos = np.array([X_H_initial, Y_H_initial])

    # MPC solver
    model, acados_solver = acados_settings(
        Tf, N, lf, lr, x0, v_ref, human_pos, weights=weights
    )
    nx = model.x.rows()     # 5
    nu = model.u.rows()     # 2

    # Vehicle simulator
    return_on = 0.0
    sim = AcadosSim()
    sim.model = model
    sim.parameter_values = np.array([
        X_H_initial,
        Y_H_initial,
        return_on,
        0.0,
        0.0,
        1.0,
    ])
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
            if raise_on_fail:
                raise RuntimeError(f"Initial guess integration failed: {status_sim}")
            return {
                "success": False,
                "failure_phase": "initial_guess",
                "integrator_status": status_sim,
            }
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
    simHJActive = np.zeros(Nsim, dtype=bool)        # 500 x 1
    simHJResidual = np.full(Nsim, np.nan)
    simHJSlack = np.zeros(Nsim)

    # Simulation
    for i in range(Nsim):

        x_current = simX[i, :]
        acados_solver.set(0, "x", x_current)

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
        hj_active = simInsideGrid[i] and simV[i] <= V_thr
        simHJActive[i] = hj_active
        if hj_active:
            M_HJ, b_HJ = hj_coefficients(relative_state, simGradV[i], brt.dynamics_parameters)
        else:
            M_HJ = np.zeros(2)
            b_HJ = 1.0


        # Activete return to right lane
        if x_current[0] > X_H_current + pass_margin:
            return_on = 1.0

        # Human prediction over MPC horizon
        for j in range(N + 1):
            if hj_active and j < HJ_STAGES:
                hj_parameters = [M_HJ[0], M_HJ[1], b_HJ]
            else:
                hj_parameters = [0.0, 0.0, 1.0]
            parameters = np.array([
                X_H_current + v_H * j * dt_ocp,
                Y_H_initial,
                return_on,
                *hj_parameters,
            ])
            acados_solver.set(j, "p", parameters)

        # Solve ocp
        start = time.perf_counter()
        status = acados_solver.solve()
        solve_time[i] = time.perf_counter() - start
        if status not in (0, 2):
            if raise_on_fail:
                acados_solver.print_statistics()
                raise RuntimeError(
                    f"OCP solver failed at step {i}, t = {i * dt:.3f} s, status = {status}"
                    )
            return {
                "success": False,
                "failure_step": i,
                "failure_time": i * dt,
                "solver_status": status,
            }

        # First optimal input
        u0 = acados_solver.get(0, "u")
        simU[i, :] = u0
        if hj_active:
            simHJResidual[i] = M_HJ @ u0 + b_HJ
        simHJSlack[i] = acados_solver.get(0, "sl")[0]

        # Ego simulator
        acados_integrator.set("x", x_current)
        acados_integrator.set("u", u0)
        status = acados_integrator.solve()

        if status not in (0, 2):
            if raise_on_fail:
                raise RuntimeError(f"Integrator failed at step {i}, status = {status}")
            return {
                "success": False,
                "failure_step": i,
                "failure_time": i * dt,
                "integrator_status": status,
            }
        
        simX[i + 1, :] = acados_integrator.get("x")
 
    ################################  
    if make_plots:
        plot_results(
            dt,
            simX,
            simU,
            v_ref,
            v_H,
            simV,
            simHJActive,
            V_thr,
        )
    animation = None
    if make_animation:
        t_x = np.arange(len(simX)) * dt
        animation = animate_simulation(
            t_x,
            simX,
            X_H_initial,
            Y_H_initial,
            v_H,
            save=True,
        )
    if make_plots or make_animation:
        print(f"\nMean OCP solve time: {1e3 * solve_time.mean():.3f} ms")
        print(f"Maximum OCP solve time: {1e3 * solve_time.max():.3f} ms")
        print(
            f"Stati dentro la griglia BRT: "
            f"{simInsideGrid.sum()}/{Nsim}"
        )
        plt.show()
    ################################

    return {
        "success": True,
        "simX": simX,
        "simU": simU,
        "simV": simV,
        "simGradV": simGradV,
        "simInsideGrid": simInsideGrid,
        "simHJActive": simHJActive,
        "simHJResidual": simHJResidual,
        "simHJSlack": simHJSlack,
        "solve_time": solve_time,
        "dt": dt,
        "v_ref": v_ref,
        "v_H": v_H,
        "X_H_initial": X_H_initial,
        "Y_H_initial": Y_H_initial,
        "scenario": scenario,
    }

if __name__ == "__main__":
    run_simulation()