import time
from datetime import datetime

import numpy as np

from acados_template import AcadosSim, AcadosSimSolver
from acados_settings import acados_settings, LANE_CENTER_1
from animation import animate_simulation, plot_results, OcpCostDiagnostics, hj_prediction_diagnostics, COST_GROUPS
from pathlib import Path
from brt_utils import BRTinterpolator, hj_coefficients
from hj_diagnostics import HJDiagnostics
import matplotlib.pyplot as plt

# BRT parameters
METRICS = ["euclidean", "dce", "ttc", "eggert"]
ALPHA_HJ = 1.5
HJ_STAGES = 3
brt_dir = Path(__file__).resolve().parents[1] / "brt"
brts = [BRTinterpolator(brt_dir / f"{metric}.npz") for metric in METRICS]
N_METRICS = len(METRICS)


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
    v_ref = scenario.get("v_ref", 8.0)
    x0 = np.array(scenario.get("x0", [0.0, 0.0, 0.0, 8.0, 0.0]), dtype=float)

    # Human
    X_H_initial = scenario.get("X_H_initial", 50.0)
    Y_H_initial = scenario.get("Y_H_initial", 0.0)
    v_H = scenario.get("v_H", 2.0)
    human_pos = np.array([X_H_initial, Y_H_initial])

    # MPC solver
    model, acados_solver = acados_settings(
        Tf, N, lf, lr, x0, v_ref, human_pos, weights=weights, n_metrics=N_METRICS,
    )
    nx = model.x.rows()     # 5
    nu = model.u.rows()     # 2
    cost_diagnostic = OcpCostDiagnostics(model, acados_solver, N, dt_ocp) if make_animation else None

    # Vehicle simulator
    sim = AcadosSim()
    sim.model = model
    sim.parameter_values = np.concatenate([
        human_pos,
        np.tile([0.0, 0.0, 1.0], N_METRICS),
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
    simV = np.full((Nsim, N_METRICS), np.nan)
    simGradV = np.full((Nsim, N_METRICS, 6), np.nan)

    simInsideGrid = np.zeros((Nsim, N_METRICS), dtype=bool)
    simHJActive = np.zeros((Nsim, N_METRICS), dtype=bool)

    simHJResidual = np.full((Nsim, N_METRICS), np.nan)
    simHJSlack = np.zeros((Nsim, N_METRICS))
    simHJSlackUpper = np.zeros((Nsim, N_METRICS))
    simOcpCost = np.full(Nsim, np.nan) if make_plots or make_animation else None
    simCostComponents = {name: np.full(Nsim, np.nan) for name in COST_GROUPS} if make_animation else None
    simHJMinResidual = np.full(Nsim, np.nan) if make_animation else None
    simHJMaxSlack = np.full(Nsim, np.nan) if make_animation else None
    simX_pred = np.empty((Nsim, N + 1, nx)) if make_animation else None

    hj_diagnostics = HJDiagnostics(
        metrics=METRICS,
        dt=dt,
        stages=HJ_STAGES,
        stage_dt=dt_ocp,
        u_lower=(-0.087, -7.0),
        u_upper=(0.087, 2.5),
    )

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
        M_HJ = np.zeros((N_METRICS, 2))
        b_HJ = np.ones(N_METRICS)
        for k, brt in enumerate(brts):
            value, gradient, inside = brt.evaluate(relative_state)
            simV[i, k] = value
            simGradV[i, k] = gradient
            simInsideGrid[i, k] = inside
            simHJActive[i, k] = inside
            if inside:
                M_HJ[k], b_HJ[k] = hj_coefficients(relative_state, gradient, brt.dynamics_parameters)
                b_HJ[k] += ALPHA_HJ * value
        hj_parameters_active = np.column_stack([M_HJ, b_HJ]).ravel()
        hj_parameters_inactive = np.tile([0.0, 0.0, 1.0], N_METRICS)

        # Human prediction over MPC horizon
        human_prediction = X_H_current + v_H * np.arange(N + 1) * dt_ocp
        for j in range(N + 1):
            if j < HJ_STAGES:
                hj_parameters = hj_parameters_active
            else:
                hj_parameters = hj_parameters_inactive
            parameters = np.concatenate([
                [human_prediction[j], Y_H_initial],
                hj_parameters,
            ])
            acados_solver.set(j, "p", parameters)
            if j < N:
                X_ref = X_E + v_ref * j * dt_ocp
                yref = np.array([
                    X_ref,
                    LANE_CENTER_1,
                    0.0,
                    v_ref,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])
                acados_solver.cost_set(j, "yref", yref)

        # Terminal reference
        X_ref_e = X_E + v_ref * Tf
        yref_e = np.array([
            X_ref_e,
            LANE_CENTER_1,
            0.0,
            v_ref,
            0.0,
        ])
        acados_solver.cost_set(N, "yref", yref_e)

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

        # Predicted MPC trajectory
        if simX_pred is not None: simX_pred[i] = np.asarray(acados_solver.get_flat("x"), dtype=float).reshape(N + 1, nx)

        # First optimal input
        u0 = acados_solver.get(0, "u")
        simU[i, :] = u0
        active = simHJActive[i]
        residual = M_HJ @ u0 + b_HJ
        simHJResidual[i, active] = residual[active]
        simHJSlack[i] = np.asarray(acados_solver.get(0, "sl"), dtype=float).ravel()
        simHJSlackUpper[i] = np.asarray(acados_solver.get(0, "su"), dtype=float).ravel()
        hj_diagnostics.record_solver(
            acados_solver,
            V=simV[i],
            alpha=ALPHA_HJ,
            available=simHJActive[i],
            state=x_current,
            relative_state=relative_state,
            time=i * dt,
            solver_status=status,
            solve_time=solve_time[i],
        )
        if simOcpCost is not None:
            if cost_diagnostic is not None:
                total_cost, components = cost_diagnostic.evaluate(acados_solver)
                simOcpCost[i] = total_cost
                for name, value in components.items():
                    simCostComponents[name][i] = value
            else:
                simOcpCost[i] = acados_solver.get_cost()

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
    # 
    # 
    # 
    if simCostComponents is not None:
        reconstructed_cost = sum(simCostComponents.values())
        max_cost_error = np.max(np.abs(simOcpCost - reconstructed_cost))
        print(
            f"Maximum cost decomposition error: "
            f"{max_cost_error:.6e}"
        )
    # 
    # 
    #  
    diagnostics_data = hj_diagnostics.arrays()
    contributions = hj_diagnostics.contributions()

    residual_matches = np.allclose(
        contributions["residual"][:, 0, :],
        simHJResidual,
        rtol=1e-8,
        atol=1e-10,
        equal_nan=True,
    )

    print(f"Residui HJ registrati correttamente: {residual_matches}")

    if not residual_matches:
        raise RuntimeError("Residui diagnostici diversi da quelli del main")

    for k, metric in enumerate(METRICS):
        active = diagnostics_data["active"][:, :, k]
        violations = contributions["soft_violated"][:, :, k]

        print(
            f"{metric}: "
            f"{np.count_nonzero(active)} condizioni attive, "
            f"{np.count_nonzero(violations)} violazioni oltre tolleranza"
        )


    run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_"
        + "_".join(METRICS)
    )

    results_directory = Path(__file__).resolve().parent / "results"

    diagnostics_path = hj_diagnostics.save(
        results_directory / f"{run_name}_hj.npz"
    )

    np.savez_compressed(
        results_directory / f"{run_name}_simulation.npz",
        simX=simX,
        simU=simU,
        dt=dt,
        v_ref=v_ref,
        X_H_initial=X_H_initial,
        Y_H_initial=Y_H_initial,
        v_H=v_H,
    )

    print(f"Diagnostica salvata in: {diagnostics_path}")

    if make_plots:
        plot_results(
            dt,
            simX,
            simU,
            v_ref,
            v_H,
            simV,
            simHJActive,
        )
        fig, ax = plt.subplots(
            figsize=(10,4),
            layout="constrained",
        )
        t_cost = np.arange(Nsim) * dt
        ax.plot(
            t_cost,
            simOcpCost,
            label="Total OCP cost",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Cost")
        ax.set_title("Optimal MPC cost")
        ax.grid(True)
        ax.legend()
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
            simX_pred=simX_pred,
            simU=simU,
            v_ref=v_ref,
            simV=simV,
            simHJActive=simHJActive,
            simHJResidual=simHJResidual,
            simHJSlack=simHJSlack,
            simHJSlackUpper=simHJSlackUpper,
            simHJMinResidual=simHJMinResidual,
            simHJMaxSlack=simHJMaxSlack,
            simOcpCost=simOcpCost,
            simCostComponents=simCostComponents,
            brt=brt,
        )
    if make_plots or make_animation:
        print(f"\nMean OCP solve time: {1e3 * solve_time.mean():.3f} ms")
        print(f"Maximum OCP solve time: {1e3 * solve_time.max():.3f} ms")
        print(
            f"Stati dentro la griglia BRT: "
            f"{simInsideGrid.sum()}/{Nsim}"
        )
        plt.close("all")
    ################################

    return {
        "success": True,
        "simX": simX,
        "simU": simU,
        "simX_pred": simX_pred,
        "simV": simV,
        "simGradV": simGradV,
        "simInsideGrid": simInsideGrid,
        "simHJActive": simHJActive,
        "simHJResidual": simHJResidual,
        "simHJSlack": simHJSlack,
        "simHJSlackUpper": simHJSlackUpper,
        "simOcpCost": simOcpCost,
        "solve_time": solve_time,
        "dt": dt,
        "v_ref": v_ref,
        "v_H": v_H,
        "X_H_initial": X_H_initial,
        "Y_H_initial": Y_H_initial,
        "scenario": scenario,
        "metrics": METRICS.copy(),
        "hj_diagnostic": hj_diagnostics,
        "diagnostics_path": diagnostics_path,
    }



if __name__ == "__main__":
    result = run_simulation(
        make_plots=False,
        make_animation=False,
    )

    t_x = np.arange(len(result["simX"])) * result["dt"]
    t_u = np.arange(len(result["simU"])) * result["dt"]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(11, 10),
        sharex=True,
        layout="constrained",
    )

    axes[0].plot(t_x, result["simX"][:, 3], label="Ego")
    axes[0].axhline(
        result["v_ref"],
        color="black",
        linestyle="--",
        label="Reference",
    )
    axes[0].set_ylabel("Velocity [m/s]")

    for k, metric in enumerate(result["metrics"]):
        axes[1].plot(t_u, result["simV"][:, k], label=metric)
        axes[2].plot(
            t_u,
            result["simHJResidual"][:, k],
            label=metric,
        )
        axes[3].plot(
            t_u,
            result["simHJSlack"][:, k],
            label=metric,
        )

        count = np.count_nonzero(result["simInsideGrid"][:, k])
        print(f"{metric}: dentro griglia {count}/{len(t_u)}")

    axes[1].set_ylabel("BRT value")
    axes[2].set_ylabel("HJ residual")
    axes[3].set_ylabel("Lower slack")
    axes[3].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[2].axhline(0.0, color="black", linestyle="--")

    plt.show()