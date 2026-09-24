import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

from ego_model import ego_model

def acados_settings(Tf, N, lf, lr, x0, v_ref, human_pos):

    # Create ocp
    ocp = AcadosOcp()

    # Model
    model = ego_model(lf, lr)
    X_H = SX.sym("X_H")
    Y_H = SX.sym("Y_H")
    model.p = vertcat(X_H, Y_H)

    r_E = 0.5 * np.hypot(lf+lr, 2.20)
    r_H = 0.5 * np.hypot(4.28, 1.80)
    safety_d = r_E + r_H

    X_E = model.x[0]
    Y_E = model.x[1]

    separation = (X_E - X_H)**2 + (Y_E - Y_H)**2 - safety_d**2

    model.con_h_expr = vertcat(separation)
    model.con_h_expr_e = vertcat(separation)

    ocp.model = model
    ocp.parameters_values = np.asarray(human_pos, dtype=float)

    # Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # Cost
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Q = np.diag([0.0, 20.0, 200.0, 20.0, 50.0])
    R = np.diag([1.0, 0.2])

    Qe = Q.copy()

    W = np.zeros((ny, ny))
    W[:nx, :nx] = Q
    W[nx:, nx:] = R

    ocp.cost.W = W
    ocp.cost.W_e = Qe

    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.array([
        0.0, 0.0, 0.0, v_ref, 0.0, 0.0, 0.0
    ])

    ocp.cost.yref_e = np.array([
        0.0, 0.0, 0.0, v_ref, 0.0
    ])

    # Input bounds
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-0.087, -7.0])
    ocp.constraints.ubu = np.array([0.087, 2.5])

    # State bounds
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx = np.array([0.0, -0.4])
    ocp.constraints.ubx = np.array([10.0, 0.4])

    # Terminal state bounds
    ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()

    # Nonlinear separation bounds
    ocp.constraints.lh = np.array([0.0])
    ocp.constraints.uh = np.array([1e15])

    # Terminal separation bounds
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([1e15])

    # Initial condition
    ocp.constraints.x0 = np.asarray(x0, dtype=float)

    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3

    # Create solver
    solver = AcadosOcpSolver(ocp)

    return model, solver