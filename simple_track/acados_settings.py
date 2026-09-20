import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from ego_model import ego_model

def acados_settings(Tf, N, lf, lr, x0, v_ref):

    # Create ocp
    ocp = AcadosOcp()

    # Model
    model = ego_model(lf, lr)
    ocp.model = model

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

    Q = np.diag([0.0, 10.0, 10.0, 1.0, 1.0])
    R = np.diag([1.0, 0.1])

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
    ocp.constraints.idxbx_e = ocp.constraints.idxbu.copy()
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()

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