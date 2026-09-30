import numpy as np
import scipy.linalg

from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin

from ego_model import ego_model

# Vehicles dimensions
EGO_LENGTH = 4.68
EGO_WIDTH = 2.20

HUMAN_LENGTH = 4.28
HUMAN_WIDTH = 1.80

# Road dimensions
ROAD_Y_MIN = -2.0
ROAD_Y_MAX = 6.0
LANE_CENTER_1 = 0.0
LANE_CENTER_2 = 4.0
LANE_HALF_DISTANCE = (LANE_CENTER_2 - LANE_CENTER_1) / 2


def acados_settings(Tf, N, lf, lr, x0, v_ref, human_pos):

    # Create ocp
    ocp = AcadosOcp()

    # Model
    model = ego_model(lf, lr)
    X_H = SX.sym("X_H")
    Y_H = SX.sym("Y_H")
    model.p = vertcat(X_H, Y_H)

    # Double circle constraints
    X_E = model.x[0]
    Y_E = model.x[1]
    PSI_E = model.x[2]
    V_E = model.x[3]
    DELTA_E = model.x[4]
    STEERING_RATE = model.u[0]
    ACCELERATION = model.u[1]

    rho_E = np.hypot(EGO_LENGTH / 4.0, EGO_WIDTH / 2.0)
    rho_H = np.hypot(HUMAN_LENGTH / 4.0, HUMAN_WIDTH / 2.0)
    safety_r = rho_E + rho_H

    XC_EGO_FRONT = X_E + EGO_LENGTH / 4.0 * cos(PSI_E)
    XC_EGO_REAR = X_E - EGO_LENGTH / 4.0 * cos(PSI_E)
    YC_EGO_FRONT = Y_E + EGO_LENGTH / 4.0 * sin(PSI_E)
    YC_EGO_REAR = Y_E - EGO_LENGTH / 4.0 * sin(PSI_E)
    XC_HUMAN_FRONT = X_H + HUMAN_LENGTH / 4.0
    XC_HUMAN_REAR = X_H - HUMAN_LENGTH / 4.0
    YC_HUMAN_FRONT = Y_H
    YC_HUMAN_REAR = Y_H
    separation_ff = (XC_EGO_FRONT - XC_HUMAN_FRONT)**2 + (YC_EGO_FRONT - YC_HUMAN_FRONT)**2 - safety_r**2
    separation_fr = (XC_EGO_FRONT - XC_HUMAN_REAR)**2 + (YC_EGO_FRONT - YC_HUMAN_REAR)**2 - safety_r**2
    separation_rf = (XC_EGO_REAR - XC_HUMAN_FRONT)**2 + (YC_EGO_REAR - YC_HUMAN_FRONT)**2 - safety_r**2
    separation_rr = (XC_EGO_REAR - XC_HUMAN_REAR)**2 + (YC_EGO_REAR - YC_HUMAN_REAR)**2 - safety_r**2

    separation = vertcat(
        separation_ff,
        separation_fr,
        separation_rf,
        separation_rr,
    )

    lane_error = (Y_E - LANE_CENTER_1) * (Y_E - LANE_CENTER_2) / (LANE_HALF_DISTANCE**2)

    model.con_h_expr = separation
    model.con_h_expr_e = separation

    ocp.model = model
    ocp.parameter_values = np.asarray(human_pos, dtype=float)

    # Dimensions
    nx = model.x.rows()
    nu = model.u.rows()

    # Cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    model.cost_y_expr = vertcat(X_E, lane_error, PSI_E, V_E, DELTA_E, STEERING_RATE, ACCELERATION)
    model.cost_y_expr_e = vertcat(X_E, lane_error, PSI_E, V_E, DELTA_E)

    Q = np.diag([0.0, 2.0, 10.0, 50.0, 0.0])
    R = np.diag([1.0, 0.2])

    Qe = Q.copy()

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe

    ocp.cost.yref = np.array([
        0.0,        # X
        0.0,        # lane error
        0.0,        # psi
        v_ref,      # v
        0.0,        # delta
        0.0,        # steering rate
        0.0,        # acceleration
    ])

    ocp.cost.yref_e = np.array([
        0.0,        # X
        0.0,        # lane error
        0.0,        # psi
        v_ref,      # v
        0.0,        # delta
    ])

    # Input bounds
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-0.087, -7.0])
    ocp.constraints.ubu = np.array([0.087, 2.5])

    # State bounds
    y_min = ROAD_Y_MIN + EGO_WIDTH / 2.0
    y_max = ROAD_Y_MAX - EGO_WIDTH / 2.0
    ocp.constraints.idxbx = np.array([1, 3, 4])
    ocp.constraints.lbx = np.array([y_min, 1.0, -0.4])
    ocp.constraints.ubx = np.array([y_max, 11.0, 0.4])

    # Terminal state bounds
    ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()

    # Collision constraints
    nh = 4
    ocp.constraints.lh = np.zeros(nh)
    ocp.constraints.uh = 1e15 * np.ones(nh)

    ocp.constraints.lh_e = np.zeros(nh)
    ocp.constraints.uh_e = 1e15 * np.ones(nh)

    # Inizial condition
    ocp.constraints.x0 = np.asarray(x0, dtype=float)

    # Solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3

    ocp.code_gen_options.code_export_directory = "codegen_ocp"

    # Create solver
    solver = AcadosOcpSolver(ocp)

    return model, solver