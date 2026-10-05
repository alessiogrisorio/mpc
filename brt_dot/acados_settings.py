import numpy as np
import scipy.linalg

from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin, exp, if_else

from ego_model import ego_model

# Vehicles dimensions
EGO_LENGTH = 4.68
EGO_WIDTH = 2.20
HUMAN_LENGTH = 4.28
HUMAN_WIDTH = 1.80

# Circle dimensions
rho_E = np.hypot(EGO_LENGTH / 4.0, EGO_WIDTH / 2.0)
rho_H = np.hypot(HUMAN_LENGTH / 4.0, HUMAN_WIDTH / 2.0)
safety_r = rho_E + rho_H

# Road dimensions
ROAD_Y_MIN = -2.0
ROAD_Y_MAX = 6.0
LANE_CENTER_1 = 0.0
LANE_CENTER_2 = 4.0
LANE_HALF_DISTANCE = (LANE_CENTER_2 - LANE_CENTER_1) / 2

# Normalizzazione pesi
psi_max = 0.2
v_max = 2.0
steering_rate_max = 0.087
acc_max = 3.0


def acados_settings(Tf, N, lf, lr, x0, v_ref, human_pos, weights=None):

    if weights is None:
        weights = {
            #"lane": 3.890810431104229,
            "lane": 3.890810431104229,
            "lane_preference": 1000.0,
            "psi": 1.4467453645031252,
            "velocity": 56.477006267018474,
            "steering_rate": 0.09888880927846047,
            "acceleration": 13.887061909157946,
        }

    # Create ocp and model
    ocp = AcadosOcp()
    model = ego_model(lf, lr)

    # Ego variables
    X_E = model.x[0]
    Y_E = model.x[1]
    PSI_E = model.x[2]
    V_E = model.x[3]
    DELTA_E = model.x[4]
    STEERING_RATE = model.u[0]
    ACCELERATION = model.u[1]

    # Variabili da aggiornare durante la simulazione
    X_H = SX.sym("X_H")
    Y_H = SX.sym("Y_H")
    M1 = SX.sym("M1")
    M2 = SX.sym("M2")
    B_HJ = SX.sym("B_HJ")
    model.p = vertcat(X_H, Y_H, M1, M2, B_HJ)
    ocp.model = model
    ocp.parameter_values = np.array([
        human_pos[0],
        human_pos[1],
        0.0,   # M1
        0.0,   # M2
        1.0,   # B_HJ
    ])

    # Lane preferences
    lane_error = (Y_E - LANE_CENTER_1) * (Y_E - LANE_CENTER_2) / (LANE_HALF_DISTANCE**2)
    lane_preference = RETURN_ON * (Y_E - LANE_CENTER_1) / (LANE_CENTER_2 - LANE_CENTER_1)

    # Costi da controllare
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    model.cost_y_expr = vertcat(X_E, lane_error, lane_preference, PSI_E, V_E, DELTA_E, STEERING_RATE, ACCELERATION)
    model.cost_y_expr_e = vertcat(X_E, lane_error, lane_preference, PSI_E, V_E, DELTA_E)

    # Pesi
    Q = np.diag([
        0.0,     # X
        weights["lane"],                   # lane error
        weights["lane_preference"],        # lane preference
        weights["psi"] / psi_max**2,       # psi
        weights["velocity"] / v_max**2,    # velocity
        0.0,                               # delta
    ])
    R = np.diag([
        weights["steering_rate"] / steering_rate_max**2,   # steering rate
        weights["acceleration"] / acc_max**2,              # acceleration
    ])
    Qe = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe

    # Reference
    ocp.cost.yref = np.array([
        0.0,        # X
        0.0,        # lane error
        0.0,        # lane preference
        0.0,        # psi
        v_ref,      # v
        0.0,        # delta
        0.0,        # steering rate
        0.0,        # acceleration
    ])
    ocp.cost.yref_e = np.array([
        0.0,        # X
        0.0,        # lane error
        0.0,        # lane preference
        0.0,        # psi
        v_ref,      # v
        0.0,        # delta
    ])

    # LIMITI FISICI
    # Stato
    y_min = ROAD_Y_MIN + EGO_WIDTH / 2.0
    y_max = ROAD_Y_MAX - EGO_WIDTH / 2.0
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.lbx = np.array([1.0, -np.pi / 12])
    ocp.constraints.ubx = np.array([11.0, np.pi / 12])
    # Terminal state
    ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
    # Input
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([-0.087, -7.0])
    ocp.constraints.ubu = np.array([0.087, 2.5])


    # Vincolo geometrico di collisione
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
    street_bounds = vertcat(
        Y_E + 0.5 * (EGO_LENGTH * sin(PSI_E) + EGO_WIDTH * cos(PSI_E)),
        Y_E + 0.5 * (EGO_LENGTH * sin(PSI_E) - EGO_WIDTH * cos(PSI_E)),
        Y_E - 0.5 * (EGO_LENGTH * sin(PSI_E) + EGO_WIDTH * cos(PSI_E)),
        Y_E - 0.5 * (EGO_LENGTH * sin(PSI_E) - EGO_WIDTH * cos(PSI_E)),
    )
    

    # Vincolo HJ
    hj_constraint = M1 * STEERING_RATE + M2 * ACCELERATION + B_HJ

    # Espressioni dei vincoli
    model.con_h_expr = vertcat(separation, hj_constraint, street_bounds)
    model.con_h_expr_0 = vertcat(separation, hj_constraint, street_bounds)
    model.con_h_expr_e = vertcat(separation, street_bounds)

    # Limiti dei vincoli collisione e HJ
    ocp.constraints.lh = np.concatenate([np.zeros(5), np.full(4, ROAD_Y_MIN)])
    ocp.constraints.uh = np.concatenate([np.full(5, 1e15), np.full(4, ROAD_Y_MAX)])

    ocp.constraints.lh_0 = ocp.constraints.lh.copy()
    ocp.constraints.uh_0 = ocp.constraints.uh.copy()

    # Vincolo terminale solo su collisione
    ocp.constraints.lh_e = np.concatenate([np.zeros(4), np.full(4, ROAD_Y_MIN)])
    ocp.constraints.uh_e = np.concatenate([np.full(4, 1e15), np.full(4, ROAD_Y_MAX)])

    # Soft constraint su HJ, indice 4
    ocp.constraints.idxsh = np.array([4])
    ocp.constraints.idxsh_0 = np.array([4])

    # Penalità del soft constraint
    ocp.cost.Zl = np.array([100.0])
    ocp.cost.Zu = np.array([100.0])
    ocp.cost.zl = np.array([100.0])
    ocp.cost.zu = np.array([100.0])
    ocp.cost.Zl_0 = ocp.cost.Zl.copy()
    ocp.cost.Zu_0 = ocp.cost.Zu.copy()
    ocp.cost.zl_0 = ocp.cost.zl.copy()
    ocp.cost.zu_0 = ocp.cost.zu.copy()

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