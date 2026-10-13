import numpy as np
import scipy.linalg

from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, cos, sin, tanh, sqrt, exp

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
TOL_ROAD = 0.0
ROAD_MIN = ROAD_Y_MIN + TOL_ROAD
ROAD_MAX = ROAD_Y_MAX - TOL_ROAD
eps = 1e-4
LANE_CENTER_1 = 0.0
LANE_CENTER_2 = 4.0
LANE_HALF_DISTANCE = (LANE_CENTER_2 - LANE_CENTER_1) / 2
left_clearance_ref = ROAD_MAX - LANE_CENTER_1 - EGO_WIDTH / 2.0
right_clearance_ref = LANE_CENTER_1 - EGO_WIDTH / 2.0 - ROAD_MIN
k_balance = left_clearance_ref / right_clearance_ref
EDGE_MARGIN = 0.3

# Normalizzazione pesi
psi_max = 0.2
v_max = 2.0
steering_rate_max = 0.087
acc_max = 3.0

# Exp penalty
K_LANE = 0.8
K_V = 0.7
K_OVERTAKE = 0.02


def acados_settings(Tf, N, lf, lr, x0, v_ref, human_pos, weights=None):

    if weights is None:
        weights = {
            "x": 12.795671099140785,
            "y": 0.187004955386393,
            "psi": 0.3,
            "velocity": 7.39806242695111,
            "steering_rate": 0.03,
            "acceleration": 0.5,
            "road_left": 1.0,
            "road_right": 1.0,
            "low_speed": 0.6997050192410854,
            "collision": 0.4897221717605145,
            "overtake": 9.334625302202161
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
    
    # Soft road penalty
    lateral_extent = EGO_LENGTH / 2.0 * sqrt(sin(PSI_E)**2 + eps**2) + EGO_WIDTH / 2.0 * sqrt(cos(PSI_E)**2 + eps**2)
    C_left = exp(K_LANE * (Y_E + lateral_extent - ROAD_MAX))
    C_right = exp(-K_LANE * k_balance * (Y_E - lateral_extent - ROAD_MIN))

    # Low speed penalty
    C_low_speed = exp(-K_V * (V_E - 1.0))

    # Geometrical coolision penalty
    C_collision = vertcat(
        exp(-separation_ff),
        exp(-separation_fr),
        exp(-separation_rf),
        exp(-separation_rr),
    )

    # Right overtake penalty
    C_overtake = exp(-(Y_E - Y_H + K_OVERTAKE * (X_E - X_H)**2))

    # Vincolo HJ
    hj_constraint = M1 * STEERING_RATE + M2 * ACCELERATION + B_HJ

    # Cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    model.cost_y_expr = vertcat(
        X_E,
        Y_E,
        PSI_E,
        V_E,
        DELTA_E,
        STEERING_RATE,
        ACCELERATION,
        C_left,
        C_right,
        C_low_speed,
        C_collision,
        C_overtake,
    )
    model.cost_y_expr_e = vertcat(
        X_E,
        Y_E,
        PSI_E,
        V_E,
        DELTA_E,
    )

    # Pesi
    Q = np.array([
        weights["x"] / 10.0**2,
        weights["y"] / LANE_HALF_DISTANCE**2,
        weights["psi"] / psi_max**2,
        weights["velocity"] / v_max**2,
        0.0,
    ])
    R = np.array([
        weights["steering_rate"] / steering_rate_max**2,
        weights["acceleration"] / acc_max**2,
    ])
    W_exp = np.array([
        weights["road_left"],
        weights["road_right"],
        weights["low_speed"],
        weights["collision"],
        weights["collision"],
        weights["collision"],
        weights["collision"],
        weights["overtake"],
    ])
    ocp.cost.W = np.diag(
        np.concatenate([Q, R, W_exp])
    )
    ocp.cost.W_e = np.diag(0.5 * Q)

    # Reference
    ocp.cost.yref = np.array([
        x0[0],
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
    ocp.cost.yref_e = np.array([
        x0[0] + v_ref * Tf,
        LANE_CENTER_1,
        0.0,
        v_ref,
        0.0,
    ])




    # Espressioni dei vincoli
    model.con_h_expr = vertcat(hj_constraint, street_bounds)
    model.con_h_expr_0 = vertcat(hj_constraint, street_bounds)
    model.con_h_expr_e = vertcat(street_bounds)

    # Limiti dei vincoli collisione e HJ
    ocp.constraints.lh = np.concatenate([np.array([0.0]), np.full(4, ROAD_MIN)])
    ocp.constraints.uh = np.concatenate([np.array([1e15]), np.full(4, ROAD_MAX)])

    ocp.constraints.lh_0 = ocp.constraints.lh.copy()
    ocp.constraints.uh_0 = ocp.constraints.uh.copy()

    # Vincolo terminale solo su collisione
    ocp.constraints.lh_e = np.full(4, ROAD_MIN)
    ocp.constraints.uh_e = np.full(4, ROAD_MAX)

    # Soft constraint su HJ, indice 4
    ocp.constraints.idxsh = np.array([0])
    ocp.constraints.idxsh_0 = np.array([0])

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