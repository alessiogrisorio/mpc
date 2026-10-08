from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, tan, atan

def ego_model(lf, lr) -> AcadosModel:

    model_name = 'ego_single_track'
    L = lf + lr

    # State
    X = SX.sym("X")
    Y = SX.sym("Y")
    psi = SX.sym("psi")
    v = SX.sym("v")
    delta = SX.sym("delta")

    x = vertcat(X, Y, psi, v, delta)

    # Control
    steering_rate = SX.sym("steering_rate")
    acceleration = SX.sym("acceleration")

    u = vertcat(steering_rate, acceleration)

    # State derivatives
    X_dot = SX.sym("X_dot")
    Y_dot = SX.sym("Y_dot")
    psi_dot = SX.sym("psi_dot")
    delta_dot = SX.sym("delta_dot")
    v_dot = SX.sym("v_dot")

    xdot = vertcat(X_dot, Y_dot, psi_dot, v_dot, delta_dot)

    # Dynamics
    beta = atan(lr / L * tan(delta))

    f_expl = vertcat(
        v * cos(psi + beta),
        v * sin(psi + beta),
        v / L * cos(beta) * tan(delta),
        acceleration,
        steering_rate,
    )
    f_impl = xdot - f_expl

    # Acados model
    model = AcadosModel()
    model.name = model_name
    model.x = x
    model.xdot = xdot
    model.u = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    return model