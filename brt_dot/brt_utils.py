import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator

class BRTinterpolator:
    def __init__(self, filepath):
        with np.load(filepath, allow_pickle=False) as data:
            coordinates = [
                data["x_rel"],
                data["y_rel"],
                data["theta_rel"],
                data["v_H"],
                data["delta_E"],
                data["v_E"],
            ]
            values = data["BRT"]
            gradients = data["gradients"]
            metadata = json.loads(data["metadata_json"].item())
            self.dynamics_parameters = metadata["dynamics"]["parameters"]
            self.theta_min = float(data["grid_lo"][2])
            self.theta_period = float(data["grid_hi"][2] - data["grid_lo"][2])
            expected_shape = tuple(len(axis) for axis in coordinates)
            if values.shape != expected_shape:
                raise ValueError("La forma del BRT non corrisponde alla griglia")
            if gradients.shape != expected_shape + (6,):
                raise ValueError("La forma dei gradienti non corrisponde alla griglia")

            self.lower = np.array([axis[0] for axis in coordinates])
            self.upper = np.array([axis[-1] for axis in coordinates])
            coordinates[2] = np.append(coordinates[2], coordinates[2][0] + self.theta_period)
            values = np.concatenate([values, values[:, :, :1, ...]], axis=2)
            gradients = np.concatenate([gradients, gradients[:, :, :1, ...]], axis=2)
            self.value_interpolator = RegularGridInterpolator(
                tuple(coordinates),
                values,
                method="linear",
                bounds_error=True,
            )
            self.gradient_interpolator = RegularGridInterpolator(
                tuple(coordinates),
                gradients,
                method="linear",
                bounds_error=True,
            )
    def evaluate(self, relative_state):
        state = np.asarray(relative_state, dtype=float).copy()
        if state.shape != (6,):
            raise ValueError("Lo stato relativo deve avere 6 componenti")
        if not np.all(np.isfinite(state)):
            return np.nan, np.full(6, np.nan), False
        state[2] = self.theta_min + (state[2] - self.theta_min) % self.theta_period
        nonperiodic = [0, 1, 3, 4, 5]
        inside_grid = bool(
            np.all(state[nonperiodic] >= self.lower[nonperiodic]) and np.all(state[nonperiodic] <= self.upper[nonperiodic])
        )
        if not inside_grid:
            return np.nan, np.full(6, np.nan), False
        point = state.reshape(1, 6)
        value = float(self.value_interpolator(point)[0])
        gradient = self.gradient_interpolator(point)[0]

        return value, gradient, True

def hj_coefficients(relative_state, gradient, dynamics_parameters):
    x_rel, y_rel, theta_rel, v_H, delta_E, v_E = relative_state
    p = np.asarray(gradient, dtype=float)
    lf = dynamics_parameters["lf"]
    lr = dynamics_parameters["lr"]
    beta_E = np.arctan(lr / (lf + lr) * np.tan(delta_E))
    omega_E = v_E * np.cos(beta_E) / (lf + lr) * np.tan(delta_E)
    drift = np.array([
        v_H * np.cos(theta_rel) - v_E * np.cos(beta_E) + y_rel * omega_E,
        v_H * np.sin(theta_rel) - v_E * np.sin(beta_E) - x_rel * omega_E,
        -omega_E,
        0.0,
        0.0,
        0.0,
    ])
    human_yaw_rate_max = dynamics_parameters["human_max_yaw_rate"]
    human_acc_max = dynamics_parameters["human_max_acceleration"]
    human_acc_min = dynamics_parameters["human_min_acceleration"]
    worst_yaw_term = -abs(p[2]) * human_yaw_rate_max
    worst_acc_term = min(p[3] * human_acc_min, p[3] * human_acc_max)
    M_HJ = p[[4, 5]]
    b_HJ = float(p @ drift + worst_yaw_term + worst_acc_term)

    return M_HJ, b_HJ