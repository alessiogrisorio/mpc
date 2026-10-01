import numpy as np
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
