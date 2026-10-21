"""HJ data and calculations. Controls are [steering_rate, acceleration].

No plotting, BRT loading, or controller modification occurs in this module.
``b`` always includes alpha * V, exactly as supplied to the solver.
Limits and polygons describe instantaneous HJ/control feasibility only;
they do not include road/state constraints or future vehicle dynamics.
"""

from pathlib import Path
import json

import numpy as np


class HJDiagnostics:
    SCHEMA_VERSION = 1
    FIELDS = (
        "time", "V", "alpha", "M", "b", "active", "u", "slack",
        "slack_upper", "state", "relative_state", "predicted_states",
        "solver_status", "solve_time",
    )

    def __init__(
        self, metrics, dt, stages=3, stage_dt=None,
        u_lower=(-0.087, -7.0), u_upper=(0.087, 2.5),
        tolerance=1e-7, coefficient_tolerance=1e-12,
    ):
        self.metrics = tuple(metrics)
        if (not self.metrics or len(set(self.metrics)) != len(self.metrics)
                or any(not isinstance(m, str) or not m for m in self.metrics)):
            raise ValueError("metrics must contain distinct, nonempty names")
        if isinstance(stages, bool) or int(stages) != stages or stages < 1:
            raise ValueError("stages must be a positive integer")
        self.stages = int(stages)
        self.dt = float(dt)
        self.stage_dt = float(dt if stage_dt is None else stage_dt)
        self.u_lower = self._array(u_lower, (2,), "u_lower")
        self.u_upper = self._array(u_upper, (2,), "u_upper")
        if np.any(self.u_lower >= self.u_upper):
            raise ValueError("Each lower control bound must be below its upper bound")
        self.tolerance = float(tolerance)
        self.coefficient_tolerance = float(coefficient_tolerance)
        if (not np.isfinite([self.dt, self.stage_dt, self.tolerance,
                             self.coefficient_tolerance]).all()
                or self.dt <= 0 or self.stage_dt <= 0
                or self.tolerance < 0 or self.coefficient_tolerance <= 0):
            raise ValueError("Invalid time steps or tolerances")
        self._records = []

    @staticmethod
    def _array(value, shape, name, finite=True):
        result = np.asarray(value, dtype=float)
        if result.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {result.shape}")
        if finite and not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result.copy()

    def __len__(self):
        return len(self._records)

    def record(
        self, *, V, alpha, M, b, active, u, slack, slack_upper=None,
        state=None, relative_state=None, predicted_states=None,
        time=None, solver_status=0, solve_time=np.nan,
    ):
        """Store one MPC solution, with one row per constrained stage.

        Shapes: V/alpha (metrics,), M (stages, metrics, 2), b/active/
        slack (stages, metrics), u (stages, 2). A scalar alpha is allowed.
        V refers to the current relative state (frozen-coefficient scheme).
        ``active`` must indicate conditions actually enabled at each stage.
        """
        s, m = self.stages, len(self.metrics)
        V = self._array(V, (m,), "V", finite=False)
        alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (m,)).copy()
        if not np.isfinite(alpha).all() or np.any(alpha < 0):
            raise ValueError("alpha must be finite and nonnegative")
        M = self._array(M, (s, m, 2), "M")
        b = self._array(b, (s, m), "b")
        active = np.asarray(active, dtype=bool)
        if active.shape != (s, m):
            raise ValueError(f"active must have shape {(s, m)}")
        if not np.isfinite(V[np.any(active, axis=0)]).all():
            raise ValueError("Active metrics must have finite V")
        u = self._array(u, (s, 2), "u")
        slack = self._array(slack, (s, m), "slack")
        slack_upper = (np.zeros((s, m)) if slack_upper is None else
                       self._array(slack_upper, (s, m), "slack_upper"))
        # Tiny negative solver slacks are retained, rather than silently clipped.
        if np.any(slack < -self.tolerance) or np.any(slack_upper < -self.tolerance):
            raise ValueError("Negative slack exceeds tolerance")
        time = len(self) * self.dt if time is None else float(time)
        if not np.isfinite(time) or (self._records and time <= self._records[-1]["time"]):
            raise ValueError("Record times must be finite and strictly increasing")
        row = dict(
            time=time, V=V, alpha=alpha, M=M, b=b, active=active.copy(),
            u=u, slack=slack, slack_upper=slack_upper,
            solver_status=int(solver_status), solve_time=float(solve_time),
        )
        for name, value, shape in (
            ("state", state, (5,)),
            ("relative_state", relative_state, (6,)),
            ("predicted_states", predicted_states, (s, 5)),
        ):
            row[name] = (np.full(shape, np.nan) if value is None else
                         self._array(value, shape, name, finite=False))
        self._records.append(row)

    def record_solver(self, solver, *, V, alpha, available, **kwargs):
        """Capture the existing solver layout [X_H,Y_H,M1_0,M2_0,b_0,...].

        Call after solve(), before updating the parameters for the next step.
        This reads the actual parameters, controls and slacks at every HJ stage.
        All recorded stages must be enabled by the main's HJ_STAGES policy;
        ``available`` is the current per-metric inside-grid mask.
        """
        m = len(self.metrics)
        available = np.asarray(available, dtype=bool)
        if available.shape != (m,):
            raise ValueError(f"available must have shape {(m,)}")
        parameters = np.asarray([
            solver.get(j, "p") for j in range(self.stages)
        ], dtype=float)
        if parameters.shape != (self.stages, 2 + 3 * m):
            raise ValueError("Unexpected solver parameter layout")
        coefficients = parameters[:, 2:].reshape(self.stages, m, 3)
        self.record(
            V=V, alpha=alpha,
            M=coefficients[:, :, :2], b=coefficients[:, :, 2],
            active=np.broadcast_to(available, (self.stages, m)),
            u=np.asarray([solver.get(j, "u") for j in range(self.stages)]),
            slack=np.asarray([solver.get(j, "sl") for j in range(self.stages)]),
            slack_upper=np.asarray([solver.get(j, "su") for j in range(self.stages)]),
            predicted_states=np.asarray([
                solver.get(j, "x") for j in range(self.stages)
            ]),
            **kwargs,
        )

    def arrays(self):
        """Axes: simulation step, HJ stage, metric, control component."""
        if not self._records:
            raise ValueError("No diagnostic samples recorded")
        return {key: np.asarray([r[key] for r in self._records])
                for key in self.FIELDS}

    def contributions(self):
        d = self.arrays()
        steering = d["M"][..., 0] * d["u"][:, :, None, 0]
        acceleration = d["M"][..., 1] * d["u"][:, :, None, 1]
        relaxation = np.broadcast_to(
            (d["alpha"] * d["V"])[:, None, :], d["b"].shape,
        ).copy()
        drift = d["b"] - relaxation
        residual = steering + acceleration + d["b"]
        result = dict(
            steering=steering, acceleration=acceleration,
            drift_and_worst_human=drift, relaxation=relaxation,
            residual=residual, slack=d["slack"],
            softened_residual=residual + d["slack"],
        )
        result = {name: np.where(d["active"], value, np.nan)
                  for name, value in result.items()}
        result["violated"] = d["active"] & (residual < -self.tolerance)
        result["soft_violated"] = d["active"] & (
            residual + d["slack"] < -self.tolerance
        )
        # This is primal proximity to a bound, not a multiplier-based diagnosis.
        result["near_boundary"] = d["active"] & (
            np.abs(residual) <= self.tolerance
        )
        return result

    def compute_limits(self, *, softened=False):
        """Conditional control bounds given the other optimized control.

        Owners are metric indices; -1 denotes the physical control bound.
        A near-zero coefficient creates no divided bound; the remaining
        inequality is tested explicitly for incompatibility.
        """
        d = self.arrays()
        M = d["M"]
        b = d["b"] + (d["slack"] if softened else 0.0)
        scale = (self.u_upper - self.u_lower) / 2.0
        magnitude = np.max(np.abs(M) * scale, axis=-1)
        result = {}
        for axis, name in enumerate(("steering_rate", "acceleration")):
            other = 1 - axis
            coefficient = M[..., axis]
            constant = M[..., other] * d["u"][:, :, None, other] + b
            zero = np.abs(coefficient) * scale[axis] <= (
                self.coefficient_tolerance * magnitude
            )
            positive = d["active"] & ~zero & (coefficient > 0)
            negative = d["active"] & ~zero & (coefficient < 0)
            boundary = np.full(b.shape, np.nan)
            np.divide(-constant, coefficient, out=boundary, where=positive | negative)
            lower = np.where(positive, boundary, -np.inf)
            upper = np.where(negative, boundary, np.inf)
            all_lower = np.concatenate([
                np.full((*b.shape[:2], 1), self.u_lower[axis]), lower,
            ], axis=-1)
            all_upper = np.concatenate([
                np.full((*b.shape[:2], 1), self.u_upper[axis]), upper,
            ], axis=-1)
            lower_bound = np.max(all_lower, axis=-1)
            upper_bound = np.min(all_upper, axis=-1)
            scale_constant = np.maximum(np.abs(constant), magnitude)
            impossible = np.any(
                d["active"] & zero &
                (constant < -self.tolerance * scale_constant), axis=-1,
            )
            result[name] = dict(
                metric_lower=lower, metric_upper=upper,
                lower=lower_bound, upper=upper_bound,
                lower_owner=np.argmax(all_lower, axis=-1) - 1,
                upper_owner=np.argmin(all_upper, axis=-1) - 1,
                feasible=(lower_bound <= upper_bound) & ~impossible,
                incompatible_zero_coefficient=impossible,
            )
        return result

    @staticmethod
    def _clip(polygon, normal, offset):
        """Clip a convex polygon by normal @ point + offset >= 0."""
        if len(polygon) == 0:
            return polygon
        output = []
        previous = polygon[-1]
        previous_value = float(normal @ previous + offset)
        for point in polygon:
            value = float(normal @ point + offset)
            if (value >= 0) != (previous_value >= 0):
                fraction = previous_value / (previous_value - value)
                output.append(previous + fraction * (point - previous))
            if value >= 0:
                output.append(point.copy())
            previous, previous_value = point, value
        return np.asarray(output, dtype=float).reshape(-1, 2)

    def control_polygon(
        self, index, stage=0, *, softened=False,
        acceleration_min=None, acceleration_max=None,
    ):
        """Return feasible vertices in physical control units, shape (n,2).

        Empty means infeasible. Lines/points are retained as feasible sets.
        Only physical control bounds and lower HJ inequalities are included;
        the solver's artificial HJ upper bound (1e15) is not modeled here.
        """
        row = self._records[index]
        if not 0 <= stage < self.stages:
            raise IndexError("Invalid HJ stage")
        lo, hi = self.u_lower.copy(), self.u_upper.copy()
        if acceleration_min is not None:
            lo[1] = max(lo[1], float(acceleration_min))
        if acceleration_max is not None:
            hi[1] = min(hi[1], float(acceleration_max))
        if np.any(lo > hi):
            return np.empty((0, 2))
        center = (self.u_lower + self.u_upper) / 2.0
        scale = (self.u_upper - self.u_lower) / 2.0
        physical = np.array([
            [lo[0], lo[1]], [hi[0], lo[1]],
            [hi[0], hi[1]], [lo[0], hi[1]],
        ])
        polygon = (physical - center) / scale
        for k in np.flatnonzero(row["active"][stage]):
            M = row["M"][stage, k]
            b = row["b"][stage, k]
            if softened:
                b += row["slack"][stage, k]
            normal = M * scale
            offset = float(M @ center + b)
            # Positive row rescaling does not change geometry or its tolerance.
            magnitude = max(float(np.max(np.abs(normal))), abs(offset))
            if magnitude == 0:
                continue
            polygon = self._clip(
                polygon, normal / magnitude,
                offset / magnitude + self.tolerance,
            )
            if not len(polygon):
                return np.empty((0, 2))
        return polygon * scale + center

    def feasibility(self, *, softened=False):
        """Check joint feasibility and whether any nonbraking command exists.

        No feasible command => HJ conflict, rather than 'braking required'.
        Braking is necessary only if feasible but no a >= 0 command exists.
        """
        shape = (len(self), self.stages)
        feasible = np.zeros(shape, dtype=bool)
        nonbraking = np.zeros(shape, dtype=bool)
        constant_speed = np.zeros(shape, dtype=bool)
        for i, row in enumerate(self._records):
            for j in range(self.stages):
                feasible[i, j] = len(self.control_polygon(i, j, softened=softened)) > 0
                nonbraking[i, j] = len(self.control_polygon(
                    i, j, softened=softened, acceleration_min=0.0,
                )) > 0
                constant_speed[i, j] = len(self.control_polygon(
                    i, j, softened=softened,
                    acceleration_min=0.0, acceleration_max=0.0,
                )) > 0
        return dict(
            feasible=feasible, constant_speed_feasible=constant_speed,
            nonbraking_feasible=nonbraking,
            braking_required=feasible & ~nonbraking,
            conflict=~feasible,
        )

    def summary(self):
        """Per-step/per-metric extrema over the recorded HJ stages."""
        d = self.arrays()
        r = self.contributions()["residual"]
        valid = np.any(d["active"], axis=1)
        minimum = np.min(np.where(d["active"], r, np.inf), axis=1)
        maximum = np.max(np.where(d["active"], d["slack"], -np.inf), axis=1)
        min_stage = np.argmin(np.where(d["active"], r, np.inf), axis=1)
        max_stage = np.argmax(np.where(d["active"], d["slack"], -np.inf), axis=1)
        return dict(
            min_residual=np.where(valid, minimum, np.nan),
            max_slack=np.where(valid, maximum, np.nan),
            min_residual_stage=np.where(valid, min_stage, -1),
            max_slack_stage=np.where(valid, max_stage, -1),
        )

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Diagnostic files must have .npz extension")
        data = self.arrays()
        metadata = dict(
            schema_version=self.SCHEMA_VERSION, metrics=self.metrics,
            dt=self.dt, stage_dt=self.stage_dt, stages=self.stages,
            u_lower=self.u_lower.tolist(), u_upper=self.u_upper.tolist(),
            tolerance=self.tolerance,
            coefficient_tolerance=self.coefficient_tolerance,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata_json=json.dumps(metadata), **data)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].item())
            if metadata.pop("schema_version") != cls.SCHEMA_VERSION:
                raise ValueError("Unsupported diagnostics schema version")
            result = cls(**metadata)
            data = {key: archive[key].copy() for key in cls.FIELDS}
        size = len(data["time"])
        if any(len(value) != size for value in data.values()):
            raise ValueError("Diagnostic arrays have inconsistent lengths")
        for i in range(size):
            result.record(**{key: value[i] for key, value in data.items()})
        return result
