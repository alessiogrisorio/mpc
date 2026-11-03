import numpy as np

from acados_settings import (
    EGO_LENGTH,
    EGO_WIDTH,
    HUMAN_LENGTH,
    HUMAN_WIDTH,
    LANE_CENTER_1,
    safety_r,
    ROAD_Y_MIN,
    ROAD_Y_MAX,
    EDGE_MARGIN,
)


# Weights to optimize
WEIGHT_BOUNDS = {
    "x": (1.0, 100.0),
    "y": (0.1, 10.0),
    "velocity": (2.0, 100.0),
    "low_speed": (0.05, 3.0),
    "collision": (0.01, 1.0),
    "overtake": (0.1, 10.0),
}

# Baseline weights
BASELINE_WEIGHTS = {
    "x": 10.0,
    "y": 0.6,
    "psi": 0.3,
    "velocity": 5.0,
    "steering_rate": 0.03,
    "acceleration": 0.5,
    "road_left": 1.0,
    "road_right": 1.0,
    "low_speed": 0.5,
    "collision": 0.05,
    "overtake": 2.0,
}


SCENARIO_RANGES = {
    "v_ref": (3.0, 9.0),
    "v_ego_initial": (3.0, 9.0),
    "v_human": (2.0, 6.0),
    "human_distance": (10.0, 20.0),
}


N_TRAIN = 6
N_VALIDATION = 15

SEED = 42


# Score parameters
V_MARGIN = 0.5

BRT_INTERACTION_BEHIND = 6.0
BRT_INTERACTION_AHEAD = 12.0

OVERTAKE_MARGIN = 1.0
LATERAL_MARGIN = 0.3


def generate_scenarios(n_scenarios, seed):

    rng = np.random.default_rng(seed)
    scenarios = []

    Tsim = 20.0
    maneuver_buffer = 5.0

    required_lead = (
        EGO_LENGTH / 4.0
        + HUMAN_LENGTH / 4.0
        + safety_r
    )

    while len(scenarios) < n_scenarios:

        v_ref = rng.uniform(*SCENARIO_RANGES["v_ref"])
        v_ego_initial = rng.uniform(*SCENARIO_RANGES["v_ego_initial"])
        v_human = rng.uniform(*SCENARIO_RANGES["v_human"])
        human_distance = rng.uniform(*SCENARIO_RANGES["human_distance"])

        relative_speed = v_ref - v_human

        if relative_speed <= 0.0:
            continue

        nominal_overtake_time = (
            human_distance + required_lead
        ) / relative_speed

        if nominal_overtake_time > Tsim - maneuver_buffer:
            continue

        scenarios.append({
            "v_ref": float(v_ref),
            "x0": [
                0.0,
                LANE_CENTER_1,
                0.0,
                float(v_ego_initial),
                0.0,
            ],
            "X_H_initial": float(human_distance),
            "Y_H_initial": float(LANE_CENTER_1),
            "v_H": float(v_human),
            "Tsim": Tsim,
        })

    return scenarios


TRAIN_SCENARIOS = generate_scenarios(
    N_TRAIN,
    SEED,
)

VALIDATION_SCENARIOS = generate_scenarios(
    N_VALIDATION,
    SEED + 1,
)


def score_components(result, scenario):

    if not result["success"]:

        return {
            "velocity": 0.0,
            "control_effort": 0.0,
            "smoothness": 0.0,
            "lane_return": 0.0,
            "hj_slack": 0.0,
            "edge": 0.0,
            "brt_margin": 0.0,
            "brt_unsafe": 0.0,
            "brt_unknown": 0.0,
            "collision": 0.0,
            "overtake": 0.0,
            "overtake_completed": False,
            "failure_penalty": 1e6,
            "total": 1e6,
        }

    simX = np.asarray(result["simX"], dtype=float)
    simU = np.asarray(result["simU"], dtype=float)
    dt = float(result["dt"])

    v_ref = float(scenario["v_ref"])
    v_H = float(scenario["v_H"])

    X_H_initial = float(scenario["X_H_initial"])
    Y_H_initial = float(scenario.get("Y_H_initial", 0.0))

    t = np.arange(len(simX)) * dt
    Tsim = float(t[-1])

    X_E = simX[:, 0]
    Y_E = simX[:, 1]
    PSI_E = simX[:, 2]

    X_H = X_H_initial + v_H * t

    # --------------------------------
    # Velocity tracking
    # --------------------------------

    velocity = np.sqrt(
        np.mean(
            ((simX[:, 3] - v_ref) / 2.0)**2
        )
    )

    # --------------------------------
    # Control effort
    # --------------------------------

    steering_effort = np.mean(
        (simU[:, 0] / 0.087)**2
    )

    acceleration_effort = np.mean(
        (simU[:, 1] / 7.0)**2
    )

    control_effort = (
        steering_effort + acceleration_effort
    )

    # --------------------------------
    # Control smoothness
    # --------------------------------

    du = np.diff(simU, axis=0)

    if len(du):

        smoothness = np.mean(
            (du[:, 0] / 0.087)**2
            + (du[:, 1] / 7.0)**2
        )

    else:
        smoothness = 0.0

    # --------------------------------
    # Return to right lane
    # --------------------------------

    lane_return = (
        abs(Y_E[-1] - LANE_CENTER_1) / 4.0
    )

    # --------------------------------
    # HJ slack
    # --------------------------------

    hj_slack = np.max(
        np.maximum(
            result["simHJSlack"],
            result["simHJSlackUpper"],
        )
    )

    # --------------------------------
    # Road boundaries
    # --------------------------------

    lateral_extent = (
        EGO_LENGTH / 2.0 * np.abs(np.sin(PSI_E))
        + EGO_WIDTH / 2.0 * np.abs(np.cos(PSI_E))
    )

    edge_error = (
        np.maximum(
            0.0,
            ROAD_Y_MIN + EDGE_MARGIN
            - (Y_E - lateral_extent),
        )
        + np.maximum(
            0.0,
            Y_E + lateral_extent
            - (ROAD_Y_MAX - EDGE_MARGIN),
        )
    ) / EDGE_MARGIN

    edge_rms = np.sqrt(
        np.mean(edge_error**2)
    )

    edge_peak = np.max(edge_error)

    # --------------------------------
    # BRT safety
    # --------------------------------

    V = np.asarray(
        result["simV"],
        dtype=float,
    )

    inside = np.asarray(
        result["simInsideGrid"],
        dtype=bool,
    )

    valid = inside & np.isfinite(V)

    # BRT samples correspond to simX[:-1]
    relative_x = X_H[:-1] - X_E[:-1]

    interaction = (
        (relative_x >= -BRT_INTERACTION_BEHIND)
        & (relative_x <= BRT_INTERACTION_AHEAD)
    )

    known_near = valid & interaction

    # --------------------------------
    # Positive BRT margin
    # --------------------------------

    # Positive BRT margin
    if np.any(valid):

        V_min = float(np.min(V[valid]))

        brt_margin_cost = (
            250.0 * float(V_min < V_MARGIN)
            + 100.0 * max(0.0, V_MARGIN - V_min)**2
        )

    else:
        brt_margin_cost = 0.0

    # --------------------------------
    # Negative BRT values
    # --------------------------------

    if np.any(valid):

        V_min = float(np.min(V[valid]))

        brt_unsafe_cost = (
            250.0 * float(V_min < -1e-3)
            + 100.0 * max(0.0, -V_min)**2
        )

    else:
        brt_unsafe_cost = 0.0

    # --------------------------------
    # BRT unavailable during interaction
    # --------------------------------

    n_interaction = np.count_nonzero(interaction)

    if n_interaction:

        unknown_fraction = (
            np.count_nonzero(interaction & ~valid)
            / n_interaction
        )

    else:
        unknown_fraction = 0.0

    brt_unknown_cost = (
        150.0 * unknown_fraction
    )

    # --------------------------------
    # Geometrical collision
    # --------------------------------

    ego_centers = np.stack([
        np.column_stack((
            X_E + EGO_LENGTH / 4.0 * np.cos(PSI_E),
            Y_E + EGO_LENGTH / 4.0 * np.sin(PSI_E),
        )),
        np.column_stack((
            X_E - EGO_LENGTH / 4.0 * np.cos(PSI_E),
            Y_E - EGO_LENGTH / 4.0 * np.sin(PSI_E),
        )),
    ], axis=1)

    human_centers = np.stack([
        np.column_stack((
            X_H + HUMAN_LENGTH / 4.0,
            np.full_like(X_H, Y_H_initial),
        )),
        np.column_stack((
            X_H - HUMAN_LENGTH / 4.0,
            np.full_like(X_H, Y_H_initial),
        )),
    ], axis=1)

    difference = (
        ego_centers[:, :, None, :]
        - human_centers[:, None, :, :]
    )

    min_clearance = (
        np.linalg.norm(difference, axis=-1)
        .min(axis=(1, 2))
        - safety_r
    )

    collision_depth = max(
        0.0,
        -float(np.min(min_clearance)),
    )

    collision_cost = (
        400.0 * float(collision_depth > 1e-3)
        + 100.0 * (collision_depth / safety_r)**2
    )

    # --------------------------------
    # Overtake completion
    # --------------------------------

    longitudinal_extent = (
        EGO_LENGTH / 2.0 * np.abs(np.cos(PSI_E))
        + EGO_WIDTH / 2.0 * np.abs(np.sin(PSI_E))
    )

    required_lead = (
        longitudinal_extent
        + HUMAN_LENGTH / 2.0
        + OVERTAKE_MARGIN
    )

    lead_margin = (
        X_E - X_H - required_lead
    )

    alongside = (
        np.abs(X_E - X_H)
        <= (EGO_LENGTH + HUMAN_LENGTH) / 2.0 + 1.0
    )

    left_pass = (
        alongside
        & (
            Y_E - Y_H_initial
            >= (EGO_WIDTH + HUMAN_WIDTH) / 2.0
            + LATERAL_MARGIN
        )
    )

    left_seen = np.maximum.accumulate(left_pass)

    pass_candidates = np.flatnonzero(
        (lead_margin >= 0.0) & left_seen
    )

    overtake_completed = bool(
        pass_candidates.size > 0
        and lead_margin[-1] >= 0.0
        and collision_depth <= 1e-3
        and brt_margin_cost == 0.0
        and brt_unsafe_cost == 0.0
        and unknown_fraction <= 0.05
    )

    if overtake_completed:

        t_pass = pass_candidates[0] * dt

        overtake_cost = (
            -100.0
            - 20.0 * (1.0 - t_pass / Tsim)
        )

    else:

        lead_deficit = max(
            0.0,
            -float(lead_margin[-1]),
        )

        overtake_cost = (
            80.0
            + 20.0 * min(
                lead_deficit / max(required_lead[-1], 1.0),
                5.0,
            )
        )

    # --------------------------------
    # Weighted score components
    # --------------------------------

    velocity_cost = 10.0 * velocity
    control_cost = 0.15 * control_effort
    smoothness_cost = 0.30 * smoothness
    lane_cost = 2.0 * lane_return
    hj_cost = 10.0 * hj_slack
    edge_cost = 2.0 * edge_rms + edge_peak

    total = (
        velocity_cost
        + control_cost
        + smoothness_cost
        + lane_cost
        + hj_cost
        + edge_cost
        + brt_margin_cost
        + brt_unsafe_cost
        + brt_unknown_cost
        + collision_cost
        + overtake_cost
    )

    return {
        "velocity": float(velocity_cost),
        "control_effort": float(control_cost),
        "smoothness": float(smoothness_cost),
        "lane_return": float(lane_cost),
        "hj_slack": float(hj_cost),
        "edge": float(edge_cost),
        "brt_margin": float(brt_margin_cost),
        "brt_unsafe": float(brt_unsafe_cost),
        "brt_unknown": float(brt_unknown_cost),
        "collision": float(collision_cost),
        "overtake": float(overtake_cost),
        "overtake_completed": overtake_completed,
        "failure_penalty": 0.0,
        "total": float(total),
    }


def score(result, scenario):

    return score_components(
        result,
        scenario,
    )["total"]