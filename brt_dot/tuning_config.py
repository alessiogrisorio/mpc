import numpy as np

from acados_settings import (
    EGO_LENGTH,
    EGO_WIDTH,
    HUMAN_LENGTH,
    safety_r,
    ROAD_Y_MIN,
    ROAD_Y_MAX,
    EDGE_MARGIN,
)


WEIGHT_BOUNDS = {
    "lane": (0.2, 20.0),
    "edge": (0.1, 100.0),
    "psi": (0.06, 6.0),
    "velocity": (50.0, 2000.0),
    "steering_rate": (0.001, 0.1),
    "acceleration": (0.18, 18.0),
}

BASELINE_WEIGHTS = {
    "lane": 3.890810431104229,
    "edge": 30.0,
    "psi": 1.4467453645031252,
    "velocity": 56.477006267018474,
    "steering_rate": 0.09888880927846047,
    "acceleration": 13.887061909157946,
}


SCENARIO_RANGES = {
    "v_ref": (3.0, 9.0),
    "v_ego_initial": (3.0, 9.0),
    "v_human": (2.0, 6.0),
    "human_distance": (10.0, 20.0),
}


N_TRAIN = 6
N_VALIDATION = 4

SEED = 42


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
                0.0,
                0.0,
                float(v_ego_initial),
                0.0,
            ],
            "X_H_initial": float(human_distance),
            "Y_H_initial": 0.0,
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
            "overtake_penalty": 0.0,
            "failure_penalty": 1e6,
            "total": 1e6,
        }

    simX = result["simX"]
    simU = result["simU"]
    dt = result["dt"]

    # Vicinanza della carrozzeria ai bordi
    lateral_extent = (
        0.5 * EGO_LENGTH * np.abs(np.sin(simX[:, 2]))
        + 0.5 * EGO_WIDTH * np.abs(np.cos(simX[:, 2]))
    )

    edge_error = (
        np.maximum(
            0.0,
            ROAD_Y_MIN + EDGE_MARGIN - (simX[:, 1] - lateral_extent),
        )
        + np.maximum(
            0.0,
            simX[:, 1] + lateral_extent - (ROAD_Y_MAX - EDGE_MARGIN),
        )
    ) / EDGE_MARGIN

    edge_rms = np.sqrt(np.mean(edge_error**2))
    edge_peak = np.max(edge_error)

    v_ref = scenario["v_ref"]
    v_H = scenario["v_H"]
    X_H_initial = scenario["X_H_initial"]

    # Velocity tracking
    velocity = np.sqrt(
        np.mean(
            ((simX[:, 3] - v_ref) / 2.0) ** 2
        )
    )

    # Control effort
    steering_effort = np.mean(
        (simU[:, 0] / 0.087) ** 2
    )

    acceleration_effort = np.mean(
        (simU[:, 1] / 7.0) ** 2
    )

    control_effort = (
        steering_effort
        + acceleration_effort
    )

    # Smoothness
    du = np.diff(simU, axis=0)

    smoothness = np.mean(
        (du[:, 0] / 0.087) ** 2
        +
        (du[:, 1] / 7.0) ** 2
    )

    # Return to right lane
    lane_return = abs(simX[-1, 1]) / 4.0

    # HJ slack
    hj_slack = np.max(
        result["simHJSlack"]
    )

    # Check completed overtake
    Tsim = dt * (len(simX) - 1)

    X_H_final = (
        X_H_initial
        + v_H * Tsim
    )

    lead = simX[-1, 0] - X_H_final

    required_lead = (
        EGO_LENGTH / 4.0
        + HUMAN_LENGTH / 4.0
        + safety_r
    )

    lead_deficit = max(
        0.0,
        required_lead - lead,
    )

    if lead_deficit > 0.0:
        overtake_penalty = (
            1000.0
            + 100.0
            * (lead_deficit / required_lead) ** 2
        )
    else:
        overtake_penalty = 0.0

    # Weighted contributions to tuning score
    velocity_cost = 1.0 * velocity
    control_cost = 0.15 * control_effort
    smoothness_cost = 0.30 * smoothness
    lane_cost = 2.0 * lane_return
    hj_cost = 10.0 * hj_slack
    edge_cost = 2.0 * edge_rms + edge_peak

    failure_penalty = 0.0

    total = (
        velocity_cost
        + control_cost
        + smoothness_cost
        + lane_cost
        + hj_cost
        + overtake_penalty
        + failure_penalty
        + edge_cost
    )

    return {
        "velocity": float(velocity_cost),
        "control_effort": float(control_cost),
        "smoothness": float(smoothness_cost),
        "lane_return": float(lane_cost),
        "hj_slack": float(hj_cost),
        "overtake_penalty": float(overtake_penalty),
        "failure_penalty": float(failure_penalty),
        "total": float(total),
        "edge": float(edge_cost),
    }


def score(result, scenario):
    return score_components(
        result,
        scenario,
    )["total"]