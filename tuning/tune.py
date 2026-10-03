import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import optuna


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, file_path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def load_project(project_name):

    project_path = ROOT / project_name

    if not project_path.exists():
        raise FileNotFoundError(
            f"Project folder not found: {project_path}"
        )

    main_path = project_path / "main.py"
    config_path = project_path / "tuning_config.py"

    if not main_path.exists():
        raise FileNotFoundError(
            f"Missing file: {main_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing file: {config_path}"
        )

    # Allows imports such as:
    # from acados_settings import ...
    sys.path.insert(
        0,
        str(project_path),
    )

    main = load_module(
        f"{project_name}_main",
        main_path,
    )

    config = load_module(
        f"{project_name}_tuning_config",
        config_path,
    )

    return main, config


def aggregate_scores(scores):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    worst_score = np.max(scores)

    return float(
        0.75 * mean_score
        + 0.15 * std_score
        + 0.25 * worst_score
    )


def run_scenarios(
    weights,
    scenarios,
    main,
    config,
):

    scores = []

    for scenario in scenarios:

        result = main.run_simulation(
            weights=weights,
            scenario=scenario,
            make_plots=False,
            make_animation=False,
            raise_on_fail=False,
        )

        scenario_score = config.score(
            result,
            scenario,
        )

        scores.append(
            scenario_score
        )

    return scores

def evaluate_weights(
    weights,
    scenarios,
    main,
    config,
):

    scores = []

    component_names = [
        "velocity",
        "control_effort",
        "smoothness",
        "lane_return",
        "hj_slack",
        "overtake_penalty",
        "failure_penalty",
    ]

    components = {
        name: []
        for name in component_names
    }

    failures = 0
    failure_details = []

    for scenario in scenarios:

        result = main.run_simulation(
            weights=weights,
            scenario=scenario,
            make_plots=False,
            make_animation=False,
            raise_on_fail=False,
        )

        if not result["success"]:
            failures += 1
            failure_details.append({
                "scenario": scenario,
                "failure_phase": result.get(
                    "failure_phase",
                    "simulation",
                ),
                "failure_time": result.get(
                    "failure_time",
                    None,
                ),
                "status": result.get(
                    "solver_status",
                    result.get(
                        "integrator_status",
                        None,
                    ),
                ),
            })

            print(
                f"  FAILURE | "
                f"scenario={scenario} | "
                f"time={result.get('failure_time', 'N/A')} | "
                f"status={result.get('solver_status', result.get('integrator_status', 'N/A'))}"
            )

        values = config.score_components(
            result,
            scenario,
        )

        scores.append(
            values["total"]
        )

        if result["success"]:
            for name in component_names:
                if name != "failure_penalty":
                    components[name].append(
                        values[name]
                    )

    component_means = {}

    for name, values in components.items():

        if len(values) > 0:
            component_means[name] = float(
                np.mean(values)
            )
        else:
            component_means[name] = float("nan")

    return {
        "scores": [
            float(x) for x in scores
        ],
        "mean": float(np.mean(scores)),
        "worst": float(np.max(scores)),
        "std": float(np.std(scores)),
        "aggregate": aggregate_scores(scores),
        "failures": failures,
        "failure_details": failure_details,
        "components": component_means,
    }

def print_evaluation(
    title,
    evaluation,
):

    print("\n" + title)
    print("-" * 50)

    print(
        f"Aggregate score : "
        f"{evaluation['aggregate']:.6f}"
    )

    print(
        f"Mean score      : "
        f"{evaluation['mean']:.6f}"
    )

    print(
        f"Worst score     : "
        f"{evaluation['worst']:.6f}"
    )

    print(
        f"Std score       : "
        f"{evaluation['std']:.6f}"
    )

    print(
        f"Failures        : "
        f"{evaluation['failures']}"
    )

    if evaluation["failure_details"]:

        print("\nFailure details:")

        for failure in evaluation["failure_details"]:

            print(
                f"  status={failure['status']} | "
                f"time={failure['failure_time']} | "
                f"scenario={failure['scenario']}"
            )

    print("\nMean score decomposition:")

    for name, value in evaluation["components"].items():
        print(
            f"  {name:20s}: {value:.6f}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "project",
        type=str,
        help="Project folder to tune",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )

    args = parser.parse_args()

    project_main, config = load_project(
        args.project
    )

    def objective(trial):

        weights = {}

        for name, bounds in config.WEIGHT_BOUNDS.items():

            weights[name] = trial.suggest_float(
                name,
                bounds[0],
                bounds[1],
                log=True,
            )

        scores = run_scenarios(
            weights,
            config.TRAIN_SCENARIOS,
            project_main,
            config,
        )

        return aggregate_scores(
            scores
        )

    sampler = optuna.samplers.TPESampler(
        seed=config.SEED
    )

    study_path = ROOT / "tuning" / "results" / f"{args.project}.db"

    study = optuna.create_study(
        study_name=args.project,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
    )

    best_weights = study.best_params

    print("\n")
    print("=" * 50)
    print("BEST WEIGHTS")
    print("=" * 50)

    for name, value in best_weights.items():
        print(
            f"{name:20s}: {value:.6g}"
        )

    # --------------------------------
    # Baseline evaluation
    # --------------------------------

    baseline_train = evaluate_weights(
        config.BASELINE_WEIGHTS,
        config.TRAIN_SCENARIOS,
        project_main,
        config,
    )

    baseline_validation = evaluate_weights(
        config.BASELINE_WEIGHTS,
        config.VALIDATION_SCENARIOS,
        project_main,
        config,
    )

    # --------------------------------
    # Best Optuna evaluation
    # --------------------------------

    best_train = evaluate_weights(
        best_weights,
        config.TRAIN_SCENARIOS,
        project_main,
        config,
    )

    best_validation = evaluate_weights(
        best_weights,
        config.VALIDATION_SCENARIOS,
        project_main,
        config,
    )

    # --------------------------------
    # Print comparison
    # --------------------------------

    print("\n")
    print("=" * 50)
    print("BASELINE")
    print("=" * 50)

    print_evaluation(
        "TRAIN",
        baseline_train,
    )

    print_evaluation(
        "VALIDATION",
        baseline_validation,
    )

    print("\n")
    print("=" * 50)
    print("BEST OPTUNA")
    print("=" * 50)

    print_evaluation(
        "TRAIN",
        best_train,
    )

    print_evaluation(
        "VALIDATION",
        best_validation,
    )


    results_dir = ROOT / "tuning" / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "project": args.project,
        "best_value": float(study.best_value),

        "baseline_weights": config.BASELINE_WEIGHTS,
        "best_weights": best_weights,

        "baseline": {
            "train": baseline_train,
            "validation": baseline_validation,
        },

        "best": {
            "train": best_train,
            "validation": best_validation,
        },
    }

    output_path = (
        results_dir
        / f"{args.project}.json"
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()