"""Run the paired synthetic and benchmark-derived operational experiments.

The LLM is not used in operational selection. All compared policies receive the
same frozen disruption state within each scenario.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json, platform, sys
import numpy as np
import pandas as pd

from benchmark_instances import create_benchmark_scenarios
from recovery_simulation import create_scenario, run_stronger_comparator_experiment

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WEIGHTS = {"delay": 0.40, "energy": 0.20, "quality": 0.20, "safety": 0.20}
BENCHMARK_NAMES = ("ft06", "la01", "la06")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "tardiness", "makespan", "processing_energy", "quality_exposure",
        "safety_exposure", "quality_violations", "safety_violations",
    ]
    rows: List[Dict[str, object]] = []
    for (dataset, policy), group in df.groupby(["dataset", "policy"], sort=True):
        row: Dict[str, object] = {
            "dataset": dataset,
            "policy": policy,
            "n": len(group),
            "completion_rate": float(group["completed"].mean()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> Dict[str, pd.DataFrame]:
    synthetic_scenarios = tuple(
        create_scenario(
            seed=4_200 + run_id,
            scenario_id=run_id,
            n_jobs=20,
            n_machines=5,
            operations_per_job=4,
            alternative_probability=0.60,
            repair_duration_bounds=(8, 24),
        )
        for run_id in range(100)
    )
    synthetic_policy, synthetic_actions = run_stronger_comparator_experiment(
        synthetic_scenarios, weights=WEIGHTS
    )
    synthetic_policy.insert(0, "dataset", "synthetic")
    synthetic_actions.insert(0, "dataset", "synthetic")
    synthetic_policy.to_csv(RESULTS_DIR / "synthetic_comparator_results.csv", index=False)
    synthetic_actions.to_csv(RESULTS_DIR / "synthetic_action_evaluations.csv", index=False)

    benchmark_scenarios, benchmark_metadata = create_benchmark_scenarios(
        DATA_DIR / "jobshop1.txt",
        BENCHMARK_NAMES,
        replications_per_instance=20,
        seed0=60_000,
        alternative_probability=0.50,
        due_date_tightness=1.00,
    )
    benchmark_policy, benchmark_actions = run_stronger_comparator_experiment(
        benchmark_scenarios, weights=WEIGHTS
    )
    metadata_df = pd.DataFrame(benchmark_metadata)
    benchmark_policy = benchmark_policy.merge(metadata_df, on="scenario_id", how="left", validate="many_to_one")
    benchmark_actions = benchmark_actions.merge(metadata_df, on="scenario_id", how="left", validate="many_to_one")
    benchmark_policy.insert(0, "dataset", "benchmark_derived")
    benchmark_actions.insert(0, "dataset", "benchmark_derived")
    benchmark_policy.to_csv(RESULTS_DIR / "benchmark_comparator_results.csv", index=False)
    benchmark_actions.to_csv(RESULTS_DIR / "benchmark_action_evaluations.csv", index=False)
    metadata_df.to_csv(RESULTS_DIR / "benchmark_manifest.csv", index=False)

    combined = pd.concat([synthetic_policy, benchmark_policy], ignore_index=True, sort=False)
    combined.to_csv(RESULTS_DIR / "combined_policy_results.csv", index=False)
    summary = _aggregate(combined)
    summary.to_csv(RESULTS_DIR / "descriptive_summary.csv", index=False)

    governance = combined[combined["policy"] == "governance_aware_no_llm"]
    action_frequency = (
        governance.groupby(["dataset", "selected_action"], dropna=False)
        .size().rename("count").reset_index()
    )
    action_frequency["proportion"] = action_frequency.groupby("dataset")["count"].transform(
        lambda values: values / values.sum()
    )
    action_frequency.to_csv(RESULTS_DIR / "governance_action_frequency.csv", index=False)

    environment = {
        "base_weights": WEIGHTS,
        "synthetic_scenarios": 100,
        "benchmark_instances": list(BENCHMARK_NAMES),
        "benchmark_replications_per_instance": 20,
        "comparators": ["right_shift", "reactive_edd", "reactive_spt", "reactive_mwkr", "reactive_min_slack"],
        "operational_selection_uses_llm": False,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (RESULTS_DIR / "operational_environment.json").write_text(json.dumps(environment, indent=2))
    return {
        "synthetic_policy": synthetic_policy,
        "benchmark_policy": benchmark_policy,
        "summary": summary,
        "action_frequency": action_frequency,
    }


if __name__ == "__main__":
    outputs = run()
    print("Operational experiments completed successfully.")
    print(outputs["summary"][["dataset", "policy", "n", "tardiness_mean"]].round(3).to_string(index=False))
