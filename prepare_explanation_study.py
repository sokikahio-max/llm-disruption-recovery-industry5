"""Prepare the stratified evidence set from the frozen action tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping
import json

import pandas as pd

from explanation_evaluation import (
    build_messages,
    estimate_cost_usd,
    estimate_tokens,
    generate_template_structured,
    payload_hash,
    render_structured_explanation,
    score_as_dict,
    score_output,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "results"


def _rejection_reasons(row: Mapping[str, Any], q_threshold: float, s_threshold: float) -> List[str]:
    reasons: List[str] = []
    if not bool(row["completed"]):
        reasons.append("incomplete_schedule")
    if float(row["max_quality_risk"]) > q_threshold + 1e-12:
        reasons.append("quality_threshold_exceeded")
    if float(row["max_safety_risk"]) > s_threshold + 1e-12:
        reasons.append("safety_threshold_exceeded")
    return reasons


def build_payload(
    dataset: str,
    scenario_id: int,
    action_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = action_rows.sort_values(["score", "action"], kind="stable")
    selected_rows = rows.loc[rows["selected"].astype(bool)]
    if len(selected_rows) != 1:
        raise ValueError(f"Expected exactly one selected action for {dataset}:{scenario_id}.")
    selected = selected_rows.iloc[0]
    q_threshold = float(config["quality_threshold"])
    s_threshold = float(config["safety_threshold"])

    candidates: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        candidates.append({
            "action": str(row["action"]),
            "feasible": bool(row["feasible"]),
            "selected": bool(row["selected"]),
            "score": round(float(row["score"]), 6),
            "tardiness": round(float(row["tardiness"]), 6),
            "energy_proxy": round(float(row["energy"]), 6),
            "quality_exposure_proxy": round(float(row["quality"]), 6),
            "safety_exposure_proxy": round(float(row["safety"]), 6),
            "max_quality_risk": round(float(row["max_quality_risk"]), 6),
            "max_safety_risk": round(float(row["max_safety_risk"]), 6),
            "rejected_reasons": _rejection_reasons(row, q_threshold, s_threshold),
        })

    scenario: Dict[str, Any] = {
        "scenario_key": f"{dataset}:{int(scenario_id)}",
        "scenario_id": int(scenario_id),
        "dataset": dataset,
        "failed_machine": int(selected["failure_machine"]),
        "failure_start": round(float(selected["failure_start"]), 6),
        "repair_duration": round(float(selected["repair_duration"]), 6),
    }
    if dataset == "synthetic":
        scenario.update({"number_of_jobs": 20, "source": "controlled synthetic job-shop generator"})
    else:
        scenario.update({
            "number_of_jobs": int(selected["n_jobs"]),
            "number_of_machines": int(selected["n_machines"]),
            "benchmark": str(selected["benchmark"]),
            "replication": int(selected["replication"]),
            "source": "OR-Library benchmark-derived disruption scenario",
        })

    return {
        "scenario": scenario,
        "decision_configuration": {
            "weights": dict(config["base_weights"]),
            "quality_threshold": q_threshold,
            "safety_threshold": s_threshold,
            "deterministic_selected_action": str(selected["action"]),
            "fallback_used": False,
        },
        "candidate_actions": candidates,
        "metric_notice": (
            "Energy, quality, and safety values are normalized experimental proxies, "
            "not physical kWh, defect probabilities, or incident probabilities."
        ),
    }


def prepare() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config" / "llm_config.json").read_text())
    frames = {
        "synthetic": pd.read_csv(DATA / "synthetic_action_evaluations.csv"),
        "benchmark_derived": pd.read_csv(DATA / "benchmark_action_evaluations.csv"),
    }

    payload_records: List[Dict[str, Any]] = []
    selection_records: List[Dict[str, Any]] = []
    template_records: List[Dict[str, Any]] = []
    score_records: List[Dict[str, Any]] = []
    cost_records: List[Dict[str, Any]] = []

    for dataset, ids in config["scenario_selection"].items():
        df = frames[dataset]
        for scenario_id in ids:
            action_rows = df.loc[df["scenario_id"] == int(scenario_id)].copy()
            if len(action_rows) != 4:
                raise ValueError(f"Expected four action rows for {dataset}:{scenario_id}, got {len(action_rows)}.")
            payload = build_payload(dataset, int(scenario_id), action_rows, config)
            scenario_key = payload["scenario"]["scenario_key"]
            p_hash = payload_hash(payload)
            payload_records.append({"scenario_key": scenario_key, "payload_hash": p_hash, "payload": payload})

            selected_row = action_rows.loc[action_rows["selected"].astype(bool)].iloc[0]
            selection_records.append({
                "scenario_key": scenario_key,
                "dataset": dataset,
                "scenario_id": int(scenario_id),
                "selected_action": str(selected_row["action"]),
                "tardiness": float(selected_row["tardiness"]),
                "energy_proxy": float(selected_row["energy"]),
                "quality_exposure_proxy": float(selected_row["quality"]),
                "safety_exposure_proxy": float(selected_row["safety"]),
                "repair_duration": float(selected_row["repair_duration"]),
                "failure_machine": int(selected_row["failure_machine"]),
                "benchmark": str(selected_row.get("benchmark", "")) if dataset != "synthetic" else "",
                "governance_rejection_count": sum(bool(row["rejected_reasons"]) for row in payload["candidate_actions"]),
                "payload_hash": p_hash,
            })

            template = generate_template_structured(payload)
            template_text = render_structured_explanation(template)
            template_records.append({
                "scenario_key": scenario_key,
                "dataset": dataset,
                "scenario_id": int(scenario_id),
                "method": "template",
                "structured_output_json": json.dumps(template, sort_keys=True),
                "rendered_explanation": template_text,
            })
            score_records.append(score_as_dict(score_output(template, payload, scenario_key, "template", 0)))

            messages = build_messages(payload)
            estimated_input, estimated_output = estimate_tokens(messages)
            per_call = estimate_cost_usd(
                estimated_input,
                estimated_output,
                float(config["input_usd_per_million_tokens"]),
                float(config["output_usd_per_million_tokens"]),
            )
            cost_records.append({
                "scenario_key": scenario_key,
                "estimated_input_tokens_per_call": estimated_input,
                "assumed_output_tokens_per_call": estimated_output,
                "repetitions": int(config["repetitions_per_scenario"]),
                "estimated_cost_usd_per_call": per_call,
                "estimated_cost_usd_all_repetitions": per_call * int(config["repetitions_per_scenario"]),
            })

    pd.DataFrame(selection_records).to_csv(RESULTS / "selected_scenarios.csv", index=False)
    pd.DataFrame(template_records).to_csv(RESULTS / "template_explanations.csv", index=False)
    pd.DataFrame(score_records).to_csv(RESULTS / "template_automated_scores.csv", index=False)
    cost_df = pd.DataFrame(cost_records)
    total = pd.DataFrame([{
        "scenario_key": "TOTAL",
        "estimated_input_tokens_per_call": cost_df["estimated_input_tokens_per_call"].sum(),
        "assumed_output_tokens_per_call": cost_df["assumed_output_tokens_per_call"].sum(),
        "repetitions": int(config["repetitions_per_scenario"]),
        "estimated_cost_usd_per_call": cost_df["estimated_cost_usd_per_call"].sum(),
        "estimated_cost_usd_all_repetitions": cost_df["estimated_cost_usd_all_repetitions"].sum(),
    }])
    pd.concat([cost_df, total], ignore_index=True).to_csv(RESULTS / "preflight_token_cost_estimate.csv", index=False)

    with (RESULTS / "evidence_payloads.jsonl").open("w", encoding="utf-8") as handle:
        for record in payload_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    status = {
        "live_llm_run_completed": False,
        "reason": "OPENAI_API_KEY was not available in the execution environment.",
        "prepared_scenarios": len(payload_records),
        "planned_calls": len(payload_records) * int(config["repetitions_per_scenario"]),
        "model": config["model"],
        "temperature": config["temperature"],
        "max_output_tokens": config["max_output_tokens"],
        "next_command": "python run_llm_explanation_study.py",
    }
    (RESULTS / "llm_preparation_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    prepare()
