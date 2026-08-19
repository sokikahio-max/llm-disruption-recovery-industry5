"""Run the controlled live LLM explanation study.

Usage:
    export OPENAI_API_KEY='...'
    python wp5_live_runner.py

The script resumes safely, never logs the API key, validates every response, and
uses the deterministic template as a communication fallback when an LLM response
is invalid. The fallback never changes the operational action.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import json
import os
import time

import pandas as pd
from openai import OpenAI

from explanation_evaluation import (
    EXPLANATION_JSON_SCHEMA,
    build_messages,
    estimate_cost_usd,
    generate_template_structured,
    pairwise_jaccard,
    render_structured_explanation,
    score_as_dict,
    score_output,
    validate_structure,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RAW_JSONL = RESULTS / "llm_call_records.jsonl"


def load_payloads() -> List[Dict[str, Any]]:
    records = []
    for line in (RESULTS / "evidence_payloads.jsonl").read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def completed_keys() -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not RAW_JSONL.exists():
        return keys
    for line in RAW_JSONL.read_text().splitlines():
        try:
            row = json.loads(line)
            keys.add((str(row["scenario_key"]), int(row["repeat"])))
        except Exception:
            continue
    return keys


def api_call(
    client: OpenAI,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    messages = build_messages(payload)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=str(config["model"]),
        messages=messages,
        temperature=float(config["temperature"]),
        max_tokens=int(config["max_output_tokens"]),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "recovery_explanation",
                "strict": True,
                "schema": EXPLANATION_JSON_SCHEMA,
            },
        },
        timeout=float(config["timeout_seconds"]),
    )
    latency = time.perf_counter() - started
    raw = response.choices[0].message.content or ""
    parsed: Optional[Dict[str, Any]] = None
    parse_error = ""
    try:
        candidate = json.loads(raw)
        if not isinstance(candidate, dict):
            raise TypeError("top-level JSON is not an object")
        parsed = candidate
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{exc}"

    usage = response.usage
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return {
        "latency_seconds": latency,
        "raw_output": raw,
        "parsed_output": parsed,
        "parse_error": parse_error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def summarize(config: Mapping[str, Any], payload_lookup: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [json.loads(line) for line in RAW_JSONL.read_text().splitlines() if line.strip()]
    flat: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    for row in rows:
        payload = payload_lookup[row["scenario_key"]]
        output = row.get("parsed_output")
        if isinstance(output, dict):
            schema_valid, validation_errors = validate_structure(output, payload)
        else:
            schema_valid, validation_errors = False, ("json_parse_failure",)
        fallback_output = generate_template_structured(payload)
        effective = output if schema_valid else fallback_output
        score = score_output(
            effective,
            payload,
            row["scenario_key"],
            "llm" if schema_valid else "template_fallback",
            int(row["repeat"]),
        )
        score_row = score_as_dict(score)
        score_row["original_llm_schema_valid"] = schema_valid
        score_row["fallback_used"] = not schema_valid
        scores.append(score_row)
        flat.append({
            "scenario_key": row["scenario_key"],
            "repeat": row["repeat"],
            "model": row["model"],
            "temperature": row["temperature"],
            "schema_valid": schema_valid,
            "fallback_used": not schema_valid,
            "validation_errors": "|".join(validation_errors),
            "latency_seconds": row["latency_seconds"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "estimated_cost_usd": row["estimated_cost_usd"],
            "rendered_explanation": render_structured_explanation(effective),
            "raw_output": row.get("raw_output", ""),
        })

    calls = pd.DataFrame(flat)
    score_df = pd.DataFrame(scores)
    calls.to_csv(RESULTS / "llm_call_metrics.csv", index=False)
    score_df.to_csv(RESULTS / "llm_automated_scores.csv", index=False)

    consistency_rows = []
    for scenario_key, group in calls.groupby("scenario_key"):
        valid = group.loc[~group["fallback_used"], "rendered_explanation"].tolist()
        consistency_rows.append({
            "scenario_key": scenario_key,
            "calls": len(group),
            "valid_calls": int((~group["fallback_used"]).sum()),
            "validity_rate": float((~group["fallback_used"]).mean()),
            "mean_pairwise_jaccard_valid_outputs": pairwise_jaccard(valid) if valid else 0.0,
            "mean_latency_seconds": float(group["latency_seconds"].mean()),
            "p95_latency_seconds": float(group["latency_seconds"].quantile(0.95)),
            "total_cost_usd": float(group["estimated_cost_usd"].sum()),
        })
    pd.DataFrame(consistency_rows).to_csv(RESULTS / "llm_consistency_summary.csv", index=False)

    overall = {
        "live_llm_run_completed": True,
        "calls": len(calls),
        "valid_json_and_schema_rate": float((~calls["fallback_used"]).mean()),
        "fallback_rate": float(calls["fallback_used"].mean()),
        "mean_latency_seconds": float(calls["latency_seconds"].mean()),
        "p50_latency_seconds": float(calls["latency_seconds"].quantile(0.50)),
        "p95_latency_seconds": float(calls["latency_seconds"].quantile(0.95)),
        "total_input_tokens": int(calls["input_tokens"].sum()),
        "total_output_tokens": int(calls["output_tokens"].sum()),
        "total_estimated_cost_usd": float(calls["estimated_cost_usd"].sum()),
        "mean_factual_reliability_score": float(score_df["factual_reliability_score"].mean()),
        "selected_action_accuracy": float(score_df["selected_action_correct"].mean()),
        "governance_f1": float(score_df["governance_f1"].mean()),
        "numeric_grounding_ratio": float(score_df["numeric_grounding_ratio"].mean()),
        "model": config["model"],
        "temperature": config["temperature"],
        "max_output_tokens": config["max_output_tokens"],
    }
    (RESULTS / "llm_run_status.json").write_text(json.dumps(overall, indent=2))


def main() -> None:
    config = json.loads((ROOT / "config" / "llm_config.json").read_text())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. No live calls were made.")
    client = OpenAI(api_key=api_key)
    payload_records = load_payloads()
    payload_lookup = {record["scenario_key"]: record["payload"] for record in payload_records}
    done = completed_keys()

    for record in payload_records:
        scenario_key = record["scenario_key"]
        payload = record["payload"]
        for repeat in range(1, int(config["repetitions_per_scenario"]) + 1):
            if (scenario_key, repeat) in done:
                continue
            last_error = ""
            call: Optional[Dict[str, Any]] = None
            for attempt in range(int(config["max_retries"]) + 1):
                try:
                    call = api_call(client, payload, config)
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}:{exc}"
                    if attempt < int(config["max_retries"]):
                        time.sleep(2 ** attempt)
            if call is None:
                call = {
                    "latency_seconds": 0.0,
                    "raw_output": "",
                    "parsed_output": None,
                    "parse_error": last_error,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            call["scenario_key"] = scenario_key
            call["payload_hash"] = record["payload_hash"]
            call["repeat"] = repeat
            call["model"] = config["model"]
            call["temperature"] = config["temperature"]
            call["max_output_tokens"] = config["max_output_tokens"]
            call["estimated_cost_usd"] = estimate_cost_usd(
                int(call["input_tokens"]),
                int(call["output_tokens"]),
                float(config["input_usd_per_million_tokens"]),
                float(config["output_usd_per_million_tokens"]),
            )
            append_jsonl(RAW_JSONL, call)
            time.sleep(float(config["pause_between_calls_seconds"]))

    summarize(config, payload_lookup)
    print(f"LLM explanation evaluation complete. Results are in {RESULTS}.")


if __name__ == "__main__":
    main()
