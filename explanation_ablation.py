"""No-LLM, deterministic-template, and LLM explanation ablation utilities.

Operational action selection remains deterministic and governance-aware in all
three variants. This isolates the explanatory contribution of the LLM from the
operational performance of the recovery framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import json
import os
import time

from recovery_simulation import DecisionResult, Scenario


ABLATION_NO_LLM = "no_llm"
ABLATION_TEMPLATE = "template_explanation"
ABLATION_LLM = "llm_explanation"


@dataclass(frozen=True)
class LLMCallRecord:
    model: str
    temperature: float
    max_output_tokens: int
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    valid: bool
    fallback_used: bool
    validation_errors: Tuple[str, ...]
    raw_output: str
    parsed_output: Optional[Mapping[str, Any]]


def _round(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def build_evidence_payload(scenario: Scenario, decision: DecisionResult) -> Dict[str, Any]:
    """Build the sole structured evidence supplied to an explanation method."""
    rows: List[Dict[str, Any]] = []
    for action, row in decision.action_table.iterrows():
        rejected_reasons: List[str] = []
        if not bool(row["completed"]):
            rejected_reasons.append("incomplete_schedule")
        if float(row["max_quality_risk"]) > decision.quality_threshold + 1e-12:
            rejected_reasons.append("quality_threshold_exceeded")
        if float(row["max_safety_risk"]) > decision.safety_threshold + 1e-12:
            rejected_reasons.append("safety_threshold_exceeded")
        rows.append(
            {
                "action": action,
                "feasible": bool(row["feasible"]),
                "selected": bool(row["selected"]),
                "score": _round(row["score"]),
                "tardiness": _round(row["tardiness"], 3),
                "energy_proxy": _round(row["energy"], 3),
                "quality_exposure_proxy": _round(row["quality"]),
                "safety_exposure_proxy": _round(row["safety"]),
                "max_quality_risk": _round(row["max_quality_risk"]),
                "max_safety_risk": _round(row["max_safety_risk"]),
                "rejected_reasons": rejected_reasons,
            }
        )

    return {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "failed_machine": scenario.failure.machine,
            "failure_start": _round(scenario.failure.start, 3),
            "repair_duration": _round(scenario.failure.duration, 3),
            "number_of_jobs": len(scenario.jobs),
        },
        "decision_configuration": {
            "weights": {key: _round(value) for key, value in decision.weights.items()},
            "quality_threshold": _round(decision.quality_threshold),
            "safety_threshold": _round(decision.safety_threshold),
            "deterministic_selected_action": decision.selected_action,
            "fallback_used": decision.fallback_used,
        },
        "candidate_actions": rows,
        "metric_notice": (
            "Energy, quality, and safety values are normalized experimental proxies, "
            "not physical kWh, defect probabilities, or incident probabilities."
        ),
    }


def generate_template_explanation(payload: Mapping[str, Any]) -> str:
    """Generate a deterministic, fully auditable explanation baseline."""
    selected_name = payload["decision_configuration"]["deterministic_selected_action"]
    action_map = {row["action"]: row for row in payload["candidate_actions"]}
    selected = action_map[selected_name]
    feasible_others = [
        row for row in payload["candidate_actions"]
        if row["feasible"] and row["action"] != selected_name
    ]
    rejected = [row for row in payload["candidate_actions"] if not row["feasible"]]

    lines = [
        f"Recommended action: {selected_name}.",
        (
            f"It is governance-feasible and has the lowest weighted score "
            f"({selected['score']:.4f}) among feasible actions. Its simulated outcomes are "
            f"tardiness {selected['tardiness']:.3f}, energy proxy "
            f"{selected['energy_proxy']:.3f}, quality-exposure proxy "
            f"{selected['quality_exposure_proxy']:.4f}, and safety-exposure proxy "
            f"{selected['safety_exposure_proxy']:.4f}."
        ),
    ]
    if feasible_others:
        runner_up = min(feasible_others, key=lambda row: (row["score"], row["action"]))
        lines.append(
            f"The closest feasible alternative is {runner_up['action']} with score "
            f"{runner_up['score']:.4f}; the recommendation follows the configured weighted "
            "trade-off rather than a claim of universal superiority."
        )
    if rejected:
        rejection_text = "; ".join(
            f"{row['action']} ({', '.join(row['rejected_reasons'])})" for row in rejected
        )
        lines.append(f"Governance rejected: {rejection_text}.")
    lines.append(str(payload["metric_notice"]))
    return " ".join(lines)


SYSTEM_PROMPT = """You are an industrial decision-support explanation layer.
Use only the structured evidence supplied by the user. Do not invent causes,
measurements, constraints, or operational facts. The deterministic governance
module has already selected the action; you must explain that selection, not
replace it. Treat proxy metrics as normalized comparison indicators. Return
strict JSON matching the requested schema."""


def build_llm_messages(payload: Mapping[str, Any]) -> List[Dict[str, str]]:
    schema = {
        "selected_action": "must equal deterministic_selected_action",
        "summary": "2-4 sentences grounded only in supplied evidence",
        "tradeoffs": [
            {
                "criterion": "delay|energy|quality|safety",
                "statement": "grounded comparison without invented numbers",
            }
        ],
        "governance_rejections": [
            {"action": "action name", "reasons": ["exact supplied rejection reason"]}
        ],
        "uncertainty_and_limits": "state that proxies and simulation limit generalization",
    }
    user_content = {
        "task": "Explain the deterministic recovery recommendation for a human operator.",
        "required_output_schema": schema,
        "evidence": payload,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, indent=2, sort_keys=True)},
    ]


def validate_llm_explanation(
    output: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Tuple[bool, Tuple[str, ...]]:
    """Validate structure, action consistency, and governance references."""
    errors: List[str] = []
    required = {
        "selected_action",
        "summary",
        "tradeoffs",
        "governance_rejections",
        "uncertainty_and_limits",
    }
    missing = sorted(required.difference(output.keys()))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")

    expected = payload["decision_configuration"]["deterministic_selected_action"]
    if output.get("selected_action") != expected:
        errors.append("selected_action_mismatch")
    if not isinstance(output.get("summary"), str) or not output.get("summary", "").strip():
        errors.append("invalid_summary")
    if not isinstance(output.get("tradeoffs"), list):
        errors.append("invalid_tradeoffs")
    if not isinstance(output.get("governance_rejections"), list):
        errors.append("invalid_governance_rejections")
    if not isinstance(output.get("uncertainty_and_limits"), str):
        errors.append("invalid_uncertainty_statement")

    valid_actions = {row["action"] for row in payload["candidate_actions"]}
    supplied_rejections = {
        row["action"]: set(row["rejected_reasons"])
        for row in payload["candidate_actions"]
        if row["rejected_reasons"]
    }
    for item in output.get("governance_rejections", []) if isinstance(output.get("governance_rejections"), list) else []:
        if not isinstance(item, Mapping):
            errors.append("malformed_rejection_item")
            continue
        action = item.get("action")
        reasons = item.get("reasons")
        if action not in valid_actions:
            errors.append(f"unknown_rejected_action:{action}")
            continue
        if action not in supplied_rejections:
            errors.append(f"unsupported_rejection:{action}")
            continue
        if not isinstance(reasons, list) or not set(reasons).issubset(supplied_rejections[action]):
            errors.append(f"unsupported_rejection_reason:{action}")

    return (not errors, tuple(errors))


def call_openai_explanation(
    payload: Mapping[str, Any],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 500,
) -> LLMCallRecord:
    """Optional live call. The experiment remains reproducible without it.

    Requires ``OPENAI_API_KEY`` and the ``openai`` package. Model can be set by
    ``OPENAI_MODEL``; the fallback value preserves the study's
    configuration for controlled re-evaluation.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; live LLM evaluation was not run.")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    from openai import OpenAI  # Imported only when live evaluation is requested.

    client = OpenAI(api_key=api_key)
    messages = build_llm_messages(payload)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        response_format={"type": "json_object"},
    )
    latency = time.perf_counter() - started
    raw = response.choices[0].message.content or ""
    parsed: Optional[Mapping[str, Any]] = None
    errors: Tuple[str, ...]
    try:
        parsed_candidate = json.loads(raw)
        if not isinstance(parsed_candidate, Mapping):
            raise TypeError("Top-level JSON is not an object.")
        parsed = parsed_candidate
        valid, errors = validate_llm_explanation(parsed, payload)
    except Exception as exc:
        valid = False
        errors = (f"json_parse_error:{type(exc).__name__}",)

    usage = getattr(response, "usage", None)
    return LLMCallRecord(
        model=model,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        latency_seconds=float(latency),
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        valid=valid,
        fallback_used=not valid,
        validation_errors=errors,
        raw_output=raw,
        parsed_output=parsed,
    )


def ablation_records(scenario: Scenario, decision: DecisionResult) -> List[Dict[str, Any]]:
    """Create aligned no-LLM/template/LLM-input records for one scenario."""
    payload = build_evidence_payload(scenario, decision)
    selected = decision.selected_result
    common = {
        "scenario_id": scenario.scenario_id,
        "selected_action": decision.selected_action,
        "tardiness": selected.tardiness,
        "energy_proxy": selected.processing_energy,
        "quality_exposure_proxy": selected.quality_exposure,
        "safety_exposure_proxy": selected.safety_exposure,
        "fallback_used": decision.fallback_used,
    }
    return [
        {
            **common,
            "ablation": ABLATION_NO_LLM,
            "explanation": "",
            "llm_messages_json": "",
        },
        {
            **common,
            "ablation": ABLATION_TEMPLATE,
            "explanation": generate_template_explanation(payload),
            "llm_messages_json": "",
        },
        {
            **common,
            "ablation": ABLATION_LLM,
            "explanation": "LIVE_CALL_NOT_RUN_IN_WP3",
            "llm_messages_json": json.dumps(build_llm_messages(payload), sort_keys=True),
        },
    ]
