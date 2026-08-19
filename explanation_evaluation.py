"""Controlled deterministic-template versus LLM explanation evaluation.

The operational action is selected upstream by the deterministic governance-aware
recovery model. This module evaluates explanation quality only. It never permits
an LLM response to change the selected action or bypass governance constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import re
import statistics

ACTIONS = ("repair_wait", "bypass", "degraded_mode", "reschedule_only")
CRITERIA = ("delay", "energy", "quality", "safety")
REJECTION_REASONS = (
    "incomplete_schedule",
    "quality_threshold_exceeded",
    "safety_threshold_exceeded",
)

EXPLANATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "selected_action",
        "summary",
        "tradeoffs",
        "closest_feasible_alternative",
        "governance_rejections",
        "operator_attention",
        "uncertainty_and_limits",
    ],
    "properties": {
        "selected_action": {"type": "string", "enum": list(ACTIONS)},
        "summary": {"type": "string"},
        "tradeoffs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criterion", "statement"],
                "properties": {
                    "criterion": {"type": "string", "enum": list(CRITERIA)},
                    "statement": {"type": "string"},
                },
            },
        },
        "closest_feasible_alternative": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action", "comparison"],
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "comparison": {"type": "string"},
                    },
                },
            ]
        },
        "governance_rejections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "reasons"],
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(REJECTION_REASONS)},
                    },
                },
            },
        },
        "operator_attention": {"type": "string"},
        "uncertainty_and_limits": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You are an industrial decision-support explanation layer.
The deterministic governance-aware module has already selected the recovery action.
Explain that decision; do not replace it, re-rank it, or recommend another action.
Use only the supplied structured evidence. Do not invent measurements, causes,
constraints, probabilities, physical units, or shop-floor facts. Treat energy,
quality, and safety values as normalized experimental proxies. Report rejected
actions only with the exact supplied rejection reasons. Keep the explanation concise
and suitable for a human operator. Return strict JSON matching the supplied schema."""


@dataclass(frozen=True)
class AutomatedScore:
    scenario_key: str
    method: str
    repeat: int
    schema_valid: bool
    selected_action_correct: bool
    runner_up_correct: bool
    governance_precision: float
    governance_recall: float
    governance_f1: float
    criterion_coverage: float
    numeric_grounding_ratio: float
    unsupported_numeric_count: int
    uncertainty_present: bool
    operator_attention_present: bool
    word_count: int
    flesch_reading_ease: float
    factual_reliability_score: float
    validation_errors: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def action_map(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row["action"]): row for row in payload["candidate_actions"]}


def selected_action(payload: Mapping[str, Any]) -> str:
    return str(payload["decision_configuration"]["deterministic_selected_action"])


def feasible_runner_up(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    selected = selected_action(payload)
    candidates = [
        row for row in payload["candidate_actions"]
        if bool(row["feasible"]) and str(row["action"]) != selected
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (float(row["score"]), str(row["action"])))


def expected_rejections(payload: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    result: Dict[str, Tuple[str, ...]] = {}
    for row in payload["candidate_actions"]:
        reasons = tuple(str(x) for x in row.get("rejected_reasons", []))
        if reasons:
            result[str(row["action"])] = reasons
    return result


def build_messages(payload: Mapping[str, Any]) -> List[Dict[str, str]]:
    task = {
        "task": "Explain the deterministic recovery recommendation for a human operator.",
        "constraints": [
            "selected_action must equal deterministic_selected_action",
            "use only supplied evidence",
            "do not describe proxies as physical measurements or probabilities",
            "include exact governance rejection reasons when present",
            "state simulation and proxy limitations",
        ],
        "output_schema": EXPLANATION_JSON_SCHEMA,
        "evidence": payload,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(task, indent=2, sort_keys=True)},
    ]


def generate_template_structured(payload: Mapping[str, Any]) -> Dict[str, Any]:
    amap = action_map(payload)
    selected_name = selected_action(payload)
    selected = amap[selected_name]
    runner = feasible_runner_up(payload)

    tradeoffs = []
    criterion_fields = {
        "delay": "tardiness",
        "energy": "energy_proxy",
        "quality": "quality_exposure_proxy",
        "safety": "safety_exposure_proxy",
    }
    for criterion, field in criterion_fields.items():
        tradeoffs.append({
            "criterion": criterion,
            "statement": (
                f"The selected action has {field.replace('_', ' ')} "
                f"{float(selected[field]):.4f} under the configured simulation."
            ),
        })

    rejections = [
        {"action": action, "reasons": list(reasons)}
        for action, reasons in expected_rejections(payload).items()
    ]
    closest = None
    if runner is not None:
        closest = {
            "action": str(runner["action"]),
            "comparison": (
                f"Its weighted score is {float(runner['score']):.4f}, compared with "
                f"{float(selected['score']):.4f} for the selected action."
            ),
        }

    return {
        "selected_action": selected_name,
        "summary": (
            f"Recommend {selected_name} because it is governance-feasible and has the "
            f"lowest weighted score ({float(selected['score']):.4f}) among feasible actions. "
            "The recommendation reflects the configured trade-off and is not a claim of "
            "universal superiority."
        ),
        "tradeoffs": tradeoffs,
        "closest_feasible_alternative": closest,
        "governance_rejections": rejections,
        "operator_attention": (
            "Before execution, verify that the monitored state and configured governance "
            "thresholds still represent current operating conditions."
        ),
        "uncertainty_and_limits": (
            "The evidence comes from a simulation, and energy, quality, and safety are "
            "normalized proxies rather than physical measurements or calibrated probabilities; these limits constrain generalization."
        ),
    }


def render_structured_explanation(output: Mapping[str, Any]) -> str:
    parts = [str(output.get("summary", "")).strip()]
    tradeoffs = output.get("tradeoffs", [])
    if isinstance(tradeoffs, list):
        for item in tradeoffs:
            if isinstance(item, Mapping):
                parts.append(f"{str(item.get('criterion', '')).title()}: {str(item.get('statement', '')).strip()}")
    closest = output.get("closest_feasible_alternative")
    if isinstance(closest, Mapping):
        parts.append(
            f"Closest feasible alternative: {closest.get('action')}. "
            f"{str(closest.get('comparison', '')).strip()}"
        )
    rejects = output.get("governance_rejections", [])
    if isinstance(rejects, list) and rejects:
        text = "; ".join(
            f"{item.get('action')} ({', '.join(item.get('reasons', []))})"
            for item in rejects if isinstance(item, Mapping)
        )
        parts.append(f"Governance rejections: {text}.")
    parts.append(f"Operator attention: {str(output.get('operator_attention', '')).strip()}")
    parts.append(str(output.get("uncertainty_and_limits", "")).strip())
    return " ".join(part for part in parts if part)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def validate_structure(output: Mapping[str, Any], payload: Mapping[str, Any]) -> Tuple[bool, Tuple[str, ...]]:
    errors: List[str] = []
    required = set(EXPLANATION_JSON_SCHEMA["required"])
    missing = sorted(required.difference(output.keys()))
    extra = sorted(set(output.keys()).difference(required))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    if extra:
        errors.append("extra_fields:" + ",".join(extra))

    if output.get("selected_action") not in ACTIONS:
        errors.append("invalid_selected_action")
    if output.get("selected_action") != selected_action(payload):
        errors.append("selected_action_mismatch")
    for field in ("summary", "operator_attention", "uncertainty_and_limits"):
        if not isinstance(output.get(field), str) or not str(output.get(field, "")).strip():
            errors.append(f"invalid_{field}")

    tradeoffs = output.get("tradeoffs")
    if not isinstance(tradeoffs, list) or not (1 <= len(tradeoffs) <= 4):
        errors.append("invalid_tradeoffs")
    else:
        seen = set()
        for item in tradeoffs:
            if not _is_mapping(item):
                errors.append("malformed_tradeoff")
                continue
            criterion = item.get("criterion")
            if criterion not in CRITERIA:
                errors.append(f"invalid_criterion:{criterion}")
            if criterion in seen:
                errors.append(f"duplicate_criterion:{criterion}")
            seen.add(criterion)
            if not isinstance(item.get("statement"), str) or not item.get("statement", "").strip():
                errors.append(f"invalid_tradeoff_statement:{criterion}")

    closest = output.get("closest_feasible_alternative")
    if closest is not None:
        if not _is_mapping(closest):
            errors.append("invalid_closest_alternative")
        else:
            if closest.get("action") not in ACTIONS:
                errors.append("invalid_closest_action")
            if not isinstance(closest.get("comparison"), str) or not closest.get("comparison", "").strip():
                errors.append("invalid_closest_comparison")

    rejects = output.get("governance_rejections")
    if not isinstance(rejects, list):
        errors.append("invalid_governance_rejections")
    else:
        for item in rejects:
            if not _is_mapping(item):
                errors.append("malformed_rejection")
                continue
            if item.get("action") not in ACTIONS:
                errors.append(f"invalid_rejected_action:{item.get('action')}")
            reasons = item.get("reasons")
            if not isinstance(reasons, list) or any(reason not in REJECTION_REASONS for reason in reasons):
                errors.append(f"invalid_rejection_reasons:{item.get('action')}")

    return (not errors, tuple(errors))


def _rejection_pairs(output: Mapping[str, Any]) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    rejects = output.get("governance_rejections", [])
    if not isinstance(rejects, list):
        return pairs
    for item in rejects:
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action", ""))
        reasons = item.get("reasons", [])
        if isinstance(reasons, list):
            for reason in reasons:
                pairs.add((action, str(reason)))
    return pairs


def governance_metrics(output: Mapping[str, Any], payload: Mapping[str, Any]) -> Tuple[float, float, float]:
    predicted = _rejection_pairs(output)
    expected = {
        (action, reason)
        for action, reasons in expected_rejections(payload).items()
        for reason in reasons
    }
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted.intersection(expected))
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def allowed_numeric_values(payload: Mapping[str, Any]) -> List[float]:
    values: List[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
        elif isinstance(value, Mapping):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(payload)
    return values


_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numeric_grounding(output: Mapping[str, Any], payload: Mapping[str, Any]) -> Tuple[float, int]:
    text = render_structured_explanation(output)
    candidates = [float(match.group(0)) for match in _NUMBER_PATTERN.finditer(text)]
    # Small integers frequently occur as grammatical counts and are not treated as factual measurements.
    factual = [value for value in candidates if abs(value) >= 5 or not float(value).is_integer()]
    if not factual:
        return 1.0, 0
    allowed = allowed_numeric_values(payload)

    def grounded(value: float) -> bool:
        return any(abs(value - permitted) <= max(1e-3, 1e-4 * max(1.0, abs(permitted))) for permitted in allowed)

    supported = sum(grounded(value) for value in factual)
    unsupported = len(factual) - supported
    return supported / len(factual), unsupported


def runner_up_is_correct(output: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    expected = feasible_runner_up(payload)
    supplied = output.get("closest_feasible_alternative")
    if expected is None:
        return supplied is None
    return isinstance(supplied, Mapping) and supplied.get("action") == expected["action"]


def criterion_coverage(output: Mapping[str, Any]) -> float:
    tradeoffs = output.get("tradeoffs", [])
    if not isinstance(tradeoffs, list):
        return 0.0
    criteria = {
        str(item.get("criterion"))
        for item in tradeoffs if isinstance(item, Mapping) and item.get("criterion") in CRITERIA
    }
    return len(criteria) / len(CRITERIA)


def uncertainty_present(output: Mapping[str, Any]) -> bool:
    text = str(output.get("uncertainty_and_limits", "")).lower()
    return ("proxy" in text or "proxies" in text) and "simulat" in text and ("limit" in text or "general" in text)


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = max(1, len(groups))
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ye")):
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"\b[A-Za-z]+\b", text)
    word_count = max(1, len(words))
    syllables = sum(_syllables(word) for word in words)
    return 206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllables / word_count)


def score_output(
    output: Mapping[str, Any],
    payload: Mapping[str, Any],
    scenario_key: str,
    method: str,
    repeat: int,
) -> AutomatedScore:
    schema_valid, errors = validate_structure(output, payload)
    precision, recall, f1 = governance_metrics(output, payload)
    grounding, unsupported_count = numeric_grounding(output, payload)
    action_correct = output.get("selected_action") == selected_action(payload)
    runner_correct = runner_up_is_correct(output, payload)
    coverage = criterion_coverage(output)
    uncertainty = uncertainty_present(output)
    operator_attention = isinstance(output.get("operator_attention"), str) and bool(output.get("operator_attention", "").strip())
    text = render_structured_explanation(output)
    words = re.findall(r"\b\w+\b", text)

    components = [
        float(schema_valid),
        float(action_correct),
        float(runner_correct),
        f1,
        coverage,
        grounding,
        float(uncertainty),
        float(operator_attention),
    ]
    reliability = 100.0 * statistics.mean(components)
    return AutomatedScore(
        scenario_key=scenario_key,
        method=method,
        repeat=int(repeat),
        schema_valid=schema_valid,
        selected_action_correct=action_correct,
        runner_up_correct=runner_correct,
        governance_precision=precision,
        governance_recall=recall,
        governance_f1=f1,
        criterion_coverage=coverage,
        numeric_grounding_ratio=grounding,
        unsupported_numeric_count=unsupported_count,
        uncertainty_present=uncertainty,
        operator_attention_present=operator_attention,
        word_count=len(words),
        flesch_reading_ease=flesch_reading_ease(text),
        factual_reliability_score=reliability,
        validation_errors="|".join(errors),
    )


def estimate_tokens(messages: Sequence[Mapping[str, str]], output_tokens_assumption: int = 260) -> Tuple[int, int]:
    """Estimate tokens without requiring tiktoken; use tiktoken when installed."""
    joined = "\n".join(str(message.get("content", "")) for message in messages)
    try:
        import tiktoken  # type: ignore
        encoding = tiktoken.get_encoding("o200k_base")
        input_tokens = len(encoding.encode(joined))
    except Exception:
        input_tokens = max(1, math.ceil(len(joined) / 4.0))
    return input_tokens, int(output_tokens_assumption)


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    return (
        input_tokens * input_usd_per_million / 1_000_000.0
        + output_tokens * output_usd_per_million / 1_000_000.0
    )


def normalized_word_set(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "is", "are", "in", "for",
        "with", "that", "this", "it", "as", "be", "by", "from", "under", "has",
    }
    return {
        token for token in re.findall(r"[a-z]+", text.lower())
        if len(token) > 2 and token not in stop
    }


def pairwise_jaccard(texts: Sequence[str]) -> float:
    sets = [normalized_word_set(text) for text in texts]
    if len(sets) < 2:
        return 1.0
    scores: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i].union(sets[j])
            scores.append(len(sets[i].intersection(sets[j])) / len(union) if union else 1.0)
    return statistics.mean(scores) if scores else 1.0


def score_as_dict(score: AutomatedScore) -> Dict[str, Any]:
    return asdict(score)
