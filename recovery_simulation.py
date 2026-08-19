"""Disruption-recovery simulation core for the manuscript reproducibility package.

The module intentionally keeps operational decision logic deterministic.
LLM-based explanation/advisory evaluation is added in a later work package.

Key design properties
---------------------
* Paired scenarios: every policy receives exactly the same jobs and failure event.
* Four distinct recovery actions: repair_wait, bypass, degraded_mode,
  and reschedule_only.
* Two explicit comparators: right_shift and reactive_edd.
* Alternative machines are capability-checked.
* Energy/quality/safety impacts are applied once, when an operation starts.
* Governance uses explicit per-operation quality and safety thresholds.
* Every scheduled operation is logged for audit and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import copy
import math

import numpy as np
import pandas as pd


ACTION_REPAIR_WAIT = "repair_wait"
ACTION_BYPASS = "bypass"
ACTION_DEGRADED = "degraded_mode"
ACTION_RESCHEDULE = "reschedule_only"

CANDIDATE_ACTIONS: Tuple[str, ...] = (
    ACTION_REPAIR_WAIT,
    ACTION_BYPASS,
    ACTION_DEGRADED,
    ACTION_RESCHEDULE,
)

POLICY_RIGHT_SHIFT = "right_shift"
POLICY_REACTIVE_EDD = "reactive_edd"
POLICY_REACTIVE_SPT = "reactive_spt"
POLICY_REACTIVE_MWKR = "reactive_mwkr"
POLICY_REACTIVE_MIN_SLACK = "reactive_min_slack"

DISPATCH_EDD = "edd"
DISPATCH_SPT = "spt"
DISPATCH_MWKR = "mwkr"
DISPATCH_MIN_SLACK = "min_slack"
DISPATCH_RULES: Tuple[str, ...] = (
    DISPATCH_EDD,
    DISPATCH_SPT,
    DISPATCH_MWKR,
    DISPATCH_MIN_SLACK,
)


@dataclass(frozen=True)
class OperationSpec:
    job_id: int
    op_index: int
    primary_machine: int
    processing_times: Mapping[int, float]
    alternative_quality_penalty: Mapping[int, float] = field(default_factory=dict)
    alternative_safety_penalty: Mapping[int, float] = field(default_factory=dict)
    base_quality_risk: float = 0.005
    base_safety_risk: float = 0.0

    @property
    def key(self) -> Tuple[int, int]:
        return (self.job_id, self.op_index)

    @property
    def eligible_machines(self) -> Tuple[int, ...]:
        return tuple(sorted(int(m) for m in self.processing_times))

    def processing_time(self, machine: int) -> float:
        if machine not in self.processing_times:
            raise ValueError(f"Machine {machine} is not eligible for operation {self.key}.")
        value = float(self.processing_times[machine])
        if value <= 0:
            raise ValueError(f"Processing time must be positive for operation {self.key}.")
        return value


@dataclass(frozen=True)
class JobSpec:
    job_id: int
    due_date: float
    operations: Tuple[OperationSpec, ...]
    release_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("A job must contain at least one operation.")
        expected = tuple(range(len(self.operations)))
        actual = tuple(op.op_index for op in self.operations)
        if actual != expected:
            raise ValueError(f"Operations for job {self.job_id} must be consecutively indexed.")


@dataclass(frozen=True)
class FailureEvent:
    machine: int
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def active(self, time: float) -> bool:
        return self.start <= time < self.end


@dataclass(frozen=True)
class SimulationConfig:
    machine_power: Tuple[float, ...]
    degraded_time_multiplier: float = 1.60
    degraded_energy_multiplier: float = 1.20
    degraded_quality_risk: float = 0.12
    degraded_safety_risk: float = 0.10
    quality_threshold: float = 0.10
    safety_threshold: float = 0.08
    max_time: float = 100_000.0

    def __post_init__(self) -> None:
        if not self.machine_power or any(p <= 0 for p in self.machine_power):
            raise ValueError("machine_power must contain positive values.")
        if self.degraded_time_multiplier < 1.0:
            raise ValueError("degraded_time_multiplier must be >= 1.")
        if self.degraded_energy_multiplier <= 0:
            raise ValueError("degraded_energy_multiplier must be positive.")
        if self.quality_threshold < 0 or self.safety_threshold < 0:
            raise ValueError("Governance thresholds must be non-negative.")


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    jobs: Tuple[JobSpec, ...]
    failure: FailureEvent
    config: SimulationConfig
    nominal_machine_order: Mapping[int, Tuple[Tuple[int, int], ...]]
    seed: int


@dataclass(frozen=True)
class ScheduledOperation:
    job_id: int
    op_index: int
    primary_machine: int
    executed_machine: int
    start: float
    end: float
    duration: float
    mode: str
    energy: float
    quality_risk: float
    safety_risk: float

    @property
    def key(self) -> Tuple[int, int]:
        return (self.job_id, self.op_index)


@dataclass
class SimulationResult:
    scenario_id: int
    policy: str
    completed: bool
    tardiness: float
    makespan: float
    processing_energy: float
    quality_exposure: float
    safety_exposure: float
    max_quality_risk: float
    max_safety_risk: float
    quality_violations: int
    safety_violations: int
    rerouted_operations: int
    degraded_operations: int
    blocked_operations: int
    schedule: List[ScheduledOperation]
    job_completion: Dict[int, float]
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_record(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "policy": self.policy,
            "completed": self.completed,
            "tardiness": self.tardiness,
            "makespan": self.makespan,
            "processing_energy": self.processing_energy,
            "quality_exposure": self.quality_exposure,
            "safety_exposure": self.safety_exposure,
            "max_quality_risk": self.max_quality_risk,
            "max_safety_risk": self.max_safety_risk,
            "quality_violations": self.quality_violations,
            "safety_violations": self.safety_violations,
            "rerouted_operations": self.rerouted_operations,
            "degraded_operations": self.degraded_operations,
            "blocked_operations": self.blocked_operations,
        }


@dataclass(frozen=True)
class DecisionResult:
    scenario_id: int
    selected_action: str
    fallback_used: bool
    weights: Mapping[str, float]
    quality_threshold: float
    safety_threshold: float
    action_table: pd.DataFrame
    selected_result: SimulationResult


# ---------------------------------------------------------------------------
# Synthetic flexible job-shop generation
# ---------------------------------------------------------------------------

def generate_jobs(
    seed: int,
    n_jobs: int = 20,
    n_machines: int = 5,
    operations_per_job: int = 4,
    alternative_probability: float = 0.60,
) -> Tuple[JobSpec, ...]:
    """Generate a reproducible flexible job-shop instance.

    Each operation has one primary machine and, with a configurable probability,
    one capability-checked alternative machine. Alternative processing times and
    risk increments are operation-specific rather than global action constants.
    """
    if n_jobs <= 0 or n_machines <= 1 or operations_per_job <= 0:
        raise ValueError("n_jobs and operations_per_job must be positive; n_machines must exceed one.")
    if operations_per_job > n_machines:
        raise ValueError("operations_per_job cannot exceed n_machines when routes use permutations.")
    if not 0.0 <= alternative_probability <= 1.0:
        raise ValueError("alternative_probability must lie in [0, 1].")

    rng = np.random.default_rng(seed)
    jobs: List[JobSpec] = []

    for job_id in range(n_jobs):
        route = rng.permutation(n_machines)[:operations_per_job]
        operations: List[OperationSpec] = []
        primary_total = 0.0

        for op_index, primary in enumerate(route):
            primary = int(primary)
            primary_time = float(rng.integers(3, 13))
            primary_total += primary_time
            processing_times: Dict[int, float] = {primary: primary_time}
            q_penalty: Dict[int, float] = {}
            s_penalty: Dict[int, float] = {}

            if rng.random() < alternative_probability:
                alternatives = [m for m in range(n_machines) if m != primary]
                alternative = int(rng.choice(alternatives))
                multiplier = float(rng.uniform(1.05, 1.45))
                processing_times[alternative] = round(primary_time * multiplier, 3)
                q_penalty[alternative] = round(float(rng.uniform(0.015, 0.085)), 4)
                s_penalty[alternative] = round(float(rng.uniform(0.005, 0.035)), 4)

            operations.append(
                OperationSpec(
                    job_id=job_id,
                    op_index=op_index,
                    primary_machine=primary,
                    processing_times=processing_times,
                    alternative_quality_penalty=q_penalty,
                    alternative_safety_penalty=s_penalty,
                    base_quality_risk=0.005,
                    base_safety_risk=0.0,
                )
            )

        # Due dates are intentionally heterogeneous and tied loosely to job work.
        due = float(round(primary_total * rng.uniform(1.8, 3.2) + rng.uniform(10, 55), 3))
        jobs.append(JobSpec(job_id=job_id, due_date=due, operations=tuple(operations)))

    return tuple(jobs)


def default_config(n_machines: int) -> SimulationConfig:
    if n_machines <= 0:
        raise ValueError("n_machines must be positive.")
    # Fixed, transparent machine-specific power rates in relative energy units/time.
    power = tuple(float(x) for x in np.linspace(2.4, 4.4, n_machines))
    return SimulationConfig(machine_power=power)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _operation_lookup(jobs: Sequence[JobSpec]) -> Dict[Tuple[int, int], OperationSpec]:
    return {op.key: op for job in jobs for op in job.operations}


def _job_lookup(jobs: Sequence[JobSpec]) -> Dict[int, JobSpec]:
    return {job.job_id: job for job in jobs}


def _priority_edd(job: JobSpec, op: OperationSpec, now: float, duration: float) -> Tuple[float, float, int, int]:
    slack = job.due_date - (now + duration)
    return (job.due_date, slack, job.job_id, op.op_index)


def _remaining_primary_work(job: JobSpec, op_index: int) -> float:
    return float(
        sum(op.processing_time(op.primary_machine) for op in job.operations[op_index:])
    )


def _dispatch_priority(
    rule: str,
    job: JobSpec,
    op: OperationSpec,
    now: float,
    duration: float,
) -> Tuple[float, ...]:
    """Return a deterministic dispatching priority tuple (smaller is better).

    Rules are classical transparent dispatching heuristics used as stronger
    rescheduling comparators. Ties are resolved by job and operation IDs.
    """
    if rule == DISPATCH_EDD:
        return _priority_edd(job, op, now, duration)
    if rule == DISPATCH_SPT:
        return (duration, job.due_date, job.job_id, op.op_index)
    remaining = _remaining_primary_work(job, op.op_index)
    if rule == DISPATCH_MWKR:
        return (-remaining, job.due_date, job.job_id, op.op_index)
    if rule == DISPATCH_MIN_SLACK:
        slack = job.due_date - now - remaining
        return (slack, job.due_date, job.job_id, op.op_index)
    raise ValueError(f"Unsupported dispatch rule: {rule}")


def _choose_bypass_machine(
    op: OperationSpec,
    now: float,
    machine_ready: Mapping[int, float],
) -> Optional[int]:
    alternatives = [m for m in op.eligible_machines if m != op.primary_machine]
    if not alternatives:
        return None
    return min(
        alternatives,
        key=lambda m: (max(now, machine_ready[m]) + op.processing_time(m), m),
    )


def _resolve_execution(
    op: OperationSpec,
    now: float,
    action: str,
    failure: FailureEvent,
    config: SimulationConfig,
    machine_ready: Mapping[int, float],
) -> Optional[Tuple[int, float, str, float, float, float]]:
    """Return machine, duration, mode, energy multiplier, quality risk, safety risk.

    A None result means the operation must wait until a later event.
    """
    primary = op.primary_machine
    failure_active = primary == failure.machine and failure.active(now)

    if not failure_active:
        duration = op.processing_time(primary)
        return (
            primary,
            duration,
            "normal",
            1.0,
            op.base_quality_risk,
            op.base_safety_risk,
        )

    if action == ACTION_BYPASS:
        alternative = _choose_bypass_machine(op, now, machine_ready)
        if alternative is None:
            return None
        duration = op.processing_time(alternative)
        q = op.base_quality_risk + float(op.alternative_quality_penalty.get(alternative, 0.0))
        s = op.base_safety_risk + float(op.alternative_safety_penalty.get(alternative, 0.0))
        return (alternative, duration, "bypass", 1.0, q, s)

    if action == ACTION_DEGRADED:
        duration = op.processing_time(primary) * config.degraded_time_multiplier
        q = op.base_quality_risk + config.degraded_quality_risk
        s = op.base_safety_risk + config.degraded_safety_risk
        return (primary, duration, "degraded", config.degraded_energy_multiplier, q, s)

    if action in {ACTION_REPAIR_WAIT, ACTION_RESCHEDULE}:
        return None

    raise ValueError(f"Unknown action: {action}")


def _next_event_time(
    now: float,
    jobs: Sequence[JobSpec],
    op_index: Mapping[int, int],
    job_ready: Mapping[int, float],
    machine_ready: Mapping[int, float],
    failure: FailureEvent,
) -> Optional[float]:
    candidates: List[float] = []
    candidates.extend(v for v in machine_ready.values() if v > now + 1e-12)
    candidates.extend(v for v in job_ready.values() if v > now + 1e-12)
    if failure.start > now + 1e-12:
        candidates.append(failure.start)
    if failure.end > now + 1e-12:
        candidates.append(failure.end)
    return min(candidates) if candidates else None


def simulate_action(
    scenario: Scenario,
    action: str,
    dispatch_rule: str = DISPATCH_EDD,
) -> SimulationResult:
    """Simulate one recovery action on one frozen scenario.

    Parameters
    ----------
    dispatch_rule:
        Transparent dispatching rule used when the action permits reactive
        resequencing. ``repair_wait`` always preserves the nominal machine
        sequence and therefore ignores this argument.
    """
    if action not in CANDIDATE_ACTIONS:
        raise ValueError(f"Unsupported action: {action}")
    if dispatch_rule not in DISPATCH_RULES:
        raise ValueError(f"Unsupported dispatch rule: {dispatch_rule}")

    jobs = scenario.jobs
    failure = scenario.failure
    config = scenario.config
    n_machines = len(config.machine_power)
    job_by_id = _job_lookup(jobs)
    op_lookup = _operation_lookup(jobs)

    op_index: Dict[int, int] = {job.job_id: 0 for job in jobs}
    job_ready: Dict[int, float] = {job.job_id: job.release_time for job in jobs}
    machine_ready: Dict[int, float] = {m: 0.0 for m in range(n_machines)}
    scheduled_keys: set[Tuple[int, int]] = set()
    blocked_keys: set[Tuple[int, int]] = set()
    records: List[ScheduledOperation] = []

    # Right-shift uses the nominal failure-free machine sequence. Other actions
    # use reactive EDD dispatch after the disruption.
    fixed_sequence = action == ACTION_REPAIR_WAIT

    now = 0.0
    total_operations = sum(len(job.operations) for job in jobs)
    iterations = 0

    while len(scheduled_keys) < total_operations and now <= config.max_time:
        iterations += 1
        if iterations > 2_000_000:
            break

        enabled: Dict[Tuple[int, int], OperationSpec] = {}
        execution: Dict[Tuple[int, int], Tuple[int, float, str, float, float, float]] = {}

        for job in jobs:
            idx = op_index[job.job_id]
            if idx >= len(job.operations) or job_ready[job.job_id] > now + 1e-12:
                continue
            op = job.operations[idx]
            resolved = _resolve_execution(op, now, action, failure, config, machine_ready)
            if resolved is None:
                if op.primary_machine == failure.machine and failure.active(now):
                    blocked_keys.add(op.key)
                continue
            enabled[op.key] = op
            execution[op.key] = resolved

        selected: Dict[int, Tuple[OperationSpec, Tuple[int, float, str, float, float, float]]] = {}

        if fixed_sequence:
            # Preserve the exact failure-free operation sequence on each machine.
            for machine in range(n_machines):
                if machine_ready[machine] > now + 1e-12:
                    continue
                order = scenario.nominal_machine_order.get(machine, tuple())
                first_remaining: Optional[Tuple[int, int]] = next(
                    (key for key in order if key not in scheduled_keys),
                    None,
                )
                if first_remaining is None or first_remaining not in enabled:
                    continue
                resolved = execution[first_remaining]
                # Repair-wait never reroutes; this check protects invariants.
                if resolved[0] != machine:
                    continue
                selected[machine] = (enabled[first_remaining], resolved)
        else:
            by_machine: Dict[int, List[Tuple[OperationSpec, Tuple[int, float, str, float, float, float]]]] = {}
            for key, op in enabled.items():
                resolved = execution[key]
                machine = resolved[0]
                if machine_ready[machine] <= now + 1e-12:
                    by_machine.setdefault(machine, []).append((op, resolved))

            for machine, candidates in by_machine.items():
                op, resolved = min(
                    candidates,
                    key=lambda item: _dispatch_priority(
                        dispatch_rule, job_by_id[item[0].job_id], item[0], now, item[1][1]
                    ),
                )
                selected[machine] = (op, resolved)

        if selected:
            for machine in sorted(selected):
                op, resolved = selected[machine]
                executed_machine, duration, mode, energy_multiplier, q_risk, s_risk = resolved
                if op.key in scheduled_keys:
                    raise RuntimeError(f"Operation {op.key} scheduled more than once.")
                if machine_ready[executed_machine] > now + 1e-12:
                    raise RuntimeError("Attempted to schedule on a busy machine.")

                start = now
                end = start + duration
                energy = duration * config.machine_power[executed_machine] * energy_multiplier

                records.append(
                    ScheduledOperation(
                        job_id=op.job_id,
                        op_index=op.op_index,
                        primary_machine=op.primary_machine,
                        executed_machine=executed_machine,
                        start=start,
                        end=end,
                        duration=duration,
                        mode=mode,
                        energy=energy,
                        quality_risk=q_risk,
                        safety_risk=s_risk,
                    )
                )
                scheduled_keys.add(op.key)
                machine_ready[executed_machine] = end
                job_ready[op.job_id] = end
                op_index[op.job_id] += 1

            next_time = _next_event_time(now, jobs, op_index, job_ready, machine_ready, failure)
            if next_time is None:
                break
            now = next_time
            continue

        next_time = _next_event_time(now, jobs, op_index, job_ready, machine_ready, failure)
        if next_time is None or next_time <= now + 1e-12:
            break
        now = next_time

    completed = len(scheduled_keys) == total_operations
    completion: Dict[int, float] = {}
    for job in jobs:
        completion[job.job_id] = float(job_ready[job.job_id]) if op_index[job.job_id] == len(job.operations) else math.inf

    finite_completion = [v for v in completion.values() if math.isfinite(v)]
    makespan = max(finite_completion) if finite_completion else math.inf
    tardiness = (
        sum(max(0.0, completion[job.job_id] - job.due_date) for job in jobs)
        if completed
        else math.inf
    )

    q_values = [r.quality_risk for r in records]
    s_values = [r.safety_risk for r in records]
    quality_violations = sum(q > config.quality_threshold + 1e-12 for q in q_values)
    safety_violations = sum(s > config.safety_threshold + 1e-12 for s in s_values)

    result = SimulationResult(
        scenario_id=scenario.scenario_id,
        policy=action,
        completed=completed,
        tardiness=float(tardiness),
        makespan=float(makespan),
        processing_energy=float(sum(r.energy for r in records)),
        quality_exposure=float(sum(q_values)),
        safety_exposure=float(sum(s_values)),
        max_quality_risk=float(max(q_values, default=0.0)),
        max_safety_risk=float(max(s_values, default=0.0)),
        quality_violations=int(quality_violations),
        safety_violations=int(safety_violations),
        rerouted_operations=sum(r.mode == "bypass" for r in records),
        degraded_operations=sum(r.mode == "degraded" for r in records),
        blocked_operations=len(blocked_keys),
        schedule=records,
        job_completion=completion,
        diagnostics={"iterations": iterations, "scheduled_operations": len(records), "dispatch_rule": dispatch_rule},
    )
    return result


def build_nominal_machine_order(
    jobs: Tuple[JobSpec, ...],
    config: SimulationConfig,
) -> Mapping[int, Tuple[Tuple[int, int], ...]]:
    """Create a feasible failure-free EDD schedule and return machine sequences."""
    far_failure = FailureEvent(machine=0, start=config.max_time * 2.0, duration=1.0)
    provisional = Scenario(
        scenario_id=-1,
        jobs=jobs,
        failure=far_failure,
        config=config,
        nominal_machine_order={m: tuple() for m in range(len(config.machine_power))},
        seed=-1,
    )
    result = simulate_action(provisional, ACTION_RESCHEDULE)
    if not result.completed:
        raise RuntimeError("Failed to construct nominal schedule.")
    order: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(len(config.machine_power))}
    for record in sorted(result.schedule, key=lambda r: (r.start, r.end, r.job_id, r.op_index)):
        order[record.executed_machine].append(record.key)
    return {m: tuple(keys) for m, keys in order.items()}


def create_scenario(
    seed: int,
    scenario_id: Optional[int] = None,
    n_jobs: int = 20,
    n_machines: int = 5,
    operations_per_job: int = 4,
    alternative_probability: float = 0.60,
    repair_duration_bounds: Tuple[int, int] = (8, 24),
) -> Scenario:
    """Create one paired scenario with a randomized, impactful failure event."""
    if repair_duration_bounds[0] <= 0 or repair_duration_bounds[1] < repair_duration_bounds[0]:
        raise ValueError("Invalid repair_duration_bounds.")

    jobs = generate_jobs(
        seed=seed,
        n_jobs=n_jobs,
        n_machines=n_machines,
        operations_per_job=operations_per_job,
        alternative_probability=alternative_probability,
    )
    config = default_config(n_machines)
    nominal_order = build_nominal_machine_order(jobs, config)

    # Obtain nominal start times to place the failure where it affects future work.
    far_failure = FailureEvent(machine=0, start=config.max_time * 2.0, duration=1.0)
    provisional = Scenario(
        scenario_id=-1,
        jobs=jobs,
        failure=far_failure,
        config=config,
        nominal_machine_order=nominal_order,
        seed=seed,
    )
    nominal_result = simulate_action(provisional, ACTION_RESCHEDULE)
    rng = np.random.default_rng(seed + 90_001)

    machine_candidates = [
        m for m in range(n_machines)
        if sum(r.executed_machine == m for r in nominal_result.schedule) >= 2
    ]
    failed_machine = int(rng.choice(machine_candidates))
    starts = sorted(r.start for r in nominal_result.schedule if r.executed_machine == failed_machine)
    quantile = float(rng.uniform(0.25, 0.60))
    failure_start = float(np.quantile(starts, quantile))
    repair_duration = float(rng.integers(repair_duration_bounds[0], repair_duration_bounds[1] + 1))

    return Scenario(
        scenario_id=seed if scenario_id is None else int(scenario_id),
        jobs=jobs,
        failure=FailureEvent(machine=failed_machine, start=failure_start, duration=repair_duration),
        config=config,
        nominal_machine_order=nominal_order,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Governance-aware deterministic selection
# ---------------------------------------------------------------------------

def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    required = ("delay", "energy", "quality", "safety")
    missing = [k for k in required if k not in weights]
    if missing:
        raise ValueError(f"Missing weights: {missing}")
    clean = {k: float(weights[k]) for k in required}
    if any(v < 0 for v in clean.values()):
        raise ValueError("Weights must be non-negative.")
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    return {k: v / total for k, v in clean.items()}


def _minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series.astype(float) - lo) / (hi - lo)


def evaluate_actions(
    scenario: Scenario,
    weights: Mapping[str, float],
    quality_threshold: Optional[float] = None,
    safety_threshold: Optional[float] = None,
    dispatch_rule: str = DISPATCH_EDD,
) -> DecisionResult:
    """Evaluate all four actions and select the minimum-score feasible action."""
    normalized_weights = _normalize_weights(weights)
    q_threshold = scenario.config.quality_threshold if quality_threshold is None else float(quality_threshold)
    s_threshold = scenario.config.safety_threshold if safety_threshold is None else float(safety_threshold)

    if dispatch_rule not in DISPATCH_RULES:
        raise ValueError(f"Unsupported dispatch rule: {dispatch_rule}")
    results = {action: simulate_action(scenario, action, dispatch_rule) for action in CANDIDATE_ACTIONS}
    rows: List[Dict[str, object]] = []
    for action, result in results.items():
        rows.append(
            {
                "action": action,
                "completed": result.completed,
                "tardiness": result.tardiness,
                "energy": result.processing_energy,
                "quality": result.quality_exposure,
                "safety": result.safety_exposure,
                "max_quality_risk": result.max_quality_risk,
                "max_safety_risk": result.max_safety_risk,
                "quality_violations": result.quality_violations,
                "safety_violations": result.safety_violations,
                "rerouted_operations": result.rerouted_operations,
                "degraded_operations": result.degraded_operations,
            }
        )
    table = pd.DataFrame(rows).set_index("action")

    # Governance is based on configured per-operation limits, not on the
    # experimental result's own threshold-derived count, so sensitivity can
    # vary thresholds without rerunning the simulation.
    table["feasible"] = (
        table["completed"]
        & (table["max_quality_risk"] <= q_threshold + 1e-12)
        & (table["max_safety_risk"] <= s_threshold + 1e-12)
    )

    for source, target in (
        ("tardiness", "delay_norm"),
        ("energy", "energy_norm"),
        ("quality", "quality_norm"),
        ("safety", "safety_norm"),
    ):
        table[target] = _minmax(table[source])

    table["score"] = (
        normalized_weights["delay"] * table["delay_norm"]
        + normalized_weights["energy"] * table["energy_norm"]
        + normalized_weights["quality"] * table["quality_norm"]
        + normalized_weights["safety"] * table["safety_norm"]
    )

    feasible = table[table["feasible"]]
    fallback_used = feasible.empty
    if not feasible.empty:
        selected = min(feasible.index, key=lambda action: (table.loc[action, "score"], action))
    else:
        # Conservative deterministic fallback. Repair-wait is always attempted
        # first; otherwise choose the action with the lowest max safety risk.
        if results[ACTION_REPAIR_WAIT].completed:
            selected = ACTION_REPAIR_WAIT
        else:
            selected = min(table.index, key=lambda action: (table.loc[action, "max_safety_risk"], action))

    table["selected"] = table.index == selected
    table = table.sort_values(["feasible", "score"], ascending=[False, True])

    return DecisionResult(
        scenario_id=scenario.scenario_id,
        selected_action=selected,
        fallback_used=fallback_used,
        weights=normalized_weights,
        quality_threshold=q_threshold,
        safety_threshold=s_threshold,
        action_table=table,
        selected_result=results[selected],
    )


def run_paired_experiment(
    n_runs: int = 100,
    seed0: int = 42,
    weights: Optional[Mapping[str, float]] = None,
    **scenario_kwargs: object,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired baselines and governance-aware selection on frozen scenarios.

    Returns
    -------
    policy_results:
        One row per scenario-policy pair for right_shift, reactive_edd, and
        governance_aware.
    action_evaluations:
        One row per scenario-candidate action, including feasibility and score.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")
    weights = weights or {"delay": 0.40, "energy": 0.20, "quality": 0.20, "safety": 0.20}

    policy_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []

    for run in range(n_runs):
        scenario = create_scenario(seed=seed0 + run, scenario_id=run, **scenario_kwargs)
        decision = evaluate_actions(scenario, weights=weights)

        baselines = {
            POLICY_RIGHT_SHIFT: simulate_action(scenario, ACTION_REPAIR_WAIT),
            POLICY_REACTIVE_EDD: simulate_action(scenario, ACTION_RESCHEDULE),
            "governance_aware": decision.selected_result,
        }
        for policy, result in baselines.items():
            row = result.to_record()
            row.update(
                {
                    "policy": policy,
                    "selected_action": decision.selected_action if policy == "governance_aware" else result.policy,
                    "failure_machine": scenario.failure.machine,
                    "failure_start": scenario.failure.start,
                    "repair_duration": scenario.failure.duration,
                    "seed": scenario.seed,
                    "fallback_used": decision.fallback_used if policy == "governance_aware" else False,
                }
            )
            policy_rows.append(row)

        action_table = decision.action_table.reset_index()
        action_table.insert(0, "scenario_id", scenario.scenario_id)
        action_table["failure_machine"] = scenario.failure.machine
        action_table["failure_start"] = scenario.failure.start
        action_table["repair_duration"] = scenario.failure.duration
        action_rows.extend(action_table.to_dict(orient="records"))

    return pd.DataFrame(policy_rows), pd.DataFrame(action_rows)


def run_stronger_comparator_experiment(
    scenarios: Sequence[Scenario],
    weights: Optional[Mapping[str, float]] = None,
    action_dispatch_rule: str = DISPATCH_EDD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate four transparent schedule-repair comparators and the framework.

    All policies are paired on exactly the same frozen disruption scenario.
    The governance-aware framework evaluates its four recovery actions using
    ``action_dispatch_rule``; the LLM is not involved in operational selection.
    """
    if action_dispatch_rule not in DISPATCH_RULES:
        raise ValueError(f"Unsupported action dispatch rule: {action_dispatch_rule}")
    weights = weights or {"delay": 0.40, "energy": 0.20, "quality": 0.20, "safety": 0.20}
    policy_rows: List[Dict[str, object]] = []
    action_rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        decision = evaluate_actions(
            scenario,
            weights=weights,
            dispatch_rule=action_dispatch_rule,
        )
        comparators = {
            POLICY_RIGHT_SHIFT: simulate_action(scenario, ACTION_REPAIR_WAIT),
            POLICY_REACTIVE_EDD: simulate_action(scenario, ACTION_RESCHEDULE, DISPATCH_EDD),
            POLICY_REACTIVE_SPT: simulate_action(scenario, ACTION_RESCHEDULE, DISPATCH_SPT),
            POLICY_REACTIVE_MWKR: simulate_action(scenario, ACTION_RESCHEDULE, DISPATCH_MWKR),
            POLICY_REACTIVE_MIN_SLACK: simulate_action(scenario, ACTION_RESCHEDULE, DISPATCH_MIN_SLACK),
            "governance_aware_no_llm": decision.selected_result,
        }
        for policy, result in comparators.items():
            row = result.to_record()
            row.update(
                {
                    "policy": policy,
                    "selected_action": decision.selected_action if policy == "governance_aware_no_llm" else result.policy,
                    "failure_machine": scenario.failure.machine,
                    "failure_start": scenario.failure.start,
                    "repair_duration": scenario.failure.duration,
                    "seed": scenario.seed,
                    "fallback_used": decision.fallback_used if policy == "governance_aware_no_llm" else False,
                    "dispatch_rule": result.diagnostics.get("dispatch_rule", "fixed_sequence"),
                }
            )
            policy_rows.append(row)

        action_table = decision.action_table.reset_index()
        action_table.insert(0, "scenario_id", scenario.scenario_id)
        action_table["failure_machine"] = scenario.failure.machine
        action_table["failure_start"] = scenario.failure.start
        action_table["repair_duration"] = scenario.failure.duration
        action_rows.extend(action_table.to_dict(orient="records"))

    return pd.DataFrame(policy_rows), pd.DataFrame(action_rows)


# ---------------------------------------------------------------------------
# Validation helpers and unit tests
# ---------------------------------------------------------------------------

def validate_schedule(scenario: Scenario, result: SimulationResult) -> List[str]:
    """Return a list of schedule invariant violations; empty means valid."""
    errors: List[str] = []
    expected_keys = {op.key for job in scenario.jobs for op in job.operations}
    actual_keys = [record.key for record in result.schedule]

    if len(actual_keys) != len(set(actual_keys)):
        errors.append("An operation was scheduled more than once.")
    if result.completed and set(actual_keys) != expected_keys:
        errors.append("Completed schedule does not contain exactly all operations.")

    # Job precedence.
    by_job: Dict[int, List[ScheduledOperation]] = {}
    for record in result.schedule:
        by_job.setdefault(record.job_id, []).append(record)
    for job_id, records in by_job.items():
        records = sorted(records, key=lambda r: r.op_index)
        for previous, current in zip(records, records[1:]):
            if current.start + 1e-12 < previous.end:
                errors.append(f"Job precedence violated for job {job_id}.")

    # Machine capacity.
    by_machine: Dict[int, List[ScheduledOperation]] = {}
    for record in result.schedule:
        by_machine.setdefault(record.executed_machine, []).append(record)
    for machine, records in by_machine.items():
        records = sorted(records, key=lambda r: (r.start, r.end))
        for previous, current in zip(records, records[1:]):
            if current.start + 1e-12 < previous.end:
                errors.append(f"Machine overlap on machine {machine}.")

    # Capability and failure-window rules.
    op_lookup = _operation_lookup(scenario.jobs)
    for record in result.schedule:
        op = op_lookup[record.key]
        if record.executed_machine not in op.eligible_machines:
            errors.append(f"Ineligible machine used for operation {record.key}.")
        if (
            result.policy in {ACTION_REPAIR_WAIT, ACTION_RESCHEDULE}
            and record.primary_machine == scenario.failure.machine
            and scenario.failure.active(record.start)
        ):
            errors.append(f"Wait/reschedule action started failed-machine operation during downtime: {record.key}.")
        if record.mode == "bypass" and record.executed_machine == record.primary_machine:
            errors.append(f"Bypass did not change machine for operation {record.key}.")

    return errors


def run_unit_tests() -> pd.DataFrame:
    """Execute deterministic self-tests and return a compact report."""
    tests: List[Tuple[str, bool, str]] = []

    def record(name: str, fn) -> None:
        try:
            fn()
            tests.append((name, True, ""))
        except Exception as exc:  # pragma: no cover - diagnostic path
            tests.append((name, False, f"{type(exc).__name__}: {exc}"))

    def test_generation_reproducible() -> None:
        a = generate_jobs(7)
        b = generate_jobs(7)
        assert a == b

    def test_all_actions_complete_and_valid() -> None:
        scenario = create_scenario(11, scenario_id=11, n_jobs=10)
        for action in CANDIDATE_ACTIONS:
            result = simulate_action(scenario, action)
            assert result.completed, action
            assert not validate_schedule(scenario, result), (action, validate_schedule(scenario, result))

    def test_impacts_applied_once() -> None:
        scenario = create_scenario(13, scenario_id=13, n_jobs=10)
        result = simulate_action(scenario, ACTION_BYPASS)
        assert len(result.schedule) == len({r.key for r in result.schedule})
        expected_energy = sum(r.energy for r in result.schedule)
        expected_quality = sum(r.quality_risk for r in result.schedule)
        assert math.isclose(result.processing_energy, expected_energy, abs_tol=1e-9)
        assert math.isclose(result.quality_exposure, expected_quality, abs_tol=1e-9)

    def test_bypass_capability() -> None:
        scenario = create_scenario(17, scenario_id=17, n_jobs=10)
        result = simulate_action(scenario, ACTION_BYPASS)
        lookup = _operation_lookup(scenario.jobs)
        for record_ in result.schedule:
            assert record_.executed_machine in lookup[record_.key].eligible_machines

    def test_governance_boundary() -> None:
        scenario = create_scenario(19, scenario_id=19, n_jobs=10)
        decision = evaluate_actions(
            scenario,
            weights={"delay": 1, "energy": 1, "quality": 1, "safety": 1},
            quality_threshold=scenario.config.degraded_quality_risk + 0.005,
            safety_threshold=scenario.config.degraded_safety_risk,
        )
        # Equality at threshold is feasible; values above are not.
        row = decision.action_table.loc[ACTION_DEGRADED]
        assert bool(row["feasible"]) == (
            row["max_quality_risk"] <= decision.quality_threshold + 1e-12
            and row["max_safety_risk"] <= decision.safety_threshold + 1e-12
        )

    def test_deterministic_selection() -> None:
        scenario = create_scenario(23, scenario_id=23, n_jobs=10)
        weights = {"delay": 0.4, "energy": 0.2, "quality": 0.2, "safety": 0.2}
        a = evaluate_actions(scenario, weights)
        b = evaluate_actions(scenario, weights)
        assert a.selected_action == b.selected_action
        pd.testing.assert_frame_equal(a.action_table, b.action_table)

    def test_paired_scenarios() -> None:
        policies, _ = run_paired_experiment(n_runs=3, seed0=31, n_jobs=8)
        counts = policies.groupby("scenario_id")["seed"].nunique()
        assert (counts == 1).all()
        assert (policies.groupby("scenario_id")["repair_duration"].nunique() == 1).all()

    for name, fn in (
        ("generator reproducibility", test_generation_reproducible),
        ("all actions complete with valid schedules", test_all_actions_complete_and_valid),
        ("one-time impact accounting", test_impacts_applied_once),
        ("bypass capability enforcement", test_bypass_capability),
        ("governance threshold boundary", test_governance_boundary),
        ("deterministic action selection", test_deterministic_selection),
        ("paired scenario identity", test_paired_scenarios),
    ):
        record(name, fn)

    return pd.DataFrame(tests, columns=["test", "passed", "details"])
