"""OR-Library benchmark-derived disruption scenarios used in the study.

The original routes and processing times are read from the public OR-Library
``jobshop1`` collection. Because the manuscript studies disruption recovery
rather than static makespan minimization, each benchmark is transparently
extended with due dates, one optional capability-checked alternative machine
per operation, risk increments, machine-power coefficients, and an injected
machine failure. The resulting scenarios are therefore described as
*benchmark-derived flexible job-shop disruption instances*, not as unchanged
classical JSSP benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np

from recovery_simulation import (
    ACTION_RESCHEDULE,
    FailureEvent,
    JobSpec,
    OperationSpec,
    Scenario,
    SimulationConfig,
    build_nominal_machine_order,
    simulate_action,
)


@dataclass(frozen=True)
class ORLibraryInstance:
    name: str
    description: str
    n_jobs: int
    n_machines: int
    jobs: Tuple[Tuple[Tuple[int, float], ...], ...]

    @property
    def operation_count(self) -> int:
        return sum(len(job) for job in self.jobs)


def load_orlibrary_instance(path: str | Path, name: str) -> ORLibraryInstance:
    """Load one named instance from OR-Library's ``jobshop1`` text file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    target = f"instance {name}".strip().lower()
    start: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == target:
            start = idx
            break
    if start is None:
        raise KeyError(f"Instance {name!r} was not found in {path}.")

    dimension_idx: Optional[int] = None
    dimension_pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
    for idx in range(start + 1, min(len(lines), start + 25)):
        match = dimension_pattern.match(lines[idx])
        if match:
            dimension_idx = idx
            break
    if dimension_idx is None:
        raise ValueError(f"Could not locate dimensions for instance {name}.")

    n_jobs, n_machines = map(int, lines[dimension_idx].split())
    description_candidates = [
        line.strip()
        for line in lines[start + 1 : dimension_idx]
        if line.strip() and set(line.strip()) != {"+"}
    ]
    description = description_candidates[-1] if description_candidates else name

    parsed_jobs: List[Tuple[Tuple[int, float], ...]] = []
    cursor = dimension_idx + 1
    while len(parsed_jobs) < n_jobs and cursor < len(lines):
        stripped = lines[cursor].strip()
        cursor += 1
        if not stripped or stripped.startswith("+"):
            continue
        values = [int(token) for token in stripped.split()]
        if len(values) != 2 * n_machines:
            raise ValueError(
                f"Instance {name}, job {len(parsed_jobs)} has {len(values)} values; "
                f"expected {2 * n_machines}."
            )
        operations = tuple(
            (int(values[pos]), float(values[pos + 1]))
            for pos in range(0, len(values), 2)
        )
        parsed_jobs.append(operations)

    if len(parsed_jobs) != n_jobs:
        raise ValueError(f"Instance {name} ended before all jobs were read.")
    if any(machine < 0 or machine >= n_machines for job in parsed_jobs for machine, _ in job):
        raise ValueError(f"Instance {name} contains an out-of-range machine index.")

    return ORLibraryInstance(
        name=name,
        description=description,
        n_jobs=n_jobs,
        n_machines=n_machines,
        jobs=tuple(parsed_jobs),
    )


def _benchmark_config(n_machines: int) -> SimulationConfig:
    power = tuple(float(x) for x in np.linspace(2.4, 4.4, n_machines))
    return SimulationConfig(
        machine_power=power,
        degraded_time_multiplier=1.60,
        degraded_energy_multiplier=1.20,
        degraded_quality_risk=0.12,
        degraded_safety_risk=0.10,
        quality_threshold=0.10,
        safety_threshold=0.08,
    )


def derive_flexible_jobs(
    instance: ORLibraryInstance,
    extension_seed: int = 2026,
    alternative_probability: float = 0.50,
    due_date_tightness: float = 1.00,
) -> Tuple[JobSpec, ...]:
    """Extend a classical JSSP instance for transparent recovery experiments.

    The benchmark's primary machine routes and processing times are preserved.
    Added alternatives and risks are seeded and operation-specific. Due dates
    are derived from a workload lower bound, making them comparable across
    benchmark sizes while retaining heterogeneous job urgency.
    """
    if not 0.0 <= alternative_probability <= 1.0:
        raise ValueError("alternative_probability must lie in [0, 1].")
    if due_date_tightness <= 0:
        raise ValueError("due_date_tightness must be positive.")

    rng = np.random.default_rng(extension_seed)
    job_work = np.array(
        [sum(duration for _, duration in job) for job in instance.jobs], dtype=float
    )
    machine_load = np.zeros(instance.n_machines, dtype=float)
    for job in instance.jobs:
        for machine, duration in job:
            machine_load[machine] += duration
    workload_lb = float(max(job_work.max(), machine_load.max()))

    jobs: List[JobSpec] = []
    for job_id, benchmark_job in enumerate(instance.jobs):
        operations: List[OperationSpec] = []
        for op_index, (primary, primary_time) in enumerate(benchmark_job):
            processing_times: Dict[int, float] = {int(primary): float(primary_time)}
            q_penalty: Dict[int, float] = {}
            s_penalty: Dict[int, float] = {}

            if rng.random() < alternative_probability:
                candidates = [m for m in range(instance.n_machines) if m != primary]
                alternative = int(rng.choice(candidates))
                multiplier = float(rng.uniform(1.08, 1.35))
                processing_times[alternative] = round(float(primary_time) * multiplier, 3)
                q_penalty[alternative] = round(float(rng.uniform(0.015, 0.075)), 4)
                s_penalty[alternative] = round(float(rng.uniform(0.005, 0.030)), 4)

            operations.append(
                OperationSpec(
                    job_id=job_id,
                    op_index=op_index,
                    primary_machine=int(primary),
                    processing_times=processing_times,
                    alternative_quality_penalty=q_penalty,
                    alternative_safety_penalty=s_penalty,
                    base_quality_risk=0.005,
                    base_safety_risk=0.0,
                )
            )

        # Workload-based due-date assignment. The random component is fixed by
        # extension_seed and bounded to ±8% of the workload lower bound.
        due = due_date_tightness * (
            0.68 * workload_lb
            + 0.55 * job_work[job_id]
            + float(rng.uniform(-0.08, 0.08)) * workload_lb
        )
        jobs.append(
            JobSpec(
                job_id=job_id,
                due_date=max(float(job_work[job_id]), round(float(due), 3)),
                operations=tuple(operations),
            )
        )

    return tuple(jobs)


def create_benchmark_scenario(
    instance: ORLibraryInstance,
    disruption_seed: int,
    scenario_id: int,
    extension_seed: int = 2026,
    alternative_probability: float = 0.50,
    due_date_tightness: float = 1.00,
    failure_quantile_bounds: Tuple[float, float] = (0.25, 0.60),
    repair_scale_bounds: Tuple[float, float] = (0.8, 2.0),
) -> Scenario:
    """Create a paired disruption scenario from one benchmark instance."""
    jobs = derive_flexible_jobs(
        instance,
        extension_seed=extension_seed,
        alternative_probability=alternative_probability,
        due_date_tightness=due_date_tightness,
    )
    config = _benchmark_config(instance.n_machines)
    nominal_order = build_nominal_machine_order(jobs, config)

    far_failure = FailureEvent(machine=0, start=config.max_time * 2.0, duration=1.0)
    provisional = Scenario(
        scenario_id=-1,
        jobs=jobs,
        failure=far_failure,
        config=config,
        nominal_machine_order=nominal_order,
        seed=disruption_seed,
    )
    nominal_result = simulate_action(provisional, ACTION_RESCHEDULE)
    if not nominal_result.completed:
        raise RuntimeError(f"Nominal benchmark schedule failed for {instance.name}.")

    rng = np.random.default_rng(disruption_seed)
    machine_candidates = [
        m
        for m in range(instance.n_machines)
        if sum(r.executed_machine == m for r in nominal_result.schedule) >= 2
    ]
    failed_machine = int(rng.choice(machine_candidates))
    starts = sorted(
        r.start for r in nominal_result.schedule if r.executed_machine == failed_machine
    )
    q_lo, q_hi = failure_quantile_bounds
    if not 0 <= q_lo <= q_hi <= 1:
        raise ValueError("Invalid failure_quantile_bounds.")
    failure_start = float(np.quantile(starts, float(rng.uniform(q_lo, q_hi))))

    durations = np.array([r.duration for r in nominal_result.schedule], dtype=float)
    repair_duration = float(
        round(np.median(durations) * float(rng.uniform(*repair_scale_bounds)), 3)
    )
    repair_duration = max(1.0, repair_duration)

    return Scenario(
        scenario_id=int(scenario_id),
        jobs=jobs,
        failure=FailureEvent(
            machine=failed_machine,
            start=failure_start,
            duration=repair_duration,
        ),
        config=config,
        nominal_machine_order=nominal_order,
        seed=disruption_seed,
    )


def create_benchmark_scenarios(
    path: str | Path,
    names: Sequence[str],
    replications_per_instance: int = 20,
    seed0: int = 60_000,
    **kwargs: object,
) -> Tuple[Tuple[Scenario, ...], Tuple[Dict[str, object], ...]]:
    """Build multiple failure replications and return scenario metadata."""
    if replications_per_instance <= 0:
        raise ValueError("replications_per_instance must be positive.")

    scenarios: List[Scenario] = []
    metadata: List[Dict[str, object]] = []
    scenario_id = 0
    for instance_offset, name in enumerate(names):
        instance = load_orlibrary_instance(path, name)
        for replication in range(replications_per_instance):
            disruption_seed = seed0 + instance_offset * 10_000 + replication
            scenario = create_benchmark_scenario(
                instance,
                disruption_seed=disruption_seed,
                scenario_id=scenario_id,
                extension_seed=2026 + instance_offset,
                **kwargs,
            )
            scenarios.append(scenario)
            metadata.append(
                {
                    "scenario_id": scenario_id,
                    "benchmark": name,
                    "benchmark_description": instance.description,
                    "n_jobs": instance.n_jobs,
                    "n_machines": instance.n_machines,
                    "operations": instance.operation_count,
                    "replication": replication,
                    "disruption_seed": disruption_seed,
                }
            )
            scenario_id += 1
    return tuple(scenarios), tuple(metadata)
