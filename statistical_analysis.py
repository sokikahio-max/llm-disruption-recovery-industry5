"""Paired inferential statistics and robustness analysis.

This module evaluates the paired operational experiment with:

* paired statistical tests, bootstrap confidence intervals, and effect sizes;
* false-discovery-rate correction across policy/metric comparisons;
* interpretable weight-profile and global-weight sensitivity;
* governance-threshold sensitivity;
* action-specific proxy-coefficient perturbation;
* stratification by benchmark and disruption severity.

The operational LLM ablation remains deliberately outside this work package:
operational decisions are deterministic and the LLM is evaluated later only as
an explanation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import json
import math
import platform
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BASE_WEIGHTS: Dict[str, float] = {
    "delay": 0.40,
    "energy": 0.20,
    "quality": 0.20,
    "safety": 0.20,
}
BASE_QUALITY_THRESHOLD = 0.10
BASE_SAFETY_THRESHOLD = 0.08
ACTION_ORDER: Tuple[str, ...] = (
    "bypass",
    "degraded_mode",
    "repair_wait",
    "reschedule_only",
)
GOVERNANCE_POLICY = "governance_aware_no_llm"
COMPARATORS: Tuple[str, ...] = (
    "right_shift",
    "reactive_edd",
    "reactive_spt",
    "reactive_mwkr",
    "reactive_min_slack",
)
CONTINUOUS_METRICS: Tuple[str, ...] = (
    "tardiness",
    "makespan",
    "processing_energy",
    "quality_exposure",
    "safety_exposure",
)

WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "manuscript_base": BASE_WEIGHTS,
    "equal_weights": {"delay": 0.25, "energy": 0.25, "quality": 0.25, "safety": 0.25},
    "delay_dominant": {"delay": 0.70, "energy": 0.10, "quality": 0.10, "safety": 0.10},
    "energy_dominant": {"delay": 0.15, "energy": 0.55, "quality": 0.15, "safety": 0.15},
    "quality_dominant": {"delay": 0.15, "energy": 0.10, "quality": 0.60, "safety": 0.15},
    "safety_dominant": {"delay": 0.15, "energy": 0.10, "quality": 0.15, "safety": 0.60},
    "risk_dominant": {"delay": 0.15, "energy": 0.10, "quality": 0.35, "safety": 0.40},
}

QUALITY_GRID: Tuple[float, ...] = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15)
SAFETY_GRID: Tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
PROXY_FACTORS: Tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


@dataclass(frozen=True)
class ActionCube:
    metadata: pd.DataFrame
    actions: Tuple[str, ...]
    completed: np.ndarray
    tardiness: np.ndarray
    energy: np.ndarray
    quality: np.ndarray
    safety: np.ndarray
    max_quality_risk: np.ndarray
    max_safety_risk: np.ndarray
    original_selected: np.ndarray

    @property
    def n_scenarios(self) -> int:
        return len(self.metadata)

    @property
    def n_actions(self) -> int:
        return len(self.actions)


def _normalize_weights(weights: Mapping[str, float]) -> np.ndarray:
    keys = ("delay", "energy", "quality", "safety")
    values = np.array([float(weights[key]) for key in keys], dtype=float)
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Weights must be non-negative and not all zero.")
    return values / values.sum()


def _minmax_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    span = hi - lo
    out = np.zeros_like(values, dtype=float)
    np.divide(values - lo, span, out=out, where=span > 1e-12)
    return out


def load_action_cube(results_dir: Path = RESULTS_DIR) -> ActionCube:
    frames = [
        pd.read_csv(results_dir / "synthetic_action_evaluations.csv"),
        pd.read_csv(results_dir / "benchmark_action_evaluations.csv"),
    ]
    action_df = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "dataset", "scenario_id", "action", "completed", "tardiness", "energy",
        "quality", "safety", "max_quality_risk", "max_safety_risk", "selected",
    }
    missing = required - set(action_df.columns)
    if missing:
        raise ValueError(f"Action evaluation data are missing columns: {sorted(missing)}")

    action_df["action"] = pd.Categorical(action_df["action"], categories=ACTION_ORDER, ordered=True)
    action_df = action_df.sort_values(["dataset", "scenario_id", "action"]).reset_index(drop=True)
    counts = action_df.groupby(["dataset", "scenario_id"], observed=True)["action"].nunique()
    if not bool((counts == len(ACTION_ORDER)).all()):
        raise ValueError("Every scenario must contain exactly four candidate actions.")

    meta_cols = [
        col for col in (
            "dataset", "scenario_id", "failure_machine", "failure_start", "repair_duration",
            "benchmark", "replication", "disruption_seed", "n_jobs", "n_machines", "operations",
        ) if col in action_df.columns
    ]
    metadata = action_df.groupby(["dataset", "scenario_id"], observed=True, sort=True).first().reset_index()[meta_cols]

    def matrix(column: str, dtype=float) -> np.ndarray:
        pivot = action_df.pivot(index=["dataset", "scenario_id"], columns="action", values=column)
        pivot = pivot.reindex(columns=ACTION_ORDER)
        pivot = pivot.sort_index()
        return pivot.to_numpy(dtype=dtype)

    selected_pivot = action_df[action_df["selected"].astype(bool)].set_index(["dataset", "scenario_id"])["action"]
    selected_pivot = selected_pivot.sort_index()
    if len(selected_pivot) != len(metadata):
        raise ValueError("Each scenario must have exactly one selected action in operational results.")

    return ActionCube(
        metadata=metadata.reset_index(drop=True),
        actions=ACTION_ORDER,
        completed=matrix("completed", bool),
        tardiness=matrix("tardiness"),
        energy=matrix("energy"),
        quality=matrix("quality"),
        safety=matrix("safety"),
        max_quality_risk=matrix("max_quality_risk"),
        max_safety_risk=matrix("max_safety_risk"),
        original_selected=selected_pivot.astype(str).to_numpy(),
    )


def _score_cube(
    tardiness: np.ndarray,
    energy: np.ndarray,
    quality: np.ndarray,
    safety: np.ndarray,
    weights: Mapping[str, float],
) -> np.ndarray:
    w = _normalize_weights(weights)
    normalized = np.stack(
        [
            _minmax_rows(tardiness),
            _minmax_rows(energy),
            _minmax_rows(quality),
            _minmax_rows(safety),
        ],
        axis=2,
    )
    return np.einsum("nak,k->na", normalized, w)


def select_actions(
    cube: ActionCube,
    weights: Mapping[str, float] = BASE_WEIGHTS,
    quality_threshold: float = BASE_QUALITY_THRESHOLD,
    safety_threshold: float = BASE_SAFETY_THRESHOLD,
    energy: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    safety: np.ndarray | None = None,
    max_quality_risk: np.ndarray | None = None,
    max_safety_risk: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected action indices, fallback flags, and action scores."""
    energy = cube.energy if energy is None else np.asarray(energy, dtype=float)
    quality = cube.quality if quality is None else np.asarray(quality, dtype=float)
    safety = cube.safety if safety is None else np.asarray(safety, dtype=float)
    max_quality_risk = cube.max_quality_risk if max_quality_risk is None else np.asarray(max_quality_risk, dtype=float)
    max_safety_risk = cube.max_safety_risk if max_safety_risk is None else np.asarray(max_safety_risk, dtype=float)

    scores = _score_cube(cube.tardiness, energy, quality, safety, weights)
    feasible = (
        cube.completed
        & (max_quality_risk <= float(quality_threshold) + 1e-12)
        & (max_safety_risk <= float(safety_threshold) + 1e-12)
    )
    masked = np.where(feasible, scores, np.inf)
    selected = np.argmin(masked, axis=1)
    fallback = ~feasible.any(axis=1)

    repair_idx = cube.actions.index("repair_wait")
    selected[fallback & cube.completed[:, repair_idx]] = repair_idx
    unresolved = fallback & ~cube.completed[:, repair_idx]
    if np.any(unresolved):
        risk_masked = np.where(cube.completed, max_safety_risk, np.inf)
        selected[unresolved] = np.argmin(risk_masked[unresolved], axis=1)
    return selected, fallback, scores


def _selected_records(
    cube: ActionCube,
    selected: np.ndarray,
    fallback: np.ndarray,
    extra: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    rows = cube.metadata.copy()
    rows["selected_action"] = np.array(cube.actions, dtype=object)[selected]
    rows["fallback_used"] = fallback.astype(bool)
    indices = np.arange(cube.n_scenarios)
    rows["tardiness"] = cube.tardiness[indices, selected]
    rows["energy"] = cube.energy[indices, selected]
    rows["quality"] = cube.quality[indices, selected]
    rows["safety"] = cube.safety[indices, selected]
    rows["max_quality_risk"] = cube.max_quality_risk[indices, selected]
    rows["max_safety_risk"] = cube.max_safety_risk[indices, selected]
    if extra:
        for key, value in extra.items():
            rows[key] = value
    return rows


def _bootstrap_mean_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> Tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0:
        return (math.nan, math.nan)
    # Chunked bootstrap keeps memory use low and is deterministic.
    means: List[np.ndarray] = []
    remaining = n_boot
    while remaining > 0:
        chunk = min(1000, remaining)
        sample_idx = rng.integers(0, n, size=(chunk, n))
        means.append(diff[sample_idx].mean(axis=1))
        remaining -= chunk
    values = np.concatenate(means)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _rank_biserial(diff: np.ndarray) -> float:
    nonzero = np.asarray(diff, dtype=float)
    nonzero = nonzero[np.abs(nonzero) > 1e-12]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    r_plus = float(ranks[nonzero > 0].sum())
    r_minus = float(ranks[nonzero < 0].sum())
    return (r_plus - r_minus) / (r_plus + r_minus)


def paired_statistical_analysis(policy_df: pd.DataFrame, seed: int = 20260723) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for dataset, dataset_df in policy_df.groupby("dataset", sort=True):
        governance = dataset_df[dataset_df["policy"] == GOVERNANCE_POLICY].set_index("scenario_id")
        for comparator in COMPARATORS:
            baseline = dataset_df[dataset_df["policy"] == comparator].set_index("scenario_id")
            common = governance.index.intersection(baseline.index).sort_values()
            if len(common) == 0:
                continue
            for metric in CONTINUOUS_METRICS:
                gov = governance.loc[common, metric].to_numpy(dtype=float)
                comp = baseline.loc[common, metric].to_numpy(dtype=float)
                diff = gov - comp  # Negative means the governance-aware policy is lower.
                n = len(diff)
                mean_diff = float(diff.mean())
                median_diff = float(np.median(diff))
                sd_diff = float(diff.std(ddof=1)) if n > 1 else math.nan
                ci_low, ci_high = _bootstrap_mean_ci(diff, rng)
                t_result = stats.ttest_rel(gov, comp, nan_policy="raise")
                nonzero_count = int(np.sum(np.abs(diff) > 1e-12))
                if nonzero_count == 0:
                    w_stat, w_p = 0.0, 1.0
                else:
                    w_result = stats.wilcoxon(
                        diff,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        method="auto",
                    )
                    w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
                cohen_dz = mean_diff / sd_diff if sd_diff and sd_diff > 1e-12 else 0.0
                comparator_mean = float(comp.mean())
                relative_change = (
                    100.0 * mean_diff / abs(comparator_mean)
                    if abs(comparator_mean) > 1e-12 else math.nan
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "comparator": comparator,
                        "metric": metric,
                        "n_pairs": n,
                        "n_nonzero_pairs": nonzero_count,
                        "governance_mean": float(gov.mean()),
                        "comparator_mean": comparator_mean,
                        "mean_paired_difference": mean_diff,
                        "median_paired_difference": median_diff,
                        "relative_mean_change_percent": relative_change,
                        "bootstrap_95ci_low": ci_low,
                        "bootstrap_95ci_high": ci_high,
                        "paired_t_statistic": float(t_result.statistic),
                        "paired_t_p": float(t_result.pvalue),
                        "wilcoxon_statistic": w_stat,
                        "wilcoxon_p": w_p,
                        "cohen_dz": float(cohen_dz),
                        "rank_biserial": float(_rank_biserial(diff)),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["paired_t_p_fdr_bh"] = multipletests(result["paired_t_p"], method="fdr_bh")[1]
        result["wilcoxon_p_fdr_bh"] = multipletests(result["wilcoxon_p"], method="fdr_bh")[1]
        result["wilcoxon_significant_fdr_0_05"] = result["wilcoxon_p_fdr_bh"] < 0.05
    return result


def benchmark_stratified_analysis(policy_df: pd.DataFrame, seed: int = 20260724) -> pd.DataFrame:
    benchmark_df = policy_df[policy_df["dataset"] == "benchmark_derived"].copy()
    if "benchmark" not in benchmark_df.columns:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for benchmark, group in benchmark_df.groupby("benchmark", sort=True):
        group = group.copy()
        group["dataset"] = f"benchmark_{benchmark}"
        analysis = paired_statistical_analysis(group, seed=seed + len(frames))
        analysis.insert(0, "benchmark", benchmark)
        frames.append(analysis)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def weight_profile_sensitivity(cube: ActionCube) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_base, _, _ = select_actions(cube)
    records: List[pd.DataFrame] = []
    for profile, weights in WEIGHT_PROFILES.items():
        selected, fallback, _ = select_actions(cube, weights=weights)
        frame = _selected_records(
            cube,
            selected,
            fallback,
            extra={
                "profile": profile,
                "w_delay": weights["delay"],
                "w_energy": weights["energy"],
                "w_quality": weights["quality"],
                "w_safety": weights["safety"],
            },
        )
        frame["matches_base_action"] = selected == selected_base
        records.append(frame)
    long_df = pd.concat(records, ignore_index=True)
    frequency = (
        long_df.groupby(["dataset", "profile", "selected_action"], observed=True)
        .size().rename("count").reset_index()
    )
    frequency["proportion"] = frequency.groupby(["dataset", "profile"])["count"].transform(lambda x: x / x.sum())
    summary = (
        long_df.groupby(["dataset", "profile"], observed=True)
        .agg(
            n=("scenario_id", "size"),
            action_stability_vs_base=("matches_base_action", "mean"),
            fallback_rate=("fallback_used", "mean"),
            tardiness_mean=("tardiness", "mean"),
            energy_mean=("energy", "mean"),
            quality_mean=("quality", "mean"),
            safety_mean=("safety", "mean"),
        )
        .reset_index()
    )
    return long_df, frequency, summary


def global_weight_sensitivity(
    cube: ActionCube,
    n_samples: int = 2000,
    seed: int = 20260725,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(4), size=n_samples)
    scores_components = np.stack(
        [
            _minmax_rows(cube.tardiness),
            _minmax_rows(cube.energy),
            _minmax_rows(cube.quality),
            _minmax_rows(cube.safety),
        ],
        axis=2,
    )
    scores = np.einsum("nak,wk->nwa", scores_components, weights)
    feasible = (
        cube.completed
        & (cube.max_quality_risk <= BASE_QUALITY_THRESHOLD + 1e-12)
        & (cube.max_safety_risk <= BASE_SAFETY_THRESHOLD + 1e-12)
    )
    masked = np.where(feasible[:, None, :], scores, np.inf)
    selected = np.argmin(masked, axis=2)  # scenario x sampled weight
    no_feasible = ~feasible.any(axis=1)
    if np.any(no_feasible):
        selected[no_feasible, :] = cube.actions.index("repair_wait")

    weight_df = pd.DataFrame(
        weights,
        columns=["w_delay", "w_energy", "w_quality", "w_safety"],
    )
    weight_df.insert(0, "weight_sample", np.arange(n_samples))

    base_selected, _, _ = select_actions(cube)
    probability_rows: List[Dict[str, object]] = []
    scenario_rows: List[Dict[str, object]] = []
    for scenario_index, meta in cube.metadata.iterrows():
        counts = np.bincount(selected[scenario_index], minlength=cube.n_actions)
        probabilities = counts / n_samples
        for action_index, action in enumerate(cube.actions):
            probability_rows.append(
                {
                    **meta.to_dict(),
                    "action": action,
                    "selection_probability": float(probabilities[action_index]),
                }
            )
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
        scenario_rows.append(
            {
                **meta.to_dict(),
                "base_action": cube.actions[base_selected[scenario_index]],
                "base_action_probability": float(probabilities[base_selected[scenario_index]]),
                "dominant_action": cube.actions[int(np.argmax(probabilities))],
                "dominant_action_probability": float(probabilities.max()),
                "number_of_selected_actions": int(np.sum(probabilities > 0)),
                "selection_entropy_nats": entropy,
            }
        )

    probability_df = pd.DataFrame(probability_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    return weight_df, probability_df, scenario_df


def threshold_sensitivity(cube: ActionCube) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_selected, _, _ = select_actions(cube)
    records: List[pd.DataFrame] = []
    for quality_threshold, safety_threshold in product(QUALITY_GRID, SAFETY_GRID):
        selected, fallback, _ = select_actions(
            cube,
            weights=BASE_WEIGHTS,
            quality_threshold=quality_threshold,
            safety_threshold=safety_threshold,
        )
        frame = _selected_records(
            cube,
            selected,
            fallback,
            extra={
                "quality_threshold": quality_threshold,
                "safety_threshold": safety_threshold,
            },
        )
        frame["matches_base_action"] = selected == base_selected
        records.append(frame)
    long_df = pd.concat(records, ignore_index=True)
    summary = (
        long_df.groupby(["dataset", "quality_threshold", "safety_threshold"], observed=True)
        .agg(
            n=("scenario_id", "size"),
            action_stability_vs_base=("matches_base_action", "mean"),
            fallback_rate=("fallback_used", "mean"),
            bypass_proportion=("selected_action", lambda x: float(np.mean(x == "bypass"))),
            degraded_proportion=("selected_action", lambda x: float(np.mean(x == "degraded_mode"))),
            repair_wait_proportion=("selected_action", lambda x: float(np.mean(x == "repair_wait"))),
            reschedule_proportion=("selected_action", lambda x: float(np.mean(x == "reschedule_only"))),
            tardiness_mean=("tardiness", "mean"),
            energy_mean=("energy", "mean"),
            quality_mean=("quality", "mean"),
            safety_mean=("safety", "mean"),
        )
        .reset_index()
    )
    return long_df, summary


def proxy_coefficient_sensitivity(cube: ActionCube) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perturb action-specific increments around repair-wait by 0.5x--2.0x.

    Uniformly multiplying an entire objective cannot affect min-max-normalized
    ranking. This analysis therefore perturbs the *incremental action-specific*
    energy, quality, and safety burdens relative to repair-wait, including the
    max-risk values used by governance. Reported operational metrics remain the
    original frozen simulation outcomes for the action selected under each
    perturbed coefficient set.
    """
    repair_idx = cube.actions.index("repair_wait")
    base_selected, _, _ = select_actions(cube)
    records: List[pd.DataFrame] = []

    energy_ref = cube.energy[:, [repair_idx]]
    quality_ref = cube.quality[:, [repair_idx]]
    safety_ref = cube.safety[:, [repair_idx]]
    max_quality_ref = cube.max_quality_risk[:, [repair_idx]]
    max_safety_ref = cube.max_safety_risk[:, [repair_idx]]

    for energy_factor, quality_factor, safety_factor in product(PROXY_FACTORS, repeat=3):
        energy_adj = energy_ref + energy_factor * (cube.energy - energy_ref)
        quality_adj = quality_ref + quality_factor * (cube.quality - quality_ref)
        safety_adj = safety_ref + safety_factor * (cube.safety - safety_ref)
        max_quality_adj = max_quality_ref + quality_factor * (cube.max_quality_risk - max_quality_ref)
        max_safety_adj = max_safety_ref + safety_factor * (cube.max_safety_risk - max_safety_ref)
        selected, fallback, _ = select_actions(
            cube,
            weights=BASE_WEIGHTS,
            energy=energy_adj,
            quality=quality_adj,
            safety=safety_adj,
            max_quality_risk=max_quality_adj,
            max_safety_risk=max_safety_adj,
        )
        frame = _selected_records(
            cube,
            selected,
            fallback,
            extra={
                "energy_factor": energy_factor,
                "quality_factor": quality_factor,
                "safety_factor": safety_factor,
            },
        )
        frame["matches_base_action"] = selected == base_selected
        records.append(frame)

    long_df = pd.concat(records, ignore_index=True)
    summary = (
        long_df.groupby(["dataset", "energy_factor", "quality_factor", "safety_factor"], observed=True)
        .agg(
            n=("scenario_id", "size"),
            action_stability_vs_base=("matches_base_action", "mean"),
            fallback_rate=("fallback_used", "mean"),
            bypass_proportion=("selected_action", lambda x: float(np.mean(x == "bypass"))),
            degraded_proportion=("selected_action", lambda x: float(np.mean(x == "degraded_mode"))),
            repair_wait_proportion=("selected_action", lambda x: float(np.mean(x == "repair_wait"))),
            reschedule_proportion=("selected_action", lambda x: float(np.mean(x == "reschedule_only"))),
            tardiness_mean=("tardiness", "mean"),
            energy_mean=("energy", "mean"),
            quality_mean=("quality", "mean"),
            safety_mean=("safety", "mean"),
        )
        .reset_index()
    )
    return long_df, summary


def severity_stratification(policy_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    right_shift = policy_df[policy_df["policy"] == "right_shift"].copy()
    scenario = right_shift[["dataset", "scenario_id", "repair_duration", "tardiness", "failure_machine"]].rename(
        columns={"tardiness": "right_shift_tardiness"}
    )
    scenario["repair_percentile"] = scenario.groupby("dataset")["repair_duration"].rank(pct=True, method="average")
    scenario["tardiness_percentile"] = scenario.groupby("dataset")["right_shift_tardiness"].rank(pct=True, method="average")
    scenario["severity_index"] = 0.5 * scenario["repair_percentile"] + 0.5 * scenario["tardiness_percentile"]
    labels = ["low", "medium", "high"]
    scenario["severity_tercile"] = scenario.groupby("dataset")["severity_index"].transform(
        lambda values: pd.qcut(values.rank(method="first"), q=3, labels=labels)
    )

    governance = policy_df[policy_df["policy"] == GOVERNANCE_POLICY].merge(
        scenario[["dataset", "scenario_id", "severity_index", "severity_tercile"]],
        on=["dataset", "scenario_id"], how="left", validate="one_to_one"
    )
    summary = (
        governance.groupby(["dataset", "severity_tercile", "selected_action"], observed=True)
        .size().rename("count").reset_index()
    )
    summary["proportion_within_severity"] = summary.groupby(["dataset", "severity_tercile"])["count"].transform(lambda x: x / x.sum())
    return scenario, summary


def create_figures(stats_df: pd.DataFrame, weight_frequency: pd.DataFrame, threshold_summary: pd.DataFrame, proxy_summary: pd.DataFrame) -> None:
    # Figure 1: Paired tardiness mean differences and bootstrap confidence intervals.
    plot_df = stats_df[stats_df["metric"] == "tardiness"].copy()
    plot_df["label"] = plot_df["dataset"].str.replace("_", " ") + " | " + plot_df["comparator"].str.replace("_", " ")
    plot_df = plot_df.sort_values(["dataset", "mean_paired_difference"])
    y = np.arange(len(plot_df))
    x = plot_df["mean_paired_difference"].to_numpy()
    xerr = np.vstack([
        x - plot_df["bootstrap_95ci_low"].to_numpy(),
        plot_df["bootstrap_95ci_high"].to_numpy() - x,
    ])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, plot_df["label"])
    ax.set_xlabel("Paired difference in tardiness (governance-aware minus comparator)")
    ax.set_title("Paired tardiness differences with bootstrap 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paired_tardiness_forest.png", dpi=220)
    plt.close(fig)

    # Figure 2: Weight-profile action proportions.
    pivot = weight_frequency.pivot_table(
        index=["dataset", "profile"], columns="selected_action", values="proportion", fill_value=0
    ).reindex(columns=ACTION_ORDER, fill_value=0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6.5))
    ax.set_ylabel("Selected-action proportion")
    ax.set_xlabel("Dataset and weight profile")
    ax.set_title("Recovery action sensitivity to interpretable weight profiles")
    ax.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "weight_profile_action_frequency.png", dpi=220)
    plt.close(fig)

    # Figure 3: Threshold sensitivity (bypass proportion), one file per dataset.
    for dataset, group in threshold_summary.groupby("dataset", sort=True):
        pivot = group.pivot(index="quality_threshold", columns="safety_threshold", values="bypass_proportion")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("Safety threshold")
        ax.set_ylabel("Quality threshold")
        ax.set_title(f"Bypass selection proportion under governance thresholds: {dataset}")
        fig.colorbar(image, ax=ax, label="Bypass proportion")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"threshold_bypass_{dataset}.png", dpi=220)
        plt.close(fig)

    # Figure 4: Proxy perturbation stability averaged over safety factors.
    for dataset, group in proxy_summary.groupby("dataset", sort=True):
        averaged = group.groupby(["energy_factor", "quality_factor"], as_index=False)["action_stability_vs_base"].mean()
        pivot = averaged.pivot(index="quality_factor", columns="energy_factor", values="action_stability_vs_base")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("Incremental energy coefficient factor")
        ax.set_ylabel("Incremental quality coefficient factor")
        ax.set_title(f"Action stability under proxy perturbation: {dataset}")
        fig.colorbar(image, ax=ax, label="Proportion matching base action")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"proxy_stability_{dataset}.png", dpi=220)
        plt.close(fig)



def run() -> Dict[str, pd.DataFrame]:
    policy_df = pd.read_csv(RESULTS_DIR / "combined_policy_results.csv")
    cube = load_action_cube()

    stats_df = paired_statistical_analysis(policy_df)
    stats_df.to_csv(RESULTS_DIR / "paired_statistical_tests.csv", index=False)

    benchmark_stats = benchmark_stratified_analysis(policy_df)
    benchmark_stats.to_csv(RESULTS_DIR / "benchmark_stratified_tests.csv", index=False)

    weight_long, weight_frequency, weight_summary = weight_profile_sensitivity(cube)
    weight_long.to_csv(RESULTS_DIR / "weight_profile_scenarios.csv", index=False)
    weight_frequency.to_csv(RESULTS_DIR / "weight_profile_action_frequency.csv", index=False)
    weight_summary.to_csv(RESULTS_DIR / "weight_profile_summary.csv", index=False)

    weight_samples, global_probability, global_scenario = global_weight_sensitivity(cube)
    weight_samples.to_csv(RESULTS_DIR / "global_weight_samples.csv", index=False)
    global_probability.to_csv(RESULTS_DIR / "global_weight_action_probabilities.csv", index=False)
    global_scenario.to_csv(RESULTS_DIR / "global_weight_scenario_stability.csv", index=False)

    threshold_long, threshold_summary = threshold_sensitivity(cube)
    threshold_long.to_csv(RESULTS_DIR / "threshold_scenarios.csv", index=False)
    threshold_summary.to_csv(RESULTS_DIR / "threshold_summary.csv", index=False)

    proxy_long, proxy_summary = proxy_coefficient_sensitivity(cube)
    proxy_long.to_csv(RESULTS_DIR / "proxy_perturbation_scenarios.csv", index=False)
    proxy_summary.to_csv(RESULTS_DIR / "proxy_perturbation_summary.csv", index=False)

    severity_scenarios, severity_actions = severity_stratification(policy_df)
    severity_scenarios.to_csv(RESULTS_DIR / "severity_scenarios.csv", index=False)
    severity_actions.to_csv(RESULTS_DIR / "severity_action_frequency.csv", index=False)

    selected_base, fallback_base, _ = select_actions(cube)
    base_records = _selected_records(cube, selected_base, fallback_base)
    base_records.to_csv(RESULTS_DIR / "recomputed_base_selection.csv", index=False)

    create_figures(stats_df, weight_frequency, threshold_summary, proxy_summary)

    config = {
        "analysis": "paired statistics and sensitivity analysis",
        "base_weights": BASE_WEIGHTS,
        "base_quality_threshold": BASE_QUALITY_THRESHOLD,
        "base_safety_threshold": BASE_SAFETY_THRESHOLD,
        "bootstrap_resamples": 5000,
        "multiple_testing_correction": "Benjamini-Hochberg FDR across all policy-metric tests",
        "primary_test": "paired Wilcoxon signed-rank",
        "secondary_test": "paired t-test",
        "global_weight_samples": 2000,
        "weight_profiles": WEIGHT_PROFILES,
        "quality_threshold_grid": QUALITY_GRID,
        "safety_threshold_grid": SAFETY_GRID,
        "proxy_increment_factors": PROXY_FACTORS,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (RESULTS_DIR / "statistical_environment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "paired_statistics": stats_df,
        "benchmark_statistics": benchmark_stats,
        "weight_summary": weight_summary,
        "global_weight_stability": global_scenario,
        "threshold_summary": threshold_summary,
        "proxy_summary": proxy_summary,
        "severity_actions": severity_actions,
        "base_records": base_records,
    }


if __name__ == "__main__":
    outputs = run()
    print("Statistical and sensitivity analyses completed successfully.")
    print("\nSelected paired tardiness results:")
    print(
        outputs["paired_statistics"]
        .query("metric == 'tardiness'")
        [[
            "dataset", "comparator", "n_pairs", "mean_paired_difference",
            "bootstrap_95ci_low", "bootstrap_95ci_high", "wilcoxon_p_fdr_bh",
            "cohen_dz", "rank_biserial",
        ]]
        .round(4)
        .to_string(index=False)
    )
    print("\nWeight-profile summary:")
    print(outputs["weight_summary"].round(4).to_string(index=False))
