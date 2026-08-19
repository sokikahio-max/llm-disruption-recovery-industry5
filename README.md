# Beyond Rescheduling: Governance-Aware Manufacturing Disruption Recovery

Companion reproducibility repository for the manuscript:

**Beyond Rescheduling: Human-Centric Governance and LLM Explanation Support for Manufacturing Disruption Recovery in Industry 5.0**

## What this repository reproduces

The repository implements the final study design in which operational recovery selection is **deterministic and governance-aware**, while the large language model (LLM) is used only as an optional **explanation layer after the operational action has been frozen**.

The code covers:

- four recovery actions: repair-wait, capability-checked bypass, degraded mode, and reschedule-only;
- 100 paired synthetic disruption scenarios;
- 60 failure-augmented scenarios derived from FT06, LA01, and LA06;
- five transparent comparators: right shift, EDD, SPT, MWKR, and minimum slack;
- paired statistical tests, bootstrap confidence intervals, effect sizes, and FDR correction;
- weight, governance-threshold, and proxy-coefficient sensitivity analyses;
- deterministic-template versus constrained-LLM explanation evaluation;
- the completed 60-call LLM evaluation, including validity, grounding, latency, token-use, cost, and consistency metrics.

## Repository structure

```text
.
├── recovery_simulation.py
├── benchmark_instances.py
├── run_operational_experiments.py
├── statistical_analysis.py
├── explanation_evaluation.py
├── prepare_explanation_study.py
├── run_llm_explanation_study.py
├── run_all.py
├── config/
│   ├── experiment_config.json
│   ├── llm_config.json
│   ├── LLM_SYSTEM_PROMPT.txt
│   └── LLM_OUTPUT_SCHEMA.json
├── data/
│   └── jobshop1.txt
├── results/
│   ├── combined_policy_results.csv
│   ├── paired_statistical_tests.csv
│   ├── weight_profile_summary.csv
│   ├── threshold_summary.csv
│   ├── proxy_perturbation_summary.csv
│   ├── template_automated_scores.csv
│   ├── llm_automated_scores.csv
│   ├── llm_run_status.json
│   └── manuscript_numerical_verification.csv
└── tests/
```

## Environment

The manuscript reports Python 3.13.5, NumPy 2.3.5, and pandas 2.2.3 for the principal experiments. Install the repository dependencies with:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the deterministic experiments

```bash
python run_operational_experiments.py
python statistical_analysis.py
python prepare_explanation_study.py
```

or run the three deterministic stages together:

```bash
python run_all.py
```

No API key is required for these stages.

## Optional: rerun the 60-call LLM explanation study

The archived `results/` directory already contains the outputs used in the manuscript. To make new API calls, set your OpenAI API key locally and run:

```bash
# macOS/Linux
export OPENAI_API_KEY="your-key"

# Windows PowerShell
$env:OPENAI_API_KEY="your-key"

python run_llm_explanation_study.py
```

or:

```bash
python run_all.py --with-llm
```

The key is read only from the environment and is never written to the repository. Because model/service behavior and pricing can change, a fresh run may not reproduce latency, token counts, cost, or wording exactly. The manuscript results correspond to the archived outputs in `results/` and the configuration in `config/llm_config.json`.

## Validation

Run:

```bash
python tests/test_operational.py
python tests/test_statistics.py
python tests/test_explanations.py
```

The file `results/manuscript_numerical_verification.csv` contains the final numerical consistency checks used to verify the reported manuscript values.

## Interpretation boundaries

Energy, quality, and safety values are simulation proxies. They are not physical kWh measurements, calibrated defect probabilities, incident probabilities, or regulatory safety limits. Governance compliance denotes satisfaction of the configured experimental thresholds only.

No human-participant evaluation is included in this repository. The LLM does not select, override, or execute the operational recovery action.

## API credentials and privacy

No API credentials are included. The shared LLM call records omit service-specific response identifiers and system fingerprints. The repository contains no human-participant data.
