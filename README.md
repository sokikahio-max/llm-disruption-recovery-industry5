# Beyond Rescheduling: Human-Centric LLM Reasoning for Manufacturing Disruption Recovery (Industry 5.0)

This repository contains the companion code (Jupyter/Colab notebook) for the manuscript:

**“Beyond Rescheduling: Human-Centric LLM Reasoning for Manufacturing Disruption Recovery in Industry 5.0.”**

The notebook implements a practical disruption-recovery prototype that combines:
- **State assessment** (delay/urgency, energy proxy, quality risk, safety risk),
- **Multi-objective decision scoring**,
- An **LLM reasoning layer** that produces ranked recovery actions with concise explanations,
- A **governance module** that enforces risk/quality constraints and supports safety-preserving fallback.

The core disruption focus is **equipment failure** (machine breakdown).

---

## Contents

- `Code_LLM_Reschedule.ipynb`  
  Main notebook containing:
  - Monte Carlo evaluation (multiple runs) comparing baseline rescheduling vs. reasoning-driven recovery
  - Visualizations (tardiness, energy proxy, quality risk comparisons)
  - Illustrative reasoning scenarios (including governance-constrained examples)

- `requirements.txt`  
  Minimal Python dependencies.

---

## How to Run

### Option A — Google Colab (recommended)
1. Upload `Code_LLM_Reschedule.ipynb` to Google Colab or open it from GitHub.
2. Install dependencies in the first cell (or run):
   ```bash
   pip install -r requirements.txt