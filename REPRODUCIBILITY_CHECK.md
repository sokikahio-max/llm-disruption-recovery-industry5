# Reproducibility checks

The packaged code was validated after repository cleanup.

- Simulator/comparator/integration tests: **13/13 passed**.
- Statistical/sensitivity tests: **7/7 passed**.
- Explanation-layer tests: **10/10 passed**.
- Manuscript numerical consistency checks: **51/51 matched**.

Archived LLM evaluation summary:

- Calls: **60**
- JSON/schema validity: **100.0%**
- Selected-action agreement: **100.0%**
- Mean factual-reliability score: **89.87**
- Mean latency: **4.31 s**
- Median latency: **4.26 s**
- 95th-percentile latency: **5.91 s**
- Input/output tokens: **103,970 / 19,126**
- Estimated API cost: **USD 0.07219**

The deterministic rerun reproduces the archived operational and statistical CSVs to numerical tolerance.
