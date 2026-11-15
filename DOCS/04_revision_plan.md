# Revision plan: fixing the pre-freeze review findings

Status: v0.2, 13 Nov 2026. Steps 1 and 2 done (step 2: skeleton, tests passing, task B splits reproducible); next: step 3. Turns the review of `01`-`03` into concrete changes. Item numbers (1.1, 2.3, ...) refer to that review.

## Decisions taken (13 Nov 2026)

| Topic | Decision |
|---|---|
| Baseline | One LLM baseline that scores options via logprobs, reached through an **OpenAI-compatible endpoint** (placeholder config; the user specifies the endpoint). Descriptive comparison only. Tasks C/D move behind it in priority. |
| Other decision models | vLLM-SR d3 and AWS Strands Decider 2B are **core but descriptive** (E8): full grid on A and B and all H1-H6 statistics for each model that passes the Phase 0 gate; no tests, outside the Holm family. In the cut line they come right after d1-2B (dropped last). Decided 13 Nov 2026. |
| Primary metric | **Brier score and log loss are primary**; ECE is descriptive only (reliability diagrams, debiased ECE in a table). |
| Freeze evidence | **OSF preregistration** of the frozen hypotheses and analysis plan, plus a git tag. The user uploads; content is reviewed before upload. |
| Multiplicity | **One primary test per hypothesis (H1-H6), Holm-corrected across the six.** Everything else is secondary or exploratory. |

## Overview and order

```
Step 1  Revise docs, structural pass   (02, 03, 01; numbers left as TBD)    <- no data or GPU needed
Step 2  Reproducible repo skeleton      (env lock, configs, fetch + checksums, Makefile)
Step 3  Metric code + synthetic tests   (known-calibration data)
Step 4  Power simulation                -> fills SESOIs / margins (TBD in step 1)
Step 5  Phase 0 on dev data             (all decision models + baseline endpoint; user runs on GPU)
Step 6  Revise docs, numeric pass       (Phase 0 decisions, SESOIs, seeds, option sets)
Step 7  Freeze: git tag + OSF upload    (user)
```

Steps 1-4 can be done in the repository now. Step 5 needs the workstation and the model card. Step 6 depends on steps 4 and 5.

---

## Step 1: Document revisions, structural pass

### 1a. `02_hypotheses.md` (major rewrite)

General changes:
- Add a **primary test table** at the top: one row per hypothesis giving the task, method, statistic, SESOI or margin, and decision rule. Apply Holm across the 6 rows (family-wise α = 0.05; report adjusted p-values from the bootstrap, or the equivalent Holm-adjusted interval levels). (2.2)
- Every hypothesis gets three explicit outcomes, **supported / refuted / inconclusive**, so that no result is left undefined. (1.1)
- Metric: Brier score and log loss replace ECE in all confirmatory statements. ECE moves to "descriptive". (1.2)
- All condition comparisons use **paired** cluster bootstrap over items (same items, all conditions resampled together).
- Define the **flip-counting rule** in the Terms section: a flip counts only if the top option differs and the probability margin in both conditions exceeds the Phase 0 noise floor ε. No subtraction of rates. (1.7)
- Define **task A polarity** in Terms: positive = hallucinated; option text and the mapping from the model's option to P(positive) are fixed in `prompts/`. (3)
- d1-2B is the confirmatory model. The other decision models (E8) and the baseline (E4) get all statistics descriptively, with no hypothesis tests.

Per hypothesis. The primary choices below are proposals; confirm them with co-authors.

| Hyp. | Current problem | Revised primary test (proposal) | Demoted to secondary / exploratory |
|---|---|---|---|
| H1 | Compares two different ECEs; gap in the verdict rules (1.1) | Task A, canonical condition: **log-loss gain** from Platt scaling fitted on the calibration pool and evaluated on test. Gain ≥ SESOI ⇒ miscalibrated. No binning, so no ECE bias. | Same on task B (temperature scaling); A-vs-B difference descriptive only |
| H2 | Several sub-tests, "A or B" (2.2) | Task B: pairwise disagreement across paraphrases at O0 (H2a), counted with the noise-floor rule. | H2a on A; H2b (order); H2c (confident flips) |
| H3 | Canonical condition is special (1.3); pooled calibrator in-distribution (1.4) | Task B, temperature scaling: **transfer matrix** over paraphrases (fit on each P_i, evaluate on every P_j). Statistic = mean off-diagonal minus mean diagonal **log loss**, paired bootstrap. | Order factor; Platt; task A. H3c becomes **leave-one-condition-out** pooled calibrator, decision rule "closes ≥ 50% of the gap, interval excludes 0 in the closing direction" |
| H4 | "Temperature or Platt" counts as a pass; interval rule missing; n = pool size (1.6, 2.2) | Task A, Platt, n = 200: relative log-loss reduction vs raw ≥ SESOI; interval combines calibration-subset and test-bootstrap variance (nested). | H4b (isotonic harm at n = 50) relabelled as **replication** of a known result (2.1) |
| H5 | H5a guaranteed by construction; H5b has no margin (1.5, 2.1) | Task B: **non-inferiority**: Brier(ensemble, calibrated) − Brier(single, calibrated) has upper bound < δ. | H5a replaced by a descriptive **disagreement-versus-K curve** (K = 1, 2, 3, 6, all) |
| H6 | H6a guaranteed by decision theory; "> half of cells" ambiguous; cells correlated (2.1) | Task A: cost-derived thresholds on **calibrated vs raw** probabilities (old H6b). Statistic = **mean expected-cost difference averaged over the 12 grid cells**, a single number with a cluster bootstrap. | Calibrated formula vs empirical cost search (rule 5) as **key secondary**; vs default rules descriptive |

Exploratory section:
- E2: rename the factors to paraphrase, order, and "interaction + residual (not separable: one observation per cell)". Define the statistic as the log-odds of the canonical top option. (2.3)
- E4: becomes "**LLM logprob baseline**" (always run); vLLM-SR d3 and Strands become E8 (core, descriptive; full grid if they pass the gate).
- Keep the debiased ECE (equal-mass and sweep variants) and reliability diagrams as descriptive.

### 1b. `03_experiment_protocol.md`

| Section | Change | Item |
|---|---|---|
| Status line | Version scheme aligned with `02`; the freeze happens **after** Phase 0 and the power simulation | 2.8, minor |
| 2 Models | Add the baseline row: "LLM logprob baseline via OpenAI-compatible endpoint, `configs/endpoints.yaml`". Add a subsection "Baseline scoring" (see below) | 2.5 |
| 3 Phase 0 | Add checks for the baseline: logprobs available, `top_logprobs` ≥ number of options, missing probability mass per item, repeatability. Phase 0 explicitly comes **before** the hypotheses are finalised (H2b depends on it) | minor, 2.5 |
| 4.1 RAGTruth | State whether to use the official split, and why if not; record which response model each item comes from; record how prevalence changes after the token filter; fix the option text and polarity | 2.4, 3 |
| 4.2 MASSIVE | **Draw both the calibration pool and the test set from the official test split** (grouped random split), so both share the same contamination status. Define the O0 option order (e.g. alphabetical) | 2.4, 3 |
| 4.4 Seeds | Replace "e.g." with one fixed master seed and a derivation rule (`seed = hash(master, split, purpose)`) | 3 |
| 6 Metrics | Brier and log loss marked primary; ECE descriptive with a debiased estimator; flip rule with ε; the "subtract the noise floor" wording removed | 1.2, 1.7 |
| 7 Analyses | Rewrite rows to match the revised `02`: transfer matrix, leave-one-condition-out, K-curve, non-inferiority, cost averaged over cells, Holm procedure | 1.3-1.5, 2.1, 2.2 |
| 7 H4 | Label budgets n ∈ {25, 50, 100, 200, 400}, capped at ≤ 50% of the pool; nested bootstrap | 1.6 |
| 9 Thresholds | Note that a review band exists in every cell of this grid, so the single-cut-off fallback applies only outside the grid | minor |
| 10 Freeze | Freeze = git tag + OSF upload of `02`, `03` and the analysis code hash; the OSF link is recorded here | 2.8 |
| 11 Schedule | Freeze moved to after Phase 0 and the power simulation; explicit buffer; cut line drops C/D first, then the baseline, then the other decision models | minor |
| 12 Threats | Add: baseline endpoint nondeterminism and model drift; contamination that differs between splits | 2.4, 2.5 |
| 13 Layout | `docs/` → `DOCS/`; add `configs/endpoints.yaml`, `Makefile`, lockfile, `data/checksums.sha256` | 3, minor |
| 14 Open items | Update the list | - |
| 15 Deferred | Recommend early DPO/Legal contact before downloading RAGTruth/ToxicChat (instead of only before publication) | 5 |

**Baseline scoring subsection (new, in 03 section 2):**
- Options are presented with single-token labels (A, B, C, ...). The probability for each option comes from the `top_logprobs` of the first answer token, renormalised over the option labels. The missing probability mass is logged per item.
- With the order factor, labels stay attached to positions and options move between them; the analysis maps back to the original option identity (same rule as for d1).
- Deterministic settings: `temperature = 0`, `max_tokens = 1`, `logprobs = true`, `top_logprobs = min(20, server max)`. Task B (18 options) needs ≥ 18; check this in Phase 0.
- Recommended: a self-hosted, OpenAI-compatible server (e.g. vLLM) with a pinned model revision. Hosted APIs cannot be pinned and their logprobs may vary between calls; if a hosted API is used, record this as a limitation.
- Endpoint config: `base_url`, `model`, `api_key_env` (the **name** of an environment variable, never the key itself), `top_logprobs`, `timeout`. The real file is git-ignored; `endpoints.example.yaml` is committed.

### 1c. `01_article_outline.md`

| Change | Item |
|---|---|
| Abstract takeaway, Discussion first bullet, recipe "100 to 200 labels": rewrite as conditional placeholders (`[if H3 supported: ...]`) | 2.7 |
| Contributions: reframe 1 around **decision models with typed probabilities** and the transfer/label-budget question, not "a new protocol"; add the baseline comparison as context | 2.6, 2.5 |
| Background: grow related work to about 0.4 pages: prompt sensitivity (Sclar et al. 2023), position bias (Zheng et al. 2023), contextual calibration (Zhao et al. 2021), calibration methods (Guo et al. 2017; Niculescu-Mizil & Caruana 2005). Verify each reference before citing | 2.6 |
| Method/Setup: Brier/log loss primary; transfer matrix; leave-one-condition-out; baseline | 1.2-1.4 |
| Figures: Fig 2 shows raw diagrams only (recalibrated ones go to 5.4); Figs 4 and 7 move to the appendix; Fig 7 becomes the K-curve; add a baseline column to Table 2 | minor, 2.1 |
| Define "cascade" or drop the word | minor |
| Limitations: single confirmatory model; baseline is descriptive; endpoint pinning | 2.5 |

---

## Step 2: Reproducible repository skeleton

| Deliverable | Verify |
|---|---|
| `pyproject.toml` + `uv.lock` (Python version pinned) | `uv sync --frozen` on a clean clone succeeds |
| `configs/` (run, analysis, `endpoints.example.yaml`); `.gitignore` covers `configs/endpoints.yaml` | `git check-ignore configs/endpoints.yaml` |
| `src/prep_data.py`: downloads pinned dataset revisions, builds splits from the master seed, writes `data/checksums.sha256` | Two runs give identical checksums |
| `src/run_grid.py` with one backend per model interface: `d1` (local HF), `openai_compat` (endpoint config); backends for vLLM-SR d3 and Strands added once Phase 0 clarifies their interfaces | Dry run on 5 dev items per backend |
| Determinism flags (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) and run metadata (commit, library versions, GPU, driver) stored with every predictions file | Metadata present in the output |
| `Makefile`: `data`, `pilot`, `grid`, `analyze`, `figures`. `analyze` and `figures` run from stored predictions only (CPU) | `make analyze` on a synthetic predictions file |
| `README.md` with the one-command reproduction path | - |

## Step 3: Metric code and synthetic tests

- `src/metrics.py`: Brier, log loss, ECE (equal-width, equal-mass, debiased), flip/disagreement with the ε rule, paired cluster bootstrap, Holm.
- `src/calibrate.py`: temperature, Platt, isotonic (fixed hyperparameters as in 03 section 8).
- Tests on synthetic data with known calibration: a perfectly calibrated generator gives log-loss gain ≈ 0; a known temperature distortion is recovered; flip counts are exact on hand-built cases; the bootstrap reaches nominal coverage on a simple case.
- Verify: `pytest` passes.

## Step 4: Power simulation (sets the TBD numbers)

- Simulate predictions at the planned sizes (about 1,000 test items, groups of 6 for task A) under plausible miscalibration and flip rates.
- For each primary test: the probability of a supported verdict at the SESOI, and the false-positive rate at zero effect, after Holm.
- Output: proposed SESOIs and margins (H1 gain, H3 gap, H4 reduction, H5 δ, H6 difference) for co-author sign-off. If a test is underpowered at any plausible SESOI, merge it into a secondary test or drop it **before** the freeze.
- Verify: a script plus a short table in `results/power/`, and the values copied into `02`.

## Step 5: Phase 0 (user on workstation)

- Read the d1-2B model card: output interface, option limit (16 vs 18), order semantics, training data (contamination of MASSIVE/RAGTruth).
- Run the Phase 0 checks for d1-2B, vLLM-SR d3, Strands Decider 2B and the baseline endpoint (user-supplied). Record which decision models pass the gate.
- Record the decisions in the deviation log.

## Step 6: Document revisions, numeric pass

Fill in the SESOIs and margins, the noise floor ε, the option set for B, the token cap for A, the pinned revisions, the seeds and the baseline model identity. Mark `02` and `03` as "frozen candidate".

## Step 7: Freeze

Co-author review, then git tag (user commits), then OSF upload (user). Write the OSF link and tag hash into `03` section 10. Test-set predictions are generated only after this step.

---

## Inputs needed from the user

1. Baseline endpoint (`base_url`, model name, environment variable name for the key), and whether it is self-hosted (recommended) or a hosted API.
2. Co-author confirmation of the proposed primary tests (table in 1a) and, after step 4, of the SESOIs.
3. d1-2B model card findings and Phase 0 runs.
4. Early contact with the DPO/Legal about dataset licences and user-generated text.
5. OSF upload at the freeze, after reviewing the content.
