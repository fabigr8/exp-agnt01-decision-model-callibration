# Experiment protocol: calibration and consistency of decision models

Status: v2.0 draft, 13 Nov 2026 (structural revision; see `04_revision_plan.md`). Companion files: `01_article_outline.md`, `02_hypotheses.md`. Dataset facts below come from the shortlist note prepared 13 Nov 2026. Nothing has been downloaded yet, so all sizes, prevalences and lengths must be measured in Phase 0. Values marked **TBD** are set in Phase 0 or by the power simulation before the freeze.

## 1. Overview

```
Phase 0  feasibility gate (dev data)                -> go / no-go per model and dataset
Phase 1  assets: data prep, paraphrases, orders, metric code, power simulation
         -> numeric pass on 02/03 -> FREEZE (git tag + OSF preregistration)
Phase 2  run grid: models x tasks x conditions x pools   -> long-format predictions
Phase 3  analyses: consistency, calibration, transfer, learning curves, ensembling, thresholds
Phase 4  figures, tables, write-up
```

Principles:
- Hypotheses, thresholds and analysis code are frozen before any test prediction exists, and the freeze is published on OSF.
- Phase 0 comes before the freeze, because several hypotheses depend on its results.
- Calibrators never see test items.
- Every run is pinned: model revision, code commit, environment lockfile, seeds.
- All results can be regenerated from the stored predictions without a GPU.

## 2. Models

| Model | Role | Status |
|---|---|---|
| d1-2B (https://huggingface.co/LiquidAI/d1-2B) | confirmatory | model card not yet read by us; read it first |
| LLM logprob baseline (OpenAI-compatible endpoint) | descriptive reference (E4), always run | endpoint and model to be specified by the user in `configs/endpoints.yaml` |
| vLLM-SR d3 | core, descriptive (E8); full grid if it passes Phase 0 | returns per-option probabilities (user check of the documentation, 13 Nov 2026); call details and order handling to verify in Phase 0 |
| AWS Strands Decider 2B | core, descriptive (E8); full grid if it passes Phase 0 | returns per-option probabilities (user check of the documentation, 13 Nov 2026); call details and order handling to verify in Phase 0 |

Pinning: record the HF revision hash (or API version string), tokenizer version, dtype, inference library and version, GPU model and driver. For the baseline, also record the server software and version and the served model revision. Hardware: workstation with RTX PRO 6000 Blackwell.

### 2.1 Baseline scoring

- **Prompting.** The baseline receives the same question text (P0..P5) and option list as d1, with each option preceded by a single-token label (A, B, C, ...). It is asked to answer with the label only.
- **Probabilities.** Request settings: `temperature = 0`, `max_tokens = 1`, `logprobs = true`, `top_logprobs = min(20, server maximum)`. P(option) = exp(logprob of its label) from the first answer token, renormalised over the option labels. The missing probability mass (1 minus the summed label probabilities) is logged per item. If a label is absent from `top_logprobs`, it gets the remaining mass split evenly among missing labels, and the item is flagged.
- **Order factor.** Labels stay attached to positions; options move between positions. Outputs are mapped back to the original option identity, as for d1.
- **Endpoint config.** `configs/endpoints.yaml` (git-ignored) with fields `base_url`, `model`, `api_key_env` (the **name** of an environment variable holding the key; never the key itself), `top_logprobs`, `timeout_s`. A template, `configs/endpoints.example.yaml`, is committed.
- **Recommendation.** Use a self-hosted OpenAI-compatible server (e.g. vLLM) with a pinned model revision. A hosted API cannot be pinned and its logprobs may vary between calls; if one is used, this is recorded as a limitation and repeatability is measured in Phase 0.

## 3. Phase 0: feasibility gate

Run on **dev data only** (MASSIVE dev split; the RAGTruth dev sources, section 4.1). The decision-model checks below apply to every decision model (d1-2B, vLLM-SR d3, Strands Decider 2B). A model enters the main grid only if all required checks pass. Phase 0 results feed the numeric pass of `02` and this file before the freeze.

| Check | What to do | Required for |
|---|---|---|
| Output interface | Does the model return a probability per option (or per level), not only a top choice or a coarse score? | calibration (required) |
| Option limit | An Ollama config of a d1 build shows `d1_set_size: 16`, `d1_attention: "set"`. Meaning unverified. If it is a maximum number of options, MASSIVE's 18 scenarios do not fit. Read the model card and test with 17 and 18 options. | task B design |
| Order semantics | Does `d1_attention: "set"` mean options are processed as an unordered set? Test: shuffle options on 200 dev items and compare outputs numerically. | H2b interpretation |
| Determinism | 200 items × 3 repeats × 3 batch sizes. Record the maximum absolute probability difference δ_max and the flip count. Sets the noise floor ε = 2 δ_max (`02`, Terms). | all |
| Length | Token counts of RAGTruth QA items under the model tokenizer; context limit and failure mode. | task A filter |
| Throughput | Items per second on the workstation for short (B) and long (A) items. | schedule |
| Input format | How question, context and options are passed; whether paraphrased instructions are free text. | design |
| Training data | Does the model card list MASSIVE, RAGTruth or derived data as training data? | contamination (section 12) |
| Baseline: logprobs | The endpoint returns `top_logprobs` with at least as many entries as options (task B needs ≥ 18 or ≥ 16). Missing probability mass per item on 200 dev items (median and 95th percentile). | baseline (required) |
| Baseline: repeatability | 200 items × 3 repeats; δ_max and flip count, as for d1. | baseline noise floor |

Decisions after Phase 0 (record in the deviation log): final option set for B (see 4.2), token cap for A, the RAGTruth split rule (4.1), ε per model, which decision models pass the gate (and why any are excluded).

## 4. Datasets and task definitions

Dataset versions (HF revision or release tag) are pinned in `configs/data.yaml`. The prepared splits are verified against `data/checksums.sha256`.

### 4.1 Task A: binary factual check (core). RAGTruth, QA subset

- **Item:** (retrieved passages, response). **Question:** is the response fully supported by the passages?
- **Polarity:** positive class = hallucinated (any annotated span; the label is derived from spans, so "no span" means "not marked"). The two option texts are fixed in `prompts/task_a/options.yaml` before the freeze, together with the mapping to P(positive).
- **Filters:**
  - QA subset only (prevalence about 29% according to the README table).
  - Drop items whose quality field flags truncated or wrongly refused answers (check the field values in Phase 0).
  - Keep items under the token cap (target 1,000 tokens; **TBD** after Phase 0).
  - Record the prevalence before and after each filter.
- **Grouping:** six responses, from different response models, share each source, so all splits are grouped by `source_id`. The response model is recorded per item.
- **Split rule (Phase 0 decision):**
  - If the official test split of the QA subset has at least 1,500 items after filtering, draw the calibration pool and the test set from it (grouped), so both share the same exposure to model training. Dev sources come from the official train split.
  - Otherwise, re-split all QA sources grouped by `source_id` and record the reason.
- **Sizes:** dev about 10% of sources (pilots only); calibration pool up to 1,000 items; test = the rest, target at least 1,000. If fewer than 1,000 test items remain, reduce the calibration pool to 500 and record it.
- **Options:** two, so the order factor has 2 levels.

### 4.2 Task B: multi-class choice (core). MASSIVE en-US, scenario level

- **Item:** one utterance.
- **Options:** 18 scenarios (e.g. alarm, weather) written as natural-language names. O0 = alphabetical order of the option names.
  - If Phase 0 shows an option limit of 16, keep the 16 most frequent scenarios and drop the rest (or merge the two rarest into "other"); document the choice.
  - Fallback dataset: CLINC150 at domain level (10 classes), if its domain labels can be recovered.
- **Splits:** official dev for pilots only. Calibration pool and test are both drawn from the **official test split** (2,974 items before any class removal): 1,000 items at random for the calibration pool, the rest (about 1,974) for test. The official train split is not used, because a model is more likely to have seen train data than test data, and the two pools should have the same exposure.
- **Grouping:** none; each utterance is its own group.
- **Order factor:** 6 permutations of the option list (4.5).

### 4.3 Extensions (only if time permits, after the other decision models and the baseline)

- **Task C, HelpSteer2 (E5):** helpfulness 0-4 as a five-level choice; filtered by the token cap; grouped by prompt; use integer labels and check that the published labels are integers. Use only helpfulness and correctness (kappa about 0.79); do not use coherence or complexity (kappa about 0.43).
- **Task D, ToxicChat 0124 `jailbreaking` (E6):** about 2% prevalence, about 204 positives overall.
  - Learning curves at n above 200 are meaningless (about 4 positives per 200 draws); report H6 only, with wide intervals.
  - About 44% of items have no human label. Decide in Phase 0 whether to restrict to human-annotated items. That changes the prevalence; recheck that it stays within 1-5%, or reconsider.

### 4.4 Pools, seeds and integrity rules

- **Pool roles:**
  - Pools are disjoint by group.
  - Calibration pool: calibrator fitting and threshold selection only.
  - Test: evaluation only.
  - Dev: pilots only; never reused after the freeze.
- **Seeds:**
  - One master seed, `20261113`, recorded in `configs/seeds.yaml`.
  - Every random step derives its own seed as `int(sha256(f"{master}:{task}:{purpose}").hexdigest()[:8], 16)`, where `purpose` names the step (e.g. `split`, `order_O5`, `learning_curve_n200`, `bootstrap_H3`).
  - No other seeds are used.

### 4.5 Question construction

**Canonical question (P0).** Written by the authors from the task definition, once, before any model run. No tuning on outputs.

**Paraphrases (P1-P5).** Template-level, not per item: the instruction text around the item is reworded. Option names are held fixed in this factor (option-name wording is exploratory, E3).
1. Generate 10 candidate paraphrases of P0 with an LLM. Record the LLM, its version, the prompt, the sampling settings and the date in `prompts/paraphrase_generation.yaml`.
2. Two authors independently check that meaning, task and label semantics are unchanged; discard any doubtful candidate.
3. Select 5 that differ clearly in wording and structure (length, voice, order of clauses).
4. Freeze the text and store it in the repository. Log all candidates and discards. The stored text is the record; LLM generation cannot be reproduced exactly.
5. Selection must not look at any model output.

**Option orders.**
- Task A (2 options): O0 = original order, O1 = reversed.
- Task B (k options): O0 = alphabetical; O1 = reversed; O2, O3, O4 = cyclic shifts by k/4, k/2, 3k/4, each rounded half up (k = 18: 5, 9, 14; k = 16: 4, 8, 12); O5 = one fixed random permutation (seed purpose `order_O5`). The same permutation applies to every item, so differences between orders are not item noise.
- Always map outputs back to the original option identity before analysis.

**Grid size.**
- Task A: 6 paraphrases × 2 orders = 12 conditions.
- Task B: 6 × 6 = 36 conditions.
- Full factorial on calibration and test pools, per model:
  - A: about 2,000 items × 12 = 24,000 forward passes.
  - B: about 3,000 items × 36 = 108,000.
- d1 reports millisecond-scale latency per item on GPUs in the vendor's blog, so this should take minutes to hours; measure in Phase 0. Baseline throughput depends on the endpoint; also measure it in Phase 0.

## 5. Run grid and logging

- One run script with one backend per model interface (`d1` local; one each for vLLM-SR d3 and Strands Decider 2B, defined after Phase 0; `openai_compat` for the baseline), configuration-driven (YAML): model revision, dataset, split, conditions, batch size, dtype.
- **Long-format output** (one row per item × condition): `model, revision, task, split, item_id, group_id, paraphrase_id, order_id, option_order, probs_by_original_option (list), top_option, margin, label, n_tokens, latency_ms, missing_mass (baseline only), run_id`.
- Save raw model outputs (logits, probabilities or logprobs as returned) separately from derived fields.
- Store run metadata with every predictions file: code commit, lockfile hash, library versions, GPU, driver, server or API version (baseline and any API-served decision model), timestamps.
- Determinism settings for d1: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed batch size.
- Repeat a 200-item subset at the end of the run to confirm stability against the Phase 0 noise floor.

## 6. Metrics

**Calibration (primary: Brier score and log loss):**
- **Log loss:** natural log, probabilities clipped to [1e-6, 1 - 1e-6].
- **Brier score:** binary (p - y)²; multi-class summed over options.
- **ECE (descriptive only):** debiased ECE with 10 equal-width bins and with 10 equal-mass bins; binary tasks on P(positive), multi-class on top-label confidence. Reliability diagrams with bin counts.

**Consistency** (computed per task, per model; disagreement rule with noise floor ε as in `02`, Terms):
- **Pairwise disagreement:** per item, the share of condition pairs that disagree. Reported over (a) paraphrases at fixed order, (b) orders at fixed paraphrase, (c) all conditions.
- **Flip rate vs canonical:** mean over items and non-canonical conditions of disagreement with c0.
- **Unanimity:** share of items where all conditions agree.
- **Probability drift:** mean absolute difference in P(canonical top option); for multi-class also mean Jensen-Shannon divergence between the condition and canonical distributions.
- **Confident flips:** flip rate among items with canonical top probability at least 0.9.

**Decisions and cost:** accuracy and coverage of the automated part; review share; expected cost per item (section 9).

**Uncertainty:**
- Cluster bootstrap over groups, 2,000 resamples, percentile 95% intervals, paired across conditions and methods.
- Calibrators are refitted inside each resample when the statistic involves fitting.
- Bootstrap p-values and the Holm procedure as in `02`.

## 7. Analyses by hypothesis

The primary test per hypothesis is defined in `02`; this table gives the procedure.

| Hyp. | Procedure |
|---|---|
| H1 | Fit Platt on the calibration pool at c0 (task A); log-loss gain on test at c0. Secondary: task B with temperature scaling; Brier. Descriptive: reliability diagrams, debiased ECE. |
| H2 | Pairwise disagreement over P0..P5 at O0 on task B (primary), with ε. Secondary: task A; orders at P0; confident flips over all conditions. |
| H3 | Paraphrase transfer matrix on task B (temperature scaling): fit at each (P_i, O0) on the calibration pool, evaluate log loss on test at each (P_j, O0); gap = off-diagonal mean - diagonal mean. Secondary: order matrix; task A with Platt; Platt on top-label confidence. H3c: leave-one-condition-out pooled calibrator, closing share. |
| H4 | Task A, Platt: in each of 2,000 bootstrap replicates draw one stratified subset of n = 200 from the calibration pool and one test resample; ratio of the n = 200 gain to the full-pool gain. Secondary learning curves: n ∈ {25, 50, 100, 200, 400} (capped at half the calibration pool, so that subsets differ), 200 subsets per n, temperature, Platt and isotonic, tasks A and B; mean log loss with the 2.5-97.5% range over subsets, and the share of subsets worse than raw. H4b at n = 50. |
| H5 | Ensemble = mean of probabilities over all conditions (also by factor: orders only, paraphrases only). Calibrate the ensemble and the single c0 call with temperature scaling on the full calibration pool. Non-inferiority on the Brier difference against margin δ (task B primary). Descriptive: split-half disagreement versus K, 200 draws per K. |
| H6 | Section 9. Primary: mean over the 12 cost cells of cost(raw formula) - cost(calibrated formula), task A, Platt calibrator from H1. Key secondary: calibrated formula versus empirical cost search. |
| Holm | Collect the six primary bootstrap p-values, apply Holm at α = 0.05, assign verdicts as defined in `02`. |

## 8. Calibration procedures

- **Temperature scaling:** minimise the negative log-likelihood of softmax(log p / T) (binary: sigmoid(logit(p) / T)) with L-BFGS, T > 0. For multi-class, calibrate the full distribution; the top option does not change.
- **Platt scaling (binary):** logistic regression on logit(p) with two parameters, no regularisation. For multi-class tasks it is applied to top-label confidence (is the top option right?) as a robustness check.
- **Isotonic regression:** monotone step function on P(positive) (binary) or top-label confidence (multi-class), out-of-bounds clipping.
- Fit on the calibration pool only. Hyperparameters (clipping, optimiser, tolerances) are fixed here and not tuned on test.

## 9. Cost-based thresholds (H6)

Binary task A (and D). Costs: c_false_block (a negative blocked), c_miss (a positive let through), c_review (one review, assumed always correct).

- **Three-way rule:** lower cut-off = c_review / c_miss; upper cut-off = 1 - c_review / c_false_block. Below lower: let through; above upper: block; between: review.
- **Fallback:** if lower is not below upper, review never pays; use the single cut-off c_false_block / (c_miss + c_false_block). With c_false_block = 1, a review band exists exactly when c_review < c_miss / (c_miss + 1). That holds in every cell of the grid below, so the fallback applies only outside this grid (e.g. task D variants).
- **Grid:** c_false_block = 1; c_miss ∈ {2, 5, 10, 25}; c_review ∈ {0.1, 0.25, 0.5}.
- **Rules compared:**
  1. Default single threshold 0.5.
  2. Default band 0.2/0.8.
  3. Cost-derived on raw probabilities.
  4. Cost-derived on calibrated probabilities (Platt, full calibration pool, as in H1).
  5. Empirical cost-minimising search for both cut-offs on the calibration pool.
- **Comparisons:** primary 4 vs 3 (averaged over cells); key secondary 4 vs 5; descriptive 4 vs 1 and 2, per cell.
- **Metric:** expected cost per test item; also coverage, accuracy of automated decisions, and review share.
- **Multi-class task B (illustrative):** review if top confidence is below 1 - c_review / c_error; same rules and grid, with c_error replacing c_miss.

## 10. Pilot and freeze

- Pilots on dev data only: check scripts, Phase 0 results and paraphrase sanity. Test the metric code on synthetic data with known calibration.
- **Power simulation** (before the freeze): simulate predictions at the planned sizes and group structure. For each primary test, estimate the power at the proposed SESOI and the false-positive rate at zero effect, after Holm. Set the TBD thresholds and margins from it. Drop or demote any test that cannot reach adequate power.
- **Freeze:**
  - Contents: hypotheses, SESOIs and margins, ε, paraphrases, orders, splits, seeds, analysis code.
  - Tag the commit and store the hash here.
  - Upload `02`, this file and the code hash to OSF as a preregistration (by the authors, after review), and store the OSF link here.
- Test predictions are generated after the tag. Changes after the tag are logged below.

**Freeze record:** git tag `TBD`, commit `TBD`, OSF link `TBD`.

**Deviation log**

| Date | Change | Reason | Affects |
|---|---|---|---|
| | | | |

## 11. Schedule

| Slot | Work |
|---|---|
| Weekend 1, Sat | Phase 0 gate (all decision models and baseline); data prep for A and B; paraphrase generation; scripts and metric tests |
| Weekend 1, Sun | Paraphrase review by two authors; power simulation; numeric pass on `02`/`03` |
| Between | Co-author sign-off on primary tests and SESOIs; **freeze** (git tag + OSF) |
| Weekend 2, Sat | Full grid on A and B for d1-2B, the decision models that passed the gate, and the baseline; H1, H2; calibration, transfer, learning curves |
| Weekend 2, Sun | Ensembling, thresholds (H5, H6); figures; Results first draft |
| Buffer | Weekend 3 reserved for overruns; extensions only if the buffer is unused |
| Between | Introduction, Background, Method written in parallel |

**Cut line:** the freeze is not rushed to keep the schedule. If behind, drop in this order: tasks C and D, the baseline, the other decision models (E8). Keep A, B and all six primary tests for d1-2B.

## 12. Threats to validity and mitigation

| Threat | Mitigation |
|---|---|
| Contamination of public datasets (MASSIVE, RAGTruth 2023-24) | Check the model card's training data; draw calibration and test pools from the same split; do not claim absolute accuracy; focus on relative effects across conditions |
| Different contamination between pools | Both pools come from the same official split (4.1, 4.2) |
| Label noise (RAGTruth label derived from spans; a later data update added hallucinations) | Report it as a floor on measurable calibration; sensitivity check dropping high-uncertainty items if identifiable |
| Paraphrases generated by one LLM, narrow style | Two-author review; state in Limitations; optionally add manually written paraphrases |
| Two tasks, one confirmatory model | Claim only for these settings; the other decision models and the baseline are run in full but reported descriptively; claims about decision models in general are limited to the models that passed the gate |
| Baseline endpoint nondeterminism or silent model updates | Self-hosted pinned server recommended; repeatability measured in Phase 0 and at the end of the run; server version logged |
| Baseline label-token scoring loses probability mass | Missing mass logged per item; flagged items reported |
| Group dependence (six responses per source) | Grouped splits and cluster bootstrap |
| Numerical nondeterminism (bf16, batch size) | Determinism check; noise floor ε in the disagreement rule |
| ECE bias and bin dependence | ECE descriptive only, debiased estimator; Brier and log loss primary |
| Many sub-tests | One primary test per hypothesis, Holm correction; everything else labelled secondary |
| Underpowered tests | Power simulation before the freeze sets SESOIs and margins |
| Rare class too thin (D) | Treat as exploratory with wide intervals |
| Model interface may not expose probabilities | Phase 0 gate; the model is excluded from calibration analyses if not |

## 13. Repository layout and deliverables

```
pyproject.toml, uv.lock   pinned environment
Makefile                  data, pilot, grid, analyze, figures (analyze/figures run from stored predictions, CPU only)
configs/                  run, analysis, data versions, seeds; endpoints.example.yaml (endpoints.yaml git-ignored)
data/                     prepared splits (not committed if the licence requires) + checksums.sha256
prompts/                  canonical question, paraphrases, candidates and discards, options, orders (frozen)
src/                      prep_data.py, run_grid.py, metrics.py, calibrate.py, analyze.py, power_sim.py
tests/                    metric and calibration tests on synthetic data
results/                  long-format predictions, power simulation, tables, figures
DOCS/                     01_article_outline.md, 02_hypotheses.md, 03_experiment_protocol.md, 04_revision_plan.md
```

Deliverables: a predictions file per model; Table 2 (hypothesis verdicts); main-text figures and appendix figures; reproducible scripts; the OSF preregistration.

## 14. Open items to settle before the freeze

1. d1-2B model card: output interface, option limit, order semantics, training data.
2. Baseline endpoint and model (`configs/endpoints.yaml`), self-hosted or hosted.
3. Interfaces of vLLM-SR d3 and Strands Decider 2B: per-option probabilities confirmed from documentation (13 Nov 2026); call details, option limit and order handling still to verify in Phase 0.
4. Option set for task B (16 versus 18).
5. Token cap, quality filter and split rule for RAGTruth.
6. Co-author confirmation of the six primary tests; SESOIs and margins after the power simulation.
7. Who reviews the paraphrases (two authors).

## 15. Deferred (not part of this experiment), with early contact recommended

- **What is deferred:**
  - Dataset licence review: several candidates have non-commercial, share-alike or no-derivatives terms.
  - Privacy review of datasets with user-generated text (RAGTruth Yelp sources, ToxicChat).
  - Internal approvals.
- **When:** contact the Data Privacy Officer and Legal **before** downloading or storing RAGTruth and ToxicChat, not only before publication, so that a restriction cannot force a late change of task. All reviews must be complete before the article or any data derived from the datasets is published.
- **Review:** all outputs should be reviewed and validated by the authors before sharing externally.
