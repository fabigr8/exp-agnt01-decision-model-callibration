# Experiment protocol: calibration and consistency of decision models

Status: v0.1 draft, 13 Nov 2026. Companion files: `01_article_outline.md`, `02_hypotheses.md`. Dataset facts below come from the shortlist note prepared 13 Nov 2026; nothing was downloaded, so all sizes, prevalences and lengths must be measured in Phase 0.

## 1. Overview

```
Phase 0  feasibility gate (dev data)     -> go / no-go per model and dataset
Phase 1  assets: data prep, paraphrases, orders, scripts -> FREEZE (commit hash)
Phase 2  run grid: models x tasks x conditions x pools   -> long-format predictions
Phase 3  analyses: consistency, calibration, transfer, learning curves, ensembling, thresholds
Phase 4  figures, tables, write-up
```

Principles: hypotheses and analysis plan frozen before test predictions exist; calibrators never see test items; every run pinned (model revision, code commit, seeds).

## 2. Models

| Model | Role | Status |
|---|---|---|
| d1-2B (https://huggingface.co/LiquidAI/d1-2B) | primary | model card not yet read by us; read it first |
| vLLM-SR d3 | optional comparison (E4) | interface unverified |
| AWS Strands Decider 2B | optional comparison (E4) | interface unverified |

Pinning: record the HF revision hash (or API version string), tokenizer version, dtype, inference library and version, GPU model and driver. Hardware: workstation with RTX PRO 6000 Blackwell.

## 3. Phase 0: feasibility gate

Run on **dev data only** (MASSIVE dev split; a dev share of RAGTruth sources). A model enters the main grid only if all required checks pass.

| Check | What to do | Required for |
|---|---|---|
| Output interface | Does the model return a probability per option (or per level), not only a top choice or a coarse score? | calibration (required) |
| Option limit | An Ollama config of a d1 build shows `d1_set_size: 16`, `d1_attention: "set"`. Meaning unverified. If it is a maximum number of options, MASSIVE's 18 scenarios do not fit. Read the model card and test with 17 and 18 options. | task B design |
| Order semantics | Does `d1_attention: "set"` mean options are processed as an unordered set? Test: shuffle options on 200 dev items and compare outputs numerically. | H2b interpretation |
| Determinism | 200 items x 3 repeats x 3 batch sizes. Record max absolute probability difference and flip count. This sets the numerical noise floor for flips. | all |
| Length | Token counts of RAGTruth QA items under the model tokenizer; context limit and failure mode. | task A filter |
| Throughput | Items per second on the workstation for short (B) and long (A) items. | schedule |
| Input format | How question, context and options are passed; whether paraphrased instructions are free text. | design |

Decisions after Phase 0 (to record in the deviation log): final option set for B (see 4.2), token cap for A, which extra models stay.

## 4. Datasets and task definitions

### 4.1 Task A: binary factual check (core). RAGTruth, QA subset

- **Item:** (retrieved passages, response). **Question:** is the response fully supported by the passages? **Positive class:** hallucinated (any annotated span; label derived from spans, so "no span" means "not marked").
- **Filters:** QA subset only (prevalence about 29% from the README table); drop items whose quality field flags truncated or wrongly refused answers (check field values in Phase 0); keep items under the token cap (target 1,000 tokens; set after Phase 0).
- **Grouping:** six responses share each source, so all splits are grouped by `source_id`.
- **Splits:** dev (about 10% of sources, pilots only); calibration pool (up to 1,000 items); test (all remaining items, target at least 1,000; if fewer than 1,000 remain, reduce the calibration pool to 500 and record it).
- **Options:** two (yes/no or supported/not supported), so the order factor has 2 levels.

### 4.2 Task B: multi-class choice (core). MASSIVE en-US, scenario level

- **Item:** one utterance. **Options:** 18 scenarios (e.g. alarm, weather) written as natural-language names. If Phase 0 shows an option limit of 16, keep the 16 most frequent scenarios and drop the rest (or merge the two rarest into "other"); document the choice. Fallback dataset: CLINC150 at domain level (10 classes) if its domain labels can be recovered.
- **Splits:** official train (calibration pool, 1,000 items drawn at random), official dev (pilots only), official test (2,974 items; after any class removal).
- **Order factor:** 6 permutations of the option list (4.5).

### 4.3 Extensions (only if time permits)

- **Task C, HelpSteer2 (E5):** helpfulness 0-4 as a five-level choice; filter by token cap; grouped by prompt; use integer labels and check that the published labels are integers. Use only helpfulness and correctness (kappa about 0.79); do not use coherence or complexity (kappa about 0.43).
- **Task D, ToxicChat 0124 `jailbreaking` (E6):** about 2% prevalence, about 204 positives overall. Learning curves at n above 200 are meaningless (about 4 positives per 200 draws); report H6 only, with wide intervals. About 44% of items have no human label; decide in Phase 0 whether to restrict to human-annotated items (prevalence changes; recheck it stays within 1-5% or reconsider).

### 4.4 Pools and integrity rules

- Pools are disjoint by group. Calibration pool: calibrator fitting and threshold selection only. Test: evaluation only. Dev: pilots only; never reused after the freeze.
- Fixed seed list (e.g. 11, 23, 37, 41, 59) recorded in the config; one master seed per split.

### 4.5 Question construction

**Canonical question (P0).** Written by the authors from the task definition, once, before any model run. No tuning on outputs.

**Paraphrases (P1-P5).** Template-level, not per item: the instruction text around the item is reworded; option names are held fixed in this factor (option-name wording is exploratory E3).
1. Generate 10 candidate paraphrases of P0 with an LLM.
2. Two authors independently check that meaning, task and label semantics are unchanged; discard any doubtful candidate.
3. Select 5 that differ clearly in wording and structure (length, voice, order of clauses).
4. Freeze text and store in the repository. Log all candidates and discards.
5. Selection must not look at any model output.

**Option orders.**
- Task A (2 options): O0 = original order, O1 = reversed.
- Task B (k options): O0 = original; O1 = reversed; O2, O3, O4 = cyclic shifts by round(k/4), round(k/2), round(3k/4); O5 = one fixed random permutation (seeded). The same permutation applies to every item, so differences between orders are not item noise.
- Always map outputs back to the original option identity before analysis.

**Grid size.** Task A: 6 paraphrases x 2 orders = 12 conditions. Task B: 6 x 6 = 36 conditions. Full factorial on calibration and test pools: A about 2,000 items x 12 = 24,000 forward passes; B about 4,000 items x 36 = 144,000. d1 reports millisecond-scale latency per item on GPUs in the vendor's blog, so this should take minutes to hours; measure in Phase 0.

## 5. Run grid and logging

- One run script per model, configuration-driven (YAML): model revision, dataset, split, conditions, batch size, dtype.
- **Long-format output** (one row per item x condition): `model, revision, task, split, item_id, group_id, paraphrase_id, order_id, option_order, probs_by_original_option (list), top_option, label, n_tokens, latency_ms, run_id`.
- Save raw model outputs (logits or probabilities as returned) separately from derived fields.
- Log code commit, library versions, GPU, driver, timestamps.
- Repeat a 200-item subset at the end of the run to confirm stability against the Phase 0 noise floor.

## 6. Metrics

**Consistency** (computed per task, per model):
- **Flip rate vs canonical:** mean over items and non-canonical conditions of 1[top option differs from the canonical top option].
- **Pairwise disagreement:** per item, the share of condition pairs with different top options; reported over (a) paraphrases at fixed order, (b) orders at fixed paraphrase, (c) all conditions.
- **Unanimity:** share of items where all conditions agree.
- **Probability drift:** mean absolute difference in P(canonical top option); for multi-class also mean Jensen-Shannon divergence between condition and canonical distributions.
- **Confident flips:** flip rate among items with canonical top probability at least 0.9.
- **Noise floor:** the Phase 0 determinism result; flips below it are not counted as inconsistency.

**Calibration:**
- **ECE:** 10 equal-width bins; binary tasks on P(positive), multi-class on top-label confidence. Robustness: 10 equal-mass bins.
- **Brier score** (multi-class: sum over options) and **log loss** (probabilities clipped to [1e-6, 1 - 1e-6]).
- Reliability diagrams with bin counts.

**Decisions and cost:** accuracy and coverage of the automated part; expected cost per item (section 9).

**Uncertainty:** cluster bootstrap over groups (source_id, prompt), 2,000 resamples, percentile 95% intervals; for learning curves, interval over calibration-set resamples plus test bootstrap reported separately.

## 7. Analyses by hypothesis

| Hyp. | Procedure |
|---|---|
| H1 | Canonical condition on test; ECE (A, B) with intervals; bootstrap the difference. |
| H2a-c | Compute pairwise disagreement over paraphrases at O0 and over orders at P0; confident-flip rate over all conditions; compare lower bounds to the thresholds in `02_hypotheses.md`. Subtract the noise floor. |
| H3 | Fit temperature scaling (Platt as robustness) on calibration pool at c0; evaluate ECE on test at every condition; compute gaps. For H3c fit on all conditions of the calibration pool pooled; compare on non-canonical conditions. |
| H4 | Learning curve: n in {25, 50, 100, 200, 500, 1000}; 200 random calibration subsets per n (without replacement, from the calibration pool; stratified by label for binary tasks); methods: temperature, Platt, isotonic; evaluate on full test at c0. Report mean ECE with 2.5-97.5% range over subsets and the share of subsets where calibrated ECE is above raw ECE. |
| H5 | Ensemble = mean of probabilities over all conditions (also by factor: orders only, paraphrases only). Split-half: repeat 200 times: partition the conditions at random into two halves, build an ensemble from each, compute flip rate; baseline is two random single conditions. Then calibrate (same method and budget) and compare Brier with paired bootstrap. |
| H6 | Section 9. |

## 8. Calibration procedures

- **Temperature scaling:** minimise negative log-likelihood of softmax(log p / T) (binary: logit(p) / T) with L-BFGS, T > 0. For multi-class, calibrate the full distribution; the top option does not change.
- **Platt scaling (binary):** logistic regression on logit(p) with two parameters, no regularisation; for multi-class tasks apply to top-label confidence (is the top option right) as a robustness check.
- **Isotonic regression:** monotone step function on P(positive) (binary) or top-label confidence (multi-class), out-of-bounds clipping.
- Fit on the calibration pool only; hyperparameters (bins, clipping, optimiser) fixed here and not tuned on test.

## 9. Cost-based thresholds (H6)

Binary task A (and D). Costs: c_false_block (a negative blocked), c_miss (a positive let through), c_review (one review, assumed always correct).

- Lower cut-off = c_review / c_miss; upper cut-off = 1 - c_review / c_false_block. Below lower: let through; above upper: block; between: review.
- If lower is not below upper, review never pays; use the single cut-off c_false_block / (c_miss + c_false_block).
- **Grid:** c_false_block = 1; c_miss in {2, 5, 10, 25}; c_review in {0.1, 0.25, 0.5}.
- **Rules compared:** (1) default single threshold 0.5; (2) default band 0.2/0.8; (3) cost-derived on raw probabilities; (4) cost-derived on calibrated probabilities (temperature or Platt, 200 labels); (5) empirical cost-minimising search on the calibration pool (cross-check only).
- **Metric:** expected cost per test item; also coverage, accuracy of automated decisions, and review share.
- **Multi-class task B (illustrative):** review if top confidence is below 1 - c_review / c_error; same rules and grid with c_error replacing c_miss.

## 10. Pilot and freeze

- Pilots on dev data only: check scripts, Phase 0 results, paraphrase sanity, metric code on synthetic data (known calibration).
- **Freeze:** hypotheses, SESOIs, paraphrases, orders, splits, seeds, analysis code. Tag the commit and store the hash in this file.
- Test predictions are generated after the tag. Changes after the tag are logged below.

**Deviation log**

| Date | Change | Reason | Affects |
|---|---|---|---|
| | | | |

## 11. Schedule (two weekends)

| Slot | Work |
|---|---|
| Weekend 1, Sat | Phase 0 gate; data prep for A and B; paraphrase generation and review; scripts |
| Weekend 1, Sun | Freeze; d1-2B full grid on A and B; consistency analysis (H2); first reliability plots (H1) |
| Weekend 2, Sat | Calibration, transfer, learning curves, ensembling, thresholds (H3-H6) |
| Weekend 2, Sun | Extensions as time allows (C, D, other models); figures; Results first draft |
| Between | Introduction, Background, Method written in parallel |

**Cut line:** if behind after Weekend 1, drop tasks C and D and the extra models; keep A, B and all of H1-H6 for d1-2B.

## 12. Threats to validity and mitigation

| Threat | Mitigation |
|---|---|
| Contamination of public datasets (MASSIVE, RAGTruth 2023-24) | State it; do not claim absolute accuracy; focus on relative effects across conditions |
| Label noise (RAGTruth label derived from spans; later data update added hallucinations) | Report as a floor on measurable calibration; sensitivity check dropping high-uncertainty items if identifiable |
| Paraphrases generated by one LLM, narrow style | Two-author review; state in Limitations; optionally add manually written paraphrases |
| Two tasks, one primary model | Claim only for these settings; extra models as descriptive |
| Group dependence (six responses per source) | Grouped splits and cluster bootstrap |
| Numerical nondeterminism (bf16, batch size) | Determinism check; noise floor subtracted |
| ECE bias and bin dependence | Large test sets, two binning schemes, report Brier and log loss |
| Rare class too thin (D) | Treat as exploratory with wide intervals |
| Model interface may not expose probabilities | Phase 0 gate; model excluded from calibration analyses if not |

## 13. Repository layout and deliverables

```
configs/        run and analysis configs (YAML)
data/           prepared splits (not committed if licence requires)
prompts/        canonical question, paraphrases, orders (frozen)
src/            prep_data.py, run_grid.py, metrics.py, calibrate.py, analyze.py
results/        long-format predictions, tables, figures
docs/           01_article_outline.md, 02_hypotheses.md, 03_experiment_protocol.md
```

Deliverables: predictions file per model; Table 2 (hypothesis verdicts); Figures 2-8; reproducible scripts.

## 14. Open items to settle before the freeze

1. d1-2B model card: output interface, option limit, order semantics.
2. Interfaces of vLLM-SR d3 and Strands Decider 2B (probabilities per option? option order handling?).
3. Option set for task B (16 versus 18).
4. Token cap and quality filter for RAGTruth.
5. Confirm SESOIs and the cost grid with co-authors.
6. Who reviews the paraphrases (two authors).

## 15. Deferred (not part of this experiment)

Dataset licence review (several candidates have non-commercial, share-alike or no-derivatives terms), privacy review of datasets with user-generated text (RAGTruth Yelp sources, ToxicChat), and internal approvals. These must be completed before the article or any data derived from the datasets is published. All outputs should be reviewed and validated by the authors before sharing externally.
