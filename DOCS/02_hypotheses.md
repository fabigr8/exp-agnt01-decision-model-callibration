# Hypotheses (frozen before any test-set run)

Status: v2.0 draft, 13 Nov 2026 (structural revision; see `04_revision_plan.md`). Values marked **TBD** are set by the power simulation (plan step 4) and Phase 0, then confirmed by co-authors before the freeze. At the freeze, this file is uploaded to OSF and tagged in git (`03_experiment_protocol.md` section 10). Any change after the freeze goes into the deviation log in `03`.

Thresholds marked **[SESOI]** are the smallest effects we consider practically meaningful. They are our a-priori choices, not derived from data. The proposed values are starting points for the power simulation.

## Terms

- **Models.** The confirmatory model is d1-2B. The other decision models (vLLM-SR d3, AWS Strands Decider 2B) are core but descriptive: each one that passes the Phase 0 gate runs the full grid, and all H1-H6 statistics are reported for it (E8), without hypothesis tests. The LLM logprob baseline (`03` section 2.1) is likewise descriptive (E4). Neither enters the Holm family.
- **Tasks.** A = binary factual check (RAGTruth QA subset); B = multi-class choice (MASSIVE en-US, scenario level).
- **Task A polarity.** Positive class = hallucinated. The option texts and the mapping from model output to P(positive) are fixed in `prompts/` before the freeze.
- **Condition** c = (p, o): question paraphrase p in {P0..P5} and option order o in {O0..O(k-1)}. Canonical condition c0 = (P0, O0). Outputs are always mapped back to the original option identity.
- **Margin.** For multi-class: probability of the top option minus probability of the runner-up. For binary: |2 P(positive) - 1|.
- **Noise floor ε.** ε = 2 × the maximum absolute probability difference observed in the Phase 0 determinism check. A difference of this size can come from numerical noise alone. **TBD (Phase 0).**
- **Disagreement (flip).** Two conditions disagree on an item if their top options differ (binary: their decisions at 0.5 differ) **and** the margin exceeds ε in both conditions. Differences with a margin at or below ε are not counted.
- **Log loss.** Mean negative natural-log probability of the true option, in nats per item; probabilities clipped to [1e-6, 1 - 1e-6].
- **Brier score.** Binary: mean (p - y)². Multi-class: mean over items of the sum over options of (p_k - y_k)².
- **ECE.** Descriptive only: reliability diagrams and debiased ECE (equal-width and equal-mass bins). It is not used in any confirmatory test.
- **Groups.** Task A: `source_id`. Task B: each utterance is its own group.
- **Intervals.** 95% cluster bootstrap over groups, 2,000 resamples, percentile intervals. All comparisons between conditions or methods are **paired**: they use the same resampled items. If a statistic involves fitting a calibrator, the calibration pool is resampled too and the calibrator is refitted in every resample.
- **Bootstrap p-value.** For a test of "θ ≥ s": p = (1 + #{θ* < s}) / (1 + 2,000), where θ* are the bootstrap replicates and s is the SESOI. For a non-inferiority test of "θ < δ": p = (1 + #{θ* ≥ δ}) / (1 + 2,000).

## Research questions

- RQ1: Are the model's raw probabilities calibrated on our tasks?
- RQ2: How consistent are decisions and probabilities under rewording and option reordering?
- RQ3: Does a calibrator transfer across wordings and orders, and how many labels does it need?
- RQ4: Can ensembling and cost-based thresholds make the decisions more reliable and cheaper?

## Primary tests

Exactly one primary test per hypothesis. The six are corrected together with **Holm** at family-wise α = 0.05.

| ID | Task | Statistic θ | Supported if | SESOI / margin (proposal) |
|---|---|---|---|---|
| H1 | A | Log-loss gain from Platt scaling at c0 | Holm-adjusted p ≤ 0.05 for θ ≥ s | s = **TBD** (proposal 0.01 nats) |
| H2 | B | Pairwise disagreement across P0..P5 at O0 | Holm-adjusted p ≤ 0.05 for θ ≥ s | s = **TBD** (proposal 0.02) |
| H3 | B | Transfer gap in log loss (paraphrase transfer matrix, temperature scaling) | Holm-adjusted p ≤ 0.05 for θ ≥ s | s = **TBD** (proposal 0.01 nats) |
| H4 | A | Share of the full-pool Platt gain reached with n = 200 labels | Holm-adjusted p ≤ 0.05 for θ ≥ s | s = **TBD** (proposal 0.8) |
| H5 | B | Brier(ensemble) - Brier(single c0), both calibrated | Holm-adjusted p ≤ 0.05 for θ < δ | δ = **TBD** (proposal 0.005) |
| H6 | A | Mean expected-cost saving of calibrated over raw cost-derived thresholds, averaged over the cost grid | Holm-adjusted p ≤ 0.05 for θ ≥ s | s = **TBD** (proposal 0.005 cost units per item) |

**Verdicts.** Each primary test has exactly one of three outcomes:
- **Supported:** the condition in the table holds.
- **Refuted:** the 95% interval lies entirely on the wrong side of the threshold. For H1-H4 and H6, the upper bound is below s. For H5, the lower bound is at or above δ.
- **Inconclusive:** anything else. Inconclusive results are reported as such, with the interval.

All secondary tests use unadjusted 95% intervals and are labelled "secondary" wherever they are reported.

## Confirmatory hypotheses

### H1: Raw probabilities are miscalibrated (RQ1)

- **Primary (task A, c0):** fit Platt scaling on the full calibration pool at c0. θ = log loss(raw) - log loss(Platt) on test at c0. A positive gain on held-out data means the raw probabilities can be improved by a monotone recalibration, i.e. they are miscalibrated.
- **Secondary:** the same on task B with temperature scaling; the same with Brier instead of log loss.
- **Descriptive:** reliability diagrams and debiased ECE for A and B. The difference between A and B is not tested, because the binary and top-label views of calibration are not comparable.
- **Why:** calibration is measured per task and per wording; being "trained for calibration" is no guarantee on our data.

### H2: Decisions are not invariant to wording (RQ2)

- **Primary (task B):** θ = mean over test items of the share of the 15 paraphrase pairs (P0..P5 at O0) that disagree, using the disagreement rule above.
- **Secondary:**
  - H2a-A: the same on task A.
  - H2b: pairwise disagreement across option orders at P0, on A and B. If the d1 model card states that options are processed as an unordered set, disagreement at or below the noise floor is the expected result and is reported as a verified property.
  - H2c: among items with canonical top-option probability ≥ 0.9, the flip rate versus c0 over all other conditions (proposal threshold 0.005).
- **Why:** a calibrated probability is only trustworthy if the answer does not depend on incidental wording.

### H3: Calibration does not fully transfer across wordings (RQ3)

- **Primary (task B, temperature scaling, order O0):** for each paraphrase i, fit a calibrator on the calibration pool at (P_i, O0). Evaluate it on the test set at every (P_j, O0). This gives a 6 × 6 matrix L[i, j] of test log loss. θ = mean of the off-diagonal entries (i ≠ j) minus mean of the diagonal entries. No single wording is privileged.
- **Secondary:**
  - H3-order: the same transfer matrix over the option orders at P0.
  - H3-A: the paraphrase matrix on task A (Platt scaling).
  - H3-Platt: the primary analysis with Platt scaling on top-label confidence.
  - **H3c (leave-one-condition-out):** for each paraphrase j, fit one calibrator on the calibration pool pooled over all other paraphrases, and evaluate it on the test set at P_j. Closing share = (off-diagonal mean - leave-one-out mean) / (off-diagonal mean - diagonal mean). Supported if the closing share is ≥ 0.5 and the interval of (off-diagonal mean - leave-one-out mean) lies above 0. This tests transfer to a wording the calibrator has never seen.
- **Why:** if consistency fails, a calibrator fitted on one phrasing describes only that phrasing.

### H4: A small label budget recovers most of the recalibration gain (RQ3)

- **Primary (task A, Platt, c0):** θ = mean over calibration subsets of [log loss(raw) - log loss(Platt fitted on n = 200)] divided by [log loss(raw) - log loss(Platt fitted on the full pool)], on test. Each of the 2,000 bootstrap replicates draws one random subset of n = 200 from the calibration pool (stratified to keep the pool's prevalence) and one cluster bootstrap of the test set.
- **Testable only if** H1 is supported. Otherwise the denominator is near 0 and H4 is reported as not testable.
- **Secondary:**
  - Learning curves for temperature, Platt and isotonic at n ∈ {25, 50, 100, 200, 400} on A and B.
  - **H4b (replication of a known result):** at n = 50, isotonic regression gives higher test log loss than temperature or Platt scaling (paired), and it does worse than raw in a larger share of subsets.
- **Why:** practitioners need to know how many labels to collect.

### H5: Averaging over perturbations does not hurt calibration (RQ4)

- **Primary (task B, non-inferiority):** the ensemble is the mean of the probabilities over all 36 conditions. The ensemble and the single call at c0 are each calibrated with temperature scaling on the full calibration pool, using the same procedure. θ = Brier(ensemble) - Brier(single), on test.
- **Secondary:** the same on task A (12 conditions); log loss instead of Brier.
- **Descriptive (replaces old H5a):** split-half disagreement as a function of ensemble size K ∈ {1, 2, 3, 6, half of all conditions}, from 200 random draws of two disjoint sets of K conditions per K. Lower disagreement at larger K is expected by construction. The curve shows how fast it falls, which tells practitioners how many calls are enough.
- **Why:** ensembling is a cheap fix for inconsistency, but only if it does not cost calibration.

### H6: Calibration pays off in cost-based decisions (RQ4)

- **Setting:** task A. Cost grid: c_false_block = 1; c_miss ∈ {2, 5, 10, 25}; c_review ∈ {0.1, 0.25, 0.5}; review is assumed always correct. Threshold formulas are in `03` section 9.
- **Primary:** θ = mean over the 12 grid cells of [expected test cost per item using the formulas on raw probabilities - the same using Platt-calibrated probabilities]. The calibrator is the same one as in H1 (full calibration pool, c0). One statistic, one bootstrap, so the correlation between cells is handled.
- **Key secondary:** calibrated formula versus empirical cost-minimising thresholds searched on the calibration pool. This is reported as a difference with interval, and it shows whether calibration adds anything over direct tuning.
- **Descriptive:** comparison with the default rules (0.5; band 0.2/0.8), per cell.
- **Why:** the threshold formulas assume that probabilities mean what they say; this is where calibration should pay off.

## Exploratory analyses (reported as exploratory, no thresholds)

- **E1 Position bias:** accuracy by position of the correct option; share of probability mass by slot.
- **E2 Variance decomposition:** within-item variance of the log-odds of the canonical top option, split into paraphrase, order, and "interaction + residual". The last two cannot be separated, because there is one observation per cell.
- **E3 Option-label wording (task B):** descriptive versus short scenario names as a third factor.
- **E4 LLM logprob baseline:** the full grid and all statistics of H1-H6 for the baseline, reported side by side with d1-2B. Descriptive; no tests and no ranking claims.
- **E5 Ordinal task C (HelpSteer2):** helpfulness as a five-level choice; calibration of the expected level.
- **E6 Rare class D (ToxicChat `jailbreaking`):** H6 repeated, with wide intervals expected (about 100 positives per half).
- **E7 Inconsistency versus miscalibration:** whether the items that flip are also the miscalibrated ones (flip rate by confidence bin).
- **E8 Other decision models (core, descriptive):** vLLM-SR d3 and AWS Strands Decider 2B. Every model that passes the Phase 0 gate runs the full grid on A and B. All H1-H6 statistics, with intervals, are reported per model next to d1-2B. No tests, no ranking claims. A model that does not expose per-option probabilities is reported as excluded, with the reason.

## Analysis rules

- **Confirmatory scope:** d1-2B, the six primary tests above. Other decision models (E8) and the baseline (E4) get the same statistics, descriptively. Everything else is secondary or exploratory and labelled as such.
- **Multiplicity:** Holm across the six primary tests. Secondary tests use unadjusted intervals and are never used to rescue a primary verdict.
- **Null results:** reported in full. A well-calibrated or order-invariant model is a legitimate finding.
- **No peeking:** test-set predictions are generated once, after the freeze. The dev set is for pilots only.
- **Deviation log:** any change after the freeze, with date and reason, goes into `03` section 10.
