# Hypotheses (frozen before any test-set run)

Status: v1.0 proposed 13 Nov 2026. Any change after the freeze goes into the deviation log in `03_experiment_protocol.md`.

Numeric thresholds marked **[SESOI]** are smallest effects we consider practically meaningful. They are our a-priori choices, not derived from data. Confirm or adjust them before the pilot freeze.

## Terms

- **Condition** c = (p, o): question paraphrase p in {P0..P5} and option order o. Canonical condition c0 = (P0, O0).
- **Flip:** the top option (after mapping back to the original option identity) differs between two conditions for the same item. For binary tasks, the decision at 0.5 on P(positive).
- **ECE:** 10 equal-width bins; on P(positive) for binary tasks, top-label for multi-class.
- **Tasks:** A = binary factual check (RAGTruth QA subset); B = multi-class choice (MASSIVE scenario level). Primary model: d1-2B.
- Intervals: 95% cluster bootstrap over item groups (source or prompt), 2,000 resamples, unless stated.

## Research questions

- RQ1: Are the model's raw probabilities calibrated on our tasks?
- RQ2: How consistent are decisions and probabilities under rewording and option reordering?
- RQ3: Does a calibrator transfer across wordings and orders, and how many labels does it need?
- RQ4: Can ensembling and cost-based thresholds make the decisions more reliable and cheaper?

## Confirmatory hypotheses

### H1: Raw probabilities are miscalibrated, and the degree depends on the task (RQ1)

- **Statement:** at the canonical condition, raw ECE on the test set is above 0.05 for at least one of tasks A and B, and raw ECE differs between A and B.
- **Test:** lower bound of the ECE interval above 0.05 for at least one task; interval of the difference ECE_A minus ECE_B excludes 0. **[SESOI: 0.05]**
- **Supported if** both conditions hold. **Not supported if** both tasks have the ECE interval upper bound below 0.05 (the model is already well calibrated here, which is itself a result).
- **Why:** calibration is measured per task and per wording; "trained for calibration" is not a guarantee on our data.

### H2: Decisions are not invariant to wording and order (RQ2)

- **H2a (paraphrase):** with option order fixed at O0, the pairwise disagreement rate across paraphrases P0..P5 is at least 2% on the test set, on task A or B. **[SESOI: 2%]**
- **H2b (order):** with paraphrase fixed at P0, the pairwise disagreement rate across option orders is at least 2%, on task A or B. **[SESOI: 2%]**
  - Note: if the d1 model card shows that options are processed as an unordered set by design, a rate indistinguishable from numerical noise (see determinism check) is the expected result and counts as a verified property, not a failure of the study. H2b is then reported separately for each model.
- **H2c (confident flips):** among items with canonical top-class probability of at least 0.9, the flip rate versus the canonical condition (over all other conditions) is at least 0.5%. **[SESOI: 0.5%]**
- **Test:** lower bound of each interval at or above its threshold.
- **Supported if** the lower bound is at or above the threshold. **Refuted if** the upper bound is below it.
- **Why:** a calibrated probability is only trustworthy if the answer does not depend on incidental wording.

### H3: Calibration does not fully transfer across conditions (RQ3)

- **Statement:** a calibrator (temperature scaling; Platt as robustness check) fitted on the canonical condition on the calibration pool gives higher test ECE at non-canonical conditions than at the canonical condition.
- **H3a (paraphrase):** mean ECE over P1..P5 (order O0) minus ECE at c0 is at least 0.01. **[SESOI: 0.01]**
- **H3b (order):** mean ECE over O1..O(k-1) (paraphrase P0) minus ECE at c0 is at least 0.01.
- **H3c (pooled calibrator):** a calibrator fitted on pooled conditions closes at least half of the gap in H3a and H3b, measured on non-canonical test conditions. This is a secondary, directional test.
- **Test:** interval of each difference; lower bound at or above 0.01 for H3a/H3b.
- **Why:** if consistency fails, a calibrator fitted on one phrasing describes only that phrasing.

### H4: Recalibration helps, and non-parametric methods need more labels (RQ3)

- **H4a:** on any task where raw ECE exceeds 0.05, temperature scaling or Platt scaling fitted on n = 200 canonical-condition labels reduces test ECE by at least 30% relative to raw (mean over 200 resamples). **[SESOI: 30%]**
- **H4b:** at n = 50, mean test ECE after isotonic regression is higher than after temperature scaling (paired by resample; interval excludes 0), and isotonic harms (ECE above raw) in a larger share of resamples than temperature scaling.
- **Supported if** both parts hold. If the model is already calibrated (H1 not supported), H4a is reported as not testable.
- **Why:** parametric methods have few parameters; isotonic regression fits a step function and needs more data.

### H5: Averaging over perturbations improves consistency without hurting calibration (RQ4)

- **Statement:** averaging probabilities over all conditions of the grid (the "full ensemble") compared with a single canonical call:
  - **H5a:** cuts split-half disagreement by at least 50% relative. Split-half disagreement = flip rate between two ensembles built from disjoint random halves of the conditions. For the single-call baseline, the rate between two randomly chosen single conditions. **[SESOI: 50%]**
  - **H5b:** after calibration with the same method and label budget, the ensemble's test Brier score is not higher than the single call's (paired bootstrap; difference interval does not exclude zero in the wrong direction).
- **Why:** ensembling is a cheap fix, and decision models are fast, so K calls per item are affordable.

### H6: Cost-derived thresholds beat default rules, but only on calibrated probabilities (RQ4)

- **Setting:** binary task A (and D if included). Cost grid: c_false_block = 1; c_miss in {2, 5, 10, 25}; c_review in {0.1, 0.25, 0.5}; review treated as always correct. Cells where the lower cut-off exceeds the upper one use the single-threshold rule from the protocol.
- **H6a:** on calibrated probabilities, cost-derived thresholds give lower expected test cost than the default rules (single threshold 0.5; band 0.2/0.8) in more than half of the grid cells (interval excludes 0).
- **H6b:** cost-derived thresholds computed on calibrated probabilities give lower expected test cost than the same formulas applied to raw probabilities in more than half of the cells.
- **Why:** the formula assumes the probabilities mean what they say; this is where calibration should pay off.

## Exploratory analyses (reported as exploratory, no thresholds)

- **E1 Position bias:** accuracy by position of the correct option; share of probability mass by slot.
- **E2 Variance decomposition:** share of within-item variance of the top-class logit attributable to paraphrase, order, their interaction and residual.
- **E3 Option-label wording (task B):** descriptive versus short scenario names as a third factor.
- **E4 Other models:** vLLM-SR d3 and AWS Strands Decider 2B, only if they pass the feasibility gate; descriptive comparison, no ranking claims.
- **E5 Ordinal task C (HelpSteer2):** helpfulness as a five-level choice, expected-level calibration.
- **E6 Rare class D (ToxicChat jailbreaking):** H6 repeated, with wide intervals expected (about 100 positives per half).
- **E7 Inconsistency versus miscalibration:** whether items that flip are also the miscalibrated ones (flip rate by confidence bin).

## Analysis rules

- **Primary model and tasks:** d1-2B on A and B. Everything else is exploratory.
- **Multiplicity:** 95% intervals are reported per test without adjustment; because there are six hypotheses on two tasks, borderline results are described as such, and all tests are reported whether or not they succeed.
- **Null results:** reported in full. A well-calibrated or order-invariant model is a legitimate finding.
- **No peeking:** test-set predictions are generated once, after the freeze; the dev set is used for pilots.
- **Deviation log:** any change after the freeze, with date and reason, goes into the protocol.
