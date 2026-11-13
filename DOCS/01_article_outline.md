# Article outline (IMRAD-style, 6-8 pages, light academic tone)

Status: draft v0.1, 13 Nov 2026. Hypothesis IDs (H1-H6) refer to `02_hypotheses.md`; experiment details to `03_experiment_protocol.md`.

## Working title

**Can You Trust the Probabilities? Calibration and Consistency of Decision Models**

Alternatives: "A 0.8 Is Only Useful If It Means 80%" / "Same Question, Different Answer: Recalibrating Decision Models".

## Positioning

- **Audience:** technical practitioners and applied researchers who use or evaluate decision models (typed answers with probabilities, no generated text).
- **Tone:** ICML-style structure, plain language. Short sentences, one idea per paragraph, no jargon without a one-line gloss.
- **Scope rule:** every claim in Results comes from our own experiment. External benchmark numbers and vendor claims are not presented as findings. Related work is kept to a few lines.
- **Naming rule:** a model is called inconsistent only where we measured it.

## Contributions (to be claimed in the Introduction)

1. A simple protocol to measure how consistent a decision model is under rewording and option reordering (flip rate, probability drift, variance decomposition).
2. Evidence on whether a calibrator fitted under one wording transfers to other wordings and orders.
3. Label-budget guidance for recalibration (learning curves for temperature, Platt and isotonic).
4. A practical recipe: measure, ensemble over perturbations, recalibrate, set cost-based thresholds.

## Section plan and page budget

| Section | Pages | Hypotheses |
|---|---|---|
| Abstract | 0.2 | all |
| 1 Introduction | 0.75 | motivation |
| 2 Background | 0.5 | - |
| 3 Method | 1.25 | - |
| 4 Experimental setup | 0.75 | - |
| 5 Results | 2.5 | H1-H6 |
| 6 Discussion and recipe | 1.0 | - |
| 7 Limitations | 0.5 | - |
| 8 Conclusion | 0.25 | - |
| References | (extra) | - |
| Appendix (online) | (extra) | - |
| **Total** | **about 7.5** | |

### Abstract (about 150 words)

- Problem: decision models return probabilities in one forward pass and are used to trigger actions.
- Question: are those probabilities calibrated on your data, and are they stable when the question is phrased differently?
- Approach: perturbation grid (paraphrase x option order) on d1-2B over two public tasks, recalibration with learning curves, cost-based thresholds.
- Result slots: flip rate, calibration transfer gap, labels needed, cost saving. Fill in after the experiment.
- Takeaway: calibrate per question version, ensemble over perturbations, set thresholds from costs.

### 1 Introduction

- What a decision model is and how it differs from a chat model: typed outputs, probabilities, single forward pass, no tokens. d1 as a running example.
- Why calibration matters: a "0.8" only helps if it means 80%, because probabilities drive automatic actions and reported rates.
- The overlooked issue: calibration is measured for one phrasing, but production phrasings and option orders vary. A model can be calibrated on average and still change its answer when the question is reworded.
- Research questions (RQ1 calibration, RQ2 consistency, RQ3 transfer and label budget, RQ4 mitigation and thresholds), contributions, roadmap.

### 2 Background

- Calibration versus discrimination (a model can rank well and be overconfident).
- Metrics in one line each: reliability diagram, ECE, Brier score, log loss.
- Recalibration toolbox: temperature scaling (does not change which option wins), Platt scaling, isotonic regression.
- Consistency in classification-by-prompt: sensitivity to wording and to option order (position bias). Keep to a few references; no vendor benchmark tables.

### 3 Method

- **Perturbation grid:** question paraphrases (6 levels) x option orders (2 for binary, 6 for multi-class). Canonical condition = P0 x O0.
- **Consistency measures:** flip rate versus canonical, pairwise disagreement, unanimity, probability drift, split-half ensemble agreement.
- **Calibration measures:** ECE (10 equal-width bins; top-label for multi-class), Brier, log loss, with cluster-bootstrap intervals.
- **Recalibration:** temperature, Platt, isotonic; learning curves over label budgets 25 to 1,000 with 200 resamples each.
- **Transfer test:** fit on canonical condition, evaluate on others; compare with a calibrator fitted on pooled conditions.
- **Mitigation and decisions:** probability averaging over perturbations; thresholds derived from error costs.

### 4 Experimental setup

- Models: d1-2B (primary). Optional: vLLM-SR d3, AWS Strands Decider 2B, only if interfaces pass the feasibility gate.
- Tasks: A, binary factual check (RAGTruth, QA subset, length-filtered); B, multi-class choice (MASSIVE en-US, scenario level). Optional extensions: C, ordinal rating (HelpSteer2); D, rare positive class (ToxicChat `jailbreaking`).
- Splits: calibration pool and held-out test grouped by source or prompt; a separate dev set for pilots.
- Hardware and pinning: one workstation, model revision, seeds, batch-size invariance check.
- Table 1: datasets (task, label, size after filtering, prevalence, options, conditions).

### 5 Results

Each subsection ends with a one-sentence answer to its hypothesis.

- 5.1 **Raw calibration** (H1): reliability diagrams and ECE per task at the canonical condition.
- 5.2 **Consistency** (H2): flip-rate heatmap (paraphrase x order), share of flips among confident items, variance decomposition.
- 5.3 **Does calibration transfer?** (H3): ECE after calibration at the canonical versus other conditions; pooled-condition calibrator.
- 5.4 **How many labels?** (H4): learning curves, harm rate of isotonic at small n.
- 5.5 **Does ensembling help?** (H5): split-half agreement and Brier, single versus ensemble, before and after calibration.
- 5.6 **Cost-based thresholds** (H6): expected cost grid, derived versus default thresholds, raw versus calibrated probabilities.

### 6 Discussion and practical recipe

- What the results mean: calibration is a property of a question version, not of a model.
- Why recalibration does not fix inconsistency (temperature scaling leaves the winning option unchanged) and what does (ensembling, per-version calibration).
- **Recipe box:** five steps a team can follow (label 100 to 200 items; freeze question wording; measure ECE and Brier on a held-out half; set two cut-offs from costs; re-check on every model or wording change).
- Where the cascade needs a human or an LLM judge: the uncertain band.

### 7 Limitations

- Two tasks and one primary model; results may not transfer to other tasks or vendors.
- Paraphrases are LLM-generated and author-checked, so their style is narrower than real-world variation.
- Contamination of public datasets cannot be ruled out.
- Label noise sets a floor on measurable calibration. Rare-class tails need many labels.
- The cost matrix is illustrative.

### 8 Conclusion

- Direct answers to RQ1 to RQ4 in three or four sentences, the minimal workflow, and open questions (fine-tuning open weights, multimodal inputs, per-language behaviour).

## Figures and tables

| ID | Content | Section |
|---|---|---|
| Fig 1 | Overview: perturbation grid, calibration split, evaluation | 3 |
| Fig 2 | Reliability diagrams, raw versus recalibrated, tasks A and B | 5.1 |
| Fig 3 | Flip-rate heatmap (paraphrase x order) | 5.2 |
| Fig 4 | Variance decomposition of the top-class logit | 5.2 |
| Fig 5 | ECE by condition after canonical calibration | 5.3 |
| Fig 6 | Learning curves (ECE versus labels) with intervals | 5.4 |
| Fig 7 | Single versus ensemble: agreement and Brier | 5.5 |
| Fig 8 | Expected cost by threshold rule across the cost grid | 5.6 |
| Table 1 | Datasets and conditions | 4 |
| Table 2 | Headline results per hypothesis (estimate, interval, verdict) | 5 |

## Style notes

- Lead each Results subsection with the finding, then the evidence.
- Report intervals, not only point estimates. State the number of items and resamples next to every figure.
- Label anything exploratory as exploratory.
- Avoid screenshots of raw examples from datasets with user-generated text.

## Writing order (parallel to the experiment)

1. Weekend 1: Introduction, Background, Method (no results needed).
2. Weekend 2: Results and Discussion after the analyses run; Abstract last.

## Before external release

- Review and validate all content before sharing; the article reflects the authors' work and is not an official company statement.
- Approvals (communications, Legal, Compliance, Data Privacy Officer) and dataset licence review are deferred for now and must be completed before publication.
