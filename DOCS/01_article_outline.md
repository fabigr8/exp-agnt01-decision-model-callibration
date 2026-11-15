# Article outline (IMRAD-style, 6-8 pages, light academic tone)

Status: draft v0.2, 13 Nov 2026 (aligned with `02` v2.0 and `03` v2.0). Hypothesis IDs (H1-H6) refer to `02_hypotheses.md`; experiment details are in `03_experiment_protocol.md`.

## Working title

**Can You Trust the Probabilities? Calibration and Consistency of Decision Models**

Alternatives: "A 0.8 Is Only Useful If It Means 80%" / "Same Question, Different Answer: Recalibrating Decision Models".

## Positioning

- **Audience:** technical practitioners and applied researchers who use or evaluate decision models (typed answers with probabilities, no generated text).
- **Tone:** ICML-style structure, plain language. Short sentences, one idea per paragraph, no jargon without a one-line gloss.
- **Scope rule:** every claim in Results comes from our own experiment. External benchmark numbers and vendor claims are not presented as findings.
- **Naming rule:** a model is called inconsistent only where we measured it.
- **Conclusion rule:** no finding is written before the analysis runs. Text that depends on a result stays a conditional placeholder `[if Hx supported: ... / else: ...]` until then.

## Contributions (to be claimed in the Introduction)

1. A study of **decision models**, which return typed probabilities in a single forward pass: are those probabilities calibrated, and are they stable under rewording and option reordering? Run on a preregistered perturbation grid; confirmatory tests on d1-2B, the same statistics for the other decision models that expose probabilities, and an LLM logprob baseline for reference.
2. Evidence on whether a calibrator fitted under one wording transfers to other wordings, including wordings it has never seen.
3. Label-budget guidance for recalibration (learning curves for temperature, Platt and isotonic).
4. Whether calibration pays off in cost-based decisions, compared with tuning thresholds directly.

The perturbation protocol itself builds on prior work on prompt sensitivity and position bias. It is presented as method, not as a contribution.

## Section plan and page budget

| Section | Pages | Hypotheses |
|---|---|---|
| Abstract | 0.2 | all |
| 1 Introduction | 0.75 | motivation |
| 2 Background and related work | 0.6 | - |
| 3 Method | 1.25 | - |
| 4 Experimental setup | 0.75 | - |
| 5 Results | 2.5 | H1-H6 |
| 6 Discussion and recipe | 0.9 | - |
| 7 Limitations | 0.5 | - |
| 8 Conclusion | 0.25 | - |
| References | (extra) | - |
| Appendix (online) | (extra) | - |
| **Total** | **about 7.7** | |

### Abstract (about 150 words)

- Problem: decision models return probabilities in one forward pass and are used to trigger actions.
- Question: are those probabilities calibrated on your data, and are they stable when the question is phrased differently?
- Approach: a preregistered perturbation grid (paraphrase × option order) on d1-2B (confirmatory) and further decision models (descriptive) over two public tasks, with an LLM logprob baseline; recalibration with learning curves; cost-based thresholds.
- Result slots: log-loss gain from recalibration, disagreement rate, transfer gap, labels needed, cost saving. Fill in after the experiment.
- Takeaway: `[written after the results; one sentence per supported or refuted hypothesis]`.

### 1 Introduction

- What a decision model is and how it differs from a chat model: typed outputs, probabilities, single forward pass, no tokens. d1 as a running example.
- Why calibration matters: a "0.8" only helps if it means 80%, because probabilities drive automatic actions and reported rates.
- The open question: calibration is usually measured for one phrasing, but production phrasings and option orders vary. Whether a decision model is affected the way prompted LLMs are is an empirical question.
- Research questions (RQ1 calibration, RQ2 consistency, RQ3 transfer and label budget, RQ4 mitigation and thresholds), contributions, preregistration (OSF link), roadmap.

### 2 Background and related work

- Calibration versus discrimination: a model can rank well and still be overconfident.
- Metrics, one line each: Brier score and log loss (primary), reliability diagram and ECE (descriptive; note its binning bias).
- Recalibration toolbox: temperature scaling (does not change which option wins), Platt scaling, isotonic regression (Guo et al. 2017; Niculescu-Mizil & Caruana 2005).
- Prompt sensitivity and position bias in LLM classification (Sclar et al. 2023; Zheng et al. 2023), and calibrating before use (Zhao et al. 2021). State what is different here: decision models with typed probabilities, transfer of calibrators, label budget, costs.
- Keep it to about 0.6 pages; no vendor benchmark tables. Verify every reference before citing.

### 3 Method

- **Perturbation grid:** question paraphrases (6 levels) × option orders (2 for binary, 6 for multi-class). Canonical condition = P0 × O0.
- **Consistency measures:** pairwise disagreement with a noise-floor rule, flip rate versus canonical, unanimity, probability drift, confident flips.
- **Calibration measures:** Brier score and log loss (primary); reliability diagrams and debiased ECE (descriptive); paired cluster-bootstrap intervals.
- **Recalibration:** temperature, Platt, isotonic; learning curves over label budgets 25 to 400.
- **Transfer test:** a transfer matrix (fit on each wording, evaluate on every other) and a leave-one-wording-out pooled calibrator.
- **Mitigation and decisions:** probability averaging over perturbations (non-inferiority on Brier; disagreement versus number of calls); thresholds derived from error costs, compared with raw probabilities and with direct threshold tuning.
- **Inference:** one primary test per hypothesis, Holm-corrected; three-way verdicts (supported, refuted, inconclusive).

### 4 Experimental setup

- Models: d1-2B (confirmatory); vLLM-SR d3 and AWS Strands Decider 2B (descriptive, full grid, if their interfaces pass the feasibility gate; exclusions reported with reasons); an LLM logprob baseline served through an OpenAI-compatible endpoint (descriptive).
- Tasks: A, binary factual check (RAGTruth, QA subset, length-filtered); B, multi-class choice (MASSIVE en-US, scenario level). Optional extensions: C, ordinal rating (HelpSteer2); D, rare positive class (ToxicChat `jailbreaking`).
- Splits: calibration pool and held-out test drawn from the same official split, grouped by source; a separate dev set for pilots.
- Preregistration, hardware and pinning: OSF link, one workstation, model revisions, lockfile, seeds, determinism check.
- Table 1: datasets (task, label, size after filtering, prevalence, options, conditions).

### 5 Results

Each subsection opens with the verdict for its primary test (supported, refuted or inconclusive), then the evidence, then secondary results labelled as such. The other decision models and the baseline are shown alongside d1-2B wherever space allows (descriptive estimates with intervals).

- 5.1 **Raw calibration** (H1): reliability diagrams (raw) and log-loss gain from recalibration per task.
- 5.2 **Consistency** (H2): disagreement heatmap (paraphrase × order); confident flips.
- 5.3 **Does calibration transfer?** (H3): transfer matrix; leave-one-wording-out calibrator.
- 5.4 **How many labels?** (H4): learning curves; raw versus recalibrated reliability diagrams; isotonic harm at small n (replication).
- 5.5 **Does ensembling hurt calibration?** (H5): Brier, single versus ensemble; disagreement versus number of calls.
- 5.6 **Cost-based thresholds** (H6): expected cost, calibrated versus raw versus directly tuned thresholds across the cost grid.

### 6 Discussion and practical recipe

- What the results mean: `[if H3 supported: calibration is a property of a question version, not of a model / else: a calibrator fitted on one wording transfers here]`.
- Recalibration and inconsistency: temperature scaling leaves the winning option unchanged, so it cannot fix flips. `[if H5 supported: ensembling over perturbations can, at no calibration cost]`.
- **Recipe box:** steps a team can follow. Label budget `[from H4: n where the learning curve flattens]`; freeze the question wording; measure Brier and log loss on a held-out half; set cut-offs from costs `[or tune directly, if H6 key secondary shows no benefit from calibration]`; re-check on every model or wording change.
- The uncertain band: items between the two cost-derived cut-offs go to a human reviewer (or an LLM judge). This two-stage setup is what we call a cascade.

### 7 Limitations

- Two tasks and one confirmatory model; the other decision models are descriptive, so statements about decision models in general rest on few models. The baseline is a reference point, not a ranking.
- Baseline probabilities come from label-token logprobs through an endpoint; missing probability mass and endpoint pinning are reported. `[if a hosted API was used: logprobs cannot be pinned]`.
- Paraphrases are LLM-generated and author-checked, so their style is narrower than real-world variation.
- Contamination of public datasets cannot be ruled out; pools are drawn from the same split to keep it equal.
- Label noise sets a floor on measurable calibration. Rare-class tails need many labels.
- The cost matrix is illustrative.

### 8 Conclusion

- Direct answers to RQ1 to RQ4 in three or four sentences, the minimal workflow, and open questions (fine-tuning open weights, multimodal inputs, per-language behaviour).

## Figures and tables

| ID | Content | Section |
|---|---|---|
| Fig 1 | Overview: perturbation grid, calibration split, evaluation | 3 |
| Fig 2 | Raw reliability diagrams, tasks A and B (all decision models and baseline) | 5.1 |
| Fig 3 | Disagreement heatmap (paraphrase × order) | 5.2 |
| Fig 4 | Transfer matrix (log loss), paraphrases, task B | 5.3 |
| Fig 5 | Learning curves (log loss versus labels) with intervals; inset: raw versus recalibrated reliability | 5.4 |
| Fig 6 | Expected cost by threshold rule across the cost grid | 5.6 |
| Table 1 | Datasets and conditions | 4 |
| Table 2 | Primary tests: estimate, interval, Holm-adjusted p, verdict (d1-2B), with descriptive estimates for the other decision models and the baseline | 5 |
| App. A | Variance decomposition (E2) | appendix |
| App. B | Single versus ensemble Brier; disagreement versus number of calls (H5) | appendix |
| App. C | Secondary tests, debiased ECE tables, power simulation | appendix |

## Style notes

- Lead each Results subsection with the verdict, then the evidence.
- Report intervals, not only point estimates. State the number of items and resamples next to every figure.
- Label anything secondary or exploratory as such.
- Avoid screenshots of raw examples from datasets with user-generated text.

## Writing order (parallel to the experiment)

1. Before the freeze: Introduction, Background, Method (no results needed).
2. After the analyses: Results and Discussion (replace the placeholders); Abstract last.

## Before external release

- Review and validate all content before sharing; the article reflects the authors' work and is not an official company statement.
- Approvals (communications, Legal, Compliance, Data Privacy Officer) and dataset licence review must be completed before publication; early contact is recommended (`03` section 15).
