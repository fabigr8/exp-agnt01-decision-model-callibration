# Prompts (frozen at the freeze; protocol 4.5)

One folder per task: `task_a/`, `task_b/`. Each holds:

- `paraphrases.yaml`: `P0: "..."` to `P5: "..."`. P0 is written by the authors from the task definition, before any model run. P1-P5 are selected per protocol 4.5.
- `paraphrase_candidates.yaml`: all LLM-generated candidates, with discards and reasons.
- `paraphrase_generation.yaml`: LLM, version, prompt, sampling settings, date.
- `options.yaml`: list of `{id, text}` in O0 order. `id` must match the `label` values in `data/task_<x>/*.jsonl`
  (task A: `hallucinated`, `supported`; task B: MASSIVE scenario names). Task B: O0 is alphabetical by `text`.
- `item_template.txt`: Python format string that renders one item from its data fields
  (task A: `{context}`, `{response}`; task B: `{text}`).

The option and item formats for d1-2B are confirmed in Phase 0 ("Input format" check).
