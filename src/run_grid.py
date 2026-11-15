"""Run one model over all conditions (paraphrase x option order) of one task split.

Usage: python src/run_grid.py --config configs/run.yaml

Inputs:  data/task_<x>/<split>.jsonl, prompts/task_<x>/{paraphrases.yaml, options.yaml, item_template.txt}
Outputs: results/predictions/<model>/task_<x>_<split>.jsonl  (long format, protocol section 5)
         results/raw/<model>/task_<x>_<split>.jsonl          (raw model outputs)
         results/predictions/<model>/task_<x>_<split>.meta.json
"""

import argparse
import json
import time
from datetime import datetime, timezone

from backends import BACKENDS
from common import ROOT, derive_seed, load_yaml, read_jsonl, run_metadata, write_jsonl
from conditions import option_orders, to_original, top_and_margin


def load_task(task):
    pdir = ROOT / "prompts" / f"task_{task.lower()}"
    paraphrases = load_yaml(pdir / "paraphrases.yaml")
    options = load_yaml(pdir / "options.yaml")  # list of {id, text}, in O0 order
    template = (pdir / "item_template.txt").read_text()
    return paraphrases, options, template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    task, split, model = cfg["task"], cfg["split"], cfg["model"]

    paraphrases, options, template = load_task(task)
    option_ids = [o["id"] for o in options]
    orders = option_orders(len(options), derive_seed(task, "order_O5"))
    p_ids = cfg.get("paraphrases") or sorted(paraphrases)
    o_ids = cfg.get("orders") or sorted(orders)

    items = read_jsonl(ROOT / "data" / f"task_{task.lower()}" / f"{split}.jsonl")
    if cfg.get("limit"):
        items = items[: cfg["limit"]]

    backend_cfg = load_yaml(ROOT / cfg["endpoint"]) if cfg.get("endpoint") else cfg
    backend = BACKENDS[cfg["backend"]](backend_cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_size = cfg.get("batch_size", 32)

    rows, raw_rows = [], []
    for p_id in p_ids:
        for o_id in o_ids:
            order = orders[o_id]
            shown = [options[i]["text"] for i in order]
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                requests = [{"question": paraphrases[p_id], "item": template.format(**it), "options": shown}
                            for it in batch]
                t0 = time.perf_counter()
                outputs = backend.predict(requests)
                latency_ms = 1000 * (time.perf_counter() - t0) / len(batch)
                for it, out in zip(batch, outputs):
                    probs = to_original(out["probs"], order)
                    top, margin = top_and_margin(probs)
                    key = {"item_id": it["item_id"], "paraphrase_id": p_id, "order_id": o_id}
                    rows.append({
                        **key, "model": model, "revision": cfg.get("revision"), "task": task, "split": split,
                        "group_id": it["group_id"], "option_order": order, "option_ids": option_ids,
                        "probs_by_original_option": probs, "top_option": option_ids[top], "margin": margin,
                        "label": it["label"], "n_tokens": out.get("n_tokens"), "latency_ms": latency_ms,
                        "missing_mass": out.get("missing_mass"), "flagged": out.get("flagged", False),
                        "run_id": run_id,
                    })
                    raw_rows.append({**key, "run_id": run_id, "raw": out.get("raw")})
            print(f"{model} task {task} {split}: {p_id} {o_id} done ({len(items)} items)")

    name = f"task_{task.lower()}_{split}"
    out_dir = ROOT / "results" / "predictions" / model
    write_jsonl(out_dir / f"{name}.jsonl", rows)
    write_jsonl(ROOT / "results" / "raw" / model / f"{name}.jsonl", raw_rows)
    meta = run_metadata({"run_id": run_id, "config": cfg, "n_items": len(items), "orders": orders})
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
