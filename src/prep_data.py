"""Download pinned dataset versions, build the splits from the master seed, write checksums.

Usage: python src/prep_data.py --task B   (A = RAGTruth QA, B = MASSIVE en-US)

Splits follow protocol section 4. Output: data/task_<x>/{dev,calib,test}.jsonl (not committed).
Each rebuild is verified against the committed data/checksums.sha256 (see verify_or_pin).
"""

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from common import ROOT, derive_seed, load_yaml, sha256_file, write_jsonl

import truststore

truststore.inject_into_ssl()  # use the OS trust store (python.org builds and corporate proxies)

RAW = ROOT / "data" / "raw"


def fetch(url, expected_sha256, name):
    """Download once into data/raw/ and verify the archive checksum (trust on first use if not yet pinned)."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    actual = sha256_file(path)
    if expected_sha256 is None:
        raise SystemExit(f"{name}: no sha256 pinned in configs/data.yaml. Downloaded file has sha256 {actual}; "
                         "pin it and re-run.")
    if actual != expected_sha256:
        raise SystemExit(f"{name}: sha256 mismatch (expected {expected_sha256}, got {actual})")
    return path


def require_fields(row, fields, source):
    missing = [f for f in fields if f not in row]
    if missing:
        raise SystemExit(f"{source}: expected fields {missing} not found; available: {sorted(row)}")


# ---------- Task B: MASSIVE en-US, scenario level (protocol 4.2) ----------

def prep_massive(cfg):
    path = fetch(cfg["url"], cfg["sha256"], Path(cfg["url"]).name)
    with tarfile.open(path) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(cfg["member_suffix"]))
        rows = [json.loads(line) for line in tar.extractfile(member) if line.strip()]
    require_fields(rows[0], ["id", "partition", "scenario", "utt"], "MASSIVE")

    drop = set(cfg.get("drop_scenarios") or [])
    items = {"dev": [], "test": []}
    for r in rows:
        if r["partition"] in items and r["scenario"] not in drop:
            items[r["partition"]].append(
                {"item_id": f"B-{r['id']}", "group_id": f"B-{r['id']}", "text": r["utt"], "label": r["scenario"]}
            )

    # Calibration pool and test both come from the official test split (protocol 4.2).
    official_test = sorted(items["test"], key=lambda x: x["item_id"])
    random.Random(derive_seed("B", "split")).shuffle(official_test)
    n_calib = cfg["calib_size"]
    return {"dev": items["dev"], "calib": official_test[:n_calib], "test": official_test[n_calib:]}


# ---------- Task A: RAGTruth QA subset (protocol 4.1) ----------

def _group_items(items):
    groups = {}
    for it in items:
        groups.setdefault(it["group_id"], []).append(it)
    return groups


def take_groups(groups, group_ids, max_items):
    """Take whole groups in the given order while the total stays at or below max_items."""
    taken, rest, n = [], [], 0
    for g in group_ids:
        if n + len(groups[g]) <= max_items:
            taken.append(g)
            n += len(groups[g])
        else:
            rest.append(g)
    return taken, rest


def ragtruth_splits(items, cfg, seed):
    """Split rule of protocol 4.1. items carry 'official_split' and 'group_id'."""
    rng = random.Random(seed)
    train = [it for it in items if it["official_split"] == "train"]
    official_test = [it for it in items if it["official_split"] == "test"]

    train_groups = _group_items(train)
    train_ids = sorted(train_groups)
    rng.shuffle(train_ids)
    n_dev_groups = round(cfg["dev_share"] * len(train_ids))
    dev_ids = train_ids[:n_dev_groups]

    if len(official_test) >= cfg["min_official_test"]:
        pool = official_test
        rule = "official_test"
    else:
        pool = [it for g in train_ids[n_dev_groups:] for it in train_groups[g]] + official_test
        rule = "resplit"

    pool_groups = _group_items(pool)
    pool_ids = sorted(pool_groups)
    rng.shuffle(pool_ids)
    calib_size = cfg["calib_size"]
    calib_ids, test_ids = take_groups(pool_groups, pool_ids, calib_size)
    if sum(len(pool_groups[g]) for g in test_ids) < cfg["min_test"]:
        calib_size = cfg["calib_size_fallback"]
        calib_ids, test_ids = take_groups(pool_groups, pool_ids, calib_size)

    splits = {
        "dev": [it for g in dev_ids for it in train_groups[g]],
        "calib": [it for g in calib_ids for it in pool_groups[g]],
        "test": [it for g in test_ids for it in pool_groups[g]],
    }
    return splits, {"split_rule": rule, "calib_size_cap": calib_size}


def prep_ragtruth(cfg):
    if not cfg.get("approved"):
        raise SystemExit("RAGTruth contains user-generated text. Set ragtruth.approved: true in configs/data.yaml "
                         "only after the Data Privacy Officer and Legal have cleared it (protocol section 15).")
    base = f"https://raw.githubusercontent.com/{cfg['repo']}/{cfg['commit']}/"
    responses = fetch(base + cfg["responses_path"], cfg["responses_sha256"], "ragtruth_response.jsonl")
    sources = fetch(base + cfg["sources_path"], cfg["sources_sha256"], "ragtruth_source_info.jsonl")
    resp = [json.loads(line) for line in open(responses) if line.strip()]
    src = {s["source_id"]: s for s in (json.loads(line) for line in open(sources) if line.strip())}
    require_fields(resp[0], ["id", "source_id", "model", "labels", "split", "quality", "response"], "RAGTruth response")
    require_fields(next(iter(src.values())), ["source_id", "task_type", "source_info"], "RAGTruth source")

    counts = {"all": len(resp)}
    qa = [r for r in resp if src[r["source_id"]]["task_type"] == cfg["task_type"]]
    counts["qa"] = len(qa)
    qa = [r for r in qa if r["quality"] in cfg["keep_quality"]]
    counts["quality"] = len(qa)

    items = []
    for r in qa:
        info = src[r["source_id"]]["source_info"]
        require_fields(info, ["question", "passages"], "RAGTruth QA source_info")
        items.append({
            "item_id": f"A-{r['id']}", "group_id": f"A-{r['source_id']}", "response_model": r["model"],
            "official_split": r["split"], "question": info["question"], "context": info["passages"],
            "response": r["response"],
            "label": "hallucinated" if r["labels"] else "supported",
        })

    if cfg.get("token_cap"):
        from transformers import AutoTokenizer  # optional dependency, only needed once the cap is set
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], revision=cfg["tokenizer_revision"])
        items = [it for it in items if len(tok("\n".join((it["question"], it["context"], it["response"])))["input_ids"]) <= cfg["token_cap"]]
    counts["token_cap"] = len(items)

    splits, info = ragtruth_splits(items, cfg, derive_seed("A", "split"))
    prevalence = {k: round(sum(it["label"] == "hallucinated" for it in v) / max(len(v), 1), 4) for k, v in splits.items()}
    print(json.dumps({"filter_counts": counts, **info, "prevalence": prevalence}, indent=2))
    return splits


CHECKSUMS = ROOT / "data" / "checksums.sha256"


def verify_or_pin(paths):
    """Compare built splits with the committed data/checksums.sha256.

    A file listed there must match exactly, otherwise the run fails. A file not yet listed is pinned
    (first build); commit data/checksums.sha256 afterwards so that later rebuilds are verified against it.
    """
    pinned = {}
    if CHECKSUMS.exists():
        for line in CHECKSUMS.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            pinned[name] = digest
    mismatched, new = [], []
    for path in paths:
        name, digest = str(path.relative_to(ROOT / "data")), sha256_file(path)
        if name not in pinned:
            pinned[name] = digest
            new.append(name)
        elif pinned[name] != digest:
            mismatched.append(name)
    if mismatched:
        raise SystemExit(f"checksum MISMATCH, the rebuild differs from the pinned splits: {mismatched}")
    if new:
        CHECKSUMS.write_text("".join(f"{pinned[n]}  {n}\n" for n in sorted(pinned)))
        print(f"pinned new checksums (commit data/checksums.sha256): {new}")
    else:
        print("all splits match data/checksums.sha256")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    splits = prep_ragtruth(cfg["ragtruth"]) if args.task == "A" else prep_massive(cfg["massive"])
    paths = []
    for name, rows in splits.items():
        path = ROOT / "data" / f"task_{args.task.lower()}" / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
        print(f"task {args.task} {name}: {len(rows)} items")
    verify_or_pin(paths)


if __name__ == "__main__":
    main()
