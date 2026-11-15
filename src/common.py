"""Shared helpers: config loading, seed derivation, checksums, JSONL I/O, run metadata."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def master_seed():
    return load_yaml(ROOT / "configs" / "seeds.yaml")["master_seed"]


def derive_seed(task, purpose, master=None):
    """Seed rule from protocol 4.4: int(sha256(f"{master}:{task}:{purpose}").hexdigest()[:8], 16)."""
    master = master_seed() if master is None else master
    return int(hashlib.sha256(f"{master}:{task}:{purpose}".encode()).hexdigest()[:8], 16)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(extra=None):
    """Everything needed to tie a predictions file to code, environment and hardware (protocol section 5)."""
    packages = {}
    for name in ("numpy", "pyyaml", "openai", "torch", "transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    lock = ROOT / "uv.lock"
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "lockfile_sha256": sha256_file(lock) if lock.exists() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": _gpu_info(),
    }
    meta.update(extra or {})
    return meta
