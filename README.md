# Decision model calibration: experiments

Code and protocol for the paper on calibration and consistency of decision models.
The study design is in `DOCS/` (`02_hypotheses.md`, `03_experiment_protocol.md`).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). All commands run from the repository root.

```
make setup     # create the environment from uv.lock
make test      # unit tests (no network, no GPU)
make data-b    # task B (MASSIVE): download pinned archive, verify sha256, build splits
make data-a    # task A (RAGTruth): blocked until cleared by DPO and Legal (configs/data.yaml)
make pilot CONFIG=configs/run.yaml   # run one model over the condition grid
```

Datasets are not committed. Each `make data-*` downloads the pinned source, checks its sha256, rebuilds the splits
and verifies them against the committed `data/checksums.sha256`; any difference stops the run.
All randomness derives from the master seed in `configs/seeds.yaml`.

## Layout

See `DOCS/03_experiment_protocol.md` section 13.

## Baseline endpoint

Copy `configs/endpoints.example.yaml` to `configs/endpoints.yaml` (git-ignored) and set the environment variable
named in `api_key_env`. Never write keys into config files.
