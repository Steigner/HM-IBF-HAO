# HM-IBF-HAO — Runbook

Three stages: **preprocess → train → evaluate**. All of them run inside the container built
from the repository's `Dockerfile`; nothing here is meant to run on the host.

For what the benchmark *is*, see [README.md](README.md).

---

## Prerequisites

Every command runs inside the container. Start it once from the repository root, on the
**host** — `run.sh`/`run.bat` talk to the Docker daemon, so don't call them again once you
are already inside; run the underlying command directly instead:

```bash
./run.sh                            # bash
.\run.bat                           # PowerShell / cmd.exe
```

`./run.sh <command>` runs a single command in the container instead of opening a shell, e.g.
`./run.sh hm-ibf --help`.

Training and `pipeline` call IRACE, which needs R from the Nix flake, so they need the
Nix-enabled image instead. It runs as its own container (`hm-ibf-hao-nix`), so it does not
disturb a plain `dev` container you may already have up:

```bash
HM_IBF_NIX=1 ./run.sh               # bash
$env:HM_IBF_NIX = "1"; .\run.bat    # PowerShell
set HM_IBF_NIX=1 & run.bat          # cmd.exe
```

Inside the container the pipeline is the `hm-ibf` command on `PATH` — no
`cargo run -p grahf-hao --bin hm-ibf-hao --` boilerplate. It builds the release binary if
needed and always runs it from `hm-ibf-hao/`, so the relative defaults (`instances/`,
`hao_run/`, `results/`) resolve the same way no matter where you call it from. It also
re-execs itself through `nix develop` for `train` and `pipeline` when Nix is available, so no
manual `nix develop --command …` step is required.

---

## Step 1: Preprocess (optional)

The benchmark instances are checked into `instances/`, so this step is only needed to
regenerate them or to build a different set. It has two modes, selected by `--source`.

### Benchmark mode (no `--source`)

Regenerates the checked-in `instances/`: the three named terrain maps, each routed between
its own published endpoints.

```bash
cd /app/hm-ibf-hao
python3 -m preprocessing.prepare_instances
```

**Needs network access on the first run.** The three source heightmaps (~190 MB) are
downloaded from
[HorizAligns-Hybrid-Optimization](https://github.com/Steigner/HorizAligns-Hybrid-Optimization)
into `external/horizaligns/` and reused afterwards; delete that directory to refetch. To
prime the cache without generating anything:

```bash
python3 -m preprocessing.fetch_maps
python3 -m preprocessing.fetch_maps --maps austria --cache-dir /tmp/maps
```

The map catalog — file name, source shape, downsample resolution, start and goal — lives in
`preprocessing/maps.py` and mirrors the upstream study's own configuration.

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | `instances` | Directory receiving the benchmark instances |
| `--maps` | all three | Regions to generate: `austria`, `italia`, `slovenia` |
| `--cache-dir` | `external/horizaligns` | Where the source heightmaps are cached |
| `--maps-base-url` | upstream raw URL | Where they are downloaded from |

### Sampled mode (`--source`)

Draws random start/goal pairs from one heightmap and splits the routes into a disjoint
train/eval set, for evaluating on unseen routes.

```bash
python3 -m preprocessing.prepare_instances --source <terrain>.npy
```

**What it does:** sample start/goal pairs far enough apart → route each with A\* → bucket the
routes by length and split them into disjoint train and eval sets → resample each raw route
into an equidistant backbone with normals and offset bounds → Douglas-Peucker the raw route to
estimate its natural dimension → write both sets.

**Output:**

```
instances_train/
  route_train_NN_config.json      backbone + cost parameters
  route_train_NN_heightmap.npy    downsampled terrain
  summary.json                    index + dimensions_allowed recommendation
instances_eval/
  route_eval_NN_config.json
  route_eval_NN_heightmap.npy
  summary.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | *unset* | Source terrain heightmap (`.npy`); selects this mode |
| `--train-dir` | `instances_train` | Directory receiving the training instances |
| `--eval-dir` | `instances_eval` | Directory receiving the evaluation instances |
| `--n-train` | `30` | Number of training instances |
| `--n-eval` | `90` | Number of evaluation instances |
| `--resolution` | `200 400` | Shape the source heightmap is downsampled to |
| `--min-distance-fraction` | `0.15` | Smallest start-goal distance, as a fraction of the diagonal |
| `--oversample-factor` | `2` | Candidate pool size, as a multiple of the instances needed |
| `--n-buckets` | `3` | Route-length buckets the candidates are spread over |

### Shared flags

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `42` | Seed of the sampling and splitting RNG |
| `--epsilon` | `1.0` | Douglas-Peucker tolerance; part of the instance name |
| `--backbone-step` | `1.0` | Arc length between backbone points, in pixels |
| `--cutting-plane-factor` | `1.0` | Multiplier on the perpendicular offset bounds |
| `--n-dimensions-allowed` | `5` | Largest number of working dimensions to recommend |
| `--tau` | `0.4` | Clothoid asymmetry recorded in every instance |

The defaults live in `preprocessing/config.py`.

**Afterwards:** the generator logs a line like

```
dimensions_allowed = [53, 60, 68]
```

That is the recommendation the Douglas-Peucker compression of the generated routes produces.
Paste it into the repository root's `params_training.conf`, keep it strictly increasing, and
re-run the verification gate. Regenerating instances invalidates every stored `elitist_*.json`
that was trained against the old dimensions.

---

## Step 2: Train

```bash
hm-ibf train
```

**What it does:** loads every instance in `--instances-dir`, evolves a graph of islands with
GRAHF, tunes each candidate with IRACE, and writes the best graphs to `--run-dir`.

**Output** (in `hao_run/` by default):

```
elitist_0.json      the best island graph
elitist_0.dot       the same graph, for graphviz
elitist_0.params    its IRACE-tuned parameters
run.log             the search log
statistics.json     per-instance median best objective
irace/              IRACE's working directory
```

The randomly generated initial population is written to `--experiments-dir`
(`experiments_hao/` by default), so a run can be reproduced from its seed.

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `42` | Seed of the outer structure search |
| `--experiments-dir` | `experiments_hao` | Directory receiving the initial population |

Everything else the search needs is a *tuning parameter* and lives in
`../params_training.conf` (see **Tuning parameters** below), not on the command line.

---

## Step 3: Evaluate

```bash
hm-ibf evaluate
```

**What it does:** rebuilds the trained elitist from `--run-dir`, replays it on every instance
in `--eval-instances-dir` for each seed, and writes one result folder per `(instance, seed)`
pair plus an aggregated summary.

Every run is cross-checked before it is exported: the best individual must agree with the
run's own best value, and its dimension must be one of `dimensions_allowed`. A mismatch fails
the stage rather than being written out.

**Output:**

```
results/<instance>_GRAHF_seed<seed>_<tag>/
  results.json      the exported offsets, their value and the run's metadata
  best_value.csv    best value against evaluation count
  avg_value.csv     mean island objective value against evaluation count
  run.log           the raw MAHF log
results_hao.csv     mean and sample standard deviation per instance
```

`results.json` uses export schema `1`: one `x`, one `solution_dim`, one `natural_dimension`.

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--first-seed` | `42` | First evaluation seed |
| `--num-seeds` | `15` | Number of consecutive seeds |
| `--elitist` | `elitist_0` | Base name of the trained artefacts |
| `--results-dir` | `results` | Directory receiving the per-run folders |
| `--summary-csv` | `results_hao.csv` | Aggregated summary |

---

## Both stages at once

```bash
hm-ibf pipeline
```

Runs `train` and then `evaluate`, sharing every global flag between them.

## Global flags

These apply to all three subcommands:

| Flag | Default | Meaning |
|---|---|---|
| `--jobs` | available parallelism | Worker threads |
| `--instances-dir` | `instances` | Instances the search trains on |
| `--eval-instances-dir` | `--instances-dir` | Instances the trained algorithm is evaluated on |
| `--run-dir` | `hao_run` | Trained graph and its tuned parameters |
| `--training-params` | `../params_training.conf` | Tuning parameters of the search |
| `--evaluation-params` | `../params_evaluation.conf` | Tuning parameters of the evaluate stage |

Point `--eval-instances-dir` at a held-out directory to evaluate on unseen routes:

```bash
hm-ibf pipeline --instances-dir instances_train --eval-instances-dir instances_eval
```

`RUST_LOG` selects the log level (`info` inside the container).

---

## Tuning parameters

Run identity and file locations are CLI flags; the *algorithm* tuning parameters live in two
TOML files at the repository root, so they can be edited without recompiling. Both are read
at startup and validated.

`params_training.conf` (read by `train` and `pipeline`):

| Setting | Shipped | Meaning |
|---|---|---|
| `epsilon` | `1e-8` | Termination tolerance against the inner problem's known optimum |
| `max_evaluations` | `100_000` | Evaluation budget of a single metaheuristic run |
| `num_repetitions` | `2` | Repetitions per instance when scoring a candidate graph |
| `num_tuning_repetitions` | `3` | Repetitions per instance during IRACE tuning |
| `num_tuning_experiments` | `10` | Minimum IRACE experiments per candidate graph |
| `num_iterations` | `4` | Generations of the outer structure search |
| `max_island_iterations` | `30` | Upper bound IRACE may assign to an island's iterations |
| `max_island_population` | `15` | Upper bound IRACE may assign to an island's population |
| `dimensions_allowed` | `[53, 60, 68]` | Working dimensions the islands may run at |
| `[grahf]` | see file | Hyper-parameters of the outer graph-level GA |

`max_evaluations` defines the benchmark, so training and evaluation deliberately share it.
`dimensions_allowed` must be non-empty and strictly increasing; it is the Douglas-Peucker
recommendation of the instance pool, so regenerate and repaste it whenever the instances
change — and retrain, because a stored elitist's tuned `dimension` may no longer be in the
set.

`params_evaluation.conf` (read by `evaluate` and `pipeline`):

| Setting | Shipped | Meaning |
|---|---|---|
| `max_iterations` | `100` | Island iteration bound when rebuilding the trained graph |
| `max_population_size` | `128` | Island population bound when rebuilding it |
| `best_value_tolerance` | `1e-9` | Tolerance of the exported-value cross-check |

Evaluation replays the stored, IRACE-tuned parameters instead of sampling new ones, so the
first two only shape the parameter space and never change a run's outcome.

---

## Verification

```bash
./run.sh verify
```

Runs `cargo fmt --check`, `cargo clippy -- -D warnings` and `cargo test` across the whole
workspace, then `ruff check`, `ruff format --check` and the full `pytest` suite. Nothing is
finished until all of it is green.

## Smoke test

`cargo test` already runs `tests/smoke.rs`, which drives the whole **evaluate** stage on a
synthetic instance. The full three-stage check needs IRACE, so it is a separate gate:

```bash
HM_IBF_NIX=1 ./run.sh smoke                 # bash
$env:HM_IBF_NIX = "1"; .\run.bat smoke      # PowerShell
```

It generates synthetic terrain, preprocesses a tiny train/eval split, writes a minimal
tuning configuration from the generator's own recommendation, runs `hm-ibf pipeline` against
it and checks every artefact. Everything lands in a scratch directory under `/tmp`, which is
removed afterwards; pass a directory to keep it. Budgets are tiny, so the objective values it
reports mean nothing — it proves the wiring, not the algorithm.

Expect around fifteen minutes. Nearly all of it is IRACE tuning the search's fifteen initial
graphs, which is why the evaluation budget cannot make it much faster.

---

## Troubleshooting

**`no instances found in <dir>`**
The directory holds no `<name>_config.json` with a matching `<name>_heightmap.npy`. Check the
path, or run step 1.

**`no terrain heightmap at <path>`**
`--source` points at a file that does not exist. Drop the flag to regenerate the named
benchmark maps instead, or use the checked-in `instances/`.

**`failed to download ...` from `preprocessing.fetch_maps`**
Benchmark mode needs network access on its first run. Prime `external/horizaligns/` on a
connected machine, or use sampled mode with `--source`.

**`Path 'flake.nix' in the repository "/app" is not tracked by Git`**
Nix flakes only see git-tracked files, so `train`/`pipeline` cannot start in a fresh clone
where the flake is still untracked. Commit it, or `git add -N flake.nix flake.lock`.

**`ModuleNotFoundError` from `irace-rs` during `train`**
The stage is running outside the Nix shell. Use the Nix-enabled container
(`HM_IBF_NIX=1 ./run.sh`); `hm-ibf` then wraps `train` and `pipeline` in `nix develop` itself.

**`stored graph uses unknown island type N`**
The stored elitist was trained against a different island set. Retrain, or restore the island
order the graph was trained with.

**`global best solution ... has unsupported dimension D`**
`dimensions_allowed` no longer matches the graph that was trained. Retrain after changing it.

**`invalid dimensions_allowed in ...`**
The list is empty, not strictly increasing, or contains a zero. `dimension()` reads its last
entry as the maximum, so the order is load-bearing.

**A component sees the wrong dimension**
It is calling `problem.dimension()`, which always reports the maximum allowed dimension. Read
`solution.len()` or `IslandDimension` from island state instead; see `islands/safe_*.rs`.

---

## Notes

- Allowed island dimensions: `dimensions_allowed` in `params_training.conf`, shipped as
  `53, 60, 68` — the Douglas-Peucker dimensions of the three benchmark maps.
- Island node weights: `0 = de`, `1 = es`, `2 = ls`, `3 = sa`, `4 = rs`, `5 = archive`.
  Particle swarm is deliberately absent; see [README.md](README.md).
- Migration transforms: `arc_linear`, `arc_pchip`, `arc_akima`, `arc_clamped_cubic`,
  `arc_total_variation`.
- Training uses a single seed by default; evaluation uses 15.
- Benchmark instances: `austria_eps1.0`, `italia_eps1.0`, `slovenia_eps1.0`.
