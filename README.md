# HM-IBF-HAO

A hyper-heuristic that **designs** island-model metaheuristics, and a horizontal alignment
benchmark to design them for.

The workspace holds two crates:

| Crate | Path | What it is |
|---|---|---|
| `grahf` | `src/` | The framework. Its search space is a *graph of islands*: node weights pick each island's metaheuristic, edge weights pick the migration policy, IRACE tunes every candidate's parameters, and the tuned algorithm is scored on the target problem. |
| `grahf-hao` | `hm-ibf-hao/` | The benchmark — **horizontal alignment optimization** — plus the `hm-ibf-hao` pipeline binary and the Python instance generator. |

## The benchmark

Route a road or railway corridor across a terrain heightmap. Each instance fixes a
*backbone*: a least-cost route between a start and a goal, resampled at equal arc lengths. A
solution is a vector of perpendicular offsets along that backbone; the offset points are
interpolated with an asymmetric clothoid spline (Walton & Meek 2005) and scored as
terrain-weighted length plus a penalty for every sample that turns tighter than the minimum
radius of curvature.

What makes it a hyper-heuristic benchmark rather than another continuous problem: islands run
at **different dimensions**. Dimension `D` places control points at `s_i = i·L/(D+1)` along
the backbone, so every dimension describes the same corridor at a different resolution, and a
migrant has to be resampled — not merely padded — to move between two islands.

The dimensions the islands may use are not picked by hand. Douglas-Peucker compression of
each map's A\* route estimates how many inflection points that route really needs, and the
pool of those estimates becomes `dimensions_allowed` in `params_training.conf` — `[53, 60,
68]` for the three benchmark maps (Italy, Slovenia, Austria).

## Requirements

Docker engine and `git` — that's all you need on the host. Clone the repository first; from
there on, everything runs inside the container built from `Dockerfile`, never on the host:

```bash
git clone https://github.com/Steigner/HM-IBF-HAO.git
cd HM-IBF-HAO
```

macOS/Linux/Git Bash:

```bash
./run.sh              # build the image if needed, start the container, open a shell
./run.sh verify       # run the full verification gate
```

Windows (PowerShell or cmd), no Git Bash or WSL required — note the `.\` prefix, which
PowerShell needs even though cmd.exe doesn't:

```powershell
.\run.bat              # build the image if needed, start the container, open a shell
.\run.bat verify       # run the full verification gate
```

`run.sh <command>` / `run.bat <command>` forwards any command into the container instead of
opening a shell, for example `./run.sh hm-ibf --help` or `.\run.bat cargo test --workspace`.

`run.sh`/`run.bat` themselves must run on the **host** — they talk to the Docker daemon. If
you already have a shell inside the container (e.g. `docker exec -it hm-ibf-hao bash`), don't
call them again from there; just run the underlying command directly, for example
`bash scripts/verify.sh` instead of `./run.sh verify`.

Training shells out to R/IRACE, which lives in the Nix flake, so it needs the Nix-enabled
image. It runs as its own container, so it does not disturb a plain `dev` container you may
already have up:

```bash
HM_IBF_NIX=1 ./run.sh hm-ibf pipeline                # bash
$env:HM_IBF_NIX = "1"; .\run.bat hm-ibf pipeline     # PowerShell
set HM_IBF_NIX=1 & run.bat hm-ibf pipeline           # cmd.exe
```

## The pipeline

**preprocess → train → evaluate**, with no manual steps in between.

Once inside the container, the pipeline is the `hm-ibf` command installed on `PATH` — no
`cargo run -p grahf-hao --bin hm-ibf-hao --` boilerplate and no need to `cd` into the right
directory first:

```bash
# 1. Regenerate the benchmark instances (from hm-ibf-hao/); downloads the terrain itself
python3 -m preprocessing.prepare_instances

# 2. Search for an island graph and tune it
hm-ibf train

# 3. Replay the trained graph over a range of seeds
hm-ibf evaluate

# or both Rust stages at once
hm-ibf pipeline
```

`hm-ibf --help` documents every flag. The instances the benchmark ships with are checked into
`hm-ibf-hao/instances/`, so steps 2 and 3 work out of the box.

Step 1 is only needed to regenerate them. With no `--source`, it fetches the three benchmark
heightmaps (Austria, Italy, Slovenia — roughly 190 MB) from the upstream
[HorizAligns-Hybrid-Optimization](https://github.com/Steigner/HorizAligns-Hybrid-Optimization)
repository into `hm-ibf-hao/external/horizaligns/` and routes each between its published
endpoints. Pass `--source <terrain>.npy` instead to sample a random train/eval split from your
own terrain; `python3 -m preprocessing.fetch_maps` primes the cache without generating
anything.

Algorithm tuning parameters live in [params_training.conf](params_training.conf) and
[params_evaluation.conf](params_evaluation.conf); only run identity and file locations are CLI
flags.

## Smoke test

An end-to-end sanity check of **preprocess → train → evaluate** on synthetic terrain with
reduced budgets — around fifteen minutes rather than the hours a real search takes, most of it
IRACE tuning the initial population. It proves the pipeline is wired together; it is not a
real training run and will not produce a useful result.

```bash
HM_IBF_NIX=1 ./run.sh smoke                 # bash
$env:HM_IBF_NIX = "1"; .\run.bat smoke      # PowerShell
```

It needs the Nix-enabled image, because training calls IRACE. The evaluate stage alone is
covered inside the verification gate by `cargo test -p grahf-hao --test smoke`, which needs
nothing extra.

See [hm-ibf-hao/README.md](hm-ibf-hao/README.md) for the benchmark's design and
[hm-ibf-hao/runbook.md](hm-ibf-hao/runbook.md) for the operational detail of each stage.

## Layout

```text
src/                     grahf framework
  components/            graph operators, island executor, migration transforms
  graph/                 directed graph type, binomial node partitioning
  problems/              algorithm-design problem, IRACE tuning, statistics
tests/                   grahf integration tests
hm-ibf-hao/
  src/                   grahf-hao library + `hm-ibf-hao` binary
  instances/             the three benchmark instances (checked in)
  preprocessing/         instance generation, map catalog and heightmap download
  tests/                 pytest suite + Rust integration and smoke tests
params_training.conf     train/pipeline algorithm tuning parameters (TOML)
params_evaluation.conf   evaluate/pipeline algorithm tuning parameters (TOML)
run.sh, run.bat          host entry point (shell, verify, smoke, arbitrary commands)
scripts/hm-ibf-entrypoint.sh   installed as `hm-ibf` on PATH inside the dev image
scripts/verify.sh        the verification gate
scripts/smoke.sh         the end-to-end smoke gate
llms.txt                 curated repository map for LLMs and coding agents
.claude/skills/          agent skills; `hm-ibf-audit` classifies a tree against HM-IBF,
                         `hm-ibf-retarget` maps the edits to swap in another problem
```

## Using another problem

The pipeline is generic over the problem type, so pointing it at a different optimization
problem (a process profile, a control schedule, a design vector) is a bounded set of edits
rather than a rewrite. The `hm-ibf-retarget` skill enumerates them and checks the result:

```bash
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py .          # the edit surface
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py . --check  # after retargeting
```

The new domain needs a resolution-independent coordinate for its decision variables — the
axis migrants are resampled along. Without one, islands at different dimensions cannot
exchange anything meaningful and a fixed-dimension optimizer is the better tool; the skill's
fit test covers this.

## Development

`AGENTS.md` is the contract: Docker-only execution, Google-style doc comments, unit tests
inline and integration tests in their own tree, 500 lines per file, and a change is done only
once `./run.sh verify` is green end to end.

## License

`GPL-3.0-or-later`. See [LICENSE](LICENSE).
