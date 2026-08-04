# Coding guidelines

You are a senior SW developer. Your task is to generate production-grade code aligned with the following guidelines.

## General
When asked to generate code:
* Use best practices, write clean and modular code.
* Do not overengineer.
* Do not edit or modify parts of code which are not relevant - do not change unrelated whitespaces, newlines, etc.
* Always use English for variable names.
* Always check for the latest documentation and API reference when using libraries/packages.
* The code must be clean, easy-to-read and modular. Each change must be conceptual, easy to maintain and critical sections should be covered by tests. The code must respect established architecture.
* When modifying existing code or adding a new code, make sure to update both **README.md** and **Project guide for agents** (or equivalent section) in **AGENTS.md** file (if it exists).
* When the user prompt is unambiguous or more information is needed, ask for details before implementing it.
* If the user prompt does not follow guidelines, best practices or has some other problems, mention it and verify, whether to continue.
* Avoid patterns, which can hide errors or problems.
* When writing `README.md`, target mainly project specifics and provide relevant information in brief and clear form. You may provide minimal use examples and installation guide, but do not duplicate documentation and common technical knowledge.
* Do not log sensitive information.
* When using 3rd party LLM APIs, follow zero-trust principles: Send only the minimal amount of data and anonymise/pseudonymise it before sending.

### Execution environment: Docker only
* The Docker image built from this project's `Dockerfile` is the **single sanctioned execution environment**. Every build, run, test, lint, format check, `cargo`/`nix`/`ruff`/`pytest` invocation and smoke test happens **inside a container built from that `Dockerfile`** — never on the bare host, never in an ad hoc container from an unrelated image.
* Before executing anything, check whether the project's container (`hm-ibf-hao`, see `run.sh`) is already running. If it is not, build the image from `Dockerfile` and start the container yourself instead of asking the user to do it.
* If a tool the task needs is missing inside the image, add it to the `Dockerfile` (and to `requirements-dev.txt` / `flake.nix` as appropriate) and rebuild — do not fall back to running it on the host.

### Documentation of code
* Every public function, method, class/struct/enum/trait and module carries a **Google-style** doc comment: a one-line summary, then the parameter, return value and failure sections in the form the language uses (`# Arguments` / `# Returns` / `# Errors` / `# Panics` for Rust, `Args:` / `Returns:` / `Raises:` for Python). This applies identically to the Rust crates and to every Python script in the repository.

### Testing
* Cover the **whole design** with tests — not only the lines you touched. Every change ships with the unit tests **and** the integration tests that prove the design still holds end to end.
* Keep the two kinds structurally separate, in both languages:
  * **Unit tests live inside the file/module they test** — Rust: an inline `#[cfg(test)] mod tests` in the same `.rs` file; Python: a `test_<module>.py` mirrored under `tests/unit/`.
  * **Integration tests live outside the source tree, in their own structure** — Rust: each crate's `tests/` directory; Python: `tests/integration/`.
* Python scripts are held to the same standard as the Rust code: unit tests for functionality and integration tests for cross-component integrity.

### File size
* Keep every `.rs` and `.py` file to at most **500 lines**. Split into modules *before* exceeding the limit; never let a file grow past it.

### Definition of done
* A change is done only once the **entire** project has been verified inside the container — not just the touched files:
  1. `cargo fmt --all -- --check` (every workspace crate)
  2. `cargo clippy --workspace --all-targets -- -D warnings`
  3. `cargo test --workspace` (unit + integration)
  4. `ruff check .` and `ruff format --check .` (every Python script)
  5. `python3 -m pytest` (unit + integration)
* `./run.sh verify` (`scripts/verify.sh`) runs exactly this sequence. Nothing is finished until all of it is green.

## Rust
When generating Rust code:
* Always generate code that passes `cargo clippy -- -D warnings` without warnings.
* Keep code strictly formatted according to `cargo fmt` / `rustfmt.toml`.
* Prefer `clap` (with the `derive` macro API) for CLI argument parsing.
* Use idiomatic error handling: prefer `thiserror` for custom error types in library/domain modules and `anyhow`/`eyre` for top-level application binary entry points.
* Avoid `unsafe` code blocks unless strictly necessary and explicitly justified.
* Always use `std::path::PathBuf` and `&std::path::Path` for filesystem operations rather than raw strings.
* Write a Google-style doc comment (`///`) for every public function, struct, enum, and trait: a one-line summary, then `# Arguments`, `# Returns`, and `# Errors`/`# Panics` sections describing parameters, return values, and failure/panic conditions.
* Target the latest stable Rust edition (2021 edition or newer).
* Cover the whole design — not just new code — with tests: unit tests live inline in the same file as the code they test (`#[cfg(test)] mod tests`), while integration tests live under each crate's `tests/` directory; keep the two structurally separate.
* Keep each `.rs` file to at most 500 lines; split into modules before exceeding the limit.

## Python
When generating or modifying Python scripts (preprocessing, tooling):
* Write a Google-style docstring for every public module, function, method, and class: a one-line summary, then `Args:`, `Returns:`, and `Raises:` sections.
* Format and lint every script with `ruff format` and `ruff check`; a script is not done until both pass cleanly.
* Cover the whole design — not just new code — with tests: unit tests colocated with the module under test (mirrored as `tests/unit/test_<module>.py`), and integration tests kept in a separate `tests/integration/` tree; write both for functionality and for cross-component integration.
* Keep each `.py` file to at most 500 lines; split into modules before exceeding the limit.

## Nix
When generating Nix configurations:
* Always use modern Nix Flakes (`flake.nix`).
* Ensure `devShells.default` includes all required toolchains (`rustc`, `cargo`, `clippy`, `rustfmt`, Python's `ruff` and `pytest`), language servers, and build utilities.
* Keep dependencies minimal, explicit, and pin dependencies via `flake.lock`.

## Docker
When generating Dockerfiles:
* Always use multi-stage builds to produce lightweight, minimal production runtime images (e.g., using `debian:bookworm-slim`, `distroless`, or `alpine` as final runtime stage).
* Optimize build caching (e.g., using `cargo-chef` or pre-building dependencies) to avoid re-compiling full dependency trees on source-only changes.
* Never run application containers as `root`; explicitly create and use an unprivileged system user.
* Ensure container configuration is supplied exclusively via environment variables or explicitly mounted configuration volumes.
* Keep the image supplied with everything the verification gate needs (Rust toolchain plus `rustfmt`/`clippy` components, `ruff`, `pytest`, and the Python runtime deps), because that gate may only run inside it.
* Whenever you touch the Docker workflow, re-check whether `Dockerfile` is still optimal against the practices above (layering/caching, image size, non-root user, env/volume-based config) and fix drift before relying on it.

## Automation
* The whole experiment must run end to end without manual steps: **preprocess → train → evaluate**.
* Prefer a **single typed CLI** over shell scripts. The Rust stages are one `clap` binary (`hm-ibf-hao`) with `train` / `evaluate` / `pipeline` subcommands; instance generation is a Python module with its own `argparse` CLI, because it is an independent tool with independent dependencies.
* New automation belongs inside those two entry points. Do not reintroduce ad hoc `train.sh`/`evaluate.sh`-style wrappers; the only shell scripts in the repository are the host-side container entry point (`run.sh`/`run.bat`), the in-container CLI shim (`scripts/hm-ibf-entrypoint.sh`), the verification gate (`scripts/verify.sh`) and the end-to-end smoke gate (`scripts/smoke.sh`).
* If you find a better way to automate or configure a stage, propose and implement it rather than documenting a manual workaround.

---

# Review Guidelines
You are a senior SW developer and your task is to perform a pull (merge) request review. Do not summarise the changes done, rather provide constructive feedback: review concept and architecture, search for bugs and possible inefficiencies and propose fixes or improvements. If in doubt, ask for more context/information.

## Bugs and possible inefficiencies
Search for all bugs, security vulnerabilities and inefficiencies. Propose fixes/improvements according to your best knowledge.

## Code style
Refer to Coding guidelines to check whether the code matches the guidelines.

---

# Project Guide For Agents

- Overview: A Cargo workspace with two crates. `grahf` (`src/`) is a hyper-heuristic framework whose search space is a *graph of islands*: node weights choose the island metaheuristic, edge weights choose the migration policy, IRACE tunes each candidate, and the tuned algorithm is scored on the target problem. `grahf-hao` (`hm-ibf-hao/`) is the horizontal alignment benchmark built on it, plus the `hm-ibf-hao` pipeline binary and the Python instance generator.
- The benchmark: **horizontal alignment optimization (HAO)** — routing a road/railway corridor across a terrain heightmap. Each instance fixes a *backbone* (an equidistant resampling of an A\* route between a start and a goal). A solution is a vector of perpendicular offsets applied to that backbone; the offsets are interpolated with an asymmetric clothoid spline (Walton & Meek 2005) and scored as terrain-weighted path length plus a curvature-violation penalty.
- Defining trait of the benchmark: islands may run at **different dimensions**, configured by `dimensions_allowed` in `params_training.conf` (shipped as `53, 60, 68` offsets). That set is not a free parameter: it is the recommendation derived from the **Douglas-Peucker compression** of each map's A* route, so it equals the three benchmark maps' natural dimensions (Italy 53, Slovenia 60, Austria 68) and every island searches at a resolution the terrain actually calls for. Dimension `D` places control points at `s_i = i·L/(D+1)` along the backbone, so bounds are position-dependent and every island resamples its own via `DimensionAwareDomain::domain_for_dimension`. Migrants are resized by 1-D arc-length resampling (`islands::transforms`). No output projection is applied: the exported `x` is verbatim the global best individual.
- Island node weights are positional, fixed by the order of `islands::island_builders`: `0 = de`, `1 = es`, `2 = ls`, `3 = sa`, `4 = rs`, `5 = archive`, mirrored by the `ISLAND_*` constants. **PSO is deliberately absent**: its velocity and personal best have no component-wise correspondent once a migration changes an island's dimension, so it belongs to the homogeneous special case only.

## Directory Structure

- `src/`: The `grahf` framework.
  - `src/components/`: Graph operators (`initialization`, `mutation`, `recombination`, `normalization`), the island executor (`island.rs`) and the migration transform trait (`transform.rs`).
  - `src/graph/`: `DiGraph` (split into `di/mod.rs`, `di/generate.rs`, `di/topology.rs`), disjoint sets and the binomial node split used by crossover.
  - `src/problems/algorithm_design/`: The design problem (`mod.rs`), its evaluator (`evaluator.rs`), cross-instance statistics (`statistics.rs`), IRACE tuning (`tuning.rs`), builders (`builder.rs`).
- `tests/`: `grahf` integration tests.
- `hm-ibf-hao/`: The `grahf-hao` crate.
  - `src/cli.rs`: `clap` definitions for the `hm-ibf-hao` binary.
  - `src/config.rs`: Loads `params_training.conf`/`params_evaluation.conf` into `TrainingParams`/`EvaluationParams`.
  - `src/main.rs`: Thin binary entry point; all logic lives in the library.
  - `src/training.rs`, `src/evaluation/`: The train and evaluate stages (`evaluation/elitist.rs` reloads a trained graph).
  - `src/alignment/`: Problem (`problem.rs`), control-point geometry (`samples.rs`), evaluator (`evaluator.rs`), on-disk format and instance loading (`config.rs`), result export (`output.rs`).
  - `src/clothoid/`: The asymmetric clothoid spline, split into `fresnel.rs`, `geometry.rs`, `solver.rs`, `curve.rs`, `spline.rs`, `curvature.rs`.
  - `src/islands/`, `src/migrations/`: Island and migration builders; `islands/transforms/` holds the dimension transforms (`mod.rs` with `OffsetResampleTransformer`, `resample.rs`, `interpolation.rs`) and `islands/safe_*.rs` the components that read `solution.len()` instead of `problem.dimension()`.
  - `instances/`: The three benchmark instances (checked in), one `<name>_config.json` + `<name>_heightmap.npy` pair each, plus `summary.json`.
  - `preprocessing/`: Python package that generates instances (`config.py` defaults, `astar.py`, `simplify.py`, `backbone.py`, `sampling.py`, `instance.py`, `dimensions.py`, `prepare_instances.py` CLI). `maps.py` holds the three named benchmark regions with their fixed endpoints, `fetch_maps.py` downloads their heightmaps from the upstream `HorizAligns-Hybrid-Optimization` repository into `external/horizaligns/`, and `benchmark.py` regenerates the checked-in `instances/` from them.
  - `tests/`: `tests/pipeline.rs` (Rust integration tests), `tests/unit/` and `tests/integration/` (pytest), `tests/conftest.py`.
- `params_training.conf`, `params_evaluation.conf`: TOML files at the repository root holding the algorithm tuning parameters of the `train`/`pipeline` and `evaluate`/`pipeline` stages respectively; loaded by `hm-ibf-hao/src/config.rs`. See **Environment & Config**.
- `scripts/verify.sh`: The verification gate.
- `scripts/smoke.sh`: The end-to-end smoke gate (`./run.sh smoke`): preprocess, train and evaluate on synthetic terrain with tiny budgets. Needs the Nix image, because training calls IRACE.
- `.claude/skills/hm-ibf-audit/`: Agent skill that classifies any source tree against the HM-IBF definition and reports deviations (`audit.py` scanner and CLI, `criteria.py` trait catalog, `model.py` dataclasses). Extend the catalog, not the scanner, when adding a trait.
- `.claude/skills/hm-ibf-retarget/`: Agent skill that retargets the pipeline to another optimization problem — a domain interview, a fit test, and the ordered edit surface with live `file:line` anchors (`retarget.py` resolver and CLI, `sites.py` change-surface catalog, `model.py` dataclasses). `--check` reports alignment assumptions that must not survive a retarget. Extend the catalog, not the resolver, when the code surface moves; every anchor is a single-line regex, so splitting an anchored signature across lines unresolves its site.
- `scripts/hm-ibf-entrypoint.sh`: Installed by `Dockerfile` as `hm-ibf` on `PATH` inside the `dev`/`dev-nix` images; builds the release binary if needed and always runs it from `hm-ibf-hao/`. For `train`/`pipeline`, re-execs itself through `nix develop` when `nix` is on `PATH` and it is not already inside a Nix shell.
- `run.sh`, `run.bat`: Host entry point; builds/starts the container and forwards commands (interactive shell, verify, arbitrary commands, including `hm-ibf` itself). `run.bat` is the same thing for PowerShell/cmd, no Git Bash/WSL required; keep the two in sync.
- `Cargo.toml`, `Cargo.lock`, `rustfmt.toml`: Rust workspace metadata and formatting.
- `pyproject.toml`: `ruff` and `pytest` configuration.
- `requirements-dev.txt`: Python environment of the container's `dev` stage.
- `flake.nix`, `flake.lock`: Nix devshell (needed for R/IRACE during training).
- `Dockerfile`, `.dockerignore`: Multi-stage container specification.
- `AGENTS.md`, `README.md`, `hm-ibf-hao/README.md`, `hm-ibf-hao/runbook.md`: Docs.
- Transient: `target/`, `.direnv/`, `result`, `hao_run/`, `experiments_hao/`, `results/`, `results_hao.csv`, `hm-ibf-hao/external/` (the downloaded heightmap cache).

## Quickstart

- `./run.sh` (from the **host**; it talks to the Docker daemon) builds the `dev` image if missing, starts the `hm-ibf-hao` container and opens a shell at `/app`. `.\run.bat` is the Windows equivalent — the `.\` prefix is required in PowerShell (unlike cmd.exe, it does not run scripts from the current directory by bare name).
- `HM_IBF_NIX=1 ./run.sh` (bash) / `$env:HM_IBF_NIX = "1"; .\run.bat` (PowerShell — `&` is not a command separator there) / `set HM_IBF_NIX=1 & run.bat` (cmd.exe) uses the `dev-nix` image/`hm-ibf-hao-nix` container instead — adds Nix/R/IRACE, needed for `train`/`pipeline`. Runs alongside a plain `dev` container under a different name.
- `./run.sh <command>` runs a single command inside the container instead of opening a shell, e.g. `./run.sh hm-ibf --help`.
- `./run.sh verify` runs the whole verification gate.
- Inside the container, the pipeline is the `hm-ibf` command installed on `PATH` — no `cargo run -p grahf-hao --bin hm-ibf-hao --` boilerplate, and always run from `hm-ibf-hao/` so relative defaults (`instances/`, `hao_run/`, `results/`) resolve correctly regardless of the caller's directory. It auto-detects the Nix-enabled container and wraps `train`/`pipeline` in `nix develop` itself.
- The container mounts the repository at `/app` and keeps `target/` in a named volume, so cargo builds are not slowed down by the host filesystem.

## Common Commands

All commands run inside the container (see **General**: Docker-only execution).

- Develop:
  - Quick check: `cargo check --workspace --all-targets`
  - Nix shell (only needed for R/IRACE): `nix develop`
- Run:
  - Help: `hm-ibf --help`
  - Train (needs the `dev-nix`/`hm-ibf-hao-nix` container): `hm-ibf train`
  - Evaluate: `hm-ibf evaluate`
  - Both: `hm-ibf pipeline`
  - Generate instances: `python3 -m preprocessing.prepare_instances --source <heightmap.npy>` (from `hm-ibf-hao/`)
- Test:
  - Full suite: `cargo test --workspace`
  - Focused test: `cargo test <test_name>`
- Format/Lint:
  - Lint check: `cargo clippy --workspace --all-targets -- -D warnings`
  - Format check: `cargo fmt --all -- --check`
  - Auto-format: `cargo fmt --all`
- Build/Release:
  - Release build: `cargo build --release --workspace`
  - Runtime image: `docker build -t hm-ibf-hao:latest .`
  - Dev image: `docker build --target dev -t hm-ibf-hao:dev .`
- Python (from the repository root):
  - Lint check: `ruff check .`
  - Format check: `ruff format --check .`
  - Auto-format: `ruff format .`
  - Tests: `python3 -m pytest` (paths come from `pyproject.toml`)

Note: Prefer the existing tooling as configured; avoid duplicating dependency or linter settings present in `Cargo.toml`, `pyproject.toml` or `flake.nix`.

## Environment & Config

- Run identity and file locations (seeds, `--jobs`, `--instances-dir`, `--eval-instances-dir`, `--run-dir`, `--experiments-dir`, evaluation's `--results-dir`/`--summary-csv`/`--elitist`) are CLI flags (`src/cli.rs`); add new settings there with a documented default, and update `hm-ibf-hao/runbook.md`. `--instances-dir` selects the training instances and `--eval-instances-dir` the evaluation ones, defaulting to the former.
- The GRAHF search's and evaluate stage's algorithm *tuning parameters* live in two TOML files at the repository root instead: `params_training.conf` (read by `train`/`pipeline`; also supplies the `max_evaluations` budget shared with `evaluate` and the `dimensions_allowed` list) and `params_evaluation.conf` (read by `evaluate`/`pipeline`). `hm-ibf-hao/src/config.rs` defines `TrainingParams`/`EvaluationParams` and loads them; the `Cli`'s `--training-params`/`--evaluation-params` flags only point at the file, defaulting to `../params_training.conf`/`../params_evaluation.conf` relative to the `hm-ibf-hao/` working directory. `TrainingParams::load` rejects a `dimensions_allowed` that is empty, not strictly increasing, or contains a zero. Add new tuning parameters as fields there (with a doc comment) and update both `.conf` files and `hm-ibf-hao/runbook.md`.
- The instance generator takes its configuration from `argparse` flags (`preprocessing/prepare_instances.py`); its defaults live in `preprocessing/config.py`.
- `RUST_LOG` selects the log level (`info` by default in the container; the flake sets `full` backtraces). The binaries log through `log`/`pretty_env_logger` — do not use `println!` in library code.
- The runtime image takes configuration exclusively from environment variables and mounted volumes (`/work/results`, `/work/hao_run`).
- Container Python lives in `/opt/venv`; add dependencies to `requirements-dev.txt` *and* to the `pythonEnv` in `flake.nix`.

## Security Guidelines

- Secrets: Never commit secrets; `.env` is ignored by VCS and by Docker.
- Inputs: Validate and sanitize all CLI and file inputs; avoid passing user input directly to shell or external processes.
- Filesystem: Use `std::path::PathBuf`, avoid writing outside the project or OS temp dirs; handle permissions and existence checks explicitly.
- Dependencies: This project is licensed `GPL-3.0-or-later` (see `LICENSE`). **Any OSI-approved open-source library may be used** — permissive (MIT, Apache-2.0, BSD, …) or copyleft (GPL, LGPL, MPL, …) — for Rust crates and Python packages alike, as long as it is compatible with GPL-3.0-or-later. Only proprietary or non-OSI "source-available" dependencies are off limits.
- Logging: Do not log secrets or PII.
- Execution: Avoid `unsafe` code blocks, dynamic code evaluation, or deserializing untrusted data without validation.

## Workflow Tips

- Rust edition: Target Rust 2021 edition as defined in `Cargo.toml`. The pinned toolchain is `RUST_VERSION` in `Dockerfile`; `Cargo.lock` requires cargo >= 1.85 (edition2024 dependencies).
- Nix shell: Run `nix develop` *inside* the Docker container (never on the bare host). It is only needed for the training stage, which shells out to R/IRACE; everything else works with the container's own toolchain.
- Tests first: Add/adjust tests for any new behaviour. Rust unit tests live inline in the file they cover; Rust integration tests live in `tests/` (per crate); Python unit tests live in `hm-ibf-hao/tests/unit/` and integration tests in `hm-ibf-hao/tests/integration/`.
- CLI changes: Update `hm-ibf-hao/src/cli.rs`. Every subcommand shares the global flags on `Cli`; clap requires argument ids to be unique across flattened groups, and `cli::tests::the_command_definition_is_valid` catches violations.
- Island set changes: adding, removing or reordering an entry of `islands::island_builders` shifts the node weight encoding and invalidates every stored `elitist_*.json`. Update the `ISLAND_*` constants, `training::initial_population` and `hm-ibf-hao/README.md` together with the list; `islands::tests::the_island_constants_match_the_builder_order` catches a forgotten constant. Editing `dimensions_allowed` in `params_training.conf` is a similar break: a stored elitist's IRACE-tuned `dimension` categorical may no longer be in the new set, so retrain after changing it.
- `dimensions_allowed` must stay strictly increasing: `HorizontalAlignment::dimension()` and `domain()` both read its last entry as the maximum, and `TrainingParams::load` rejects anything else. Regenerating instances prints a fresh recommendation (`dimensions_allowed = [...]`); paste it into `params_training.conf` and re-run the gate. `tests/integration/test_shipped_instances.py` fails if the shipped set and the recommendation drift apart.
- Problem changes: when the objective, the decision-variable encoding or the instance schema changes — and always when swapping in a different domain — work from `.claude/skills/hm-ibf-retarget/` rather than grepping. Two invariants dominate: the objective must read `solution.len()` (never `problem.dimension()`, which returns the *maximum* allowed dimension by contract), and a migration transform must return exactly `target_dim` elements.
- Export schema: `hm-ibf-hao/src/alignment/output.rs` writes schema `1`; bump `OUTPUT_EXPORT_SCHEMA_VERSION` when the payload changes.
- Documentation: Update `README.md` for user-facing changes and this **Project Guide For Agents** section for agent-facing guidance. For the `hm-ibf-hao/` crate, **always keep `hm-ibf-hao/README.md` and `hm-ibf-hao/runbook.md` current** whenever preprocessing, training or evaluation behaviour, flags or outputs change — a change that leaves either of them stale is not done.
- Definition of done: inside the container, run `./run.sh verify` (or `scripts/verify.sh` directly). See **Definition of done** under **General** for the exact sequence. Nothing is finished until all of it is green — not just the parts you touched. For a change that touches the stage wiring, also run `HM_IBF_NIX=1 ./run.sh smoke`; `cargo test -p grahf-hao --test smoke` already covers the evaluate stage inside the gate.
- Automation: the preprocess → train → evaluate pipeline runs end-to-end without manual steps; see **Automation** under **Coding guidelines**.

## Known Pitfalls

- Host execution: `cargo`/`pytest` on the bare host will not match the pinned toolchain. Always go through `run.sh`.
- Toolchain floor: building with cargo < 1.85 fails on `hashbrown` ("feature `edition2024` is required"). Bump `RUST_VERSION` rather than editing `Cargo.lock`.
- `rustfmt.toml` sets `imports_granularity` and `group_imports`, which are nightly-only. Stable `cargo fmt` prints a warning for each and ignores them; the exit code is unaffected.
- Nix lock drift: If system packages or Rust dependencies fail to build in Nix, update the flake via `nix flake update`.
- Nix flake vs. bind mount: `/app` is bind-mounted from the host, so the `dev-nix` image's `git config --global --add safe.directory /app` is required or `nix develop`'s libgit2-based flake fetcher refuses it ("repository path is not owned by current user"). Keep this if the Dockerfile's user/mount setup changes.
- Docker context: `.dockerignore` excludes `target/`, `.git/` and experiment output; keep it in sync when adding generated directories.
- Instance regeneration needs network access in benchmark mode: `preprocessing.prepare_instances` without `--source` downloads roughly 190 MB of heightmaps from the upstream `HorizAligns-Hybrid-Optimization` repository into `hm-ibf-hao/external/horizaligns/` (cached; delete to refetch). Offline runs must pass `--source <heightmap.npy>`, which selects the sampled train/eval mode instead. The checked-in `instances/` are the reproducible fallback for both.
- Nix flakes only see git-tracked files: in a fresh clone where `flake.nix` is still untracked, `hm-ibf train`/`pipeline` fails with `Path 'flake.nix' ... is not tracked by Git`. Commit it, or `git add -N flake.nix flake.lock`.
- Dimension mismatch: components must read `solution.len()` (or `IslandDimension` from island state), never `problem.dimension()`, which always reports the maximum allowed dimension. The `islands/safe_*.rs` modules exist for exactly this reason.

## When Adding Code

- Structure: Keep modules cohesive and small; separate CLI (`cli.rs`), stage orchestration (`training.rs`, `evaluation/`) and domain logic (`alignment/`, `clothoid/`, `islands/`). Keep each `.rs`/`.py` file to at most 500 lines; split into modules before exceeding the limit.
- Types & docs: Leverage Rust's strong type system, custom `enum`s for state, and write Google-style doc comments (summary + `# Arguments`/`# Returns`/`# Errors`/`# Panics`) for public constructs; the same docstring standard applies to every Python module, function, method and class.
- Errors: `thiserror`/`eyre` in library code, `anyhow`/`eyre` at the binary entry point. Do not swallow errors: fall back only where the fallback is correct and documented, and prefer failing loudly over guessing.
- Tests: Cover new logic with unit tests colocated in the same file/module and integration tests kept separately. Run the whole gate before considering a change done.
- Config: Introduce new settings as CLI flags with safe defaults and document them in `hm-ibf-hao/runbook.md`.
- Dependencies: Keep Cargo/Python dependencies minimal; any OSI-approved open-source license is acceptable given this project's own `GPL-3.0-or-later` license.
