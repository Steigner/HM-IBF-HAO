---
name: hm-ibf-retarget
description: Retarget the HM-IBF pipeline to a different optimization problem - swap the objective function, the decision-variable encoding, the instances and the migration transform while keeping heterogeneous migration intact. Use when a user arrives with their own domain (enzymes, reaction profiles, scheduling, control, antenna design) and wants to run `hm-ibf` on it, when asked to change the objective function or the fitness evaluation, to add a new problem alongside the horizontal alignment benchmark, or to judge whether a domain fits the benchmark at all.
---

# Retargeting HM-IBF to a new problem

The framework is already generic. `training::run`, `island_builders`, `load_elitist`, every
island and every migration policy are bound only by
`P: RealValuedProblem + DimensionAwareDomain` and never mention terrain. Retargeting is
therefore **not a rewrite**: it is a finite, enumerable set of edits in three layers —
rewrite the domain, re-bind the concrete types, touch nothing else.

Everything runs inside the container (`AGENTS.md`), so every command goes through `./run.sh`.

## Run

Print the ordered edit surface with live `file:line` anchors:

```bash
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py .
```

After retargeting, verify no alignment assumption survived:

```bash
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py . --check
```

Machine-readable, and a self-test for the resolver itself:

```bash
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py . --json
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py --self-test
```

Exit codes: `plan` returns `2` if a site failed to resolve, `check` returns `2` on a
high-severity leftover, `--self-test` returns `1` on failure.

## Step 1 — the domain interview

**Do not edit anything before these five are answered.** They are not preferences; each one
maps onto exactly one site, and guessing produces a run that completes and reports numbers
that mean nothing.

| # | Question | Site it decides |
|---|---|---|
| 1 | **Block.** Which group of variables describes one repeatable element? How many variables is that (`B`)? | `block_size`, `dimensions_conf` |
| 2 | **Objective.** What is `f(x)`, minimised, on one scalar scale? Which constraints are penalties and at what weight? | `objective` |
| 3 | **Decoder.** How does one block become something physically comparable across solutions? | `decoder` |
| 4 | **Backbone.** What resolution-independent coordinate `t ∈ [0, 1]` says *how far along* a block sits? | `backbone` |
| 5 | **Topology.** Are the variables periodic, bounded-linear, or constrained (sum-to-one, monotone)? | `variable_topology`, `topology_bound` |

Ask all five at once, in the user's own vocabulary. If the user cannot answer 4, go to the
fit test before writing code.

The alignment's own answers are the worked reference: `B = 1` (one scalar offset per control
point), `f(x) = terrain-weighted length + 1000·(curvature violations)`, an asymmetric
clothoid spline as decoder, an A\* route resampled at equal arc lengths as backbone, and
bounded-linear offsets clipped to the terrain.

## Step 2 — the fit test

The benchmark's defining property is that islands run at **different dimensions that are
different resolutions of the same object**, and a migrant is re-expressed rather than copied.
Formally the design is `(G, h, mu, D, tau)`; when `D` is constant over the nodes every `tau`
collapses to the identity and you are back to a plain island model. That needs a shared axis.
Check three things and say so plainly if one fails:

1. **Is there a resolution axis?** `53` and `68` variables must be a coarse and a fine
   description of one thing. If the dimensions are genuinely different problems (10 enzymes
   vs. 50 enzymes in a portfolio), migration has no meaning and a fixed-dimension optimiser
   is the honest answer.
2. **Is the objective scale-free across dimensions?** If more variables mechanically score
   better, the outer search selects the largest dimension and heterogeneity collapses. The
   alignment buys this with a *fixed* sampling step (`CLOTHOID_POINT_STEP`): make the
   discretisation depend on the block count and comparability is gone.
3. **Are the variables continuous?** The encoding is `Vec<f64>` with `Element = f64`.
   Categorical variables (an amino acid identity, a discrete catalyst choice) need a
   documented continuous relaxation plus a decoding rule, and the objective must stay
   meaningful between grid points. Say this out loud rather than rounding silently.

Failing the test is a real finding. Report it, propose the nearest thing that does work
(fixed-dimension GRAHF, or a different backbone), and stop.

## Step 3 — the edit surface

Run `retarget.py .` and work its three layers top to bottom. Each entry carries the change
to make and the invariant that must survive it. Two rules that the report repeats and that
are worth stating up front:

- **The objective must read `solution.len()`, never `problem.dimension()`.**
  `dimension()` returns the *maximum* allowed dimension by contract; islands read their real
  size from `IslandDimension` in state. An objective that reads the problem's dimension
  scores every island as if it were the largest one.
- **`tau` must return exactly `target_dim` elements** for every input the islands can
  produce, including degenerate ones. The caller inserts the result into the target
  population without re-checking.

## Step 4 — verify

```bash
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py . --check   # no leftovers
./run.sh python3 .claude/skills/hm-ibf-audit/audit.py .                 # still HM-IBF
./run.sh verify                                                          # the gate
```

Then a real run, because none of the above executes the new objective end to end:

```bash
./run.sh cargo test -p grahf-hao --test smoke   # the evaluate stage, no IRACE needed
HM_IBF_NIX=1 ./run.sh smoke                     # preprocess -> train -> evaluate
```

A retarget is done when the gate is green, the audit still returns an `HM-IBF *` verdict, and
one run has written a `results.json` whose `solution_dim` is an allowed island dimension.

## Worked example: enzymatic process profiles

A user arrives with enzymes and a fed-batch reaction they want optimised. The interview
resolves to:

| Horizontal alignment | Enzyme process |
|---|---|
| `B = 1` scalar offset per control point | `B = 3` set-point variables (temperature, pH, feed rate) |
| `D` control points along the backbone | `M` control set-points along the reaction |
| `dimensions_allowed = [53, 60, 68]` from Douglas-Peucker | `[3, 6, 9, 12, 15]` = 1..5 set-points |
| Clothoid spline: offsets → a drivable corridor | Kinetic model: set-point → reactor state |
| Backbone: A\* route resampled at equal arc lengths | Backbone: normalised reaction coordinate (conversion, or elapsed time / batch time) |
| `weighted_length + 1000·curvature_violations` | `-titre + GAMMA·max_constraint_violation` |
| Offsets clipped to the terrain | Clamp to the operating envelope |
| Instance = heightmap + start/goal | Instance = substrate load, enzyme batch, quality targets |

Two islands then genuinely mean something different: one searches a 2-set-point profile,
another a 5-set-point profile, and a migrant crossing that edge is resampled along the
reaction coordinate rather than index-to-index. That is the whole point of the framework.

The same interview run on **enzyme sequence design** usually fails the fit test at question 4
unless the residues are anchored to a structural backbone: sequence length is not a
resolution of one object, and residue identity is categorical.

## Traps

- **The variable topology is the one that bites.** The alignment's offsets are
  bounded-linear, so `transforms/resample.rs` interpolates them directly. On a *periodic*
  domain (angles, phases, headings) that is wrong at the wrap boundary and corrupts every
  migrant **silently** — no panic, no NaN, just plausible-looking results. Unwrap by whole
  periods before interpolating and re-wrap afterwards, in every method of the catalog.
- **The bounds are position dependent.** `domain_for_dimension` resamples the backbone; it is
  not `domain()` sliced. A retarget that replaces it with a slice hands each island the
  bounds of positions it never uses, and the only symptom is bad fitness.
- **Editing `dimensions_allowed` invalidates every stored `elitist_*.json`.** A trained
  elitist carries an IRACE-tuned `dimension` categorical that may no longer be in the set.
  Retrain; do not reuse `hao_run/` across a retarget.
- **Do not touch the island set while retargeting.** Reordering `islands::island_builders`
  shifts the node-weight encoding. It is a separate change with its own migration path
  (`AGENTS.md`, *Island set changes*). PSO's absence is deliberate, not an oversight — see
  the audit skill.
- **The export is verbatim.** `OUTPUT_TRANSFORM_NOT_APPLIED` and `select_global_best_solution`
  promise that `x` is exactly the individual that produced `best_value`. Adding an output
  projection is a design change, not a retarget — and it needs a schema bump.
- **Single objective only.** `AlgorithmDesignProblem` is built on `SingleObjective`. A
  multi-objective domain must be scalarised in the objective, deliberately and documented.
- **Training needs the Nix container.** `train`/`pipeline` shell out to R/IRACE:
  `HM_IBF_NIX=1 ./run.sh` (bash) or `$env:HM_IBF_NIX = "1"; .\run.bat` (PowerShell). Nix
  flakes only see git-tracked files, so `flake.nix` must be committed (or `git add -N`) in a
  fresh clone.

## Gotchas

- **`--check` fires on everything in this repository, and that is correct.** This *is* the
  alignment benchmark; all seven residues are expected here. It is a post-retarget check, not
  a health check — run it against the retargeted tree.
- **`--check` skips `.claude/`.** The catalog's own pattern strings would otherwise match
  every residue it searches for. Never remove that entry from `SKIPPED_DIRECTORIES`.
- **`NOT FOUND` after a retarget is usually good news** — it means the site was renamed out
  of alignment vocabulary. `NOT FOUND` on an untouched tree means the catalog has drifted from
  the code; fix `sites.py`.
- **The catalog is anchored, not line-numbered.** Anchors are regexes over single lines, so
  reformatting that splits a signature across lines will unresolve a site. Widen the anchor
  in `sites.py` rather than reformatting the source back.
- **`global_dimension_read` is a `note`, not a defect.** `VectorProblem::dimension` and the
  documented island fallbacks legitimately call it. Read each hit.
- **`offsets` is a broad residue pattern.** It fires on any variable called `offsets`, which
  is deliberate: in this codebase that word *is* the encoding. Expect false positives in a
  retargeted tree that happens to reuse the name, and read the hits.
- **Retarget in place, or alongside?** In place (replace `alignment/` and `clothoid/`) is the
  default and the smaller diff. Keep both only if the alignment benchmark must stay runnable —
  then `evaluation::run` and `main.rs::evaluate` have to become generic too, which widens the
  diff into shared files. Ask before choosing the second.

## Keeping it green

The gate lints this directory (`ruff check .` covers `.claude/`), and `pytest` does not reach
it — correctness is covered by `--self-test`, as in `hm-ibf-audit`. After editing:

```bash
./run.sh ruff check .claude/skills/hm-ibf-retarget/
./run.sh ruff format --check .claude/skills/hm-ibf-retarget/
./run.sh python3 .claude/skills/hm-ibf-retarget/retarget.py --self-test
```

Extend `sites.py` (pure data) when the code surface changes; `model.py` holds the
dataclasses, `retarget.py` the resolver. Each file stays under the 500-line limit from
`AGENTS.md`.
