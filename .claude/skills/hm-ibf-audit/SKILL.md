---
name: hm-ibf-audit
description: Audit, classify and review whether a repository implements an HM-IBF (heterogeneous migration island-based framework) and report where it deviates. Use when asked whether code is HM-IBF, whether islands really migrate across dimensions, to check backbone/arc-length resampling, transform catalogs, dimension-agnostic operators, two-level (architecture + hyperparameter) design, or to review an island-model implementation against the HM-IBF definition.
---

# HM-IBF conformance audit

Classifies a source tree against the seven structural traits of an HM-IBF and prints a
verdict with `file:line` evidence for every trait.

The discriminator is **heterogeneous migration**: a plain island model copies a migrant
verbatim; an HM-IBF re-expresses it, because neighbouring islands may run at different
dimensions that are different *resolutions of the same object*. A repo can have islands,
a graph and migration and still not be HM-IBF.

Formally, a design is the tuple `(G, h, mu, D, tau)` — a digraph of islands, a solver per
node, a migration policy per edge, a **per-island working dimension** `D`, and a transform
`tau` that reconciles `D(u)` with `D(v)` before insertion. Constant `D` makes every `tau` the
identity and recovers the homogeneous island model as a special case. That is the line this
skill draws.

Paths below are relative to the repository root. Per `AGENTS.md` everything runs inside the
container, so every command goes through `./run.sh`.

## Run

```bash
./run.sh python3 .claude/skills/hm-ibf-audit/audit.py .
```

Audit a subtree (useful to separate a framework crate from its benchmark):

```bash
./run.sh python3 .claude/skills/hm-ibf-audit/audit.py src
```

Machine-readable, for chaining into further analysis:

```bash
./run.sh python3 .claude/skills/hm-ibf-audit/audit.py . --json
```

Verify the classifier itself before trusting a surprising verdict:

```bash
./run.sh python3 .claude/skills/hm-ibf-audit/audit.py --self-test
```

Exit codes: `0` = an `HM-IBF *` verdict, `2` = anything else, `1` = failed self-test.

## Verdicts

| Label | Meaning |
|---|---|
| `HM-IBF - FULL` | All seven traits implemented. |
| `HM-IBF - CORE COMPLETE` | Heterogeneous migration works; the two-level design loop is incomplete. |
| `PARTIAL HM-IBF` | Migration across dimensions started but unfinished. |
| `HOMOGENEOUS ISLAND MODEL - NOT HM-IBF` | Islands and migration exist, but tau is identity. |
| `NOT AN ISLAND MODEL` | No island graph found. |

Per-trait marks: `[x]` implemented, `[~]` partial, `[ ]` absent. Per-signal marks: `+` proven
by code, `?` **described in docs only**, `-` not found.

## What it checks

`C1` graph topology · `C2` heterogeneous migration (tau on the edge) · `C3` backbone /
arc-length resampling · `C4` swappable transform catalog · `C5` dimension-agnostic islands
and operators · `C6` two-level automatic design · `C7` freeze into a deterministic solver.

Plus deviation probes: stateful PSO across dimension changes, missing identity bypass,
index-to-index resampling, missing phase unwrapping, global-dimension reads, an optional
(silently skippable) transformer, an unchecked post-transform length, and a compiled-in
dimension set.

The traits are pure data in `.claude/skills/hm-ibf-audit/criteria.py` — add a `Criterion` or
`Deviation` there and the engine picks it up with no other change. `model.py` holds the
dataclasses, `audit.py` the scanner.

## How this repository answers the seven traits

Useful as a reference point when auditing something else — this is what a `FULL` verdict
looks like in practice:

| Trait | Where it lives here |
|---|---|
| `C1` | `grahf`'s `DiIntGraph`; node weights index `islands::island_builders`, edge weights `migrations::migration_builders`. |
| `C2` | `components/transform.rs` + `islands/transforms/`; equal dimensions short-circuit to a copy, migrants are re-evaluated by the target island. |
| `C3` | `alignment/problem.rs::sample_backbone` — offset `i` of a `D`-dimensional solution sits at `s_i = i·L/(D+1)`, i.e. `t_i = (i+1)/(D+1)`. |
| `C4` | `TransformMethod`: linear, PCHIP, Akima, clamped cubic, total-variation denoising — IRACE picks one per migration edge. |
| `C5` | `IslandDimension` in island state, `DimensionAwareDomain::domain_for_dimension`, and the `islands/safe_*.rs` operators that read `solution.len()`. |
| `C6` | EDGE evolves the graph on the z-score; IRACE tunes each candidate against the **raw** median best objective. Two levels, two objectives. |
| `C7` | `evaluation/elitist.rs` reloads the frozen graph and its tuned parameters and runs without the design loop. |

**PSO is deliberately absent from the island set**, and that is a conformance point rather
than a gap. Unlike DE or boundary handling, PSO carries per-particle auxiliary state — a
velocity and a personal best — that has no component-wise correspondent at another
resolution. A safe variant that zeroes the velocity and collapses the personal best after
every migration runs without out-of-bounds access, but the particle then restarts with no
inertia and no memory, which is precisely what separates PSO from a random restart. If a tree
under audit keeps PSO among dimension-agnostic islands, the `stateful_pso` probe is telling
you something real.

## Manual review checklist

**A `FULL` verdict is a presence check, not a correctness review.** Every signal is a
line-level regex, so the seven traits can all be `[x]` while the migration path is wrong.
When the task is to *review* rather than to classify, open the migration gate and the
transform entry point and confirm these by reading — none of them are lexically decidable:

1. **Is the transform gate per-individual?** A gate that samples one representative
   (`selected.first()`, `population[0]`) and then transforms the whole batch is wrong for a
   population that mixes lengths: if the representative already matches the target dimension
   the gate stays shut and the mismatched migrants pass through untransformed. The predicate
   must be `any`, or the decision must move inside the per-individual loop.
2. **What happens when the transform is skipped?** Find the fallback arm. Skipping the
   transform while `D(source) != D(target)` must be an error, not a `false`.
3. **Is the output length checked?** The `SolutionTransformer` contract promises
   `target_dim`, but degenerate inputs usually return an empty vector. Verify the caller
   asserts it before insertion.
4. **Do the tuned dimensions and the runtime dimension agree?** The migration edge's target
   dimension and the target island's own working dimension come from separate reads of the
   same parameter set; confirm they cannot diverge.
5. **Are the dimension fields actually read?** Fields carried on the edge purely to describe
   the tuning (e.g. a `source_dimension` that nothing consumes) are dead weight that makes
   the encoding look richer than it is.
6. **Are the per-dimension bounds really per-dimension?** `domain_for_dimension` must
   resample the reference, not slice the maximum-dimension domain: different dimensions place
   their samples at different positions, so a slice hands an island bounds belonging to
   positions it never uses. This one fails silently, as bad fitness.
7. **Is the objective scale-free across dimensions?** Anything that makes the evaluation
   finer as the dimension grows — a discretisation derived from the block count, a per-sample
   bonus — turns the outer search into a search for the largest dimension.
8. **Is the reference derived from the instance alone?** A backbone built from any candidate
   solution or intermediate result is look-ahead, and the comparison against fixed-dimension
   baselines stops being fair.

## Gotchas

- **Documentation can never satisfy a trait.** A `.md` hit is corroboration only; a signal
  counts as found solely on non-doc evidence. This is load-bearing — before it was added,
  `README.md` and `AGENTS.md` alone satisfied C4, C6 and C7. The `prose_only` self-test case
  guards it: a README describing every trait must still classify `NOT AN ISLAND MODEL`.
- **The auditor detects itself.** Its own regex literals match every signal it searches for,
  so `.claude` is in `SKIPPED_DIRECTORIES`. Never remove that entry; auditing a repo whose
  implementation lives under `.claude` will silently under-report.
- **Deviation probes ignore whole-line comments and docs.** A comment reading
  `// use solution.len() instead of problem.dimension()` otherwise fires the exact anti-pattern
  it warns against. Criteria signals *do* accept comments, since doc comments are legitimate
  evidence of design intent.
- **Trailing comments are not stripped**, only whole-line ones. A deviation pattern in a
  trailing `// ...` can still fire.
- **`global_dimension_reads` fires on this repo and that is correct.** The hits in
  `hm-ibf-hao/src/islands/mod.rs` and `safe_boundary.rs` are documented fallbacks for islands
  that did not override the dimension. It is a `note`, not a defect — read the lines before
  acting.
- **`unwrap_missing` fires on this repo and that is also correct.** The alignment's offsets
  are bounded-linear scalars, not angles, so there is nothing to unwrap. The probe is there
  for trees whose variables *are* periodic; read the encoding before acting on it.
- **`hardcoded_dimension_set` should stay silent here.** The allowed dimensions live in
  `params_training.conf` and are validated at load time. If it fires, someone reintroduced a
  compiled-in constant, and every stored elitist now depends on a recompile.
- **Auditing `src` alone yields `PARTIAL HM-IBF`.** The `grahf` framework holds the graph and
  the `SolutionTransformer` hook, but the backbone and dimension-safe operators live in
  `hm-ibf-hao/`. Audit the repo root for the real verdict.
- Signals are lexical. Renaming `backbone` to a domain word (`spine`, `medial_axis`) or
  writing the catalog in another language will under-report; extend `criteria.py` rather than
  trusting a low score.
- **A trait can be satisfied by prose inside a source file.** Only `.md` files count as
  documentation; a Rust `//!` or `///` comment is code evidence. Never read a green `C4` as
  proof that the catalog exists — follow the signal to the file and check for real function
  bodies.
- **The scanner matches one line at a time.** An expression split across lines (a `let` whose
  `.first()` lands on the next line) cannot be caught by any `Deviation`. That is why the
  migration gate is on the **Manual review checklist** instead of being a probe.

## Troubleshooting

- `python3: command not found` — you piped the output into the **host** shell. Keep the whole
  pipeline inside the container: `./run.sh bash -c '... | python3 -c "..."'`.
- Verdict looks wrong → run `--self-test` first, then `--json` and read the `evidence` arrays
  to see which files actually matched.

## Keeping it green

The gate lints this directory (`ruff check .` covers `.claude/`). After editing:

```bash
./run.sh ruff check .claude/skills/hm-ibf-audit/
./run.sh ruff format --check .claude/skills/hm-ibf-audit/
./run.sh python3 .claude/skills/hm-ibf-audit/audit.py --self-test
```

Each file stays under the 500-line limit from `AGENTS.md`.
