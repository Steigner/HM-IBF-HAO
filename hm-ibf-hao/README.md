# grahf-hao — the horizontal alignment benchmark

The problem GRAHF designs algorithms for, plus the `hm-ibf-hao` pipeline binary and the
Python instance generator. Heterogeneous migration between island dimensions is configured by
`dimensions_allowed` in [`../params_training.conf`](../params_training.conf), shipped as
`53, 60, 68` offsets.

For how to *run* it, see [runbook.md](runbook.md). This file describes what it is.

## The problem

An instance is a corridor across a terrain heightmap:

- **Backbone** — a least-cost (A\*) route between a start and a goal, resampled at equal arc
  lengths. It is derived from the raw route, not from its simplification, so it follows the
  terrain faithfully.
- **Solution** — a vector of `D` perpendicular offsets. Offset `i` displaces the backbone
  along its local normal at `s_i = i·L/(D+1)`; the backbone's own endpoints are fixed
  boundary conditions and carry no offset.
- **Objective** — the offset points, bracketed by the backbone endpoints, form a control
  polygon. It is interpolated with an asymmetric clothoid spline and scored as

  ```
  f(x) = terrain-weighted length + 1000 · (samples violating the minimum radius of curvature)
  ```

  A segment above `height_limit` has to be tunnelled and a segment steeper than
  `gradient_change_limit` needs earthworks; a segment that is both pays both multipliers.

Lower is better and `f(x) ≥ 0`, with a known (unreachable) optimum of zero.

## Why it is a hyper-heuristic benchmark

Islands may run at **different dimensions** — `dimensions_allowed` in the repository
root's `params_training.conf`, shipped as `[53, 60, 68]`.

That set is a *recommendation derived from the Douglas-Peucker compression* of each
instance's raw route: the compression estimates how many interior inflection points a route
really needs (its `natural_dimension`), and the pool of those estimates becomes the allowed
set. For the three benchmark maps it is exactly their own natural dimensions — Italy 53,
Slovenia 60, Austria 68 — so every island searches in the same native space a
single-dimension reference method would use.

Three consequences the design has to respect:

1. **Bounds are position-dependent.** Dimension `D` places its control points at different
   arc lengths than dimension `D'`, so the offset bounds differ too. Every island resamples
   its own bounds through `DimensionAwareDomain::domain_for_dimension`; slicing the
   maximum-dimension domain would hand it the bounds of positions it never uses.
2. **Migrants must be resampled.** `islands::transforms` treats the offsets as a 1-D signal
   sampled at `t_i = (i+1)/(D+1)` and resamples it; IRACE picks the method per migration
   edge from `arc_linear`, `arc_pchip`, `arc_akima`, `arc_clamped_cubic` and
   `arc_total_variation`. The result is then clamped into the target dimension's own bounds.
3. **Components must not ask the problem for its dimension.** `HorizontalAlignment::dimension()`
   reports the *maximum* allowed dimension, because that is what sizes the declared search
   space. Components read `solution.len()`, or `IslandDimension` from island state — that is
   what the `islands/safe_*.rs` modules exist for.

`dimensions_allowed` must stay strictly increasing: `dimension()` and `domain()` both read
its last entry as the maximum, and `TrainingParams::load` rejects anything else.

No output projection is applied: the exported `x` is verbatim the run's global best
individual, at whatever dimension the winning island ran.

## Island encoding

Node weights are positional, fixed by the order of `islands::island_builders` and mirrored by
the `ISLAND_*` constants:

| Weight | Constant | Island |
|---:|---|---|
| 0 | `ISLAND_DE` | Differential evolution |
| 1 | `ISLAND_ES` | Evolution strategy |
| 2 | `ISLAND_LS` | Local search |
| 3 | `ISLAND_SA` | Simulated annealing |
| 4 | `ISLAND_RS` | Random search |
| 5 | `ISLAND_ARCHIVE` | Passive archive (no search, tracks diversity) |

**Particle swarm is deliberately not in the set.** Unlike the operators above, which act on
the solution vector alone, PSO carries per-particle auxiliary state — a velocity and a
personal best — that has no component-wise correspondent at another resolution. A variant
that zeroes the velocity and collapses the personal best after every migration runs without
out-of-bounds access, but the particle then restarts with neither inertia nor memory, which
is exactly what separates PSO from a random restart. It stays available in the homogeneous
setting, where every island shares one dimension.

The archive is a **dependent island**: it only tracks diversity and never searches, so a
graph consisting of archives alone is infeasible.

Adding, removing or reordering an entry shifts the encoding and **invalidates every stored
`elitist_*.json`**. Update the constants, `training::initial_population` and this table
together with the list.

## Heterogeneous migration

When a migrant crosses to an island of a different dimension it is resized by
`islands::transforms`:

1. the offset vector is read as samples of a scalar profile along the backbone, offset `i` of
   a `D`-dimensional solution sitting at the resolution-independent coordinate
   `t_i = (i+1)/(D+1)`,
2. that profile is reconstructed at the target island's coordinates `t_j = (j+1)/(D'+1)` by
   the IRACE-selected method (`arc_linear`, `arc_pchip`, `arc_akima`, `arc_clamped_cubic`,
   `arc_total_variation`),
3. the result is clamped into the *target* dimension's own offset bounds, which sit at
   different arc lengths than the source's,
4. equal source and target dimensions short-circuit to a verbatim copy, so a homogeneous edge
   pays nothing.

Unlike a per-block projection, no decoding is needed: the offsets are attached to fixed
backbone positions by construction, so the coordinate of a sample follows from its index and
the dimension alone.

## Layout

```
src/
  cli.rs              clap definitions for the hm-ibf-hao binary
  config.rs           loads ../params_training.conf and ../params_evaluation.conf
  main.rs             thin entry point; all logic lives in the library
  training.rs         the GRAHF structure search
  evaluation/         replaying a trained elitist; log_series.rs extracts the progress CSVs
  alignment/          the problem: problem.rs, evaluator.rs, config.rs (loading), output.rs
  clothoid/           the asymmetric clothoid spline
  islands/            island builders, dimension transforms, safe_* components
  migrations/         migration condition, selection and replacement builders
  problems/           the DimensionAwareDomain trait
instances/            the three benchmark instances (checked in)
preprocessing/        the Python instance generator, map catalog and heightmap download
tests/                pipeline.rs + smoke.rs (Rust), unit/ and integration/ (pytest)
```

## Instances

Each instance is a pair of files plus a shared index:

- `<name>_config.json` — the backbone, the route it came from and the cost parameters.
- `<name>_heightmap.npy` — the terrain the backbone refers to, as `float32`.
- `summary.json` — the set's index; the training set also carries the recommended
  `dimensions_allowed`.

A directory entry counts as an instance only when both files are present, which is what keeps
`summary.json` out of the benchmark.

`preprocessing/` regenerates the set. In its default *benchmark* mode it downloads the three
named terrain maps (`preprocessing/maps.py`) from the upstream
[HorizAligns-Hybrid-Optimization](https://github.com/Steigner/HorizAligns-Hybrid-Optimization)
repository and routes each between its published endpoints, reproducing exactly what is
checked in. With `--source <terrain>.npy` it instead samples a random train/eval split from
one heightmap of your own.

The heightmaps are not part of this repository — the checked-in `instances/` are the
reproducible fallback. After regenerating, paste the printed `dimensions_allowed = [...]` line
into the repository root's `params_training.conf` and re-run the verification gate.

## Output

Evaluation writes one folder per `(instance, seed)` under `--results-dir`:

- `results.json` — schema `1`; carries the exported offsets as `x`, their objective value,
  the evaluation count and the instance's natural dimension. Bump
  `OUTPUT_EXPORT_SCHEMA_VERSION` in `alignment/output.rs` when the payload changes.
- `best_value.csv` — the run's best value against its evaluation count.
- `avg_value.csv` — the mean objective value across every island's population, against the
  same evaluation count. Both series are extracted from the run log by `evaluation/log_series.rs`.
- `run.log` — the raw MAHF log of the run.

An aggregated `results_hao.csv` summarizes the mean and sample standard deviation per
instance across all seeds.
