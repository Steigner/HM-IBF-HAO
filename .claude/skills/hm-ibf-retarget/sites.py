"""Catalog of every place a retarget to a new optimization problem has to touch.

The pipeline is already generic over the problem type: `training::run`, `island_builders`,
`load_elitist` and every island operator are bound only by
`P: RealValuedProblem + DimensionAwareDomain`. Retargeting is therefore not a rewrite but a
finite, enumerable set of edits, listed here as pure data.

Three layers, ordered by how much judgement each needs:

* ``rewrite`` - the domain semantics. Nothing here transfers to another problem.
* ``bind`` - mechanical, but skipping one entry produces a run that fails late or lies.
* ``keep`` - already generic. Editing these is how a retarget stops being an HM-IBF.

:mod:`retarget` anchors each entry to a live line; extend this catalog, not the resolver.
"""

from __future__ import annotations

from model import Layer, Residue, Site

LAYERS: tuple[Layer, ...] = (
    Layer(
        key="rewrite",
        title="REWRITE - the domain lives here",
        note=(
            "Each entry encodes an assumption about routing a corridor across terrain. Replace "
            "the body, keep the signature's contract. The five that decide whether migration "
            "still means anything are the block size, the objective, the decoder, the backbone "
            "and the variable topology."
        ),
    ),
    Layer(
        key="bind",
        title="BIND - mechanical, but load-bearing",
        note=(
            "Concrete type bindings, validation and the export payload. No design decisions "
            "here, yet skipping one produces a run that fails on the last instance or writes a "
            "payload that no longer describes what was optimized."
        ),
    ),
    Layer(
        key="keep",
        title="KEEP - already generic, do not touch",
        note=(
            "These are bound by `P: RealValuedProblem + DimensionAwareDomain` and never mention "
            "terrain or alignments. Editing them to 'make the new domain fit' is the failure "
            "mode: it specialises the framework instead of the benchmark."
        ),
    ),
)

SITES: tuple[Site, ...] = (
    # ---------------------------------------------------------------- rewrite
    Site(
        key="block_size",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/problem.rs",
        anchor=r"pub fn control_polygon",
        title="Block size B - decision variables per indivisible unit",
        change=(
            "The alignment's block size is 1: every decision variable is one scalar offset at "
            "one backbone position. If the new domain's repeatable element carries several "
            "variables (a set-point triple, a joint pose, a stage recipe), introduce a `B` "
            "constant and make this function decode `solution.len() / B` blocks."
        ),
        contract=(
            "A solution of length D encodes D/B blocks. Every allowed dimension must stay a "
            "positive multiple of B, and the flattening must stay block-major: block i owns "
            "indices [i*B, (i+1)*B)."
        ),
    ),
    Site(
        key="instance_schema",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/config.rs",
        anchor=r"^pub struct AlignmentConfig",
        title="Instance descriptor - what a benchmark case is",
        change=(
            "Replace the terrain, the endpoints and the route with the new domain's case data. "
            "Keep `name` (it labels run folders and the summary) and keep whatever field the "
            "backbone needs to build its parameterisation."
        ),
        contract=(
            "Stays `serde`-round-trippable: instances are read from JSON on disk, and the "
            "loader pairs `<name>_config.json` with a sibling payload file."
        ),
    ),
    Site(
        key="objective",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/problem.rs",
        anchor=r"pub fn evaluate_solution",
        title="Objective function f(x)",
        change=(
            "Replace the `weighted_length + CURVATURE_PENALTY * violations` body with the new "
            "objective."
        ),
        contract=(
            "THE load-bearing invariant of the whole framework: read `offsets.len()`, never "
            "`problem.dimension()`. A 53-variable and a 68-variable solution are scored on the "
            "same scale, otherwise the outer search just selects for whichever dimension the "
            "objective happens to flatter, and heterogeneous migration becomes noise."
        ),
    ),
    Site(
        key="decoder",
        layer="rewrite",
        path="hm-ibf-hao/src/clothoid/spline.rs",
        anchor=r"pub fn interpolate_path_with_tau",
        title="Decoder - decision variables to the observable space",
        change=(
            "Replace the asymmetric clothoid spline with the map from the control polygon to "
            "the space the objective measures (a reactor state trajectory, a joint path, a "
            "load profile). The whole `clothoid/` module goes with it."
        ),
        contract=(
            "Pure and deterministic, and its sampling density must not depend on the number of "
            "control points - `CLOTHOID_POINT_STEP` is fixed for exactly that reason. It is "
            "called once per evaluation, so it must be cheap and side-effect free."
        ),
    ),
    Site(
        key="bounds",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/problem.rs",
        anchor=r"fn domain\(&self\)",
        title="Search-space bounds at the maximum dimension",
        change=(
            "Delegates to `domain_for_dimension(self.dimension())`. Usually unchanged; change "
            "it only if the new domain's maximum-dimension bounds are not the per-dimension "
            "ones evaluated at the maximum."
        ),
        contract="Length must equal `self.dimension()`; islands slice it, never extend it.",
    ),
    Site(
        key="max_dimension",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/problem.rs",
        anchor=r"fn dimension\(&self\)",
        title="Declared dimension - the maximum, not the working one",
        change=(
            "Usually unchanged: it returns the largest entry of `dimensions_allowed`. Read the "
            "doc comment before touching it - returning anything else silently breaks every "
            "island that runs below the maximum."
        ),
        contract=(
            "Islands read their own working dimension from `IslandDimension` in state, set by "
            "`RandomSpreadWithDimension::init`. This method is the ceiling, not the runtime "
            "size, which is why `dimensions_allowed` must stay strictly increasing."
        ),
    ),
    Site(
        key="dimension_aware_domain",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/problem.rs",
        anchor=r"^impl DimensionAwareDomain for",
        title="Per-dimension bounds",
        change=(
            "The alignment overrides this because its bounds are *position dependent*: "
            "dimension D places control points at different arc lengths than D', so the "
            "terrain leaves different room at each. Keep the override if the new domain's "
            "bounds depend on where in the solution a variable sits; drop it for the default "
            "body, which cycles `domain()` modulo its length, only when they repeat with "
            "period B."
        ),
        contract=(
            "Must return exactly `dim` ranges. Getting this wrong does not crash - it samples "
            "initial populations outside the feasible set and the error only shows up as bad "
            "fitness."
        ),
    ),
    Site(
        key="backbone",
        layer="rewrite",
        path="hm-ibf-hao/src/alignment/problem.rs",
        anchor=r"pub fn sample_backbone",
        title="Backbone - the resolution-independent parameter t in [0, 1]",
        change=(
            "Replace the equidistant resampling of the A* route with the new domain's shared "
            "axis. Sample `i` of a `D`-dimensional solution sits at `t_i = (i+1)/(D+1)`, i.e. "
            "`s_i = i*L/(D+1)` in arc length, and the two endpoints are excluded because they "
            "are fixed boundary conditions."
        ),
        contract=(
            "This is what makes the framework HM-IBF rather than an island model. It answers "
            "'how far along is this block' in a way that does not depend on how many blocks "
            "the island uses. The backbone must be a preprocessing artefact of the instance "
            "alone - derived from no candidate solution and no intermediate result - or it "
            "becomes look-ahead. If the new domain has no such axis - variables are unordered, "
            "exchangeable, or purely categorical - say so and stop; the benchmark is not a fit."
        ),
    ),
    Site(
        key="variable_topology",
        layer="rewrite",
        path="hm-ibf-hao/src/islands/transforms/resample.rs",
        anchor=r"pub fn arc_linear",
        title="Variable topology - bounded-linear vs. periodic",
        change=(
            "Offsets are bounded-linear scalars, so the resamplers interpolate them directly. "
            "For a periodic domain - angles, phases, headings - this is WRONG at the wrap "
            "boundary: the signal has to be unwrapped by whole periods before interpolation "
            "and re-wrapped afterwards. Add that here, in every method, or the artificial "
            "jumps become real geometry."
        ),
        contract=(
            "Every method in the catalog must share one coordinate contract, "
            "`t_i = (i+1)/(D+1)`, and differ only in how the profile between samples is "
            "reconstructed."
        ),
    ),
    Site(
        key="topology_bound",
        layer="rewrite",
        path="hm-ibf-hao/src/islands/transforms/mod.rs",
        anchor=r"impl<P> SolutionTransformer<P> for OffsetResampleTransformer",
        title="Post-transform bounding back into the feasible set",
        change=(
            "The clamp into `domain_for_dimension(target_dim)` is what puts an overshooting "
            "spline back inside the terrain. Replace it with the new domain's feasibility "
            "restoration: a clamp, a projection, or a renormalisation for sum-constrained "
            "variables."
        ),
        contract=(
            "Runs on every migrant. The target dimension's bounds are not the source's - they "
            "sit at different positions along the backbone - so a migrant that was admissible "
            "where it came from can still arrive infeasible."
        ),
    ),
    Site(
        key="tau",
        layer="rewrite",
        path="hm-ibf-hao/src/islands/transforms/mod.rs",
        anchor=r"pub fn transform_with_optional_params",
        title="tau - the migration transform along the backbone",
        change=(
            "Rename to the new domain and rewrite the resampling if its samples are not a 1-D "
            "scalar profile. The identity bypass for equal dimensions stays."
        ),
        contract=(
            "MUST return exactly `target_dim` elements for every input the islands can produce. "
            "The caller inserts the result into the target island's population without "
            "re-checking the length. Migrants are then marked unevaluated and re-scored by the "
            "target island, so the fitness stays consistent with the target dimension."
        ),
    ),
    Site(
        key="transform_catalog",
        layer="rewrite",
        path="hm-ibf-hao/src/islands/transforms/mod.rs",
        anchor=r"^pub enum TransformMethod",
        title="The catalog IRACE picks a migration method from",
        change=(
            "Keep at least two genuinely different reconstructions. The shipped five - linear, "
            "PCHIP, Akima, clamped cubic and total-variation denoising - span exact "
            "interpolants and smoothing operators, which is the point: the configurator picks "
            "per edge how much structure a migrant keeps."
        ),
        contract=(
            "`all_names` and `from_name` must stay in sync, and an unknown name must fall back "
            "to a defined method rather than refusing to migrate - the name comes from a stored "
            "parameter file, and refusing would silently change the algorithm being replayed."
        ),
    ),
    Site(
        key="instance_generator",
        layer="rewrite",
        path="hm-ibf-hao/preprocessing/instance.py",
        anchor=r"^def build_config",
        title="Instance generator",
        change=(
            "Replace the A*-route-and-backbone assembly with the new domain's case generation, "
            "and `maps.py`/`fetch_maps.py` with wherever the source data comes from. Emit the "
            "JSON schema the new instance descriptor deserialises."
        ),
        contract=(
            "Stays a standalone package with its own dependencies - it is not part of the "
            "binary. Keep the deterministic per-instance seed so instances are reproducible."
        ),
    ),
    Site(
        key="dimension_estimator",
        layer="rewrite",
        path="hm-ibf-hao/preprocessing/dimensions.py",
        anchor=r"^def select_dimensions_allowed",
        title="Where the allowed dimensions come from",
        change=(
            "The alignment derives them from the Douglas-Peucker compression of each raw "
            "route: the compression estimates how many inflection points a case really needs, "
            "and the pool of those estimates becomes the recommendation. Replace it with the "
            "new domain's complexity estimator - the answer to 'how many blocks does this "
            "instance actually deserve'."
        ),
        contract=(
            "The output is a recommendation, not a hard constraint: it is pasted into "
            "`params_training.conf`, which the loader validates. Keep the two consistent, or "
            "the islands search at a resolution the instances never asked for."
        ),
    ),
    # ------------------------------------------------------------------- bind
    Site(
        key="dimension_validation",
        layer="bind",
        path="hm-ibf-hao/src/config.rs",
        anchor=r"^fn validate_dimensions_allowed",
        title="dimensions_allowed validation",
        change=(
            "Add the new domain's block invariant if `B > 1`: every entry must be a positive "
            "multiple of the block size."
        ),
        contract=(
            "Non-empty and strictly increasing. This is the only place that catches a "
            "mis-specified dimension set before it becomes a mid-run panic or a silently "
            "undersized search space."
        ),
    ),
    Site(
        key="dimensions_conf",
        layer="bind",
        path="params_training.conf",
        anchor=r"^dimensions_allowed",
        title="The allowed island dimensions themselves",
        change=(
            "Set to the new domain's dimensions. Pick a range wide enough that migration "
            "between different dimensions is actually exercised - a single entry turns the "
            "framework back into a homogeneous island model."
        ),
        contract=(
            "Changing this invalidates every stored `elitist_*.json`: a trained elitist's "
            "IRACE-tuned `dimension` categorical may no longer be in the set. Retrain."
        ),
    ),
    Site(
        key="train_binding",
        layer="bind",
        path="hm-ibf-hao/src/main.rs",
        anchor=r"HorizontalAlignment::load_instances",
        title="Train stage type binding",
        change=(
            "Swap the concrete problem, evaluator and transformer. `training::run` itself is "
            "generic and needs no change."
        ),
    ),
    Site(
        key="eval_elitist_binding",
        layer="bind",
        path="hm-ibf-hao/src/evaluation/mod.rs",
        anchor=r"load_elitist::<HorizontalAlignment>",
        title="Evaluate stage type binding",
        change="Swap the concrete problem type; `load_elitist` is generic.",
    ),
    Site(
        key="eval_dimension_gate",
        layer="bind",
        path="hm-ibf-hao/src/evaluation/mod.rs",
        anchor=r"dimensions_allowed\.contains",
        title="Exported-dimension gate",
        change=(
            "Keep it. It asserts the exported best individual's length is an allowed island "
            "dimension - the end-to-end proof that the transform chain preserved lengths."
        ),
        contract=(
            "Do not relax this to a warning. It is the check that catches a tau which returned "
            "the wrong length."
        ),
    ),
    Site(
        key="eval_grouping",
        layer="bind",
        path="hm-ibf-hao/src/evaluation/mod.rs",
        anchor=r"summary\.natural_dimension = ",
        title="Result grouping axis of the summary",
        change=(
            "Replace `natural_dimension` with the new domain's instance-difficulty axis. It "
            "becomes a column of the summary CSV."
        ),
    ),
    Site(
        key="export_payload",
        layer="bind",
        path="hm-ibf-hao/src/alignment/output.rs",
        anchor=r"pub struct RunMetadata",
        title="results.json payload",
        change=(
            "Replace `natural_dimension` and `backbone_length` with the new domain's "
            "descriptors. Keep `solution_dim`, `x`, `best_value` and `n_evals` - downstream "
            "analysis reads them."
        ),
    ),
    Site(
        key="export_schema",
        layer="bind",
        path="hm-ibf-hao/src/alignment/output.rs",
        anchor=r"^pub const OUTPUT_EXPORT_SCHEMA_VERSION",
        title="Export schema version",
        change="Bump it. Any change to the payload above changes the schema.",
    ),
    Site(
        key="export_validation",
        layer="bind",
        path="hm-ibf-hao/src/alignment/output.rs",
        anchor=r"pub fn select_global_best_solution",
        title="Global-best extraction and its validity checks",
        change=(
            "Point the encoding check at the new block-count decoder, if the new domain has "
            "one. Keep the finite-objective and non-empty checks."
        ),
        contract=(
            "The exported `x` is the verbatim global best individual - no output projection is "
            "applied. If the new domain needs one, that is a design change, not a retarget: "
            "`OUTPUT_TRANSFORM_NOT_APPLIED` and the export doc comments claim otherwise."
        ),
    ),
    Site(
        key="cli_defaults",
        layer="bind",
        path="hm-ibf-hao/src/cli.rs",
        anchor=r"^pub const DEFAULT_SUMMARY_CSV",
        title="CLI defaults and naming",
        change=(
            "Rename the run directory, summary file and binary naming to the new domain. "
            "Cosmetic, but these strings appear in every runbook command."
        ),
    ),
    Site(
        key="smoke_fixture",
        layer="bind",
        path="hm-ibf-hao/tests/smoke.rs",
        anchor=r"^fn smoke_config",
        title="The end-to-end smoke fixture",
        change=(
            "Rebuild the synthetic instance from the new descriptor and re-tune the elitist "
            "params to an island the new set actually has. This is the test that proves the "
            "evaluate stage still runs end to end; a retarget that leaves it stale removes "
            "the only executable check of the wiring."
        ),
    ),
    # ------------------------------------------------------------------- keep
    Site(
        key="keep_training",
        layer="keep",
        path="hm-ibf-hao/src/training.rs",
        anchor=r"^pub fn run<P, O>",
        title="GRAHF structure search",
        change="No change. Already generic over the problem and its evaluator.",
    ),
    Site(
        key="keep_island_builders",
        layer="keep",
        path="hm-ibf-hao/src/islands/mod.rs",
        anchor=r"^pub fn island_builders<P",
        title="Island builder set and the node-weight encoding",
        change=(
            "No change. Adding, removing or reordering a builder shifts the node weight "
            "encoding and invalidates every stored elitist - that is an island-set change, a "
            "separate task from retargeting the problem. PSO is absent by design: its velocity "
            "and personal best have no meaning across a dimension change."
        ),
    ),
    Site(
        key="keep_dimension_trait",
        layer="keep",
        path="hm-ibf-hao/src/problems/mod.rs",
        anchor=r"^pub trait DimensionAwareDomain",
        title="The dimension-aware domain trait",
        change=(
            "No change. Implement it for the new problem instead; read the coherence note "
            "before considering a blanket impl."
        ),
    ),
    Site(
        key="keep_migrations",
        layer="keep",
        path="hm-ibf-hao/src/migrations/mod.rs",
        anchor=r"pub fn migration_builders",
        title="Migration policy set and the edge-weight encoding",
        change="No change. Domain-independent: selection, condition and replacement policies.",
    ),
    Site(
        key="keep_safe_operators",
        layer="keep",
        path="hm-ibf-hao/src/islands/safe_de.rs",
        anchor=r"pub struct SafeDEBinomialCrossover",
        title="Dimension-safe operators",
        change=(
            "No change. These exist precisely because they read `solution.len()` instead of "
            "`problem.dimension()`. A retarget that reintroduces a global dimension read here "
            "breaks mixed-dimension populations."
        ),
    ),
)

RESIDUES: tuple[Residue, ...] = (
    Residue(
        key="clothoid_decoder",
        pattern=r"clothoid|fresnel|Fresnel|curvature_radius|CURVATURE_PENALTY",
        severity="high",
        message=(
            "The clothoid decoder is still referenced. The objective is still measuring a road "
            "alignment: its length, its curvature violations, or both."
        ),
    ),
    Residue(
        key="terrain_objective",
        pattern=r"heightmap|height_limit|tunnel_factor|gradient_factor|weighted_length",
        severity="high",
        message=(
            "The terrain cost model survives. `f(x)` is still charging tunnels and earthworks, "
            "so the reported values describe a corridor rather than the new domain."
        ),
    ),
    Residue(
        key="offset_backbone",
        pattern=r"backbone_offset_bounds|backbone_normals|perpendicular offset|\boffsets\b",
        severity="high",
        message=(
            "The perpendicular-offset encoding is still live. Solutions are still being read as "
            "displacements of a fixed route along its local normals - which is only meaningful "
            "if the new domain really is a corridor."
        ),
    ),
    Residue(
        key="route_vocabulary",
        pattern=r"natural_dimension|path_astar|path_simplified|Douglas.?Peucker",
        severity="note",
        message=(
            "Route vocabulary survives in the export payload, the summary grouping or the "
            "instance schema. Harmless at runtime, but the results no longer describe what was "
            "optimized."
        ),
    ),
    Residue(
        key="hao_naming",
        pattern=r"HorizontalAlignment|AlignmentEvaluator|AlignmentConfig|hao_run|results_hao",
        severity="note",
        message=(
            "Alignment naming survives in types, defaults or paths. Cosmetic - but check "
            "`params_training.conf`, the runbook and `run.sh` for stale command lines."
        ),
    ),
    Residue(
        key="global_dimension_read",
        pattern=r"problem\.dimension\(\)",
        severity="note",
        message=(
            "A global dimension read. Legitimate in `VectorProblem::dimension` itself and in "
            "the documented island fallbacks; anywhere else it breaks islands running below "
            "the maximum. Read each hit before acting."
        ),
    ),
    Residue(
        key="unbumped_export_schema",
        pattern=r"OUTPUT_EXPORT_SCHEMA_VERSION: u32 = 1",
        severity="note",
        message=(
            "The export schema is still at the alignment version 1 while the payload changed. "
            "Bump it so downstream readers can tell the formats apart."
        ),
    ),
)
