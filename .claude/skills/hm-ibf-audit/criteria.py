"""The HM-IBF knowledge base: the traits to look for and the anti-patterns to report.

This module is pure data. Adding a trait means appending a :class:`~model.Criterion` here;
the scanning engine in :mod:`audit` needs no change.

The traits follow the HM-IBF definition: an island model whose topology is a labelled
directed graph, whose neighbouring islands may work at different dimensions, and whose
migrants are therefore re-expressed by a transformation along a shared geometric backbone
rather than copied index-to-index.

Formally a design is the tuple `(G, h, mu, D, tau)`: the digraph, the solver on each node,
the migration policy on each edge, the *per-island* working dimension, and the
transformation that reconciles `D(u)` with `D(v)` before insertion. When `D` is constant
every `tau` collapses to the identity and the homogeneous island model is recovered as the
special case — which is exactly what the `heterogeneous_migration` and `backbone` criteria
below are there to tell apart.
"""

from __future__ import annotations

from model import Criterion, Deviation, Signal

CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="topology",
        title="C1  Topology is a labelled directed graph",
        question="Are islands nodes of a digraph whose edges are migration channels?",
        min_strong=2,
        signals=(
            Signal("directed graph type", r"\bDiGraph\b|petgraph|nx\.DiGraph|networkx"),
            Signal("node / edge weight labelling", r"node_weight|edge_weight"),
            Signal("island catalog indexed by node weight", r"island_builders|ISLAND_[A-Z]+"),
            Signal("migration policy catalog on edges", r"migration_builders|MigrationBuilder"),
        ),
    ),
    Criterion(
        key="heterogeneous_migration",
        title="C2  Heterogeneous migration (tau applied on the edge)",
        question="Is a migrant re-expressed when D(source) != D(target)?",
        min_strong=3,
        signals=(
            Signal(
                "transform hook in the migration path",
                r"SolutionTransformer|TransformRequest|MigrationTransformer",
            ),
            Signal("source/target dimension carried by the edge", r"source_dim|target_dim"),
            Signal(
                "identity bypass when the dimensions agree",
                r"is_identity|source_dim\w*\s*==\s*target_dim\w*",
            ),
            Signal("migrants marked unevaluated and re-evaluated", r"new_unevaluated|unevaluated"),
        ),
    ),
    Criterion(
        key="backbone",
        title="C3  Backbone as a shared geometric reference",
        question="Is migration resampled along a normalized arc-length coordinate?",
        min_strong=3,
        signals=(
            Signal("backbone reference object", r"\bbackbone\b"),
            Signal("arc-length parameterisation", r"arc.?length|cumulative"),
            Signal(
                "projection of a sample onto the reference",
                r"project_t|project_onto|sample_backbone|domain_for_dimension",
            ),
            Signal(
                "coordinate normalized to [0, 1]",
                r"clamp\(0\.0,\s*1\.0\)|\[0,\s*1\]|normali[sz]ed position",
            ),
            # The contract that makes two dimensions describe the same object: sample `i`
            # of a `D`-dimensional solution sits at `t_i = (i+1)/(D+1)` along the backbone,
            # so a resample is a change of density rather than a change of meaning.
            Signal(
                "resolution-independent sample position",
                r"\(i \+ 1\) / \(D \+ 1\)|i\+1\)/\(D\+1|i as f64 \* .*/ \(dimension as f64 \+ 1"
                r"|s_i = i|t_i = ",
            ),
        ),
    ),
    Criterion(
        key="transform_catalog",
        title="C4  Swappable transformation catalog",
        question="Is the interpolation/smoothing method a tunable parameter?",
        min_strong=4,
        signals=(
            Signal("linear interpolation", r"resample_signal|linear interpolat"),
            Signal("PCHIP", r"\bPCHIP\b|\bPchip\b"),
            Signal("Akima", r"\bAkima\b"),
            Signal("clamped cubic spline", r"CT_Spline|CtSpline|clamped.?cubic|CubicSpline"),
            Signal("Douglas-Peucker simplification", r"Douglas.?Peucker"),
            Signal("Whittaker-Eilers / V-spline", r"Whittaker|VSpline|V-spline"),
            Signal("total-variation denoising (ADMM)", r"TVDenoise|total.?variation|\bADMM\b"),
        ),
    ),
    Criterion(
        key="dimension_agnostic",
        title="C5  Dimension-agnostic islands and operators",
        question="Is the dimension per-island rather than a global constant?",
        min_strong=3,
        signals=(
            Signal(
                "per-island working dimension",
                r"IslandDimension|island_dimension|DIMENSIONS_ALLOWED|dimensions_allowed"
                r"|working dimension",
            ),
            Signal(
                "the dimension is a tuned categorical, not a constant",
                r"with_categorical\(\s*\"dimension\"|\"dimension\":|categorical.*dimension",
            ),
            Signal(
                "operators read the individual's own length",
                r"solution\.len\(\)|len\(solution\)|child_solution\.len\(\)",
            ),
            Signal("bounds derived per working dimension", r"domain_for_dimension|bounds_for_dim"),
            Signal(
                "explicitly dimension-safe component variants",
                r"Safe[A-Z]\w+|IslandDimensionSaturation|safe_de|safe_boundary",
            ),
        ),
    ),
    Criterion(
        key="two_level_design",
        title="C6  Two-level automatic design",
        question="Is architecture search separated from hyperparameter tuning?",
        min_strong=3,
        signals=(
            Signal("lower-level tuner", r"\bIRACE\b|\birace\b|optuna|\bSMAC\b|smac3"),
            Signal("upper-level architecture search", r"algorithm_design|hyper.?heuristic"),
            Signal("graph variation operators", r"recombination|graph.*mutation|binomial.*split"),
            Signal(
                "cross-instance normalization (z-score / MAD)",
                r"z.?score|median absolute deviation|standardi[sz]|median_deviation",
            ),
            # The two levels must not optimize the same quantity: the tuner minimizes the
            # raw median best objective, the architecture search the z-score of it.
            Signal(
                "the tuner scores the raw objective, not the normalized one",
                r"MedianBestObjectiveValue|median best objective|performance_measure",
            ),
        ),
    ),
    Criterion(
        key="freeze_inference",
        title="C7  Freeze into a deterministic inference solver",
        question="Can the elitist architecture run without the design loop?",
        min_strong=2,
        signals=(
            Signal("elitist architecture artefact", r"load_elitist|read_builder_graph|elitist"),
            Signal("tuned parameters restored from disk", r"read_params|\.params\b"),
            Signal("standalone evaluation stage", r"\bevaluate\b|inference|frozen"),
        ),
    ),
)

DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        key="stateful_pso",
        pattern=r"particle_swarm|\bPsoIsland\b|\bvelocity\b|personal_best|\bpbest\b",
        trigger="present",
        severity="high",
        message=(
            "PSO-style auxiliary state (velocity / personal best) is present. Those quantities "
            "are meaningless once an individual changes dimension: a velocity recorded at one "
            "resolution has no component-wise correspondent at another. A 'safe' variant that "
            "zeroes the velocity and collapses the personal best after every migration runs "
            "without out-of-bounds access but loses exactly the inertia and memory that "
            "distinguish PSO from a random restart. Either exclude PSO from the "
            "dimension-agnostic island set, or resize/reset its state inside the migration path."
        ),
    ),
    Deviation(
        key="hardcoded_dimension_set",
        pattern=r"(?:const|static)\s+DIMENSIONS_ALLOWED|DIMENSIONS_ALLOWED\s*:\s*\[u32",
        trigger="present",
        severity="note",
        message=(
            "The set of allowed island dimensions is compiled in rather than configured. It is "
            "a property of the instance pool - for a route it comes from the compression of the "
            "raw path - so it changes whenever the instances do, and a stored elitist's tuned "
            "`dimension` categorical becomes invalid with it. Prefer a settings file the "
            "loader validates."
        ),
    ),
    Deviation(
        key="no_identity_bypass",
        pattern=r"is_identity|source_dim\w*\s*==\s*target_dim\w*",
        trigger="absent",
        severity="high",
        message=(
            "No short-circuit for equal source and target dimension was found, so every "
            "migration pays the transform cost and homogeneous edges are not exact copies."
        ),
    ),
    Deviation(
        key="index_resampling",
        pattern=r"arc.?length|project_t|\bbackbone\b",
        trigger="absent",
        severity="high",
        message=(
            "No arc-length / backbone parameterisation was found. Migration most likely copies "
            "index-to-index, which misaligns samples that represent the same object at "
            "different resolutions."
        ),
    ),
    Deviation(
        key="unwrap_missing",
        pattern=r"unwrap_angles|np\.unwrap|phase.?unwrap",
        trigger="absent",
        severity="note",
        message=(
            "No phase unwrapping found. If the encoded values are periodic or bounded (angles), "
            "resampling the wrapped signal introduces artificial jumps at the wrap boundary."
        ),
    ),
    Deviation(
        key="optional_transformer",
        pattern=(
            r"Option<Arc<dyn SolutionTransformer|transformer:\s*Option<"
            r"|try_borrow::<MigrationTransformer\w*>\(\)\.ok\(\)"
        ),
        trigger="present",
        severity="high",
        message=(
            "The migration transform is optional, so a missing transformer degrades migration "
            "to a verbatim copy instead of failing. Across islands of differing dimension that "
            "silently inserts a wrong-length migrant and turns the framework back into the "
            "homogeneous special case. Read every construction site: if any path can build the "
            "island graph without a transformer while the islands may differ in dimension, make "
            "the transformer mandatory or assert on the mismatch."
        ),
    ),
    Deviation(
        key="unchecked_transform_length",
        pattern=(
            r"(?:assert|ensure|debug_assert|expect)\w*!?\s*\(?[^\n]*"
            r"(?:len\(\)\s*==\s*\w*target|target_dim\w*\s*==\s*[^\n]*len\(\))"
        ),
        trigger="absent",
        severity="high",
        message=(
            "Nothing asserts that the transform actually returned `target_dim` elements. The "
            "`SolutionTransformer` contract promises that length, but implementations typically "
            "return an empty vector for degenerate input, and an unchecked result is inserted "
            "into the target population as-is. Assert the post-transform length on the migration "
            "edge, where the violation is still attributable."
        ),
    ),
    Deviation(
        key="global_dimension_reads",
        pattern=r"problem\.dimension\(\)|self\.dimension\b",
        trigger="present",
        severity="note",
        message=(
            "A problem-level dimension is read somewhere. Confirm no variation or boundary "
            "operator uses it in place of the individual's own length; a documented fallback "
            "for islands that did not override the dimension is fine."
        ),
    ),
)
