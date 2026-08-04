"""Command line entry point of the instance generator.

Two modes, selected by whether `--source` is given:

* **benchmark** (the default) regenerates the checked-in `instances/`: the three named
  terrain maps of :mod:`preprocessing.maps`, each routed between its own fixed endpoints.
  The source heightmaps are downloaded on demand by :mod:`preprocessing.fetch_maps`.
* **sampled** (`--source <terrain>.npy`) draws random start/goal pairs from one heightmap
  and splits the routes into a disjoint train/eval set, for evaluating on unseen routes.

Both modes print the `dimensions_allowed` recommendation derived from the Douglas-Peucker
compression of the generated routes. Run either from the crate directory::

    python3 -m preprocessing.prepare_instances
    python3 -m preprocessing.prepare_instances --source terrain.npy
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import astar, dimensions, instance
from preprocessing.benchmark import generate_benchmark
from preprocessing.config import PreprocessingConfig
from preprocessing.fetch_maps import DEFAULT_BASE_URL, DEFAULT_CACHE_DIR, MapFetchError
from preprocessing.maps import MAP_NAMES
from preprocessing.sampling import (
    Candidate,
    bucket_by_length,
    sample_start_goal_pairs,
    split_train_eval,
)

LOGGER = logging.getLogger("preprocessing")

#: Prefix of the sampled instance names, before the set name and the index.
_NAME_PREFIX = "route"


def build_parser(defaults: PreprocessingConfig | None = None) -> argparse.ArgumentParser:
    """Build the argument parser of the generator.

    Args:
        defaults: Configuration supplying the flag defaults; a fresh one when omitted.

    Returns:
        The parser.
    """
    defaults = defaults or PreprocessingConfig()
    parser = argparse.ArgumentParser(
        prog="prepare_instances",
        description="Generate horizontal alignment instances from terrain heightmaps.",
    )

    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "NumPy .npy heightmap to sample random routes from; "
            "omit it to regenerate the named benchmark maps instead"
        ),
    )

    benchmark = parser.add_argument_group("benchmark mode (no --source)")
    benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("instances"),
        help="directory receiving the benchmark instances (default: %(default)s)",
    )
    benchmark.add_argument(
        "--maps",
        nargs="+",
        choices=MAP_NAMES,
        default=list(MAP_NAMES),
        help="regions to generate (default: all)",
    )
    benchmark.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="directory the source heightmaps are cached in (default: %(default)s)",
    )
    benchmark.add_argument(
        "--maps-base-url",
        default=DEFAULT_BASE_URL,
        help="base URL the source heightmaps are downloaded from",
    )

    sampled = parser.add_argument_group("sampled mode (--source)")
    sampled.add_argument(
        "--train-dir",
        type=Path,
        default=Path("instances_train"),
        help="directory receiving the training instances (default: %(default)s)",
    )
    sampled.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("instances_eval"),
        help="directory receiving the evaluation instances (default: %(default)s)",
    )
    sampled.add_argument(
        "--n-train", type=int, default=defaults.n_train, help="number of training instances"
    )
    sampled.add_argument(
        "--n-eval", type=int, default=defaults.n_eval, help="number of evaluation instances"
    )
    sampled.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLUMNS"),
        default=list(defaults.target_resolution),
        help="shape the source heightmap is downsampled to; benchmark maps carry their own",
    )
    sampled.add_argument(
        "--min-distance-fraction",
        type=float,
        default=defaults.min_distance_fraction,
        help="smallest start-goal distance, as a fraction of the map diagonal",
    )
    sampled.add_argument(
        "--oversample-factor",
        type=int,
        default=defaults.oversample_factor,
        help="candidate pool size, as a multiple of the instances needed",
    )
    sampled.add_argument(
        "--n-buckets",
        type=int,
        default=defaults.n_buckets,
        help="number of route-length buckets the candidates are spread over",
    )

    parser.add_argument("--seed", type=int, default=defaults.seed, help="seed of the sampling RNG")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=defaults.epsilon,
        help="Douglas-Peucker tolerance used to estimate the natural dimension",
    )
    parser.add_argument(
        "--backbone-step",
        type=float,
        default=defaults.backbone_step,
        help="arc length between consecutive backbone points, in pixels",
    )
    parser.add_argument(
        "--cutting-plane-factor",
        type=float,
        default=defaults.cutting_plane_factor,
        help="multiplier applied to the perpendicular offset bounds",
    )
    parser.add_argument(
        "--n-dimensions-allowed",
        type=int,
        default=defaults.n_dimensions_allowed,
        help="largest number of working dimensions to recommend",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=defaults.tau,
        help="clothoid asymmetry parameter recorded in every instance",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> PreprocessingConfig:
    """Turn parsed arguments into a validated configuration.

    Args:
        args: The parsed arguments.

    Returns:
        The configuration.

    Raises:
        ValueError: If the arguments describe an ungeneratable instance set.
    """
    config = dataclasses.replace(
        PreprocessingConfig(),
        n_train=args.n_train,
        n_eval=args.n_eval,
        epsilon=args.epsilon,
        cutting_plane_factor=args.cutting_plane_factor,
        backbone_step=args.backbone_step,
        target_resolution=(int(args.resolution[0]), int(args.resolution[1])),
        min_distance_fraction=args.min_distance_fraction,
        oversample_factor=args.oversample_factor,
        n_buckets=args.n_buckets,
        n_dimensions_allowed=args.n_dimensions_allowed,
        seed=args.seed,
        tau=args.tau,
    )
    config.validate()
    return config


def load_heightmap(source: Path, config: PreprocessingConfig) -> np.ndarray:
    """Load a source heightmap and downsample it.

    Args:
        source: The `.npy` file holding the terrain.
        config: Supplies the target resolution.

    Returns:
        The downsampled heightmap.

    Raises:
        FileNotFoundError: If `source` does not exist.
        ValueError: If the file does not hold a two-dimensional array.
    """
    if not source.is_file():
        raise FileNotFoundError(f"no terrain heightmap at {source}; pass --source <heightmap.npy>")

    heightmap = np.load(source)
    LOGGER.info("loaded %s of shape %s", source, heightmap.shape)

    downsampled = astar.downsample(heightmap, *config.target_resolution)
    LOGGER.info("downsampled to %s", downsampled.shape)
    return downsampled


def generate(
    heightmap: np.ndarray,
    config: PreprocessingConfig,
    train_dir: Path,
    eval_dir: Path,
) -> list[int]:
    """Generate both sampled instance sets and write them to disk.

    Args:
        heightmap: The downsampled terrain to route across.
        config: The generator configuration.
        train_dir: Directory receiving the training instances.
        eval_dir: Directory receiving the evaluation instances.

    Returns:
        The recommended working dimensions, derived from the combined pool.

    Raises:
        RuntimeError: If the terrain yields too few routable candidates.
    """
    rng = np.random.default_rng(config.seed)
    min_distance = config.min_distance_fraction * float(np.hypot(*heightmap.shape))

    LOGGER.info(
        "sampling %d candidate start/goal pairs at least %.1f px apart",
        config.n_candidates,
        min_distance,
    )
    pairs = sample_start_goal_pairs(heightmap.shape, rng, min_distance, config.n_candidates)

    LOGGER.info("routing %d candidates and bucketing them by length", len(pairs))
    candidates = bucket_by_length(pairs, heightmap, config.n_buckets)
    train, evaluation = split_train_eval(candidates, config.n_train, config.n_eval, rng)
    LOGGER.info("split into %d train and %d eval instances", len(train), len(evaluation))

    train_configs = _write_set(train, heightmap, config, train_dir, "train")
    eval_configs = _write_set(evaluation, heightmap, config, eval_dir, "eval")

    pool = [payload["natural_dimension"] for payload in train_configs + eval_configs]
    dimensions_allowed = dimensions.select_dimensions_allowed(pool, config.n_dimensions_allowed)

    instance.write_summary(train_dir, train_configs, config.seed, dimensions_allowed)
    instance.write_summary(eval_dir, eval_configs, config.seed)

    LOGGER.info(
        "pool natural dimensions: min=%d median=%d max=%d",
        min(pool),
        int(np.median(pool)),
        max(pool),
    )
    LOGGER.info(
        "paste into params_training.conf: %s",
        dimensions.format_toml_setting(dimensions_allowed),
    )
    return dimensions_allowed


def main(argv: list[str] | None = None) -> int:
    """Run the generator in whichever mode the arguments select.

    Args:
        argv: Command line arguments; `sys.argv[1:]` when omitted.

    Returns:
        A process exit code: zero on success, one on a configuration, data or download
        error.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        config = config_from_args(args)
        if args.source is None:
            generate_benchmark(
                args.output_dir,
                config,
                args.maps,
                args.cache_dir,
                args.maps_base_url,
            )
        else:
            heightmap = load_heightmap(args.source, config)
            generate(heightmap, config, args.train_dir, args.eval_dir)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, MapFetchError) as error:
        LOGGER.error("%s", error)
        return 1

    return 0


def _write_set(
    candidates: list[Candidate],
    heightmap: np.ndarray,
    config: PreprocessingConfig,
    output_dir: Path,
    set_name: str,
) -> list[dict[str, Any]]:
    """Build and write every instance of one sampled set.

    Args:
        candidates: The candidates of the set.
        heightmap: The downsampled terrain.
        config: The generator configuration.
        output_dir: Directory receiving the instances.
        set_name: Name of the set, used in the instance names.

    Returns:
        The written instance payloads.
    """
    configs = []
    for index, candidate in enumerate(candidates):
        name = f"{_NAME_PREFIX}_{set_name}_{index:02d}"
        instance_config = instance.build_config(candidate, heightmap.shape, name, config)
        instance.write_instance(instance_config, heightmap, output_dir)
        configs.append(instance_config)
    return configs


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
