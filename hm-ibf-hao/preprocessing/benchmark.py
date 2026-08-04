"""Generation of the three checked-in benchmark instances from the named terrain maps.

This is the path that reproduces `instances/`. Unlike the sampled generator, which draws
random start/goal pairs to build a larger train/eval split, every instance here comes from a
fixed region and a fixed pair of endpoints (:mod:`preprocessing.maps`), so regenerating the
set yields exactly the instances the benchmark ships with.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import astar, dimensions, instance
from preprocessing.config import PreprocessingConfig
from preprocessing.fetch_maps import DEFAULT_BASE_URL, DEFAULT_CACHE_DIR, ensure_heightmap
from preprocessing.maps import BENCHMARK_MAPS, BenchmarkMap, instance_name, map_by_name
from preprocessing.sampling import Candidate

LOGGER = logging.getLogger(__name__)


def build_map_instance(
    entry: BenchmarkMap,
    heightmap: np.ndarray,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Route one benchmark map and assemble its instance payload.

    Args:
        entry: The benchmark map, supplying the fixed start and goal.
        heightmap: The already downsampled terrain of that map.
        config: The generator configuration supplying the cost parameters.

    Returns:
        The instance payload, matching the `AlignmentConfig` the Rust benchmark reads.

    Raises:
        RuntimeError: If no route exists between the map's start and goal.
        ValueError: If the endpoints lie outside the heightmap or the route cannot form a
            backbone.
    """
    route = astar.find_route(heightmap, entry.start, entry.goal)
    if not route:
        raise RuntimeError(
            f"no route between {entry.start} and {entry.goal} on {entry.name}; "
            "the terrain or the resolution does not match the catalog"
        )

    candidate = Candidate(
        start=entry.start,
        goal=entry.goal,
        route=route,
        length=astar.route_length(route),
        # The named maps are not bucketed by length: each one is its own instance.
        bucket=0,
    )
    return instance.build_config(
        candidate,
        heightmap.shape,
        instance_name(entry, config.epsilon),
        config,
    )


def generate_benchmark(
    output_dir: Path,
    config: PreprocessingConfig,
    names: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_url: str = DEFAULT_BASE_URL,
) -> list[int]:
    """Generate the benchmark instance set and write it to disk.

    Args:
        output_dir: Directory receiving the instances and their summary.
        config: The generator configuration; only `epsilon`, `backbone_step`,
            `cutting_plane_factor`, the cost parameters and `tau` are read, because the
            resolution and the endpoints come from the map catalog.
        names: Region names to generate; every catalog entry when omitted.
        cache_dir: Directory the source heightmaps are cached in.
        base_url: Base URL the source heightmaps are downloaded from.

    Returns:
        The recommended working dimensions, derived from the generated pool.

    Raises:
        KeyError: If a requested region name is not in the catalog.
        MapFetchError: If a source heightmap cannot be downloaded.
        RuntimeError: If a map's start and goal cannot be connected.
    """
    entries = [map_by_name(name) for name in names] if names else list(BENCHMARK_MAPS)

    configs: list[dict[str, Any]] = []
    for entry in entries:
        source = ensure_heightmap(entry, cache_dir, base_url)
        LOGGER.info("routing %s at resolution %s", entry.name, entry.resolution)
        heightmap = astar.downsample(np.load(source), *entry.resolution)

        instance_config = build_map_instance(entry, heightmap, config)
        instance.write_instance(instance_config, heightmap, output_dir)
        configs.append(instance_config)

    pool = [payload["natural_dimension"] for payload in configs]
    dimensions_allowed = dimensions.select_dimensions_allowed(pool, config.n_dimensions_allowed)
    instance.write_summary(output_dir, configs, config.seed, dimensions_allowed)

    LOGGER.info("natural dimensions from Douglas-Peucker (eps=%s): %s", config.epsilon, pool)
    LOGGER.info(
        "paste into params_training.conf: %s",
        dimensions.format_toml_setting(dimensions_allowed),
    )
    return dimensions_allowed
