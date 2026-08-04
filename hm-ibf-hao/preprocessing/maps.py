"""Catalog of the three benchmark terrain maps and their fixed route endpoints.

The horizontal alignment benchmark is defined on three European regions whose elevation
data is published by the upstream `HorizAligns-Hybrid-Optimization` study. Each entry below
mirrors that study's own ``conf_example`` configuration verbatim — the source heightmap, the
resolution it is downsampled to, and the start/goal cells of the route — so that the
instances generated here describe the same three routes the baseline was measured on.

The elevation data itself is not part of this repository; :mod:`preprocessing.fetch_maps`
downloads it on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Repository the elevation data is published in.
UPSTREAM_REPOSITORY = "https://github.com/Steigner/HorizAligns-Hybrid-Optimization"

Cell = tuple[int, int]


@dataclass(frozen=True)
class BenchmarkMap:
    """One benchmark region, with the route the alignment is optimized along.

    Attributes:
        name: Short region name; the stem of every instance generated from this map.
        filename: Name of the source `.npy` heightmap in the upstream repository.
        source_shape: `(rows, columns)` the upstream heightmap has, used to verify a
            download landed intact.
        resolution: `(rows, columns)` the heightmap is downsampled to before routing.
        start: Route start, in downsampled heightmap cells.
        goal: Route goal, in downsampled heightmap cells.
    """

    name: str
    filename: str
    source_shape: tuple[int, int]
    resolution: tuple[int, int]
    start: Cell
    goal: Cell


#: The three benchmark regions, in the order their instances are generated.
BENCHMARK_MAPS: tuple[BenchmarkMap, ...] = (
    BenchmarkMap(
        name="austria",
        filename="austria_2000_6000_heightmap_wo_norm.npy",
        source_shape=(2000, 6000),
        resolution=(200, 600),
        start=(50, 140),
        goal=(140, 460),
    ),
    BenchmarkMap(
        name="italia",
        filename="italia_4000_6000_heightmap_wo_norm.npy",
        source_shape=(4000, 6000),
        resolution=(400, 600),
        start=(130, 160),
        goal=(225, 425),
    ),
    BenchmarkMap(
        name="slovenia",
        filename="slovenia_2000_6000_heightmap_wo_norm.npy",
        source_shape=(2000, 6000),
        resolution=(200, 400),
        start=(175, 0),
        goal=(160, 290),
    ),
)

#: Every catalog name, for command line choices.
MAP_NAMES: tuple[str, ...] = tuple(entry.name for entry in BENCHMARK_MAPS)


def map_by_name(name: str) -> BenchmarkMap:
    """Look a benchmark map up by its region name.

    Args:
        name: The region name, one of :data:`MAP_NAMES`.

    Returns:
        The catalog entry.

    Raises:
        KeyError: If no map carries that name.
    """
    for entry in BENCHMARK_MAPS:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown benchmark map {name!r}; known maps are {', '.join(MAP_NAMES)}")


def instance_name(entry: BenchmarkMap, epsilon: float) -> str:
    """Return the instance name a map generates at a given simplification tolerance.

    The tolerance is part of the name because it decides the route's natural dimension, and
    therefore which working dimensions the islands are recommended to run at.

    Args:
        entry: The benchmark map.
        epsilon: The Douglas-Peucker tolerance the instance was generated with.

    Returns:
        The instance name, e.g. `austria_eps1.0`.
    """
    return f"{entry.name}_eps{epsilon}"
