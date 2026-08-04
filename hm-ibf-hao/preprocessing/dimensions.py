"""Recommendation of the working dimensions the islands are allowed to run at.

The recommendation is derived from the **Douglas-Peucker compression** of each instance's
raw route: the compression estimates how many interior inflection points a route really
needs (its `natural_dimension`, see :mod:`preprocessing.simplify`), and this module turns a
pool of those per-instance estimates into the `dimensions_allowed` list that
`params_training.conf` carries. Every island therefore searches at a resolution the terrain
itself calls for, rather than at an arbitrary one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_dimensions_allowed(natural_dimensions: Sequence[int], count: int) -> list[int]:
    """Pick representative working dimensions from a pool of instances.

    A pool no larger than `count` is taken verbatim, which is the case for the three
    benchmark maps: their Douglas-Peucker dimensions *are* the recommendation, so every
    island operates in the same native search space as the single-dimension reference
    methods. A larger pool is reduced to its minimum, its median and its largest distinct
    values, so the islands can still specialize anywhere between the simplest and the most
    intricate route. Duplicates are dropped, so the result may be shorter than `count`.

    Args:
        natural_dimensions: The natural dimension of every instance in the pool.
        count: Largest number of dimensions to recommend.

    Returns:
        The selected dimensions, sorted ascending and deduplicated.

    Raises:
        ValueError: If the pool is empty or `count` is not positive.
    """
    if not natural_dimensions:
        raise ValueError("the pool of natural dimensions is empty")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    distinct = sorted(set(natural_dimensions))
    if len(distinct) <= count:
        return distinct

    ordered = sorted(natural_dimensions)
    selected = {ordered[0], int(np.median(ordered))}

    n_largest = max(count - len(selected), 0)
    if n_largest:
        selected.update(distinct[-n_largest:])

    return sorted(selected)


def format_toml_setting(dimensions: Sequence[int]) -> str:
    """Render selected dimensions as the setting `params_training.conf` expects.

    Args:
        dimensions: The selected dimensions.

    Returns:
        A `dimensions_allowed` assignment, ready to paste into `params_training.conf`.
    """
    values = ", ".join(str(dimension) for dimension in dimensions)
    return f"dimensions_allowed = [{values}]"
