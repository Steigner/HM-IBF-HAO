"""Backbone construction: equidistant resampling, normals and offset bounds."""

from __future__ import annotations

from typing import Any

import numpy as np

#: Lengths below this are treated as zero when normalizing a direction.
_DEGENERATE_LENGTH = 1e-10

#: Fallback normal for a stretch of route with no discernible direction.
_FALLBACK_NORMAL = np.array([0.0, 1.0])


def build(
    route: list[tuple[int, int]],
    heightmap_shape: tuple[int, int],
    cutting_plane_factor: float,
    backbone_step: float,
) -> dict[str, Any]:
    """Build the backbone of a route.

    The backbone is resampled from the raw route rather than from its simplification, so it
    follows the terrain faithfully; equal arc-length spacing then gives the Rust benchmark a
    parametrization it can subsample at any working dimension.

    Args:
        route: The raw route, as pixel coordinates on the downsampled heightmap.
        heightmap_shape: `(rows, columns)` of the downsampled heightmap.
        cutting_plane_factor: Multiplier applied to the perpendicular offset bounds.
        backbone_step: Arc length between consecutive backbone points, in pixels.

    Returns:
        A dict with the `points`, `cumulative_distances`, `total_length`, `normals` and
        `offset_bounds` of the backbone, ready to be serialized into an instance config.

    Raises:
        ValueError: If the route has fewer than two points or no length at all.
    """
    if len(route) < 2:
        raise ValueError(f"a backbone needs at least two route points, got {len(route)}")

    points = np.asarray(route, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError("the route has no length; start and goal may coincide")

    count = max(2, int(total_length / backbone_step) + 1)
    distances = np.linspace(0.0, total_length, count)
    resampled = np.column_stack(
        [
            np.interp(distances, cumulative, points[:, 0]),
            np.interp(distances, cumulative, points[:, 1]),
        ]
    )

    normals = compute_normals(resampled)
    bounds = compute_offset_bounds(resampled, normals, heightmap_shape, cutting_plane_factor)

    return {
        "points": resampled.tolist(),
        "cumulative_distances": distances.tolist(),
        "total_length": total_length,
        "normals": normals.tolist(),
        "offset_bounds": bounds,
    }


def compute_normals(points: np.ndarray) -> np.ndarray:
    """Compute the unit normal at each backbone point.

    The tangent comes from a central difference, or a one-sided difference at the ends; the
    normal is that tangent rotated a quarter turn counter-clockwise.

    Args:
        points: The backbone points, shaped `(n, 2)`.

    Returns:
        The unit normals, shaped `(n, 2)`.
    """
    count = len(points)
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if count > 2:
        tangents[1:-1] = points[2:] - points[:-2]

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    degenerate = lengths < _DEGENERATE_LENGTH

    normals[~degenerate] /= lengths[~degenerate, None]
    normals[degenerate] = _FALLBACK_NORMAL
    return normals


def compute_offset_bounds(
    points: np.ndarray,
    normals: np.ndarray,
    heightmap_shape: tuple[int, int],
    cutting_plane_factor: float,
) -> list[list[float]]:
    """Compute how far each backbone point may be displaced along its normal.

    The nominal reach is the mean segment length scaled by `cutting_plane_factor`; it is
    then clipped so the displaced point stays inside the heightmap.

    Args:
        points: The backbone points, shaped `(n, 2)`.
        normals: The unit normals, shaped `(n, 2)`.
        heightmap_shape: `(rows, columns)` of the downsampled heightmap.
        cutting_plane_factor: Multiplier applied to the nominal reach.

    Returns:
        One `[min_offset, max_offset]` pair per backbone point.
    """
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean_segment = float(np.mean(segment_lengths)) if len(segment_lengths) else 1.0
    reach = mean_segment * cutting_plane_factor

    return [
        [
            -_clip_reach(point, normal, reach, heightmap_shape, positive=False),
            _clip_reach(point, normal, reach, heightmap_shape, positive=True),
        ]
        for point, normal in zip(points, normals, strict=True)
    ]


def _clip_reach(
    point: np.ndarray,
    normal: np.ndarray,
    reach: float,
    heightmap_shape: tuple[int, int],
    *,
    positive: bool,
) -> float:
    """Shorten a displacement until it stays inside the heightmap.

    Args:
        point: The backbone point.
        normal: Its unit normal.
        reach: The nominal displacement.
        heightmap_shape: `(rows, columns)` of the heightmap.
        positive: Whether to displace along the normal or against it.

    Returns:
        The largest admissible displacement, never negative.
    """
    sign = 1.0 if positive else -1.0
    end = point + sign * reach * normal
    limit = reach

    for axis, extent in enumerate(heightmap_shape):
        if abs(normal[axis]) <= _DEGENERATE_LENGTH:
            continue
        if end[axis] < 0.0:
            limit = min(limit, abs(point[axis] / normal[axis]))
        elif end[axis] >= extent:
            limit = min(limit, abs((extent - 1 - point[axis]) / normal[axis]))

    return max(0.0, float(limit))
