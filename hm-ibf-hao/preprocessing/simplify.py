"""Douglas-Peucker simplification, used to estimate a route's geometric complexity."""

from __future__ import annotations

import numpy as np

Point = tuple[int, int]


def point_line_distance(point: Point, start: Point, end: Point) -> float:
    """Return the perpendicular distance from a point to a line segment's carrier line.

    Args:
        point: The point to measure.
        start: The first endpoint of the segment.
        end: The second endpoint of the segment.

    Returns:
        The distance; the straight-line distance to `start` when the segment is degenerate.
    """
    if start == end:
        return float(np.hypot(point[0] - start[0], point[1] - start[1]))

    numerator = abs(
        (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
    )
    denominator = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    return float(numerator / denominator)


def simplify(points: list[Point], epsilon: float) -> list[Point]:
    """Simplify a polyline with the Douglas-Peucker algorithm.

    Args:
        points: The polyline to simplify.
        epsilon: Largest perpendicular deviation a dropped point may have.

    Returns:
        The simplified polyline, always keeping both endpoints.

    Raises:
        ValueError: If `epsilon` is not positive.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if len(points) < 3:
        return list(points)

    # Iterative rather than recursive: a raw route can be thousands of points long, which
    # is enough to exhaust the interpreter's stack.
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    pending = [(0, len(points) - 1)]

    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue

        furthest, distance = _furthest_point(points, first, last)
        if distance > epsilon:
            keep[furthest] = True
            pending.append((first, furthest))
            pending.append((furthest, last))

    return [point for point, kept in zip(points, keep, strict=True) if kept]


def natural_dimension(simplified: list[Point]) -> int:
    """Return the number of interior inflection points of a simplified route.

    The two endpoints are fixed boundary conditions of the alignment problem and are never
    optimized, so they do not count towards the route's complexity.

    Args:
        simplified: The simplified route.

    Returns:
        The interior point count, never below two.
    """
    return max(2, len(simplified) - 2)


def _furthest_point(points: list[Point], first: int, last: int) -> tuple[int, float]:
    """Find the interior point furthest from the chord between two indices.

    Args:
        points: The polyline.
        first: Index of the chord's first endpoint.
        last: Index of the chord's second endpoint.

    Returns:
        The index of the furthest interior point and its distance.
    """
    furthest = first
    largest = 0.0
    for index in range(first + 1, last):
        distance = point_line_distance(points[index], points[first], points[last])
        if distance > largest:
            furthest = index
            largest = distance
    return furthest, largest
