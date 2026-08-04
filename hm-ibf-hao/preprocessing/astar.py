"""Least-cost pathfinding across a terrain heightmap."""

from __future__ import annotations

import heapq
from collections.abc import Iterator

import numpy as np

#: The eight grid neighbours of a cell.
_NEIGHBOUR_OFFSETS = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

#: Base cost of entering a neighbouring cell, before the elevation change is added.
_STEP_COST = 1.0

Cell = tuple[int, int]


def downsample(heightmap: np.ndarray, target_rows: int, target_columns: int) -> np.ndarray:
    """Downsample a heightmap by averaging over rectangular blocks.

    The block size is the integer ratio between the source and the target shape, so the
    result is at least as large as the target but never interpolates between cells.

    Args:
        heightmap: The source heightmap.
        target_rows: Desired number of rows.
        target_columns: Desired number of columns.

    Returns:
        The downsampled heightmap.

    Raises:
        ValueError: If the heightmap is not two-dimensional or the target is not positive.
    """
    if heightmap.ndim != 2:
        raise ValueError(f"the heightmap must be 2-D, got shape {heightmap.shape}")
    if target_rows < 1 or target_columns < 1:
        raise ValueError(
            f"the target shape must be positive, got ({target_rows}, {target_columns})"
        )

    rows, columns = heightmap.shape
    row_factor = max(1, rows // target_rows)
    column_factor = max(1, columns // target_columns)
    new_rows = rows // row_factor
    new_columns = columns // column_factor

    trimmed = heightmap[: new_rows * row_factor, : new_columns * column_factor]
    blocks = trimmed.reshape(new_rows, row_factor, new_columns, column_factor)
    return blocks.mean(axis=(1, 3))


def neighbours(cell: Cell, shape: tuple[int, int]) -> Iterator[Cell]:
    """Yield the in-bounds grid neighbours of a cell.

    Args:
        cell: The cell to expand.
        shape: `(rows, columns)` of the grid.

    Yields:
        Each neighbouring cell that lies inside the grid.
    """
    row, column = cell
    for d_row, d_column in _NEIGHBOUR_OFFSETS:
        candidate = (row + d_row, column + d_column)
        if 0 <= candidate[0] < shape[0] and 0 <= candidate[1] < shape[1]:
            yield candidate


def euclidean_distance(a: Cell, b: Cell) -> float:
    """Return the straight-line distance between two cells.

    Args:
        a: The first cell.
        b: The second cell.

    Returns:
        The euclidean distance.
    """
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def step_cost(heightmap: np.ndarray, current: Cell, following: Cell) -> float:
    """Return the cost of moving between two adjacent cells.

    Climbing and descending cost the same: the route is penalized for elevation change,
    not for its direction.

    Args:
        heightmap: The terrain.
        current: The cell being left.
        following: The cell being entered.

    Returns:
        The step cost.
    """
    return float(abs(heightmap[following] - heightmap[current])) + _STEP_COST


def find_route(heightmap: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    """Find the least-cost route between two cells with A*.

    Args:
        heightmap: The terrain.
        start: The start cell.
        goal: The goal cell.

    Returns:
        The route from `start` to `goal` inclusive, or an empty list if none exists.

    Raises:
        ValueError: If either cell lies outside the heightmap.
    """
    shape = heightmap.shape
    for name, cell in (("start", start), ("goal", goal)):
        if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
            raise ValueError(f"{name} {cell} lies outside a heightmap of shape {shape}")

    frontier: list[tuple[float, Cell]] = [(0.0, start)]
    came_from: dict[Cell, Cell] = {}
    cost_so_far: dict[Cell, float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct(came_from, start, goal)

        for following in neighbours(current, shape):
            new_cost = cost_so_far[current] + step_cost(heightmap, current, following)
            if following not in cost_so_far or new_cost < cost_so_far[following]:
                cost_so_far[following] = new_cost
                heapq.heappush(
                    frontier, (new_cost + euclidean_distance(following, goal), following)
                )
                came_from[following] = current

    return []


def route_length(route: list[Cell]) -> float:
    """Return the arc length of a route.

    Args:
        route: The route, as a list of cells.

    Returns:
        The summed length of its segments; zero for a route of fewer than two cells.
    """
    if len(route) < 2:
        return 0.0

    points = np.asarray(route, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _reconstruct(came_from: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell]:
    """Walk the predecessor map back from the goal.

    Args:
        came_from: Predecessor of each visited cell.
        start: The cell the search began at.
        goal: The cell the search reached.

    Returns:
        The route from `start` to `goal` inclusive.
    """
    route = [goal]
    current = goal
    while current in came_from:
        current = came_from[current]
        route.append(current)
    route.reverse()
    if route[0] != start:
        route.insert(0, start)
    return route
