"""Sampling of start/goal pairs and the disjoint train/eval split."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from preprocessing import astar

LOGGER = logging.getLogger(__name__)

#: Rejection-sampling attempts allowed per requested pair before giving up.
_ATTEMPTS_PER_PAIR = 1000

Cell = tuple[int, int]


@dataclass(frozen=True)
class Candidate:
    """One sampled route, ready to be turned into an instance.

    Attributes:
        start: The route's start cell.
        goal: The route's goal cell.
        route: The least-cost route between them.
        length: The route's arc length.
        bucket: Index of the length bucket the route was assigned to.
    """

    start: Cell
    goal: Cell
    route: list[Cell]
    length: float
    bucket: int


def sample_start_goal_pairs(
    shape: tuple[int, int],
    rng: np.random.Generator,
    min_distance: float,
    count: int,
) -> list[tuple[Cell, Cell]]:
    """Rejection-sample start/goal pairs that are far enough apart.

    Args:
        shape: `(rows, columns)` of the heightmap.
        rng: The random generator to draw from.
        min_distance: Smallest admissible distance between a start and its goal.
        count: Number of pairs to sample.

    Returns:
        The sampled pairs.

    Raises:
        ValueError: If `count` is negative.
        RuntimeError: If the map is too small to yield `count` pairs.
    """
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")

    rows, columns = shape
    pairs: list[tuple[Cell, Cell]] = []
    attempts = 0
    max_attempts = max(1, count) * _ATTEMPTS_PER_PAIR

    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        start = (int(rng.integers(0, rows)), int(rng.integers(0, columns)))
        goal = (int(rng.integers(0, rows)), int(rng.integers(0, columns)))
        if astar.euclidean_distance(start, goal) >= min_distance:
            pairs.append((start, goal))

    if len(pairs) < count:
        raise RuntimeError(
            f"only sampled {len(pairs)}/{count} start/goal pairs at least "
            f"{min_distance:.1f} px apart after {attempts} attempts; lower "
            f"--min-distance-fraction or use a larger heightmap"
        )

    return pairs


def bucket_by_length(
    pairs: list[tuple[Cell, Cell]],
    heightmap: np.ndarray,
    n_buckets: int,
) -> list[Candidate]:
    """Route every pair and spread the results over length buckets.

    Bucket boundaries are quantiles of the whole pool, so every bucket is non-empty and the
    train and eval sets end up drawing from comparable length distributions.

    Args:
        pairs: The start/goal pairs to route.
        heightmap: The terrain to route across.
        n_buckets: Number of buckets to spread the routes over.

    Returns:
        One candidate per pair that could be routed, in input order.

    Raises:
        ValueError: If `n_buckets` is not positive.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be at least 1, got {n_buckets}")

    routed: list[tuple[Cell, Cell, list[Cell], float]] = []
    for start, goal in pairs:
        route = astar.find_route(heightmap, start, goal)
        if not route:
            LOGGER.warning("no route between %s and %s; skipping", start, goal)
            continue
        routed.append((start, goal, route, astar.route_length(route)))

    if not routed:
        return []

    lengths = np.array([length for *_, length in routed])
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    boundaries = np.quantile(lengths, quantiles) if quantiles.size else np.array([])

    return [
        Candidate(
            start=start,
            goal=goal,
            route=route,
            length=length,
            bucket=int(np.searchsorted(boundaries, length, side="right")),
        )
        for start, goal, route, length in routed
    ]


def split_train_eval(
    candidates: list[Candidate],
    n_train: int,
    n_eval: int,
    rng: np.random.Generator,
) -> tuple[list[Candidate], list[Candidate]]:
    """Partition candidates into disjoint train and eval sets.

    Both sets are drawn evenly from every length bucket, so neither ends up specialized to
    short or to long routes.

    Args:
        candidates: The bucketed candidates.
        n_train: Number of training instances required.
        n_eval: Number of evaluation instances required.
        rng: The random generator used to shuffle.

    Returns:
        The training and the evaluation candidates.

    Raises:
        ValueError: If there are no candidates to split.
        RuntimeError: If a bucket holds too few candidates to satisfy both sets.
    """
    if not candidates:
        raise ValueError("there are no candidates to split")

    n_buckets = max(candidate.bucket for candidate in candidates) + 1
    buckets: list[list[Candidate]] = [[] for _ in range(n_buckets)]
    for candidate in candidates:
        buckets[candidate.bucket].append(candidate)
    for bucket in buckets:
        rng.shuffle(bucket)

    train: list[Candidate] = []
    evaluation: list[Candidate] = []
    for index, bucket in enumerate(buckets):
        # The remainder goes to the earliest buckets so the totals come out exact.
        train_count = n_train // n_buckets + (1 if index < n_train % n_buckets else 0)
        eval_count = n_eval // n_buckets + (1 if index < n_eval % n_buckets else 0)
        needed = train_count + eval_count
        if len(bucket) < needed:
            raise RuntimeError(
                f"bucket {index} holds {len(bucket)} candidates but {needed} are needed "
                f"(train={train_count}, eval={eval_count}); raise --oversample-factor"
            )
        train.extend(bucket[:train_count])
        evaluation.extend(bucket[train_count:needed])

    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation
