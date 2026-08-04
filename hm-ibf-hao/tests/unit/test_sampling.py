"""Unit tests for :mod:`preprocessing.sampling`."""

from __future__ import annotations

import numpy as np
import pytest
from preprocessing import astar, sampling
from preprocessing.sampling import Candidate


def make_candidates(counts: dict[int, int]) -> list[Candidate]:
    """Build placeholder candidates, `counts[bucket]` many per bucket."""
    candidates = []
    for bucket, count in counts.items():
        for index in range(count):
            candidates.append(
                Candidate(
                    start=(0, index),
                    goal=(1, index),
                    route=[(0, index), (1, index)],
                    length=float(bucket * 100 + index),
                    bucket=bucket,
                )
            )
    return candidates


def test_sampled_pairs_respect_the_minimum_distance(rng):
    pairs = sampling.sample_start_goal_pairs((40, 60), rng, 20.0, 25)

    assert len(pairs) == 25
    for start, goal in pairs:
        assert astar.euclidean_distance(start, goal) >= 20.0


def test_sampled_pairs_stay_inside_the_map(rng):
    pairs = sampling.sample_start_goal_pairs((10, 12), rng, 5.0, 20)

    for start, goal in pairs:
        for cell in (start, goal):
            assert 0 <= cell[0] < 10
            assert 0 <= cell[1] < 12


def test_sampling_is_reproducible_for_a_seed():
    first = sampling.sample_start_goal_pairs((30, 30), np.random.default_rng(3), 10.0, 8)
    second = sampling.sample_start_goal_pairs((30, 30), np.random.default_rng(3), 10.0, 8)

    assert first == second


def test_sampling_nothing_yields_nothing(rng):
    assert sampling.sample_start_goal_pairs((10, 10), rng, 1.0, 0) == []


def test_a_negative_count_is_rejected(rng):
    with pytest.raises(ValueError, match="not be negative"):
        sampling.sample_start_goal_pairs((10, 10), rng, 1.0, -1)


def test_an_unreachable_minimum_distance_is_reported(rng):
    with pytest.raises(RuntimeError, match="min-distance-fraction"):
        sampling.sample_start_goal_pairs((5, 5), rng, 1000.0, 3)


def test_bucketing_assigns_every_routed_pair_a_bucket(terrain, rng):
    pairs = sampling.sample_start_goal_pairs(terrain.shape, rng, 25.0, 12)

    candidates = sampling.bucket_by_length(pairs, terrain, 3)

    assert len(candidates) == len(pairs)
    assert all(0 <= candidate.bucket < 3 for candidate in candidates)


def test_bucketing_spreads_candidates_over_every_bucket(terrain, rng):
    pairs = sampling.sample_start_goal_pairs(terrain.shape, rng, 25.0, 30)

    candidates = sampling.bucket_by_length(pairs, terrain, 3)

    assert len({candidate.bucket for candidate in candidates}) == 3


def test_a_single_bucket_holds_everything(terrain, rng):
    pairs = sampling.sample_start_goal_pairs(terrain.shape, rng, 25.0, 6)

    candidates = sampling.bucket_by_length(pairs, terrain, 1)

    assert all(candidate.bucket == 0 for candidate in candidates)


def test_bucketing_records_the_route_and_its_length(terrain, rng):
    pairs = sampling.sample_start_goal_pairs(terrain.shape, rng, 25.0, 4)

    candidates = sampling.bucket_by_length(pairs, terrain, 2)

    for candidate in candidates:
        assert candidate.route[0] == candidate.start
        assert candidate.route[-1] == candidate.goal
        assert candidate.length == pytest.approx(astar.route_length(candidate.route))


def test_bucketing_nothing_yields_nothing(terrain):
    assert sampling.bucket_by_length([], terrain, 3) == []


def test_a_non_positive_bucket_count_is_rejected(terrain):
    with pytest.raises(ValueError, match="at least 1"):
        sampling.bucket_by_length([], terrain, 0)


def test_the_split_is_disjoint_and_complete(rng):
    candidates = make_candidates({0: 6, 1: 6})

    train, evaluation = sampling.split_train_eval(candidates, 4, 6, rng)

    assert len(train) == 4
    assert len(evaluation) == 6
    train_keys = {(c.start, c.goal, c.bucket) for c in train}
    eval_keys = {(c.start, c.goal, c.bucket) for c in evaluation}
    assert not train_keys & eval_keys


def test_both_sets_draw_from_every_bucket(rng):
    candidates = make_candidates({0: 8, 1: 8, 2: 8})

    train, evaluation = sampling.split_train_eval(candidates, 6, 9, rng)

    assert {c.bucket for c in train} == {0, 1, 2}
    assert {c.bucket for c in evaluation} == {0, 1, 2}


def test_the_split_is_reproducible_for_a_seed():
    candidates = make_candidates({0: 5, 1: 5})

    first = sampling.split_train_eval(candidates, 2, 4, np.random.default_rng(11))
    second = sampling.split_train_eval(candidates, 2, 4, np.random.default_rng(11))

    assert first == second


def test_an_empty_candidate_pool_is_rejected(rng):
    with pytest.raises(ValueError, match="no candidates"):
        sampling.split_train_eval([], 1, 1, rng)


def test_a_bucket_that_is_too_small_is_reported(rng):
    candidates = make_candidates({0: 2, 1: 2})

    with pytest.raises(RuntimeError, match="oversample-factor"):
        sampling.split_train_eval(candidates, 4, 4, rng)
