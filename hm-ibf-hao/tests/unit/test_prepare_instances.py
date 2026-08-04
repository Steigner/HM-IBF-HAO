"""Unit tests for :mod:`preprocessing.prepare_instances`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from preprocessing import prepare_instances


def parse(argv: list[str]):
    """Parse arguments, always supplying the required source."""
    return prepare_instances.build_parser().parse_args(["--source", "terrain.npy", *argv])


def test_omitting_the_source_selects_the_benchmark_mode():
    args = prepare_instances.build_parser().parse_args([])

    assert args.source is None
    assert args.output_dir == Path("instances")
    assert args.maps == ["austria", "italia", "slovenia"]


def test_an_unknown_benchmark_map_is_rejected():
    with pytest.raises(SystemExit):
        prepare_instances.build_parser().parse_args(["--maps", "atlantis"])


def test_the_documented_defaults_are_parsed():
    args = parse([])

    assert args.source == Path("terrain.npy")
    assert args.train_dir == Path("instances_train")
    assert args.eval_dir == Path("instances_eval")
    assert args.seed == 42
    assert args.resolution == [200, 400]


def test_flags_override_the_defaults():
    args = parse(["--n-train", "3", "--n-eval", "4", "--seed", "9", "--resolution", "50", "60"])

    assert args.n_train == 3
    assert args.n_eval == 4
    assert args.seed == 9
    assert args.resolution == [50, 60]


def test_the_configuration_mirrors_the_arguments():
    config = prepare_instances.config_from_args(
        parse(["--n-train", "3", "--tau", "0.6", "--resolution", "50", "60"])
    )

    assert config.n_train == 3
    assert config.tau == 0.6
    assert config.target_resolution == (50, 60)


def test_an_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="tau"):
        prepare_instances.config_from_args(parse(["--tau", "2.0"]))


def test_a_missing_source_is_reported(tmp_path, small_config):
    with pytest.raises(FileNotFoundError, match="--source"):
        prepare_instances.load_heightmap(tmp_path / "absent.npy", small_config)


def test_a_source_is_loaded_and_downsampled(tmp_path, small_config, terrain):
    source = tmp_path / "terrain.npy"
    np.save(source, terrain)

    loaded = prepare_instances.load_heightmap(source, small_config)

    assert loaded.shape == small_config.target_resolution


def test_main_reports_a_missing_source(tmp_path):
    exit_code = prepare_instances.main(["--source", str(tmp_path / "absent.npy")])

    assert exit_code == 1


def test_main_reports_a_failed_benchmark_download(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise prepare_instances.MapFetchError("offline")

    monkeypatch.setattr(prepare_instances, "generate_benchmark", fail)

    exit_code = prepare_instances.main(["--output-dir", str(tmp_path / "instances")])

    assert exit_code == 1


def test_main_reports_an_invalid_configuration(tmp_path, terrain):
    source = tmp_path / "terrain.npy"
    np.save(source, terrain)

    exit_code = prepare_instances.main(["--source", str(source), "--tau", "3.0"])

    assert exit_code == 1


def test_main_reports_a_terrain_that_is_too_small(tmp_path):
    source = tmp_path / "tiny.npy"
    np.save(source, np.zeros((4, 4)))

    exit_code = prepare_instances.main(
        [
            "--source",
            str(source),
            "--train-dir",
            str(tmp_path / "train"),
            "--eval-dir",
            str(tmp_path / "eval"),
            "--resolution",
            "4",
            "4",
            "--min-distance-fraction",
            "0.9",
        ]
    )

    assert exit_code == 1
