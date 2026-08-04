"""Unit tests for :mod:`preprocessing.fetch_maps`.

Nothing here touches the network: every test either works on an already cached file or
substitutes the download step, so the suite stays runnable offline.
"""

from __future__ import annotations

import io
import urllib.error
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from preprocessing import fetch_maps
from preprocessing.maps import map_by_name


def write_npy_bytes(destination, shape):
    """Write a `.npy` payload to an exact path, the way a download does.

    `np.save` appends `.npy` to a path that lacks the suffix, which would miss the partial
    file the downloader actually writes.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.zeros(shape, dtype=np.float32))
    Path(destination).write_bytes(buffer.getvalue())


@pytest.fixture
def entry():
    """A catalog entry shrunk to a shape a test can materialize."""
    return replace(map_by_name("austria"), filename="tiny.npy", source_shape=(3, 4))


def cache_heightmap(cache_dir, entry, shape=None):
    """Write a heightmap of the entry's declared shape into the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / entry.filename
    np.save(path, np.zeros(shape or entry.source_shape, dtype=np.float32))
    return path


def test_an_already_cached_heightmap_is_reused(tmp_path, entry, monkeypatch):
    cache_heightmap(tmp_path, entry)

    def fail(*_args, **_kwargs):
        raise AssertionError("a cached heightmap must not be downloaded again")

    monkeypatch.setattr(fetch_maps, "_download", fail)

    assert fetch_maps.ensure_heightmap(entry, tmp_path) == tmp_path / entry.filename


def test_a_download_is_verified_and_moved_into_place(tmp_path, entry, monkeypatch):
    def download(_url, destination):
        write_npy_bytes(destination, entry.source_shape)

    monkeypatch.setattr(fetch_maps, "_download", download)

    path = fetch_maps.ensure_heightmap(entry, tmp_path / "cache")

    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".part").exists()


def test_a_download_of_the_wrong_shape_is_rejected_and_not_cached(tmp_path, entry, monkeypatch):
    def download(_url, destination):
        write_npy_bytes(destination, (2, 2))

    monkeypatch.setattr(fetch_maps, "_download", download)

    with pytest.raises(fetch_maps.MapFetchError, match="expected"):
        fetch_maps.ensure_heightmap(entry, tmp_path / "cache")

    assert not (tmp_path / "cache" / entry.filename).exists()
    assert not list((tmp_path / "cache").glob("*.part"))


def test_a_failed_download_reports_its_url(tmp_path, entry, monkeypatch):
    def download(_url, _destination):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(fetch_maps, "_download", download)

    with pytest.raises(fetch_maps.MapFetchError, match=entry.filename):
        fetch_maps.ensure_heightmap(entry, tmp_path / "cache")


def test_a_file_that_is_not_an_npy_array_is_rejected(tmp_path, entry):
    (tmp_path / entry.filename).write_text("not an array", encoding="utf-8")

    with pytest.raises(fetch_maps.MapFetchError, match="readable"):
        fetch_maps.verify_heightmap(tmp_path / entry.filename, entry)


def test_fetching_several_maps_returns_one_path_each(tmp_path, monkeypatch):
    def already_cached(entry, cache_dir, _base_url):
        return cache_dir / entry.filename

    monkeypatch.setattr(fetch_maps, "ensure_heightmap", already_cached)

    paths = fetch_maps.fetch_maps(["austria", "slovenia"], tmp_path)

    assert set(paths) == {"austria", "slovenia"}


def test_the_cli_reports_a_failed_download(tmp_path, monkeypatch):
    def download(_url, _destination):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(fetch_maps, "_download", download)

    exit_code = fetch_maps.main(["--maps", "austria", "--cache-dir", str(tmp_path / "cache")])

    assert exit_code == 1


def test_the_cli_defaults_to_every_map():
    args = fetch_maps.build_parser().parse_args([])

    assert args.maps == list(fetch_maps.MAP_NAMES)
    assert args.base_url.startswith("https://")
