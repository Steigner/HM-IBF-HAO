"""Download the benchmark terrain heightmaps from the upstream study.

`preprocessing.prepare_instances` needs the elevation data published in
https://github.com/Steigner/HorizAligns-Hybrid-Optimization. Those three `.npy` files total
roughly 190 MB and are not part of this repository, so this module fetches them on demand
into a local cache directory and verifies that what arrived has the shape the catalog
declares.

Only the files named in :mod:`preprocessing.maps` are ever requested, and only from the
configured base URL, so no caller-supplied string reaches the network layer as a path.

Run it directly to prime the cache::

    python3 -m preprocessing.fetch_maps
"""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from preprocessing.maps import BENCHMARK_MAPS, MAP_NAMES, BenchmarkMap, map_by_name

LOGGER = logging.getLogger(__name__)

#: Directory the heightmaps are cached in, relative to the crate directory.
DEFAULT_CACHE_DIR = Path("external/horizaligns")

#: Repository path the heightmaps are served from, appended to the raw content host.
_UPSTREAM_PATH = "Steigner/HorizAligns-Hybrid-Optimization/master/scripts/heightmaps"

#: Base URL the heightmaps are downloaded from.
DEFAULT_BASE_URL = f"https://raw.githubusercontent.com/{_UPSTREAM_PATH}"

#: Seconds to wait for the download to make progress before giving up.
DOWNLOAD_TIMEOUT = 300


class MapFetchError(RuntimeError):
    """Raised when a heightmap cannot be downloaded or does not match the catalog."""


def _download(url: str, destination: Path) -> None:
    """Stream one URL to a file.

    Args:
        url: The URL to fetch; always built from the configured base URL and a catalog file
            name, never from caller-supplied text.
        destination: File the response body is written to.

    Raises:
        urllib.error.URLError: If the request fails.
        OSError: If the response cannot be read or the file cannot be written.
    """
    # Streamed rather than read whole: the heightmaps are tens of megabytes each.
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response,  # noqa: S310
        destination.open("wb") as file,
    ):
        shutil.copyfileobj(response, file)


def ensure_heightmap(
    entry: BenchmarkMap,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    """Ensure one benchmark heightmap is present in the cache.

    An already cached file is reused as-is and never re-downloaded; delete it to pick up an
    upstream change. A download is written to a temporary file first and only renamed into
    place after it has been verified, so an interrupted run cannot leave a truncated cache
    entry behind.

    Args:
        entry: The benchmark map to fetch.
        cache_dir: Directory holding the cached heightmaps; created if missing.
        base_url: Base URL the file is downloaded from.

    Returns:
        Path of the cached `.npy` heightmap.

    Raises:
        MapFetchError: If the download fails or the file does not hold an array of the
            shape the catalog declares.
    """
    destination = cache_dir / entry.filename
    if destination.is_file():
        LOGGER.info("reusing cached %s", destination)
        return destination

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{entry.filename}"
    partial = destination.with_suffix(destination.suffix + ".part")

    LOGGER.info("downloading %s (%s)", url, entry.name)
    try:
        _download(url, partial)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise MapFetchError(f"failed to download {url}: {error}") from error

    try:
        verify_heightmap(partial, entry)
    except MapFetchError:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(destination)
    LOGGER.info("cached %s", destination)
    return destination


def verify_heightmap(path: Path, entry: BenchmarkMap) -> None:
    """Check that a downloaded file holds the heightmap the catalog describes.

    Args:
        path: The downloaded `.npy` file.
        entry: The catalog entry it should match.

    Raises:
        MapFetchError: If the file is not a readable `.npy` array of the declared shape.
    """
    try:
        heightmap = np.load(path, mmap_mode="r")
    except (ValueError, OSError) as error:
        raise MapFetchError(f"{path} is not a readable .npy heightmap: {error}") from error

    if tuple(heightmap.shape) != entry.source_shape:
        raise MapFetchError(
            f"{path} has shape {tuple(heightmap.shape)}, expected {entry.source_shape}; "
            "the upstream layout may have changed"
        )


def fetch_maps(
    names: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Path]:
    """Ensure several benchmark heightmaps are cached.

    Args:
        names: Region names to fetch; every catalog entry when omitted.
        cache_dir: Directory holding the cached heightmaps.
        base_url: Base URL the files are downloaded from.

    Returns:
        The cached path of each requested map, keyed by region name.

    Raises:
        KeyError: If a requested name is not in the catalog.
        MapFetchError: If a download fails or a file does not match the catalog.
    """
    entries = [map_by_name(name) for name in names] if names else list(BENCHMARK_MAPS)
    return {entry.name: ensure_heightmap(entry, cache_dir, base_url) for entry in entries}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the downloader.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="fetch_maps",
        description="Download the benchmark terrain heightmaps from the upstream study.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        choices=MAP_NAMES,
        default=list(MAP_NAMES),
        help="regions to download (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="directory the heightmaps are cached in (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="base URL the heightmaps are downloaded from",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the downloader.

    Args:
        argv: Command line arguments; `sys.argv[1:]` when omitted.

    Returns:
        A process exit code: zero on success, one if a download failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        cached = fetch_maps(args.maps, args.cache_dir, args.base_url)
    except (KeyError, MapFetchError) as error:
        LOGGER.error("%s", error)
        return 1

    for name, path in cached.items():
        LOGGER.info("%s -> %s", name, path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    raise SystemExit(main())
