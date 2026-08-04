"""Instance generator for the HM-IBF-HAO horizontal alignment benchmark.

An instance is a route across a terrain heightmap, described by a *backbone*: the
least-cost route between a start and a goal, resampled at equal arc lengths. The Rust
benchmark optimizes perpendicular offsets along that backbone, so preprocessing also
stores the local normals and the offset bounds the terrain leaves room for.

The package is split by concern:

* :mod:`preprocessing.config` - the tunable defaults, as one dataclass.
* :mod:`preprocessing.astar` - least-cost pathfinding and heightmap downsampling.
* :mod:`preprocessing.simplify` - Douglas-Peucker simplification.
* :mod:`preprocessing.backbone` - backbone resampling, normals and offset bounds.
* :mod:`preprocessing.maps` - the three named benchmark regions and their endpoints.
* :mod:`preprocessing.fetch_maps` - downloading and caching their terrain heightmaps.
* :mod:`preprocessing.benchmark` - regenerating the checked-in benchmark instances.
* :mod:`preprocessing.sampling` - start/goal sampling and the train/eval split.
* :mod:`preprocessing.instance` - assembling and writing one instance.
* :mod:`preprocessing.dimensions` - choosing the islands' working dimensions.
* :mod:`preprocessing.prepare_instances` - the command line entry point.
"""
