"""Session fixtures of lineage-resolved trajectory results on the 1K PBMC store.

The session pseudotime runs from Leiden cluster 1 to cluster 2. Its fate map
takes clusters 2, 3, and 4 as sinks, which gives three lineages that share
the cells of the other clusters. The PBMC data hold no differentiation
process; these results test mechanics, provenance, and reuse.
"""

import pytest

LINEAGE_SINKS = [2, 3, 4]


@pytest.fixture(scope="session")
def lineage_fate(datastore, pseudotime_scoring, leiden_clustering):
    return datastore.trajectory.fate(
        pseudotime_scoring,
        leiden_clustering,
        sinks=LINEAGE_SINKS,
    )


@pytest.fixture(scope="session")
def lineage_markers(datastore, lineage_fate, mark_hvgs):
    return datastore.trajectory.lineage_markers(lineage_fate, features=mark_hvgs)


@pytest.fixture(scope="session")
def lineage_drivers(datastore, lineage_fate, mark_hvgs):
    return datastore.trajectory.drivers(lineage_fate, features=mark_hvgs)


@pytest.fixture(scope="session")
def lineage_aggregation(datastore, lineage_fate, mark_hvgs):
    return datastore.trajectory.lineage_aggregation(
        lineage_fate,
        features=mark_hvgs,
        n_bins=12,
        n_clusters=5,
    )


@pytest.fixture(scope="session")
def lineage_curves(datastore, lineage_fate):
    return datastore.trajectory.curves(lineage_fate, n_points=40)


@pytest.fixture(scope="session")
def lineage_tree(datastore, connectivity_graph, leiden_clustering):
    return datastore.trajectory.lineage_tree(
        connectivity_graph,
        leiden_clustering,
        root=1,
    )
