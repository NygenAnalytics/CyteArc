import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import PerturbationClusters
from cytearc.perturbation import clusters as clusters_module
from cytearc.perturbation.clusters import (
    cluster_memory_bytes,
    cluster_profiles,
    embed_profiles,
    hdbscan_clusters,
    leaf_order,
    nearest_profiles,
    profile_distances,
)
from cytearc.storage.artifacts import ArtifactRef, inspect_artifact
from tests.storage_helpers import write_count_store


def _planted(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Profiles of 4 planted groups of 8 perturbations, and 6 others.

    The groups share a component, so they correlate at about 0.5 with each
    other and the others, unrelated, sit farther from every group.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=200) + rng.normal(size=(4, 200))
    grouped = np.repeat(centres, 8, axis=0) + rng.normal(scale=0.3, size=(32, 200))
    values = np.vstack([grouped, rng.normal(size=(6, 200))])
    return values, np.r_[np.repeat(np.arange(4), 8), np.full(6, -1)]


def _same_partition(labels: np.ndarray, planted: np.ndarray) -> bool:
    """Whether the labels group the planted groups exactly, up to their names."""
    pairs = set(zip(labels.tolist(), planted.tolist(), strict=True))
    return len(pairs) == len(set(labels.tolist())) == len(set(planted.tolist()))


@pytest.mark.parametrize("metric", ["correlation", "euclidean"])
def test_distances_equal_scipys(metric) -> None:
    values, _ = _planted()

    distances = profile_distances(values.astype(np.float32), metric)

    expected = squareform(pdist(values.astype(np.float32).astype(np.float64), metric))
    np.testing.assert_allclose(distances, expected, atol=1e-10)
    np.testing.assert_array_equal(distances, distances.T)
    assert (np.diag(distances) == 0).all() and distances.dtype == np.float64


def test_a_constant_profile_correlates_with_none() -> None:
    values, _ = _planted()
    values[3] = 0.1

    distances = profile_distances(values, "correlation")

    assert (np.delete(distances[3], 3) == 1).all() and distances[3, 3] == 0
    assert (profile_distances(np.zeros((3, 0)), "correlation") == 1 - np.eye(3)).all()


@pytest.mark.parametrize(
    ("values", "metric", "message"),
    [
        (np.ones((3, 2)), "cosine", "metric must be 'correlation' or 'euclidean'"),
        (np.ones(3), "correlation", "finite, perturbations x features"),
        (np.full((3, 2), np.nan), "correlation", "finite, perturbations x features"),
    ],
)
def test_distances_refuse_what_they_cannot_measure(values, metric, message) -> None:
    with pytest.raises(ValueError, match=message):
        profile_distances(values, metric)


@pytest.mark.parametrize("selection", ["eom", "leaf"])
def test_clusters_equal_a_direct_scikit_learn_call(selection) -> None:
    values, planted = _planted()
    distances = profile_distances(values, "correlation")

    labels, probability = hdbscan_clusters(
        distances, min_cluster_size=4, min_samples=2, selection=selection
    )

    model = HDBSCAN(
        metric="precomputed",
        min_cluster_size=4,
        min_samples=2,
        cluster_selection_method=selection,
        copy=True,
    ).fit(squareform(pdist(values, "correlation")))
    np.testing.assert_array_equal(labels, model.labels_)
    np.testing.assert_allclose(probability, model.probabilities_)
    assert labels.dtype == np.int64 and probability.dtype == np.float64
    if selection == "eom":
        assert _same_partition(labels, planted)
        assert (probability[labels < 0] == 0).all()


def test_leaf_order_is_average_linkages() -> None:
    values, planted = _planted()
    distances = profile_distances(values, "correlation")

    order = leaf_order(distances)

    expected = leaves_list(linkage(pdist(values, "correlation"), method="average"))
    np.testing.assert_array_equal(order, expected)
    # Each planted group sits together in the order.
    for group in range(4):
        positions = np.flatnonzero(planted[order] == group)
        assert positions.max() - positions.min() == 7


def test_nearest_profiles_skip_themselves_and_break_ties_by_order() -> None:
    distances = np.array(
        [
            [0.0, 1.0, 1.0, 0.5],
            [1.0, 0.0, 0.2, 0.2],
            [1.0, 0.2, 0.0, 0.3],
            [0.5, 0.2, 0.3, 0.0],
        ]
    )

    indices, nearest = nearest_profiles(distances, 2)

    assert indices.tolist() == [[3, 1], [2, 3], [1, 3], [1, 2]]
    np.testing.assert_allclose(
        nearest, [[0.5, 1.0], [0.2, 0.2], [0.2, 0.3], [0.2, 0.3]]
    )
    assert nearest.dtype == np.float32


def test_layout_depends_only_on_the_seed_and_separates_the_groups() -> None:
    values, planted = _planted()
    distances = profile_distances(values, "correlation")

    layout = embed_profiles(values, distances, metric="correlation", seed=0)

    assert layout.shape == (38, 2) and layout.dtype == np.float32
    assert np.isfinite(layout).all()
    np.testing.assert_array_equal(
        embed_profiles(values, distances, metric="correlation", seed=0), layout
    )
    assert not np.array_equal(
        embed_profiles(values, distances, metric="correlation", seed=1), layout
    )
    centres = np.stack([layout[planted == group].mean(axis=0) for group in range(4)])
    spread = max(
        np.linalg.norm(layout[planted == group] - centres[group], axis=1).max()
        for group in range(4)
    )
    assert spread < pdist(centres).min()
    euclidean = profile_distances(values, "euclidean")
    assert embed_profiles(values, euclidean, metric="euclidean", seed=0).shape == (
        38,
        2,
    )


def test_a_layout_of_equal_profiles_starts_from_its_noise() -> None:
    values = np.tile(np.arange(5.0), (6, 1))

    layout = embed_profiles(
        values, profile_distances(values, "correlation"), metric="correlation", seed=0
    )

    assert layout.shape == (6, 2) and np.isfinite(layout).all()


def test_clustering_refuses_too_few_perturbations_or_features() -> None:
    for values in (np.ones((2, 5)), np.ones((5, 1))):
        with pytest.raises(ValueError, match="at least 3 perturbations and 2"):
            cluster_profiles(
                values,
                metric="correlation",
                min_cluster_size=2,
                min_samples=1,
                selection="eom",
                seed=0,
            )


def test_the_memory_estimate_bounds_the_clusterings_peak() -> None:
    rng = np.random.default_rng(2)
    settings = {
        "metric": "correlation",
        "min_cluster_size": 4,
        "min_samples": 2,
        "selection": "eom",
        "seed": 0,
    }
    # The first call imports and compiles what the layout needs.
    cluster_profiles(rng.normal(size=(20, 10)), **settings)
    for shape in ((600, 40), (150, 900)):
        values = rng.normal(size=shape)
        tracemalloc.start()
        try:
            cluster_profiles(values, **settings)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak <= cluster_memory_bytes(*shape)


def test_the_clusters_type_is_exported() -> None:
    assert PerturbationClusters is clusters_module.PerturbationClusters


# 12 targets of 30 cells in three groups of four, each group lowering eight
# genes of its own, and 150 control cells.
N_TARGETS, PER_TARGET, N_CONTROLS, N_GENES = 12, 30, 150, 40
TARGETS = [f"T{k:02d}" for k in range(N_TARGETS)]
PLANTED = np.arange(N_TARGETS) // 4


def _screen() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    guide_of = np.r_[
        np.repeat(np.arange(N_TARGETS), PER_TARGET), np.full(N_CONTROLS, N_TARGETS)
    ]
    library = rng.lognormal(np.log(3000), 0.3, len(guide_of))
    base = rng.normal(-3.5, 0.6, N_GENES)
    mu = np.exp(base[None, :] + np.log(library)[:, None])
    for target in range(N_TARGETS):
        genes = np.arange(8) + 8 * PLANTED[target]
        mu[np.ix_(guide_of == target, genes)] *= 0.25
    guides = np.zeros((len(guide_of), N_TARGETS + 1), dtype=np.uint16)
    guides[np.arange(len(guide_of)), guide_of] = 20
    return rng.poisson(mu).astype(np.uint16), guides


@dataclass(frozen=True)
class _Shared:
    path: Path
    store: DataStore
    profiles: ArtifactRef


@pytest.fixture(scope="module")
def shared(tmp_path_factory) -> _Shared:
    """One store of the screen with its profiles."""
    path = tmp_path_factory.mktemp("clusters") / "store.zarr"
    rna, guides = _screen()
    write_count_store(str(path), {"RNA": rna, "CRISPR": guides}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=2
    )
    store._get_assay("CRISPR").feats.insert(
        "target_gene_name", np.asarray([*TARGETS, "NT"], dtype=object)
    )
    calls = store.perturbation.call_guides(
        store.select_measured_cells("CRISPR"), method="fixed", min_umi=10
    )
    labels = store.perturbation.labels(calls, controls=["NT"])
    features = store.features.snapshot(feature_indexes=np.arange(N_GENES))
    profiles = store.perturbation.profiles(
        labels, features, min_cells=10, id_column=None
    )
    return _Shared(path, store, profiles)


def _profile_values(store: DataStore, profiles: ArtifactRef) -> pd.DataFrame:
    return store.perturbation.load_profiles(profiles).values.astype(np.float64)


def test_clusters_recover_the_planted_groups_as_scikit_learn_does(shared) -> None:
    store, profiles = shared.store, shared.profiles

    clusters = store.perturbation.cluster(profiles, min_cluster_size=3)

    assert (clusters.kind, clusters.assay) == ("perturbation_clusters", "RNA")
    result = store.perturbation.load_clusters(clusters)
    assert isinstance(result, PerturbationClusters)
    assert (result.metric, result.min_cluster_size, result.min_samples) == (
        "correlation",
        3,
        2,
    )
    assert (result.selection, result.seed, result.from_assay) == ("eom", 0, "RNA")
    table = result.table
    assert table.index.tolist() == TARGETS
    assert table.columns.tolist() == ["cluster", "probability", "n_cells", "x", "y"]
    assert _same_partition(table["cluster"].to_numpy(), PLANTED)
    assert (table["n_cells"] == PER_TARGET).all()
    assert np.isfinite(table[["x", "y"]].to_numpy()).all()
    values = _profile_values(store, profiles).to_numpy()
    expected = squareform(pdist(values, "correlation"))
    model = HDBSCAN(
        metric="precomputed",
        min_cluster_size=3,
        min_samples=2,
        cluster_selection_method="eom",
        copy=True,
    ).fit(expected)
    np.testing.assert_array_equal(table["cluster"].to_numpy(), model.labels_)
    np.testing.assert_allclose(table["probability"].to_numpy(), model.probabilities_)
    np.testing.assert_allclose(result.distances.to_numpy(), expected, atol=1e-10)
    assert result.distances.index.tolist() == TARGETS
    assert result.distances.columns.tolist() == TARGETS
    order = leaves_list(linkage(pdist(values, "correlation"), method="average"))
    assert result.order.tolist() == [TARGETS[k] for k in order]
    status = inspect_artifact(store.zw, clusters)
    assert set(status.inputs) == {"perturbation_profiles", "perturbations_fingerprint"}
    assert status.inputs["perturbations_fingerprint"] is None
    assert status.parameters == {
        "metric": "correlation",
        "min_cluster_size": 3,
        "min_samples": 2,
        "selection": "eom",
        "seed": 0,
    }
    # The profiles of the screen leave no perturbation out.
    assert store.perturbation.load_profiles(profiles).left_out.empty


def test_cluster_layouts_depend_only_on_the_seed(shared) -> None:
    store, profiles = shared.store, shared.profiles
    first = store.perturbation.load_clusters(
        store.perturbation.cluster(profiles, min_cluster_size=3)
    )

    again = store.perturbation.load_clusters(
        store.perturbation.cluster(profiles, min_cluster_size=3, invalidate_cache=True)
    )
    seeded = store.perturbation.load_clusters(
        store.perturbation.cluster(profiles, min_cluster_size=3, seed=1)
    )
    largest = store.perturbation.load_clusters(
        store.perturbation.cluster(profiles, min_cluster_size=3, seed=2**32 - 1)
    )

    pd.testing.assert_frame_equal(again.table, first.table)
    assert seeded.seed == 1
    pd.testing.assert_series_equal(seeded.table["cluster"], first.table["cluster"])
    assert not np.array_equal(seeded.table["x"], first.table["x"])
    # The largest seed lays the perturbations out too.
    assert largest.seed == 2**32 - 1
    assert np.isfinite(largest.table[["x", "y"]].to_numpy()).all()


def test_clusters_of_named_perturbations(shared) -> None:
    store, profiles = shared.store, shared.profiles
    named = ["T05", "T01", "T00", "T07", "T02", "T03", "T04", "T06", "T01"]

    clusters = store.perturbation.cluster(
        profiles, perturbations=pd.Series(named), min_cluster_size=3
    )

    result = store.perturbation.load_clusters(clusters)
    assert result.table.index.tolist() == TARGETS[:8]
    assert _same_partition(result.table["cluster"].to_numpy(), PLANTED[:8])
    assert result.distances.shape == (8, 8)
    assert sorted(result.order.tolist()) == TARGETS[:8]
    status = inspect_artifact(store.zw, clusters)
    assert status.inputs["perturbations_fingerprint"] is not None
    # Every profile, named or not, is the same input.
    assert store.perturbation.cluster(
        profiles, perturbations=TARGETS, min_cluster_size=3
    ) == store.perturbation.cluster(profiles, min_cluster_size=3)


def test_clusters_by_euclidean_distance(shared) -> None:
    store, profiles = shared.store, shared.profiles

    result = store.perturbation.load_clusters(
        store.perturbation.cluster(profiles, metric="euclidean", min_cluster_size=3)
    )

    values = _profile_values(store, profiles).to_numpy()
    np.testing.assert_allclose(
        result.distances.to_numpy(), squareform(pdist(values)), atol=1e-9
    )
    assert result.metric == "euclidean"
    assert _same_partition(result.table["cluster"].to_numpy(), PLANTED)


def test_clusters_are_reused_read_only_and_refused_when_new(shared) -> None:
    store, profiles = shared.store, shared.profiles
    clusters = store.perturbation.cluster(profiles, min_cluster_size=3)
    reader = DataStore(
        str(shared.path),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )

    assert reader.perturbation.cluster(profiles, min_cluster_size=3) == clusters
    with pytest.raises(PermissionError, match="cluster_perturbations"):
        reader.perturbation.cluster(profiles, min_cluster_size=5)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"metric": "cosine"}, ValueError, "metric must be 'correlation' or"),
        ({"min_cluster_size": 1}, ValueError, "min_cluster_size must be at least 2"),
        ({"min_samples": 0}, ValueError, "min_samples must be at least 1"),
        ({"min_samples": 13}, ValueError, "min_samples=13 must be at most the 12"),
        ({"selection": "top"}, ValueError, "selection must be 'eom' or 'leaf'"),
        ({"seed": -1}, ValueError, "seed must be at least 0"),
        # scikit-learn's PCA accepts 32-bit unsigned random states only.
        ({"seed": 2**32}, ValueError, "seed must be at most 4294967295"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
        ({"perturbations": "T00"}, TypeError, "perturbations must be a sequence"),
        ({"perturbations": 3}, TypeError, "perturbations must be a sequence"),
        ({"perturbations": [1, 2, 3]}, TypeError, "perturbations must be a sequence"),
        (
            {"perturbations": ["T00", "NT", "T99"]},
            ValueError,
            "without a profile: 'NT', 'T99'",
        ),
        ({"perturbations": ["T00", "T01"]}, ValueError, "not 2 and 40"),
    ],
)
def test_cluster_refuses_arguments_it_cannot_use(
    shared, arguments, error, message
) -> None:
    with pytest.raises(error, match=message):
        shared.store.perturbation.cluster(shared.profiles, **arguments)


def test_cluster_refuses_inputs_and_budgets_it_cannot_use(shared, monkeypatch) -> None:
    store, profiles = shared.store, shared.profiles
    clusters = store.perturbation.cluster(profiles, min_cluster_size=3)
    with pytest.raises(TypeError, match="profiles must be an ArtifactRef"):
        store.perturbation.cluster("profiles")
    with pytest.raises(ValueError, match="not 'perturbation_clusters'"):
        store.perturbation.cluster(clusters)
    with pytest.raises(TypeError, match="clusters must be an ArtifactRef"):
        store.perturbation.load_clusters("clusters")
    with pytest.raises(ValueError, match="not 'perturbation_profiles'"):
        store.perturbation.load_clusters(profiles)
    # The clustering takes about 30 kB, and the distances on loading 12 kB.
    monkeypatch.setattr(store, "memoryBytes", 10_000)
    with pytest.raises(MemoryError, match="Clustering 12 perturbations in 40"):
        store.perturbation.cluster(profiles, min_cluster_size=6)
    with pytest.raises(MemoryError, match="distances between 12 profiles"):
        store.perturbation.load_clusters(clusters)
