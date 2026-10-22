"""PCA artifacts: the solver, the stored explained variance, and the revision."""

import numpy as np
import pytest
from scipy.linalg import subspace_angles

import cytearc.datastore._operations.graph as graph_operations
from cytearc.datastore.datastore import DataStore
from cytearc.embeddings.reduction import pca_fit_resident_bytes
from cytearc.matrix import ChunkedArray
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from cytearc.storage.budget import ResourceBudget
from cytearc.storage.operation_revisions import (
    OPERATION_REVISIONS,
    effective_revision,
)
from cytearc.utils.logging import logger
from tests.storage_helpers import write_count_store

N_CELLS = 150
# More genes than the exact Gram solver takes, and fewer.
WIDE_GENES = 4_200
NARROW_GENES = 40


def _counts(n_genes: int) -> np.ndarray:
    """Return Poisson counts of cells drawn from four expression programs."""
    rng = np.random.default_rng(17)
    programs = rng.gamma(0.5, 2.0, size=(4, n_genes))
    labels = rng.integers(0, 4, size=N_CELLS)
    depth = rng.lognormal(0.0, 0.3, size=(N_CELLS, 1))
    return rng.poisson(programs[labels] * depth).astype(np.uint16)


def _normalized_store(directory, n_genes: int) -> tuple[DataStore, ArtifactRef]:
    zarr_loc = str(directory / f"genes_{n_genes}.zarr")
    write_count_store(zarr_loc, {"RNA": _counts(n_genes)}, "uint16")
    store = DataStore(
        zarr_loc, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    cells = store.snapshot_cell_selection("I")
    features = store.features.universe(from_assay="RNA")
    return store, store.features.normalize(cells, features)


@pytest.fixture(scope="module")
def wide(tmp_path_factory) -> tuple[DataStore, ArtifactRef]:
    return _normalized_store(tmp_path_factory.mktemp("wide_pca"), WIDE_GENES)


@pytest.fixture(scope="module")
def narrow(tmp_path_factory) -> tuple[DataStore, ArtifactRef]:
    return _normalized_store(tmp_path_factory.mktemp("narrow_pca"), NARROW_GENES)


def _input(store: DataStore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return ArtifactRef.from_dict(store.artifacts.inspect(ref).inputs[name])


def _fitted_values(store: DataStore, pca: ArtifactRef) -> np.ndarray:
    """Return the standardized normalized values that ``pca`` was fitted on."""
    normalized = store.artifacts.load(_input(store, pca, "normalized"))["data"][:]
    scaling = store.artifacts.load(_input(store, pca, "feature_scaling"))
    return (np.asarray(normalized, dtype=np.float64) - scaling["mean"][:]) / scaling[
        "scale"
    ][:]


def _exact_pca(values: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading principal axes and the variance of every component."""
    centered = values - values.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    return axes[:dims].T, np.square(singular_values) / (len(values) - 1)


@pytest.mark.parametrize(
    ("genes", "parameters"),
    [
        ("narrow", {"dims": 6, "feat_scaling": True}),
        ("wide", {"dims": 6, "feat_scaling": True, "solver": "randomized"}),
    ],
)
def test_pca_stores_the_exact_components_and_their_variance(
    request, genes: str, parameters: dict
) -> None:
    store, normalized = request.getfixturevalue(genes)

    pca = store.reduction.pca(normalized, dims=6, local_cache=False)

    status = store.artifacts.inspect(pca)
    assert status.parameters == parameters
    # The revision of IncrementalPCA fits applies to no new fit.
    assert (status.revision, status.current_revision, status.is_current) == (
        1,
        1,
        True,
    )
    group = store.artifacts.load(pca)
    fitted = _fitted_values(store, pca)
    axes, variance = _exact_pca(fitted, 6)
    loadings = np.asarray(group["loadings"][:])
    # The randomized solver of more than 4096 features stops within a small
    # fraction of a degree of the exact subspace; the Gram solver is exact.
    limit = 1e-8 if genes == "narrow" else 0.1
    assert np.degrees(subspace_angles(loadings, axes)).max() < limit
    for name in ("variance", "variance_ratio"):
        assert group[name].shape == (6,)
        assert group[name].dtype == np.dtype(np.float64)
    np.testing.assert_allclose(group["variance"][:], variance[:6], rtol=1e-6)
    np.testing.assert_allclose(
        group["variance_ratio"][:], variance[:6] / variance.sum(), rtol=1e-6
    )
    np.testing.assert_allclose(
        group["data"][:],
        (fitted - group["center"][:]) @ loadings,
        rtol=1e-5,
        atol=1e-5,
    )


def _rewrite_as_earlier_release(
    store: DataStore, ref: ArtifactRef, **parameters: int
) -> None:
    """Rewrite a PCA artifact as a release before 1.3.0 stored it.

    Those releases stored no explained variance, and recorded
    ``incremental_block_rows`` for IncrementalPCA fits over several blocks.
    """
    group = artifact_group(store.zw, ref)
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = {**provenance["parameters"], **parameters}
    group.attrs["provenance"] = provenance
    del group["variance"]
    del group["variance_ratio"]


def _warnings_of(run) -> list[str]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        run()
    finally:
        logger.remove(sink)
    return messages


def test_incremental_pca_fits_are_stale_and_computed_again(narrow) -> None:
    store, normalized = narrow
    fitted = store.reduction.pca(
        normalized, dims=3, local_cache=False, invalidate_cache=True
    )
    _rewrite_as_earlier_release(store, fitted, incremental_block_rows=20)

    recomputed = store.reduction.pca(normalized, dims=3, local_cache=False)

    (revision,) = OPERATION_REVISIONS["run_pca"]
    old = store.artifacts.inspect(fitted)
    assert (old.revision, old.current_revision, old.is_current) == (1, 2, False)
    assert old.superseded_by == (revision,)
    markdown = store.artifacts.lineage(fitted).to_markdown()
    assert "- Status: `stale`" in markdown
    assert f"- Revision: `1, current 2: {revision.change}`" in markdown
    # A new fit records no block size and no revision, so it is current.
    assert recomputed != fitted
    new = store.artifacts.inspect(recomputed)
    assert new.parameters == {"dims": 3, "feat_scaling": True}
    assert (new.revision, new.is_current) == (1, True)
    assert store.reduction.pca(normalized, dims=3, local_cache=False) == recomputed


def test_earlier_pca_of_other_fits_is_reused_without_an_elbow_plot(
    narrow, monkeypatch
) -> None:
    store, normalized = narrow
    fitted = store.reduction.pca(
        normalized, dims=5, local_cache=False, invalidate_cache=True
    )
    # Single-block and Gram fits of earlier releases recorded the provenance of
    # a new fit, and no explained variance.
    _rewrite_as_earlier_release(store, fitted)
    shown: list[dict] = []
    monkeypatch.setattr("cytearc.plotting.elbow", lambda **kwargs: shown.append(kwargs))
    reused: list[ArtifactRef] = []

    messages = _warnings_of(
        lambda: reused.append(
            store.reduction.pca(
                normalized, dims=5, local_cache=False, show_elbow_plot=True
            )
        )
    )

    assert reused == [fitted]
    status = store.artifacts.inspect(fitted)
    assert (status.revision, status.current_revision, status.is_current) == (
        1,
        1,
        True,
    )
    assert status.superseded_by == ()
    assert shown == []
    assert messages == [
        f"PCA artifact {fitted.artifact_id[:12]} was stored before release 1.3.0 "
        "without its explained variance, so no elbow plot is available. Pass "
        "invalidate_cache=True to fit it again."
    ]


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"dims": 21, "feat_scaling": True, "incremental_block_rows": 1000}, 2),
        ({"dims": 21, "feat_scaling": True}, 1),
        ({"dims": 21, "feat_scaling": True, "solver": "randomized"}, 1),
        # A record of another release may lack any parameter.
        ({}, 1),
    ],
)
def test_pca_revision_applies_only_to_incremental_pca_fits(
    parameters: dict, expected: int
) -> None:
    inputs = {"normalized": {"artifact_id": "a" * 64}}

    assert effective_revision("run_pca", "reduction", parameters, inputs) == expected
    (revision,) = OPERATION_REVISIONS["run_pca"]
    assert (revision.revision, revision.release) == (2, "1.3.0")


def test_pca_raises_memory_error_before_it_scales_or_fits(wide, monkeypatch) -> None:
    store, normalized = wide
    resources = store.resources
    before = {
        kind: store.artifacts.list(kind=kind, from_assay="RNA", complete_only=False)
        for kind in ("feature_scaling", "reduction")
    }
    data = ChunkedArray(
        store.artifacts.load(normalized)["data"],
        block_size=N_CELLS,
    )
    required = (
        pca_fit_resident_bytes(WIDE_GENES, N_CELLS, 9)
        + data._resident_bytes()
        + data._block_task_bytes()
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a fit over budget must not start")

    monkeypatch.setattr(graph_operations, "_feature_scaling_statistics", forbidden)
    monkeypatch.setattr("cytearc.embeddings.reduction.fit_pca", forbidden)
    store.resources = ResourceBudget(required - 1, 1)
    try:
        with pytest.raises(
            MemoryError,
            match=(
                f"PCA with the randomized solver needs about {required} bytes to "
                f"fit 9 components of {WIDE_GENES} features in blocks of "
                f"{N_CELLS} rows"
            ),
        ):
            store.reduction.pca(normalized, dims=9, local_cache=False)
    finally:
        store.resources = resources

    assert {
        kind: store.artifacts.list(kind=kind, from_assay="RNA", complete_only=False)
        for kind in ("feature_scaling", "reduction")
    } == before


def test_pca_fit_budget_admits_exactly_its_requirement() -> None:
    values = np.zeros((40, 30), dtype=np.float32)
    data = ChunkedArray.from_numpy(values, block_size=10)
    required = (
        pca_fit_resident_bytes(30, 10, 3)
        + data._resident_bytes()
        + data._block_task_bytes()
    )

    graph_operations._require_pca_fit_budget(data, ResourceBudget(required, 1), dims=3)
    with pytest.raises(MemoryError, match="PCA with the gram solver needs about"):
        graph_operations._require_pca_fit_budget(
            data, ResourceBudget(required - 1, 1), dims=3
        )
