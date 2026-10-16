import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc.datastore.datastore import DataStore
from cytearc.regulons.grn import GrnParameters, infer_grn
from cytearc.regulons.results import GrnResult
from cytearc.storage.artifacts import ArtifactRef, artifact_group
from tests.storage_helpers import write_count_store

N_CELLS = 150
FAST = {"max_stages": 150, "early_stop_window": 10}
TFS = ["RNA0", "RNA1", "RNA2", "RNA3"]


def _counts() -> np.ndarray:
    rng = np.random.default_rng(3)
    regulators = rng.poisson(2.0, size=(N_CELLS, 4))
    targets = np.column_stack(
        [
            regulators[:, 0] * 2 + rng.poisson(0.5, N_CELLS),
            regulators[:, 1] + regulators[:, 2] + rng.poisson(0.5, N_CELLS),
            rng.poisson(1.0, N_CELLS),
            np.zeros(N_CELLS, dtype=np.int64),
        ]
    )
    return np.hstack([regulators, targets])


def _open(zarr_loc: str, **kwargs) -> DataStore:
    return DataStore(
        zarr_loc, default_assay="RNA", min_features_per_cell=-1, nthreads=2, **kwargs
    )


def _grn(store: DataStore) -> ArtifactRef:
    """The network of every cell and gene, reused once any test computed it."""
    return store.regulons.grn(
        store.snapshot_cell_selection("everyone"),
        features=store.features.universe(),
        tfs=TFS,
        **FAST,
    )


def _copy(store_path: str, tmp_path) -> str:
    """A copy of the store that a test may change without affecting others."""
    path = str(tmp_path / "copy.zarr")
    shutil.copytree(store_path, path)
    return path


@pytest.fixture(scope="module")
def store_path(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("regulons") / "store.zarr")
    # The ADT assay is one that GRN inference refuses.
    write_count_store(path, {"RNA": _counts(), "ADT": _counts()[:, :2]}, "uint16")
    store = _open(path)
    store.cells.insert("everyone", np.ones(store.cells.N, dtype=bool), overwrite=True)
    return path


def test_grn_artifact_matches_direct_inference(store_path):
    store = _open(store_path)
    cells = store.snapshot_cell_selection("everyone")
    universe = store.features.universe()

    ref = store.regulons.grn(cells, features=universe, tfs=TFS, random_seed=4, **FAST)
    result = store.regulons.load_grn(ref)

    counts = _counts().astype(np.float32)
    names = [f"RNA{index}" for index in range(counts.shape[1])]
    expected = infer_grn(
        counts,
        names,
        TFS,
        seed=4,
        parameters=GrnParameters(maxStages=150, earlyStopWindow=10),
    )
    assert isinstance(result, GrnResult)
    assert result.regulators == tuple(TFS)
    assert result.random_seed == 4
    assert result.cell_selection == cells
    pd.testing.assert_frame_equal(result.adjacencies, expected)


def test_grn_reuses_identical_requests_and_separates_seeds(store_path):
    store = _open(store_path)
    cells = store.snapshot_cell_selection("everyone")
    universe = store.features.universe()

    first = store.regulons.grn(cells, features=universe, tfs=TFS, **FAST)
    again = store.regulons.grn(
        cells, features=universe, tfs=list(reversed(TFS)) + ["RNA0"], nthreads=1, **FAST
    )
    reseeded = store.regulons.grn(
        cells, features=universe, tfs=TFS, random_seed=9, **FAST
    )

    assert again == first
    assert reseeded != first


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"cell_selection": "everyone"}, TypeError, "cell_selection must be"),
        ({"features": "universe"}, TypeError, "features must be"),
        ({"tfs": "RNA0"}, TypeError, "sequence of names"),
        ({"tfs": ["RNA0", 0]}, TypeError, "sequence of names"),
        ({"tfs": []}, ValueError, "at least one"),
        ({"tfs": ["NOPE"]}, ValueError, "None of tfs"),
        ({"from_assay": "ADT"}, TypeError, "RNAassay"),
    ],
)
def test_grn_rejects_invalid_requests(store_path, changes, error, message):
    store = _open(store_path)
    request = {
        "cell_selection": store.snapshot_cell_selection("everyone"),
        "features": store.features.universe(),
        "tfs": TFS,
        **FAST,
    }

    with pytest.raises(error, match=message):
        store.regulons.grn(**(request | changes))


def test_grn_requires_a_writable_store_and_two_cells(store_path):
    store = _open(store_path)
    store.cells.insert("first", np.arange(N_CELLS) == 0, overwrite=True)
    universe = store.features.universe()

    with pytest.raises(ValueError, match="two selected cells"):
        store.regulons.grn(
            store.snapshot_cell_selection("first"), features=universe, tfs=TFS
        )
    cells = store.snapshot_cell_selection("everyone")
    read_only = DataStore(store_path, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError):
        read_only.regulons.grn(cells, features=universe, tfs=TFS, **FAST)


def test_grn_refuses_counts_beyond_the_memory_budget(store_path):
    computed = _grn(_open(store_path))
    store = _open(store_path, mem_budget="1K")
    cells = store.snapshot_cell_selection("everyone")
    universe = store.features.universe()

    # A new network must fit the budget; an existing one is returned as is.
    with pytest.raises(MemoryError, match="memory budget"):
        store.regulons.grn(cells, features=universe, tfs=TFS, random_seed=77, **FAST)
    assert store.regulons.grn(cells, features=universe, tfs=TFS, **FAST) == computed


def test_grn_leaves_out_genes_with_repeated_names(store_path, tmp_path):
    stored = _grn(_open(store_path))
    path = _copy(store_path, tmp_path)
    # Two genes now share a name, so the stored network no longer applies.
    zarr.open_group(path, mode="r+")["RNA/featureData/names"][5] = "RNA4"
    store = _open(path)

    ref = _grn(store)

    links = store.regulons.load_grn(ref).adjacencies
    assert ref != stored
    assert "RNA4" not in {*links["TF"], *links["target"]}


@pytest.mark.parametrize(
    ("grn", "error", "message"),
    [
        ("grn", TypeError, "ArtifactRef"),
        (
            ArtifactRef(scope="datastore", kind="cell_selection", artifact_id="0" * 64),
            ValueError,
            "grn_adjacencies",
        ),
        (
            ArtifactRef(
                scope="assay", assay="RNA", kind="grn_adjacencies", artifact_id="0" * 64
            ),
            ValueError,
            "missing or incomplete",
        ),
    ],
)
def test_load_grn_requires_a_stored_grn_artifact(store_path, grn, error, message):
    with pytest.raises(error, match=message):
        _open(store_path).regulons.load_grn(grn)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda group: group.attrs.update(layout="dense"), "unsupported layout"),
        (
            lambda group: group.attrs.update(
                provenance={**group.attrs["provenance"], "parameters": {}}
            ),
            "invalid provenance",
        ),
        (lambda group: group.attrs.update(cell_digest="0"), "cells do not match"),
        (lambda group: group["importance"].resize((0,)), "misaligned arrays"),
        (lambda group: group.attrs.update(regulator_digest="0"), "regulators"),
        (lambda group: group["tf_index"].set_basic_selection(0, -1), "tf_index"),
    ],
)
def test_load_grn_rejects_corrupt_artifacts(store_path, tmp_path, corrupt, message):
    ref = _grn(_open(store_path))
    path = _copy(store_path, tmp_path)
    corrupt(artifact_group(zarr.open_group(path, mode="r+"), ref))

    with pytest.raises(ValueError, match=message):
        _open(path).regulons.load_grn(ref)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"assay": "ADT"}, "result assay"),
        ({"adjacencies": pd.DataFrame(columns=["TF", "target"])}, "columns"),
        ({"method": "genie3"}, "Unknown GRN method"),
        ({"regulators": ()}, "at least one regulator"),
    ],
)
def test_grn_result_rejects_inconsistent_fields(store_path, changes, message):
    store = _open(store_path)
    result = store.regulons.load_grn(_grn(store))

    with pytest.raises(ValueError, match=message):
        replace(result, **changes)


def test_profiles_are_log_cpm_of_each_groups_summed_counts(store_path):
    store = _open(store_path)
    groups = np.arange(N_CELLS) % 3
    store.cells.insert("thirds", groups, overwrite=True)

    profiles = store.regulons.profiles("thirds")

    counts = _counts().astype(np.float64)
    sums = np.vstack([counts[groups == group].sum(axis=0) for group in range(3)])
    expected = np.log1p(sums / sums.sum(axis=1, keepdims=True) * 1e6)
    assert profiles.columns.tolist() == [f"RNA{index}" for index in range(8)]
    np.testing.assert_allclose(profiles.to_numpy(), expected)
