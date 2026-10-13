import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.regulons.grn import GrnParameters, infer_grn, pearson_correlations
from cytearc.regulons.results import GrnResult
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


@pytest.fixture(scope="module")
def store_path(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("regulons") / "store.zarr")
    write_count_store(path, {"RNA": _counts()}, "uint16")
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
    expected["correlation"] = pearson_correlations(counts, names, expected)
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


def test_grn_requires_a_writable_store_and_known_regulators(store_path):
    store = _open(store_path)
    cells = store.snapshot_cell_selection("everyone")
    universe = store.features.universe()

    with pytest.raises(ValueError, match="None of tfs"):
        store.regulons.grn(cells, features=universe, tfs=["NOPE"], **FAST)
    with pytest.raises(TypeError, match="sequence of names"):
        store.regulons.grn(cells, features=universe, tfs="RNA0", **FAST)

    read_only = DataStore(store_path, default_assay="RNA", zarr_mode="r")
    with pytest.raises(PermissionError):
        read_only.regulons.grn(cells, features=universe, tfs=TFS, **FAST)


def test_grn_refuses_counts_beyond_the_memory_budget(store_path):
    store = _open(store_path, mem_budget="1K")
    cells = store.snapshot_cell_selection("everyone")
    universe = store.features.universe()

    with pytest.raises(MemoryError, match="memory budget"):
        store.regulons.grn(cells, features=universe, tfs=TFS, **FAST)
