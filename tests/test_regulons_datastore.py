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
from tests.storage_helpers import record_normalized_values, write_count_store

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


def test_grn_records_the_values_that_it_reads(store_path, tmp_path):
    store = _open(store_path)
    counts = _grn(store)
    assert store.artifacts.inspect(counts).parameters["values"] == "raw_counts"
    # The same stored values, recorded as normalized before import.
    path = _copy(store_path, tmp_path)
    record_normalized_values(path, "RNA", "normalized")
    normalized = _open(path)
    assert not normalized.RNA.holdsCounts
    ref = _grn(normalized)
    assert ref != counts
    status = normalized.artifacts.inspect(ref)
    assert status.parameters["values"] == "normalized_values"
    # Equal values give an equal network, whatever they hold.
    pd.testing.assert_frame_equal(
        normalized.regulons.load_grn(ref).adjacencies,
        store.regulons.load_grn(counts).adjacencies,
    )


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
    # Rows are labelled by the groups' integer labels, as compute_bulk labels
    # its columns.
    assert profiles.index.tolist() == [0, 1, 2]
    assert profiles.index.name == "group"
    np.testing.assert_allclose(profiles.to_numpy(), expected)


def test_grn_names_the_budget_of_its_matrix_and_links(store_path):
    from cytearc.utils.memory_budget import MemoryBudgetError

    store = _open(store_path)
    cells = store.snapshot_cell_selection("everyone")
    universe = store.features.universe()

    def infer(budget: int) -> MemoryBudgetError | None:
        small = _open(store_path, mem_budget=budget)
        try:
            small.regulons.grn(
                cells, features=universe, tfs=TFS, random_seed=78, **FAST
            )
        except MemoryBudgetError as refusal:
            return refusal
        return None

    refusal = infer(1024)
    assert refusal is not None
    assert (refusal.budget_bytes, refusal.item) == (
        1024,
        "the expression matrix and the links of GRN inference",
    )
    assert f"needs about {refusal.needed_bytes} bytes" in str(refusal)
    below = infer(refusal.needed_bytes - 1)
    assert below is not None and below.needed_bytes == refusal.needed_bytes
    later = infer(refusal.needed_bytes)
    assert later is None or later.item != refusal.item


def test_grn_runs_as_many_worker_processes_as_the_budget_holds(
    store_path, tmp_path, monkeypatch
):
    from cytearc.regulons import grn
    from cytearc.utils.logging import logger
    from cytearc.utils.memory_budget import MemoryBudgetError

    path = _copy(store_path, tmp_path)
    workers: list[int] = []
    grn_links = grn._grn_links

    def record_workers(*args, **kwargs):
        workers.append(kwargs["nthreads"])
        return grn_links(*args, **kwargs)

    monkeypatch.setattr(grn, "_grn_links", record_workers)

    def infer(budget: int, seed: int) -> None:
        store = DataStore(
            path,
            default_assay="RNA",
            min_features_per_cell=-1,
            nthreads=4,
            mem_budget=budget,
        )
        store.regulons.grn(
            store.snapshot_cell_selection("everyone"),
            features=store.features.universe(),
            tfs=TFS,
            random_seed=seed,
            **FAST,
        )

    with pytest.raises(MemoryBudgetError) as refused:
        infer(1024, 80)
    # The refusal names the expression, the links and one fit in this
    # process; each worker process holds its own copy of the four
    # regulators and its own fit instead.
    fit = grn.grn_fit_bytes(N_CELLS, len(TFS), FAST["max_stages"])
    worker = grn.grn_process_bytes(N_CELLS, len(TFS), FAST["max_stages"])
    held = refused.value.needed_bytes - fit
    assert worker > fit
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        infer(held + 3 * worker, 81)
        infer(held + 3 * worker - 1, 82)
        infer(held + 2 * worker - 1, 83)
    finally:
        logger.remove(sink)

    assert workers == [3, 2, 1]
    reductions = [message for message in messages if "fits targets on" in message]
    assert len(reductions) == 3
    assert (
        f"on 3 of 4 workers: each worker process holds about {worker} bytes"
        in (reductions[0])
    )


def test_grn_identity_and_links_do_not_depend_on_the_worker_processes(
    store_path, tmp_path, monkeypatch
):
    from cytearc.regulons import grn

    pools: list[int] = []
    fit_in_processes = grn._fit_in_processes

    def record_pool(*args, **kwargs):
        pools.append(kwargs["processes"])
        return fit_in_processes(*args, **kwargs)

    monkeypatch.setattr(grn, "_IN_PROCESS_SECONDS", 0.0)
    monkeypatch.setattr(grn, "_fit_in_processes", record_pool)
    # A seed of its own, which no other test's network shares.
    request = {"features": None, "tfs": TFS, "random_seed": 37, **FAST}

    def infer(path: str, nthreads: int) -> tuple[DataStore, ArtifactRef]:
        store = DataStore(
            path, default_assay="RNA", min_features_per_cell=-1, nthreads=nthreads
        )
        ref = store.regulons.grn(
            store.snapshot_cell_selection("everyone"),
            **(request | {"features": store.features.universe()}),
        )
        return store, ref

    on_workers, computed = infer(_copy(store_path, tmp_path / "workers"), 3)
    assert pools == [3]
    # The thread count is an execution setting, so one thread reuses the
    # network that three worker processes inferred.
    assert infer(str(on_workers.zarr_loc), 1)[1] == computed
    # A copy of the store infers the same network in this process alone,
    # recorded with the same parameters.
    alone, recomputed = infer(_copy(store_path, tmp_path / "alone"), 1)
    assert pools == [3]
    assert (
        alone.artifacts.inspect(recomputed).parameters
        == on_workers.artifacts.inspect(computed).parameters
    )
    pd.testing.assert_frame_equal(
        alone.regulons.load_grn(recomputed).adjacencies,
        on_workers.regulons.load_grn(computed).adjacencies,
    )
