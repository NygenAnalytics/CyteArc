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
from cytearc.utils.arrays import sort_categories
from cytearc.utils.memory_budget import MemoryLedger
from tests.memory_helpers import climb_refusals
from tests.regulon_helpers import csi_by_definition, rss_by_definition
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


def _scored_store(
    folder, *, n_cells: int, n_sets: int, chunk_rows: int | None = None
) -> tuple[str, ArtifactRef]:
    """A store of synthetic counts and its AUCell scores of every cell, and their artifact.

    The scores are of ``n_sets`` sets of 15 genes and a set ``"silent"`` of
    10 genes that no cell expresses, which score 0 in every cell. The cells
    carry ``"groups"``, seven groups and one of a single cell, and
    ``"many"``, 40 groups numbered from 0. With ``chunk_rows``, the scores
    are stored again in chunks of that many cells, as scores of many cells
    are stored in chunks of up to 128 MiB.
    """
    rng = np.random.default_rng(n_cells + n_sets)
    counts = rng.poisson(rng.uniform(0.2, 4.0, size=150), size=(n_cells, 150))
    counts[:, 140:] = 0
    path = str(folder / "scored.zarr")
    write_count_store(path, {"RNA": counts}, "uint16")
    store = _open(path)
    store.cells.insert("everyone", np.ones(n_cells, dtype=bool), overwrite=True)
    groups = np.asarray([f"g{code}" for code in rng.integers(0, 7, n_cells)])
    groups[0] = "lone"
    store.cells.insert("groups", groups.astype(object), overwrite=True)
    store.cells.insert("many", rng.integers(0, 40, n_cells), overwrite=True)
    targets = [rng.choice(140, 15, replace=False) for _ in range(n_sets)]
    net = pd.DataFrame(
        {
            "source": np.repeat([f"set{index}" for index in range(n_sets)], 15).tolist()
            + ["silent"] * 10,
            "target": [f"RNA{gene}" for genes in targets for gene in genes]
            + [f"RNA{gene}" for gene in range(140, 150)],
        }
    )
    auc = store.scores.aucell(
        net,
        store.snapshot_cell_selection("everyone"),
        features=store.features.universe(),
        tmin=5,
    )
    if chunk_rows is not None:
        _store_scores_in_chunks(path, auc, chunk_rows)
    return path, auc


def _store_scores_in_chunks(path: str, auc: ArtifactRef, rows: int) -> None:
    """Store the scores of an enrichment artifact again in chunks of ``rows`` cells."""
    group = artifact_group(zarr.open_group(path, mode="r+"), auc)
    values = np.asarray(group["scores"][:])
    attributes = dict(group["scores"].attrs)
    del group["scores"]
    scores = group.create_array(
        "scores", shape=values.shape, dtype=values.dtype, chunks=(rows, values.shape[1])
    )
    scores[:] = values
    scores.attrs.update(attributes)


def _activity(store: DataStore, auc: ArtifactRef, column: str):
    """The scores of every scored cell as a table, and the cells' labels in ``column``."""
    activity = store.scores.load(auc).to_pandas()
    labels = np.asarray(store.cells.fetch_all(column))[activity.index.to_numpy()]
    return activity.to_numpy(), labels


@pytest.fixture(scope="module", params=[(240, 5), (1200, 24)], ids=["6x240", "25x1200"])
def scored(request, tmp_path_factory) -> tuple[str, ArtifactRef]:
    n_cells, n_sets = request.param
    return _scored_store(
        tmp_path_factory.mktemp("scored"), n_cells=n_cells, n_sets=n_sets
    )


class _RecordingLedger(MemoryLedger):
    """A ledger that records each charge and the most bytes held at once, peaks included."""

    def __init__(self, *arguments, **options) -> None:
        super().__init__(*arguments, **options)
        self.charges: list[tuple[str, int]] = []
        self.most = 0

    def hold(self, item, nbytes, *, remedy=None):
        held = super().hold(item, nbytes, remedy=remedy)
        self.charges.append((item, int(nbytes)))
        self.most = max(self.most, self.held_bytes)
        return held

    def peak(self, item, nbytes, *, remedy=None):
        super().peak(item, nbytes, remedy=remedy)
        self.charges.append((item, int(nbytes)))
        self.most = max(self.most, self.held_bytes + int(nbytes))


_BLOCK = "the scores read one block of cells at a time"


@pytest.fixture
def ledgers(monkeypatch) -> list[_RecordingLedger]:
    """The ledgers of the specificity calls, which record their charges."""
    from cytearc.datastore._operations import regulons

    made: list[_RecordingLedger] = []

    def recording(*arguments, **options) -> _RecordingLedger:
        made.append(_RecordingLedger(*arguments, **options))
        return made[-1]

    monkeypatch.setattr(regulons, "MemoryLedger", recording)
    return made


@pytest.fixture
def blocks_read(monkeypatch) -> list[int]:
    """The cells of each block that the regulon specificity kernels sum in their first pass."""
    from cytearc.regulons import specificity

    rows: list[int] = []
    add_totals = specificity._add_totals

    def counted(values, totals):
        rows.append(len(values))
        return add_totals(values, totals)

    monkeypatch.setattr(specificity, "_add_totals", counted)
    return rows


@pytest.fixture(scope="module")
def chunked(tmp_path_factory) -> dict[int, tuple[str, ArtifactRef]]:
    """Scores of 200 regulons over 400 and 4,000 cells, stored in chunks of 100 cells."""
    return {
        n_cells: _scored_store(
            tmp_path_factory.mktemp(f"chunked{n_cells}"),
            n_cells=n_cells,
            n_sets=199,
            chunk_rows=100,
        )
        for n_cells in (400, 4000)
    }


@pytest.mark.parametrize("column", ["groups", "many"])
def test_specificity_streams_the_scores_that_regulon_specificity_reads(scored, column):
    path, auc = scored
    store = _open(path)

    rss = store.regulons.specificity(auc, column)

    values, labels = _activity(store, auc, column)
    expected = rss_by_definition(values, labels)
    # The groups in their natural order, so 2 before 10.
    assert rss.index.name == "group"
    assert rss.index.tolist() == sort_categories(expected.index.tolist())
    assert rss.columns.tolist() == store.scores.load(auc).source_names.tolist()
    np.testing.assert_allclose(
        rss.to_numpy(), expected.loc[rss.index].to_numpy(), rtol=0, atol=1e-12
    )
    assert rss["silent"].isna().all()
    assert not rss.drop(columns="silent").isna().any(axis=None)


def test_specificity_reads_a_selection_of_the_scored_cells_and_label_artifacts(
    scored,
):
    path, auc = scored
    store = _open(path)
    everyone = store.snapshot_cell_selection("everyone")
    half = np.arange(store.cells.N) % 2 == 0
    store.cells.insert("half", half, overwrite=True)
    some = store.snapshot_cell_selection("half")

    rss = store.regulons.specificity(auc, "groups", cell_selection=some)

    values, labels = _activity(store, auc, "groups")
    expected = rss_by_definition(values[half], labels[half])
    np.testing.assert_allclose(
        rss.to_numpy(), expected.loc[rss.index].to_numpy(), rtol=0, atol=1e-12
    )
    # Labels frozen as an artifact give the scores of their column.
    frozen = store.snapshot_cluster_labels("groups", cell_selection=everyone)
    by_column = store.regulons.specificity(auc, "groups")
    by_artifact = store.regulons.specificity(auc, frozen)
    assert by_artifact.index.tolist() == by_column.index.tolist()
    np.testing.assert_array_equal(by_artifact.to_numpy(), by_column.to_numpy())
    # An artifact that labels only some of the scored cells serves them alone.
    narrow = store.snapshot_cluster_labels("groups", cell_selection=some)
    with pytest.raises(ValueError, match="holds no label for some cells to score"):
        store.regulons.specificity(auc, narrow)
    np.testing.assert_array_equal(
        store.regulons.specificity(auc, narrow, cell_selection=some).to_numpy(),
        rss.to_numpy(),
    )


def test_specificity_does_not_depend_on_the_chunks_of_the_stored_scores(
    scored, tmp_path, blocks_read
):
    path, auc = scored
    store = _open(path)
    rss = store.regulons.specificity(auc, "many")
    csi = store.regulons.connection_specificity(auc)

    for rows in (3, 37):
        copy = _copy(path, tmp_path / f"chunks{rows}")
        _store_scores_in_chunks(copy, auc, rows)
        rechunked = _open(copy)
        blocks_read.clear()

        np.testing.assert_allclose(
            rechunked.regulons.specificity(auc, "many").to_numpy(),
            rss.to_numpy(),
            rtol=0,
            atol=1e-12,
        )
        # Each block holds the cells of one stored chunk.
        assert max(blocks_read) == rows
        np.testing.assert_array_equal(
            rechunked.regulons.connection_specificity(auc).to_numpy(), csi.to_numpy()
        )


def test_specificity_refuses_unlabelled_or_unscored_cells_and_negative_scores(
    scored, tmp_path
):
    path, auc = scored
    copy = _copy(path, tmp_path)
    store = _open(copy)
    labels = np.asarray(store.cells.fetch_all("groups"), dtype=object)
    labels[3] = None
    store.cells.insert("unlabelled", labels, overwrite=True)
    with pytest.raises(
        ValueError, match=f"groups holds no label for 1 of the {store.cells.N} cells"
    ):
        store.regulons.specificity(auc, "unlabelled")

    # Scores of some of the cells hold none of the others.
    store.cells.insert("first", np.arange(store.cells.N) < 100, overwrite=True)
    everyone = store.snapshot_cell_selection("everyone")
    first_auc = store.scores.aucell(
        pd.DataFrame({"source": "set0", "target": [f"RNA{gene}" for gene in range(9)]}),
        store.snapshot_cell_selection("first"),
        features=store.scores.load(auc).feature_selection,
        tmin=5,
    )
    with pytest.raises(ValueError, match="holds cells that scores did not score"):
        store.regulons.specificity(first_auc, "groups", cell_selection=everyone)
    with pytest.raises(ValueError, match="holds cells that scores did not score"):
        store.regulons.connection_specificity(first_auc, cell_selection=everyone)

    # A score below 0, as chromVAR's z-scores can be, has no share of activity.
    artifact_group(zarr.open_group(copy, mode="r+"), auc)["scores"][5, 1] = -0.25
    with pytest.raises(
        ValueError,
        match="The scores of the aucell artifact must be finite and non-negative",
    ):
        store.regulons.specificity(auc, "groups")


def test_specificity_rejects_invalid_requests(scored):
    path, auc = scored
    store = _open(path)
    everyone = store.snapshot_cell_selection("everyone")

    with pytest.raises(TypeError, match="scores must be an ArtifactRef"):
        store.regulons.specificity("auc", "groups")
    with pytest.raises(ValueError, match="not a cell_selection artifact"):
        store.regulons.connection_specificity(everyone)
    with pytest.raises(TypeError, match="ArtifactRef or a cell column name"):
        store.regulons.specificity(auc, 3)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.regulons.specificity(auc, "groups", cell_selection="everyone")


def test_connection_specificity_streams_the_correlations_of_the_scores(scored):
    path, auc = scored
    store = _open(path)
    half = np.arange(store.cells.N) % 2 == 0
    store.cells.insert("half", half, overwrite=True)
    some = store.snapshot_cell_selection("half")

    csi = store.regulons.connection_specificity(auc)
    part = store.regulons.connection_specificity(auc, margin=0.2, cell_selection=some)

    values, _ = _activity(store, auc, "groups")
    names = store.scores.load(auc).source_names.tolist()
    assert csi.index.tolist() == csi.columns.tolist() == names
    np.testing.assert_array_equal(csi.to_numpy(), csi_by_definition(values))
    np.testing.assert_array_equal(part.to_numpy(), csi_by_definition(values[half], 0.2))
    with pytest.raises(ValueError, match="margin must be finite"):
        store.regulons.connection_specificity(auc, margin=float("inf"))
    with pytest.raises(TypeError, match="margin must be a real number"):
        store.regulons.connection_specificity(auc, margin="0.05")


def test_specificity_charges_a_block_of_scores_that_follows_the_budget_not_the_cells(
    chunked, ledgers, blocks_read
):
    # One cell's float32 scores of 200 regulons.
    per_row = 4 * 200
    charged: dict[int, list[int]] = {}
    for n_cells, (path, auc) in chunked.items():
        blocks_read.clear()
        _open(path).regulons.specificity(auc, "groups")
        charged[n_cells] = [
            size for item, size in ledgers[-1].charges if item == _BLOCK
        ]
        # A block holds the 100 cells of one stored chunk, in each pass.
        assert len(charged[n_cells]) == 2 and max(blocks_read) == 100
    assert charged[400] == charged[4000]

    path, auc = chunked[400]
    default = _open(path).regulons.specificity(auc, "groups")

    def score(budget: int) -> pd.DataFrame:
        blocks_read.clear()
        return _open(path, mem_budget=budget).regulons.specificity(auc, "groups")

    # The budget that the refusals name, from 1 KiB, is the least that runs:
    # its last refusal is the block's, so it streams the scores a cell at a
    # time, within its budget, and gives the scores of the default budget.
    refusals, scores = climb_refusals(score, start=1024)
    ledger = ledgers[-1]
    assert refusals[-1].item == _BLOCK
    assert ledger.most <= ledger.budget == refusals[-1].needed_bytes
    assert set(blocks_read) == {1} and len(blocks_read) == 400
    blocks = [size for item, size in ledger.charges if item == _BLOCK]
    assert charged[400][0] - blocks[0] == (100 - 1) * per_row
    np.testing.assert_allclose(
        scores.to_numpy(), default.to_numpy(), rtol=0, atol=1e-12
    )
    # Its budget holds less than the cells' float32 scores.
    assert ledger.budget < 400 * per_row


def test_connection_specificity_charges_a_block_of_scores_and_their_deviations(
    chunked, ledgers
):
    # One cell's float32 scores of 200 regulons and their float64 deviations.
    per_row = 4 * 200 + 8 * 200
    charged: dict[int, list[int]] = {}
    for n_cells, (path, auc) in chunked.items():
        _open(path).regulons.connection_specificity(auc)
        charges = ledgers[-1].charges
        charged[n_cells] = [size for item, size in charges if item == _BLOCK]
        # The sums of products of deviations of every pair of regulons and a
        # block's products, float64, beside three float64 values of each
        # regulon, whatever the cells.
        correlations = 8 * (2 * 200 * 200 + 3 * 200)
        assert ("the correlations of 200 regulons", correlations) in charges
    assert len(charged[400]) == 1 and charged[400] == charged[4000]

    path, auc = chunked[400]
    default = _open(path).regulons.connection_specificity(auc)

    def correlate(budget: int) -> pd.DataFrame:
        return _open(path, mem_budget=budget).regulons.connection_specificity(auc)

    refusals, csi = climb_refusals(correlate, start=1024)
    ledger = ledgers[-1]
    assert refusals[-1].item == _BLOCK
    assert ledger.most <= ledger.budget == refusals[-1].needed_bytes
    blocks = [size for item, size in ledger.charges if item == _BLOCK]
    assert charged[400][0] - blocks[0] == (100 - 1) * per_row
    np.testing.assert_array_equal(csi.to_numpy(), default.to_numpy())
    # A budget without room for the correlations names fewer regulons too.
    refused = [r for r in refusals if r.item == "the correlations of 200 regulons"]
    assert len(refused) == 1
    assert str(refused[0]).endswith(", or score fewer regulons.")
