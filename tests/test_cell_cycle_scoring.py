"""Cell-cycle and gene-set scoring: repeated genes, renamed genes, and reads.

Stores here name their RNA features after the default cell-cycle genes, so
scoring follows the lists exactly as it does on real data.
"""

import shutil
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import cytearc.storage.operation_revisions as operation_revisions
from cytearc import DataStore
from cytearc.assay.feature_summary import (
    ensure_feature_summary,
    feature_summary_selected_count,
    feature_summary_values,
)
from cytearc.assay.rna import RNAassay
from cytearc.features.scoring import binned_sampling
from cytearc.quality_control import cell_cycle_previous_symbols
from cytearc.quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    match_gene_names,
    s_phase_genes,
)
from cytearc.storage import feature_major_rows
from cytearc.storage.operation_revisions import _repeats_cell_cycle_gene
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.storage.stores import load_zarr
from cytearc.utils.logging import logger
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts

N_CELLS = 150
FILLERS = [f"GENE{index}" for index in range(40)]
# The store predates the renames of PIMREG and JPT1.
RENAMED = {"PIMREG": "FAM64A", "JPT1": "HN1"}


def _write_named_store(path: Path, names: list[str], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.05, 3.0, size=len(names))
    counts = rng.poisson(means, size=(N_CELLS, len(names))).astype(np.uint16)
    counts[:, 0] += 1
    root = load_zarr(zarr_loc=str(path), mode="w")
    cell_ids = np.array([f"cell{index}" for index in range(N_CELLS)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    array = create_zarr_count_assay(
        root,
        "RNA",
        None,
        N_CELLS,
        feat_ids=np.array([f"ENSG{index:05d}" for index in range(len(names))]),
        feat_names=np.array(names),
        dtype="uint16",
    )
    array[:] = counts
    finalize_test_counts(array)
    finalize_writer_counts_t(root, "RNA", None)


def _open(path: Path) -> DataStore:
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


@pytest.fixture(scope="module")
def renamed_template(tmp_path_factory) -> Path:
    """Default genes under their 2019 symbols except PIMREG and JPT1."""
    path = tmp_path_factory.mktemp("cell_cycle") / "renamed.zarr"
    names = [RENAMED.get(gene, gene) for gene in [*s_phase_genes, *g2m_phase_genes]]
    _write_named_store(path, [*names, *FILLERS])
    _open(path)
    return path


@pytest.fixture(scope="module")
def current_template(tmp_path_factory) -> Path:
    """Default genes under their 2019 symbols, so no alias is needed."""
    path = tmp_path_factory.mktemp("cell_cycle") / "current.zarr"
    _write_named_store(path, [*s_phase_genes, *g2m_phase_genes, *FILLERS], seed=1)
    _open(path)
    return path


def _copy(template: Path, tmp_path: Path) -> DataStore:
    target = tmp_path / "store.zarr"
    shutil.copytree(template, target)
    return _open(target)


def _scores(store: DataStore, ref) -> dict[str, np.ndarray]:
    group = store.artifacts.load(ref)
    return {
        name: np.asarray(group[name][:]) for name in ("s_score", "g2m_score", "phase")
    }


def _feature_means(store: DataStore, cells) -> np.ndarray:
    """Return the log1p feature means that cell-cycle scoring bins by."""
    summary = ensure_feature_summary(store.zw, store.RNA, cells, log_transform=True)
    n_cells = feature_summary_selected_count(store.zw, cells)
    return np.asarray(
        feature_summary_values(store.zw, summary, n_selected=n_cells)["avg"]
    )


# Gene matching with renamed genes


def test_genes_match_names_without_case_then_by_another_hgnc_symbol():
    names = np.array(
        ["Mcm5", "MCM5", "Fam64a", "MLF1IP", "CENPU", "CDC2", "TOP2A", "RecA"]
    )

    match = match_gene_names(
        names,
        ["MCM5", "PIMREG", "CENPU", "MLF1IP", "CDK1", "TOP2", "RAD51", "POLA1", "X"],
    )

    # MCM5 names two features. PIMREG falls back to its previous symbol, the
    # previous symbol TOP2 to the current TOP2A, and RAD51 past RAD51A to its
    # second previous symbol. CENPU and MLF1IP both exist, so neither needs
    # the other. CDC2 also names another gene, so CDK1 never matches it, and
    # no symbol of POLA1 is present.
    assert match.indices == [0, 1, 2, 4, 3, 6, 7]
    assert match.aliases == {"PIMREG": "Fam64a", "TOP2": "TOP2A", "RAD51": "RecA"}
    assert match.unmatched == ["CDK1", "POLA1", "X"]


def test_previous_symbols_name_one_gene_each():
    symbols = [
        symbol
        for current, previous in cell_cycle_previous_symbols.items()
        for symbol in (current, *previous)
    ]
    defaults = {*s_phase_genes, *g2m_phase_genes}

    # The renames of Seurat's 2019 list, among others.
    assert cell_cycle_previous_symbols["CENPU"] == ("MLF1IP",)
    assert cell_cycle_previous_symbols["PIMREG"] == ("FAM64A",)
    assert cell_cycle_previous_symbols["JPT1"] == ("HN1",)
    assert set(cell_cycle_previous_symbols) <= defaults
    assert len(symbols) == len({symbol.upper() for symbol in symbols})
    # Symbols that also name other genes are left out.
    assert not {"CDC2", "TS", "CCNL1", "CDC25", "SD"} & set(symbols)
    previous = set(symbols) - set(cell_cycle_previous_symbols)
    assert not previous & defaults
    with pytest.raises(TypeError):
        cell_cycle_previous_symbols["CDK1"] = ("CDC2",)  # type: ignore[index]


def test_renamed_default_genes_match_their_previous_symbols(renamed_template, tmp_path):
    store = _copy(renamed_template, tmp_path)
    cells = store.snapshot_cell_selection("I")
    names = np.asarray(store.RNA.feats.fetch_all("names")).astype(str)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        ref = store.scores.cell_cycle(cells)
    finally:
        logger.remove(sink)

    parameters = store.artifacts.inspect(ref).parameters
    assert parameters["gene_aliases"] == {"PIMREG": "FAM64A", "JPT1": "HN1"}
    expected = [
        int(np.flatnonzero(names == RENAMED.get(gene, gene))[0])
        for gene in g2m_phase_genes
    ]
    assert parameters["g2m_gene_indices"] == expected
    assert parameters["s_gene_indices"] == list(range(len(s_phase_genes)))
    assert (
        "2 of 54 g2m_genes matched another HGNC symbol of the same gene: "
        "'PIMREG' as 'FAM64A', 'JPT1' as 'HN1'"
    ) in messages
    assert not [message for message in messages if "were not found" in message]


def test_scores_that_need_no_alias_record_none(current_template, tmp_path):
    store = _copy(current_template, tmp_path)
    cells = store.snapshot_cell_selection("I")

    ref = store.scores.cell_cycle(cells)

    parameters = store.artifacts.inspect(ref).parameters
    assert "gene_aliases" not in parameters
    assert parameters["g2m_gene_indices"] == list(
        range(len(s_phase_genes), len(s_phase_genes) + len(g2m_phase_genes))
    )
    assert store.artifacts.inspect(ref).revision == 1


# Repeated genes


def test_a_repeated_gene_counts_once_in_cell_cycle_scores(current_template, tmp_path):
    store = _copy(current_template, tmp_path)
    cells = store.snapshot_cell_selection("I")
    s_genes = list(s_phase_genes)
    # MCM5 eleven times, and S genes again under other cases.
    repeated = ["MCM5"] * 10 + s_genes + [gene.lower() for gene in s_genes[:5]]

    plain = store.scores.cell_cycle(cells, s_genes=s_genes, ctrl_size=43)
    weighted = store.scores.cell_cycle(cells, s_genes=repeated, ctrl_size=43)

    assert weighted != plain
    status = store.artifacts.inspect(weighted)
    assert status.parameters["s_gene_indices"][:11] == [0] * 11
    assert len(status.parameters["s_gene_indices"]) == 10 + 43 + 5
    assert status.revision == 2
    assert store.artifacts.inspect(plain).revision == 1
    plain_scores, weighted_scores = _scores(store, plain), _scores(store, weighted)
    for name, values in plain_scores.items():
        np.testing.assert_array_equal(weighted_scores[name], values)

    # The S score is the mean of the 43 S genes minus their controls, which
    # repeating MCM5 used to weigh eleven times.
    means = _feature_means(store, cells)
    controls = binned_sampling(pd.Series(means), list(range(43)), 43, 50, 4466)
    normed = np.log1p(
        1000
        * store.RNA.rawData[:, :].compute().astype(np.float64)
        / store.cells.fetch_all("RNA_nCounts")[:, None]
    )
    expected = normed[:, :43].mean(axis=1) - normed[:, controls].mean(axis=1)
    np.testing.assert_allclose(plain_scores["s_score"], expected, rtol=0, atol=1e-12)
    old = (
        10 * normed[:, 0] + normed[:, :43].sum(axis=1) + normed[:, :5].sum(axis=1)
    ) / 58
    assert np.abs(old - normed[:, controls].mean(axis=1) - expected).max() > 1e-3


def test_score_features_counts_a_repeated_name_once(current_template, tmp_path):
    store = _copy(current_template, tmp_path)
    genes = ["MCM5", "PCNA", "TYMS"]

    single = store.RNA.score_features(genes, "I", 5, 10, 7, log_transform=True)
    repeated = store.RNA.score_features(
        genes + ["mcm5", "MCM5", "Pcna"], "I", 5, 10, 7, log_transform=True
    )

    np.testing.assert_array_equal(repeated, single)


def test_repeated_index_predicate_is_total_over_recorded_parameters():
    assert _repeats_cell_cycle_gene("cell_cycle", {"s_gene_indices": [1, 2, 1]}, {})
    assert _repeats_cell_cycle_gene(
        "cell_cycle", {"s_gene_indices": [1, 2], "g2m_gene_indices": [3, 3]}, {}
    )
    for parameters in (
        {},
        {"s_gene_indices": [1, 2], "g2m_gene_indices": [3, 4]},
        {"s_gene_indices": "1,1", "g2m_gene_indices": None},
        {"s_gene_indices": [True, True], "g2m_gene_indices": [1.0, 1.0]},
        {"s_gene_indices": [[1], [1]]},
        {"s_gene_indices": [{"a": 1}, {"a": 1}]},
    ):
        assert not _repeats_cell_cycle_gene("cell_cycle", parameters, {})


def test_scores_of_revision_1_are_recomputed_only_when_they_repeat_a_gene(
    current_template, tmp_path, monkeypatch
):
    store = _copy(current_template, tmp_path)
    cells = store.snapshot_cell_selection("I")
    repeated = ["MCM5", *s_phase_genes]
    monkeypatch.setattr(
        operation_revisions, "OPERATION_REVISIONS", MappingProxyType({})
    )
    legacy_plain = store.scores.cell_cycle(cells)
    legacy_repeated = store.scores.cell_cycle(cells, s_genes=repeated)
    assert store.artifacts.inspect(legacy_repeated).revision == 1
    monkeypatch.undo()

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        plain = store.scores.cell_cycle(cells)
        current = store.scores.cell_cycle(cells, s_genes=repeated)
    finally:
        logger.remove(sink)

    # The scoped revision leaves scores without a repeated gene current.
    assert plain == legacy_plain
    assert store.artifacts.inspect(plain).is_current
    assert current != legacy_repeated
    assert store.artifacts.inspect(current).revision == 2
    legacy_status = store.artifacts.inspect(legacy_repeated)
    assert not legacy_status.is_current
    assert [entry.revision for entry in legacy_status.superseded_by] == [2]
    (line,) = [message for message in messages if "Recomputing" in message]
    assert line.startswith(
        f"Recomputing run_cell_cycle_scoring: artifact "
        f"{legacy_repeated.artifact_id[:12]} is revision 1, current 2: A gene index"
    )


# One pass over the counts


def test_cell_cycle_scoring_reads_both_gene_sets_in_one_pass(
    current_template, tmp_path, monkeypatch
):
    store = _copy(current_template, tmp_path)
    cells = store.snapshot_cell_selection("I")
    s_genes, g2m_genes = list(s_phase_genes), list(g2m_phase_genes)
    means = _feature_means(store, cells)
    cell_idx = store.cells.active_index("I")
    # Each set on its own, as earlier releases scored them.
    separate = [
        store.RNA._score_feature_indices(
            np.arange(start, stop),
            cell_idx,
            means,
            ctrl_size=43,
            n_bins=50,
            rand_seed=4466,
            log_transform=True,
        )
        for start, stop in ((0, 43), (43, 97))
    ]
    passes: list[int] = []
    original = RNAassay._mean_normed_union

    def counted(self, *args, **kwargs):
        passes.append(len(args[3]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RNAassay, "_mean_normed_union", counted)

    ref = store.scores.cell_cycle(
        cells, s_genes=s_genes, g2m_genes=g2m_genes, rand_seed=4466
    )

    # One read holds the targets and controls of both phases.
    assert passes == [4]
    scores = _scores(store, ref)
    np.testing.assert_array_equal(scores["s_score"], separate[0])
    np.testing.assert_array_equal(scores["g2m_score"], separate[1])


@pytest.mark.parametrize("log_transform", [False, True])
def test_group_means_from_counts_and_counts_t_are_identical(
    current_template, tmp_path, monkeypatch, log_transform
):
    store = _copy(current_template, tmp_path)
    groups = {"a": np.array([3, 70, 5]), "b": np.array([90, 1, 44, 12, 3])}
    cells = np.arange(0, N_CELLS, 3)
    used: list[str] = []
    real_iter = feature_major_rows.iter_feature_major_rows

    def from_counts_t(*args, **kwargs):
        used.append("countsT")
        return real_iter(*args, **kwargs)

    monkeypatch.setattr(feature_major_rows, "iter_feature_major_rows", from_counts_t)
    monkeypatch.setattr(
        feature_major_rows, "feature_major_row_scratch", lambda *args, **kwargs: None
    )
    from_counts = store.RNA._mean_normed_feature_groups(
        cells, groups, log_transform=log_transform
    )
    assert used == []
    monkeypatch.setattr(
        feature_major_rows, "feature_major_row_scratch", lambda *args, **kwargs: 1 << 20
    )
    transposed = store.RNA._mean_normed_feature_groups(
        cells, groups, log_transform=log_transform
    )
    assert used == ["countsT"]

    raw = store.RNA.rawData[:, :].compute().astype(np.float64)[cells]
    normed = 1000 * raw / store.cells.fetch_all("RNA_nCounts")[cells, None]
    if log_transform:
        normed = np.log1p(normed)
    for key, indices in groups.items():
        np.testing.assert_array_equal(transposed[key], from_counts[key])
        np.testing.assert_allclose(
            transposed[key], normed[:, indices].mean(axis=1), rtol=0, atol=1e-12
        )


def test_unordered_cells_are_read_from_counts(current_template, tmp_path, monkeypatch):
    store = _copy(current_template, tmp_path)
    groups = {"a": np.array([3, 70, 5])}
    ordered = np.arange(0, N_CELLS, 2)
    shuffled = np.random.default_rng(0).permutation(ordered)
    probed: list[bool] = []
    real_scratch = feature_major_rows.feature_major_row_scratch

    def scratch(*args, **kwargs):
        probed.append(True)
        return real_scratch(*args, **kwargs)

    monkeypatch.setattr(feature_major_rows, "feature_major_row_scratch", scratch)

    means = store.RNA._mean_normed_feature_groups(shuffled, groups)
    assert probed == []
    reference = store.RNA._mean_normed_feature_groups(ordered, groups)

    # Cells out of order can only be read from counts, in their own order.
    order = np.searchsorted(ordered, shuffled)
    np.testing.assert_array_equal(means["a"], reference["a"][order])
    assert probed == [True]
