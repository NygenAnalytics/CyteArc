"""Every clone summary on every clone source.

Each source saves the synthetic truth's clones its own way, in a store of the
truth's cells, and every summary must measure the truth's clones whatever
saved them. The expected values are counted cell by cell from the truth, so
they never come from the code under test. A pull request that adds a clone
source adds a row to ``SOURCES``, and to ``KEPT_APART`` when its clones keep
labels apart; one that adds a summary adds a test here.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc.datastore.datastore import DataStore
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import (
    coupling_reference,
    import_truth_clones,
    import_truth_receptors,
    overlap_reference,
    write_truth_store,
)
from tests.clone_truth import N_CELLS, clone_truth, number_label, receptor_truth
from tests.storage_helpers import insert_nullable_cell_column

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()
RECEPTORS = receptor_truth(TRUTH)
SAMPLES = ("d2", "d4")


def _imported_clones(store: DataStore, folder: Path) -> ArtifactRef:
    return import_truth_clones(store, TRUTH, folder)


def _clonotypes(store: DataStore, folder: Path) -> ArtifactRef:
    # Under receptor_arms="any" a cell that lost one of its clone's chains
    # still joins the clone, so within each receptor type the clonotypes are
    # the truth's clones; its multichain and ambiguous cells are receptor
    # types of their own.
    receptors = import_truth_receptors(store, RECEPTORS, folder)
    return store.repertoire.clonotypes(receptors, receptor_arms="any")


#: How each clone source saves the truth's clones, in the order the stack adds
#: them.
SOURCES: dict[str, Callable[[DataStore, Path], ArtifactRef]] = {
    "imported clones": _imported_clones,
    "clonotypes": _clonotypes,
}
#: The labels that a source's clones keep apart, by key column, for each of the
#: truth's cells: every summary measures the cells of each label apart, so T
#: cells and B cells never share a group.
KEPT_APART: dict[str, dict[str, np.ndarray]] = {
    "clonotypes": {
        "receptor_type": RECEPTORS.expected["receptor_type"].to_numpy(dtype=object)
    },
}


@pytest.fixture(scope="module", params=list(SOURCES))
def source(request, tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp(request.param.replace(" ", "_"))
    store = write_truth_store(folder / "store.zarr", TRUTH)
    store.cells.insert("is_d2", TRUTH.samples == "d2")
    store.cells.insert("is_d4", TRUTH.samples == "d4")
    insert_nullable_cell_column(
        store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
    )
    return SimpleNamespace(
        store=store,
        clones=SOURCES[request.param](store, folder),
        kept_apart=KEPT_APART.get(request.param, {}),
    )


def _groups(source, labels: np.ndarray | None = None) -> np.ndarray:
    """Each of the truth's cells' group, named as the summaries name it.

    A group is named by the cell's labels joined by commas: the labels given,
    then those that the source's clones keep apart; without any, every cell
    is in ``"all cells"``.
    """
    columns = [] if labels is None else [np.asarray(labels, dtype=object)]
    columns += [
        np.asarray(values, dtype=object) for values in source.kept_apart.values()
    ]
    if not columns:
        return np.full(N_CELLS, "all cells", dtype=object)
    return np.asarray(
        [", ".join(str(label) for label in parts) for parts in zip(*columns)],
        dtype=object,
    )


def _by_group(
    source, labels: np.ndarray | None = None, cells: np.ndarray | None = None
) -> dict[object, list[int]]:
    return TRUTH.group_clone_sizes(_groups(source, labels), cells=cells)


def _group_cells(
    source, labels: np.ndarray, cells: np.ndarray | None = None
) -> dict[object, dict[int, int]]:
    """Each group's cells of each of the truth's clones, counted cell by cell.

    The cells of a group share the labels that the source's clones keep
    apart, so within a group each of the truth's clones is one of the
    source's clones.
    """
    return TRUTH.group_clone_cells(_groups(source, labels), cells=cells)


def _kept_labels(source, labels: np.ndarray) -> dict[object, tuple[str, ...]]:
    """The labels that the source's clones keep apart, of each group by name."""
    kept = [np.asarray(values, dtype=object) for values in source.kept_apart.values()]
    found: dict[object, tuple[str, ...]] = {}
    for cell, group in enumerate(_groups(source, labels)):
        stratum = tuple(str(values[cell]) for values in kept)
        assert found.setdefault(group, stratum) == stratum
    return found


def _paired(groups, kept: dict[object, tuple[str, ...]]) -> set[frozenset]:
    """The pairs of groups that share the labels the clones keep apart."""
    return {
        frozenset((first, second))
        for first in groups
        for second in groups
        if str(first) < str(second) and kept[first] == kept[second]
    }


def _keys(source) -> list[tuple | None]:
    """Each of the truth's cells' clone as the source saves it.

    A clone of the truth within the labels that the source keeps apart, as
    its clone and those labels, or None for a cell without a clone.
    """
    kept = [np.asarray(values, dtype=object) for values in source.kept_apart.values()]
    return [
        None if clone < 0 else (int(clone), *(str(values[cell]) for values in kept))
        for cell, clone in enumerate(TRUTH.clones.tolist())
    ]


def _source_clones(source) -> tuple[dict[tuple, str], dict[tuple, int]]:
    """The source's label and rank of each clone of ``_keys``.

    They come from the source's cells table, which must give each such clone
    one label of its own and no label to a cell without a clone; the rank is
    the clone's place in the source's table of clones, largest first.
    """
    loaded = source.store.clones.load_clones(source.clones)
    ranks = {str(label): rank for rank, label in enumerate(loaded.clones["clone"])}
    of_cell = loaded.cells["clone"].reindex(TRUTH.cell_ids)
    labels: dict[tuple, str] = {}
    for key, label in zip(_keys(source), of_cell.tolist(), strict=True):
        if key is None:
            assert pd.isna(label)
            continue
        assert labels.setdefault(key, str(label)) == str(label)
    assert len(set(labels.values())) == len(labels)
    return labels, {key: ranks[label] for key, label in labels.items()}


def test_clone_abundance_counts_the_clones_of_each_size(source) -> None:
    result = source.store.plots.clone_abundance(
        source.clones, group_by="sample", show=False
    )
    sizes = result.tables["sizes"]
    expected = _by_group(source, TRUTH.samples)
    assert set(sizes["group"]) == set(expected)
    for group, found in expected.items():
        rows = sizes[sizes["group"] == group]
        assert dict(zip(rows["clone_size"], rows["n_clones"], strict=True)) == {
            size: found.count(size) for size in set(found)
        }
    plt.close(result.figure)


def test_expansion_counts_each_clone_within_each_sample(source) -> None:
    store = source.store
    loaded = store.clones.load_expansion(
        store.clones.expansion(source.clones, group_by="sample")
    )
    cells = loaded.cells.loc[TRUTH.cell_ids]
    np.testing.assert_array_equal(
        cells["group_clone_size"],
        TRUTH.cell_clone_sizes(_groups(source, TRUTH.samples)),
    )
    expected = _by_group(source, TRUTH.samples)
    totals = loaded.categories.groupby("group")["cells"].sum().to_dict()
    assert totals == {group: sum(found) for group, found in expected.items()}
    clones = loaded.categories.groupby("group")["clones"].sum().to_dict()
    assert clones == {group: len(found) for group, found in expected.items()}


def test_diversity_measures_the_clone_sizes_of_each_sample(source) -> None:
    result = source.store.clones.diversity(source.clones, group_by="sample")
    indices = result.indices.set_index("group")
    expected = _by_group(source, TRUTH.samples)
    assert sorted(indices.index) == sorted(expected)
    for group, sizes in expected.items():
        found = np.asarray(sizes)
        shares = found / found.sum()
        row = indices.loc[group]
        assert row[["cells", "clones", "singletons", "doubletons"]].tolist() == [
            found.sum(),
            len(found),
            np.count_nonzero(found == 1),
            np.count_nonzero(found == 2),
        ]
        assert row["shannon"] == pytest.approx(-np.sum(shares * np.log(shares)))
        assert row["inverse_simpson"] == pytest.approx(1 / np.sum(shares**2))


def test_rarefaction_reaches_each_samples_observed_diversity(source) -> None:
    result = source.store.clones.rarefaction(
        source.clones, group_by="sample", orders=(0, 2)
    )
    observed = result.curves[result.curves["method"] == "observed"]
    expected = _by_group(source, TRUTH.samples)
    assert set(observed["group"]) == set(expected)
    for group, sizes in expected.items():
        found = np.asarray(sizes)
        shares = found / found.sum()
        rows = observed[observed["group"] == group].set_index("order")
        assert rows.loc[0, "size"] == found.sum()
        assert rows.loc[0, "diversity"] == len(found)
        assert rows.loc[2, "diversity"] == pytest.approx(1 / np.sum(shares**2))


def test_overlap_compares_the_clones_that_samples_share(source) -> None:
    result = source.store.clones.overlap(source.clones, group_by="sample")
    by_group = _group_cells(source, TRUTH.samples)
    kept = _kept_labels(source, TRUTH.samples)
    columns = ["group", "sample", *source.kept_apart, "cells", "clones"]
    assert sorted(map(tuple, result.group_sizes[columns].values.tolist())) == sorted(
        (group, group.split(", ")[0], *kept[group], sum(found.values()), len(found))
        for group, found in by_group.items()
    )
    # Only groups of the same labels kept apart are compared.
    pairs = result.pairs.to_dict("records")
    assert {frozenset((row["group_1"], row["group_2"])) for row in pairs} == (
        _paired(by_group, kept)
    )
    for row in pairs:
        first, second = by_group[row["group_1"]], by_group[row["group_2"]]
        for column, value in overlap_reference(first, second).items():
            assert row[column] == pytest.approx(value, rel=1e-12)
    if not source.kept_apart:
        (row,) = pairs
        assert (row["group_1"], row["group_2"]) == SAMPLES


def test_track_follows_each_samples_largest_clones(source) -> None:
    tracked = source.store.clones.track(source.clones, group_by="sample", top=2)
    labels, ranks = _source_clones(source)
    if not source.kept_apart:
        assert labels == {(clone,): number_label(clone) for (clone,) in labels}
    by_group = _group_cells(source, TRUTH.samples)
    kept = _kept_labels(source, TRUTH.samples)
    # Each group's two largest clones, ties to the larger clone overall.
    chosen: set[tuple] = set()
    for group, found in by_group.items():
        keys = [(clone, *kept[group]) for clone in found]
        ranked = sorted(keys, key=lambda key: (-found[key[0]], ranks[key]))
        chosen |= set(ranked[:2])
    order = sorted(chosen, key=ranks.__getitem__)
    assert tracked.tracked == tuple(labels[key] for key in order)
    # Each clone is followed through the groups of its own labels kept apart,
    # zeros included.
    columns = ["clone", "sample", *source.kept_apart, "cells"]
    assert sorted(map(tuple, tracked.cells[columns].values.tolist())) == sorted(
        (labels[key], group.split(", ")[0], *kept[group], found.get(key[0], 0))
        for key in order
        for group, found in by_group.items()
        if kept[group] == key[1:]
    )
    assert tuple(tracked.cells["clone"].drop_duplicates()) == tracked.tracked


def test_coupling_couples_the_cell_types_that_share_clones(source) -> None:
    result = source.store.clones.coupling(
        source.clones, group_by="cell_type", method="abundance_cosine"
    )
    by_group = _group_cells(source, TRUTH.cell_types)
    sizes = result.group_sizes
    assert dict(zip(sizes["group"], sizes["cells"], strict=True)) == {
        group: sum(found.values()) for group, found in by_group.items()
    }
    # Only groups of the same labels kept apart are coupled.
    assert {
        frozenset(pair) for pair in result.pairs[["group_1", "group_2"]].values
    } == _paired(by_group, _kept_labels(source, TRUTH.cell_types))
    for row in result.pairs.itertuples(index=False):
        assert row.coupling == pytest.approx(
            coupling_reference(by_group, row.group_1, row.group_2, "abundance_cosine"),
            rel=1e-12,
        )
    assert np.isfinite(result.pairs["z_score"]).all()
    assert sorted(result.order) == sorted(by_group)


def test_fate_bias_follows_early_cells_to_their_clones_late_cells(source) -> None:
    store = source.store
    early, late = TRUTH.samples == "d2", TRUTH.samples == "d4"
    loaded = store.clones.load_fate_bias(
        store.clones.fate_bias(
            source.clones,
            group_by="cell_type",
            early=store.snapshot_cell_selection("is_d2"),
            late=store.snapshot_cell_selection("is_d4"),
        )
    )
    labels, _ranks = _source_clones(source)
    keys = _keys(source)
    fates = _groups(source, TRUTH.cell_types)
    # Each fate holds its late cells, those without a clone included unless
    # the clones keep labels apart, which a cell without a clone does not take.
    with_fate = [
        cell
        for cell in np.flatnonzero(late)
        if keys[cell] is not None or not source.kept_apart
    ]
    assert dict(
        zip(loaded.fates["fate"], loaded.fates["late_cells"], strict=True)
    ) == dict(Counter(fates[cell] for cell in with_fate))
    clone_fates: dict[tuple, Counter] = {}
    for cell in np.flatnonzero(late):
        if keys[cell] is not None:
            clone_fates.setdefault(keys[cell], Counter())[fates[cell]] += 1
    names = loaded.fates["fate"].tolist()
    assert loaded.cells.index.tolist() == TRUTH.cell_ids[early].tolist()
    for (cell_id, row), cell in zip(
        loaded.cells.iterrows(), np.flatnonzero(early), strict=True
    ):
        found = clone_fates.get(keys[cell], Counter()) if keys[cell] else Counter()
        total = sum(found.values())
        assert row["late_cells"] == total, cell_id
        if total:
            assert row[names].tolist() == pytest.approx(
                [found[name] / total for name in names]
            )
    assert set(loaded.clone_fates["clone"].astype(str)) == {
        labels[key] for key in clone_fates
    }


def test_summaries_report_each_label_their_clones_keep_apart(source) -> None:
    store = source.store
    columns = ["sample", *source.kept_apart]
    clonal = TRUTH.clones >= 0
    labels = [TRUTH.samples, *source.kept_apart.values()]
    # Each group's labels in the key columns, from the truth's cells with a
    # clone.
    expected = sorted(
        {
            tuple(str(values[cell]) for values in labels)
            for cell in np.flatnonzero(clonal)
        }
    )
    diversity = store.clones.diversity(source.clones, group_by="sample").indices
    assert sorted(map(tuple, diversity[columns].values.tolist())) == expected
    curves = store.clones.rarefaction(source.clones, group_by="sample").curves
    assert sorted(set(map(tuple, curves[columns].values.tolist()))) == expected
    expansion = store.clones.load_expansion(
        store.clones.expansion(source.clones, group_by="sample")
    )
    assert sorted(set(map(tuple, expansion.categories[columns].values.tolist()))) == (
        expected
    )
    if source.kept_apart:
        # T cells and B cells of one sample are measured apart.
        types = {key[:2] for key in expected}
        assert {("d2", "TCR"), ("d2", "BCR")} <= types


def test_summaries_keep_to_a_cell_selection(source) -> None:
    store = source.store
    d2 = TRUTH.samples == "d2"
    selection = store.snapshot_cell_selection("is_d2")
    expected = _by_group(source, cells=d2)
    diversity = store.clones.diversity(source.clones, cell_selection=selection)
    assert {
        group: [cells, clones]
        for group, cells, clones in diversity.indices[
            ["group", "cells", "clones"]
        ].values.tolist()
    } == {group: [sum(found), len(found)] for group, found in expected.items()}
    expansion = store.clones.load_expansion(
        store.clones.expansion(source.clones, cell_selection=selection)
    )
    assert expansion.cells.index.tolist() == TRUTH.cell_ids[d2].tolist()
    np.testing.assert_array_equal(
        expansion.cells["group_clone_size"],
        TRUTH.cell_clone_sizes(_groups(source), cells=d2)[d2],
    )
    by_type = _group_cells(source, TRUTH.cell_types, cells=d2)
    cells_of = {group: sum(found.values()) for group, found in by_type.items()}
    overlap = store.clones.overlap(
        source.clones, group_by="cell_type", cell_selection=selection
    )
    sizes = overlap.group_sizes
    assert dict(zip(sizes["group"], sizes["cells"], strict=True)) == cells_of
    # The source's largest clone, followed through the cells of d2.
    labels, ranks = _source_clones(source)
    largest = min(ranks, key=ranks.__getitem__)
    tracked = store.clones.track(
        source.clones, labels=[labels[largest]], cell_selection=selection
    )
    in_d2 = sum(key == largest for key, in_d2 in zip(_keys(source), d2) if in_d2)
    assert tracked.cells[["clone", "cells"]].values.tolist() == [
        [labels[largest], in_d2]
    ]
    coupling = store.clones.coupling(
        source.clones, group_by="cell_type", cell_selection=selection, permutations=0
    )
    sizes = coupling.group_sizes
    assert dict(zip(sizes["group"], sizes["cells"], strict=True)) == cells_of


def test_summaries_refuse_a_grouping_that_leaves_out_every_clone(source) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")

    def fate_bias(clones, **options):
        return store.clones.fate_bias(clones, early=early, late=late, **options)

    for summary in (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
        store.clones.overlap,
        store.clones.track,
        store.clones.coupling,
        fate_bias,
    ):
        with pytest.raises(ValueError, match="No cell has both a clone and a group"):
            summary(source.clones, group_by="unlabelled")


def test_summaries_name_missing_and_incomplete_clones(source) -> None:
    store = source.store
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")

    def fate_bias(clones):
        return store.clones.fate_bias(
            clones, group_by="cell_type", early=early, late=late
        )

    summaries = (
        store.clones.expansion,
        store.clones.diversity,
        store.clones.rarefaction,
        lambda clones: store.clones.overlap(clones, group_by="sample"),
        store.clones.track,
        lambda clones: store.clones.coupling(
            clones, group_by="cell_type", permutations=0
        ),
        fate_bias,
    )
    # Each saved result, with the loader that reads it.
    saved = {
        store.clones.expansion(source.clones): store.clones.load_expansion,
        fate_bias(source.clones): store.clones.load_fate_bias,
    }
    # Artifacts that the store does not hold raise KeyError.
    for summary in summaries:
        with pytest.raises(KeyError, match="Artifact does not exist"):
            summary(_unsaved(source.clones))
    for result, load in saved.items():
        with pytest.raises(KeyError, match="Artifact does not exist"):
            load(_unsaved(result))
    # An import that stopped before it finished leaves incomplete clones,
    # which raise RuntimeError, also where a saved result reads them.
    loads_of_saved = tuple(
        lambda _ref, result=result, load=load: load(result)
        for result, load in saved.items()
    )
    # Clonotypes also read their receptors' types, through every summary.
    status = inspect_artifact(store.zw, source.clones)
    receptors = (
        [status.input_ref("receptors")] if "receptors" in (status.inputs or {}) else []
    )
    summaries_of_clones = tuple(
        lambda _ref, summary=summary: summary(source.clones) for summary in summaries
    )
    for ref, readers in (
        (source.clones, (*summaries, *loads_of_saved)),
        *((result, (load,)) for result, load in saved.items()),
        *((receptor_ref, summaries_of_clones) for receptor_ref in receptors),
    ):
        attributes = artifact_group(store.zw, ref).attrs
        attributes["complete"] = False
        try:
            for read in readers:
                with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                    read(ref)
        finally:
            attributes["complete"] = True
    for result, load in saved.items():
        assert len(load(result).cells) > 0


def _unsaved(ref: ArtifactRef) -> ArtifactRef:
    """A reference of the same kind to an artifact that no store holds."""
    return ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)
