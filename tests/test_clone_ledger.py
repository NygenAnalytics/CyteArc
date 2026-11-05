"""Clone calling's steps refuse budgets too small for them, and the clone operations give their results.

Calling's steps refuse a budget below what they would build before they
build it, the clones of a metadata column read in blocks at a small budget
are the clones of one read, a column's stamp follows its labels and its
missing mask, and the clonotypes within types of clonotypes across receptor
types are named, cleared and followed as fate bias and tracking measure them.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.clones import calling, sequences
from cytearc.clones.assignments import ranked_clone_labels
from cytearc.clones.calling import cell_sets, identical_sets, set_clones
from cytearc.datastore._operations import clones as clone_operations
from cytearc.metadata.rows import metadata_column_stamp
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.utils.memory_budget import (
    MemoryBudgetError,
    MemoryLedger,
)
from tests.clone_helpers import every_cell, write_cells_store
from tests.storage_helpers import insert_nullable_cell_column


def test_identical_sets_refuse_a_budget_below_the_sets_hashes() -> None:
    # A megabyte cannot hold the hashes of 400,000 cells' sets.
    n_cells = 400_000
    rng = np.random.default_rng(3)
    cells = np.repeat(np.arange(n_cells), 3)
    sets = cell_sets(cells, rng.integers(0, 3_000, size=len(cells)), n_cells)
    with pytest.raises(MemoryBudgetError, match="the sets' hashes"):
        identical_sets(sets, ledger=MemoryLedger(1_000_000, "Calling clones"))


def test_keeping_the_edges_between_remaining_nodes_refuses_a_small_budget() -> None:
    # A megabyte cannot hold a copy of 400,000 edges.
    n_edges = 400_000
    edges = calling.Edges(np.arange(n_edges), np.arange(n_edges) + 1)
    present = np.ones(n_edges + 1, dtype=bool)
    with pytest.raises(MemoryBudgetError, match="the edges kept"):
        calling._kept_edges(MemoryLedger(1_000_000, "Calling clones"), edges, present)


def test_removing_doublets_from_a_chain_of_nodes_keeps_its_edges() -> None:
    # 200,000 cells in 100,000 nodes of two cells, each node's set sharing
    # one of three items with the next: 99,999 edges at a similarity of a
    # third, which removing doublets keeps.
    n_cells = 200_000
    cells = np.repeat(np.arange(n_cells), 2)
    first = np.arange(n_cells) // 2
    items = np.stack([first, first + 1], axis=1).ravel()
    sets = cell_sets(cells, items, n_cells)

    clones = set_clones(
        sets,
        method="jaccard",
        similarity=0.3,
        inclusive=True,
        doublets=True,
        ledger=MemoryLedger(None, "x"),
    )
    assert clones.n_edges == 99_999


def _bounded_intervals(
    intervals: list[tuple[str, int, int]], items: Callable[[str], bool]
) -> list[tuple[str, int, int]]:
    """The intervals of the charges that ``items`` names, each checked from both sides.

    Each traced interval stays within its plan and the call's objects, and
    each plan within half again what it traced and NumPy's fixed objects.
    """
    chosen = [entry for entry in intervals if items(entry[0])]
    for item, planned, traced in chosen:
        assert traced <= planned + AUDIT_ALLOWANCE, (item, planned, traced)
        assert planned <= 1.5 * traced + NUMPY_STEP_BYTES, (item, planned, traced)
    return chosen


def _prefix_rows(shape: str) -> tuple[list[list[int]], np.ndarray | None]:
    """Sets of each shape the graph takes, and the cells' groups."""
    rng = np.random.default_rng(len(shape))
    if shape == "one item most cells hold":
        # TREX after correction: 1 to 5 barcodes of their own, and in nine
        # cells of ten the barcode that most merged into.
        rows = [
            [0] * (rng.random() < 0.9) + list(range(5 * cell + 1, 5 * cell + 2 + size))
            for cell, size in enumerate(rng.integers(0, 5, size=60_000).tolist())
        ]
        return rows, None
    if shape == "one node meets many":
        # A set of 40,000 tags, each shared with one set of two that also
        # holds the barcode that most hold: at 0.7 its prefix keeps 12,000
        # tags, and filling the edges sorts the 12,000 nodes it meets.
        hub = list(range(1, 40_001))
        return [hub] + [[0, tag] for tag in hub], None
    # Clones of 10 cells each lose some of their clone's tags, so cells of a
    # clone link: CellTag's 4 to 12 tags of 3,000, or 40 of 400, whose
    # nodes meet many others and sort their ranks with a copy.
    n_tags, pool, n_cells = (
        (8, 3_000, 60_000) if shape == "celltag" else (40, 400, 3_000)
    )
    rows = []
    for clone in range(n_cells // 10):
        tags = rng.choice(
            pool,
            size=int(rng.integers(n_tags // 2, 3 * n_tags // 2 + 1)),
            replace=False,
        )
        for _cell in range(10):
            kept = tags[rng.random(len(tags)) < 0.85]
            rows.append(sorted(set(kept.tolist()) | {int(rng.integers(0, pool))}))
    groups = rng.integers(0, 4, size=n_cells) if shape == "celltag" else None
    return rows, groups


@pytest.mark.parametrize(
    ("shape", "similarity", "way"),
    [
        ("one item most cells hold", 0.7, "prefixes"),
        ("one node meets many", 0.7, "prefixes"),
        ("celltag", 0.7, "product"),
        pytest.param("celltag", 0.7, "prefixes", marks=pytest.mark.slow),
        pytest.param("large sets", 0.5, "prefixes", marks=pytest.mark.slow),
        pytest.param("large sets", 0.5, "product", marks=pytest.mark.slow),
    ],
)
def test_the_graph_holds_each_step_within_its_plan(
    monkeypatch, shape, similarity, way
) -> None:
    # Coding the items within their groups, choosing each node's prefix,
    # meeting the nodes through the prefixes or through every item, and
    # filling the edges are each charged before they allocate, and none
    # overstates what it holds by half.
    monkeypatch.setattr(
        calling, "_prefixes_are_faster", lambda *steps: way == "prefixes"
    )
    rows, groups = _prefix_rows(shape)
    cells = np.repeat(np.arange(len(rows)), [len(row) for row in rows])
    sets = cell_sets(
        cells, np.concatenate([np.asarray(row) for row in rows]), len(rows)
    )
    nodes = identical_sets(sets, groups)

    def call() -> object:
        return calling.jaccard_edges(
            nodes, similarity, inclusive=False, ledger=MemoryLedger(None, "x")
        )

    edges = call()
    _result, intervals = interval_peaks(call)
    meeting = (
        "the nodes that the prefixes reach"
        if way == "prefixes"
        else "the nodes that share an item"
    )
    steps = {
        "the lists of each item's nodes",
        "the prefixes of the nodes' sets",
        meeting,
    }
    chosen = _bounded_intervals(
        intervals,
        lambda item: item in steps or item.endswith(" edges of the graph"),
    )
    assert {item for item, _planned, _traced in chosen} == steps | {
        f"the {len(edges)} edges of the graph"
    }
    assert ("one" in shape) or len(edges) > 1_000


def _keyed_inputs(shape: str) -> tuple[np.ndarray, np.ndarray, int, bool, int, int]:
    """Texts of each shape the keyed search takes: codes, lengths, limit, edits, segments and split."""
    rng = np.random.default_rng(len(shape))
    if shape == "edits":
        amino = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
        lengths = rng.integers(10, 17, size=30_000)
        codes = np.zeros((30_000, 16), dtype=np.uint8)
        for row, length in enumerate(lengths.tolist()):
            codes[row, :length] = amino[rng.integers(0, 20, size=length)]
        # A fifth of them one substitution from another.
        codes[::5] = codes[1::5]
        codes[::5, 3] = amino[rng.integers(0, 20, size=len(codes[::5]))]
        lengths[::5] = lengths[1::5]
        return codes, lengths.astype(np.int64), 1, True, 0, 0
    if shape == "copies":
        # 300 copies of a text share each of its 5 keys: runs of 300 keys
        # outgrow the rows of a run, and their 224,250 pairs the buffer.
        codes = np.frombuffer(b"CASSF" * 300, dtype=np.uint8).reshape(300, 5).copy()
        return codes, np.full(300, 5, dtype=np.int64), 1, False, 5, 0
    if shape == "letters":
        # 300 copies of one letter share its one key: the buffer doubles to
        # 65,536 for their 44,850 pairs, each found once, so copying the
        # pairs out of the buffer holds more than its last growth did.
        codes = np.full((300, 1), ord("A"), dtype=np.uint8)
        return codes, np.ones(300, dtype=np.int64), 1, False, 1, 0
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    codes = bases[rng.integers(0, 4, size=(40_000, 28))]
    variants = codes[rng.integers(0, 40_000, size=10_000)]
    changed = rng.random(variants.shape) < 0.1
    variants[changed] = bases[rng.integers(0, 4, size=int(changed.sum()))]
    codes = np.ascontiguousarray(np.vstack([codes, variants]))
    lengths = np.full(len(codes), 28, dtype=np.int64)
    split = 30_000 if shape == "split" else 0
    return codes, lengths, 5, False, 7, split


@pytest.mark.parametrize("shape", ["barcodes", "split", "copies", "letters", "edits"])
def test_each_step_of_the_keyed_search_holds_what_it_charges(shape) -> None:
    # The packed codes of the texts, the keys of a pass, the buffer of the
    # pairs that it finds and the rows of a run, while they grow, and the
    # pairs copied out of the buffer, or under edits the candidate pairs,
    # are each charged before they are made and none overstates what it
    # holds by half; the search is called alone, so its ledger holds
    # nothing else.
    codes, lengths, limit, edits, segments, split = _keyed_inputs(shape)
    table = sequences._zobrist_table(codes.shape[1])

    def call() -> tuple[np.ndarray, np.ndarray]:
        return sequences._keyed_pairs(
            codes,
            lengths,
            limit,
            edits,
            table,
            MemoryLedger(None, "x"),
            segments,
            split,
        )

    first, _second = call()
    _result, intervals = interval_peaks(call)
    steps = (
        {
            "the variant keys of one pass",
            "the candidate pairs of texts that share variant keys",
        }
        if edits
        else {
            "the packed codes of the texts",
            "the variant keys of one pass",
            "the pairs within max_distance",
        }
        | ({"the packed codes of a run of keys"} if shape == "copies" else set())
    )
    chosen = _bounded_intervals(
        intervals, lambda item: item in steps or item == "the pairs found"
    )
    assert {item for item, _planned, _traced in chosen} == steps | {"the pairs found"}
    assert len(first)
    if shape in ("copies", "letters"):
        # Every pair of copies, which the buffer doubled to hold.
        assert len(first) == 300 * 299 // 2
        assert sum(item == "the pairs within max_distance" for item, *_ in chosen) > 5


# The labels of clones ranked by size.


@pytest.mark.parametrize("n_clones", [0, 1, 9, 10, 99, 100, 12_345])
def test_ranked_labels_name_each_clone_by_its_rank(n_clones) -> None:
    labels = ranked_clone_labels(n_clones)
    assert labels.tolist() == [f"clone_{rank}" for rank in range(1, n_clones + 1)]


# Clonotypes across receptor types, as fate bias and tracking measure them.


def _receptor_type_input(n_cells: int) -> Any:
    """Clonotypes of three cells each, of four receptor types in turn, and their groups' strata."""
    import pandas as pd

    from cytearc.storage.refs import ArtifactRef

    labels = np.asarray(["BCR", "TCR", "ambiguous", "multichain"], dtype=object)
    unsaved = ArtifactRef(scope="datastore", kind="clonotypes", artifact_id="f" * 64)
    codes = np.where(np.arange(n_cells) % 9 == 0, -1, np.arange(n_cells) // 3)
    source = clone_operations._CloneInput(
        ref=unsaved,
        cell_selection=unsaved,
        codes=codes,
        clone_ids=None,
        kept_apart=(("receptor_type", labels[np.arange(n_cells) % 4]),),
        nbytes=0,
    )
    keys = pd.DataFrame(
        {"receptor_type": labels.tolist()}, index=[f"g{i}" for i in range(4)]
    )
    return source, keys, np.arange(4, dtype=np.int64)


def _parts_of_clonotypes(n_cells: int, per_part: int, spanning: float) -> tuple:
    """Cells of clonotypes within types, a share of which pair up into clonotypes of two types."""
    rng = np.random.default_rng(per_part)
    within = rng.integers(0, n_cells // per_part, n_cells)
    within[::50] = -1
    present = np.unique(within[within >= 0])
    within = np.where(within >= 0, np.searchsorted(present, within), -1)
    n_parts = len(present)
    n_pairs = int(spanning * n_parts / 2)
    part_clonotype = np.concatenate(
        [np.repeat(np.arange(n_pairs), 2), np.arange(n_pairs, n_parts - n_pairs)]
    )
    part_type = np.concatenate(
        [np.tile([0, 1], n_pairs), np.ones(n_parts - 2 * n_pairs, dtype=np.int64)]
    )
    shuffled = rng.permutation(n_parts)
    part_clonotype, part_type = part_clonotype[shuffled], part_type[shuffled]
    at = np.maximum(within, 0)
    codes = np.where(within >= 0, part_clonotype[at], -1)
    types = np.where(within >= 0, part_type[at], 2)
    return codes, within, types


@pytest.mark.parametrize(
    ("n_cells", "per_part", "spanning"),
    [(40_000, 3, 0.1), (400_000, 1, 0.0), (400_000, 10, 0.5), (200_000, 1, 1.0)],
    ids=["a tenth spanning", "none spanning", "half spanning", "all spanning"],
)
def test_naming_clonotypes_within_types_adds_a_text_where_a_clonotype_holds_several(
    n_cells, per_part, spanning
) -> None:
    from cytearc.repertoire.clonotypes import within_type_parts

    codes, within, types = _parts_of_clonotypes(n_cells, per_part, spanning)
    clonotypes, added = within_type_parts(
        codes,
        within,
        types,
        ["BCR", "TCR", "no IR"],
        ledger=MemoryLedger(1 << 40, "Naming clonotypes within types"),
    )
    # Each clonotype within a type takes its cells' clonotype, and adds a
    # text where its clonotype holds another.
    assert (clonotypes[within[within >= 0]] == codes[codes >= 0]).all()
    shared = np.bincount(clonotypes)[clonotypes] > 1
    assert [bool(text) for text in added.tolist()] == shared.tolist()


@pytest.mark.parametrize("n_cells", [40_000, 400_000])
def test_clearing_the_labels_of_cells_without_a_clone_keeps_the_labels_given(
    n_cells,
) -> None:
    source, _keys, _strata = _receptor_type_input(n_cells)
    labels = source.kept_apart[0][1]
    cleared = clone_operations._cleared_labels(source.codes, source.kept_apart)
    # A cell without a clone loses its label, and the labels given stay.
    expected = np.where(source.codes < 0, None, labels)
    assert cleared[0][1].tolist() == expected.tolist()
    assert labels[0] == "BCR" and source.codes[0] == -1


@pytest.mark.parametrize("n_cells", [40_000, 400_000])
def test_a_named_clone_is_followed_through_the_stratum_of_its_first_cell(
    n_cells,
) -> None:
    source, keys, strata = _receptor_type_input(n_cells)
    # Clone 5 starts at cell 15, of BCR, clone 77 at cell 231, of TCR, and the
    # last clone at a cell of its type.
    chosen = np.asarray([5, 77, n_cells // 3 - 2], dtype=np.int64)
    first = 3 * chosen
    first[first % 9 == 0] += 1
    found = clone_operations._fallback_strata(source, keys, strata, chosen)
    assert found.tolist() == (first % 4).tolist()


# Clones of a cell metadata column.


@pytest.fixture(scope="module")
def column_store(tmp_path_factory) -> Any:
    """A store of 60,000 cells with columns of short, long and four-byte clone labels."""
    path = tmp_path_factory.mktemp("from_column") / "store.zarr"
    n_cells = 60_000
    store = write_cells_store(path, [f"c{i:07d}" for i in range(n_cells)])
    rng = np.random.default_rng(11)
    clone_of = rng.integers(0, 6_000, n_cells)
    store.cells.insert("short", np.asarray([f"k{c}" for c in clone_of]))
    store.cells.insert("long", np.asarray([f"{'ACGT' * 40}_{c}" for c in clone_of]))
    store.cells.insert(
        "four_byte", np.asarray(["\U0001f600" * 12 + str(c) for c in clone_of])
    )
    store.cells.insert("numbers", clone_of.astype(np.int64))
    store.cells.insert("few", np.arange(n_cells) < 48)
    return store


def test_the_stamp_of_a_column_follows_its_stored_chunks(column_store) -> None:
    store = column_store
    cells = every_cell(store)
    first = store.clones.from_column("short", cell_selection=cells)
    stamp = metadata_column_stamp(store.cells, "short")
    # The same column gives the same stamp and reuses the clones.
    assert metadata_column_stamp(store.cells, "short") == stamp
    assert store.clones.from_column("short", cell_selection=cells) == first
    # An edited label changes the stamp and gives new clones.
    labels = store.cells.fetch_all("short").copy()
    labels[7] = "edited"
    store.cells.insert("short", labels, overwrite=True)
    assert metadata_column_stamp(store.cells, "short") != stamp
    assert store.clones.from_column("short", cell_selection=cells) != first


def test_cells_whose_labels_are_missing_join_no_clone_until_the_mask_changes(
    tmp_path,
) -> None:
    # The stored labels of cells 0 and 4 are labels like the others, which
    # only the missing mask hides: those cells join no clone. Marking cell 2
    # missing edits the mask alone, which changes the stamp and gives new
    # clones without it.
    store = write_cells_store(tmp_path / "store.zarr", [f"c{i}" for i in range(8)])
    labels = np.asarray(["k1", "k1", "k2", "k2", "k3", "k3", "k1", "k2"])
    missing = np.isin(np.arange(8), [0, 4])
    insert_nullable_cell_column(store, "partial", labels, missing)
    cells = every_cell(store)
    first = store.clones.from_column("partial", cell_selection=cells)
    loaded = store.clones.load_clones(first).cells
    assert loaded["clone"].isna().tolist() == missing.tolist()
    assert loaded["clone_size"].tolist() == [0, 2, 3, 3, 0, 1, 2, 3]
    stamp = metadata_column_stamp(store.cells, "partial")
    store.zw["cellData"][MISSING_MASK_PREFIX + "partial"][:] = np.isin(
        np.arange(8), [0, 2, 4]
    )
    assert metadata_column_stamp(store.cells, "partial") != stamp
    second = store.clones.from_column("partial", cell_selection=cells)
    assert second != first
    assert store.clones.load_clones(second).cells["clone_size"].tolist() == [
        0,
        2,
        0,
        2,
        0,
        1,
        2,
        2,
    ]


@pytest.mark.parametrize(
    ("column", "bytes_per_cell"), [("short", 300), ("four_byte", 500), ("numbers", 200)]
)
def test_clones_of_a_column_read_in_blocks_are_the_clones_of_one_read(
    column_store, column, bytes_per_cell
) -> None:
    # A few hundred bytes a cell read the labels of 60,000 cells in more and
    # smaller blocks than 16 GiB does, and give the same clones.
    store = column_store
    cells = every_cell(store)
    small = bytes_per_cell * store.cells.N
    loaded = {}
    for budget in (small, 1 << 34):
        store.memoryBytes = budget
        try:
            ref = store.clones.from_column(
                column, cell_selection=cells, invalidate_cache=True
            )
        finally:
            store.memoryBytes = 1 << 34
        loaded[budget] = store.clones.load_clones(ref)
    pd.testing.assert_frame_equal(loaded[small].cells, loaded[1 << 34].cells)
    pd.testing.assert_frame_equal(loaded[small].clones, loaded[1 << 34].clones)


# The groups' labels of clonal expansion.


def test_the_expansions_group_labels_read_back_as_the_grouping_gave_them(
    column_store,
) -> None:
    store = column_store
    store.cells.insert(
        "sample", np.asarray([f"s{i % 3}" for i in range(store.cells.N)])
    )
    store.cells.insert(
        "donor", np.asarray([f"d\U0001f600{i % 2}" for i in range(store.cells.N)])
    )
    clones = store.clones.from_column("short", cell_selection=every_cell(store))
    saved = store.clones.expansion(clones, group_by=["sample", "donor"])
    group = store.artifacts.load(saved)
    # The labels are an array of each group's label in each key column, and
    # no attribute holds them.
    assert "group_keys" not in group.attrs and group.attrs["n_groups"] == 6
    assert group["group_labels"].shape == (12,)
    keys = store.clones.load_expansion(saved).group_keys
    assert keys.columns.tolist() == ["sample", "donor"]
    assert keys.values.tolist() == [
        [sample, donor]
        for sample in ("s0", "s1", "s2")
        for donor in ("d\U0001f6000", "d\U0001f6001")
    ]
    # Without key columns, the one group has no labels to store.
    alone = store.clones.expansion(clones)
    assert "group_labels" not in store.artifacts.load(alone)
    assert store.clones.load_expansion(alone).group_keys.shape == (1, 0)
