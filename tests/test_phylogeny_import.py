"""Cell phylogenies imported from every format, and the clones they are cut into.

The synthetic truth's phylogeny splits, below a root and its one child, into
one clade per clone, so a cut at time 1 or at depth 1 must give back the
truth's clones; other cuts are derived by hand from how the clades are built
in ``tests/clone_truth.py``. Real trees from PEtracer and ConvexML are cut as
pycea 0.4.0 cuts them, in ``tests/pycea_0_4_0_golden.json``.
"""

import gzip
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from cytearc.clones import CellPhylogeny
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    copy_clone_store,
    every_cell,
    loaded_partition,
    open_clone_store,
    write_truth_store,
)
from tests.clone_truth import (
    LEAF_TIME,
    N_CELLS,
    N_OUTSIDE,
    PHYLOGENY_CUT_DEPTH,
    PHYLOGENY_CUT_TIME,
    TruthTree,
    clone_truth,
    newick_text,
    truth_forest,
    truth_phylogeny,
    write_h5td,
    write_newick,
    write_tree_zarr,
)
from tests.storage_helpers import write_count_store

TRUTH = clone_truth()
TREE = truth_phylogeny(TRUTH)
DATA = Path(__file__).parent / "data" / "phylogenies"
GOLDEN = json.loads((Path(__file__).parent / "pycea_0_4_0_golden.json").read_text())
N_IN_TREES = int(np.count_nonzero(TRUTH.clones >= 0))
LABELS = [str(clone + 1) for clone in range(TRUTH.n_clones)]


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("phylogeny") / "store.zarr"
    write_truth_store(path, TRUTH)
    return path


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    """One writable copy for imports that save artifacts of their own."""
    folder = tmp_path_factory.mktemp("phylogeny_shared")
    store = copy_clone_store(template, folder)
    return SimpleNamespace(store=store, cells=every_cell(store), folder=folder)


@pytest.fixture
def store(template, tmp_path):
    return copy_clone_store(template, tmp_path)


def _import(store, paths, **options):
    """Import trees, collecting the warnings the import gives."""
    options.setdefault("cell_selection", every_cell(store))
    options.setdefault("time_unit", "days")
    with collect_warnings() as messages:
        ref = store.clones.import_phylogeny(paths, **options)
    return ref, list(messages)


def _cut(store, phylogeny, **cut):
    with collect_warnings() as messages:
        ref = store.clones.phylogeny_clones(phylogeny, **cut)
    return ref, list(messages)


def _members(clone: int) -> list[str]:
    """A clone's cells in the order its clade lists them, outside barcodes left out."""
    return [str(cell) for cell in TRUTH.cell_ids[TRUTH.clones == clone]]


def _write_newick(path: Path, compress: bool = False) -> Path:
    return write_newick([TREE], path, compress=compress)


_FORMATS = {
    "newick": lambda folder: _write_newick(folder / "truth_tree.nwk"),
    "newick gzip": lambda folder: _write_newick(folder / "truth_tree.nwk.gz", True),
    "h5td": lambda folder: write_h5td([TREE], folder / "truth_tree.h5td", lengths=True),
    "zarr": lambda folder: write_tree_zarr([TREE], folder / "truth_tree.zarr"),
    "treedata 0.4.0 h5td": lambda _folder: DATA / "truth_tree_treedata_0_4_0.h5td",
    "treedata 0.4.0 zarr": lambda _folder: DATA / "truth_tree_treedata_0_4_0.zarr",
}


@pytest.mark.parametrize("name", list(_FORMATS))
def test_each_format_recovers_the_truths_clones(shared, tmp_path, name) -> None:
    path = _FORMATS[name](tmp_path)
    phylogeny, messages = _import(shared.store, path, cell_selection=shared.cells)
    attributes = shared.store.artifacts.load(phylogeny).attrs
    # The three barcodes outside the store are leaves of clone 1's clade.
    assert messages == [
        f"{N_OUTSIDE} of {N_IN_TREES + N_OUTSIDE} barcodes of the tree file match "
        "no selected cell, for example "
        + ", ".join(repr(f"d2__{barcode}-1") for barcode in TRUTH.outside)
        + ". Check barcode_prefix and cell_selection."
    ]
    assert {
        key: attributes[key] for key in sorted(attributes) if key.startswith("n_")
    } == {
        "n_cells_in_no_tree": N_CELLS - N_IN_TREES,
        "n_cells_in_trees": N_IN_TREES,
        "n_file_barcodes": N_IN_TREES + N_OUTSIDE,
        "n_leaves": N_IN_TREES + N_OUTSIDE,
        "n_leaves_without_a_cell": N_OUTSIDE,
        "n_nodes": len(TREE.names),
        "n_trees": 1,
        "n_unmatched_barcodes": N_OUTSIDE,
        "n_unnamed_leaves": 0,
    }
    assert attributes["time_unit"] == "days"
    assert attributes["leaf_time_spread"] == 0.0
    clones, messages = _cut(shared.store, phylogeny, time=PHYLOGENY_CUT_TIME)
    assert messages == []
    assert loaded_partition(shared.store, clones) == TRUTH.partition()
    loaded = shared.store.clones.load_clones(clones)
    # Clades are named by their roots, which carry the clones' labels.
    assert loaded.clones["clone"].tolist() == LABELS
    assert loaded.source == "phylogeny_clones"


def test_depth_one_cuts_the_truth_once_the_root_merges_with_its_child(shared) -> None:
    path = _write_newick(shared.folder / "depth_tree.nwk")
    phylogeny, _messages = _import(shared.store, path, cell_selection=shared.cells)
    clones, messages = _cut(shared.store, phylogeny, depth=PHYLOGENY_CUT_DEPTH)
    assert messages == []
    assert loaded_partition(shared.store, clones) == TRUTH.partition()
    assert shared.store.clones.load_clones(clones).clones["clone"].tolist() == LABELS


@pytest.mark.parametrize(
    "paths",
    [
        lambda folder: [
            write_newick([tree], folder / f"{tree.name}.nwk")
            for tree in truth_forest(TRUTH)
        ],
        lambda _folder: [DATA / "truth_forest_treedata_0_4_0.h5td"],
        lambda folder: [write_tree_zarr(truth_forest(TRUTH), folder / "forest.zarr")],
    ],
    ids=["one newick file per clone", "treedata 0.4.0 h5td", "zarr"],
)
def test_one_tree_per_clone_cuts_into_clones_named_by_tree(
    shared, tmp_path, paths
) -> None:
    files = paths(tmp_path)
    phylogeny, _messages = _import(shared.store, files, cell_selection=shared.cells)
    loaded = shared.store.clones.load_phylogeny(phylogeny)
    stem = "" if len(files) > 1 else Path(files[0]).name.split(".")[0] + "_"
    assert loaded.trees["tree"].tolist() == [f"{stem}clone_{k}" for k in range(1, 9)]
    # Each tree's root starts at time 0, so cutting at 0 gives one clone per tree.
    clones, _messages = _cut(shared.store, phylogeny, time=0)
    assert loaded_partition(shared.store, clones) == TRUTH.partition()
    assert shared.store.clones.load_clones(clones).clones["clone"].tolist() == [
        f"{stem}clone_{k}/{k}" for k in range(1, 9)
    ]


@pytest.fixture(scope="module")
def truth_phylogeny_ref(shared):
    path = _write_newick(shared.folder / "cuts.nwk")
    phylogeny, _messages = _import(shared.store, path, cell_selection=shared.cells)
    return phylogeny


def _clone_sets(store, clones) -> dict[str, frozenset[str]]:
    cells = store.clones.load_clones(clones).cells
    labelled = cells["clone"].dropna()
    return {
        str(label): frozenset(str(cell) for cell in members)
        for label, members in labelled.groupby(labelled, observed=True).groups.items()
    }


@pytest.mark.parametrize(
    ("cut", "label"),
    [
        ({"time": 0}, "root"),
        ({"depth": 0}, "root"),
        ({"time": 0.25}, "founder"),
        ({"time": 0.5}, "founder"),
    ],
)
def test_cuts_before_the_clades_make_one_clone_of_the_tree(
    shared, truth_phylogeny_ref, cut, label
) -> None:
    clones, _messages = _cut(shared.store, truth_phylogeny_ref, **cut)
    assert _clone_sets(shared.store, clones) == {
        label: frozenset(str(cell) for cell in TRUTH.cell_ids[TRUTH.clones >= 0])
    }


def test_a_cut_just_past_a_clade_root_splits_that_clade_alone(
    shared, truth_phylogeny_ref
) -> None:
    # Clade 1 starts at time 1 and the others from 1.125 on, so at 1.0625
    # only clade 1 splits: into its first two leaves and the chain below.
    clones, _messages = _cut(shared.store, truth_phylogeny_ref, time=1.0625)
    members = _members(0)
    expected = {
        LABELS[clone]: frozenset(_members(clone)) for clone in range(1, TRUTH.n_clones)
    }
    expected[members[0]] = frozenset(members[:1])
    expected[members[1]] = frozenset(members[1:2])
    expected["c1n0"] = frozenset(members[2:])
    assert _clone_sets(shared.store, clones) == expected


def test_depth_two_splits_each_polytomy_and_leaves_out_a_single_child_clade(
    shared, truth_phylogeny_ref
) -> None:
    clones, messages = _cut(shared.store, truth_phylogeny_ref, depth=2)
    expected: dict[str, frozenset[str]] = {}
    for clone in range(TRUTH.n_clones):
        members = _members(clone)
        if len(members) == 1:
            # Clade 8 is a root of one child: its leaf stays at depth 1.
            continue
        for leaf in members[:2]:
            expected[leaf] = frozenset([leaf])
        rest = members[2:]
        if len(rest) == 1:
            expected[rest[0]] = frozenset(rest)
        elif rest:
            # The second clone's chain hangs below a node of one child.
            expected["c2single" if clone == 1 else f"c{clone + 1}n0"] = frozenset(rest)
    assert _clone_sets(shared.store, clones) == expected
    assert messages == [
        f"1 of the {N_IN_TREES} cells in the trees end before the cut at depth 2, "
        "so they are in no clone"
    ]
    attributes = shared.store.artifacts.load(clones).attrs
    assert (attributes["n_clones"], attributes["n_cells_before_cut"]) == (19, 1)


def test_cutting_at_the_leaves_makes_each_cell_a_clone_named_by_its_leaf(
    shared, truth_phylogeny_ref
) -> None:
    clones, _messages = _cut(shared.store, truth_phylogeny_ref, time=LEAF_TIME)
    cells = [str(cell) for cell in TRUTH.cell_ids[TRUTH.clones >= 0]]
    assert _clone_sets(shared.store, clones) == {
        cell: frozenset([cell]) for cell in cells
    }
    with pytest.raises(
        ValueError, match="No cell of the trees is in a clone at time 4.5 days"
    ):
        shared.store.clones.phylogeny_clones(truth_phylogeny_ref, time=4.5)


def _tree(name: str, nodes: list[tuple[str, int, float]]) -> TruthTree:
    names, parents, times = zip(*nodes, strict=True)
    return TruthTree(name, names, parents, times)


def _cell_ids(count: int) -> list[str]:
    return [str(cell) for cell in TRUTH.cell_ids[:count]]


def test_cells_that_end_before_the_cut_are_counted_and_warned(store, tmp_path) -> None:
    a, b, c, d = _cell_ids(4)
    # c and d end at 3, a and b at 4: the tree is 1 day from ultrametric.
    tree = _tree(
        "uneven",
        [("r", -1, 0.0), ("x", 0, 1.0), (a, 1, 4.0), (b, 1, 4.0), ("y", 0, 2.0)]
        + [(c, 4, 3.0), (d, 4, 3.0)],
    )
    phylogeny, _messages = _import(store, write_newick([tree], tmp_path / "u.nwk"))
    assert store.artifacts.load(phylogeny).attrs["leaf_time_spread"] == 1.0
    clones, messages = _cut(store, phylogeny, time=3.5)
    assert _clone_sets(store, clones) == {a: frozenset([a]), b: frozenset([b])}
    assert messages == [
        "2 of the 4 cells in the trees end before the cut at time 3.5 days, so they "
        "are in no clone"
    ]
    attributes = store.artifacts.load(clones).attrs
    assert attributes["n_cells_before_cut"] == 2
    assert attributes["leaf_time_spread"] == 1.0
    assert attributes["time_unit"] == "days"


def test_a_path_that_crosses_the_cut_twice_joins_the_first_clade(
    store, tmp_path
) -> None:
    a, b, c = _cell_ids(3)
    # x at 5, then y back at 4 below a negative length, then its leaves.
    path = tmp_path / "negative.nwk"
    path.write_text(f"((({a}:2,{b}:2)y:-1,{c}:1)x:5)r;")
    phylogeny, _messages = _import(store, path)
    clones, _messages = _cut(store, phylogeny, time=4.5)
    assert _clone_sets(store, clones) == {"x": frozenset([a, b, c])}


def test_trees_without_lengths_are_cut_at_depths_only(store, tmp_path) -> None:
    a, b, c = _cell_ids(3)
    path = tmp_path / "topology.nwk"
    path.write_text(f"(({a},{b})x,{c})r;")
    phylogeny, _messages = _import(store, path)
    assert store.artifacts.load(phylogeny).attrs["leaf_time_spread"] is None
    with pytest.raises(ValueError, match="5 nodes of the phylogeny have no time"):
        store.clones.phylogeny_clones(phylogeny, time=1.0)
    clones, _messages = _cut(store, phylogeny, depth=1)
    assert _clone_sets(store, clones) == {"x": frozenset([a, b]), c: frozenset([c])}
    nodes = store.clones.load_phylogeny(phylogeny).nodes
    assert nodes["time"].isna().all()


def test_the_time_unit_is_declared_never_inferred(store, tmp_path) -> None:
    path = _write_newick(tmp_path / "t.nwk")
    cells = every_cell(store)
    with pytest.raises(TypeError, match="time_unit"):
        store.clones.import_phylogeny(path, cell_selection=cells)
    with pytest.raises(ValueError, match="time_unit must name the unit"):
        store.clones.import_phylogeny(path, cell_selection=cells, time_unit="  ")
    with pytest.raises(TypeError, match="time_unit must be text"):
        store.clones.import_phylogeny(path, cell_selection=cells, time_unit=3)
    phylogeny, _messages = _import(store, path, time_unit=" hours ")
    assert store.clones.load_phylogeny(phylogeny).time_unit == "hours"
    untimed, _messages = _import(store, path, time_unit=None)
    assert untimed != phylogeny
    assert store.clones.load_phylogeny(untimed).time_unit is None
    with pytest.raises(ValueError, match="imported with time_unit=None"):
        store.clones.phylogeny_clones(untimed, time=1.0)
    clones, _messages = _cut(store, untimed, depth=PHYLOGENY_CUT_DEPTH)
    assert loaded_partition(store, clones) == TRUTH.partition()


def test_unnamed_or_shared_names_label_clones_by_their_root_position(
    store, tmp_path
) -> None:
    a, b, c, d = _cell_ids(4)
    path = tmp_path / "unnamed.nwk"
    # The internal nodes have no names, or support values that repeat.
    path.write_text(f"(({a}:1,{b}:1):1,({c}:1,{d}:1)95:1)95;")
    phylogeny, _messages = _import(store, path)
    clones, _messages = _cut(store, phylogeny, depth=1)
    nodes = store.clones.load_phylogeny(phylogeny).nodes
    # The clades' roots sit at positions 1 and 4 of the tree.
    assert nodes.loc[[1, 4], "node"].tolist() == [1, 4]
    assert _clone_sets(store, clones) == {
        "node1": frozenset([a, b]),
        "node4": frozenset([c, d]),
    }


def test_several_trees_of_one_file_are_named_by_their_position(store, tmp_path) -> None:
    a, b, c, d = _cell_ids(4)
    path = tmp_path / "pair.nwk.gz"
    path.write_bytes(gzip.compress(f"({a},{b})x;\n({c},{d})y;\n".encode()))
    phylogeny, _messages = _import(store, path, time_unit=None)
    assert store.clones.load_phylogeny(phylogeny).trees["tree"].tolist() == [
        "pair_1",
        "pair_2",
    ]
    clones, _messages = _cut(store, phylogeny, depth=0)
    assert _clone_sets(store, clones) == {
        "pair_1/x": frozenset([a, b]),
        "pair_2/y": frozenset([c, d]),
    }


def test_two_trees_of_one_name_are_refused(store, tmp_path) -> None:
    a, b = _cell_ids(2)
    first, second = tmp_path / "s1", tmp_path / "s2"
    first.mkdir()
    second.mkdir()
    (first / "tree.nwk").write_text(f"({a});")
    (second / "tree.nwk").write_text(f"({b});")
    with pytest.raises(
        ValueError,
        match="Two trees are named 'tree', of '.*/s[12]/tree.nwk' and '.*/s[12]/tree.nwk'",
    ):
        _import(store, [first / "tree.nwk", second / "tree.nwk"])


def test_a_tree_key_with_a_slash_is_refused(store, tmp_path) -> None:
    a, b = _cell_ids(2)
    tree = _tree("a/b", [("r", -1, 0.0), (a, 0, 1.0), (b, 0, 1.0)])
    other = _tree("c", [("s", -1, 0.0), (_cell_ids(3)[2], 0, 1.0)])
    path = write_h5td([tree, other], tmp_path / "slash.h5td")
    with pytest.raises(ValueError, match="has a '/' in its name"):
        _import(store, path)


def test_two_leaves_that_name_one_cell_are_refused(store, tmp_path) -> None:
    a, b = _cell_ids(2)
    # Two trees over the same cells, as bootstrap replicates are.
    path = tmp_path / "replicates.nwk"
    path.write_text(f"({a},{b});\n({b},{a});\n")
    with pytest.raises(ValueError, match="2 cells are named by several leaves"):
        _import(store, path)
    # A pattern can make two leaves name one cell too.
    patterned = tmp_path / "patterned.nwk"
    patterned.write_text(f"('x:{a}','y:{a}');")
    with pytest.raises(ValueError, match=f"'{a}' by 'x:{a}' and 'y:{a}'"):
        _import(store, patterned, barcode_pattern=r":(?P<barcode>.+)$")


def test_trees_that_name_no_selected_cell_are_refused(store, tmp_path) -> None:
    path = tmp_path / "strangers.nwk"
    path.write_text("(x,y,z);")
    with pytest.raises(
        ValueError,
        match="No leaf of the tree file names a selected cell, for example 'x', 'y', 'z'",
    ):
        _import(store, path)


def test_imports_are_reused_by_their_bytes_and_tree_names(store, tmp_path) -> None:
    path = _write_newick(tmp_path / "truth_tree.nwk")
    phylogeny, first = _import(store, path)
    again, reused = _import(store, path)
    assert again == phylogeny and reused == first
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(path, moved / "truth_tree.nwk")
    assert _import(store, moved / "truth_tree.nwk")[0] == phylogeny
    # The name of a file names its tree, so a renamed file is another import.
    renamed = tmp_path / "renamed.nwk"
    shutil.copy(path, renamed)
    assert _import(store, renamed)[0] != phylogeny
    assert _import(store, path, time_unit="hours")[0] != phylogeny
    assert _import(store, path, invalidate_cache=True)[0] != phylogeny
    record = store.artifacts.inspect(phylogeny)
    assert record.parameters == {"time_unit": "days"}
    (source,) = record.inputs["sources"]
    assert source == {
        "format": "newick",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "barcode_prefix": None,
        "barcode_suffix": None,
        "barcode_pattern": None,
        "tree_name": "truth_tree",
    }
    clones, _messages = _cut(store, phylogeny, time=1.0)
    assert _cut(store, phylogeny, time=1)[0] == clones
    assert _cut(store, phylogeny, depth=1)[0] != clones
    assert _cut(store, phylogeny, time=1.0, invalidate_cache=True)[0] != clones
    assert store.artifacts.inspect(clones).parameters == {"time": 1.0, "depth": None}


def test_read_only_stores_reuse_saved_work_and_refuse_new(store, tmp_path) -> None:
    path = _write_newick(tmp_path / "truth_tree.nwk")
    phylogeny, first = _import(store, path)
    clones, _messages = _cut(store, phylogeny, time=1.0)
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
    again, warned = _import(reader, path, cell_selection=every_cell(store))
    assert again == phylogeny and warned == first
    assert _cut(reader, phylogeny, time=1.0)[0] == clones
    with pytest.raises(
        PermissionError, match="requires a DataStore opened with zarr_mode='r\\+'"
    ):
        reader.clones.phylogeny_clones(phylogeny, time=2.0)
    other = tmp_path / "other.nwk"
    shutil.copy(path, other)
    with pytest.raises(PermissionError, match="zarr_mode='r\\+'"):
        reader.clones.import_phylogeny(
            other, cell_selection=every_cell(store), time_unit="days"
        )


def test_loaded_phylogenies_list_every_node_and_tree(
    shared, truth_phylogeny_ref
) -> None:
    loaded = shared.store.clones.load_phylogeny(truth_phylogeny_ref)
    assert isinstance(loaded, CellPhylogeny)
    assert loaded.time_unit == "days"
    nodes = loaded.nodes
    assert len(nodes) == len(TREE.names)
    assert nodes["name"].tolist() == list(TREE.names)
    assert nodes["parent"].tolist() == list(TREE.parents)
    assert nodes["time"].tolist() == list(TREE.times)
    first = nodes.iloc[:3]
    assert first[["name", "depth", "is_leaf"]].values.tolist() == [
        ["root", 0, False],
        ["founder", 0, False],
        ["1", 1, False],
    ]
    leaves = nodes[nodes["is_leaf"]]
    expected = [
        name if name in set(TRUTH.cell_ids.tolist()) else pd.NA
        for name in leaves["name"]
    ]
    assert leaves["cell"].tolist() == expected
    assert nodes.loc[~nodes["is_leaf"], "cell"].isna().all()
    assert loaded.trees.to_dict("records") == [
        {
            "tree": "cuts",
            "n_nodes": len(TREE.names),
            "n_leaves": N_IN_TREES + N_OUTSIDE,
            "n_cells": N_IN_TREES,
            "root_time": 0.0,
            "earliest_leaf_time": LEAF_TIME,
            "latest_leaf_time": LEAF_TIME,
        }
    ]


def test_writing_the_names_counts_what_the_join_holds(tmp_path) -> None:
    # Long node names make their writing the import's largest check, which
    # must count what the join holds beside the trees: its lookup and each
    # selected cell's ID as text, whether or not a leaf names the cell.
    import re
    import sys
    import warnings

    from cytearc.datastore._operations.cell_barcodes import LOOKUP_BYTES_PER_CELL
    from cytearc.datastore.datastore import DataStore

    n_cells, n_leaves = 25_000, 20_000
    ids = [f"cell{cell:06d}-1" for cell in range(n_cells)]
    write_count_store(
        str(tmp_path / "names.zarr"),
        {"RNA": np.ones((n_cells, 1), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = DataStore(
        str(tmp_path / "names.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    store.cells.insert("in_tree", np.arange(n_cells) < n_leaves)
    pairs = ",".join(
        f"({ids[cell]}:1,{ids[cell + 1]}:1){'n' * 190}{cell:09d}:1"
        for cell in range(0, n_leaves, 2)
    )
    path = tmp_path / "names.nwk"
    path.write_text(f"({pairs})root;")

    def names_estimate(selection) -> int:
        """The bytes that writing the names holds, from its refusal."""
        budget = 1
        for _attempt in range(20):
            store.memoryBytes = budget
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    store.clones.import_phylogeny(
                        path,
                        cell_selection=selection,
                        time_unit="days",
                        invalidate_cache=True,
                    )
            except MemoryError as error:
                found = re.search(r"about (\d+) bytes", str(error))
                assert found is not None, str(error)
                if str(error).startswith("Writing the names"):
                    return int(found.group(1))
                budget = int(found.group(1))
                continue
            raise AssertionError("writing the names was never refused")
        raise AssertionError("the budget did not settle")

    in_tree = names_estimate(store.snapshot_cell_selection("in_tree"))
    every_cell_of_the_store = names_estimate(store.snapshot_cell_selection())
    beyond = ids[n_leaves:]
    assert every_cell_of_the_store - in_tree == LOOKUP_BYTES_PER_CELL * len(
        beyond
    ) + sum(map(sys.getsizeof, beyond))


def test_missing_and_incomplete_phylogenies_are_refused(store, tmp_path) -> None:
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.refs import ArtifactRef

    phylogeny, _messages = _import(store, _write_newick(tmp_path / "t.nwk"))
    readers = (
        lambda ref: store.clones.load_phylogeny(ref),
        lambda ref: store.clones.phylogeny_clones(ref, time=1.0, invalidate_cache=True),
    )
    missing = ArtifactRef(
        scope=phylogeny.scope, kind=phylogeny.kind, artifact_id="f" * 64
    )
    for read in readers:
        with pytest.raises(KeyError, match="Artifact does not exist"):
            read(missing)
    # An import that stopped before it finished leaves incomplete trees.
    attributes = artifact_group(store.zw, phylogeny).attrs
    attributes["complete"] = False
    try:
        for read in readers:
            with pytest.raises(RuntimeError, match="Artifact is incomplete"):
                read(phylogeny)
    finally:
        attributes["complete"] = True
    assert len(store.clones.load_phylogeny(phylogeny).nodes) == len(TREE.names)


def test_wrong_inputs_are_refused(shared, truth_phylogeny_ref, tmp_path) -> None:
    store = shared.store
    clones, _messages = _cut(store, truth_phylogeny_ref, time=1.0)
    with pytest.raises(
        ValueError, match="phylogeny must be a cell_phylogeny artifact, not 'clones'"
    ):
        store.clones.load_phylogeny(clones)
    with pytest.raises(ValueError, match="phylogeny must be a cell_phylogeny artifact"):
        store.clones.phylogeny_clones(clones, time=1.0)
    with pytest.raises(TypeError, match="phylogeny must be an ArtifactRef"):
        store.clones.phylogeny_clones("tree", time=1.0)
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.phylogeny_clones(truth_phylogeny_ref, time=1.0, invalidate_cache=1)
    with pytest.raises(ValueError, match="Pass exactly one of time"):
        store.clones.phylogeny_clones(truth_phylogeny_ref)
    path = _write_newick(tmp_path / "t.nwk")
    for options, error, message in (
        (
            {"cell_selection": "cells"},
            TypeError,
            "cell_selection must be an ArtifactRef",
        ),
        ({"format": 3}, TypeError, "format must be a format name or None"),
        ({"format": "nexus"}, ValueError, "format must be one of newick, treedata"),
        ({"format": "treedata"}, ValueError, "which holds newick trees, not treedata"),
        ({"invalidate_cache": "yes"}, TypeError, "invalidate_cache must be a boolean"),
    ):
        with pytest.raises(error, match=message):
            _import(store, path, **options)


def _cells_store(folder: Path, ids: list[str]):
    path = folder / "store.zarr"
    counts = np.ones((len(ids), 3), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=ids)
    store = open_clone_store(path)
    store.cells.insert("every_cell", np.ones(len(ids), dtype=bool))
    return store


def _canonical(store, clones, leaves: list[str]) -> str:
    """The SHA-256 of a cut's clades as the golden script writes them."""
    groups = _clone_sets(store, clones)
    partition = sorted(sorted(members) for members in groups.values())
    inside = {cell for members in partition for cell in members}
    text = json.dumps(
        {"clades": partition, "outside": sorted(set(leaves) - inside)},
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _golden_cuts(store, phylogeny, case: str, leaves: list[str]) -> None:
    golden = GOLDEN["cases"][case]
    for files_digest in golden["files"].values():
        assert len(files_digest) == 64
    for kind, option in (("times", "time"), ("depths", "depth")):
        for cut, expected in golden[kind].items():
            value = float(cut) if option == "time" else int(cut)
            with collect_warnings():
                clones = store.clones.phylogeny_clones(phylogeny, **{option: value})
            assert _canonical(store, clones, leaves) == expected["sha256"], (
                case,
                option,
                cut,
            )
            sizes = store.clones.load_clones(clones).clones["n_cells"].value_counts()
            assert {str(size): int(n) for size, n in sizes.items()} == expected[
                "clades_by_size"
            ]
            n_before = store.artifacts.load(clones).attrs["n_cells_before_cut"]
            assert n_before == expected["n_outside"]


def test_petracer_trees_cut_as_pycea_cuts_them(tmp_path) -> None:
    ids = (
        gzip.decompress((DATA / "E7.5-R1_barcodes.tsv.gz").read_bytes())
        .decode()
        .split()
    )
    store = _cells_store(tmp_path, ids)
    files = [DATA / "E7.5-R1-C1.nwk.gz", DATA / "E7.5-R1-C2.nwk.gz"]
    for file in files:
        assert (
            hashlib.sha256(file.read_bytes()).hexdigest()
            == GOLDEN["cases"]["petracer"]["files"][file.name]
        )
    phylogeny, messages = _import(store, files)
    assert messages == []
    attributes = store.artifacts.load(phylogeny).attrs
    # 310 of the embryo's 1,342 cells are in neither clone's tree.
    assert (attributes["n_cells_in_trees"], attributes["n_cells_in_no_tree"]) == (
        1032,
        310,
    )
    assert attributes["n_trees"] == 2
    # The lengths' six digits leave the leaves 16 millionths of a day apart.
    assert attributes["leaf_time_spread"] == pytest.approx(1.6e-5, abs=1e-6)
    loaded = store.clones.load_phylogeny(phylogeny)
    leaves = loaded.nodes.loc[loaded.nodes["is_leaf"], "name"].tolist()
    _golden_cuts(store, phylogeny, "petracer", leaves)
    clones, _messages = _cut(store, phylogeny, time=0)
    # The trees' nodes have names of their own, which name their clones.
    assert sorted(store.clones.load_clones(clones).clones["clone"]) == [
        "E7.5-R1-C1/0D4vcs3c",
        "E7.5-R1-C2/"
        + loaded.nodes.loc[loaded.nodes["tree"] == "E7.5-R1-C2", "name"].iloc[0],
    ]


def test_convexml_tree_cuts_as_pycea_cuts_it(tmp_path) -> None:
    path = DATA / "convexml_tree_0.nwk"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == GOLDEN["cases"]["convexml"]["files"][path.name]
    )
    from cytearc.readers.phylogeny import read_trees, tree_file_layout

    trees = read_trees(tree_file_layout(path), memory_bytes=10**9)
    leaves = trees.names[trees.leaf].tolist()
    # Ten cells that no leaf names, and every leaf a cell.
    store = _cells_store(tmp_path, [*leaves, *(f"other{i}" for i in range(10))])
    phylogeny, _messages = _import(store, path)
    assert store.artifacts.load(phylogeny).attrs["n_cells_in_no_tree"] == 10
    _golden_cuts(store, phylogeny, "convexml", leaves)


def test_an_hdf5_file_holds_its_trees_in_obst(store, tmp_path) -> None:
    a, b = _cell_ids(2)
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.zeros(3))
    with pytest.raises(ValueError, match="has no 'obst' element"):
        _import(store, path)
    tree = _tree("t", [("r", -1, 0.0), (a, 0, 2.0), (b, 0, 2.0)])
    phylogeny, _messages = _import(store, write_h5td([tree], tmp_path / "t.h5td"))
    clones, _messages = _cut(store, phylogeny, time=1.0)
    assert _clone_sets(store, clones) == {a: frozenset([a]), b: frozenset([b])}


def test_the_text_of_a_tree_keeps_the_truths_shape() -> None:
    text = newick_text(TREE)
    assert text.startswith("((") and text.endswith(")root;")
    assert text.count("(") == text.count(")")
    # Every leaf ends at the same time, so the tree is ultrametric.
    assert set(
        TREE.times[node]
        for node, children in enumerate(TREE.children())
        if not children
    ) == {LEAF_TIME}


def test_cut_clones_read_as_cell_values_and_phylogenies_do_not(
    shared, truth_phylogeny_ref
) -> None:
    clones, _messages = _cut(shared.store, truth_phylogeny_ref, time=PHYLOGENY_CUT_TIME)
    loaded = shared.store.artifacts.load_values(clones)
    expected = np.asarray(
        [LABELS[clone] if clone >= 0 else "" for clone in TRUTH.clones.tolist()]
    )
    np.testing.assert_array_equal(loaded.values, expected)
    assert loaded.cell_ids.tolist() == TRUTH.cell_ids.tolist()
    sizes = shared.store.artifacts.load_values(clones, value="clone_size").values
    assert sizes.tolist() == TRUTH.cell_clone_sizes().tolist()
    with pytest.raises(ValueError, match="cell_phylogeny"):
        shared.store.artifacts.load_values(truth_phylogeny_ref)


def test_a_zarr_store_is_identified_by_every_file_and_path(tmp_path) -> None:
    from cytearc.datastore._operations.cell_barcodes import barcode_files, store_sha256
    from cytearc.datastore._operations.phylogeny import _tree_file_sha256

    first = write_tree_zarr([TREE], tmp_path / "first.zarr")
    second = tmp_path / "second.zarr"
    shutil.copytree(first, second)
    assert store_sha256(first) == store_sha256(second)
    assert len(store_sha256(first)) == 64
    # Renaming a file inside the store, or editing one, changes the identity.
    nodes = next(
        path
        for path in sorted(second.rglob("*"))
        if path.is_file() and path.parent.name == "nodes"
    )
    renamed = nodes.with_name(nodes.name + "_moved")
    nodes.rename(renamed)
    assert store_sha256(first) != store_sha256(second)
    renamed.rename(nodes)
    assert store_sha256(first) == store_sha256(second)
    (second / "zarr.json").write_text((second / "zarr.json").read_text() + " ")
    assert store_sha256(first) != store_sha256(second)
    (file,) = barcode_files(first, what="tree file", identify=_tree_file_sha256)
    assert file.sha256 == store_sha256(first)
    newick = _write_newick(tmp_path / "truth_tree.nwk")
    assert _tree_file_sha256(newick) == hashlib.sha256(newick.read_bytes()).hexdigest()


def test_a_file_without_a_tree_extension_names_its_tree_whole(store, tmp_path) -> None:
    a, b = _cell_ids(2)
    path = tmp_path / "lineage.v2"
    path.write_text(f"({a},{b})root;")
    phylogeny, _messages = _import(store, path, time_unit=None)
    assert store.clones.load_phylogeny(phylogeny).trees["tree"].tolist() == [
        "lineage.v2"
    ]


def test_the_join_counts_the_cells_ids_once(tmp_path) -> None:
    # The join counts the IDs that it holds as text, so an import runs within
    # the budget that its check before it reads the IDs names; counting them
    # again beside the join would refuse it.
    import re
    import warnings

    from cytearc.datastore.datastore import DataStore

    n_cells = 3_000
    ids = [f"{'sample_with_a_long_name_' * 3}{cell:08d}-1" for cell in range(n_cells)]
    write_count_store(
        str(tmp_path / "long.zarr"),
        {"RNA": np.ones((n_cells, 1), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = DataStore(
        str(tmp_path / "long.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    pairs = ",".join(
        f"({ids[cell]}:1,{ids[cell + 1]}:1)g{cell}:1" for cell in range(0, 2_000, 2)
    )
    path = tmp_path / "long.nwk"
    path.write_text(f"({pairs})root;")
    cells = store.snapshot_cell_selection()
    refusals: list[str] = []
    budget = 1
    for _attempt in range(20):
        store.memoryBytes = budget
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                store.clones.import_phylogeny(
                    path, cell_selection=cells, time_unit="days", invalidate_cache=True
                )
            break
        except MemoryError as error:
            refusals.append(str(error))
            found = re.search(r"about (\d+) bytes", str(error))
            assert found is not None and int(found.group(1)) > budget, str(error)
            budget = int(found.group(1))
    assert refusals[-1].startswith(f"Joining 3001 nodes of 1 trees to {n_cells} cells")


def test_a_budget_too_small_is_refused_before_anything_is_saved(
    store, tmp_path
) -> None:
    path = _write_newick(tmp_path / "truth_tree.nwk")
    store.memoryBytes = 10_000
    with pytest.raises(
        MemoryError,
        match=r"holds about \d+ bytes, more than the memory budget of 10000 bytes",
    ):
        _import(store, path)
    store.memoryBytes = 10**10
    assert store.artifacts.list(kind="cell_phylogeny", scope="datastore") == []
    phylogeny, _messages = _import(store, path)
    store.memoryBytes = 10_000
    for call in (
        lambda: store.clones.load_phylogeny(phylogeny),
        lambda: store.clones.phylogeny_clones(phylogeny, time=1.0),
    ):
        with pytest.raises(MemoryError, match="Raise mem_budget"):
            call()


def test_saved_phylogenies_that_lost_an_array_or_a_count_are_not_used(
    store, tmp_path
) -> None:
    path = _write_newick(tmp_path / "truth_tree.nwk")
    from cytearc.storage.artifacts import artifact_group

    phylogeny, _messages = _import(store, path)
    group = artifact_group(store.zw, phylogeny)
    group.attrs["n_cells_in_trees"] = "many"
    with pytest.raises(ValueError, match="'n_cells_in_trees' attribute is not a count"):
        store.clones.load_phylogeny(phylogeny)
    # Arrays whose lengths disagree are not reused; the import runs again.
    del group["tree_unique_names"]
    group.create_array("tree_unique_names", data=np.zeros(3, dtype=bool))
    again, _messages = _import(store, path)
    assert again != phylogeny
    assert store.clones.load_phylogeny(again).trees["tree"].tolist() == ["truth_tree"]
    # An artifact without one of its arrays is not reused either.
    del artifact_group(store.zw, again)["tree_names"]
    assert _import(store, path)[0] not in (phylogeny, again)


def test_a_reused_import_reads_neither_cells_nor_trees(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import phylogeny as operations
    from cytearc.readers import phylogeny as reader

    path = _write_newick(tmp_path / "truth_tree.nwk")
    phylogeny, first = _import(store, path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a reused import read what it saved")

    monkeypatch.setattr(type(store), "_clone_selection_indices", refuse)
    monkeypatch.setattr(reader, "read_trees", refuse)
    monkeypatch.setattr(
        operations._PhylogenyOperationsMixin, "_imported_forest", refuse
    )
    again, reused = _import(
        store, path, cell_selection=phylogeny_cells(store, phylogeny)
    )
    assert again == phylogeny and reused == first


def phylogeny_cells(store, phylogeny):
    return store.artifacts.inspect(phylogeny).input_ref("cell_selection")


def test_a_tree_whose_root_alone_is_unnamed_labels_clones_by_position(
    store, tmp_path
) -> None:
    a, b, c, d = _cell_ids(4)
    path = tmp_path / "rootless.nwk"
    path.write_text(f"(({a},{b})n1,({c},{d})n2);")
    phylogeny, _messages = _import(store, path, time_unit=None)
    clones, _messages = _cut(store, phylogeny, depth=1)
    assert _clone_sets(store, clones) == {
        "node1": frozenset([a, b]),
        "node4": frozenset([c, d]),
    }


def test_support_values_never_name_clones(store, tmp_path) -> None:
    a, b, c, d = _cell_ids(4)
    path = tmp_path / "support.nwk"
    path.write_text(f"(({a}:1,{b}:1)95:1,({c}:1,{d}:1)80:1)100;")
    phylogeny, _messages = _import(store, path)
    clones, _messages = _cut(store, phylogeny, depth=1)
    assert _clone_sets(store, clones) == {
        "node1": frozenset([a, b]),
        "node4": frozenset([c, d]),
    }


def test_unnamed_leaves_count_as_leaves_without_a_cell_and_not_as_barcodes(
    store, tmp_path
) -> None:
    a, b = _cell_ids(2)
    path = tmp_path / "gaps.nwk"
    path.write_text(f"({a},,{b},());")
    phylogeny, messages = _import(store, path, time_unit=None)
    attributes = store.artifacts.load(phylogeny).attrs
    assert messages == []
    assert {
        name: attributes[name]
        for name in (
            "n_file_barcodes",
            "n_unmatched_barcodes",
            "n_unnamed_leaves",
            "n_leaves_without_a_cell",
            "n_leaves",
        )
    } == {
        "n_file_barcodes": 2,
        "n_unmatched_barcodes": 0,
        "n_unnamed_leaves": 2,
        "n_leaves_without_a_cell": 2,
        "n_leaves": 4,
    }


def test_trees_of_unnamed_leaves_are_refused_without_examples(store, tmp_path) -> None:
    path = tmp_path / "blank.nwk"
    path.write_text("(,(,));")
    with pytest.raises(
        ValueError,
        match="names a selected cell, since no leaf has a name; name the leaves",
    ):
        _import(store, path, time_unit=None)


def test_a_slash_from_the_prefix_is_named_as_the_prefix(store, tmp_path) -> None:
    path = _write_newick(tmp_path / "truth_tree.nwk")
    with pytest.raises(
        ValueError, match="remove the '/' from barcode_prefix or from the tree's key"
    ):
        _import(store, path, barcode_prefix="x/")


def test_time_cuts_say_why_nodes_have_no_time(store, tmp_path) -> None:
    a, b = _cell_ids(2)
    path = tmp_path / "half.nwk"
    path.write_text(f"({a}:1,{b});")
    phylogeny, _messages = _import(store, path)
    with pytest.raises(
        ValueError,
        match="^1 node of the phylogeny has no time, since a branch above it has",
    ):
        store.clones.phylogeny_clones(phylogeny, time=0.5)


def test_directories_are_hashed_only_as_zarr_stores(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import cell_barcodes
    from cytearc.datastore._operations import phylogeny as phylogeny_operations

    folder = tmp_path / "outs"
    folder.mkdir()
    (folder / "big.bin").write_bytes(b"x" * 1000)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a directory was hashed before its format was known")

    monkeypatch.setattr(cell_barcodes, "store_sha256", refuse)
    monkeypatch.setattr(phylogeny_operations, "store_sha256", refuse)
    # Clone files and call files are files, as before.
    with pytest.raises(IsADirectoryError):
        store.clones.import_clones(folder, cell_selection=every_cell(store))
    with pytest.raises(ValueError, match="is a directory but not a Zarr store"):
        _import(store, folder)
