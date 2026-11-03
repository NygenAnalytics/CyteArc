"""Reading Newick and TreeData trees, node by node, against hand-derived trees.

Every expected parent, name and length below is written out from the text of
the tree, in preorder, so the reader's own order never decides what is
right. Real files come from TreeData written by treedata 0.4.0 and from
PEtracer and ConvexML.
"""

import gzip
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from cytearc.clones.phylogeny import node_times
from cytearc.readers.phylogeny import (
    FORMATS,
    FileTrees,
    read_trees,
    tree_file_layout,
)

DATA = Path(__file__).parent / "data" / "phylogenies"
NAN = float("nan")


def _write(tmp_path: Path, text: str | bytes, name: str = "tree.nwk") -> Path:
    path = tmp_path / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_bytes(text.encode("utf-8"))
    return path


def _read(path: Path, **options: object) -> FileTrees:
    return read_trees(tree_file_layout(path, **options), memory_bytes=10**9)


def _parse(tmp_path: Path, text: str | bytes) -> FileTrees:
    return _read(_write(tmp_path, text))


def _lengths(trees: FileTrees) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in trees.length]


def test_a_tree_reads_in_preorder_with_its_labels_and_lengths(tmp_path) -> None:
    trees = _parse(tmp_path, "((A:1,B:2)C:3,D:4)E;")
    # E, then C and its leaves A and B, then D.
    assert trees.names.tolist() == ["E", "C", "A", "B", "D"]
    assert trees.parent.tolist() == [-1, 0, 1, 1, 0]
    assert trees.leaf.tolist() == [False, False, True, True, True]
    assert _lengths(trees) == [None, 3.0, 1.0, 2.0, 4.0]
    assert trees.tree_offsets.tolist() == [0, 5]
    assert trees.tree_keys == ("1",)
    assert np.isnan(trees.time).all()
    assert trees.parent.dtype == np.int64 and trees.length.dtype == np.float64


def test_polytomies_single_child_nodes_and_a_stem_keep_their_shape(tmp_path) -> None:
    trees = _parse(tmp_path, "(((A:1)X:1)Y:1,B:3,C:3,D:3)R:0.5;")
    assert trees.names.tolist() == ["R", "Y", "X", "A", "B", "C", "D"]
    assert trees.parent.tolist() == [-1, 0, 1, 2, 0, 0, 0]
    # The root's own length is the stem above it.
    assert _lengths(trees) == [0.5, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    assert times.tolist() == [0.5, 1.5, 2.5, 3.5, 3.5, 3.5, 3.5]
    wide = _parse(tmp_path, "(" + ",".join(f"L{i}:1" for i in range(67)) + ");")
    assert wide.parent.tolist() == [-1] + [0] * 67
    assert wide.names[1:].tolist() == [f"L{i}" for i in range(67)]


def test_zero_negative_and_scientific_lengths_read_as_written(tmp_path) -> None:
    trees = _parse(tmp_path, "((A:0,B:-0.5):0.0,C:1e-3,D:2.5E+1,E:-0,F:+7):1_000;")
    assert _lengths(trees) == [1000.0, 0.0, 0.0, -0.5, 0.001, 25.0, -0.0, 7.0]
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    # B's negative length puts it before its parent.
    assert times[3] == 999.5 and times[1] == 1000.0


def test_quoted_labels_hold_marks_spaces_and_doubled_quotes(tmp_path) -> None:
    trees = _parse(
        tmp_path,
        "('a b':1,'it''s':2,'x,y(z):[w];':3,'':4,'''':5)'root node':0;",
    )
    assert trees.names.tolist() == ["root node", "a b", "it's", "x,y(z):[w];", "", "'"]
    assert _lengths(trees) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # A quoted length reads like an unquoted one.
    assert _lengths(_parse(tmp_path, "(A:'1.5');")) == [None, 1.5]


def test_comments_are_skipped_wherever_they_fall_and_may_nest(tmp_path) -> None:
    trees = _parse(
        tmp_path,
        "[&R] ((A[&rate=1]:1,B:[&x]2[&&NHX:S=x])C[a [nested [deeper]] note]:1,"
        "'D[not a comment]':1)[&label=E]E;",
    )
    assert trees.names.tolist() == ["E", "C", "A", "B", "D[not a comment]"]
    assert _lengths(trees) == [None, 1.0, 1.0, 2.0, 1.0]


def test_whitespace_between_marks_is_dropped_and_inside_labels_kept(tmp_path) -> None:
    trees = _parse(tmp_path, " (\r\n A b : 1 ,\n\t B : 2 ) \r\n ;\r\n")
    assert trees.names.tolist() == ["", "A b", "B"]
    assert _lengths(trees) == [None, 1.0, 2.0]
    # A byte order mark before the text is not a label.
    assert _parse(tmp_path, b"\xef\xbb\xbf(A,B);").names.tolist() == ["", "A", "B"]


def test_underscores_and_unicode_labels_are_kept_as_written(tmp_path) -> None:
    trees = _parse(
        tmp_path, "(exp1_PA-02A.LIG-P3-B11_RT-P3-G10:1,κύτταρο_1:2,'λ x':3);"
    )
    assert trees.names.tolist() == [
        "",
        "exp1_PA-02A.LIG-P3-B11_RT-P3-G10",
        "κύτταρο_1",
        "λ x",
    ]


def test_several_trees_read_one_after_another(tmp_path) -> None:
    trees = _parse(tmp_path, "(A:1,B:1);\n\n;(C:2,(D:1,E:1)F:1)G;\n[end]\n")
    assert trees.names.tolist() == ["", "A", "B", "G", "C", "F", "D", "E"]
    # Each tree's parents count within the file's nodes.
    assert trees.parent.tolist() == [-1, 0, 0, -1, 3, 3, 5, 5]
    assert trees.tree_offsets.tolist() == [0, 3, 8]
    assert trees.tree_keys == ("1", "2")
    # The ';' after the last tree may be missing, as treedata allows.
    assert _parse(tmp_path, "(A:1,B:2)").names.tolist() == ["", "A", "B"]
    assert _parse(tmp_path, "A;").parent.tolist() == [-1]


def test_empty_groups_make_unnamed_leaves(tmp_path) -> None:
    trees = _parse(tmp_path, "(,(,):1,());")
    assert trees.parent.tolist() == [-1, 0, 0, 2, 2, 0, 5]
    assert trees.leaf.tolist() == [False, True, False, True, True, False, True]
    assert trees.names.tolist() == [""] * 7


def test_gzip_text_reads_like_plain_text(tmp_path) -> None:
    text = "((A:1,B:2)C:3,D:4)E;\n"
    plain = _read(_write(tmp_path, text, "plain.nwk"))
    path = tmp_path / "packed.nwk.gz"
    path.write_bytes(gzip.compress(text.encode(), mtime=0))
    layout = tree_file_layout(path)
    assert layout.compression == "gzip"
    assert layout.n_text_bytes == len(text)
    packed = _read(path)
    assert packed.names.tolist() == plain.names.tolist()
    assert _lengths(packed) == _lengths(plain)


def test_a_tree_a_million_nodes_deep_reads_without_recursion(tmp_path) -> None:
    depth = 1_000_000
    trees = _parse(tmp_path, "(" * depth + "A:1" + "):1" * depth + ";")
    assert trees.n_nodes == depth + 1
    np.testing.assert_array_equal(trees.parent, np.arange(-1, depth, dtype=np.int64))
    assert trees.names[-1] == "A" and trees.leaf.sum() == 1
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    assert times[-1] == depth + 1


def test_a_polytomy_of_a_million_leaves_reads_in_one_group(tmp_path) -> None:
    width = 1_000_000
    trees = _parse(tmp_path, "(" + ",".join(f"c{i}:0.5" for i in range(width)) + ")r;")
    assert trees.n_nodes == width + 1
    assert (trees.parent[1:] == 0).all()
    assert trees.names[1] == "c0" and trees.names[-1] == f"c{width - 1}"
    assert (trees.length[1:] == 0.5).all()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("((A,B)", "parentheses never close at its end"),
        ("((A,B);", "an unexpected ';' inside parentheses or after ':' at byte 6"),
        ("(A,B));", r"a '\)' without its '\(' at byte 5"),
        ("A,B;", "an unexpected ',' outside parentheses at byte 1"),
        ("(A,B)C(D);", r"an unexpected '\(' after a label.* at byte 6"),
        ("(A B C:1:2);", "a second ':' for one node at byte 8"),
        ("(A:1 B);", "branch length that is not a number at byte 3"),
        ("(A:,B);", "no branch length after ':' at byte 3"),
        ("(A:);", "no branch length after ':' at byte 3"),
        ("('A'B,C);", "a second label for one node.* at byte 4"),
        ("(A:1C,D);", "branch length that is not a number"),
        ("(A:inf,B);", "branch length that is not finite"),
        ("(A:-Infinity,B);", "branch length that is not finite"),
        ("(A:1e999,B);", "branch length that is not finite"),
        ("('A,B);", "a quoted label that never closes at byte 1"),
        ("(A[note,B);", r"a comment '\[' that never closes at byte 2"),
        ("(A,(B;C));", "an unexpected ';' inside parentheses or after ':' at byte 5"),
        ("(A:;", "an unexpected ';' inside parentheses or after ':'"),
    ],
)
def test_malformed_newick_is_refused_with_its_place(tmp_path, text, message) -> None:
    with pytest.raises(ValueError, match=message):
        _parse(tmp_path, text)


def test_errors_quote_the_text_around_the_mistake(tmp_path) -> None:
    text = "(" + ",".join(f"leaf{i}:1" for i in range(1000)) + "));"
    with pytest.raises(
        ValueError,
        match=r"\)' without its '\(' at byte 9891 of its text: '7:1,leaf998:1,leaf999:1\)\);'",
    ):
        _parse(tmp_path, text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "holds no tree"),
        ("  \n\t ", "holds no tree"),
        ("[only a comment]\n;;", "holds no tree"),
    ],
)
def test_text_without_a_tree_is_refused(tmp_path, text, message) -> None:
    with pytest.raises(ValueError, match=message):
        _parse(tmp_path, text)


def test_labels_that_are_not_utf8_are_refused(tmp_path) -> None:
    with pytest.raises(
        ValueError, match="label that is not UTF-8 text at byte 1 of its text"
    ):
        _parse(tmp_path, b"(\xff\xfe:1,B:1);")


def test_nexus_files_are_refused_before_they_are_read(tmp_path) -> None:
    path = _write(tmp_path, "#NEXUS\nbegin trees;\ntree one = (A,B);\nend;\n")
    with pytest.raises(ValueError, match="is a NEXUS file; export its trees"):
        tree_file_layout(path)


def test_the_layout_bounds_the_nodes_and_counts_the_text(tmp_path) -> None:
    text = "('a,b':1,[c,d](C,D):1);(E,F);"
    layout = tree_file_layout(_write(tmp_path, text))
    trees = read_trees(layout, memory_bytes=10**9)
    assert layout.format == "newick" and layout.container is None
    assert layout.n_text_bytes == len(text.encode())
    # Marks in quotes and comments count too, so the bound holds every node.
    assert layout.max_nodes >= trees.n_nodes == 8
    assert layout.max_trees == text.count(";") + 1
    assert layout.record() == {"format": "newick"}


def test_a_file_that_grows_after_its_layout_is_refused(tmp_path) -> None:
    path = _write(tmp_path, "(A,B);")
    layout = tree_file_layout(path)
    path.write_text("(A,B,C);")
    with pytest.raises(ValueError, match="changed while it was read"):
        read_trees(layout, memory_bytes=10**9)


def test_reading_refuses_a_budget_too_small_before_it_reads(tmp_path) -> None:
    layout = tree_file_layout(_write(tmp_path, "((A:1,B:2)C:3,D:4)E;"))
    with pytest.raises(
        MemoryError,
        match=r"holds about \d+ bytes, more than the memory budget of 100 bytes",
    ):
        read_trees(layout, memory_bytes=100)
    # The names are counted once the scan has found them.
    long_names = "(" + ",".join("x" * 4000 + str(i) for i in range(50)) + ");"
    layout = tree_file_layout(_write(tmp_path, long_names, "long.nwk"))
    with pytest.raises(MemoryError, match="names and branch lengths of 51 nodes"):
        read_trees(layout, memory_bytes=layout.read_bytes + 100)


@pytest.mark.parametrize("format", [None, *FORMATS])
def test_format_names_what_the_file_must_hold(tmp_path, format) -> None:
    path = _write(tmp_path, "(A,B);")
    if format == "treedata":
        with pytest.raises(
            ValueError, match="is text, which holds newick trees, not treedata"
        ):
            tree_file_layout(path, format=format)
    else:
        assert tree_file_layout(path, format=format).format == "newick"


def test_unknown_formats_and_missing_files_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="format must be one of newick, treedata"):
        tree_file_layout(_write(tmp_path, "(A,B);"), format="nexus")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        tree_file_layout(tmp_path / "absent.nwk")
    folder = tmp_path / "folder"
    folder.mkdir()
    with pytest.raises(ValueError, match="is a directory but not a Zarr store"):
        tree_file_layout(folder)


# TreeData


def _h5td(path: Path, obst: object, *, raw: bool = False) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "obst",
            data=obst if raw else json.dumps(obst),
            dtype=h5py.string_dtype(),
        )
    return path


def _graph(edges: list[tuple[str, str]], **times: float) -> dict:
    nodes = {name: {} for edge in edges for name in edge}
    for name, time in times.items():
        nodes[name]["time"] = time
    adjacency: dict[str, dict] = {name: {} for name in nodes}
    for source, target in edges:
        adjacency[source][target] = {}
    return {"nodes": nodes, "edges": adjacency}


def test_treedata_written_by_treedata_reads_with_node_times(tmp_path) -> None:
    for name in ("truth_tree_treedata_0_4_0.h5td", "truth_tree_treedata_0_4_0.zarr"):
        layout = tree_file_layout(DATA / name)
        assert (layout.format, layout.container) == (
            "treedata",
            "zarr" if name.endswith("zarr") else "hdf5",
        )
        trees = read_trees(layout, memory_bytes=10**9)
        assert trees.tree_keys == ("tree",)
        assert trees.names[:3].tolist() == ["root", "founder", "1"]
        assert trees.parent[:3].tolist() == [-1, 0, 1]
        times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
        # Node times win over the length of 1 given on every other edge.
        assert times[:3].tolist() == [0.0, 0.5, 1.0]
        assert set(times[trees.leaf].tolist()) == {4.0}
        assert trees.n_nodes == 61 and int(trees.leaf.sum()) == 35


def test_treedata_forests_read_tree_by_tree_in_natural_key_order(tmp_path) -> None:
    trees = _read(DATA / "truth_forest_treedata_0_4_0.h5td")
    assert trees.tree_keys == tuple(f"clone_{k}" for k in range(1, 9))
    assert len(trees.tree_offsets) == 9
    roots = trees.tree_offsets[:-1]
    assert trees.names[roots].tolist() == [str(k) for k in range(1, 9)]
    assert (trees.parent[roots] == -1).all()
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    # clone_2's root lost its time and starts at 0, beside its children's times.
    assert times[roots].tolist() == [0.0] * 8
    graphs = {
        "clone_10": _graph([("r10", "a")]),
        "clone_2": _graph([("r2", "b")]),
        "clone_1": _graph([("r1", "c")]),
    }
    path = _h5td(tmp_path / "ordered.h5td", graphs)
    assert _read(path).tree_keys == ("clone_1", "clone_2", "clone_10")


def test_treedata_times_fall_back_to_lengths_and_stay_unknown_without_them(
    tmp_path,
) -> None:
    graph = {
        "nodes": {"r": {}, "a": {"time": 5.0}, "b": {}, "c": {}, "d": {"time": None}},
        "edges": {
            "r": {"a": {"length": 1.0}, "b": {"length": 2.0}},
            "b": {"c": {}, "d": {"length": 0.5}},
        },
    }
    trees = _read(_h5td(tmp_path / "t.h5td", {"t": graph}))
    assert trees.names.tolist() == ["r", "a", "b", "c", "d"]
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    # a keeps its own time; b adds its length to r's 0; c has neither; d's null
    # time is unknown and its length counts.
    assert times[0] == 0.0 and times[1] == 5.0 and times[2] == 2.0
    assert np.isnan(times[3]) and times[4] == 2.5


def test_treedata_drops_attributes_other_than_time_and_length(tmp_path) -> None:
    graph = _graph([("r", "a"), ("r", "b")], r=0.0, a=1.0, b=2.0)
    graph["nodes"]["a"]["characters"] = list(range(10_000))
    graph["nodes"]["b"]["meta"] = {"time": 99.0, "nested": [1, 2]}
    trees = _read(_h5td(tmp_path / "t.h5td", {"t": graph}))
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    assert times.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("obst", "message"),
    [
        ("not json", "has an 'obst' that is not JSON"),
        ("[1, 2]", "does not map tree keys to trees"),
        ('{"t": {"nodes": {}}}', "has no 'nodes' and 'edges'"),
        ('{"t": {"nodes": {"a": 1}, "edges": {}}}', "has no 'nodes' and 'edges'"),
        ('{"t": 5}', "does not map tree keys to trees"),
        ('{"t": {"nodes": {"a": {}}, "edges": {"a": [1]}}}', "has no 'nodes' and"),
        (
            '{"t": {"nodes": {"a": {}}, "edges": {"a": {"b": 5}}}}',
            "edges from 'a' that do not map targets to objects",
        ),
        ("{}", "holds no tree in 'obst'"),
        (
            '{"t": {"nodes": {"time": 5}, "edges": {}}}',
            "has a node 'time' whose attributes are not an object: 5",
        ),
        ('{"t": {"nodes": {}, "edges": {}}}', "holds no tree in 'obst'"),
    ],
)
def test_malformed_treedata_json_is_refused(tmp_path, obst, message) -> None:
    with pytest.raises(ValueError, match=message):
        _read(_h5td(tmp_path / "t.h5td", obst, raw=True))


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([("r", "a"), ("s", "a")], "gives the node 'a' two parents, 'r' and 's'"),
        ([("r", "a"), ("s", "b")], "has 2 roots, not one, such as \\['r', 's'\\]"),
        ([("a", "b"), ("b", "a")], "has 0 roots, not one; its edges form a cycle"),
        (
            [("r", "a"), ("b", "c"), ("c", "b")],
            "has 2 nodes that its root does not reach",
        ),
    ],
)
def test_treedata_graphs_that_are_not_trees_are_refused(
    tmp_path, edges, message
) -> None:
    graph = _graph(edges)
    with pytest.raises(ValueError, match=message):
        _read(_h5td(tmp_path / "t.h5td", {"t": graph}))


@pytest.mark.parametrize(
    ("value", "message"),
    [("later", "a time that is not a number: 'later'"), (True, "not a number: True")],
)
def test_treedata_times_that_are_not_numbers_are_refused(
    tmp_path, value, message
) -> None:
    graph = _graph([("r", "a")])
    graph["nodes"]["a"]["time"] = value
    with pytest.raises(ValueError, match=message):
        _read(_h5td(tmp_path / "t.h5td", {"t": graph}))


def test_nan_lengths_and_times_are_unknown_as_treedata_writes_them(tmp_path) -> None:
    newick = _parse(tmp_path, "(A:nan,B:NaN,C:1);")
    assert np.isnan(newick.length[1:3]).all() and newick.length[3] == 1.0
    # treedata writes NaN into its JSON where a branch has no length.
    path = tmp_path / "nan.h5td"
    text = (
        '{"t": {"nodes": {"r": {"time": 0.0}, "a": {"time": NaN}, "b": {}}, '
        '"edges": {"r": {"a": {"length": NaN}, "b": {"length": 2.0}}, "a": {}, "b": {}}}}'
    )
    trees = _read(_h5td(path, text, raw=True))
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    assert times[0] == 0.0 and np.isnan(times[1]) and times[2] == 2.0
    zarr_path = _columnar(
        tmp_path / "nan.zarr",
        {
            "t": {
                "nodes": ["r", "a"],
                "edges": [["r", "a"]],
                "node_attrs": {"time": np.asarray([0.0, np.nan])},
                "edge_attrs": {"length": np.asarray([3.0])},
            }
        },
    )
    zarr_trees = _read(zarr_path)
    zarr_times = node_times(
        zarr_trees.parent, zarr_trees.length, zarr_trees.time, zarr_trees.tree_offsets
    )
    # a's NaN time is unknown, so its edge's length gives it one.
    assert zarr_times.tolist() == [0.0, 3.0]


def test_treedata_infinite_lengths_are_refused(tmp_path) -> None:
    graph = _graph([("r", "a")])
    graph["edges"]["r"]["a"] = {"length": float("inf")}
    with pytest.raises(ValueError, match="a length that is not finite"):
        _read(_h5td(tmp_path / "t.h5td", {"t": graph}))


def test_hdf5_files_without_trees_are_refused(tmp_path) -> None:
    path = tmp_path / "cells.h5ad"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="has no 'obst' element"):
        tree_file_layout(path)
    with h5py.File(path, "a") as handle:
        handle.create_dataset("obst", data=np.zeros(3))
    with pytest.raises(ValueError, match="does not store 'obst' as one JSON string"):
        tree_file_layout(path)
    with pytest.raises(
        ValueError, match="is an HDF5 file, which holds treedata trees, not newick"
    ):
        tree_file_layout(path, format="newick")


def test_hdf5_text_is_bounded_by_the_bytes_outside_fixed_size_data(tmp_path) -> None:
    graph = _graph([("r", "a")], r=0.0, a=1.0)
    path = _h5td(tmp_path / "t.h5td", {"t": graph})
    with h5py.File(path, "a") as handle:
        handle.create_dataset("X", data=np.zeros((500, 500)))
    layout = tree_file_layout(path)
    text = json.dumps({"t": graph})
    assert len(text) <= layout.n_text_bytes < path.stat().st_size - 500 * 500 * 8 + 4096


def _columnar(path: Path, trees: dict[str, dict]) -> Path:
    """A TreeData Zarr store with one group of columns per tree, as treedata 0.3+ writes."""
    root = zarr.open_group(path, mode="w", zarr_format=3)
    obst = root.create_group("obst")
    strings = np.dtypes.StringDType()
    for key, tree in trees.items():
        group = obst.create_group(key)
        group.create_array("nodes", data=np.asarray(tree["nodes"], dtype=strings))
        edges = np.asarray(tree["edges"], dtype=strings)
        group.create_array(
            "edges", data=edges.reshape(-1, 2) if edges.size == 0 else edges
        )
        for kind in ("node_attrs", "edge_attrs"):
            if kind in tree:
                attributes = group.create_group(kind)
                for name, values in tree[kind].items():
                    attributes.create_array(name, data=np.asarray(values))
    return path


def test_zarr_columns_read_as_numbers_or_json_text(tmp_path) -> None:
    path = _columnar(
        tmp_path / "t.zarr",
        {
            "t": {
                "nodes": ["r", "a", "b"],
                "edges": [["r", "a"], ["r", "b"]],
                "node_attrs": {
                    "time": np.asarray(
                        ["0", "", "null"], dtype=np.dtypes.StringDType()
                    ),
                    "characters": np.asarray(
                        ["[1]", "[2]", "[3]"], dtype=np.dtypes.StringDType()
                    ),
                },
                "edge_attrs": {"length": np.asarray([1.5, 2.0])},
            }
        },
    )
    layout = tree_file_layout(path)
    assert (layout.container, layout.max_nodes) == ("zarr", 3)
    trees = read_trees(layout, memory_bytes=10**9)
    assert trees.names.tolist() == ["r", "a", "b"]
    times = node_times(trees.parent, trees.length, trees.time, trees.tree_offsets)
    assert times.tolist() == [0.0, 1.5, 2.0]


@pytest.mark.parametrize(
    ("tree", "message"),
    [
        (
            {"nodes": ["r", "a"], "edges": [["r", "a", "x"]]},
            "no 'edges' array of source and target pairs",
        ),
        (
            {
                "nodes": ["r", "a"],
                "edges": [["r", "a"]],
                "node_attrs": {"time": np.asarray([1.0])},
            },
            "a 'time' attribute that does not hold one value per element",
        ),
        (
            {
                "nodes": ["r", "a"],
                "edges": [["r", "a"]],
                "edge_attrs": {
                    "length": np.asarray(["1.5x"], dtype=np.dtypes.StringDType())
                },
            },
            "a 'length' that is not JSON that Python can read: '1.5x'",
        ),
        (
            {
                "nodes": ["r", "a"],
                "edges": [["r", "a"]],
                "edge_attrs": {
                    "length": np.asarray([" [1, 2]"], dtype=np.dtypes.StringDType())
                },
            },
            r"a 'length' that is not a number: ' \[1, 2\]'",
        ),
    ],
)
def test_malformed_zarr_trees_are_refused(tmp_path, tree, message) -> None:
    path = _columnar(tmp_path / "t.zarr", {"t": tree})
    with pytest.raises(ValueError, match=message):
        _read(path)


def test_zarr_stores_without_trees_or_nodes_are_refused(tmp_path) -> None:
    zarr.open_group(tmp_path / "empty.zarr", mode="w", zarr_format=3).create_group("X")
    with pytest.raises(ValueError, match="has no 'obst' element"):
        tree_file_layout(tmp_path / "empty.zarr")
    root = zarr.open_group(tmp_path / "bare.zarr", mode="w", zarr_format=3)
    root.create_group("obst").create_group("t")
    with pytest.raises(ValueError, match="has no one-dimensional 'nodes' array"):
        tree_file_layout(tmp_path / "bare.zarr")


def test_zarr_text_is_counted_as_it_is_read(tmp_path) -> None:
    names = ["n" * 5_000 + str(i) for i in range(20)]
    path = _columnar(
        tmp_path / "t.zarr",
        {"t": {"nodes": names, "edges": [[names[0], name] for name in names[1:]]}},
    )
    layout = tree_file_layout(path)
    with pytest.raises(MemoryError, match=re.escape("Reading the tree 't' of")):
        read_trees(layout, memory_bytes=layout.read_bytes + 10_000)
    assert _read(path).n_nodes == 20


def test_real_newick_trees_read_with_every_node(tmp_path) -> None:
    petracer = _read(DATA / "E7.5-R1-C1.nwk.gz")
    # PEtracer's root at time 0 has one child, and every node has a name.
    assert (petracer.n_nodes, int(petracer.leaf.sum())) == (1484, 850)
    assert petracer.names[0] == "0D4vcs3c" and petracer.parent[1] == 0
    assert np.count_nonzero(petracer.parent == 0) == 1
    assert len(set(petracer.names.tolist())) == petracer.n_nodes
    convexml = _read(DATA / "convexml_tree_0.nwk")
    assert (convexml.n_nodes, int(convexml.leaf.sum())) == (800, 400)
    times = node_times(
        convexml.parent, convexml.length, convexml.time, convexml.tree_offsets
    )
    # ConvexML's tree is ultrametric: every leaf ends at 1.
    np.testing.assert_allclose(times[convexml.leaf], 1.0, rtol=0, atol=1e-12)


def test_text_that_cannot_be_read_or_changes_size_is_refused(tmp_path) -> None:
    packed = gzip.compress(b"((A:1,B:2)C:3,D:4)E;" * 50, mtime=0)
    cut = tmp_path / "cut.nwk.gz"
    cut.write_bytes(packed[: len(packed) // 2])
    with pytest.raises(ValueError, match="cannot be read"):
        tree_file_layout(cut)
    path = _write(tmp_path, "(A,B,C,D);")
    layout = tree_file_layout(path)
    path.write_text("(A,B);")
    with pytest.raises(ValueError, match="changed while it was read"):
        read_trees(layout, memory_bytes=10**9)


def test_layouts_bound_names_of_other_text_by_four_bytes_a_byte(tmp_path) -> None:
    plain = tree_file_layout(_write(tmp_path, "(AB,CD);", "plain.nwk"))
    greek = tree_file_layout(_write(tmp_path, "(αβ,γδ);", "greek.nwk"))
    assert plain.ascii_text and not greek.ascii_text
    assert plain.n_text_bytes == 8 and greek.n_text_bytes == 12
    # The bound of names grows from 49 bytes a node and one a byte to 80 and four.
    nodes = plain.max_nodes
    assert greek.max_nodes == nodes == 4
    assert greek.trees_bytes - plain.trees_bytes == 31 * nodes + 4 * 12 - 8
    assert greek.peak_bytes - greek.trees_bytes == greek.read_bytes


def test_treedata_json_that_is_not_utf8_is_refused(tmp_path) -> None:
    path = tmp_path / "bytes.h5td"
    with h5py.File(path, "w") as handle:
        # An ASCII string dataset hands back its bytes as they are stored.
        handle.create_dataset(
            "obst", data=b'{"t": "\xff\xfe"}', dtype=h5py.string_dtype("ascii")
        )
    with pytest.raises(ValueError, match="'obst' that is not UTF-8"):
        _read(path)


def test_a_zarr_tree_of_one_node_has_no_edges(tmp_path) -> None:
    path = _columnar(tmp_path / "one.zarr", {"t": {"nodes": ["only"], "edges": []}})
    trees = _read(path)
    assert trees.names.tolist() == ["only"] and trees.parent.tolist() == [-1]


def test_zarr_stores_that_cannot_be_opened_are_refused(tmp_path) -> None:
    folder = tmp_path / "broken.zarr"
    folder.mkdir()
    (folder / "zarr.json").write_text("{not json")
    with pytest.raises(ValueError, match="cannot be opened"):
        tree_file_layout(folder)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        # No '(' counted, then thousands: the stack of open groups would overrun.
        ("A" * 10_000 + ";", "(" * 5_000 + "B" * 5_000 + ";"),
        # Two nodes counted, then thousands of ',': the nodes would overrun.
        ("A" * 10_001 + ";", "(" + "B," * 4_999 + "B);"),
        # One tree counted, then two, ended by ';' or by the end of the text:
        # the trees' ends would overrun.
        ("(A,B)" + "C" * 10_000, "A;" * 5_002 + "B"),
        ("(A,B)" + "C" * 10_000, "A;" + "B" * 10_003),
        # One node counted, then a second tree's root.
        ("A" * 10_003, "A;" + "B" * 10_001),
    ],
)
def test_a_file_rewritten_at_the_same_size_is_refused_not_overrun(
    tmp_path, before, after
) -> None:
    path = _write(tmp_path, before)
    layout = tree_file_layout(path)
    path.write_text(after)
    assert path.stat().st_size == layout.n_text_bytes
    with pytest.raises(ValueError, match="changed while it was read"):
        read_trees(layout, memory_bytes=10**9)


def test_a_damaged_gzip_stream_is_refused_as_unreadable(tmp_path) -> None:
    packed = bytearray(gzip.compress(b"((A:1,B:2)C:3,D:4)E;" * 200, mtime=0))
    packed[30] ^= 0xFF
    path = tmp_path / "damaged.nwk.gz"
    path.write_bytes(bytes(packed))
    with pytest.raises(ValueError, match="cannot be read"):
        _read(path)
    # Damaged after its layout counted it, the stream is refused as it is read.
    path.write_bytes(gzip.compress(b"((A:1,B:2)C:3,D:4)E;" * 200, mtime=0))
    layout = tree_file_layout(path)
    path.write_bytes(bytes(packed))
    with pytest.raises(ValueError, match="cannot be read"):
        read_trees(layout, memory_bytes=10**9)


def test_a_label_after_a_length_or_a_group_after_a_group_is_refused(tmp_path) -> None:
    with pytest.raises(
        ValueError, match="a label after its length at byte 6 of its text"
    ):
        _parse(tmp_path, "(:1[c]B,C);")
    with pytest.raises(
        ValueError,
        match=r"an unexpected '\(' after a label, a length or a closed group at byte 5 of",
    ):
        _parse(tmp_path, "(A,B)(C,D);")


def test_unquoted_numbers_on_internal_nodes_are_support_not_names(tmp_path) -> None:
    trees = _parse(tmp_path, "((c0:1,c1:1)95:1,(c2:1,c3:1)'80':1,(7,8)-1.5e2)100;")
    # As treedata reads them, the root's 100, 95 and -1.5e2 are support values;
    # the quoted '80' and the leaves' numbers are names.
    assert trees.names.tolist() == ["", "", "c0", "c1", "80", "c2", "c3", "", "7", "8"]


def test_stores_of_treedata_before_0_3_are_refused_before_they_are_read(
    tmp_path,
) -> None:
    root = zarr.open_group(tmp_path / "old.zarr", mode="w", zarr_format=2)
    obst = root.create_array("obst", shape=(), dtype=str)
    obst[()] = json.dumps({"t": _graph([("r", "a")], r=0.0, a=1.0)})
    with pytest.raises(ValueError, match="before 0.3.0.*write it again"):
        tree_file_layout(tmp_path / "old.zarr")
    # A store rewritten that way after its layout is refused as it is read.
    path = _columnar(tmp_path / "t.zarr", {"t": {"nodes": ["r"], "edges": []}})
    layout = tree_file_layout(path)
    root = zarr.open_group(path, mode="w", zarr_format=2)
    root.create_array("obst", shape=(), dtype=str)
    with pytest.raises(ValueError, match="changed while it was read"):
        read_trees(layout, memory_bytes=10**9)


def test_files_of_the_earliest_treedata_name_their_layout(tmp_path) -> None:
    path = tmp_path / "early.h5td"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw.treedata", data="{}", dtype=h5py.string_dtype())
    with pytest.raises(ValueError, match="'raw.treedata'.*write it again"):
        tree_file_layout(path)


def test_the_json_scan_counts_objects_members_and_the_values_held() -> None:
    from cytearc.readers._treedata import _scan_json

    # Worked by hand: the open containers hold at most 7 values, at the ','
    # before "c"; the strings' marks, an escaped quote among them, count for
    # nothing.
    text = '{"a": [1, [2, 3], {"b": "}{,:[\\"]"}], "c": 4}'
    data = np.frombuffer(text.encode(), dtype=np.uint8)
    shape = _scan_json(data, np.empty(8, dtype=np.int64), np.empty(8, dtype=np.bool_))
    assert shape == (2, 3, 7, len(text), ord("}"))
    greek = np.frombuffer('{"λ": 1}'.encode(), dtype=np.uint8)
    shape = _scan_json(greek, np.empty(8, dtype=np.int64), np.empty(8, dtype=np.bool_))
    assert shape == (1, 1, 1, 8, 0xCE)
    deep = np.frombuffer(b"[[[]]]", dtype=np.uint8)
    assert (
        _scan_json(deep, np.empty(2, dtype=np.int64), np.empty(2, dtype=np.bool_))[0]
        == -1
    )


def test_json_that_python_cannot_read_is_refused_as_such(tmp_path) -> None:
    graph = _graph([("r", "a")])
    graph["nodes"]["a"]["meta"] = "PLACEHOLDER"
    text = json.dumps({"t": graph})
    deep = text.replace('"PLACEHOLDER"', "[" * 9_000 + "]" * 9_000)
    with pytest.raises(
        ValueError, match="whose objects and arrays nest more than 8192"
    ):
        _read(_h5td(tmp_path / "deep.h5td", deep, raw=True))
    huge = text.replace('"PLACEHOLDER"', "1" * 5_000)
    with pytest.raises(ValueError, match="not JSON that Python can read: Exceeds"):
        _read(_h5td(tmp_path / "huge.h5td", huge, raw=True))


def test_times_given_as_lists_or_long_text_are_refused_briefly(tmp_path) -> None:
    graph = _graph([("r", "a")])
    graph["nodes"]["a"]["time"] = [1.0] * 1_000
    with pytest.raises(ValueError, match="a time that is not a number: a list$"):
        _read(_h5td(tmp_path / "list.h5td", {"t": graph}))
    graph["nodes"]["a"]["time"] = "x" * 1_000
    with pytest.raises(ValueError) as refused:
        _read(_h5td(tmp_path / "text.h5td", {"t": graph}))
    # Quoted values keep their first 38 and last 39 characters of 80.
    assert str(refused.value).endswith(
        "not a number: '" + "x" * 37 + "..." + "x" * 38 + "'"
    )
