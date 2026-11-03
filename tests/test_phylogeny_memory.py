"""Importing, loading and cutting phylogenies run within the smallest budget they accept.

The budget climbs from one byte to each refusal's own estimate until the step
runs, and the peak that tracemalloc records while it runs must not exceed
that budget, as the clone summaries' memory tests check. The trees stress
each estimate: a 59 MB tree on one line, long and non-ASCII names, many
small files, TreeData with character states to drop, attributes that hold objects, a star of leaves and in Zarr, and cuts into a few
clones or one clone per cell.
"""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from cytearc.readers.phylogeny import read_trees, tree_file_layout
from tests.memory_helpers import budget_that_runs, stated_bytes
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.slow

N_CELLS = 60_000
N_LEAVES = 50_000
N_OUTSIDE = 2_000


def _random_tree(names: list[str], rng: np.random.Generator) -> str:
    """Newick text of a random tree over ``names``, on one line.

    Groups of two to five, now and then up to 67, merge until one is left,
    with lengths that make every leaf end at time 10.
    """
    texts = [f"{name}" for name in names]
    heights = [0.0] * len(texts)
    node = 0
    while len(texts) > 1:
        size = int(rng.choice([2, 2, 2, 3, 5, 67]))
        size = min(size, len(texts))
        picked = sorted(rng.choice(len(texts), size=size, replace=False).tolist())
        height = max(heights[i] for i in picked) + 1.0
        merged = (
            "("
            + ",".join(f"{texts[i]}:{height - heights[i]:g}" for i in picked)
            + f")n{node}"
        )
        node += 1
        for i in reversed(picked):
            texts[i] = texts[-1]
            heights[i] = heights[-1]
            texts.pop()
            heights.pop()
        texts.append(merged)
        heights.append(height)
    return texts[0] + ";"


def _h5td(path: Path, names: list[str], rng: np.random.Generator) -> Path:
    """A TreeData file of one tree with character states on every node."""
    nodes = {"root": {"time": 0.0, "characters": [0] * 42}}
    edges: dict[str, dict] = {"root": {}}
    for start in range(0, len(names), 50):
        group = f"g{start}"
        nodes[group] = {"time": 1.0, "characters": rng.integers(0, 9, 42).tolist()}
        edges["root"][group] = {"length": 1.0}
        edges[group] = {}
        for name in names[start : start + 50]:
            nodes[name] = {"time": 2.0, "characters": rng.integers(0, 9, 42).tolist()}
            edges[group][name] = {}
            edges[name] = {}
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "obst",
            data=json.dumps({"tree": {"nodes": nodes, "edges": edges}}),
            dtype=h5py.string_dtype(),
        )
        handle.create_dataset("X", data=np.zeros((len(names), 20), dtype=np.float32))
    return path


def _star_h5td(
    path: Path, names: list[str], *, nested: bool, beside: bool = False
) -> Path:
    """A TreeData file of one root over every leaf, as a star-shaped clone is.

    With ``nested`` every node also holds an attribute that is an object, as
    well as its character states. With ``beside`` the tree also holds, beside
    its nodes and edges, a million empty lists, which cost the most for their
    text, and an object of 300,000 keys, which the parse keeps to the end.
    """
    extra = {"meta": {"source": "test"}} if nested else {}
    nodes = {"root": {"time": 0.0, **extra}}
    edges: dict[str, dict] = {"root": {}}
    for index, name in enumerate(names):
        nodes[name] = {"time": 1.0, "characters": [index % 9] * 120, **extra}
        edges["root"][name] = {"length": 1.0}
        edges[name] = {}
    tree: dict[str, object] = {"nodes": nodes, "edges": edges}
    if beside:
        tree["meta"] = {
            "lists": [[] for _ in range(1_000_000)],
            "keys": {f"key{index}": 0 for index in range(300_000)},
        }
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "obst", data=json.dumps({"tree": tree}), dtype=h5py.string_dtype()
        )
    return path


def _tree_zarr(path: Path, names: list[str]) -> Path:
    """A TreeData Zarr store of one tree, its columns as treedata 0.3+ writes them."""
    import zarr

    groups = [f"g{start}" for start in range(0, len(names), 50)]
    nodes = ["root", *groups, *names]
    edges = [("root", group) for group in groups] + [
        (groups[index // 50], name) for index, name in enumerate(names)
    ]
    root = zarr.open_group(path, mode="w", zarr_format=3)
    tree = root.create_group("obst").create_group("tree")
    strings = np.dtypes.StringDType()
    tree.create_array("nodes", data=np.asarray(nodes, dtype=strings), chunks=(4_096,))
    tree.create_array("edges", data=np.asarray(edges, dtype=strings), chunks=(4_096, 2))
    times = [0.0] + [1.0] * len(groups) + [2.0] * len(names)
    tree.create_group("node_attrs").create_array(
        "time", data=np.asarray(times), chunks=(4_096,)
    )
    return path


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("phylogeny_memory")
    rng = np.random.default_rng(11)
    ids = [f"cell{i:06d}-1" for i in range(N_CELLS)]
    path = folder / "store.zarr"
    write_count_store(
        str(path),
        {"RNA": np.ones((N_CELLS, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    cells = store.snapshot_cell_selection("every_cell")
    leaves = ids[:N_LEAVES] + [f"outside{i}-1" for i in range(N_OUTSIDE)]
    rng.shuffle(leaves)
    big = folder / "big.nwk"
    big.write_text(_random_tree(leaves, rng))
    # Long names in Greek, of which the store's cells are the prefix-less part.
    wide = folder / "wide.nwk"
    prefix = "κύτταρο_" * 30
    wide.write_text(_random_tree([prefix + name for name in ids[:5_000]], rng))
    small = folder / "small"
    small.mkdir()
    files = []
    for index in range(200):
        tree = small / f"clone_{index}.nwk"
        tree.write_text(_random_tree(ids[index * 50 : index * 50 + 50], rng))
        files.append(tree)
    trees = read_trees(tree_file_layout(big), memory_bytes=10**10)
    # Every leaf ends at the root's height, the length of its longest path.
    leaf_time = float(
        np.nansum(trees.length[_path_to_root(trees, len(trees.parent) - 1)])
    )
    return SimpleNamespace(
        store=store,
        cells=cells,
        big=big,
        wide=wide,
        files=files,
        h5td=_h5td(folder / "tree.h5td", ids[:20_000], rng),
        zarr=_tree_zarr(folder / "tree.zarr", ids[20_000:40_000]),
        star=_star_h5td(folder / "star.h5td", ids[:50_000], nested=False),
        nested=_star_h5td(folder / "nested.h5td", ids[:20_000], nested=True),
        beside=_star_h5td(
            folder / "beside.h5td", ids[:5_000], nested=False, beside=True
        ),
        leaf_time=leaf_time,
    )


def _path_to_root(trees, node: int) -> list[int]:
    path = [node]
    while trees.parent[path[-1]] >= 0:
        path.append(int(trees.parent[path[-1]]))
    return path


def _imported(data: SimpleNamespace, path: Path, **options: object):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return data.store.clones.import_phylogeny(
            path, cell_selection=data.cells, time_unit="days", **options
        )


def _cut(**cut: object) -> Callable[[SimpleNamespace], object]:
    def run(data: SimpleNamespace) -> object:
        import warnings

        phylogeny = _imported(data, data.big)
        options = {
            name: data.leaf_time - 1e-6 if value == "leaves" else value
            for name, value in cut.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return data.store.clones.phylogeny_clones(
                phylogeny, invalidate_cache=True, **options
            )

    return run


CASES: dict[str, Callable[[SimpleNamespace], object]] = {
    "import of 52,000 leaves on one line": lambda data: _imported(
        data, data.big, invalidate_cache=True
    ),
    "import of 5,000 leaves with 240-character Greek names": lambda data: _imported(
        data,
        data.wide,
        barcode_pattern=r"_(?P<barcode>cell\d+-1)$",
        invalidate_cache=True,
    ),
    "import of 200 files": lambda data: _imported(
        data, data.files, invalidate_cache=True
    ),
    "import of TreeData with character states": lambda data: _imported(
        data, data.h5td, invalidate_cache=True
    ),
    "import of a TreeData Zarr store": lambda data: _imported(
        data, data.zarr, invalidate_cache=True
    ),
    "import of a star of 50,000 leaves in TreeData": lambda data: _imported(
        data, data.star, invalidate_cache=True
    ),
    "import of TreeData whose attributes hold objects": lambda data: _imported(
        data, data.nested, invalidate_cache=True
    ),
    "import of TreeData holding lists and keys beside its tree": lambda data: _imported(
        data, data.beside, invalidate_cache=True
    ),
    "load of 52,000 leaves": lambda data: data.store.clones.load_phylogeny(
        _imported(data, data.big)
    ),
    "cut into a few clones": _cut(time=2.0),
    "cut into one clone per cell": _cut(time="leaves"),
    "cut at a depth": _cut(depth=6),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_traced_peak_fits_the_smallest_budget_accepted(data, case) -> None:
    run = CASES[case]
    # Compiled kernels and saved inputs are made before anything is traced.
    run(data)
    # The budget climbs from one byte to the bytes that each refusal states.
    # Imports and cuts are made afresh, and loads read, on each run.
    budget, _refusals, _result, peak = budget_that_runs(
        data.store,
        lambda _store: run(data),
        refusal_budget=stated_bytes,
        max_refusals=100,
    )
    print(f"{case}: peak {peak} budget {budget} ratio {peak / budget:.2f}")
    assert peak <= budget, (case, peak, budget)


def _one_line_tree(path: Path, n_leaves: int, rng: np.random.Generator) -> None:
    """A tree of ``n_leaves`` with DNA Typewriter's names, written on one line.

    The leaves hang in polytomies of up to 67 below a backbone that is as
    deep as there are polytomies, so the tree is wide and deep at once.
    """
    with open(path, "w", encoding="ascii") as handle:
        n_groups = 0
        written = 0
        opened = 0
        while written < n_leaves:
            size = int(min(rng.integers(2, 68), n_leaves - written))
            handle.write("(" if opened == 0 else ",(")
            opened += 1
            leaves = ",".join(
                f"exp1_P{chr(65 + i % 26)}-{i % 100:02d}A.LIG-P3-B11_RT-P{i:07d}"
                f":{rng.random():.8f}"
                for i in range(written, written + size)
            )
            handle.write(f"({leaves})g{n_groups}:{rng.random():.6f}")
            written += size
            n_groups += 1
        handle.write("):1.0" * (opened - 1) + ");")


def test_a_59_megabyte_tree_on_one_line_parses_within_its_budget(tmp_path) -> None:
    path = tmp_path / "typewriter.nwk"
    _one_line_tree(path, 1_280_000, np.random.default_rng(3))
    text_bytes = path.stat().st_size
    assert text_bytes > 59_000_000
    with open(path, "rb") as handle:
        assert handle.read().count(b"\n") == 0
    layout = tree_file_layout(path)
    # The backbone is about 38,000 groups deep, which recursion could not read.
    trees = read_trees(layout, memory_bytes=10**11)
    assert int(trees.leaf.sum()) == 1_280_000
    depth = np.zeros(trees.n_nodes, dtype=np.int64)
    for node in range(1, trees.n_nodes):
        depth[node] = depth[trees.parent[node]] + 1
    assert depth.max() > 30_000
    del trees, depth
    # The budget climbs from one byte to the bytes that each refusal states.
    budget, _refusals, trees, peak = budget_that_runs(
        lambda budget: budget,
        lambda budget: read_trees(layout, memory_bytes=budget),
        refusal_budget=stated_bytes,
    )
    print(f"59 MB tree: peak {peak} budget {budget} ratio {peak / budget:.2f}")
    assert peak <= budget
    assert trees.names[2].startswith("exp1_P")
