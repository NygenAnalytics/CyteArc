"""Each clone summary runs within the smallest memory budget that it accepts.

The budget climbs from one byte to each refusal's own estimate until the
summary runs, and the peak that tracemalloc records while it runs must not
exceed that budget. The test never repeats the budgets' formulas, so it
checks that whatever a summary estimates bounds what it allocates, on inputs
that stress each estimate: long and emoji labels, long clone labels, a few
measured cells of many clone cells, a label artifact and a cell selection of
more cells than the clones hold, a label for each of many cells, many sizes,
many pairs of groups, one-cell clones, 13 key columns, and clone labels
held beside many rows. Clonotypes, whose receptor types every summary keeps
apart, are measured alike, as are the steps that define and load them, also
within those labels and for cells with long IDs, the labels that figures give
them, and clonotype clusters, whose smallest budget is found by bisection.
"""

import re
from collections.abc import Callable
from types import SimpleNamespace

import numba
import numpy as np
import pytest

import cytearc.datastore._operations.repertoire as repertoire_operations
import cytearc.repertoire.clonotypes as clonotypes_module
from cytearc.datastore.datastore import DataStore
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import logger
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    smallest_budget,
    traced_peak,
    write_clone_contigs,
    write_lines,
)
from tests.storage_helpers import write_count_store

pytestmark = pytest.mark.slow

N_CELLS = 10_000
N_MEASURED = 500
N_LONG_IDS = 2_000
# Clonotypes of one cell each, as in most T cell repertoires, cost more for
# their clonotypes than for their cells; with more cells than one chunk of
# cell IDs holds, the IDs' chunk does not hide their cost.
N_SINGLETONS = 50_000
# Cells with contigs beside their chains, whose junctions no slot holds.
N_EXTRA_CELLS = 10_000
# Clonotypes of one cell each, named by long junctions of wide characters.
N_LABELLED = 5_000


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> SimpleNamespace:
    rng = np.random.default_rng(7)
    path = tmp_path_factory.mktemp("clone_memory") / "store.zarr"
    ids = np.asarray([f"cell{i:06d}-1" for i in range(N_CELLS)])
    counts = rng.poisson(2.0, size=(N_CELLS, 4)).astype(np.uint16) + 1
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=ids)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    draws = rng.integers(0, 8, size=N_CELLS)
    store.cells.insert(
        "donor", np.asarray([f"donor_with_a_long_name_{k:04d}" for k in draws])
    )
    store.cells.insert(
        "condition", np.asarray([f"condition_named_at_length_{k:04d}" for k in draws])
    )
    store.cells.insert(
        "emoji", np.asarray(["\U0001f600" * 17 + f"{k:03d}" for k in draws])
    )
    # A label of its own for each of the first 2,000 cells, and none for the
    # others, makes groups that cost more than their cells.
    own = np.asarray([f"g{i:06d}" if i < 2_000 else "" for i in range(N_CELLS)])
    store.cells.insert("own", own)
    # Twelve copies of those labels give as many key columns and no more
    # groups.
    for copy in range(12):
        store.cells.insert(f"own_{copy}", own)
    for n_first in (220, 300, 800):
        store.cells.insert(f"first_{n_first}", np.arange(N_CELLS) < n_first)
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.cells.insert("first", np.arange(N_CELLS) < N_MEASURED)
    # Late cells for fate bias, none of them among the first.
    store.cells.insert("after_first", np.arange(N_CELLS) >= N_MEASURED)
    store.cells.insert(
        "next_300", (np.arange(N_CELLS) >= N_MEASURED) & (np.arange(N_CELLS) < 800)
    )
    # Zipf clone sizes, about one cell in ten without a clone.
    sizes = np.minimum(rng.zipf(1.8, size=N_CELLS), 300)
    clone_of = np.repeat(np.arange(len(sizes)), sizes)[:N_CELLS]
    rng.shuffle(clone_of)
    labels = np.asarray([f"clone{c}" for c in clone_of], dtype=object)
    labels[rng.random(N_CELLS) < 0.1] = ""
    store.cells.insert("clone", labels)
    store.cells.insert(
        "long_clone", np.asarray([f"{'ACGT' * 40}_{i}" for i in range(N_CELLS)])
    )
    emoji = rng.integers(0x1F300, 0x1F5FF, size=(N_CELLS, 40))
    store.cells.insert(
        "emoji_clone",
        np.asarray(["".join(map(chr, row)) + str(i) for i, row in enumerate(emoji)]),
    )
    # Clones of 100 early and 100 late cells: the first half of the cells is
    # early and the second late, and the late cells fall in 200 fates, so
    # each early cell's fractions are the largest arrays of its fate bias.
    half = N_CELLS // 2
    store.cells.insert("first_half", np.arange(N_CELLS) < half)
    store.cells.insert("second_half", np.arange(N_CELLS) >= half)
    store.cells.insert(
        "paired_clone",
        np.asarray([f"p{(i % half) // 100}" for i in range(N_CELLS)], dtype=object),
    )
    store.cells.insert(
        "state",
        np.asarray([f"f{k:03d}" for k in rng.integers(0, 200, size=N_CELLS)]),
    )
    store.cells.insert(
        "label_of_each_cell", np.asarray([f"c{i:06d}" for i in range(N_CELLS)])
    )
    every = store.snapshot_cell_selection("every_cell")
    first = store.snapshot_cell_selection("first")
    # Label artifacts of a label for each of the first cells: their small
    # chunks leave the pairs or rows of many groups the largest estimate.
    own_labels = {
        n_first: store.snapshot_cluster_labels(
            "own", cell_selection=store.snapshot_cell_selection(f"first_{n_first}")
        )
        for n_first in (220, 300, 800)
    }
    next_300 = store.snapshot_cell_selection("next_300")
    # Each clone's cells carry its receptor, so its clonotypes are the clones,
    # with T cell and B cell clones.
    contigs = write_clone_contigs(
        path.parent, "contigs.csv", ids, np.where(labels == "", -1, clone_of)
    )
    receptors = store.repertoire.import_receptors(contigs, cell_selection=every)
    clonotypes = store.repertoire.clonotypes(receptors)
    return SimpleNamespace(
        store=store,
        folder=path.parent,
        every=every,
        first=first,
        after_first=store.snapshot_cell_selection("after_first"),
        next_300=next_300,
        # A fate of its own for each of the 300 cells after the first.
        own_next_300=store.snapshot_cluster_labels("own", cell_selection=next_300),
        receptors=receptors,
        clonotypes=clonotypes,
        clonotype_ids=store.repertoire.load_clonotypes(clonotypes)
        .clonotypes["clonotype"]
        .tolist(),
        own_clonotypes=store.repertoire.clonotypes(
            receptors, within="label_of_each_cell"
        ),
        clones=store.clones.from_column("clone", cell_selection=every),
        long=store.clones.from_column("long_clone", cell_selection=every),
        emoji=store.clones.from_column("emoji_clone", cell_selection=every),
        few=store.clones.from_column("clone", cell_selection=first),
        labels=store.snapshot_cluster_labels("donor", cell_selection=every),
        own_labels=own_labels,
        own_labels_300=own_labels[300],
        named=sorted(set(labels.tolist()) - {""})[::3][:1_000],
        first_half=store.snapshot_cell_selection("first_half"),
        second_half=store.snapshot_cell_selection("second_half"),
        paired=store.clones.from_column("paired_clone", cell_selection=every),
    )


def _expansion(**options: object) -> Callable[[SimpleNamespace], object]:
    def run(data: SimpleNamespace) -> object:
        return data.store.clones.expansion(
            data.clones, invalidate_cache=True, **options
        )

    return run


def _loaded(clones: str, **options: str) -> Callable[[SimpleNamespace], object]:
    def run(data: SimpleNamespace) -> object:
        selected = {name: getattr(data, value) for name, value in options.items()}
        saved = data.store.clones.expansion(getattr(data, clones), **selected)
        return data.store.clones.load_expansion(saved)

    return run


def _coupling(clones: str, **options: object) -> Callable[[SimpleNamespace], object]:
    def run(data: SimpleNamespace) -> object:
        chosen = {
            name: getattr(data, value) if name == "groups" else value
            for name, value in options.items()
        }
        return data.store.clones.coupling(getattr(data, clones), **chosen)

    return run


def _four_threads(run: Callable[[SimpleNamespace], object]) -> Callable:
    def on_four(data: SimpleNamespace) -> object:
        data.store.nthreads = 4
        try:
            return run(data)
        finally:
            data.store.nthreads = 1

    return on_four


def _fate_bias(
    clones: str, late: str = "after_first", **options: object
) -> Callable[[SimpleNamespace], object]:
    def run(data: SimpleNamespace) -> object:
        chosen = {
            name: getattr(data, value) if name == "groups" else value
            for name, value in options.items()
        }
        return data.store.clones.fate_bias(
            getattr(data, clones),
            early=data.first,
            late=getattr(data, late),
            invalidate_cache=True,
            **chosen,
        )

    return run


def _loaded_fate_bias(
    clones: str, late: str = "after_first", **options: object
) -> Callable[[SimpleNamespace], object]:
    def run(data: SimpleNamespace) -> object:
        chosen = {
            name: getattr(data, value) if name == "groups" else value
            for name, value in options.items()
        }
        saved = data.store.clones.fate_bias(
            getattr(data, clones), early=data.first, late=getattr(data, late), **chosen
        )
        return data.store.clones.load_fate_bias(saved)

    return run


def _saved_fate_bias(
    clones: str, **options: str
) -> Callable[[SimpleNamespace], object]:
    """Load the fate bias that the first run saves, so later runs only load it.

    The first run, which warms the caches at the full budget, measures the
    fate bias of the first cells; the smallest budget accepted and the traced
    peak are then those of loading alone.
    """
    saved: list[ArtifactRef] = []

    def run(data: SimpleNamespace) -> object:
        if not saved:
            saved.append(
                data.store.clones.fate_bias(
                    getattr(data, clones),
                    early=data.first,
                    late=data.after_first,
                    **options,
                )
            )
        return data.store.clones.load_fate_bias(saved[0])

    return run


CASES: dict[str, Callable[[SimpleNamespace], object]] = {
    "expansion of every cell": _expansion(),
    "expansion by two columns of long labels": _expansion(
        group_by=["donor", "condition"]
    ),
    "expansion by emoji labels": _expansion(group_by="emoji"),
    "expansion by a label for each of many cells": _expansion(group_by="own"),
    "expansion of a few of the clones' cells": lambda data: data.store.clones.expansion(
        data.long, cell_selection=data.first, invalidate_cache=True
    ),
    "loaded expansion of 160-character clone labels": _loaded("long"),
    "loaded expansion of emoji clone labels": _loaded("emoji"),
    "loaded expansion of a few of the clones' cells": _loaded(
        "long", cell_selection="first"
    ),
    "loaded expansion of a few cells of short clone labels": _loaded(
        "clones", cell_selection="first"
    ),
    "diversity by two columns of long labels": lambda data: data.store.clones.diversity(
        data.clones, group_by=["donor", "condition"]
    ),
    "diversity by a label for each of many cells": lambda data: (
        data.store.clones.diversity(data.clones, group_by="own")
    ),
    "diversity of a few cells by labels of every cell": lambda data: (
        data.store.clones.diversity(data.few, groups=data.labels)
    ),
    "diversity of a few cells within a selection of every cell": lambda data: (
        data.store.clones.diversity(data.few, cell_selection=data.every)
    ),
    "rarefaction at 5,000 sizes": lambda data: data.store.clones.rarefaction(
        data.clones, orders=(0, 1), sizes=range(1, 5_001)
    ),
    "overlap by two columns of long labels": lambda data: data.store.clones.overlap(
        data.clones, group_by=["donor", "condition"]
    ),
    "overlap by a label for each of 300 cells": lambda data: data.store.clones.overlap(
        data.clones, groups=data.own_labels[300]
    ),
    "overlap of one-cell clones by a label for each of 300 cells": lambda data: (
        data.store.clones.overlap(data.long, groups=data.own_labels[300])
    ),
    "overlap of a few cells by labels of every cell": lambda data: (
        data.store.clones.overlap(data.few, groups=data.labels)
    ),
    "track of a few cells by 13 key columns": lambda data: data.store.clones.track(
        data.few,
        group_by=[f"own_{copy}" for copy in range(12)],
        groups=data.own_labels[800],
        top=1,
    ),
    "track of emoji clone labels by a label for each of 220 cells": lambda data: (
        data.store.clones.track(data.emoji, groups=data.own_labels[220], top=1)
    ),
    "track of 1,000 named clones": lambda data: data.store.clones.track(
        data.clones, labels=data.named
    ),
    "track of a few cells by labels of every cell": lambda data: (
        data.store.clones.track(data.few, groups=data.labels)
    ),
    "coupling by two columns of long labels": _coupling(
        "clones", group_by=["donor", "condition"]
    ),
    "coupling by a label for each of 300 cells": _coupling(
        "clones", groups="own_labels_300", permutations=50
    ),
    "coupling on four threads by a label for each of 300 cells": _four_threads(
        _coupling("clones", groups="own_labels_300", permutations=50)
    ),
    "coupling of one-cell clones by a label for each of 300 cells": _coupling(
        "long", groups="own_labels_300", method="jaccard", permutations=50
    ),
    "coupling of a few cells by labels of every cell": _coupling(
        "few", groups="labels", method="normalized_covariance"
    ),
    "fate bias by two columns of long labels": _fate_bias(
        "clones", group_by=["donor", "condition"]
    ),
    "fate bias by a label for each of 300 late cells": _fate_bias(
        "clones", late="next_300", groups="own_next_300"
    ),
    "fate bias of a few late cells of many clone cells": _fate_bias(
        "long", late="next_300", group_by="emoji"
    ),
    "loaded fate bias of 160-character clone labels": _loaded_fate_bias(
        "long", group_by="donor"
    ),
    "loaded fate bias of a few early cells of short clone labels": _saved_fate_bias(
        "clones", group_by="donor"
    ),
    "loaded fate bias of emoji clone labels by a label for each of 300 late cells": (
        _loaded_fate_bias("emoji", late="next_300", groups="own_next_300")
    ),
    "fate bias of 200 fates of clones of 100 early and 100 late cells": lambda data: (
        data.store.clones.fate_bias(
            data.paired,
            group_by="state",
            early=data.first_half,
            late=data.second_half,
            invalidate_cache=True,
        )
    ),
    "loaded fate bias of 200 fates of clones of 100 early and 100 late cells": (
        lambda data: data.store.clones.load_fate_bias(
            data.store.clones.fate_bias(
                data.paired,
                group_by="state",
                early=data.first_half,
                late=data.second_half,
            )
        )
    ),
    "expansion of clonotypes": lambda data: data.store.clones.expansion(
        data.clonotypes, invalidate_cache=True
    ),
    "expansion of clonotypes by two columns of long labels": lambda data: (
        data.store.clones.expansion(
            data.clonotypes, group_by=["donor", "condition"], invalidate_cache=True
        )
    ),
    "loaded expansion of the clonotypes of a few cells": lambda data: (
        data.store.clones.load_expansion(
            data.store.clones.expansion(data.clonotypes, cell_selection=data.first)
        )
    ),
    "diversity of clonotypes by emoji labels": lambda data: data.store.clones.diversity(
        data.clonotypes, group_by="emoji"
    ),
    "rarefaction of clonotypes": lambda data: data.store.clones.rarefaction(
        data.clonotypes, orders=(0, 1)
    ),
    "clonotypes loaded as clones": lambda data: data.store.clones.load_clones(
        data.clonotypes
    ),
    "expansion of clonotypes within a label of each cell": lambda data: (
        data.store.clones.expansion(data.own_clonotypes, invalidate_cache=True)
    ),
    "diversity of clonotypes within a label of each cell": lambda data: (
        data.store.clones.diversity(data.own_clonotypes)
    ),
    "clonotypes within a label of each cell loaded as clones": lambda data: (
        data.store.clones.load_clones(data.own_clonotypes)
    ),
    "overlap of clonotypes by two columns of long labels": lambda data: (
        data.store.clones.overlap(data.clonotypes, group_by=["donor", "condition"])
    ),
    "track of clonotypes by a label for each of 300 cells": lambda data: (
        data.store.clones.track(data.clonotypes, groups=data.own_labels[300], top=1)
    ),
    "coupling of clonotypes by two columns of long labels": _coupling(
        "clonotypes", group_by=["donor", "condition"], permutations=50
    ),
    "fate bias of clonotypes by long labels": _fate_bias(
        "clonotypes", group_by="donor"
    ),
    "loaded fate bias of the clonotypes of a few early cells": _saved_fate_bias(
        "clonotypes", group_by="donor"
    ),
    "clonotypes exported as a clone matrix": lambda data: (
        data.store.clones.export_clones(
            data.clonotypes, data.folder / "clonotypes.mtx", format="matrix"
        )
    ),
    "clonotypes": lambda data: data.store.repertoire.clonotypes(
        data.receptors, invalidate_cache=True
    ),
    "clonotypes within two columns of long labels": lambda data: (
        data.store.repertoire.clonotypes(
            data.receptors, within=["donor", "condition"], invalidate_cache=True
        )
    ),
    "clonotypes within emoji labels": lambda data: data.store.repertoire.clonotypes(
        data.receptors, within="emoji", invalidate_cache=True
    ),
    "clonotypes within a label of each cell": lambda data: (
        data.store.repertoire.clonotypes(
            data.receptors, within="label_of_each_cell", invalidate_cache=True
        )
    ),
    "loaded clonotypes": lambda data: data.store.repertoire.load_clonotypes(
        data.clonotypes
    ),
    "loaded clonotypes within a label of each cell": lambda data: (
        data.store.repertoire.load_clonotypes(
            data.store.repertoire.clonotypes(
                data.receptors, within="label_of_each_cell"
            )
        )
    ),
    "figure labels of the 20 largest clonotypes": lambda data: (
        data.store._clone_figure_labels(data.clonotypes, data.clonotype_ids[:20])
    ),
    "figure labels of every clonotype": lambda data: data.store._clone_figure_labels(
        data.clonotypes, data.clonotype_ids
    ),
}

# Clonotype clusters by residues, unless a case compares nucleotides.
CLUSTER_CASES: dict[str, dict[str, object]] = {
    "within one substitution": {"sequence": "aa", "metric": "hamming", "cutoff": 1},
    "within one edit": {"sequence": "aa", "metric": "levenshtein", "cutoff": 1},
    "of nucleotides within 15 percent and a V gene": {
        "metric": "normalized_hamming",
        "cutoff": 15,
        "same_v_gene": True,
    },
    "of either arm with every chain": {
        "sequence": "aa",
        "metric": "hamming",
        "cutoff": 1,
        "receptor_arms": "any",
        "dual_ir": "all",
    },
    "within a label of each cell": {
        "sequence": "aa",
        "metric": "hamming",
        "cutoff": 1,
        "within": "label_of_each_cell",
    },
}


def _smallest_budget(store: DataStore, run: Callable[[], object]) -> int:
    """Raise the budget to each refusal's estimate until ``run`` runs."""
    budget = 1
    for _attempt in range(100):
        store.memoryBytes = budget
        try:
            run()
            return budget
        except MemoryError as error:
            found = re.search(r"about (\d+) bytes", str(error))
            assert found is not None, str(error)
            estimate = int(found.group(1))
            # A refusal names a larger estimate than the budget it refused.
            assert estimate > budget, str(error)
            budget = estimate
    raise AssertionError("the budget did not settle")


def _traced_at_smallest_budget(
    store: DataStore, run: Callable[[], object]
) -> tuple[int, int]:
    """The peak that tracemalloc records at the smallest budget accepted."""
    full = store.memoryBytes
    try:
        # Compiled kernels and caches are warmed before anything is traced.
        run()
        budget = _smallest_budget(store, run)
        store.memoryBytes = budget
        # Python grows its table of interned strings in one allocation, 15 MB
        # when the table passes about 350,000 strings, during whichever run
        # adds the next one. A peak above the budget is therefore traced
        # again: a summary that holds too much does so in both runs.
        _result, peak = traced_peak(run, budget)
    finally:
        store.memoryBytes = full
    return peak, budget


@pytest.mark.parametrize("case", list(CASES))
def test_the_traced_peak_fits_the_smallest_budget_accepted(data, case) -> None:
    peak, budget = _traced_at_smallest_budget(data.store, lambda: CASES[case](data))
    assert peak <= budget, (case, peak, budget)


# The two summaries are run, each at the smallest budget it accepts, in a fresh
# process whose Zarr reads no earlier test left running. Each finished task of
# Zarr's pool is held until the next finishes, as the thread that ran it holds
# its chunk, as stored and decoded, on a busy machine until after the next
# read has begun.
_HELD_READ_SCRIPT = """
import json, re, sys, threading, tracemalloc
from concurrent.futures import thread
from cytearc.datastore.datastore import DataStore

store = DataStore(sys.argv[1], default_assay="RNA", min_features_per_cell=0, nthreads=1)
every = store.snapshot_cell_selection("every_cell")
first = store.snapshot_cell_selection("first")
long = store.clones.from_column("long_clone", cell_selection=every)
few = store.clones.from_column("clone", cell_selection=first)
cases = {
    "expansion of a few of the clones' cells": lambda: store.clones.expansion(
        long, cell_selection=first, invalidate_cache=True
    ),
    "diversity of a few cells within a selection of every cell": lambda: (
        store.clones.diversity(few, cell_selection=every)
    ),
}
lock, held, run = threading.Lock(), [None], thread._WorkItem.run

def run_and_hold(item, *args):
    run(item, *args)
    with lock:
        held[0] = item

results = {}
for name, call in cases.items():
    call()
    budget = 1
    while True:
        store.memoryBytes = budget
        try:
            call()
            break
        except MemoryError as error:
            budget = int(re.search(r"about (\\d+) bytes", str(error)).group(1))
    thread._WorkItem.run = run_and_hold
    tracemalloc.start()
    call()
    results[name] = [tracemalloc.get_traced_memory()[1], budget]
    tracemalloc.stop()
    thread._WorkItem.run = run
print(json.dumps(results))
"""


def test_the_budgets_count_the_read_that_zarr_still_holds(data) -> None:
    import json
    import subprocess
    import sys

    # Matching the clones to a few measured cells, and grouping a few cells
    # within a selection of every cell, read cell selections whose stored
    # chunk outweighs the cells they select; the read before can still hold
    # such a chunk beside them.
    traced = subprocess.run(
        [sys.executable, "-c", _HELD_READ_SCRIPT, str(data.store.zarr_loc)],
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(traced.stdout.strip().splitlines()[-1])

    assert set(results) == {
        "expansion of a few of the clones' cells",
        "diversity of a few cells within a selection of every cell",
    }
    for case, (peak, budget) in results.items():
        assert peak <= budget, (case, peak, budget)


# Grouping 400 cells, spread over 20,000, by text columns of 24 characters in
# a fresh process. While the cells are grouped, every task that Zarr's pool
# ran for a row read is held until the next row read ends, as master's test of
# the held codes read holds them, so each read has the read before beside it.
_CHUNKED_GROUPS_SCRIPT = """
import json, re, sys, threading, tracemalloc
from concurrent.futures import thread
import numpy as np
import cytearc.metadata.rows as rows
from cytearc.datastore._operations import clones as operations
from cytearc.datastore.datastore import DataStore
from tests.storage_helpers import write_count_store

n_cells, path = 20_000, sys.argv[1]
rng = np.random.default_rng(3)
ids = np.asarray([f"cell{i:06d}-1" for i in range(n_cells)])
write_count_store(path, {"RNA": np.ones((n_cells, 2), dtype=np.uint16)}, "uint16", cell_ids=ids)
store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
alphabet = np.asarray(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
chunk_rows = {"in_10_chunks": 2_000, "whole_a": n_cells, "whole_b": n_cells}
for name, rows_per_chunk in chunk_rows.items():
    labels = np.asarray(["".join(row) for row in rng.choice(alphabet, size=(n_cells, 24))])
    store.zw["cellData"].create_array(name, data=labels, chunks=(rows_per_chunk,))
store.cells.insert("few", np.arange(n_cells) % 50 == 0)
store.cells.insert("clone", np.asarray([f"clone{i % 7}" for i in range(n_cells)]))
store = DataStore(path, default_assay="RNA", min_features_per_cell=0, nthreads=1)
for name, rows_per_chunk in chunk_rows.items():
    assert store.cells._get_array(name).chunks == (rows_per_chunk,)
clones = store.clones.from_column("clone", cell_selection=store.snapshot_cell_selection("few"))
cases = {
    "a column in 10 chunks": ["in_10_chunks"],
    "two columns of one chunk each": ["whole_a", "whole_b"],
}
lock, reading, held = threading.Lock(), [0], {}
run, read_rows, cell_groups = thread._WorkItem.run, rows._read_array_rows, operations.cell_groups

def run_and_hold(item, *args):
    with lock:
        read = reading[0]
    run(item, *args)
    with lock:
        if read >= reading[0] - 1:
            held.setdefault(read, []).append(item)

def read_and_count(array, indices):
    values = read_rows(array, indices)
    with lock:
        held.pop(reading[0] - 1, None)
        reading[0] += 1
    return values

def grouped_with_reads_held(*args, **kwargs):
    thread._WorkItem.run, rows._read_array_rows = run_and_hold, read_and_count
    try:
        return cell_groups(*args, **kwargs)
    finally:
        thread._WorkItem.run, rows._read_array_rows = run, read_rows
        with lock:
            held.clear()
            reading[0] = 0

results, full = {}, store.memoryBytes
for case, columns in cases.items():
    def call():
        return store.clones.diversity(clones, group_by=columns)
    store.memoryBytes = full
    call()
    budget = 1
    while True:
        store.memoryBytes = budget
        try:
            call()
            break
        except MemoryError as error:
            budget = int(re.search(r"about (\\d+) bytes", str(error)).group(1))
    peaks = []
    for hold in (False, True):
        operations.cell_groups = grouped_with_reads_held if hold else cell_groups
        tracemalloc.start()
        call()
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    operations.cell_groups = cell_groups
    results[case] = [*peaks, budget]
print(json.dumps(results))
"""


def test_grouping_by_columns_of_several_chunks_holds_its_budget(tmp_path) -> None:
    import json
    import subprocess
    import sys

    # The grouping reads each column a stored chunk at a time, rather than
    # every chunk its cells touch at once, and the read of a chunk can still
    # hold the chunk read before it, of the same column or the column before.
    traced = subprocess.run(
        [sys.executable, "-c", _CHUNKED_GROUPS_SCRIPT, str(tmp_path / "store.zarr")],
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(traced.stdout.strip().splitlines()[-1])

    # A chunk of 2,000 or 20,000 labels of 24 characters decodes to 96 bytes a
    # label; a peak that grew by half a decoded chunk held the read before.
    decoded = {
        "a column in 10 chunks": 2_000 * 96,
        "two columns of one chunk each": 20_000 * 96,
    }
    assert set(results) == set(decoded)
    for case, (peak, held_peak, budget) in results.items():
        assert peak <= budget, (case, peak, budget)
        assert held_peak <= budget, (case, held_peak, budget)
        assert held_peak - peak > decoded[case] // 2, (case, peak, held_peak)


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 4,
    reason="the coupling plans four workers only where Numba has four threads",
)
def test_the_null_on_four_threads_stays_within_the_bytes_it_plans(data) -> None:
    """With room for four workers, the coupling's plan bounds its traced peak.

    The smallest budget accepted measures the null on one worker, so this
    case gives the coupling room for four workers and 64 permutations at
    once, and checks the peak against the bytes that its plan reports.
    """
    store, threads = data.store, data.store.nthreads
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    store.nthreads = 4

    def run() -> object:
        return store.clones.coupling(
            data.clones, groups=data.own_labels_300, permutations=200
        )

    try:
        run()
        (line,) = [message for message in messages if "coupling of" in message]
        found = re.search(r"holding about (\d+) bytes", line)
        assert found is not None, line
        planned = int(found.group(1))
        messages.clear()
        # A peak above the plan is traced again, as the summaries' peaks are.
        _result, peak = traced_peak(run, planned)
    finally:
        store.nthreads = threads
        logger.remove(sink)
    traced = [message for message in messages if "coupling of" in message]
    assert traced and all(message == line for message in traced), traced
    assert "on 4 threads, 64 at once" in line, line
    assert peak <= planned, (peak, line)


@pytest.fixture(scope="module")
def long_ids(tmp_path_factory) -> SimpleNamespace:
    """Clonotypes of cells whose IDs, 80 characters, are wider than all else."""
    path = tmp_path_factory.mktemp("clonotype_ids") / "store.zarr"
    ids = np.asarray(
        [f"{'sample_with_a_long_name_' * 3}{i:08d}-1" for i in range(N_LONG_IDS)]
    )
    assert ids.dtype == np.dtype("<U82")
    counts = np.ones((N_LONG_IDS, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=ids)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    contigs = write_clone_contigs(
        path.parent, "contigs.csv", ids, np.arange(N_LONG_IDS) // 4
    )
    every = store.snapshot_cell_selection()
    clonotypes = store.repertoire.clonotypes(
        store.repertoire.import_receptors(contigs, cell_selection=every)
    )
    return SimpleNamespace(store=store, clonotypes=clonotypes)


def test_loaded_clonotypes_of_long_cell_ids_fit_the_smallest_budget(
    long_ids,
) -> None:
    peak, budget = _traced_at_smallest_budget(
        long_ids.store,
        lambda: long_ids.store.repertoire.load_clonotypes(long_ids.clonotypes),
    )
    assert peak <= budget, (peak, budget)


@pytest.fixture(scope="module")
def singletons(tmp_path_factory) -> SimpleNamespace:
    """Clonotypes of one cell each, every cell with two chains of each arm."""
    path = tmp_path_factory.mktemp("clonotype_singletons") / "store.zarr"
    ids = np.asarray([f"c{i}" for i in range(N_SINGLETONS)])
    counts = np.ones((N_SINGLETONS, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=ids)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    contigs = write_clone_contigs(
        path.parent,
        "contigs.csv",
        ids,
        np.arange(N_SINGLETONS),
        chains_per_arm=2,
    )
    every = store.snapshot_cell_selection()
    clonotypes = store.repertoire.clonotypes(
        store.repertoire.import_receptors(contigs, cell_selection=every)
    )
    return SimpleNamespace(store=store, clonotypes=clonotypes)


def test_loaded_clonotypes_of_one_cell_each_fit_the_smallest_budget(
    singletons,
) -> None:
    store, clonotypes = singletons.store, singletons.clonotypes
    loaded = store.repertoire.load_clonotypes(clonotypes)
    assert len(loaded.clonotypes) == N_SINGLETONS
    # Each of the 50,000 clonotypes has four chains to decode, so their
    # values pass through several batches.
    assert loaded.clonotypes["VJ_2_junction_aa"].notna().all()
    peak, budget = _traced_at_smallest_budget(
        store, lambda: store.repertoire.load_clonotypes(clonotypes)
    )
    assert peak <= budget, (peak, budget)


def _cluster_peak(
    store: DataStore, receptors: object, options: dict[str, object], monkeypatch
) -> tuple[int, int]:
    """The traced peak of clusters at the smallest budget they accept, and it.

    The search's fixed reservation would hide what clusters hold for their
    cells, contigs, junctions and links, so it shrinks to 1 MiB, with the
    chunks of candidate pairs of cells that it covers.
    """
    for module in (repertoire_operations, clonotypes_module):
        monkeypatch.setattr(module, "CLUSTER_SEARCH_BYTES", 1 << 20)
    monkeypatch.setattr(clonotypes_module, "_PAIRS_PER_CHUNK", 1 << 12)

    def run() -> object:
        return store.repertoire.clonotypes(receptors, invalidate_cache=True, **options)

    def attempt(budget: int) -> object:
        store.memoryBytes = budget
        return run()

    full = store.memoryBytes
    try:
        # Compiled kernels and caches are warmed before anything is traced.
        run()
        # Refusals of the search and of linking name what they hold beside
        # the definition, so the smallest budget is found by bisection.
        budget = smallest_budget(attempt, 1 << 28)
        store.memoryBytes = budget
        peak = traced_peak(run)
    finally:
        store.memoryBytes = full
    return peak, budget


@pytest.mark.parametrize("case", list(CLUSTER_CASES))
def test_clonotype_clusters_fit_the_smallest_budget_accepted(
    data, monkeypatch, case
) -> None:
    peak, budget = _cluster_peak(
        data.store, data.receptors, CLUSTER_CASES[case], monkeypatch
    )
    assert peak <= budget, (case, peak, budget)


@pytest.fixture(scope="module")
def extra_contigs(tmp_path_factory) -> SimpleNamespace:
    """Cells of an alpha and a beta chain and eight contigs that are not chains.

    Every contig has a junction of 45 bases of its own; the import keeps the
    non-productive contigs, which no slot holds, as Cell Ranger's unfiltered
    annotations list them.
    """
    rng = np.random.default_rng(13)
    path = tmp_path_factory.mktemp("extra_contigs") / "store.zarr"
    ids = [f"cell{i:06d}-1" for i in range(N_EXTRA_CELLS)]
    counts = np.ones((N_EXTRA_CELLS, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=np.asarray(ids))
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    residues = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    bases = np.asarray(list("ACGT"))
    rows = []
    for cell, name in enumerate(ids):
        for contig in range(10):
            locus = ("TRA", "TRB")[contig % 2]
            residue_text = "".join(rng.choice(residues, size=15))
            base_text = "".join(rng.choice(bases, size=45))
            productive = "true" if contig < 2 else "false"
            rows.append(
                f"{name},true,{name}_contig_{contig + 1},true,500,{locus},"
                f"{locus}V1,,{locus}J1,{locus}C,true,{productive},{residue_text},"
                f"{base_text},100,{9 if contig < 2 else 2},ct{cell},ct{cell}_{contig + 1}"
            )
    contigs = write_lines(path.parent, "contigs.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=store.snapshot_cell_selection()
    )
    return SimpleNamespace(store=store, receptors=receptors)


def test_clusters_of_cells_with_contigs_beside_their_chains_fit_the_budget(
    extra_contigs, monkeypatch
) -> None:
    # Four in five junctions belong to contigs that no slot holds.
    peak, budget = _cluster_peak(
        extra_contigs.store,
        extra_contigs.receptors,
        {"sequence": "nt", "metric": "hamming", "cutoff": 1},
        monkeypatch,
    )
    assert peak <= budget, (peak, budget)


@pytest.fixture(scope="module")
def wide_junctions(tmp_path_factory) -> SimpleNamespace:
    """Clonotypes named by beta chains of 300 characters beyond the BMP.

    Each cell is a clonotype of its own chains, so every label decodes a
    junction of its own, whose bytes set the labels' smallest budget. Each
    character takes four bytes as stored and four in a Python string.
    """
    rng = np.random.default_rng(17)
    path = tmp_path_factory.mktemp("wide_junctions") / "store.zarr"
    ids = [f"cell{i:06d}-1" for i in range(N_LABELLED)]
    counts = np.ones((N_LABELLED, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=np.asarray(ids))
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    wide = np.asarray([chr(0x1F600 + k) for k in range(20)])
    residues = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    rows = []
    for cell, name in enumerate(ids):
        alpha = "C" + "".join(rng.choice(residues, size=12)) + "F"
        beta = "C" + "".join(rng.choice(wide, size=300)) + "F"
        chains = (("TRA", alpha), ("TRB", beta))
        for contig, (locus, junction) in enumerate(chains):
            rows.append(
                f"{name},true,{name}_contig_{contig + 1},true,500,{locus},"
                f"{locus}V1,,{locus}J1,{locus}C,true,true,{junction},,100,9,"
                f"ct{cell},ct{cell}_{contig + 1}"
            )
    contigs = write_lines(path.parent, "contigs.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=store.snapshot_cell_selection()
    )
    clonotypes = store.repertoire.clonotypes(receptors, sequence="aa")
    return SimpleNamespace(store=store, clonotypes=clonotypes)


def test_labels_of_long_wide_junctions_fit_the_smallest_budget(
    wide_junctions,
) -> None:
    store, clonotypes = wide_junctions.store, wide_junctions.clonotypes
    ids = [f"clonotype_{rank}" for rank in range(1, N_LABELLED + 1)]
    labels = store._clone_figure_labels(clonotypes, ids).tolist()
    assert len(set(labels)) == N_LABELLED
    assert {len(label) for label in labels} == {302}
    peak, budget = _traced_at_smallest_budget(
        store, lambda: store._clone_figure_labels(clonotypes, ids)
    )
    assert peak <= budget, (peak, budget)
