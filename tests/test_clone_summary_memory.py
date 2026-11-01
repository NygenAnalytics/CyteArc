"""Each clone summary runs within the smallest memory budget that it accepts.

The budget climbs from one byte to each refusal's own estimate until the
summary runs, and the peak that tracemalloc records while it runs must not
exceed that budget. The test never repeats the budgets' formulas, so it
checks that whatever a summary estimates bounds what it allocates, on inputs
that stress each estimate: long and emoji labels, long clone labels, a few
measured cells of many clone cells, a label artifact and a cell selection of
more cells than the clones hold, a label for each of many cells, and many
sizes.
"""

import re
import tracemalloc
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from cytearc.datastore.datastore import DataStore
from tests.storage_helpers import write_count_store

pytestmark = pytest.mark.slow

N_CELLS = 10_000
N_MEASURED = 500


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
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    store.cells.insert("first", np.arange(N_CELLS) < N_MEASURED)
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
    every = store.snapshot_cell_selection("every_cell")
    first = store.snapshot_cell_selection("first")
    return SimpleNamespace(
        store=store,
        every=every,
        first=first,
        clones=store.clones.from_column("clone", cell_selection=every),
        long=store.clones.from_column("long_clone", cell_selection=every),
        emoji=store.clones.from_column("emoji_clone", cell_selection=every),
        few=store.clones.from_column("clone", cell_selection=first),
        labels=store.snapshot_cluster_labels("donor", cell_selection=every),
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


@pytest.mark.parametrize("case", list(CASES))
def test_the_traced_peak_fits_the_smallest_budget_accepted(data, case) -> None:
    store, full = data.store, data.store.memoryBytes
    run = CASES[case]
    try:
        # Compiled kernels and caches are warmed before anything is traced.
        run(data)
        budget = _smallest_budget(store, lambda: run(data))
        store.memoryBytes = budget
        tracemalloc.start()
        try:
            run(data)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    finally:
        store.memoryBytes = full
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
