"""Recorder imports, loads and exports stay within the smallest budget they accept.

The smallest budget that each step accepts is found from its refusals,
which state the bytes the step would hold: the search climbs from 1 MiB to
the bytes that each refusal states, or a quarter more where that is more,
then narrows the gap to the last budget refused to within one percent. tracemalloc then records the step's peak at
the smallest budget accepted, which must not pass the largest budget
refused plus one byte, below which no accepted budget falls. The shapes are
those whose costs differ: a wide character matrix of few distinct tokens,
CARLIN alleles each of its own long text, and a long table of a few rows
per cell among many integrations, whose matrix is mostly missing.
"""

import re
import tracemalloc
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.clone_helpers import open_clone_store
from tests.storage_helpers import write_count_store

pytestmark = pytest.mark.slow

_RNG = np.random.default_rng(19)
# Each refusal states the bytes that the step would hold.
_STATED = re.compile(r"holds? about (\d+) bytes")


def _store(folder: Path, cell_ids: list[str]):
    folder.mkdir(parents=True, exist_ok=True)
    write_count_store(
        str(folder / "store.zarr"),
        {"RNA": np.ones((len(cell_ids), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cell_ids,
    )
    store = open_clone_store(folder / "store.zarr")
    return store, store.snapshot_cell_selection()


def _refused(store, run: Callable[[], object], budget: int) -> int | None:
    """The bytes that the step's refusal states at ``budget``, or None if it fits."""
    store.memoryBytes = budget
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run()
    except MemoryError as error:
        found = _STATED.search(str(error))
        assert found, str(error)
        stated = int(found.group(1))
        # A refusal states more than the budget it refuses.
        assert stated > budget, (stated, budget)
        return stated
    return None


def _bounds(store, run: Callable[[], object]) -> tuple[int, int]:
    """The largest budget refused and the smallest accepted, within one percent."""
    low, high = 0, 1 << 20
    # Chunks grow with the budget, and so do the bytes that a refusal
    # states, so the climb takes at least a quarter more at each step.
    while (stated := _refused(store, run, high)) is not None:
        low, high = high, max(stated, high + high // 4)
    assert low, "the step fits 1 MiB, too small a shape to measure"
    while high - low > high // 100:
        middle = (low + high) // 2
        if _refused(store, run, middle) is None:
            high = middle
        else:
            low = middle
    return low, high


def _traced(store, run: Callable[[], object], budget: int) -> int:
    store.memoryBytes = budget
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracemalloc.start()
        try:
            run()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()


def _check(store, run: Callable[[], object]) -> None:
    full = store.memoryBytes
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compiled kernels and caches are warmed before anything is traced.
            run()
        low, high = _bounds(store, run)
        peak = _traced(store, run, high)
    finally:
        store.memoryBytes = full
    # Every accepted budget is above the largest refused one.
    assert peak <= low + 1, (peak, low, high)


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    """PEtracer's characters of 1,000 cells by 100 integrations: 300,000 records."""
    folder = tmp_path_factory.mktemp("wide")
    cells = [f"cell{index:06d}" for index in range(1_000)]
    header = ["cell_barcode"] + [
        f"intID{k}-{site}" for k in range(100) for site in ("EMX1", "HEK3", "RNF2")
    ]
    tokens = _RNG.choice(np.array(list("*-12345678!"), dtype=object), size=(1_000, 300))
    path = folder / "characters.csv"
    path.write_text(
        ",".join(header)
        + "\n"
        + "".join(
            cell + "," + ",".join(row) + "\n"
            for cell, row in zip(cells, tokens.tolist(), strict=True)
        )
    )
    store, cells_ref = _store(folder, cells)
    return store, cells_ref, path, folder


@pytest.fixture(scope="module")
def carlin(tmp_path_factory):
    """A CARLIN pair of 30,000 cells, each allele its own text of about 150 characters."""
    folder = tmp_path_factory.mktemp("carlin")
    cells = sorted({"".join(_RNG.choice(list("ACGT"), 16)) for _ in range(30_000)})
    pair = folder / "pair"
    pair.mkdir()
    (pair / "AlleleAnnotations.txt").write_text(
        "\n".join(
            ",".join(
                f"{10 + 9 * j + cell % 7}_{40 + 9 * j + cell % 5}del" for j in range(12)
            )
            + f",{cell}_{cell}insACGT"
            for cell in range(len(cells))
        )
    )
    (pair / "AlleleColonies.txt").write_text("\n".join(cells))
    store, cells_ref = _store(folder, [f"{cell}-1" for cell in cells])
    return store, cells_ref, pair, folder


@pytest.fixture(scope="module")
def long(tmp_path_factory):
    """PEtracer's long table of 20,000 cells with 4 rows each among 300 integrations."""
    folder = tmp_path_factory.mktemp("long")
    cells = [f"cell{index:06d}" for index in range(20_000)]
    marks = ("ACAGT", "None", "GATAG", "None", "", "GTCAG")
    lines = ["intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount"]
    for index, cell in enumerate(cells):
        for k in range(4):
            lines.append(
                f"intID{(index * 31 + k * 7) % 300},{cell},{marks[k]},"
                f"{marks[(k + 2) % 6]},None,{k + 2},{9 * k + 5}"
            )
    path = folder / "alleles.csv"
    path.write_text("\n".join(lines) + "\n")
    store, cells_ref = _store(folder, cells)
    return store, cells_ref, path, folder


def _import(store, cells, path, **options):
    return store.clones.import_alleles(
        path, cell_selection=cells, invalidate_cache=True, **options
    )


@pytest.mark.parametrize("name", ["wide", "carlin", "long"])
def test_imports_fit_the_smallest_budget_they_accept(request, name) -> None:
    store, cells, path, _folder = request.getfixturevalue(name)
    options = {"integrations": "CA", "barcode_suffix": "-1"} if name == "carlin" else {}
    _check(store, lambda: _import(store, cells, path, **options))


@pytest.mark.parametrize("name", ["wide", "carlin", "long"])
def test_loads_fit_the_smallest_budget_they_accept(request, name) -> None:
    store, cells, path, _folder = request.getfixturevalue(name)
    options = {"integrations": "CA", "barcode_suffix": "-1"} if name == "carlin" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(path, cell_selection=cells, **options)
    _check(store, lambda: store.clones.load_alleles(alleles))


@pytest.mark.parametrize("name", ["wide", "carlin", "long"])
@pytest.mark.parametrize("format", ["cassiopeia", "laml"])
def test_exports_fit_the_smallest_budget_they_accept(request, name, format) -> None:
    store, cells, path, folder = request.getfixturevalue(name)
    options = {"integrations": "CA", "barcode_suffix": "-1"} if name == "carlin" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(path, cell_selection=cells, **options)
    targets = iter(range(1_000))

    def export() -> object:
        suffix = ".zarr" if format == "cassiopeia" else ".csv"
        target = folder / f"export_{format}_{next(targets)}{suffix}"
        return store.clones.export_characters(alleles, target, format=format)

    _check(store, export)


def test_a_budget_too_small_refuses_before_anything_is_written(wide) -> None:
    store, cells, path, _folder = wide
    full = store.memoryBytes
    before = set(store.artifacts.list(kind="lineage_alleles", scope="datastore"))
    try:
        store.memoryBytes = 30_000_000
        with pytest.raises(MemoryError, match="Importing alleles holds about"):
            _import(store, cells, path)
    finally:
        store.memoryBytes = full
    assert (
        set(store.artifacts.list(kind="lineage_alleles", scope="datastore")) == before
    )
