"""Recorder imports, loads and exports stay within the smallest budget they accept.

The smallest budget that each step accepts is found from its refusals,
which state the bytes the step would hold: the search climbs from 1 MiB to
the bytes that each refusal states, or a quarter more where that is more,
then narrows the gap to the last budget refused to within one percent. tracemalloc then records the step's peak at
the smallest budget accepted, which must not pass the largest budget
refused plus one byte, below which no accepted budget falls. The shapes are
those whose costs differ: a wide character matrix of few distinct tokens,
CARLIN alleles each of its own long text, a long table of a few rows per
cell among many integrations, whose matrix is mostly missing, and alleles of
20,000 characters each, ASCII or ending in a character of Latin-1, within
U+FFFF or beyond it, or ASCII but for one allele that starts with an emoji,
whose text outweighs the rest.
"""

import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from cytearc.utils.memory_budget import MemoryBudgetError
from tests.memory_helpers import at_budget, budget_bounds, stated_bytes, traced_peak
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.slow

_RNG = np.random.default_rng(19)
# Each refusal states the bytes that the step would hold.
_stated = partial(stated_bytes, pattern=r"holds? about (\d+) bytes")


def _store(folder: Path, cell_ids: list[str]):
    folder.mkdir(parents=True, exist_ok=True)
    write_count_store(
        str(folder / "store.zarr"),
        {"RNA": np.ones((len(cell_ids), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cell_ids,
    )
    store = open_store(folder / "store.zarr")
    return store, store.snapshot_cell_selection()


def _quietly(run: Callable[[], object]) -> object:
    """Run a step with its warnings ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run()


def _check(store, run: Callable[[], object]) -> None:
    full = store.memoryBytes
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compiled kernels and caches are warmed before anything is traced.
            run()
        # The largest budget refused and the smallest accepted, within one
        # percent: chunks grow with the budget, and so do the bytes that a
        # refusal states, so the climb takes at least a quarter more at each
        # step. Each run recomputes its step: imports invalidate the saved
        # one, and exports write a file of their own.
        low, high = budget_bounds(
            at_budget(store, lambda: _quietly(run)),
            1 << 20,
            parts=100,
            refusal_bytes=_stated,
        )
        store.memoryBytes = high
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, peak = traced_peak(run, low + 1)
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


def _allele_table(folder: Path, alleles: list[str]):
    """A table of one allele per cell, read through columns."""
    cells = [f"cell{index:06d}" for index in range(len(alleles))]
    path = folder / "alleles.tsv"
    path.write_text(
        "cell\tallele\n"
        + "".join(
            f"{cell}\t{text}\n" for cell, text in zip(cells, alleles, strict=True)
        ),
        encoding="utf-8",
    )
    store, cells_ref = _store(folder, cells)
    return store, cells_ref, path, folder


@pytest.fixture(scope="module")
def long_text(tmp_path_factory):
    """1,000 cells, each allele its own ASCII text of 20,000 characters.

    The alleles' text, 20 MB, outweighs what an import, a load and an export
    hold whatever their size, so each step's bytes per byte of text set its
    budget.
    """
    letters = np.array(list("ACGT"))
    alleles = [
        "".join(_RNG.choice(letters, 19_988)) + f"{cell:012d}" for cell in range(1_000)
    ]
    return _allele_table(tmp_path_factory.mktemp("long_text"), alleles)


@pytest.fixture(scope="module")
def wide_text(tmp_path_factory):
    """400 cells, each allele 20,000 characters that end in an emoji.

    One character beyond U+FFFF makes Python hold each of the allele's
    characters in four bytes, four times the bytes that UTF-8 stores.
    """
    letters = np.array(list("ACGT"))
    alleles = [
        "".join(_RNG.choice(letters, 19_987)) + f"{cell:012d}😀" for cell in range(400)
    ]
    return _allele_table(tmp_path_factory.mktemp("wide_text"), alleles)


@pytest.fixture(scope="module")
def latin_text(tmp_path_factory):
    """1,000 cells, each allele 20,000 characters that end in é.

    A character beyond ASCII up to U+00FF leaves Python's strings at one byte
    per character, but each string caches its UTF-8 as merging encodes it.
    """
    letters = np.array(list("ACGT"))
    alleles = [
        "".join(_RNG.choice(letters, 19_987)) + f"{cell:012d}é" for cell in range(1_000)
    ]
    return _allele_table(tmp_path_factory.mktemp("latin_text"), alleles)


@pytest.fixture(scope="module")
def plane_text(tmp_path_factory):
    """600 cells, each allele 20,000 characters that end in 語.

    A character beyond U+00FF and within U+FFFF makes Python hold each of the
    allele's characters in two bytes.
    """
    letters = np.array(list("ACGT"))
    alleles = [
        "".join(_RNG.choice(letters, 19_987)) + f"{cell:012d}語" for cell in range(600)
    ]
    return _allele_table(tmp_path_factory.mktemp("plane_text"), alleles)


@pytest.fixture(scope="module")
def one_wide(tmp_path_factory):
    """1,000 ASCII alleles of 20,000 characters and one that starts with an emoji.

    The emoji's allele sorts last, so only the last chunks of the stored text
    hold a byte from 0xF0, and only the last chunk of an import's file. Yet
    decoding the text, and joining it as an import merges, widens every
    character to four bytes.
    """
    letters = np.array(list("ACGT"))
    alleles = [
        "".join(_RNG.choice(letters, 19_988)) + f"{cell:012d}" for cell in range(1_000)
    ]
    alleles.append("😀" + "".join(_RNG.choice(letters, 19_999)))
    return _allele_table(tmp_path_factory.mktemp("one_wide"), alleles)


_TEXT_OPTIONS = {"columns": {"cell": "cell", "allele": "allele"}, "integrations": "I"}
_SHAPES = [
    "wide",
    "carlin",
    "long",
    "long_text",
    "latin_text",
    "plane_text",
    "wide_text",
    "one_wide",
]
_OPTIONS = {
    "carlin": {"integrations": "CA", "barcode_suffix": "-1"},
    "long_text": _TEXT_OPTIONS,
    "latin_text": _TEXT_OPTIONS,
    "plane_text": _TEXT_OPTIONS,
    "wide_text": _TEXT_OPTIONS,
    "one_wide": _TEXT_OPTIONS,
}


def _import(store, cells, path, **options):
    return store.clones.import_alleles(
        path, cell_selection=cells, invalidate_cache=True, **options
    )


@pytest.mark.parametrize("name", _SHAPES)
def test_imports_fit_the_smallest_budget_they_accept(request, name) -> None:
    store, cells, path, _folder = request.getfixturevalue(name)
    options = _OPTIONS.get(name, {})
    _check(store, lambda: _import(store, cells, path, **options))


@pytest.mark.parametrize("name", _SHAPES)
def test_loads_fit_the_smallest_budget_they_accept(request, name) -> None:
    store, cells, path, _folder = request.getfixturevalue(name)
    options = _OPTIONS.get(name, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(path, cell_selection=cells, **options)
    _check(store, lambda: store.clones.load_alleles(alleles))


@pytest.mark.parametrize("name", _SHAPES)
@pytest.mark.parametrize("format", ["treedata", "laml"])
def test_exports_fit_the_smallest_budget_they_accept(request, name, format) -> None:
    store, cells, path, folder = request.getfixturevalue(name)
    options = _OPTIONS.get(name, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(path, cell_selection=cells, **options)
    targets = iter(range(1_000))

    def export() -> object:
        suffix = ".zarr" if format == "treedata" else ".csv"
        target = folder / f"export_{format}_{next(targets)}{suffix}"
        return store.clones.export_characters(alleles, target, format=format)

    _check(store, export)


def test_a_budget_too_small_refuses_before_anything_is_written(wide) -> None:
    store, cells, path, _folder = wide
    full = store.memoryBytes
    before = set(store.artifacts.list(kind="lineage_alleles", scope="datastore"))
    try:
        store.memoryBytes = 30_000_000
        with pytest.raises(
            MemoryBudgetError,
            match="Importing alleles for 1000 selected cells holds about",
        ):
            _import(store, cells, path)
    finally:
        store.memoryBytes = full
    assert (
        set(store.artifacts.list(kind="lineage_alleles", scope="datastore")) == before
    )
