"""Joining the barcodes of imported files to the cells of a store.

Every importer joins its files to cells through one helper, so the same cases
run on each: prefixes, suffixes, patterns, barcodes that match no cell, the
warning repeated on reuse, the order of files and files listed twice. Each
importer is a row of ``IMPORTERS``; a later importer adds a row.
"""

import gc
import hashlib
import json
import os
import re
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytearc.datastore._operations.cell_barcodes import (
    UNMATCHED_EXAMPLES,
    BarcodeFile,
    CellBarcodeJoin,
    barcode_files,
    barcode_pattern,
    derived_id_bytes,
    join_bytes,
    unmatched_barcodes_warning,
)
from cytearc.utils.logging import collect_warnings, logger
from tests.barcode_join_helpers import (
    CASES,
    build_guide_store,
    build_join_store,
    import_case,
    write_clone_file,
    write_guide_call_file,
)

_REFERENCE = Path(__file__).parent / "guide_calls_import_reference.json"
# Attributes that differ between two writes of one artifact.
_VOLATILE = {"artifact_id", "created_at_ns", "cytearc_version"}


def _stored(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.dtype.kind in "OUS":
        return {
            "dtype": "text",
            "shape": list(values.shape),
            "values": values.astype(str).tolist(),
        }
    flat = values.reshape(-1)
    if values.dtype.kind == "f":
        listed = [None if np.isnan(value) else float(value) for value in flat]
    else:
        listed = flat.tolist()
    return {"dtype": values.dtype.str, "shape": list(values.shape), "values": listed}


def test_imported_guide_calls_match_those_saved_before_the_shared_join(
    tmp_path,
) -> None:
    # The reference was written by import_guide_calls before it joined
    # barcodes through the shared helper; every array and attribute must
    # come out the same.
    reference = json.loads(_REFERENCE.read_text())
    store, selection = build_guide_store(tmp_path / "store.zarr")
    assert set(reference) == set(CASES)
    for name in CASES:
        # Each case's file names a barcode outside the selection: a cell of
        # the store, which the import logs, or no cell, which it warns about.
        with collect_warnings():
            ref = import_case(store, selection, name, tmp_path)
        group = store.artifacts.load(ref)
        attributes = {
            key: group.attrs[key] for key in group.attrs if key not in _VOLATILE
        }
        attributes["provenance"]["inputs"]["cell_selection"].pop("artifact_id")
        expected = reference[name]
        assert store.artifacts.inspect(ref).parameters == expected["parameters"]
        assert attributes == expected["attributes"]
        assert {key: _stored(group[key][:]) for key in group.array_keys()} == expected[
            "arrays"
        ]


def _shared_warning(
    files: str, n_unmatched: int, n_barcodes: int, examples: list[str], options: bool
) -> str:
    """The shared join's warning about barcodes that match no selected cell."""
    return (
        f"{n_unmatched} of {n_barcodes} barcodes of {files} match no selected "
        "cell, for example "
        + ", ".join(repr(value) for value in examples)
        + ". Check barcode_prefix and cell_selection."
    )


def _guide_call_warning(
    files: str, n_unmatched: int, n_barcodes: int, examples: list[str], options: bool
) -> str:
    """Guide calls warn about barcodes that match no cell of the store.

    Barcodes of cells outside the selection are logged instead, and with a
    suffix or a pattern the warning names every option.
    """
    rule = (
        "A cell matches a barcode when its ID is barcode_prefix, the barcode or "
        "the part of it that barcode_pattern keeps, and barcode_suffix, so check "
        "these options against the cell IDs."
        if options
        else "A cell matches a barcode when its ID is barcode_prefix followed by "
        "the barcode, so check barcode_prefix when the cell IDs carry a sample "
        "prefix."
    )
    return (
        f"{n_unmatched} of {n_barcodes} barcodes of {files} match no cell of the "
        "store, for example " + ", ".join(repr(value) for value in examples) + ". "
        f"{rule}"
    )


@dataclass(frozen=True)
class Importer:
    """How the join cases import files with one importer and read what joined.

    Attributes:
        name: The importer.
        several_files: Whether it imports several files at once.
        write: Writes a file whose rows name the given barcodes.
        run: Imports paths for a cell selection with join options.
        joined: The IDs of the cells that the import's rows named.
        files: How the warning names one file and several.
        warning: The warning about barcodes that match no cell, from the
            files' name, the counts, the examples and whether a suffix or a
            pattern was given.
    """

    name: str
    several_files: bool
    write: Callable[[Path, list[str]], Path]
    run: Callable[..., Any]
    joined: Callable[[Any, Any], set[str]]
    files: tuple[str, str]
    warning: Callable[[str, int, int, list[str], bool], str] = _shared_warning


def _guide_calls(store, paths, **options):
    (path,) = paths
    return store.perturbation.import_guide_calls(path, **options)


def _called_cells(store, ref) -> set[str]:
    cells = store.perturbation.load_guide_calls(ref).cells
    return set(cells.index[cells["n_guides"] > 0])


def _clones(store, paths, **options):
    return store.clones.import_clones(paths, **options)


def _cloned_cells(store, ref) -> set[str]:
    cells = store.clones.load_clones(ref).cells
    return set(cells.index[cells["clone"].notna()])


IMPORTERS = {
    "import_guide_calls": Importer(
        name="import_guide_calls",
        several_files=False,
        write=write_guide_call_file,
        run=_guide_calls,
        joined=_called_cells,
        files=("the call file", "the call file"),
        warning=_guide_call_warning,
    ),
    "import_clones": Importer(
        name="import_clones",
        several_files=True,
        write=write_clone_file,
        run=_clones,
        joined=_cloned_cells,
        files=("the clone file", "the clone files"),
    ),
}
MULTI_FILE = [name for name, importer in IMPORTERS.items() if importer.several_files]


@pytest.fixture(scope="module")
def join_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("join") / "store.zarr"
    build_join_store(path)
    return path


@pytest.fixture
def joined(join_template, tmp_path):
    """A copy of the join store and its selection of every cell."""
    import shutil

    from tests.perturbation_helpers import open_store

    target = tmp_path / "store.zarr"
    shutil.copytree(join_template, target)
    store = open_store(target)
    return store, store.snapshot_cell_selection("joined")


def _attributes(store, ref) -> dict:
    attributes = store.artifacts.load(ref).attrs
    return {
        name: attributes[name]
        for name in (
            "n_file_barcodes",
            "n_unmatched_barcodes",
            "unmatched_barcode_examples",
        )
    }


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_a_prefix_and_a_suffix_name_cells_by_bare_barcodes(
    joined, tmp_path, name
) -> None:
    store, _ = joined
    importer = IMPORTERS[name]
    path = importer.write(tmp_path / "file.txt", ["AAAC", "CCCG", "TTTA"])

    # A file's barcodes reach one sample's cells.
    with collect_warnings() as messages:
        ref = importer.run(
            store,
            [path],
            cell_selection=store.snapshot_cell_selection("s1"),
            barcode_prefix="s1__",
            barcode_suffix="-1",
        )

    assert importer.joined(store, ref) == {"s1__AAAC-1", "s1__CCCG-1", "s1__TTTA-1"}
    assert _attributes(store, ref) == {
        "n_file_barcodes": 3,
        "n_unmatched_barcodes": 0,
        "unmatched_barcode_examples": [],
    }
    assert list(messages) == []
    parameters = store.artifacts.inspect(ref).parameters
    recorded = (
        parameters
        if name == "import_guide_calls"
        else (store.artifacts.inspect(ref).inputs["sources"][0])
    )
    assert recorded["barcode_suffix"] == "-1"


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_a_pattern_keeps_the_barcode_group_and_others_match_no_cell(
    joined, tmp_path, name
) -> None:
    store, _ = joined
    importer = IMPORTERS[name]
    path = importer.write(
        tmp_path / "file.txt", ["lane1:AAAC:x", "lane1:GGGT:y", "AAAC", "lane1::z"]
    )

    with collect_warnings() as messages:
        ref = importer.run(
            store,
            [path],
            cell_selection=store.snapshot_cell_selection("s2"),
            barcode_prefix="s2__",
            barcode_suffix="-1",
            barcode_pattern=r"^lane1:(?P<barcode>[ACGT]+):",
        )

    # "AAAC" and "lane1::z" do not match the pattern and name no cell.
    assert importer.joined(store, ref) == {"s2__AAAC-1", "s2__GGGT-1"}
    assert _attributes(store, ref) == {
        "n_file_barcodes": 4,
        "n_unmatched_barcodes": 2,
        "unmatched_barcode_examples": ["AAAC", "lane1::z"],
    }
    assert list(messages) == [
        importer.warning(importer.files[0], 2, 4, ["AAAC", "lane1::z"], True)
    ]


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_a_pattern_finds_the_barcode_anywhere_in_a_barcode(
    joined, tmp_path, name
) -> None:
    # The pattern is searched for, as re.search does, rather than matched
    # from the first character.
    store, _ = joined
    importer = IMPORTERS[name]
    path = importer.write(tmp_path / "file.txt", ["x_AAAC-1", "lane9_GGGT-1"])

    with collect_warnings() as messages:
        ref = importer.run(
            store,
            [path],
            cell_selection=store.snapshot_cell_selection("s2"),
            barcode_prefix="s2__",
            barcode_suffix="-1",
            barcode_pattern=r"_(?P<barcode>[ACGT]{4})-1$",
        )

    assert importer.joined(store, ref) == {"s2__AAAC-1", "s2__GGGT-1"}
    assert list(messages) == []


def test_guide_calls_take_an_empty_suffix_for_none(joined, tmp_path) -> None:
    # An empty prefix or suffix adds nothing, as None does, and the identity
    # of calls imported without a suffix or a pattern omits both.
    store, _ = joined
    cells = store.snapshot_cell_selection("s2")
    path = write_guide_call_file(tmp_path / "calls.csv", ["s2__GGGT-1"])

    empty = store.perturbation.import_guide_calls(
        path, cell_selection=cells, barcode_prefix="", barcode_suffix=""
    )
    none = store.perturbation.import_guide_calls(path, cell_selection=cells)

    assert empty == none
    assert store.artifacts.inspect(none).parameters == {"barcode_prefix": None}
    suffixed = store.perturbation.import_guide_calls(
        write_guide_call_file(tmp_path / "bare.csv", ["s2__GGGT"]),
        cell_selection=cells,
        barcode_suffix="-1",
    )
    assert store.artifacts.inspect(suffixed).parameters == {
        "barcode_prefix": None,
        "barcode_suffix": "-1",
    }


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_unmatched_barcodes_are_counted_and_warned_again_on_reuse(
    joined, tmp_path, name
) -> None:
    store, _ = joined
    # Without a prefix, a file's barcodes reach the cells of one sample.
    cells = store.snapshot_cell_selection("s2")
    importer = IMPORTERS[name]
    outside = [f"x{index}" for index in range(7)]
    path = importer.write(tmp_path / "file.txt", ["s2__GGGT-1", *outside])

    with collect_warnings() as first:
        ref = importer.run(store, [path], cell_selection=cells)
    with collect_warnings() as reused:
        assert importer.run(store, [path], cell_selection=cells) == ref
    from tests.perturbation_helpers import open_store

    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    with collect_warnings() as read_only:
        assert importer.run(reader, [path], cell_selection=cells) == ref

    assert _attributes(store, ref) == {
        "n_file_barcodes": 8,
        "n_unmatched_barcodes": 7,
        "unmatched_barcode_examples": outside[:UNMATCHED_EXAMPLES],
    }
    expected = [
        importer.warning(importer.files[0], 7, 8, outside[:UNMATCHED_EXAMPLES], False)
    ]
    assert list(first) == list(reused) == list(read_only) == expected


@pytest.mark.parametrize("name", sorted(IMPORTERS))
@pytest.mark.parametrize(
    ("pattern", "error", "message"),
    [
        (r"^(?P<cell>[ACGT]+)", ValueError, "has no named group 'barcode'"),
        (r"(?P<barcode>[ACGT", ValueError, "not a valid regular expression"),
        (3, TypeError, "barcode_pattern must be a regular expression"),
    ],
)
def test_a_pattern_must_name_the_barcode(
    joined, tmp_path, name, pattern, error, message
) -> None:
    store, cells = joined
    importer = IMPORTERS[name]
    path = importer.write(tmp_path / "file.txt", ["AAAC"])
    with pytest.raises(error, match=message):
        importer.run(store, [path], cell_selection=cells, barcode_pattern=pattern)


@pytest.mark.parametrize("name", MULTI_FILE)
def test_each_file_takes_its_own_prefix(joined, tmp_path, name) -> None:
    store, cells = joined
    importer = IMPORTERS[name]
    first = importer.write(tmp_path / "s1.txt", ["AAAC", "CCCG"])
    second = importer.write(tmp_path / "s2.txt", ["AAAC", "GGGT"])

    with collect_warnings() as messages:
        ref = importer.run(
            store,
            [first, second],
            cell_selection=cells,
            barcode_prefix=["s1__", "s2__"],
            barcode_suffix="-1",
        )

    # The two files' equal barcode AAAC names two cells.
    assert importer.joined(store, ref) == {
        "s1__AAAC-1",
        "s1__CCCG-1",
        "s2__AAAC-1",
        "s2__GGGT-1",
    }
    assert _attributes(store, ref)["n_file_barcodes"] == 4
    assert list(messages) == []


@pytest.mark.parametrize("name", MULTI_FILE)
def test_files_in_another_order_give_the_same_artifact(joined, tmp_path, name) -> None:
    store, cells = joined
    importer = IMPORTERS[name]
    first = importer.write(tmp_path / "s1.txt", ["AAAC", "CCCG"])
    second = importer.write(tmp_path / "s2.txt", ["AAAC", "GGGT"])
    options = {"cell_selection": cells, "barcode_suffix": "-1"}

    ref = importer.run(
        store, [first, second], barcode_prefix=["s1__", "s2__"], **options
    )

    assert (
        importer.run(store, [second, first], barcode_prefix=["s2__", "s1__"], **options)
        == ref
    )
    sources = store.artifacts.inspect(ref).inputs["sources"]
    digests = [source["sha256"] for source in sources]
    assert digests == sorted(digests)
    assert digests == sorted(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)
    )


@pytest.mark.parametrize("name", MULTI_FILE)
def test_a_file_listed_twice_with_the_same_prefix_and_suffix_is_refused(
    joined, tmp_path, name
) -> None:
    store, cells = joined
    importer = IMPORTERS[name]
    path = importer.write(tmp_path / "s.txt", ["AAAC"])
    copy = tmp_path / "copy.txt"
    copy.write_bytes(path.read_bytes())

    with pytest.raises(ValueError, match="listed twice with the same barcode_prefix"):
        importer.run(store, [path, copy], cell_selection=cells, barcode_suffix="-1")
    # The same bytes under two prefixes name two samples' cells.
    ref = importer.run(
        store,
        [path, copy],
        cell_selection=cells,
        barcode_prefix=["s1__", "s2__"],
        barcode_suffix="-1",
    )
    assert importer.joined(store, ref) == {"s1__AAAC-1", "s2__AAAC-1"}
    with pytest.raises(ValueError, match="lists 3 values for 2 clone files"):
        importer.run(
            store, [path, copy], cell_selection=cells, barcode_prefix=["a", "b", "c"]
        )


def test_guide_calls_refuse_two_barcodes_that_name_one_cell(joined, tmp_path) -> None:
    store, _ = joined
    cells = store.snapshot_cell_selection("s1")
    path = write_guide_call_file(tmp_path / "calls.csv", ["AAAC-1", "AAAC-2"])
    with pytest.raises(
        ValueError,
        match="'AAAC-1' and 'AAAC-2' of the call file name one cell, 's1__AAAC-1'",
    ):
        store.perturbation.import_guide_calls(
            path,
            cell_selection=cells,
            barcode_prefix="s1__",
            barcode_suffix="-1",
            barcode_pattern=r"^(?P<barcode>[ACGT]+)-",
        )
    for option in ("barcode_prefix", "barcode_suffix"):
        with pytest.raises(TypeError, match=f"{option} must be text or None"):
            store.perturbation.import_guide_calls(
                path, cell_selection=cells, **{option: ["s1__"]}
            )


def _logged_guide_calls(store, path, **options) -> tuple[Any, list[str], list[str]]:
    """Import guide calls; return them, the warnings, and the notes on unselected cells."""
    notes: list[str] = []
    sink = logger.add(
        lambda message: notes.append(message.record["message"]),
        level="INFO",
        filter=lambda record: "outside the selected cells" in record["message"],
    )
    try:
        with collect_warnings() as messages:
            calls = store.perturbation.import_guide_calls(path, **options)
    finally:
        logger.remove(sink)
    return calls, list(messages), notes


def test_guide_calls_find_cells_outside_the_selection_through_the_options(
    joined, tmp_path
) -> None:
    store, _ = joined
    # QC kept two of the first sample's cells, so s1__TTTA-1 is a cell of
    # the store outside the selection.
    store.cells.insert("kept", np.asarray([True, True, False, False, False]))
    path = write_guide_call_file(
        tmp_path / "calls.csv",
        ["a:AAAC:1", "b:TTTA:1", "c:TTTA:2", "AAAC", "d:GGGG:1"],
    )
    options = {
        "cell_selection": store.snapshot_cell_selection("kept"),
        "barcode_prefix": "s1__",
        "barcode_suffix": "-1",
        "barcode_pattern": r"^[a-d]:(?P<barcode>[ACGT]+):",
    }

    calls, messages, notes = _logged_guide_calls(store, path, **options)

    # Two barcodes name s1__TTTA-1 through the pattern, which is no error for
    # a cell outside the selection; AAAC does not match the pattern, and no
    # cell is named s1__GGGG-1.
    assert _called_cells(store, calls) == {"s1__AAAC-1"}
    attributes = store.artifacts.load(calls).attrs
    assert {
        name: attributes[name]
        for name in (
            "n_file_barcodes",
            "n_unmatched_barcodes",
            "unmatched_barcode_examples",
            "n_unselected_barcodes",
            "unselected_barcode_examples",
        )
    } == {
        "n_file_barcodes": 5,
        "n_unmatched_barcodes": 2,
        "unmatched_barcode_examples": ["AAAC", "d:GGGG:1"],
        "n_unselected_barcodes": 2,
        "unselected_barcode_examples": ["b:TTTA:1", "c:TTTA:2"],
    }
    assert messages == [
        _guide_call_warning("the call file", 2, 5, ["AAAC", "d:GGGG:1"], True)
    ]
    assert notes == [
        "2 of 5 barcodes of the call file are cells of the store outside the "
        "selected cells, for example 'b:TTTA:1', 'c:TTTA:2'; the import labels "
        "the selected cells only"
    ]
    # A reused import reports both again.
    assert _logged_guide_calls(store, path, **options) == (calls, messages, notes)


def test_guide_calls_refuse_cells_that_the_suffix_cannot_reach(tmp_path) -> None:
    from tests.perturbation_helpers import open_store, written
    from tests.storage_helpers import write_count_store

    # A 10x aggregate names the cells of its second library -2.
    ids = ("AAAC-1", "CCCG-1", "GGGT-2")
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((3, 2)), "CRISPR": np.full((3, 1), 5)},
        "uint16",
        cell_ids=ids,
    )
    store = open_store(tmp_path / "store.zarr")
    store.cells.insert("every", np.ones(3, dtype=bool))
    store.cells.insert("first", np.asarray([True, True, False]))
    every = store.snapshot_cell_selection("every")
    first = store.snapshot_cell_selection("first")
    path = write_guide_call_file(tmp_path / "calls.csv", ["AAAC", "CCCG"])
    saved = written(store)

    # GGGT-2 would be labelled Negative, since no barcode can name it.
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            "The call file's barcodes reach only the cells whose IDs end with "
            "barcode_suffix '-1', but 1 of the cells of cell_selection, such as "
            "'GGGT-2', does not; select only the cells of that sample, or call the "
            "guides from their counts with perturbation.call_guides"
        )
        + "$",
    ):
        store.perturbation.import_guide_calls(
            path, cell_selection=every, barcode_suffix="-1"
        )
    assert written(store) == saved
    calls = store.perturbation.import_guide_calls(
        path, cell_selection=first, barcode_suffix="-1"
    )
    assert _called_cells(store, calls) == {"AAAC-1", "CCCG-1"}


@pytest.mark.slow
def test_guide_calls_with_every_option_run_within_the_budget_that_their_refusal_names(
    tmp_path,
) -> None:
    from tests.perturbation_helpers import check_budget, open_store
    from tests.storage_helpers import write_count_store

    # 3,000 cells of one sample with 40-letter barcodes; the file names 2,400
    # selected cells, 300 cells outside the selection and 300 barcodes of no
    # cell, each wrapped in text that the pattern removes.
    n_cells = 3_000
    rng = np.random.default_rng(5)
    letters = np.asarray(list("ACGT"))
    barcodes = sorted({"".join(rng.choice(letters, 40)) for _ in range(n_cells + 300)})
    ids = [f"s1__{barcode}-1" for barcode in barcodes[:n_cells]]
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path),
        {
            "RNA": rng.poisson(2.0, size=(n_cells, 3)),
            "CRISPR": rng.poisson(3.0, size=(n_cells, 4)),
        },
        "uint16",
        cell_ids=ids,
    )
    store = open_store(path)
    store.cells.insert("kept", np.arange(n_cells) < 2_700)
    kept = store.snapshot_cell_selection("kept")
    named = barcodes[:2_400] + barcodes[2_700:3_300]
    calls = write_guide_call_file(
        tmp_path / "calls.csv", [f"lane1:{barcode}:x" for barcode in named]
    )

    budget, refusals, _ = check_budget(
        path,
        lambda opened: opened.perturbation.import_guide_calls(
            calls,
            cell_selection=kept,
            barcode_prefix="s1__",
            barcode_suffix="-1",
            barcode_pattern=r"^lane1:(?P<barcode>[ACGT]+):",
            invalidate_cache=True,
        ),
    )

    # One refusal, before the file is parsed, names the budget that runs.
    assert len(refusals) == 1
    assert refusals[0].startswith("Importing the guide calls of 2700 cells holds about")
    reference = open_store(path).perturbation.load_guide_calls(
        open_store(path).perturbation.import_guide_calls(
            calls,
            cell_selection=kept,
            barcode_prefix="s1__",
            barcode_suffix="-1",
            barcode_pattern=r"^lane1:(?P<barcode>[ACGT]+):",
        )
    )
    assert int((reference.cells["n_guides"] > 0).sum()) == 2_400


def _file(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_files_order_by_their_bytes_then_their_options(tmp_path) -> None:
    paths = [_file(tmp_path, f"{index}.txt", f"file {index}\n") for index in range(4)]
    digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    files = barcode_files(paths[::-1], barcode_prefix=["d", "c", "b", "a"])

    assert [file.sha256 for file in files] == sorted(digests.values())
    assert {os.fspath(file.path): file.barcode_prefix for file in files} == {
        os.fspath(path): prefix
        for path, prefix in zip(paths[::-1], "dcba", strict=True)
    }
    same = barcode_files(
        [paths[0], paths[0]],
        barcode_prefix=["b", "a"],
        barcode_pattern=["(?P<barcode>.)", None],
    )
    assert [file.barcode_prefix for file in same] == ["a", "b"]
    assert same[0].record() == {
        "sha256": digests[paths[0]],
        "barcode_prefix": "a",
        "barcode_suffix": None,
        "barcode_pattern": None,
    }


def test_an_empty_prefix_adds_nothing_and_identifies_the_file_alike(tmp_path) -> None:
    path = _file(tmp_path, "a.txt", "x\n")
    (empty,) = barcode_files(path, barcode_prefix="", barcode_suffix="")
    (none,) = barcode_files(path)
    assert empty.record() == none.record()
    with pytest.raises(ValueError, match="listed twice"):
        barcode_files([path, path], barcode_prefix=["", None])


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"paths": 3}, TypeError, "paths must be a path or a sequence of paths"),
        ({"paths": [3]}, TypeError, "paths must be a path or a sequence of paths"),
        ({"paths": b"a.txt"}, TypeError, "paths must be a path"),
        ({"paths": []}, ValueError, "paths must name at least one file"),
        ({"barcode_prefix": 3}, TypeError, "barcode_prefix must be text"),
        ({"barcode_prefix": [3]}, TypeError, "barcode_prefix must be text"),
        ({"barcode_suffix": ["a", "b"]}, ValueError, "lists 2 values for 1 files"),
        ({"barcode_pattern": [3]}, TypeError, "barcode_pattern must be a regular"),
    ],
)
def test_barcode_files_refuse_arguments_they_cannot_use(
    tmp_path, arguments, error, message
) -> None:
    call = {"paths": _file(tmp_path, "a.txt", "x\n"), **arguments}
    with pytest.raises(error, match=message):
        barcode_files(call.pop("paths"), **call)
    with pytest.raises(FileNotFoundError):
        barcode_files(tmp_path / "absent.txt")


def _plain(path: str = "f", **options) -> BarcodeFile:
    return BarcodeFile(
        path=path,
        sha256="0" * 64,
        barcode_prefix=options.get("prefix"),
        barcode_suffix=options.get("suffix"),
        barcode_pattern=options.get("pattern"),
    )


def test_a_pattern_is_text_with_a_barcode_group() -> None:
    assert barcode_pattern(None) is None
    assert barcode_pattern("(?P<barcode>.+)-1") == "(?P<barcode>.+)-1"
    with pytest.raises(
        TypeError, match="barcode_pattern must be a regular expression or None"
    ):
        barcode_pattern(3)


def test_a_file_derives_the_cell_id_of_each_barcode() -> None:
    barcodes = np.asarray(["x:AAAC:1", "AAAC", "x::2", "y:CCCG:3"], dtype=object)

    ids, named = _plain(
        prefix="s1__", suffix="-1", pattern=r"^x:(?P<barcode>[ACGT]+):"
    ).cell_ids(barcodes)

    # Barcodes that the pattern does not match name no cell; their IDs keep
    # the whole barcode, so that they can still be counted.
    assert ids.tolist() == ["s1__AAAC-1", "s1__AAAC-1", "s1__x::2-1", "s1__y:CCCG:3-1"]
    assert named.tolist() == [True, False, False, False]
    plain_ids, plain_named = _plain().cell_ids(barcodes)
    assert plain_ids.tolist() == barcodes.tolist()
    assert plain_named.all()


_JOIN_OPTIONS = {
    "none": ({}, "s1__{}-1"),
    "prefix": ({"prefix": "s1__"}, "{}-1"),
    "suffix": ({"suffix": "-1"}, "s1__{}"),
    "prefix and suffix": ({"prefix": "s1__", "suffix": "-1"}, "{}"),
    "pattern": ({"pattern": r"^x_(?P<barcode>.+)$"}, "x_s1__{}-1"),
    "every option": (
        {"prefix": "s1__", "suffix": "-1", "pattern": r"^x_(?P<barcode>[ACGT]+)_y$"},
        "x_{}_y",
    ),
}


def _traced(call: Callable[[], Any]) -> int:
    """Return the most bytes that ``call`` held beside what was held before it."""
    gc.collect()
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        call()
        return tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("name", sorted(_JOIN_OPTIONS))
def test_the_join_holds_no_more_than_its_estimate(name, matched) -> None:
    # 20,000 barcodes of 120 letters, every one or none of them a selected
    # cell; barcodes that match no cell are hashed and counted, which holds
    # the most.
    options, layout = _JOIN_OPTIONS[name]
    cores = [
        format(index, "020b").replace("0", "A").replace("1", "C") + "G" * 100
        for index in range(20_000)
    ]
    barcodes = np.asarray([layout.format(core) for core in cores], dtype=object)
    cell_ids = np.asarray(
        [f"s1__{core}-1" if matched else f"other{core}" for core in cores],
        dtype=object,
    )
    file = _plain(**options)
    join = CellBarcodeJoin(cell_ids, memory_bytes=10**12)
    # The lookup's table, which held_bytes counts, is built on the first use.
    join.cells(file, barcodes[:1])
    n_chars = sum(map(len, barcodes))

    peak = _traced(lambda: join.cells(file, barcodes))
    derived = _traced(lambda: file.cell_ids(barcodes))

    assert (join.n_unmatched_rows == 0) == matched
    assert peak <= join_bytes(file, len(barcodes), n_chars)
    assert derived <= derived_id_bytes(file, len(barcodes), n_chars)


def test_the_join_counts_distinct_barcodes_and_rows() -> None:
    join = CellBarcodeJoin(np.asarray(["a-1", "b-1", "c-1"]), memory_bytes=10**6)

    first = join.cells(
        _plain(suffix="-1"), np.asarray(["a", "b", "z", "z", "a"], dtype=object)
    )
    second = join.cells(
        _plain(suffix="-1"), np.asarray(["y", "z", "x", "w", "v", "u"], dtype=object)
    )

    assert first.tolist() == [0, 1, -1, -1, 0]
    assert second.tolist() == [-1] * 6
    assert (join.n_rows, join.n_unmatched_rows) == (11, 8)
    # a and b, and z, y, x, w, v, u by their cell IDs.
    assert (join.n_file_barcodes, join.n_unmatched_barcodes) == (8, 6)
    assert join.examples == ["z", "y", "x", "w", "v"]
    assert join.attributes() == {
        "n_file_barcodes": 8,
        "n_unmatched_barcodes": 6,
        "unmatched_barcode_examples": ["z", "y", "x", "w", "v"],
    }


def test_one_barcode_under_two_prefixes_counts_as_two() -> None:
    join = CellBarcodeJoin(np.asarray(["s1__a"]), memory_bytes=10**6)
    join.cells(_plain(prefix="s1__"), np.asarray(["a", "b"], dtype=object))
    join.cells(_plain(prefix="s2__"), np.asarray(["a", "b"], dtype=object))
    assert (join.n_file_barcodes, join.n_unmatched_barcodes) == (4, 3)
    # The examples are barcodes as the files hold them, each once.
    assert join.examples == ["b", "a"]


def test_no_barcode_gives_no_count() -> None:
    join = CellBarcodeJoin(np.asarray(["a"]), memory_bytes=10**6)
    assert (
        join.cells(
            _plain(pattern="(?P<barcode>.)"), np.asarray([], dtype=object)
        ).tolist()
        == []
    )
    assert (join.n_file_barcodes, join.n_rows) == (0, 0)
    assert unmatched_barcodes_warning(join.attributes(), "the file") is None


def test_the_join_refuses_repeated_cell_ids_and_budgets_too_small() -> None:
    with pytest.raises(ValueError, match="cells' IDs repeat"):
        CellBarcodeJoin(np.asarray(["a", "a"]), memory_bytes=10**6)
    with pytest.raises(MemoryError, match="memory budget of 500 bytes"):
        CellBarcodeJoin(
            np.asarray(["a"] * 0 + ["a", "b", "c"]), memory_bytes=500, held_bytes=300
        )
    join = CellBarcodeJoin(np.asarray(["a"]), memory_bytes=200)
    with pytest.raises(
        MemoryError,
        match="Check barcode_prefix, barcode_suffix, barcode_pattern and cell_selection",
    ):
        join.cells(
            _plain(), np.asarray([f"x{index}" for index in range(20)], dtype=object)
        )


def test_the_join_counts_the_text_of_long_cell_ids() -> None:
    # One ID of 10,000 characters holds more than the budget by itself.
    with pytest.raises(MemoryError, match="memory budget of 5000 bytes"):
        CellBarcodeJoin(np.asarray(["x" * 10_000]), memory_bytes=5_000)
    join = CellBarcodeJoin(np.asarray(["x" * 10_000]), memory_bytes=20_000)
    assert join.held_bytes >= 10_000


def test_very_long_and_unicode_cell_ids_join_exactly() -> None:
    long_id = "s1__" + "ACGT" * 2500 + "-1"
    join = CellBarcodeJoin(np.asarray([long_id, "s1__κύτταρο-1"]), memory_bytes=10**6)
    positions = join.cells(
        _plain(prefix="s1__", suffix="-1"),
        np.asarray(["ACGT" * 2500, "κύτταρο", "ACGT" * 2499], dtype=object),
    )
    assert positions.tolist() == [0, 1, -1]
    assert join.examples == ["ACGT" * 2499]
