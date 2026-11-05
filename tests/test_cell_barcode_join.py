"""Joining the barcodes of imported files to the cells of a store.

Every importer joins its files to cells through one helper, so the same cases
run on each: prefixes, suffixes, patterns, barcodes that match no cell, the
warning repeated on reuse, the order of files and files listed twice. Each
importer is a row of ``IMPORTERS``; a later importer adds a row.
"""

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytearc.datastore._operations.cell_barcodes import (
    BarcodeFile,
    CellBarcodeJoin,
    barcode_example,
    barcode_files,
    barcode_pattern,
    unmatched_barcode_rules,
)
from cytearc.metadata.arguments import JOIN_DIAGNOSTICS
from cytearc.storage.diagnostics import EXAMPLES
from cytearc import DiagnosticWarning
from cytearc.utils.logging import collect_warnings, logger
from cytearc.utils.memory_budget import (
    MemoryLedger,
)
from tests.barcode_join_helpers import (
    CASES,
    build_guide_store,
    build_join_store,
    import_case,
    write_allele_file,
    write_barcode_file,
    write_clone_file,
    write_contig_file,
    write_guide_call_file,
    write_joined_barcode_file,
    write_joined_clone_file,
    write_tree_file,
    write_variant_file,
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
    # come out the same, the join's counts and examples moved into the
    # diagnostics, each example beside the cell ID it derives.
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
    """The shared join's warning about barcodes that match no cell of the store."""
    return (
        f"{n_unmatched} of {n_barcodes} barcodes of {files} match no cell of the "
        "store, for example "
        + ", ".join(repr(value) for value in examples)
        + ". Check barcode_prefix, barcode_suffix and barcode_pattern against the "
        "store's cell IDs."
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


def _receptors(store, paths, **options):
    return store.repertoire.import_receptors(paths, **options)


def _cells_with_contigs(store, ref) -> set[str]:
    cells = store.repertoire.load_receptors(ref, fields=[]).cells
    return set(cells.index[cells["n_contigs"] > 0])


def _barcodes(store, paths, **options):
    return store.clones.import_barcodes(paths, **options)


def _cells_with_barcodes(store, ref) -> set[str]:
    cells = store.clones.load_barcodes(ref).cells
    return set(cells.index[cells["n_barcodes"] > 0])


def _variant_counts(store, paths, **options):
    return store.variants.import_counts(paths, **options)


def _listed_cells(store, ref) -> set[str]:
    cells = store.variants.load_counts(ref, sites=[]).cells
    return set(cells.index[cells["listed"]])


def _alleles(store, paths, **options):
    return store.clones.import_alleles(paths, **options)


def _cells_with_alleles(store, ref) -> set[str]:
    cells = store.clones.load_alleles(ref).cells
    return set(cells.index[cells["n_records"] > 0])


def _phylogeny(store, paths, **options):
    return store.clones.import_phylogeny(paths, time_unit="days", **options)


def _leaf_cells(store, ref) -> set[str]:
    nodes = store.clones.load_phylogeny(ref).nodes
    return set(nodes["ids"].dropna())


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
    "import_receptors": Importer(
        name="import_receptors",
        several_files=True,
        write=write_contig_file,
        run=_receptors,
        joined=_cells_with_contigs,
        files=("the contig file", "the contig files"),
    ),
    "import_barcodes": Importer(
        name="import_barcodes",
        several_files=True,
        write=write_barcode_file,
        run=_barcodes,
        joined=_cells_with_barcodes,
        files=("the barcode file", "the barcode files"),
    ),
    "import_counts": Importer(
        name="import_counts",
        several_files=True,
        write=write_variant_file,
        run=_variant_counts,
        joined=_listed_cells,
        files=("the variant file", "the variant files"),
    ),
    "import_alleles": Importer(
        name="import_alleles",
        several_files=True,
        write=write_allele_file,
        run=_alleles,
        joined=_cells_with_alleles,
        files=("the allele file", "the allele files"),
    ),
    "import_phylogeny": Importer(
        name="import_phylogeny",
        several_files=True,
        write=write_tree_file,
        run=_phylogeny,
        joined=_leaf_cells,
        files=("the tree file", "the tree files"),
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

    from tests.storage_helpers import open_store

    target = tmp_path / "store.zarr"
    shutil.copytree(join_template, target)
    store = open_store(target)
    return store, store.snapshot_cell_selection("joined")


def _joined(store, ref) -> dict:
    """The join's diagnostics that an import saved."""
    diagnostics = store.artifacts.inspect(ref).diagnostics
    return {name: diagnostics[name] for name in JOIN_DIAGNOSTICS}


def _counts(
    n_file: int,
    unmatched: list[str] | int = 0,
    unselected: list[str] | int = 0,
    *,
    n_unmatched: int | None = None,
    n_unselected: int | None = None,
) -> dict:
    """The join's diagnostics of ``n_file`` barcodes with these examples of each kind."""
    unmatched = [] if isinstance(unmatched, int) else unmatched
    unselected = [] if isinstance(unselected, int) else unselected
    return {
        "fileBarcodes": n_file,
        "unmatchedBarcodes": len(unmatched) if n_unmatched is None else n_unmatched,
        "unselectedBarcodes": (
            len(unselected) if n_unselected is None else n_unselected
        ),
        "unmatchedBarcodeExamples": unmatched,
        "unselectedBarcodeExamples": unselected,
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
    assert _joined(store, ref) == _counts(3)
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

    # "AAAC" and "lane1::z" do not match the pattern and name no cell, which
    # their examples say.
    assert importer.joined(store, ref) == {"s2__AAAC-1", "s2__GGGT-1"}
    examples = [
        "AAAC -> (barcode_pattern does not match)",
        "lane1::z -> (barcode_pattern does not match)",
    ]
    assert _joined(store, ref) == _counts(4, examples)
    assert list(messages) == [importer.warning(importer.files[0], 2, 4, examples, True)]


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
    from tests.storage_helpers import open_store

    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    with collect_warnings() as read_only:
        assert importer.run(reader, [path], cell_selection=cells) == ref

    # Each barcode is the ID of the cell it names, which its example shows
    # once.
    assert _joined(store, ref) == _counts(8, outside[:EXAMPLES], n_unmatched=7)
    expected = [importer.warning(importer.files[0], 7, 8, outside[:EXAMPLES], False)]
    assert list(first) == list(reused) == list(read_only) == expected


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_barcodes_of_cells_outside_the_selection_are_unselected_and_give_no_warning(
    joined, tmp_path, name
) -> None:
    # The second sample's cells are selected; the file names one of them,
    # the first sample's three cells, which QC filtering could have left out,
    # and two barcodes of no cell of the store.
    store, _ = joined
    cells = store.snapshot_cell_selection("s2")
    importer = IMPORTERS[name]
    path = importer.write(
        tmp_path / "file.txt",
        ["s2__GGGT", "s1__TTTA", "nowhere", "s1__AAAC", "s1__CCCG", "s9__AAAC"],
    )

    def run(store: Any) -> tuple[Any, list[str], list[Warning]]:
        with pytest.warns() as caught, collect_warnings() as messages:
            ref = importer.run(store, [path], cell_selection=cells, barcode_suffix="-1")
        return ref, list(messages), [record.message for record in caught]

    ref, first, _issued = run(store)
    from tests.storage_helpers import open_store

    # The examples show each barcode beside the cell ID it derives, in the
    # order in which the importer joins them, which the variant import sorts.
    unmatched = ["nowhere -> nowhere-1", "s9__AAAC -> s9__AAAC-1"]
    unselected = [
        "s1__AAAC -> s1__AAAC-1",
        "s1__CCCG -> s1__CCCG-1",
        "s1__TTTA -> s1__TTTA-1",
    ]
    assert importer.joined(store, ref) == {"s2__GGGT-1"}
    saved = _joined(store, ref)
    for kind in ("unmatchedBarcodeExamples", "unselectedBarcodeExamples"):
        saved[kind] = sorted(saved[kind])
    assert saved == _counts(6, unmatched, unselected)
    # Only the barcodes of no cell warn, and a reused import, writable or
    # read-only, warns alike.
    assert first == [importer.warning(importer.files[0], 2, 6, unmatched, True)]
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    for again in (store, reader):
        reused, messages, warnings = run(again)
        assert reused == ref
        assert messages == first
        assert [str(warning) for warning in warnings] == first
    (warning,) = warnings
    assert isinstance(warning, DiagnosticWarning)
    assert (warning.operation, warning.diagnostic, warning.value) == (
        store.artifacts.inspect(ref).operation,
        "unmatchedBarcodes",
        2,
    )


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
    assert _joined(store, ref)["fileBarcodes"] == 4
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
    # "the clone files" names the files of a listing as the errors do.
    noun = importer.files[1].removeprefix("the ")
    with pytest.raises(ValueError, match=f"lists 3 values for 2 {noun}"):
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
    # a cell outside the selection, and count once, as every importer counts
    # barcodes by the cell ID they name; AAAC does not match the pattern, and
    # no cell is named s1__GGGG-1.
    assert _called_cells(store, calls) == {"s1__AAAC-1"}
    unmatched = ["AAAC -> (barcode_pattern does not match)", "d:GGGG:1 -> s1__GGGG-1"]
    assert _joined(store, calls) == _counts(4, unmatched, ["b:TTTA:1 -> s1__TTTA-1"])
    assert messages == [_guide_call_warning("the call file", 2, 4, unmatched, True)]
    assert notes == [
        "1 of 4 barcodes of the call file are cells of the store outside the "
        "selected cells, for example 'b:TTTA:1 -> s1__TTTA-1'; the import labels "
        "the selected cells only"
    ]
    # A reused import reports both again.
    assert _logged_guide_calls(store, path, **options) == (calls, messages, notes)


def test_guide_calls_refuse_cells_that_the_suffix_cannot_reach(tmp_path) -> None:
    from tests.storage_helpers import open_store, write_count_store, written

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
def test_guide_calls_with_every_option_refuse_a_small_budget_before_writing(
    tmp_path,
) -> None:
    from cytearc.utils.memory_budget import MemoryBudgetError
    from tests.storage_helpers import open_store, write_count_store, written

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

    # A budget of 1 MB cannot hold the guide IDs of 2,700 cells, and the
    # refusal names that step before anything is written.
    small = open_store(path, mem_budget=1_000_000)
    saved = written(small)
    with pytest.raises(
        MemoryBudgetError,
        match=r"^Importing the guide calls of 2700 cells holds about \d+ bytes for "
        r"the guide IDs, more than the memory budget of 1000000 bytes\. Raise "
        r"mem_budget",
    ):
        small.perturbation.import_guide_calls(
            calls,
            cell_selection=kept,
            barcode_prefix="s1__",
            barcode_suffix="-1",
            barcode_pattern=r"^lane1:(?P<barcode>[ACGT]+):",
        )
    assert written(open_store(path)) == saved
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


# The importers that charge every step to one ledger. The variant import
# reads a GoT table whole before its ledger exists, so its pass over the
# store's cell IDs reads mgatk files, in tests/test_variant_import.py.
LEDGER_IMPORTERS = sorted(name for name in IMPORTERS if name != "import_counts")


# A store of many cells, ten of them selected, whose files name cells outside
# the selection, so that the join reads every one of the store's cell IDs.
_WIDE_CELLS = 60_000
_WIDE_SELECTED = 10


@pytest.fixture(scope="module")
def wide_template(tmp_path_factory) -> Path:
    from tests.storage_helpers import open_store, write_count_store

    path = tmp_path_factory.mktemp("wide") / "store.zarr"
    rng = np.random.default_rng(3)
    write_count_store(
        str(path),
        {
            "RNA": rng.poisson(2.0, size=(_WIDE_CELLS, 3)),
            "CRISPR": rng.poisson(3.0, size=(_WIDE_CELLS, 2)),
        },
        "uint16",
        cell_ids=[f"s1__{index:016d}-1" for index in range(_WIDE_CELLS)],
    )
    store = open_store(path)
    store.cells.insert("kept", np.arange(_WIDE_CELLS) < _WIDE_SELECTED)
    return path


@pytest.mark.parametrize("name", LEDGER_IMPORTERS)
def test_the_pass_over_the_stores_cell_ids_counts_every_barcode_of_the_file(
    wide_template, tmp_path, name
) -> None:
    import shutil

    from tests.storage_helpers import open_store

    shutil.copytree(wide_template, tmp_path / "store.zarr")
    store = open_store(tmp_path / "store.zarr")
    selection = store.snapshot_cell_selection("kept")
    importer = IMPORTERS[name]
    # Two selected cells, 20,000 cells outside the selection and 2,000
    # barcodes of no cell.
    path = importer.write(
        tmp_path / "file.txt",
        [f"{index:016d}" for index in range(_WIDE_SELECTED - 2, 20_010)]
        + [f"x{index:015d}" for index in range(2_000)],
    )

    # The join hashes each of the store's 60,000 cell IDs to tell the cells
    # outside the selection from the barcodes of no cell.
    with collect_warnings():
        ref = importer.run(
            store,
            [path],
            cell_selection=selection,
            barcode_prefix="s1__",
            barcode_suffix="-1",
        )
    diagnostics = store.artifacts.inspect(ref).diagnostics
    assert (
        diagnostics["fileBarcodes"],
        diagnostics["unmatchedBarcodes"],
        diagnostics["unselectedBarcodes"],
    ) == (22_002, 2_000, 20_000)


@pytest.mark.parametrize("letter", ["a", "κ", "\U0001f600"])
def test_the_join_counts_the_barcodes_beyond_the_selection_over_chunked_store_ids(
    tmp_path, letter
) -> None:
    import zarr

    # 200,000 cells of the store in chunks of 100,000, each hashed in two
    # parts, ten selected; the barcodes name 5,000 cells outside the selection
    # and 5,000 of no cell.
    ids = np.asarray([f"s1__{letter}{index:09d}-1" for index in range(200_000)])
    group = zarr.open_group(tmp_path / "ids.zarr", mode="w")
    array = group.create_array(
        "ids", shape=ids.shape, dtype=ids.dtype, chunks=(100_000,)
    )
    array[:] = ids
    selected = ids[:10].astype(object)
    barcodes = np.asarray(
        [f"{letter}{index:09d}" for index in range(10, 5_010)]
        + [f"none{index}" for index in range(5_000)],
        dtype=object,
    )
    file = _plain(prefix="s1__", suffix="-1")

    join = CellBarcodeJoin(
        selected, ledger=MemoryLedger(None, "Joining"), store_ids=array
    )
    join.cells(file, barcodes)
    assert (join.n_unselected_barcodes, join.n_unmatched_barcodes) == (5_000, 5_000)


def test_the_join_reads_the_stores_cell_ids_once_and_only_when_it_needs_them() -> None:
    class Counted(np.ndarray):
        reads = 0

        def __getitem__(self, key):  # type: ignore[no-untyped-def]
            if isinstance(key, slice):
                Counted.reads += 1
            return np.asarray(super().__getitem__(key))

    store = np.asarray(["a-1", "b-1", "c-1", "d-1"]).view(Counted)
    file = _plain(suffix="-1")
    # Barcodes of selected cells need no store's IDs; the first barcode
    # outside the selection reads them, and later chunks reuse their hashes.
    join = CellBarcodeJoin(
        np.asarray(["a-1", "b-1"]), ledger=MemoryLedger(None, "J"), store_ids=store
    )
    join.cells(file, np.asarray(["a", "b"], dtype=object))
    assert Counted.reads == 0
    join.cells(file, np.asarray(["c", "z"], dtype=object))
    join.cells(file, np.asarray(["d", "y", "c"], dtype=object))
    assert Counted.reads == 1
    assert join.diagnostics() == _counts(
        6, ["z -> z-1", "y -> y-1"], ["c -> c-1", "d -> d-1"]
    )
    # A selection of every cell of the store leaves none outside it.
    Counted.reads = 0
    every = CellBarcodeJoin(
        np.asarray(store), ledger=MemoryLedger(None, "J"), store_ids=store
    )
    every.cells(file, np.asarray(["z", "a"], dtype=object))
    assert Counted.reads == 0
    assert every.n_unmatched_barcodes == 1


def test_the_pass_keeps_eight_bytes_for_each_of_the_stores_ids(tmp_path) -> None:
    import zarr

    from cytearc.datastore._operations.cell_barcodes import store_pass_bytes

    group = zarr.open_group(tmp_path / "ids.zarr", mode="w")
    arrays = {}
    for n_ids in (100_000, 100_010):
        ids = np.asarray([f"s1__{index:09d}-1" for index in range(n_ids)])
        arrays[n_ids] = group.create_array(
            f"ids{n_ids}", shape=ids.shape, dtype=ids.dtype, chunks=(50_000,)
        )
        arrays[n_ids][:] = ids
    # Ten more IDs in a band of the same rows add their ten hashes alone.
    assert (
        store_pass_bytes(arrays[100_010], 10) - store_pass_bytes(arrays[100_000], 10)
        == 80
    )
    # A read on more threads can leave more chunks behind.
    assert store_pass_bytes(arrays[100_000], 10, codec_threads=4) >= (
        store_pass_bytes(arrays[100_000], 10)
    )
    # A selection of every cell of the store reads none of them.
    assert store_pass_bytes(arrays[100_000], 100_000) == 0


@pytest.mark.parametrize("order", ["named first", "unnamed first", "two chunks"])
def test_a_barcode_that_the_pattern_does_not_match_counts_apart_whatever_the_order(
    order,
) -> None:
    # "AAAC" names no cell, as the pattern does not match it, and "AAAC-1"
    # names AAAC, a cell of the store outside the selection: both count,
    # each as its kind, in any order of the rows and over chunks alike.
    file = _plain(pattern=r"^(?P<barcode>[ACGT]+)-1$")
    rows = {
        "named first": [["AAAC-1", "AAAC", "TTTT"]],
        "unnamed first": [["AAAC", "TTTT", "AAAC-1"]],
        "two chunks": [["AAAC", "TTTT"], ["AAAC-1"]],
    }[order]
    join = CellBarcodeJoin(
        np.asarray(["CCCG"]),
        ledger=MemoryLedger(None, "J"),
        store_ids=np.asarray(["AAAC", "CCCG"]),
    )
    for chunk in rows:
        join.cells(file, np.asarray(chunk, dtype=object))
    saved = join.diagnostics()
    saved["unmatchedBarcodeExamples"] = sorted(saved["unmatchedBarcodeExamples"])
    assert saved == _counts(
        3,
        [
            "AAAC -> (barcode_pattern does not match)",
            "TTTT -> (barcode_pattern does not match)",
        ],
        ["AAAC-1 -> AAAC"],
    )


def test_a_barcode_of_no_cell_and_one_that_the_pattern_does_not_match_count_once() -> (
    None
):
    # "GGGG-1" names GGGG, no cell of the store, and "GGGG", which the
    # pattern does not match, counts by the same ID: one barcode of no cell,
    # with one example.
    file = _plain(pattern=r"^(?P<barcode>[ACGT]+)-1$")
    join = CellBarcodeJoin(
        np.asarray(["CCCG"]),
        ledger=MemoryLedger(None, "J"),
        store_ids=np.asarray(["AAAC", "CCCG"]),
    )
    join.cells(file, np.asarray(["GGGG-1", "GGGG", "CCCG-1"], dtype=object))
    assert join.diagnostics() == _counts(2, ["GGGG-1 -> GGGG"])


def test_a_barcode_that_the_pattern_does_not_match_names_no_cell_of_the_store() -> None:
    # The pattern does not match "AAAC", whose whole barcode with the
    # prefix and suffix is the ID of a cell outside the selection; it names
    # no cell, so it is unmatched.
    join = CellBarcodeJoin(
        np.asarray(["s1__CCCG-1"]),
        ledger=MemoryLedger(None, "J"),
        store_ids=np.asarray(["s1__AAAC-1", "s1__CCCG-1"]),
    )
    join.cells(
        _plain(prefix="s1__", suffix="-1", pattern=r"^x:(?P<barcode>[ACGT]+)$"),
        np.asarray(["x:CCCG", "AAAC"], dtype=object),
    )
    assert join.diagnostics() == _counts(
        2, ["AAAC -> (barcode_pattern does not match)"]
    )


def test_an_example_shows_the_barcode_beside_the_id_it_derives() -> None:
    assert barcode_example("AAAC", "s1__AAAC-1") == "AAAC -> s1__AAAC-1"
    assert barcode_example("s1__AAAC-1", "s1__AAAC-1") == "s1__AAAC-1"
    assert (
        barcode_example("x", "s1__x-1", named=False)
        == "x -> (barcode_pattern does not match)"
    )


def test_the_join_refuses_store_ids_that_are_not_text_and_a_pass_beyond_the_budget() -> (
    None
):
    with pytest.raises(TypeError, match="store's cell IDs must be one-dimensional"):
        CellBarcodeJoin(
            np.asarray(["a"]), ledger=MemoryLedger(None, "J"), store_ids=np.arange(3)
        )
    # The join starts with what its pass over the store's 1,000 IDs will
    # hold, which 10 bytes an ID cannot hold.
    store = np.asarray([f"cell{index}" for index in range(1_000)])
    ledger = MemoryLedger(10_000, "Joining file barcodes")
    with pytest.raises(
        MemoryError,
        match=r"holds about \d+ bytes for the pass over the store's cell IDs, more "
        r"than the memory budget of \d+ bytes\. Raise mem_budget to at least \d+ "
        r"bytes\.$",
    ):
        CellBarcodeJoin(store[:1], ledger=ledger, store_ids=store)


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


def _join(
    cell_ids: Any, ledger: MemoryLedger, store_ids: Any = None
) -> CellBarcodeJoin:
    """A join of ``cell_ids`` in a store of ``store_ids``, by default those cells alone."""
    return CellBarcodeJoin(
        cell_ids,
        ledger=ledger,
        store_ids=np.asarray(cell_ids if store_ids is None else store_ids, dtype=str),
    )


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


@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("name", sorted(_JOIN_OPTIONS))
def test_the_join_matches_long_barcodes_under_every_option(name, matched) -> None:
    # 20,000 barcodes of 120 letters, every one or none of them a selected
    # cell; barcodes that match no cell are hashed and counted.
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
    join = _join(cell_ids, MemoryLedger(None, "Joining"))
    join.cells(file, barcodes)
    assert (join.n_unmatched_rows == 0) == matched


def test_the_join_counts_distinct_barcodes_and_rows() -> None:
    join = _join(np.asarray(["a-1", "b-1", "c-1"]), MemoryLedger(None, "Joining"))

    first = join.cells(
        _plain(suffix="-1"), np.asarray(["a", "b", "z", "z", "a"], dtype=object)
    )
    second = join.cells(
        _plain(suffix="-1"), np.asarray(["y", "z", "x", "w", "v", "u"], dtype=object)
    )

    assert first.tolist() == [0, 1, -1, -1, 0]
    assert second.tolist() == [-1] * 6
    assert (join.n_rows, join.n_unmatched_rows) == (11, 8)
    # a and b, and z, y, x, w, v, u by their cell IDs; the store holds the
    # selected cells alone, so no barcode names a cell outside the selection.
    assert (join.n_file_barcodes, join.n_unmatched_barcodes) == (8, 6)
    assert join.diagnostics() == _counts(
        8, ["z -> z-1", "y -> y-1", "x -> x-1", "w -> w-1", "v -> v-1"], n_unmatched=6
    )


def test_one_barcode_under_two_prefixes_counts_as_two() -> None:
    join = _join(np.asarray(["s1__a"]), MemoryLedger(None, "Joining"))
    join.cells(_plain(prefix="s1__"), np.asarray(["a", "b"], dtype=object))
    join.cells(_plain(prefix="s2__"), np.asarray(["a", "b"], dtype=object))
    assert (join.n_file_barcodes, join.n_unmatched_barcodes) == (4, 3)
    # The examples show the barcodes as the files hold them beside the IDs
    # they derive, each once.
    assert join.unmatched_examples == ["b -> s1__b", "a -> s2__a", "b -> s2__b"]


def test_no_barcode_gives_no_count() -> None:
    join = _join(np.asarray(["a"]), MemoryLedger(None, "Joining"))
    assert (
        join.cells(
            _plain(pattern="(?P<barcode>.)"), np.asarray([], dtype=object)
        ).tolist()
        == []
    )
    assert (join.n_file_barcodes, join.n_rows) == (0, 0)
    (rule,) = unmatched_barcode_rules("the file")
    assert not rule.applies(join.diagnostics())


def _join_ledger(budget: int | None, held: int = 0) -> MemoryLedger:
    """A ledger of ``budget`` bytes that holds ``held`` of them for the caller."""
    ledger = MemoryLedger(budget, "Joining file barcodes")
    ledger.hold("what the caller holds", held)
    return ledger


def test_the_join_refuses_repeated_cell_ids_and_budgets_too_small() -> None:
    with pytest.raises(ValueError, match="cells' IDs repeat"):
        _join(np.asarray(["a", "a"]), _join_ledger(None))
    # A budget that the caller's 300 bytes fill leaves no room for the IDs.
    with pytest.raises(
        MemoryError,
        match="bytes for the selected cells' IDs as text, more than the memory "
        "budget of 301 bytes",
    ):
        _join(np.asarray(["a", "b", "c"]), _join_ledger(301, 300))
    with pytest.raises(TypeError, match="IDs must be text"):
        _join(np.arange(3), _join_ledger(None))
    join = _join(np.asarray(["a"]), _join_ledger(200))
    with pytest.raises(
        MemoryError,
        match="check barcode_prefix, barcode_suffix, barcode_pattern and cell_selection",
    ):
        join.cells(
            _plain(), np.asarray([f"x{index}" for index in range(20)], dtype=object)
        )


def test_the_join_counts_each_distinct_barcode_of_no_cell_once_across_chunks() -> None:
    # Five barcodes that name no cell, then a chunk of six others and one of
    # the five.
    first = np.asarray([f"x{k}" for k in range(5)], dtype=object)
    second = np.asarray([f"y{k}" for k in range(6)] + ["x0"], dtype=object)
    join = _join(np.asarray(["a"]), _join_ledger(None))
    join.cells(_plain(), first)
    join.cells(_plain(), second)
    assert join.n_unmatched_barcodes == 11


@pytest.mark.parametrize(
    ("n_barcodes", "chunk_rows"),
    [
        (100_000, 5_000),
        pytest.param(1_000_000, 50_000, marks=pytest.mark.slow),
    ],
)
def test_the_hashes_of_barcodes_that_match_no_cell_count_each_one_across_chunks(
    n_barcodes, chunk_rows
) -> None:
    # Barcodes outside the selection, in chunks: the join merges each chunk's
    # hashes into those it holds, within a budget of 64 bytes a barcode.
    cell_ids = np.asarray([f"c{index}" for index in range(10)], dtype=object)
    chunks = [
        np.asarray(
            [f"out{index:08d}" for index in range(start, start + chunk_rows)],
            dtype=object,
        )
        for start in range(0, n_barcodes, chunk_rows)
    ]

    def run(budget: int) -> CellBarcodeJoin:
        join = _join(cell_ids, _join_ledger(budget))
        for chunk in chunks:
            join.cells(_plain(), chunk)
        return join

    assert run(64 * n_barcodes).n_unmatched_barcodes == n_barcodes


def test_the_join_counts_the_text_of_long_cell_ids() -> None:
    # One ID of 10,000 characters holds more than the budget by itself.
    with pytest.raises(MemoryError, match="memory budget of 5000 bytes"):
        _join(np.asarray(["x" * 10_000]), _join_ledger(5_000))
    _join(np.asarray(["x" * 10_000]), _join_ledger(20_000))


def test_very_long_and_unicode_cell_ids_join_exactly() -> None:
    long_id = "s1__" + "ACGT" * 2500 + "-1"
    join = _join(np.asarray([long_id, "s1__κύτταρο-1"]), MemoryLedger(None, "Joining"))
    positions = join.cells(
        _plain(prefix="s1__", suffix="-1"),
        np.asarray(["ACGT" * 2500, "κύτταρο", "ACGT" * 2499], dtype=object),
    )
    assert positions.tolist() == [0, 1, -1]
    assert join.unmatched_examples == [f"{'ACGT' * 2499} -> s1__{'ACGT' * 2499}-1"]


# The importers whose cell barcode may join several columns, with a writer of
# a file whose cells join a sample column to a barcode column and the options
# that read it.
JOINED_CELLS = {
    "import_clones": (
        write_joined_clone_file,
        {"columns": {"cell": ["sample", "barcode"], "clone": "clone"}},
    ),
    "import_barcodes": (
        write_joined_barcode_file,
        {
            "format": "pairs",
            "columns": {"cell": ["sample", "barcode"], "barcode": "lb", "umis": "n"},
        },
    ),
}


@pytest.mark.parametrize("name", sorted(JOINED_CELLS))
@pytest.mark.parametrize(
    "parts",
    [("s1", ""), ("", "CCCG-1"), (" ", "CCCG-1")],
    ids=["barcode", "sample", "blank"],
)
def test_a_cell_joined_from_columns_without_a_part_is_refused(
    joined, tmp_path, name, parts
) -> None:
    store, selection = joined
    importer = IMPORTERS[name]
    write, options = JOINED_CELLS[name]
    whole = write(tmp_path / "whole.tsv", [("s1", "AAAC-1"), ("s2", "GGGT-1")])
    ref = importer.run(store, [whole], cell_selection=selection, **options)
    assert importer.joined(store, ref) == {"s1__AAAC-1", "s2__GGGT-1"}
    # A part left empty would join to "s1__" or "__CCCG-1", which names no
    # cell, so the row is refused with its line rather than counted unmatched.
    path = write(tmp_path / "part.tsv", [("s1", "AAAC-1"), parts])
    with pytest.raises(
        ValueError,
        match=r"lack a part of their cell barcode \('sample', 'barcode'\) on line 3$",
    ):
        importer.run(store, [path], cell_selection=selection, **options)
