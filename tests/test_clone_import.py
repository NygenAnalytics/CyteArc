"""Clones imported from every supported format recover a synthetic truth.

The module writes a store of the truth's cells once, as a template. Tests that
only import share one copy, since each import saves its own artifact; tests
that check reuse, read-only stores or the memory budget copy the template.
"""

import gzip
import math
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from cytearc import DiagnosticWarning, ExportSummary
from cytearc.clones import Clones
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from cytearc.storage.operation_revisions import OPERATION_REVISIONS
from cytearc.utils.digest import file_sha256
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    LISTED,
    assert_joined,
    every_cell,
    loaded_partition,
    write_cells_store,
    write_truth_store,
)
from tests.storage_helpers import (
    by_cell_id,
    computed_anew,
    copy_store,
    open_store,
    rewrite_as_saved_by_1_4,
)
from tests.clone_truth import (
    CLONE_SIZES,
    N_CELLS,
    N_OUTSIDE,
    clone_truth,
    write_anndata_h5ad,
    write_cassiopeia,
    write_celltag_multi,
    write_celltagr,
    write_cospar,
    write_cospar_h5ad,
    write_larry,
    write_mito_h5ad,
    write_numbat,
    write_sticr,
    write_trex,
    write_wide,
)

TRUTH = clone_truth()
N_IN_CLONES = sum(CLONE_SIZES)
# The IDs that the files' droplets outside the store name.
_OUTSIDE_IDS = [f"d2__{barcode}-1" for barcode in TRUTH.outside]
# Clone labels as each writer numbers the truth's clones from 0.
NUMBERED = [str(clone + 1) for clone in range(len(CLONE_SIZES))]


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clone_import") / "store.zarr"
    write_truth_store(path, TRUTH)
    return path


@pytest.fixture(scope="module")
def shared(template, tmp_path_factory) -> SimpleNamespace:
    """One writable copy for the imports that save artifacts of their own."""
    store = copy_store(template, tmp_path_factory.mktemp("clone_shared"))
    return SimpleNamespace(store=store, cells=every_cell(store))


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


# Each case: the writer, the import's options, the format it should detect,
# the clone labels in truth order, and how many distinct barcodes the file
# lists: only the cells with a clone, or every cell, with or without the
# barcodes outside the store.
_LISTS_CLONED = N_IN_CLONES + N_OUTSIDE
_LISTS_EVERY = N_CELLS + N_OUTSIDE
_CASES: dict[str, tuple[Callable[[Path], Path], dict, str, list[str], int]] = {
    "cospar": (
        lambda folder: write_cospar(TRUTH, folder / "clones.csv"),
        {},
        "cospar",
        NUMBERED,
        _LISTS_CLONED,
    ),
    "cospar_gzip": (
        lambda folder: write_cospar(TRUTH, folder / "clones.csv.gz", compress=True),
        {},
        "cospar",
        NUMBERED,
        _LISTS_CLONED,
    ),
    **{
        f"trex_{generation}": (
            lambda folder, generation=generation: write_trex(
                TRUTH, folder / "clones.txt", generation=generation
            ),
            {},
            "trex",
            NUMBERED,
            _LISTS_CLONED,
        )
        for generation in ("2022", "2023", "current")
    },
    "celltag_multi": (
        lambda folder: write_celltag_multi(TRUTH, folder / "clones.csv"),
        {},
        "celltag_multi",
        [f"r1_{label}" for label in NUMBERED],
        _LISTS_CLONED,
    ),
    "celltag_multi_without_index": (
        lambda folder: write_celltag_multi(
            TRUTH, folder / "clones.csv", index_column=False
        ),
        {},
        "celltag_multi",
        [f"r1_{label}" for label in NUMBERED],
        _LISTS_CLONED,
    ),
    "celltagr": (
        lambda folder: write_celltagr(TRUTH, folder / "clones.csv"),
        {},
        "celltagr",
        NUMBERED,
        _LISTS_CLONED,
    ),
    "sticr": (
        lambda folder: write_sticr(TRUTH, folder / "meta.tsv"),
        {},
        "sticr",
        [f"GW15_Rep1_{label}" for label in NUMBERED],
        _LISTS_EVERY,
    ),
    "cassiopeia": (
        lambda folder: write_cassiopeia(TRUTH, folder / "alleles.txt.gz"),
        {},
        "cassiopeia",
        NUMBERED,
        _LISTS_EVERY,
    ),
    "numbat": (
        lambda folder: write_numbat(TRUTH, folder / "clone_post_2.tsv"),
        {},
        "numbat",
        NUMBERED,
        N_CELLS,
    ),
    "wide": (
        lambda folder: write_wide(TRUTH, folder / "celltag_data.txt"),
        {"columns": {"clone": "CellTagD2"}},
        "table",
        NUMBERED,
        _LISTS_EVERY,
    ),
    "larry_cells_by_clones": (
        lambda folder: write_larry(TRUTH, folder, orientation="cells_by_clones"),
        {"columns": {"cell": ["Library", "Cell barcode"]}},
        "matrix",
        [str(clone) for clone in range(len(CLONE_SIZES))],
        _LISTS_EVERY,
    ),
    "larry_clones_by_cells": (
        lambda folder: write_larry(TRUTH, folder, orientation="clones_by_cells"),
        {"columns": {"cell": ["Library", "Cell barcode"]}},
        "matrix",
        [str(clone) for clone in range(len(CLONE_SIZES))],
        _LISTS_EVERY,
    ),
    **{
        f"cospar_h5ad_{layout}": (
            lambda folder, layout=layout: write_cospar_h5ad(
                TRUTH, folder / "adata.h5ad", layout=layout
            ),
            {},
            "cospar_h5ad",
            [str(clone) for clone in range(len(CLONE_SIZES))],
            _LISTS_EVERY,
        )
        for layout in ("csr", "csc", "dense")
    },
    "mito": (
        lambda folder: write_mito_h5ad(TRUTH, folder / "afm.h5ad"),
        {},
        "mito",
        [f"MT_{label}" for label in NUMBERED],
        _LISTS_EVERY,
    ),
    # anndata writes the cell names and text columns as nullable string
    # arrays, groups of values and a mask.
    "mito_by_anndata": (
        lambda folder: write_anndata_h5ad(TRUTH, folder / "afm.h5ad", content="mito"),
        {},
        "mito",
        [f"MT_{label}" for label in NUMBERED],
        _LISTS_EVERY,
    ),
    "cospar_h5ad_by_anndata": (
        lambda folder: write_anndata_h5ad(
            TRUTH, folder / "adata.h5ad", content="cospar"
        ),
        {},
        "cospar_h5ad",
        [str(clone) for clone in range(len(CLONE_SIZES))],
        _LISTS_EVERY,
    ),
    "h5ad_columns_by_anndata": (
        lambda folder: write_anndata_h5ad(
            TRUTH, folder / "adata.h5ad", content="columns"
        ),
        {"columns": {"clone": "clone", "probability": "probability"}},
        "h5ad",
        NUMBERED,
        _LISTS_EVERY,
    ),
}


def _folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    return folder


@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_format_recovers_the_truth(shared, tmp_path, case) -> None:
    write, options, detected, labels, n_listed = _CASES[case]
    path = write(_folder(tmp_path, case))

    with (
        pytest.warns(UserWarning, match="match no cell of the store")
        if (n_listed > N_CELLS)
        else collect_warnings() as messages
    ):
        ref = shared.store.clones.import_clones(
            path, cell_selection=shared.cells, **options
        )

    assert ref.kind == "clones"
    assert ref.scope == "datastore"
    (source,) = shared.store.artifacts.inspect(ref).inputs["sources"]
    assert source["format"] == detected
    assert loaded_partition(shared.store, ref) == TRUTH.partition()
    loaded = shared.store.clones.load_clones(ref)
    # Clones rank by size; the truth's clones are numbered in that order, and
    # its two clones of two cells by label.
    assert loaded.clones["n_cells"].tolist() == list(CLONE_SIZES)
    assert loaded.clones["clone"].tolist() == labels
    attributes = shared.store.artifacts.load(ref).attrs
    diagnostics = shared.store.artifacts.inspect(ref).diagnostics
    # Every file but Numbat's lists the barcodes outside the store, each on a
    # row of its own, or as a cell of a matrix.
    assert_joined(
        diagnostics,
        n_file=n_listed,
        unmatched=_OUTSIDE_IDS if n_listed != N_CELLS else (),
    )
    # Cassiopeia's table holds two or three rows per barcode, one per intBC,
    # as its writer alternates them; every other format a row, or a
    # matrix's cell, per barcode.
    assert diagnostics["fileRows"] == (
        sum(2 + position % 2 for position in range(n_listed))
        if case == "cassiopeia"
        else n_listed
    )
    assert attributes["n_clones"] == len(CLONE_SIZES)
    assert diagnostics["cellsInSeveralClones"] == 0
    if n_listed == N_CELLS:
        assert list(messages) == []


def test_numbat_posteriors_become_the_clone_probability(shared, tmp_path) -> None:
    path = write_numbat(TRUTH, tmp_path / "clone_post_2.tsv")

    ref = shared.store.clones.import_clones(path, cell_selection=shared.cells)

    cells = shared.store.clones.load_clones(ref).cells
    in_clone = TRUTH.clones >= 0
    np.testing.assert_allclose(
        cells["clone_probability"].to_numpy()[in_clone], TRUTH.probabilities[in_clone]
    )
    assert np.isnan(cells["clone_probability"].to_numpy()[~in_clone]).all()
    values = shared.store.artifacts.load_values(ref, value="clone_probability")
    np.testing.assert_array_equal(values.values, cells["clone_probability"].to_numpy())
    assert values.categorical is False


def test_cell_names_that_anndata_writes_as_nullable_strings_are_read(
    shared, tmp_path
) -> None:
    path = write_anndata_h5ad(TRUTH, tmp_path / "adata.h5ad", content="columns")
    # The fixture stores its names and labels as current anndata does, or the
    # rows of the format matrix written by anndata would test nothing new.
    with h5py.File(path, "r") as handle:
        assert isinstance(handle["obs/_index"], h5py.Group)
        assert isinstance(handle["obs/clone"], h5py.Group)

    with pytest.warns(UserWarning, match="match no cell of the store"):
        ref = shared.store.clones.import_clones(
            path,
            cell_selection=shared.cells,
            columns={"clone": "clone", "probability": "probability"},
        )

    cells = shared.store.clones.load_clones(ref).cells
    in_clone = TRUTH.clones >= 0
    np.testing.assert_allclose(
        cells["clone_probability"].to_numpy()[in_clone], TRUTH.probabilities[in_clone]
    )
    assert np.isnan(cells["clone_probability"].to_numpy()[~in_clone]).all()


def test_an_anndata_file_with_a_cell_without_a_name_is_refused(
    shared, tmp_path
) -> None:
    path = write_mito_h5ad(TRUTH, tmp_path / "afm.h5ad")
    with h5py.File(path, "r+") as handle:
        obs = handle["obs"]
        names = obs["_index"][:]
        del obs["_index"]
        index = obs.create_group("_index")
        index.attrs["encoding-type"] = "nullable-string-array"
        index.attrs["encoding-version"] = "0.1.0"
        index.create_dataset("values", data=names)
        mask = np.zeros(len(names), dtype=bool)
        mask[4] = True
        index.create_dataset("mask", data=mask)

    with pytest.raises(ValueError, match="cells without a name"):
        shared.store.clones.import_clones(path, cell_selection=shared.cells)


def test_clones_select_cells_and_read_as_per_cell_values(shared, tmp_path) -> None:
    ref = shared.store.clones.import_clones(
        write_trex(TRUTH, tmp_path / "clones.txt"), cell_selection=shared.cells
    )

    largest = shared.store.select_cells(ref, include=["1"])
    selected = shared.store.cells.fetch_all("ids")[
        np.asarray(shared.store.artifacts.load(largest)["values"][:])
    ]
    assert set(selected.tolist()) == set(TRUTH.cell_ids[TRUTH.clones == 0].tolist())
    codes = shared.store.artifacts.load_values(ref, value="codes")
    np.testing.assert_array_equal(codes.values, TRUTH.clones)
    sizes = shared.store.artifacts.load_values(ref, value="clone_size").values
    expected = np.where(
        TRUTH.clones >= 0, np.asarray(CLONE_SIZES + (0,))[TRUTH.clones], 0
    )
    np.testing.assert_array_equal(sizes, expected)
    with pytest.raises(ValueError, match="no 'clone_probability' value array"):
        shared.store.artifacts.load_values(ref, value="clone_probability")


def _per_sample_files(folder: Path, *, bare: bool) -> list[Path]:
    """One TREX file per sample, naming cells by bare barcode or by full ID."""
    paths = []
    for sample in ("d2", "d4"):
        cells = TRUTH.samples == sample
        text = TRUTH.barcodes if bare else None
        path = folder / f"{sample}.clones.txt"
        write_trex(TRUTH, path, cells=cells, cell_text=text)
        paths.append(path)
    return paths


def test_files_of_each_sample_join_by_prefix_and_keep_their_clones_apart(
    shared, tmp_path
) -> None:
    paths = _per_sample_files(tmp_path, bare=True)

    # Each file lists the three barcodes outside the store, under its prefix.
    listed = N_IN_CLONES + 2 * N_OUTSIDE
    with pytest.warns(UserWarning, match=f"6 of {listed} barcodes of the clone files"):
        ref = shared.store.clones.import_clones(
            paths,
            cell_selection=shared.cells,
            barcode_prefix=["d2__", "d4__"],
            barcode_suffix="-1",
        )

    # Equal clone numbers of two files are two clones, labeled by prefix.
    assert loaded_partition(shared.store, ref) == TRUTH.partition(
        split_by=TRUTH.samples
    )
    labels = set(shared.store.clones.load_clones(ref).clones["clone"])
    assert {label.split("__")[0] for label in labels} == {"d2", "d4"}
    sources = shared.store.artifacts.inspect(ref).inputs["sources"]
    assert sorted(source["namespace"] for source in sources) == ["d2__", "d4__"]


def test_files_without_prefixes_are_namespaced_by_their_names(shared, tmp_path) -> None:
    paths = _per_sample_files(tmp_path, bare=False)

    with pytest.warns(UserWarning, match="match no cell of the store"):
        ref = shared.store.clones.import_clones(paths, cell_selection=shared.cells)

    labels = shared.store.clones.load_clones(ref).clones["clone"].tolist()
    assert {label.split("_")[0] for label in labels} == {"d2", "d4"}
    assert loaded_partition(shared.store, ref) == TRUTH.partition(
        split_by=TRUTH.samples
    )


def test_files_that_share_a_namespace_are_refused(shared, tmp_path) -> None:
    first = _folder(tmp_path, "a")
    second = _folder(tmp_path, "b")
    write_trex(TRUTH, first / "clones.txt", cells=TRUTH.samples == "d2")
    write_trex(TRUTH, second / "clones.txt", cells=TRUTH.samples == "d4")

    # Files named alike are renamed; a prefix would change the cells they name.
    with pytest.raises(
        ValueError,
        match="share the namespace 'clones_' .* rename one of the files so",
    ):
        shared.store.clones.import_clones(
            [first / "clones.txt", second / "clones.txt"], cell_selection=shared.cells
        )
    with pytest.raises(
        ValueError, match="share the namespace 's1' .* give each file its own"
    ):
        shared.store.clones.import_clones(
            [first / "clones.txt", second / "clones.txt"],
            cell_selection=shared.cells,
            barcode_prefix="s1",
        )


def test_files_without_a_name_before_their_extensions_need_a_prefix(
    shared, tmp_path
) -> None:
    hidden = write_trex(TRUTH, tmp_path / ".clones.txt", cells=TRUTH.samples == "d2")
    other = write_trex(TRUTH, tmp_path / "d4.txt", cells=TRUTH.samples == "d4")
    with pytest.raises(ValueError, match="has no name before its extensions"):
        shared.store.clones.import_clones([hidden, other], cell_selection=shared.cells)


def test_clone_ids_that_are_numbers_in_one_file_and_text_in_another(
    shared, tmp_path
) -> None:
    numbers = write_trex(TRUTH, tmp_path / "numbers.txt", cells=TRUTH.samples == "d2")
    text = tmp_path / "text.csv"
    rows = [
        f"{cell},r1_{clone + 1}"
        for cell, clone, sample in zip(
            TRUTH.cell_ids, TRUTH.clones, TRUTH.samples, strict=True
        )
        if clone >= 0 and sample == "d4"
    ]
    text.write_text("Cell_ID,Clone_ID\n" + "\n".join(rows) + "\n")

    with pytest.warns(UserWarning, match="match no cell of the store"):
        ref = shared.store.clones.import_clones(
            [numbers, text], cell_selection=shared.cells
        )

    loaded = shared.store.clones.load_clones(ref)
    assert loaded_partition(shared.store, ref) == TRUTH.partition(
        split_by=TRUTH.samples
    )
    assert all(
        label.startswith(("numbers_", "text_r1_")) for label in loaded.clones["clone"]
    )


def _conflicting_table(path: Path) -> Path:
    lines = ["Cell_ID,Clone_ID"]
    for cell, clone in zip(TRUTH.cell_ids, TRUTH.clones, strict=True):
        if clone >= 0:
            lines.append(f"{cell},{clone + 1}")
    # Two cells listed again in another clone; one of them twice in it.
    first, second = TRUTH.cell_ids[TRUTH.clones == 0][:2]
    lines += [f"{first},2", f"{second},3", f"{second},3", f"{first},1"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_cell_in_two_clones_is_refused_with_examples(shared, tmp_path) -> None:
    path = _conflicting_table(tmp_path / "clones.csv")
    first, second = TRUTH.cell_ids[TRUTH.clones == 0][:2]

    with pytest.raises(
        ValueError, match="2 cells are listed in two different clones"
    ) as raised:
        shared.store.clones.import_clones(path, cell_selection=shared.cells)
    message = str(raised.value)
    assert f"{str(first)!r} in '1' and '2'" in message
    assert f"{str(second)!r} in '1' and '3'" in message
    assert "multiple='drop'" in message


def test_dropped_cells_have_no_clone_and_are_counted(shared, tmp_path) -> None:
    path = _conflicting_table(tmp_path / "clones.csv")
    dropped = set(TRUTH.cell_ids[TRUTH.clones == 0][:2].tolist())

    ref = shared.store.clones.import_clones(
        path, cell_selection=shared.cells, multiple="drop"
    )

    cells = shared.store.clones.load_clones(ref).cells
    assert cells.loc[sorted(dropped), "clone"].isna().all()
    assert (cells.loc[sorted(dropped), "clone_size"] == 0).all()
    expected = {
        frozenset(group - dropped) for group in TRUTH.partition() if group - dropped
    }
    assert loaded_partition(shared.store, ref) == expected
    assert shared.store.artifacts.inspect(ref).diagnostics["cellsInSeveralClones"] == 2
    assert shared.store.artifacts.inspect(ref).parameters == {"multiple": "drop"}


def test_repeated_rows_of_one_cell_and_clone_count_the_cell_once(
    shared, tmp_path
) -> None:
    # Cassiopeia lists each cell once per target site.
    path = write_cassiopeia(TRUTH, tmp_path / "alleles.txt", compress=False)
    with pytest.warns(UserWarning, match=f"{N_OUTSIDE} of {N_CELLS + N_OUTSIDE} "):
        ref = shared.store.clones.import_clones(path, cell_selection=shared.cells)
    assert shared.store.clones.load_clones(ref).clones["n_cells"].tolist() == list(
        CLONE_SIZES
    )


def test_a_subset_of_cells_holds_only_its_clones(shared, tmp_path) -> None:
    shared.store.cells.insert(
        "day_two", by_cell_id(shared.store.cells, TRUTH.samples == "d2"), overwrite=True
    )
    day_two = shared.store.snapshot_cell_selection("day_two")
    path = write_trex(TRUTH, tmp_path / "clones.txt")

    with pytest.warns(UserWarning, match="match no cell of the store"):
        ref = shared.store.clones.import_clones(path, cell_selection=day_two)

    cells = shared.store.clones.load_clones(ref).cells
    assert cells.index.tolist() == TRUTH.cell_ids[TRUTH.samples == "d2"].tolist()
    assert loaded_partition(shared.store, ref) == TRUTH.partition(TRUTH.samples == "d2")
    # The day-four cells of the file's clones are cells of the store outside
    # the selection, and the droplets outside the store match no cell.
    in_clones = TRUTH.clones >= 0
    assert_joined(
        shared.store.artifacts.inspect(ref).diagnostics,
        n_file=N_IN_CLONES + N_OUTSIDE,
        unmatched=_OUTSIDE_IDS,
        unselected=TRUTH.cell_ids[(TRUTH.samples == "d4") & in_clones].tolist(),
    )


def test_import_is_reused_by_contents_in_any_file_order(store, tmp_path) -> None:
    cells = every_cell(store)
    paths = _per_sample_files(tmp_path, bare=True)
    options = {"cell_selection": cells, "barcode_suffix": "-1"}

    with pytest.warns(UserWarning):
        first = store.clones.import_clones(
            paths, barcode_prefix=["d2__", "d4__"], **options
        )
    with pytest.warns(UserWarning):
        swapped = store.clones.import_clones(
            paths[::-1], barcode_prefix=["d4__", "d2__"], **options
        )
    assert swapped == first
    moved = [tmp_path / "moved_d2.clones.txt", tmp_path / "moved_d4.clones.txt"]
    for source, target in zip(paths, moved, strict=True):
        target.write_bytes(source.read_bytes())
    with pytest.warns(UserWarning):
        assert (
            store.clones.import_clones(
                moved, barcode_prefix=["d2__", "d4__"], **options
            )
            == first
        )
    with pytest.warns(UserWarning):
        again = store.clones.import_clones(
            paths, barcode_prefix=["d2__", "d4__"], invalidate_cache=True, **options
        )
    assert again != first


def test_a_single_file_moved_is_reused_and_an_edited_one_is_not(
    store, tmp_path
) -> None:
    cells = every_cell(store)
    path = write_cospar(TRUTH, tmp_path / "a.csv", cells=TRUTH.samples == "d2")
    with pytest.warns(UserWarning):
        first = store.clones.import_clones(path, cell_selection=cells)
    moved = tmp_path / "b.csv"
    moved.write_bytes(path.read_bytes())
    with pytest.warns(UserWarning):
        assert store.clones.import_clones(moved, cell_selection=cells) == first
    moved.write_text(moved.read_text().replace(",1\n", ",9\n", 1))
    with pytest.warns(UserWarning):
        assert store.clones.import_clones(moved, cell_selection=cells) != first


def test_a_reused_import_warns_again_and_a_read_only_store_reads_it(
    store, tmp_path
) -> None:
    cells = every_cell(store)
    path = write_trex(TRUTH, tmp_path / "clones.txt")

    with collect_warnings() as first_messages:
        saved = store.clones.import_clones(path, cell_selection=cells)
    with collect_warnings() as reused_messages:
        assert store.clones.import_clones(path, cell_selection=cells) == saved
    assert list(first_messages) == list(reused_messages)
    assert list(first_messages) == [
        f"{N_OUTSIDE} of {N_IN_CLONES + N_OUTSIDE} barcodes of the clone file match "
        "no cell of the store, for example "
        + ", ".join(repr(f"d2__{barcode}-1") for barcode in TRUTH.outside)
        + ". Check barcode_prefix, barcode_suffix and barcode_pattern against the "
        "store's cell IDs."
    ]
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    with collect_warnings() as reader_messages:
        assert reader.clones.import_clones(path, cell_selection=cells) == saved
    assert list(reader_messages) == list(first_messages)
    other = write_cospar(TRUTH, tmp_path / "other.csv")
    with pytest.raises(PermissionError, match="import_clones"):
        reader.clones.import_clones(other, cell_selection=cells)


def test_import_refuses_a_budget_too_small_and_writes_nothing(store, tmp_path) -> None:
    cells = every_cell(store)
    path = write_trex(TRUTH, tmp_path / "clones.txt")
    store.memoryBytes = 10_000

    with pytest.raises(MemoryError, match="memory budget of 10000 bytes"):
        store.clones.import_clones(path, cell_selection=cells)
    assert store.artifacts.list(scope="datastore", kind="clones") == []


def test_import_refuses_more_clone_labels_than_the_budget_holds(
    store, tmp_path, monkeypatch
) -> None:
    import cytearc.clones.assignments as assignments

    # Each label is taken to hold a terabyte, so the first chunk's labels
    # outgrow the budget.
    monkeypatch.setattr(assignments, "_LABEL_ENTRY_BYTES", 2**40)
    for path in (
        write_trex(TRUTH, tmp_path / "clones.txt"),
        write_larry(TRUTH, _folder(tmp_path, "larry")),
    ):
        with pytest.raises(MemoryError, match=r"bytes for \d+ clone labels"):
            store.clones.import_clones(
                path,
                cell_selection=every_cell(store),
                columns={"cell": ["Library", "Cell barcode"]}
                if path.name.endswith(".mtx.gz")
                else None,
            )
    assert store.artifacts.list(scope="datastore", kind="clones") == []


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"cell_selection": "every_cell"}, TypeError, "cell_selection must be"),
        ({"multiple": "keep"}, ValueError, "multiple must be one of error, drop"),
        ({"invalidate_cache": 1}, TypeError, "invalidate_cache must be a boolean"),
        ({"format": 3}, TypeError, "format must be a format name"),
        ({"format": "larry"}, ValueError, "format must be one of"),
        ({"format": "numbat"}, ValueError, "has no column 'cell'"),
        ({"columns": ["cell"]}, TypeError, "columns must map"),
        ({"columns": {"barcode": "x"}}, ValueError, "unknown roles 'barcode'"),
        ({"columns": {"clone": 3}}, TypeError, r"columns\['clone'\] must be"),
        (
            {"columns": {"cell": "cell_id", "clone": "missing"}},
            ValueError,
            "has no column 'missing'",
        ),
        ({"columns": {"clone": "clone_nr"}}, ValueError, "has no row names"),
        ({"barcode_prefix": 3}, TypeError, "barcode_prefix must be text"),
        ({"barcode_suffix": ["a", "b"]}, ValueError, "lists 2 values for 1 clone"),
        ({"barcode_pattern": "(.+)-1"}, ValueError, "no named group 'barcode'"),
        ({"barcode_pattern": "(?P<barcode>"}, ValueError, "not a valid regular"),
    ],
)
def test_import_refuses_arguments_it_cannot_use(
    shared, tmp_path, arguments, error, message
) -> None:
    path = write_trex(TRUTH, tmp_path / "clones.txt")
    call = {"cell_selection": shared.cells, **arguments}
    with pytest.raises(error, match=message):
        shared.store.clones.import_clones(path, **call)


def test_import_refuses_files_it_cannot_read(shared, tmp_path) -> None:
    cells = shared.cells
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="is empty"):
        shared.store.clones.import_clones(empty, cell_selection=cells)
    unknown = tmp_path / "unknown.tsv"
    unknown.write_text("barcode\tgroup\nAAAC\t1\n")
    with pytest.raises(ValueError, match="match no clone table that CyteArc knows"):
        shared.store.clones.import_clones(unknown, cell_selection=cells)
    with pytest.raises(FileNotFoundError):
        shared.store.clones.import_clones(tmp_path / "absent.csv", cell_selection=cells)
    with pytest.raises(ValueError, match="paths must name at least one clone file"):
        shared.store.clones.import_clones([], cell_selection=cells)
    with pytest.raises(TypeError, match="paths must be a path"):
        shared.store.clones.import_clones(3, cell_selection=cells)
    nobody = shared.store.cells.insert(
        "nobody",
        by_cell_id(shared.store.cells, np.zeros(N_CELLS, dtype=bool)),
        overwrite=True,
    )
    assert nobody is None
    with pytest.raises(ValueError, match="selects no cell"):
        shared.store.clones.import_clones(
            write_trex(TRUTH, tmp_path / "clones.txt"),
            cell_selection=shared.store.snapshot_cell_selection("nobody"),
        )


def _write_lines(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _write_mito_of_no_cell(path: Path) -> Path:
    import anndata

    obs = pd.DataFrame(
        {"MiTo_clone": pd.Categorical([], categories=["MT_1"])},
        index=pd.Index([], dtype=str),
    )
    anndata.AnnData(obs=obs).write_h5ad(path)
    return path


def _write_matrix_of_no_cell(folder: Path) -> Path:
    # A cell list of a header alone beside a matrix of no row.
    _write_lines(folder / "cells.tsv", "cell\n")
    return _write_lines(
        folder / "clones.mtx",
        "%%MatrixMarket matrix coordinate integer general\n0 3 0\n",
    )


# Files that hold no row, each with the options that read it and whether its
# format gives probabilities: a header alone, and with blank lines; Numbat's
# table and an AnnData column of probabilities, whose clones save each cell's
# probability; an AnnData file of no cell; and a clone matrix of no cell.
_ROWLESS: dict[str, tuple[Callable[[Path], Path], dict, bool]] = {
    "trex": (
        lambda folder: _write_lines(folder / "clones.txt", "clone_nr\tcell_id\n"),
        {},
        False,
    ),
    "cospar_blank_lines": (
        lambda folder: _write_lines(folder / "clones.csv", "Cell_ID,Clone_ID\n\n\n"),
        {},
        False,
    ),
    "numbat": (
        lambda folder: _write_lines(
            folder / "clone_post_2.tsv", "cell\tclone_opt\tGT_opt\tp_opt\n"
        ),
        {},
        True,
    ),
    "table_with_probabilities": (
        lambda folder: _write_lines(folder / "calls.tsv", "barcode\tlineage\tp\n"),
        {"columns": {"cell": "barcode", "clone": "lineage", "probability": "p"}},
        True,
    ),
    "mito": (lambda folder: _write_mito_of_no_cell(folder / "afm.h5ad"), {}, False),
    "matrix": (_write_matrix_of_no_cell, {}, False),
}


def _no_row_warning(names: list[str]) -> str:
    """A pattern of the import's warning of files without a row, of these names."""
    files, remedy = (
        ("the clone file", "a file that lists")
        if len(names) == 1
        else ("the clone files", "files that list")
    )
    return (
        f"The import read no row from {files} "
        + ", ".join(f"'[^']*{re.escape(name)}'" for name in names)
        + rf", so no selected cell has a clone\. Pass {remedy} the cells' clones\."
    )


def _empty_files_warning(names: list[str], n_files: int) -> str:
    """A pattern of the import's warning of some of its ``n_files`` files that hold no row."""
    files, those, remedy = (
        (
            "the clone file",
            "that file gives",
            "file you meant in its place, or leave it",
        )
        if len(names) == 1
        else (
            "the clone files",
            "those files give",
            "files you meant in their place, or leave them",
        )
    )
    return (
        f"The import read no row from {files} "
        + ", ".join(f"'[^']*{re.escape(name)}'" for name in names)
        + rf", {len(names)} of its {n_files} clone files, so {those} no cell a "
        rf"clone\. Pass the {remedy} out\."
    )


def _rows_per_file(store, ref: ArtifactRef, rows: dict[Path, int]) -> list[int]:
    """The rows of each file, by its path, in the order of the import's recorded files."""
    by_digest = {file_sha256(path): count for path, count in rows.items()}
    sources = store.artifacts.inspect(ref).inputs["sources"]
    return [by_digest[source["sha256"]] for source in sources]


#: The refusal of the clones of an import of one file that read no row.
_NO_ROW_REFUSAL = (
    "^The clones give no cell a clone: their import read no row from the clone "
    "file; import a file that lists the cells' clones$"
)


@pytest.mark.parametrize("case", sorted(_ROWLESS))
def test_an_import_that_reads_no_row_warns_naming_its_file_again_on_reuse(
    shared, tmp_path, case
) -> None:
    write, options, with_probability = _ROWLESS[case]
    path = write(_folder(tmp_path, case))

    refs = []
    for _call in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refs.append(
                shared.store.clones.import_clones(
                    path, cell_selection=shared.cells, **options
                )
            )
        assert [type(item.message) for item in caught] == [DiagnosticWarning]
        warning = caught[0].message
        assert isinstance(warning, DiagnosticWarning)
        assert re.fullmatch(_no_row_warning([path.name]), str(warning))
        assert (warning.operation, warning.diagnostic, warning.value) == (
            "import_clones",
            "rowsPerFile",
            [0],
        )
    assert refs[0] == refs[1]
    loaded = shared.store.clones.load_clones(refs[0])
    assert loaded.clones.empty
    assert loaded.cells.index.tolist() == TRUTH.cell_ids.tolist()
    assert loaded.cells["clone"].isna().all()
    assert (loaded.cells["clone_size"] == 0).all()
    # A format that gives probabilities saves one per cell, none known.
    assert ("clone_probability" in loaded.cells) is with_probability
    if with_probability:
        assert loaded.cells["clone_probability"].isna().all()
    diagnostics = shared.store.artifacts.inspect(refs[0]).diagnostics
    assert (diagnostics["fileRows"], diagnostics["cellsWithClone"]) == (0, 0)
    assert diagnostics["rowsPerFile"] == [0]
    assert_joined(diagnostics, n_file=0)


def test_probabilities_that_no_selected_cell_takes_are_saved_unknown(
    store, tmp_path
) -> None:
    d2 = TRUTH.samples == "d2"
    store.cells.insert("only_d2", by_cell_id(store.cells, d2))
    # Numbat's one row names a cell of d4, outside the selection: the import
    # read a row, so it gives no warning, and no selected cell takes the row's
    # probability.
    outside = str(TRUTH.cell_ids[np.flatnonzero(~d2)[0]])
    numbat = _write_lines(
        tmp_path / "clone_post_2.tsv",
        f"cell\tclone_opt\tGT_opt\tp_opt\n{outside}\t2\t1a\t0.9\n",
    )
    with collect_warnings() as messages:
        ref = store.clones.import_clones(
            numbat, cell_selection=store.snapshot_cell_selection("only_d2")
        )
    assert messages == []
    diagnostics = store.artifacts.inspect(ref).diagnostics
    assert (diagnostics["fileRows"], diagnostics["cellsWithClone"]) == (1, 0)
    cells = store.clones.load_clones(ref).cells
    assert cells.index.tolist() == TRUTH.cell_ids[d2].tolist()
    assert cells["clone"].isna().all()
    assert cells["clone_probability"].isna().all()
    # TREX's rows beside a Numbat header alone keep TREX's clones, no cell
    # has a probability, and the warning names the header.
    first, second, third = (str(cell) for cell in TRUTH.cell_ids[:3])
    trex = _write_lines(
        tmp_path / "trex.txt",
        f"clone_nr\tcell_id\n1\t{first}\n1\t{second}\n2\t{third}\n",
    )
    header = _write_lines(tmp_path / "numbat.tsv", "cell\tclone_opt\tGT_opt\tp_opt\n")
    with collect_warnings() as messages:
        alone = store.clones.import_clones(trex, cell_selection=every_cell(store))
    assert messages == []
    with pytest.warns(DiagnosticWarning) as caught:
        both = store.clones.import_clones(
            [trex, header], cell_selection=every_cell(store)
        )
    assert len(caught) == 1
    assert re.fullmatch(_empty_files_warning(["numbat.tsv"], 2), str(caught[0].message))
    assert store.artifacts.inspect(both).diagnostics["rowsPerFile"] == _rows_per_file(
        store, both, {trex: 3, header: 0}
    )
    partition = {frozenset({first, second}), frozenset({third})}
    assert loaded_partition(store, alone) == loaded_partition(store, both) == partition
    loaded = store.clones.load_clones(both)
    # Several files' labels start with their names.
    assert loaded.clones["clone"].tolist() == ["trex_1", "trex_2"]
    assert len(loaded.cells) == N_CELLS
    assert loaded.cells["clone_probability"].isna().all()
    assert "clone_probability" not in store.clones.load_clones(alone).cells


def test_only_an_import_that_reads_no_row_warns_of_its_files(shared, tmp_path) -> None:
    store, cells = shared.store, shared.cells
    trex = _write_lines(tmp_path / "trex.txt", "clone_nr\tcell_id\n")
    cospar = _write_lines(tmp_path / "cospar.csv", "Cell_ID,Clone_ID\n")
    # The warning names every file of the import, and the summaries name the
    # files of the import that read no row.
    with pytest.warns(DiagnosticWarning) as caught:
        empty = store.clones.import_clones([trex, cospar], cell_selection=cells)
    assert len(caught) == 1
    assert re.fullmatch(
        _no_row_warning(["trex.txt", "cospar.csv"]), str(caught[0].message)
    )
    with pytest.raises(
        ValueError,
        match="their import read no row from the clone files; import files that "
        "list the cells' clones$",
    ):
        store.clones.expansion(empty)
    # An empty file beside one row of a selected cell is named alone, again
    # when the import is reused, and the clones are the row's.
    first = TRUTH.cell_ids[0]
    one = _write_lines(tmp_path / "one.csv", f"Cell_ID,Clone_ID\n{first},A\n")
    for _call in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            beside = store.clones.import_clones([trex, one], cell_selection=cells)
        assert [type(item.message) for item in caught] == [DiagnosticWarning]
        warning = caught[0].message
        assert isinstance(warning, DiagnosticWarning)
        assert re.fullmatch(_empty_files_warning(["trex.txt"], 2), str(warning))
        expected = _rows_per_file(store, beside, {trex: 0, one: 1})
        assert (warning.diagnostic, warning.value) == ("rowsPerFile", expected)
    assert loaded_partition(store, beside) == {frozenset({first})}
    # Two empty files of three are named in one warning.
    with pytest.warns(DiagnosticWarning) as caught:
        three = store.clones.import_clones([trex, cospar, one], cell_selection=cells)
    assert len(caught) == 1
    named = re.fullmatch(
        "The import read no row from the clone files '([^']*)', '([^']*)', 2 of "
        "its 3 clone files, so those files give no cell a clone. Pass the files "
        "you meant in their place, or leave them out.",
        str(caught[0].message),
    )
    assert named is not None
    assert {Path(name).name for name in named.groups()} == {"trex.txt", "cospar.csv"}
    assert store.artifacts.inspect(three).diagnostics["rowsPerFile"] == _rows_per_file(
        store, three, {trex: 0, cospar: 0, one: 1}
    )
    # A row of no cell of the store is named only by the barcodes' warning,
    # and a row of a cell outside the selection gives no warning.
    stray = _write_lines(tmp_path / "stray.csv", "Cell_ID,Clone_ID\nstray,A\n")
    with collect_warnings() as messages, pytest.warns(DiagnosticWarning):
        store.clones.import_clones(stray, cell_selection=cells)
    assert len(messages) == 1
    assert messages[0].startswith("1 of 1 barcodes of the clone file match no cell")
    store.cells.insert(
        "not_first",
        by_cell_id(store.cells, TRUTH.cell_ids != first),
        overwrite=True,
    )
    with collect_warnings() as messages:
        store.clones.import_clones(
            one, cell_selection=store.snapshot_cell_selection("not_first")
        )
    assert messages == []


def test_the_warnings_name_five_files_and_count_the_others(shared, tmp_path) -> None:
    store, cells = shared.store, shared.cells
    # Seven tables of a header alone, told apart by their prefixes, which
    # order them too.
    paths = [
        _write_lines(tmp_path / f"e{index}.csv", "Cell_ID,Clone_ID\n")
        for index in range(7)
    ]
    prefixes = [f"p{index}_" for index in range(7)]
    with pytest.warns(DiagnosticWarning) as caught:
        empty = store.clones.import_clones(
            paths, cell_selection=cells, barcode_prefix=prefixes
        )
    assert len(caught) == 1
    first_five = ", ".join(f"'[^']*e{index}\\.csv'" for index in range(5))
    assert re.fullmatch(
        f"The import read no row from the clone files {first_five} and 2 more, so "
        "no selected cell has a clone\\. Pass files that list the cells' clones\\.",
        str(caught[0].message),
    )
    # The saved counts keep every file.
    assert store.artifacts.inspect(empty).diagnostics["rowsPerFile"] == [0] * 7
    # Five of them are named, with none more.
    with pytest.warns(DiagnosticWarning) as caught:
        store.clones.import_clones(
            paths[:5], cell_selection=cells, barcode_prefix=prefixes[:5]
        )
    assert len(caught) == 1
    assert re.fullmatch(
        f"The import read no row from the clone files {first_five}, so no selected "
        "cell has a clone\\. Pass files that list the cells' clones\\.",
        str(caught[0].message),
    )
    # Six of them beside a file with a row.
    one = _write_lines(
        tmp_path / "one.csv", f"Cell_ID,Clone_ID\n{TRUTH.cell_ids[0]},A\n"
    )
    with pytest.warns(DiagnosticWarning) as caught:
        beside = store.clones.import_clones(
            [*paths[:6], one],
            cell_selection=cells,
            barcode_prefix=[*prefixes[:6], None],
        )
    assert len(caught) == 1
    assert re.fullmatch(
        f"The import read no row from the clone files {first_five} and 1 more, 6 of "
        "its 7 clone files, so those files give no cell a clone\\. Pass the files "
        "you meant in their place, or leave them out\\.",
        str(caught[0].message),
    )
    assert store.artifacts.inspect(beside).diagnostics["rowsPerFile"] == _rows_per_file(
        store, beside, {**dict.fromkeys(paths[:6], 0), one: 1}
    )


def _clone_summaries(store, clones: ArtifactRef) -> dict[str, Callable[[], object]]:
    """Every clone summary of some clones of the truth's store, as a call.

    Each takes options that the truth's clones pass: groups of samples, or of
    cell types where pairs of groups are compared, and fate bias follows the
    cells of d2 to the cell types of d4's cells.
    """
    for sample in ("d2", "d4"):
        store.cells.insert(
            f"is_{sample}",
            by_cell_id(store.cells, TRUTH.samples == sample),
            overwrite=True,
        )
    early = store.snapshot_cell_selection("is_d2")
    late = store.snapshot_cell_selection("is_d4")
    return {
        "expansion": lambda: store.clones.expansion(clones, group_by="sample"),
        "expansion of d2": lambda: store.clones.expansion(clones, cell_selection=early),
        "diversity": lambda: store.clones.diversity(clones, group_by="sample"),
        "rarefaction": lambda: store.clones.rarefaction(clones, group_by="sample"),
        "overlap": lambda: store.clones.overlap(clones, group_by="cell_type"),
        "track": lambda: store.clones.track(clones, group_by="sample", top_n=3),
        "track of labels": lambda: store.clones.track(clones, labels=["1"]),
        "coupling": lambda: store.clones.coupling(
            clones, group_by="cell_type", n_permutations=0
        ),
        "fate_bias": lambda: store.clones.fate_bias(
            clones, group_by="cell_type", early=early, late=late
        ),
    }


def test_summaries_refuse_clones_whose_import_read_no_row(store, tmp_path) -> None:
    path = _write_lines(tmp_path / "clones.csv", "Cell_ID,Clone_ID\n")
    with pytest.warns(DiagnosticWarning, match="read no row"):
        empty = store.clones.import_clones(path, cell_selection=every_cell(store))
    summaries = _clone_summaries(store, empty)
    saved = store.artifacts.list(scope="datastore")

    for name, summary in summaries.items():
        with pytest.raises(ValueError, match=_NO_ROW_REFUSAL):
            summary()
        assert store.artifacts.list(scope="datastore") == saved, name
    # The export refuses them in either format before it writes a file.
    out = _folder(tmp_path, "out")
    for format, name in (("cospar", "clones.csv"), ("matrix", "clones.mtx")):
        with pytest.raises(ValueError, match=_NO_ROW_REFUSAL):
            store.clones.export_clones(empty, out / name, format=format)
    assert list(out.iterdir()) == []
    # load_clones reads them, and the abundance figure says why it draws none.
    assert store.clones.load_clones(empty).clones.empty
    with pytest.raises(
        ValueError, match="^No cell has a clone, so there is nothing to plot$"
    ):
        store.plots.clone_abundance(empty, show=False)


def test_summaries_of_clones_of_no_selected_cell_keep_their_refusals(
    store, tmp_path
) -> None:
    d2 = TRUTH.samples == "d2"
    store.cells.insert("only_d2", by_cell_id(store.cells, d2))
    # The file's one row names a cell of d4, so the clones of d2's cells hold
    # none of its rows; and a column without labels saves clones of no cell
    # that record no rows.
    cell = TRUTH.cell_ids[np.flatnonzero(~d2)[0]]
    path = _write_lines(tmp_path / "d4.csv", f"Cell_ID,Clone_ID\n{cell},1\n")
    with collect_warnings() as messages:
        elsewhere = store.clones.import_clones(
            path, cell_selection=store.snapshot_cell_selection("only_d2")
        )
    assert messages == []
    assert store.artifacts.inspect(elsewhere).diagnostics["fileRows"] == 1
    store.cells.insert(
        "no_lineage", by_cell_id(store.cells, np.full(N_CELLS, "", dtype=object))
    )
    unlabeled = store.clones.from_column("no_lineage", cell_selection=every_cell(store))

    for source, clones in (("elsewhere", elsewhere), ("unlabeled", unlabeled)):
        assert store.artifacts.inspect(clones).diagnostics["cellsWithClone"] == 0
        for name, summary in _clone_summaries(store, clones).items():
            refusal = (
                f"^None of the {int(d2.sum())} cells of early has a clone; pass "
                "early cells that the clones give a clone, or clones imported on "
                "the early cells too$"
                if name == "fate_bias"
                else "^No cell has both a clone and a group$"
            )
            with pytest.raises(ValueError, match=refusal):
                summary()
        # Their export writes the header alone, as before.
        out = tmp_path / f"{source}.csv"
        assert store.clones.export_clones(clones, out).rows == 0
        assert out.read_text() == "Cell_ID,Clone_ID\n"


def test_one_row_is_enough_for_the_clone_summaries(store, tmp_path) -> None:
    d2 = TRUTH.samples == "d2"
    cell = str(TRUTH.cell_ids[np.flatnonzero(d2)[0]])
    path = _write_lines(tmp_path / "one.csv", f"Cell_ID,Clone_ID\n{cell},1\n")
    with collect_warnings() as messages:
        clones = store.clones.import_clones(path, cell_selection=every_cell(store))
    assert messages == []
    summaries = _clone_summaries(store, clones)

    # The one cell is the one clone of one cell, in d2.
    for name in ("expansion", "expansion of d2"):
        expansion = store.clones.load_expansion(summaries[name]())
        assert expansion.cells["category"].dropna().astype(str).to_dict() == {
            cell: "1 cell"
        }
    indices = summaries["diversity"]().indices
    assert indices[
        ["group", "n_cells", "n_clones", "n_singletons"]
    ].values.tolist() == [["d2", 1, 1, 1]]
    curves = summaries["rarefaction"]().curves
    observed = curves[curves["method"] == "observed"]
    assert observed[["group", "order", "size", "diversity"]].values.tolist() == [
        ["d2", order, 1, 1.0] for order in (0, 1, 2)
    ]
    for name, group in (("track", "d2"), ("track of labels", "all cells")):
        table = summaries[name]().table
        assert table.values.tolist() == [["1", group, 1, 1.0]], name
    # The summaries that compare groups, and fate bias, which follows early
    # cells to late ones, refuse the one cell for their own reasons.
    for name, refusal in (
        ("overlap", "Overlap needs at least two groups of cells with a clone"),
        ("coupling", "Coupling needs at least two groups of cells with a clone"),
        (
            "fate_bias",
            f"None of the {int((~d2).sum())} cells of late has a clone; pass late "
            "cells that the clones give a clone",
        ),
    ):
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            summaries[name]()


def _write_mito_unassigned(path: Path) -> Path:
    import anndata

    obs = pd.DataFrame(
        {"MiTo_clone": pd.Categorical(["unassigned"] * len(LISTED))},
        index=pd.Index(LISTED, dtype=str),
    )
    anndata.AnnData(obs=obs).write_h5ad(path)
    return path


def _write_na_labels(path: Path, cells: list[str]) -> Path:
    """Write CoSpar's clone table of an NA label for each of ``cells``."""
    return _write_lines(
        path, "Cell_ID,Clone_ID\n" + "".join(f"{cell},NA\n" for cell in cells)
    )


def _write_cospar_h5ad_without_entries(path: Path) -> Path:
    import anndata
    from scipy.sparse import csr_matrix

    anndata.AnnData(
        obs=pd.DataFrame(index=pd.Index(LISTED, dtype=str)),
        obsm={"X_clone": csr_matrix((len(LISTED), 3), dtype=np.float32)},
    ).write_h5ad(path)
    return path


def _write_matrix_without_entries(folder: Path) -> Path:
    _write_lines(folder / "cells.tsv", "cell\n" + "".join(f"{c}\n" for c in LISTED))
    return _write_lines(
        folder / "clones.mtx",
        f"%%MatrixMarket matrix coordinate integer general\n{len(LISTED)} 3 0\n",
    )


# Files whose rows name the ten cells but give none of them a clone, with how
# the warning says so: MiTo's cells left unassigned, Cassiopeia's rows of
# lineage group 0, two for each cell, and CoSpar's table of NA labels label
# the cells with no clone, and a clone matrix that lists the cells and
# CoSpar's AnnData clone matrix hold no entry for them.
_CLONELESS: dict[str, tuple[Callable[[Path], Path], str]] = {
    "mito_unassigned": (
        lambda folder: _write_mito_unassigned(folder / "afm.h5ad"),
        "labels them with no clone",
    ),
    "cassiopeia_group_0": (
        lambda folder: _write_lines(
            folder / "alleles.tsv",
            "cellBC\tintBC\tallele\tlineageGrp\n"
            + "".join(f"{c}\t{i}\tx\t0\n" for c in LISTED for i in ("AAAA", "CCCC")),
        ),
        "labels them with no clone",
    ),
    "matrix_without_entries": (
        _write_matrix_without_entries,
        "holds no entry for them",
    ),
    "cospar_h5ad_without_entries": (
        lambda folder: _write_cospar_h5ad_without_entries(folder / "clones.h5ad"),
        "holds no entry for them",
    ),
    "cospar_na": (
        lambda folder: _write_na_labels(folder / "clones.csv", LISTED),
        "labels them with no clone",
    ),
}

#: The refusal of the clones of one table whose rows label the ten cells with no clone.
_NO_CLONE_REFUSAL = (
    "^The clones give no cell a clone: their clone file names 10 selected cells "
    "but labels them with no clone; import a file that lists the cells' clones$"
)


@pytest.mark.parametrize("case", sorted(_CLONELESS))
def test_files_whose_rows_give_no_selected_cell_a_clone_warn_naming_them(
    shared, tmp_path, case
) -> None:
    write, cause = _CLONELESS[case]
    path = write(_folder(tmp_path, case))

    refs = []
    for _call in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refs.append(
                shared.store.clones.import_clones(path, cell_selection=shared.cells)
            )
        assert [type(item.message) for item in caught] == [DiagnosticWarning]
        warning = caught[0].message
        assert isinstance(warning, DiagnosticWarning)
        # The selection holds every cell, so the warning suggests no other.
        assert re.fullmatch(
            f"The clone file '[^']*{re.escape(path.name)}' names 10 selected cells "
            rf"but {cause}\. Pass a file that lists the cells' clones\.",
            str(warning),
        )
        assert (warning.operation, warning.diagnostic, warning.value) == (
            "import_clones",
            "cellsWithClone",
            0,
        )
    assert refs[0] == refs[1]
    diagnostics = shared.store.artifacts.inspect(refs[0]).diagnostics
    assert (diagnostics["cellsWithClone"], diagnostics["cellsInSeveralClones"]) == (
        0,
        0,
    )
    assert_joined(diagnostics, n_file=10)
    # Cassiopeia's table holds two rows for each cell.
    assert diagnostics["rowsPerFile"] == [20 if case == "cassiopeia_group_0" else 10]
    assert shared.store.clones.load_clones(refs[0]).clones.empty
    # The summaries name the same cause.
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone file names 10 selected "
        f"cells but {cause}; import a file that lists the cells' clones$",
    ):
        shared.store.clones.expansion(refs[0])


def test_summaries_and_the_export_name_files_whose_rows_give_no_clone(
    store, tmp_path
) -> None:
    write, cause = _CLONELESS["cospar_na"]
    path = write(tmp_path)
    with pytest.warns(DiagnosticWarning, match=cause):
        clones = store.clones.import_clones(path, cell_selection=every_cell(store))
    summaries = _clone_summaries(store, clones)
    saved = store.artifacts.list(scope="datastore")

    for name, summary in summaries.items():
        with pytest.raises(ValueError, match=_NO_CLONE_REFUSAL):
            summary()
        assert store.artifacts.list(scope="datastore") == saved, name
    out = _folder(tmp_path, "out")
    for format, name in (("cospar", "clones.csv"), ("matrix", "clones.mtx")):
        with pytest.raises(ValueError, match=_NO_CLONE_REFUSAL):
            store.clones.export_clones(clones, out / name, format=format)
    assert list(out.iterdir()) == []
    assert store.clones.load_clones(clones).clones.empty
    with pytest.raises(
        ValueError, match="^No cell has a clone, so there is nothing to plot$"
    ):
        store.plots.clone_abundance(clones, show=False)
    # Files of no clone, of which one holds no row: the warning of the empty
    # file names it, that of no clone names the files that hold rows, and the
    # summaries name the import's files.
    first = write(_folder(tmp_path, "first"))
    second = _write_na_labels(_folder(tmp_path, "second") / "more.csv", LISTED[:4])
    empty = _write_lines(tmp_path / "empty.csv", "Cell_ID,Clone_ID\n")
    with pytest.warns(DiagnosticWarning) as caught:
        several = store.clones.import_clones(
            [first, second, empty], cell_selection=every_cell(store)
        )
    messages = sorted(str(item.message) for item in caught)
    assert len(messages) == 2
    assert re.fullmatch(_empty_files_warning(["empty.csv"], 3), messages[1])
    named = re.fullmatch(
        "The clone files '([^']*)', '([^']*)' name 10 selected cells but label "
        "them with no clone. Pass files that list the cells' clones.",
        messages[0],
    )
    assert named is not None
    assert {Path(name).name for name in named.groups()} == {"clones.csv", "more.csv"}
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone files name 10 selected "
        "cells but label them with no clone; import files that list the cells' "
        "clones$",
    ):
        store.clones.expansion(several)


def test_the_cause_of_no_clone_fits_the_layouts_of_the_files_that_hold_rows(
    shared, tmp_path
) -> None:
    store, cells = shared.store, shared.cells
    matrix = _write_matrix_without_entries(_folder(tmp_path, "matrix"))
    labels = _write_na_labels(tmp_path / "labels.csv", LISTED[:4])
    # A matrix without entries beside a table of NA labels gives the cells
    # no clone in two ways, which the warning and the summaries both name.
    with pytest.warns(DiagnosticWarning) as caught:
        mixed = store.clones.import_clones([matrix, labels], cell_selection=cells)
    assert len(caught) == 1
    named = re.fullmatch(
        "The clone files '([^']*)', '([^']*)' name 10 selected cells but give them "
        "no clone, as no matrix holds an entry for them and no other file labels "
        "them with a clone. Pass files that list the cells' clones.",
        str(caught[0].message),
    )
    assert named is not None
    assert {Path(name).name for name in named.groups()} == {"clones.mtx", "labels.csv"}
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone files name 10 selected "
        "cells but give them no clone, as no matrix holds an entry for them and no "
        "other file labels them with a clone; import files that list the cells' "
        "clones$",
    ):
        store.clones.expansion(mixed)
    # Beside a table of a header alone, the matrix is the one file that holds
    # rows, so the warning and the refusal name the matrix's cause and count
    # one file.
    empty = _write_lines(tmp_path / "empty.csv", "Cell_ID,Clone_ID\n")
    with pytest.warns(DiagnosticWarning) as caught:
        beside = store.clones.import_clones([matrix, empty], cell_selection=cells)
    messages = sorted(str(item.message) for item in caught)
    assert len(messages) == 2
    assert re.fullmatch(
        "The clone file '[^']*clones\\.mtx' names 10 selected cells but holds no "
        "entry for them\\. Pass a file that lists the cells' clones\\.",
        messages[0],
    )
    assert re.fullmatch(_empty_files_warning(["empty.csv"], 2), messages[1])
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone file names 10 selected "
        "cells but holds no entry for them; import a file that lists the cells' "
        "clones$",
    ):
        store.clones.expansion(beside)
    # One selected cell is named as one.
    one = _write_na_labels(tmp_path / "one.csv", LISTED[:1])
    with pytest.warns(DiagnosticWarning) as caught:
        single = store.clones.import_clones(one, cell_selection=cells)
    assert len(caught) == 1
    assert re.fullmatch(
        "The clone file '[^']*one\\.csv' names 1 selected cell but labels it with "
        "no clone\\. Pass a file that lists the cells' clones\\.",
        str(caught[0].message),
    )
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone file names 1 selected "
        "cell but labels it with no clone; import a file that lists the cells' "
        "clones$",
    ):
        store.clones.expansion(single)


def test_clones_that_some_listed_cell_takes_or_that_drop_left_out_give_no_warning(
    store, tmp_path
) -> None:
    # One labeled cell among unassigned ones is enough for the summaries.
    labeled = _write_lines(
        tmp_path / "one_label.csv",
        "Cell_ID,Clone_ID\n"
        + "".join(f"{c},{'A' if c == LISTED[0] else 'NA'}\n" for c in LISTED),
    )
    with collect_warnings() as messages:
        clones = store.clones.import_clones(labeled, cell_selection=every_cell(store))
    assert messages == []
    expansion = store.clones.load_expansion(store.clones.expansion(clones))
    assert expansion.cells["category"].dropna().astype(str).to_dict() == {
        LISTED[0]: "1 cell"
    }
    # Every listed cell in two clones, which multiple="drop" leaves out: the
    # file gave clones, so the import gives no warning, and the summaries and
    # the export name the cells that drop left out.
    doubled = _write_lines(
        tmp_path / "doubled.csv",
        "Cell_ID,Clone_ID\n" + "".join(f"{c},A\n{c},B\n" for c in LISTED[:3]),
    )
    with collect_warnings() as messages:
        dropped = store.clones.import_clones(
            doubled, cell_selection=every_cell(store), multiple="drop"
        )
    assert messages == []
    diagnostics = store.artifacts.inspect(dropped).diagnostics
    assert (diagnostics["cellsWithClone"], diagnostics["cellsInSeveralClones"]) == (
        0,
        3,
    )
    out = tmp_path / "dropped.csv"
    for call in (
        lambda: store.clones.expansion(dropped),
        lambda: store.clones.diversity(dropped),
        lambda: store.clones.export_clones(dropped, out),
    ):
        with pytest.raises(
            ValueError,
            match="^The clones give no cell a clone: multiple='drop' left out the 3 "
            "cells that their clone file lists in several clones; import it with "
            "multiple='error' to name such cells$",
        ):
            call()
    assert not out.exists()
    # A cell that two files list in two clones is the files' one cell left out.
    first = _write_lines(tmp_path / "a.csv", f"Cell_ID,Clone_ID\n{LISTED[0]},A\n")
    second = _write_lines(tmp_path / "b.csv", f"Cell_ID,Clone_ID\n{LISTED[0]},B\n")
    with collect_warnings() as messages:
        across = store.clones.import_clones(
            [first, second], cell_selection=every_cell(store), multiple="drop"
        )
    assert messages == []
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: multiple='drop' left out the 1 "
        "cell that their clone files list in several clones; import them with "
        "multiple='error' to name such cells$",
    ):
        store.clones.expansion(across)
    # multiple="error" names such a cell, as the refusal says, as one.
    with pytest.raises(
        ValueError,
        match=f"^1 cell is listed in two different clones, '{LISTED[0]}' in 'a_A' "
        "and 'b_B'\\. Pass multiple='drop' to leave it without a clone$",
    ):
        store.clones.import_clones([first, second], cell_selection=every_cell(store))


def test_files_that_give_clones_only_to_cells_outside_the_selection_suggest_them(
    store, tmp_path
) -> None:
    d2 = TRUTH.samples == "d2"
    store.cells.insert("only_d2", by_cell_id(store.cells, d2))
    store.cells.insert("only_d4", by_cell_id(store.cells, ~d2))
    early = store.snapshot_cell_selection("only_d2")
    n_early, n_late = int(d2.sum()), int((~d2).sum())
    late_cells = [str(cell) for cell in TRUTH.cell_ids[~d2]]
    # The selected cells of d2 have NA labels, and the cells of d4, outside
    # the selection, a clone; the warning counts the selected cells alone and
    # suggests the others if the file gives them clones.
    path = _write_lines(
        tmp_path / "every.csv",
        "Cell_ID,Clone_ID\n"
        + "".join(
            f"{cell},{'NA' if is_early else 'A'}\n"
            for cell, is_early in zip(TRUTH.cell_ids, d2, strict=True)
        ),
    )
    with pytest.warns(DiagnosticWarning) as caught:
        clones = store.clones.import_clones(path, cell_selection=early)
    assert len(caught) == 1
    assert re.fullmatch(
        f"The clone file '[^']*every\\.csv' names {n_early} selected cells but "
        "labels them with no clone\\. Pass a file that lists the cells' clones, or, "
        f"if it gives clones to the {n_late} cells outside the selection that it "
        "names, a cell_selection of those cells\\.",
        str(caught[0].message),
    )
    with pytest.raises(
        ValueError,
        match=f"^The clones give no cell a clone: their clone file names {n_early} "
        "selected cells but labels them with no clone; import a file that lists "
        f"the cells' clones, or, if it gives clones to the {n_late} cells outside "
        "the selection that it names, import it on a cell_selection of those cells$",
    ):
        store.clones.expansion(clones)
    # The cells to which the file gives a clone take it, without a warning.
    with collect_warnings() as messages:
        late = store.clones.import_clones(
            path, cell_selection=store.snapshot_cell_selection("only_d4")
        )
    assert messages == []
    assert loaded_partition(store, late) == {frozenset(late_cells)}
    # Two files, one of the selected cells' NA labels and one of clones of
    # cells outside the selection, are named together.
    labels = _write_na_labels(
        tmp_path / "labels.csv", [str(cell) for cell in TRUTH.cell_ids[d2]]
    )
    others = _write_lines(
        tmp_path / "others.csv",
        "Cell_ID,Clone_ID\n" + "".join(f"{cell},A\n" for cell in late_cells),
    )
    with pytest.warns(DiagnosticWarning) as caught:
        both = store.clones.import_clones([labels, others], cell_selection=early)
    assert len(caught) == 1
    named = re.fullmatch(
        f"The clone files '([^']*)', '([^']*)' name {n_early} selected cells but "
        "label them with no clone. Pass files that list the cells' clones, or, if "
        f"they give clones to the {n_late} cells outside the selection that they "
        "name, a cell_selection of those cells.",
        str(caught[0].message),
    )
    assert named is not None
    assert {Path(name).name for name in named.groups()} == {"labels.csv", "others.csv"}
    with pytest.raises(
        ValueError,
        match=f"^The clones give no cell a clone: their clone files name {n_early} "
        "selected cells but label them with no clone; import files that list the "
        f"cells' clones, or, if they give clones to the {n_late} cells outside the "
        "selection that they name, import them on a cell_selection of those cells$",
    ):
        store.clones.expansion(both)
    # Clones of cells outside the store name no cell outside the selection,
    # so neither the warning nor the refusal suggests a selection.
    stray = _write_lines(tmp_path / "stray.csv", "Cell_ID,Clone_ID\nstray-1,A\n")
    with pytest.warns(DiagnosticWarning) as caught:
        unmatched = store.clones.import_clones([labels, stray], cell_selection=early)
    messages = sorted(str(item.message) for item in caught)
    assert len(messages) == 2
    assert messages[0].startswith("1 of ")
    named = re.fullmatch(
        f"The clone files '([^']*)', '([^']*)' name {n_early} selected cells but "
        "label them with no clone. Pass files that list the cells' clones.",
        messages[1],
    )
    assert named is not None
    assert {Path(name).name for name in named.groups()} == {"labels.csv", "stray.csv"}
    with pytest.raises(
        ValueError,
        match=f"^The clones give no cell a clone: their clone files name {n_early} "
        "selected cells but label them with no clone; import files that list the "
        "cells' clones$",
    ):
        store.clones.expansion(unmatched)
    # One selected cell beside one cell outside the selection are each named
    # as one.
    store.cells.insert(
        "all_but_first", by_cell_id(store.cells, TRUTH.cell_ids != LISTED[0])
    )
    pair = _write_na_labels(tmp_path / "pair.csv", LISTED[:2])
    with pytest.warns(DiagnosticWarning) as caught:
        single = store.clones.import_clones(
            pair, cell_selection=store.snapshot_cell_selection("all_but_first")
        )
    assert len(caught) == 1
    assert re.fullmatch(
        "The clone file '[^']*pair\\.csv' names 1 selected cell but labels it with "
        "no clone\\. Pass a file that lists the cells' clones, or, if it gives a "
        "clone to the cell outside the selection that it names, a cell_selection of "
        "that cell\\.",
        str(caught[0].message),
    )
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone file names 1 selected "
        "cell but labels it with no clone; import a file that lists the cells' "
        "clones, or, if it gives a clone to the cell outside the selection that it "
        "names, import it on a cell_selection of that cell$",
    ):
        store.clones.expansion(single)


def test_an_import_saved_before_its_rows_per_file_still_reads_and_is_imported_anew(
    store, tmp_path
) -> None:
    first, second, third = LISTED[:3]
    path = _write_lines(
        tmp_path / "clones.csv",
        f"Cell_ID,Clone_ID\n{first},A\n{second},A\n{third},B\n",
    )
    saved = store.clones.import_clones(path, cell_selection=every_cell(store))
    status = rewrite_as_saved_by_1_4(store, saved, revision=1, without="rowsPerFile")
    assert (status.revision, status.current_revision, status.is_current) == (
        1,
        2,
        False,
    )
    assert status.diagnostics["rowsPerFile"] is None

    assert store.artifacts.inspect(saved).diagnostics["fileRows"] == 3
    # It still serves the calls that pass it.
    partition = {frozenset({first, second}), frozenset({third})}
    assert loaded_partition(store, saved) == partition
    expansion = store.clones.load_expansion(store.clones.expansion(saved))
    assert expansion.cells["category"].dropna().astype(str).to_dict() == {
        first: "2 cells",
        second: "2 cells",
        third: "1 cell",
    }
    # The import is computed anew, which the log explains, and counts the
    # file's rows.
    (revision,) = OPERATION_REVISIONS["import_clones"]
    again = computed_anew(
        store,
        saved,
        lambda: store.clones.import_clones(path, cell_selection=every_cell(store)),
        operation="import_clones",
        revision=1,
        current=2,
        change=revision.change,
    )
    assert store.artifacts.inspect(again).diagnostics["rowsPerFile"] == [3]
    assert loaded_partition(store, again) == partition
    # The summaries name the cause of a clone matrix without entries saved so,
    # from the layout of its one file.
    matrix = _write_matrix_without_entries(_folder(tmp_path, "matrix"))
    with pytest.warns(DiagnosticWarning, match="holds no entry for them"):
        unlisted = store.clones.import_clones(matrix, cell_selection=every_cell(store))
    rewrite_as_saved_by_1_4(store, unlisted, revision=1, without="rowsPerFile")
    with pytest.raises(
        ValueError,
        match="^The clones give no cell a clone: their clone file names 10 selected "
        "cells but holds no entry for them; import a file that lists the cells' "
        "clones$",
    ):
        store.clones.expansion(unlisted)


def test_an_import_of_this_revision_whose_rows_per_file_read_none_is_imported_anew(
    store, tmp_path
) -> None:
    first, second = LISTED[:2]
    path = _write_lines(
        tmp_path / "clones.csv", f"Cell_ID,Clone_ID\n{first},A\n{second},A\n"
    )
    saved = store.clones.import_clones(path, cell_selection=every_cell(store))
    group = artifact_group(store.zw, saved)
    group.attrs["diagnostics"] = {**group.attrs["diagnostics"], "rowsPerFile": None}

    # Only an import saved before its rows per file reads them as None, so
    # this one is refused, and the same call imports it anew.
    with pytest.raises(
        ValueError,
        match="The diagnostic 'rowsPerFile' of import_clones must be a list of "
        "counts, one for each file, not None. To read it, call "
        "ds.clones.import_clones again with the same arguments",
    ):
        store.artifacts.inspect(saved).diagnostics
    with collect_warnings() as messages:
        again = store.clones.import_clones(path, cell_selection=every_cell(store))
    assert messages == []
    assert again != saved
    assert store.artifacts.inspect(again).diagnostics["rowsPerFile"] == [2]
    assert loaded_partition(store, again) == {frozenset({first, second})}


def test_unicode_and_very_long_labels_survive_the_round_trip(store, tmp_path) -> None:
    cells = every_cell(store)
    long_label = "κλώνος-" + "x" * 5000
    labels = {0: long_label, 1: "克隆 one", 2: "clone, with comma"}
    rows = [
        (cell, labels.get(int(clone), f"c{clone}"))
        for cell, clone in zip(TRUTH.cell_ids, TRUTH.clones, strict=True)
        if clone >= 0
    ]
    path = tmp_path / "clones.csv"
    pd.DataFrame(rows, columns=["Cell_ID", "Clone_ID"]).to_csv(path, index=False)

    ref = store.clones.import_clones(path, cell_selection=cells)

    assert loaded_partition(store, ref) == TRUTH.partition()
    loaded = store.clones.load_clones(ref)
    assert loaded.clones["clone"].tolist()[:3] == [
        long_label,
        "克隆 one",
        "clone, with comma",
    ]
    for format, name in (("cospar", "out.csv"), ("matrix", "out.mtx")):
        folder = _folder(tmp_path, format)
        store.clones.export_clones(ref, folder / name, format=format)
        again = store.clones.import_clones(folder / name, cell_selection=cells)
        assert store.clones.load_clones(again).cells.equals(loaded.cells)


def test_load_clones_reads_the_tables_largest_clone_first(shared, tmp_path) -> None:
    ref = shared.store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_2.tsv"), cell_selection=shared.cells
    )

    loaded = shared.store.clones.load_clones(ref)

    assert isinstance(loaded, Clones)
    assert loaded.source == "import_clones"
    assert loaded.cells.index.name == "ids"
    assert loaded.cells.index.tolist() == TRUTH.cell_ids.tolist()
    assert list(loaded.cells.columns) == ["clone", "clone_size", "clone_probability"]
    assert list(loaded.cells["clone"].cat.categories) == NUMBERED
    assert loaded.cells["clone_size"].dtype == np.int64
    assert loaded.clones.columns.tolist() == ["clone", "n_cells"]
    in_clone = TRUTH.clones >= 0
    sizes = np.asarray(CLONE_SIZES)[TRUTH.clones[in_clone]]
    np.testing.assert_array_equal(
        loaded.cells["clone_size"].to_numpy()[in_clone], sizes
    )
    assert (loaded.cells["clone_size"].to_numpy()[~in_clone] == 0).all()


def test_load_and_export_refuse_other_artifacts(shared, tmp_path) -> None:
    other = shared.store.snapshot_cluster_labels("sample", cell_selection=shared.cells)
    for method in (
        shared.store.clones.load_clones,
        lambda ref: shared.store.clones.export_clones(ref, tmp_path / "x.csv"),
    ):
        with pytest.raises(
            ValueError, match="clones must be a clones or clonotypes artifact"
        ):
            method(other)
        with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
            method("clones")
    ref = shared.store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_2.tsv"), cell_selection=shared.cells
    )
    with pytest.raises(ValueError, match="format must be one of cospar, matrix"):
        shared.store.clones.export_clones(ref, tmp_path / "x.csv", format="h5ad")
    with pytest.raises(TypeError, match="path must be a path"):
        shared.store.clones.export_clones(ref, 3)


def test_load_and_export_name_missing_and_incomplete_clones(
    shared, store, tmp_path
) -> None:
    def readers(target):
        return (
            target.clones.load_clones,
            lambda ref: target.clones.export_clones(ref, tmp_path / "out.csv"),
        )

    # Clones saved in one store are missing from another.
    elsewhere = shared.store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_2.tsv"), cell_selection=shared.cells
    )
    for read in readers(store):
        with pytest.raises(KeyError, match="Artifact does not exist"):
            read(elsewhere)
    # An import that stopped before it finished leaves an incomplete artifact.
    saved = store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_3.tsv"),
        cell_selection=every_cell(store),
    )
    artifact_group(store.zw, saved).attrs["complete"] = False
    for read in readers(store):
        with pytest.raises(RuntimeError, match="Artifact is incomplete"):
            read(saved)
    assert not (tmp_path / "out.csv").exists()


def test_load_refuses_tables_larger_than_the_budget(store, tmp_path) -> None:
    ref = store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_2.tsv"),
        cell_selection=every_cell(store),
    )
    store.memoryBytes = 1_000
    with pytest.raises(MemoryError, match="memory budget of 1000 bytes"):
        store.clones.load_clones(ref)
    with pytest.raises(MemoryError, match="memory budget of 1000 bytes"):
        store.clones.export_clones(ref, tmp_path / "out.csv")


@pytest.mark.parametrize("name", ["clones.csv", "clones.csv.gz"])
def test_the_exported_table_is_cospars_and_imports_back(shared, tmp_path, name) -> None:
    ref = shared.store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_2.tsv"), cell_selection=shared.cells
    )
    path = tmp_path / name

    written = shared.store.clones.export_clones(ref, path)

    assert written == ExportSummary(
        path=path, rows=N_IN_CLONES, diagnostics={"cells": N_IN_CLONES}
    )
    opened = gzip.open(path, "rt") if name.endswith(".gz") else open(path)
    with opened as handle:
        table = pd.read_csv(handle, dtype=str)
    assert table.columns.tolist() == ["Cell_ID", "Clone_ID"]
    expected = {
        (str(cell), str(clone + 1))
        for cell, clone in zip(TRUTH.cell_ids, TRUTH.clones, strict=True)
        if clone >= 0
    }
    assert set(map(tuple, table.to_numpy().tolist())) == expected
    again = shared.store.clones.import_clones(path, cell_selection=shared.cells)
    reloaded = shared.store.clones.load_clones(again)
    original = shared.store.clones.load_clones(ref)
    # Probabilities are not exported; the clones and their labels are.
    assert reloaded.cells.equals(original.cells.drop(columns="clone_probability"))
    assert reloaded.clones.equals(original.clones)


def test_the_exported_matrix_lists_its_cells_and_clones_and_imports_back(
    shared, tmp_path
) -> None:
    ref = shared.store.clones.import_clones(
        write_trex(TRUTH, tmp_path / "clones.txt"), cell_selection=shared.cells
    )
    folder = _folder(tmp_path, "matrix")

    written = shared.store.clones.export_clones(
        ref, folder / "clones.mtx.gz", format="matrix"
    )

    assert (written.path, written.rows) == (folder / "clones.mtx.gz", N_IN_CLONES)
    assert written.diagnostics == {"cells": N_IN_CLONES}
    with gzip.open(folder / "clones.mtx.gz", "rt") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "%%MatrixMarket matrix coordinate integer general"
    assert lines[2] == f"{N_IN_CLONES} {len(CLONE_SIZES)} {N_IN_CLONES}"
    cells = pd.read_csv(folder / "cells.tsv", sep="\t", dtype=str)["cell"].tolist()
    clones = pd.read_csv(folder / "clones.tsv", sep="\t", dtype=str)["clone"].tolist()
    assert clones == NUMBERED
    entries = [tuple(map(int, line.split())) for line in lines[3:]]
    assert all(value == 1 for _row, _column, value in entries)
    named = {(cells[row - 1], clones[column - 1]) for row, column, _ in entries}
    assert named == {
        (str(cell), str(clone + 1))
        for cell, clone in zip(TRUTH.cell_ids, TRUTH.clones, strict=True)
        if clone >= 0
    }
    again = shared.store.clones.import_clones(
        folder / "clones.mtx.gz", cell_selection=shared.cells
    )
    assert shared.store.clones.load_clones(again).cells.equals(
        shared.store.clones.load_clones(ref).cells
    )


@pytest.mark.parametrize(
    ("named", "refused"),
    [
        ({0: "NA", 1: "nan", 2: "NaN"}, "'NA', 'nan', 'NaN'"),
        # The tokens that every table reads as missing, which clone tables
        # read too; from_column strips a label, as a clone table's reader does.
        ({0: "None", 1: "NULL", 2: "N/A"}, "'None', 'NULL', 'N/A'"),
        ({0: " null", 1: "#N/A", 2: "<NA>"}, "'null', '#N/A', '<NA>'"),
    ],
)
def test_labels_that_a_clone_table_reads_as_no_clone_export_only_as_a_matrix(
    store, tmp_path, named, refused
) -> None:
    cells = every_cell(store)
    labels = np.asarray(
        [
            named.get(int(clone), f"L{clone}") if clone >= 0 else ""
            for clone in TRUTH.clones
        ],
        dtype=object,
    )
    store.cells.insert("lineage", by_cell_id(store.cells, labels))
    ref = store.clones.from_column("lineage", cell_selection=cells)

    # A clone table would read these labels back as cells without a clone.
    with pytest.raises(ValueError, match=f"{re.escape(refused)} read as no clone"):
        store.clones.export_clones(ref, tmp_path / "clones.csv")
    assert not (tmp_path / "clones.csv").exists()

    folder = _folder(tmp_path, "matrix")
    store.clones.export_clones(ref, folder / "clones.mtx", format="matrix")
    again = store.clones.import_clones(folder / "clones.mtx", cell_selection=cells)
    assert store.clones.load_clones(again).cells.equals(
        store.clones.load_clones(ref).cells
    )


def test_labels_that_hold_line_breaks_export_only_as_a_table(store, tmp_path) -> None:
    cells = every_cell(store)
    rows = [
        (cell, "two\nlines" if clone == 0 else f"c{clone}")
        for cell, clone in zip(TRUTH.cell_ids, TRUTH.clones, strict=True)
        if clone >= 0
    ]
    path = tmp_path / "clones.csv"
    pd.DataFrame(rows, columns=["Cell_ID", "Clone_ID"]).to_csv(path, index=False)
    ref = store.clones.import_clones(path, cell_selection=cells)

    # The matrix's clone list holds one label per line.
    folder = _folder(tmp_path, "matrix")
    with pytest.raises(ValueError, match=re.escape("'two\\nlines' holds a line break")):
        store.clones.export_clones(ref, folder / "clones.mtx", format="matrix")
    assert list(folder.iterdir()) == []

    store.clones.export_clones(ref, tmp_path / "out.csv")
    again = store.clones.import_clones(tmp_path / "out.csv", cell_selection=cells)
    assert store.clones.load_clones(again).cells.equals(
        store.clones.load_clones(ref).cells
    )


def test_editing_a_matrix_cell_list_imports_new_clones(store, tmp_path) -> None:
    # The matrix names its cells by row in cells.tsv, so the identity of its
    # import holds the bytes of that list too.
    cells = every_cell(store)
    ref = store.clones.import_clones(
        write_trex(TRUTH, tmp_path / "clones.txt"), cell_selection=cells
    )
    folder = _folder(tmp_path, "matrix")
    store.clones.export_clones(ref, folder / "clones.mtx", format="matrix")
    matrix = store.clones.import_clones(folder / "clones.mtx", cell_selection=cells)
    listed = (folder / "cells.tsv").read_text().splitlines()
    first, last = listed[1], listed[-1]
    listed[1], listed[-1] = last, first
    (folder / "cells.tsv").write_text("\n".join(listed) + "\n")

    edited = store.clones.import_clones(folder / "clones.mtx", cell_selection=cells)

    assert edited != matrix
    assert loaded_partition(store, edited) != loaded_partition(store, matrix)


def test_clones_come_from_a_cell_metadata_column(store) -> None:
    cells = every_cell(store)
    labels = np.where(
        TRUTH.clones >= 0, np.char.add("L", TRUTH.clones.astype(str)), ""
    ).astype(object)
    labels[np.flatnonzero(TRUTH.clones < 0)[:2]] = None
    store.cells.insert("lineage", by_cell_id(store.cells, labels))

    ref = store.clones.from_column("lineage", cell_selection=cells)

    assert ref.kind == "clones"
    assert loaded_partition(store, ref) == TRUTH.partition()
    loaded = store.clones.load_clones(ref)
    assert loaded.source == "import_clones_from_column"
    assert loaded.clones["n_cells"].tolist() == list(CLONE_SIZES)
    assert store.artifacts.inspect(ref).parameters == {"column": "lineage"}
    assert store.clones.from_column("lineage", cell_selection=cells) == ref
    # Editing the column gives new clones.
    edited = labels.copy()
    edited[np.flatnonzero(TRUTH.clones == 0)[0]] = "L7"
    store.cells.insert("lineage", by_cell_id(store.cells, edited), overwrite=True)
    changed = store.clones.from_column("lineage", cell_selection=cells)
    assert changed != ref
    assert store.clones.load_clones(changed).clones["n_cells"].tolist()[0] == 8


def test_numeric_label_columns_name_clones_by_whole_numbers(store) -> None:
    cells = every_cell(store)
    values = np.where(TRUTH.clones >= 0, TRUTH.clones + 10.0, np.nan)
    store.cells.insert("lineage_number", by_cell_id(store.cells, values))

    ref = store.clones.from_column("lineage_number", cell_selection=cells)

    assert store.clones.load_clones(ref).clones["clone"].tolist() == [
        str(clone + 10) for clone in range(len(CLONE_SIZES))
    ]
    store.cells.insert("fraction", by_cell_id(store.cells, np.full(N_CELLS, 0.5)))
    with pytest.raises(TypeError, match="text or whole numbers, not 0.5"):
        store.clones.from_column("fraction", cell_selection=cells)
    with pytest.raises(ValueError, match="'absent' is not a cell metadata column"):
        store.clones.from_column("absent", cell_selection=cells)
    with pytest.raises(TypeError, match="column must be the name"):
        store.clones.from_column(3, cell_selection=cells)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.clones.from_column("lineage_number", cell_selection="cells")
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.from_column(
            "lineage_number", cell_selection=cells, invalidate_cache=1
        )


def test_labels_of_a_column_larger_than_the_budget_are_refused(store) -> None:
    store.cells.insert(
        "lineage_text", by_cell_id(store.cells, TRUTH.clones.astype(str).astype(object))
    )
    store.memoryBytes = 1_000
    with pytest.raises(MemoryError, match="memory budget of 1000 bytes"):
        store.clones.from_column("lineage_text", cell_selection=every_cell(store))


def test_a_column_without_labels_saves_no_clone(store) -> None:
    store.cells.insert(
        "empty_lineage", by_cell_id(store.cells, np.full(N_CELLS, "", dtype=object))
    )
    ref = store.clones.from_column("empty_lineage", cell_selection=every_cell(store))
    loaded = store.clones.load_clones(ref)
    assert loaded.clones.empty
    assert (loaded.cells["clone_size"] == 0).all()
    assert isinstance(ref, ArtifactRef)
    assert math.isclose(store.artifacts.load(ref).attrs["n_clones"], 0)


# Cells of the long-label store, each in its own clone. The last label is
# 5,000 characters long, past the rows that a table's layout samples, and the
# store keeps text at the width of the longest, so each cell's label takes
# 20,000 bytes there: 40 MB in all.
_N_LONG = 2_000
_BUDGET = 16_000_000


def _long_label_store(tmp_path: Path) -> SimpleNamespace:
    ids = [f"cell{index:05d}" for index in range(_N_LONG)]
    store = write_cells_store(tmp_path / "store.zarr", ids)
    labels = np.asarray([f"L{index}" for index in range(_N_LONG)], dtype=object)
    labels[-1] = "X" * 5_000
    store.cells.insert("lineage", by_cell_id(store.cells, labels))
    table = tmp_path / "clones.csv"
    table.write_text(
        "Cell_ID,Clone_ID\n"
        + "".join(f"{cell},{label}\n" for cell, label in zip(ids, labels, strict=True))
    )
    return SimpleNamespace(store=store, cells=every_cell(store), table=table, ids=ids)


@pytest.mark.slow
def test_long_labels_import_as_clones_of_their_own(tmp_path) -> None:
    long = _long_label_store(tmp_path)
    # A budget that holds the stored labels saves them. The cell table keeps
    # the column in a chunk of 100,000 rows, so reading it would take 2 GB
    # whatever the budget.
    long.store.memoryBytes = 1_000_000_000
    saved = long.store.clones.import_clones(long.table, cell_selection=long.cells)
    loaded = long.store.clones.load_clones(saved)
    assert loaded.clones["n_cells"].tolist() == [1] * _N_LONG
    assert loaded.clones["clone"].tolist()[-1] == "X" * 5_000


def test_a_table_whose_first_rows_are_long_needs_room_for_a_chunk(tmp_path) -> None:
    # The layout samples the first rows, so the long label sizes every row
    # of a chunk, and 1,000 of them do not fit the budget.
    long = _long_label_store(tmp_path)
    header, *rows = long.table.read_text().splitlines()
    first = tmp_path / "long_first.csv"
    first.write_text("\n".join([header, rows[-1], *rows[:-1]]) + "\n")
    long.store.memoryBytes = _BUDGET

    with pytest.raises(
        MemoryError, match="bytes for a chunk of 1000 rows of a clone file"
    ):
        long.store.clones.import_clones(first, cell_selection=long.cells)


@pytest.mark.parametrize("budget", [21_000_000, 27_000_000])
@pytest.mark.parametrize(
    ("row", "n_fields"), [("X9,1,extra", 3), ("X9", 1)], ids=["long", "short"]
)
def test_a_row_of_another_width_is_refused_wherever_the_budget_ends_a_chunk(
    store, tmp_path, monkeypatch, budget, row, n_fields
) -> None:
    # The budget sizes the chunks of rows, and pandas, reading alone, drops
    # the extra field of a row that starts a chunk and pads a short row with
    # empty text. Placed first in the second chunk of each budget, the row is
    # refused by its line all the same.
    import cytearc.readers.clone_tables as tables

    reader = tables.iter_clone_rows
    used: list[int] = []

    def recording(layout, chunk_rows, **options):
        used.append(chunk_rows)
        yield from reader(layout, chunk_rows, **options)

    monkeypatch.setattr(tables, "iter_clone_rows", recording)
    cells = every_cell(store)
    ids = store.cells.fetch_all("ids").tolist()
    rows = [f"{cell},1" for cell in ids] + [
        f"X{index},1" for index in range(60_000 - len(ids))
    ]
    store.memoryBytes = budget
    valid = tmp_path / "valid.csv"
    valid.write_text("Cell_ID,Clone_ID\n" + "\n".join(rows) + "\n")
    with collect_warnings():
        store.clones.import_clones(valid, cell_selection=cells)
    chunk_rows = used[-1]
    assert len(ids) < chunk_rows < len(rows) - 1
    rows[chunk_rows] = row
    bad = tmp_path / "bad.csv"
    bad.write_text("Cell_ID,Clone_ID\n" + "\n".join(rows) + "\n")
    with (
        collect_warnings(),
        pytest.raises(
            ValueError,
            match=rf"is malformed: line {chunk_rows + 2} holds {n_fields} "
            rf"fields?, but its header names 2 columns$",
        ),
    ):
        store.clones.import_clones(bad, cell_selection=cells)
