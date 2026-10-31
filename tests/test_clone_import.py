"""Clones imported from every supported format recover a synthetic truth.

The module writes a store of the truth's cells once, as a template. Tests that
only import share one copy, since each import saves its own artifact; tests
that check reuse, read-only stores or the memory budget copy the template.
"""

import gzip
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from cytearc.clones import Clones
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    copy_clone_store,
    every_cell,
    loaded_partition,
    open_clone_store,
    write_truth_store,
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
    store = copy_clone_store(template, tmp_path_factory.mktemp("clone_shared"))
    return SimpleNamespace(store=store, cells=every_cell(store))


@pytest.fixture
def store(template, tmp_path):
    return copy_clone_store(template, tmp_path)


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
        pytest.warns(UserWarning, match="match no selected cell")
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
    assert attributes["n_file_barcodes"] == n_listed
    # Every file but Numbat's lists the barcodes outside the store.
    assert attributes["n_unmatched_barcodes"] == (
        0 if n_listed == N_CELLS else N_OUTSIDE
    )
    assert attributes["n_clones"] == len(CLONE_SIZES)
    assert attributes["n_cells_in_several_clones"] == 0
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

    with pytest.warns(UserWarning, match="match no selected cell"):
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

    # Equal clone numbers of two files are two clones, labelled by prefix.
    assert loaded_partition(shared.store, ref) == TRUTH.partition(
        split_by=TRUTH.samples
    )
    labels = set(shared.store.clones.load_clones(ref).clones["clone"])
    assert {label.split("__")[0] for label in labels} == {"d2", "d4"}
    sources = shared.store.artifacts.inspect(ref).inputs["sources"]
    assert sorted(source["namespace"] for source in sources) == ["d2__", "d4__"]


def test_files_without_prefixes_are_namespaced_by_their_names(shared, tmp_path) -> None:
    paths = _per_sample_files(tmp_path, bare=False)

    with pytest.warns(UserWarning, match="match no selected cell"):
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

    with pytest.raises(ValueError, match="share the namespace 'clones_'"):
        shared.store.clones.import_clones(
            [first / "clones.txt", second / "clones.txt"], cell_selection=shared.cells
        )
    with pytest.raises(ValueError, match="share the namespace 's1'"):
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

    with pytest.warns(UserWarning, match="match no selected cell"):
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
    assert shared.store.artifacts.load(ref).attrs["n_cells_in_several_clones"] == 2
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
    shared.store.cells.insert("day_two", TRUTH.samples == "d2", overwrite=True)
    day_two = shared.store.snapshot_cell_selection("day_two")
    path = write_trex(TRUTH, tmp_path / "clones.txt")

    with pytest.warns(UserWarning, match="match no selected cell"):
        ref = shared.store.clones.import_clones(path, cell_selection=day_two)

    cells = shared.store.clones.load_clones(ref).cells
    assert cells.index.tolist() == TRUTH.cell_ids[TRUTH.samples == "d2"].tolist()
    assert loaded_partition(shared.store, ref) == TRUTH.partition(TRUTH.samples == "d2")
    attributes = shared.store.artifacts.load(ref).attrs
    n_day_two = int(np.count_nonzero((TRUTH.samples == "d2") & (TRUTH.clones >= 0)))
    assert attributes["n_unmatched_barcodes"] == N_IN_CLONES - n_day_two + N_OUTSIDE


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
        "no selected cell, for example "
        + ", ".join(repr(f"d2__{barcode}-1") for barcode in TRUTH.outside)
        + ". Check barcode_prefix and cell_selection."
    ]
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
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
    import cytearc.datastore._operations.clones as clone_operations

    # Each label is taken to hold a terabyte, so the first chunk's labels
    # outgrow the budget.
    monkeypatch.setattr(clone_operations, "_BYTES_PER_LABEL", 2**40)
    for path in (
        write_trex(TRUTH, tmp_path / "clones.txt"),
        write_larry(TRUTH, _folder(tmp_path, "larry")),
    ):
        with pytest.raises(MemoryError, match="clone labels holds about"):
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
        "nobody", np.zeros(N_CELLS, dtype=bool), overwrite=True
    )
    assert nobody is None
    with pytest.raises(ValueError, match="selects no cell"):
        shared.store.clones.import_clones(
            write_trex(TRUTH, tmp_path / "clones.txt"),
            cell_selection=shared.store.snapshot_cell_selection("nobody"),
        )


def test_a_header_without_rows_saves_no_clone(shared, tmp_path) -> None:
    path = tmp_path / "clones.txt"
    path.write_text("clone_nr\tcell_id\n")

    ref = shared.store.clones.import_clones(path, cell_selection=shared.cells)

    loaded = shared.store.clones.load_clones(ref)
    assert loaded.clones.empty
    assert loaded.cells["clone"].isna().all()
    assert shared.store.artifacts.load(ref).attrs["n_file_barcodes"] == 0


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
    for format, name in (("table", "out.csv"), ("matrix", "out.mtx")):
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
        with pytest.raises(ValueError, match="clones must be a clones artifact"):
            method(other)
        with pytest.raises(TypeError, match="clones must be an ArtifactRef"):
            method("clones")
    ref = shared.store.clones.import_clones(
        write_numbat(TRUTH, tmp_path / "clone_post_2.tsv"), cell_selection=shared.cells
    )
    with pytest.raises(ValueError, match="format must be one of table, matrix"):
        shared.store.clones.export_clones(ref, tmp_path / "x.csv", format="h5ad")
    with pytest.raises(TypeError, match="path must be a path"):
        shared.store.clones.export_clones(ref, 3)


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

    assert written == N_IN_CLONES
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

    assert written == N_IN_CLONES
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


def test_clones_come_from_a_cell_metadata_column(store) -> None:
    cells = every_cell(store)
    labels = np.where(
        TRUTH.clones >= 0, np.char.add("L", TRUTH.clones.astype(str)), ""
    ).astype(object)
    labels[np.flatnonzero(TRUTH.clones < 0)[:2]] = None
    store.cells.insert("lineage", labels)

    ref = store.clones.from_column("lineage", cell_selection=cells)

    assert ref.kind == "clones"
    assert loaded_partition(store, ref) == TRUTH.partition()
    loaded = store.clones.load_clones(ref)
    assert loaded.source == "clones_from_column"
    assert loaded.clones["n_cells"].tolist() == list(CLONE_SIZES)
    assert store.artifacts.inspect(ref).parameters == {"column": "lineage"}
    assert store.clones.from_column("lineage", cell_selection=cells) == ref
    # Editing the column gives new clones.
    edited = labels.copy()
    edited[np.flatnonzero(TRUTH.clones == 0)[0]] = "L7"
    store.cells.insert("lineage", edited, overwrite=True)
    changed = store.clones.from_column("lineage", cell_selection=cells)
    assert changed != ref
    assert store.clones.load_clones(changed).clones["n_cells"].tolist()[0] == 8


def test_numeric_label_columns_name_clones_by_whole_numbers(store) -> None:
    cells = every_cell(store)
    values = np.where(TRUTH.clones >= 0, TRUTH.clones + 10.0, np.nan)
    store.cells.insert("lineage_number", values)

    ref = store.clones.from_column("lineage_number", cell_selection=cells)

    assert store.clones.load_clones(ref).clones["clone"].tolist() == [
        str(clone + 10) for clone in range(len(CLONE_SIZES))
    ]
    store.cells.insert("fraction", np.full(N_CELLS, 0.5))
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
    store.cells.insert("lineage_text", TRUTH.clones.astype(str).astype(object))
    store.memoryBytes = 1_000
    with pytest.raises(MemoryError, match="memory budget of 1000 bytes"):
        store.clones.from_column("lineage_text", cell_selection=every_cell(store))


def test_a_column_without_labels_saves_no_clone(store) -> None:
    store.cells.insert("empty_lineage", np.full(N_CELLS, "", dtype=object))
    ref = store.clones.from_column("empty_lineage", cell_selection=every_cell(store))
    loaded = store.clones.load_clones(ref)
    assert loaded.clones.empty
    assert (loaded.cells["clone_size"] == 0).all()
    assert isinstance(ref, ArtifactRef)
    assert math.isclose(store.artifacts.load(ref).attrs["n_clones"], 0)
