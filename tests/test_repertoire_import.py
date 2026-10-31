"""Importing immune receptor contigs and loading them back.

Hand-written contig rows of eight cells pin the chain ranking, the labels and
the stored fields, and a synthetic truth, written as Cell Ranger's contig
annotations and as AIRR files with the noise each holds, checks that every
encoding recovers each cell's chains. The join of barcodes to cells runs in
``tests/test_cell_barcode_join.py`` with the other importers.
"""

import gzip
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cytearc.repertoire import ImmuneReceptors
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    CELLRANGER_HEADER,
    N_RECEPTOR_CELLS,
    TCR_ROWS,
    airr_file,
    bcr_file,
    copy_clone_store,
    every_cell,
    receptor_store,
    tcr_file,
    write_lines,
    write_truth_store,
)
from tests.clone_truth import (
    N_OUTSIDE,
    RECEPTOR_SLOTS,
    SAMPLES,
    clone_truth,
    receptor_truth,
    write_airr_contigs,
    write_cellranger_contigs,
)

N_CELLS = N_RECEPTOR_CELLS
TRUTH = clone_truth()
RECEPTORS = receptor_truth(TRUTH)


@pytest.fixture(scope="module")
def receptor_template(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("receptor_template")
    receptor_store(folder)
    return folder / "store.zarr"


@pytest.fixture(autouse=True)
def _receptor_store_copy(receptor_template, tmp_path) -> None:
    """Give each test a copy of the receptor store, which receptor_store opens."""
    shutil.copytree(receptor_template, tmp_path / "store.zarr")


def test_import_classifies_each_cells_chains(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()

    ref = store.repertoire.import_receptors(
        [tcr_file(tmp_path), bcr_file(tmp_path), airr_file(tmp_path)],
        cell_selection=cells,
    )

    assert ref.kind == "immune_receptors" and ref.scope == "datastore"
    status = store.artifacts.inspect(ref)
    assert status.operation == "import_immune_receptors"
    assert status.parameters == {"productive_only": True}
    sources = status.inputs["sources"]
    assert {source["format"] for source in sources} == {"cellranger", "airr"}
    assert sources == sorted(sources, key=lambda source: source["sha256"])

    loaded = store.repertoire.load_receptors(ref)
    assert isinstance(loaded, ImmuneReceptors)
    assert loaded.productive_only is True
    table = loaded.cells
    assert table.index.tolist() == [f"cell{index}" for index in range(N_CELLS)]
    assert table["receptor_type"].tolist() == [
        "ambiguous",  # alpha and beta chains with a heavy chain
        "TCR",
        "TCR",
        "multichain",
        "BCR",
        "no IR",  # its one contig is outside a called cell
        "BCR",
        "no IR",
    ]
    assert table["receptor_subtype"].tolist() == [
        "ambiguous",
        "TRA+TRB",
        "TRA+TRB",
        "multichain",
        "IGH+IGK",
        "no IR",
        "IGH+IGL",
        "no IR",
    ]
    assert table["chain_pairing"].tolist() == [
        "ambiguous",
        "extra VJ",
        "orphan VDJ",
        "multichain",
        "single pair",
        "no IR",
        "single pair",
        "no IR",
    ]
    assert table["n_vj"].tolist() == [1, 2, 0, 0, 1, 0, 1, 0]
    assert table["n_vdj"].tolist() == [2, 1, 1, 3, 1, 0, 1, 0]
    assert table["n_contigs"].tolist() == [3, 3, 2, 3, 2, 1, 2, 0]
    assert table["multichain"].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    # The more expressed alpha chain of cell1 fills VJ_1.
    assert table.loc["cell1", "VJ_1_junction_aa"] == "CAVG"
    assert table.loc["cell1", "VJ_2_junction_aa"] == "CAVH"
    # cell0's beta chain has more UMIs than its heavy chain.
    assert table.loc["cell0", "VDJ_1_junction_aa"] == "CASSF"
    assert table.loc["cell0", "VDJ_2_locus"] == "IGH"
    assert table.loc["cell6", "VJ_1_v_call"] == "IGLV1-40*01"
    assert table.loc["cell6", "VJ_1_locus"] == "IGL"  # inferred from its calls
    assert pd.isna(table.loc["cell2", "VJ_1_junction_aa"])
    assert table.loc["cell0", "VDJ_1_umi_count"] == 8
    assert pd.isna(table.loc["cell7", "VDJ_1_umi_count"])
    assert "VJ_1_d_call" not in table.columns and "VDJ_1_d_call" in table.columns

    chains = loaded.chains
    assert len(chains) == 16
    assert chains["ids"].tolist()[:3] == ["cell0", "cell0", "cell0"]
    assert set(chains.columns) >= {
        "ids",
        "slot",
        "chain",
        "sequence",
        "sequence_alignment",
    }
    by_id = chains.set_index("sequence_id")
    assert by_id.loc["cell2_contig_2", "chain"] == np.False_  # non-productive
    assert by_id.loc["cell3_contig_3", "chain"] == np.True_  # a chain in no slot
    assert by_id.loc["cell3_contig_3", "slot"] == ""
    # The slots of the contigs whose IDs are unique, derived from the rows:
    # cell1's alpha and cell3's beta chains rank by UMIs, and cell2's
    # non-productive and cell5's flagged contigs fill no slot.
    expected_slots = {
        "cell1_contig_1": "VJ_1",
        "cell1_contig_2": "VJ_2",
        "cell1_contig_3": "VDJ_1",
        "cell2_contig_1": "VDJ_1",
        "cell2_contig_2": "",
        "cell3_contig_1": "VDJ_1",
        "cell3_contig_2": "VDJ_2",
        "cell4_contig_1": "VDJ_1",
        "cell4_contig_2": "VJ_1",
        "cell5_contig_1": "",
        "cell6_contig_1": "VDJ_1",
        "cell6_contig_2": "VJ_1",
    }
    unique = by_id.loc[~by_id.index.duplicated(keep=False), "slot"].to_dict()
    assert {name: unique[name] for name in expected_slots} == expected_slots
    assert by_id.loc["cell5_contig_1", "is_cell"] == np.False_
    assert by_id.loc["cell6_contig_1", "sequence_alignment"] == "ACGT-ACGT"
    assert pd.isna(by_id.loc["cell6_contig_2", "sequence_alignment"])
    assert by_id.loc["cell6_contig_1", "duplicate_count"] == 12
    assert by_id.loc["cell6_contig_1", "v_identity"] == 0.95
    assert np.isnan(by_id.loc["cell6_contig_2", "v_identity"])

    group = store.artifacts.load(ref)
    assert group.attrs["n_file_contigs"] == 17
    assert group.attrs["n_unmatched_contigs"] == 1
    # cell0 to cell6 and outside-1, each counted once across the files.
    assert group.attrs["n_file_barcodes"] == 8
    assert group.attrs["n_unmatched_barcodes"] == 1
    assert group.attrs["unmatched_barcode_examples"] == ["outside-1"]


def test_unmatched_barcodes_are_named_in_a_warning(tmp_path) -> None:
    store = receptor_store(tmp_path)
    with collect_warnings() as messages:
        ref = store.repertoire.import_receptors(
            tcr_file(tmp_path), cell_selection=store.snapshot_cell_selection()
        )
    # cell0, cell1, cell2, cell3, cell5 and outside-1.
    assert list(messages) == [
        "1 of 6 barcodes of the contig file match no selected cell, for example "
        "'outside-1'. Check barcode_prefix and cell_selection."
    ]
    attributes = store.artifacts.load(ref).attrs
    assert (attributes["n_file_contigs"], attributes["n_unmatched_contigs"]) == (12, 1)


def test_unmatched_barcodes_count_once_per_sample(tmp_path) -> None:
    store = receptor_store(tmp_path)
    path = tcr_file(tmp_path)
    # Under a second prefix no barcode of the file names a cell, so its
    # outside-1 is a barcode of its own beside the first file's.
    ref = store.repertoire.import_receptors(
        [path, path],
        cell_selection=store.snapshot_cell_selection(),
        barcode_prefix=["", "s2__"],
    )
    attributes = store.artifacts.load(ref).attrs
    barcodes = {row.split(",")[0] for row in TCR_ROWS}
    assert attributes["n_unmatched_barcodes"] == len(barcodes) + 1
    assert attributes["n_unmatched_contigs"] == len(TCR_ROWS) + 1


def test_import_is_reused_by_file_contents_in_any_order(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    first = store.repertoire.import_receptors(
        [tcr_file(tmp_path), bcr_file(tmp_path)], cell_selection=cells
    )
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(tmp_path / "tcr.csv", moved / "renamed.csv")
    shutil.copy(tmp_path / "bcr.csv", moved / "bcr.csv")
    assert (
        store.repertoire.import_receptors(
            [moved / "bcr.csv", moved / "renamed.csv"], cell_selection=cells
        )
        == first
    )
    edited = tcr_file(
        tmp_path, [row.replace("CASSF,", "CASSW,") for row in TCR_ROWS], "edited.csv"
    )
    changed = store.repertoire.import_receptors(
        [edited, bcr_file(tmp_path)], cell_selection=cells
    )
    assert changed != first
    assert (
        store.repertoire.load_receptors(changed).cells.loc["cell0", "VDJ_1_junction_aa"]
        == "CASSW"
    )
    again = store.repertoire.import_receptors(
        [tcr_file(tmp_path), bcr_file(tmp_path)],
        cell_selection=cells,
        invalidate_cache=True,
    )
    assert again != first


def test_full_ties_go_to_the_contig_id_in_any_row_order(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    # Two alpha chains that tie on counts and both junctions.
    first = "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVF,TGTGCTGTGTTT,100,5,ct1,ct1_1"
    second = "cell0,true,cell0_contig_2,true,500,TRA,TRAV2,,TRAJ1,TRAC,true,true,CAVF,TGTGCTGTGTTT,100,5,ct1,ct1_1"
    for name, rows in (("ab.csv", [first, second]), ("ba.csv", [second, first])):
        receptors = store.repertoire.import_receptors(
            write_lines(tmp_path, name, CELLRANGER_HEADER, rows), cell_selection=cells
        )
        table = store.repertoire.load_receptors(receptors).cells
        assert table.loc["cell0", ["VJ_1_v_call", "VJ_2_v_call"]].tolist() == [
            "TRAV1",
            "TRAV2",
        ]


def test_compressed_files_read_whatever_their_name(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    # Gzip-compressed Cell Ranger and AIRR files without the .gz suffix.
    for plain in (tcr_file(tmp_path), airr_file(tmp_path)):
        packed = plain.with_name(f"packed_{plain.name}")
        packed.write_bytes(gzip.compress(plain.read_bytes()))
        expected, found = (
            store.repertoire.load_receptors(
                store.repertoire.import_receptors(path, cell_selection=cells)
            ).cells
            for path in (plain, packed)
        )
        pd.testing.assert_frame_equal(found, expected)


def test_non_productive_chains_count_when_asked(tmp_path) -> None:
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        tcr_file(tmp_path),
        cell_selection=store.snapshot_cell_selection(),
        productive_only=False,
    )
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.productive_only is False
    assert loaded.cells.loc["cell2", "chain_pairing"] == "single pair"
    assert loaded.cells.loc["cell2", "VJ_1_junction_aa"] == "CAVI"


def test_barcode_prefixes_join_the_cells_of_merged_stores(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    # cell ids are "cell0"...; strip the "cell" of the barcodes and prefix it back.
    rows = [row.replace("cell", "", 1) for row in TCR_ROWS]
    path = tcr_file(tmp_path, rows, "prefixed.csv")
    ref = store.repertoire.import_receptors(
        path, cell_selection=cells, barcode_prefix="cell"
    )
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.cells.loc["cell0", "VDJ_1_junction_aa"] == "CASSF"
    assert store.artifacts.inspect(ref).inputs["sources"][0]["barcode_prefix"] == "cell"
    both = store.repertoire.import_receptors(
        [path, bcr_file(tmp_path)], cell_selection=cells, barcode_prefix=["cell", None]
    )
    assert (
        store.repertoire.load_receptors(both).cells.loc["cell4", "receptor_type"]
        == "BCR"
    )
    with pytest.raises(ValueError, match="lists 1 values for 2 contig files"):
        store.repertoire.import_receptors(
            [path, bcr_file(tmp_path)], cell_selection=cells, barcode_prefix=["cell"]
        )


def test_a_cell_selection_limits_the_imported_cells(tmp_path) -> None:
    store = receptor_store(tmp_path)
    keep = np.zeros(N_CELLS, dtype=bool)
    keep[[0, 4]] = True
    store.cells.insert("chosen", keep)
    ref = store.repertoire.import_receptors(
        [tcr_file(tmp_path), bcr_file(tmp_path)],
        cell_selection=store.snapshot_cell_selection("chosen"),
    )
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.cells.index.tolist() == ["cell0", "cell4"]
    assert store.artifacts.load(ref).attrs["n_unmatched_contigs"] == 10
    values = store.artifacts.load_values(ref, value="chain_pairing")
    assert values.to_pandas().tolist() == ["ambiguous", "single pair"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (TCR_ROWS + [TCR_ROWS[0]], "Contig ID 'cell0_contig_1' repeats"),
        ([TCR_ROWS[0].replace("cell0_contig_1", "")], "have no sequence_id"),
    ],
)
def test_contig_ids_are_required_and_unique_within_a_file(
    tmp_path, rows, message
) -> None:
    store = receptor_store(tmp_path)
    with pytest.raises(ValueError, match=message):
        store.repertoire.import_receptors(
            tcr_file(tmp_path, rows), cell_selection=store.snapshot_cell_selection()
        )
    assert store.artifacts.list(kind="immune_receptors", scope="datastore") == []


def test_one_contig_id_may_appear_in_two_files(tmp_path) -> None:
    store = receptor_store(tmp_path)
    # cell0_contig_1 is the ID of an alpha chain and of a heavy chain.
    ref = store.repertoire.import_receptors(
        [tcr_file(tmp_path), bcr_file(tmp_path)],
        cell_selection=store.snapshot_cell_selection(),
    )
    chains = store.repertoire.load_receptors(
        ref, fields=["sequence_id", "locus"]
    ).chains
    assert (chains["sequence_id"] == "cell0_contig_1").sum() == 2


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"paths": 3}, TypeError, "paths must be a path"),
        ({"paths": []}, ValueError, "at least one contig file"),
        ({"paths": [3]}, TypeError, "paths must be a path"),
        ({"barcode_prefix": 3}, TypeError, "barcode_prefix must be text"),
        ({"barcode_prefix": [3]}, TypeError, "barcode_prefix must be text"),
        ({"barcode_suffix": 3}, TypeError, "barcode_suffix must be text"),
        ({"barcode_pattern": "(?P<cell>.+)"}, ValueError, "no named group 'barcode'"),
        ({"format": 3}, TypeError, "format must be a format name or None"),
        ({"format": "10x"}, ValueError, "format must be 'cellranger' or 'airr'"),
        ({"productive_only": 1}, TypeError, "productive_only must be a boolean"),
        ({"invalidate_cache": "yes"}, TypeError, "invalidate_cache must be a boolean"),
        ({"cell_selection": "I"}, TypeError, "cell_selection must be an ArtifactRef"),
    ],
)
def test_import_arguments_are_checked(tmp_path, kwargs, error, message) -> None:
    store = receptor_store(tmp_path)
    arguments = {
        "paths": tcr_file(tmp_path),
        "cell_selection": store.snapshot_cell_selection(),
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        store.repertoire.import_receptors(arguments.pop("paths"), **arguments)


def test_a_file_listed_twice_is_refused(tmp_path) -> None:
    store = receptor_store(tmp_path)
    path = tcr_file(tmp_path)
    with pytest.raises(ValueError, match="listed twice"):
        store.repertoire.import_receptors(
            [path, path], cell_selection=store.snapshot_cell_selection()
        )


def test_an_unknown_file_format_is_refused(tmp_path) -> None:
    store = receptor_store(tmp_path)
    other = write_lines(tmp_path, "other.csv", "a,b", ["1,2"])
    with pytest.raises(ValueError, match="neither an AIRR Rearrangement file"):
        store.repertoire.import_receptors(
            other, cell_selection=store.snapshot_cell_selection()
        )


def test_a_named_format_reads_files_as_their_header_would(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    paths = [tcr_file(tmp_path), airr_file(tmp_path)]
    detected = store.repertoire.import_receptors(paths, cell_selection=cells)

    # The format is part of each file's identity, not of how it was found.
    named = store.repertoire.import_receptors(
        paths[0], cell_selection=cells, format="cellranger"
    )
    assert named == store.repertoire.import_receptors(paths[0], cell_selection=cells)
    assert named != detected
    # A file read as the other format fails on the fields that format needs.
    with pytest.raises(ValueError, match="The AIRR file has no cell_id field"):
        store.repertoire.import_receptors(paths[0], cell_selection=cells, format="airr")
    with pytest.raises(ValueError, match="lacks the columns 'barcode', 'contig_id'"):
        store.repertoire.import_receptors(
            paths[1], cell_selection=cells, format="cellranger"
        )


def test_a_file_without_contig_rows_leaves_every_cell_without_a_receptor(
    tmp_path,
) -> None:
    store = receptor_store(tmp_path)
    for name, header in (
        ("empty.csv", CELLRANGER_HEADER),
        ("empty.tsv", "cell_id\tsequence_id\tlocus\tjunction_aa\tproductive"),
    ):
        with collect_warnings() as messages:
            ref = store.repertoire.import_receptors(
                write_lines(tmp_path, name, header, []),
                cell_selection=store.snapshot_cell_selection(),
            )
        loaded = store.repertoire.load_receptors(ref)
        assert loaded.cells["receptor_type"].tolist() == ["no IR"] * N_CELLS
        assert loaded.cells["n_contigs"].tolist() == [0] * N_CELLS
        assert len(loaded.chains) == 0
        attributes = store.artifacts.load(ref).attrs
        assert (attributes["n_file_contigs"], attributes["n_file_barcodes"]) == (0, 0)
        assert list(messages) == []


def test_import_refuses_an_empty_cell_selection(tmp_path) -> None:
    store = receptor_store(tmp_path)
    store.cells.insert("nobody", np.zeros(N_CELLS, dtype=bool))
    with pytest.raises(ValueError, match="selects no cell"):
        store.repertoire.import_receptors(
            tcr_file(tmp_path), cell_selection=store.snapshot_cell_selection("nobody")
        )


def test_import_refuses_contigs_larger_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.repertoire.import_receptors(tcr_file(tmp_path), cell_selection=cells)
    assert store.artifacts.list(kind="immune_receptors", scope="datastore") == []
    # A budget below what the selected cells take fails before reading.
    monkeypatch.setattr(store, "memoryBytes", 500)
    with pytest.raises(MemoryError, match="Importing contigs for 8 selected cells"):
        store.repertoire.import_receptors(tcr_file(tmp_path), cell_selection=cells)


def test_the_memory_check_is_three_times_the_encoded_contigs_and_400_bytes_per_cell(
    tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import repertoire as operations
    from cytearc.metadata import records

    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    path = tcr_file(tmp_path)
    encoded: list[int] = []
    add = records.RecordEncoder.add

    def recording_add(self, *arguments, **options):
        add(self, *arguments, **options)
        encoded.append(self.nbytes)

    monkeypatch.setattr(records.RecordEncoder, "add", recording_add)
    saved = store.repertoire.import_receptors(path, cell_selection=cells)
    # The peak that the import documents: three times its encoded contigs and
    # 400 bytes per selected cell.
    estimate = 3 * max(encoded) + 400 * N_CELLS
    monkeypatch.setattr(store, "memoryBytes", estimate - 1)
    with pytest.raises(MemoryError, match=f"holds about {estimate} bytes at its peak"):
        store.repertoire.import_receptors(
            path, cell_selection=cells, invalidate_cache=True
        )
    assert store.artifacts.list(kind="immune_receptors", scope="datastore") == [saved]
    monkeypatch.setattr(store, "memoryBytes", estimate)
    assert (
        store.repertoire.import_receptors(
            path, cell_selection=cells, invalidate_cache=True
        )
        != saved
    )

    # Below the cells' share, the import refuses before it reads a cell ID.
    def unread(*_arguments, **_options):
        raise AssertionError("the cell IDs were read")

    monkeypatch.setattr(operations, "read_metadata_rows_chunkwise", unread)
    monkeypatch.setattr(store, "memoryBytes", 400 * N_CELLS - 1)
    with pytest.raises(MemoryError, match="holds about 3200 bytes before the first"):
        store.repertoire.import_receptors(
            path, cell_selection=cells, invalidate_cache=True
        )


def _refused_estimate(run, store, monkeypatch, budget: int) -> int:
    """Return the bytes that a run names when it refuses ``budget``."""
    import re

    with monkeypatch.context() as patched:
        patched.setattr(store, "memoryBytes", budget)
        with pytest.raises(MemoryError) as refused:
            run()
    found = re.search(r"holds? about (\d+) bytes", str(refused.value))
    assert found is not None, str(refused.value)
    return int(found.group(1))


@pytest.mark.slow
def test_the_memory_checks_cover_what_a_sparse_import_and_load_allocate(
    tmp_path, monkeypatch
) -> None:
    import tracemalloc

    from cytearc.datastore.datastore import DataStore
    from tests.storage_helpers import write_count_store

    # One cell in 50 has a contig, as when a store of every cell of a sample
    # meets the contigs of its B cells, so what the import holds for each
    # selected cell outweighs its contigs.
    n_cells = 50_000
    write_count_store(
        str(tmp_path / "sparse.zarr"),
        {"RNA": np.ones((n_cells, 1), dtype=np.uint16)},
        "uint16",
    )
    store = DataStore(
        str(tmp_path / "sparse.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    rows = [
        f"cell{cell},true,cell{cell}_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,"
        f"true,true,CASS{cell % 50}F,TGTGCC{cell % 50},40,3,ct{cell},ct{cell}_1"
        for cell in range(0, n_cells, 50)
    ]
    path = write_lines(tmp_path, "sparse.csv", CELLRANGER_HEADER, rows)
    cells = store.snapshot_cell_selection()
    # Warm the process, so one-off costs such as lazy imports are paid.
    small = receptor_store(tmp_path)
    small.repertoire.import_receptors(
        tcr_file(tmp_path), cell_selection=small.snapshot_cell_selection()
    )

    def run_import():
        return store.repertoire.import_receptors(path, cell_selection=cells)

    # The share of the cells passes the first check, so the second names the
    # whole estimate.
    held = _refused_estimate(run_import, store, monkeypatch, 1)
    estimate = _refused_estimate(run_import, store, monkeypatch, held)
    tracemalloc.start()
    try:
        ref = run_import()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert estimate > held
    assert peak <= estimate, (peak, estimate)

    def run_load():
        return store.repertoire.load_receptors(ref, fields=[])

    load_estimate = _refused_estimate(run_load, store, monkeypatch, 1)
    run_load()
    tracemalloc.start()
    try:
        run_load()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= load_estimate, (peak, load_estimate)


def test_zero_counts_load_as_zero_and_missing_counts_as_missing(tmp_path) -> None:
    store = receptor_store(tmp_path)
    rows = [
        # cell0's alpha chain has no UMI and no read; its beta chain's counts
        # are missing.
        TCR_ROWS[0].replace(",100,5,", ",0,0,"),
        TCR_ROWS[1].replace(",200,8,", ",,,"),
    ]
    ref = store.repertoire.import_receptors(
        tcr_file(tmp_path, rows, "zero.csv"),
        cell_selection=store.snapshot_cell_selection(),
    )
    loaded = store.repertoire.load_receptors(ref)
    chains = loaded.chains.set_index("sequence_id")
    assert chains.loc["cell0_contig_1", "umi_count"] == 0
    assert chains.loc["cell0_contig_1", "consensus_count"] == 0
    assert pd.isna(chains.loc["cell0_contig_2", "umi_count"])
    assert pd.isna(chains.loc["cell0_contig_2", "consensus_count"])
    assert loaded.cells.loc["cell0", "VJ_1_umi_count"] == 0
    assert pd.isna(loaded.cells.loc["cell0", "VDJ_1_umi_count"])


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(lambda stored: [*stored, "sequence"], id="field-without-array"),
        pytest.param(lambda stored: "junction_aa", id="fields-not-a-list"),
    ],
)
def test_a_saved_import_with_incomplete_contigs_is_imported_again(
    tmp_path, fields
) -> None:
    from cytearc.storage.artifacts import artifact_group

    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    path = tcr_file(tmp_path)
    saved = store.repertoire.import_receptors(path, cell_selection=cells)
    attributes = artifact_group(store.zw, saved).attrs
    attributes["fields"] = fields(list(attributes["fields"]))
    again = store.repertoire.import_receptors(path, cell_selection=cells)
    assert again != saved
    assert store.repertoire.load_receptors(again).cells["receptor_type"].tolist()[
        :4
    ] == ["TCR", "TCR", "TCR", "multichain"]


def test_contig_arrays_are_chunked_by_bytes(tmp_path, monkeypatch) -> None:
    from cytearc.datastore._operations import repertoire as operations
    from cytearc.storage.artifacts import artifact_group

    monkeypatch.setattr(operations, "_CONTIG_CHUNK_BYTES", 16)
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    ref = store.repertoire.import_receptors(tcr_file(tmp_path), cell_selection=cells)
    group = artifact_group(store.zw, ref)
    # Per-contig codes of 4 bytes and packed bytes fill chunks of 16 bytes;
    # per-cell arrays keep chunks of cells.
    assert group["contig_junction_aa"].chunks == (4,)
    assert group["contig_junction_aa_values"].chunks == (16,)
    assert group["contig_cells"].dtype == np.int32
    assert group["chain_slots"].dtype == np.int32
    n_cells = group["n_vj"].shape[0]
    assert group["n_vj"].chunks == (n_cells,)
    loaded = store.repertoire.load_receptors(ref)
    assert loaded.chains["junction_aa"].notna().any()


def test_a_read_only_store_refuses_to_import_new_receptors(tmp_path) -> None:
    store = receptor_store(tmp_path)
    cells = store.snapshot_cell_selection()
    saved = store.repertoire.import_receptors(tcr_file(tmp_path), cell_selection=cells)
    reader = receptor_store(tmp_path, zarr_mode="r")
    assert (
        reader.repertoire.import_receptors(tcr_file(tmp_path), cell_selection=cells)
        == saved
    )
    with pytest.raises(PermissionError, match="import_immune_receptors"):
        reader.repertoire.import_receptors(bcr_file(tmp_path), cell_selection=cells)


def test_loading_reads_the_requested_fields(tmp_path) -> None:
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        tcr_file(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    loaded = store.repertoire.load_receptors(ref, fields=["junction_aa", "umi_count"])
    assert list(loaded.chains.columns) == [
        "ids",
        "slot",
        "chain",
        "junction_aa",
        "umi_count",
    ]
    # The table of cells holds the loaded slot fields.
    assert [name for name in loaded.cells.columns if name.startswith("VDJ_1")] == [
        "VDJ_1_junction_aa",
        "VDJ_1_umi_count",
    ]
    assert "raw_clonotype_id" in loaded.fields
    # Text has the string dtype, also in a field that no contig holds.
    without_d = store.repertoire.import_receptors(
        tcr_file(tmp_path, [row.replace("TRBD1", "") for row in TCR_ROWS], "no_d.csv"),
        cell_selection=store.snapshot_cell_selection(),
    )
    calls = store.repertoire.load_receptors(
        without_d, fields=["v_call", "d_call"]
    ).chains
    assert calls["v_call"].dtype == "string" and calls["d_call"].dtype == "string"
    assert calls["d_call"].isna().all()
    empty = store.repertoire.load_receptors(ref, fields=[])
    assert list(empty.chains.columns) == ["ids", "slot", "chain"]
    assert list(empty.cells.columns) == [
        "receptor_type",
        "receptor_subtype",
        "chain_pairing",
        "multichain",
        "n_vj",
        "n_vdj",
        "n_contigs",
    ]


@pytest.mark.parametrize(
    ("fields", "error", "message"),
    [
        (["sequence"], ValueError, "store no field 'sequence'"),
        ("junction_aa", TypeError, "fields must be a sequence"),
    ],
)
def test_loading_refuses_fields_that_are_not_stored(
    tmp_path, fields, error, message
) -> None:
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        tcr_file(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    with pytest.raises(error, match=message):
        store.repertoire.load_receptors(ref, fields=fields)


def test_loading_refuses_tables_larger_than_the_memory_budget(
    tmp_path, monkeypatch
) -> None:
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        tcr_file(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="Pass fields="):
        store.repertoire.load_receptors(ref)


def test_loading_refuses_other_artifact_kinds(tmp_path) -> None:
    store = receptor_store(tmp_path)
    with pytest.raises(ValueError, match="immune_receptors artifact"):
        store.repertoire.load_receptors(store.snapshot_cell_selection())
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.repertoire.load_receptors("receptors")


def test_receptors_select_cells_like_labels(tmp_path) -> None:
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        [tcr_file(tmp_path), bcr_file(tmp_path)],
        cell_selection=store.snapshot_cell_selection(),
    )
    t_cells = store.select_cells(ref, include=["TCR"])
    selected = store.artifacts.load_values(ref, cell_selection=t_cells)
    assert selected.cell_ids.tolist() == ["cell1", "cell2"]
    paired = store.select_cells(ref, value="chain_pairing", include=["single pair"])
    assert store.artifacts.load_values(
        ref, cell_selection=paired
    ).cell_ids.tolist() == ["cell4"]
    many = store.select_cells(ref, value="n_vdj", low=2, keep_bounds=True)
    assert store.artifacts.load_values(ref, cell_selection=many).cell_ids.tolist() == [
        "cell0",
        "cell3",
    ]


def test_loading_refuses_a_corrupt_list_of_fields(tmp_path) -> None:
    store = receptor_store(tmp_path)
    ref = store.repertoire.import_receptors(
        tcr_file(tmp_path), cell_selection=store.snapshot_cell_selection()
    )
    from cytearc.storage.artifacts import artifact_group

    artifact_group(store.zw, ref).attrs["fields"] = ["not_a_field"]
    with pytest.raises(ValueError, match="no valid list of fields"):
        store.repertoire.load_receptors(ref)


@pytest.fixture(scope="module")
def truth_store(tmp_path_factory) -> SimpleNamespace:
    """A store of the truth's cells, which the truth imports share."""
    path = tmp_path_factory.mktemp("receptor_truth") / "store.zarr"
    store = write_truth_store(path, TRUTH)
    return SimpleNamespace(store=store, cells=every_cell(store), path=path)


def _write_truth(folder: Path, encoding: str) -> list[Path]:
    """Write each sample's TCR and BCR contigs in one encoding."""
    paths = []
    for sample in SAMPLES:
        if encoding.startswith("airr"):
            path = folder / f"{sample}_airr_rearrangement.tsv"
            write_airr_contigs(
                RECEPTORS, path, sample=sample, with_locus=encoding == "airr"
            )
            paths.append(path)
            continue
        for receptor in ("TCR", "BCR"):
            path = folder / f"{sample}_{receptor}_{encoding}.csv"
            write_cellranger_contigs(
                RECEPTORS,
                path,
                sample=sample,
                receptor=receptor,
                all_contigs=encoding == "all",
                release="4" if encoding == "release_4" else "5",
                compress=encoding == "gzip",
            )
            paths.append(path)
    return paths


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Values as text, a missing one as "-", so tables compare cell by cell."""
    values = frame.astype(object)
    return values.where(values.notna(), "-").astype(str)


@pytest.mark.parametrize(
    "encoding", ["filtered", "all", "release_4", "gzip", "airr", "airr_without_locus"]
)
def test_every_encoding_recovers_the_receptors_of_the_truth(
    truth_store, tmp_path, encoding
) -> None:
    paths = _write_truth(tmp_path, encoding)
    prefixes = [f"{path.name.split('_')[0]}__" for path in paths]

    with collect_warnings() as messages:
        ref = truth_store.store.repertoire.import_receptors(
            paths, cell_selection=truth_store.cells, barcode_prefix=prefixes
        )

    table = truth_store.store.repertoire.load_receptors(ref).cells
    expected = RECEPTORS.expected
    columns = [
        "receptor_type",
        "receptor_subtype",
        "chain_pairing",
        "n_vj",
        "n_vdj",
        *(f"{slot}_junction_aa" for slot in RECEPTOR_SLOTS),
    ]
    pd.testing.assert_frame_equal(
        _plain(table.loc[expected.index, columns]), _plain(expected[columns])
    )
    # A cell's contigs are those its sample's files list for it.
    listed = RECEPTORS.contigs[
        RECEPTORS.contigs["is_cell"] & RECEPTORS.contigs["high_confidence"]
        if encoding != "all"
        else np.ones(len(RECEPTORS.contigs), dtype=bool)
    ]
    counts = listed[listed["cell"] >= 0].groupby("cell").size()
    assert table["n_contigs"].tolist() == [
        int(counts.get(cell, 0)) for cell in range(len(expected))
    ]
    attributes = truth_store.store.artifacts.load(ref).attrs
    assert attributes["n_unmatched_barcodes"] == N_OUTSIDE
    assert attributes["n_unmatched_contigs"] == N_OUTSIDE
    assert attributes["n_file_contigs"] == len(listed)
    assert len(messages) == 1 and messages[0].startswith(
        f"{N_OUTSIDE} of {attributes['n_file_barcodes']} barcodes of the contig "
        "files match no selected cell"
    )


def test_tcr_files_alone_leave_a_heavy_chain_unseen(truth_store, tmp_path) -> None:
    paths = [
        write_cellranger_contigs(
            RECEPTORS, tmp_path / f"{sample}.csv", sample=sample, receptor="TCR"
        )
        for sample in SAMPLES
    ]
    ref = truth_store.store.repertoire.import_receptors(
        paths,
        cell_selection=truth_store.cells,
        barcode_prefix=[f"{sample}__" for sample in SAMPLES],
    )
    table = truth_store.store.repertoire.load_receptors(ref).cells
    expected = RECEPTORS.expected
    # The cell whose heavy chain the BCR file holds pairs its T cell chains.
    ambiguous = expected.index[expected["profile"] == "ambiguous"]
    assert table.loc[ambiguous, "chain_pairing"].tolist() == ["single pair"]
    assert table.loc[ambiguous, "receptor_type"].tolist() == ["TCR"]
    # B cells have no chain in the TCR files.
    b_cells = expected.index[expected["receptor_type"] == "BCR"]
    assert set(table.loc[b_cells, "receptor_type"]) == {"no IR"}


def test_a_merged_store_needs_each_samples_prefix(truth_store, tmp_path) -> None:
    store = copy_clone_store(truth_store.path, tmp_path / "merged")
    paths = _write_truth(tmp_path, "filtered")
    with collect_warnings() as messages:
        ref = store.repertoire.import_receptors(paths, cell_selection=every_cell(store))
    attributes = store.artifacts.load(ref).attrs
    # Without prefixes no barcode names a cell of the merged store.
    assert attributes["n_unmatched_barcodes"] == attributes["n_file_barcodes"]
    assert set(store.repertoire.load_receptors(ref).cells["receptor_type"]) == {"no IR"}
    assert "Check barcode_prefix and cell_selection" in messages[0]
