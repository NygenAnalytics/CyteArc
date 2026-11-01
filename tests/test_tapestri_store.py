"""Writing a store from a Tapestri file of single-cell DNA and protein.

The files come from the synthetic truth of ``tests/variant_truth.py`` in the
layout that Mission Bio documents; the store must hold the truth's amplicon
and protein reads, cell metadata and file metadata exactly.
"""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from tests.clone_helpers import loaded_partition, smallest_budget, traced_peak
from tests.variant_truth import (
    PROTEIN_METADATA,
    TAPESTRI_AMPLICONS,
    TAPESTRI_ANTIBODIES,
    TAPESTRI_CLONE_VARIANTS,
    TAPESTRI_METADATA,
    tapestri_truth,
    write_large_tapestri,
    write_tapestri,
)

TRUTH = tapestri_truth()


def _write_store(file: Path, store: Path, **options):
    from cytearc.readers import TapestriReader
    from cytearc.writers import TapestriToZarr

    reader = TapestriReader(str(file))
    try:
        TapestriToZarr(reader, str(store), nthreads=1, **options).dump()
    finally:
        reader.close()


def _open(store: Path, **options):
    from cytearc.datastore.datastore import DataStore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DataStore(str(store), min_features_per_cell=0, nthreads=1, **options)


def _counts(assay) -> np.ndarray:
    return np.asarray(assay.rawData.compute())


def test_a_store_holds_the_amplicon_and_protein_reads(tmp_path) -> None:
    file = write_tapestri(TRUTH, tmp_path / "run.dna+protein.h5")
    _write_store(file, tmp_path / "store.zarr")
    ds = _open(tmp_path / "store.zarr", default_assay="ADT")
    assert ds.assay_names == ("ADT", "CNV") or set(ds.assay_names) == {"ADT", "CNV"}
    assert ds.cells.fetch_all("ids").tolist() == TRUTH.row_ids
    np.testing.assert_array_equal(_counts(ds.CNV), TRUTH.amplicon_reads)
    np.testing.assert_array_equal(_counts(ds.ADT), TRUTH.protein_reads)
    assert ds.cells.fetch_all("sample_name").tolist() == TRUTH.samples.tolist()
    np.testing.assert_array_equal(ds.cells.fetch_all("filtered"), TRUTH.filtered)
    amplicons = ds.CNV.feats
    assert amplicons.fetch_all("ids").tolist() == [a[0] for a in TAPESTRI_AMPLICONS]
    assert amplicons.fetch_all("CHROM").tolist() == [a[1] for a in TAPESTRI_AMPLICONS]
    assert amplicons.fetch_all("start_pos").tolist() == [
        a[2] for a in TAPESTRI_AMPLICONS
    ]
    assert amplicons.fetch_all("end_pos").tolist() == [a[3] for a in TAPESTRI_AMPLICONS]
    assert set(amplicons.fetch_all("feature_type")) == {"Amplicon"}
    proteins = ds.ADT.feats
    assert proteins.fetch_all("ids").tolist() == [a[0] for a in TAPESTRI_ANTIBODIES]
    assert proteins.fetch_all("antibody_sequence").tolist() == [
        a[1] for a in TAPESTRI_ANTIBODIES
    ]
    assert set(proteins.fetch_all("feature_type")) == {"Antibody Capture"}
    # The amplicons' intervals reach their own assay only.
    assert "start_pos" not in proteins.columns
    assert "antibody_sequence" not in amplicons.columns
    attributes = zarr.open_group(str(tmp_path / "store.zarr"), mode="r").attrs
    assert attributes["assayTypes"] == {"ADT": "ADT", "CNV": "Assay"}
    recorded = attributes["tapestri"]
    assert recorded["samples"] == ["d2", "d4"]
    for name, metadata in (
        ("dna_read_counts", TAPESTRI_METADATA),
        ("dna_variants", TAPESTRI_METADATA),
        ("protein_read_counts", PROTEIN_METADATA),
    ):
        assert {key: recorded["assays"][name][key] for key in metadata} == metadata
    assert recorded["assays"]["dna_read_counts"]["sample_name"] == [["d2"], ["d4"]]
    assert recorded["file"] == {"date_created": "2026-01-01", "sdk_version": "3.1.2"}


def test_the_log_names_tapestri_as_the_source(tmp_path) -> None:
    from cytearc.utils.logging import logger

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        _write_store(write_tapestri(TRUTH, tmp_path / "run.h5"), tmp_path / "a.zarr")
    finally:
        logger.remove(sink)
    assert f"Wrote {TRUTH.n_rows} cells and 7 features from Tapestri to 2 assay(s)" in (
        messages
    )


def test_a_file_without_proteins_gives_the_amplicons_alone(tmp_path) -> None:
    file = write_tapestri(TRUTH, tmp_path / "run.dna.h5", protein=False)
    _write_store(file, tmp_path / "store.zarr", workspace="tapestri")
    ds = _open(tmp_path / "store.zarr", workspace="tapestri")
    assert list(ds.assay_names) == ["CNV"]
    np.testing.assert_array_equal(_counts(ds.CNV), TRUTH.amplicon_reads)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert "tapestri" in root["tapestri"].attrs
    assert "tapestri" not in root.attrs


def test_assay_types_replace_the_defaults(tmp_path) -> None:
    file = write_tapestri(TRUTH, tmp_path / "run.h5")
    _write_store(file, tmp_path / "store.zarr", assay_types={"CNV": "CUSTOM"})
    attributes = zarr.open_group(str(tmp_path / "store.zarr"), mode="r").attrs
    assert attributes["assayTypes"]["CNV"] == "CUSTOM"


def test_the_writer_takes_a_tapestri_reader(tmp_path) -> None:
    from cytearc.writers import TapestriToZarr

    with pytest.raises(TypeError, match="reader must be a TapestriReader, not object"):
        TapestriToZarr(object(), str(tmp_path / "store.zarr"))  # type: ignore[arg-type]
    assert not (tmp_path / "store.zarr").exists()


def _edit(path: Path, edit) -> Path:
    with h5py.File(path, "r+") as handle:
        edit(handle)
    return path


def _replace(group, name: str, values) -> None:
    del group[name]
    group.create_dataset(name, data=values)


def _reverse_cells(handle, assay: str) -> None:
    rows = handle[f"assays/{assay}/ra"]
    for name in ("barcode", "sample_name"):
        _replace(rows, name, rows[name][:][::-1])


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda h: [
                h.__delitem__(f"assays/{name}")
                for name in ("dna_read_counts", "protein_read_counts")
            ],
            "holds neither assays/dna_read_counts nor assays/protein_read_counts",
        ),
        (
            # The same cells in another order.
            lambda h: _reverse_cells(h, "protein_read_counts"),
            "assays/protein_read_counts of .* lists other cells than "
            "assays/dna_read_counts, or in another order",
        ),
        (
            lambda h: _reverse_cells(h, "dna_variants"),
            "assays/dna_variants of .* lists other cells",
        ),
        (
            lambda h: _replace(
                h["assays/dna_variants/ra"],
                "filtered",
                np.full(TRUTH.n_rows, 2, np.uint8),
            ),
            "filtered of .* must hold a 0 or 1 for each cell",
        ),
        (
            lambda h: _replace(
                h["assays/dna_read_counts/layers"],
                "read_counts",
                np.zeros((TRUTH.n_rows, 2), dtype=np.int64),
            ),
            "read_counts of .* is 52 by 2, but the assay lists 52 cells and 4 features",
        ),
        (
            lambda h: _replace(
                h["assays/dna_read_counts/ca"],
                "id",
                np.asarray([b"AML_NRAS_1"] * len(TAPESTRI_AMPLICONS)),
            ),
            "feature IDs of assays/dna_read_counts must contain unique values",
        ),
        (
            lambda h: _replace(
                h["assays/dna_read_counts/ra"],
                "barcode",
                np.asarray([b"AAAA"] * TRUTH.n_rows),
            ),
            "lists the barcode 'AAAA' of the sample 'd2' twice",
        ),
        (
            lambda h: h["assays/dna_read_counts"].__delitem__("ra"),
            "lacks assays/dna_read_counts/ra",
        ),
        (
            lambda h: h["assays/dna_read_counts/ra"].__delitem__("barcode"),
            "lacks assays/dna_read_counts/ra/barcode",
        ),
        (
            lambda h: _replace(
                h["assays/dna_read_counts/ra"], "sample_name", np.asarray([b"d2"] * 3)
            ),
            "ra of .* names 52 barcodes but 3 samples",
        ),
        (
            lambda h: h["assays/dna_read_counts"].__delitem__("layers"),
            "lacks assays/dna_read_counts/layers",
        ),
        (
            lambda h: h["assays/dna_read_counts"].__delitem__("ca"),
            "lacks assays/dna_read_counts/ca",
        ),
        (
            lambda h: (
                _replace(
                    h["assays/protein_read_counts/ca"], "id", np.asarray([], "S1")
                ),
                _replace(
                    h["assays/protein_read_counts/layers"],
                    "read_counts",
                    np.zeros((TRUTH.n_rows, 0), dtype=np.int64),
                ),
            ),
            "assays/protein_read_counts of .* lists no features",
        ),
    ],
)
def test_the_reader_refuses_files_it_cannot_read(tmp_path, edit, message) -> None:
    from cytearc.readers import TapestriReader

    file = _edit(write_tapestri(TRUTH, tmp_path / "run.h5"), edit)
    with pytest.raises(ValueError, match=message):
        TapestriReader(str(file))


def test_a_misaligned_feature_column_is_refused(tmp_path) -> None:
    from cytearc.readers import TapestriReader

    file = _edit(
        write_tapestri(TRUTH, tmp_path / "run.h5"),
        lambda h: _replace(h["assays/dna_read_counts/ca"], "start_pos", np.arange(3)),
    )
    reader = TapestriReader(str(file))
    try:
        with pytest.raises(ValueError, match=r"ca/start_pos of .* has shape \(3,\)"):
            list(reader.get_feature_columns())
    finally:
        reader.close()


def test_assay_groups_that_the_store_does_not_hold_are_named(tmp_path) -> None:
    from cytearc.readers import TapestriReader

    file = write_tapestri(TRUTH, tmp_path / "run.h5")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TapestriReader(str(file)).close()
    assert not [w for w in caught if "does not hold" in str(w.message)]
    _edit(file, lambda h: h.create_group("assays/rna_read_counts"))
    with pytest.warns(
        UserWarning,
        match="holds assays/rna_read_counts, which the store does not hold: it "
        "holds the amplicon and protein reads",
    ):
        TapestriReader(str(file)).close()


def test_a_feature_column_of_both_assays_is_one_column(tmp_path) -> None:
    from cytearc.readers import TapestriReader

    amplicons = [f"amplicon {index}" for index in range(len(TAPESTRI_AMPLICONS))]
    antibodies = [f"antibody {index}" for index in range(len(TAPESTRI_ANTIBODIES))]

    def add(handle) -> None:
        for assay, notes in (
            ("dna_read_counts", amplicons),
            ("protein_read_counts", antibodies),
        ):
            handle[f"assays/{assay}/ca"].create_dataset(
                "note", data=np.char.encode(np.asarray(notes))
            )

    file = _edit(write_tapestri(TRUTH, tmp_path / "run.h5"), add)
    reader = TapestriReader(str(file))
    try:
        notes = [
            values for key, values in reader.get_feature_columns() if key == "note"
        ]
    finally:
        reader.close()
    assert len(notes) == 1
    assert notes[0].tolist() == amplicons + antibodies
    _write_store(file, tmp_path / "store.zarr")
    ds = _open(tmp_path / "store.zarr", default_assay="ADT")
    assert ds.CNV.feats.fetch_all("note").tolist() == amplicons
    assert ds.ADT.feats.fetch_all("note").tolist() == antibodies


def test_a_store_imports_its_variants_and_calls_their_clones(tmp_path) -> None:
    file = write_tapestri(TRUTH, tmp_path / "run.h5")
    _write_store(file, tmp_path / "store.zarr")
    ds = _open(tmp_path / "store.zarr", default_assay="ADT")
    cells = ds.snapshot_cell_selection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = ds.variants.import_counts(file, cell_selection=cells)
    # The store names its cells as the import names the file's.
    attributes = ds.artifacts.load(counts).attrs
    assert attributes["n_unmatched_barcodes"] == 0
    assert attributes["n_listed_cells"] == TRUTH.n_rows
    genotypes = ds.variants.genotypes(counts, method="source")
    clones = ds.variants.genotype_clones(genotypes, within="sample_name")
    ids = TRUTH.row_ids
    expected = {
        frozenset(
            ids[row]
            for row in range(len(TRUTH.clones.clones))
            if TRUTH.clones.clones[row] == clone and TRUTH.samples[row] == sample
        )
        for clone in TAPESTRI_CLONE_VARIANTS
        for sample in ("d2", "d4")
    } - {frozenset()}
    assert loaded_partition(ds, clones) == expected


@pytest.mark.parametrize("read_dtype", [np.int64, np.int32, np.uint16])
def test_a_batch_of_reads_holds_what_the_reader_reports(tmp_path, read_dtype) -> None:
    """The writer plans each batch with the reader's staging bytes beside
    the sparse batch it keeps, so the batch's peak must fit both."""
    import gc
    import tracemalloc

    from cytearc.readers import TapestriReader
    from cytearc.utils.arrays import sparse_matrix_bytes

    file = write_large_tapestri(
        tmp_path / "large.h5",
        n_cells=3_000,
        n_variants=2,
        n_amplicons=300,
        read_dtype=read_dtype,
    )
    reader = TapestriReader(str(file))
    try:
        for rows in (100, 1_000, 3_000):
            batches = reader.consume(rows)
            gc.collect()
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                matrix = next(batches)
                peak = tracemalloc.get_traced_memory()[1] - before
            finally:
                tracemalloc.stop()
            staging = reader.producer_staging_bytes(rows, 0)
            assert matrix.shape == (rows, reader.nFeatures)
            assert peak <= staging + sparse_matrix_bytes(matrix), (rows, peak)
            # The staging is the batch's own: the peak reaches most of it.
            assert peak >= staging * 0.75, (rows, peak, staging)
            del matrix, batches
    finally:
        reader.close()


@pytest.mark.slow
def test_the_traced_peak_of_a_store_fits_the_smallest_budget_accepted(
    tmp_path,
) -> None:
    from cytearc.readers import TapestriReader
    from cytearc.storage.count_matrix import CountMatrixPolicy
    from cytearc.writers import TapestriToZarr

    file = write_large_tapestri(
        tmp_path / "large.h5", n_cells=20_000, n_variants=2, n_amplicons=300
    )
    # The default layout's shards need far more than the import itself.
    policy = CountMatrixPolicy(unitBytes=1 << 20, chunkBytes=1 << 16)
    attempt = 0

    def write(budget: int) -> None:
        nonlocal attempt
        attempt += 1
        reader = TapestriReader(str(file))
        try:
            TapestriToZarr(
                reader,
                str(tmp_path / f"store{attempt}.zarr"),
                mem_budget=budget,
                nthreads=1,
                policy=policy,
            ).dump()
        finally:
            reader.close()

    budget = smallest_budget(write, 1 << 30)
    peak = traced_peak(lambda: write(budget))
    assert peak <= budget, (peak, budget)
