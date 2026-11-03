"""Importing Tapestri and DAb-seq genotypes as variant counts, and their calls.

The files are written from the synthetic truth of ``tests/variant_truth.py``
in the layouts that Mission Bio documents and the DAb-seq pipeline writes,
and every import must recover the truth's depth, reads and calls exactly.
"""

import hashlib
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import every_cell, loaded_partition, write_cells_store
from tests.memory_helpers import traced_peak
from tests.storage_helpers import copy_store
from tests.variant_helpers import (
    DAB_SEQ_ITD_SITE,
    TAPESTRI_ALLELES,
    TAPESTRI_PASSING,
    TAPESTRI_REFERENCES,
    assert_tables,
    dab_seq_ids,
    expected_dab_seq,
    expected_tapestri,
)
from tests.variant_truth import (
    TAPESTRI_CLONE_VARIANTS,
    TAPESTRI_VARIANTS,
    tapestri_truth,
    write_dab_seq,
    write_got,
    variant_truth,
    write_tapestri,
)

TRUTH = tapestri_truth()
_SKIPPED = re.escape(
    "has 1 chosen variants with an empty ALT and 0 whose alleles are not bases"
)


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tapestri") / "store.zarr"
    write_cells_store(path, TRUTH.row_ids)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


@pytest.fixture(scope="module")
def dab_seq_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("dab_seq") / "store.zarr"
    write_cells_store(path, dab_seq_ids(TRUTH))
    return path


@pytest.fixture
def dab_seq_store(dab_seq_template, tmp_path):
    return copy_store(dab_seq_template, tmp_path)


def _import(store, path, **options):
    with collect_warnings() as messages, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = store.variants.import_counts(
            path, cell_selection=every_cell(store), **options
        )
    return ref, list(messages)


def _source_record(store, ref) -> dict:
    return store.artifacts.inspect(ref).inputs["sources"][0]


def test_tapestri_counts_are_the_passing_variants_read_from_af_and_dp(
    store, tmp_path
) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.dna+protein.h5")
    ref, messages = _import(store, path)
    loaded = store.variants.load_counts(ref)
    depth, counts, calls = expected_tapestri(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert_tables(loaded.calls, calls)
    # One site per position: the alternatives of chr1:1000 share its depth,
    # and the deletion and SNV of chr1:2000 are alleles of its longer
    # reference.
    assert loaded.sites.index.tolist() == list(TAPESTRI_REFERENCES)
    assert loaded.sites["reference"].tolist() == list(TAPESTRI_REFERENCES.values())
    alleles = {
        (str(site), allele, role)
        for site, allele, role in loaded.alleles.itertuples(index=False)
    }
    assert alleles == {
        *(
            (site, reference, "reference")
            for site, reference in TAPESTRI_REFERENCES.items()
        ),
        *((*TAPESTRI_ALLELES[column], "alternative") for column in TAPESTRI_PASSING),
    }
    attributes = store.artifacts.load(ref).attrs
    assert attributes["format"] == "tapestri"
    assert attributes["count_unit"] == "reads"
    assert attributes["depth_definition"] == "all"
    assert attributes["count_fields"] == []
    record = _source_record(store, ref)
    assert {
        key: record[key]
        for key in (
            "n_variants",
            "n_passing_variants",
            "n_imported_variants",
            "n_skipped_empty_alternative",
            "n_skipped_other_alleles",
            "skipped_examples",
            "samples",
            "panel_name",
            "genome_version",
            "pipeline_version",
            "gq_cutoff",
            "dp_cutoff",
            "af_cutoff",
        )
    } == {
        "n_variants": 10,
        "n_passing_variants": 7,
        "n_imported_variants": 6,
        "n_skipped_empty_alternative": 1,
        "n_skipped_other_alleles": 0,
        "skipped_examples": ["chr7:300:A/"],
        "samples": ["d2", "d4"],
        "panel_name": "AML",
        "genome_version": "hg19",
        "pipeline_version": "2.0.2",
        "gq_cutoff": 30,
        "dp_cutoff": 10,
        "af_cutoff": 20,
    }
    assert any(re.search(_SKIPPED, message) for message in messages)


def test_the_cells_of_two_samples_with_one_barcode_stay_apart(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    loaded = store.variants.load_counts(_import(store, path)[0])
    twin = TRUTH.row_ids[TRUTH.twin_row]
    first = TRUTH.row_ids[0]
    assert twin.split("__")[1] == first.split("__")[1]
    assert twin != first
    depth = loaded.depth.set_index(["ids", "site"])["depth"]
    for row, cell in ((0, first), (TRUTH.twin_row, twin)):
        assert int(depth[(cell, "chr7:400")]) == int(TRUTH.depth[row, 9])


def test_variants_add_failing_variants_and_positions(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    ref, _ = _import(store, path, include_variants=["chr2:500"])
    loaded = store.variants.load_counts(ref)
    columns = (*TAPESTRI_PASSING, 6)
    depth, counts, calls = expected_tapestri(TRUTH, columns)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert_tables(loaded.calls, calls)
    # The one read of the overlap cell counts toward both alternatives of
    # chr2:500, as their allele frequencies of 100% each say.
    cell = TRUTH.row_ids[TRUTH.overlap_row]
    shown = loaded.counts[
        (loaded.counts["ids"] == cell) & (loaded.counts["site"] == "chr2:500")
    ]
    assert sorted(zip(shown["allele"], shown["count"], strict=True)) == [
        ("C", 1),
        ("G", 1),
    ]
    assert (
        int(
            loaded.depth[
                (loaded.depth["ids"] == cell) & (loaded.depth["site"] == "chr2:500")
            ]["depth"].iloc[0]
        )
        == 1
    )
    record = _source_record(store, ref)
    assert (record["n_imported_variants"], record["n_added_variants"]) == (7, 1)
    assert (
        record["added_variants_sha256"] == hashlib.sha256(b"chr2:500:A/C").hexdigest()
    )
    # Naming the variant rather than its position adds the same variant, and
    # naming a passing one adds none, so both reuse their imports.
    assert (
        _import(store, path, include_variants=["chr2:500:A/C", "chr1:1000"])[0] == ref
    )
    plain, _ = _import(store, path)
    assert plain != ref
    assert _import(store, path, include_variants=["chr7:400:G/A"])[0] == plain
    assert _source_record(store, plain)["n_added_variants"] == 0
    # With chr7:400 failing too, adding it or chr2:500:A/C imports as many
    # variants, which their names tell apart.
    failing = _edit(
        write_tapestri(TRUTH, tmp_path / "failing.h5"),
        lambda h: h[f"{_COLUMNS}/filtered"].__setitem__(9, 1),
    )
    first, _ = _import(store, failing, include_variants=["chr7:400:G/A"])
    second, _ = _import(store, failing, include_variants=["chr2:500:A/C"])
    assert first != second
    assert store.variants.load_counts(first).sites.index.tolist() == [
        "chr1:1000",
        "chr1:2000",
        "chr2:500",
        "chr7:400",
    ]


def test_variants_that_cannot_be_read_are_skipped_or_refused(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    # A position takes its unreadable variants too, which are skipped.
    with pytest.warns(
        UserWarning,
        match=re.escape("1 chosen variants with an empty ALT and 1 whose alleles"),
    ):
        ref = store.variants.import_counts(
            path, cell_selection=every_cell(store), include_variants=["chr2:800"]
        )
    record = _source_record(store, ref)
    assert record["n_skipped_other_alleles"] == 1
    assert record["skipped_examples"] == ["chr2:800:*/C+T", "chr7:300:A/"]
    # Named exactly, they are refused.
    for name, reason in (
        ("chr2:800:*/C+T", "alleles that are not bases"),
        ("chr7:300:A/", "an empty ALT"),
    ):
        with pytest.raises(
            ValueError, match=re.escape(f"{name!r} of {path} has {reason}")
        ):
            store.variants.import_counts(
                path, cell_selection=every_cell(store), include_variants=[name]
            )
    with pytest.raises(ValueError, match="'chr9:1', which is neither a variant"):
        store.variants.import_counts(
            path, cell_selection=every_cell(store), include_variants=["chr9:1"]
        )
    for wrong in ("chr2:500", [1], b"chr2:500"):
        with pytest.raises(TypeError, match="variants must be a sequence"):
            store.variants.import_counts(
                path, cell_selection=every_cell(store), include_variants=wrong
            )


def test_variants_are_refused_for_other_formats(store, tmp_path) -> None:
    got = write_got(variant_truth(), tmp_path / "CALR.tsv", style="geo_total")
    with pytest.raises(
        ValueError, match="variants chooses the variants of Tapestri and DAb-seq"
    ):
        store.variants.import_counts(
            got, cell_selection=every_cell(store), include_variants=["chr1:1"]
        )


def test_a_tapestri_import_is_reused_and_warns_again(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    first, _ = _import(store, path)
    # The format is recognised from the file, so naming it reuses the import.
    with pytest.warns(UserWarning, match=_SKIPPED):
        again = store.variants.import_counts(
            path, cell_selection=every_cell(store), format="tapestri"
        )
    assert again == first
    assert _import(store, path, invalidate_cache=True)[0] != first


def test_dab_seq_counts_are_its_reads_and_calls(dab_seq_store, tmp_path) -> None:
    path = write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5")
    with pytest.warns(
        UserWarning,
        match=re.escape("2 chosen variants with an empty ALT and 1 whose alleles"),
    ):
        ref = dab_seq_store.variants.import_counts(
            path, cell_selection=every_cell(dab_seq_store)
        )
    loaded = dab_seq_store.variants.load_counts(ref)
    depth, counts, calls = expected_dab_seq(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert_tables(loaded.calls.drop(columns="quality"), calls.drop(columns="quality"))
    # A missing GQ is NaN, and every other quality is the file's.
    merged = loaded.calls.merge(calls, on=["ids", "site", "allele"])
    np.testing.assert_array_equal(merged["quality_x"], merged["quality_y"])
    assert merged["quality_x"].isna().sum() == (calls["quality"].isna()).sum() > 0
    assert loaded.sites.index.tolist() == [*TAPESTRI_REFERENCES, DAB_SEQ_ITD_SITE]
    attributes = dab_seq_store.artifacts.load(ref).attrs
    assert attributes["format"] == "dab_seq"
    assert loaded.cells.index.name == "ids"
    # include_variants names a DAb-seq variant too, here as chrom:pos:ref/alt;
    # every variant is read already, so it adds none and reuses the import.
    assert _import(dab_seq_store, path, include_variants=["chr1:1000:G/A"])[0] == ref
    record = _source_record(dab_seq_store, ref)
    assert (
        record["n_variants"],
        record["n_imported_variants"],
        record["n_skipped_empty_alternative"],
        record["n_skipped_other_alleles"],
    ) == (11, 8, 2, 1)


def test_dab_seq_variants_named_without_a_gene_read_alike(
    dab_seq_store, tmp_path
) -> None:
    # The pipeline's non-human mode names variants <chrom>:<pos>:<ref>/<alt>.
    def strip(handle) -> None:
        names = _texts(handle, "VARIANTS")
        _replace(
            handle,
            "VARIANTS",
            np.asarray(
                [
                    name if name.startswith("FLT3-ITD:") else name.split(":", 1)[1]
                    for name in names
                ]
            ),
        )

    path = _edit(write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5"), strip)
    with h5py.File(path, "r") as handle:
        assert _texts(handle, "VARIANTS")[0] == "chr1:1000:G/A"
    loaded = dab_seq_store.variants.load_counts(_import(dab_seq_store, path)[0])
    depth, counts, calls = expected_dab_seq(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert_tables(loaded.calls.drop(columns="quality"), calls.drop(columns="quality"))


def _add_itds(handle, names: list[str]) -> None:
    """Rename DAb-seq's FLT3-ITD column, the last, to the first name, and add
    one column per other name, each carried by the cells after the previous
    one's."""
    variants = _texts(handle, "VARIANTS")
    layers = {name: handle[name][:] for name in ("AD", "RD", "DP", "GQ", "GT")}
    variants[-1] = names[0]
    for name in names[1:]:
        variants.append(name)
        for layer, values in layers.items():
            layers[layer] = np.column_stack([values, np.roll(values[:, -1], 1)])
    for layer, values in layers.items():
        _replace(handle, layer, values)
    _replace(handle, "VARIANTS", np.asarray(variants))


@pytest.mark.parametrize(
    "names",
    [
        # GATK's DP at chr7:400 and ITDseek's depth never meet.
        ["FLT3-ITD:chr7:400:G/GCTC"],
        ["FLT3-ITD:chr13:28608251:T/TCTC", "FLT3-ITD:chr13:28608251:T/TCTCGG"],
    ],
)
def test_flt3_itd_variants_are_sites_of_their_own(
    dab_seq_store, tmp_path, names
) -> None:
    path = _edit(
        write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5"), lambda h: _add_itds(h, names)
    )
    loaded = dab_seq_store.variants.load_counts(_import(dab_seq_store, path)[0])
    assert loaded.sites.index.tolist() == [*TAPESTRI_REFERENCES, *names]
    ids = np.asarray(dab_seq_ids(TRUTH), dtype=object)
    expected_depth, _counts, _calls = expected_dab_seq(TRUTH)
    coordinates = expected_depth[expected_depth["site"] != DAB_SEQ_ITD_SITE]
    assert_tables(loaded.depth[~loaded.depth["site"].isin(names)], coordinates)
    carriers = np.flatnonzero(TRUTH.clones.clones == 4)
    for name in names:
        alternative = name.rsplit("/", 1)[1]
        rows = ids[carriers].tolist()
        shown = {
            kind: frame[frame["site"] == name]
            for kind, frame in (
                ("depth", loaded.depth),
                ("counts", loaded.counts),
                ("calls", loaded.calls),
            )
        }
        assert_tables(
            shown["depth"],
            pd.DataFrame({"ids": rows, "site": name, "depth": 60}),
        )
        # 25 reads of the ITD and no reads of the reference, since RD holds
        # ITDseek's depth.
        assert_tables(
            shown["counts"],
            pd.DataFrame(
                {"ids": rows, "site": name, "allele": alternative, "count": 25}
            ),
        )
        assert_tables(
            shown["calls"],
            pd.DataFrame(
                {
                    "ids": rows,
                    "site": name,
                    "allele": alternative,
                    "source": "GT",
                    "call": "1",
                    "quality": 100.0,
                }
            ),
        )
        carriers = (carriers + 1) % TRUTH.n_rows


def _batch_tables(
    source, chunk_rows: int
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The blocks of a source's batches, and the depth, counts and calls they
    give by cell ID, site label and allele."""
    sites = source.sites()
    labels = sites.labels()
    allele_sites = labels[sites.allele_site]
    depth, counts, calls = [], [], []
    blocks = 0
    for batch in source.batches(chunk_rows):
        cells = np.asarray(batch.barcodes, dtype=object)[batch.barcode_codes]
        if batch.kind == "cells":
            blocks += 1
            assert len(cells) == len(batch.barcodes)
        elif batch.kind == "depth":
            depth.extend(zip(cells, labels[batch.keys], batch.values, strict=True))
        elif batch.kind == "counts":
            counts.extend(
                zip(
                    cells,
                    allele_sites[batch.keys],
                    sites.allele[batch.keys],
                    batch.values,
                    strict=True,
                )
            )
        else:
            calls.extend(
                zip(
                    cells,
                    allele_sites[batch.keys],
                    sites.allele[batch.keys],
                    [batch.call_source] * len(cells),
                    batch.calls,
                    batch.call_quality,
                    strict=True,
                )
            )
    return (
        blocks,
        pd.DataFrame(depth, columns=["ids", "site", "depth"]),
        pd.DataFrame(counts, columns=["ids", "site", "allele", "count"]),
        pd.DataFrame(
            calls, columns=["ids", "site", "allele", "source", "call", "quality"]
        ),
    )


@pytest.mark.parametrize(("chunk_rows", "n_blocks"), [(1, 52), (20, 26), (10_000, 1)])
@pytest.mark.parametrize("file_format", ["tapestri", "dab_seq"])
def test_a_source_gives_the_truth_in_blocks_of_any_size(
    tmp_path, file_format, chunk_rows, n_blocks
) -> None:
    from cytearc.readers.tapestri import dab_seq_source, tapestri_source

    # A block holds chunk_rows // (variants + 2) cells: 6 variants of
    # Tapestri's and 8 of DAb-seq's, so 1 cell at a time, then 2, then all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if file_format == "tapestri":
            source = tapestri_source(write_tapestri(TRUTH, tmp_path / "run.h5"))
            expected = expected_tapestri(TRUTH)
        else:
            source = dab_seq_source(write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5"))
            expected = expected_dab_seq(TRUTH)
    blocks, *tables = _batch_tables(source, chunk_rows)
    assert blocks == n_blocks
    for loaded, wanted in zip(tables, expected, strict=True):
        assert_tables(loaded, wanted)


def test_the_store_reader_gives_every_read_in_batches(tmp_path) -> None:
    from cytearc.readers import TapestriReader

    reader = TapestriReader(str(write_tapestri(TRUTH, tmp_path / "run.h5")))
    try:
        batches = [matrix.toarray() for matrix in reader.consume(7)]
    finally:
        reader.close()
    # 52 cells in batches of 7, the last of 3.
    assert [len(batch) for batch in batches] == [7] * 7 + [3]
    np.testing.assert_array_equal(
        np.vstack(batches), np.hstack([TRUTH.amplicon_reads, TRUTH.protein_reads])
    )


def test_tapestri_and_dab_seq_files_are_recognised(tmp_path) -> None:
    from cytearc.readers._variant_files import hdf5_format

    assert hdf5_format(str(write_tapestri(TRUTH, tmp_path / "a.h5"))) == "tapestri"
    assert hdf5_format(str(write_dab_seq(TRUTH, tmp_path / "b.hdf5"))) == "dab_seq"
    with h5py.File(tmp_path / "c.h5", "w") as handle:
        handle.create_dataset("GT", data=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="none of giftwrap's counts.*DAb-seq"):
        hdf5_format(str(tmp_path / "c.h5"))


def _edit(path: Path, edit) -> Path:
    with h5py.File(path, "r+") as handle:
        edit(handle)
    return path


def _replace(group, name: str, values) -> None:
    del group[name]
    array = np.asarray(values)
    if array.dtype.kind == "U":
        array = np.char.encode(array, "utf-8")
    group.create_dataset(name, data=array)


_LAYERS = "assays/dna_variants/layers"
_COLUMNS = "assays/dna_variants/ca"


def _texts(handle, name):
    return [value.decode() for value in handle[name][:]]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda h: _replace(
                h[_COLUMNS], "POS", np.r_[999, h[f"{_COLUMNS}/POS"][1:]]
            ),
            "'chr1:1000:G/A' at column 0 .* disagrees with its CHROM '1', POS 999",
        ),
        (
            lambda h: h[f"{_LAYERS}/AF"].__setitem__((0, 9), 33.3),
            "do not give a whole number of reads as AF × DP / 100",
        ),
        (
            lambda h: h[f"{_LAYERS}/AF"].__setitem__((0, 9), 100.5),
            "the allele frequency 100.5; frequencies lie from 0 to 100%",
        ),
        (
            lambda h: h[f"{_LAYERS}/DP"].__setitem__((0, 9), -2),
            "the depth -2",
        ),
        (
            lambda h: h[f"{_LAYERS}/NGT"].__setitem__((0, 9), 4),
            "the genotype 4; genotypes are 0 to 3",
        ),
        (
            lambda h: h[f"{_LAYERS}/DP"].__setitem__(
                (0, 1), int(h[f"{_LAYERS}/DP"][0, 0]) + 1
            ),
            "'chr1:1000:G/A' and 'chr1:1000:G/T' .* share a position but give "
            "the cell 'd2__GGGTAAAGCAGTTCGT' the DP",
        ),
        (
            lambda h: h[f"{_COLUMNS}/filtered"].__setitem__(slice(None), 1),
            "has no variant to import",
        ),
        (
            lambda h: h[f"{_COLUMNS}/filtered"].__setitem__(0, 2),
            "filtered of .* must hold 0 or 1",
        ),
        (
            lambda h: h["assays/dna_variants"].__delitem__("ca"),
            "lacks assays/dna_variants/ca",
        ),
        (
            lambda h: h["assays/dna_variants"].__delitem__("ra"),
            "lacks assays/dna_variants/ra",
        ),
        (
            lambda h: _replace(h[_COLUMNS], "POS", np.arange(3, dtype=np.int32)),
            "The arrays of assays/dna_variants/ca of .* do not align",
        ),
    ],
)
def test_malformed_tapestri_files_are_refused(store, tmp_path, edit, message) -> None:
    path = _edit(write_tapestri(TRUTH, tmp_path / "run.h5"), edit)
    with pytest.raises(ValueError, match=message):
        _import(store, path)


def _float32_frequencies(handle, frequency: float = 33.3) -> None:
    """Store AF as float32, the first cell reading chr7:400 1,000 times at
    ``frequency``."""
    values = handle[f"{_LAYERS}/AF"][:]
    values[0, 9] = frequency
    handle[f"{_LAYERS}/DP"][0, 9] = 1_000
    del handle[f"{_LAYERS}/AF"]
    handle.create_dataset(f"{_LAYERS}/AF", data=values.astype(np.float32))


def test_float32_allele_frequencies_give_whole_reads(store, tmp_path) -> None:
    from dataclasses import replace

    # float32 holds 33.3 as 33.29999924, so 1,000 reads give 332.9999924
    # alternative reads, which float64's rounding alone would refuse.
    path = _edit(write_tapestri(TRUTH, tmp_path / "run.h5"), _float32_frequencies)
    loaded = store.variants.load_counts(_import(store, path)[0])
    depth, alternative = TRUTH.depth.copy(), TRUTH.alternative.copy()
    depth[0, 9], alternative[0, 9] = 1_000, 333
    expected = expected_tapestri(replace(TRUTH, depth=depth, alternative=alternative))
    assert_tables(loaded.depth, expected[0])
    assert_tables(loaded.counts, expected[1])
    # Half a read off a whole number is still refused.
    off = _edit(
        write_tapestri(TRUTH, tmp_path / "off.h5"),
        lambda h: _float32_frequencies(h, 33.35),
    )
    with pytest.raises(ValueError, match="do not give a whole number of reads"):
        _import(store, off)


def _rename_variant(handle, column: int, reference: str, alternative: str) -> None:
    variant = TAPESTRI_VARIANTS[column]
    ids = _texts(handle, f"{_COLUMNS}/id")
    refs = _texts(handle, f"{_COLUMNS}/REF")
    alts = _texts(handle, f"{_COLUMNS}/ALT")
    ids[column] = f"chr{variant.chrom}:{variant.position}:{reference}/{alternative}"
    refs[column] = reference
    alts[column] = alternative
    for name, values in (("id", ids), ("REF", refs), ("ALT", alts)):
        _replace(handle[_COLUMNS], name, np.asarray(values))


@pytest.mark.parametrize(
    ("reference", "alternative", "message"),
    [
        # References that share no first base cannot be one site's.
        ("A", "G", "references 'TC' and 'A', which differ beyond their shared"),
        # TCA/TA is the deletion of TC/T written with one more base.
        ("TCA", "TA", "name the same allele 'TA' of one site"),
        ("T", "TC", None),
        ("TC", "TC", "has the reference as its alternative"),
    ],
)
def test_the_alleles_of_one_position_must_read_as_one_site(
    store, tmp_path, reference, alternative, message
) -> None:
    path = _edit(
        write_tapestri(TRUTH, tmp_path / "run.h5"),
        lambda h: _rename_variant(h, 4, reference, alternative),
    )
    if message is None:
        # T/TC, an insertion after the T, is TCC beside the reference TC.
        loaded = store.variants.load_counts(_import(store, path)[0], cell_selection=[])
        assert set(loaded.alleles[loaded.alleles["site"] == "chr1:2000"]["allele"]) == {
            "TC",
            "T",
            "TCC",
        }
        return
    with pytest.raises(ValueError, match=re.escape(message)):
        _import(store, path)


def test_tapestri_layers_and_cells_must_align(store, tmp_path) -> None:
    def narrow(handle) -> None:
        del handle[f"{_LAYERS}/NGT"]
        handle.create_dataset(f"{_LAYERS}/NGT", data=np.zeros((TRUTH.n_rows, 3)))

    path = _edit(write_tapestri(TRUTH, tmp_path / "a.h5"), narrow)
    with pytest.raises(ValueError, match="layers/NGT of .* is 52 by 3, but the file"):
        _import(store, path)

    def repeat(handle) -> None:
        barcodes = _texts(handle, "assays/dna_variants/ra/barcode")
        barcodes[1] = barcodes[0]
        _replace(handle["assays/dna_variants/ra"], "barcode", np.asarray(barcodes))

    path = _edit(write_tapestri(TRUTH, tmp_path / "b.h5"), repeat)
    with pytest.raises(
        ValueError,
        match="lists the barcode 'GGGTAAAGCAGTTCGT' of the sample 'd2' twice",
    ):
        _import(store, path)
    dab_seq = write_dab_seq(TRUTH, tmp_path / "c.hdf5")
    with pytest.raises(ValueError, match="lacks assays/dna_variants.*format='dab_seq'"):
        _import(store, dab_seq, format="tapestri")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda h: _replace(
                h,
                "VARIANTS",
                np.asarray(["NRAS:chr1:1000:G>A", *_texts(h, "VARIANTS")[1:]]),
            ),
            r"VARIANTS\[0\] of .* is 'NRAS:chr1:1000:G>A', neither <gene>:<chrom>",
        ),
        (
            lambda h: h["RD"].__setitem__((0, 1), int(h["RD"][0, 0]) + 1),
            "share a position but give the cell 'GGGTAAAGCAGTTCGT-1-d2' the "
            "reference reads",
        ),
        (
            lambda h: h["AD"].__setitem__((0, 0), -2),
            "the value -2; DAb-seq writes a missing value as -1",
        ),
        (
            lambda h: h["GT"].__setitem__((0, 0), 7),
            "the genotype 7; genotypes are 0 to 3",
        ),
        (
            lambda h: _replace(
                h,
                "CELL_BARCODES",
                np.asarray([_texts(h, "CELL_BARCODES")[0]] * TRUTH.n_rows),
            ),
            "CELL_BARCODES of .* lists the barcode 'GGGTAAAGCAGTTCGT-1-d2' twice",
        ),
    ],
)
def test_malformed_dab_seq_files_are_refused(
    dab_seq_store, tmp_path, edit, message
) -> None:
    path = _edit(write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5"), edit)
    with pytest.raises(ValueError, match=message):
        _import(dab_seq_store, path)


def _call_name(
    genotype: int,
    depth: int,
    quality: float = 0.0,
    *,
    min_quality: float | None = None,
    min_depth: int | None = None,
) -> str | None:
    """Tapestri's NGT as a call, None for a cell without reads, whose call
    the import leaves out; a call below either rule is filtered."""
    if not depth:
        return None
    if genotype == 3:
        return "filtered"
    if (min_quality is not None and not quality >= min_quality) or (
        min_depth is not None and depth < min_depth
    ):
        return "filtered"
    return ("wild_type", "heterozygous", "homozygous_alternative")[genotype]


#: Tapestri's genotype units, by column of ``TAPESTRI_VARIANTS``, in the
#: order of the counts' alleles. A unit is labelled by its site and alleles,
#: Tapestri's name unless the reference was extended, as chr1:2000:T/G is.
_TAPESTRI_UNITS = {
    0: "chr1:1000:G/A",
    1: "chr1:1000:G/T",
    4: "chr1:2000:TC/GC",
    3: "chr1:2000:TC/T",
    5: "chr2:500:A/G",
    9: "chr7:400:G/A",
}


def _expected_calls(
    units: dict[int, str],
    *,
    min_quality: float | None = None,
    min_depth: int | None = None,
) -> pd.DataFrame:
    """Each cell's call and NGT label at each unit, from the truth."""
    ids = TRUTH.row_ids
    return pd.DataFrame(
        [
            (ids[row], unit, call, str(int(TRUTH.genotypes[row, column])))
            for column, unit in units.items()
            for row in range(TRUTH.n_rows)
            if (
                call := _call_name(
                    int(TRUTH.genotypes[row, column]),
                    int(TRUTH.depth[row, column]),
                    float(TRUTH.quality[row, column]),
                    min_quality=min_quality,
                    min_depth=min_depth,
                )
            )
            is not None
        ],
        columns=["ids", "site", "call", "label"],
    )


def test_source_genotypes_of_tapestri_counts_follow_ngt(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    counts, _ = _import(store, path)
    genotypes = store.variants.genotypes(counts, method="source")
    loaded = store.variants.load_genotypes(genotypes)
    # One genotype per variant.
    assert loaded.sites.index.tolist() == list(_TAPESTRI_UNITS.values())
    assert loaded.sites["reference"].tolist() == ["G", "G", "TC", "TC", "A", "G"]
    assert loaded.sites["alternative"].tolist() == ["A", "T", "GC", "T", "G", "A"]
    assert set(loaded.sites["source"]) == {"NGT"}
    ids = TRUTH.row_ids
    assert_tables(
        loaded.calls.astype({"call": str, "label": str}),
        _expected_calls(_TAPESTRI_UNITS),
    )
    calls = loaded.calls.set_index(["ids", "site"])
    # No call with reads is filtered, and without reads missing, as is the
    # wild-type call that Tapestri writes without reads.
    assert calls.loc[(ids[TRUTH.no_call_row], "chr7:400:G/A"), "call"] == "filtered"
    assert (ids[TRUTH.uncovered_row], "chr7:400:G/A") not in calls.index
    assert int(TRUTH.genotypes[TRUTH.wild_type_row, 0]) == 0
    assert (ids[TRUTH.wild_type_row], "chr1:1000:G/A") not in calls.index
    assert calls.loc[(ids[TRUTH.overlap_row], "chr2:500:A/G"), "call"] == "filtered"


@pytest.mark.parametrize(
    ("min_quality", "min_depth"),
    [(60, None), (None, 33), (45.5, 30), (0, 1)],
)
def test_source_calls_below_a_quality_or_depth_are_filtered(
    store, tmp_path, min_quality, min_depth
) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    counts, _ = _import(store, path)
    plain = store.variants.genotypes(counts, method="source")
    ref = store.variants.genotypes(
        counts, method="source", min_quality=min_quality, min_depth=min_depth
    )
    assert ref != plain
    loaded = store.variants.load_genotypes(ref)
    expected = _expected_calls(
        _TAPESTRI_UNITS, min_quality=min_quality, min_depth=min_depth
    )
    assert_tables(loaded.calls.astype({"call": str, "label": str}), expected)
    if (min_quality, min_depth) != (0, 1):
        # The rules filter some calls, and keep the labels Tapestri wrote.
        assert len(expected) == len(store.variants.load_genotypes(plain).calls)
        assert (expected["call"] == "filtered").sum() > (
            _expected_calls(_TAPESTRI_UNITS)["call"] == "filtered"
        ).sum()
    # The rules have one home, the provenance parameters, which the record
    # reads.
    rules = store.variants.load_genotypes(ref).rules
    assert rules == {
        name: value
        for name, value in store.artifacts.inspect(ref).parameters.items()
        if name in rules
    }
    assert (rules["min_quality"], rules["min_depth"]) == (
        None if min_quality is None else float(min_quality),
        min_depth,
    )
    assert (
        store.variants.genotypes(
            counts, method="source", min_quality=min_quality, min_depth=min_depth
        )
        == ref
    )


def test_a_call_without_a_quality_fails_every_min_quality(
    dab_seq_store, tmp_path
) -> None:
    row = int(np.flatnonzero(TRUTH.clones.clones == 4)[0])
    path = _edit(
        write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5"),
        lambda h: h["GQ"].__setitem__((row, 5), -1),
    )
    counts, _ = _import(dab_seq_store, path)
    cell = (dab_seq_ids(TRUTH)[row], "chr2:500:A/G")
    for options, call in (
        ({}, "heterozygous"),
        ({"min_quality": 0}, "filtered"),
    ):
        loaded = dab_seq_store.variants.load_genotypes(
            dab_seq_store.variants.genotypes(counts, method="source", **options)
        )
        assert loaded.calls.set_index(["ids", "site"]).loc[cell, "call"] == call


def test_the_rules_of_several_alternatives_reach_each_of_them(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    counts, _ = _import(store, path, include_variants=["chr2:500"])
    units = {**_TAPESTRI_UNITS, 6: "chr2:500:A/C"}
    for options in ({}, {"min_depth": 33}, {"min_quality": 50}):
        loaded = store.variants.load_genotypes(
            store.variants.genotypes(counts, method="source", **options)
        )
        assert set(loaded.sites.index) == set(units.values())
        assert_tables(
            loaded.calls.astype({"call": str, "label": str}),
            _expected_calls(units, **options),
        )
    # The overlap cell's one read leaves both alternatives of chr2:500
    # filtered, beside a cell that reads them and calls one wild type.
    calls = loaded.calls.set_index(["ids", "site"])["call"].astype(str)
    overlap = TRUTH.row_ids[TRUTH.overlap_row]
    assert (
        calls[(overlap, "chr2:500:A/G")]
        == calls[(overlap, "chr2:500:A/C")]
        == ("filtered")
    )


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"min_quality": -1}, ValueError, "min_quality must be a finite number of 0"),
        ({"min_quality": float("nan")}, ValueError, "finite number of 0 or more"),
        ({"min_quality": "30"}, TypeError, "min_quality must be a number or None"),
        ({"min_quality": True}, TypeError, "min_quality must be a number or None"),
        ({"min_depth": 0}, ValueError, "min_depth must be at least 1"),
        ({"min_depth": 1.5}, TypeError, "min_depth must be a whole number"),
        (
            {"min_depth": 10, "method": "threshold", "min_mutant_umis": 1},
            ValueError,
            "method='threshold' uses no min_depth",
        ),
        (
            {"min_umis": 2},
            ValueError,
            "method='source' uses no min_umis; it adopts the calls that the tool "
            "wrote, those of too low a quality or depth filtered by min_quality",
        ),
    ],
)
def test_source_call_rules_are_checked(
    store, tmp_path, options, error, message
) -> None:
    counts, _ = _import(store, write_tapestri(TRUTH, tmp_path / "run.h5"))
    with pytest.raises(error, match=re.escape(message)):
        store.variants.genotypes(counts, **{"method": "source", **options})


def test_genotype_clones_of_tapestri_calls_are_the_truth_clones(
    store, tmp_path
) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    genotypes = store.variants.genotypes(_import(store, path)[0], method="source")
    clones = store.variants.genotype_clones(genotypes)
    ids = TRUTH.row_ids
    truth = {
        frozenset(
            ids[row]
            for row in range(len(TRUTH.clones.clones))
            if int(TRUTH.clones.clones[row]) == clone
        )
        for clone in TAPESTRI_CLONE_VARIANTS
    }
    assert loaded_partition(store, clones) == truth


def test_dab_seq_genotypes_adopt_gt_and_flt3_itd(dab_seq_store, tmp_path) -> None:
    path = write_dab_seq(TRUTH, tmp_path / "GSM0.hdf5")
    counts, _ = _import(dab_seq_store, path)
    genotypes = dab_seq_store.variants.genotypes(counts, method="source")
    loaded = dab_seq_store.variants.load_genotypes(genotypes)
    assert loaded.sites.index.tolist() == [
        "chr1:1000:G/A",
        "chr1:1000:G/T",
        "chr1:2000:TC/GC",
        "chr1:2000:TC/T",
        "chr2:500:A/C",
        "chr2:500:A/G",
        "chr7:400:G/A",
        DAB_SEQ_ITD_SITE,
    ]
    assert set(loaded.sites["source"]) == {"GT"}
    # FLT3-ITD is called only where ITDseek found it, since the wild-type
    # calls that DAb-seq writes elsewhere have no reads.
    itd = loaded.calls[loaded.calls["site"] == DAB_SEQ_ITD_SITE]
    clone_4 = {
        dab_seq_ids(TRUTH)[row] for row in np.flatnonzero(TRUTH.clones.clones == 4)
    }
    assert set(itd["ids"].astype(str)) == clone_4
    assert set(itd["call"].astype(str)) == {"heterozygous"}


def test_threshold_rules_refuse_tapestri_counts(store, tmp_path) -> None:
    counts, _ = _import(store, write_tapestri(TRUTH, tmp_path / "run.h5"))
    with pytest.raises(
        ValueError, match="Tapestri's files hold no reads of the reference"
    ):
        store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)


def test_units_of_several_alleles_list_their_site_apart() -> None:
    from cytearc.datastore._operations.variants import _unit_ranks
    from cytearc.variants.counts import Contig, SiteTable

    table = SiteTable(
        contigs=(Contig("chr1"),),
        site_contig=np.zeros(3, dtype=np.int32),
        site_position=np.asarray([5, 9, 12]),
        site_name=np.full(3, "", dtype=object),
        site_channel=np.full(3, "", dtype=object),
        allele_site=np.asarray([0, 0, 0, 1, 1, 2, 2]),
        allele=np.asarray(["A", "C", "G", "T", "A", "C", "T"], dtype=object),
        allele_role=np.asarray(
            ["reference", "alternative", "alternative"]
            + ["reference", "alternative"] * 2,
            dtype=object,
        ),
    )
    # Site 0 is genotyped at both its alternatives, site 2 whole.
    site_rank, allele_rank, site_units, most = _unit_ranks(
        table, np.asarray([0, 0, 2]), np.asarray([1, 2, -1])
    )
    assert site_rank.tolist() == [-1, -1, 2]
    assert allele_rank.tolist() == [-1, 0, 1, -1, -1, 2, 2]
    assert site_units is not None
    offsets, units = site_units
    assert offsets.tolist() == [0, 2, 2, 2]
    assert units.tolist() == [0, 1]
    assert most == 2
    _site_rank, _allele_rank, none, one = _unit_ranks(
        table, np.asarray([1, 2]), np.asarray([-1, -1])
    )
    assert none is None and one == 1


def _large_store(tmp_path, file: Path):
    with h5py.File(file, "r") as handle:
        rows = handle["assays/dna_variants/ra"]
        ids = [
            f"{sample.decode()}__{barcode.decode()}"
            for sample, barcode in zip(
                rows["sample_name"][:], rows["barcode"][:], strict=True
            )
        ]
    return write_cells_store(tmp_path / "large.zarr", ids)


def _smallest_and_peak(store, run) -> tuple[int, int]:
    from tests.memory_helpers import at_budget, smallest_budget, traced_peak

    store.memoryBytes = 1 << 33
    run()
    budget = smallest_budget(at_budget(store, run), 1 << 30)
    store.memoryBytes = budget
    return budget, traced_peak(run, budget)[1]


@pytest.mark.slow
def test_the_traced_peak_of_a_tapestri_import_fits_the_smallest_budget_accepted(
    tmp_path,
) -> None:
    from tests.variant_truth import write_large_tapestri

    file = write_large_tapestri(tmp_path / "large.h5", n_cells=8_000, n_variants=40)
    store = _large_store(tmp_path, file)
    cells = every_cell(store)

    def run() -> None:
        store.variants.import_counts(file, cell_selection=cells, invalidate_cache=True)

    budget, peak = _smallest_and_peak(store, run)
    assert peak <= budget, (peak, budget)


# The arrays of the variant counts that the import appends to a band at a time.
_LONG_ARRAYS = (
    "depth_sites",
    "depth",
    "count_alleles",
    "counts",
    "call_alleles",
    "call_sources",
    "call_values",
    "call_quality",
)


def _chunk_writes(monkeypatch) -> Counter[tuple[str, int]]:
    """Count the writes that the store receives for each chunk of the variant counts' arrays."""
    import zarr.storage

    writes: Counter[tuple[str, int]] = Counter()
    write = zarr.storage.LocalStore.set

    async def counted(self, key, value) -> None:
        parts = key.split("/")
        if parts[:2] == ["artifacts", "variant_counts"] and parts[-2:-1] == ["c"]:
            writes[parts[3], int(parts[-1])] += 1
        await write(self, key, value)

    monkeypatch.setattr(zarr.storage.LocalStore, "set", counted)
    return writes


@pytest.fixture(scope="module")
def banded(tmp_path_factory) -> tuple[Path, Path]:
    """A Tapestri file of 300 cells and 20 variants, at most 18,000 rows of depth, counts and calls, and a store of its cells."""
    from tests.variant_truth import write_large_tapestri

    folder = tmp_path_factory.mktemp("banded")
    file = write_large_tapestri(folder / "large.h5", n_cells=300, n_variants=20)
    _large_store(folder, file)
    return file, folder / "large.zarr"


def _import_at_smallest_budget(
    banded, tmp_path, writes: Counter[tuple[str, int]]
) -> tuple[int, BaseException | None, Any]:
    """Import the file at the smallest budget that its refusals lead to.

    The refusals are followed from one byte: each states the bytes to try
    next, except the parse's, which states what the import holds before it
    reads a row, so a quarter more is tried then. The import's traced peak
    must fit the budget that runs it, and ``writes`` keeps the chunk writes
    of that import alone.

    Returns:
        That budget, the refusal one byte below it, None if it ran, and the
        stored counts.
    """
    from tests.memory_helpers import stated_bytes

    file, location = banded
    store = copy_store(location, tmp_path)
    cells = every_cell(store)

    def run() -> Any:
        return store.variants.import_counts(
            file, cell_selection=cells, invalidate_cache=True
        )

    budget = 1
    while True:
        store.memoryBytes = budget
        writes.clear()
        try:
            ref, peak = traced_peak(run, budget)
        except MemoryError as refusal:
            stated = stated_bytes(refusal)
            budget = stated if stated > budget else budget + max(1, budget // 4)
        else:
            break
    assert peak <= budget, (peak, budget)
    ran = Counter(writes)
    store.memoryBytes = budget - 1
    try:
        run()
    except MemoryError as refusal:
        below: BaseException | None = refusal
    else:
        below = None
    writes.clear()
    writes.update(ran)
    return budget, below, store.artifacts.load(ref)


def test_the_smallest_budget_accepted_writes_each_stored_chunk_once(
    banded, tmp_path, monkeypatch
) -> None:
    from tests.memory_helpers import stated_bytes

    # Appending a band rewrites the stored chunks that it ends in, so a band
    # holds a stored chunk's rows at least. The 300 cells' rows fit one chunk
    # of each array, so one band holds them all and writes each chunk once,
    # or not at all where Zarr skips a chunk of zeros, its fill value.
    writes = _chunk_writes(monkeypatch)
    budget, below, group = _import_at_smallest_budget(banded, tmp_path, writes)

    for name in _LONG_ARRAYS:
        values = group[name][:]
        assert 0 < len(values) <= group[name].chunks[0], name
        assert writes[name, 0] == int(np.any(values)), (name, writes)
        assert (name, 1) not in writes, name
    assert writes["depth", 0] == writes["counts", 0] == 1
    # The check of the bands refuses one byte less, naming the budget.
    assert re.match(r"Writing the variant counts in bands of \d+ rows ", str(below))
    assert stated_bytes(below) == budget


def test_a_band_holds_a_chunk_of_rows_when_the_rows_outnumber_a_chunk(
    banded, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import variants as operations
    from tests.memory_helpers import stated_bytes

    # Chunks of at most 12,000 rows, fewer than the 300 cells hold: a band
    # holds at least 12,000 rows, less the 60 rows of one cell where a band
    # ends, so two bands hold the at most 18,000 rows, and each writes a
    # chunk at most once.
    monkeypatch.setattr(operations, "_ROW_BAND", 12_000)
    writes = _chunk_writes(monkeypatch)
    budget, below, group = _import_at_smallest_budget(banded, tmp_path, writes)

    assert max(group[name].chunks[0] for name in _LONG_ARRAYS) <= 12_000
    assert max(writes.values()) == 2, writes
    assert writes["depth", 0] == writes["counts", 0] == 2
    assert str(below).startswith("Writing the variant counts in bands of 12000 rows ")
    assert stated_bytes(below) == budget


@pytest.mark.slow
@pytest.mark.parametrize(
    "rules", [{}, {"min_quality": 30, "min_depth": 30}], ids=["adopted", "filtered"]
)
def test_source_genotypes_of_several_alternatives_fit_the_smallest_budget_accepted(
    tmp_path, rules
) -> None:
    from tests.variant_truth import write_large_tapestri

    # Three variants share each position, so each depth row covers three
    # genotyped units, which the depth that the rules compare covers too.
    file = write_large_tapestri(
        tmp_path / "large.h5", n_cells=8_000, n_variants=30, alternatives=3
    )
    store = _large_store(tmp_path, file)
    counts = store.variants.import_counts(file, cell_selection=every_cell(store))
    loaded = store.variants.load_counts(counts, cell_selection=[])
    assert len(loaded.sites) == 10 and len(loaded.alleles) == 40

    def run() -> None:
        store.variants.genotypes(
            counts, method="source", invalidate_cache=True, **rules
        )

    budget, peak = _smallest_and_peak(store, run)
    assert peak <= budget, (peak, budget)


def test_a_spanning_deletion_is_read_as_its_allele(store, tmp_path) -> None:
    # VCF's * stands for a deletion that spans the position, beside a
    # reference of any length, so it is not extended.
    def edit(handle) -> None:
        _rename_variant(handle, 9, "G", "*")
        _rename_variant(handle, 4, "T", "*")

    path = _edit(write_tapestri(TRUTH, tmp_path / "run.h5"), edit)
    loaded = store.variants.load_counts(_import(store, path)[0])
    alleles = loaded.alleles.groupby("site", observed=True)["allele"].apply(set)
    assert alleles["chr7:400"] == {"G", "*"}
    assert alleles["chr1:2000"] == {"TC", "T", "*"}
    shown = loaded.counts[loaded.counts["allele"] == "*"]
    expected = {
        (TRUTH.row_ids[row], site): int(TRUTH.alternative[row, column])
        for column, site in ((9, "chr7:400"), (4, "chr1:2000"))
        for row in range(TRUTH.n_rows)
        if TRUTH.alternative[row, column]
    }
    assert {
        (str(cell), str(site)): int(count)
        for cell, site, count in zip(
            shown["ids"], shown["site"], shown["count"], strict=True
        )
    } == expected


def test_unit_labels_name_an_allele_without_a_reference() -> None:
    from cytearc.datastore._operations.variants import _unit_alleles, _unit_labels
    from cytearc.variants.counts import Contig, SiteTable

    # A site with coordinates whose files name no reference allele.
    table = SiteTable(
        contigs=(Contig("chr1"),),
        site_contig=np.zeros(2, dtype=np.int32),
        site_position=np.asarray([5, 9]),
        site_name=np.full(2, "", dtype=object),
        site_channel=np.full(2, "", dtype=object),
        allele_site=np.asarray([0, 1, 1]),
        allele=np.asarray(["C", "A", "G"], dtype=object),
        allele_role=np.asarray(
            ["alternative", "reference", "alternative"], dtype=object
        ),
    )
    genotyped = np.asarray([0, 1, 1])
    units = np.asarray([0, 2, -1])
    alleles = _unit_alleles(table, genotyped, units)
    assert alleles == [("", "C"), ("A", "G"), ("", "")]
    assert _unit_labels(table.labels(), genotyped, units, alleles) == (
        "chr1:5:C",
        "chr1:9:A/G",
        "chr1:9",
    )


def test_calls_and_labels_are_coded_whatever_their_number() -> None:
    from cytearc.datastore._operations.variants import (
        _factorized,
        _label_codes,
        _Vocabulary,
    )

    vocabulary = _Vocabulary()
    assert vocabulary.codes(np.asarray(["2", "0", "2"], dtype=object)).tolist() == [
        0,
        1,
        0,
    ]
    assert vocabulary.code("0") == 1 and vocabulary.code("3") == 2
    assert vocabulary.codes(np.asarray([], dtype=object)).tolist() == []
    # Stored, the texts are sorted and the codes follow them.
    texts, remap = vocabulary.sorted()
    assert texts == ["0", "2", "3"] and remap.tolist() == [1, 0, 2]
    assert _label_codes(
        {"": 0, "1": 1, "3": 2}, np.asarray(["3", "", "3"])
    ).tolist() == [
        2,
        0,
        2,
    ]
    with pytest.raises(ValueError, match="missing where text was expected"):
        _factorized(np.asarray(["0", None], dtype=object))


def test_a_source_must_give_each_call_one_quality() -> None:
    from cytearc.datastore._operations.variants import (
        _VariantOperationsMixin,
        _Vocabulary,
    )
    from cytearc.variants.counts import (
        CALL_DTYPE,
        DEPTH_DTYPE,
        EntryBatch,
        count_dtype,
    )
    from cytearc.variants.sorting import CellRowSorter

    counts_dtype = count_dtype(())
    sorter = CellRowSorter(
        {
            "counts": (counts_dtype, ("cell", "key")),
            "depth": (DEPTH_DTYPE, ("cell", "key")),
            "calls": (CALL_DTYPE, ("cell", "key", "source")),
        },
        n_cells=2,
        buffer_bytes=1 << 20,
    )

    def add(quality: np.ndarray, call_fields: tuple[str, ...] = ("quality",)) -> None:
        _VariantOperationsMixin._add_variant_batch(
            sorter,
            EntryBatch(
                kind="calls",
                barcodes=np.asarray(["a", "b"], dtype=object),
                barcode_codes=np.asarray([0, 1]),
                keys=np.asarray([0, 0]),
                call_source="NGT",
                calls=np.asarray(["0", "1"], dtype=object),
                call_quality=quality,
            ),
            cells=np.asarray([0, 1]),
            site_map=np.asarray([0]),
            allele_map=np.asarray([0]),
            counts_dtype=counts_dtype,
            listed=np.zeros(2, dtype=bool),
            call_sources=_Vocabulary(),
            call_values=_Vocabulary(),
            path="run.h5",
            field_paths={},
            call_fields=call_fields,
        )

    with sorter:
        with pytest.raises(ValueError, match="run.h5 gives 1 call qualities for 2"):
            add(np.asarray([30.0]))
        # A source whose format lists no call quality may not give one.
        with pytest.raises(ValueError, match="run.h5 gives call qualities, which"):
            add(np.asarray([30.0, 99.0]), call_fields=())
        add(np.asarray([30.0, 99.0]))
        add(np.empty(0))
        rows = next(sorter.bands(10))[2]["calls"]
    assert rows["quality"][np.isfinite(rows["quality"])].tolist() == [30.0, 99.0]
    assert int(np.isnan(rows["quality"]).sum()) == 2


def _write_texts(group, name: str, values: list[str], variable_length: bool) -> None:
    if variable_length:
        group.create_dataset(
            name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype()
        )
    else:
        group.create_dataset(name, data=np.char.encode(np.asarray(values)))


def _barcodes(n_cells: int) -> list[str]:
    return [
        "".join("ACGT"[(cell >> (2 * digit)) & 3] for digit in range(18))
        for cell in range(n_cells)
    ]


def _alternatives(n_variants: int, length: int) -> list[str]:
    """``G`` for a SNV, or an insertion of ``length`` bases after the ``A``."""
    if length == 1:
        return ["G"] * n_variants
    rng = np.random.default_rng(5)
    return ["A" + "".join(rng.choice(list("ACGT"), length)) for _ in range(n_variants)]


def _wide_tapestri(
    path: Path,
    *,
    n_cells: int,
    n_variants: int,
    sample: str,
    alternative_length: int = 1,
    variable_length: bool = False,
) -> Path:
    """A Tapestri file of many candidate variants of which two pass."""
    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as handle:
        variants = handle.create_group("assays/dna_variants")
        columns = variants.create_group("ca")
        positions = 1_000 + np.arange(n_variants)
        alternatives = _alternatives(n_variants, alternative_length)
        names = [
            f"chr1:{position}:A/{alternative}"
            for position, alternative in zip(positions, alternatives, strict=True)
        ]
        for name, values in (
            ("id", names),
            ("CHROM", ["1"] * n_variants),
            ("REF", ["A"] * n_variants),
            ("ALT", alternatives),
        ):
            _write_texts(columns, name, values, variable_length)
        columns.create_dataset("POS", data=positions.astype(np.int32))
        filtered = np.ones(n_variants, dtype=np.uint8)
        filtered[[0, n_variants // 2]] = 0
        columns.create_dataset("filtered", data=filtered)
        rows = variants.create_group("ra")
        _write_texts(rows, "barcode", _barcodes(n_cells), variable_length)
        _write_texts(rows, "sample_name", [sample] * n_cells, variable_length)
        depth = rng.poisson(20, size=(n_cells, n_variants)).astype(np.int16)
        for name, values in (
            ("DP", depth),
            ("AF", np.zeros(depth.shape)),
            ("NGT", np.zeros(depth.shape, dtype=np.int8)),
            ("GQ", np.full(depth.shape, 99, dtype=np.int8)),
        ):
            variants.create_dataset(
                f"layers/{name}",
                data=values,
                chunks=(min(64, n_cells), 64),
                compression="gzip",
            )
    return path


def _wide_dab_seq(
    path: Path,
    *,
    n_cells: int,
    n_variants: int,
    alternative_length: int,
    variable_length: bool,
) -> Path:
    """A DAb-seq file of many variants, every cell without reads."""
    with h5py.File(path, "w") as handle:
        positions = 1_000 + np.arange(n_variants)
        alternatives = _alternatives(n_variants, alternative_length)
        _write_texts(
            handle,
            "VARIANTS",
            [
                f"GENE:chr1:{position}:A/{alternative}"
                for position, alternative in zip(positions, alternatives, strict=True)
            ],
            variable_length,
        )
        _write_texts(
            handle,
            "CELL_BARCODES",
            [f"{barcode}-1-s" for barcode in _barcodes(n_cells)],
            variable_length,
        )
        for name, dtype in (
            ("AD", np.int16),
            ("RD", np.int16),
            ("DP", np.int16),
            ("GQ", np.int8),
            ("GT", np.int8),
        ):
            handle.create_dataset(
                name, data=np.zeros((n_cells, n_variants), dtype), chunks=(50, 64)
            )
    return path


@pytest.mark.parametrize(
    ("file_format", "alternative_length", "variable_length"),
    [
        ("tapestri", 1, False),
        pytest.param("tapestri", 1_000, False, marks=pytest.mark.slow),
        pytest.param("tapestri", 1_000, True, marks=pytest.mark.slow),
        ("dab_seq", 1, False),
        pytest.param("dab_seq", 1_000, False, marks=pytest.mark.slow),
        pytest.param("dab_seq", 1_000, True, marks=pytest.mark.slow),
    ],
)
def test_describing_a_file_holds_no_more_than_the_smallest_budget_accepted(
    tmp_path, file_format, alternative_length, variable_length
) -> None:
    from cytearc.readers.tapestri import dab_seq_source, tapestri_source
    from tests.memory_helpers import traced_peak

    # SNVs make what the description holds for each variant most of it, and
    # insertions of 1,000 bases the variants' texts, however the file stores
    # them.
    options = {
        "n_cells": 50,
        "n_variants": 2_000,
        "alternative_length": alternative_length,
        "variable_length": variable_length,
    }
    if file_format == "tapestri":
        path = _wide_tapestri(tmp_path / "wide.h5", sample="s", **options)
        describe = tapestri_source
    else:
        path = _wide_dab_seq(tmp_path / "wide.hdf5", **options)
        describe = dab_seq_source
    with pytest.raises(MemoryError, match=r"holds about \d+ bytes") as refused:
        describe(path, memory_bytes=1)
    found = re.search(r"holds about (\d+) bytes", str(refused.value))
    assert found is not None
    # The smallest budget accepted is the one the refusal names.
    budget = int(found.group(1))
    with pytest.raises(MemoryError):
        describe(path, memory_bytes=budget - 1)
    _source, peak = traced_peak(lambda: describe(path, memory_bytes=budget), budget)
    assert peak <= budget, (peak, budget)


@pytest.mark.slow
def test_the_traced_peak_of_a_wide_tapestri_import_fits_the_smallest_budget(
    tmp_path,
) -> None:
    # Describing 50,000 candidate variants holds about 31 MB, far more than
    # their two that pass and the 200 cells need, which without the check of
    # the description would let the import accept a budget of 17 MB.
    path = _wide_tapestri(
        tmp_path / "wide.h5", n_cells=200, n_variants=50_000, sample="s"
    )
    store = _large_store(tmp_path, path)
    cells = every_cell(store)

    def run() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            store.variants.import_counts(
                path, cell_selection=cells, invalidate_cache=True
            )

    budget, peak = _smallest_and_peak(store, run)
    assert peak <= budget, (peak, budget)


def test_a_source_refuses_a_cell_whose_sample_or_barcode_is_empty(tmp_path) -> None:
    # Each block of cells joins their samples to their barcodes as it reads
    # them, so an empty part is refused there as the store reader refuses it.
    from cytearc.readers.tapestri import tapestri_source

    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        source = tapestri_source(path)
    with h5py.File(path, "a") as handle:
        rows = handle["assays/dna_variants/ra"]
        barcodes = rows["barcode"][:]
        barcodes[3] = b""
        del rows["barcode"]
        rows.create_dataset("barcode", data=barcodes)
    # The row counts in the file, whichever block of cells holds it.
    for cells in (10_000, 3, 1):
        with pytest.raises(
            ValueError,
            match="has cells without a sample name or a barcode, the first in row 4;",
        ):
            list(source.batches(cells))
