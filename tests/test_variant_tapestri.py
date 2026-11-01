"""Importing Tapestri and DAb-seq genotypes as variant counts, and their calls.

The files are written from the synthetic truth of ``tests/variant_truth.py``
in the layouts that Mission Bio documents and the DAb-seq pipeline writes,
and every import must recover the truth's depth, reads and calls exactly.
"""

import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    copy_clone_store,
    every_cell,
    loaded_partition,
    write_cells_store,
)
from tests.variant_helpers import (
    DABSEQ_ITD_SITE,
    TAPESTRI_ALLELES,
    TAPESTRI_PASSING,
    TAPESTRI_REFERENCES,
    assert_tables,
    dabseq_ids,
    expected_dabseq,
    expected_tapestri,
)
from tests.variant_truth import (
    TAPESTRI_CLONE_VARIANTS,
    TAPESTRI_VARIANTS,
    tapestri_truth,
    write_dabseq,
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
    return copy_clone_store(template, tmp_path)


@pytest.fixture(scope="module")
def dabseq_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("dabseq") / "store.zarr"
    write_cells_store(path, dabseq_ids(TRUTH))
    return path


@pytest.fixture
def dabseq_store(dabseq_template, tmp_path):
    return copy_clone_store(dabseq_template, tmp_path)


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
    }
    assert any(re.search(_SKIPPED, message) for message in messages)


def test_the_cells_of_two_samples_with_one_barcode_stay_apart(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    loaded = store.variants.load_counts(_import(store, path)[0])
    twin = TRUTH.row_ids[TRUTH.twin_row]
    first = TRUTH.row_ids[0]
    assert twin.split("__")[1] == first.split("__")[1]
    assert twin != first
    depth = loaded.depth.set_index(["cell", "site"])["depth"]
    for row, cell in ((0, first), (TRUTH.twin_row, twin)):
        assert int(depth[(cell, "chr7:400")]) == int(TRUTH.depth[row, 9])


def test_variants_add_failing_variants_and_positions(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    ref, _ = _import(store, path, variants=["chr2:500"])
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
        (loaded.counts["cell"] == cell) & (loaded.counts["site"] == "chr2:500")
    ]
    assert sorted(zip(shown["allele"], shown["count"], strict=True)) == [
        ("C", 1),
        ("G", 1),
    ]
    assert (
        int(
            loaded.depth[
                (loaded.depth["cell"] == cell) & (loaded.depth["site"] == "chr2:500")
            ]["depth"].iloc[0]
        )
        == 1
    )
    assert _source_record(store, ref)["n_imported_variants"] == 7
    # Naming a passing variant, or a variant by its full name, adds nothing,
    # but the list is part of the identity.
    named, _ = _import(store, path, variants=["chr2:500:A/C"])
    assert named != ref
    assert_tables(store.variants.load_counts(named).counts, counts)


def test_variants_that_cannot_be_read_are_skipped_or_refused(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    # A position takes its unreadable variants too, which are skipped.
    with pytest.warns(
        UserWarning,
        match=re.escape("1 chosen variants with an empty ALT and 1 whose alleles"),
    ):
        ref = store.variants.import_counts(
            path, cell_selection=every_cell(store), variants=["chr2:800"]
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
                path, cell_selection=every_cell(store), variants=[name]
            )
    with pytest.raises(ValueError, match="'chr9:1', which is neither a variant"):
        store.variants.import_counts(
            path, cell_selection=every_cell(store), variants=["chr9:1"]
        )
    for wrong in ("chr2:500", [1], b"chr2:500"):
        with pytest.raises(TypeError, match="variants must be a sequence"):
            store.variants.import_counts(
                path, cell_selection=every_cell(store), variants=wrong
            )


def test_variants_are_refused_for_other_formats(store, tmp_path) -> None:
    got = write_got(variant_truth(), tmp_path / "CALR.tsv", style="geo_total")
    with pytest.raises(
        ValueError, match="variants chooses the variants of Tapestri and DAb-seq"
    ):
        store.variants.import_counts(
            got, cell_selection=every_cell(store), variants=["chr1:1"]
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


def test_dabseq_counts_are_its_reads_and_calls(dabseq_store, tmp_path) -> None:
    path = write_dabseq(TRUTH, tmp_path / "GSM0.hdf5")
    with pytest.warns(
        UserWarning,
        match=re.escape("2 chosen variants with an empty ALT and 1 whose alleles"),
    ):
        ref = dabseq_store.variants.import_counts(
            path, cell_selection=every_cell(dabseq_store)
        )
    loaded = dabseq_store.variants.load_counts(ref)
    depth, counts, calls = expected_dabseq(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert_tables(loaded.calls.drop(columns="quality"), calls.drop(columns="quality"))
    # A missing GQ is NaN, and every other quality is the file's.
    merged = loaded.calls.merge(calls, on=["cell", "site", "allele"])
    np.testing.assert_array_equal(merged["quality_x"], merged["quality_y"])
    assert merged["quality_x"].isna().sum() == (calls["quality"].isna()).sum() > 0
    assert loaded.sites.index.tolist() == [*TAPESTRI_REFERENCES, DABSEQ_ITD_SITE]
    attributes = dabseq_store.artifacts.load(ref).attrs
    assert attributes["format"] == "dabseq"
    record = _source_record(dabseq_store, ref)
    assert (
        record["n_variants"],
        record["n_imported_variants"],
        record["n_skipped_empty_alternative"],
        record["n_skipped_other_alleles"],
    ) == (11, 8, 2, 1)


def test_tapestri_and_dabseq_files_are_recognised(tmp_path) -> None:
    from cytearc.readers._variant_files import hdf5_format

    assert hdf5_format(str(write_tapestri(TRUTH, tmp_path / "a.h5"))) == "tapestri"
    assert hdf5_format(str(write_dabseq(TRUTH, tmp_path / "b.hdf5"))) == "dabseq"
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
            lambda h: _replace(h[_COLUMNS], "POS", np.arange(3, dtype=np.int32)),
            "The arrays of assays/dna_variants/ca of .* do not align",
        ),
    ],
)
def test_malformed_tapestri_files_are_refused(store, tmp_path, edit, message) -> None:
    path = _edit(write_tapestri(TRUTH, tmp_path / "run.h5"), edit)
    with pytest.raises(ValueError, match=message):
        _import(store, path)


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
        loaded = store.variants.load_counts(_import(store, path)[0], cells=[])
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
    dabseq = write_dabseq(TRUTH, tmp_path / "c.hdf5")
    with pytest.raises(ValueError, match="lacks assays/dna_variants.*format='dabseq'"):
        _import(store, dabseq, format="tapestri")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda h: _replace(
                h,
                "VARIANTS",
                np.asarray(["NRAS:chr1:1000:G>A", *_texts(h, "VARIANTS")[1:]]),
            ),
            r"VARIANTS\[0\] of .* is 'NRAS:chr1:1000:G>A', not <gene>:<chrom>",
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
def test_malformed_dabseq_files_are_refused(
    dabseq_store, tmp_path, edit, message
) -> None:
    path = _edit(write_dabseq(TRUTH, tmp_path / "GSM0.hdf5"), edit)
    with pytest.raises(ValueError, match=message):
        _import(dabseq_store, path)


def _call_name(genotype: int, depth: int) -> str | None:
    """Tapestri's NGT as a call, None for a cell that the calls leave out."""
    if genotype == 3:
        return "filtered" if depth > 0 else None
    return ("wild_type", "heterozygous", "homozygous_alternative")[genotype]


def test_source_genotypes_of_tapestri_counts_follow_ngt(store, tmp_path) -> None:
    path = write_tapestri(TRUTH, tmp_path / "run.h5")
    counts, _ = _import(store, path)
    genotypes = store.variants.genotypes(counts, method="source")
    loaded = store.variants.load_genotypes(genotypes)
    # One genotype per variant, named as Tapestri names it, in the order of
    # the counts' alleles.
    units = {
        0: "chr1:1000:G/A",
        1: "chr1:1000:G/T",
        4: "chr1:2000:TC/GC",
        3: "chr1:2000:TC/T",
        5: "chr2:500:A/G",
        9: "chr7:400:G/A",
    }
    assert loaded.sites.index.tolist() == list(units.values())
    assert loaded.sites["reference"].tolist() == ["G", "G", "TC", "TC", "A", "G"]
    assert loaded.sites["alternative"].tolist() == ["A", "T", "GC", "T", "G", "A"]
    assert set(loaded.sites["source"]) == {"NGT"}
    ids = TRUTH.row_ids
    expected = pd.DataFrame(
        [
            (ids[row], unit, call, str(int(TRUTH.genotypes[row, column])))
            for column, unit in units.items()
            for row in range(TRUTH.n_rows)
            if (
                call := _call_name(
                    int(TRUTH.genotypes[row, column]), int(TRUTH.depth[row, column])
                )
            )
            is not None
        ],
        columns=["cell", "site", "call", "label"],
    )
    assert_tables(loaded.calls.astype({"call": str, "label": str}), expected)
    calls = loaded.calls.set_index(["cell", "site"])
    # No call with reads is filtered, without reads missing, and a wild-type
    # call without reads is kept as the file wrote it.
    assert calls.loc[(ids[TRUTH.no_call_row], "chr7:400:G/A"), "call"] == "filtered"
    assert (ids[TRUTH.uncovered_row], "chr7:400:G/A") not in calls.index
    assert calls.loc[(ids[TRUTH.wild_type_row], "chr1:1000:G/A"), "call"] == "wild_type"
    assert calls.loc[(ids[TRUTH.overlap_row], "chr2:500:A/G"), "call"] == "filtered"


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


def test_dabseq_genotypes_adopt_gt_and_flt3_itd(dabseq_store, tmp_path) -> None:
    path = write_dabseq(TRUTH, tmp_path / "GSM0.hdf5")
    counts, _ = _import(dabseq_store, path)
    genotypes = dabseq_store.variants.genotypes(counts, method="source")
    loaded = dabseq_store.variants.load_genotypes(genotypes)
    assert loaded.sites.index.tolist() == [
        "chr1:1000:G/A",
        "chr1:1000:G/T",
        "chr1:2000:TC/GC",
        "chr1:2000:TC/T",
        "chr2:500:A/C",
        "chr2:500:A/G",
        "chr7:400:G/A",
        "chr13:28608251:T/TCTC",
    ]
    assert set(loaded.sites["source"]) == {"GT"}
    itd = loaded.calls[loaded.calls["site"] == "chr13:28608251:T/TCTC"]
    clone_4 = {
        dabseq_ids(TRUTH)[row] for row in np.flatnonzero(TRUTH.clones.clones == 4)
    }
    assert set(itd[itd["call"] == "heterozygous"]["cell"].astype(str)) == clone_4
    assert len(itd) == TRUTH.n_rows
    assert (itd[itd["call"] != "heterozygous"]["call"] == "wild_type").all()


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
    from tests.clone_helpers import smallest_budget, traced_peak

    store.memoryBytes = 1 << 33
    run()

    def at(budget: int) -> None:
        store.memoryBytes = budget
        run()

    budget = smallest_budget(at, 1 << 30)
    store.memoryBytes = budget
    return budget, traced_peak(run)


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


@pytest.mark.slow
def test_source_genotypes_of_several_alternatives_fit_the_smallest_budget_accepted(
    tmp_path,
) -> None:
    from tests.variant_truth import write_large_tapestri

    # Three variants share each position, so each depth row covers three
    # genotyped units.
    file = write_large_tapestri(
        tmp_path / "large.h5", n_cells=8_000, n_variants=30, alternatives=3
    )
    store = _large_store(tmp_path, file)
    counts = store.variants.import_counts(file, cell_selection=every_cell(store))
    loaded = store.variants.load_counts(counts, cells=[])
    assert len(loaded.sites) == 10 and len(loaded.alleles) == 40

    def run() -> None:
        store.variants.genotypes(counts, method="source", invalidate_cache=True)

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
            shown["cell"], shown["site"], shown["count"], strict=True
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
        _sorted_unique,
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
    values = np.asarray([5, 3, 5, 9, 3, 3], dtype=np.int64)
    assert _sorted_unique(values).tolist() == [3, 5, 9]
    assert _sorted_unique(values[:1]).tolist() == [5]
    assert _sorted_unique(values[:0]).tolist() == []


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

    def add(quality: np.ndarray) -> None:
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
        )

    with sorter:
        with pytest.raises(ValueError, match="run.h5 gives 1 call qualities for 2"):
            add(np.asarray([30.0]))
        add(np.asarray([30.0, 99.0]))
        add(np.empty(0))
        rows = next(sorter.bands(10))[2]["calls"]
    assert rows["quality"][np.isfinite(rows["quality"])].tolist() == [30.0, 99.0]
    assert int(np.isnan(rows["quality"]).sum()) == 2
