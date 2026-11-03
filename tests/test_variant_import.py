"""Importing per-cell allele counts of mitochondrial and targeted variants.

Each format is written from one synthetic truth by ``tests/variant_truth.py``,
and every import must recover the truth's depth, counts and calls exactly.
"""

import gzip
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.metadata.rows import text_read_bytes
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import every_cell
from tests.memory_helpers import at_budget, smallest_budget, traced_peak
from tests.storage_helpers import copy_store
from tests.variant_helpers import (
    assert_tables,
    expected_gift,
    expected_got,
    expected_got_calls,
    expected_mito,
    expected_variant_sites,
    mito_label,
    mito_reference_digest,
    write_variant_store,
)
from tests.variant_truth import (
    BASES,
    CONTIG,
    N_POSITIONS,
    REFERENCE,
    SITE_POSITIONS,
    UNCOVERED_CELL,
    variant_truth,
    write_cellsnp,
    write_giftwrap_h5,
    write_giftwrap_tables,
    write_got,
    write_got_multi,
    write_gotcha,
    write_mgatk,
    write_mito_afm,
    write_mquad,
    write_vartrix,
)

TRUTH = variant_truth()
_UNMATCHED = "3 of 51 barcodes of the variant file match no selected cell"


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("variants") / "store.zarr"
    write_variant_store(path, TRUTH)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


def _import(store, path, **options):
    with collect_warnings() as messages, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = store.variants.import_counts(
            path, cell_selection=every_cell(store), **options
        )
    return ref, list(messages)


def _refused_estimate(store, run, budget: int) -> int:
    """The bytes that ``run`` names when it refuses ``budget``."""
    import re

    full = store.memoryBytes
    store.memoryBytes = budget
    try:
        with pytest.raises(MemoryError) as refused:
            run()
    finally:
        store.memoryBytes = full
    found = re.search(r"holds about (\d+) bytes", str(refused.value))
    assert found is not None, str(refused.value)
    return int(found.group(1))


def _large_mgatk(
    folder: Path, store_ids: list[str], seed: int, n_positions: int = 1_000
) -> dict[str, np.ndarray]:
    """An mgatk folder of many positions, its cells in shuffled order.

    Returns each cell's reads per position and base, forward and reverse,
    shaped (cells, positions, bases, strands).
    """
    rng = np.random.default_rng(seed)
    reference = rng.choice(list("ACGT"), size=n_positions)
    reference[100] = "N"
    reads = rng.poisson(0.6, size=(len(store_ids), n_positions, 4, 2))
    folder.mkdir(parents=True)
    order = rng.permutation(len(store_ids))
    for base_index, base in enumerate("ACGT"):
        frames = []
        for cell in order.tolist():
            forward = reads[cell, :, base_index, 0]
            reverse = reads[cell, :, base_index, 1]
            kept = (forward + reverse) > 0
            frames.append(
                pd.DataFrame(
                    {
                        "position": np.flatnonzero(kept) + 1,
                        "barcode": store_ids[cell],
                        "forward": forward[kept],
                        "reverse": reverse[kept],
                    }
                )
            )
        text = pd.concat(frames).to_csv(header=False, index=False)
        (folder / f"big.{base}.txt.gz").write_bytes(
            gzip.compress(text.encode(), mtime=0)
        )
    (folder / "chrM_refAllele.txt").write_text(
        "".join(f"{position + 1}\t{base}\n" for position, base in enumerate(reference))
    )
    return {"reads": reads, "reference": reference}


def test_mgatk_counts_every_base_and_strand_of_every_cell(store, tmp_path) -> None:
    folder = write_mgatk(TRUTH, tmp_path / "final")
    ref, messages = _import(store, folder)

    loaded = store.variants.load_counts(ref)
    depth, counts = expected_mito(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert loaded.calls.empty
    assert (loaded.format, loaded.count_unit, loaded.depth_definition) == (
        "mgatk",
        "reads",
        "all",
    )
    # Every position but the N is a site, with four bases, the reference's
    # uppercased base its reference.
    assert loaded.sites.index.tolist() == [mito_label(p) for p in SITE_POSITIONS]
    assert loaded.sites["reference"].tolist() == [
        REFERENCE[p - 1] for p in SITE_POSITIONS
    ]
    assert len(loaded.alleles) == 4 * len(SITE_POSITIONS)
    assert loaded.contigs.to_dict("records") == [
        {
            "name": CONTIG,
            "length": N_POSITIONS,
            "sequence_sha256": mito_reference_digest(),
        }
    ]
    assert messages == [
        _UNMATCHED
        + ", for example "
        + ", ".join(repr(f"d2__{barcode}-1") for barcode in TRUTH.clones.outside)
        + ". Check barcode_prefix and cell_selection."
    ]


def test_mean_depth_covered_sites_and_listed_cells_come_from_the_truth(
    store, tmp_path
) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    cells = store.variants.load_counts(ref).cells
    reads = TRUTH.mito.sum(axis=(2, 3))[: TRUTH.n_cells][
        :, np.asarray(SITE_POSITIONS) - 1
    ]
    np.testing.assert_allclose(
        cells["mean_depth"].to_numpy(), reads.sum(axis=1) / len(SITE_POSITIONS)
    )
    np.testing.assert_array_equal(
        cells["n_covered_sites"].to_numpy(), (reads > 0).sum(axis=1)
    )
    assert cells["listed"].all()
    # The cell without reads is in the depth table: listed, at depth 0.
    assert cells["mean_depth"].iloc[UNCOVERED_CELL] == 0.0
    assert cells["n_covered_sites"].iloc[UNCOVERED_CELL] == 0
    # Site sums cover every cell.
    sites = store.variants.load_counts(ref, cell_selection=[]).sites
    np.testing.assert_array_equal(sites["depth"].to_numpy(), reads.sum(axis=0))
    np.testing.assert_array_equal(sites["n_cells"].to_numpy(), (reads > 0).sum(axis=0))


def test_without_a_depth_table_cells_with_reads_are_the_listed_ones(
    store, tmp_path
) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final", depth_table=False))
    cells = store.variants.load_counts(ref).cells
    assert not cells["listed"].iloc[UNCOVERED_CELL]
    assert np.isnan(cells["mean_depth"].iloc[UNCOVERED_CELL])
    assert cells["listed"].sum() == TRUTH.n_cells - 1


def test_six_column_files_keep_qualities_and_need_a_format(store, tmp_path) -> None:
    folder = write_mgatk(TRUTH, tmp_path / "final", six_columns=True)
    with pytest.raises(ValueError, match="pass format='mgatk' or format='maegatk'"):
        store.variants.import_counts(folder, cell_selection=every_cell(store))
    ref, _ = _import(store, folder, format="mgatk")
    loaded = store.variants.load_counts(ref)
    _depth, counts = expected_mito(TRUTH)
    assert_tables(loaded.counts, counts)
    # Each strand's mean quality, missing where the strand has no reads.
    merged = loaded.counts.assign(cell=loaded.counts["ids"].astype(str))
    for row in merged.sample(40, random_state=0).itertuples():
        truth_row = TRUTH.row_ids.index(row.cell)
        position = int(row.site.split(":")[1])
        base = BASES.index(row.allele)
        for strand, value, count in (
            (0, row.quality_forward, row.forward),
            (1, row.quality_reverse, row.reverse),
        ):
            expected = TRUTH.quality[truth_row, position - 1, base, strand]
            if count:
                assert value == pytest.approx(expected, abs=1e-4)
            else:
                assert np.isnan(value)


def test_maegatk_strands_are_swapped_back(store, tmp_path) -> None:
    folder = write_mgatk(TRUTH, tmp_path / "final", six_columns=True, swapped=True)
    ref, _ = _import(store, folder, format="maegatk")
    loaded = store.variants.load_counts(ref)
    _depth, counts = expected_mito(TRUTH)
    assert_tables(loaded.counts, counts)
    assert loaded.count_unit == "molecules"
    record = store.artifacts.inspect(ref).inputs["sources"][0]
    assert (record["format"], record["strands_swapped"]) == ("maegatk", True)
    # Each strand's mean quality swaps back with its count, in every row.
    rows = loaded.counts.assign(cell=loaded.counts["ids"].astype(str))
    truth_rows = np.asarray([TRUTH.row_ids.index(cell) for cell in rows["cell"]])
    positions = rows["site"].astype(str).str.split(":").str[1].astype(int).to_numpy()
    bases = np.asarray([BASES.index(allele) for allele in rows["allele"]])
    for strand, quality, count in (
        (0, "quality_forward", "forward"),
        (1, "quality_reverse", "reverse"),
    ):
        expected = TRUTH.quality[truth_rows, positions - 1, bases, strand]
        values = rows[quality].to_numpy()
        read = rows[count].to_numpy() > 0
        np.testing.assert_allclose(values[read], expected[read], atol=1e-4)
        assert np.isnan(values[~read]).all()


def test_old_mgatk_needs_swap_strands_and_keeps_them_otherwise(store, tmp_path) -> None:
    folder = write_mgatk(TRUTH, tmp_path / "final", swapped=True)
    plain, _ = _import(store, folder)
    swapped, _ = _import(store, folder, swap_strands=True)
    _depth, counts = expected_mito(TRUTH)
    assert_tables(store.variants.load_counts(swapped).counts, counts)
    _depth, as_written = expected_mito(TRUTH, swapped=True)
    assert_tables(store.variants.load_counts(plain).counts, as_written)
    assert store.artifacts.inspect(swapped).parameters["swap_strands"] is True
    with pytest.raises(ValueError, match="swap_strands applies to mgatk files"):
        store.variants.import_counts(
            write_mgatk(TRUTH, tmp_path / "maegatk", six_columns=True),
            cell_selection=every_cell(store),
            format="maegatk",
            swap_strands=True,
        )


def test_cell_order_and_spills_to_disk_do_not_change_the_counts(
    template, tmp_path
) -> None:
    # Cells in another order than the store's, and a budget so small that
    # the rows spill to sorted runs on disk, give the same artifact. The
    # budget covers the read of the cells' IDs, which decodes a stored chunk
    # of 100,000 of them, and leaves room for the chunks that writing
    # rewrites; the rows outgrow the sort buffer that the rest leaves.
    roomy = copy_store(template, tmp_path / "roomy")
    tight = copy_store(template, tmp_path / "tight")
    ids = [str(cell) for cell in roomy.cells.fetch_all("ids")]
    folder = tmp_path / "final"
    drawn = _large_mgatk(folder, ids, seed=4, n_positions=4_000)
    tight.memoryBytes = 24_000_000
    groups = []
    for store in (roomy, tight):
        ref, _ = _import(store, folder)
        groups.append(store.artifacts.load(ref))
    assert groups[0].attrs["n_sorted_runs"] == 0
    assert groups[1].attrs["n_sorted_runs"] > 3
    for name in (
        "depth_offsets",
        "depth_sites",
        "depth",
        "count_offsets",
        "count_alleles",
        "counts",
        "counts_forward",
        "counts_reverse",
        "mean_depth",
        "site_depth",
    ):
        np.testing.assert_array_equal(groups[0][name][:], groups[1][name][:])
    # The stored rows are those drawn, with the N's position left out.
    reads = drawn["reads"]
    kept = drawn["reference"] != "N"
    per_site = reads[:, kept].sum(axis=(2, 3))
    np.testing.assert_array_equal(groups[1]["site_depth"][:], per_site.sum(axis=0))
    covered = per_site > 0
    np.testing.assert_array_equal(groups[1]["depth"][:], per_site[covered])
    # Index the strand after the mask: a mask and an integer split by a slice
    # would move the masked axis first.
    forward = reads[:, kept][..., 0].reshape(len(ids), -1)
    reverse = reads[:, kept][..., 1].reshape(len(ids), -1)
    counted = (forward + reverse) > 0
    np.testing.assert_array_equal(groups[1]["counts_forward"][:], forward[counted])
    np.testing.assert_array_equal(groups[1]["counts_reverse"][:], reverse[counted])
    np.testing.assert_array_equal(
        np.diff(groups[1]["count_offsets"][:]), counted.sum(axis=1)
    )


def test_cellsnp_lite_reference_is_dp_less_ad_and_depth_adds_other_bases(
    store, tmp_path
) -> None:
    folder = write_cellsnp(TRUTH, tmp_path / "cellsnp", compress=True)
    with pytest.raises(
        ValueError,
        match="does not say whether cellsnp_lite counted reads, UMIs or molecules",
    ):
        store.variants.import_counts(folder, cell_selection=every_cell(store))
    ref, _ = _import(store, folder, count_unit="umis")
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_variant_sites(TRUTH, other=True)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert (loaded.count_unit, loaded.depth_definition) == ("umis", "all")
    assert set(loaded.alleles["role"]) == {"reference", "alternative", "other"}
    assert store.artifacts.inspect(ref).parameters["count_unit"] == "umis"


def test_cellsnp_lite_without_other_bases_counts_reference_and_alternative(
    store, tmp_path
) -> None:
    ref, _ = _import(
        store, write_cellsnp(TRUTH, tmp_path / "c", other=False), count_unit="reads"
    )
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_variant_sites(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert loaded.depth_definition == "reference_and_alternative"


def test_mquad_reads_its_passed_variants_and_contig_from_bic_params(
    store, tmp_path
) -> None:
    ref, _ = _import(store, write_mquad(TRUTH, tmp_path / "mquad"), count_unit="reads")
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_variant_sites(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert loaded.contigs["name"].tolist() == [CONTIG]
    assert loaded.depth_definition == "reference_and_alternative"


def test_mquad_without_bic_params_has_an_unnamed_contig(store, tmp_path) -> None:
    ref, _ = _import(
        store, write_mquad(TRUTH, tmp_path / "mquad", bic=False), count_unit="reads"
    )
    loaded = store.variants.load_counts(ref)
    assert loaded.contigs["name"].tolist() == [""]
    position = TRUTH.variant_sites()[0][0]
    assert str(position) in loaded.sites.index


def test_vartrix_counts_reference_and_alternative_and_skips_multiallelic_records(
    store, tmp_path
) -> None:
    ref, _ = _import(
        store, write_vartrix(TRUTH, tmp_path / "vartrix"), count_unit="umis"
    )
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_variant_sites(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert loaded.contigs["length"].tolist() == [N_POSITIONS]
    record = store.artifacts.inspect(ref).inputs["sources"][0]
    assert record["n_skipped_records"] == 1


def test_mito_allele_frequency_matrix_keeps_site_depth_without_reference(
    store, tmp_path
) -> None:
    ref, _ = _import(store, write_mito_afm(TRUTH, tmp_path / "afm.h5ad"))
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_variant_sites(TRUTH, reference=False, contig="")
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert loaded.count_unit == "molecules"
    assert loaded.contigs["name"].tolist() == [""]
    record = store.artifacts.inspect(ref).inputs["sources"][0]
    assert (record["scLT_system"], record["pp_method"]) == ("MAESTER", "maegatk")


@pytest.mark.parametrize("layout", ["before 0.3.0", "0.3.0"])
def test_giftwrap_counts_umis_reads_and_support_per_gapfill(
    store, tmp_path, layout
) -> None:
    ref, _ = _import(
        store, write_giftwrap_h5(TRUTH, tmp_path / "counts.1.h5", layout=layout)
    )
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_gift(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    # Support that the file leaves out is the matrix's zero.
    assert (loaded.counts["support"] == 0).any()
    roles = loaded.alleles.assign(site=loaded.alleles["site"].astype(str))
    role_of = {(row.site, row.allele): row.role for row in roles.itertuples()}
    assert role_of[("JAK2 c.1849G>T", "CAC")] == "reference"
    assert role_of[("JAK2 c.1849G>T", "AAC")] == "alternative"
    assert role_of[("JAK2 c.1849G>T", "CAA")] == "other"
    assert role_of[("ASXL1 c.1772dup", "GTAAG")] == "alternative"
    assert role_of[("JAK2 0bp", "")] == "other"
    assert store.artifacts.inspect(ref).inputs["sources"][0]["layout"] == layout


def test_giftwrap_per_umi_tables_give_the_counts_of_the_hdf5_file(
    store, tmp_path
) -> None:
    tables, _ = _import(store, write_giftwrap_tables(TRUTH, tmp_path / "gift"))
    loaded = store.variants.load_counts(tables)
    depth, counts = expected_gift(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    h5, _ = _import(store, write_giftwrap_h5(TRUTH, tmp_path / "counts.1.h5"))
    for name in ("counts", "counts_reads", "depth", "allele_sites", "alleles"):
        np.testing.assert_array_equal(
            store.artifacts.load(tables)[name][:], store.artifacts.load(h5)[name][:]
        )


@pytest.mark.parametrize(
    ("style", "name", "file_format", "ambiguous"),
    [
        ("ironthrone", "CALR.summTable.txt", "ironthrone", True),
        ("xv", "Patient1_CALR_summTable.txt", "got", False),
        ("geo_calls", "GSM1_CALR.txt.gz", "got", False),
        ("geo_total", "GSM2_CALR.tsv", "got", False),
    ],
)
def test_got_tables_count_wild_type_and_mutant_umis(
    store, tmp_path, style, name, file_format, ambiguous
) -> None:
    path = write_got(TRUTH, tmp_path / name, style=style)
    ref, _ = _import(store, path)
    loaded = store.variants.load_counts(ref)
    site = name.split(".summTable")[0].removesuffix("_summTable.txt")
    site = site.split(".")[0] if style != "ironthrone" else "CALR"
    if style == "xv":
        site = "Patient1_CALR"
    depth, counts = expected_got(TRUTH, site=site, ambiguous=ambiguous)
    assert loaded.sites.index.tolist() == [site]
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert (loaded.format, loaded.count_unit) == (file_format, "umis")
    assert loaded.depth_definition == (
        "all" if ambiguous else "reference_and_alternative"
    )
    if style == "geo_total":
        assert_tables(
            loaded.calls,
            expected_got_calls(TRUTH, site=site, source="Genotype"),
        )
        # The NA rows list cells that GoT did not cover.
        assert loaded.cells["listed"].all()
    else:
        assert loaded.calls.empty


def test_a_got_table_of_calls_alone_keeps_the_calls(store, tmp_path) -> None:
    ref, _ = _import(
        store, write_got(TRUTH, tmp_path / "GSM3_CALR.csv", style="geo_call_only")
    )
    loaded = store.variants.load_counts(ref)
    assert loaded.depth.empty and loaded.counts.empty
    assert_tables(
        loaded.calls, expected_got_calls(TRUTH, site="GSM3_CALR", source="Genotype")
    )
    assert loaded.cells["listed"].all()
    assert (loaded.cells["mean_depth"] == 0).all()


def test_got_multi_keeps_each_channel_and_the_target_calls(store, tmp_path) -> None:
    ref, _ = _import(store, write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv"))
    loaded = store.variants.load_counts(ref)
    assert loaded.sites.index.tolist() == [
        "CALR",
        "CALR (geneseq)",
        "CALR (probebc)",
        "SF3B1",
        "SF3B1 (geneseq)",
        "SF3B1 (probebc)",
    ]
    for target, name in enumerate(("CALR", "SF3B1")):
        for channel, extra in (("geneseq", 0), ("probebc", 1)):
            label = f"{name} ({channel})"
            depth, counts = expected_got(
                TRUTH, site=label, target=target, ambiguous=True, extra_wild_type=extra
            )
            assert_tables(loaded.depth[loaded.depth["site"] == label], depth)
            assert_tables(
                loaded.counts[loaded.counts["site"] == label][
                    ["ids", "site", "allele", "count"]
                ],
                counts,
            )
        calls = loaded.calls[loaded.calls["site"] == name]
        for source in ("genotype_geneseq", "genotype_probebc", "genotype_merged"):
            assert_tables(
                calls[calls["source"] == source],
                expected_got_calls(TRUTH, site=name, target=target, source=source),
            )
    # Each cell's stored calls follow the allele, then the source's position
    # among the sorted source names, whatever order the file's columns have.
    group = store.artifacts.load(ref)
    names = group["call_source_names"][:].astype(str).tolist()
    assert names == ["genotype_geneseq", "genotype_merged", "genotype_probebc"]
    offsets = group["call_offsets"][:]
    rows = np.stack(
        [
            np.repeat(np.arange(len(offsets) - 1), np.diff(offsets)),
            group["call_alleles"][:],
            group["call_sources"][:],
        ]
    )
    assert len(np.unique(rows[2])) == 3
    np.testing.assert_array_equal(np.lexsort(rows[::-1]), np.arange(rows.shape[1]))
    # Reads come with the counts: each call's reads, the probe barcode
    # channel's WT reads one more.
    reads = loaded.counts.assign(cell=loaded.counts["ids"].astype(str))
    row = reads[(reads["site"] == "CALR (probebc)") & (reads["allele"] == "WT")].iloc[0]
    truth_row = TRUTH.row_ids.index(row["cell"])
    assert row["reads"] == TRUTH.got_reads[truth_row, 0, 0] + 1


def test_got_multi_stage_tables_import_together(store, tmp_path) -> None:
    stage1 = write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv")
    refined = write_got_multi(
        TRUTH, tmp_path / "got_multi_out_refined.csv", refined=True
    )
    ref, _ = _import(store, [stage1, refined])
    calls = store.variants.load_counts(ref).calls
    assert set(calls["source"]) == {
        "genotype_geneseq",
        "genotype_probebc",
        "genotype_merged",
        "genotype_final",
    }
    final = calls[calls["source"] == "genotype_final"]
    assert "FalsePositive" in set(final["call"])


@pytest.mark.parametrize("style", ["labeling", "genes"])
def test_gotcha_counts_are_reads(store, tmp_path, style) -> None:
    name = "TP53_gotcha.csv" if style == "labeling" else "metadata_cells.csv"
    ref, _ = _import(store, write_gotcha(TRUTH, tmp_path / name, style=style))
    loaded = store.variants.load_counts(ref)
    assert loaded.count_unit == "reads"
    if style == "labeling":
        depth, counts = expected_got(TRUTH, site="TP53_gotcha")
        assert_tables(
            loaded.calls,
            expected_got_calls(TRUTH, site="TP53_gotcha", source="genotype_pred"),
        )
    else:
        depth, counts = expected_got(TRUTH, site="CALR")
        depth2, counts2 = expected_got(TRUTH, site="SF3B1", target=1)
        depth = pd.concat([depth, depth2])
        counts = pd.concat([counts, counts2])
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)


def test_several_libraries_join_with_their_own_prefixes(template, tmp_path) -> None:
    # Each library's files name cells by bare barcode; the prefix names them.
    store = copy_store(template, tmp_path)
    ids = TRUTH.row_ids
    samples = [identifier.split("__")[0] for identifier in ids]
    folders = []
    for sample in ("d2", "d4"):
        rows = [row for row in range(TRUTH.n_cells) if samples[row] == sample]
        folder = write_mgatk(TRUTH, tmp_path / sample, rows=rows)
        for path in folder.glob("*.gz"):
            text = gzip.decompress(path.read_bytes()).decode()
            path.write_bytes(
                gzip.compress(text.replace(f"{sample}__", "").encode(), mtime=0)
            )
        depth_table = folder / "sample.depthTable.txt"
        depth_table.write_text(depth_table.read_text().replace(f"{sample}__", ""))
        folders.append(folder)
    with collect_warnings() as messages:
        ref = store.variants.import_counts(
            folders, cell_selection=every_cell(store), barcode_prefix=["d2__", "d4__"]
        )
    assert list(messages) == []
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_mito(TRUTH)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    assert len(store.artifacts.inspect(ref).inputs["sources"]) == 2


def test_a_cell_counted_by_two_files_is_refused(store, tmp_path) -> None:
    first = write_mgatk(TRUTH, tmp_path / "a")
    second = tmp_path / "b"
    shutil.copytree(first, second)
    (second / "sample.depthTable.txt").write_text("extra\t0\n")
    with pytest.raises(ValueError, match="is counted twice for the allele"):
        store.variants.import_counts([first, second], cell_selection=every_cell(store))


def test_counts_on_another_reference_are_refused(store, tmp_path) -> None:
    first = write_mgatk(TRUTH, tmp_path / "a", rows=range(10))
    second = write_mgatk(TRUTH, tmp_path / "b", rows=range(10, 20))
    reference = second / f"{CONTIG}_refAllele.txt"
    reference.write_text(reference.read_text() + f"{N_POSITIONS + 1}\tA\n")
    with pytest.raises(ValueError, match="with different references"):
        store.variants.import_counts([first, second], cell_selection=every_cell(store))
    # Base files beyond their own reference are refused too.
    shorter = write_mgatk(TRUTH, tmp_path / "c")
    lines = (shorter / f"{CONTIG}_refAllele.txt").read_text().splitlines()[:-3]
    (shorter / f"{CONTIG}_refAllele.txt").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="outside the .* positions of its reference"):
        store.variants.import_counts(shorter, cell_selection=every_cell(store))


def test_formats_of_one_import_must_agree(store, tmp_path) -> None:
    with pytest.raises(ValueError, match="several formats, got, mgatk"):
        store.variants.import_counts(
            [
                write_mgatk(TRUTH, tmp_path / "final"),
                write_got(TRUTH, tmp_path / "CALR.tsv", style="xv"),
            ],
            cell_selection=every_cell(store),
        )


def test_a_reused_import_warns_again_and_a_changed_file_imports_again(
    store, tmp_path
) -> None:
    folder = write_mgatk(TRUTH, tmp_path / "final")
    first, messages = _import(store, folder)
    again, reused = _import(store, folder)
    assert again == first and reused == messages
    moved = tmp_path / "moved"
    shutil.copytree(folder, moved)
    assert _import(store, moved)[0] == first
    (moved / "sample.depthTable.txt").write_text(
        (moved / "sample.depthTable.txt").read_text() + "extra-1\t0.00\n"
    )
    assert _import(store, moved)[0] != first
    assert _import(store, folder, invalidate_cache=True)[0] != first


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"format": "vcf"}, ValueError, "format must be one of"),
        ({"count_unit": "cells"}, ValueError, "count_unit must be one of"),
        ({"count_unit": "umis"}, ValueError, "mgatk counts reads, not umis"),
        ({"swap_strands": 1}, TypeError, "swap_strands must be a boolean"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
        (
            {"cell_selection": "cells"},
            TypeError,
            "cell_selection must be an ArtifactRef",
        ),
    ],
)
def test_import_refuses_arguments_it_cannot_use(
    store, tmp_path, options, error, message
) -> None:
    folder = write_mgatk(TRUTH, tmp_path / "final")
    call = {"cell_selection": every_cell(store), **options}
    with pytest.raises(error, match=message):
        store.variants.import_counts(folder, **call)


def test_missing_paths_and_files_are_named(store, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        store.variants.import_counts(
            tmp_path / "absent", cell_selection=every_cell(store)
        )
    folder = write_mgatk(TRUTH, tmp_path / "final")
    (folder / f"{CONTIG}_refAllele.txt").unlink()
    with pytest.raises(FileNotFoundError, match="holds no reference file"):
        store.variants.import_counts(folder, cell_selection=every_cell(store))


def test_a_read_only_store_reuses_saved_counts_and_refuses_new_ones(
    store, tmp_path
) -> None:
    from tests.storage_helpers import open_store

    folder = write_mgatk(TRUTH, tmp_path / "final")
    ref, messages = _import(store, folder)
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    again, reused = _import(reader, folder)
    assert again == ref and reused == messages
    with pytest.raises(PermissionError):
        reader.variants.import_counts(
            write_got(TRUTH, tmp_path / "CALR.tsv", style="xv"),
            cell_selection=every_cell(reader),
        )


def test_a_memory_budget_too_small_for_the_cells_is_refused(store, tmp_path) -> None:
    store.memoryBytes = 2_000
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.variants.import_counts(
            write_mgatk(TRUTH, tmp_path / "final"), cell_selection=every_cell(store)
        )


def test_the_import_counts_the_read_of_cell_ids_before_it_reads_them(
    store, tmp_path, monkeypatch
) -> None:
    import cytearc.datastore._operations.variants as operations

    path = write_got(TRUTH, tmp_path / "CALR.tsv", style="xv")

    def run() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return store.variants.import_counts(path, cell_selection=every_cell(store))

    reading = _refused_estimate(store, run, 1)
    # A read of IDs decodes a whole stored chunk of them, however few cells
    # the selection holds.
    ids = store.cells._get_array("ids")
    assert reading > ids.chunks[0] * ids.dtype.itemsize

    def unread(*_arguments, **_options):
        raise AssertionError("the cell IDs were read")

    with monkeypatch.context() as patched:
        patched.setattr(operations, "read_metadata_rows_chunkwise", unread)
        store.memoryBytes = reading - 1
        with pytest.raises(MemoryError, match=f"holds about {reading} bytes"):
            run()
    store.memoryBytes = reading
    ref = run()
    store.memoryBytes = 1 << 30
    assert store.variants.load_counts(ref, sites=[]).cells.shape[0] == len(
        TRUTH.clones.cell_ids
    )


def test_the_parse_shares_what_the_join_leaves(template, tmp_path) -> None:
    # A store whose IDs are one chunk of their own size, as columns written
    # from AnnData files are stored, reads them cheaply, so the least parse,
    # 1,000 rows, sets the import's smallest budget beside what the join
    # holds: its lookup, the IDs as text and the bytes held for each cell.
    import sys

    from cytearc.datastore._operations import variants as operations
    from cytearc.datastore._operations.cell_barcodes import LOOKUP_BYTES_PER_CELL
    from cytearc.storage.arrays import create_metadata_column
    from tests.storage_helpers import open_store

    path = tmp_path / "store" / template.name
    shutil.copytree(template, path)
    ids = np.asarray(open_store(path).cells.fetch_all("ids"))
    create_metadata_column(
        open_store(path).cells._group,
        "ids",
        data=ids,
        dtype=ids.dtype,
        chunkSize=len(ids),
        overwrite=True,
    )
    store = open_store(path)
    assert store.cells._get_array("ids").chunks == (len(ids),)
    got = write_got(TRUTH, tmp_path / "CALR.tsv", style="xv")

    def run() -> object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return store.variants.import_counts(got, cell_selection=every_cell(store))

    reading = _refused_estimate(store, run, 1)
    held = _refused_estimate(store, run, reading)
    assert held == (LOOKUP_BYTES_PER_CELL + operations._IMPORT_BYTES_PER_CELL) * len(
        ids
    ) + sum(map(sys.getsizeof, ids.astype(object).tolist()))
    store.memoryBytes = held + (
        operations._PARSE_SHARE
        * operations._PARSE_BYTES_PER_ROW
        * operations._MIN_CHUNK_ROWS
    )
    assert store.variants.load_counts(run(), sites=[]).cells["listed"].any()


def test_load_counts_counts_the_ids_it_reads_and_finds_cells_among(
    store, tmp_path, monkeypatch
) -> None:
    import cytearc.datastore._operations.variants as operations

    ref, _ = _import(store, write_got(TRUTH, tmp_path / "CALR.tsv", style="xv"))
    ids = store.cells._get_array("ids")
    chunk = ids.chunks[0] * ids.dtype.itemsize
    named = [str(TRUTH.clones.cell_ids[1])]
    # Loading every cell reads their IDs, a stored chunk at a time, and
    # finding cells by ID reads the IDs of every cell of the counts.
    loads = {
        "every cell": lambda: store.variants.load_counts(ref),
        "cells by ID": lambda: store.variants.load_counts(ref, cell_selection=named),
    }

    def unread(*_arguments, **_options):
        raise AssertionError("the cell IDs were read")

    for name, load in loads.items():
        needed = _refused_estimate(store, load, 1)
        assert needed > chunk, name
        with monkeypatch.context() as patched:
            patched.setattr(operations, "read_metadata_rows_chunkwise", unread)
            store.memoryBytes = needed - 1
            with pytest.raises(MemoryError, match=f"holds about {needed} bytes"):
                load()
    # Without a cell, no ID is read, so a budget far below a chunk of IDs
    # loads the sites.
    store.memoryBytes = chunk // 10
    assert store.variants.load_counts(ref, cell_selection=[]).cells.empty


def test_load_counts_names_counts_of_another_kind_missing_or_incomplete(
    store, tmp_path
) -> None:
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.refs import ArtifactRef

    ref, _ = _import(store, write_got(TRUTH, tmp_path / "CALR.tsv", style="xv"))
    with pytest.raises(ValueError, match="counts must be a variant_counts artifact"):
        store.variants.load_counts(every_cell(store))
    missing = ArtifactRef(scope=ref.scope, kind=ref.kind, artifact_id="f" * 64)
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store.variants.load_counts(missing)
    # An import that stopped before it finished leaves incomplete counts.
    attributes = artifact_group(store.zw, ref).attrs
    attributes["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store.variants.load_counts(ref)
    attributes["complete"] = True
    assert len(store.variants.load_counts(ref).cells) == len(TRUTH.clones.cell_ids)


def test_select_cells_and_load_values_read_the_mean_depth(store, tmp_path) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    values = store.artifacts.load_values(ref)
    mean_depth = store.variants.load_counts(ref).cells["mean_depth"].to_numpy()
    np.testing.assert_allclose(values.values, mean_depth)
    # A bound that one cell's mean depth equals keeps the cells above it only.
    bound = float(np.sort(mean_depth)[len(mean_depth) // 2])
    deep = store.select_cells(ref, low=bound)
    chosen = store.artifacts.load(deep)["values"][:]
    assert chosen.sum() == int((mean_depth > bound).sum())
    assert not chosen[mean_depth == bound].any()
    listed = store.artifacts.load_values(ref, value="listed")
    assert listed.values.all()


def test_load_counts_keeps_the_sites_and_cells_it_is_given(store, tmp_path) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    full = store.variants.load_counts(ref)
    chosen_ids = [str(cell) for cell in TRUTH.clones.cell_ids[[3, 1, 40]]]
    positions = [SITE_POSITIONS[4], SITE_POSITIONS[0]]
    by_position = store.variants.load_counts(
        ref, sites=positions, cell_selection=chosen_ids
    )
    by_label = store.variants.load_counts(
        ref,
        sites=[mito_label(position) for position in positions],
        cell_selection=chosen_ids,
    )
    labels = sorted(
        {mito_label(position) for position in positions},
        key=lambda t: int(t.split(":")[1]),
    )
    for loaded in (by_position, by_label):
        assert loaded.sites.index.tolist() == labels
        # Cells come back in the artifact's order, whatever order they are named in.
        assert loaded.cells.index.tolist() == [
            str(cell) for cell in TRUTH.clones.cell_ids[[1, 3, 40]]
        ]
        for name in ("depth", "counts"):
            frame = getattr(full, name)
            expected = frame[
                frame["site"].astype(str).isin(labels)
                & frame["ids"].astype(str).isin(chosen_ids)
            ]
            assert_tables(getattr(loaded, name), expected)
        assert set(loaded.alleles["site"].astype(str)) == set(labels)
    # A site or a cell named twice is loaded once.
    twice = store.variants.load_counts(
        ref,
        sites=[*positions, positions[0]],
        cell_selection=[*chosen_ids, chosen_ids[1]],
    )
    assert twice.sites.index.tolist() == labels
    assert twice.cells.index.tolist() == by_position.cells.index.tolist()
    for name in ("sites", "cells", "depth", "counts"):
        assert_tables(getattr(twice, name), getattr(by_position, name))
    # A cell selection within the counts' cells selects the same way.
    selection = store.select_cells(ref, low=0.0)
    selected = store.variants.load_counts(
        ref, cell_selection=selection, sites=positions
    )
    # A bound leaves out values equal to it: the cell without reads.
    covered = full.cells.index[full.cells["mean_depth"] > 0].tolist()
    assert selected.cells.index.tolist() == covered
    # Without cells, only the sites and their sums; without sites, only cells.
    sites_only = store.variants.load_counts(ref, cell_selection=[])
    assert sites_only.depth.empty and sites_only.counts.empty and sites_only.cells.empty
    assert sites_only.sites["depth"].tolist() == full.sites["depth"].tolist()
    cells_only = store.variants.load_counts(ref, sites=[])
    assert cells_only.depth.empty and len(cells_only.cells) == TRUTH.n_cells


def test_load_counts_refuses_sites_and_cells_it_does_not_hold(store, tmp_path) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    for sites, error, message in (
        (["chrM:999"], ValueError, "has no site 'chrM:999'"),
        ([999], ValueError, "has no site 999"),
        ("chrM:1", TypeError, "sites must be a sequence of site labels"),
        ([1.5], TypeError, "sites must hold site labels or positions, not float"),
    ):
        with pytest.raises(error, match=message):
            store.variants.load_counts(ref, sites=sites)
    with pytest.raises(ValueError, match="The counts hold no cell 'nobody'"):
        store.variants.load_counts(ref, cell_selection=["nobody"])
    with pytest.raises(
        TypeError, match="^cell_selection must be a cell selection ArtifactRef"
    ):
        store.variants.load_counts(ref, cell_selection="d2__AAAC-1")
    store.cells.insert("first_half", np.arange(TRUTH.n_cells) < TRUTH.n_cells // 2)
    half = store.snapshot_cell_selection("first_half")
    with collect_warnings(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        half_counts = store.variants.import_counts(
            write_got(TRUTH, tmp_path / "CALR.tsv", style="xv"), cell_selection=half
        )
    with pytest.raises(
        ValueError, match="^cell_selection selects .* cells that the counts do not hold"
    ):
        store.variants.load_counts(half_counts, cell_selection=every_cell(store))
    with pytest.raises(TypeError, match="counts must be an ArtifactRef"):
        store.variants.load_counts("counts")
    with pytest.raises(
        ValueError, match="must be a variant_counts artifact, not 'cell_selection'"
    ):
        store.variants.load_counts(half)
    # Positions name sites only on an artifact of one contig.
    targets, _ = _import(
        store, write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv")
    )
    with pytest.raises(
        ValueError,
        match="Sites are named by position only on an artifact of one contig",
    ):
        store.variants.load_counts(targets, sites=[1])
    store.memoryBytes = 5_000
    with pytest.raises(MemoryError, match="Raise mem_budget, or pass fewer cells"):
        store.variants.load_counts(ref)


def test_import_refuses_files_without_rows_values_out_of_range_and_empty_selections(
    store, tmp_path
) -> None:
    empty = tmp_path / "CALR.tsv"
    empty.write_text("BC\twtUMIs\tmutUMIs\n")
    with pytest.raises(
        ValueError,
        match="No row of counts, depth, calls or cells is in the variant file",
    ):
        store.variants.import_counts(empty, cell_selection=every_cell(store))
    huge = tmp_path / "SF3B1.tsv"
    cell = str(TRUTH.clones.cell_ids[0])
    huge.write_text(f"BC\twtUMIs\tmutUMIs\n{cell}\t{2**33}\t1\n")
    with pytest.raises(ValueError, match="A counts value lies outside 0 to"):
        store.variants.import_counts(huge, cell_selection=every_cell(store))
    reads = tmp_path / "ironthrone_out_pp.csv"
    text = write_got_multi(TRUTH, reads).read_text().splitlines()
    fields = text[1].split(",")
    fields[2] = str(2**33)
    reads.write_text("\n".join([text[0], ",".join(fields), *text[2:]]) + "\n")
    with pytest.raises(ValueError, match="A value of reads lies outside 0 to"):
        store.variants.import_counts(reads, cell_selection=every_cell(store))
    with pytest.raises(ValueError, match="swap_strands applies to mgatk files"):
        store.variants.import_counts(
            write_got(TRUTH, tmp_path / "x.tsv", style="xv"),
            cell_selection=every_cell(store),
            swap_strands=True,
        )
    store.cells.insert("nothing", np.zeros(TRUTH.n_cells, dtype=bool))
    with pytest.raises(ValueError, match="selects no cell"):
        store.variants.import_counts(
            write_got(TRUTH, tmp_path / "y.tsv", style="xv"),
            cell_selection=store.snapshot_cell_selection("nothing"),
        )


def test_two_files_that_give_one_cell_two_depths_or_calls_are_refused(
    store, tmp_path
) -> None:
    import h5py

    first = write_mito_afm(TRUTH, tmp_path / "a.h5ad")
    second = tmp_path / "b.h5ad"
    shutil.copy(first, second)
    with h5py.File(second, "a") as handle:
        indptr = handle["layers/AD/indptr"][:]
        del handle["layers/AD"]["data"], handle["layers/AD"]["indices"]
        handle["layers/AD"].create_dataset("data", data=np.zeros(0, dtype=np.int16))
        handle["layers/AD"].create_dataset("indices", data=np.zeros(0, dtype=np.int32))
        handle["layers/AD/indptr"][:] = np.zeros_like(indptr)
    with pytest.raises(ValueError, match="has two depths at the site"):
        store.variants.import_counts([first, second], cell_selection=every_cell(store))
    calls = [tmp_path / name / "GSM3_CALR.csv" for name in ("one", "two")]
    for path in calls:
        path.parent.mkdir()
        write_got(TRUTH, path, style="geo_call_only")
    # The same calls twice are kept once; a call that differs is refused.
    calls[1].write_text(calls[1].read_text() + "\n")
    ref = store.variants.import_counts(calls, cell_selection=every_cell(store))
    assert_tables(
        store.variants.load_counts(ref).calls,
        expected_got_calls(TRUTH, site="GSM3_CALR", source="Genotype"),
    )
    called = sum(TRUTH.got_call(row, 0) != "NA" for row in range(TRUTH.n_cells))
    assert store.artifacts.load(ref).attrs["n_repeated_calls"] == called
    flipped = (
        calls[1]
        .read_text()
        .replace('"MUT"', '"tmp"')
        .replace('"WT"', '"MUT"')
        .replace('"tmp"', '"WT"')
    )
    calls[1].write_text(flipped)
    with pytest.raises(
        ValueError, match="has two different calls of Genotype for the allele 'MUT'"
    ):
        store.variants.import_counts(calls, cell_selection=every_cell(store))


def test_an_artifact_that_lost_a_table_is_not_reused(store, tmp_path) -> None:
    import zarr

    from cytearc.storage.artifacts import artifact_path

    folder = write_mgatk(TRUTH, tmp_path / "final")
    first, _ = _import(store, folder)

    def writable(ref) -> zarr.Group:
        return zarr.open_group(store.zarr_loc, mode="r+")[artifact_path(ref)]

    del writable(first)["call_values"]
    second, _ = _import(store, folder)
    assert second != first
    writable(second).attrs["count_fields"] = "forward"
    third, _ = _import(store, folder)
    assert third not in (first, second)
    del writable(third)["counts_forward"]
    fourth, _ = _import(store, folder)
    assert fourth not in (first, second, third)
    # mgatk's calls carry no quality, so its counts store none; counts that
    # list a call quality they lost, or that list no call fields, as an
    # earlier version saved them, are imported again.
    assert store.artifacts.load(fourth).attrs["call_fields"] == []
    assert "call_quality" not in store.artifacts.load(fourth)
    writable(fourth).attrs["call_fields"] = ["quality"]
    fifth, _ = _import(store, folder)
    assert fifth not in (first, second, third, fourth)
    del writable(fifth).attrs["call_fields"]
    assert _import(store, folder)[0] not in (first, second, third, fourth, fifth)


def test_repeated_calls_must_agree_in_their_quality() -> None:
    from cytearc.datastore._operations.variants import _distinct_calls
    from cytearc.variants.counts import CALL_DTYPE

    rows = np.zeros(3, dtype=CALL_DTYPE)
    rows["value"] = 1
    rows["quality"] = [np.nan, np.nan, 30.0]
    rows["key"] = [0, 0, 1]
    # Two repeats without a quality are one call; NaN never equals NaN.
    kept, conflict, dropped = _distinct_calls(rows)
    assert conflict is None and dropped == 1 and len(kept) == 2
    rows["key"] = 0
    kept, conflict, dropped = _distinct_calls(rows)
    assert conflict == 2 and dropped == 0
    rows["quality"] = [20.0, 20.0, 99.0]
    rows["key"] = [0, 0, 0]
    assert _distinct_calls(rows)[1] == 2


def test_load_counts_refuses_rows_beyond_the_memory_budget(store, tmp_path) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    # The cells, their IDs among them, fit; their rows do not.
    cells = _refused_estimate(store, lambda: store.variants.load_counts(ref), 1)
    store.memoryBytes = cells + 60_000
    with pytest.raises(MemoryError, match="Pass fewer sites or cells"):
        store.variants.load_counts(ref)
    assert len(store.variants.load_counts(ref, sites=[SITE_POSITIONS[0]]).sites) == 1


def test_threads_parse_the_base_files_into_the_same_artifact(
    template, tmp_path
) -> None:
    from tests.storage_helpers import open_store

    folder = write_mgatk(TRUTH, tmp_path / "final")
    groups, warned = [], []
    for threads in (1, 4):
        target = tmp_path / f"threads{threads}"
        shutil.copytree(template, target / template.name)
        store = open_store(target / template.name, nthreads=threads)
        ref, messages = _import(store, folder)
        groups.append(store.artifacts.load(ref))
        warned.append(messages)
    assert warned[0] == warned[1]
    assert (
        groups[0].attrs["unmatched_barcode_examples"]
        == groups[1].attrs["unmatched_barcode_examples"]
    )
    for name in ("depth_offsets", "depth", "count_alleles", "counts", "counts_forward"):
        np.testing.assert_array_equal(groups[0][name][:], groups[1][name][:])
    # A malformed file stops the threads with its error.
    (folder / "sample.G.txt.gz").write_bytes(gzip.compress(b"1,x,1,1\n2,x,y,1\n"))
    store = open_store(tmp_path / "threads4" / template.name, nthreads=4)
    with pytest.raises(ValueError, match="holds a row that is not 4 fields"):
        store.variants.import_counts(folder, cell_selection=every_cell(store))


def _pad_until_sorted(
    padded: Path, digest, *, before: str | None = None, after: str | None = None
) -> None:
    """Append blank lines, which the readers skip, to a file of a source until
    the source's SHA-256 sorts before or after another's, so an import takes
    its sources in the canonical order a test wants."""
    for _ in range(500):
        value = digest()
        if (before is None or value < before) and (after is None or value > after):
            return
        padded.write_text(padded.read_text() + "\n")
    raise AssertionError("no padding gave the order wanted")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_got_multi(truth) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GoT-Multi's UMIs and reads per channel, and the calls of its stage-1
    and refined tables, as ``write_got_multi`` writes them."""
    from tests.variant_truth import TARGETS

    ids = truth.row_ids
    counts, calls = [], []
    for row in range(truth.n_cells):
        for target, name in enumerate(TARGETS):
            umis = truth.got[row, target].tolist()
            if umis[0] < 0:
                continue
            reads = truth.got_reads[row, target].tolist()
            for channel, extra in (("geneseq", 0), ("probebc", 1)):
                for index, allele in enumerate(("WT", "MUT", "ambiguous")):
                    count = umis[index] + (extra if index == 0 else 0)
                    if count:
                        counts.append(
                            (
                                ids[row],
                                f"{name} ({channel})",
                                allele,
                                count,
                                reads[index] + (extra if index == 0 else 0),
                            )
                        )
            call = truth.got_call(row, target)
            final = "FalsePositive" if call == "MUT" and row % 4 == 0 else call
            for source, value in (
                ("genotype_geneseq", call),
                ("genotype_probebc", call),
                ("genotype_merged", call),
                ("genotype_final", final),
            ):
                calls.append((ids[row], name, "MUT", source, value))
    return (
        pd.DataFrame(counts, columns=["ids", "site", "allele", "count", "reads"]),
        pd.DataFrame(calls, columns=["ids", "site", "allele", "source", "call"]),
    )


@pytest.mark.parametrize("refined_first", [True, False])
def test_got_multi_stage_tables_keep_reads_and_calls_in_either_order(
    store, tmp_path, refined_first
) -> None:
    # The refined table holds calls alone, so it must not decide which
    # fields the counts keep, whichever source sorts first.
    stage1 = write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv")
    (tmp_path / "refined").mkdir()
    refined = write_got_multi(
        TRUTH, tmp_path / "refined" / "got_multi_out_refined.csv", refined=True
    )
    side = "before" if refined_first else "after"
    _pad_until_sorted(refined, lambda: _sha256(refined), **{side: _sha256(stage1)})
    ref, _ = _import(store, [stage1, refined])
    sources = store.artifacts.inspect(ref).inputs["sources"]
    assert [source["files"]["table"] for source in sources] == (
        [_sha256(refined), _sha256(stage1)]
        if refined_first
        else [_sha256(stage1), _sha256(refined)]
    )
    group = store.artifacts.load(ref)
    assert group.attrs["count_fields"] == ["reads"]
    loaded = store.variants.load_counts(ref)
    counts, calls = _expected_got_multi(TRUTH)
    assert_tables(loaded.counts, counts)
    # Calls that first appear in the later table, such as FalsePositive,
    # keep their text whatever order the tables come in.
    assert_tables(loaded.calls, calls)
    # The refined table's genotype_final is read, not its genotype_merged.
    assert group.attrs["n_repeated_calls"] == 0


@pytest.mark.parametrize("six_first", [True, False])
def test_folders_whose_counts_carry_different_fields_are_refused(
    store, tmp_path, six_first
) -> None:
    from cytearc.readers._variant_files import source_sha256
    from cytearc.readers.mgatk import mgatk_source

    def digest(folder: Path) -> str:
        reader = mgatk_source(folder, format="mgatk", swap_strands=False)
        return source_sha256(reader.files)[0]

    half = TRUTH.n_cells // 2
    four = write_mgatk(TRUTH, tmp_path / "four", rows=range(half))
    six = write_mgatk(
        TRUTH, tmp_path / "six", rows=range(half, TRUTH.n_cells), six_columns=True
    )
    side = "before" if six_first else "after"
    _pad_until_sorted(
        six / "sample.depthTable.txt", lambda: digest(six), **{side: digest(four)}
    )
    with pytest.raises(
        ValueError,
        match=r"give their counts different fields: .*six gives quality_forward and "
        r".*four does not",
    ):
        store.variants.import_counts(
            [four, six], cell_selection=every_cell(store), format="mgatk"
        )


def test_cellsnp_folders_with_and_without_other_bases_are_refused(
    store, tmp_path
) -> None:
    folders = [
        write_cellsnp(TRUTH, tmp_path / "with_other"),
        write_cellsnp(TRUTH, tmp_path / "without_other", other=False),
    ]
    for order in (folders, folders[::-1]):
        with pytest.raises(
            ValueError,
            match="define depth differently.*cellsnp-lite folders with and without OTH",
        ):
            store.variants.import_counts(
                order, cell_selection=every_cell(store), count_unit="umis"
            )


def test_every_source_is_checked_against_the_unit_and_the_others() -> None:
    from types import SimpleNamespace

    from cytearc.datastore._operations.variants import _VariantOperationsMixin

    def source(path: str, unit: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            reader=SimpleNamespace(
                path=path,
                format="mito",
                count_unit=unit,
                depth_definition="all",
                depth_roles=None,
                reference_from_depth=False,
                alleles_overlap=False,
                open_alleles=False,
                count_fields=(),
                call_fields=(),
            )
        )

    rules = _VariantOperationsMixin._variant_import_rules
    # A stated unit is checked in every source, not only in the first.
    with pytest.raises(ValueError, match="mito counts molecules, not reads"):
        rules([source("a.h5ad", None), source("b.h5ad", "molecules")], "reads")
    with pytest.raises(
        ValueError,
        match="count in different units: a.h5ad counts molecules and b.h5ad reads",
    ):
        rules([source("a.h5ad", "molecules"), source("b.h5ad", "reads")], None)
    assert rules([source("a.h5ad", None)], "umis").count_unit == "umis"


def test_one_genome_under_two_contig_names_is_refused(store, tmp_path) -> None:
    # Deposits prefix the reference file, which names mgatk's contig, so
    # each library would count every position at a site of its own.
    half = TRUTH.n_cells // 2
    folders = []
    for sample, rows in (("d2", range(half)), ("d4", range(half, TRUTH.n_cells))):
        folder = write_mgatk(TRUTH, tmp_path / sample, rows=rows)
        (folder / f"{CONTIG}_refAllele.txt").rename(
            folder / f"{sample}_{CONTIG}_refAllele.txt"
        )
        folders.append(folder)
    with pytest.raises(
        ValueError,
        match="one reference sequence under the contig names 'd2_chrM' and 'd4_chrM'",
    ):
        store.variants.import_counts(folders, cell_selection=every_cell(store))


@pytest.mark.parametrize(
    ("options", "unit"),
    [
        ({}, "molecules"),
        ({"system": "mtscATAC", "pipeline": "mgatk", "molecules": True}, "molecules"),
        ({"system": "RedeeM", "pipeline": "RedeemV"}, "molecules"),
        ({"system": "mtscATAC", "pipeline": "mgatk"}, None),
        ({"system": None, "pipeline": None}, None),
    ],
)
def test_a_mito_matrix_states_its_unit_or_needs_count_unit(
    store, tmp_path, options, unit
) -> None:
    path = write_mito_afm(TRUTH, tmp_path / "afm.h5ad", **options)
    if unit is None:
        # mgatk reads that MiTo left as reads, or a matrix without uns.
        with pytest.raises(ValueError, match="does not say whether mito counted"):
            store.variants.import_counts(path, cell_selection=every_cell(store))
        ref, _ = _import(store, path, count_unit="reads")
        assert store.variants.load_counts(ref, cell_selection=[]).count_unit == "reads"
        return
    with pytest.raises(ValueError, match="mito counts molecules, not reads"):
        store.variants.import_counts(
            path, cell_selection=every_cell(store), count_unit="reads"
        )
    ref, _ = _import(store, path)
    assert store.variants.load_counts(ref, cell_selection=[]).count_unit == unit


def test_short_tables_are_one_chunk_no_longer_than_their_rows(store, tmp_path) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    group = store.artifacts.load(ref)
    _depth, counts = expected_mito(TRUTH)
    # mgatk gives count rows only and a band derives depth from them, so no
    # array of depth or counts needs more rows than the counts given.
    for name in (
        "depth_sites",
        "depth",
        "count_alleles",
        "counts",
        "counts_forward",
        "counts_reverse",
    ):
        assert group[name].shape[0] <= group[name].chunks[0] <= len(counts)
    calls_ref, _ = _import(
        store, write_got(TRUTH, tmp_path / "CALR.tsv", style="geo_total")
    )
    calls = store.artifacts.load(calls_ref)
    n_calls = len(expected_got_calls(TRUTH, site="CALR", source="Genotype"))
    for name in ("call_alleles", "call_sources", "call_values"):
        assert calls[name].shape[0] == calls[name].chunks[0] == n_calls
    # GoT tables give no quality of their calls, so none is stored.
    assert "call_quality" not in calls
    assert calls.attrs["call_fields"] == []


def test_long_tables_are_chunked_in_bands_of_a_million_rows(tmp_path) -> None:
    import zarr

    from cytearc.datastore._operations.variants import _Appender

    group = zarr.open_group(str(tmp_path / "g.zarr"), mode="w", use_consolidated=False)
    appender = _Appender(
        group,
        {
            "long": (np.uint32, 3_000_000),
            "short": (np.int32, 10),
            "empty": (np.int32, 0),
        },
    )
    assert [appender.arrays[name].chunks for name in ("long", "short", "empty")] == [
        (1 << 20,),
        (10,),
        (1,),
    ]
    appender.append("short", np.arange(10, dtype=np.int32))
    np.testing.assert_array_equal(group["short"][:], np.arange(10))


def test_load_counts_counts_the_chunks_a_read_decompresses(store, tmp_path) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    group = store.artifacts.load(ref)
    # Every long array holds 4-byte values in one chunk here, which a read of
    # even one cell decompresses whole.
    chunk_bytes = 4 * max(
        group[name].chunks[0]
        for name in ("depth", "counts", "counts_forward", "call_values")
    )
    one_cell = [str(TRUTH.clones.cell_ids[0])]
    one_site = [SITE_POSITIONS[0]]
    # The one cell is read by a selection, and its ID as text, which decodes
    # a stored chunk of IDs, beside the chunks of the long arrays.
    store.cells.insert(
        "is_first", np.arange(len(store.cells.fetch_all("ids"))) == 0, overwrite=True
    )
    first = store.snapshot_cell_selection("is_first")
    ids_bytes = text_read_bytes(store.cells._get_array("ids"), 1)
    store.memoryBytes = ids_bytes + 2 * chunk_bytes
    with pytest.raises(MemoryError, match="before it keeps a row"):
        store.variants.load_counts(ref, sites=one_site, cell_selection=first)
    store.memoryBytes = ids_bytes + 2 * chunk_bytes + 10_000
    loaded = store.variants.load_counts(ref, sites=one_site, cell_selection=first)
    assert loaded.cells.index.tolist() == one_cell
    # Without rows to read, no chunk is decompressed.
    store.memoryBytes = 2 * chunk_bytes
    assert store.variants.load_counts(ref, cell_selection=[]).cells.empty


def test_a_small_budget_parses_the_base_files_on_one_thread(
    template, tmp_path, monkeypatch
) -> None:
    import cytearc.readers.mgatk as mgatk
    from tests.storage_helpers import open_store

    real = mgatk.mgatk_source
    threads: list[int] = []

    def recording(path, **options):
        threads.append(options["threads"])
        return real(path, **options)

    monkeypatch.setattr(mgatk, "mgatk_source", recording)
    folder = write_mgatk(TRUTH, tmp_path / "final")
    shutil.copytree(template, tmp_path / "store" / template.name)
    store = open_store(tmp_path / "store" / template.name, nthreads=4)
    for budget in (512 << 20, 2 << 30):
        store.memoryBytes = budget
        _import(store, folder, invalidate_cache=True)
    assert threads == [1, 4]


def test_load_counts_takes_arrays_and_indexes_of_sites_and_cells(
    store, tmp_path
) -> None:
    ref, _ = _import(store, write_mgatk(TRUTH, tmp_path / "final"))
    ids = [str(cell) for cell in TRUTH.clones.cell_ids[[3, 1]]]
    positions = [SITE_POSITIONS[4], SITE_POSITIONS[0]]
    listed = store.variants.load_counts(ref, sites=positions, cell_selection=ids)
    for sites, cells in (
        (np.asarray(positions), np.asarray(ids, dtype=object)),
        (pd.Index(positions), pd.Index(ids)),
        (np.asarray([mito_label(p) for p in positions]), np.asarray(ids, dtype=str)),
    ):
        loaded = store.variants.load_counts(ref, sites=sites, cell_selection=cells)
        assert_tables(loaded.counts, listed.counts)
        assert loaded.cells.index.tolist() == listed.cells.index.tolist()
    with pytest.raises(TypeError, match="sites must hold site labels or positions"):
        store.variants.load_counts(ref, sites=np.zeros((2, 2), dtype=int))


def test_cellsnp_lite_uppercases_the_alleles_of_its_vcf(store, tmp_path) -> None:
    ref, _ = _import(
        store,
        write_cellsnp(TRUTH, tmp_path / "cellsnp", lowercase=True),
        count_unit="umis",
    )
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_variant_sites(TRUTH, other=True)
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)


def test_giftwrap_support_in_any_column_order_follows_its_counts(
    store, tmp_path
) -> None:
    ref, _ = _import(
        store,
        write_giftwrap_h5(TRUTH, tmp_path / "counts.1.h5", unsorted_support=True),
    )
    _depth, counts = expected_gift(TRUTH)
    assert_tables(store.variants.load_counts(ref).counts, counts)


def test_vartrix_records_split_at_one_position_are_refused(store, tmp_path) -> None:
    folder = write_vartrix(TRUTH, tmp_path / "vartrix")
    vcf = folder / "variants.vcf"
    vcf.write_text(
        vcf.read_text().replace(
            f"{CONTIG}\t2\t.\tA\tC,G",
            f"{CONTIG}\t2\t.\tA\tC\t.\t.\t.\n{CONTIG}\t2\t.\tA\tG",
        )
    )
    # The truth's own variants may lie at chrM:2 too, so either pair is named.
    with pytest.raises(
        ValueError,
        match="lists two records at chrM:2, A>[CG] and A>[CG], as a VCF with split "
        "multi-allelic sites does",
    ):
        store.variants.import_counts(
            folder, cell_selection=every_cell(store), count_unit="umis"
        )


def test_a_got_table_with_a_byte_order_mark_reads_alike(store, tmp_path) -> None:
    path = write_got(TRUTH, tmp_path / "CALR.tsv", style="xv")
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    ref, _ = _import(store, path)
    loaded = store.variants.load_counts(ref)
    depth, counts = expected_got(TRUTH, site="CALR")
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)


def test_got_multi_reads_come_with_every_call_that_has_umis(store, tmp_path) -> None:
    import re

    path = write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    def written(name: str, table: pd.DataFrame) -> Path:
        (tmp_path / name).mkdir()
        target = tmp_path / name / "ironthrone_out_pp.csv"
        table.to_csv(target, index=False)
        return target

    partial = written("partial", frame.drop(columns="amb_reads_per_umi_total_geneseq"))
    with pytest.raises(
        ValueError,
        match="lacks the column 'amb_reads_per_umi_total_geneseq', which its other "
        "reads_per_umi_total columns need",
    ):
        store.variants.import_counts(partial, cell_selection=every_cell(store))
    covered = frame["wt_calls_geneseq"].astype(int) > 0
    row = int(np.flatnonzero(covered)[0])
    empty = frame.copy()
    empty.loc[row, "wt_reads_per_umi_total_geneseq"] = ""
    with pytest.raises(
        ValueError,
        match="'wt_reads_per_umi_total_geneseq' of .* is empty for the cell "
        + re.escape(repr(frame.loc[row, "BC"]))
        + ", which has UMIs of that call",
    ):
        store.variants.import_counts(
            written("empty", empty), cell_selection=every_cell(store)
        )
    # Reads beside no UMIs are never stored, so an empty one there reads.
    idle = frame.copy()
    unused = int(np.flatnonzero(frame["amb_calls_geneseq"].astype(int) == 0)[0])
    idle.loc[unused, "amb_reads_per_umi_total_geneseq"] = ""
    ref, _ = _import(store, written("idle", idle))
    counts, _calls = _expected_got_multi(TRUTH)
    assert_tables(store.variants.load_counts(ref).counts, counts)


def test_a_budget_without_room_for_the_rewritten_chunks_writes_nothing(
    store, tmp_path
) -> None:
    # The rows parse and spill within the budget that reading the cells' IDs
    # takes, but appending them rewrites chunks of about four times the
    # 536,000 rows' 4-byte values.
    ids = [str(cell) for cell in store.cells.fetch_all("ids")]
    folder = tmp_path / "final"
    _large_mgatk(folder, ids, seed=4, n_positions=4_000)

    def run() -> object:
        return store.variants.import_counts(folder, cell_selection=every_cell(store))

    store.memoryBytes = _refused_estimate(store, run, 1)
    with pytest.raises(MemoryError, match="rewrites stored chunks of about"):
        run()
    assert store.artifacts.list(kind="variant_counts") == []


@pytest.mark.slow
def test_the_traced_peaks_of_import_and_load_fit_the_smallest_budget_accepted(
    tmp_path,
) -> None:
    """With 80-character IDs, reading them dominates what each step holds."""
    from tests.storage_helpers import open_store, write_count_store

    n_cells = 20_000
    ids = [f"{'sample_with_a_long_name_' * 3}{cell:08d}-1" for cell in range(n_cells)]
    write_count_store(
        str(tmp_path / "long.zarr"),
        {"RNA": np.ones((n_cells, 1), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = open_store(tmp_path / "long.zarr")
    got = tmp_path / "CALR.tsv"
    got.write_text(
        "BC\twtUMIs\tmutUMIs\n"
        + "".join(
            f"{ids[cell]}\t{1 + cell % 3}\t{cell % 2}\n"
            for cell in range(0, n_cells, 100)
        )
    )
    cells = store.snapshot_cell_selection()
    counts = store.variants.import_counts(got, cell_selection=cells)
    steps = {
        "import": lambda: store.variants.import_counts(
            got, cell_selection=cells, invalidate_cache=True
        ),
        "load": lambda: store.variants.load_counts(counts),
        "load by ID": lambda: store.variants.load_counts(
            counts, cell_selection=ids[:50:10]
        ),
    }
    for name, run in steps.items():
        store.memoryBytes = 1 << 33
        run()
        budget = smallest_budget(at_budget(store, run), 1 << 33)
        store.memoryBytes = budget
        _result, peak = traced_peak(run, budget)
        assert peak <= budget, (name, peak, budget)


@pytest.mark.slow
def test_loading_a_few_cells_of_many_fits_the_smallest_budget_accepted(
    tmp_path,
) -> None:
    """Counts of 500,000 cells hold offsets and per-cell arrays for each one."""
    from tests.storage_helpers import open_store, write_count_store

    n_cells = 500_000
    ids = [f"c{cell}" for cell in range(n_cells)]
    write_count_store(
        str(tmp_path / "many.zarr"),
        {"RNA": np.ones((n_cells, 1), dtype=np.uint16)},
        "uint16",
        cell_ids=ids,
    )
    store = open_store(tmp_path / "many.zarr")
    got = tmp_path / "CALR.tsv"
    got.write_text(
        "BC\twtUMIs\tmutUMIs\n"
        + "".join(f"{ids[cell]}\t1\t0\n" for cell in range(0, n_cells, 2_000))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        counts = store.variants.import_counts(
            got, cell_selection=store.snapshot_cell_selection()
        )
    store.cells.insert("first", np.arange(n_cells) == 0)
    first = store.snapshot_cell_selection("first")
    for name, run in {
        "one cell": lambda: store.variants.load_counts(counts, cell_selection=first),
        "no cell": lambda: store.variants.load_counts(counts, cell_selection=[]),
    }.items():
        store.memoryBytes = 1 << 33
        run()
        budget = smallest_budget(at_budget(store, run), 1 << 33)
        store.memoryBytes = budget
        _result, peak = traced_peak(run, budget)
        assert peak <= budget, (name, peak, budget)
