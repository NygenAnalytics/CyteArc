"""Reading real variant files, and the files that each reader refuses.

Real files come from the tools' own repositories under permissive licenses,
listed in THIRD_PARTY_NOTICES.md. Their expected values are taken with pandas
and SciPy from the files, or from values that the tools' own tests pin.
"""

import gzip
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

from cytearc.readers._delimited import first_lines
from cytearc.readers._matrix_market import (
    matrix_market_entries,
    matrix_market_header,
)
from cytearc.readers._variant_files import (
    folder_format,
    hdf5_format,
    read_barcodes,
    read_vcf,
)
from cytearc.readers.cellsnp import cellsnp_source
from cytearc.readers.giftwrap import giftwrap_source
from cytearc.readers.got import got_source, got_table_format
from cytearc.readers.mgatk import mgatk_source
from cytearc.readers.mito_afm import mito_afm_source
from cytearc.readers.vartrix import vartrix_source
from tests.storage_helpers import open_store, write_count_store
from tests.network_helpers import download_or_skip
from tests.variant_helpers import assert_tables

DATA = Path(__file__).parent / "data" / "variants"


def _store(path: Path, cell_ids: list[str]):
    rng = np.random.default_rng(0)
    write_count_store(
        str(path),
        {"RNA": rng.poisson(2.0, size=(len(cell_ids), 4)) + 1},
        "uint16",
        cell_ids=cell_ids,
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store, store.snapshot_cell_selection("every_cell")


def _import(store, cells, path, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = store.variants.import_counts(path, cell_selection=cells, **options)
    return store.variants.load_counts(ref), store.artifacts.load(ref)


def _base_table(folder: Path, base: str, *, six: bool) -> pd.DataFrame:
    path = next(folder.glob(f"*{base}.txt.gz"))
    names = (
        ["position", "cell", "fw", "qfw", "rv", "qrv"]
        if six
        else ["position", "cell", "fw", "rv"]
    )
    table = pd.read_csv(path, header=None, names=names)
    table["base"] = base
    return table


def _mgatk_oracle(folder: Path, *, six: bool, swapped: bool, contig: str, reference):
    """Depth and counts from the four files, the N's position left out."""
    table = pd.concat([_base_table(folder, base, six=six) for base in "ACGT"])
    table = table[[reference[position - 1] != "N" for position in table["position"]]]
    table["site"] = [
        f"{contig}:{position}" if contig else str(position)
        for position in table["position"]
    ]
    table["count"] = table["fw"] + table["rv"]
    forward, reverse = ("rv", "fw") if swapped else ("fw", "rv")
    counts = pd.DataFrame(
        {
            "ids": table["cell"],
            "site": table["site"],
            "allele": table["base"],
            "count": table["count"],
            "forward": table[forward],
            "reverse": table[reverse],
        }
    )
    depth = (
        table.groupby(["cell", "site"], sort=False)["count"].sum().reset_index()
    ).rename(columns={"count": "depth", "cell": "ids"})
    return depth, counts


def _reference(path: Path) -> str:
    text = gzip.decompress(path.read_bytes()).decode()
    return "".join(line.split("\t")[1] for line in text.splitlines()).upper()


def test_signac_fixture_gives_the_frequencies_signac_pins(tmp_path) -> None:
    folder = DATA / "signac_test_mgatk"
    cells = pd.read_csv(folder / "test.depthTable.txt", sep="\t", header=None)[
        0
    ].tolist()
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, group = _import(store, selection, folder)
    # Signac's ReadMGATK reads 11 cells.
    assert loaded.cells["listed"].sum() == 11
    depth = loaded.depth.assign(
        cell=loaded.depth["ids"].astype(str), site=loaded.depth["site"].astype(str)
    ).set_index(["cell", "site"])["depth"]
    counts = loaded.counts.assign(
        cell=loaded.counts["ids"].astype(str), site=loaded.counts["site"].astype(str)
    ).set_index(["cell", "site", "allele"])["count"]

    def frequency(cell: str, variant: str) -> float:
        position, change = int(variant[:-3]), variant[-3:]
        site = f"chrM:{position}"
        assert loaded.sites.loc[site, "reference"] == change[0]
        return counts.get((cell, site, change[2]), 0) / depth[(cell, site)]

    # The nonzero entries of the matrix that Signac's test-mito.R pins for
    # AlleleFreq of 627G>A, 1888G>A and 1888G>C.
    pinned = {
        ("ACAGGCCGTGGTCGAA-1", "1888G>A"): 0.0947368421052632,
        ("AAACTGCAGAGTCCGA-1", "627G>A"): 0.0955631399317406,
        ("AAACTGCAGAGTCCGA-1", "1888G>C"): 0.00273972602739726,
        ("ACTAGGTAGTGTCGGA-1", "1888G>A"): 0.108108108108108,
        ("ACCATCCTCTTGTCGC-1", "627G>A"): 0.0833333333333333,
        ("ACCATCCTCTTGTCGC-1", "1888G>C"): 0.00409836065573771,
        ("ACAGAAATCTATCTCA-1", "1888G>A"): 0.0256410256410256,
        ("ACAGAAATCTATCTCA-1", "1888G>C"): 0.00233100233100233,
        ("AATGGCTCAACGTAGG-1", "627G>A"): 0.087431693989071,
        ("ACCAAACGTTGTGAGG-1", "627G>A"): 0.13768115942029,
        ("AACCGATTCAAGAGAT-1", "1888G>A"): 0.0308788598574822,
        ("ACTAGGTCAATGCCAT-1", "1888G>A"): 0.072,
        ("AAAGGGCAGCTACGCC-1", "627G>A"): 0.0254237288135593,
        ("AAAGGGCAGCTACGCC-1", "1888G>C"): 0.00689655172413793,
    }
    for (cell, variant), value in pinned.items():
        assert frequency(cell, variant) == pytest.approx(value, rel=1e-12)
    # The datasets fork recomputed 627G>A in AAACTGCAGAGTCCGA-1 as 28 of 293.
    assert counts[("AAACTGCAGAGTCCGA-1", "chrM:627", "A")] == 28
    assert depth[("AAACTGCAGAGTCCGA-1", "chrM:627")] == 293
    reference = _reference(folder / "chrM_refAllele.txt.gz")
    oracle_depth, oracle_counts = _mgatk_oracle(
        folder, six=False, swapped=False, contig="chrM", reference=reference
    )
    assert_tables(loaded.depth, oracle_depth)
    assert_tables(loaded.counts, oracle_counts)
    # rCRS: 16,569 positions with the N at 3107; 418 soft-masked bases read uppercased.
    assert len(loaded.sites) == 16_568 and "chrM:3107" not in loaded.sites.index
    assert loaded.contigs["length"].tolist() == [16_569]
    assert group.attrs["contigs"][0]["name"] == "chrM"


def test_mito_mtscatac_subset_reads_as_mgatk_with_an_unnamed_contig(tmp_path) -> None:
    folder = DATA / "mito_mtscatac"
    cells = sorted(set(pd.read_csv(folder / "pL1000.A.txt.gz", header=None)[1]))
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, folder)
    reference = _reference(folder / "pL1000.refAllele.txt.gz")
    depth, counts = _mgatk_oracle(
        folder, six=False, swapped=False, contig="", reference=reference
    )
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    # nf-MiTo's refAllele.txt names no contig.
    assert loaded.contigs["name"].tolist() == [""]
    assert loaded.format == "mgatk"


def test_mito_maester_subset_reads_as_maegatk_with_strands_swapped_back(
    tmp_path,
) -> None:
    folder = DATA / "mito_maester"
    cells = sorted(set(pd.read_csv(folder / "A.txt.gz", header=None)[1]))
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, folder, format="maegatk")
    reference = _reference(folder / "chrM_refAllele.txt.gz")
    depth, counts = _mgatk_oracle(
        folder, six=True, swapped=True, contig="chrM", reference=reference
    )
    assert_tables(loaded.depth, depth)
    assert_tables(loaded.counts, counts)
    # maegatk's forward quality column belongs to its reverse count.
    table = _base_table(folder, "A", six=True)
    row = table[(table["fw"] > 0) & (table["rv"] > 0)].iloc[0]
    loaded_row = loaded.counts[
        (loaded.counts["ids"].astype(str) == row["cell"])
        & (loaded.counts["site"].astype(str) == f"chrM:{row['position']}")
        & (loaded.counts["allele"] == "A")
    ].iloc[0]
    assert loaded_row["quality_forward"] == pytest.approx(row["qrv"], abs=1e-4)
    assert loaded_row["quality_reverse"] == pytest.approx(row["qfw"], abs=1e-4)


def test_cellsnp_lite_subset_derives_the_reference_from_dp(tmp_path) -> None:
    folder = DATA / "cellsnp_lite"
    cells = (folder / "cellSNP.samples.tsv").read_text().split()
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, folder, count_unit="umis")
    records = pd.read_csv(
        folder / "cellSNP.base.vcf.gz", sep="\t", comment="#", header=None
    )
    sites = [
        f"{chrom}:{pos}" for chrom, pos in zip(records[0], records[1], strict=True)
    ]

    def matrix(tag: str) -> np.ndarray:
        return sp.coo_matrix(
            scipy.io.mmread(folder / f"cellSNP.tag.{tag}.mtx")
        ).toarray()

    ad, dp, oth = matrix("AD"), matrix("DP"), matrix("OTH")
    depth_rows, count_rows = [], []
    for record in range(60):
        for cell in range(40):
            total = dp[record, cell] + oth[record, cell]
            if total:
                depth_rows.append((cells[cell], sites[record], total))
            for allele, value in (
                (records[3][record], dp[record, cell] - ad[record, cell]),
                (records[4][record], ad[record, cell]),
                ("other", oth[record, cell]),
            ):
                if value:
                    count_rows.append((cells[cell], sites[record], allele, value))
    assert_tables(
        loaded.depth, pd.DataFrame(depth_rows, columns=["ids", "site", "depth"])
    )
    assert_tables(
        loaded.counts,
        pd.DataFrame(count_rows, columns=["ids", "site", "allele", "count"]),
    )
    # Contigs follow natural order.
    names = sorted({str(name) for name in records[0]}, key=int)
    assert loaded.contigs["name"].tolist() == names


def test_mquad_example_reads_ad_and_dp_of_its_passed_variants(tmp_path) -> None:
    folder = DATA / "mquad"
    cells = (folder / "cellSNP.samples.tsv").read_text().split()
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, folder, count_unit="reads")
    names = (folder / "passed_variant_names.txt").read_text().split()
    ad = scipy.io.mmread(folder / "passed_ad.mtx").toarray()
    dp = scipy.io.mmread(folder / "passed_dp.mtx").toarray()
    depth_rows, count_rows = [], []
    for variant, name in enumerate(names):
        site = f"chrM:{name[:-3]}"
        for cell, cell_id in enumerate(cells):
            if dp[variant, cell]:
                depth_rows.append((cell_id, site, int(dp[variant, cell])))
            for allele, value in (
                (name[-3], dp[variant, cell] - ad[variant, cell]),
                (name[-1], ad[variant, cell]),
            ):
                if value:
                    count_rows.append((cell_id, site, allele, int(value)))
    assert_tables(
        loaded.depth, pd.DataFrame(depth_rows, columns=["ids", "site", "depth"])
    )
    assert_tables(
        loaded.counts,
        pd.DataFrame(count_rows, columns=["ids", "site", "allele", "count"]),
    )
    assert len(loaded.sites) == 12


def test_vartrix_test_matrices_give_hand_derived_counts(tmp_path) -> None:
    folder = DATA / "vartrix"
    cells = read_barcodes(str(folder / "barcodes.tsv")).tolist()
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, folder, count_unit="reads")
    # test.vcf: 1:200 C>G, 17:200 G>A, 2:200 G>C, 7:200 A>G. The alt matrix
    # holds zeros for rows 1 and 2 and 7 reads of row 3 in cell 18; the ref
    # matrix 1 read of row 1 in cell 20, 6 and 1 of row 2 in cells 20 and 15.
    assert loaded.sites.index.tolist() == ["1:200", "2:200", "7:200", "17:200"]
    assert_tables(
        loaded.depth,
        pd.DataFrame(
            [
                (cells[19], "1:200", 1),
                (cells[19], "17:200", 6),
                (cells[14], "17:200", 1),
                (cells[17], "2:200", 7),
            ],
            columns=["ids", "site", "depth"],
        ),
    )
    assert_tables(
        loaded.counts,
        pd.DataFrame(
            [
                (cells[19], "1:200", "C", 1),
                (cells[19], "17:200", "G", 6),
                (cells[14], "17:200", "G", 1),
                (cells[17], "2:200", "C", 7),
            ],
            columns=["ids", "site", "allele", "count"],
        ),
    )
    assert loaded.contigs["length"].tolist() == [400, 400, 400, 400]


def test_giftwrap_sample_subset_matches_its_matrices(tmp_path) -> None:
    path = DATA / "giftwrap_counts.1.filtered.h5"
    with h5py.File(path, "r") as handle:
        cells = [value.decode() for value in handle["matrix/barcode"][:]]
        probes = handle["matrix/probe"][:].astype(str)

        def matrix(name: str) -> sp.csr_matrix:
            group = handle["matrix"][name]
            return sp.csr_matrix(
                (group["data"][:], group["indices"][:], group["indptr"][:]),
                shape=tuple(group.attrs["shape"]),
            )

        umis, reads, support = (
            matrix("data"),
            matrix("total_reads"),
            matrix("percent_supporting"),
        )
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, path)
    rows, columns = umis.nonzero()
    expected = pd.DataFrame(
        {
            "ids": np.asarray(cells)[rows],
            "site": probes[columns, 0],
            "allele": probes[columns, 1],
            "count": np.asarray(umis[rows, columns]).ravel(),
            "reads": np.asarray(reads[rows, columns]).ravel(),
            "support": np.asarray(support[rows, columns]).ravel(),
        }
    )
    assert_tables(loaded.counts, expected)
    depth = (
        expected.groupby(["ids", "site"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "depth"})
    )
    assert_tables(loaded.depth, depth)
    # The sample's manifest names no wild type or mutant, so every gapfill is "other".
    assert set(loaded.alleles["role"]) == {"other"}
    assert loaded.count_unit == "umis"


@pytest.mark.parametrize(
    ("name", "site"),
    [
        ("Patient1_TET2.2847_summTable.txt", "Patient1_TET2.2847"),
        ("Patient5_ZRSR2.500_summTable.txt", "Patient5_ZRSR2.500"),
    ],
)
def test_xv_seq_got_tables_count_wild_type_and_mutant_umis(
    tmp_path, name, site
) -> None:
    path = DATA / "got" / name
    table = pd.read_csv(path, sep="\t")
    assert got_table_format(path) == "got"
    store, selection = _store(tmp_path / "store.zarr", table["BC"].tolist())
    loaded, _group = _import(store, selection, path)
    counts = pd.concat(
        [
            pd.DataFrame(
                {
                    "ids": table["BC"],
                    "site": site,
                    "allele": "WT",
                    "count": table["wtUMIs"],
                }
            ),
            pd.DataFrame(
                {
                    "ids": table["BC"],
                    "site": site,
                    "allele": "MUT",
                    "count": table["mutUMIs"],
                }
            ),
        ]
    )
    assert_tables(loaded.counts, counts[counts["count"] > 0])
    depth = pd.DataFrame(
        {"ids": table["BC"], "site": site, "depth": table["wtUMIs"] + table["mutUMIs"]}
    )
    assert_tables(loaded.depth, depth[depth["depth"] > 0])
    assert loaded.alleles["role"].tolist() == ["alternative", "reference"]


# The formats each folder or file is recognised as.
def test_formats_are_recognised_from_files(tmp_path) -> None:
    assert folder_format(str(DATA / "signac_test_mgatk")) == "mgatk"
    assert folder_format(str(DATA / "mito_maester")) == "mgatk"
    assert folder_format(str(DATA / "cellsnp_lite")) == "cellsnp_lite"
    assert folder_format(str(DATA / "mquad")) == "mquad"
    assert folder_format(str(DATA / "vartrix")) == "vartrix"
    assert hdf5_format(str(DATA / "giftwrap_counts.1.filtered.h5")) == "giftwrap"
    with pytest.raises(ValueError, match="holds none of the files"):
        folder_format(str(tmp_path))
    with h5py.File(tmp_path / "other.h5", "w") as handle:
        handle.create_group("matrix")
    with pytest.raises(
        ValueError, match="none of giftwrap's counts, MiTo's allele frequency matrix"
    ):
        hdf5_format(str(tmp_path / "other.h5"))


def _copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def test_mgatk_reader_refuses_malformed_folders(tmp_path) -> None:
    folder = _copy(DATA / "signac_test_mgatk", tmp_path / "a")
    (folder / "test.T.txt.gz").unlink()
    with pytest.raises(FileNotFoundError, match="holds no T base file"):
        mgatk_source(folder, format=None, swap_strands=False)
    folder = _copy(DATA / "signac_test_mgatk", tmp_path / "b")
    shutil.copy(folder / "test.A.txt.gz", folder / "other.A.txt.gz")
    with pytest.raises(ValueError, match="holds several A base file"):
        mgatk_source(folder, format=None, swap_strands=False)
    folder = _copy(DATA / "signac_test_mgatk", tmp_path / "c")
    (folder / "test.A.txt.gz").write_bytes(gzip.compress(b"1,AAAC-1,3,4,5\n"))
    with pytest.raises(ValueError, match="holds 5 fields per row"):
        mgatk_source(folder, format=None, swap_strands=False)
    (folder / "test.A.txt.gz").write_bytes(gzip.compress(b"1,AAAC-1,3,30.5\n"))
    with pytest.raises(ValueError, match="as mgatk wrote before 0.4.7"):
        mgatk_source(folder, format=None, swap_strands=False)
    (folder / "test.A.txt.gz").write_bytes(gzip.compress(b"1,AAAC-1,3,4,5,6\n"))
    with pytest.raises(ValueError, match="hold different columns"):
        mgatk_source(folder, format=None, swap_strands=False)
    folder = _copy(DATA / "signac_test_mgatk", tmp_path / "d")
    with pytest.raises(ValueError, match="maegatk always writes six"):
        mgatk_source(folder, format="maegatk", swap_strands=False)
    with pytest.raises(ValueError, match="format must be one of mgatk, maegatk"):
        mgatk_source(folder, format="cellsnp_lite", swap_strands=False)
    reference = folder / "chrM_refAllele.txt.gz"
    reference.write_bytes(gzip.compress(b"1\tG\n2\tR\n"))
    with pytest.raises(ValueError, match="Position 2 of .* holds 'R'"):
        mgatk_source(folder, format=None, swap_strands=False)
    reference.write_bytes(gzip.compress(b"1\tG\n3\tA\n"))
    with pytest.raises(ValueError, match="must run from 1 in order"):
        mgatk_source(folder, format=None, swap_strands=False)
    reference.write_bytes(gzip.compress(b"1\tG\tx\n"))
    with pytest.raises(ValueError, match="must hold a position and a base"):
        mgatk_source(folder, format=None, swap_strands=False)


def test_mgatk_rows_are_checked_as_they_are_read(tmp_path) -> None:
    folder = _copy(DATA / "signac_test_mgatk", tmp_path / "a")
    (folder / "test.A.txt.gz").write_bytes(
        gzip.compress(b"1,AAAC-1,3,4\n2,AAAC-1,x,1\n")
    )
    source = mgatk_source(folder, format=None, swap_strands=False)
    with pytest.raises(ValueError, match="holds a row that is not 4 fields"):
        list(source.batches(10))
    (folder / "test.A.txt.gz").write_bytes(gzip.compress(b"1,AAAC-1,-3,4\n"))
    with pytest.raises(ValueError, match="holds a negative count"):
        list(mgatk_source(folder, format=None, swap_strands=False).batches(10))
    (folder / "test.A.txt.gz").write_bytes(gzip.compress(b"0,AAAC-1,3,4\n"))
    with pytest.raises(ValueError, match="counts position 0, outside"):
        list(mgatk_source(folder, format=None, swap_strands=False).batches(10))
    (folder / "test.A.txt.gz").write_bytes(
        gzip.compress(f"1,AAAC-1,{2**33},4\n".encode())
    )
    with pytest.raises(ValueError, match="holds a count above"):
        list(mgatk_source(folder, format=None, swap_strands=False).batches(10))
    (folder / "test.depthTable.txt").write_text("AAAC-1\t3\textra\n")
    with pytest.raises(ValueError, match="must hold a barcode and a mean depth"):
        list(mgatk_source(folder, format=None, swap_strands=False).batches(10))
    # An empty base file holds no rows.
    (folder / "test.depthTable.txt").write_text("AAAC-1\t0\n")
    (folder / "test.A.txt.gz").write_bytes(gzip.compress(b""))
    batches = list(mgatk_source(folder, format=None, swap_strands=False).batches(10))
    assert batches[0].kind == "cells"


def _renamed_mgatk(folder: Path, prefix: str, reference: str) -> Path:
    """Signac's mgatk files with the base files and depth table under
    ``prefix`` and the reference named ``reference``."""
    for path in sorted(folder.iterdir()):
        if path.name.startswith("test."):
            path.rename(folder / (prefix + path.name[len("test.") :]))
    text = gzip.decompress((folder / "chrM_refAllele.txt.gz").read_bytes())
    (folder / "chrM_refAllele.txt.gz").unlink()
    target = folder / reference
    target.write_bytes(gzip.compress(text) if reference.endswith(".gz") else text)
    return folder


@pytest.mark.parametrize(
    ("prefix", "name", "contig"),
    [
        # mgatk's own folder: the reference carries no prefix.
        ("test.", "chrM_refAllele.txt", "chrM"),
        ("test.", "hg19_chrM_refAllele.txt", "hg19_chrM"),
        # Deposits prefix every file alike.
        ("TF1_filtered.", "TF1_filtered.chrM_refAllele.txt.gz", "chrM"),
        ("CRC_v12-mtMask_mgatk.", "CRC_v12-mtMask_mgatk.MT_refAllele.txt", "MT"),
        ("lib.v2.", "lib.v2.chrM_refAllele.txt", "chrM"),
        # A contig whose name holds an accession's version keeps it whole.
        ("test.", "NC_012920.1_refAllele.txt", "NC_012920.1"),
        ("sample.", "sample.NC_012920.1_refAllele.txt", "NC_012920.1"),
        ("", "NC_012920.1_refAllele.txt", "NC_012920.1"),
        # nf-MiTo's reference names no contig.
        ("", "refAllele.txt", ""),
        ("pL1000.", "pL1000.refAllele.txt", ""),
    ],
)
def test_the_contig_is_named_after_the_reference_file_without_its_prefix(
    tmp_path, prefix, name, contig
) -> None:
    folder = _renamed_mgatk(
        _copy(DATA / "signac_test_mgatk", tmp_path / "a"), prefix, name
    )
    source = mgatk_source(folder, format=None, swap_strands=False)
    assert source.sites().contigs[0].name == contig


@pytest.mark.parametrize(
    ("prefix", "name", "carried"),
    [
        # A prefix the base files do not carry, or one beside theirs.
        ("lib1.", "lib2.chrM_refAllele.txt", "'lib1.'"),
        ("", "pL1000.chrM_refAllele.txt", "no prefix"),
        ("lib1.", "lib1.lib2.chrM_refAllele.txt", "'lib1.'"),
    ],
)
def test_a_reference_whose_name_leaves_its_contig_unclear_is_refused(
    tmp_path, prefix, name, carried
) -> None:
    folder = _renamed_mgatk(
        _copy(DATA / "signac_test_mgatk", tmp_path / "a"), prefix, name
    )
    with pytest.raises(
        ValueError, match=f"{name.split('_')[0]}.* the base files carry {carried}"
    ):
        mgatk_source(folder, format=None, swap_strands=False)


def test_base_files_with_different_prefixes_are_refused(tmp_path) -> None:
    folder = _copy(DATA / "signac_test_mgatk", tmp_path / "a")
    (folder / "test.G.txt.gz").rename(folder / "other.G.txt.gz")
    with pytest.raises(
        ValueError, match="carry different prefixes, 'other.', 'test.'; they must come"
    ):
        mgatk_source(folder, format=None, swap_strands=False)


@pytest.mark.parametrize(
    ("fixture", "contig"),
    [("signac_test_mgatk", "chrM"), ("mito_mtscatac", ""), ("mito_maester", "chrM")],
)
def test_the_fixture_folders_name_their_contigs(fixture, contig) -> None:
    format = "maegatk" if fixture == "mito_maester" else None
    source = mgatk_source(DATA / fixture, format=format, swap_strands=False)
    assert [entry.name for entry in source.sites().contigs] == [contig]


def test_cellsnp_and_mquad_readers_refuse_inconsistent_folders(tmp_path) -> None:
    folder = _copy(DATA / "cellsnp_lite", tmp_path / "a")
    (folder / "cellSNP.samples.tsv").write_text("A\nB\n")
    with pytest.raises(
        ValueError,
        match="is a 60 by 40 matrix, but its folder lists 60 variants and 2 cells",
    ):
        cellsnp_source(folder, format="cellsnp_lite")
    folder = _copy(DATA / "cellsnp_lite", tmp_path / "b")
    vcf = (
        gzip.decompress((folder / "cellSNP.base.vcf.gz").read_bytes())
        .decode()
        .splitlines()
    )
    for change, message in (
        (
            lambda line: (
                line.replace("\tA\t", "\tA,T\t", 1) if "\tG\tA\t" in line else line
            ),
            "each record needs one",
        ),
        (
            lambda line: line.replace("\tG\tA\t", "\tG\tG\t"),
            "the same reference and alternative",
        ),
    ):
        changed = [change(line) if not line.startswith("#") else line for line in vcf]
        (folder / "cellSNP.base.vcf.gz").write_bytes(
            gzip.compress(("\n".join(changed) + "\n").encode())
        )
        with pytest.raises(ValueError, match=message):
            cellsnp_source(folder, format="cellsnp_lite")
    records = [line for line in vcf if not line.startswith("#")]
    duplicated = (
        [line for line in vcf if line.startswith("#")] + [records[0]] * 2 + records[2:]
    )
    (folder / "cellSNP.base.vcf.gz").write_bytes(
        gzip.compress(("\n".join(duplicated) + "\n").encode())
    )
    with pytest.raises(ValueError, match="lists two variants at 1:"):
        cellsnp_source(folder, format="cellsnp_lite")
    with pytest.raises(ValueError, match="format must be cellsnp_lite or mquad"):
        cellsnp_source(folder, format="vartrix")
    folder = _copy(DATA / "mquad", tmp_path / "c")
    (folder / "cellSNP.samples.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="MQuad writes none"):
        cellsnp_source(folder, format="mquad")
    folder = _copy(DATA / "mquad", tmp_path / "d")
    (folder / "passed_variant_names.txt").write_text("chrM_2956_C_A\n" * 12)
    with pytest.raises(ValueError, match="MQuad names variants as position"):
        cellsnp_source(folder, format="mquad")
    folder = _copy(DATA / "mquad", tmp_path / "e")
    bic = pd.read_csv(folder / "BIC_params.csv")
    bic.iloc[1:].to_csv(folder / "BIC_params.csv", index=False)
    with pytest.raises(ValueError, match="does not list the passed variant '2956C>A'"):
        cellsnp_source(folder, format="mquad")
    bic.assign(
        variant_name=bic["variant_name"]
        .str.replace("chrM_", "MT_", n=1)
        .where(bic.index > 0, bic["variant_name"])
    ).to_csv(folder / "BIC_params.csv", index=False)
    with pytest.raises(
        ValueError, match="places the passed variants on several contigs"
    ):
        cellsnp_source(folder, format="mquad")
    bic.assign(variant_name="2956").to_csv(folder / "BIC_params.csv", index=False)
    with pytest.raises(ValueError, match="names them as contig, position"):
        cellsnp_source(folder, format="mquad")


def test_vartrix_reader_needs_its_four_files(tmp_path) -> None:
    folder = _copy(DATA / "vartrix", tmp_path / "a")
    (folder / "test_coverage_ref.mtx").rename(folder / "test_coverage_alt.mtx")
    with pytest.raises(ValueError, match="the reference one with 'ref' in its name"):
        vartrix_source(folder)
    folder = _copy(DATA / "vartrix", tmp_path / "b")
    shutil.copy(folder / "barcodes.tsv", folder / "more.tsv")
    with pytest.raises(ValueError, match="must hold one barcode list"):
        vartrix_source(folder)
    folder = _copy(DATA / "vartrix", tmp_path / "c")
    (folder / "barcodes.tsv").write_text("A\nB\n")
    with pytest.raises(ValueError, match="is a 4 by 20 matrix"):
        vartrix_source(folder)
    folder = _copy(DATA / "vartrix", tmp_path / "d")
    vcf = (folder / "test.vcf").read_text().replace("\tC\tG\t", "\tC\tG,T\t")
    (folder / "test.vcf").write_text(vcf)
    with pytest.raises(ValueError, match="which has several alternative alleles"):
        list(vartrix_source(folder).batches(10))
    (folder / "test.vcf").write_text(
        (DATA / "vartrix" / "test.vcf").read_text().replace("\tC\tG\t", "\tC\tC\t")
    )
    with pytest.raises(ValueError, match="the same reference and alternative allele"):
        vartrix_source(folder)
    folder = _copy(DATA / "vartrix", tmp_path / "e")
    (folder / "test_coverage.mtx").write_text(
        (folder / "test_coverage.mtx").read_text().replace("3 18 7", "3 18 0.25")
    )
    with pytest.raises(ValueError, match="counts must be finite non-negative integers"):
        list(vartrix_source(folder).batches(10))


def test_matrix_market_and_vcf_helpers_refuse_malformed_files(tmp_path) -> None:
    path = tmp_path / "m.mtx"
    for text, message in (
        ("not a matrix\n", "coordinate matrix format, but .* begins 'not a matrix'"),
        (
            "%%MatrixMarket matrix array real general\n2 2\n",
            "must use coordinate matrix format",
        ),
        (
            "%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 1\n",
            "integer or real coordinate values, but .* holds pattern values",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n%\n",
            "dimensions line must contain three integers, but .* ends before it",
        ),
        (
            "%%MatrixMarket matrix coordinate integer general\n2 x 1\n",
            "dimensions line must contain three integers, but .* has '2 x 1'",
        ),
    ):
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            matrix_market_header(str(path))
    for body, message in (
        ("3 1 1\n", "outside the declared dimensions, 2 by 2"),
        ("1 1 -1\n", "finite non-negative integers, but .* holds -1"),
        ("1 1 a\n", "Could not parse Matrix Market coordinates"),
        ("1 1 1\n1 2 1\n", "header declares 1 entries, but 2 were read"),
    ):
        path.write_text(
            "%%MatrixMarket matrix coordinate integer general\n2 2 1\n" + body
        )
        with pytest.raises(ValueError, match=message):
            list(
                matrix_market_entries(matrix_market_header(str(path)), 10, counts=True)
            )
    vcf = tmp_path / "v.vcf"
    vcf.write_text("1\t2\t.\tA\tC\n")
    with pytest.raises(ValueError, match="no #CHROM header line before"):
        read_vcf(str(vcf))
    vcf.write_text("##fileformat=VCFv4.2\n")
    with pytest.raises(ValueError, match="has no #CHROM header line"):
        read_vcf(str(vcf))
    vcf.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\tx\t.\tA\tC\n")
    with pytest.raises(ValueError, match="Line 2 of .* holds the position 'x'"):
        read_vcf(str(vcf))
    barcodes = tmp_path / "b.tsv"
    barcodes.write_text("A\nB\nA\n")
    with pytest.raises(ValueError, match="The barcode 'A' repeats"):
        read_barcodes(str(barcodes))


def test_giftwrap_reader_refuses_files_that_do_not_hold_its_matrices(tmp_path) -> None:
    path = tmp_path / "counts.h5"
    shutil.copy(DATA / "giftwrap_counts.1.filtered.h5", path)
    with h5py.File(path, "a") as handle:
        del handle["matrix/probe"]
    with pytest.raises(ValueError, match="lacks matrix/probe"):
        giftwrap_source(path)
    shutil.copy(DATA / "giftwrap_counts.1.filtered.h5", path)
    with h5py.File(path, "a") as handle:
        handle["matrix/data"].attrs["shape"] = np.asarray([5, 5])
    with pytest.raises(ValueError, match="are 5 by 5"):
        giftwrap_source(path)
    shutil.copy(DATA / "giftwrap_counts.1.filtered.h5", path)
    with h5py.File(path, "a") as handle:
        indices = handle["matrix/total_reads/indices"][:]
        indices[0] = indices[0] + 1
        handle["matrix/total_reads/indices"][:] = indices
    with pytest.raises(ValueError, match="must hold the entries of matrix/data"):
        giftwrap_source(path)


def test_giftwrap_tables_refuse_unknown_probes_and_bad_values(tmp_path) -> None:
    from tests.variant_truth import variant_truth, write_giftwrap_tables

    folder = write_giftwrap_tables(variant_truth(), tmp_path / "gift")
    reads = folder / "GSM0_GAP_probe_reads.tsv.gz"
    original = gzip.decompress(reads.read_bytes()).decode()

    def write_first_row(column: int, value: str) -> None:
        lines = original.splitlines()
        fields = lines[1].split("\t")
        fields[column] = value
        lines[1] = "\t".join(fields)
        reads.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))

    for column, value, message in (
        (1, "9", "names the probe index 9, which"),
        (1, "1.5", "probe_idx that is not a whole number"),
        (6, "1.5", "percent_supporting outside 0 to 1"),
        (0, "999", "cell_idx that is not a whole number"),
        (5, "0", "pcr_duplicate_count that is not a whole number from 1"),
    ):
        write_first_row(column, value)
        with pytest.raises(ValueError, match=message):
            giftwrap_source(folder)
    reads.write_bytes(gzip.compress(original.encode()))
    manifest = folder / "GSM0_GAP_manifest.tsv.gz"
    text = gzip.decompress(manifest.read_bytes()).decode()
    manifest.write_bytes(gzip.compress(text.replace("\tindex", "\tposition").encode()))
    with pytest.raises(ValueError, match="lacks the manifest column 'index'"):
        giftwrap_source(folder)
    lines = text.splitlines()
    manifest.write_bytes(
        gzip.compress(("\n".join([*lines, lines[-1]]) + "\n").encode())
    )
    with pytest.raises(ValueError, match="must hold distinct numbers"):
        giftwrap_source(folder)
    manifest.write_bytes(gzip.compress(b"# key=1\n"))
    with pytest.raises(ValueError, match="holds no probes"):
        giftwrap_source(folder)


def test_giftwrap_manifest_with_one_gapfill_as_both_reads_it_as_the_reference(
    tmp_path,
) -> None:
    from tests.variant_truth import variant_truth, write_giftwrap_tables

    folder = write_giftwrap_tables(variant_truth(), tmp_path / "gift")
    manifest = folder / "GSM0_GAP_manifest.tsv.gz"
    text = (
        gzip.decompress(manifest.read_bytes())
        .decode()
        .replace("\tAAC\tCAC\t", "\tCAC\tCAC\t")
    )
    manifest.write_bytes(gzip.compress(text.encode()))
    source = giftwrap_source(folder)
    table = source.sites()
    role = {
        (str(table.site_name[site]), str(allele)): str(role)
        for site, allele, role in zip(
            table.allele_site, table.allele, table.allele_role, strict=True
        )
    }
    assert role[("JAK2 c.1849G>T", "CAC")] == "reference"
    assert role[("JAK2 c.1849G>T", "AAC")] == "other"
    assert source.notes["same_gapfills"] == ["JAK2 c.1849G>T"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "is empty"),
        (
            "BC\tWT.calls\tMUT.calls\nA\t1\n",
            "header of .* names 3 columns, but its first row holds 2 fields",
        ),
        # A later row of another width is refused by its line, as every table's.
        (
            "BC\tWT.calls\tMUT.calls\nA\t1\t2\n\nB\t1\n",
            "is malformed: line 4 holds 2 fields, but its header names 3 columns",
        ),
        ("BC\tGenotype\nA\tWT\n\t\tMUT\n", "line 3 holds 3 fields"),
        ("BC\tGenotype\nA\tWT\n\tMUT\n", "has rows without a cell barcode on line 3$"),
        ("cell_name\tGenotype\nA\tWT\n", "has no column of cell barcodes"),
        ("BC\tBC\tGenotype\nA\tA\tWT\n", "names the column 'BC' twice"),
    ],
)
def test_got_tables_refuse_malformed_text(tmp_path, text, message) -> None:
    path = tmp_path / "t.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        got_source(path, format="got")


def test_got_tables_read_r_row_names_as_clone_tables_do(tmp_path) -> None:
    """A delimiter after each row's last field gives one field more than the
    header, as R's row names do, so a table whose rows of one field more all
    end in a delimiter is refused, while one row holding a value after its
    named columns makes them row names."""
    cells = ["AAAC-1", "AAAG-1"]
    store, selection = _store(tmp_path / "store.zarr", cells)
    path = tmp_path / "CALR.tsv"
    path.write_text("BC\tWT.calls\tMUT.calls\nAAAC-1\t1\t2\t\nAAAG-1\t0\t3\t\n")
    with pytest.raises(ValueError, match="end the header with the delimiter too"):
        got_source(path, format="got")
    counts = pd.DataFrame(
        {
            "ids": ["AAAC-1", "AAAC-1", "AAAG-1"],
            "site": "CALR",
            "allele": ["WT", "MUT", "MUT"],
            "count": [1, 2, 3],
        }
    )
    # A header that ends in the delimiter too names an empty last column.
    path.write_text("BC\tWT.calls\tMUT.calls\t\nAAAC-1\t1\t2\t\nAAAG-1\t0\t3\t\n")
    loaded, _group = _import(store, selection, path)
    assert_tables(loaded.counts, counts)
    # R leaves a missing last value empty with na=""; the second row's value
    # shows that the first field of each row is a row name.
    path.write_text(
        "BC\tWT.calls\tMUT.calls\tGenotype\n1\tAAAC-1\t1\t2\t\n2\tAAAG-1\t0\t3\tMUT\n"
    )
    loaded, _group = _import(store, selection, path, invalidate_cache=True)
    assert_tables(loaded.counts, counts)
    assert loaded.calls[["ids", "call"]].to_numpy().tolist() == [["AAAG-1", "MUT"]]


def test_got_formats_and_their_columns(tmp_path) -> None:
    path = tmp_path / "t.tsv"
    path.write_text("BC\tnum.WT.call\nA\t1\n")
    with pytest.raises(ValueError, match="pass format to name its format"):
        got_table_format(path)
    with pytest.raises(
        ValueError, match="counts only one of the wild type and the mutant"
    ):
        got_source(path, format="got")
    path.write_text("BC\tWT.calls\tMUT.calls\tTotal.calls\nA\t1\t2\t4\n")
    with pytest.raises(ValueError, match="Total.calls of .* is not the sum"):
        got_source(path, format="got")
    path.write_text("BC\tWT.calls\tMUT.calls\nA\t1.5\t2\n")
    with pytest.raises(ValueError, match="holds '1.5'; counts are whole numbers"):
        got_source(path, format="got")
    path.write_text("BC\tsomething\nA\t1\n")
    with pytest.raises(ValueError, match="holds neither GoT counts nor calls"):
        got_source(path, format="got")
    with pytest.raises(ValueError, match="lacks the summTable column 'WT.calls'"):
        got_source(path, format="ironthrone")
    with pytest.raises(ValueError, match="lacks GoT-Multi's 'target' column"):
        got_source(path, format="got_multi")
    with pytest.raises(ValueError, match="holds no WTcount and MUTcount"):
        got_source(path, format="gotcha")
    with pytest.raises(ValueError, match="format must be one of"):
        got_source(path, format="mgatk")
    path.write_text("BC,target,genotype_merged\nA,,WT\n")
    with pytest.raises(ValueError, match="has a row without a target"):
        got_source(path, format="got_multi")
    path.write_text("BC,target,genotype_merged,wt_calls_geneseq\nA,X,WT,1\n")
    with pytest.raises(ValueError, match="lacks the column"):
        got_source(path, format="got_multi")
    path.write_text('"","WTcount","genotype_pred"\n"A",1,"WT"\n')
    with pytest.raises(ValueError, match="has WTcount but no MUTcount"):
        got_source(path, format="gotcha")
    path.write_text('"X","TP53_WTcount"\n"A",1\n')
    with pytest.raises(ValueError, match="has TP53_WTcount but no TP53_MUTcount"):
        got_source(path, format="gotcha")
    unnamed = tmp_path / ".txt"
    unnamed.write_text("BC\tGenotype\nA\tWT\n")
    with pytest.raises(ValueError, match="has no name to name its target"):
        got_source(unnamed, format="got")


def test_mito_reader_takes_the_largest_dp_of_a_sites_variants(tmp_path) -> None:
    from tests.variant_truth import variant_truth, write_mito_afm

    path = write_mito_afm(variant_truth(), tmp_path / "afm.h5ad")
    with h5py.File(path, "a") as handle:
        var = handle["var"]
        for name, value in (("pos", var["pos"][0]), ("ref", var["ref"][0])):
            data = var[name][:]
            data[1] = value
            del var[name]
            var.create_dataset(name, data=data)
        alt = var["alt"][:].astype(str)
        alternative = next(
            base for base in "ACGT" if base not in (var["ref"][0].decode(), alt[0])
        )
        alt[1] = alternative
        del var["alt"]
        var.create_dataset("alt", data=alt.astype(h5py.string_dtype()))
        dp = handle["layers/DP"][:]
        dp[:, 1] = dp[:, 0] + 3
        handle["layers/DP"][:] = dp
    source = mito_afm_source(path)
    table = source.sites()
    # The first two variants share a site with three alleles.
    assert table.n_sites == 15 and table.n_alleles == 31
    depth = next(batch for batch in source.batches(10_000) if batch.kind == "depth")
    first_site = depth.keys == 0
    np.testing.assert_array_equal(
        depth.values[first_site], (dp[:, 0] + 3)[depth.barcode_codes[first_site]]
    )


def test_mito_reader_refuses_files_that_are_not_mitos(tmp_path) -> None:
    from tests.variant_truth import variant_truth, write_mito_afm

    path = write_mito_afm(variant_truth(), tmp_path / "afm.h5ad")
    for edit, message in (
        (lambda handle: handle["layers"].pop("DP"), "lacks MiTo's layers/DP"),
        (lambda handle: handle["var"].pop("pos"), "lacks MiTo's 'pos' column"),
    ):
        broken = tmp_path / "broken.h5ad"
        shutil.copy(path, broken)
        with h5py.File(broken, "a") as handle:
            edit(handle)
        with pytest.raises(ValueError, match=message):
            mito_afm_source(broken)
    broken = tmp_path / "shape.h5ad"
    shutil.copy(path, broken)
    with h5py.File(broken, "a") as handle:
        dp = handle["layers/DP"][:]
        del handle["layers/DP"]
        handle["layers"].create_dataset("DP", data=dp[:, :3])
    with pytest.raises(ValueError, match="layers/DP of .* is"):
        mito_afm_source(broken)
    broken = tmp_path / "negative.h5ad"
    shutil.copy(path, broken)
    with h5py.File(broken, "a") as handle:
        dp = handle["layers/DP"][:]
        dp[0, 0] = -1
        handle["layers/DP"][:] = dp
    with pytest.raises(ValueError, match="not a whole number of 0 or more"):
        list(mito_afm_source(broken).batches(10_000))
    broken = tmp_path / "csc.h5ad"
    shutil.copy(path, broken)
    with h5py.File(broken, "a") as handle:
        handle["layers/AD"].attrs["encoding-type"] = "csc_matrix"
    with pytest.raises(ValueError, match="MiTo writes CSR"):
        list(mito_afm_source(broken).batches(10_000))


def test_giftwrap_hdf5_with_inconsistent_probes_or_shapes_is_refused(tmp_path) -> None:
    from tests.variant_truth import variant_truth, write_giftwrap_h5

    original = write_giftwrap_h5(variant_truth(), tmp_path / "counts.1.h5")

    def edited(name: str, edit) -> Path:
        path = tmp_path / name
        shutil.copy(original, path)
        with h5py.File(path, "a") as handle:
            edit(handle)
        return path

    def repeat_name(handle) -> None:
        names = handle["probe_metadata/name"][:]
        names[1] = names[0]
        handle["probe_metadata/name"][:] = names

    with pytest.raises(ValueError, match="lists the probe 'JAK2 c.1849G>T' twice"):
        giftwrap_source(edited("repeat.h5", repeat_name))

    def flat_probes(handle) -> None:
        probes = handle["matrix/probe"][:]
        del handle["matrix/probe"]
        handle["matrix"].create_dataset("probe", data=probes[:, 0])

    with pytest.raises(ValueError, match="must hold a probe and a gapfill per column"):
        giftwrap_source(edited("flat.h5", flat_probes))

    def short_indptr(handle) -> None:
        indptr = handle["matrix/percent_supporting/indptr"][:]
        del handle["matrix/percent_supporting/indptr"]
        handle["matrix/percent_supporting"].create_dataset("indptr", data=indptr[:-1])

    with pytest.raises(
        ValueError, match="matrix/percent_supporting of .* has another number of cells"
    ):
        giftwrap_source(edited("short.h5", short_indptr))

    def outside(handle) -> None:
        for part in ("data", "total_reads"):
            indices = handle[f"matrix/{part}/indices"][:]
            indices[0] = 99
            handle[f"matrix/{part}/indices"][:] = indices

    with pytest.raises(
        ValueError, match="has entries outside its 6 probe and gapfill columns"
    ):
        list(giftwrap_source(edited("outside.h5", outside)).batches(1_000))

    def without_metadata(handle) -> None:
        del handle["probe_metadata"]

    # Without a manifest every probe comes from the counts, and every gapfill
    # is "other".
    source = giftwrap_source(edited("bare.h5", without_metadata))
    assert set(source.sites().allele_role.tolist()) == {"other"}
    assert source.sites().n_sites == 3


def test_giftwrap_tables_need_a_barcode_column(tmp_path) -> None:
    from tests.variant_truth import variant_truth, write_giftwrap_tables

    folder = write_giftwrap_tables(variant_truth(), tmp_path / "gift")
    (folder / "GSM0_GAP_barcodes.tsv.gz").write_bytes(
        gzip.compress(b"cell\tplex\nA\t1\n")
    )
    with pytest.raises(ValueError, match="lacks its 'barcode' column"):
        giftwrap_source(folder)


def test_mito_reader_refuses_var_and_obs_it_cannot_read(tmp_path) -> None:
    from tests.variant_truth import variant_truth, write_mito_afm

    original = write_mito_afm(variant_truth(), tmp_path / "afm.h5ad")

    def edited(name: str, edit) -> Path:
        path = tmp_path / name
        shutil.copy(original, path)
        with h5py.File(path, "a") as handle:
            edit(handle)
        return path

    def replace(handle, name: str, values) -> None:
        del handle["var"][name]
        handle["var"].create_dataset(name, data=values)

    def missing_ref(handle) -> None:
        refs = handle["var/ref"][:]
        del handle["var/ref"]
        group = handle["var"].create_group("ref")
        group.attrs["encoding-type"] = "categorical"
        group.attrs["encoding-version"] = "0.2.0"
        group.create_dataset(
            "codes", data=np.r_[-1, np.zeros(len(refs) - 1)].astype(np.int8)
        )
        group.create_dataset(
            "categories", data=np.asarray(["A"], dtype=h5py.string_dtype())
        )

    with pytest.raises(ValueError, match="'ref' column of .* has missing values"):
        mito_afm_source(edited("missing.h5ad", missing_ref))
    with pytest.raises(ValueError, match="obs of .* has no index of names"):
        mito_afm_source(
            edited(
                "noindex.h5ad", lambda handle: handle["obs"].attrs.__delitem__("_index")
            )
        )
    positions = None

    def zero_position(handle) -> None:
        nonlocal positions
        positions = handle["var/pos"][:]
        values = positions.copy()
        values[0] = 0
        replace(handle, "pos", values)

    with pytest.raises(ValueError, match="holds a position below 1"):
        mito_afm_source(edited("zero.h5ad", zero_position))

    def two_references(handle) -> None:
        values = handle["var/pos"][:]
        values[1] = values[0]
        replace(handle, "pos", values)
        refs = handle["var/ref"][:].astype(str)
        refs[1] = next(base for base in "ACGT" if base != refs[0])
        replace(handle, "ref", refs.astype(h5py.string_dtype()))

    with pytest.raises(ValueError, match="the reference bases"):
        mito_afm_source(edited("tworefs.h5ad", two_references))

    def alt_is_ref(handle) -> None:
        replace(handle, "alt", handle["var/ref"][:])

    with pytest.raises(ValueError, match="whose alternative is its reference base"):
        mito_afm_source(edited("altref.h5ad", alt_is_ref))


def test_text_helpers_refuse_binary_text_and_short_vcf_records(tmp_path) -> None:
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ValueError, match="is not a text file"):
        first_lines(str(binary), 1)
    vcf = tmp_path / "v.vcf"
    vcf.write_text("#CHROM\tPOS\tID\n1\t2\t.\n")
    with pytest.raises(
        ValueError, match="need the columns CHROM, POS, ID, REF and ALT"
    ):
        read_vcf(str(vcf))
    vcf.write_text("##contig=<ID=chrM,length=16569>\n#CHROM\tPOS\tID\tREF\tALT\n")
    records = read_vcf(str(vcf))
    assert len(records.pos) == 0 and records.contig_lengths == {"chrM": 16569}


# Files that tests marked integration download; each skips when its host is
# unreachable. Figshare 31323481 is CC BY 4.0; GEO states no license;
# GoT-Multi-ML's example outputs carry no license, so they are read, never
# checked in.
_FIGSHARE = "https://ndownloader.figshare.com/files/{}"
_GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{}nnn/{}/suppl/{}"
_GOT_MULTI = (
    "https://raw.githubusercontent.com/AnnaNamLab/GoT-Multi-ML/"
    "c4a6721d64c0710e6fc2fa3356f9ad4f55e796e8/example_run/got_multi_ml_run/{}"
)


def _geo(sample: str, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return download_or_skip(_GEO.format(sample[:-3], sample, name), folder / name)


@pytest.mark.integration
def test_the_reference_agrees_with_mgatk_0_9_0_variant_statistics(tmp_path) -> None:
    # Every variant that mgatk 0.9.0 scored in the pL1000 run lies on a site
    # whose reference base is the variant's own, and none at the N.
    folder = tmp_path / "final"
    folder.mkdir()
    download_or_skip(_FIGSHARE.format(61834360), folder / "chrM_refAllele.txt")
    stats = pd.read_csv(
        download_or_skip(_FIGSHARE.format(61834378), tmp_path / "variant_stats.tsv.gz"),
        sep="\t",
    )
    for base in "ACGT":
        (folder / f"pL1000.{base}.txt.gz").write_bytes(gzip.compress(b""))
    sites = mgatk_source(folder, format=None, swap_strands=False).sites()
    reference = dict(
        zip(
            sites.site_position[
                sites.allele_site[sites.allele_role == "reference"]
            ].tolist(),
            sites.allele[sites.allele_role == "reference"].tolist(),
            strict=True,
        )
    )
    assert len(stats) > 1_000
    assert sites.contigs[0].name == "chrM" and sites.contigs[0].length == 16_569
    for position, change in zip(stats["position"], stats["nucleotide"], strict=True):
        assert reference[int(position)] == change[0]
        assert change[2] != change[0]
    assert 3107 not in reference


@pytest.mark.integration
def test_gift_per_umi_tables_on_geo_count_umis_and_reads(tmp_path) -> None:
    folder = tmp_path / "BC8"
    stem = "GSM9714721_MPN_16plex_1_lane1_BC8"
    reads = _geo("GSM9714721", f"{stem}_GAP_probe_reads.tsv.gz", folder)
    barcodes = _geo("GSM9714721", f"{stem}_GAP_barcodes.tsv.gz", folder)
    manifest = _geo("GSM9714721", f"{stem}_BC16_GAP_manifest.tsv.gz", folder)
    cells = pd.read_csv(barcodes, sep="\t")["barcode"].tolist()
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, group = _import(store, selection, folder)
    table = pd.read_csv(reads, sep="\t", keep_default_na=False, dtype={"gapfill": str})
    probes = pd.read_csv(manifest, sep="\t", comment="#", keep_default_na=False)
    name = dict(zip(probes["index"], probes["name"], strict=True))
    grouped = (
        table.groupby(["cell_idx", "probe_idx", "gapfill"])
        .agg(count=("umi", "size"), reads=("pcr_duplicate_count", "sum"))
        .reset_index()
    )
    expected = pd.DataFrame(
        {
            "ids": np.asarray(cells)[grouped["cell_idx"]],
            "site": [name[index] for index in grouped["probe_idx"]],
            "allele": grouped["gapfill"],
            "count": grouped["count"],
            "reads": grouped["reads"],
        }
    )
    assert_tables(loaded.counts, expected)
    roles = loaded.alleles.assign(site=loaded.alleles["site"].astype(str))
    role = {(row.site, row.allele): row.role for row in roles.itertuples()}
    assert role[("JAK2 c.1849G>T", "CAC")] == "reference"
    assert role[("JAK2 c.1849G>T", "AAC")] == "alternative"
    same = store.artifacts.inspect(
        store.artifacts.list(kind="variant_counts", scope="datastore")[0]
    ).inputs["sources"][0]["same_gapfills"]
    assert "PPM1D c.1636_1639" in same
    assert group.attrs["n_unmatched_barcodes"] == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ("sample", "name", "columns"),
    [
        ("GSM3309839", "GSM3309839_ET02.CALR.txt.gz", ("num.WT.call", "num.MUT.call")),
        ("GSM6198716", "GSM6198716_MDS04.SF3B1.tsv.gz", ("WT.calls", "MUT.calls")),
    ],
)
def test_got_tables_on_geo_count_wild_type_and_mutant_umis(
    tmp_path, sample, name, columns
) -> None:
    path = _geo(sample, name, tmp_path / "geo")
    table = pd.read_csv(path, sep="\t")
    cells = table["BC"].tolist() if "BC" in table.columns else table.index.tolist()
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, path)
    site = name.split(".tsv")[0].split(".txt")[0]
    counts = []
    for allele, column in zip(("WT", "MUT"), columns, strict=True):
        values = table[column]
        kept = values.notna() & (values > 0)
        counts.append(
            pd.DataFrame(
                {
                    "ids": np.asarray(cells)[kept.to_numpy()],
                    "site": site,
                    "allele": allele,
                    "count": values[kept].astype(int).to_numpy(),
                }
            )
        )
    assert_tables(loaded.counts, pd.concat(counts))
    assert loaded.cells["listed"].all()
    if "Genotype" in table.columns:
        called = table["Genotype"].notna()
        assert len(loaded.calls) == int(called.sum())


@pytest.mark.integration
def test_got_multi_example_outputs_import_together(tmp_path) -> None:
    stage1 = download_or_skip(
        _GOT_MULTI.format("ironthrone_out_pp.csv"), tmp_path / "ironthrone_out_pp.csv"
    )
    refined = download_or_skip(
        _GOT_MULTI.format("ml_denoise/got_multi_out_refined.csv"),
        tmp_path / "got_multi_out_refined.csv",
    )
    counts_table = pd.read_csv(stage1)
    final_table = pd.read_csv(refined)
    cells = sorted(set(counts_table["BC"]) | set(final_table["BC"]))
    store, selection = _store(tmp_path / "store.zarr", cells)
    loaded, _group = _import(store, selection, [stage1, refined])
    for channel in ("geneseq", "probebc"):
        for allele, prefix in (("WT", "wt"), ("MUT", "mut"), ("ambiguous", "amb")):
            column = counts_table[f"{prefix}_calls_{channel}"]
            kept = column > 0
            expected = pd.DataFrame(
                {
                    "ids": counts_table["BC"][kept],
                    "site": counts_table["target"][kept] + f" ({channel})",
                    "allele": allele,
                    "count": column[kept],
                }
            )
            chosen = loaded.counts[
                loaded.counts["site"].astype(str).str.endswith(f"({channel})")
                & (loaded.counts["allele"] == allele)
            ]
            assert_tables(chosen[["ids", "site", "allele", "count"]], expected)
    finals = loaded.calls[loaded.calls["source"] == "genotype_final"]
    # The refined table repeats two cells' calls at a target, which count once.
    called = final_table.dropna(subset=["genotype_final"])
    assert len(finals) == len(called.drop_duplicates(["BC", "target"]))
    assert _group.attrs["n_repeated_calls"] == int(
        called.duplicated(["BC", "target"]).sum()
    )


def test_giftwrap_tables_read_rows_ending_in_tabs_as_every_table_reads_them(
    tmp_path,
) -> None:
    # A tab after each row's last field gives one field more than the header,
    # as R's row names do, so the table is refused rather than read with its
    # columns shifted; a header that ends in a tab too names an empty column.
    from tests.variant_truth import variant_truth, write_giftwrap_tables

    folder = write_giftwrap_tables(variant_truth(), tmp_path / "gift")
    expected = giftwrap_source(folder)
    path = folder / "GSM0_GAP_barcodes.tsv.gz"
    header, *rows = gzip.decompress(path.read_bytes()).decode().splitlines()
    ended = [f"{row}\t" for row in rows]
    path.write_bytes(gzip.compress(("\n".join([header, *ended]) + "\n").encode()))
    with pytest.raises(ValueError, match="end the header with the delimiter too"):
        giftwrap_source(folder)
    path.write_bytes(
        gzip.compress(("\n".join([f"{header}\t", *ended]) + "\n").encode())
    )
    found = giftwrap_source(folder)
    assert found._barcodes.tolist() == expected._barcodes.tolist()
    # A row of another width than the first is refused by its line.
    path.write_bytes(
        gzip.compress(("\n".join([header, rows[0], rows[1] + "\tx"]) + "\n").encode())
    )
    with pytest.raises(ValueError, match="is malformed: line 3 holds"):
        giftwrap_source(folder)


def test_got_tables_read_every_missing_value_token_as_not_covered(tmp_path) -> None:
    # GoT's tables read missing values as every table read by its column
    # names does: NULL or N/A counts read as NA counts do.
    path = tmp_path / "t.tsv"
    path.write_text("BC\twtUMIs\tmutUMIs\nA\t1\tNA\nB\tNA\t2\n")
    expected = list(got_source(path, format="got").batches(10))
    for token in ("NULL", "N/A", "None", "null", ""):
        path.write_text(f"BC\twtUMIs\tmutUMIs\nA\t1\t{token}\nB\t{token}\t2\n")
        found = list(got_source(path, format="got").batches(10))
        assert len(found) == len(expected)
        for batch, wanted in zip(found, expected, strict=True):
            assert batch.kind == wanted.kind
            assert np.array_equal(batch.barcode_codes, wanted.barcode_codes)
            for name in ("keys", "values"):
                left, right = getattr(batch, name), getattr(wanted, name)
                assert (left is None) == (right is None)
                if left is not None:
                    assert np.array_equal(left, right)
