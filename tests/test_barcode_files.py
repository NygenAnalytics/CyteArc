"""Lineage barcodes imported from files that the tools published.

pycashier's reference outputs (MIT), with and without their header, and the
molecule tables of 31 cells of TREX's expected outputs (MIT), in the current
format, corrected, and as TREX wrote them in 2022, are checked in under
``tests/data/barcodes/``. No licensed per-read CellTag file exists, so its
rows are written here in CellTag-multi's layout. BARtab's counts and
annotation of GEO GSM9076301 (301 and 194 KB) are downloaded; that test skips
when GEO is unreachable or cuts a file short. Every expectation is counted
with pandas from the raw files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.network_helpers import download_or_skip
from tests.storage_helpers import open_store, write_count_store

DATA = Path(__file__).parent / "data" / "barcodes"
_BARTAB = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9076nnn/GSM9076301/suppl/"
    "GSM9076301_SG05-Capture-1-Rep1_S1{}"
)


def _store(path: Path, cell_ids: list[str]):
    counts = np.ones((len(cell_ids), 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=cell_ids)
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store, store.snapshot_cell_selection("every_cell")


def _records(store, ref) -> pd.DataFrame:
    records = store.clones.load_barcodes(ref).records
    return pd.DataFrame(
        {
            "cell": records["ids"].astype(str).to_numpy(dtype=object),
            "barcode": records["barcode"].astype(str).to_numpy(dtype=object),
            "umi": pd.Series(
                [None if pd.isna(umi) else str(umi) for umi in records["umi"]],
                dtype=object,
            ),
            "umis": records["umis"].fillna(-1).to_numpy(dtype=np.int64),
            "reads": records["reads"].fillna(-1).to_numpy(dtype=np.int64),
        }
    )


@pytest.mark.parametrize(
    "name",
    [
        "pycashier_test.umi_cell_labeled.barcode.tsv",
        "pycashier_test.cell_record_labeled.barcode.tsv",
    ],
)
def test_pycashiers_reads_join_10x_cells_through_a_suffix(tmp_path, name) -> None:
    path = DATA / name
    with_header = "umi_cell_labeled" in name
    table = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        header=0 if with_header else None,
        names=None if with_header else ["info", "umi", "cell", "barcode"],
    )
    cell_ids = sorted(f"{cell}-1" for cell in table["cell"].unique())
    store, cells = _store(tmp_path / "store.zarr", cell_ids)

    ref = store.clones.import_barcodes(path, cell_selection=cells, barcode_suffix="-1")

    expected = (
        table.groupby(["cell", "umi", "barcode"])
        .size()
        .rename("reads")
        .reset_index()
        .assign(cell=lambda frame: frame["cell"] + "-1")
    )
    records = _records(store, ref)
    assert sorted(
        zip(
            records["cell"],
            records["umi"],
            records["barcode"],
            records["reads"],
            strict=True,
        )
    ) == sorted(
        zip(
            expected["cell"],
            expected["umi"],
            expected["barcode"],
            expected["reads"],
            strict=True,
        )
    )
    # Partial barcodes of 11 and 19 bases are kept as the file holds them.
    assert sorted(len(barcode) for barcode in records["barcode"]) == [
        11,
        19,
        20,
        20,
        20,
    ]
    sources = store.artifacts.inspect(ref).inputs["sources"]
    assert sources[0]["format"] == "pycashier"


@pytest.mark.parametrize(
    ("name", "barcode", "reads"),
    [
        ("trex_molecules.txt", "clone_id", True),
        ("trex_molecules_corrected.txt", "clone_id", True),
        ("trex_molecules_2022.txt", "clone_id", False),
    ],
)
def test_trex_molecules_of_every_release_keep_partial_clone_ids(
    tmp_path, name, barcode, reads
) -> None:
    path = DATA / name
    table = pd.read_csv(path, sep="\t", dtype=str)
    table.columns = [column.lstrip("#") for column in table.columns]
    cell_ids = sorted(f"{cell}-1" for cell in table["cell_id"].unique())
    store, cells = _store(tmp_path / "store.zarr", cell_ids)

    ref = store.clones.import_barcodes(path, cell_selection=cells, barcode_suffix="-1")

    records = _records(store, ref)
    expected_reads = (
        table["read_count"].astype(int).tolist() if reads else [-1] * len(table)
    )
    assert sorted(
        zip(
            records["cell"],
            records["umi"],
            records["barcode"],
            records["reads"],
            strict=True,
        )
    ) == sorted(
        zip(
            table["cell_id"] + "-1",
            table["umi"],
            table[barcode],
            expected_reads,
            strict=True,
        )
    )
    # cloneIDs hold "-" where TREX covered no base, and each cell's UMIs
    # are its molecules.
    assert any("-" in value for value in table[barcode]) == any(
        "-" in value for value in records["barcode"]
    )
    loaded = store.clones.load_barcodes(ref)
    per_cell = table.groupby("cell_id").size()
    assert loaded.cells["n_umis"].to_dict() == {
        f"{cell}-1": int(count) for cell, count in per_cell.items()
    }
    assert loaded.cells["n_barcodes"].to_dict() == {
        f"{cell}-1": int(count)
        for cell, count in table.groupby("cell_id")[barcode].nunique().items()
    }


def test_trex_corrected_molecules_hold_the_corrected_clone_ids(tmp_path) -> None:
    table = pd.read_csv(DATA / "trex_molecules_corrected.txt", sep="\t", dtype=str)
    # Error correction changed some cloneIDs.
    assert (table["original_clone_id"] != table["clone_id"]).any()
    store, cells = _store(
        tmp_path / "store.zarr", sorted(table["#cell_id"].unique().tolist())
    )
    ref = store.clones.import_barcodes(
        DATA / "trex_molecules_corrected.txt", cell_selection=cells
    )
    records = _records(store, ref)
    assert set(records["barcode"]) == set(table["clone_id"])
    assert not set(records["barcode"]) & (
        set(table["original_clone_id"]) - set(table["clone_id"])
    )


# CellTag-multi's per-read rows: three reads of one molecule, two of another
# CellTag in the same cell, one read of a cell outside the store, and the
# same cells' scATAC-seq reads, which have no UMI.
_CELLTAG_RNA = """Cell.BC\tUMI\tCell.Tag
ACGTTGCAACGTTGCA-1\tAACCGGTTAACC\tAAAACCCCGGGGTTTTAAAACCCCGGGG
ACGTTGCAACGTTGCA-1\tAACCGGTTAACC\tAAAACCCCGGGGTTTTAAAACCCCGGGG
GGATCCAAGGATCCAA-1\tCCAATTGGCCAA\tAAAACCCCGGGGTTTTAAAACCCCGGGG
ACGTTGCAACGTTGCA-1\tGTGTGTACACAC\tTTTTGGGGCCCCAAAATTTTGGGGCCCC
ACGTTGCAACGTTGCA-1\tAACCGGTTAACC\tAAAACCCCGGGGTTTTAAAACCCCGGGG
ACGTTGCAACGTTGCA-1\tGTGTGTACACAC\tTTTTGGGGCCCCAAAATTTTGGGGCCCC
TTTTCCCCGGGGAAAA-1\tTATATAGCGCGC\tTTTTGGGGCCCCAAAATTTTGGGGCCCC
"""
_CELLTAG_ATAC = """Cell.BC\tCell.Tag
ACGTTGCAACGTTGCA-1\tAAAACCCCGGGGTTTTAAAACCCCGGGG
ACGTTGCAACGTTGCA-1\tAAAACCCCGGGGTTTTAAAACCCCGGGG
GGATCCAAGGATCCAA-1\tTTTTGGGGCCCCAAAATTTTGGGGCCCC
"""


def test_celltag_multi_reads_collapse_into_molecules_and_atac_into_pairs(
    tmp_path,
) -> None:
    rna = tmp_path / "parse_rna.txt"
    rna.write_text(_CELLTAG_RNA)
    atac = tmp_path / "parse_atac.txt"
    atac.write_text(_CELLTAG_ATAC)
    store, cells = _store(
        tmp_path / "store.zarr", ["ACGTTGCAACGTTGCA-1", "GGATCCAAGGATCCAA-1"]
    )

    with pytest.warns(UserWarning, match="1 of 3 barcodes of the barcode files"):
        ref = store.clones.import_barcodes(
            [rna, atac], cell_selection=cells, library=["RNA", "ATAC"]
        )

    tag_a = "AAAACCCCGGGGTTTTAAAACCCCGGGG"
    tag_b = "TTTTGGGGCCCCAAAATTTTGGGGCCCC"
    records = _records(store, ref)
    assert records.values.tolist() == [
        ["ACGTTGCAACGTTGCA-1", tag_a, None, -1, 2],
        ["ACGTTGCAACGTTGCA-1", tag_a, "AACCGGTTAACC", 1, 3],
        ["ACGTTGCAACGTTGCA-1", tag_b, "GTGTGTACACAC", 1, 2],
        ["GGATCCAAGGATCCAA-1", tag_b, None, -1, 1],
        ["GGATCCAAGGATCCAA-1", tag_a, "CCAATTGGCCAA", 1, 1],
    ]
    loaded = store.clones.load_barcodes(ref)
    assert loaded.records["library"].astype(str).tolist() == [
        "ATAC",
        "RNA",
        "RNA",
        "ATAC",
        "RNA",
    ]
    assert loaded.cells["n_barcodes"].tolist() == [2, 2]


def _download(name: str, folder: Path) -> Path:
    """Download one of BARtab's files, or skip when GEO cannot serve it whole."""
    return download_or_skip(_BARTAB.format(name), folder / f"GSM9076301{name}")


@pytest.mark.integration
def test_bartabs_counts_and_annotation_match_counts_taken_from_the_raw_files(
    tmp_path,
) -> None:
    counts_path = _download(".counts.tsv.gz", tmp_path)
    annotation_path = _download("_cell_barcode_annotation.tsv.gz", tmp_path)
    counts = pd.read_csv(counts_path, sep="\t", dtype={"count": np.int64})
    annotation = pd.read_csv(annotation_path, sep="\t", dtype=str)
    assert len(counts) == 40_582
    assert counts["gene"].nunique() == 419
    cell_ids = sorted(set(counts["cell"]) | set(annotation.index))
    store, cells = _store(tmp_path / "store.zarr", cell_ids)

    counted = store.clones.import_barcodes(counts_path, cell_selection=cells)
    annotated = store.clones.import_barcodes(annotation_path, cell_selection=cells)

    records = _records(store, counted)
    assert sorted(
        zip(records["cell"], records["barcode"], records["umis"], strict=True)
    ) == sorted(zip(counts["cell"], counts["gene"], counts["count"], strict=True))
    # BARtab's annotation keeps each cell's barcodes with at least 0.3 times
    # the UMIs of its top barcode, as its README defines the filter.
    top = counts.groupby("cell")["count"].transform("max")
    kept = counts[counts["count"] >= 0.3 * top]
    annotated_records = _records(store, annotated)
    assert sorted(
        zip(
            annotated_records["cell"],
            annotated_records["barcode"],
            annotated_records["umis"],
            strict=True,
        )
    ) == sorted(zip(kept["cell"], kept["gene"], kept["count"], strict=True))
    loaded = store.clones.load_barcodes(annotated)
    assert int((loaded.cells["n_barcodes"] > 0).sum()) == 18_850
