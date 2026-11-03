"""Every reader reads a file by its bytes, whatever its name says.

A gzip file whose name lacks ``.gz`` and a plain file named ``.gz`` read as
the file itself does, for each reader of the stack and for master's readers
of Matrix Market folders, gene references, Cell Ranger's call file and gene
annotations, which all open their files through ``cytearc.utils.files``. A
byte-order mark at the start of a file's text reads the same in every
format: as no part of the text.
"""

import gzip
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
KINDS = ["plain named .gz", "gzip without .gz", "with a byte-order mark"]


def _plain_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def _variant(source: Path, folder: Path, kind: str) -> Path:
    """Copy a file, or a folder's files, plain under ``.gz`` names,
    gzip-compressed under names without it, or as they are with a byte-order
    mark before their text."""
    folder.mkdir()
    files = sorted(source.iterdir()) if source.is_dir() else [source]
    for path in files:
        data = _plain_bytes(path)
        bare = path.name.removesuffix(".gz")
        if kind == "plain named .gz":
            (folder / f"{bare}.gz").write_bytes(data)
        elif kind == "gzip without .gz":
            (folder / bare).write_bytes(gzip.compress(data, mtime=0))
        else:
            marked = b"\xef\xbb\xbf" + data
            compressed = path.read_bytes()[:2] == b"\x1f\x8b"
            (folder / path.name).write_bytes(
                gzip.compress(marked, mtime=0) if compressed else marked
            )
    if source.is_dir():
        return folder
    return next(folder.iterdir())


def _clone_rows(path: Path) -> Any:
    from cytearc.readers.clone_tables import clone_layout, iter_clone_rows

    return [
        (rows.barcodes.tolist(), rows.clones.tolist())
        for rows in iter_clone_rows(clone_layout(path), 1_000)
    ]


def _barcode_rows(path: Path) -> Any:
    from cytearc.readers.lineage_barcodes import barcode_layout, iter_barcode_rows

    return [
        (rows.cells.tolist(), rows.barcodes.tolist())
        for rows in iter_barcode_rows(barcode_layout(path, library="lib"), 1_000)
    ]


def _allele_rows(path: Path) -> Any:
    from cytearc.readers.recorder_alleles import allele_layout, iter_allele_rows

    return [
        (rows.barcodes.tolist(), rows.tokens.tolist(), rows.rows.tolist())
        for rows in iter_allele_rows(allele_layout(path), 1_000)
    ]


def _batches(source: Any) -> Any:
    found = []
    for batch in source.batches(1_000):
        found.append(
            (
                batch.kind,
                np.asarray(batch.barcodes).tolist(),
                None if batch.keys is None else np.asarray(batch.keys).tolist(),
                None if batch.values is None else np.asarray(batch.values).tolist(),
            )
        )
    return found


def _got(path: Path) -> Any:
    from cytearc.readers.got import got_source

    return _batches(got_source(path, format="got"))


def _cellsnp(path: Path) -> Any:
    from cytearc.readers.cellsnp import cellsnp_source

    return _batches(cellsnp_source(path, format="cellsnp_lite"))


def _vartrix(path: Path) -> Any:
    from cytearc.readers.vartrix import vartrix_source

    return _batches(vartrix_source(path))


def _mgatk(path: Path) -> Any:
    from cytearc.readers.mgatk import mgatk_source

    return _batches(mgatk_source(path, format="mgatk", swap_strands=False))


def _trees(path: Path) -> Any:
    from cytearc.readers.phylogeny import read_trees, tree_file_layout

    trees = read_trees(tree_file_layout(path), memory_bytes=1 << 30)
    return trees.names.tolist(), trees.parent.tolist()


# Each reader and a real file or folder of its format.
READERS: dict[str, tuple[Callable[[Path], Any], Path]] = {
    "clone table": (_clone_rows, DATA / "clones" / "trex_clones.txt"),
    "barcode table": (_barcode_rows, DATA / "barcodes" / "trex_molecules.txt"),
    "allele table": (
        _allele_rows,
        DATA / "recorders" / "petracer_barcoded_tracing_alleles.csv",
    ),
    "GoT table": (_got, DATA / "variants" / "got" / "Patient1_TET2.2847_summTable.txt"),
    "cellsnp-lite folder": (_cellsnp, DATA / "variants" / "cellsnp_lite"),
    "VarTrix folder": (_vartrix, DATA / "variants" / "vartrix"),
    "mgatk folder": (_mgatk, DATA / "variants" / "signac_test_mgatk"),
    "Newick tree": (_trees, DATA / "phylogenies" / "convexml_tree_0.nwk"),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reader", sorted(READERS))
def test_every_reader_reads_a_file_by_its_bytes(tmp_path, reader, kind) -> None:
    read, source = READERS[reader]
    original = tmp_path / "original"
    if source.is_dir():
        shutil.copytree(source, original)
    else:
        original.mkdir()
        original = Path(shutil.copy(source, original))
    expected = read(original)
    assert expected
    assert read(_variant(original, tmp_path / "variant", kind)) == expected


_CONTIGS = {
    "airr": (
        "cell_id\tsequence_id\tlocus\tv_call\tj_call\tjunction_aa\tproductive\n"
        "c1\tc1_1\tTRB\tTRBV1\tTRBJ1\tCASS\tT\nc2\tc2_1\tTRA\tTRAV1\tTRAJ1\tCAVF\tF\n"
    ),
    "cellranger": (
        "barcode,contig_id,chain,cdr3,productive\n"
        "AAAC-1,AAAC-1_contig_1,TRB,CASS,true\nAAAG-1,AAAG-1_contig_1,TRA,CAVF,false\n"
    ),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("format", sorted(_CONTIGS))
def test_contig_readers_read_a_file_by_its_bytes(tmp_path, format, kind) -> None:
    from cytearc.readers.airr import iter_airr_rearrangements
    from cytearc.readers.cellranger import iter_vdj_contigs

    read = iter_airr_rearrangements if format == "airr" else iter_vdj_contigs
    original = tmp_path / "contigs.tsv"
    original.write_text(_CONTIGS[format])

    def contigs(path: Path) -> list:
        return [chunk.to_dict("list") for _lines, chunk in read(path)]

    variant = _variant(original, tmp_path / "variant", kind)
    assert contigs(variant) == contigs(original) != []


@pytest.mark.parametrize("kind", KINDS)
def test_masters_matrix_market_folders_read_their_files_by_their_bytes(
    tmp_path, kind
) -> None:
    import zarr
    from zarr.storage import MemoryStore

    from cytearc.readers import MtxReader, inspect_mtx
    from cytearc.storage.count_matrix import CountMatrixPolicy
    from cytearc.writers import MtxToZarr

    folder = tmp_path / "original"
    folder.mkdir()
    (folder / "matrix.mtx").write_text(
        "%%MatrixMarket matrix coordinate integer general\n2 3 3\n1 1 4\n2 2 5\n1 3 6\n"
    )
    (folder / "features.tsv").write_text(
        "g0\tGene0\tGene Expression\ng1\tGene1\tGene Expression\n"
    )
    (folder / "barcodes.tsv").write_text("c0\nc1\nc2\n")

    def counts(path: Path) -> tuple:
        reader = MtxReader(inspect_mtx(path)[0])
        store = MemoryStore()
        try:
            MtxToZarr(
                reader,
                store,
                mem_budget="64M",
                nthreads=1,
                policy=CountMatrixPolicy(unitBytes=16, chunkBytes=8),
            ).dump()
            group = zarr.open_group(store=store, mode="r")
            return (
                group["cellData/ids"][:].tolist(),
                group["RNA/featureData/ids"][:].tolist(),
                np.asarray(group["RNA/counts"][:]).tolist(),
            )
        finally:
            reader.close()

    variant = _variant(folder, tmp_path / "variant", kind)
    assert counts(variant) == counts(folder)


@pytest.mark.parametrize("kind", KINDS)
def test_masters_gene_annotations_read_their_files_by_their_bytes(
    tmp_path, kind
) -> None:
    from cytearc.features.genomic.annotation import read_gene_annotation

    original = tmp_path / "genes.gtf"
    original.write_text(
        "#!genome-build GRCh38\n"
        'chr1\tsrc\tgene\t11\t20\t.\t+\t.\tgene_id "g1"; gene_name "A1";\n'
        'chr1\tsrc\tgene\t31\t40\t.\t-\t.\tgene_id "g2"; gene_name "B2";\n'
    )
    variant = _variant(original, tmp_path / "variant", kind)
    expected = read_gene_annotation(original)
    assert expected["gene_id"].tolist() == ["g1", "g2"]
    assert read_gene_annotation(variant).equals(expected)


@pytest.mark.parametrize("kind", KINDS)
def test_masters_call_file_reads_by_its_bytes(tmp_path, kind) -> None:
    from cytearc.readers.cellranger import (
        protospacer_calls_size,
        read_protospacer_calls,
    )

    original = tmp_path / "protospacer_calls_per_cell.csv"
    original.write_text(
        "cell_barcode,num_features,feature_call,num_umis\n"
        "AAAC-1,1,g1,5\nAAAG-1,2,g1|g2,3|4\n"
    )
    variant = _variant(original, tmp_path / "variant", kind)
    expected = read_protospacer_calls(original)
    found = read_protospacer_calls(variant)
    assert found.barcodes.tolist() == expected.barcodes.tolist() == ["AAAC-1", "AAAG-1"]
    assert found.umis.tolist() == expected.umis.tolist()
    # The size bounds the memory of parsing, so it counts a mark's bytes.
    size = protospacer_calls_size(variant)
    original_size = protospacer_calls_size(original)
    marked = 3 if kind == "with a byte-order mark" else 0
    assert size.n_bytes == original_size.n_bytes + marked
    assert (size.n_rows, size.n_pipes, size.most_calls) == (
        original_size.n_rows,
        original_size.n_pipes,
        original_size.most_calls,
    )


def test_masters_gene_reference_reads_a_plain_download_named_gz(
    tmp_path, monkeypatch
) -> None:
    import cytearc.features.gene_reference as reference

    text = (
        "##gff-version 3\n"
        "1\tensembl\tgene\t11\t20\t.\t+\t.\tID=gene:ENSG01;Name=A1;biotype=x\n"
        "MT\tensembl\tgene\t31\t40\t.\t-\t.\tID=gene:ENSG02;Name=MT-B2\n"
    )

    def listing(url: str, *, timeout: float) -> str:
        return '<a href="Homo_sapiens.GRCh38.115.gff3.gz">current</a>'

    def download(url: str, destination: Path, *, timeout: float) -> None:
        # Ensembl's name ends in .gz, and a mirror may serve the text plain.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)

    monkeypatch.setattr(reference, "_directory_listing", listing)
    monkeypatch.setattr(reference, "_download_gff3", download)
    found = reference.ensure_reference("homo_sapiens", cacheDir=tmp_path / "plain")
    assert found.geneId == ("ENSG01", "ENSG02")
    assert found.chromosome == ("1", "MT")

    def compressed(url: str, destination: Path, *, timeout: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(gzip.compress(text.encode(), mtime=0))

    monkeypatch.setattr(reference, "_download_gff3", compressed)
    again = reference.ensure_reference("homo_sapiens", cacheDir=tmp_path / "gzip")
    assert again.geneId == found.geneId and again.symbol == found.symbol
