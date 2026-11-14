"""Repertoire operations on the shapes that stress them give their results.

The shapes are clonotypes of one cell each with two chains of each arm,
beta chains of 300 characters beyond the BMP, junctions of 302 ASCII residues
whose nucleotides fill many stored chunks, and AIRR exports from contigs
stored in small chunks: Cell Ranger files of T and B cells whose contig IDs
repeat across the files, with gene calls that hold @, # and a double quote,
cells of 78-digit IDs, and AIRR files of sequences and alignments.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import cytearc.datastore._operations.record_tables as record_tables
from cytearc.datastore.datastore import DataStore
from cytearc.storage.refs import ArtifactRef
from tests.clone_helpers import CELLRANGER_HEADER, write_clone_contigs, write_lines
from tests.storage_helpers import open_store, write_count_store

pytestmark = pytest.mark.slow


# Clonotypes of one cell each, as in most T cell repertoires, of more cells
# than one chunk of cell IDs holds.
N_SINGLETONS = 50_000
# Clonotypes of one cell each, named by long junctions of wide characters.
N_LABELED = 5_000
# Cells of long beta chains, whose junctions span many stored chunks.
N_LONG_JUNCTIONS = 20_000


@pytest.fixture(scope="module")
def singletons(tmp_path_factory) -> SimpleNamespace:
    """Clonotypes of one cell each, every cell with two chains of each arm."""
    path = tmp_path_factory.mktemp("clonotype_singletons") / "store.zarr"
    ids = np.asarray([f"c{i}" for i in range(N_SINGLETONS)])
    counts = np.ones((N_SINGLETONS, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=ids)
    store = open_store(path)
    contigs = write_clone_contigs(
        path.parent,
        "contigs.csv",
        ids,
        np.arange(N_SINGLETONS),
        chains_per_arm=2,
    )
    every = store.snapshot_cell_selection()
    clonotypes = store.repertoire.clonotypes(
        store.repertoire.import_receptors(contigs, cell_selection=every)
    )
    return SimpleNamespace(store=store, clonotypes=clonotypes)


def test_loaded_clonotypes_of_one_cell_each_decode_every_chain(singletons) -> None:
    store, clonotypes = singletons.store, singletons.clonotypes
    loaded = store.repertoire.load_clonotypes(clonotypes)
    assert len(loaded.clonotypes) == N_SINGLETONS
    # Each of the 50,000 clonotypes has four chains to decode, so their
    # values pass through several batches.
    assert loaded.clonotypes["VJ_2_junction_aa"].notna().all()


@pytest.fixture(scope="module")
def wide_junctions(tmp_path_factory) -> SimpleNamespace:
    """Clonotypes named by beta chains of 300 characters beyond the BMP.

    Each cell is a clonotype of its own chains, so every label decodes a
    junction of its own. Each character takes four bytes as stored and four
    in a Python string.
    """
    rng = np.random.default_rng(17)
    path = tmp_path_factory.mktemp("wide_junctions") / "store.zarr"
    ids = [f"cell{i:06d}-1" for i in range(N_LABELED)]
    counts = np.ones((N_LABELED, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=np.asarray(ids))
    store = open_store(path)
    wide = np.asarray([chr(0x1F600 + k) for k in range(20)])
    residues = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    rows = []
    for cell, name in enumerate(ids):
        alpha = "C" + "".join(rng.choice(residues, size=12)) + "F"
        beta = "C" + "".join(rng.choice(wide, size=300)) + "F"
        chains = (("TRA", alpha), ("TRB", beta))
        for contig, (locus, junction) in enumerate(chains):
            rows.append(
                f"{name},true,{name}_contig_{contig + 1},true,500,{locus},"
                f"{locus}V1,,{locus}J1,{locus}C,true,true,{junction},,100,9,"
                f"ct{cell},ct{cell}_{contig + 1}"
            )
    contigs = write_lines(path.parent, "contigs.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=store.snapshot_cell_selection()
    )
    clonotypes = store.repertoire.clonotypes(receptors, sequence="aa")
    return SimpleNamespace(store=store, clonotypes=clonotypes)


def test_labels_of_long_wide_junctions_name_each_clonotype(wide_junctions) -> None:
    store, clonotypes = wide_junctions.store, wide_junctions.clonotypes
    ids = [f"clonotype_{rank}" for rank in range(1, N_LABELED + 1)]
    labels = store._clone_figure_labels(clonotypes, ids).tolist()
    assert len(set(labels)) == N_LABELED
    assert {len(label) for label in labels} == {302}


def test_lengths_of_long_wide_junctions_count_their_characters(wide_junctions) -> None:
    store, clonotypes = wide_junctions.store, wide_junctions.clonotypes
    lengths = store.repertoire.spectratype(clonotypes).table
    # Every beta chain's 302 characters take four bytes each.
    assert dict(zip(lengths["locus"], lengths["length"], strict=True)) == {
        "TRA": 14,
        "TRB": 302,
    }


@pytest.fixture(scope="module")
def long_junctions(tmp_path_factory) -> SimpleNamespace:
    """Cells of beta chains of 302 residues, whose nucleotides span many chunks.

    The junctions are ASCII, and their 18 MB of nucleotides fill nine stored
    chunks.
    """
    rng = np.random.default_rng(19)
    path = tmp_path_factory.mktemp("long_junctions") / "store.zarr"
    ids = [f"cell{i:06d}-1" for i in range(N_LONG_JUNCTIONS)]
    counts = np.ones((N_LONG_JUNCTIONS, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=np.asarray(ids))
    store = open_store(path)
    residues = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    bases = np.asarray(list("ACGT"))
    rows = []
    for cell, name in enumerate(ids):
        for contig, (locus, length) in enumerate((("TRA", 14), ("TRB", 302))):
            residue_text = "".join(rng.choice(residues, size=length))
            base_text = "".join(rng.choice(bases, size=3 * length))
            rows.append(
                f"{name},true,{name}_contig_{contig + 1},true,500,{locus},{locus}V1,,"
                f"{locus}J1,{locus}C,true,true,{residue_text},{base_text},100,9,"
                f"ct{cell},ct{cell}_{contig + 1}"
            )
    contigs = write_lines(path.parent, "contigs.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=store.snapshot_cell_selection()
    )
    return SimpleNamespace(store=store, receptors=receptors)


@pytest.mark.parametrize("sequence", ["aa", "nt"])
def test_lengths_of_long_ascii_junctions_read_over_many_chunks(
    long_junctions, sequence
) -> None:
    store, receptors = long_junctions.store, long_junctions.receptors
    lengths = store.repertoire.spectratype(receptors, sequence=sequence).table
    factor = 1 if sequence == "aa" else 3
    assert dict(zip(lengths["locus"], lengths["length"], strict=True)) == {
        "TRA": 14 * factor,
        "TRB": 302 * factor,
    }


# AIRR exports, of contigs stored in chunks of 64 KiB, so that every pass
# crosses stored chunks and their windows move: Cell Ranger files of T and B
# cells whose contig IDs repeat across the files, with gene calls that hold @,
# # and a double quote, of many clonotypes; cells of 78-digit IDs; and AIRR
# files of sequences and alignments, with identities that are not whole.
N_EXPORT_CELLS = 20_000
N_SEQUENCE_CELLS = 8_000
_EXPORT_CHUNK_BYTES = 64 * 2**10


def _export_store(folder: Path, ids: list[str]) -> DataStore:
    path = folder / "store.zarr"
    counts = np.ones((len(ids), 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=np.asarray(ids))
    return open_store(path)


def _import_in_small_chunks(store: DataStore, files: list[Path]) -> ArtifactRef:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(record_tables, "_RECORD_CHUNK_BYTES", _EXPORT_CHUNK_BYTES)
        return store.repertoire.import_receptors(
            files, cell_selection=store.snapshot_cell_selection()
        )


def _cellranger_export(folder: Path, *, id_width: int, both: bool) -> SimpleNamespace:
    rng = np.random.default_rng(id_width)
    ids = [f"c{index:0{id_width}d}-1" for index in range(N_EXPORT_CELLS)]
    store = _export_store(folder, ids)
    residues = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    tcr, bcr = [], []
    for cell, name in enumerate(ids):
        clone = cell // 3
        # Each clone's own junctions, so cells of a clone share a clonotype.
        bases = "".join("ACGT"[(clone >> (2 * k)) & 3] for k in range(10))
        v_call = (
            f'"TRAV{clone % 40}@""B#"'
            if both and cell % 50 == 0
            else f"TRAV{clone % 40}"
        )
        for contig, locus in enumerate(("TRA", "TRB")):
            junction = "C" + "".join(rng.choice(residues, size=12)) + "F"
            tcr.append(
                f"{name},true,{name}_contig_{contig + 1},true,500,{locus},"
                f"{v_call if locus == 'TRA' else 'TRBV1'},,{locus}J1,{locus}C,true,true,"
                f"{junction},TGT{bases}{contig}TTC,100,9,ct{clone},ct{clone}_{contig + 1}"
            )
        if both:
            bcr.append(
                f"{name},true,{name}_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,"
                f"{'false' if cell % 7 == 0 else 'true'},CARW,TGT{bases}GTTC,100,9,"
                f"cb{clone},cb{clone}_1"
            )
    files = [write_lines(folder, "tcr.csv", CELLRANGER_HEADER, tcr)]
    if bcr:
        files.append(write_lines(folder, "bcr.csv", CELLRANGER_HEADER, bcr))
    receptors = _import_in_small_chunks(store, files)
    return SimpleNamespace(
        store=store,
        receptors=receptors,
        clonotypes=store.repertoire.clonotypes(receptors),
    )


@pytest.fixture(scope="module")
def cellranger_export(tmp_path_factory) -> SimpleNamespace:
    """T and B cells whose contig_1 IDs repeat, some gene calls of @, # and quotes."""
    return _cellranger_export(
        tmp_path_factory.mktemp("cellranger_export"), id_width=12, both=True
    )


@pytest.fixture(scope="module")
def long_id_export(tmp_path_factory) -> SimpleNamespace:
    """T cells of 81-character IDs."""
    return _cellranger_export(
        tmp_path_factory.mktemp("long_id_export"), id_width=78, both=False
    )


@pytest.fixture(scope="module")
def sequence_export(tmp_path_factory) -> SimpleNamespace:
    """B cells of AIRR contigs with sequences, alignments and identities."""
    folder = tmp_path_factory.mktemp("sequence_export")
    rng = np.random.default_rng(23)
    ids = [f"c{index:012d}-1" for index in range(N_SEQUENCE_CELLS)]
    store = _export_store(folder, ids)
    letters = np.asarray(list("ACGT"))[rng.integers(0, 4, size=2 * len(ids) * 1460)]
    text = "".join(letters.tolist())
    header = (
        "cell_id\tsequence_id\tlocus\tv_call\td_call\tj_call\tc_call\tjunction"
        "\tjunction_aa\tproductive\tumi_count\tv_identity\tsequence"
        "\tsequence_alignment\tgermline_alignment\tv_cigar\tj_cigar"
    )
    rows = []
    for cell, name in enumerate(ids):
        for contig, locus in enumerate(("IGH", "IGK")):
            at = (2 * cell + contig) * 1460
            bases = "".join("ACGT"[(cell >> (2 * k)) & 3] for k in range(9))
            rows.append(
                f"{name}\t{name}_contig_{contig + 1}\t{locus}\t{locus}V1*01\t\t"
                f"{locus}J1*01\t{locus}C\tTGT{bases}{contig}\tCARW\tT\t{cell % 9 + 1}"
                f"\t{0.85 + (cell % 13) / 100}\t{text[at : at + 500]}"
                f"\t{text[at + 500 : at + 980]}\t{text[at + 980 : at + 1460]}"
                "\t12S468M\t3S40M"
            )
    receptors = _import_in_small_chunks(
        store, [write_lines(folder, "airr.tsv", header, rows)]
    )
    return SimpleNamespace(
        store=store,
        receptors=receptors,
        clonotypes=store.repertoire.clonotypes(receptors),
    )


EXPORT_CASES = {
    "repeated IDs and calls of @, # and quotes": ("cellranger_export", "receptors", {}),
    "clonotypes of repeated IDs": ("cellranger_export", "clonotypes", {}),
    "chains of repeated IDs, compressed": (
        "cellranger_export",
        "receptors",
        {"chains_only": True, "suffix": ".tsv.gz"},
    ),
    "long cell IDs": ("long_id_export", "clonotypes", {}),
    "sequences and alignments": ("sequence_export", "receptors", {}),
    "chains of sequences": ("sequence_export", "clonotypes", {"chains_only": True}),
}


@pytest.mark.parametrize("case", list(EXPORT_CASES))
def test_airr_exports_across_stored_chunks_name_each_contig_once(
    request, tmp_path, case
) -> None:
    fixture, source, options = EXPORT_CASES[case]
    data = request.getfixturevalue(fixture)
    options = dict(options)
    path = tmp_path / f"export{options.pop('suffix', '.tsv')}"
    data.store.repertoire.export_airr(getattr(data, source), path, **options)
    written = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE
    )
    assert written["sequence_id"].is_unique
