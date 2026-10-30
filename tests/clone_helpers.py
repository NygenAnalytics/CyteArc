"""Stores and checks shared by the tests of clones."""

import shutil
from collections.abc import Mapping, Sequence
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.clone_truth import (
    N_CELLS,
    SAMPLES,
    CloneTruth,
    ReceptorTruth,
    write_cellranger_contigs,
    write_cospar,
)
from tests.storage_helpers import write_count_store

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore
    from cytearc.storage.refs import ArtifactRef


def open_clone_store(path: Path, **options: object) -> "DataStore":
    """Open a store of the truth's cells on one thread."""
    from cytearc.datastore.datastore import DataStore

    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        **{"nthreads": 1, **options},  # type: ignore[arg-type]
    )


def write_truth_store(path: Path, truth: CloneTruth) -> "DataStore":
    """Write a store whose cells are the truth's, with their sample, time and type."""
    counts = np.random.default_rng(5).poisson(3.0, size=(N_CELLS, 6)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=truth.cell_ids)
    store = open_clone_store(path)
    store.cells.insert("sample", truth.samples.astype(object))
    store.cells.insert("time_point", truth.time_points)
    store.cells.insert("cell_type", truth.cell_types.astype(object))
    store.cells.insert("every_cell", np.ones(N_CELLS, dtype=bool))
    return store


def write_cells_store(path: Path, cell_ids: Sequence[str]) -> "DataStore":
    """Write a store of these cells, each with one count of three features.

    The ``every_cell`` column selects every cell.
    """
    counts = np.ones((len(cell_ids), 3), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=list(cell_ids))
    store = open_clone_store(path)
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store


def copy_clone_store(template: Path, folder: Path) -> "DataStore":
    """Copy a template store into ``folder`` and open the copy."""
    target = folder / template.name
    shutil.copytree(template, target)
    return open_clone_store(target)


def every_cell(store: "DataStore") -> "ArtifactRef":
    return store.snapshot_cell_selection("every_cell")


def ids_read_bytes(store: "DataStore", n_cells: int) -> int:
    """The bytes of reading the IDs of ``n_cells`` cells of ``store`` as text."""
    from cytearc.metadata.rows import text_read_bytes

    return text_read_bytes(store.cells._get_array("ids"), n_cells)


def loaded_partition(store: "DataStore", clones: "ArtifactRef") -> set[frozenset[str]]:
    """The saved clones as sets of cell IDs, from the cells table of load_clones."""
    cells = store.clones.load_clones(clones).cells
    labelled = cells["clone"].dropna()
    return {
        frozenset(str(cell) for cell in members)
        for members in labelled.groupby(labelled, observed=True).groups.values()
    }


def import_truth_clones(
    store: "DataStore",
    truth: CloneTruth,
    folder: Path,
    *,
    cell_selection: "ArtifactRef | None" = None,
) -> "ArtifactRef":
    """Save the truth's clones by importing CoSpar's table of them.

    The table also lists barcodes outside the store, about which the import
    warns, as it does again when it reuses saved clones.
    """
    path = write_cospar(truth, folder / "truth_clones.csv")
    selection = every_cell(store) if cell_selection is None else cell_selection
    with pytest.warns(UserWarning, match="match no selected cell"):
        return store.clones.import_clones(path, cell_selection=selection)


def overlap_reference(x: Mapping[object, int], y: Mapping[object, int]) -> dict:
    """The overlap indices of two groups, from each one's cells of each clone.

    Set by set, as the published definitions read: ``shared`` counts the
    clones of both groups, the presence indices compare the two sets of
    clones, and the abundance indices weigh each clone by its cells.
    """
    shared = set(x) & set(y)
    s1, s2, both = len(x), len(y), len(shared)
    n1, n2 = sum(x.values()), sum(y.values())
    dot = sum(x[c] * y[c] for c in shared)
    q1, q2 = sum(v * v for v in x.values()), sum(v * v for v in y.values())
    return {
        "shared": both,
        "jaccard": both / (s1 + s2 - both),
        "sorensen": 2 * both / (s1 + s2),
        "overlap": both / min(s1, s2),
        "cosine": both / sqrt(s1 * s2),
        "morisita_horn": 2 * dot / ((q1 / n1**2 + q2 / n2**2) * n1 * n2),
        "abundance_cosine": dot / sqrt(q1 * q2),
        "shared_fraction_1": sum(x[c] for c in shared) / n1,
        "shared_fraction_2": sum(y[c] for c in shared) / n2,
    }


def coupling_reference(
    table: Mapping[object, Mapping[object, int]],
    first: object,
    second: object,
    method: str,
) -> float:
    """One pair's fate coupling, from each group's cells of each clone.

    As CoSpar's definitions read: ``share_cosine`` divides each group's cells
    of a clone by the group's cells and each clone's values by their sum over
    every group of ``table``, and ``normalized_covariance`` counts every clone
    that the table holds.
    """
    clones = sorted({clone for counts in table.values() for clone in counts}, key=str)
    x = np.asarray([table[first].get(c, 0) for c in clones], dtype=float)
    y = np.asarray([table[second].get(c, 0) for c in clones], dtype=float)
    if method == "abundance_cosine":
        return float(x @ y / sqrt((x @ x) * (y @ y)))
    if method == "share_cosine":
        frequency = {
            group: np.asarray([counts.get(c, 0) for c in clones]) / sum(counts.values())
            for group, counts in table.items()
        }
        total = sum(frequency.values())
        p, q = frequency[first] / total, frequency[second] / total
        return float(p @ q / sqrt((p @ p) * (q @ q)))
    if method == "normalized_covariance":
        return float(np.cov(x, y)[0, 1] / (x.mean() * y.mean()))
    if method == "jaccard":
        both = len(set(table[first]) & set(table[second]))
        return both / len(set(table[first]) | set(table[second]))
    raise ValueError(f"no reference for {method!r}")


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini and Hochberg's adjusted p-values, step by step from the largest."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return adjusted


def import_truth_receptors(
    store: "DataStore",
    receptors: ReceptorTruth,
    folder: Path,
    *,
    cell_selection: "ArtifactRef | None" = None,
) -> "ArtifactRef":
    """Import the truth's receptors from each sample's contig annotations.

    Each sample's T cell and B cell contigs of called cells go in files of
    their own, as ``cellranger multi`` writes them, joined to the store's
    cells with the sample's prefix. The files also list barcodes outside the
    store, about which the import warns.
    """
    paths = [
        write_cellranger_contigs(
            receptors,
            folder / f"{sample}_{receptor}_contigs.csv",
            sample=sample,
            receptor=receptor,
        )
        for sample in SAMPLES
        for receptor in ("TCR", "BCR")
    ]
    selection = every_cell(store) if cell_selection is None else cell_selection
    with pytest.warns(UserWarning, match="match no selected cell"):
        return store.repertoire.import_receptors(
            paths,
            cell_selection=selection,
            barcode_prefix=[f"{sample}__" for sample in SAMPLES for _ in range(2)],
        )


# The cells of the receptor store are cell0 to cell7.
N_RECEPTOR_CELLS = 8
CELLRANGER_HEADER = (
    "barcode,is_cell,contig_id,high_confidence,length,chain,v_gene,d_gene,j_gene,"
    "c_gene,full_length,productive,cdr3,cdr3_nt,reads,umis,raw_clonotype_id,"
    "raw_consensus_id"
)
# T cell contigs of cells cell0 to cell5 and of a barcode outside the store.
TCR_ROWS = [
    # cell0: one alpha and one beta chain.
    "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVF,TGTGCTGTGTTT,100,5,ct1,ct1_1",
    "cell0,true,cell0_contig_2,true,500,TRB,TRBV1,TRBD1,TRBJ1,TRBC1,true,true,CASSF,TGTGCCAGCAGCTTT,200,8,ct1,ct1_2",
    # cell1: a second alpha chain with fewer UMIs.
    "cell1,true,cell1_contig_1,true,500,TRA,TRAV2,,TRAJ2,TRAC,true,true,CAVG,TGTGCTGTGGGT,90,6,ct2,ct2_1",
    "cell1,true,cell1_contig_2,true,500,TRA,TRAV3,,TRAJ3,TRAC,true,true,CAVH,TGTGCTGTGCAT,50,2,ct2,ct2_2",
    "cell1,true,cell1_contig_3,true,500,TRB,TRBV2,,TRBJ2,TRBC2,true,true,CASSG,TGTGCCAGCAGCGGT,300,9,ct2,ct2_3",
    # cell2: a beta chain and a non-productive alpha chain.
    "cell2,true,cell2_contig_1,true,500,TRB,TRBV3,,TRBJ3,TRBC1,true,true,CASSH,TGTGCCAGCAGCCAT,80,4,ct3,ct3_1",
    "cell2,true,cell2_contig_2,true,500,TRA,TRAV4,,TRAJ4,TRAC,true,false,CAVI,TGTGCTGTGATT,30,3,ct3,ct3_2",
    # cell3: three beta chains.
    "cell3,true,cell3_contig_1,true,500,TRB,TRBV4,,TRBJ4,TRBC1,true,true,CASSI,TGTGCCAGCAGCATT,10,3,ct4,ct4_1",
    "cell3,true,cell3_contig_2,true,500,TRB,TRBV5,,TRBJ5,TRBC1,true,true,CASSK,TGTGCCAGCAGCAAA,10,2,ct4,ct4_2",
    "cell3,true,cell3_contig_3,true,500,TRB,TRBV6,,TRBJ6,TRBC1,true,true,CASSL,TGTGCCAGCAGCCTT,10,1,ct4,ct4_3",
    # cell5: a low-confidence contig outside a called cell.
    "cell5,false,cell5_contig_1,false,300,TRB,TRBV7,,TRBJ7,TRBC1,true,true,CASSM,TGTGCCAGCAGCATG,5,1,None,None",
    # A barcode that is not a cell of the store.
    "outside-1,true,outside-1_contig_1,true,500,TRB,TRBV8,,TRBJ8,TRBC1,true,true,CASSN,TGTGCCAGCAGCAAT,40,3,ct5,ct5_1",
]
# B cell contigs: cell4 is a B cell and cell0 also has a heavy chain.
BCR_ROWS = [
    "cell4,true,cell4_contig_1,true,500,IGH,IGHV1,IGHD1,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
    "cell4,true,cell4_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,400,15,cb1,cb1_2",
    "cell0,true,cell0_contig_1,true,500,IGH,IGHV2,,IGHJ2,IGHG1,true,true,CARX,TGTGCGAGATAA,50,2,cb2,cb2_1",
]
AIRR_HEADER = (
    "cell_id\tsequence_id\tv_call\td_call\tj_call\tc_call\tjunction\tjunction_aa"
    "\tproductive\tduplicate_count\tconsensus_count\tv_identity\tsequence"
    "\tsequence_alignment"
)
# B cell contigs of cell6, without a locus field.
AIRR_ROWS = [
    "cell6\tcell6_contig_1\tIGHV3-30*01\t\tIGHJ4*02\tIGHG1\tTGTGCG\tCARW\tT\t12\t300\t0.95\tACGTACGT\tACGT-ACGT",
    "cell6\tcell6_contig_2\tIGLV1-40*01\t\tIGLJ2*01\tIGLC2\tTGTCAG\tCQSY\tT\t9\t200\t\tGGCC\t",
]


# One codon for each residue, so distinct junction residues have distinct
# nucleotides.
_CODONS = dict(
    zip(
        "ACDEFGHIKLMNPQRSTVWY",
        "GCT TGT GAT GAA TTT GGT CAT ATT AAA CTG ATG AAT CCG CAG CGT AGC ACC GTG "
        "TGG TAT".split(),
        strict=True,
    )
)


def write_clone_contigs(
    folder: Path,
    name: str,
    cell_ids: Sequence[str],
    clone_of: np.ndarray,
    *,
    chains_per_arm: int = 1,
) -> Path:
    """Write Cell Ranger contig annotations in which each clone has its receptor.

    Every fifth clone is a B cell clone with heavy and kappa chains, and the
    others are T cell clones with alpha and beta chains, ``chains_per_arm`` of
    each, the first with the most UMIs. Each cell carries every chain of its
    clone, and a cell of clone -1 has no contig. A clone's junctions spell its
    number in residues, so clonotypes of identical chains are the clones.
    """
    residues = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for cell, clone in zip(cell_ids, np.asarray(clone_of).tolist(), strict=True):
        if clone < 0:
            continue
        digits = "".join(residues[clone // 20**power % 20] for power in range(5))
        loci = ("IGK", "IGH") if clone % 5 == 4 else ("TRA", "TRB")
        index = 0
        for arm, locus in enumerate(loci):
            for chain in range(chains_per_arm):
                index += 1
                junction_aa = f"C{digits}{residues[chain]}{'FW'[arm]}"
                junction = "".join(_CODONS[residue] for residue in junction_aa)
                rows.append(
                    f"{cell},true,{cell}_contig_{index},true,500,{locus},{locus}V1,,"
                    f"{locus}J1,{locus}C,true,true,{junction_aa},{junction},100,"
                    f"{9 - chain},ct{clone},ct{clone}_{index}"
                )
    return write_lines(folder, name, CELLRANGER_HEADER, rows)


def receptor_store(folder: Path, **options: object) -> "DataStore":
    """Open the store of cells cell0 to cell7 in ``folder``, writing it once."""
    from cytearc.datastore.datastore import DataStore

    path = folder / "store.zarr"
    if not path.exists():
        counts = np.random.default_rng(5).poisson(2.0, size=(N_RECEPTOR_CELLS, 6)) + 1
        write_count_store(str(path), {"RNA": counts}, "uint16")
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        **{"nthreads": 1, **options},  # type: ignore[arg-type]
    )


def write_lines(folder: Path, name: str, header: str, rows: list[str]) -> Path:
    """Write a header and rows, one per line, as ``folder / name``."""
    path = folder / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def tcr_file(folder: Path, rows: list[str] = TCR_ROWS, name: str = "tcr.csv") -> Path:
    """Cell Ranger's contig annotations of the T cell rows."""
    return write_lines(folder, name, CELLRANGER_HEADER, rows)


def bcr_file(folder: Path) -> Path:
    """Cell Ranger's contig annotations of the B cell rows."""
    return write_lines(folder, "bcr.csv", CELLRANGER_HEADER, BCR_ROWS)


def airr_file(folder: Path) -> Path:
    """An AIRR Rearrangement file of cell6's B cell rows."""
    return write_lines(folder, "bcr.tsv", AIRR_HEADER, AIRR_ROWS)


def levenshtein_distance(first: str, second: str) -> int:
    """The substitutions, insertions and deletions between two texts, row by row."""
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(
                min(
                    previous[column - 1] + (left != right),
                    previous[column] + 1,
                    current[column - 1] + 1,
                )
            )
        previous = current
    return previous[-1]


def within_distance(first: str, second: str, metric: str, cutoff: int) -> bool:
    """Whether two texts are within ``cutoff`` by ``metric``, as scirpy measures it.

    ``normalized_hamming`` rounds the substitutions' percentage of the length
    half up, in integers.
    """
    if metric == "identity":
        return first == second
    if metric == "levenshtein":
        return levenshtein_distance(first, second) <= cutoff
    if len(first) != len(second):
        return False
    mismatches = sum(a != b for a, b in zip(first, second, strict=True))
    if metric == "hamming":
        return mismatches <= cutoff
    return not first or (200 * mismatches + len(first)) // (2 * len(first)) <= cutoff
