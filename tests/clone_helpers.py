"""Stores and checks shared by the tests of clones."""

import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tests.clone_truth import (
    DARLIN_ARRAYS,
    SAMPLES,
    AlleleCallingTruth,
    CallingTruth,
    CloneTruth,
    ReceptorTruth,
    write_carlin_pair,
    write_cellranger_contigs,
    write_celltag_reads,
    write_cospar,
    write_pycashier,
    write_trex_molecules,
)
from tests.storage_helpers import open_store, write_count_store

if TYPE_CHECKING:
    from cytearc.datastore.datastore import DataStore
    from cytearc.storage.refs import ArtifactRef


def write_truth_store(path: Path, truth: CloneTruth) -> "DataStore":
    """Write a store whose cells are the truth's, with their sample, time and type."""
    n_cells = len(truth.cell_ids)
    counts = np.random.default_rng(5).poisson(3.0, size=(n_cells, 6)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=truth.cell_ids)
    store = open_store(path)
    store.cells.insert("sample", truth.samples.astype(object))
    store.cells.insert("time_point", truth.time_points)
    store.cells.insert("cell_type", truth.cell_types.astype(object))
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    return store


def write_cells_store(path: Path, cell_ids: Sequence[str]) -> "DataStore":
    """Write a store of these cells, each with one count of three features.

    The ``every_cell`` column selects every cell.
    """
    counts = np.ones((len(cell_ids), 3), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16", cell_ids=list(cell_ids))
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(len(cell_ids), dtype=bool))
    return store


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


def splitmix64(value: int) -> int:
    """SplitMix64's finaliser (Steele, Lea and Flood 2014) on Python integers, from its published constants."""
    mask = 2**64 - 1
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


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
# cell0 and cell1 share an alpha and a beta chain. cell2 has the same alpha
# residues from other nucleotides, cell3 the beta chain alone, and cell4 a
# second alpha chain beside the pair. cell5 and cell6 are B cells of one
# clone, and cell7 has no contig.
CLONAL_ROWS = [
    "cell0,true,cell0_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
    "cell0,true,cell0_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
    "cell1,true,cell1_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,70,7,ct1,ct1_1",
    "cell1,true,cell1_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,60,6,ct1,ct1_2",
    "cell2,true,cell2_contig_1,true,500,TRA,TRAV2,,TRAJ1,TRAC,true,true,CAVA,TGTGCAGTAGCA,50,5,ct2,ct2_1",
    "cell2,true,cell2_contig_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,40,4,ct2,ct2_2",
    "cell3,true,cell3_contig_1,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,30,3,ct3,ct3_1",
    "cell4,true,cell4_contig_1,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct4,ct4_1",
    "cell4,true,cell4_contig_2,true,500,TRA,TRAV3,,TRAJ2,TRAC,true,true,CAVB,TGTGCTGTGTAC,20,2,ct4,ct4_2",
    "cell4,true,cell4_contig_3,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct4,ct4_3",
    "cell5,true,cell5_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
    "cell5,true,cell5_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,400,15,cb1,cb1_2",
    "cell6,true,cell6_contig_1,true,500,IGH,IGHV1,,IGHJ1,IGHG1,true,true,CARW,TGTGCGAGATGG,300,12,cb1,cb1_1",
    "cell6,true,cell6_contig_2,true,500,IGK,IGKV1,,IGKJ1,IGKC,true,true,CQQY,TGTCAGCAGTAT,200,10,cb1,cb1_2",
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
    path = folder / "store.zarr"
    if not path.exists():
        counts = np.random.default_rng(5).poisson(2.0, size=(N_RECEPTOR_CELLS, 6)) + 1
        write_count_store(str(path), {"RNA": counts}, "uint16")
    return open_store(path, **options)


def write_lines(folder: Path, name: str, header: str, rows: list[str]) -> Path:
    """Write a header and rows, one per line, as ``folder / name``."""
    path = folder / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def clonal_receptors(
    folder: Path, rows: list[str] = CLONAL_ROWS
) -> tuple["DataStore", "ArtifactRef"]:
    """The store of cells cell0 to cell7 and the receptors of Cell Ranger rows."""
    store = receptor_store(folder)
    path = write_lines(folder, "clonal.csv", CELLRANGER_HEADER, rows)
    receptors = store.repertoire.import_receptors(
        path, cell_selection=store.snapshot_cell_selection()
    )
    return store, receptors


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


def clone_partition(codes: np.ndarray) -> set[frozenset[int]]:
    """The positions of the cells of each clone, from each cell's clone code, -1 for none."""
    return {
        frozenset(np.flatnonzero(codes == clone).tolist())
        for clone in range(int(codes.max()) + 1 if len(codes) else 0)
    }


def link_count_variance(
    n_texts: int, first: np.ndarray, second: np.ndarray
) -> tuple[int, float]:
    """The pairs linked, and the variance of that count over draws of the texts.

    Each pair's link varies on its own, and with every pair that shares one of
    its texts: the cherries, two linked pairs that share a text, estimate how
    much more often such pairs link together than apart.
    """
    count = len(first)
    pairs = n_texts * (n_texts - 1) / 2
    rate = count / pairs
    linked = np.bincount(np.concatenate([first, second]), minlength=n_texts)
    cherries = float(np.sum(linked * (linked - 1) / 2))
    together = n_texts * (n_texts - 1) * (n_texts - 2) / 2 * rate**2
    return count, pairs * rate * (1 - rate) + 2 * max(0.0, cherries - together)


def within_distance(first: str, second: str, metric: str, cutoff: int) -> bool:
    """Whether two texts are within ``cutoff`` by ``metric``, as scirpy measures it.

    ``normalized_hamming`` rounds the substitutions' percentage of the length
    half up, computed as an exact fraction. A cutoff of 0 is identity under
    every metric, as scirpy's ``ir_dist`` makes it.
    """
    if metric == "identity" or cutoff == 0:
        return first == second
    if metric == "levenshtein":
        return levenshtein_distance(first, second) <= cutoff
    if len(first) != len(second):
        return False
    mismatches = sum(a != b for a, b in zip(first, second, strict=True))
    if metric == "hamming":
        return mismatches <= cutoff
    if not first:
        return True
    percent = Fraction(100 * mismatches, len(first))
    return math.floor(percent + Fraction(1, 2)) <= cutoff


def count_units(
    units: pd.DataFrame, keys: list[str], values: list[str], count: str
) -> pd.DataFrame:
    """Count units by keys and values, with each count's share of its keys.

    The pandas reference that the counts of chains are checked against, as
    #55 counted gene usage before its counts moved to integer codes.

    Args:
        units: One row per unit, such as a cell's chain, with the ``keys``
            and ``values`` columns; rows with a missing value are left out.
        keys: The columns whose combinations share one total, such as the
            group and locus.
        values: The counted columns, such as the V gene.
        count: The name of the count column.

    Returns:
        One row per combination of keys and values with a unit, in the
        natural order of the keys and by count, most first: the keys, the
        values, ``count`` and ``fraction``.
    """
    kept = units.dropna(subset=values)
    if kept.empty:
        return pd.DataFrame(columns=[*keys, *values, count, "fraction"])
    table = (
        kept.groupby([*keys, *values], sort=False, dropna=False)
        .size()
        .rename(count)
        .reset_index()
    )
    totals = table.groupby(keys, sort=False, dropna=False)[count].transform("sum")
    table["fraction"] = table[count] / totals
    table[count] = table[count].astype(np.int64)
    return table.sort_values(
        [*keys, count, *values],
        ascending=[*([True] * len(keys)), False, *([True] * len(values))],
        kind="stable",
    ).reset_index(drop=True)


#: The file layout in which each barcode-calling preset reads the truth.
CALLING_LAYOUTS = ("trex", "pycashier", "celltag", "celltag_multi")


def import_calling_truth(
    store: "DataStore",
    calling: CallingTruth,
    folder: Path,
    layout: str,
    *,
    cell_selection: "ArtifactRef | None" = None,
    samples: Sequence[str] = SAMPLES,
) -> "ArtifactRef":
    """Import the calling truth's molecules as one tool writes them, one file per sample.

    ``"trex"`` writes TREX's ``molecules.txt`` of one library, ``"pycashier"``
    pycashier's reads, ``"celltag"`` the Biddy et al. workflow's reads of
    one library, and ``"celltag_multi"`` CellTag-multi's scRNA-seq reads of
    d2 and its scATAC-seq reads, without UMIs, of d4, as two libraries. The
    files are listed in the order of ``samples``.
    """
    barcodes = calling.barcodes
    prefixes = [f"{sample}__" for sample in samples]
    assays = {"d2": "RNA", "d4": "ATAC"}
    if layout == "trex":
        paths = [
            write_trex_molecules(
                barcodes, folder / f"{sample}_molecules.txt", sample=sample
            )
            for sample in samples
        ]
        options: dict[str, object] = {"barcode_suffix": "-1", "library": "trex"}
    elif layout == "pycashier":
        paths = [
            write_pycashier(
                barcodes,
                folder / f"{sample}.umi_cell_labeled.barcode.tsv",
                sample=sample,
            )
            for sample in samples
        ]
        options = {"barcode_suffix": "-1", "library": "larry"}
    elif layout == "celltag":
        paths = [
            write_celltag_reads(
                barcodes,
                folder / f"{sample}.celltag.parsed.tsv.gz",
                sample=sample,
                biddy=True,
            )
            for sample in samples
        ]
        options = {"library": "rna"}
    elif layout == "celltag_multi":
        paths = [
            write_celltag_reads(
                barcodes,
                folder / f"{sample}_parse.txt.gz",
                sample=sample,
                assay=assays[sample],
            )
            for sample in samples
        ]
        options = {"library": [assays[sample] for sample in samples]}
    else:
        raise ValueError(f"layout must be one of {', '.join(CALLING_LAYOUTS)}")
    selection = every_cell(store) if cell_selection is None else cell_selection
    return store.clones.import_barcodes(
        paths, cell_selection=selection, barcode_prefix=prefixes, **options
    )


def import_allele_truth(
    store: "DataStore",
    calling: AlleleCallingTruth,
    truth: CloneTruth,
    folder: Path,
    *,
    cell_selection: "ArtifactRef | None" = None,
    samples: Sequence[str] = SAMPLES,
    arrays: Sequence[str] = DARLIN_ARRAYS,
) -> "ArtifactRef":
    """Import the allele calling truth as CARLIN pairs, one per sample and array.

    Each sample's pairs join its cells through the sample's prefix, and the
    first pair of d2 lists droplets outside the store, about which the import
    warns. The pairs are listed sample by sample, in the order of ``samples``
    and ``arrays``.
    """
    import warnings

    paths, integrations, prefixes = [], [], []
    for sample in samples:
        for array in arrays:
            pair = folder / f"{sample}_{array}"
            write_carlin_pair(
                calling.recorders, truth, pair, sample=sample, array=array
            )
            paths.append(pair)
            integrations.append(array)
            prefixes.append(f"{sample}__")
    selection = every_cell(store) if cell_selection is None else cell_selection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return store.clones.import_alleles(
            paths,
            cell_selection=selection,
            integrations=integrations,
            barcode_prefix=prefixes,
            barcode_suffix="-1",
        )


def traced_steps(
    monkeypatch: pytest.MonkeyPatch, call: Callable[[], object], budget: int
) -> tuple[list[tuple[str, int]], int]:
    """Each memory check of allele clone calling with the bytes it states, and the call's traced peak.

    The bytes that a check states are the call's total under ``budget``:
    those of the operation's checks as they state them, those of the
    domain's beside what the operation holds, and those of
    ``clones/calling.py`` beside what the domain holds. A first call finds
    the checks and warms the compiled kernels. The traced call then holds,
    from each check to the next, the bytes by which that check falls short
    of the largest, allocated but never written, so its peak passes the
    largest check exactly when a step holds more than its own check states.
    The peak is traced as :func:`tests.memory_helpers.traced_peak`
    traces it, again where it passes the largest check, so ``call`` must
    compute afresh each time.

    Returns:
        Each check's step and the bytes it states, from the first call, and
        the traced call's peak.
    """
    import gc

    import cytearc.clones.allele_clones as domain
    import cytearc.clones.calling as calling
    import cytearc.datastore._operations.recorder_alleles as operations
    from tests.memory_helpers import traced_peak

    checks: list[tuple[str, int]] = []
    allowance: list[np.ndarray] = []
    ceiling = [0]

    def note(what: str, stated: int) -> None:
        checks.append((str(what), int(stated)))
        allowance.clear()
        if ceiling[0]:
            # A check of the traced call that states more than the first
            # call's largest is held to its own bytes, and fails below.
            allowance.append(np.empty(max(0, ceiling[0] - int(stated)), dtype=np.uint8))

    room_check = domain._Room.check
    calling_check = calling._check
    operation_check = operations._check_calling

    def checked_room(self, needed: int, what: str) -> None:
        held = 0 if self.limit is None else budget - self.limit
        note(what, held + self.held + needed)
        room_check(self, needed, what)

    def checked_calling(needed: int, max_bytes: int | None, what: str) -> None:
        note(what, needed + (0 if max_bytes is None else budget - max_bytes))
        calling_check(needed, max_bytes, what)

    def checked_operation(needed: int, limit: int, what: str) -> None:
        note(what, needed)
        operation_check(needed, limit, what)

    def run() -> None:
        checks.clear()
        try:
            call()
        finally:
            allowance.clear()

    with monkeypatch.context() as patched:
        patched.setattr(domain._Room, "check", checked_room)
        patched.setattr(calling, "_check", checked_calling)
        patched.setattr(operations, "_check_calling", checked_operation)
        run()
        found = list(checks)
        ceiling[0] = max((stated for _what, stated in found), default=0)
        gc.collect()
        _result, peak = traced_peak(run, ceiling[0])
        # The traced call makes the checks that the first call made.
        assert checks == found, (checks, found)
    return found, peak
