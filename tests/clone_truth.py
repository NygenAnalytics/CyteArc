"""A seeded synthetic lineage tracing experiment, written in every clone format.

``clone_truth`` draws cells with a sample, a time point and a cell type, and
clones of known sizes. The writers emit the experiment as each tool writes
its files, with the quirks an importer must handle: headers that start with
``#``, R row names, unnamed index columns, gzip, a missing final newline,
``NA`` and barcodes of droplets that are not cells. ``receptor_truth`` gives
the clones T cell and B cell receptor chains and fixes, for each cell, the
chains and labels that an import of its contigs must give. ``barcode_truth``
gives the clones lineage barcodes and each cell its molecules and their reads,
with sequencing errors, ambient reads and droplets outside the store, which
writers emit as reads, molecules or barcode counts in each tool's format.
Tests check that an import recovers the truth, which this module fixes and the
code under test never computes.
"""

import gzip
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SAMPLES = ("d2", "d4")
TIME_POINTS = {"d2": 2, "d4": 4}
CELL_TYPES = ("Undifferentiated", "Neutrophil", "Monocyte")
# Clone sizes, largest first; the remaining cells have no clone.
CLONE_SIZES = (9, 6, 5, 4, 3, 2, 2, 1)
N_CELLS = 48
# Barcodes that the files list in clones but that name no cell of the store,
# such as droplets that cell calling rejected.
N_OUTSIDE = 3
TREX_GENERATIONS = ("2022", "2023", "current")


@dataclass(frozen=True, slots=True)
class CloneTruth:
    """The cells and clones of a synthetic experiment.

    Attributes:
        barcodes: Each cell's 16-nucleotide barcode.
        samples: Each cell's sample.
        cell_types: Each cell's type.
        clones: Each cell's clone, as its rank from 0 (largest first), -1
            for a cell without a clone.
        probabilities: Each cell's probability of its clone.
        outside: Barcodes that the files list in clone 0 but that name no
            cell of the store.
    """

    barcodes: np.ndarray
    samples: np.ndarray
    cell_types: np.ndarray
    clones: np.ndarray
    probabilities: np.ndarray
    outside: np.ndarray

    @property
    def cell_ids(self) -> np.ndarray:
        """Each cell's ID in the store: sample, two underscores, barcode and ``-1``."""
        return np.asarray(
            [
                f"{sample}__{barcode}-1"
                for sample, barcode in zip(self.samples, self.barcodes, strict=True)
            ]
        )

    @property
    def time_points(self) -> np.ndarray:
        return np.asarray([TIME_POINTS[sample] for sample in self.samples])

    @property
    def n_clones(self) -> int:
        return int(self.clones.max()) + 1

    def partition(
        self,
        cells: np.ndarray | None = None,
        *,
        split_by: np.ndarray | None = None,
    ) -> set[frozenset[str]]:
        """The clones as sets of cell IDs.

        Args:
            cells: The cells to keep, as a mask, or None for every cell.
            split_by: A key per cell whose values keep cells of one clone
                apart, such as the sample of each file when each sample's
                file is imported with its own namespace.
        """
        keep = np.ones(len(self.clones), dtype=bool) if cells is None else cells
        keys = (
            np.zeros(len(self.clones), dtype=object) if split_by is None else split_by
        )
        ids = self.cell_ids
        groups: dict[tuple[int, object], set[str]] = {}
        for position in np.flatnonzero(keep & (self.clones >= 0)).tolist():
            key = (int(self.clones[position]), keys[position])
            groups.setdefault(key, set()).add(str(ids[position]))
        return {frozenset(group) for group in groups.values()}

    def _counted(
        self, groups: np.ndarray | None, cells: np.ndarray | None
    ) -> tuple[list[tuple[object, int] | None], Counter]:
        """Each cell's group and clone, None when it is not counted, and the counts."""
        keep = np.ones(N_CELLS, dtype=bool) if cells is None else cells
        labels = np.zeros(N_CELLS, dtype=object) if groups is None else groups
        pairs: list[tuple[object, int] | None] = [
            (label, clone) if counted and clone >= 0 else None
            for label, clone, counted in zip(
                labels.tolist(), self.clones.tolist(), keep.tolist(), strict=True
            )
        ]
        return pairs, Counter(pair for pair in pairs if pair is not None)

    def cell_clone_sizes(
        self, groups: np.ndarray | None = None, cells: np.ndarray | None = None
    ) -> np.ndarray:
        """Each cell's clone size within its group, counted cell by cell.

        Args:
            groups: Each cell's group, or None for one group of every cell.
            cells: The cells counted, as a mask, or None for every cell.

        Returns:
            The cells of each counted cell's clone in its group, 0 for a cell
            without a clone or outside ``cells``.
        """
        pairs, counts = self._counted(groups, cells)
        return np.asarray([0 if pair is None else counts[pair] for pair in pairs])

    def group_clone_sizes(
        self, groups: np.ndarray | None = None, cells: np.ndarray | None = None
    ) -> dict[object, list[int]]:
        """Each group's clone sizes, largest first, counted cell by cell."""
        _pairs, counts = self._counted(groups, cells)
        sizes: dict[object, list[int]] = {}
        for (group, _clone), size in counts.items():
            sizes.setdefault(group, []).append(size)
        return {group: sorted(found, reverse=True) for group, found in sizes.items()}

    def group_clone_cells(
        self, groups: np.ndarray | None = None, cells: np.ndarray | None = None
    ) -> dict[object, dict[int, int]]:
        """Each group's cells of each of its clones, counted cell by cell.

        Returns:
            The groups in sorted order, each with its clones by rank and their
            cells in the group.
        """
        _pairs, counts = self._counted(groups, cells)
        found: dict[object, dict[int, int]] = {}
        for (group, clone), size in sorted(counts.items()):
            found.setdefault(group, {})[clone] = size
        return found


def clone_truth(seed: int = 0) -> CloneTruth:
    """Draw a synthetic experiment with clones of the sizes in ``CLONE_SIZES``."""
    rng = np.random.default_rng(seed)
    n_barcodes = N_CELLS + N_OUTSIDE
    barcodes: set[str] = set()
    while len(barcodes) < n_barcodes:
        barcodes.add("".join(rng.choice(list("ACGT"), size=16)))
    drawn = np.asarray(sorted(barcodes))[rng.permutation(n_barcodes)]
    clones = np.full(N_CELLS, -1, dtype=np.int64)
    members = rng.permutation(N_CELLS)
    start = 0
    for clone, size in enumerate(CLONE_SIZES):
        clones[members[start : start + size]] = clone
        start += size
    return CloneTruth(
        barcodes=drawn[:N_CELLS],
        samples=rng.choice(SAMPLES, size=N_CELLS),
        cell_types=rng.choice(CELL_TYPES, size=N_CELLS),
        clones=clones,
        probabilities=np.round(rng.uniform(0.5, 1.0, size=N_CELLS), 6),
        outside=drawn[N_CELLS:],
    )


def number_label(clone: int) -> str:
    """A clone label as TREX and LARRY number clones, from 1."""
    return str(clone + 1)


def without_single_cell_clones(truth: CloneTruth) -> CloneTruth:
    """The truth with the cell of each clone of one cell left without a clone.

    A source that joins each cell to the cells that share its evidence, such
    as clones called from variants, calls no clone of one cell. The other
    clones keep their order by size, numbered from 0 again.
    """
    clones = truth.clones
    sizes = np.bincount(clones[clones >= 0], minlength=truth.n_clones)
    kept = sizes >= 2
    numbers = np.cumsum(kept) - 1
    clone_of = np.where(clones >= 0, clones, 0)
    return replace(
        truth, clones=np.where((clones >= 0) & kept[clone_of], numbers[clone_of], -1)
    )


def _write(
    path: Path,
    lines: Iterable[str],
    *,
    compress: bool = False,
    final_newline: bool = True,
) -> Path:
    text = "\n".join(lines) + ("\n" if final_newline else "")
    data = text.encode("utf-8")
    if compress:
        path.write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_bytes(data)
    return path


def _rows(
    truth: CloneTruth, cells: np.ndarray | None, *, with_outside: bool
) -> list[tuple[str, int]]:
    """Each listed cell's ID and clone, cells without a clone included."""
    keep = np.ones(N_CELLS, dtype=bool) if cells is None else cells
    rows = [
        (str(cell_id), int(clone))
        for cell_id, clone in zip(truth.cell_ids[keep], truth.clones[keep], strict=True)
    ]
    if with_outside:
        rows += [(f"d2__{barcode}-1", 0) for barcode in truth.outside]
    return rows


def write_cospar(
    truth: CloneTruth,
    path: Path,
    *,
    cells: np.ndarray | None = None,
    compress: bool = False,
) -> Path:
    """CoSpar's ``Cell_ID,Clone_ID``, one row per cell with a clone."""
    rows = [row for row in _rows(truth, cells, with_outside=True) if row[1] >= 0]
    return _write(
        path,
        [
            "Cell_ID,Clone_ID",
            *(f"{cell},{number_label(clone)}" for cell, clone in rows),
        ],
        compress=compress,
    )


def write_trex(
    truth: CloneTruth,
    path: Path,
    *,
    generation: str = "current",
    cell_text: Sequence[str] | None = None,
    cells: np.ndarray | None = None,
) -> Path:
    """TREX's ``clones.txt`` of one release, one row per cell with a clone.

    The 2022 releases wrote ``#clone_id,cell_id`` as CSV, mid-2023 briefly
    ``clone#`` and TREX since ``clone_nr``, tab-separated. ``cell_text``
    replaces each cell's ID, such as its bare barcode, in cell order.
    """
    rows = _rows(truth, cells, with_outside=True)
    if cell_text is not None:
        keep = np.ones(N_CELLS, dtype=bool) if cells is None else cells
        texts = list(np.asarray(cell_text)[keep]) + [
            str(barcode) for barcode in truth.outside
        ]
        rows = [(text, clone) for text, (_cell, clone) in zip(texts, rows, strict=True)]
    rows = sorted((row for row in rows if row[1] >= 0), key=lambda row: row[1])
    if generation == "2022":
        header, separator = "#clone_id,cell_id", ","
    elif generation == "2023":
        header, separator = "clone#\tcell_id", "\t"
    elif generation == "current":
        header, separator = "clone_nr\tcell_id", "\t"
    else:
        raise ValueError(f"TREX generations are {TREX_GENERATIONS}")
    return _write(
        path,
        [header, *(f"{number_label(clone)}{separator}{cell}" for cell, clone in rows)],
    )


def write_celltag_multi(
    truth: CloneTruth, path: Path, *, index_column: bool = True
) -> Path:
    """CellTag-multi's clone table with its unnamed index column and annotations.

    Clone IDs carry the replicate, as ``r1_<n>``, and each row repeats its
    clone's size, as the iEP table on Zenodo does.
    """
    rows = [row for row in _rows(truth, None, with_outside=True) if row[1] >= 0]
    sizes = np.bincount([clone for _cell, clone in rows])
    header = "clone_id,cell.bc,edge.den,day,assay,clone.size (RNA & ATAC)"
    lines = ["," + header if index_column else header]
    for position, (cell, clone) in enumerate(rows):
        fields = f"r1_{clone + 1},{cell},1.0,D12,RNA,{sizes[clone]}"
        lines.append(f"{27000 + position},{fields}" if index_column else fields)
    return _write(path, lines)


def write_celltagr(truth: CloneTruth, path: Path) -> Path:
    """CellTagR's clone composition, ``clone.id`` and ``cell.barcode``, R-quoted."""
    rows = [row for row in _rows(truth, None, with_outside=True) if row[1] >= 0]
    return _write(
        path,
        [
            '"clone.id","cell.barcode"',
            *(f'{clone + 1},"{cell}"' for cell, clone in rows),
        ],
    )


def write_sticr(truth: CloneTruth, path: Path) -> Path:
    """STICR's cell metadata: every cell, ``VBC`` empty without a clone.

    The UMI counts are padded with spaces, as the Cell Browser tables are.
    """
    lines = ["cell_Id\tnCount_RNA\tVBC\tAssigned_STICR_Barcode_UMI"]
    for cell, clone in _rows(truth, None, with_outside=True):
        label = f"GW15_Rep1_{clone + 1}" if clone >= 0 else ""
        lines.append(f"{cell}\t{1000 + clone}\t{label}\t {7 + clone:3d}")
    return _write(path, lines)


def write_cassiopeia(truth: CloneTruth, path: Path, *, compress: bool = True) -> Path:
    """Cassiopeia's allele table: two or three rows per cell, one per intBC.

    A cell without a clone has lineage group 0, which Cassiopeia writes for
    cells it assigned to no group.
    """
    lines = ["cellBC\tintBC\tallele\tr1\tr2\tr3\tlineageGrp\tUMI\treadCount"]
    for position, (cell, clone) in enumerate(_rows(truth, None, with_outside=True)):
        for site in range(2 + position % 2):
            lines.append(
                f"{cell}\tINT{site}\tAAC[None]GGT\tAAC[None]\tGGT\tNone\t"
                f"{clone + 1}\t{3 + site}\t{40 + site}"
            )
    return _write(path, lines, compress=compress)


def write_numbat(truth: CloneTruth, path: Path) -> Path:
    """Numbat's ``clone_post_{i}.tsv``: each cell's clone and its posterior.

    Cells without a clone are written ``NA``; ``GT_opt`` holds R's quoted
    empty string for the normal clone.
    """
    lines = ["cell\tclone_opt\tGT_opt\tp_opt\tp_1\tp_2\tcompartment_opt"]
    for position, (cell, clone) in enumerate(_rows(truth, None, with_outside=False)):
        if clone < 0:
            lines.append(f"{cell}\tNA\tNA\tNA\tNA\tNA\tNA")
            continue
        probability = truth.probabilities[position]
        genotype = '""' if clone == 0 else "1a,5b"
        lines.append(
            f"{cell}\t{number_label(clone)}\t{genotype}\t{probability}\t"
            f"{probability}\t{1 - probability:.6f}\ttumor"
        )
    return _write(path, lines)


def write_wide(truth: CloneTruth, path: Path, *, rounds: int = 3) -> Path:
    """The Biddy et al. workflow's ``celltag_data.txt``: one clone column per round.

    R's ``write.table`` layout: space-separated, quoted, a header one field
    shorter than the rows, which start with the row name, and ``NA`` for a
    cell without a clone. The truth's clones fill the last round; earlier
    rounds give every cell another clone or none.
    """
    names = [f'"CellTagD{round_index}"' for round_index in range(rounds)]
    lines = [" ".join(names)]
    for position, (cell, clone) in enumerate(_rows(truth, None, with_outside=True)):
        earlier = [
            str(position % 4 + 1) if position % 3 else "NA" for _ in range(rounds - 1)
        ]
        last = number_label(clone) if clone >= 0 else "NA"
        lines.append(" ".join([f'"{cell}"', *earlier, last]))
    return _write(path, lines)


def write_larry(
    truth: CloneTruth,
    folder: Path,
    *,
    orientation: str = "cells_by_clones",
    stem: str = "GSM0_stateFate_test_",
) -> Path:
    """LARRY's GEO files: a binary clone matrix and the metadata of its cells.

    The metadata names each cell by ``Library`` and ``Cell barcode``, which
    joined by ``"__"`` give the cell's ID, and ends without a newline, as
    GEO's files do. ``orientation`` is ``"cells_by_clones"`` (the in vitro
    set) or ``"clones_by_cells"`` (the in vivo and cytokine sets). Returns
    the matrix's path.
    """
    rows = _rows(truth, None, with_outside=True)
    metadata = ["Library\tCell barcode\tTime point\tCell type annotation"]
    for position, (cell, _clone) in enumerate(rows):
        library, barcode = cell.split("__", 1)
        time_point = TIME_POINTS[library]
        cell_type = (
            truth.cell_types[position] if position < N_CELLS else "Undifferentiated"
        )
        metadata.append(f"{library}\t{barcode}\t{float(time_point)}\t{cell_type}")
    _write(
        folder / f"{stem}metadata.txt.gz", metadata, compress=True, final_newline=False
    )
    entries = [
        (row + 1, clone + 1) for row, (_cell, clone) in enumerate(rows) if clone >= 0
    ]
    n_cells, n_clones = len(rows), truth.n_clones
    if orientation == "cells_by_clones":
        shape, pairs = (n_cells, n_clones), entries
    elif orientation == "clones_by_cells":
        shape, pairs = (n_clones, n_cells), [(clone, cell) for cell, clone in entries]
    else:
        raise ValueError("orientation is cells_by_clones or clones_by_cells")
    matrix = folder / f"{stem}clone_matrix.mtx.gz"
    _write(
        matrix,
        [
            "%%MatrixMarket matrix coordinate integer general",
            "%",
            f"{shape[0]} {shape[1]} {len(pairs)}",
            *(f"{row} {column} 1" for row, column in pairs),
        ],
        compress=True,
    )
    return matrix


def _obs_frame(handle: h5py.File, names: Sequence[str]) -> h5py.Group:
    obs = handle.create_group("obs")
    obs.attrs["_index"] = "_index"
    obs.attrs["encoding-type"] = "dataframe"
    obs.attrs["encoding-version"] = "0.2.0"
    obs.attrs["column-order"] = np.asarray([], dtype=h5py.string_dtype())
    obs.create_dataset("_index", data=np.asarray(names, dtype=h5py.string_dtype()))
    return obs


def write_categorical(
    group: h5py.Group, name: str, labels: Sequence[str | None]
) -> None:
    """Write an AnnData categorical column; None is a missing label."""
    categories = sorted({label for label in labels if label is not None})
    codes = np.asarray(
        [categories.index(label) if label is not None else -1 for label in labels],
        dtype=np.int8,
    )
    column = group.create_group(name)
    column.attrs["encoding-type"] = "categorical"
    column.attrs["encoding-version"] = "0.2.0"
    column.attrs["ordered"] = False
    column.create_dataset("codes", data=codes)
    column.create_dataset(
        "categories", data=np.asarray(categories, dtype=h5py.string_dtype())
    )


def write_cospar_h5ad(truth: CloneTruth, path: Path, *, layout: str = "csr") -> Path:
    """An AnnData file with CoSpar's ``obsm['X_clone']``, cells by clones.

    ``layout`` stores the matrix as ``"csr"``, ``"csc"`` or ``"dense"``.
    Clones are numbered by their column from 0, as CoSpar numbers them.
    """
    rows = _rows(truth, None, with_outside=True)
    dense = np.zeros((len(rows), truth.n_clones), dtype=np.float32)
    for row, (_cell, clone) in enumerate(rows):
        if clone >= 0:
            dense[row, clone] = 1.0
    with h5py.File(path, "w") as handle:
        _obs_frame(handle, [cell for cell, _clone in rows])
        obsm = handle.create_group("obsm")
        if layout == "dense":
            obsm.create_dataset("X_clone", data=dense)
            return path
        if layout not in ("csr", "csc"):
            raise ValueError("layout is csr, csc or dense")
        major = dense if layout == "csr" else dense.T
        indptr = np.concatenate([[0], np.cumsum((major != 0).sum(axis=1))])
        indices = np.concatenate([np.flatnonzero(line) for line in major])
        group = obsm.create_group("X_clone")
        group.attrs["encoding-type"] = f"{layout}_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray(dense.shape)
        group.create_dataset("data", data=np.ones(len(indices), dtype=np.float32))
        group.create_dataset("indices", data=indices.astype(np.int32))
        group.create_dataset("indptr", data=indptr.astype(np.int64))
    return path


def write_mito_h5ad(truth: CloneTruth, path: Path) -> Path:
    """An AnnData file with MiTo's ``obs['MiTo_clone']``, ``"unassigned"`` for none."""
    rows = _rows(truth, None, with_outside=True)
    with h5py.File(path, "w") as handle:
        obs = _obs_frame(handle, [cell for cell, _clone in rows])
        write_categorical(
            obs,
            "MiTo_clone",
            [
                f"MT_{clone + 1}" if clone >= 0 else "unassigned"
                for _cell, clone in rows
            ],
        )
    return path


def write_anndata_h5ad(truth: CloneTruth, path: Path, *, content: str) -> Path:
    """An AnnData file that anndata itself writes, as current releases store it.

    anndata stores a text index and text columns as nullable string arrays,
    groups of values and a mask, which the h5py writers above do not. With
    ``content="mito"`` the clones are MiTo's ``obs['MiTo_clone']``, with
    ``"cospar"`` CoSpar's sparse ``obsm['X_clone']`` numbered from 0, and with
    ``"columns"`` the obs column ``clone`` in pandas' string dtype, NA for no
    clone, beside each cell's ``probability``, NaN for none.
    """
    import anndata
    import pandas as pd
    from scipy import sparse

    rows = _rows(truth, None, with_outside=True)
    probability = dict(zip(truth.cell_ids.tolist(), truth.probabilities, strict=True))
    index = pd.Index([cell for cell, _clone in rows], dtype="string")
    obs = pd.DataFrame(index=index)
    obsm = {}
    if content == "mito":
        obs["MiTo_clone"] = pd.Categorical(
            [
                f"MT_{number_label(clone)}" if clone >= 0 else "unassigned"
                for _cell, clone in rows
            ]
        )
    elif content == "cospar":
        entries = [
            (row, clone) for row, (_cell, clone) in enumerate(rows) if clone >= 0
        ]
        obsm["X_clone"] = sparse.csr_matrix(
            (
                np.ones(len(entries), dtype=np.float32),
                ([row for row, _clone in entries], [clone for _row, clone in entries]),
            ),
            shape=(len(rows), truth.n_clones),
        )
    elif content == "columns":
        obs["clone"] = pd.array(
            [number_label(clone) if clone >= 0 else None for _cell, clone in rows],
            dtype="string",
        )
        obs["probability"] = np.asarray(
            [
                probability.get(cell, np.nan) if clone >= 0 else np.nan
                for cell, clone in rows
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("content is mito, cospar or columns")
    # Releases of anndata before pandas 3 write nullable strings, as current
    # releases always do, only once that is allowed.
    with anndata.settings.override(allow_write_nullable_strings=True):
        anndata.AnnData(obs=obs, obsm=obsm).write_h5ad(path)
    return path


# Receptors of the truth's clones: clones 0 to 5 are T cell clones with an
# alpha (TRA) and a beta (TRB) chain, and clones 6 and 7 B cell clones with a
# heavy chain (IGH) and a kappa (IGK) or lambda (IGL) light chain.
B_CELL_LIGHT_LOCI = {6: "IGK", 7: "IGL"}
RECEPTOR_SLOTS = ("VJ_1", "VJ_2", "VDJ_1", "VDJ_2")
# Each cell's receptor profile: how its contigs differ from its clone's two
# chains. A cell of a clone has the clone's chains ("pair"), lacks its VJ or
# VDJ chain ("orphan_vdj", "orphan_vj"), has a second alpha chain with fewer
# UMIs ("extra_vj"), a non-productive alpha chain with more UMIs
# ("nonproductive_vj"), two more beta chains ("multichain"), or a heavy chain
# beside its T cell receptor ("ambiguous"). A cell without a clone has no
# contig ("none"), or one contig that Cell Ranger's all_contig_annotations.csv
# alone lists, outside a called cell ("outside_cell") or of low confidence
# ("low_confidence").
RECEPTOR_PROFILES = (
    "pair",
    "orphan_vdj",
    "orphan_vj",
    "extra_vj",
    "nonproductive_vj",
    "multichain",
    "ambiguous",
    "none",
    "outside_cell",
    "low_confidence",
)
# The gene calls that the chains of each locus draw from: V, D, J and C.
_GENE_CALLS = {
    "TRA": (
        ("TRAV1-2*01", "TRAV12-2*01", "TRAV29/DV5*01"),
        (),
        ("TRAJ33*01", "TRAJ20*01"),
        ("TRAC",),
    ),
    "TRB": (
        ("TRBV20-1*01", "TRBV6-5*01"),
        ("TRBD1*01", "TRBD2*01"),
        ("TRBJ2-7*01", "TRBJ1-1*01"),
        ("TRBC1", "TRBC2"),
    ),
    "IGH": (
        ("IGHV3-30*01", "IGHV1-69*01"),
        ("IGHD3-10*01",),
        ("IGHJ4*02", "IGHJ6*02"),
        ("IGHM", "IGHG1"),
    ),
    "IGK": (("IGKV1-39*01", "IGKV3-20*01"), (), ("IGKJ1*01", "IGKJ2*01"), ("IGKC",)),
    "IGL": (("IGLV2-14*01", "IGLV1-40*01"), (), ("IGLJ2*01", "IGLJ3*02"), ("IGLC2",)),
}
_AMINO_ACIDS = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))


@dataclass(frozen=True, slots=True)
class ReceptorTruth:
    """The receptor contigs of the truth's cells and what an import must give.

    Attributes:
        contigs: One row per contig: ``cell`` (the cell's position among the
            truth's cells, -1 for a barcode outside the store), ``sample``,
            ``barcode`` (as Cell Ranger writes it, with ``-1``),
            ``contig_id``, ``locus``, ``v_call``, ``d_call``, ``j_call``,
            ``c_call`` (None where a chain has no such gene), ``junction``,
            ``junction_aa``, ``productive``, ``umis``, ``reads``,
            ``is_cell``, ``high_confidence`` and ``sequence``.
        expected: One row per cell, indexed by cell ID, of what an import of
            all its contigs with ``productive_only=True`` gives:
            ``profile``, ``receptor_type``, ``receptor_subtype``,
            ``chain_pairing``, ``n_vj``, ``n_vdj``, and the junction
            amino-acid sequence of each slot, such as ``VDJ_1_junction_aa``,
            None for an empty slot.
    """

    contigs: pd.DataFrame
    expected: pd.DataFrame

    def sample_contigs(
        self,
        sample: str,
        *,
        receptor: str | None = None,
        all_contigs: bool = False,
    ) -> pd.DataFrame:
        """The contigs of one sample's barcodes that a V(D)J file lists.

        Args:
            sample: The sample.
            receptor: ``"TCR"`` for the TRA and TRB contigs, ``"BCR"`` for
                IGH, IGK and IGL, or None for both.
            all_contigs: List every contig, as ``all_contig_annotations.csv``
                does, rather than those of called cells with high confidence.
        """
        rows = self.contigs[self.contigs["sample"] == sample]
        if receptor is not None:
            loci = ("TRA", "TRB") if receptor == "TCR" else ("IGH", "IGK", "IGL")
            rows = rows[rows["locus"].isin(loci)]
        if not all_contigs:
            rows = rows[rows["is_cell"] & rows["high_confidence"]]
        return rows


def _receptor_profiles(truth: CloneTruth) -> np.ndarray:
    profiles = np.full(N_CELLS, "none", dtype=object)
    profiles[truth.clones >= 0] = "pair"
    for clone, changed in (
        (0, ("orphan_vdj", "extra_vj", "nonproductive_vj", "multichain", "ambiguous")),
        (1, ("orphan_vj",)),
        (6, ("orphan_vdj",)),
    ):
        members = np.flatnonzero(truth.clones == clone)
        profiles[members[: len(changed)]] = changed
    unclonal = np.flatnonzero(truth.clones < 0)
    profiles[unclonal[:2]] = ("outside_cell", "low_confidence")
    return profiles


def _expected_labels(profile: str, clone: int) -> tuple[str, str, str, int, int]:
    """The receptor type, subtype, chain pairing and chains of each arm.

    Written from scirpy's documented categories for each profile.
    """
    if profile in ("none", "outside_cell", "low_confidence"):
        return "no IR", "no IR", "no IR", 0, 0
    if profile == "multichain":
        return "multichain", "multichain", "multichain", 1, 3
    if profile == "ambiguous":
        return "ambiguous", "ambiguous", "ambiguous", 1, 2
    light = B_CELL_LIGHT_LOCI.get(clone)
    receptor_type = "TCR" if light is None else "BCR"
    subtype = "TRA+TRB" if light is None else f"IGH+{light}"
    if profile in ("pair", "nonproductive_vj"):
        return receptor_type, subtype, "single pair", 1, 1
    if profile == "orphan_vdj":
        return receptor_type, "TRA+TRB" if light is None else "IGH", "orphan VDJ", 0, 1
    if profile == "orphan_vj":
        return receptor_type, subtype, "orphan VJ", 1, 0
    if profile == "extra_vj":
        return receptor_type, subtype, "extra VJ", 2, 1
    raise ValueError(f"Unknown receptor profile {profile!r}")


class _ChainDraws:
    """Draws distinct junctions and gene calls of chains."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._junctions: set[str] = set()

    def chain(self, locus: str) -> dict[str, object]:
        """A productive chain of ``locus`` with its genes and junction."""
        while True:
            length = int(self._rng.integers(9, 16))
            junction_aa = (
                "C" + "".join(self._rng.choice(_AMINO_ACIDS, length - 2)) + "F"
            )
            if junction_aa not in self._junctions:
                self._junctions.add(junction_aa)
                break
        bases = "".join(self._rng.choice(list("ACGT"), 3 * (length - 2)))
        v_calls, d_calls, j_calls, c_calls = _GENE_CALLS[locus]
        return {
            "locus": locus,
            "v_call": str(self._rng.choice(v_calls)),
            "d_call": str(self._rng.choice(d_calls)) if d_calls else None,
            "j_call": str(self._rng.choice(j_calls)),
            "c_call": str(self._rng.choice(c_calls)),
            "junction": "TGT" + bases + "TTT",
            "junction_aa": junction_aa,
            "productive": True,
        }

    def mutated(self, chain: dict[str, object], rng: np.random.Generator) -> dict:
        """The chain with one substitution in its nucleotide and residue junctions.

        The junctions keep their conserved ends, and the residues stay
        distinct from every junction drawn.
        """
        bases = str(chain["junction"])
        position = int(rng.integers(3, len(bases) - 3))
        base = str(
            rng.choice([letter for letter in "ACGT" if letter != bases[position]])
        )
        residues = str(chain["junction_aa"])
        while True:
            at = int(rng.integers(1, len(residues) - 1))
            residue = str(
                rng.choice(
                    [letter for letter in _AMINO_ACIDS if letter != residues[at]]
                )
            )
            junction_aa = residues[:at] + residue + residues[at + 1 :]
            if junction_aa not in self._junctions:
                self._junctions.add(junction_aa)
                break
        return {
            **chain,
            "junction": bases[:position] + base + bases[position + 1 :],
            "junction_aa": junction_aa,
        }


def receptor_truth(
    truth: CloneTruth, seed: int = 0, *, somatic_mutations: bool = False
) -> ReceptorTruth:
    """Give the truth's clones receptor chains and each cell its contigs.

    Each cell's contigs follow its profile in ``RECEPTOR_PROFILES``. A clone's
    chains carry 8 to 20 UMIs in each cell, so a second alpha chain (3 UMIs),
    a non-productive one (40), extra beta chains (2 and 1) and a heavy chain
    beside a T cell receptor (4) rank as the expected slots say. Each barcode
    outside the store has one beta chain.

    With ``somatic_mutations``, the cells of each B cell clone after its
    first carry the clone's chains with one substitution in each junction,
    of nucleotides and of residues, as somatic hypermutation gives a B cell
    clone's later cells: identical junctions split such a clone, and
    junctions within one substitution keep it whole. The other draws stay
    those of the truth without mutations.
    """
    rng = np.random.default_rng([seed, 5])
    draws = _ChainDraws(rng)
    clone_chains = []
    for clone in range(truth.n_clones):
        light = B_CELL_LIGHT_LOCI.get(clone)
        vj, vdj = ("TRA", "TRB") if light is None else (light, "IGH")
        clone_chains.append((draws.chain(vj), draws.chain(vdj)))
    # Each mutated cell's chains, by cell.
    variants: dict[int, tuple[dict, dict]] = {}
    if somatic_mutations:
        mutations = np.random.default_rng([seed, 7])
        for clone in B_CELL_LIGHT_LOCI:
            later = np.flatnonzero(truth.clones == clone)[1:]
            if len(later):
                vj, vdj = clone_chains[clone]
                mutated = (draws.mutated(vj, mutations), draws.mutated(vdj, mutations))
                variants.update(dict.fromkeys(later.tolist(), mutated))
    profiles = _receptor_profiles(truth)
    rows: list[dict[str, object]] = []
    expected_rows = []

    def add(cell: int, sample: str, barcode: str, chain: dict, umis: int, **flags):
        rows.append(
            {
                "cell": cell,
                "sample": sample,
                "barcode": barcode,
                "contig_id": f"{barcode}_contig_{sum(row['barcode'] == barcode for row in rows) + 1}",
                **chain,
                "umis": umis,
                "reads": 37 * umis,
                "is_cell": flags.get("is_cell", True),
                "high_confidence": flags.get("high_confidence", True),
                "sequence": "".join(rng.choice(list("ACGT"), 30)),
            }
        )

    for cell in range(N_CELLS):
        sample = str(truth.samples[cell])
        barcode = f"{truth.barcodes[cell]}-1"
        profile = str(profiles[cell])
        clone = int(truth.clones[cell])
        slots: dict[str, str | None] = dict.fromkeys(RECEPTOR_SLOTS)
        if clone >= 0:
            vj, vdj = variants.get(cell, clone_chains[clone])
            if profile == "nonproductive_vj":
                add(
                    cell,
                    sample,
                    barcode,
                    {**draws.chain(vj["locus"]), "productive": False},
                    40,
                )
            if profile != "orphan_vdj":
                add(cell, sample, barcode, vj, int(rng.integers(8, 21)))
                slots["VJ_1"] = str(vj["junction_aa"])
            if profile != "orphan_vj":
                add(cell, sample, barcode, vdj, int(rng.integers(8, 21)))
                slots["VDJ_1"] = str(vdj["junction_aa"])
            if profile == "extra_vj":
                second = draws.chain(vj["locus"])
                add(cell, sample, barcode, second, 3)
                slots["VJ_2"] = str(second["junction_aa"])
            elif profile == "multichain":
                extra = [draws.chain(vdj["locus"]) for _ in range(2)]
                add(cell, sample, barcode, extra[0], 2)
                add(cell, sample, barcode, extra[1], 1)
                slots["VDJ_2"] = str(extra[0]["junction_aa"])
            elif profile == "ambiguous":
                heavy = draws.chain("IGH")
                add(cell, sample, barcode, heavy, 4)
                slots["VDJ_2"] = str(heavy["junction_aa"])
        elif profile == "outside_cell":
            add(cell, sample, barcode, draws.chain("TRB"), 2, is_cell=False)
        elif profile == "low_confidence":
            add(cell, sample, barcode, draws.chain("TRB"), 1, high_confidence=False)
        receptor_type, subtype, pairing, n_vj, n_vdj = _expected_labels(profile, clone)
        expected_rows.append(
            {
                "ids": str(truth.cell_ids[cell]),
                "profile": profile,
                "receptor_type": receptor_type,
                "receptor_subtype": subtype,
                "chain_pairing": pairing,
                "n_vj": n_vj,
                "n_vdj": n_vdj,
                **{f"{slot}_junction_aa": slots[slot] for slot in RECEPTOR_SLOTS},
            }
        )
    for barcode in truth.outside:
        add(-1, SAMPLES[0], f"{barcode}-1", draws.chain("TRB"), 5)
    return ReceptorTruth(
        contigs=pd.DataFrame(rows),
        expected=pd.DataFrame(expected_rows).set_index("ids"),
    )


# Cell Ranger's contig annotation columns, in the order it writes them.
_CELLRANGER_VDJ_COLUMNS = (
    "barcode",
    "is_cell",
    "contig_id",
    "high_confidence",
    "length",
    "chain",
    "v_gene",
    "d_gene",
    "j_gene",
    "c_gene",
    "full_length",
    "productive",
    "cdr3",
    "cdr3_nt",
    "reads",
    "umis",
    "raw_clonotype_id",
    "raw_consensus_id",
)


def write_cellranger_contigs(
    receptors: ReceptorTruth,
    path: Path,
    *,
    sample: str,
    receptor: str | None = None,
    all_contigs: bool = False,
    release: str = "5",
    compress: bool = False,
) -> Path:
    """Cell Ranger's contig annotations of one sample's barcodes.

    ``filtered_contig_annotations.csv`` lists the contigs of the barcodes that
    Cell Ranger calls cells, of high confidence; ``all_contigs=True`` writes
    ``all_contig_annotations.csv``, every contig. ``receptor`` keeps the
    contigs of ``"TCR"`` or ``"BCR"`` loci, as the ``vdj_t`` and ``vdj_b``
    outputs of ``cellranger multi`` hold them. Release ``"5"`` writes flags
    as ``true`` and ``false`` and a missing D gene empty; release ``"4"``
    writes ``True``, ``False`` and ``None``, as Cell Ranger 4.0.0 did.
    """
    true, false, missing = (
        ("true", "false", "") if release == "5" else ("True", "False", "None")
    )
    lines = [",".join(_CELLRANGER_VDJ_COLUMNS)]
    rows = receptors.sample_contigs(sample, receptor=receptor, all_contigs=all_contigs)
    for row in rows.itertuples(index=False):
        called = row.is_cell and row.high_confidence
        clonotype = f"clonotype{row.cell + 2}" if called else "None"
        fields = (
            row.barcode,
            true if row.is_cell else false,
            row.contig_id,
            true if row.high_confidence else false,
            "500",
            row.locus,
            row.v_call,
            missing if pd.isna(row.d_call) else row.d_call,
            row.j_call,
            row.c_call,
            true,
            true if row.productive else false,
            row.junction_aa,
            row.junction,
            str(row.reads),
            str(row.umis),
            clonotype,
            f"{clonotype}_consensus_1" if called else "None",
        )
        lines.append(",".join(fields))
    return _write(path, lines, compress=compress)


def write_airr_contigs(
    receptors: ReceptorTruth,
    path: Path,
    *,
    sample: str,
    with_locus: bool = True,
    compress: bool = False,
) -> Path:
    """An AIRR Rearrangement file of one sample's called barcodes.

    The file lists the contigs of the barcodes that Cell Ranger calls cells,
    of high confidence, as ``airr_rearrangement.tsv`` does: tab-separated,
    flags ``T`` and ``F``, a missing value empty, and each contig's sequence.
    Without ``locus``, an import infers each contig's locus from its gene
    calls.
    """
    columns = [
        "cell_id",
        "sequence_id",
        *(["locus"] if with_locus else []),
        "v_call",
        "d_call",
        "j_call",
        "c_call",
        "junction",
        "junction_aa",
        "productive",
        "umi_count",
        "consensus_count",
        "sequence",
    ]
    lines = ["\t".join(columns)]
    for row in receptors.sample_contigs(sample).itertuples(index=False):
        values = {
            "cell_id": row.barcode,
            "sequence_id": row.contig_id,
            "locus": row.locus,
            "v_call": row.v_call,
            "d_call": "" if pd.isna(row.d_call) else row.d_call,
            "j_call": row.j_call,
            "c_call": row.c_call,
            "junction": row.junction,
            "junction_aa": row.junction_aa,
            "productive": "T" if row.productive else "F",
            "umi_count": str(row.umis),
            "consensus_count": str(row.reads),
            "sequence": row.sequence,
        }
        lines.append("\t".join(str(values[name]) for name in columns))
    return _write(path, lines, compress=compress)


# Lineage barcodes of the truth's clones: clone k carries 1 + k % 3 barcodes
# of 20 bases, the longest that pycashier keeps, and each UMI 12 bases.
BARCODE_LENGTH = 20
UMI_LENGTH = 12
# The kinds of molecule: a clone's barcode in its cell ("signal"), a read of
# one with a sequencing error in its barcode ("error"), a clone's barcode in a
# cell without a clone ("ambient"), and a clone's barcode in a droplet that is
# not a cell of the store ("outside").
MOLECULE_KINDS = ("signal", "error", "ambient", "outside")


def _sequence(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


@dataclass(frozen=True, slots=True)
class BarcodeTruth:
    """The lineage barcode molecules of the truth's cells, with read-level noise.

    Attributes:
        clone_barcodes: Each clone's barcodes, by its rank.
        molecules: One row per molecule: ``cell`` (the cell's position among
            the truth's cells, -1 for a droplet outside the store),
            ``sample``, ``cell_barcode`` (its 16 bases, without ``-1``),
            ``umi``, ``barcode`` as sequenced, ``true_barcode`` as an error
            correction gives it, ``reads`` and ``kind``.
        unlabelled: The ``sample`` and ``cell_barcode`` of each cell without a
            molecule.
    """

    clone_barcodes: tuple[tuple[str, ...], ...]
    molecules: pd.DataFrame
    unlabelled: pd.DataFrame

    def sample_molecules(self, sample: str) -> pd.DataFrame:
        """The molecules of one sample's cells, and for ``d2`` of the droplets outside."""
        return self.molecules[self.molecules["sample"] == sample]

    def expected_molecules(
        self, truth: CloneTruth, *, corrected: bool = False
    ) -> pd.DataFrame:
        """The molecules of the store's cells that an import gives, sorted.

        Columns ``cell`` (the cell's ID), ``barcode``, ``umi`` and ``reads``;
        ``corrected`` gives each molecule its true barcode.
        """
        cells = self.molecules[self.molecules["cell"] >= 0]
        return (
            pd.DataFrame(
                {
                    "cell": truth.cell_ids[cells["cell"].to_numpy()],
                    "barcode": cells["true_barcode" if corrected else "barcode"],
                    "umi": cells["umi"],
                    "reads": cells["reads"].astype(np.int64),
                }
            )
            .sort_values(["cell", "barcode", "umi"])
            .reset_index(drop=True)
        )

    def expected_pairs(
        self, truth: CloneTruth, *, corrected: bool = False
    ) -> pd.DataFrame:
        """Each store cell's barcodes with their UMIs and reads, sorted."""
        molecules = self.expected_molecules(truth, corrected=corrected)
        return (
            molecules.groupby(["cell", "barcode"], sort=True)
            .agg(umis=("umi", "size"), reads=("reads", "sum"))
            .reset_index()
            .astype({"umis": np.int64, "reads": np.int64})
        )

    def reads(self, sample: str, seed: int = 0) -> pd.DataFrame:
        """One row per read of a sample's molecules, in a shuffled order."""
        molecules = self.sample_molecules(sample)
        rows = molecules.loc[molecules.index.repeat(molecules["reads"])]
        order = np.random.default_rng([seed, 13]).permutation(len(rows))
        return rows.iloc[order].reset_index(drop=True)


def barcode_truth(truth: CloneTruth, seed: int = 0) -> BarcodeTruth:
    """Give the truth's clones lineage barcodes and each cell its molecules.

    A cell of a clone has two to five UMIs of each of its clone's barcodes,
    each read 3 to 12 times; every other such cell also has a one-read
    molecule whose barcode carries a sequencing error. Every other cell
    without a clone has one ambient read of clone 0's barcode, and each
    droplet outside the store two molecules of it.
    """
    rng = np.random.default_rng([seed, 11])
    seen: set[str] = set()
    clone_barcodes = []
    for clone in range(truth.n_clones):
        drawn = []
        while len(drawn) < 1 + clone % 3:
            barcode = _sequence(rng, BARCODE_LENGTH)
            if barcode not in seen:
                seen.add(barcode)
                drawn.append(barcode)
        clone_barcodes.append(tuple(drawn))
    rows: list[dict[str, object]] = []
    used: dict[str, set[str]] = {}

    def add(cell: int, sample: str, cell_barcode: str, barcode: str, **values):
        taken = used.setdefault(cell_barcode, set())
        while True:
            umi = _sequence(rng, UMI_LENGTH)
            if umi not in taken:
                taken.add(umi)
                break
        rows.append(
            {
                "cell": cell,
                "sample": sample,
                "cell_barcode": cell_barcode,
                "umi": umi,
                "barcode": values.get("sequenced", barcode),
                "true_barcode": barcode,
                "reads": values["reads"],
                "kind": values["kind"],
            }
        )

    for cell in range(N_CELLS):
        sample = str(truth.samples[cell])
        cell_barcode = str(truth.barcodes[cell])
        clone = int(truth.clones[cell])
        if clone < 0:
            if cell % 2 == 0:
                add(
                    cell,
                    sample,
                    cell_barcode,
                    clone_barcodes[0][0],
                    reads=1,
                    kind="ambient",
                )
            continue
        for barcode in clone_barcodes[clone]:
            for _ in range(int(rng.integers(2, 6))):
                add(
                    cell,
                    sample,
                    cell_barcode,
                    barcode,
                    reads=int(rng.integers(3, 13)),
                    kind="signal",
                )
        if cell % 2:
            barcode = clone_barcodes[clone][0]
            position = int(rng.integers(0, BARCODE_LENGTH))
            base = "ACGT"[("ACGT".index(barcode[position]) + 1) % 4]
            add(
                cell,
                sample,
                cell_barcode,
                barcode,
                sequenced=barcode[:position] + base + barcode[position + 1 :],
                reads=1,
                kind="error",
            )
    for barcode in truth.outside:
        for _ in range(2):
            add(
                -1,
                SAMPLES[0],
                str(barcode),
                clone_barcodes[0][0],
                reads=4,
                kind="outside",
            )
    molecules = pd.DataFrame(rows)
    labelled = set(molecules["cell"])
    quiet = [cell for cell in range(N_CELLS) if cell not in labelled]
    return BarcodeTruth(
        clone_barcodes=tuple(clone_barcodes),
        molecules=molecules,
        unlabelled=pd.DataFrame(
            {"sample": truth.samples[quiet], "cell_barcode": truth.barcodes[quiet]}
        ),
    )


def write_trex_molecules(
    barcodes: BarcodeTruth, path: Path, *, sample: str, generation: str = "current"
) -> Path:
    """TREX's molecule tables of one sample's cells, named by bare barcodes.

    ``generation`` is ``"current"`` (``molecules.txt`` with ``read_count``),
    ``"corrected"`` (``molecules_corrected.txt``, whose ``clone_id`` is the
    corrected barcode), ``"2022"`` (without ``read_count``) or ``"reads"``
    (``reads.txt``, one row per read, under the header of 2022's molecules).
    """
    molecules = barcodes.sample_molecules(sample)
    if generation == "reads":
        rows = barcodes.reads(sample)
        lines = ["#cell_id\tumi\tclone_id"] + [
            f"{row.cell_barcode}\t{row.umi}\t{row.barcode}"
            for row in rows.itertuples(index=False)
        ]
        return _write(path, lines)
    if generation == "corrected":
        header = "#cell_id\tumi\toriginal_clone_id\tclone_id\tread_count"
        lines = [header] + [
            f"{row.cell_barcode}\t{row.umi}\t{row.barcode}\t{row.true_barcode}\t"
            f"{row.reads}"
            for row in molecules.itertuples(index=False)
        ]
        return _write(path, lines)
    with_reads = generation == "current"
    if generation not in ("current", "2022"):
        raise ValueError("generation is current, corrected, 2022 or reads")
    header = "#cell_id\tumi\tclone_id" + ("\tread_count" if with_reads else "")
    lines = [header] + [
        f"{row.cell_barcode}\t{row.umi}\t{row.barcode}"
        + (f"\t{row.reads}" if with_reads else "")
        for row in molecules.sort_values(["cell_barcode", "barcode", "umi"]).itertuples(
            index=False
        )
    ]
    return _write(path, lines)


def write_pycashier(
    barcodes: BarcodeTruth, path: Path, *, sample: str, header: bool = True
) -> Path:
    """pycashier's per-read table of one sample, ``-1`` stripped from each cell.

    Releases since 2024.1001 write a header; earlier ones wrote none.
    """
    rows = barcodes.reads(sample)
    lines = (["info\tumi\tcell\tbarcode"] if header else []) + [
        f"@A00842:201:H3JGCDSX3:3:{index}:{1000 + index}:{2000 + index}\t{row.umi}\t"
        f"{row.cell_barcode}\t{row.barcode}"
        for index, row in enumerate(rows.itertuples(index=False))
    ]
    return _write(path, lines)


def write_celltag_reads(
    barcodes: BarcodeTruth,
    path: Path,
    *,
    sample: str,
    assay: str = "RNA",
    biddy: bool = False,
) -> Path:
    """CellTag-multi's per-read table of one sample, gzip-compressed, ``-1`` kept.

    ``assay="ATAC"`` writes no UMIs, as scATAC-seq has none; ``biddy`` writes
    the Biddy et al. workflow's columns.
    """
    rows = barcodes.reads(sample, seed=1)
    if biddy:
        lines = ["Read.ID\tRead.Seq\tCell.BC\tUMI\tCell.Tag\tGene"] + [
            f"read{index}\tACGT{row.barcode}\t{row.cell_barcode}-1\t{row.umi}\t"
            f"{row.barcode}\tGFP"
            for index, row in enumerate(rows.itertuples(index=False))
        ]
    elif assay == "ATAC":
        lines = ["Cell.BC\tCell.Tag"] + [
            f"{row.cell_barcode}-1\t{row.barcode}"
            for row in rows.itertuples(index=False)
        ]
    else:
        lines = ["Cell.BC\tUMI\tCell.Tag"] + [
            f"{row.cell_barcode}-1\t{row.umi}\t{row.barcode}"
            for row in rows.itertuples(index=False)
        ]
    return _write(path, lines, compress=True)


def write_fatemap(
    barcodes: BarcodeTruth, path: Path, *, sample: str, header: bool = True
) -> Path:
    """FateMap's ``stepThreeStarcodeShavedReads.txt`` of one sample.

    One row per molecule and original barcode, read by its corrected 50-base
    column; the first molecule is listed twice, with two original barcodes,
    as starcode leaves molecules that differed in their raw barcode.
    """
    molecules = barcodes.sample_molecules(sample)
    lines = ["\t".join(_FATEMAP_COLUMNS)] if header else []
    for index, row in enumerate(molecules.itertuples(index=False)):
        copies = 2 if index == 0 else 1
        for copy in range(copies):
            original = row.barcode if copy == 0 else row.barcode[::-1]
            lines.append(
                "\t".join(
                    [
                        row.cell_barcode,
                        row.umi,
                        original,
                        row.true_barcode,
                        row.true_barcode[:16],
                        row.true_barcode[:12],
                        row.true_barcode[:10],
                        "1",
                    ]
                )
            )
    return _write(path, lines)


_FATEMAP_COLUMNS = (
    "cellID",
    "UMI",
    "OriginalBarcode",
    "BC50StarcodeD8",
    "BC40StarcodeD8",
    "BC30StarcodeD8",
    "BC30StarcodeD6",
    "SampleNum",
)


def _sample_pairs(barcodes: BarcodeTruth, sample: str) -> pd.DataFrame:
    """UMIs and reads of each cell and sequenced barcode of one sample."""
    molecules = barcodes.sample_molecules(sample)
    return (
        molecules.groupby(["cell_barcode", "barcode"], sort=True)
        .agg(umis=("umi", "size"), reads=("reads", "sum"))
        .reset_index()
    )


def write_quicat(barcodes: BarcodeTruth, path: Path, *, sample: str) -> Path:
    """QuiCAT's per-sample ``<sample>_output.csv``: UMIs per cell and barcode.

    Cells keep 10x's ``-1``, and the sample's cells without barcodes have a
    row with an empty barcode and count, as QuiCAT's exploded lists write.
    """
    pairs = _sample_pairs(barcodes, sample)
    lines = ["sample,cell,barcode,counts"] + [
        f"{sample},{row.cell_barcode}-1,{row.barcode},{row.umis}"
        for row in pairs.itertuples(index=False)
    ]
    quiet = barcodes.unlabelled[barcodes.unlabelled["sample"] == sample]
    lines += [f"{sample},{cell}-1,," for cell in quiet["cell_barcode"]]
    return _write(path, lines)


def _merged_pairs(barcodes: BarcodeTruth) -> tuple[pd.DataFrame, list[str]]:
    """Every sample's pairs with QuiCAT's merged cell names, and the barcodes."""
    frames = []
    for sample in SAMPLES:
        pairs = _sample_pairs(barcodes, sample)
        pairs["cell"] = pairs["cell_barcode"] + f"-1_{sample}"
        pairs["sample"] = sample
        frames.append(pairs)
    merged = pd.concat(frames, ignore_index=True)
    return merged, sorted(merged["barcode"].unique())


def write_quicat_wide(barcodes: BarcodeTruth, path: Path) -> Path:
    """QuiCAT's merged ``barcodes_output.csv``: one column per barcode, zeros filled."""
    merged, names = _merged_pairs(barcodes)
    wide = merged.pivot_table(
        index=["sample", "cell"], columns="barcode", values="umis", fill_value=0
    ).reindex(columns=names, fill_value=0)
    lines = ["sample,cell," + ",".join(names)] + [
        f"{sample},{cell}," + ",".join(str(int(value)) for value in values)
        for (sample, cell), values in zip(wide.index, wide.to_numpy(), strict=True)
    ]
    return _write(path, lines)


def write_quicat_h5ad(
    barcodes: BarcodeTruth,
    path: Path,
    *,
    layout: str = "csr",
    with_anndata: bool = False,
) -> Path:
    """QuiCAT's ``barcodes_output.h5ad``: cells by barcodes, float UMI counts.

    With ``with_anndata``, anndata itself writes the file; current releases
    store each index as a nullable string array, a group of values and a
    mask, where the file written here holds a dataset.
    """
    merged, names = _merged_pairs(barcodes)
    cells = sorted(merged["cell"].unique())
    dense = np.zeros((len(cells), len(names)), dtype=np.float64)
    rows = pd.Index(cells).get_indexer(merged["cell"])
    columns = pd.Index(names).get_indexer(merged["barcode"])
    dense[rows, columns] = merged["umis"].to_numpy(dtype=np.float64)
    if with_anndata:
        import anndata
        from scipy import sparse

        samples = pd.Categorical([cell.rsplit("_", 1)[1] for cell in cells])
        matrices = {
            "dense": dense,
            "csr": sparse.csr_matrix(dense),
            "csc": sparse.csc_matrix(dense),
        }
        # Releases of anndata before pandas 3 write nullable strings, as current
        # releases always do, only once that is allowed.
        with anndata.settings.override(allow_write_nullable_strings=True):
            anndata.AnnData(
                X=matrices[layout],
                obs=pd.DataFrame(
                    {"sample": samples}, index=pd.Index(cells, dtype="string")
                ),
                var=pd.DataFrame(index=pd.Index(names, dtype="string")),
            ).write_h5ad(path)
        return path
    with h5py.File(path, "w") as handle:
        obs = _obs_frame(handle, cells)
        write_categorical(obs, "sample", [cell.rsplit("_", 1)[1] for cell in cells])
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.attrs["column-order"] = np.asarray([], dtype=h5py.string_dtype())
        var.create_dataset("_index", data=np.asarray(names, dtype=h5py.string_dtype()))
        if layout == "dense":
            handle.create_dataset("X", data=dense)
            return path
        major = dense if layout == "csr" else dense.T
        group = handle.create_group("X")
        group.attrs["encoding-type"] = f"{layout}_matrix"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = np.asarray(dense.shape)
        indptr = np.concatenate([[0], np.cumsum((major != 0).sum(axis=1))])
        indices = np.concatenate([np.flatnonzero(line) for line in major])
        group.create_dataset("data", data=major[major != 0])
        group.create_dataset("indices", data=indices.astype(np.int32))
        group.create_dataset("indptr", data=indptr.astype(np.int64))
    return path


def write_bartab_counts(barcodes: BarcodeTruth, path: Path, *, sample: str) -> Path:
    """BARtab's ``counts.tsv`` of one sample: UMIs per corrected barcode and cell."""
    molecules = barcodes.sample_molecules(sample)
    pairs = (
        molecules.groupby(["true_barcode", "cell_barcode"], sort=True)["umi"]
        .size()
        .reset_index()
    )
    lines = ["gene\tcell\tcount"] + [
        f"{row.true_barcode}\t{row.cell_barcode}\t{row.umi}"
        for row in pairs.itertuples(index=False)
    ]
    return _write(path, lines, compress=True)


def write_bartab_annotation(
    barcodes: BarcodeTruth, path: Path, *, sample: str, legacy: bool = False
) -> Path:
    """BARtab's cell barcode annotation of one sample: one row per cell.

    The header names two of the three fields, R's row-name layout; barcodes
    and their UMIs are joined by ``;``, or, as in early files, by ``,`` in a
    table separated by spaces, whose cells keep ``-1``.
    """
    molecules = barcodes.sample_molecules(sample)
    pairs = (
        molecules.groupby(["cell_barcode", "true_barcode"], sort=True)["umi"]
        .size()
        .reset_index()
    )
    separator, joiner = (" ", ",") if legacy else ("\t", ";")
    suffix = "-1" if legacy else ""
    lines = [separator.join(["barcode", "bc.umi.count"])]
    for cell, rows in pairs.groupby("cell_barcode", sort=True):
        lines.append(
            separator.join(
                [
                    f"{cell}{suffix}",
                    joiner.join(rows["true_barcode"]),
                    joiner.join(str(count) for count in rows["umi"]),
                ]
            )
        )
    return _write(path, lines)


def pack_umis(umis: Sequence[str]) -> np.ndarray:
    """Pack UMIs two bits per base, the last base lowest, as Cell Ranger does."""
    values = np.zeros(len(umis), dtype=np.uint32)
    for index, umi in enumerate(umis):
        value = 0
        for base in umi:
            value = value * 4 + "ACGT".index(base)
        values[index] = value
    return values


def write_molecule_info(
    barcodes: BarcodeTruth,
    path: Path,
    *,
    sample: str,
    library_type: str = "Custom",
    umi_length: int | None = UMI_LENGTH,
) -> Path:
    """A Cell Ranger ``molecule_info.h5`` of one sample in its documented layout.

    Gene expression molecules come first in each cell, then the lineage
    library's, whose features are the corrected barcodes, named ``LB<k>``
    with their sequence. Barcodes are stored without ``-1``; each molecule
    records GEM well 1. ``umi_length`` is recorded in ``metrics_json``, or
    left out with None.
    """
    import json

    molecules = barcodes.sample_molecules(sample)
    lineage = sorted(molecules["true_barcode"].unique())
    feature_ids = ["ENSG0001", "ENSG0002"] + [
        f"LB{index}" for index in range(len(lineage))
    ]
    feature_types = ["Gene Expression"] * 2 + [library_type] * len(lineage)
    sequences = ["", ""] + lineage
    cell_barcodes = sorted(molecules["cell_barcode"].unique())
    barcode_index = {barcode: index for index, barcode in enumerate(cell_barcodes)}
    rows = []
    for cell_barcode, cell_molecules in molecules.groupby("cell_barcode", sort=True):
        # One gene expression molecule of each cell, in library 0.
        rows.append((barcode_index[cell_barcode], "A" * UMI_LENGTH, 0, 7, 0))
        for row in cell_molecules.itertuples(index=False):
            rows.append(
                (
                    barcode_index[cell_barcode],
                    row.umi,
                    2 + lineage.index(row.true_barcode),
                    int(row.reads),
                    1,
                )
            )
    table = pd.DataFrame(
        rows, columns=["barcode", "umi", "feature", "count", "library"]
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["file_version"] = 5
        handle.attrs["filetype"] = "molecule"
        handle.create_dataset("barcode_idx", data=table["barcode"].to_numpy(np.uint64))
        handle.create_dataset("umi", data=pack_umis(table["umi"].tolist()))
        handle.create_dataset("feature_idx", data=table["feature"].to_numpy(np.uint32))
        handle.create_dataset("count", data=table["count"].to_numpy(np.uint32))
        handle.create_dataset("gem_group", data=np.ones(len(table), dtype=np.uint16))
        handle.create_dataset("library_idx", data=table["library"].to_numpy(np.uint16))
        handle.create_dataset("umi_type", data=np.zeros(len(table), dtype=np.uint32))
        handle.create_dataset("barcodes", data=np.asarray(cell_barcodes, dtype="S16"))
        features = handle.create_group("features")
        features.create_dataset("id", data=np.asarray(feature_ids, dtype="S16"))
        features.create_dataset("name", data=np.asarray(feature_ids, dtype="S16"))
        features.create_dataset(
            "feature_type", data=np.asarray(feature_types, dtype="S32")
        )
        features.create_dataset("sequence", data=np.asarray(sequences, dtype="S32"))
        libraries = [
            {"library_id": 0, "library_type": "Gene Expression", "gem_group": 1},
            {"library_id": 1, "library_type": library_type, "gem_group": 1},
        ]
        handle.create_dataset(
            "library_info", data=np.asarray([json.dumps(libraries)], dtype="S4096")
        )
        metrics: dict[str, object] = {"cellranger_version": "cellranger-8.0.0"}
        if umi_length is not None:
            metrics["chemistry_umi"] = [{"length": umi_length, "read_type": "R1"}]
        handle.create_dataset(
            "metrics_json", data=json.dumps(metrics), dtype=h5py.string_dtype()
        )
    return path


# The barcodes of each clone of the truth for barcode calling, by rank: clone
# 1 carries one barcode, so a method that needs two per cell leaves it out.
CALLING_BARCODES = (4, 1, 5, 2, 4, 3, 4, 2)
# The noise that each preset's rules must remove from the calling truth.
CALLING_NOISE = (
    "amplified",
    "ambient",
    "dropout",
    "indel",
    "partial",
    "contaminant",
    "crowded",
)
# Barcodes that the crowded cells carry, more than any CellTag preset allows.
N_CROWDED_BARCODES = 30
N_CROWDED_CELLS = 3


@dataclass(frozen=True, slots=True)
class CallingTruth:
    """Lineage barcode molecules of the truth's clones, with noise to remove.

    Attributes:
        barcodes: The molecules, in the layout of ``barcode_truth``, so that
            its writers emit them in each tool's format.
        allowlist: The barcodes of the library: the clones' and the crowded
            cells', and neither their errors nor the contaminant.
        noise: The kinds of noise that the molecules hold.
    """

    barcodes: BarcodeTruth
    allowlist: tuple[str, ...]
    noise: tuple[str, ...]

    def expected_partition(
        self, truth: CloneTruth, *, min_barcodes: int = 1, min_size: int = 1
    ) -> set[frozenset[str]]:
        """The truth's clones that a preset calls, as sets of cell IDs.

        A clone is called when it carries at least ``min_barcodes`` barcodes
        and holds at least ``min_size`` cells; its cells are those of the
        truth, whatever noise they carry.
        """
        ids = truth.cell_ids
        called = set()
        for clone, members in enumerate(self.barcodes.clone_barcodes):
            cells = np.flatnonzero(truth.clones == clone)
            if len(members) >= min_barcodes and len(cells) >= min_size:
                called.add(frozenset(str(ids[cell]) for cell in cells))
        return called


def _variant(rng: np.random.Generator, barcode: str, kind: str) -> str:
    """A sequencing or PCR error of a barcode: a substitution, or an indel."""
    position = int(rng.integers(1, len(barcode) - 1))
    if kind == "substitution":
        base = "ACGT"[("ACGT".index(barcode[position]) + 1) % 4]
        return barcode[:position] + base + barcode[position + 1 :]
    # A deletion and a base added at the end keep the length, two edits away.
    base = "ACGT"[("ACGT".index(barcode[-1]) + 2) % 4]
    return barcode[:position] + barcode[position + 1 :] + base


def calling_truth(
    truth: CloneTruth,
    *,
    noise: Sequence[str] = ("amplified", "ambient"),
    barcode_length: int = 28,
    seed: int = 0,
) -> CallingTruth:
    """Give the truth's clones barcodes and each cell its molecules, with noise.

    Clone ``k`` carries ``CALLING_BARCODES[k]`` barcodes. Each cell of a
    clone holds four to seven UMIs of each, read 12 to 30 times, which pass
    every preset's thresholds. Every cell of a clone also holds a one-read
    molecule of a barcode one substitution from one of its own, and the
    kinds of ``noise`` add:

    - ``"amplified"``: a molecule of 12 reads one substitution away, a PCR
      error that read filters keep, in every other cell of a clone;
    - ``"ambient"``: a one-read molecule of clone 0's barcode in every other
      cell without a clone and in every third cell of another clone;
    - ``"dropout"``: every third cell of a clone of four or more barcodes
      lacks the clone's last barcode;
    - ``"indel"``: a molecule of 3 reads two edits from one of its barcodes,
      in every other cell of a clone;
    - ``"partial"``: a molecule of 4 reads whose barcode's last six bases are
      ``-``, as TREX writes bases that no read covered, in every other cell
      of a clone;
    - ``"contaminant"``: three cells without a clone hold five UMIs of each
      of two barcodes outside the allowlist;
    - ``"crowded"``: three cells without a clone hold five UMIs of each of
      the same 30 allowlisted barcodes.
    """
    unknown = sorted(set(noise) - set(CALLING_NOISE))
    if unknown:
        raise ValueError(f"unknown noise {unknown}")
    rng = np.random.default_rng([seed, 23])
    seen: set[str] = set()

    def drawn() -> str:
        while True:
            barcode = _sequence(rng, barcode_length)
            if barcode not in seen:
                seen.add(barcode)
                return barcode

    clone_barcodes = tuple(
        tuple(drawn() for _ in range(count))
        for count in CALLING_BARCODES[: truth.n_clones]
    )
    crowded = tuple(drawn() for _ in range(N_CROWDED_BARCODES))
    contaminants = (drawn(), drawn())
    rows: list[dict[str, object]] = []
    used: dict[str, set[str]] = {}

    def add(
        cell: int, barcode: str, reads: int, kind: str, sequenced: str = ""
    ) -> None:
        cell_barcode = str(truth.barcodes[cell])
        taken = used.setdefault(cell_barcode, set())
        while True:
            umi = _sequence(rng, UMI_LENGTH)
            if umi not in taken:
                taken.add(umi)
                break
        rows.append(
            {
                "cell": cell,
                "sample": str(truth.samples[cell]),
                "cell_barcode": cell_barcode,
                "umi": umi,
                "barcode": sequenced or barcode,
                "true_barcode": barcode,
                "reads": reads,
                "kind": kind,
            }
        )

    unclonal = np.flatnonzero(truth.clones < 0).tolist()
    for cell in range(N_CELLS):
        clone = int(truth.clones[cell])
        if clone < 0:
            continue
        members = clone_barcodes[clone]
        rank = int(np.count_nonzero(np.flatnonzero(truth.clones == clone) < cell))
        carried = members
        if "dropout" in noise and len(members) >= 4 and rank % 3 == 1:
            carried = members[:-1]
        for barcode in carried:
            for _ in range(int(rng.integers(4, 8))):
                add(cell, barcode, int(rng.integers(12, 31)), "signal")
        source = carried[int(rng.integers(0, len(carried)))]
        # Each error of a cell is a barcode of its own, of one UMI.
        variants: set[str] = set()

        def variant(kind: str) -> str:
            while True:
                drawn_variant = _variant(rng, source, kind)
                if drawn_variant not in variants:
                    variants.add(drawn_variant)
                    return drawn_variant

        add(cell, source, 1, "error", variant("substitution"))
        if "amplified" in noise and rank % 2 == 0:
            add(cell, source, 12, "amplified", variant("substitution"))
        if "indel" in noise and rank % 2 == 1:
            add(cell, source, 3, "indel", variant("indel"))
        if "partial" in noise and rank % 2 == 1:
            add(cell, source, 4, "partial", source[:-6] + "-" * 6)
        if "ambient" in noise and clone != 0 and rank % 3 == 2:
            add(cell, clone_barcodes[0][0], 1, "ambient")
    for position, cell in enumerate(unclonal):
        if "ambient" in noise and position % 2 == 0:
            add(cell, clone_barcodes[0][0], 1, "ambient")
    chosen = unclonal[1::2]
    if "contaminant" in noise:
        for cell in chosen[:N_CROWDED_CELLS]:
            for contaminant in contaminants:
                for _ in range(5):
                    add(cell, contaminant, 20, "contaminant")
    if "crowded" in noise:
        for cell in chosen[N_CROWDED_CELLS : 2 * N_CROWDED_CELLS]:
            for barcode in crowded:
                for _ in range(5):
                    add(cell, barcode, 20, "crowded")
    molecules = pd.DataFrame(rows)
    labelled = set(molecules["cell"])
    quiet = [cell for cell in range(N_CELLS) if cell not in labelled]
    return CallingTruth(
        barcodes=BarcodeTruth(
            clone_barcodes=clone_barcodes,
            molecules=molecules,
            unlabelled=pd.DataFrame(
                {"sample": truth.samples[quiet], "cell_barcode": truth.barcodes[quiet]}
            ),
        ),
        allowlist=tuple(sorted({b for c in clone_barcodes for b in c} | set(crowded))),
        noise=tuple(noise),
    )


# Lineage recorders -----------------------------------------------------------

#: PEtracer's eight lineage marks at each site, the 5-mers that Koblan, Yost
#: et al. (Science 2025) insert, written out here so that the reader's own
#: table is checked against them.
PETRACER_MARKS = {
    "RNF2": ("ACAGT", "ACTTA", "TTCCT", "TATAT", "GTTCA", "TGCCA", "TCCAA", "ACTCC"),
    "HEK3": ("GATAG", "AATCG", "GCAAG", "GCGCC", "CTTTG", "ATCAA", "CTCTC", "ATTTA"),
    "EMX1": ("GGACA", "ACAAT", "CCCTA", "AGTAC", "CCGAT", "CCTTT", "ATCAA", "ATTCG"),
}
PETRACER_SITES = ("RNF2", "HEK3", "EMX1")
PETRACER_INTEGRATIONS = ("intID101", "intID205", "intID318", "intID4")
# An insertion outside the eight marks, as PEtracer's kinetics pegRNA writes.
PETRACER_OTHER = "GTCAG"
#: DNA Typewriter's tapes of 2022 and their slots: three contracted ones, as
#: in the published HEK293T line.
TYPEWRITER_TAPES = {"ATTTGGTT": 5, "TGCGATTT": 5, "TGGACGAC": 4, "TTCACGTA": 2}
TYPEWRITER_SYMBOLS = ("CCGGGA", "TATGGA", "CATGGA", "GCCGGA", "ATAGGA", "GATGGA")
#: DNA Typewriter's tapes of 2026: six slots, one contracted to four.
CHAIN_TAPES = {"CGGGGAATTGTA": 6, "TCCGGAAGACCC": 6, "GTGTAAATCGGC": 4}
CHAIN_SYMBOLS = ("AAG", "GCC", "CAC", "CCC", "GATG", "ACT")
DARLIN_ARRAYS = ("CA", "TA", "RA")
# CARLIN events, as Custom_CARLIN writes them, of which alleles are built.
_DARLIN_EVENTS = (
    "23_268del",
    "22_23insTA",
    "50_130del",
    "103A>T",
    "185_238del",
    "23_265delinsG",
    "77_83del",
    "212_267del",
    "9_35del",
    "158_161del",
)


def recorder_counts(cell: int, integration: int) -> tuple[int, int]:
    """The UMIs and reads that every writer gives a cell's integration."""
    umis = 2 + (cell + 3 * integration) % 9
    return umis, 10 * umis + cell % 5


@dataclass(frozen=True, slots=True)
class RecorderTruth:
    """The edits of the truth's cells on three kinds of recorder, drawn along clones.

    Attributes:
        independent: PEtracer-like sites, one row per cell, integration and
            site: ``cell`` (its position among the truth's cells),
            ``integration``, ``site``, ``read`` (whether the integration was
            read), ``state`` (``"unedited"``, ``"mark"``, ``"unknown_mark"``
            or ``"missing"``) and ``mark`` (``1`` to ``8``, 0 for none).
        second: A second, weaker allele of the first integration of some
            cells, in the columns of ``independent``.
        tapes: DNA Typewriter's tapes of 2022, one row per cell and tape:
            ``cell``, ``integration``, ``edits`` (its edited slots, -1 when
            the tape was not read) and ``symbols``, the symbols of its slots
            in order.
        chains: DNA Typewriter's tapes of 2026, in the columns of ``tapes``.
        arrays: DARLIN's arrays, one row per cell and array: ``cell``,
            ``integration``, ``read`` and ``allele`` (``[]`` unedited, ``X``
            an unknown allele).
    """

    independent: pd.DataFrame
    second: pd.DataFrame
    tapes: pd.DataFrame
    chains: pd.DataFrame
    arrays: pd.DataFrame


def recorder_truth(truth: CloneTruth, seed: int = 0) -> RecorderTruth:
    """Draw the truth's recorder edits: a clone's cells share its founder's edits.

    A clone's founder edits each site with a chance of one half, each of its
    cells adds edits of its own to unedited sites with a chance of one
    fifth, and a cell without a clone draws every site as a founder does.
    Every seventh cell's private edit at its first PEtracer site, where it
    has one, is an insertion outside the eight marks. Tapes extend their
    founder's chain by up to two symbols per cell. Each integration of a cell
    is not read with a chance of one tenth, and each PEtracer site of a read
    integration with a chance of one in twenty.
    """
    rng = np.random.default_rng([seed, 19])
    n_clones = truth.n_clones
    founders_marks = rng.integers(1, 9, size=(n_clones, 4, 3))
    founders_edited = rng.random((n_clones, 4, 3)) < 0.5
    founder_chains = {
        name: [
            list(rng.choice(TYPEWRITER_SYMBOLS, size=int(rng.integers(0, length))))
            for _clone in range(n_clones)
        ]
        for name, length in TYPEWRITER_TAPES.items()
    }
    founder_symbols = {
        name: [
            list(rng.choice(CHAIN_SYMBOLS, size=int(rng.integers(0, length))))
            for _clone in range(n_clones)
        ]
        for name, length in CHAIN_TAPES.items()
    }
    founder_events = [
        [
            sorted(rng.choice(len(_DARLIN_EVENTS), size=2, replace=False).tolist())
            for _array in DARLIN_ARRAYS
        ]
        for _clone in range(n_clones)
    ]
    independent, second, tapes, chains, arrays = [], [], [], [], []
    for cell in range(N_CELLS):
        clone = int(truth.clones[cell])
        for k, integration in enumerate(PETRACER_INTEGRATIONS):
            read = bool(rng.random() >= 0.1)
            for s, site in enumerate(PETRACER_SITES):
                if clone >= 0 and founders_edited[clone, k, s]:
                    state, mark = "mark", int(founders_marks[clone, k, s])
                elif rng.random() < (0.2 if clone >= 0 else 0.5):
                    state, mark = "mark", int(rng.integers(1, 9))
                    if cell % 7 == 0 and k == 0 and s == 0:
                        state, mark = "unknown_mark", 0
                else:
                    state, mark = "unedited", 0
                if read and rng.random() < 0.05:
                    state, mark = "missing", 0
                independent.append(
                    {
                        "cell": cell,
                        "integration": integration,
                        "site": site,
                        "read": read,
                        "state": state,
                        "mark": mark,
                    }
                )
            if k == 0 and read and cell % 5 == 0:
                for s, site in enumerate(PETRACER_SITES):
                    second.append(
                        {
                            "cell": cell,
                            "integration": integration,
                            "site": site,
                            "read": True,
                            "state": "mark",
                            "mark": 1 + (cell + s) % 8,
                        }
                    )
        for name, length in TYPEWRITER_TAPES.items():
            chain = list(founder_chains[name][clone]) if clone >= 0 else []
            chain += list(rng.choice(TYPEWRITER_SYMBOLS, size=int(rng.integers(0, 3))))
            chain = chain[:length]
            read = bool(rng.random() >= 0.1)
            tapes.append(
                {
                    "cell": cell,
                    "integration": name,
                    "edits": len(chain) if read else -1,
                    "symbols": tuple(str(symbol) for symbol in chain),
                }
            )
        for name, length in CHAIN_TAPES.items():
            chain = list(founder_symbols[name][clone]) if clone >= 0 else []
            chain += list(rng.choice(CHAIN_SYMBOLS, size=int(rng.integers(0, 3))))
            chain = chain[:length]
            read = bool(rng.random() >= 0.1)
            chains.append(
                {
                    "cell": cell,
                    "integration": name,
                    "edits": len(chain) if read else -1,
                    "symbols": tuple(str(symbol) for symbol in chain),
                }
            )
        for a, array in enumerate(DARLIN_ARRAYS):
            events = set(founder_events[clone][a]) if clone >= 0 else set()
            if rng.random() < 0.3:
                events.add(int(rng.integers(0, len(_DARLIN_EVENTS))))
            allele = (
                ",".join(_DARLIN_EVENTS[event] for event in sorted(events))
                if events
                else "[]"
            )
            if cell % 11 == 3 and a == 0:
                allele = "X"
            arrays.append(
                {
                    "cell": cell,
                    "integration": array,
                    "read": bool(rng.random() >= 0.15),
                    "allele": allele,
                }
            )
    return RecorderTruth(
        independent=pd.DataFrame(independent),
        second=pd.DataFrame(second),
        tapes=pd.DataFrame(tapes),
        chains=pd.DataFrame(chains),
        arrays=pd.DataFrame(arrays),
    )


def _records(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Expected records, in the columns an import's records are compared in."""
    columns = ["cell", "integration", "site", "state", "allele", "umis", "reads"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(columns[:5], na_position="first").reset_index(drop=True)


def _site_token(state: str, mark: int, site: str, cell: int) -> str:
    """A PEtracer site as its 2025 tables write it."""
    if state == "mark":
        return PETRACER_MARKS[site][mark - 1]
    if state == "unknown_mark":
        return PETRACER_OTHER
    if state == "missing":
        return ""
    # An unedited EMX1 reads CTTGGG in every other cell.
    return "CTTGGG" if site == "EMX1" and cell % 2 else "None"


def write_petracer_long(
    recorders: RecorderTruth, truth: CloneTruth, path: Path, *, sample: str
) -> Path:
    """PEtracer's long table of 2025 for one sample's cells, as GEO holds it.

    ``intID``, ``cellBC`` (the 10x barcode with ``-1``), the three sites,
    ``UMI`` and ``readCount``, then figshare's extra columns; the weaker
    second alleles follow their cell's rows with fewer UMIs, and a droplet
    outside the store has a row of its own.
    """
    lines = ["intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount,sample,reads_per_umi"]
    for table, weaker in ((recorders.independent, False), (recorders.second, True)):
        for (cell, integration), sites in table.groupby(
            ["cell", "integration"], sort=True
        ):
            if truth.samples[cell] != sample or not bool(sites["read"].iloc[0]):
                continue
            umis, reads = recorder_counts(
                int(cell), PETRACER_INTEGRATIONS.index(integration)
            )
            if weaker:
                umis, reads = 1, 3
            tokens = [
                _site_token(row.state, int(row.mark), row.site, int(cell))
                for row in sites.itertuples()
            ]
            lines.append(
                f"{integration},{truth.barcodes[cell]}-1,{','.join(tokens)},{umis},"
                f"{reads},{sample},{reads / umis}"
            )
    if sample == SAMPLES[0]:
        lines.append(f"intID101,{truth.outside[0]}-1,None,None,None,4,40,{sample},10.0")
    return _write(path, lines)


def expected_petracer_long(
    recorders: RecorderTruth, truth: CloneTruth, *, sample: str
) -> pd.DataFrame:
    """The records that importing ``write_petracer_long`` gives."""
    rows: list[dict[str, object]] = []
    for table, weaker in ((recorders.independent, False), (recorders.second, True)):
        for row in table.itertuples():
            if truth.samples[row.cell] != sample or not row.read:
                continue
            umis, reads = recorder_counts(
                row.cell, PETRACER_INTEGRATIONS.index(row.integration)
            )
            if weaker:
                umis, reads = 1, 3
            allele = (
                PETRACER_MARKS[row.site][row.mark - 1]
                if row.state == "mark"
                else PETRACER_OTHER
                if row.state == "unknown_mark"
                else None
            )
            rows.append(
                {
                    "cell": truth.cell_ids[row.cell],
                    "integration": row.integration,
                    "site": row.site,
                    "state": row.state,
                    "allele": allele,
                    "umis": umis,
                    "reads": reads,
                }
            )
    return _records(rows)


def write_petracer_characters(
    recorders: RecorderTruth, truth: CloneTruth, path: Path, *, layout: str = "geo"
) -> Path:
    """PEtracer's character matrix of 2026, of every cell, gzip-compressed.

    ``layout="geo"`` names the cell column ``cell_barcode``, as GEO's files
    do; ``"zenodo"`` leaves it unnamed, as the Zenodo matrix does, and the
    cells of the second sample, another embryo, lack the last integration,
    so its fields are empty. ``*`` is unedited, ``-`` missing, ``1`` to
    ``8`` a mark and ``!`` another insertion. Cell IDs are the store's.
    """
    columns = [
        f"{integration}-{site}"
        for integration in PETRACER_INTEGRATIONS
        for site in sorted(PETRACER_SITES)
    ]
    header = ("" if layout == "zenodo" else "cell_barcode") + "," + ",".join(columns)
    states = recorders.independent.set_index(["cell", "integration", "site"])
    lines = [header]
    for cell in range(N_CELLS):
        fields = [str(truth.cell_ids[cell])]
        for integration in PETRACER_INTEGRATIONS:
            for site in sorted(PETRACER_SITES):
                row = states.loc[(cell, integration, site)]
                if (
                    layout == "zenodo"
                    and integration == PETRACER_INTEGRATIONS[-1]
                    and truth.samples[cell] == SAMPLES[1]
                ):
                    fields.append("")
                elif not row["read"] or row["state"] == "missing":
                    fields.append("-")
                elif row["state"] == "mark":
                    fields.append(str(row["mark"]))
                elif row["state"] == "unknown_mark":
                    fields.append("!")
                else:
                    fields.append("*")
        lines.append(",".join(fields))
    return _write(path, lines, compress=True)


def expected_petracer_characters(
    recorders: RecorderTruth, truth: CloneTruth, *, layout: str = "geo"
) -> pd.DataFrame:
    """The records that importing ``write_petracer_characters`` gives."""
    rows: list[dict[str, object]] = []
    for row in recorders.independent.itertuples():
        state = row.state if row.read else "missing"
        # "!" is a placeholder of an unknown mark, which names no allele.
        allele = str(row.mark) if state == "mark" else None
        if (
            layout == "zenodo"
            and row.integration == PETRACER_INTEGRATIONS[-1]
            and truth.samples[row.cell] == SAMPLES[1]
        ):
            state, allele = "not_applicable", None
        rows.append(
            {
                "cell": truth.cell_ids[row.cell],
                "integration": row.integration,
                "site": row.site,
                "state": state,
                "allele": allele,
                "umis": -1,
                "reads": -1,
            }
        )
    return _records(rows)


# Cells whose 2022 rows carry a quirk of the published table: a sixth, empty
# slot field; a blank first slot before an edit; and an insertion beyond the
# fourth slot of the contracted TGGACGAC tape.
_RAGGED_CELL, _GAP_CELL, _BEYOND_CELL = 1, 2, 4


def _tape_slots(edits: int, symbols: Sequence[str], n_slots: int) -> list[str]:
    return [symbols[slot] if slot < edits else "" for slot in range(n_slots)]


def write_typewriter(
    recorders: RecorderTruth, truth: CloneTruth, path: Path, *, sample: str
) -> Path:
    """DNA Typewriter's table of 2022 for one sample's cells.

    ``Cell`` (the 10x barcode with ``-1``), ``TargetBC``, ``Site1`` to
    ``Site5`` and ``nUMI``; a slot holds its 6-base insertion and is blank
    when unedited, so the slots beyond a contracted tape are blank too.
    """
    lines = ["Cell,TargetBC,Site1,Site2,Site3,Site4,Site5,nUMI"]
    for row in recorders.tapes.itertuples():
        if truth.samples[row.cell] != sample or row.edits < 0:
            continue
        slots = _tape_slots(row.edits, row.symbols, 5)
        if row.cell == _GAP_CELL and row.integration == "ATTTGGTT":
            slots = ["", "TATGGA", "", "", ""]
        if row.cell == _BEYOND_CELL and row.integration == "TGGACGAC":
            slots = _tape_slots(4, ("CCGGGA",) * 4, 5)[:4] + ["GATGGA"]
        umis, _reads = recorder_counts(
            row.cell, list(TYPEWRITER_TAPES).index(row.integration)
        )
        fields = [f"{truth.barcodes[row.cell]}-1", row.integration, *slots]
        if row.cell == _RAGGED_CELL:
            fields.append("")
        lines.append(",".join([*fields, str(umis)]))
    return _write(path, lines)


def expected_typewriter(
    recorders: RecorderTruth, truth: CloneTruth, *, sample: str
) -> pd.DataFrame:
    """The records that importing ``write_typewriter`` with the tapes' slots gives."""
    rows: list[dict[str, object]] = []
    for row in recorders.tapes.itertuples():
        if truth.samples[row.cell] != sample or row.edits < 0:
            continue
        length = TYPEWRITER_TAPES[row.integration]
        edits, symbols = row.edits, row.symbols
        states = ["mark" if slot < edits else "unedited" for slot in range(5)]
        alleles = [symbols[slot] if slot < edits else None for slot in range(5)]
        if row.cell == _GAP_CELL and row.integration == "ATTTGGTT":
            states = ["missing", "mark", "unedited", "unedited", "unedited"]
            alleles = [None, "TATGGA", None, None, None]
        if row.cell == _BEYOND_CELL and row.integration == "TGGACGAC":
            states = ["mark"] * 4 + ["unedited"]
            alleles = ["CCGGGA"] * 4 + [None]
        umis, _reads = recorder_counts(
            row.cell, list(TYPEWRITER_TAPES).index(row.integration)
        )
        for slot in range(5):
            beyond = slot >= length
            rows.append(
                {
                    "cell": truth.cell_ids[row.cell],
                    "integration": row.integration,
                    "site": str(slot + 1),
                    "state": "not_applicable" if beyond else states[slot],
                    "allele": None if beyond else alleles[slot],
                    "umis": umis,
                    "reads": -1,
                }
            )
    return _records(rows)


def write_typewriter_chains(
    recorders: RecorderTruth, truth: CloneTruth, path: Path, *, form: str = "chain"
) -> Path:
    """DNA Typewriter's per-cell table of 2026, of every cell, gzip-compressed.

    ``form="chain"`` writes each tape as its symbols joined by ``-``, ``-``
    for none and nothing for no call, beside ``n_loci``, ``n_doublet_loci``
    and ``mean_dominance``; ``"slots"`` writes six slots joined by ``|``,
    ``U`` unedited and ``NA`` for no call, and adds ``pass_qc``. Cell IDs
    are the store's.
    """
    names = list(CHAIN_TAPES)
    header = [
        "cell" if form == "chain" else "cell_id",
        *names,
        "n_loci",
        "n_doublet_loci",
        "mean_dominance",
    ]
    if form == "slots":
        header.append("pass_qc")
    chains = recorders.chains.set_index(["cell", "integration"])
    lines = ["\t".join(header)]
    for cell in range(N_CELLS):
        fields = [str(truth.cell_ids[cell])]
        loci = 0
        for name in names:
            row = chains.loc[(cell, name)]
            edits, symbols = int(row["edits"]), tuple(row["symbols"])
            if edits < 0:
                fields.append("" if form == "chain" else "NA")
                continue
            loci += 1
            if form == "chain":
                fields.append("-".join(symbols[:edits]) if edits else "-")
            else:
                fields.append("|".join([*symbols[:edits], *["U"] * (6 - edits)]))
        fields += [str(loci), "0", "1.000"]
        if form == "slots":
            fields.append("1" if loci >= 2 else "0")
        lines.append("\t".join(fields))
    return _write(path, lines, compress=True)


def expected_typewriter_chains(
    recorders: RecorderTruth, truth: CloneTruth
) -> pd.DataFrame:
    """The records that importing ``write_typewriter_chains`` with the tapes' slots gives."""
    rows: list[dict[str, object]] = []
    for row in recorders.chains.itertuples():
        length = CHAIN_TAPES[row.integration]
        for slot in range(6):
            if slot >= length:
                state, allele = "not_applicable", None
            elif row.edits < 0:
                state, allele = "missing", None
            elif slot < row.edits:
                state, allele = "mark", row.symbols[slot]
            else:
                state, allele = "unedited", None
            rows.append(
                {
                    "cell": truth.cell_ids[row.cell],
                    "integration": row.integration,
                    "site": str(slot + 1),
                    "state": state,
                    "allele": allele,
                    "umis": -1,
                    "reads": -1,
                }
            )
    return _records(rows)


def write_carlin_pair(
    recorders: RecorderTruth,
    truth: CloneTruth,
    folder: Path,
    *,
    sample: str,
    array: str,
) -> Path:
    """CARLIN's AlleleAnnotations.txt and AlleleColonies.txt of one sample and array.

    One line per distinct allele, most cells first, each cell barcode bare
    and listed once; the first allele's line also lists the droplets
    outside the store, and neither file ends with a newline.
    """
    folder.mkdir(parents=True, exist_ok=True)
    table = recorders.arrays
    chosen = table[(table["integration"] == array) & table["read"]]
    chosen = chosen[[truth.samples[cell] == sample for cell in chosen["cell"]]]
    groups = chosen.groupby("allele", sort=True)["cell"].apply(list)
    order = sorted(groups.index, key=lambda allele: (-len(groups[allele]), allele))
    annotations, colonies = [], []
    for position, allele in enumerate(order):
        barcodes = [str(truth.barcodes[cell]) for cell in groups[allele]]
        if position == 0 and sample == SAMPLES[0]:
            barcodes += [str(barcode) for barcode in truth.outside]
        annotations.append(allele)
        colonies.append(",".join(barcodes))
    _write(folder / "AlleleAnnotations.txt", annotations, final_newline=False)
    return _write(folder / "AlleleColonies.txt", colonies, final_newline=False)


def expected_darlin(
    recorders: RecorderTruth,
    truth: CloneTruth,
    *,
    samples: Sequence[str],
    arrays: Sequence[str],
) -> pd.DataFrame:
    """The records that importing ``write_carlin_pair`` of samples and arrays gives."""
    rows: list[dict[str, object]] = []
    for row in recorders.arrays.itertuples():
        if (
            not row.read
            or row.integration not in arrays
            or truth.samples[row.cell] not in samples
        ):
            continue
        state = (
            "unedited"
            if row.allele == "[]"
            else "unknown_mark"
            if row.allele == "X"
            else "mark"
        )
        rows.append(
            {
                "cell": truth.cell_ids[row.cell],
                "integration": row.integration,
                "site": "",
                "state": state,
                # CARLIN's "X" is a placeholder, which names no allele.
                "allele": row.allele if state == "mark" else None,
                "umis": -1,
                "reads": -1,
            }
        )
    return _records(rows)


def write_darlin_step4(
    recorders: RecorderTruth,
    truth: CloneTruth,
    folder: Path,
    *,
    sample: str,
    array: str,
) -> Path:
    """The ``darlin`` CLI's step4_final.tsv and step4_annotated.tsv of one sample and array.

    Columns as its documentation names them. Every third cell has a second
    lineage barcode whose allele is unedited, written ``[]``, or empty, as
    darlin-core wrote it before 1.1.0, in every other such cell. Each row of
    the final table has ``n_UMIs`` molecules in the annotated table, each
    with reads of its own.
    """
    folder.mkdir(parents=True, exist_ok=True)
    table = recorders.arrays
    chosen = table[(table["integration"] == array) & table["read"]]
    final = ["n_UMIs\tCR\tmutation\taligned_LR\taligned_ref"]
    molecules = ["CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref"]
    for row in chosen.itertuples():
        if truth.samples[row.cell] != sample:
            continue
        alleles = [row.allele]
        if row.cell % 3 == 0:
            alleles.append("[]" if row.cell % 2 else "")
        for k, allele in enumerate(alleles):
            barcode = str(truth.barcodes[row.cell])
            lineage = f"LB{row.cell:03d}{k}{array}"
            aligned = f"ACGT-{lineage}-AC"
            umis, reads = recorder_counts(row.cell, DARLIN_ARRAYS.index(array) + k)
            final.append(f"{umis}\t{barcode}\t{allele}\t{aligned}\tACGTACGTAC")
            for umi in range(umis):
                molecules.append(
                    f"{barcode}\t{lineage}\tUMI{umi:04d}\t{1 + (umi + row.cell) % 4}\t{allele}\t{aligned}\tACGTACGTAC"
                )
    _write(folder / "step4_annotated.tsv", molecules)
    return _write(folder / "step4_final.tsv", final)


def expected_darlin_step4(
    recorders: RecorderTruth, truth: CloneTruth, *, sample: str, array: str
) -> pd.DataFrame:
    """The records that importing ``write_darlin_step4`` gives, reads summed."""
    rows: list[dict[str, object]] = []
    for row in recorders.arrays.itertuples():
        if (
            row.integration != array
            or not row.read
            or truth.samples[row.cell] != sample
        ):
            continue
        alleles = [row.allele]
        if row.cell % 3 == 0:
            alleles.append("[]" if row.cell % 2 else "")
        for k, allele in enumerate(alleles):
            umis, _reads = recorder_counts(row.cell, DARLIN_ARRAYS.index(array) + k)
            reads = sum(1 + (umi + row.cell) % 4 for umi in range(umis))
            state = (
                "unedited"
                if allele in ("[]", "")
                else "unknown_mark"
                if allele == "X"
                else "mark"
            )
            rows.append(
                {
                    "cell": truth.cell_ids[row.cell],
                    "integration": array,
                    "site": "",
                    "state": state,
                    "allele": None if state == "unedited" else allele,
                    "umis": umis,
                    "reads": reads,
                }
            )
    return _records(rows)


def _cassiopeia_token(state: str, mark: int, site_index: int) -> str:
    """A cut site as Cassiopeia writes it, with 5 bases around the event."""
    context = ("CGCCG", "ATTCG", "GATAT")[site_index]
    if state in ("mark", "unknown_mark"):
        size = mark if state == "mark" else 11
        return f"{context}[{100 + 30 * site_index + size}:{size}D]AATGG"
    if state == "missing":
        return "NC" if site_index % 2 else ""
    return f"{context}[None]CGGAG"


def write_cassiopeia_table(
    recorders: RecorderTruth, truth: CloneTruth, path: Path, *, sample: str
) -> Path:
    """Cassiopeia's allele table of one sample, in KP-Tracer's published layout.

    ``cellBC`` is ``<sample>.<barcode>-1``; an integration's sites are
    ``r1`` to ``r3``, uncut where they hold ``[None]``, missing where empty
    or ``NC``; ``allele`` joins them; a second, weaker allele of some cells
    follows. The PEtracer-like truth gives the edits.
    """
    header = "cellBC\tintBC\tr1\tr2\tr3\tallele\tsampleID\tUMI\treadCount\tTumor"
    lines = [header]
    for table, weaker in ((recorders.independent, False), (recorders.second, True)):
        for (cell, integration), sites in table.groupby(
            ["cell", "integration"], sort=True
        ):
            if truth.samples[cell] != sample or not bool(sites["read"].iloc[0]):
                continue
            tokens = [
                _cassiopeia_token(row.state, int(row.mark), index)
                for index, row in enumerate(sites.itertuples())
            ]
            umis, reads = recorder_counts(
                int(cell), PETRACER_INTEGRATIONS.index(integration)
            )
            if weaker:
                umis, reads = 1, 3
            lines.append(
                f"{sample}.{truth.barcodes[cell]}-1\t{integration}\t"
                + "\t".join(tokens)
                + f"\t{''.join(tokens)}\t{sample}\t{umis}\t{reads}\tT1"
            )
    return _write(path, lines)


def expected_cassiopeia_table(
    recorders: RecorderTruth, truth: CloneTruth, *, sample: str
) -> pd.DataFrame:
    """The records that importing ``write_cassiopeia_table`` gives."""
    rows: list[dict[str, object]] = []
    for table, weaker in ((recorders.independent, False), (recorders.second, True)):
        for row in table.itertuples():
            if truth.samples[row.cell] != sample or not row.read:
                continue
            index = PETRACER_SITES.index(row.site)
            token = _cassiopeia_token(row.state, row.mark, index)
            state = "mark" if row.state in ("mark", "unknown_mark") else row.state
            umis, reads = recorder_counts(
                row.cell, PETRACER_INTEGRATIONS.index(row.integration)
            )
            if weaker:
                umis, reads = 1, 3
            rows.append(
                {
                    "cell": truth.cell_ids[row.cell],
                    "integration": row.integration,
                    "site": f"r{index + 1}",
                    "state": state,
                    "allele": token if state == "mark" else None,
                    "umis": umis,
                    "reads": reads,
                }
            )
    return _records(rows)


def write_gestalt_table(
    recorders: RecorderTruth, truth: CloneTruth, path: Path, *, long: bool = False
) -> Path:
    """A GESTALT-family table of every cell: ``NONE`` unedited, ``UNKNOWN`` missing.

    Wide, one row per cell and integration with a column per target; or,
    with ``long``, one row per target, as macsGESTALT writes it, with its
    UMIs. Edits read ``<k>D+<site>``. Cell IDs are the store's.
    """

    def token(row) -> str:
        if not row.read or row.state == "missing":
            return "UNKNOWN"
        if row.state == "unedited":
            return "NONE"
        return f"{row.mark if row.state == 'mark' else 99}D+{row.site}"

    if long:
        lines = ["intID\tcellID\ttarget\ttarget_edit\tUMIs_crispr_intID_cell"]
        for row in recorders.independent.itertuples():
            umis, _reads = recorder_counts(
                row.cell, PETRACER_INTEGRATIONS.index(row.integration)
            )
            lines.append(
                f"{row.integration}\t{truth.cell_ids[row.cell]}\t{row.site}\t{token(row)}\t{umis}"
            )
        return _write(path, lines)
    lines = ["cell\tintegration\ttarget1\ttarget2\ttarget3"]
    for (cell, integration), sites in recorders.independent.groupby(
        ["cell", "integration"], sort=True
    ):
        lines.append(
            f"{truth.cell_ids[cell]}\t{integration}\t"
            + "\t".join(token(row) for row in sites.itertuples())
        )
    return _write(path, lines)


def expected_gestalt_table(
    recorders: RecorderTruth, truth: CloneTruth, *, long: bool = False
) -> pd.DataFrame:
    """The records that importing ``write_gestalt_table`` through columns gives."""
    rows: list[dict[str, object]] = []
    for row in recorders.independent.itertuples():
        state = "missing" if not row.read or row.state == "missing" else row.state
        if state == "unknown_mark":
            state = "mark"
        allele = None
        if state == "mark":
            allele = f"{row.mark if row.state == 'mark' else 99}D+{row.site}"
        umis = recorder_counts(row.cell, PETRACER_INTEGRATIONS.index(row.integration))[
            0
        ]
        rows.append(
            {
                "cell": truth.cell_ids[row.cell],
                "integration": row.integration,
                "site": row.site
                if long
                else f"target{PETRACER_SITES.index(row.site) + 1}",
                "state": state,
                "allele": allele,
                "umis": umis if long else -1,
                "reads": -1,
            }
        )
    return _records(rows)


# Allele clone calling ----------------------------------------------------------

#: The noise that the rules of allele clone calling must remove: a homoplasic
#: allele, rare in the bank but found in three of its mice, in one cell of
#: each of two clones ("homoplasy"), and an allele that the bank lacks in one
#: cell of each of two other clones ("merged").
ALLELE_NOISE = ("homoplasy", "merged")
#: The bank frequency of the alleles that independently edited cells often
#: hold, and of the rare alleles that the bank lists.
COMMON_FREQUENCY = 0.05
RARE_FREQUENCY = 1e-6
#: The clones that each kind of noise joins: both rare alleles of each clone
#: and the allele that joins them make five, more than three.
HOMOPLASIC_CLONES = (0, 3)
MERGED_CLONES = (2, 5)


@dataclass(frozen=True, slots=True)
class AlleleCallingTruth:
    """DARLIN arrays of the truth's cells for calling clones from rare alleles.

    Clone ``k`` holds two rare alleles: one on its own array,
    ``DARLIN_ARRAYS[k % 3]``, in every cell but its second, and one on the
    next array in its first two cells and every other cell after, so its
    second cell joins the others only through its first, which holds both.
    The bank lists the first rare allele of even clones and the second of
    odd ones, at ``RARE_FREQUENCY`` in one mouse, and lacks the others. The
    third array holds one of two common alleles in even clones, at
    ``COMMON_FREQUENCY`` in three mice, and is unedited in odd ones, read in
    every third cell. Cells without a clone hold common alleles, unedited
    arrays, CARLIN's placeholder ``X`` or nothing, so that only rare alleles
    could give them a clone.

    Attributes:
        recorders: The arrays, in ``RecorderTruth.arrays``, so that
            ``write_carlin_pair`` writes them; the other tables are empty.
        banks: Each array's bank, as DARLIN's banks hold them: ``allele``,
            ``observed_count``, ``sample_count``, ``sample_id`` and
            ``normalized_count``.
        noise: The kinds of noise, of ``ALLELE_NOISE``.
        rare: Each array's alleles that the design makes rare, the
            homoplasic one among them.
        homoplasic: The homoplasic allele of RA, or None without it.
    """

    recorders: RecorderTruth
    banks: dict[str, pd.DataFrame]
    noise: tuple[str, ...]
    rare: dict[str, frozenset[str]]
    homoplasic: str | None

    def expected_partition(
        self,
        truth: CloneTruth,
        *,
        max_samples: int | None = None,
        max_alleles: int | None = None,
        groups: np.ndarray | None = None,
    ) -> set[frozenset[str]]:
        """The clones that the design's rare alleles give, as sets of cell IDs.

        Cells of a group that hold one of the design's rare alleles of an
        array are joined, one cell at a time; the homoplasic allele is rare
        unless ``max_samples`` is below 3, the bank's mice that hold it. A
        joined set of more than ``max_alleles`` distinct rare alleles leaves
        its cells without a clone. ``groups`` keeps the cells of each group
        apart, as ``within`` does. Without noise, the clones are the truth's.
        """
        rare = {array: set(alleles) for array, alleles in self.rare.items()}
        if max_samples is not None and max_samples < 3:
            rare["RA"] -= {self.homoplasic} if self.homoplasic else set()
        keys = np.zeros(N_CELLS, dtype=object) if groups is None else groups
        parent = list(range(N_CELLS))

        def root(cell: int) -> int:
            while parent[cell] != cell:
                cell = parent[cell]
            return cell

        holders: dict[tuple[object, str, str], int] = {}
        held: dict[int, set[tuple[str, str]]] = {}
        for row in self.recorders.arrays.itertuples():
            if not row.read or row.allele not in rare[row.integration]:
                continue
            cell = int(row.cell)
            held.setdefault(cell, set()).add((row.integration, row.allele))
            key = (keys[cell], row.integration, row.allele)
            if key in holders:
                parent[root(cell)] = root(holders[key])
            else:
                holders[key] = cell
        members: dict[int, set[int]] = {}
        for cell in held:
            members.setdefault(root(cell), set()).add(cell)
        ids = truth.cell_ids
        found = set()
        for cells in members.values():
            alleles = set().union(*(held[cell] for cell in cells))
            if max_alleles is None or len(alleles) <= max_alleles:
                found.add(frozenset(str(ids[cell]) for cell in cells))
        return found


def _allele_text(rng: np.random.Generator, seen: set[str]) -> str:
    """A DARLIN allele of two events, as Custom_CARLIN writes them, not seen before."""
    while True:
        start = int(rng.integers(20, 120))
        end = start + int(rng.integers(5, 140))
        insertion = "".join(rng.choice(list("ACGT"), size=int(rng.integers(1, 4))))
        allele = f"{start}_{end}del,{end + 20}_{end + 21}ins{insertion}"
        if allele not in seen:
            seen.add(allele)
            return allele


def allele_calling_truth(
    truth: CloneTruth, *, noise: Sequence[str] = (), seed: int = 0
) -> AlleleCallingTruth:
    """Give the truth's clones rare alleles on DARLIN's arrays, with a bank.

    ``"homoplasy"`` gives the third cells of clones 0 and 3, on their third
    array, RA for both, an allele that the bank lists as rare but in three
    mice; ``"merged"`` gives the third cell of clone 2 and the first of clone
    5, on their third array, TA for both, an allele that the bank lacks.
    """
    unknown = sorted(set(noise) - set(ALLELE_NOISE))
    if unknown:
        raise ValueError(f"unknown noise {unknown}")
    rng = np.random.default_rng([seed, 20])
    seen: set[str] = set()
    common = {
        array: (_allele_text(rng, seen), _allele_text(rng, seen))
        for array in DARLIN_ARRAYS
    }
    first_rare = [_allele_text(rng, seen) for _clone in range(truth.n_clones)]
    second_rare = [_allele_text(rng, seen) for _clone in range(truth.n_clones)]
    homoplasic = _allele_text(rng, seen)
    lacking = _allele_text(rng, seen)
    rows: list[dict[str, object]] = []
    rare: dict[str, set[str]] = {array: set() for array in DARLIN_ARRAYS}
    listed: dict[str, list[tuple[str, float, int]]] = {
        array: [
            ("[]", 1.0, 3),
            *((allele, COMMON_FREQUENCY, 3) for allele in common[array]),
        ]
        for array in DARLIN_ARRAYS
    }

    def add(cell: int, array: str, allele: str | None) -> None:
        rows.append(
            {
                "cell": cell,
                "integration": array,
                "read": allele is not None,
                "allele": "[]" if allele is None else allele,
            }
        )

    for clone in range(truth.n_clones):
        own, following, third = (DARLIN_ARRAYS[(clone + step) % 3] for step in range(3))
        rare[own].add(first_rare[clone])
        rare[following].add(second_rare[clone])
        if clone % 2 == 0:
            listed[own].append((first_rare[clone], RARE_FREQUENCY, 1))
        else:
            listed[following].append((second_rare[clone], RARE_FREQUENCY, 1))
        members = np.flatnonzero(truth.clones == clone).tolist()
        for position, cell in enumerate(members):
            add(cell, own, None if position == 1 else first_rare[clone])
            add(
                cell,
                following,
                second_rare[clone] if position < 2 or position % 2 == 0 else None,
            )
            allele = None
            if position % 3 == 0:
                allele = common[third][position % 2] if clone % 2 == 0 else "[]"
            if "homoplasy" in noise and clone in HOMOPLASIC_CLONES and position == 2:
                allele = homoplasic
            if "merged" in noise and (
                (clone == MERGED_CLONES[0] and position == 2)
                or (clone == MERGED_CLONES[1] and position == 0)
            ):
                allele = lacking
            add(cell, third, allele)
    if "homoplasy" in noise:
        listed["RA"].append((homoplasic, RARE_FREQUENCY, 3))
        rare["RA"].add(homoplasic)
    if "merged" in noise:
        rare["TA"].add(lacking)
    # Cells without a clone read common alleles, unedited arrays, CARLIN's
    # placeholder or nothing, in turn.
    choices = ("[]", "common0", "X", None, "common1")
    for order, cell in enumerate(np.flatnonzero(truth.clones < 0).tolist()):
        for step, array in enumerate(DARLIN_ARRAYS):
            choice = choices[(order + step) % len(choices)]
            if choice is not None and choice.startswith("common"):
                choice = common[array][int(choice[-1])]
            add(cell, array, choice)
    banks = {
        array: pd.DataFrame(
            {
                "allele": [allele for allele, _rate, _mice in entries],
                "observed_count": [
                    max(1, round(rate * 1_000_000)) for _allele, rate, _mice in entries
                ],
                "sample_count": [mice for _allele, _rate, mice in entries],
                "sample_id": [
                    ",".join(f"LL{583 + mouse}" for mouse in range(mice))
                    for _allele, _rate, mice in entries
                ],
                "normalized_count": [rate for _allele, rate, _mice in entries],
            }
        )
        for array, entries in listed.items()
    }
    empty = pd.DataFrame()
    return AlleleCallingTruth(
        recorders=RecorderTruth(
            independent=empty,
            second=empty,
            tapes=empty,
            chains=empty,
            arrays=pd.DataFrame(rows),
        ),
        banks=banks,
        noise=tuple(noise),
        rare={array: frozenset(alleles) for array, alleles in rare.items()},
        homoplasic=homoplasic if "homoplasy" in noise else None,
    )
