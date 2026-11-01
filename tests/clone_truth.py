"""A seeded synthetic lineage tracing experiment, written in every clone format.

``clone_truth`` draws cells with a sample, a time point and a cell type, and
clones of known sizes. The writers emit the experiment as each tool writes
its files, with the quirks an importer must handle: headers that start with
``#``, R row names, unnamed index columns, gzip, a missing final newline,
``NA`` and barcodes of droplets that are not cells. ``receptor_truth`` gives
the clones T cell and B cell receptor chains and fixes, for each cell, the
chains and labels that an import of its contigs must give. Tests check that an
import recovers the truth, which this module fixes and the code under test
never computes.
"""

import gzip
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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


def receptor_truth(truth: CloneTruth, seed: int = 0) -> ReceptorTruth:
    """Give the truth's clones receptor chains and each cell its contigs.

    Each cell's contigs follow its profile in ``RECEPTOR_PROFILES``. A clone's
    chains carry 8 to 20 UMIs in each cell, so a second alpha chain (3 UMIs),
    a non-productive one (40), extra beta chains (2 and 1) and a heavy chain
    beside a T cell receptor (4) rank as the expected slots say. Each barcode
    outside the store has one beta chain.
    """
    rng = np.random.default_rng([seed, 5])
    draws = _ChainDraws(rng)
    clone_chains = []
    for clone in range(truth.n_clones):
        light = B_CELL_LIGHT_LOCI.get(clone)
        vj, vdj = ("TRA", "TRB") if light is None else (light, "IGH")
        clone_chains.append((draws.chain(vj), draws.chain(vdj)))
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
            vj, vdj = clone_chains[clone]
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
