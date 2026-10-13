"""cisTarget ranking databases and motif-to-transcription-factor annotations."""

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "RankingDatabase",
    "read_motif_annotations",
]

#: Names that a cisTarget database gives the column of its row identifiers.
INDEX_COLUMNS = ("motifs", "tracks", "regions", "genes")

#: ``.tbl`` columns that annotation filtering and pruning read.
ANNOTATION_COLUMNS = (
    "#motif_id",
    "gene_name",
    "motif_similarity_qvalue",
    "orthologous_identity",
    "description",
)


def _polars() -> Any:
    try:
        import polars
    except ImportError as exc:
        raise ImportError(
            "Reading cisTarget databases needs polars; install cytearc[regulons]"
        ) from exc
    return polars


class RankingDatabase:
    """A cisTarget rankings database read column by column.

    A database is a Feather version 2 (Arrow IPC) file whose rows are motifs
    or tracks and whose columns are ranked features, genes or regions, plus
    one index column naming the rows. Each value is the zero-based rank of a
    feature in that motif's ranking of all ``total`` features. Only the
    columns of requested features are read and decompressed, so databases far
    larger than memory can be queried.

    Args:
        path: Database file, such as
            ``hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather``.
        name: Display name; the file name when omitted.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, name: str | None = None
    ) -> None:
        polars = _polars()
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Ranking database {self.path} does not exist")
        with open(self.path, "rb") as handle:
            header = handle.read(6)
            handle.seek(-6, os.SEEK_END)
            footer = handle.read(6)
        if header != b"ARROW1" or footer != b"ARROW1":
            raise ValueError(
                f"{self.path.name} is not a Feather version 2 file; convert legacy "
                "Feather version 1 databases first"
            )
        self.name = name or self.path.name
        names = list(polars.read_ipc_schema(self.path))
        index = [column for column in names if column in INDEX_COLUMNS]
        if len(index) != 1:
            raise ValueError(
                f"{self.path.name} must have exactly one index column named one of "
                f"{INDEX_COLUMNS}, found {index}"
            )
        self.indexColumn = index[0]
        self.featureNames: tuple[str, ...] = tuple(
            column for column in names if column != self.indexColumn
        )
        if len(set(self.featureNames)) != len(self.featureNames):
            raise ValueError(f"{self.path.name} repeats a feature column")
        self._position = {name: index for index, name in enumerate(self.featureNames)}
        index_frame = polars.read_ipc(self.path, columns=[self.indexColumn])
        self.rowNames = np.asarray(
            index_frame.get_column(self.indexColumn).to_list(), dtype=object
        ).astype(str)
        if len(self.rowNames) < 2:
            raise ValueError(f"{self.path.name} must rank at least two motifs")

    @property
    def total(self) -> int:
        """Number of ranked features."""
        return len(self.featureNames)

    def __contains__(self, feature: object) -> bool:
        return feature in self._position

    def __repr__(self) -> str:
        return (
            f"RankingDatabase({self.name!r}, rows={len(self.rowNames)}, "
            f"features={self.total})"
        )

    def ranks(self, features: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """Read the ranks of the features that the database ranks.

        Args:
            features: Feature names; names the database does not rank are
                skipped, as ctxcore does.

        Returns:
            The found feature names in database column order, and their
            ``(rows, found)`` int32 ranks.
        """
        polars = _polars()
        wanted = {str(feature) for feature in features}
        found = sorted(
            (self._position[name] for name in wanted if name in self._position)
        )
        names = np.asarray([self.featureNames[index] for index in found], dtype=object)
        ranks = np.empty((len(self.rowNames), len(found)), dtype=np.int32)
        if len(found) == 0:
            return names.astype(str), ranks
        frame = polars.read_ipc(self.path, columns=names.tolist())
        if frame.height != len(self.rowNames):
            raise ValueError(f"{self.path.name} returned misaligned rank columns")
        for index, name in enumerate(names):
            ranks[:, index] = frame.get_column(name).to_numpy()
        if np.any(ranks < 0) or np.any(ranks >= self.total + 1):
            raise ValueError(f"{self.path.name} holds ranks outside 0..{self.total}")
        return names.astype(str), ranks


def _annotation_category(description: pd.Series) -> pd.Series:
    """Classify cisTarget annotation descriptions as pycistarget does."""
    text = description.fillna("").astype(str)
    similar = text.str.contains("similar")
    orthologous = text.str.contains("orthologous")
    category = pd.Series("other", index=description.index, dtype=object)
    category[text == "gene is directly annotated"] = "direct"
    category[similar & ~orthologous] = "similarity"
    category[~similar & orthologous] = "orthology"
    category[similar & orthologous] = "similarity_and_orthology"
    return category


def read_motif_annotations(
    path: str | os.PathLike[str],
    *,
    motif_similarity_fdr: float = 0.001,
    orthologous_identity_threshold: float = 0.0,
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a cisTarget motif-to-TF annotation table (``.tbl``).

    Rows whose motif similarity q-value exceeds ``motif_similarity_fdr`` or
    whose orthologous identity is below ``orthologous_identity_threshold``
    are dropped, as pySCENIC and pycistarget do.

    Args:
        path: Annotation table, such as ``motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl``.
        motif_similarity_fdr: Largest motif similarity q-value kept.
        orthologous_identity_threshold: Smallest orthologous identity kept.
        categories: Keep only these annotation categories: ``"direct"``,
            ``"similarity"``, ``"orthology"``, and ``"similarity_and_orthology"``.

    Returns:
        A DataFrame with ``motif``, ``TF``, ``motif_similarity_qvalue``,
        ``orthologous_identity``, ``description``, and ``category`` columns.
    """
    table = pd.read_csv(path, sep="\t", usecols=list(ANNOTATION_COLUMNS))
    table = table.rename(columns={"#motif_id": "motif", "gene_name": "TF"})
    table = table.loc[
        (table["motif_similarity_qvalue"] <= motif_similarity_fdr)
        & (table["orthologous_identity"] >= orthologous_identity_threshold)
    ].copy()
    table["category"] = _annotation_category(table["description"])
    if categories is not None:
        unknown = set(categories) - {
            "direct",
            "similarity",
            "orthology",
            "similarity_and_orthology",
        }
        if unknown:
            raise ValueError(f"Unknown annotation categories: {sorted(unknown)}")
        table = table.loc[table["category"].isin(list(categories))]
    return table.reset_index(drop=True)
