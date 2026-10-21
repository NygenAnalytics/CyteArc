import csv
import gc
import gzip
import os
import shutil
import stat
import tempfile
import weakref
import zipfile
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TextIO

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arrays import cumulative_nnz, max_window_nnz
from ..utils.count_values import CountValueRange, new_count_ranges
from .cellranger import CrReader
from ._text import (
    DuplicateFeatureIds,
    duplicate_feature_ids_option,
    require_unique_identifiers,
    unique_feature_identifiers,
)

type MatrixOrientation = Literal["featuresByCells", "cellsByFeatures"]
type CoordinateOrder = Literal["cellMajor", "featureMajor", "unordered"]

_CHANGED_COUNT_MESSAGE = (
    "A Matrix Market count exceeds the largest count found when the reader "
    "scanned the file; the file changed"
)
_CHANGED_UNORDERED_ROWS_MESSAGE = (
    "Unordered Matrix Market preparation produced inconsistent row pointers; "
    "the file changed after the reader scanned it"
)


@dataclass(frozen=True, slots=True)
class MtxCandidate:
    """One complete Matrix Market matrix, feature, and cell selection."""

    source: str
    matrixPath: str
    featurePath: str
    cellPath: str
    matrixOrientation: MatrixOrientation
    nCells: int
    nFeatures: int
    nEntries: int
    archivePath: str | None = None
    cellMetadataPath: str | None = None
    featureReferencePath: str | None = None
    cellIdKeys: tuple[str, ...] = ()
    relatedFiles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MtxHeader:
    field: Literal["integer", "real"]
    nRows: int
    nColumns: int
    nEntries: int


@dataclass(frozen=True, slots=True)
class _ResolvedTriplet:
    matrix: str
    features: str
    cells: str
    parseLayout: bool
    cellMetadata: str | None
    featureReference: str | None
    related: tuple[str, ...]


def _without_gzip(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def _matrix_basename(name: str) -> str:
    return PurePosixPath(_without_gzip(name)).name.lower()


def _is_matrix_path(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".mtx") or lowered.endswith(".mtx.gz")


def _is_parse_matrix(name: str) -> bool:
    return _matrix_basename(name) in {"count_matrix.mtx", "dge.mtx"}


def _is_bed_sidecar(name: str) -> bool:
    return PurePosixPath(_without_gzip(name)).name.lower().endswith(".bed")


def _local_absolute(path: str | Path) -> Path:
    """Return an absolute path without resolving symbolic links.

    Sidecars are matched by their names, so a symbolic link keeps the name it
    has in its directory rather than the name of its target.
    """
    return Path(os.path.abspath(Path(path).expanduser()))


def _matrix_prefix(name: str) -> str:
    stem = PurePosixPath(_without_gzip(name)).stem
    return stem[: -len("matrix")] if stem.lower().endswith("matrix") else ""


def _same_parent_names(
    names: tuple[str, ...],
    matrix_name: str,
) -> dict[str, str]:
    parent = PurePosixPath(matrix_name).parent
    return {
        PurePosixPath(name).name.lower(): name
        for name in names
        if PurePosixPath(name).parent == parent
    }


def _first_name(
    available: dict[str, str],
    names: tuple[str, ...],
) -> str | None:
    for name in names:
        match = available.get(name.lower())
        if match is not None:
            return match
    return None


def _related_files(available: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            value
            for key, value in available.items()
            if ("guide" in key or "protospacer" in key or "crispr_analysis" in key)
            and key.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz"))
        )
    )


def _resolve_triplet(
    names: tuple[str, ...],
    matrix_name: str,
) -> _ResolvedTriplet | None:
    available = _same_parent_names(names, matrix_name)
    parse_layout = _is_parse_matrix(matrix_name)
    if parse_layout:
        features = _first_name(
            available,
            (
                "all_genes.csv.gz",
                "all_genes.csv",
                "genes.csv.gz",
                "genes.csv",
            ),
        )
        cells = _first_name(
            available,
            ("cell_metadata.csv.gz", "cell_metadata.csv"),
        )
        if features is None or cells is None:
            return None
        return _ResolvedTriplet(
            matrix=matrix_name,
            features=features,
            cells=cells,
            parseLayout=True,
            cellMetadata=cells,
            featureReference=None,
            related=_related_files(available),
        )

    prefix = _matrix_prefix(matrix_name).lower()
    features = _first_name(
        available,
        tuple(
            f"{prefix}{suffix}"
            for suffix in (
                "features.tsv.gz",
                "features.tsv",
                "genes.tsv.gz",
                "genes.tsv",
                "peaks.bed.gz",
                "peaks.bed",
            )
        ),
    )
    cells = _first_name(
        available,
        (f"{prefix}barcodes.tsv.gz", f"{prefix}barcodes.tsv"),
    )
    if features is None or cells is None:
        return None
    feature_reference = _first_name(
        available,
        (
            f"{prefix}feature_reference.csv.gz",
            f"{prefix}feature_reference.csv",
            "feature_reference.csv.gz",
            "feature_reference.csv",
        ),
    )
    return _ResolvedTriplet(
        matrix=matrix_name,
        features=features,
        cells=cells,
        parseLayout=False,
        cellMetadata=None,
        featureReference=feature_reference,
        related=_related_files(available),
    )


@contextmanager
def _open_path_text(path: str) -> Iterator[TextIO]:
    if path.lower().endswith(".gz"):
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with open(path, mode="rt", encoding="utf-8", newline="") as handle:
            yield handle


def _zip_member_mode(info: zipfile.ZipInfo) -> int:
    return int(info.external_attr >> 16)


def _validated_zip_infos(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe ZIP archive member: {info.filename}")
        normalized = str(member)
        if normalized in infos:
            raise ValueError(f"Duplicate ZIP archive member: {info.filename}")
        if info.flag_bits & 0x1:
            raise ValueError(f"Encrypted ZIP archive member: {info.filename}")
        mode = _zip_member_mode(info)
        if stat.S_ISLNK(mode):
            raise ValueError(f"ZIP archive links are not supported: {info.filename}")
        file_type = stat.S_IFMT(mode)
        if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"Unsupported ZIP archive member: {info.filename}")
        if not info.is_dir() and normalized.lower().endswith(".zip"):
            raise ValueError("A Matrix Market ZIP cannot contain another ZIP")
        infos[normalized] = info
    return infos


@contextmanager
def _open_zip_text(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> Iterator[TextIO]:
    with archive.open(info, mode="r") as raw:
        if info.filename.lower().endswith(".gz"):
            with gzip.GzipFile(fileobj=raw, mode="rb") as decoded:
                with TextIOWrapper(decoded, encoding="utf-8", newline="") as text:
                    yield text
        else:
            with TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                yield text


def _read_header(open_text: Callable[[], Any]) -> _MtxHeader:
    with open_text() as handle:
        banner = handle.readline().strip().split()
        if len(banner) != 5 or [part.lower() for part in banner[:3]] != [
            "%%matrixmarket",
            "matrix",
            "coordinate",
        ]:
            raise ValueError("Matrix Market input must use coordinate matrix format")
        field = banner[3].lower()
        symmetry = banner[4].lower()
        if field not in {"integer", "real"}:
            raise ValueError(
                "Matrix Market counts must use integer or real coordinate values"
            )
        if symmetry != "general":
            raise ValueError("Matrix Market counts must use general symmetry")
        dimensions = ""
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                dimensions = stripped
                break
        fields = dimensions.split()
        if len(fields) != 3:
            raise ValueError(
                "Matrix Market dimensions line must contain three integers"
            )
        try:
            n_rows, n_columns, n_entries = (int(value) for value in fields)
        except ValueError as exc:
            raise ValueError(
                "Matrix Market dimensions line must contain three integers"
            ) from exc
        if n_rows < 0 or n_columns < 0 or n_entries < 0:
            raise ValueError("Matrix Market dimensions cannot be negative")
        return _MtxHeader(
            field=field,
            nRows=n_rows,
            nColumns=n_columns,
            nEntries=n_entries,
        )


def _count_sidecar_rows(
    open_text: Callable[[], Any],
    *,
    header: bool,
) -> int:
    with open_text() as handle:
        rows = sum(1 for line in handle if line.strip())
    return max(0, rows - int(header))


def _csv_header(open_text: Callable[[], Any]) -> tuple[str, ...]:
    with open_text() as handle:
        # An empty file has no header.
        return tuple(str(value).strip() for value in next(csv.reader(handle), ()))


def _candidate_from_triplet(
    *,
    source: str,
    archive_path: str | None,
    triplet: _ResolvedTriplet,
    open_text: Callable[[str], Any],
) -> MtxCandidate:
    header = _read_header(lambda: open_text(triplet.matrix))
    feature_rows = _count_sidecar_rows(
        lambda: open_text(triplet.features),
        header=triplet.parseLayout,
    )
    cell_rows = _count_sidecar_rows(
        lambda: open_text(triplet.cells),
        header=triplet.parseLayout,
    )
    preferred: MatrixOrientation = (
        "cellsByFeatures" if triplet.parseLayout else "featuresByCells"
    )
    features_by_cells = feature_rows == header.nRows and cell_rows == header.nColumns
    cells_by_features = cell_rows == header.nRows and feature_rows == header.nColumns
    if preferred == "featuresByCells" and features_by_cells:
        orientation: MatrixOrientation = "featuresByCells"
    elif preferred == "cellsByFeatures" and cells_by_features:
        orientation = "cellsByFeatures"
    elif features_by_cells and not cells_by_features:
        orientation = "featuresByCells"
    elif cells_by_features and not features_by_cells:
        orientation = "cellsByFeatures"
    else:
        raise ValueError(
            f"Matrix dimensions ({header.nRows}, {header.nColumns}) do not "
            f"match {feature_rows} feature rows and {cell_rows} cell rows"
        )
    cell_id_keys = (
        tuple(
            key
            for key in ("bc_wells", "bc_index")
            if key in _csv_header(lambda: open_text(triplet.cells))
        )
        if triplet.parseLayout
        else ()
    )
    n_cells = header.nColumns if orientation == "featuresByCells" else header.nRows
    n_features = header.nRows if orientation == "featuresByCells" else header.nColumns
    return MtxCandidate(
        source=source,
        matrixPath=triplet.matrix,
        featurePath=triplet.features,
        cellPath=triplet.cells,
        matrixOrientation=orientation,
        nCells=n_cells,
        nFeatures=n_features,
        nEntries=header.nEntries,
        archivePath=archive_path,
        cellMetadataPath=triplet.cellMetadata,
        featureReferencePath=triplet.featureReference,
        cellIdKeys=cell_id_keys,
        relatedFiles=triplet.related,
    )


def inspect_mtx(source: str | Path) -> tuple[MtxCandidate, ...]:
    """Return complete Matrix Market import candidates found in one source."""
    source_path = _local_absolute(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    candidates: list[MtxCandidate] = []
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            infos = _validated_zip_infos(archive)
            names = tuple(
                sorted(name for name, info in infos.items() if not info.is_dir())
            )

            @contextmanager
            def open_member(name: str) -> Iterator[TextIO]:
                with _open_zip_text(archive, infos[name]) as handle:
                    yield handle

            for matrix_name in (name for name in names if _is_matrix_path(name)):
                triplet = _resolve_triplet(names, matrix_name)
                if triplet is None:
                    continue
                candidates.append(
                    _candidate_from_triplet(
                        source=str(source_path),
                        archive_path=str(source_path),
                        triplet=triplet,
                        open_text=open_member,
                    )
                )
    else:
        directory = source_path if source_path.is_dir() else source_path.parent
        names = tuple(
            sorted(str(path) for path in directory.iterdir() if path.is_file())
        )
        selected_matrix = str(source_path) if source_path.is_file() else None
        for matrix_name in (name for name in names if _is_matrix_path(name)):
            if selected_matrix is not None and matrix_name != selected_matrix:
                continue
            triplet = _resolve_triplet(names, matrix_name)
            if triplet is None:
                continue
            candidates.append(
                _candidate_from_triplet(
                    source=str(source_path),
                    archive_path=None,
                    triplet=triplet,
                    open_text=lambda name: _open_path_text(name),
                )
            )

    if not candidates:
        raise ValueError(
            "No complete Matrix Market matrix, feature, and cell triplet found"
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.matrixPath,
                candidate.featurePath,
                candidate.cellPath,
            ),
        )
    )


def _explicit_candidate(
    matrix_path: str,
    feature_path: str,
    cell_path: str,
    *,
    feature_reference_path: str | None,
) -> MtxCandidate:
    matrix = str(_local_absolute(matrix_path))
    features = str(_local_absolute(feature_path))
    cells = str(_local_absolute(cell_path))
    for path in (matrix, features, cells):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    parse_layout = _is_parse_matrix(matrix)
    triplet = _ResolvedTriplet(
        matrix=matrix,
        features=features,
        cells=cells,
        parseLayout=parse_layout,
        cellMetadata=None,
        featureReference=feature_reference_path,
        related=(),
    )
    return replace(
        _candidate_from_triplet(
            source=matrix,
            archive_path=None,
            triplet=triplet,
            open_text=lambda name: _open_path_text(name),
        ),
        featureReferencePath=(
            str(_local_absolute(feature_reference_path))
            if feature_reference_path is not None
            else None
        ),
    )


def _column_name(values: tuple[str, ...], preferences: tuple[str, ...]) -> str:
    normalized = {str(value).lower(): str(value) for value in values}
    for preference in preferences:
        if preference in normalized:
            return normalized[preference]
    raise ValueError(
        "Could not identify a required column; available columns are "
        + ", ".join(values)
    )


def _remove_prepared_rows(
    directory: Path,
    arrays: tuple[np.ndarray | np.memmap, ...],
) -> None:
    """Remove the prepared rows of a reader that was never closed.

    POSIX systems remove files that are still memory-mapped, but Windows does
    not, so only when the first removal fails are the maps closed, as
    ``release`` closes them, and the removal tried again.
    """
    try:
        shutil.rmtree(directory)
    except OSError:
        for array in arrays:
            _MtxEngine._close_array(array)
        shutil.rmtree(directory, ignore_errors=True)


class _MtxEngine:
    def __init__(
        self,
        candidate: MtxCandidate,
        *,
        cell_id_key: str | None,
        separator: str,
        index_offset: int,
        is_filtered: bool,
        filtering_cutoff: int,
        temp_dir: str | None,
        duplicate_feature_ids: DuplicateFeatureIds,
    ) -> None:
        self.candidate = candidate
        self.separator = separator
        self.indexOffset = int(index_offset)
        self.isFiltered = bool(is_filtered)
        self.filteringCutoff = int(filtering_cutoff)
        self.tempDir = None if temp_dir is None else str(Path(temp_dir))
        self.duplicateFeatureIds = duplicate_feature_ids_option(duplicate_feature_ids)
        if self.filteringCutoff < 0:
            raise ValueError("filtering_cutoff cannot be negative")
        if self.tempDir is not None and not Path(self.tempDir).is_dir():
            raise FileNotFoundError(self.tempDir)

        self._archiveDirectory: Path | None = None
        self._archiveTemporary: Any | None = None
        self._archivePaths: dict[str, str] = {}
        self._csrDirectory: Path | None = None
        self._csrFinalizer: weakref.finalize[Any, Any] | None = None
        self._csrData: np.ndarray | np.memmap | None = None
        self._csrIndices: np.ndarray | np.memmap | None = None
        self._csrIndptr: np.ndarray | np.memmap | None = None
        # The row slots that the scan counted for each kept cell of an
        # unordered matrix, which hold its entries before they are summed.
        self._slotPointers: np.ndarray | None = None
        self._importLinesInMem = 100_000
        self.temporaryDiskBytes = 0
        try:
            self._ensure_archive_extracted()

            self.matrixPath = self._local_path(candidate.matrixPath)
            self.featurePath = self._local_path(candidate.featurePath)
            self.cellPath = self._local_path(candidate.cellPath)
            self.header = _read_header(lambda: _open_path_text(self.matrixPath))
            self.matrixEntryCount = self.header.nEntries
            self.rawCellCount = candidate.nCells
            self.nFeatures = candidate.nFeatures
            (
                self._sourceFeatureIds,
                self._rawFeatureNames,
                self._rawFeatureTypes,
            ) = self._read_features()
            (
                self._rawCellNames,
                self._rawCellColumns,
                self.selectedCellIdKey,
            ) = self._read_cells(cell_id_key)
            self._rawFeatureIds = unique_feature_identifiers(
                self._sourceFeatureIds,
                "Feature IDs",
                self.duplicateFeatureIds,
            )
            require_unique_identifiers(self._rawCellNames, "Cell IDs")

            (
                row_nnz,
                cell_totals,
                cell_maxima,
                feature_maxima,
                self.coordinateOrder,
            ) = self._scan_matrix(self._importLinesInMem)
            if self.isFiltered:
                valid = np.arange(self.rawCellCount, dtype=np.int64)
            else:
                valid = np.flatnonzero(cell_totals > self.filteringCutoff).astype(
                    np.int64,
                    copy=False,
                )
            self._set_valid_cells(valid)
            self._featureColumns = self._feature_reference_columns()
            self._keptFeatureMaxima: np.ndarray | None
            if self.coordinateOrder == "unordered":
                # The totals are exact in any order, and the sum of a cell's
                # duplicate coordinates never exceeds its total, so the
                # largest kept total bounds every count of the sorted rows.
                self._slotPointers = cumulative_nnz(row_nnz[valid])
                kept_total_maximum = int(cell_totals[valid].max(initial=0))
                del row_nnz, cell_totals, cell_maxima, feature_maxima
                pointers, kept_maxima, self._keptFeatureMaxima = (
                    self._build_unordered_rows(
                        self._importLinesInMem,
                        kept_total_maximum,
                        expected=None,
                    )
                )
                self._set_row_counts(pointers, int(kept_maxima.max(initial=0)))
            else:
                self._set_row_counts(
                    cumulative_nnz(row_nnz[valid]),
                    int(cell_maxima[valid].max(initial=0)),
                )
                # The scan's feature maxima cover every raw cell, so they are
                # the kept cells' maxima only when no cell is dropped.
                self._keptFeatureMaxima = (
                    feature_maxima if valid.size == self.rawCellCount else None
                )
        except BaseException:
            self.release()
            raise

    def _ensure_archive_extracted(self) -> None:
        if self.candidate.archivePath is None or self._archiveDirectory is not None:
            return
        temporary = tempfile.TemporaryDirectory(
            prefix="cytearc-mtx-archive-",
            dir=self.tempDir,
        )
        directory = Path(temporary.name)
        self._archiveTemporary = temporary
        required = {
            self.candidate.matrixPath,
            self.candidate.featurePath,
            self.candidate.cellPath,
            self.candidate.cellMetadataPath,
            self.candidate.featureReferencePath,
        }
        members = {name for name in required if name is not None}
        try:
            with zipfile.ZipFile(self.candidate.archivePath) as archive:
                infos = _validated_zip_infos(archive)
                missing = sorted(members.difference(infos))
                if missing:
                    raise ValueError(
                        "ZIP archive is missing selected members: " + ", ".join(missing)
                    )
                for index, member in enumerate(sorted(members)):
                    suffixes = "".join(PurePosixPath(member).suffixes)
                    destination = directory / f"{index}{suffixes}"
                    with archive.open(infos[member], mode="r") as source:
                        with destination.open("wb") as target:
                            shutil.copyfileobj(source, target)
                    self._archivePaths[member] = str(destination)
            self._archiveDirectory = directory
        except BaseException:
            temporary.cleanup()
            self._archiveTemporary = None
            self._archivePaths.clear()
            raise

    def _local_path(self, path: str) -> str:
        return self._archivePaths.get(path, path)

    def _set_valid_cells(self, valid: np.ndarray) -> None:
        self.validCellIndexes = valid
        self.nCells = int(valid.size)
        self._cellNames = self._rawCellNames[valid]
        self._cellColumns = {
            name: values[valid] for name, values in self._rawCellColumns.items()
        }

    def _set_row_counts(self, pointers: np.ndarray, count_maximum: int) -> None:
        """Keep the row pointers and the largest count of the kept cells."""
        self._cumulativeRowNnz = pointers
        self.countMaximum = int(count_maximum)
        # Batches hold the kept counts in the narrowest dtype that holds them.
        self.matrixDtype: np.dtype[Any] = np.min_scalar_type(self.countMaximum)

    def _read_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if _is_bed_sidecar(self.candidate.featurePath):
            ids = self._read_bed_features()
            names = ids.copy()
            types = np.full(ids.size, "Peaks", dtype=object)
        elif _is_parse_matrix(self.candidate.matrixPath):
            frame = pd.read_csv(self.featurePath, compression="infer")
            columns = tuple(str(value) for value in frame.columns)
            id_column = _column_name(
                columns,
                ("gene_id", "feature_id", "id", "gene"),
            )
            try:
                name_column = _column_name(
                    columns,
                    ("gene_name", "feature_name", "name", "gene_symbol"),
                )
            except ValueError:
                name_column = id_column
            ids = frame[id_column].astype(str).to_numpy()
            names = frame[name_column].astype(str).to_numpy()
            types = np.full(ids.size, "Gene Expression", dtype=object)
        else:
            frame = pd.read_csv(
                self.featurePath,
                sep="\t",
                header=None,
                dtype=str,
                keep_default_na=False,
                compression="infer",
            )
            ids = frame.iloc[:, 0].astype(str).to_numpy()
            names = (
                frame.iloc[:, 1].astype(str).to_numpy()
                if frame.shape[1] > 1
                else ids.copy()
            )
            types = (
                frame.iloc[:, 2].astype(str).to_numpy()
                if frame.shape[1] > 2
                else np.full(ids.size, "Gene Expression", dtype=object)
            )
        if ids.size != self.nFeatures:
            raise ValueError(
                f"Feature sidecar has {ids.size} rows, expected {self.nFeatures}"
            )
        return ids, names, types

    def _read_bed_features(self) -> np.ndarray:
        """Return ``chrom:start-end`` identifiers for peak BED rows."""
        frame = pd.read_csv(
            self.featurePath,
            sep="\t",
            header=None,
            dtype=str,
            keep_default_na=False,
            compression="infer",
        )
        if frame.shape[1] < 3:
            raise ValueError(
                "Peak BED sidecar must contain chrom, start, and end columns"
            )
        chrom = frame.iloc[:, 0].astype(str).to_numpy()
        try:
            start = frame.iloc[:, 1].astype(np.int64).to_numpy()
            end = frame.iloc[:, 2].astype(np.int64).to_numpy()
        except ValueError as exc:
            raise ValueError(
                "Peak BED sidecar start and end columns must hold integers"
            ) from exc
        return np.asarray(
            [f"{c}:{s}-{e}" for c, s, e in zip(chrom, start, end, strict=True)],
            dtype=object,
        )

    def _read_cells(
        self,
        cell_id_key: str | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], str | None]:
        if _is_parse_matrix(self.candidate.matrixPath):
            frame = pd.read_csv(self.cellPath, compression="infer")
            aliases = [key for key in ("bc_wells", "bc_index") if key in frame.columns]
            if cell_id_key is None:
                if len(aliases) != 1:
                    reason = "absent" if not aliases else "ambiguous"
                    raise ValueError(
                        f"Parse cell-ID aliases are {reason}; pass cell_id_key "
                        "explicitly"
                    )
                selected_key = aliases[0]
            else:
                selected_key = cell_id_key
                if selected_key not in frame.columns:
                    raise KeyError(f"Cell ID column {selected_key!r} was not found")
            names = frame[selected_key].astype(str).to_numpy()
            columns = {
                str(name): frame[name].to_numpy()
                for name in frame.columns
                if name != selected_key
            }
        else:
            try:
                frame = pd.read_csv(
                    self.cellPath,
                    sep="\t",
                    header=None,
                    dtype=str,
                    keep_default_na=False,
                    compression="infer",
                )
            except pd.errors.EmptyDataError:
                # A sidecar without lines lists no cells.
                names = np.empty(0, dtype=str)
            else:
                names = frame.iloc[:, 0].astype(str).to_numpy()
            columns = {}
            selected_key = None
        if names.size != self.rawCellCount:
            raise ValueError(
                f"Cell sidecar has {names.size} rows, expected {self.rawCellCount}"
            )
        return names, columns, selected_key

    def _feature_reference_columns(self) -> dict[str, np.ndarray]:
        """Align optional feature-reference columns to the matrix features.

        A 10x Feature Reference lists only Feature Barcode features, so rows of
        other features hold ``None``. The reference never replaces the name or
        type read from the feature sidecar.
        """
        reference_path = self.candidate.featureReferencePath
        if reference_path is None:
            return {}
        path = self._local_path(reference_path)
        frame = pd.read_csv(path, compression="infer")
        columns = tuple(str(value) for value in frame.columns)
        id_column = _column_name(
            columns,
            ("id", "feature_id", "gene_id"),
        )
        reference_ids = frame[id_column].astype(str).to_numpy()
        require_unique_identifiers(reference_ids, "Feature-reference IDs")
        # The reference names features by their IDs in the feature sidecar,
        # before any repeated ID is renamed.
        unknown = sorted(set(reference_ids).difference(self._sourceFeatureIds))
        if unknown:
            raise ValueError(
                "Feature reference lists IDs that are not matrix features: "
                + ", ".join(unknown[:5])
            )
        positions = {value: index for index, value in enumerate(reference_ids)}
        order = np.fromiter(
            (positions.get(str(value), -1) for value in self._sourceFeatureIds),
            dtype=np.int64,
            count=self.nFeatures,
        )
        present = order >= 0
        result: dict[str, np.ndarray] = {}
        for name in frame.columns:
            resolved_name = str(name)
            if resolved_name == id_column or resolved_name.lower() in {
                "name",
                "feature_type",
            }:
                continue
            source = frame[name].to_numpy(dtype=object)
            values = np.full(self.nFeatures, None, dtype=object)
            values[present] = [
                None if pd.isna(value) else value for value in source[order[present]]
            ]
            result[resolved_name] = values
        return result

    def _raw_chunks(
        self,
        lines_in_mem: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.header.nEntries == 0:
            return
        separator = r"\s+" if self.separator == " " else self.separator
        count_dtype: Any = np.int64 if self.header.field == "integer" else np.float64
        try:
            stream = pd.read_csv(
                self.matrixPath,
                comment="%",
                sep=separator,
                header=0,
                names=["axis1", "axis2", "count"],
                chunksize=lines_in_mem,
                dtype={
                    "axis1": np.int64,
                    "axis2": np.int64,
                    "count": count_dtype,
                },
                compression="infer",
            )
            for frame in stream:
                axis1 = frame["axis1"].to_numpy(dtype=np.int64, copy=False)
                axis2 = frame["axis2"].to_numpy(dtype=np.int64, copy=False)
                raw_values = frame["count"].to_numpy(copy=False)
                if (
                    not np.isfinite(raw_values).all()
                    or np.any(raw_values < 0)
                    or np.any(raw_values != np.floor(raw_values))
                ):
                    raise ValueError(
                        "Matrix Market counts must be finite non-negative integers"
                    )
                values = raw_values.astype(np.uint64, copy=False)
                if self.candidate.matrixOrientation == "featuresByCells":
                    features = axis1 + self.indexOffset
                    cells = axis2 + self.indexOffset
                else:
                    cells = axis1 + self.indexOffset
                    features = axis2 + self.indexOffset
                if (
                    np.any(features < 0)
                    or np.any(features >= self.nFeatures)
                    or np.any(cells < 0)
                    or np.any(cells >= self.rawCellCount)
                ):
                    raise ValueError(
                        "Matrix Market coordinate is outside the declared dimensions"
                    )
                yield features, cells, values
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Matrix Market"):
                raise
            raise ValueError("Could not parse Matrix Market coordinates") from exc

    @staticmethod
    def _order_violation(
        first: np.ndarray,
        second: np.ndarray,
        previous: tuple[int, int] | None,
    ) -> int | None:
        if previous is None:
            joined_first = first
            joined_second = second
            first_new_offset = 1
        else:
            joined_first = np.concatenate(
                (np.array([previous[0]], dtype=np.int64), first)
            )
            joined_second = np.concatenate(
                (np.array([previous[1]], dtype=np.int64), second)
            )
            first_new_offset = 0
        invalid = (joined_first[1:] < joined_first[:-1]) | (
            (joined_first[1:] == joined_first[:-1])
            & (joined_second[1:] < joined_second[:-1])
        )
        positions = np.flatnonzero(invalid)
        return None if positions.size == 0 else int(positions[0] + first_new_offset)

    def _scan_matrix(
        self,
        lines_in_mem: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CoordinateOrder]:
        """Return the entries, total, and largest count of each raw cell, the
        largest count of each feature, and the coordinate order.

        Cell-major and feature-major coordinates keep duplicate coordinates
        adjacent, so the counts are summed over duplicates, as the import
        stores them. Coordinates in any other order are unordered. Their
        totals are exact, but duplicates that are not adjacent count as
        separate entries and are not summed into the largest counts, so
        construction takes the entries and largest counts of the kept cells
        from the sorted rows instead.
        """
        row_nnz = np.zeros(self.rawCellCount, dtype=np.int64)
        cell_totals = np.zeros(self.rawCellCount, dtype=np.uint64)
        cell_maxima = np.zeros(self.rawCellCount, dtype=np.uint64)
        feature_maxima = np.zeros(self.nFeatures, dtype=np.uint64)
        previous: tuple[int, int] | None = None
        pending: tuple[int, int, np.uint64] | None = None
        parsed = 0
        cell_major = True
        feature_major = True

        def account(
            features: np.ndarray,
            cells: np.ndarray,
            values: np.ndarray,
        ) -> None:
            np.add.at(row_nnz, cells, 1)
            np.add.at(cell_totals, cells, values)
            np.maximum.at(cell_maxima, cells, values)
            np.maximum.at(feature_maxima, features, values)

        for features, cells, values in self._raw_chunks(lines_in_mem):
            if features.size == 0:
                continue
            if cell_major or feature_major:
                if cell_major:
                    cell_previous = (
                        None if previous is None else (previous[1], previous[0])
                    )
                    cell_major = (
                        self._order_violation(cells, features, cell_previous) is None
                    )
                if feature_major:
                    feature_major = (
                        self._order_violation(features, cells, previous) is None
                    )
                previous = (int(features[-1]), int(cells[-1]))
            parsed += int(features.size)

            if pending is not None:
                features = np.concatenate(
                    (np.array([pending[0]], dtype=np.int64), features)
                )
                cells = np.concatenate((np.array([pending[1]], dtype=np.int64), cells))
                values = np.concatenate(
                    (np.array([pending[2]], dtype=np.uint64), values)
                )
            changes = (features[1:] != features[:-1]) | (cells[1:] != cells[:-1])
            starts = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.flatnonzero(changes).astype(np.int64) + 1,
                )
            )
            reduced = np.add.reduceat(values, starts)
            complete = max(0, starts.size - 1)
            if complete:
                account(
                    features[starts[:complete]],
                    cells[starts[:complete]],
                    reduced[:complete],
                )
            last = int(starts[-1])
            pending = (
                int(features[last]),
                int(cells[last]),
                np.uint64(reduced[-1]),
            )

        if parsed != self.header.nEntries:
            raise ValueError(
                f"Matrix Market header declares {self.header.nEntries} entries, "
                f"but {parsed} were read"
            )
        if pending is not None:
            account(
                np.array([pending[0]], dtype=np.int64),
                np.array([pending[1]], dtype=np.int64),
                np.array([pending[2]], dtype=np.uint64),
            )
        order: CoordinateOrder = (
            "cellMajor"
            if cell_major
            else "featureMajor"
            if feature_major
            else "unordered"
        )
        return row_nnz, cell_totals, cell_maxima, feature_maxima, order

    def kept_feature_maxima(self) -> np.ndarray:
        """Return the largest count of each feature over the kept cells.

        When the reader dropped cells from a sorted matrix, the file is read
        once more to find them. Construction finds them for an unordered
        matrix while it sorts the rows.
        """
        if self._keptFeatureMaxima is None:
            kept = np.zeros(self.rawCellCount, dtype=bool)
            kept[self.validCellIndexes] = True
            maxima = np.zeros(self.nFeatures, dtype=np.uint64)
            for features, cells, values in self._coalesced_chunks(
                self._importLinesInMem
            ):
                selected = kept[cells]
                np.maximum.at(maxima, features[selected], values[selected])
            self._keptFeatureMaxima = maxima
        return self._keptFeatureMaxima

    def _coalesced_chunks(
        self,
        lines_in_mem: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield the coordinates with each run of adjacent duplicates summed.

        The order that the scan found is checked again, so a sorted file that
        was reordered after the scan raises. Unordered coordinates are
        yielded in file order.
        """
        previous: tuple[int, int] | None = None
        pending: tuple[int, int, np.uint64] | None = None
        parsed = 0
        for features, cells, values in self._raw_chunks(lines_in_mem):
            if features.size == 0:
                continue
            if self.coordinateOrder != "unordered":
                first, second = (
                    (cells, features)
                    if self.coordinateOrder == "cellMajor"
                    else (features, cells)
                )
                invalid = self._order_violation(first, second, previous)
                if invalid is not None:
                    raise ValueError(
                        f"Matrix Market {self.coordinateOrder} order is invalid at "
                        f"entry {parsed + invalid + 1}"
                    )
                previous = (int(first[-1]), int(second[-1]))
            parsed += int(features.size)

            if pending is not None:
                features = np.concatenate(
                    (np.array([pending[0]], dtype=np.int64), features)
                )
                cells = np.concatenate((np.array([pending[1]], dtype=np.int64), cells))
                values = np.concatenate(
                    (np.array([pending[2]], dtype=np.uint64), values)
                )
            changes = (features[1:] != features[:-1]) | (cells[1:] != cells[:-1])
            starts = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.flatnonzero(changes).astype(np.int64) + 1,
                )
            )
            reduced = np.add.reduceat(values, starts)
            complete = max(0, starts.size - 1)
            if complete:
                yield (
                    features[starts[:complete]],
                    cells[starts[:complete]],
                    reduced[:complete],
                )
            last = int(starts[-1])
            pending = (
                int(features[last]),
                int(cells[last]),
                np.uint64(reduced[-1]),
            )
        if parsed != self.header.nEntries:
            raise ValueError(
                f"Matrix Market header declares {self.header.nEntries} entries, "
                f"but {parsed} were read"
            )
        if pending is not None:
            yield (
                np.array([pending[0]], dtype=np.int64),
                np.array([pending[1]], dtype=np.int64),
                np.array([pending[2]], dtype=np.uint64),
            )

    def _checked_values(self, values: np.ndarray) -> np.ndarray:
        """Return the counts of kept cells in ``matrixDtype``.

        Construction found the largest count, so a larger one means that the
        file changed since then.
        """
        if values.size and int(values.max()) > self.countMaximum:
            raise OverflowError(_CHANGED_COUNT_MESSAGE)
        return values.astype(self.matrixDtype, copy=False)

    def _prepare_feature_major(self, lines_in_mem: int = 1_000_000) -> None:
        if self._csrIndptr is not None:
            return
        parent = (
            Path(self.tempDir)
            if self.tempDir is not None
            else Path(tempfile.gettempdir())
        )
        retained_nnz = int(self._cumulativeRowNnz[-1])
        index_dtype = (
            np.int32
            if max(self.nFeatures, retained_nnz) <= np.iinfo(np.int32).max
            else np.int64
        )
        required = int(
            retained_nnz * (self.matrixDtype.itemsize + np.dtype(index_dtype).itemsize)
            + (self.nCells + 1) * np.dtype(np.int64).itemsize
        )
        self.temporaryDiskBytes = required
        free = shutil.disk_usage(parent).free
        if required > free:
            raise OSError(
                f"Feature-major Matrix Market preparation needs {required} "
                f"temporary bytes, but {free} bytes are free in {parent}"
            )
        directory = Path(tempfile.mkdtemp(prefix="cytearc-mtx-csr-", dir=str(parent)))
        try:
            indptr = np.memmap(
                directory / "indptr.bin",
                mode="w+",
                dtype=np.int64,
                shape=(self.nCells + 1,),
            )
            indptr[:] = self._cumulativeRowNnz
            data: np.ndarray | np.memmap
            indices: np.ndarray | np.memmap
            if retained_nnz:
                data = np.memmap(
                    directory / "data.bin",
                    mode="w+",
                    dtype=self.matrixDtype,
                    shape=(retained_nnz,),
                )
                indices = np.memmap(
                    directory / "indices.bin",
                    mode="w+",
                    dtype=index_dtype,
                    shape=(retained_nnz,),
                )
            else:
                data = np.empty(0, dtype=self.matrixDtype)
                indices = np.empty(0, dtype=index_dtype)
            cell_map = np.full(self.rawCellCount, -1, dtype=np.int64)
            cell_map[self.validCellIndexes] = np.arange(
                self.nCells,
                dtype=np.int64,
            )
            cursor = np.asarray(self._cumulativeRowNnz[:-1]).copy()
            for features, cells, values in self._coalesced_chunks(lines_in_mem):
                output_cells = cell_map[cells]
                selected = output_cells >= 0
                if not selected.any():
                    continue
                output_cells = output_cells[selected]
                selected_features = features[selected]
                selected_values = values[selected]
                order = np.argsort(output_cells, kind="stable")
                grouped_cells = output_cells[order]
                edges = np.concatenate(
                    (
                        np.array([0], dtype=np.int64),
                        np.flatnonzero(grouped_cells[1:] != grouped_cells[:-1]).astype(
                            np.int64
                        )
                        + 1,
                        np.array([grouped_cells.size], dtype=np.int64),
                    )
                )
                widths = np.diff(edges)
                positions = cursor[grouped_cells]
                positions += np.arange(grouped_cells.size, dtype=np.int64)
                positions -= np.repeat(edges[:-1], widths)
                cursor[grouped_cells[edges[:-1]]] += widths
                data[positions] = self._checked_values(selected_values[order])
                indices[positions] = selected_features[order]
            if not np.array_equal(cursor, self._cumulativeRowNnz[1:]):
                raise RuntimeError(
                    "Feature-major CSR preparation produced inconsistent row pointers"
                )
            for array in (data, indices, indptr):
                if isinstance(array, np.memmap):
                    array.flush()
            self._keep_rows(directory, data, indices, indptr)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def _keep_rows(
        self,
        directory: Path,
        data: np.ndarray | np.memmap,
        indices: np.ndarray | np.memmap,
        indptr: np.ndarray | np.memmap,
    ) -> None:
        """Hold prepared rows until :meth:`release` removes their directory.

        A reader that is never closed removes the directory when it is
        garbage collected or when the interpreter exits.
        """
        self._csrDirectory = directory
        # The finalizer holds the arrays, never the engine, so that it can
        # close their maps where mapped files cannot be removed.
        self._csrFinalizer = weakref.finalize(
            self, _remove_prepared_rows, directory, (data, indices, indptr)
        )
        self._csrData = data
        self._csrIndices = indices
        self._csrIndptr = indptr

    def _build_unordered_rows(
        self,
        lines_in_mem: int,
        limit: int,
        *,
        expected: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sort the coordinates of an unordered matrix into temporary rows.

        Each kept cell's coordinates are written to the slots that the scan
        counted for it. Each row is then sorted by feature, its duplicate
        coordinates are summed, and it is moved down over the slots that its
        duplicates freed, so the rows end up packed in kept-cell order. The
        rows hold the narrowest dtype that holds ``limit``, which bounds
        every count and sum; a larger one means that the file changed.

        Args:
            lines_in_mem: Matrix Market lines parsed at a time, and entries
                sorted at a time.
            limit: Largest count that a row may hold.
            expected: Row pointers that construction found, which a rebuild
                must reproduce; None at construction.

        Returns:
            The row pointers of the kept cells, the largest count of each
            kept cell, and the largest count of each feature over the kept
            cells.
        """
        assert self._slotPointers is not None
        slot_nnz = int(self._slotPointers[-1])
        value_dtype = np.min_scalar_type(limit)
        index_dtype = (
            np.int32
            if max(self.nFeatures, slot_nnz) <= np.iinfo(np.int32).max
            else np.int64
        )
        parent = (
            Path(self.tempDir)
            if self.tempDir is not None
            else Path(tempfile.gettempdir())
        )
        required = int(
            slot_nnz * (value_dtype.itemsize + np.dtype(index_dtype).itemsize)
        )
        self.temporaryDiskBytes = required
        free = shutil.disk_usage(parent).free
        if required > free:
            raise OSError(
                f"Unordered Matrix Market preparation needs {required} temporary "
                f"bytes, but {free} bytes are free in {parent}"
            )
        directory = Path(tempfile.mkdtemp(prefix="cytearc-mtx-csr-", dir=str(parent)))
        try:
            data: np.ndarray | np.memmap
            indices: np.ndarray | np.memmap
            if slot_nnz:
                data = np.memmap(
                    directory / "data.bin",
                    mode="w+",
                    dtype=value_dtype,
                    shape=(slot_nnz,),
                )
                indices = np.memmap(
                    directory / "indices.bin",
                    mode="w+",
                    dtype=index_dtype,
                    shape=(slot_nnz,),
                )
            else:
                data = np.empty(0, dtype=value_dtype)
                indices = np.empty(0, dtype=index_dtype)
            self._scatter_unordered(lines_in_mem, limit, data, indices)
            row_counts, cell_maxima, feature_maxima = self._sort_unordered_rows(
                lines_in_mem,
                limit,
                data,
                indices,
            )
            pointers = cumulative_nnz(row_counts)
            if expected is not None:
                if not np.array_equal(pointers, expected):
                    raise RuntimeError(_CHANGED_UNORDERED_ROWS_MESSAGE)
                pointers = expected
            for array in (data, indices):
                if isinstance(array, np.memmap):
                    array.flush()
            self._keep_rows(directory, data, indices, pointers)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return pointers, cell_maxima, feature_maxima

    def _scatter_unordered(
        self,
        lines_in_mem: int,
        limit: int,
        data: np.ndarray | np.memmap,
        indices: np.ndarray | np.memmap,
    ) -> None:
        """Write each kept cell's coordinates into the slots counted for it."""
        assert self._slotPointers is not None
        cell_map = np.full(self.rawCellCount, -1, dtype=np.int64)
        cell_map[self.validCellIndexes] = np.arange(self.nCells, dtype=np.int64)
        cursor = np.asarray(self._slotPointers[:-1]).copy()
        ends = self._slotPointers[1:]
        for features, cells, values in self._coalesced_chunks(lines_in_mem):
            output_cells = cell_map[cells]
            selected = output_cells >= 0
            if not selected.any():
                continue
            output_cells = output_cells[selected]
            selected_features = features[selected]
            selected_values = values[selected]
            # A count above every kept total can only come from a changed file.
            if int(selected_values.max()) > limit:
                raise OverflowError(_CHANGED_COUNT_MESSAGE)
            order = np.argsort(output_cells)
            grouped_cells = output_cells[order]
            edges = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.flatnonzero(grouped_cells[1:] != grouped_cells[:-1]).astype(
                        np.int64
                    )
                    + 1,
                    np.array([grouped_cells.size], dtype=np.int64),
                )
            )
            widths = np.diff(edges)
            first_cells = grouped_cells[edges[:-1]]
            starts = cursor[first_cells]
            if bool(np.any(starts + widths > ends[first_cells])):
                raise RuntimeError(_CHANGED_UNORDERED_ROWS_MESSAGE)
            positions = np.repeat(starts - edges[:-1], widths)
            positions += np.arange(grouped_cells.size, dtype=np.int64)
            cursor[first_cells] += widths
            data[positions] = selected_values[order]
            indices[positions] = selected_features[order]
        if not np.array_equal(cursor, ends):
            raise RuntimeError(_CHANGED_UNORDERED_ROWS_MESSAGE)

    def _sort_unordered_rows(
        self,
        lines_in_mem: int,
        limit: int,
        data: np.ndarray | np.memmap,
        indices: np.ndarray | np.memmap,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sort each row by feature, sum its duplicates, and pack the rows.

        Rows are read in blocks of at most ``lines_in_mem`` slots, or of one
        row when it holds more. A row never grows, so each block is written
        at or before the slots it was read from, and never over slots that
        are still to be read.

        Returns:
            The number of distinct features and the largest count of each
            kept cell, and the largest count of each feature over the kept
            cells.
        """
        assert self._slotPointers is not None
        slots = self._slotPointers
        row_counts = np.zeros(self.nCells, dtype=np.int64)
        cell_maxima = np.zeros(self.nCells, dtype=np.uint64)
        feature_maxima = np.zeros(self.nFeatures, dtype=np.uint64)
        width = max(1, self.nFeatures)
        # A block's keys, its local row times the feature count plus the
        # feature, must fit int64.
        max_block_rows = int(np.iinfo(np.int64).max // width)
        written = 0
        row = 0
        while row < self.nCells:
            start = int(slots[row])
            stop_row = int(np.searchsorted(slots, start + lines_in_mem, side="right"))
            stop_row = min(
                self.nCells, max(row + 1, stop_row - 1), row + max_block_rows
            )
            stop = int(slots[stop_row])
            if stop > start:
                block_features = np.array(indices[start:stop], dtype=np.int64)
                block_values = np.array(data[start:stop], dtype=np.uint64)
                local_rows = np.repeat(
                    np.arange(stop_row - row, dtype=np.int64),
                    np.diff(slots[row : stop_row + 1]),
                )
                keys = local_rows * width + block_features
                order = np.argsort(keys)
                keys = keys[order]
                heads = np.flatnonzero(
                    np.concatenate((np.array([True]), keys[1:] != keys[:-1]))
                )
                sums = np.add.reduceat(block_values[order], heads)
                if int(sums.max()) > limit:
                    raise OverflowError(_CHANGED_COUNT_MESSAGE)
                keys = keys[heads]
                rows = keys // width
                columns = keys - rows * width
                row_counts[row:stop_row] = np.bincount(rows, minlength=stop_row - row)
                np.maximum.at(cell_maxima[row:stop_row], rows, sums)
                np.maximum.at(feature_maxima, columns, sums)
                end = written + int(sums.size)
                data[written:end] = sums
                indices[written:end] = columns
                written = end
            row = stop_row
        return row_counts, cell_maxima, feature_maxima

    def _prepare_unordered(self, lines_in_mem: int) -> None:
        """Sort the rows again after a release removed them."""
        if self._csrIndptr is not None:
            return
        # The counts of the kept cells were found when the rows were first
        # sorted, so the rebuilt rows hold them in matrixDtype.
        self._build_unordered_rows(
            lines_in_mem,
            self.countMaximum,
            expected=self._cumulativeRowNnz,
        )

    def configure_import_lines(self, lines_in_mem: int) -> None:
        self._importLinesInMem = int(lines_in_mem)

    def prepare(self, lines_in_mem: int | None = None) -> None:
        resolved_lines = (
            self._importLinesInMem if lines_in_mem is None else int(lines_in_mem)
        )
        self.configure_import_lines(resolved_lines)
        self._ensure_archive_extracted()
        if self.candidate.archivePath is not None:
            self.matrixPath = self._local_path(self.candidate.matrixPath)
        if self.coordinateOrder == "featureMajor":
            try:
                self._prepare_feature_major(resolved_lines)
            except BaseException:
                self.release()
                raise
        elif self.coordinateOrder == "unordered":
            try:
                self._prepare_unordered(resolved_lines)
            except BaseException:
                self.release()
                raise

    @staticmethod
    def _close_array(array: np.ndarray | np.memmap | None) -> None:
        # Only a memory map holds a file mapping to close.
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def release(self) -> None:
        for array in (self._csrData, self._csrIndices, self._csrIndptr):
            self._close_array(array)
        self._csrData = None
        self._csrIndices = None
        self._csrIndptr = None
        gc.collect()
        if self._csrFinalizer is not None:
            self._csrFinalizer.detach()
            self._csrFinalizer = None
        if self._csrDirectory is not None:
            shutil.rmtree(self._csrDirectory, ignore_errors=True)
            self._csrDirectory = None
        if self._archiveTemporary is not None:
            self._archiveTemporary.cleanup()
            self._archiveTemporary = None
            self._archiveDirectory = None
            self._archivePaths.clear()

    def resident_bytes(self) -> int:
        slot_bytes = 0 if self._slotPointers is None else self._slotPointers.nbytes
        return int(
            self.validCellIndexes.nbytes + self._cumulativeRowNnz.nbytes + slot_bytes
        )

    def max_window_nnz(self, window_rows: int) -> int:
        return max_window_nnz(self._cumulativeRowNnz, window_rows)

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lines_in_mem <= 0:
            raise ValueError("lines_in_mem must be positive")
        parser_value_bytes = (
            2 * np.dtype(np.int64).itemsize + np.dtype(np.uint64).itemsize
        )
        source_nnz = self.max_window_nnz(batch_size)
        cell_map_bytes = (
            self.rawCellCount * np.dtype(np.int64).itemsize
            if self.coordinateOrder == "cellMajor"
            else 0
        )
        # The rows that construction sorts for an unordered matrix hold a
        # dtype that holds every kept total, and each batch is cast from it.
        row_value_bytes = (
            self._csrData.dtype.itemsize
            if self.coordinateOrder == "unordered"
            and self._csrData is not None
            and self._csrData.dtype != self.matrixDtype
            else 0
        )
        return int(
            lines_in_mem * (4 * parser_value_bytes + np.dtype(np.bool_).itemsize)
            + source_nnz
            * (
                2 * np.dtype(np.int64).itemsize
                + self.matrixDtype.itemsize
                + row_value_bytes
            )
            + cell_map_bytes
        )

    def consume(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> Generator[coo_matrix, None, None]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lines_in_mem <= 0:
            raise ValueError("lines_in_mem must be positive")
        try:
            if self.coordinateOrder in ("featureMajor", "unordered"):
                self.prepare(lines_in_mem)
                source = self._consume_rows(batch_size)
            else:
                self._ensure_archive_extracted()
                if self.candidate.archivePath is not None:
                    self.matrixPath = self._local_path(self.candidate.matrixPath)
                source = self._consume_cell_major(batch_size, lines_in_mem)
            yield from source
        finally:
            self.release()

    def _consume_rows(
        self,
        batch_size: int,
    ) -> Generator[coo_matrix, None, None]:
        """Yield batches of the rows prepared for a feature-major or an
        unordered matrix."""
        # consume prepares the CSR arrays before it reads them.
        assert self._csrData is not None
        assert self._csrIndices is not None
        assert self._csrIndptr is not None
        for start in range(0, self.nCells, batch_size):
            end = min(start + batch_size, self.nCells)
            pointers = np.asarray(self._csrIndptr[start : end + 1])
            value_start = int(pointers[0])
            value_end = int(pointers[-1])
            local = pointers - value_start
            values = np.array(self._csrData[value_start:value_end], copy=True)
            if values.dtype != self.matrixDtype:
                values = self._checked_values(values)
            matrix = csr_matrix(
                (
                    values,
                    np.array(self._csrIndices[value_start:value_end], copy=True),
                    local,
                ),
                shape=(end - start, self.nFeatures),
            )
            yield matrix.tocoo(copy=False)

    def _consume_cell_major(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> Generator[coo_matrix, None, None]:
        cell_map = np.full(self.rawCellCount, -1, dtype=np.int64)
        cell_map[self.validCellIndexes] = np.arange(
            self.nCells,
            dtype=np.int64,
        )
        batch_count = (self.nCells + batch_size - 1) // batch_size
        current_batch = 0
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []

        def matrix_for(batch_index: int) -> coo_matrix:
            start = batch_index * batch_size
            end = min(start + batch_size, self.nCells)
            rows = (
                np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
            )
            columns = (
                np.concatenate(column_parts)
                if column_parts
                else np.empty(0, dtype=np.int64)
            )
            values = (
                np.concatenate(value_parts)
                if value_parts
                else np.empty(0, dtype=self.matrixDtype)
            )
            return coo_matrix(
                (values, (rows, columns)),
                shape=(end - start, self.nFeatures),
                dtype=self.matrixDtype,
            )

        for features, cells, values in self._coalesced_chunks(lines_in_mem):
            output_cells = cell_map[cells]
            selected = output_cells >= 0
            output_cells = output_cells[selected]
            features = features[selected]
            values = self._checked_values(values[selected])
            if output_cells.size == 0:
                continue
            batch_ids = output_cells // batch_size
            position = 0
            while position < output_cells.size:
                batch_id = int(batch_ids[position])
                while current_batch < batch_id:
                    yield matrix_for(current_batch)
                    row_parts.clear()
                    column_parts.clear()
                    value_parts.clear()
                    current_batch += 1
                end = int(np.searchsorted(batch_ids, batch_id, side="right"))
                row_parts.append(
                    output_cells[position:end] - current_batch * batch_size
                )
                column_parts.append(features[position:end])
                value_parts.append(values[position:end])
                position = end
        while current_batch < batch_count:
            yield matrix_for(current_batch)
            row_parts.clear()
            column_parts.clear()
            value_parts.clear()
            current_batch += 1

    def feature_ids(self) -> list[str]:
        return [str(value) for value in self._rawFeatureIds]

    def feature_names(self) -> list[str]:
        return [str(value) for value in self._rawFeatureNames]

    def feature_types(self) -> list[str]:
        return [str(value) for value in self._rawFeatureTypes]

    def cell_names(self) -> list[str]:
        return [str(value) for value in self._cellNames]

    def cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._cellColumns.items()

    def feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._featureColumns.items()


class MtxReader(CrReader):
    """Read a selected Matrix Market triplet as cell-row sparse batches.

    Construction reads the matrix to validate it and to find its coordinate
    order, the entries of each cell, and the largest count of each kept cell
    and each feature, with duplicate coordinates summed. Batches hold the
    counts in the narrowest unsigned dtype that holds the largest kept count.

    Coordinates may come in any order. A matrix sorted by cell or by feature
    is read once at construction. Any other order, such as the descending
    features within each cell that Cell Ranger 1 and 2 write, is unordered,
    and construction reads it once more to write the entries of each kept
    cell to temporary rows on disk, sorted by feature and with duplicate
    coordinates summed wherever they are in the file. Imports read the cells
    of an unordered or feature-major matrix from these rows; a feature-major
    matrix is sorted into them when an import starts. ``temporaryDiskBytes``
    reports the size of the rows, which the end of an import and ``close``
    remove.

    Args:
        matrix_path: A candidate returned by :func:`~cytearc.inspect_mtx`, or
            the path of a Matrix Market file.
        feature_path: Feature sidecar of a matrix path. Not allowed with a
            candidate.
        cell_path: Cell sidecar of a matrix path. Not allowed with a
            candidate.
        feature_reference_path: Optional 10x Feature Reference CSV whose
            columns are added to the features it lists.
        cell_id_key: Parse Biosciences cell metadata column that holds the
            cell IDs. Required unless the metadata holds exactly one of
            ``bc_wells`` and ``bc_index``.
        mtx_separator: Column delimiter in the MTX file.
        index_offset: Value added to each MTX coordinate. The default turns
            one-based coordinates into zero-based indexes.
        is_filtered: Whether the cells are already filtered. When False,
            only cells whose total count exceeds ``filtering_cutoff`` are
            kept.
        filtering_cutoff: Total count that a cell must exceed to be kept
            when ``is_filtered`` is False.
        temp_dir: Parent directory of the temporary files: the selected
            members of a ZIP archive and the sorted rows. None uses the
            system temporary directory.
        duplicate_feature_ids: ``"error"`` rejects repeated feature IDs.
            ``"make_unique"`` keeps the first occurrence of each ID and
            renames each later one, in file order and across assays, by
            appending ``-1``, ``-2``, and so on (skipping names that are
            already IDs), as AnnData's ``var_names_make_unique`` does, and
            logs a warning. Feature names are never changed, and repeated
            cell IDs always raise.

    Attributes:
        coordinateOrder: ``"cellMajor"`` or ``"featureMajor"`` for sorted
            coordinates, otherwise ``"unordered"``.
        temporaryDiskBytes: Bytes of the temporary rows that the reader
            prepared most recently.
    """

    def __init__(
        self,
        matrix_path: MtxCandidate | str,
        feature_path: str | None = None,
        cell_path: str | None = None,
        *,
        feature_reference_path: str | None = None,
        cell_id_key: str | None = None,
        mtx_separator: str = " ",
        index_offset: int = -1,
        is_filtered: bool = True,
        filtering_cutoff: int = 500,
        temp_dir: str | None = None,
        duplicate_feature_ids: DuplicateFeatureIds = "error",
    ) -> None:
        if isinstance(matrix_path, MtxCandidate):
            if feature_path is not None or cell_path is not None:
                raise ValueError("Do not pass explicit sidecars with an MtxCandidate")
            candidate = replace(
                matrix_path,
                featureReferencePath=(
                    feature_reference_path
                    if feature_reference_path is not None
                    else matrix_path.featureReferencePath
                ),
            )
        else:
            if feature_path is None or cell_path is None:
                raise ValueError(
                    "Explicit Matrix Market construction requires feature_path "
                    "and cell_path"
                )
            candidate = _explicit_candidate(
                matrix_path,
                feature_path,
                cell_path,
                feature_reference_path=feature_reference_path,
            )
        self._engine = _MtxEngine(
            candidate,
            cell_id_key=cell_id_key,
            separator=mtx_separator,
            index_offset=index_offset,
            is_filtered=is_filtered,
            filtering_cutoff=filtering_cutoff,
            temp_dir=temp_dir,
            duplicate_feature_ids=duplicate_feature_ids,
        )
        self.validBarcodeIdx = self._engine.validCellIndexes
        self.matrixEntryCount = self._engine.matrixEntryCount
        self.coordinateOrder = self._engine.coordinateOrder
        self.selectedCellIdKey = self._engine.selectedCellIdKey
        self.temporaryDiskBytes = self._engine.temporaryDiskBytes
        try:
            super().__init__(self._handle_version())
        except BaseException:
            # The rows of an unordered matrix exist from construction on.
            self._engine.release()
            raise
        self.nCells = self._engine.nCells

    def _handle_version(self) -> dict[str, str]:
        return {
            "feature_ids": "feature_ids",
            "feature_names": "feature_names",
            "feature_types": "feature_types",
            "cell_names": "cell_names",
        }

    def _read_dataset(self, key: str) -> list[str]:
        values = {
            "feature_ids": self._engine.feature_ids,
            "feature_names": self._engine.feature_names,
            "feature_types": self._engine.feature_types,
            "cell_names": self._engine.cell_names,
        }
        return values[key]()

    @property
    def matrix_dtype(self) -> np.dtype[Any]:
        return self._engine.matrixDtype

    def count_value_ranges(
        self, maxBytes: int, featureGroups: np.ndarray | None = None
    ) -> list[CountValueRange]:
        """Return the range of the counts of each group of features.

        Matrix Market counts are non-negative integers, and construction
        already scanned them with duplicate coordinates summed and kept the
        largest count of each feature, so this does not use ``maxBytes``.
        When the reader dropped cells from a sorted matrix, several groups
        need one more read of the file to find the largest counts of the kept
        cells.
        """
        value_ranges = new_count_ranges(featureGroups)
        if featureGroups is None or len(value_ranges) == 1:
            value_ranges[0].maximum = self._engine.countMaximum
            return value_ranges
        maxima = self._engine.kept_feature_maxima()
        for code, value_range in enumerate(value_ranges):
            value_range.maximum = int(maxima[featureGroups == code].max(initial=0))
        return value_ranges

    def consume(
        self,
        batch_size: int,
        lines_in_mem: int = 100000,
    ) -> Generator[coo_matrix, None, None]:
        """Yield chunks of cell rows from the Matrix Market file.

        Args:
            batch_size: Number of cells per yielded chunk.
            lines_in_mem: Number of Matrix Market lines parsed at a time.

        Yields:
            scipy.sparse.coo_matrix chunks in ``matrix_dtype``.
        """
        yield from self._engine.consume(batch_size, lines_in_mem)

    def max_window_nnz(self, window_rows: int) -> int:
        return self._engine.max_window_nnz(window_rows)

    def producer_staging_bytes(
        self,
        batch_size: int,
        lines_in_mem: int,
    ) -> int:
        return self._engine.producer_staging_bytes(batch_size, lines_in_mem)

    def get_cell_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from self._engine.cell_columns()

    def get_feature_columns(self) -> Iterator[tuple[str, np.ndarray]]:
        yield from super().get_feature_columns()
        yield from self._engine.feature_columns()

    def _set_sparse_import_lines_in_mem(self, lines_in_mem: int) -> None:
        self._engine.configure_import_lines(lines_in_mem)

    def _prepare_sparse_import(self) -> None:
        self._engine.prepare()
        self.temporaryDiskBytes = self._engine.temporaryDiskBytes

    def _release_sparse_import(self) -> None:
        self._engine.release()

    def _sparse_import_resident_bytes(self) -> int:
        return self._engine.resident_bytes()

    def close(self) -> None:
        self._engine.release()


class CrDirReader(MtxReader):
    """Read a Cell Ranger output directory in Matrix Market format.

    The directory must contain exactly one complete matrix, feature, and
    barcode triplet, such as ``matrix.mtx.gz``, ``features.tsv.gz``, and
    ``barcodes.tsv.gz``. Use :func:`~cytearc.inspect_mtx` and
    :class:`~cytearc.readers.MtxReader` to choose among several triplets. All
    reading and streaming methods come from
    :class:`~cytearc.readers.MtxReader`.

    Cell Ranger 1 and 2 list the features of each barcode in descending
    order, so construction sorts their matrices into temporary rows, as
    :class:`~cytearc.readers.MtxReader` sorts every unordered matrix.

    Args:
        loc: Path to the directory that contains the Cell Ranger output.
        mtx_separator: Column delimiter in the MTX file.
        index_offset: Value added to each MTX coordinate. The default turns
            one-based coordinates into zero-based indexes.
        is_filtered: Whether the barcodes are already filtered. When False,
            only barcodes whose total count exceeds ``filtering_cutoff`` are
            kept.
        filtering_cutoff: Total count that a barcode must exceed to be kept
            when ``is_filtered`` is False.
        temp_dir: Parent directory of the temporary files that
            :class:`~cytearc.readers.MtxReader` describes. None uses the
            system temporary directory.
        duplicate_feature_ids: ``"error"`` rejects repeated feature IDs, and
            ``"make_unique"`` renames each later occurrence, as
            :class:`~cytearc.readers.MtxReader` describes.
    """

    def __init__(
        self,
        loc: str,
        mtx_separator: str = " ",
        index_offset: int = -1,
        is_filtered: bool = True,
        filtering_cutoff: int = 500,
        *,
        temp_dir: str | None = None,
        duplicate_feature_ids: DuplicateFeatureIds = "error",
    ) -> None:
        candidates = inspect_mtx(loc)
        if len(candidates) != 1:
            raise ValueError(
                "CrDirReader requires one complete Matrix Market triplet; "
                "use inspect_mtx and MtxReader to select among candidates"
            )
        super().__init__(
            candidates[0],
            mtx_separator=mtx_separator,
            index_offset=index_offset,
            is_filtered=is_filtered,
            filtering_cutoff=filtering_cutoff,
            temp_dir=temp_dir,
            duplicate_feature_ids=duplicate_feature_ids,
        )
