import math
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from scipy.sparse import coo_matrix

from ..storage.types import as_zarr_array, as_zarr_group
from ..readers import H5adReader
from ..readers.h5ad import CategoryCodes, TableColumn, _H5adAssayFeatures
from ..storage import ArtifactRef
from ..storage.budget import ResourceBudget
from ..storage.count_dtype import count_storage_dtype
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..utils.digest import file_sha256
from ..utils.logging import logger

from ..utils.memory_budget import MemoryBudgetError
from ..utils.progress import iter_progress
from ..utils.arguments import bool_argument
from ..utils.matrix_values import ImportValues, MatrixValuesKind


# Categories longer than this list are not stored with their column: AnnData
# makes a categorical of any text that repeats a value, so a column of nearly
# unique text can hold about one category per row.
_MAX_STORED_CATEGORIES = 10_000
# A dense matrix is read about this many values at a time, as its value scan
# reads it, or in whole rows of its HDF5 chunks. Each batch is copied into a
# band of every assay, so a batch need not hold a destination row band.
_DENSE_BATCH_VALUES = 1 << 20


def _stored_levels(dtype: np.dtype[Any], categories: Sequence[Any]) -> list[Any] | None:
    """Return categories as the stored column holds its values, or None.

    A text column holds text, and a boolean, integer, or float column holds
    categories of that type. Categories that the column's type cannot hold
    exactly, such as text categories of a numeric column, return None.
    """
    if dtype.kind == "U":
        return [
            category.decode("utf-8") if isinstance(category, bytes) else str(category)
            for category in categories
        ]
    checks = {
        "b": lambda value: isinstance(value, bool),
        "i": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "f": lambda value: (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ),
    }
    check = checks.get("i" if dtype.kind == "u" else dtype.kind)
    if check is None or not all(check(category) for category in categories):
        return None
    if dtype.kind == "f":
        return [float(category) for category in categories]
    return list(categories)


def _categorical_attributes(
    dtype: np.dtype[Any],
    categories: Any,
    *,
    colors: tuple[str, ...] | None,
    has_missing: bool,
    display: bool,
) -> dict[str, Any]:
    """Return the attributes that keep a source categorical with its column.

    ``levels`` holds every category in its source order, unused ones
    included, ``ordered`` whether they are ordered, and for cell columns
    ``display`` the categorical display contract that plots honor, with the
    source colors when the file holds them. A categorical whose categories
    the column cannot hold, or that has more than 10,000 categories, gets no
    attributes.
    """
    if len(categories.values) > _MAX_STORED_CATEGORIES:
        return {}
    levels = _stored_levels(dtype, categories.values)
    if levels is None:
        return {}
    attributes: dict[str, Any] = {"levels": levels, "ordered": categories.ordered}
    if display:
        from ..metadata.artifacts import levels_display

        contract = levels_display(levels, colors, has_missing)
        if contract is not None:
            attributes["display"] = contract
    return attributes


def _column_rows(column: TableColumn, rows: np.ndarray | None) -> TableColumn:
    """Return the given rows of a source column, or all of them for None."""
    if rows is None:
        return column
    if isinstance(column, CategoryCodes):
        return CategoryCodes(column.codes[rows], column.categories)
    values, missing = column
    return values[rows], missing[rows]


def _write_column(
    group: Any,
    name: str,
    column: TableColumn,
    *,
    profile: StorageProfile,
    categorical: bool = False,
    ordered: bool = False,
) -> np.ndarray:
    """Write a source column as a metadata column and return its missing rows.

    A categorical is written from its codes, so its categories are converted
    once each rather than once per row. With ``categorical`` it is stored as
    a categorical column, its codes beside its categories, which ``ordered``
    records as ordered or not; otherwise as a column of its labels.
    """
    from ._store import write_categorical_column, write_metadata_column

    if isinstance(column, CategoryCodes):
        if categorical:
            return write_categorical_column(
                group,
                name,
                column.codes,
                column.categories,
                ordered=ordered,
                profile=profile,
            )
        write_metadata_column(
            group, name, column.codes, profile=profile, categories=column.categories
        )
        return column.missing
    values, missing = column
    write_metadata_column(group, name, values, missing, profile=profile)
    return missing


def _count_matrix_bands(
    matrix: Any,
    buffers: dict[str, Any],
    assay_names: tuple[str, ...],
    projection: tuple[np.ndarray, np.ndarray] | None,
    linear: Mapping[str, MatrixValuesKind],
) -> Iterator[tuple[str, Any, int]]:
    """Split a source batch into the bands of each assay.

    ``linear`` maps an assay to the log kind whose log its values undo.
    """
    from ..utils.arrays import sparse_matrix_bytes
    from ..utils.matrix_values import linear_coo

    chunk = matrix.tocoo(copy=False)
    source_bytes = sparse_matrix_bytes(matrix, chunk)
    if projection is None:
        if len(assay_names) != 1:
            raise RuntimeError("Multi-assay projection was not initialized")
        assay_name = assay_names[0]
        buffer = buffers[assay_name]
        if assay_name in linear:
            chunk = linear_coo(
                chunk, linear[assay_name], buffer.dtype, assay=assay_name
            )
        for band in buffer.add(chunk):
            producer_bytes = source_bytes + sum(
                item.residentBytes for item in buffers.values()
            )
            yield assay_name, band, producer_bytes
            # The consumer holds the band it was handed; keeping it here would
            # hold its values beside the rest of the batch.
            del band
        return

    codes, columns = projection
    batch_codes = codes[chunk.col]
    batch_columns = columns[chunk.col]
    source_bytes += batch_codes.nbytes + batch_columns.nbytes
    for code, assay_name in enumerate(assay_names):
        buffer = buffers[assay_name]
        selected = batch_codes == code
        projected = coo_matrix(
            (
                chunk.data[selected],
                (chunk.row[selected], batch_columns[selected]),
            ),
            shape=(chunk.shape[0], buffer.nColumns),
        )
        # An assay's columns follow the order of its source features, so the
        # projection of a canonical batch is canonical.
        projected.has_canonical_format = bool(chunk.has_canonical_format)
        if assay_name in linear:
            projected = linear_coo(
                projected, linear[assay_name], buffer.dtype, assay=assay_name
            )
        for band in buffer.add(projected):
            producer_bytes = (
                source_bytes
                + selected.nbytes
                + sparse_matrix_bytes(projected)
                + sum(item.residentBytes for item in buffers.values())
            )
            yield assay_name, band, producer_bytes
            del band


def _finished_count_bands(
    buffers: dict[str, Any],
) -> Iterator[tuple[str, Any, int]]:
    for assay_name, buffer in buffers.items():
        for band in buffer.finish():
            producer_bytes = sum(item.residentBytes for item in buffers.values())
            yield assay_name, band, producer_bytes


def _dense_count_bands(
    reader: H5adReader,
    batch_rows: int,
    row_start: int,
    row_end: int,
    buffers: Mapping[str, Any],
    columns: Mapping[str, np.ndarray] | None,
    linear: Mapping[str, MatrixValuesKind],
    stop: Any | None = None,
) -> Iterator[tuple[str, Any]]:
    """Copy dense rows ``[row_start, row_end)`` into the bands of each assay.

    Yields each assay's complete bands from its :class:`DenseBandBuffer` in
    ``buffers``. ``columns`` holds the source columns of each assay, or None
    for one assay of every column, and ``linear`` maps an assay to the log
    kind whose log its values undo. A ``stop`` that is set ends the rows
    early.
    """
    from ..utils.matrix_values import linear_values

    for values in reader.consume_dense_rows(batch_rows, row_start, row_end):
        if stop is not None and stop.is_set():
            return
        for assay_name, buffer in buffers.items():
            part = values if columns is None else values[:, columns[assay_name]]
            if assay_name in linear:
                part = linear_values(
                    part, linear[assay_name], buffer.dtype, assay=assay_name
                )
            if part.dtype.kind == "f" and buffer.dtype.kind == "f":
                # A sparse batch leaves out negative zeros, so the zeros that
                # dense files stored were positive, as adding zero makes them.
                np.add(part, 0, out=part)
            for band in buffer.add(part):
                yield assay_name, band
            del part
        del values
    for buffer in buffers.values():
        buffer.finish()


def _producer_process_bytes(producers: int, reserveBytes: int) -> int:
    """Return the bytes that producer processes hold while the parent writes.

    A single producer is a generator in the writing process and is suspended
    while bands are written. Several producers run in their own processes and
    keep buffering during the writes, so each holds its whole reserve.
    """
    return int(producers) * int(reserveBytes) if int(producers) > 1 else 0


def _run_worker(
    reader_kwargs: dict[str, Any],
    connection: Any,
    work: Any,
) -> None:
    """Run one H5AD worker body and report failures over its pipe."""
    reader: H5adReader | None = None
    try:
        reader = H5adReader(**reader_kwargs)
        work(reader)
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
    finally:
        if reader is not None:
            reader.close()
        connection.close()


def _worker_messages(
    connections: dict[Any, int],
    workers: list[Any],
    role: str,
) -> Iterator[tuple[int, str, Any]]:
    """Yield ``(worker, kind, payload)`` messages until every worker is done.

    A worker fails only when it exits with a non-zero code; a clean exit whose
    ``done`` message is still in its pipe is read on the next wait.
    """
    from multiprocessing.connection import wait

    active = dict(connections)
    while active:
        ready = cast(list[Any], wait(tuple(active), timeout=0.5))
        if not ready:
            failed = [
                workers[index]
                for index in active.values()
                if workers[index].exitcode not in (None, 0)
            ]
            if failed:
                details = ", ".join(
                    f"{worker.name} exitcode={worker.exitcode}" for worker in failed
                )
                raise RuntimeError(f"H5AD {role} process failed: {details}")
            continue
        for connection in ready:
            index = active[connection]
            try:
                kind, payload = connection.recv()
            except EOFError as exc:
                raise RuntimeError(
                    f"H5AD {role} {index} closed without a result"
                ) from exc
            if kind == "error":
                raise RuntimeError(f"H5AD {role} {index} failed: {payload}")
            if kind == "done":
                active.pop(connection)
                connection.close()
            yield index, kind, payload


def _stop_workers(
    stop: Any,
    workers: list[Any],
    connections: list[Any],
) -> None:
    """Stop workers, draining their pipes so a blocked send can finish."""
    from multiprocessing.connection import wait

    stop.set()
    deadline = time.monotonic() + 5.0
    while any(worker.is_alive() for worker in workers) and time.monotonic() < deadline:
        open_connections = [
            connection for connection in connections if not connection.closed
        ]
        ready = cast(
            list[Any],
            wait(tuple(open_connections), timeout=0.05) if open_connections else [],
        )
        for connection in ready:
            try:
                connection.recv()
            except (EOFError, OSError):
                pass
        for worker in workers:
            worker.join(timeout=0)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=1.0)
        if worker.is_alive():
            worker.kill()
            worker.join()
    for connection in connections:
        connection.close()


def _read_h5ad_process_window(
    reader_kwargs: dict[str, Any],
    batch_size: int,
    row_start: int,
    row_end: int,
    destination_specs: dict[str, Any],
    assay_names: tuple[str, ...],
    projection: tuple[np.ndarray, np.ndarray] | None,
    connection: Any,
    stop: Any,
    linear: Mapping[str, MatrixValuesKind],
) -> None:
    from ..storage.sharding import SparseShardBuffer

    def send_band(item: tuple[str, Any, int]) -> bool:
        if stop.is_set():
            return False
        connection.send(("band", item))
        return not stop.is_set()

    def work(reader: H5adReader) -> None:
        buffers = {
            assay_name: SparseShardBuffer(
                destination,
                startRow=row_start,
                endRow=row_end,
            )
            for assay_name, destination in destination_specs.items()
        }
        for matrix in reader.consume_row_range(batch_size, row_start, row_end):
            if stop.is_set():
                return
            for item in _count_matrix_bands(
                matrix,
                buffers,
                assay_names,
                projection,
                linear,
            ):
                if not send_band(item):
                    return
                # The band was sent; its copy here would sit beside the next one.
                del item
        for item in _finished_count_bands(buffers):
            if not send_band(item):
                return
        connection.send(("done", None))

    _run_worker(reader_kwargs, connection, work)


def _write_h5ad_process_window(
    reader_kwargs: dict[str, Any],
    batch_size: int,
    row_start: int,
    row_end: int,
    zarr_location: str,
    storage_options: dict[str, Any] | None,
    workspace: str | None,
    assay_names: tuple[str, ...],
    projection: tuple[np.ndarray, np.ndarray] | None,
    resources: ResourceBudget,
    resident_bytes: int,
    producer_reserve_bytes: int,
    io: StorageIoPolicy | None,
    connection: Any,
    stop: Any,
    linear: Mapping[str, MatrixValuesKind],
) -> None:
    from ..storage.execution import execution_report_scope
    from ..storage.identity import CountSummary
    from ..storage.schema import load_count_array
    from ..storage.sharding import (
        SparseShardBuffer,
        SparseWriteBand,
        write_sparse_bands,
    )
    from ..storage.stores import load_zarr

    def work(reader: H5adReader) -> None:
        root = load_zarr(
            zarr_location,
            mode="a",
            storage_options=storage_options,
        )
        destinations = {
            assay_name: load_count_array(root, assay_name, workspace)
            for assay_name in assay_names
        }
        buffers = {
            assay_name: SparseShardBuffer(
                destination,
                startRow=row_start,
                endRow=row_end,
            )
            for assay_name, destination in destinations.items()
        }
        summaries = {
            assay_name: CountSummary(destination, rows=(row_start, row_end))
            for assay_name, destination in destinations.items()
        }

        def writes() -> Iterator[SparseWriteBand]:
            for matrix in reader.consume_row_range(batch_size, row_start, row_end):
                if stop.is_set():
                    return
                for assay_name, band, producer_bytes in _count_matrix_bands(
                    matrix,
                    buffers,
                    assay_names,
                    projection,
                    linear,
                ):
                    yield SparseWriteBand(
                        destinations[assay_name],
                        band,
                        producer_bytes,
                    )
                    del band
            if stop.is_set():
                return
            for assay_name, band, producer_bytes in _finished_count_bands(buffers):
                yield SparseWriteBand(
                    destinations[assay_name],
                    band,
                    producer_bytes,
                )

        with execution_report_scope() as reports:
            write_sparse_bands(
                writes(),
                resources=resources,
                residentBytes=resident_bytes
                + sum(summary.nbytes for summary in summaries.values()),
                producerReserveBytes=producer_reserve_bytes,
                io=io,
                countSummaries={
                    destinations[name].path: summary
                    for name, summary in summaries.items()
                },
            )
        if not stop.is_set():
            windows = {name: summary.window() for name, summary in summaries.items()}
            connection.send(("done", (reports, windows)))

    _run_worker(reader_kwargs, connection, work)


def _read_dense_h5ad_process_window(
    reader_kwargs: dict[str, Any],
    batch_rows: int,
    row_start: int,
    row_end: int,
    destination_specs: dict[str, Any],
    columns: Mapping[str, np.ndarray] | None,
    connection: Any,
    stop: Any,
    linear: Mapping[str, MatrixValuesKind],
) -> None:
    """Send the dense bands of each assay over a row window to the parent."""
    from ..storage.sharding import DenseBandBuffer

    def work(reader: H5adReader) -> None:
        buffers = {
            assay_name: DenseBandBuffer(spec, startRow=row_start, endRow=row_end)
            for assay_name, spec in destination_specs.items()
        }
        for item in _dense_count_bands(
            reader, batch_rows, row_start, row_end, buffers, columns, linear, stop
        ):
            if stop.is_set():
                return
            connection.send(("band", item))
            if stop.is_set():
                return
        if not stop.is_set():
            connection.send(("done", None))

    _run_worker(reader_kwargs, connection, work)


def _write_dense_h5ad_process_window(
    reader_kwargs: dict[str, Any],
    batch_rows: int,
    row_start: int,
    row_end: int,
    zarr_location: str,
    storage_options: dict[str, Any] | None,
    workspace: str | None,
    assay_names: tuple[str, ...],
    columns: Mapping[str, np.ndarray] | None,
    plan: Any,
    connection: Any,
    stop: Any,
    linear: Mapping[str, MatrixValuesKind],
) -> None:
    """Write the dense bands of each assay over a row window with ``plan``."""
    from ..storage.execution import execution_report_scope
    from ..storage.identity import CountSummary
    from ..storage.schema import load_count_array
    from ..storage.sharding import DenseBandBuffer, DenseWriteBand, write_dense_bands
    from ..storage.stores import load_zarr

    def work(reader: H5adReader) -> None:
        root = load_zarr(zarr_location, mode="a", storage_options=storage_options)
        destinations = {
            assay_name: load_count_array(root, assay_name, workspace)
            for assay_name in assay_names
        }
        buffers = {
            assay_name: DenseBandBuffer(destination, startRow=row_start, endRow=row_end)
            for assay_name, destination in destinations.items()
        }
        summaries = {
            assay_name: CountSummary(destination, rows=(row_start, row_end))
            for assay_name, destination in destinations.items()
        }
        writes = (
            DenseWriteBand(destinations[assay_name], band)
            for assay_name, band in _dense_count_bands(
                reader, batch_rows, row_start, row_end, buffers, columns, linear, stop
            )
        )
        with execution_report_scope() as reports:
            write_dense_bands(
                writes,
                plan=plan,
                countSummaries={
                    destinations[name].path: summary
                    for name, summary in summaries.items()
                },
            )
        if not stop.is_set():
            windows = {name: summary.window() for name, summary in summaries.items()}
            connection.send(("done", (reports, windows)))

    _run_worker(reader_kwargs, connection, work)


def _producer_messages(
    target: Callable[..., None],
    windows: Sequence[tuple[int, int]],
    arguments: Callable[[int, int], tuple[Any, ...]],
    linear: Mapping[str, MatrixValuesKind],
    *,
    role: str,
    duplex: bool,
) -> Generator[tuple[int, str, Any], None, None]:
    """Run ``target`` in a spawned process for each row window, yielding messages.

    A window's process takes the leading arguments that ``arguments`` gives
    for the window's rows, then its end of a pipe, the shared stop event, and
    ``linear``. The messages end once every process is done, and closing the
    iterator stops the processes.
    """
    from multiprocessing import get_context

    context = get_context("spawn")
    stop = context.Event()
    connections: dict[Any, int] = {}
    workers: list[Any] = []
    try:
        for index, (row_start, row_end) in enumerate(windows):
            parent_connection, child_connection = context.Pipe(duplex=duplex)
            worker = context.Process(
                target=target,
                args=(*arguments(row_start, row_end), child_connection, stop, linear),
                name=f"h5ad-{role}-{index}",
                daemon=True,
            )
            worker.start()
            child_connection.close()
            connections[parent_connection] = index
            workers.append(worker)
        yield from _worker_messages(connections, workers, role)
    finally:
        _stop_workers(stop, workers, list(connections))


def _merge_window_writes(
    messages: Generator[tuple[int, str, Any], None, None],
    summaries: dict[str, Any],
) -> None:
    """Record the reports and merge the count summaries that writer processes send.

    A writer sends only its done message; its errors raise here.
    """
    from contextlib import closing

    from ..storage.execution import record_execution_report

    with closing(messages):
        for _index, _kind, payload in messages:
            reports, summary_windows = payload
            for report in reports:
                record_execution_report(report)
            for assay_name, window in summary_windows.items():
                summaries[assay_name].merge(window)


def _validate_assay_names(names: tuple[str, ...]) -> None:
    from ..storage.schema import validate_assay_name

    for name in names:
        validate_assay_name(name)


@dataclass(frozen=True, slots=True)
class H5adImportResult:
    """Immutable outputs created while importing one H5AD file.

    Attributes:
        assayNames: Assay groups written to the destination store.
        analysisAssay: Assay owning explicitly selected analytical outputs.
        cellSelection: Artifact covering the imported cell axis.
        embeddingArtifacts: Artifacts of the selected ``obsm`` arrays keyed by
            exact ``obsm`` key: ``embedding`` artifacts for the ``"umap"`` and
            ``"tsne"`` roles, and ``imported_coordinates`` artifacts, which
            ``graph.ann_index`` accepts, for the ``"coordinates"`` role.
        clusterArtifacts: Cluster artifacts keyed by exact ``obs`` key.
    """

    assayNames: tuple[str, ...]
    analysisAssay: str | None
    cellSelection: ArtifactRef
    embeddingArtifacts: Mapping[str, ArtifactRef]
    clusterArtifacts: Mapping[str, ArtifactRef]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "embeddingArtifacts",
            MappingProxyType(dict(self.embeddingArtifacts)),
        )
        object.__setattr__(
            self,
            "clusterArtifacts",
            MappingProxyType(dict(self.clusterArtifacts)),
        )


class _H5adCellIdSource(Sequence[str | bytes]):
    def __init__(self, reader: H5adReader) -> None:
        self._reader = reader
        self.shape = (reader.nCells,)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("H5AD cell IDs support only contiguous reads")
            return self._reader._cell_ids_block(start, stop)
        position = range(len(self))[index]
        return self._reader._cell_ids_block(position, position + 1)[0]


class H5adToZarr:
    """A class for converting data in anndata's H5ad format to Zarr hierarchy.

    Args:
        h5ad: Reader for the source H5AD file.
        zarr_loc: The file name for the Zarr hierarchy or a store
        assay_name: the name of the assay (e. g. 'RNA')
        assay_split_key: A var column used to split features into assays.
        assay_name_map: Feature type to assay name overrides.
        workspace: An optional workspace id. None uses the legacy layout
                   without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix layout: a ``CountMatrixPolicy``, used exactly;
                None, the default layout; or ``"fit"``, the largest halving
                of the default ``unitBytes`` and ``chunkBytes`` whose counts
                write and ``countsT`` fit ``mem_budget``. A policy or the
                default that does not fit raises ``CountLayoutMemoryError``,
                a ``MemoryError``, before the destination is created, with
                the largest halving that fits as its ``fittingPolicy``.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
        analysis_assay: Imported assay that owns explicitly selected H5AD
                        embeddings and clusters. Required for multi-assay
                        imports with analytical outputs.
        assay_type: Preset assay type, such as ``RNA``, for a single imported
                    assay whose name is not a preset. When None, the assay
                    name decides the type. Not allowed with ``assay_split_key``.
        overwrite: If True, replace a CyteArc store that no ``DataStore`` has opened.
        values: What the matrix holds, ``"counts"``, ``"normalized"``,
            ``"log_normalized"`` or ``"log2_normalized"``, as the import
            reference's normalized values section describes. None decides
            from the values.
        categorical_columns: If True, store each categorical column of
            ``obs`` and ``var`` as a categorical column, its codes beside
            every one of its categories, instead of a column of its labels.
            Every read returns the same labels either way. Off by default
            in this release, because releases before categorical columns
            cannot read such a store.

    Each assay stores its counts in the dtype that
    :func:`~cytearc.storage.count_dtype.count_storage_dtype` resolves from the
    canonical source values of its own features, so the same counts import
    identically from every H5AD encoding.

    Raises:
        FileExistsError: If ``zarr_loc`` is not empty and may not be replaced.
        TypeError: If ``categorical_columns`` is not a boolean.

    Attributes:
        h5ad: A h5ad object (h5 file with added AnnData structure).
        assayName: Name of the imported assay; None with ``assay_split_key``.
        z: The Zarr hierarchy (array or group).
        storageDtypes: Count storage dtype of each imported assay.
        policy: The count-matrix layout that every imported assay was written
            with.
    """

    def __init__(
        self,
        h5ad: H5adReader,
        zarr_loc: ZarrLocation,
        assay_name: str | None = None,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | Literal["fit"] | None = None,
        io: StorageIoPolicy | None = None,
        assay_split_key: str | None = None,
        assay_name_map: dict[str, str] | None = None,
        analysis_assay: str | None = None,
        assay_type: str | None = None,
        *,
        overwrite: bool = False,
        values: ImportValues = None,
        categorical_columns: bool = False,
    ) -> None:
        overwrite = bool_argument(overwrite, "overwrite")
        self._categoricalColumns = bool_argument(
            categorical_columns, "categorical_columns"
        )
        from ..utils.matrix_values import import_values_argument

        import_values_argument(values)
        from ..storage.budget import resolve_budget
        from ..storage.schema import create_zarr_count_assay, normalized_values_record
        from ..storage.sharding import count_layout_request
        from ._store import resolve_import_values
        from ..storage.destinations import check_destination, create_destination
        from ..assay.classification import validate_assay_type

        layout_request = count_layout_request(policy)

        validate_assay_type(
            assay_type, assay="RNA" if assay_name is None else assay_name
        )
        if assay_type is not None and assay_split_key is not None:
            raise ValueError(
                "assay_type applies to a single assay and cannot be combined "
                "with assay_split_key"
            )
        self.h5ad = h5ad
        self.workspace = workspace
        self.storage_options = storage_options
        self._parallelWriteLocation = zarr_loc if isinstance(zarr_loc, str) else None
        self.assaySplitKey = assay_split_key
        self.assayNameMap = assay_name_map
        self.assayFeatures: dict[str, _H5adAssayFeatures] | None
        if assay_split_key is not None:
            if assay_name is not None:
                logger.warning(
                    "`assay_name` is ignored when `assay_split_key` is provided"
                )
            self.assayName = None
            self.assayFeatures = self.h5ad.assay_feature_slices(
                assay_split_key,
                assay_name_map,
            )
            self.assayNames = tuple(self.assayFeatures)
        elif assay_name is None:
            logger.debug("Using RNA as the default assay name")
            self.assayName = "RNA"
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        else:
            self.assayName = assay_name
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        _validate_assay_names(self.assayNames)
        has_analysis = bool(self.h5ad.embeddingRoles or self.h5ad.clusterKeys)
        if analysis_assay is None:
            if has_analysis and len(self.assayNames) != 1:
                raise ValueError(
                    "analysis_assay is required when importing H5AD analytical "
                    "outputs into a multi-assay store"
                )
            resolved_analysis_assay = self.assayNames[0] if has_analysis else None
        else:
            if not has_analysis:
                raise ValueError(
                    "analysis_assay requires at least one explicitly selected "
                    "H5AD analytical output"
                )
            if analysis_assay not in self.assayNames:
                raise ValueError("analysis_assay must name an imported assay")
            resolved_analysis_assay = analysis_assay
        self.analysisAssay = resolved_analysis_assay
        self.assayTypes = (
            {} if assay_type is None else {name: assay_type for name in self.assayNames}
        )
        self.resources = resolve_budget(mem_budget, nthreads)
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.io = io
        # A declared membership column that is malformed fails here, before
        # the destination is read or created.
        self._membership = self._declared_membership()
        # A destination that would be refused fails before the passes over
        # the source below; create_destination checks it again.
        check_destination(
            zarr_loc, overwrite=overwrite, storage_options=storage_options
        )
        self._sourceDigest = (
            bytes.fromhex(file_sha256(self.h5ad.h5adFn)) if has_analysis else None
        )
        self._projection = (
            None
            if self.assayFeatures is None
            else self._assay_feature_projection(self.assayFeatures)
        )
        value_ranges = dict(
            zip(
                self.assayNames,
                self.h5ad.count_value_ranges(
                    self.resources.memoryBytes,
                    None if self._projection is None else self._projection[0],
                    processes=self._value_scan_processes(),
                ),
                strict=True,
            )
        )
        self.storageDtypes = {
            name: count_storage_dtype(self.h5ad.sourceMatrixDtype, value_range)
            for name, value_range in value_ranges.items()
        }
        logger.debug(f"Resolved H5AD count storage dtypes={self.storageDtypes}")
        self.h5ad.materialize_csc(self.resources.memoryBytes)
        try:
            self.values = resolve_import_values(
                values,
                value_ranges,
                self._first_cells,
                {name: self.assayTypes.get(name) for name in self.assayNames},
                self.storageDtypes,
            )
        except ValueError as error:
            others = self._other_count_matrices()
            if others:
                raise ValueError(
                    f"{error}. The file also holds counts in {others}; pass "
                    "matrix_key to import them"
                ) from error
            raise
        if any(decision.kind != "counts" for decision in self.values.values()):
            others = self._other_count_matrices()
            if others:
                logger.warning(
                    f"{self.h5ad.matrixKey} holds normalized values, and {others} "
                    "hold counts; pass matrix_key to H5adReader to import the counts"
                )
        # The assays whose log the writer undoes, by log kind.
        self._linear = {
            name: decision.kind
            for name, decision in self.values.items()
            if decision.kind in ("log_normalized", "log2_normalized")
        }
        # A layout that does not fit fails here, before the destination exists.
        self.policy = self._fit_count_layout(layout_request)
        # The destination must be empty, or with overwrite an unprepared store.
        self.z = create_destination(
            zarr_loc, overwrite=overwrite, storage_options=storage_options
        )
        self._ini_cell_data()
        for resolved_assay_name in self.assayNames:
            if self.assayFeatures is None:
                feature_ids = self.h5ad.feat_ids()
                feature_names = self.h5ad.feat_names()
            else:
                features = self.assayFeatures[resolved_assay_name]
                feature_ids = features.featureIds
                feature_names = features.featureNames
            create_zarr_count_assay(
                z=self.z,
                assay_name=resolved_assay_name,
                workspace=workspace,
                n_cells=self.h5ad.nCells,
                feat_ids=feature_ids,
                feat_names=feature_names,
                dtype=self.storageDtypes[resolved_assay_name],
                profile=self.profile,
                policy=self.policy,
                normalized_values=normalized_values_record(
                    self.values[resolved_assay_name]
                ),
            )
        self._ini_feature_data()
        self.root = (
            self.z
            if self.workspace is None
            else as_zarr_group(self.z[self.workspace], name=self.workspace)
        )

    def _other_count_matrices(self) -> list[str]:
        """Return the file's other matrices whose sampled values are counts."""
        return [
            key for key in self.h5ad._count_matrix_keys() if key != self.h5ad.matrixKey
        ]

    def _first_cells(self) -> dict[str, Any]:
        """Return the values of each assay's first cells, which decide what they are."""
        from scipy.sparse import csr_matrix, vstack

        from ..utils.matrix_values import DETECTION_CELLS

        rows = min(DETECTION_CELLS, self.h5ad.nCells)
        blocks = list(self.h5ad.consume_row_range(max(1, rows), 0, rows))
        cells = vstack(blocks).tocoo() if blocks else coo_matrix((0, 0))
        if self._projection is None:
            return {self.assayNames[0]: csr_matrix(cells)}
        codes, columns = self._projection
        first: dict[str, Any] = {}
        for code, name in enumerate(self.assayNames):
            inside = codes[cells.col] == code
            first[name] = csr_matrix(
                (cells.data[inside], (cells.row[inside], columns[cells.col[inside]])),
                shape=(cells.shape[0], int(np.count_nonzero(codes == code))),
            )
        return first

    def _declared_membership(self) -> dict[str, np.ndarray]:
        """Read the membership of each imported assay that the file declares.

        Returns:
            The values of each declared membership column of an imported
            assay, one boolean per cell.

        Raises:
            ValueError: If a declared column is absent from ``obs``, is not
                boolean, or has missing values.
        """
        from ..metadata.membership import checked_membership_values

        declared = self.h5ad.assay_membership()
        restored: dict[str, np.ndarray] = {}
        for assay in self.assayNames:
            column = declared.get(assay)
            if column is None:
                continue
            if not self.h5ad._check_exists(self.h5ad.cellAttrsKey, column):
                raise ValueError(
                    f"The file declares {column!r} as the membership of assay "
                    f"{assay!r}, but {self.h5ad.cellAttrsKey} has no such column"
                )
            values, missing = self.h5ad._read_column(self.h5ad.cellAttrsKey, column)
            restored[assay] = checked_membership_values(values, missing, assay=assay)
        return restored

    def _ini_cell_data(self) -> None:
        from ..metadata.membership import reserved_membership_columns
        from ..storage.metadata_keys import (
            assay_membership_column,
            metadata_column_keys,
        )
        from ..storage.schema import create_cell_data
        from ._store import keyed_metadata_columns, write_membership_column

        ids = self.h5ad.cell_ids()
        g = create_cell_data(
            root=self.z,
            workspace=self.workspace,
            ids=ids,
            names=ids,
            profile=self.profile,
        )
        # The membership column name of every imported assay is reserved. A
        # declared membership is restored below; any other column of that
        # name is skipped with a warning.
        reserved = reserved_membership_columns(self.assayNames)
        restored = {assay_membership_column(assay) for assay in self._membership}
        # Keys are planned over every decodable obs column, including the ID
        # and cluster columns left out below, so a renamed column takes the
        # same key whichever columns an import selects.
        keys = metadata_column_keys(
            self.h5ad._source_column_names(self.h5ad.cellAttrsKey),
            taken=[*g.keys(), *reserved],
        )
        for key, (name, column, categories) in keyed_metadata_columns(
            (
                (name, (name, column, categories))
                for name, column, categories in self.h5ad._cell_columns()
                if name not in restored
            ),
            keys,
            "cell",
            membership=reserved,
        ):
            missing = _write_column(
                g,
                key,
                column,
                profile=self.profile,
                categorical=self._categoricalColumns,
                ordered=categories is not None and categories.ordered,
            )
            if categories is not None:
                self._keep_categories(
                    g,
                    key,
                    categories,
                    colors=self.h5ad._category_colors(name, categories),
                    missing=missing,
                    display=True,
                )
        for assay, members in self._membership.items():
            write_membership_column(g, assay, members, profile=self.profile)

    @staticmethod
    def _keep_categories(
        group: Any,
        key: str,
        categories: Any,
        *,
        colors: tuple[str, ...] | None,
        missing: np.ndarray,
        display: bool,
    ) -> None:
        """Record a source categorical's categories on its stored column.

        A categorical column records them on its codes, whose attributes are
        the column's. The attributes follow the type of the labels, which a
        categorical column's categories hold, so either layout records the
        same ones.
        """
        from ..storage.categorical import CategoricalColumn

        array = as_zarr_array(group[key], name=key)
        labels = CategoricalColumn.open(group, key, values=array)
        array.attrs.update(
            _categorical_attributes(
                np.dtype(array.dtype) if labels is None else labels.dtype,
                categories,
                colors=colors,
                has_missing=bool(np.asarray(missing, dtype=bool).any()),
                display=display,
            )
        )

    def _ini_feature_data(self) -> None:
        from ..storage.metadata_keys import metadata_column_keys
        from ._store import keyed_metadata_columns

        targets: list[tuple[Any, np.ndarray | None]] = []
        for assay_name in self.assayNames:
            if self.workspace is None:
                group_path = f"{assay_name}/featureData"
            else:
                group_path = f"{self.workspace}/{assay_name}/featureData"
            feat_group = as_zarr_group(self.z[group_path], name=group_path)
            feature_indexes = (
                None
                if self.assayFeatures is None
                else self.assayFeatures[assay_name].featureIndexes
            )
            targets.append((feat_group, feature_indexes))

        keys = metadata_column_keys(
            self.h5ad._source_column_names(self.h5ad.featureAttrsKey),
            taken={key for group, _indexes in targets for key in group.keys()},
        )
        # Stream one column at a time so a single decoded var column is held in
        # memory rather than every column for the full feature axis at once.
        for column_name, (column, categories) in keyed_metadata_columns(
            (
                (name, (column, categories))
                for name, column, categories in self.h5ad._feature_columns()
            ),
            keys,
            "feature",
        ):
            for feat_group, feature_indexes in targets:
                # Each assay of a split types its categories from its own
                # codes, as it types the labels of a column of them.
                selected_missing = _write_column(
                    feat_group,
                    column_name,
                    _column_rows(column, feature_indexes),
                    profile=self.profile,
                    categorical=self._categoricalColumns,
                    ordered=categories is not None and categories.ordered,
                )
                if categories is not None:
                    self._keep_categories(
                        feat_group,
                        column_name,
                        categories,
                        colors=None,
                        missing=selected_missing,
                        display=False,
                    )

    def dump(self, batch_size: int | None = None) -> H5adImportResult:
        """Write h5ad matrix data into Zarr ``counts`` and RNA ``countsT``.

        Args:
            batch_size: Number of source cells per batch, at most one
                        destination row band, which is the default for a
                        sparse matrix. A dense matrix is copied into dense
                        row bands, so its batches hold about a million values
                        or whole rows of its HDF5 chunks by default, and this
                        caps them.

        Raises:
            ValueError: If ``batch_size`` is not positive.

        Returns:
            Explicit selection, embedding, and cluster artifact references.
        """
        self._write_counts(batch_size=batch_size)
        from .counts_t import finalize_writer_counts_t_many

        finalize_writer_counts_t_many(
            self.z,
            self.assayNames,
            self.workspace,
            assay_types=self.assayTypes,
            resources=self.resources,
            profile=self.profile,
            io=self.io,
        )
        cell_selection = self._write_cell_selection()
        cluster_artifacts = self._write_cluster_artifacts(
            cell_selection,
            batch_size,
        )
        embedding_artifacts = self._write_embedding_artifacts(
            cell_selection,
            batch_size,
        )
        return H5adImportResult(
            assayNames=self.assayNames,
            analysisAssay=self.analysisAssay,
            cellSelection=cell_selection,
            embeddingArtifacts=embedding_artifacts,
            clusterArtifacts=cluster_artifacts,
        )

    def _analysis_block_rows(
        self,
        requested: int | None,
        *,
        row_bytes: int,
    ) -> int:
        from ._store import bounded_block_rows

        return bounded_block_rows(
            requested,
            row_bytes=row_bytes,
            memory_bytes=int(self.resources.memoryBytes),
        )

    def _write_cell_selection(self) -> ArtifactRef:
        from ._store import resolve_import_cell_selection

        return resolve_import_cell_selection(self.root, source="h5ad", inputs={})

    def _write_embedding_artifacts(
        self,
        cell_selection: ArtifactRef,
        requested_rows: int | None,
    ) -> dict[str, ArtifactRef]:
        from ..embeddings.imported import (
            write_imported_coordinates,
            write_imported_embedding,
        )

        if not self.h5ad.embeddingRoles:
            return {}
        # Construction resolves both whenever the reader selected an output.
        assert self.analysisAssay is not None and self._sourceDigest is not None

        from ._store import fingerprint_row_blocks, floating_payload_dtype

        artifacts: dict[str, ArtifactRef] = {}
        cell_ids = _H5adCellIdSource(self.h5ad)
        for key, role in self.h5ad.embeddingRoles.items():
            source = self.h5ad._obsm_array(key)
            source_shape = (int(source.shape[0]), int(source.shape[1]))
            dtype = floating_payload_dtype(source.dtype, "Embedding payload")
            block_rows = self._analysis_block_rows(
                requested_rows,
                row_bytes=int(source.shape[1]) * int(dtype.itemsize),
            )
            fingerprint = fingerprint_row_blocks(
                self.h5ad._iter_obsm_blocks(key, block_rows, dtype),
                source_shape,
                dtype,
                label=f"Embedding key {key!r}",
            )

            def blocks(
                source_key: str = key,
                rows: int = block_rows,
                resolved_dtype: np.dtype[Any] = dtype,
            ) -> Iterator[np.ndarray]:
                return self.h5ad._iter_obsm_blocks(
                    source_key,
                    rows,
                    resolved_dtype,
                )

            if role == "coordinates":
                # Graph coordinates, such as a scVI latent space, which graph
                # construction, embeddings, and mapping accept.
                artifacts[key] = write_imported_coordinates(
                    self.root,
                    assay=self.analysisAssay,
                    dimreduc_key=key,
                    role=role,
                    coordinates=blocks,
                    coordinate_shape=source_shape,
                    coordinate_dtype=dtype,
                    source_digest=self._sourceDigest,
                    payload_fingerprints={"data": fingerprint},
                    source_cell_ids=cell_ids,
                    cell_selection=cell_selection,
                    block_rows=block_rows,
                )
                continue
            artifacts[key] = write_imported_embedding(
                self.root,
                assay=self.analysisAssay,
                dimreduc_key=key,
                role=role,
                coordinates=blocks,
                coordinate_shape=source_shape,
                coordinate_dtype=dtype,
                source_digest=self._sourceDigest,
                payload_fingerprints={"values": fingerprint},
                source_cell_ids=cell_ids,
                cell_selection=cell_selection,
                block_rows=block_rows,
            )
        return artifacts

    @staticmethod
    def _value_is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float | np.floating):
            return not bool(np.isfinite(value))
        return False

    @staticmethod
    def _decode_cluster_text(value: Any) -> str:
        from ._store import decode_text

        return decode_text(value)

    def _cluster_dtype_and_missing(
        self,
        key: str,
        block_rows: int,
    ) -> tuple[np.dtype[Any], bool]:
        # The reader admits only boolean, numeric, and text cluster keys.
        source_dtype = self.h5ad._cell_column_value_dtype(key)
        has_missing = False
        if source_dtype.kind not in "OSU":
            for start in range(0, self.h5ad.nCells, block_rows):
                stop = min(start + block_rows, self.h5ad.nCells)
                values, missing = self.h5ad._cell_column_block(key, start, stop)
                has_missing = has_missing or bool(np.asarray(missing, dtype=bool).any())
                if np.asarray(values).dtype.kind == "O":
                    has_missing = has_missing or any(
                        self._value_is_missing(value) for value in values
                    )
            return np.dtype(source_dtype.str), has_missing

        maximum = 1
        for start in range(0, self.h5ad.nCells, block_rows):
            stop = min(start + block_rows, self.h5ad.nCells)
            values, raw_missing = self.h5ad._cell_column_block(key, start, stop)
            missing = np.asarray(raw_missing, dtype=bool).copy()
            for index, value in enumerate(values):
                if missing[index] or self._value_is_missing(value):
                    missing[index] = True
                    continue
                maximum = max(maximum, len(self._decode_cluster_text(value)))
            has_missing = has_missing or bool(missing.any())
        return np.dtype(f"U{maximum}"), has_missing

    def _cluster_block(
        self,
        key: str,
        start: int,
        stop: int,
        dtype: np.dtype[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_values, raw_missing = self.h5ad._cell_column_block(key, start, stop)
        values = np.asarray(raw_values)
        missing = np.asarray(raw_missing, dtype=bool).copy()
        if values.dtype.kind == "O":
            for index, value in enumerate(values):
                if self._value_is_missing(value):
                    missing[index] = True

        if dtype.kind == "U":
            normalized = np.asarray(
                [
                    "" if missing[index] else self._decode_cluster_text(value)
                    for index, value in enumerate(values)
                ],
                dtype=dtype,
            )
            return normalized, missing

        normalized = np.zeros(values.shape, dtype=dtype)
        valid = ~missing
        normalized[valid] = np.asarray(values[valid], dtype=dtype)
        if dtype.kind == "f":
            nonfinite = ~np.isfinite(normalized)
            missing |= nonfinite
            normalized[missing] = 0
        return normalized, missing

    def _write_cluster_artifacts(
        self,
        cell_selection: ArtifactRef,
        requested_rows: int | None,
    ) -> dict[str, ArtifactRef]:
        from ..storage.arrays import (
            MISSING_MASK_PREFIX,
            MetadataBlock,
            create_streamed_metadata_column,
        )
        from ..storage.artifact_writer import (
            ArrayRequirement,
            artifact_transaction,
            plan_artifact,
        )
        from ._store import DEFAULT_IMPORT_BLOCK_ROWS

        if not self.h5ad.clusterKeys:
            return {}
        # Construction resolves both whenever the reader selected an output.
        assert self.analysisAssay is not None and self._sourceDigest is not None

        artifacts: dict[str, ArtifactRef] = {}
        block_rows = self._analysis_block_rows(requested_rows, row_bytes=64)
        for key in self.h5ad.clusterKeys:
            dtype, has_missing = self._cluster_dtype_and_missing(key, block_rows)
            requirements = [
                ArrayRequirement(
                    "values",
                    shape=(self.h5ad.nCells,),
                    dtype=dtype,
                )
            ]
            if has_missing:
                requirements.append(
                    ArrayRequirement(
                        f"{MISSING_MASK_PREFIX}values",
                        shape=(self.h5ad.nCells,),
                        dtype=bool,
                    )
                )
            planned = plan_artifact(
                self.root,
                scope="assay",
                assay=self.analysisAssay,
                kind="cluster_labels",
                operation="import_cluster_labels",
                parameters={"source": "h5ad", "source_key": key},
                inputs={
                    "source_digest": self._sourceDigest,
                    "cell_selection": cell_selection,
                },
                execution_options={"block_rows": block_rows},
                required_arrays=tuple(requirements),
            )
            if planned.reused:
                artifacts[key] = planned.ref
                continue
            with artifact_transaction(self.root, planned) as group:

                def blocks() -> Iterator[MetadataBlock]:
                    for start in range(0, self.h5ad.nCells, block_rows):
                        stop = min(start + block_rows, self.h5ad.nCells)
                        values, missing = self._cluster_block(
                            key,
                            start,
                            stop,
                            dtype,
                        )
                        yield MetadataBlock(
                            start,
                            values,
                            missing if has_missing else None,
                        )

                values = create_streamed_metadata_column(
                    group,
                    "values",
                    shape=self.h5ad.nCells,
                    dtype=dtype,
                    blocks=blocks(),
                    chunkSize=min(
                        DEFAULT_IMPORT_BLOCK_ROWS,
                        max(1, self.h5ad.nCells),
                    ),
                    hasMissing=has_missing,
                    profile=self.profile,
                )
                categories = self.h5ad._cell_column_categories(key)
                if categories is not None:
                    # The labels keep their source categories, order, and
                    # colors, which exports and plots read.
                    values.attrs.update(
                        _categorical_attributes(
                            dtype,
                            categories,
                            colors=self.h5ad._category_colors(key, categories),
                            has_missing=has_missing,
                            display=True,
                        )
                    )
            artifacts[key] = planned.ref
        return artifacts

    def _assay_widths(self) -> dict[str, int]:
        """Return the feature count of each imported assay."""
        if self.assayFeatures is None:
            return {name: int(self.h5ad.nFeatures) for name in self.assayNames}
        return {
            name: int(self.assayFeatures[name].featureIndexes.size)
            for name in self.assayNames
        }

    def _count_resident_bytes(self) -> int:
        """Return the bytes that the counts write holds from start to end.

        They are the reader's row counts or converted rows, the count
        summaries of every cell, and the features of each assay.
        """
        from ..storage.identity import CountSummary

        resident = (
            self.h5ad.materialized_csr_bytes()
            + max(0, int(self.h5ad._sparse_import_resident_bytes()))
            + sum(
                CountSummary.nbytes_for(self.h5ad.nCells, width)
                for width in self._assay_widths().values()
            )
        )
        if self.assayFeatures is not None:
            resident += sum(
                assay.featureIndexes.nbytes for assay in self.assayFeatures.values()
            )
        if self._projection is not None:
            resident += sum(array.nbytes for array in self._projection)
        return resident

    def _count_import_requirements(
        self,
    ) -> tuple[int, Callable[[int], int], bool]:
        """Return the resident and extra producer bytes of a sparse counts write.

        The third value says whether every source batch arrives marked
        canonical, which a split into assays and an undone log keep, so the
        write sorts and sums none. The layout fit and the write plan the
        same import from these.
        """
        self.h5ad._prepare_sparse_import()
        resident = self._count_resident_bytes()
        source_itemsize = self.h5ad.consumeDtype.itemsize
        projection_value_bytes = max(
            [
                source_itemsize,
                *(dtype.itemsize for dtype in self.storageDtypes.values()),
            ]
        )

        def extra_producer_bytes(rows: int) -> int:
            if self.assayFeatures is None:
                return 0
            source_values = max(0, int(self.h5ad.max_batch_nnz(rows)))
            return source_values * (
                projection_value_bytes
                + 4 * np.dtype(np.int64).itemsize
                + np.dtype(np.bool_).itemsize
            )

        return resident, extra_producer_bytes, self.h5ad._canonical_batches()

    def _dense_batch_rows(self, bandRows: int, requested: int | None = None) -> int:
        """Return the rows of each batch that a dense counts write reads.

        A batch holds about a million values, as the value scan reads them,
        or one row of the matrix's HDF5 chunks when that holds more. It holds
        no more rows than ``requested``, nor than leave its producer holding
        more bytes than ``bandRows`` rows of every assay hold in storage, so a
        batch never outweighs the bands that it fills. Within these limits a
        batch of a chunked matrix holds whole rows of chunks, which HDF5 then
        decompresses once each.
        """
        band_bytes = int(bandRows) * sum(
            width * np.dtype(self.storageDtypes[assay_name]).itemsize
            for assay_name, width in self._assay_widths().items()
        )
        value_rows = _DENSE_BATCH_VALUES // max(1, int(self.h5ad.nFeatures))
        chunk_rows = self.h5ad._dense_chunk_rows()
        if chunk_rows is not None:
            value_rows = max(value_rows, chunk_rows)
        rows = max(
            1,
            min(
                int(bandRows),
                value_rows,
                band_bytes // max(1, self._dense_producer_bytes(1)),
            ),
        )
        if chunk_rows is not None and rows >= chunk_rows:
            rows -= rows % chunk_rows
        return rows if requested is None else max(1, min(rows, int(requested)))

    def _dense_producer_bytes(self, rows: int) -> int:
        """Return the bytes that a dense counts write holds for ``rows`` source rows.

        The producer holds the batch in the source dtype and transforms the
        columns of one assay at a time: a copy of them when the matrix splits
        into assays, beside either the float64 products and exponentials of
        undoing a log, or the truncated values and comparison that check a
        cast to an integer storage dtype, or that cast's result.
        """
        source_itemsize = int(np.dtype(self.h5ad.sourceMatrixDtype).itemsize)
        widths = self._assay_widths()

        def transform_bytes(assay_name: str) -> int:
            per_value = source_itemsize if self.assayFeatures is not None else 0
            storage = np.dtype(self.storageDtypes[assay_name])
            if assay_name in self._linear:
                per_value += 2 * np.dtype(np.float64).itemsize
            elif storage.kind in "biu":
                per_value += max(
                    source_itemsize + np.dtype(np.bool_).itemsize, storage.itemsize
                )
            return widths[assay_name] * per_value

        return int(rows) * (
            int(self.h5ad.nFeatures) * source_itemsize
            + max(transform_bytes(assay_name) for assay_name in self.assayNames)
        )

    def _dense_columns(self) -> dict[str, np.ndarray] | None:
        """Return the source columns of each assay, or None for one assay of all."""
        if self.assayFeatures is None:
            return None
        return {
            assay_name: self.assayFeatures[assay_name].featureIndexes
            for assay_name in self.assayNames
        }

    def _fit_count_layout(
        self, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the ``requested`` count layout if its writes fit the budget.

        None fits the default layout to the budget. A dense matrix is admitted as its write runs, through dense row bands
        of every assay, and a sparse one through the sparse band writer.
        """
        from ..storage.sharding import (
            dense_counts_admission,
            fit_count_layout,
            sparse_counts_admission,
        )
        from .counts_t import counts_t_assays

        if self.h5ad.matrixOrientation == "dense":
            admit_counts = dense_counts_admission(
                self._count_resident_bytes(),
                lambda band_rows: self._dense_producer_bytes(
                    self._dense_batch_rows(band_rows)
                ),
            )
        else:
            resident, extra_producer_bytes, canonical_batches = (
                self._count_import_requirements()
            )
            admit_counts = sparse_counts_admission(
                nRows=self.h5ad.nCells,
                maxWindowNnz=self.h5ad.max_batch_nnz,
                sourceDtype=self.h5ad.consumeDtype,
                residentBytes=resident,
                producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
                extraProducerBytes=extra_producer_bytes,
                canonicalBatches=canonical_batches,
                alignedBatches=True,
            )
        widths = self._assay_widths()
        return fit_count_layout(
            {
                name: (widths[name], self.storageDtypes[name])
                for name in self.assayNames
            },
            nCells=self.h5ad.nCells,
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays(self.assayNames, self.assayTypes),
            admitCounts=admit_counts,
            requested=requested,
        )

    def _write_counts(self, batch_size: int | None = None) -> None:
        """Write and finalize cell-major ``counts`` (profiling stage split helper)."""
        from ..storage.identity import CountSummary, finalize_counts
        from ..storage.schema import load_count_array

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        destinations = {
            assay_name: load_count_array(self.z, assay_name, self.workspace)
            for assay_name in self.assayNames
        }
        summaries = {
            assay_name: CountSummary(destination)
            for assay_name, destination in destinations.items()
        }
        logger.debug(
            f"Writing counts with up to {self.resources.workers} row-band writer(s)"
        )
        started = time.perf_counter()
        if self.h5ad.matrixOrientation == "dense":
            self._write_dense_counts(batch_size, destinations, summaries)
        else:
            self._write_sparse_counts(batch_size, destinations, summaries)
        counts_seconds = time.perf_counter() - started
        # counts is the durable physical orientation for H5AD imports. Public
        # dump() always finalizes paired RNA countsT after this step.
        logger.debug(f"Counts written in {counts_seconds:.1f}s")
        logger.info(
            f"Wrote {self.h5ad.nCells} cells and {self.h5ad.nFeatures} features "
            f"from H5AD to {len(destinations)} assay(s)"
        )
        for assay_name, summary in summaries.items():
            finalize_counts(destinations[assay_name], summary=summary)

    def _value_scan_processes(self) -> int:
        """Return how many processes may scan the values for their range at once.

        Nothing is written while the values are scanned, so every worker may
        read, at most ``io.readWorkers`` of them. The scan starts fewer when
        the budget or the size of the matrix calls for fewer.
        """
        limit = max(1, int(self.resources.workers))
        if self.io is not None and self.io.readWorkers is not None:
            limit = min(limit, max(1, int(self.io.readWorkers)))
        return limit

    def _producer_window_limit(self, shard_rows: int) -> int:
        """Return how many producers may read disjoint row windows.

        Half the workers read, at most ``io.readWorkers`` of them, and each
        window holds at least one band of ``shard_rows`` rows.
        """
        limit = max(1, (int(self.resources.workers) + 1) // 2)
        if self.io is not None and self.io.readWorkers is not None:
            limit = min(limit, max(1, int(self.io.readWorkers)))
        if self.h5ad.matrixOrientation == "csc":
            limit = 1
        return max(1, min(limit, (self.h5ad.nCells + shard_rows - 1) // shard_rows))

    def _write_dense_counts(
        self,
        batch_size: int | None,
        destinations: dict[str, Any],
        summaries: dict[str, Any],
    ) -> None:
        """Write a dense matrix's counts through dense row bands of every assay.

        Each batch of about a million values is copied into a band of every
        assay, so the write holds bands, never a sparse copy of the values.
        Producer processes read and write disjoint row windows when the
        budget holds their writes side by side: each process writes its own
        window when the destination is a path, and otherwise sends its bands
        to this process to write.
        """
        from ..storage.identity import CountSummary
        from ..storage.layout import array_shard_rows
        from ..storage.sharding import (
            aligned_row_windows,
            filling_band_bytes,
            plan_dense_band_writes,
            row_band_task_count,
            write_dense_bands,
        )

        arrays = tuple(destinations.values())
        n_cells = self.h5ad.nCells
        band_rows = [array_shard_rows(array) for array in arrays]
        band_bytes = [
            rows * int(array.shape[1]) * np.dtype(array.dtype).itemsize
            for rows, array in zip(band_rows, arrays, strict=True)
        ]
        batch_rows = self._dense_batch_rows(min(band_rows), batch_size)
        producer_bytes = self._dense_producer_bytes(batch_rows)
        resident = self._count_resident_bytes()
        filling = filling_band_bytes(arrays)

        def window_tasks(start: int, stop: int) -> int:
            return sum(row_band_task_count(stop - start, rows) for rows in band_rows)

        tasks = window_tasks(0, n_cells)
        # The layout fit admitted this plan with one worker.
        plan = plan_dense_band_writes(
            arrays,
            self.resources,
            tasks,
            self.io,
            residentBytes=resident + producer_bytes + filling,
        )
        shard_rows = math.lcm(*band_rows)
        direct = isinstance(self._parallelWriteLocation, str)
        workers = int(self.resources.workers)
        n_producers = 1
        write_workers = workers
        workers_per_process: int | None = None
        for candidate in range(self._producer_window_limit(shard_rows), 1, -1):
            candidate_windows = aligned_row_windows(n_cells, shard_rows, candidate)
            if direct:
                # Each writer process holds its batch, the bands it fills, and
                # the count summaries of its window.
                window_rows = max(stop - start for start, stop in candidate_windows)
                candidate_resources = ResourceBudget(
                    (self.resources.memoryBytes - resident) // candidate,
                    max(1, workers // candidate),
                )
                candidate_tasks = max(
                    window_tasks(start, stop) for start, stop in candidate_windows
                )
                candidate_resident = (
                    resident
                    + sum(
                        CountSummary.nbytes_for(window_rows, array.shape[1])
                        for array in arrays
                    )
                    + producer_bytes
                    + filling
                )
            else:
                # Each reader process holds its batch, a band of every assay,
                # and the pickled copy of the band it sends; this process
                # unpickles one band beside its writes.
                held = producer_bytes + sum(band_bytes) + max(band_bytes)
                candidate_resources = ResourceBudget(
                    self.resources.memoryBytes, max(1, workers - candidate)
                )
                candidate_tasks = tasks
                candidate_resident = resident + candidate * held + max(band_bytes)
            try:
                candidate_plan = plan_dense_band_writes(
                    arrays,
                    candidate_resources,
                    candidate_tasks,
                    self.io,
                    residentBytes=candidate_resident,
                )
            except MemoryBudgetError:
                continue
            n_producers = len(candidate_windows)
            plan = candidate_plan
            if direct:
                workers_per_process = candidate_resources.workers
                write_workers = n_producers * workers_per_process
            else:
                write_workers = candidate_resources.workers
            break
        windows = aligned_row_windows(n_cells, shard_rows, n_producers)
        direct_process_writes = direct and n_producers > 1
        self._lastDenseBatchRows = batch_rows
        self._lastImportProducerCount = n_producers
        self._lastImportWorkersPerProcess = workers_per_process
        self._lastImportWriteWorkers = write_workers
        logger.info(
            f"Resolved H5AD dense source batch rows={batch_rows} "
            f"write_tasks={tasks} producers={n_producers} writers={write_workers} "
            f"workers_per_process={workers_per_process}"
        )
        if direct_process_writes:
            _merge_window_writes(
                _producer_messages(
                    _write_dense_h5ad_process_window,
                    windows,
                    lambda row_start, row_end: (
                        self.h5ad._clone_kwargs(),
                        batch_rows,
                        row_start,
                        row_end,
                        self._parallelWriteLocation,
                        self.storage_options,
                        self.workspace,
                        self.assayNames,
                        self._dense_columns(),
                        plan,
                    ),
                    self._linear,
                    role="writer",
                    duplex=False,
                ),
                summaries,
            )
            return
        write_dense_bands(
            self._dense_count_writes(batch_rows, destinations, windows),
            plan=plan,
            countSummaries={
                destinations[name].path: summary for name, summary in summaries.items()
            },
            msg="Writing counts",
            total=tasks,
        )

    def _dense_count_writes(
        self,
        batch_rows: int,
        destinations: dict[str, Any],
        windows: list[tuple[int, int]],
    ) -> Iterator[Any]:
        """Yield the dense band writes of every assay, from this process or readers.

        One window is read here; several are read by producer processes,
        which send their bands.
        """
        from contextlib import closing

        from ..storage.sharding import DenseBandBuffer, DenseWriteBand

        if len(windows) <= 1:
            buffers = {
                assay_name: DenseBandBuffer(destination)
                for assay_name, destination in destinations.items()
            }
            for assay_name, band in _dense_count_bands(
                self.h5ad,
                batch_rows,
                0,
                self.h5ad.nCells,
                buffers,
                self._dense_columns(),
                self._linear,
            ):
                yield DenseWriteBand(destinations[assay_name], band)
            return
        destination_specs = self._destination_specs(destinations)
        with closing(
            _producer_messages(
                _read_dense_h5ad_process_window,
                windows,
                lambda row_start, row_end: (
                    self.h5ad._clone_kwargs(),
                    batch_rows,
                    row_start,
                    row_end,
                    destination_specs,
                    self._dense_columns(),
                ),
                self._linear,
                role="producer",
                duplex=True,
            )
        ) as messages:
            for _index, kind, payload in messages:
                if kind == "done":
                    continue
                assay_name, band = payload
                yield DenseWriteBand(destinations[assay_name], band)

    def _write_sparse_counts(
        self,
        batch_size: int | None,
        destinations: dict[str, Any],
        summaries: dict[str, Any],
    ) -> None:
        """Write a sparse matrix's counts through sparse bands of every assay."""
        from ..storage.identity import CountSummary
        from ..storage.layout import array_shard_rows
        from ..storage.sharding import (
            SparseShardBuffer,
            aligned_row_windows,
            resolve_sparse_import_batch,
            write_sparse_bands,
        )

        buffers = {
            assay_name: SparseShardBuffer(destination)
            for assay_name, destination in destinations.items()
        }
        projection = self._projection
        resident_source_bytes, extra_producer_bytes, canonical_batches = (
            self._count_import_requirements()
        )

        plan = resolve_sparse_import_batch(
            tuple(destinations.values()),
            nRows=self.h5ad.nCells,
            resources=self.resources,
            maxWindowNnz=self.h5ad.max_batch_nnz,
            sourceDtype=self.h5ad.consumeDtype,
            batchRows=batch_size,
            residentBytes=resident_source_bytes,
            producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
            extraProducerBytes=extra_producer_bytes,
            canonicalBatches=canonical_batches,
            alignedBatches=True,
        )
        self._lastImportPlan = plan
        resolved_batch_rows = plan.batchRows
        shard_rows = math.lcm(
            *(array_shard_rows(destination) for destination in destinations.values())
        )
        available_windows = self._producer_window_limit(shard_rows)
        n_producers = 1
        process_resources: ResourceBudget | None = None
        process_plan = None
        # One producer writes with the plan above, so only wider splits are
        # planned. A split whose budget cannot hold it fails admission.
        for candidate in range(available_windows, 1, -1):
            direct_candidate = isinstance(self._parallelWriteLocation, str)
            if direct_candidate:
                available_memory = self.resources.memoryBytes - resident_source_bytes
                candidate_resources = ResourceBudget(
                    available_memory // candidate,
                    max(1, int(self.resources.workers) // candidate),
                )
                # Each writer process also holds the count summaries of its window.
                window_rows = max(
                    stop - start
                    for start, stop in aligned_row_windows(
                        self.h5ad.nCells, shard_rows, candidate
                    )
                )
                candidate_resident = resident_source_bytes + sum(
                    CountSummary.nbytes_for(window_rows, destination.shape[1])
                    for destination in destinations.values()
                )
                candidate_batch_rows = None
            else:
                candidate_resources = self.resources
                candidate_resident = resident_source_bytes + _producer_process_bytes(
                    candidate, plan.producerReserveBytes
                )
                candidate_batch_rows = resolved_batch_rows
            try:
                candidate_plan = resolve_sparse_import_batch(
                    tuple(destinations.values()),
                    nRows=self.h5ad.nCells,
                    resources=candidate_resources,
                    maxWindowNnz=self.h5ad.max_batch_nnz,
                    sourceDtype=self.h5ad.consumeDtype,
                    batchRows=candidate_batch_rows,
                    residentBytes=candidate_resident,
                    producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
                    extraProducerBytes=extra_producer_bytes,
                    canonicalBatches=canonical_batches,
                    alignedBatches=True,
                )
                if (
                    direct_candidate
                    and batch_size is not None
                    and candidate_plan.batchRows > resolved_batch_rows
                ):
                    candidate_plan = resolve_sparse_import_batch(
                        tuple(destinations.values()),
                        nRows=self.h5ad.nCells,
                        resources=candidate_resources,
                        maxWindowNnz=self.h5ad.max_batch_nnz,
                        sourceDtype=self.h5ad.consumeDtype,
                        batchRows=resolved_batch_rows,
                        residentBytes=candidate_resident,
                        producerStagingBytes=self.h5ad.producer_batch_staging_bytes,
                        extraProducerBytes=extra_producer_bytes,
                        canonicalBatches=canonical_batches,
                        alignedBatches=True,
                    )
            except MemoryBudgetError:
                continue
            n_producers = candidate
            process_resources = candidate_resources if direct_candidate else None
            process_plan = candidate_plan if direct_candidate else None
            break
        windows = aligned_row_windows(
            self.h5ad.nCells,
            shard_rows,
            n_producers,
        )
        n_producers = max(1, len(windows))
        self._lastImportProducerCount = n_producers
        direct_process_writes = (
            n_producers > 1
            and process_resources is not None
            and isinstance(self._parallelWriteLocation, str)
        )
        if direct_process_writes:
            assert process_resources is not None
            assert process_plan is not None
            self._lastImportPlan = process_plan
            resolved_batch_rows = process_plan.batchRows
            workers_per_process = process_resources.workers
            write_workers = n_producers * workers_per_process
        else:
            workers_per_process = None
            write_workers = (
                int(self.resources.workers)
                if n_producers == 1
                else max(1, int(self.resources.workers) - n_producers)
            )
        self._lastImportWorkersPerProcess = workers_per_process
        self._lastImportWriteWorkers = write_workers
        write_resources = ResourceBudget(self.resources.memoryBytes, write_workers)
        logger.info(
            f"Resolved H5AD source batch rows={resolved_batch_rows} "
            f"write_tasks={plan.writeTasks} producers={n_producers} "
            f"writers={write_workers} workers_per_process={workers_per_process}"
        )
        if direct_process_writes:
            assert process_resources is not None
            self._write_parallel_count_windows(
                resolved_batch_rows,
                projection,
                windows,
                resources=process_resources,
                residentBytes=resident_source_bytes,
                producerReserveBytes=self._lastImportPlan.producerReserveBytes,
                summaries=summaries,
            )
        else:
            extra_producer_resident = _producer_process_bytes(
                n_producers, plan.producerReserveBytes
            )
            write_sparse_bands(
                self._count_shard_tasks(
                    resolved_batch_rows,
                    buffers,
                    destinations,
                    projection,
                    windows=windows,
                ),
                resources=write_resources,
                residentBytes=resident_source_bytes + extra_producer_resident,
                producerReserveBytes=plan.producerReserveBytes,
                total=plan.writeTasks,
                io=self.io,
                countSummaries={
                    destinations[name].path: summary
                    for name, summary in summaries.items()
                },
            )

    def _write_parallel_count_windows(
        self,
        batch_size: int,
        projection: tuple[np.ndarray, np.ndarray] | None,
        windows: list[tuple[int, int]],
        *,
        resources: ResourceBudget,
        residentBytes: int,
        producerReserveBytes: int,
        summaries: dict[str, Any],
    ) -> None:
        _merge_window_writes(
            _producer_messages(
                _write_h5ad_process_window,
                windows,
                lambda row_start, row_end: (
                    self.h5ad._clone_kwargs(),
                    batch_size,
                    row_start,
                    row_end,
                    self._parallelWriteLocation,
                    self.storage_options,
                    self.workspace,
                    self.assayNames,
                    projection,
                    resources,
                    residentBytes,
                    producerReserveBytes,
                    self.io,
                ),
                self._linear,
                role="writer",
                duplex=False,
            ),
            summaries,
        )

    def _count_shard_tasks(
        self,
        batch_size: int,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
        windows: list[tuple[int, int]],
    ) -> Iterator[Any]:
        """Yield complete row-band writes from disjoint H5AD producers."""
        ranges = windows or [(0, self.h5ad.nCells)]
        if len(ranges) <= 1:
            yield from self._emit_count_bands(
                self.h5ad,
                batch_size,
                buffers,
                destinations,
                projection,
                row_start=ranges[0][0],
                row_end=ranges[0][1],
            )
            return
        yield from self._parallel_count_bands(
            batch_size,
            destinations,
            projection,
            ranges,
        )

    def _parallel_count_bands(
        self,
        batch_size: int,
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
        windows: list[tuple[int, int]],
    ) -> Iterator[Any]:
        from contextlib import closing

        from ..storage.sharding import SparseWriteBand

        destination_specs = self._destination_specs(destinations)
        # A producer sends bands and then done; its errors raise here.
        with closing(
            _producer_messages(
                _read_h5ad_process_window,
                windows,
                lambda row_start, row_end: (
                    self.h5ad._clone_kwargs(),
                    batch_size,
                    row_start,
                    row_end,
                    destination_specs,
                    self.assayNames,
                    projection,
                ),
                self._linear,
                role="producer",
                duplex=True,
            )
        ) as messages:
            for _index, kind, payload in messages:
                if kind == "done":
                    continue
                assay_name, band, producer_bytes = payload
                yield SparseWriteBand(
                    destinations[assay_name],
                    band,
                    producer_bytes,
                )

    @staticmethod
    def _destination_specs(destinations: dict[str, Any]) -> dict[str, Any]:
        """Return the geometry of each count array, which producer processes take."""
        from ..storage.geometry import array_geometry
        from ..storage.layout import ZarrArraySpec

        specs: dict[str, Any] = {}
        for assay_name, destination in destinations.items():
            geometry = array_geometry(destination)
            # Count arrays are always chunked.
            assert geometry is not None
            specs[assay_name] = ZarrArraySpec(
                shape=geometry.shape,
                chunks=geometry.chunks,
                shards=geometry.shards,
                dtype=destination.dtype,
                compressors=(),
            )
        return specs

    def _emit_count_bands(
        self,
        reader: Any,
        batch_size: int,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
        *,
        row_start: int,
        row_end: int,
    ) -> Iterator[Any]:
        """Yield complete row-band writes from one reader over a row window."""
        n_rows = max(0, row_end - row_start)
        n_batches = (n_rows + batch_size - 1) // batch_size if n_rows else 0
        stream = iter_progress(
            reader.consume_row_range(batch_size, row_start, row_end),
            total=n_batches,
            desc="Writing counts",
        )
        for matrix in stream:
            yield from self._emit_count_matrix_bands(
                matrix,
                buffers,
                destinations,
                projection,
            )
        yield from self._finish_count_buffers(buffers, destinations)

    def _emit_count_matrix_bands(
        self,
        matrix: Any,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
        projection: tuple[np.ndarray, np.ndarray] | None,
    ) -> Iterator[Any]:
        from ..storage.sharding import SparseWriteBand

        for assay_name, band, producer_bytes in _count_matrix_bands(
            matrix,
            buffers,
            self.assayNames,
            projection,
            self._linear,
        ):
            yield SparseWriteBand(
                destinations[assay_name],
                band,
                producer_bytes,
            )
            del band

    def _finish_count_buffers(
        self,
        buffers: dict[str, Any],
        destinations: dict[str, Any],
    ) -> Iterator[Any]:
        from ..storage.sharding import SparseWriteBand

        for assay_name, band, producer_bytes in _finished_count_bands(buffers):
            yield SparseWriteBand(
                destinations[assay_name],
                band,
                producer_bytes,
            )

    def _assay_feature_projection(
        self,
        assay_features: dict[str, _H5adAssayFeatures],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map each source feature to its assay code and assay-local column."""
        codes = np.full(int(self.h5ad.nFeatures), -1, dtype=np.int64)
        columns = np.zeros(int(self.h5ad.nFeatures), dtype=np.int64)
        for code, assay_name in enumerate(self.assayNames):
            indexes = assay_features[assay_name].featureIndexes
            codes[indexes] = code
            columns[indexes] = np.arange(indexes.size, dtype=np.int64)
        return codes, columns


__all__ = ["H5adImportResult", "H5adToZarr"]
