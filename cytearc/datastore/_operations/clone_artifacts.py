"""The ``clones`` artifact that every clone source saves.

Clone import, clones from a cell column, barcode clones and clones called
from variants all save the same artifact: per cell the clone label in
``values``, the clone's rank in ``codes`` and its size in ``clone_size``, and
per clone ``clone_ids`` and ``clone_sizes``, beside the per-cell arrays that a
source adds, such as each cell's probability of its clone. These helpers plan
it, check that its labels fit the memory budget as the store saves them,
write it and log it, once for every operation mixin that saves clones.
"""

import io
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

import numpy as np
import zarr

from ...clones.assignments import CloneAssignment
from ...graph.arguments import ArgumentRecord
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import ArtifactRef
from ...utils.logging import logger

__all__ = [
    "CLONE_PROBABILITY",
    "SOURCE_ARRAYS",
    "CloneArguments",
    "check_clone_write_budget",
    "log_clones",
    "plan_clones",
    "write_clones",
    "zarr_write_buffer_bytes",
]

#: The per-cell array of each cell's probability of its clone.
CLONE_PROBABILITY = "clone_probability"
#: The per-cell arrays that a clone source may add, with the dtype that the
#: table of loaded cells holds: each cell's probability of its clone, its
#: barcodes, whether calling removed it as a doublet, and whether it left it
#: out as ambiguous, its calls agreeing with two clones that disagree.
SOURCE_ARRAYS: Mapping[str, Any] = {
    CLONE_PROBABILITY: np.float64,
    "n_barcodes": np.int64,
    "doublet": bool,
    "ambiguous": bool,
}


class CloneArguments(Protocol):
    """The arguments of an operation that saves clones, as ``plan_clones`` reads them."""

    operation: ClassVar[str]
    artifact_kind: ClassVar[str]

    @property
    def invalidate_cache(self) -> bool: ...

    def to_record(self) -> ArgumentRecord: ...


def plan_clones(
    root: zarr.Group,
    arguments: CloneArguments,
    cell_selection: ArtifactRef,
    n_cells: int,
    *,
    with_probability: bool,
    required_attributes: tuple[str, ...],
    source_arrays: Mapping[str, str] | None = None,
) -> PlannedArtifact:
    """Reuse saved clones with these arguments, or plan new ones.

    ``source_arrays`` names further per-cell arrays that the source saves,
    each one of ``SOURCE_ARRAYS``, with their dtype kinds, such as barcode
    clones' ``n_barcodes`` and ``doublet``; saved clones are reused only with
    them.
    """
    record = arguments.to_record()
    arrays: dict[str, tuple[tuple[int | None, ...], str | None]] = {
        "values": ((n_cells,), None),
        "codes": ((n_cells,), "i"),
        "clone_size": ((n_cells,), "i"),
    }
    if with_probability:
        arrays[CLONE_PROBABILITY] = ((n_cells,), "f")
    for name, kind in (source_arrays or {}).items():
        arrays[name] = ((n_cells,), kind)

    def has_clones(_ref: ArtifactRef, group: zarr.Group) -> bool:
        ids = group.get("clone_ids")
        sizes = group.get("clone_sizes")
        return (
            isinstance(ids, zarr.Array)
            and isinstance(sizes, zarr.Array)
            and ids.shape == sizes.shape
        )

    return plan_cell_data_artifact(
        root,
        scope="datastore",
        kind=arguments.artifact_kind,
        operation=arguments.operation,
        parameters=record.parameters,
        inputs=record.inputs,
        execution_options=record.execution_options,
        cell_selection=cell_selection,
        arrays=arrays,
        invalidate_cache=arguments.invalidate_cache,
        required_attributes=required_attributes,
        reuse_validator=has_clones,
    )


def zarr_write_buffer_bytes() -> int:
    """The bytes that Zarr's writes hold in file buffers at once.

    Zarr writes each chunk and each array's metadata through a buffered file,
    up to its async concurrency of them at once. Each buffer is counted as
    one of io's default buffers, 8 KiB before Python 3.14 and 128 KiB from it,
    which covers the buffer that Python gives a file on a file system whose
    blocks are no larger.
    """
    return int(zarr.config.get("async.concurrency")) * io.DEFAULT_BUFFER_SIZE


def check_clone_write_budget(
    memory_bytes: int, held: int, clone_ids: np.ndarray, n_cells: int
) -> None:
    """Refuse clones whose labels would not fit the memory budget as stored.

    The store saves text at the width of the longest value, so each cell's
    label takes four bytes per character of the longest as it is written,
    beside its code and Zarr's file buffers (``zarr_write_buffer_bytes``);
    ``held`` is what the caller holds beside them.

    Raises:
        MemoryError: If the clones would not fit ``memory_bytes``.
    """
    width = max((len(str(label)) for label in clone_ids.tolist()), default=0)
    peak = held + n_cells * (8 + 4 * width) + zarr_write_buffer_bytes()
    if peak > memory_bytes:
        raise MemoryError(
            f"Saving the clones of {n_cells} cells holds about {peak} bytes, "
            f"more than the memory budget of {memory_bytes} bytes, since the "
            "store saves each cell's clone label at the width of the longest, "
            f"{width} characters. Raise mem_budget."
        )


def write_clones(
    root: zarr.Group,
    planned: PlannedArtifact,
    assignment: CloneAssignment,
    attributes: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Write planned clones with their attributes and the arrays their source adds.

    Raises:
        ValueError: If ``source_arrays`` names an array that is not one of
            ``SOURCE_ARRAYS``, or the assignment's probabilities already.
    """
    arrays: dict[str, np.ndarray] = {
        "values": assignment.values,
        "codes": assignment.codes.astype(np.int32),
        "clone_size": assignment.clone_size,
        "clone_ids": assignment.clone_ids,
        "clone_sizes": assignment.clone_sizes.astype(np.int64),
    }
    if assignment.probabilities is not None:
        arrays[CLONE_PROBABILITY] = assignment.probabilities.astype(np.float64)
    for name, values in (source_arrays or {}).items():
        if name not in SOURCE_ARRAYS:
            raise ValueError(
                f"{name!r} is not a per-cell array that a clone source adds beside "
                f"its clones; those are {', '.join(SOURCE_ARRAYS)}"
            )
        if name in arrays:
            raise ValueError(
                f"{name!r} is written from the assignment's probabilities already; "
                "pass it once"
            )
        arrays[name] = values
    write_cell_data_artifact(root, planned, arrays, attributes=dict(attributes))


def log_clones(assignment: CloneAssignment, n_cells: int) -> None:
    """Log how many clones hold how many cells."""
    n_assigned = int(np.count_nonzero(assignment.codes >= 0))
    largest = int(assignment.clone_sizes[0]) if len(assignment.clone_sizes) else 0
    logger.info(
        f"Saved {len(assignment.clone_ids)} clones of {n_assigned} of {n_cells} "
        f"cells; the largest holds {largest} cells"
    )
