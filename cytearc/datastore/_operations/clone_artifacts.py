"""The ``clones`` artifact that every clone source saves.

Clone import, clones from a cell column, barcode clones, clones called from
variants, allele clones and the clones of a phylogeny's cut all save the same
artifact: per cell the clone label in ``values``, the clone's rank in
``codes`` and its size in ``clone_size``, and per clone ``clone_ids`` and
``clone_sizes``, beside the per-cell arrays that a source adds, such as each
cell's probability of its clone. These helpers plan it, state what saving it
holds, refuse a save that would not fit the memory budget, write it and log
it, once for every operation mixin that saves clones.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

import numpy as np
import zarr

from ...clones.assignments import CloneAssignment
from ...graph.arguments import ArgumentRecord
from ...metadata.artifacts import (
    cell_data_write_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import ArtifactRef
from ...utils.logging import logger

__all__ = [
    "CLONE_PROBABILITY",
    "SOURCE_ARRAYS",
    "CloneArguments",
    "clone_write_bytes",
    "log_clones",
    "plan_clones",
    "write_clones",
]

#: The per-cell array of each cell's probability of its clone.
CLONE_PROBABILITY = "clone_probability"
#: The per-cell arrays that a clone source may add, with the dtype that the
#: table of loaded cells holds: each cell's probability of its clone, its
#: barcodes, whether calling removed it as a doublet, whether it left it out
#: as ambiguous, its calls agreeing with two clones that disagree, and its
#: distinct rare recorder alleles.
SOURCE_ARRAYS: Mapping[str, Any] = {
    CLONE_PROBABILITY: np.float64,
    "n_barcodes": np.int64,
    "doublet": bool,
    "ambiguous": bool,
    "n_alleles": np.int64,
}
# Bytes of a reference to a Python object in an array of objects.
_REFERENCE_BYTES = 8


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


def clone_write_bytes(
    n_cells: int,
    n_clones: int,
    width: int,
    *,
    with_probability: bool = False,
    source_arrays: Mapping[str, Any] | None = None,
) -> int:
    """The most bytes that saving clones holds, from their assignment on.

    ``width`` is the characters of the longest label, at which the store saves
    every cell's label, and ``source_arrays`` maps each per-cell array that
    the source adds to its dtype. Saving holds the assignment, each cell's
    code as int64 with each clone's label and size and, when the source gives
    them, each cell's probability, and the source's arrays as passed.
    :func:`write_clones` builds each cell's label as a reference to its
    clone's, its code as int32 and its clone's size, gathering the sizes
    beside a mask of the cells with a clone and two int64 arrays, and copies
    the clones' sizes and the probabilities; writing them holds what
    :func:`~cytearc.metadata.artifacts.cell_data_write_bytes` states. The
    labels' text is the caller's to count.
    """
    label = np.dtype(f"U{max(1, width)}")
    sources = {name: np.dtype(dtype) for name, dtype in (source_arrays or {}).items()}
    probability = 8 * n_cells if with_probability else 0
    given = (
        8 * n_cells
        + (_REFERENCE_BYTES + 8) * n_clones
        + probability
        + sum(n_cells * int(dtype.itemsize) for dtype in sources.values())
    )
    built = n_cells * (_REFERENCE_BYTES + 4 + 8) + 8 * n_clones + probability
    gathering = n_cells * (_REFERENCE_BYTES + 4 + 8 + 1 + 8 + 8)
    arrays: dict[str, tuple[tuple[int, ...], Any]] = {
        "values": ((n_cells,), label),
        "codes": ((n_cells,), np.int32),
        "clone_size": ((n_cells,), np.int64),
        "clone_ids": ((n_clones,), label),
        "clone_sizes": ((n_clones,), np.int64),
    }
    if with_probability:
        arrays[CLONE_PROBABILITY] = ((n_cells,), np.float64)
    arrays.update({name: ((n_cells,), dtype) for name, dtype in sources.items()})
    return given + max(gathering, built + cell_data_write_bytes(arrays))


def write_clones(
    root: zarr.Group,
    planned: PlannedArtifact,
    assignment: CloneAssignment,
    attributes: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray] | None = None,
    *,
    memory_bytes: int,
    held_bytes: int,
) -> None:
    """Write planned clones with their attributes and the arrays their source adds.

    The save is refused before it writes anything when ``held_bytes``, what
    the caller holds beside the assignment, such as its labels' text, and
    what saving holds, as :func:`clone_write_bytes` states, pass
    ``memory_bytes``.

    Raises:
        ValueError: If ``source_arrays`` names an array that is not one of
            ``SOURCE_ARRAYS``, or the assignment's probabilities already.
        MemoryError: If saving the clones would not fit ``memory_bytes``.
    """
    sources = dict(source_arrays or {})
    for name in sources:
        if name not in SOURCE_ARRAYS:
            raise ValueError(
                f"{name!r} is not a per-cell array that a clone source adds beside "
                f"its clones; those are {', '.join(SOURCE_ARRAYS)}"
            )
        if name == CLONE_PROBABILITY and assignment.probabilities is not None:
            raise ValueError(
                f"{name!r} is written from the assignment's probabilities already; "
                "pass it once"
            )
    n_cells = len(assignment.codes)
    width = max((len(str(label)) for label in assignment.clone_ids.tolist()), default=0)
    needed = held_bytes + clone_write_bytes(
        n_cells,
        len(assignment.clone_ids),
        width,
        with_probability=assignment.probabilities is not None,
        source_arrays={
            name: np.asarray(values).dtype for name, values in sources.items()
        },
    )
    if needed > memory_bytes:
        raise MemoryError(
            f"Saving the clones of {n_cells} cells holds about {needed} bytes, more "
            f"than the memory budget of {memory_bytes} bytes, since the store saves "
            "each cell's clone label at the width of the longest, "
            f"{width} characters. Raise mem_budget."
        )
    arrays: dict[str, np.ndarray] = {
        "values": assignment.values,
        "codes": assignment.codes.astype(np.int32),
        "clone_size": assignment.clone_size,
        "clone_ids": assignment.clone_ids,
        "clone_sizes": assignment.clone_sizes.astype(np.int64),
    }
    if assignment.probabilities is not None:
        arrays[CLONE_PROBABILITY] = assignment.probabilities.astype(np.float64)
    arrays.update(sources)
    write_cell_data_artifact(root, planned, arrays, attributes=dict(attributes))


def log_clones(assignment: CloneAssignment, n_cells: int) -> None:
    """Log how many clones hold how many cells."""
    n_assigned = int(np.count_nonzero(assignment.codes >= 0))
    largest = int(assignment.clone_sizes[0]) if len(assignment.clone_sizes) else 0
    logger.info(
        f"Saved {len(assignment.clone_ids)} clones of {n_assigned} of {n_cells} "
        f"cells; the largest holds {largest} cells"
    )
