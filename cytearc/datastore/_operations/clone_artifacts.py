"""The ``clones`` artifact that every clone source saves.

Clone import, clones from a cell column and clones called from variants all
save the same artifact: per cell the clone label in ``values``, the clone's
rank in ``codes`` and its size in ``clone_size``, and per clone
``clone_ids`` and ``clone_sizes``. These helpers plan it, check that its
labels fit the memory budget as the store saves them, write it and log it,
once for every operation mixin that saves clones.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import zarr

from ...clones.assignments import CloneAssignment
from ...graph.arguments import OperationArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import ArtifactRef
from ...utils.logging import logger

__all__ = [
    "CLONE_PROBABILITY",
    "check_clone_write_budget",
    "log_clones",
    "plan_clones",
    "write_clones",
]

#: The per-cell array of each cell's probability of its clone.
CLONE_PROBABILITY = "clone_probability"


def plan_clones(
    root: zarr.Group,
    arguments: OperationArguments,
    cell_selection: ArtifactRef,
    n_cells: int,
    *,
    with_probability: bool,
    required_attributes: tuple[str, ...],
    invalidate_cache: bool,
) -> PlannedArtifact:
    """Reuse saved clones with these arguments, or plan new ones."""
    record = arguments.to_record()
    arrays: dict[str, tuple[tuple[int | None, ...], str | None]] = {
        "values": ((n_cells,), None),
        "codes": ((n_cells,), "i"),
        "clone_size": ((n_cells,), "i"),
    }
    if with_probability:
        arrays[CLONE_PROBABILITY] = ((n_cells,), "f")

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
        invalidate_cache=invalidate_cache,
        required_attributes=required_attributes,
        reuse_validator=has_clones,
    )


def check_clone_write_budget(
    memory_bytes: int, held: int, clone_ids: np.ndarray, n_cells: int
) -> None:
    """Refuse clones whose labels would not fit the memory budget as stored.

    The store saves text at the width of the longest value, so each cell's
    label takes four bytes per character of the longest as it is written,
    beside its code; ``held`` is what the caller holds beside them.

    Raises:
        MemoryError: If the clones would not fit ``memory_bytes``.
    """
    width = max((len(str(label)) for label in clone_ids.tolist()), default=0)
    peak = held + n_cells * (8 + 4 * width)
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
) -> None:
    """Write planned clones with their attributes."""
    arrays: dict[str, np.ndarray] = {
        "values": assignment.values,
        "codes": assignment.codes.astype(np.int32),
        "clone_size": assignment.clone_size,
        "clone_ids": assignment.clone_ids,
        "clone_sizes": assignment.clone_sizes.astype(np.int64),
    }
    if assignment.probabilities is not None:
        arrays[CLONE_PROBABILITY] = assignment.probabilities.astype(np.float64)
    write_cell_data_artifact(root, planned, arrays, attributes=dict(attributes))


def log_clones(assignment: CloneAssignment, n_cells: int) -> None:
    """Log how many clones hold how many cells."""
    n_assigned = int(np.count_nonzero(assignment.codes >= 0))
    largest = int(assignment.clone_sizes[0]) if len(assignment.clone_sizes) else 0
    logger.info(
        f"Saved {len(assignment.clone_ids)} clones of {n_assigned} of {n_cells} "
        f"cells; the largest holds {largest} cells"
    )
