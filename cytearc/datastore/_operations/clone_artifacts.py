"""The ``clones`` artifact that every clone source saves.

Clone import, clones from a cell column, barcode clones, clones called from
variants, allele clones and the clones of a phylogeny's cut all save the same
artifact: per cell the clone label in ``values``, the clone's rank in
``codes`` and its size in ``clone_size``, and per clone ``clone_ids`` and
``clone_sizes``, beside the per-cell arrays that a source adds, such as each
cell's probability of its clone. These helpers plan it, state what saving it
holds, charge the save to the operation's memory ledger before it writes
anything, write it and log it, once for every operation mixin that saves
clones.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

import numpy as np
import zarr

from ...clones.assignments import CloneAssignment
from ...graph.arguments import ArgumentRecord
from ...metadata.arguments import CLONE_DIAGNOSTICS
from ...metadata.artifacts import (
    cell_data_write_bytes,
    headroom_bytes,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...storage.artifact_writer import PlannedArtifact
from ...storage.artifacts import ArtifactRef
from ...utils.logging import logger
from ...utils.arguments import quoted_value
from ...utils.memory_budget import MemoryLedger

__all__ = [
    "CLONE_PROBABILITY",
    "SITE_ARRAYS",
    "SOURCE_ARRAYS",
    "CloneArguments",
    "clone_diagnostics",
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
#: The arrays of the sites that a clone source may save beside its clones:
#: genotype clones' site labels and each within group's dropout and error
#: probability at each site, a row per group.
SITE_ARRAYS: tuple[str, ...] = (
    "site_labels",
    "dropout_probability",
    "error_probability",
)
# Bytes of a reference to a Python object in an array of objects.
_REFERENCE_BYTES = 8
# Python's objects of a save while it gathers the clone sizes, whatever the
# number of cells, which the writer's own allowance covers once it writes.
_GATHERING_FIXED_BYTES = 64 << 10


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
    site_arrays: tuple[str, ...] = (),
) -> PlannedArtifact:
    """Reuse saved clones with these arguments, or plan new ones.

    ``source_arrays`` names further per-cell arrays that the source saves,
    each one of ``SOURCE_ARRAYS``, with their dtype kinds, such as barcode
    clones' ``n_barcodes`` and ``doublet``, and ``site_arrays`` the arrays of
    the sites that it saves, of ``SITE_ARRAYS``; saved clones are reused only
    with them.
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
            and all(isinstance(group.get(name), zarr.Array) for name in site_arrays)
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
    site_arrays: Mapping[str, tuple[tuple[int, ...], Any]] | None = None,
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
    :func:`~cytearc.metadata.artifacts.cell_data_write_bytes` states. What
    the save builds and gathers is counted exactly as it is held, so the
    cost adds :func:`~cytearc.metadata.artifacts.headroom_bytes` of it.
    ``site_arrays`` maps each array of the sites that the source saves to its
    shape and stored dtype, a text array at its widest value's width, written
    after the clones' arrays from what the caller holds. The labels' text is
    the caller's to count.
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
    arrays.update(dict(site_arrays or {}))
    return given + max(
        gathering + headroom_bytes(gathering) + _GATHERING_FIXED_BYTES,
        built + headroom_bytes(built) + cell_data_write_bytes(arrays),
    )


def _stored_dtype(values: np.ndarray) -> np.dtype:
    """The dtype that the store saves ``values`` at: text at its widest value's width, numbers as they are."""
    values = np.asarray(values)
    if values.dtype.kind in {"O", "U", "S"}:
        width = max((len(str(value)) for value in values.tolist()), default=0)
        return np.dtype(f"U{max(1, width)}")
    return values.dtype


def clone_diagnostics(codes: np.ndarray, sizes: np.ndarray) -> dict[str, int | float]:
    """Return the diagnostics that every source's clones save, from each cell's clone and each clone's size.

    ``cellsWithClone`` counts the cells given a clone, -1 marking a cell
    without one in ``codes``, and ``largestCloneShare`` is the share of them
    that the largest clone holds, 0 when no cell is given one. The clone
    writer saves them with the clones of every source, and clonotypes, which
    their own writer saves, save them too.
    """
    n_with_clone = int(np.count_nonzero(np.asarray(codes) >= 0))
    largest = int(np.max(sizes)) if len(sizes) else 0
    return {
        "cellsWithClone": n_with_clone,
        "largestCloneShare": largest / n_with_clone if n_with_clone else 0.0,
    }


def write_clones(
    root: zarr.Group,
    planned: PlannedArtifact,
    assignment: CloneAssignment,
    attributes: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray] | None = None,
    *,
    ledger: MemoryLedger,
    diagnostics: Mapping[str, Any] | None = None,
    site_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Write planned clones with their attributes and the arrays their source adds.

    The clones save the diagnostics that their operation declares: those of
    :func:`clone_diagnostics`, which the writer adds, beside the source's own
    ``diagnostics``, such as the join's counts of an import. What saving
    holds, as :func:`clone_write_bytes` states, is charged to ``ledger``
    beside what the caller holds there, such as its labels' text, before
    anything is written.

    Raises:
        ValueError: If ``source_arrays`` names an array that is not one of
            ``SOURCE_ARRAYS``, or the assignment's probabilities already;
            ``site_arrays`` names one that is not one of ``SITE_ARRAYS``;
            ``diagnostics`` names one that the writer saves; or the
            diagnostics are not those that the operation declares.
        MemoryBudgetError: If saving the clones would pass the ledger's
            budget.
    """
    given = dict(diagnostics or {})
    written = [name for name in CLONE_DIAGNOSTICS if name in given]
    if written:
        raise ValueError(
            f"The clone writer saves {', '.join(written)} for every source; leave "
            "them out of the source's diagnostics"
        )
    sources = dict(source_arrays or {})
    for name in sources:
        if name not in SOURCE_ARRAYS:
            raise ValueError(
                f"{quoted_value(name)} is not a per-cell array that a clone source adds beside "
                f"its clones; those are {', '.join(SOURCE_ARRAYS)}"
            )
        if name == CLONE_PROBABILITY and assignment.probabilities is not None:
            raise ValueError(
                f"{quoted_value(name)} is written from the assignment's probabilities already; "
                "pass it once"
            )
    sites = dict(site_arrays or {})
    for name in sites:
        if name not in SITE_ARRAYS:
            raise ValueError(
                f"{quoted_value(name)} is not an array of the sites that a clone source "
                f"saves; those are {', '.join(SITE_ARRAYS)}"
            )
    n_cells = len(assignment.codes)
    width = max((len(str(label)) for label in assignment.clone_ids.tolist()), default=0)
    # The store saves each cell's clone label at the width of the longest.
    ledger.peak(
        f"writing the clones of {n_cells} cells, each label {width} characters "
        "wide as stored",
        clone_write_bytes(
            n_cells,
            len(assignment.clone_ids),
            width,
            with_probability=assignment.probabilities is not None,
            source_arrays={
                name: np.asarray(values).dtype for name, values in sources.items()
            },
            site_arrays={
                name: (np.shape(values), _stored_dtype(values))
                for name, values in sites.items()
            },
        ),
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
    arrays.update(sites)
    write_cell_data_artifact(
        root,
        planned,
        arrays,
        attributes=dict(attributes),
        diagnostics={
            **clone_diagnostics(assignment.codes, assignment.clone_sizes),
            **given,
        },
    )


def log_clones(assignment: CloneAssignment, n_cells: int) -> None:
    """Log how many clones hold how many cells."""
    n_assigned = int(np.count_nonzero(assignment.codes >= 0))
    largest = int(assignment.clone_sizes[0]) if len(assignment.clone_sizes) else 0
    logger.info(
        f"Saved {len(assignment.clone_ids)} clones of {n_assigned} of {n_cells} "
        f"cells; the largest holds {largest} cells"
    )
