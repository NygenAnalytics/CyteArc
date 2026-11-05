"""Steps that the clone operations share, each charged to the operation's memory ledger.

Clone import, the clone summaries, lineage barcodes, recorder alleles,
phylogenies and receptors read the same kinds of stored arrays and name their
files and clones alike: integer codes read whole as int64, the IDs of a
cell-aligned artifact's cells, the labels of clones ranked by size and the
words for the files of an import. Operation modules may not import each
other, so each step lives here once.
"""

from typing import Any

import numpy as np
import zarr

from ...clones.assignments import ranked_clone_label_bytes, ranked_clone_labels
from ...metadata.arguments import ClonotypeArguments
from ...metadata.rows import read_text_rows, read_whole
from ...metadata.selection import (
    CellSelectionRead,
    plan_cell_selection_read,
    read_cell_selection,
)
from ...storage.artifacts import ArtifactRef, inspect_artifact
from ...utils.memory_budget import MemoryLedger

__all__ = [
    "CODES_WITHIN_TYPE",
    "REPRESENTATIVES_WITHIN_TYPE",
    "artifact_cell_ids",
    "files_text",
    "hold_ranked_clone_labels",
    "measured_within_types",
    "plan_artifact_cells",
    "read_codes",
]

#: The array in which clonotypes defined across receptor types save each
#: cell's clonotype within its receptor type, as ``within_receptor_type=True``
#: codes it, -1 for none, which every clone summary measures.
CODES_WITHIN_TYPE = "codes_within_type"
#: The array in which clonotypes defined across receptor types save the
#: representative cell of each clonotype within a receptor type, by its code,
#: as ``within_receptor_type=True`` chooses it, which names it in figures.
REPRESENTATIVES_WITHIN_TYPE = "representatives_within_type"


def files_text(noun: str, n_files: int) -> str:
    """The words for the files of an import in its messages, such as "the clone files"."""
    return f"the {noun} file" if n_files == 1 else f"the {noun} files"


def measured_within_types(clones: ArtifactRef, status: Any, group: zarr.Group) -> bool:
    """Whether the clone summaries measure a clone artifact by each cell's clonotype within its receptor type.

    Clonotypes defined across receptor types are measured so, by the codes
    that they save in ``CODES_WITHIN_TYPE``, so their summaries equal those
    of clonotypes defined within types.

    Args:
        clones: The clone artifact.
        status: Its record, as ``inspect_artifact`` gives it.
        group: Its group.

    Raises:
        ValueError: If clonotypes defined across receptor types were saved
            without each cell's clonotype within its type.
    """
    if clones.kind != ClonotypeArguments.artifact_kind or (status.parameters or {}).get(
        "within_receptor_type", True
    ):
        return False
    if CODES_WITHIN_TYPE not in group:
        raise ValueError(
            "clonotypes was defined across receptor types by an earlier version "
            "of CyteArc, without each cell's clonotype within its receptor type, "
            "which the clone summaries measure; define the clonotypes again with "
            "ds.repertoire.clonotypes, which saves it"
        )
    return True


def read_codes(
    ledger: MemoryLedger, group: zarr.Group, name: str, *, item: str
) -> np.ndarray:
    """Read the whole integer array ``name`` of ``group`` as int64, keeping it charged as ``item``.

    The array is read through :func:`~cytearc.metadata.rows.read_whole`; an
    array stored narrower is widened into an int64 array charged before it
    is made, and the stored values are let go.
    """
    stored = read_whole(ledger, group, name, item=item)
    if stored.dtype == np.int64:
        return stored
    if stored.dtype.kind not in "iub":
        raise TypeError(f"{item} must be stored as integers, not as {stored.dtype}")
    ledger.hold(item, 8 * stored.size)
    codes = stored.astype(np.int64)
    stored_bytes = stored.nbytes
    del stored
    ledger.release(stored_bytes)
    return codes


def hold_ranked_clone_labels(ledger: MemoryLedger, n_clones: int) -> np.ndarray:
    """The labels ``clone_1`` to ``clone_{n_clones}``, charged to ``ledger`` before they are made."""
    ledger.hold("the clones' labels", ranked_clone_label_bytes(n_clones))
    return ranked_clone_labels(n_clones)


def plan_artifact_cells(root: Any, artifact: ArtifactRef) -> CellSelectionRead:
    """Check a cell-aligned artifact's cell selection, for :func:`artifact_cell_ids`, before anything is charged.

    The artifact's record and the selection's identity check are read
    outside the budget, once in the operation's scope.
    """
    selection = inspect_artifact(root, artifact).input_ref("cell_selection")
    return plan_cell_selection_read(root, selection)


def artifact_cell_ids(
    ledger: MemoryLedger,
    root: Any,
    cell_ids: Any,
    planned: CellSelectionRead,
    *,
    item: str = "the cells' IDs",
) -> np.ndarray:
    """The IDs of a cell-aligned artifact's cells, in the order of its arrays, kept charged as ``item``.

    The artifact's cell selection, as :func:`plan_artifact_cells` planned
    it, is read through
    :func:`~cytearc.metadata.selection.read_cell_selection` and the IDs of
    its cells from ``cell_ids``, the store's stored IDs, through
    :func:`~cytearc.metadata.rows.read_text_rows`, as Python strings.
    """
    cell_index = read_cell_selection(
        ledger, root, planned, item="the positions of the artifact's cells"
    )
    ids = read_text_rows(ledger, cell_ids, cell_index, item=item)
    index_bytes = cell_index.nbytes
    del cell_index
    ledger.release(index_bytes)
    return ids
