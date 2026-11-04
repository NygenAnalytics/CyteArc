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
from ...metadata.rows import read_text_rows, read_whole
from ...metadata.selection import plan_cell_selection_read, read_cell_selection
from ...storage.artifacts import ArtifactRef, inspect_artifact
from ...utils.memory_budget import MemoryLedger

__all__ = [
    "artifact_cell_ids",
    "files_text",
    "hold_ranked_clone_labels",
    "read_codes",
]


def files_text(noun: str, n_files: int) -> str:
    """The words for the files of an import in its messages, such as "the clone files"."""
    return f"the {noun} file" if n_files == 1 else f"the {noun} files"


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


def artifact_cell_ids(
    ledger: MemoryLedger,
    root: Any,
    cell_ids: Any,
    artifact: ArtifactRef,
    *,
    item: str = "the cells' IDs",
) -> np.ndarray:
    """The IDs of a cell-aligned artifact's cells, in the order of its arrays, kept charged as ``item``.

    The artifact's cell selection is read through
    :func:`~cytearc.metadata.selection.read_cell_selection` and the IDs of
    its cells from ``cell_ids``, the store's stored IDs, through
    :func:`~cytearc.metadata.rows.read_text_rows`, as Python strings.
    """
    selection = inspect_artifact(root, artifact).input_ref("cell_selection")
    planned = plan_cell_selection_read(root, selection)
    cell_index = read_cell_selection(
        ledger, root, planned, item="the positions of the artifact's cells"
    )
    ids = read_text_rows(ledger, cell_ids, cell_index, item=item)
    index_bytes = cell_index.nbytes
    del cell_index
    ledger.release(index_bytes)
    return ids
