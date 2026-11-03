"""Write character matrices of lineage recorder alleles for tree builders.

Tree builders read a lineage recorder's alleles as a character matrix:
cells by characters, one character per site of an integration, with integer
states. ``write_treedata_characters`` writes the TreeData ``.zarr`` store
that Cassiopeia 3.0 reads, and ``write_laml_characters`` the CSV that LAML
reads, with its priors and a table that names each state's allele.
"""

import csv
import os
from pathlib import Path

import numpy as np

from ..clones.recorders import CharacterMatrix
from ..readers._treedata import write_character_store
from ..utils.arrays import sorted_unique

__all__ = [
    "laml_paths",
    "laml_row_bytes",
    "write_treedata_characters",
    "write_laml_characters",
]

# Characters that LAML's CSV and the Newick trees it writes cannot hold in a
# cell's or a character's name.
_LAML_FORBIDDEN = frozenset(",;:()[]'\"\t\r\n ")
# Bytes that each state of a block of LAML's text takes as it is written:
# its position among the states, the check of that position, a reference
# to the state's text, which each state shares, and the reference in the
# block's rows as lists, 24 in all as traced, beside the row's joined text.
_LAML_BYTES_PER_STATE = 40
# At most this many cells of LAML's text are written at once.
_LAML_MAX_BLOCK = 50_000


def write_treedata_characters(
    path: str | os.PathLike[str], cell_ids: np.ndarray, matrix: CharacterMatrix
) -> None:
    """Write a character matrix as the TreeData ``.zarr`` store Cassiopeia reads.

    States are 0 unedited, -1 missing and 1 to k edits, which
    ``uns['character_states']`` names.

    Raises:
        FileExistsError: If ``path`` exists.
    """
    write_character_store(
        path,
        cell_ids,
        matrix.characters,
        matrix.codes,
        matrix.states.loc[:, ["character", "state", "allele"]],
        missing_state=-1,
        unmodified_state=0,
    )


def laml_paths(path: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    """The matrix, priors and state table that a LAML export writes.

    The priors and the state table sit beside the matrix, named after its
    stem: ``characters.csv`` gives ``characters_priors.csv`` and
    ``characters_states.csv``.
    """
    target = Path(path)
    stem = target.name[: -len(target.suffix)] if target.suffix else target.name
    return (
        target,
        target.with_name(f"{stem}_priors.csv"),
        target.with_name(f"{stem}_states.csv"),
    )


def laml_row_bytes(n_characters: int) -> int:
    """The bytes that one cell's row of LAML's text holds as it is written."""
    return _LAML_BYTES_PER_STATE * (max(1, int(n_characters)) + 1)


def _refuse_laml_names(names: list[str], what: str) -> None:
    for name in names:
        if not name or _LAML_FORBIDDEN & set(name):
            raise ValueError(
                f"The {what} {name!r} holds a character that LAML's CSV or the "
                "Newick trees it writes cannot hold: a comma, semicolon, colon, "
                "bracket, quote or space"
            )


def write_laml_characters(
    path: str | os.PathLike[str],
    cell_ids: np.ndarray,
    matrix: CharacterMatrix,
    *,
    block_cells: int = _LAML_MAX_BLOCK,
) -> None:
    """Write a character matrix as LAML's CSV, with its priors and state table.

    The matrix has a header ``cell,<characters>``, then one row per cell,
    with 0 unedited, ``?`` missing and 1 to k edits. The priors are LAML's
    headerless rows of a character's position from 0, a state and its
    probability; the state table names each state's allele, with a header.
    Files of these names are replaced.

    Args:
        path: The matrix to write.
        cell_ids: Each row's cell.
        matrix: The character matrix.
        block_cells: The cells written at once.

    Raises:
        ValueError: If a cell or character name holds a character that LAML's
            files cannot hold.
    """
    cells = [str(cell) for cell in np.asarray(cell_ids, dtype=object).tolist()]
    characters = list(matrix.characters)
    _refuse_laml_names(cells, "cell")
    _refuse_laml_names(characters, "character")
    target, priors_path, states_path = laml_paths(path)
    codes = np.asarray(matrix.codes)
    # Each state's text is made once and shared by every cell that holds it:
    # missing, unedited and each edited state of the state table.
    values = sorted_unique(
        np.concatenate(
            [[-1, 0], np.asarray(matrix.states["state"], dtype=np.int64)]
        ).astype(np.int64)
    )
    texts = np.asarray(
        ["?" if value < 0 else str(value) for value in values.tolist()], dtype=object
    )
    step = max(1, min(_LAML_MAX_BLOCK, int(block_cells)))
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(["cell", *characters]) + "\n")
        for start in range(0, len(cells), step):
            block = codes[start : start + step]
            positions = np.minimum(np.searchsorted(values, block), len(values) - 1)
            if not np.array_equal(values[positions], block):
                raise ValueError(
                    "The character matrix holds a state that its state table "
                    "does not list"
                )
            rows = texts[positions].tolist()
            del positions
            handle.writelines(
                ",".join([cell, *row]) + "\n"
                for cell, row in zip(cells[start : start + step], rows, strict=True)
            )
            del rows
    priors = matrix.priors()
    with open(priors_path, "w", encoding="utf-8", newline="") as handle:
        handle.writelines(
            f"{character},{state},{probability!r}\n"
            for character, state, probability in zip(
                priors["character"].tolist(),
                priors["state"].tolist(),
                priors["probability"].tolist(),
                strict=True,
            )
        )
    with open(states_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["character", "state", "allele", "kind", "n_cells"])
        writer.writerows(
            matrix.states.loc[
                :, ["character", "state", "allele", "kind", "n_cells"]
            ].itertuples(index=False, name=None)
        )
