"""The clone writers refuse inputs that would write a broken file."""

import numpy as np
import pytest

from cytearc.writers.clone_tables import write_clone_matrix, write_clone_table


def test_a_clone_table_needs_one_label_per_cell(tmp_path) -> None:
    with pytest.raises(ValueError, match="one clone label per cell"):
        write_clone_table(tmp_path / "t.csv", np.asarray(["a", "b"]), np.asarray(["x"]))
    assert not (tmp_path / "t.csv").exists()


def test_a_clone_matrix_needs_one_clone_per_cell_among_its_clones(tmp_path) -> None:
    with pytest.raises(ValueError, match="one clone per cell"):
        write_clone_matrix(
            tmp_path / "m.mtx",
            np.asarray(["a", "b"]),
            np.asarray([0]),
            np.asarray(["x"]),
        )
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
        write_clone_matrix(
            tmp_path / "m.mtx", np.asarray(["a"]), np.asarray([1]), np.asarray(["x"])
        )
    assert not (tmp_path / "m.mtx").exists()


def test_an_empty_matrix_lists_no_cell(tmp_path) -> None:
    written = write_clone_matrix(
        tmp_path / "m.mtx",
        np.asarray([], dtype=object),
        np.asarray([], dtype=int),
        np.asarray([], dtype=object),
    )
    assert written == 0
    assert (tmp_path / "m.mtx").read_text().splitlines()[2] == "0 0 0"
    assert (tmp_path / "cells.tsv").read_text() == '"cell"\n'


@pytest.mark.parametrize(
    ("cells", "labels", "what"),
    [(["a\nb"], ["x"], "cell ID"), (["a"], ["x\ry"], "clone label")],
)
def test_a_clone_matrix_refuses_names_that_hold_line_breaks(
    tmp_path, cells, labels, what
) -> None:
    # Each of the matrix's lists holds one name per line.
    with pytest.raises(ValueError, match=f"The {what} .* holds a line break"):
        write_clone_matrix(
            tmp_path / "m.mtx",
            np.asarray(cells, dtype=object),
            np.asarray([0]),
            np.asarray(labels, dtype=object),
        )
    assert list(tmp_path.iterdir()) == []
