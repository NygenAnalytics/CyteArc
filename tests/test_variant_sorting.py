"""Sorting rows keyed by cell in bands, in memory and through runs on disk."""

from pathlib import Path

import numpy as np
import pytest

from cytearc.variants.counts import CALL_DTYPE, count_dtype
from cytearc.variants.sorting import CellRowSorter, sort_rows

COUNTS = count_dtype(("forward",))


def _random_rows(n_rows: int, n_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.zeros(n_rows, dtype=COUNTS)
    # Distinct keys per cell, so the order of a sorted band is unique.
    pairs = rng.choice(n_cells * 1000, size=n_rows, replace=False)
    rows["cell"] = pairs // 1000
    rows["key"] = pairs % 1000
    rows["count"] = rng.integers(1, 50, n_rows)
    rows["forward"] = rows["count"] // 2
    return rows


def _sorted(rows: np.ndarray) -> np.ndarray:
    return rows[np.lexsort((rows["key"], rows["cell"]))]


def _collect(
    sorter: CellRowSorter, max_rows: int
) -> tuple[list[tuple[int, int]], np.ndarray]:
    bounds, pieces = [], []
    for first, stop, rows in sorter.bands(max_rows):
        bounds.append((first, stop))
        band = rows["counts"]
        assert np.all((band["cell"] >= first) & (band["cell"] < stop))
        pieces.append(band)
    return bounds, np.concatenate(pieces)


@pytest.mark.parametrize("buffer_bytes", [10**9, 2_000, 1])
def test_rows_in_any_order_come_back_sorted_by_cell_and_key(
    buffer_bytes, tmp_path
) -> None:
    rows = _random_rows(5_000, 40, seed=1)
    with CellRowSorter(
        {"counts": (COUNTS, ("cell", "key"))},
        n_cells=40,
        buffer_bytes=buffer_bytes,
        temp_dir=tmp_path,
    ) as sorter:
        for chunk in np.array_split(rows, 37):
            sorter.add("counts", chunk)
        if buffer_bytes == 10**9:
            assert sorter.n_runs == 0
        else:
            assert sorter.n_runs >= 2
        np.testing.assert_array_equal(
            sorter.rows_per_cell("counts"), np.bincount(rows["cell"], minlength=40)
        )
        bounds, collected = _collect(sorter, max_rows=700)
    np.testing.assert_array_equal(collected, _sorted(rows))
    # Bands cover the cells in order, each within the row limit.
    assert bounds[0][0] == 0 and bounds[-1][1] == 40
    assert all(
        left[1] == right[0] for left, right in zip(bounds, bounds[1:], strict=False)
    )
    per_cell = np.bincount(rows["cell"], minlength=40)
    assert all(per_cell[first:stop].sum() <= 700 for first, stop in bounds)
    # The runs are removed when the sorter closes.
    assert not list(tmp_path.iterdir())


def test_a_cell_with_more_rows_than_a_band_is_its_own_band() -> None:
    rows = np.zeros(10, dtype=COUNTS)
    rows["cell"] = [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
    rows["key"] = [0, 5, 4, 3, 2, 1, 0, 2, 1, 0]
    with CellRowSorter(
        {"counts": (COUNTS, ("cell", "key"))}, n_cells=4, buffer_bytes=10**6
    ) as sorter:
        sorter.add("counts", rows)
        assert sorter.band_plan(3).tolist() == [0, 1, 2, 4]
        bands = [
            (first, stop, len(band["counts"])) for first, stop, band in sorter.bands(3)
        ]
    assert bands == [(0, 1, 1), (1, 2, 6), (2, 4, 3)]


def test_streams_share_their_bands_and_calls_sort_by_source() -> None:
    calls = np.zeros(4, dtype=CALL_DTYPE)
    calls["cell"] = [1, 0, 1, 1]
    calls["key"] = [2, 2, 2, 1]
    calls["source"] = [1, 0, 0, 0]
    counts = _random_rows(30, 3, seed=2)
    with CellRowSorter(
        {
            "counts": (COUNTS, ("cell", "key")),
            "calls": (CALL_DTYPE, ("cell", "key", "source")),
        },
        n_cells=3,
        buffer_bytes=1,
    ) as sorter:
        sorter.add("calls", calls)
        sorter.add("counts", counts)
        merged_calls = np.concatenate(
            [rows["calls"] for _f, _s, rows in sorter.bands(10**6)]
        )
    assert merged_calls[["cell", "key", "source"]].tolist() == [
        (0, 2, 0),
        (1, 1, 0),
        (1, 2, 0),
        (1, 2, 1),
    ]


def test_the_sorter_refuses_rows_it_cannot_place() -> None:
    with pytest.raises(ValueError, match="must sort by 'cell' first"):
        CellRowSorter({"counts": (COUNTS, ("key", "cell"))}, n_cells=2, buffer_bytes=10)
    with pytest.raises(ValueError, match="must sort by 'cell' first"):
        CellRowSorter(
            {"counts": (COUNTS, ("cell", "allele"))}, n_cells=2, buffer_bytes=10
        )
    sorter = CellRowSorter(
        {"counts": (COUNTS, ("cell", "key"))}, n_cells=2, buffer_bytes=10
    )
    with pytest.raises(ValueError, match="must have the dtype"):
        sorter.add("counts", np.zeros(1, dtype=count_dtype(())))
    outside = np.zeros(1, dtype=COUNTS)
    outside["cell"] = 2
    with pytest.raises(ValueError, match="name a cell out of range"):
        sorter.add("counts", outside)
    sorter.add("counts", np.zeros(0, dtype=COUNTS))
    with pytest.raises(ValueError, match="max_rows must be at least 1"):
        list(sorter.bands(0))
    with pytest.raises(ValueError, match="cannot be added once bands are read"):
        sorter.add("counts", np.zeros(1, dtype=COUNTS))
    sorter.close()


def test_a_folder_that_cannot_be_removed_is_logged(monkeypatch, tmp_path) -> None:
    import shutil

    from cytearc.utils.logging import collect_warnings

    sorter = CellRowSorter(
        {"counts": (COUNTS, ("cell", "key"))},
        n_cells=3,
        buffer_bytes=1,
        temp_dir=tmp_path,
    )
    sorter.add("counts", _random_rows(5, 3, seed=3))
    assert sorter.n_runs == 1

    def refuse(path: Path) -> None:
        raise OSError("busy")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with collect_warnings() as messages:
        sorter.close()
    assert len(messages) == 1 and "Could not remove the temporary rows" in messages[0]
    # Closing again does nothing.
    sorter.close()


def test_sort_rows_keeps_equal_keys_in_order() -> None:
    rows = np.zeros(4, dtype=COUNTS)
    rows["cell"] = [1, 0, 1, 0]
    rows["key"] = [3, 3, 3, 1]
    rows["count"] = [1, 2, 3, 4]
    assert sort_rows(rows, ("cell", "key"))["count"].tolist() == [4, 2, 1, 3]
    assert sort_rows(rows[:1], ("cell", "key"))["count"].tolist() == [1]


def test_sort_rows_keeps_cells_apart_whatever_their_keys() -> None:
    # mgatk's alleles reach 4 * 16,568 - 1 = 66,271 on rCRS, beyond 16 bits,
    # so the cell must sit above every key in the combined sort key.
    rows = np.zeros(6, dtype=COUNTS)
    rows["cell"] = [1, 0, 2, 1, 0, 2]
    rows["key"] = [10, 66_271, 3, 66_000, 3, 2**31 - 1]
    rows["count"] = np.arange(6)
    ordered = sort_rows(rows, ("cell", "key"))
    assert list(
        zip(ordered["cell"].tolist(), ordered["key"].tolist(), strict=True)
    ) == [
        (0, 3),
        (0, 66_271),
        (1, 10),
        (1, 66_000),
        (2, 3),
        (2, 2**31 - 1),
    ]
    np.testing.assert_array_equal(ordered, _sorted(rows))


def test_bands_can_be_read_again() -> None:
    rows = _random_rows(50, 5, seed=4)
    with CellRowSorter(
        {"counts": (COUNTS, ("cell", "key"))}, n_cells=5, buffer_bytes=10**6
    ) as sorter:
        sorter.add("counts", rows)
        first = np.concatenate([band["counts"] for _f, _s, band in sorter.bands(20)])
        again = np.concatenate([band["counts"] for _f, _s, band in sorter.bands(1_000)])
    np.testing.assert_array_equal(first, again)
