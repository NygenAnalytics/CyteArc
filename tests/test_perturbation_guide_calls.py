import gzip
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.perturbation import guide_calls
from cytearc.perturbation.guide_calls import (
    MULTIPLE,
    NEGATIVE,
    guide_call_arrays,
    guide_calls_frames,
    import_settings,
    summarize_guide_counts,
    validated_guide_ids,
)
from cytearc.readers import cellranger
from tests.perturbation_helpers import use_python_kernels
from cytearc.readers.cellranger import protospacer_calls_size, read_protospacer_calls

GUIDES = np.asarray(["ACE2-1", "ACE2-2", "DPP4-1", "Non-Targeting-1"], dtype=object)


def test_guide_ids_are_returned_as_text() -> None:
    ids = validated_guide_ids(np.asarray([b"g1", "g2"], dtype=object))
    assert ids.tolist() == ["b'g1'", "g2"]
    assert validated_guide_ids(GUIDES).tolist() == GUIDES.tolist()


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([], "at least one guide"),
        (["g1", " "], "must not be empty"),
        (["g1", "g2", "g1"], "must be unique"),
        (["g1", "Negative"], "reserved"),
        (["Multiple", "g1"], "reserved"),
    ],
)
def test_guide_ids_that_cannot_label_cells_are_refused(ids, message) -> None:
    with pytest.raises(ValueError, match=message):
        validated_guide_ids(np.asarray(ids, dtype=object))


@pytest.mark.parametrize(
    ("path", "prefix", "message"),
    [
        (7, None, "^path must be a path to Cell Ranger's call file$"),
        ("", None, "^path must be a path to Cell Ranger's call file$"),
        (b"calls.csv", None, "^path must be a path to Cell Ranger's call file$"),
        ("calls.csv", 1, "^barcode_prefix must be text or None$"),
        (Path("calls.csv"), b"L1__", "^barcode_prefix must be text or None$"),
    ],
)
def test_import_settings_refuse_what_the_import_cannot_use(
    path, prefix, message
) -> None:
    # The pipeline recipe runs the same check on its guide_calling settings.
    with pytest.raises(TypeError, match=message):
        import_settings(path, prefix)


def test_import_settings_take_text_or_path_objects() -> None:
    # Whether the path names a file is checked when the file is read.
    for path in ("missing.csv", Path("missing.csv")):
        for prefix in (None, "", "L1__"):
            import_settings(path, prefix)


def test_count_summary_breaks_ties_by_assay_position() -> None:
    counts = np.asarray(
        [
            [0, 5, 5, 1],  # a tie between guides 1 and 2: guide 1 wins
            [0, 0, 0, 0],  # no guide UMIs
            [9, 0, 0, 0],
            [2, 2, 2, 2],  # a four-way tie: guide 0 wins
            [0, 0, 1, 3],
        ],
        dtype=np.uint16,
    )
    # Blocks of two, two, and one rows, as a stream delivers them.
    summary = summarize_guide_counts([counts[:2], counts[2:4], counts[4:]], 5, 4)
    assert summary.top_code.tolist() == [1, -1, 0, 0, 3]
    assert summary.top_umi.tolist() == [5, 0, 9, 2, 3]
    assert summary.total_umi.tolist() == [11, 0, 9, 8, 4]
    np.testing.assert_allclose(
        summary.top_fraction, [5 / 11, np.nan, 1.0, 0.25, 0.75], equal_nan=True
    )


def test_count_summary_checks_the_stream() -> None:
    counts = np.ones((3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="4 columns"):
        summarize_guide_counts([counts[:, :3]], 3, 4)
    with pytest.raises(TypeError, match="unsigned integers"):
        summarize_guide_counts([counts.astype(np.int32)], 3, 4)
    with pytest.raises(ValueError, match="hold 2 rows, but there are 3 cells"):
        summarize_guide_counts([counts[:2]], 3, 4)
    with pytest.raises(ValueError, match="more than 3 rows"):
        summarize_guide_counts([counts, counts[:1]], 3, 4)


def test_count_summary_of_no_cells_is_empty() -> None:
    summary = summarize_guide_counts([], 0, 4)
    assert len(summary.top_code) == 0
    assert len(summary.top_fraction) == 0


def test_call_arrays_order_calls_and_label_cells() -> None:
    # Cell 0: guides 2 (40 UMIs) and 0 (40 UMIs) tie, so guide 0 comes first,
    # then guide 3 (7). Cell 1 has no call. Cell 2 has guide 1 only.
    arrays = guide_call_arrays(
        cells=[0, 2, 0, 0],
        codes=[2, 1, 3, 0],
        umis=[40, 12, 7, 40],
        n_cells=3,
        guide_ids=GUIDES,
    )
    assert arrays.values.tolist() == [MULTIPLE, NEGATIVE, "ACE2-2"]
    assert arrays.n_guides.tolist() == [3, 0, 1]
    assert arrays.guide_codes.tolist() == [[0, 2, 3], [-1, -1, -1], [1, -1, -1]]
    assert arrays.guide_umis.tolist() == [[40, 40, 7], [0, 0, 0], [12, 0, 0]]
    assert arrays.guide_codes.dtype == np.int32
    assert arrays.guide_umis.dtype == np.uint32


def _sorted_rows(cells, codes, umis, n_cells):
    """Each cell's calls as (guide, UMIs), most UMIs first, then by guide."""
    rows: list[list[tuple[int, int]]] = [[] for _ in range(n_cells)]
    for cell, code, count in zip(cells, codes, umis, strict=True):
        rows[int(cell)].append((int(code), int(count)))
    return [sorted(row, key=lambda call: (-call[1], call[0])) for row in rows]


@pytest.mark.parametrize("compiled", [True, False])
def test_call_arrays_order_each_cells_calls_whatever_their_order(
    monkeypatch, compiled
) -> None:
    if not compiled:
        use_python_kernels(monkeypatch, guide_calls)
    rng = np.random.default_rng(41)
    guide_ids = np.asarray([f"g{index}" for index in range(30)], dtype=object)
    for trial in range(200):
        n_cells = int(rng.integers(1, 40))
        held = rng.random((n_cells, len(guide_ids))) < rng.uniform(0, 0.5)
        cells, codes = np.nonzero(held)
        order = rng.permutation(len(cells))
        # Few distinct counts make ties common; some calls hold the uint32
        # maximum.
        umis = rng.integers(0, [3, 50, 2**32][trial % 3], size=len(cells))
        umis[rng.random(len(umis)) < 0.05] = 2**32 - 1
        dtype = ["int64", "uint32", "float64"][trial % 3]
        arrays = guide_call_arrays(
            cells[order],
            codes[order],
            umis[order].astype(dtype),
            n_cells=n_cells,
            guide_ids=guide_ids,
        )
        expected = _sorted_rows(cells, codes, umis, n_cells)
        width = max(1, max(len(row) for row in expected))
        assert arrays.guide_codes.shape == (n_cells, width)
        for cell, row in enumerate(expected):
            assert arrays.guide_codes[cell].tolist() == [code for code, _ in row] + [
                -1
            ] * (width - len(row))
            assert arrays.guide_umis[cell].tolist() == [umi for _, umi in row] + [0] * (
                width - len(row)
            )
        assert arrays.n_guides.tolist() == [len(row) for row in expected]
        assert arrays.values.tolist() == [
            NEGATIVE if not row else guide_ids[row[0][0]] if len(row) == 1 else MULTIPLE
            for row in expected
        ]
        # A guide called again in one cell, with any count, is refused.
        if len(cells):
            with pytest.raises(ValueError, match="more than once"):
                guide_call_arrays(
                    np.append(cells, cells[-1]),
                    np.append(codes, codes[-1]),
                    np.append(umis, 7),
                    n_cells=n_cells,
                    guide_ids=guide_ids,
                )


def test_call_arrays_hold_little_beside_their_padded_arrays() -> None:
    n_cells, per_cell, n_guides = 20_000, 8, 40
    rng = np.random.default_rng(42)
    cells = np.repeat(np.arange(n_cells, dtype=np.int64), per_cell)
    codes = (
        np.argsort(rng.random((n_cells, n_guides)), axis=1)[:, :per_cell]
        .reshape(-1)
        .astype(np.int64)
    )
    umis = rng.integers(1, 4, len(cells)).astype(np.uint32)
    guide_ids = np.asarray([f"g{index}" for index in range(n_guides)], dtype=object)
    guide_call_arrays(cells[:2], codes[:2], umis[:2], n_cells=1, guide_ids=guide_ids)
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        arrays = guide_call_arrays(
            cells, codes, umis, n_cells=n_cells, guide_ids=guide_ids
        )
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()
    padded = arrays.guide_codes.nbytes + arrays.guide_umis.nbytes
    # Beside the padded slots, a cell holds its count, its label as a
    # reference and the masks that choose the label; a call holds nothing,
    # where sorting every call held about 75 bytes per call.
    assert peak - padded <= 32 * n_cells


def test_count_summary_refuses_counts_beyond_uint32_as_calling_does() -> None:
    counts = np.zeros((2, 4), dtype=np.uint64)
    counts[1, 2] = 2**33 + 7
    with pytest.raises(
        ValueError, match="^Guide UMI counts must be at most 4294967295$"
    ):
        summarize_guide_counts([counts], 2, 4)
    counts[1, 2] = 2**32 - 1
    assert summarize_guide_counts([counts], 2, 4).top_umi.tolist() == [0, 2**32 - 1]


def test_call_arrays_without_calls_keep_one_padded_column() -> None:
    arrays = guide_call_arrays([], [], [], n_cells=2, guide_ids=GUIDES)
    assert arrays.values.tolist() == [NEGATIVE, NEGATIVE]
    assert arrays.guide_codes.shape == (2, 1)
    assert arrays.guide_codes.tolist() == [[-1], [-1]]


@pytest.mark.parametrize(
    ("cells", "codes", "umis", "message"),
    [
        ([0, 0], [1, 1], [3, 4], "more than once"),
        ([0], [1, 2], [3], "one cell, one guide"),
        ([3], [1], [3], "Cell positions"),
        ([0], [4], [3], "Guide positions"),
        ([0], [1], [-1], "must not be negative"),
        ([0], [1], [2.5], "whole numbers"),
        ([0], [1], [2**32], "at most"),
    ],
)
def test_call_arrays_refuse_malformed_calls(cells, codes, umis, message) -> None:
    with pytest.raises(ValueError, match=message):
        guide_call_arrays(cells, codes, umis, n_cells=3, guide_ids=GUIDES)


@pytest.mark.parametrize(
    ("cells", "codes"),
    [([0.0], [1]), ([0], [1.0])],
)
def test_call_arrays_refuse_positions_that_are_not_integers(cells, codes) -> None:
    with pytest.raises(ValueError, match="positions must be integers"):
        guide_call_arrays(cells, codes, [3], n_cells=3, guide_ids=GUIDES)


def test_frames_list_each_cell_and_each_call() -> None:
    arrays = guide_call_arrays(
        [0, 0, 2], [2, 0, 1], [5, 9, 3], n_cells=3, guide_ids=GUIDES
    )
    cells, calls = guide_calls_frames(
        np.asarray(["a", "b", "c"]),
        values=arrays.values,
        n_guides=arrays.n_guides,
        top_guide=np.asarray(["ACE2-1", "", "ACE2-2"], dtype=object),
        top_umi=np.asarray([9, 0, 3]),
        top_fraction=np.asarray([9 / 14, np.nan, 1.0]),
        total_umi=np.asarray([14, 0, 3]),
        guide_codes=arrays.guide_codes,
        guide_umis=arrays.guide_umis,
        guide_ids=GUIDES,
    )
    assert cells.index.tolist() == ["a", "b", "c"]
    assert cells["call"].tolist() == [MULTIPLE, NEGATIVE, "ACE2-2"]
    assert calls.to_dict("list") == {
        "ids": ["a", "a", "c"],
        "guide": ["ACE2-1", "DPP4-1", "ACE2-2"],
        "umis": [9, 5, 3],
    }


def _write(path, text: str) -> str:
    path.write_text(text)
    return str(path)


def test_cell_ranger_call_file_is_read(tmp_path) -> None:
    text = (
        "cell_barcode,num_features,feature_call,num_umis\n"
        "AAACCCAGTCGATTCA-1,1,Non-Targeting-1,3678\n"
        "GTTACCCAGTAGTCTC-1,2,ACE2-1|DPP4-1,619|2869\n"
        "TTTGGTTGTTTGGTTG-1,0,None,0\n"
        "CCCAAAGGGTTTCCCA-1,0,,\n"
    )
    for path in (
        _write(tmp_path / "calls.csv", text),
        str(tmp_path / "calls.csv.gz"),
    ):
        if path.endswith(".gz"):
            with gzip.open(path, "wt") as handle:
                handle.write(text)
        calls = read_protospacer_calls(path)
        assert calls.barcodes.tolist() == [
            "AAACCCAGTCGATTCA-1",
            "GTTACCCAGTAGTCTC-1",
            "TTTGGTTGTTTGGTTG-1",
            "CCCAAAGGGTTTCCCA-1",
        ]
        assert calls.rows.tolist() == [0, 1, 1]
        assert calls.features.tolist() == ["Non-Targeting-1", "ACE2-1", "DPP4-1"]
        assert calls.umis.tolist() == [3678, 619, 2869]


def test_call_file_without_rows_has_no_calls(tmp_path) -> None:
    calls = read_protospacer_calls(
        _write(
            tmp_path / "calls.csv", "cell_barcode,num_features,feature_call,num_umis\n"
        )
    )
    assert len(calls.barcodes) == 0
    assert len(calls.features) == 0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("A-1,1,g1\n", "lacks the columns 'num_umis'"),
        ("A-1,1,g1,3\nA-1,1,g2,4\n", "repeat on lines 3"),
        (" ,1,g1,3\n", "Empty cell barcodes on lines 2"),
        ("A-1,two,g1,3\n", "whole number of at least 0 on lines 2"),
        ("A-1,-1,g1,3\n", "whole number of at least 0"),
        ("A-1,2,g1|g2,3\n", "disagrees with the number"),
        ("A-1,1,g1,3.5\n", "UMI counts must be whole numbers"),
        ("A-1,1,g1,many\n", "UMI counts must be whole numbers"),
        ("A-1,2,g1|,3|4\n", "Empty feature IDs"),
        ("A-1,2,g1|g1,3|4\n", "called twice in one row on lines 2"),
        ("A-1,0,g1,3\n", "num_features 0 list calls"),
    ],
)
def test_malformed_call_files_are_refused(tmp_path, body, message) -> None:
    header = (
        "cell_barcode,num_features,feature_call\n"
        if "lacks" in message
        else "cell_barcode,num_features,feature_call,num_umis\n"
    )
    path = _write(tmp_path / "calls.csv", header + body)
    with pytest.raises(ValueError, match=message):
        read_protospacer_calls(path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # Blank lines are rows, so the bad UMI count on line 5 is not
        # reported on line 3.
        ("A-1,1,g1,5\n\n\nC-1,1,g1,x\n", "blank lines among its rows, on lines 3, 4;"),
        ("A-1,1,g1,5\n   \nC-1,1,g1,4\n", "blank lines among its rows, on lines 3;"),
        # A longer row neither shifts the columns nor loses its last field.
        (
            "A-1,1,g1,5,extra\n",
            "^Line 2 of the call file holds 5 fields, but its header names 4$",
        ),
        ("A-1,1,g1,5\nC-1,1,g1,5,extra\n", "^Line 3 of the call file holds 5 fields"),
        ("A-1,1,g1,5,\n", "^Line 2 of the call file holds 5 fields"),
        # A UMI count beyond uint32 is refused before it overflows.
        (
            "A-1,1,g1,99999999999999999999\n",
            "UMI counts must be at most 4294967295 on lines 2$",
        ),
        ("A-1,1,g1,4294967296\n", "UMI counts must be at most 4294967295 on lines 2$"),
        ('"A-1,1,g1,5\n', "^The call file cannot be read as CSV: "),
    ],
)
def test_call_file_messages_name_the_line_and_the_problem(
    tmp_path, body, message
) -> None:
    path = _write(
        tmp_path / "calls.csv",
        "cell_barcode,num_features,feature_call,num_umis\n" + body,
    )
    with pytest.raises(ValueError, match=message):
        read_protospacer_calls(path)


def test_call_file_reads_the_largest_counts_and_ends_without_blank_rows(
    tmp_path,
) -> None:
    header = "cell_barcode,num_features,feature_call,num_umis\n"
    path = _write(
        tmp_path / "calls.csv",
        header + "A-1,1,g1,4294967295\r\nC-1,2,g1|g2,3|4\r\n\r\n\r\n",
    )
    calls = read_protospacer_calls(path)
    assert calls.barcodes.tolist() == ["A-1", "C-1"]
    assert calls.umis.tolist() == [4294967295, 3, 4]
    # A file of blank lines after its header has no rows.
    assert (
        len(
            read_protospacer_calls(
                _write(tmp_path / "blank.csv", header + "\n\n")
            ).barcodes
        )
        == 0
    )


def test_a_num_features_beyond_any_row_disagrees_without_overflow(
    tmp_path, recwarn
) -> None:
    path = _write(
        tmp_path / "calls.csv",
        "cell_barcode,num_features,feature_call,num_umis\nA-1,99999999999999999999,g1,5\n",
    )
    with pytest.raises(ValueError, match="disagrees with the number"):
        read_protospacer_calls(path)
    assert not [entry for entry in recwarn if "cast" in str(entry.message)]


def test_a_gzip_call_file_is_read_whatever_its_name(tmp_path) -> None:
    text = "cell_barcode,num_features,feature_call,num_umis\nA-1,2,g1|g2,5|3\n"
    path = tmp_path / "calls.csv"
    path.write_bytes(gzip.compress(text.encode()))
    calls = read_protospacer_calls(path)
    assert calls.features.tolist() == ["g1", "g2"]
    assert protospacer_calls_size(path) == protospacer_calls_size(
        _write(tmp_path / "plain.csv", text)
    )


@pytest.mark.parametrize("scan_bytes", [3, 1 << 20])
def test_the_size_scan_bounds_the_rows_and_calls(
    tmp_path, monkeypatch, scan_bytes
) -> None:
    # Small reads split lines and runs of separators across reads.
    monkeypatch.setattr(cellranger, "_SCAN_BYTES", scan_bytes)
    rows = [
        "A-1,1,g1,5",
        "B-1,3,g1|g2|g3,5|6|7",
        "C-1,0,,",
        "D-1,2,g10|g20,1|2",
    ]
    text = "cell_barcode,num_features,feature_call,num_umis\n" + "\n".join(rows)
    for name, ending in (
        ("plain.csv", ""),
        ("ended.csv", "\n"),
        ("windows.csv", "\r\n"),
    ):
        body = text.replace("\n", ending or "\n") if ending == "\r\n" else text + ending
        path = _write(tmp_path / name, body)
        size = protospacer_calls_size(path)
        calls = read_protospacer_calls(path)
        assert size.n_bytes == len(body.encode())
        assert size.n_pipes == 2 * 2 + 2 * 1
        # The rows and calls of the file, or one more row when the last
        # line ends.
        assert size.n_rows in (len(calls.barcodes), len(calls.barcodes) + 1)
        assert size.n_calls >= len(calls.features)
        assert size.most_calls == int(np.bincount(calls.rows).max())


def test_call_file_frames_round_trip_through_pandas(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "cell_barcode": [f"BC{index}-1" for index in range(50)],
            "num_features": [1 + index % 3 for index in range(50)],
        }
    )
    frame["feature_call"] = [
        "|".join(f"g{offset}" for offset in range(count))
        for count in frame["num_features"]
    ]
    frame["num_umis"] = [
        "|".join(str(10 * (offset + 1)) for offset in range(count))
        for count in frame["num_features"]
    ]
    path = tmp_path / "calls.csv"
    frame.to_csv(path, index=False)
    calls = read_protospacer_calls(path)
    assert len(calls.features) == int(frame["num_features"].sum())
    assert (
        np.bincount(calls.rows, minlength=50).tolist() == frame["num_features"].tolist()
    )
