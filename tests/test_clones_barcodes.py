"""Collapsing barcode rows into molecules and pairs, and summarising them.

``BarcodeCounter`` is checked against pandas' group-by of the same rows,
however they are split into chunks and however often the chunks merge, and
the summaries against counts derived by hand.
"""

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.barcodes import (
    FIELDS,
    BarcodeCounter,
    barcode_pairs,
    cell_summaries,
    library_summaries,
    lineage_barcode_frames,
    pair_keys,
    repeated_keys,
)
from cytearc.metadata.records import decode_text


def _random_reads(seed: int, n_rows: int) -> pd.DataFrame:
    """Reads that repeat a few hundred molecules many times, shuffled."""
    rng = np.random.default_rng(seed)
    molecules = pd.DataFrame(
        {
            "cell": rng.integers(0, 40, 400),
            "library": rng.choice(["lib_a", "lib_b"], 400),
            "barcode": rng.choice(["ACGT", "ACGA", "TTTT", "GGCC", "κ-tag"], 400),
            "umi": rng.choice([f"U{index:03d}" for index in range(60)], 400),
        }
    )
    return molecules.iloc[rng.integers(0, len(molecules), n_rows)].reset_index(
        drop=True
    )


def _table_frame(counted) -> pd.DataFrame:
    table = counted.table
    return pd.DataFrame(
        {
            "cell": table.cells,
            "library": np.asarray(counted.libraries, dtype=object)[table.sources],
            "barcode": decode_text(
                table.text["barcode"].codes, table.text["barcode"].categories
            ),
            "umi": pd.Series(
                decode_text(table.text["umi"].codes, table.text["umi"].categories),
                dtype=object,
            ),
            "umis": table.values["umis"],
            "reads": table.values["reads"],
        }
    )


def _count(rows: pd.DataFrame, chunk: int, merge_rows: int, **options):
    counter = BarcodeCounter(merge_rows=merge_rows)
    for start in range(0, len(rows), chunk):
        part = rows.iloc[start : start + chunk]
        counter.add(
            part["cell"].to_numpy(),
            part["library"].to_numpy(dtype=object),
            part["barcode"].to_numpy(dtype=object),
            umis=None if "umi" not in part else part["umi"].to_numpy(dtype=object),
            **{
                key: None if value is None else part[value].to_numpy()
                for key, value in options.items()
                if key in ("umi_counts", "reads")
            },
            rows_are_reads=options.get("rows_are_reads", False),
        )
    return counter, counter.finish()


@pytest.mark.parametrize(("chunk", "merge_rows"), [(17, 1), (250, 40), (5_000, 10**9)])
def test_reads_collapse_into_molecules_whatever_the_chunks_and_merges(
    chunk, merge_rows
) -> None:
    reads = _random_reads(1, 5_000)

    _counter, counted = _count(reads, chunk, merge_rows, rows_are_reads=True)

    expected = (
        reads.groupby(["cell", "library", "barcode", "umi"])
        .size()
        .rename("reads")
        .reset_index()
        .sort_values(["cell", "library", "barcode", "umi"])
        .reset_index(drop=True)
    )
    observed = _table_frame(counted)
    assert observed[["cell", "library", "barcode", "umi"]].values.tolist() == (
        expected[["cell", "library", "barcode", "umi"]].values.tolist()
    )
    assert observed["reads"].tolist() == expected["reads"].tolist()
    assert (observed["umis"] == 1).all()
    assert counted.libraries == ("lib_a", "lib_b")
    assert counted.levels == ("molecules", "molecules")
    assert counted.table.kinds == FIELDS


def test_the_records_and_their_codes_do_not_depend_on_row_order() -> None:
    reads = _random_reads(2, 3_000)
    _counter, first = _count(reads, 300, 100, rows_are_reads=True)
    _counter, second = _count(
        reads.sample(frac=1, random_state=3), 77, 10**9, rows_are_reads=True
    )
    for name in ("barcode", "umi"):
        np.testing.assert_array_equal(
            first.table.text[name].codes, second.table.text[name].codes
        )
        np.testing.assert_array_equal(
            first.table.text[name].data, second.table.text[name].data
        )
        categories = first.table.text[name].categories.tolist()
        assert categories == sorted(categories)
    np.testing.assert_array_equal(first.table.cells, second.table.cells)
    np.testing.assert_array_equal(first.table.sources, second.table.sources)
    np.testing.assert_array_equal(
        first.table.values["reads"], second.table.values["reads"]
    )


def test_merging_bounds_the_rows_held_by_the_records() -> None:
    # Ten thousand reads of 400 molecules: rows wait until they outnumber the
    # records held, so the counter holds at most about twice the molecules.
    reads = _random_reads(4, 10_000)
    counter = BarcodeCounter(merge_rows=200)
    held = []
    for start in range(0, len(reads), 50):
        part = reads.iloc[start : start + 50]
        counter.add(
            part["cell"].to_numpy(),
            part["library"].to_numpy(dtype=object),
            part["barcode"].to_numpy(dtype=object),
            umis=part["umi"].to_numpy(dtype=object),
            rows_are_reads=True,
        )
        held.append(counter.n_rows)
        assert counter.peak_bytes > counter.nbytes > 0
    n_molecules = len(reads.drop_duplicates())
    assert max(held) <= 2 * n_molecules + 50 < len(reads) / 10
    assert len(counter.finish().table.cells) == n_molecules


def test_molecule_rows_sum_their_reads_and_pairs_their_umis() -> None:
    counter = BarcodeCounter()
    # Two rows of one molecule, one with unknown reads.
    counter.add(
        [0, 0, 0, 1],
        ["m"] * 4,
        ["A", "A", "A", "A"],
        umis=["u1", "u1", "u2", "u1"],
        reads=[3, 4, 5, -1],
    )
    counter.add([1], ["m"], ["A"], umis=["u1"], reads=[2])
    # Pairs: UMIs sum, and an unknown count makes the sum unknown.
    counter.add([0, 0, 2, 2], ["p"] * 4, ["B", "B", "C", "C"], umi_counts=[2, 3, 1, -1])
    counter.add([2], ["p"], ["D"])
    counted = counter.finish()
    frame = _table_frame(counted)
    assert frame.values.tolist() == [
        [0, "m", "A", "u1", 1, 7],
        [0, "m", "A", "u2", 1, 5],
        [0, "p", "B", None, 5, -1],
        [1, "m", "A", "u1", 1, -1],
        [2, "p", "C", None, -1, -1],
        [2, "p", "D", None, -1, -1],
    ]
    assert counted.levels == ("molecules", "pairs")


def test_reads_without_umis_count_pairs_by_their_reads() -> None:
    counter = BarcodeCounter()
    counter.add([1, 1, 1, 0], ["atac"] * 4, ["X", "X", "Y", "X"], rows_are_reads=True)
    frame = _table_frame(counter.finish())
    assert frame[["cell", "barcode", "umis", "reads"]].values.tolist() == [
        [0, "X", -1, 1],
        [1, "X", -1, 2],
        [1, "Y", -1, 1],
    ]


def test_a_library_holds_molecules_or_pairs_but_not_both() -> None:
    counter = BarcodeCounter()
    counter.add([0], ["lib"], ["A"], umis=["u"])
    with pytest.raises(ValueError, match="'lib' holds molecules in one file"):
        counter.add([0], ["lib"], ["A"], umi_counts=[3])
    with pytest.raises(ValueError, match="'lib' holds molecules in one file"):
        counter.add_library("lib", molecules=False)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"cells": [0, 1]}, "one cell, one library and one barcode"),
        ({"cells": [-1]}, "position among the cells"),
        ({"barcodes": [None]}, "barcode is missing"),
        ({"umis": [None]}, "UMI is missing"),
        ({"umis": ["a", "b"]}, "one UMI"),
        ({"umis": ["a"], "umi_counts": [1]}, "which count no UMIs"),
        ({"rows_are_reads": True, "reads": [1]}, "count their reads by rows"),
        ({"reads": [-2]}, "read counts must be at least 0"),
        ({"umi_counts": [1, 2]}, "one of the UMI counts"),
        ({"libraries": [None]}, "needs a library name"),
    ],
)
def test_the_counter_refuses_rows_it_cannot_count(arguments, message) -> None:
    call = {"cells": [0], "libraries": ["lib"], "barcodes": ["A"], **arguments}
    counter = BarcodeCounter()
    with pytest.raises(ValueError, match=message):
        counter.add(
            call.pop("cells"), call.pop("libraries"), call.pop("barcodes"), **call
        )


def test_the_counter_is_used_once_and_needs_a_positive_merge() -> None:
    with pytest.raises(ValueError, match="merge_rows must be at least 1"):
        BarcodeCounter(merge_rows=0)
    counter = BarcodeCounter()
    counter.finish()
    with pytest.raises(RuntimeError, match="has finished"):
        counter.finish()
    with pytest.raises(RuntimeError, match="has finished"):
        counter.add([0], ["lib"], ["A"])


def test_a_counter_without_rows_lists_its_libraries_and_no_record() -> None:
    counter = BarcodeCounter()
    counter.add_library("empty", molecules=True)
    counter.add([], [], [], umis=[], reads=[])
    counted = counter.finish()
    assert counted.table.n_records == 0
    assert counted.libraries == ("empty",)
    assert library_summaries(counted) == [
        {
            "library": "empty",
            "level": "molecules",
            "records": 0,
            "has_umis": False,
            "has_reads": False,
        }
    ]
    summaries = cell_summaries(counted.table, 3)
    assert {name: values.tolist() for name, values in summaries.items()} == {
        "n_barcodes": [0, 0, 0],
        "n_umis": [0, 0, 0],
        "n_reads": [0, 0, 0],
        "n_records": [0, 0, 0],
    }
    pairs = barcode_pairs(counted.table)
    assert len(pairs.cells) == 0


def _example():
    counter = BarcodeCounter()
    # Cell 0: barcode A in two molecules, B in one; cell 2: A as a pair of 4
    # UMIs and B as a pair without counts; cell 1 has nothing.
    counter.add(
        [0, 0, 0], ["rna"] * 3, ["A", "A", "B"], umis=["x", "y", "x"], reads=[2, 3, 1]
    )
    counter.add([2, 2], ["counts"] * 2, ["A", "B"], umi_counts=[4, -1], reads=[9, -1])
    return counter.finish()


def test_cells_count_their_distinct_barcodes_umis_and_reads() -> None:
    counted = _example()
    summaries = cell_summaries(counted.table, 3)
    assert {name: values.tolist() for name, values in summaries.items()} == {
        "n_barcodes": [2, 0, 2],
        "n_umis": [3, 0, -1],
        "n_reads": [6, 0, -1],
        "n_records": [3, 0, 2],
    }
    assert summaries["n_barcodes"].dtype == np.int32
    assert library_summaries(counted) == [
        {
            "library": "counts",
            "level": "pairs",
            "records": 2,
            "has_umis": False,
            "has_reads": False,
        },
        {
            "library": "rna",
            "level": "molecules",
            "records": 3,
            "has_umis": True,
            "has_reads": True,
        },
    ]


def test_records_collapse_into_pairs_of_a_cell_and_a_barcode() -> None:
    pairs = barcode_pairs(_example().table)
    assert pairs.cells.tolist() == [0, 0, 2, 2]
    assert pairs.sources.tolist() == [1, 1, 0, 0]
    assert pairs.barcodes.tolist() == [0, 1, 0, 1]
    assert pairs.umis.tolist() == [2, 1, 4, -1]
    assert pairs.reads.tolist() == [5, 1, 9, -1]
    assert pairs.molecules.tolist() == [2, 1, 0, 0]


def test_the_record_tables_name_cells_libraries_and_unknown_counts() -> None:
    counted = _example()
    summaries = cell_summaries(counted.table, 3)
    loaded = lineage_barcode_frames(
        ["c0", "c1", "c2"], summaries, counted.table, library_summaries(counted)
    )
    assert loaded.cells.index.tolist() == ["c0", "c1", "c2"]
    assert loaded.cells.index.name == "ids"
    assert loaded.cells["n_umis"].tolist() == [3, 0, -1]
    records = loaded.records
    assert records["cell"].astype(str).tolist() == ["c0", "c0", "c0", "c2", "c2"]
    assert list(records["cell"].cat.categories) == ["c0", "c1", "c2"]
    assert records["library"].astype(str).tolist() == ["rna"] * 3 + ["counts"] * 2
    assert records["umi"].isna().tolist() == [False, False, False, True, True]
    assert str(records["umis"].dtype) == "Int64"
    assert records["umis"].isna().tolist() == [False, False, False, False, True]
    assert records["reads"].fillna(-1).tolist() == [2, 3, 1, 9, -1]
    assert loaded.libraries.to_dict("list") == {
        "library": ["counts", "rna"],
        "level": ["pairs", "molecules"],
        "records": [2, 3],
        "has_umis": [False, True],
        "has_reads": [False, True],
    }
    pairs = loaded.pairs()
    assert pairs["barcode"].astype(str).tolist() == ["A", "B", "A", "B"]
    assert pairs["umis"].fillna(-1).tolist() == [2, 1, 4, -1]
    assert pairs["molecules"].tolist() == [2, 1, 0, 0]
    assert pairs["cell"].cat.categories.tolist() == ["c0", "c1", "c2"]


def test_long_and_unicode_barcodes_survive_merges() -> None:
    long_barcode = "ACGT" * 2_500
    counter = BarcodeCounter(merge_rows=1)
    for _ in range(3):
        counter.add([0, 1], ["l", "l"], [long_barcode, "κύτταρο"], rows_are_reads=True)
    frame = _table_frame(counter.finish())
    assert frame["barcode"].tolist() == [long_barcode, "κύτταρο"]
    assert frame["reads"].tolist() == [3, 3]


def test_pairs_of_texts_key_apart_and_repeats_count_once_per_copy() -> None:
    keys = pair_keys(["AAAC", "AAAC", "CCCG", "GG"], ["TT", "TA", "TT", "AAAC"])
    # Each pair keys apart, a pair and its swap included.
    assert len({tuple(key) for key in keys.tolist()}) == 4
    assert (pair_keys(["GG"], ["AAAC"]) != pair_keys(["AAAC"], ["GG"])).any()
    again = pair_keys(["AAAC", "AAAC", "AAAC"], ["TT", "TT", "GC"])
    assert again[0].tolist() == keys[0].tolist()
    # The two copies of AAAC and TT repeat the first; AAAC and GC is new.
    assert repeated_keys([keys, again]) == 2
    assert repeated_keys([keys]) == 0
    assert repeated_keys([]) == 0


def test_finding_repeated_keys_holds_the_bytes_the_import_counts() -> None:
    import tracemalloc

    from cytearc.datastore._operations import lineage_barcodes as operations

    n_keys = 200_000
    cells = np.asarray(
        [f"C{index % 5_000:05d}" for index in range(n_keys)], dtype=object
    )
    umis = np.asarray([f"U{index:08d}" for index in range(n_keys)], dtype=object)
    tracemalloc.start()
    try:
        keys = [
            pair_keys(cells[start : start + 20_000], umis[start : start + 20_000])
            for start in range(0, n_keys, 20_000)
        ]
        tracemalloc.reset_peak()
        assert repeated_keys(keys) == 0
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The keys themselves and what finding their repeats adds.
    assert peak <= operations._KEY_BYTES * n_keys


def test_adding_many_small_chunks_costs_time_linear_in_the_chunks(monkeypatch) -> None:
    from cytearc.clones import barcodes as barcodes_module

    lengths = []
    length = barcodes_module._Rows.__len__

    def counted(self) -> int:
        lengths.append(1)
        return length(self)

    monkeypatch.setattr(barcodes_module._Rows, "__len__", counted)
    counter = BarcodeCounter(merge_rows=10**9)
    n_chunks = 2_000
    for chunk in range(n_chunks):
        counter.add([chunk % 7], ["lib"], [f"B{chunk % 5}"], umi_counts=[1])
        counter.peak_bytes
    # Walking the waiting chunks on each call took about 6 million lengths.
    assert len(lengths) < 20 * n_chunks
    assert counter.n_rows == n_chunks


def test_the_counters_totals_follow_the_rows_it_holds() -> None:
    rng = np.random.default_rng(11)
    counter = BarcodeCounter(merge_rows=40)
    for _chunk in range(60):
        size = int(rng.integers(1, 30))
        counter.add(
            rng.integers(0, 20, size),
            ["lib"] * size,
            [f"B{value}" for value in rng.integers(0, 15, size)],
            umis=[f"U{value}" for value in rng.integers(0, 9, size)],
            rows_are_reads=True,
        )
        parts = ([] if counter._held is None else [counter._held]) + counter._waiting
        n_rows = sum(len(part) for part in parts)
        n_text = sum(len(part.barcode_values) + len(part.umi_values) for part in parts)
        assert counter.n_rows == n_rows
        assert counter.nbytes == sum(part.nbytes for part in parts)
        assert counter.peak_bytes == counter.nbytes + 110 * n_rows + 48 * n_text
