import numpy as np
import pandas as pd
import pytest

from cytearc.repertoire import contigs as contigs_module
from cytearc.repertoire.contigs import (
    FIELDS,
    ContigEncoder,
    chunk_rows_for,
    decode_ragged,
    decode_text,
    encode_ragged,
    encode_text,
    iter_present,
    parse_flags,
    parse_integers,
    parse_numbers,
    parse_text,
)
from tests.perturbation_helpers import use_python_kernels


def test_text_parsing_turns_missing_tokens_into_none() -> None:
    parsed = parse_text([" TRBV1 ", "", "None", "NA", "nan", "null", None, "x"])
    assert parsed.tolist() == ["TRBV1", None, None, None, None, None, None, "x"]
    assert parsed.dtype == object


def test_flags_accept_every_spelling_of_true_and_false() -> None:
    flags = parse_flags(
        [
            "T",
            "F",
            "true",
            "false",
            "True",
            "False",
            "TRUE",
            "FALSE",
            "1",
            "0",
            "",
            "None",
        ],
        name="productive",
    )
    assert flags.tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, -1, -1]
    assert flags.dtype == np.int8


def test_unknown_flags_are_refused_with_their_lines() -> None:
    with pytest.raises(
        ValueError, match=r"productive must be true or false.*'maybe' on lines 4, 6"
    ):
        parse_flags(["T", "F", "maybe", "T", "maybe"], name="productive", first_line=2)


def test_integers_parse_whole_numbers_and_mark_missing_values() -> None:
    parsed = parse_integers(["3", "", "0", " 12 ", "None", "4.0"], name="umi_count")
    assert parsed.tolist() == [3, -1, 0, 12, -1, 4]
    assert parsed.dtype == np.int64


@pytest.mark.parametrize("value", ["-1", "2.5", "many", "inf"])
def test_integers_refuse_values_that_are_not_counts(value) -> None:
    with pytest.raises(ValueError, match="umi_count must be a whole number"):
        parse_integers(["1", value], name="umi_count", first_line=10)


def test_integers_refuse_values_beyond_int64() -> None:
    with pytest.raises(ValueError, match="too large"):
        parse_integers(["1", str(2**64)], name="umi_count")


def test_numbers_parse_reals_and_refuse_non_finite_values() -> None:
    parsed = parse_numbers(["0.95", "", "1"], name="v_identity")
    np.testing.assert_array_equal(parsed, [0.95, np.nan, 1.0])
    with pytest.raises(ValueError, match="v_identity must be a finite number.*line 3"):
        parse_numbers(["0.9", "inf"], name="v_identity")
    with pytest.raises(ValueError, match="'x'"):
        parse_numbers(["x"], name="v_identity")


def test_line_numbers_name_at_most_five_lines() -> None:
    with pytest.raises(ValueError, match=r"lines 2, 3, 4, 5, 6 and 2 more"):
        parse_flags(["?"] * 7, name="is_cell")


def test_text_encoding_sorts_distinct_values_and_marks_missing() -> None:
    encoded = encode_text(["b", None, "a", "b", float("nan")])
    assert encoded.categories.tolist() == ["a", "b"]
    assert encoded.codes.tolist() == [1, -1, 0, 1, -1]
    assert encoded.codes.dtype == np.int32
    assert decode_text(encoded.codes, encoded.categories).tolist() == [
        "b",
        None,
        "a",
        "b",
        None,
    ]
    empty = encode_text([None, None])
    assert empty.codes.tolist() == [-1, -1] and len(empty.categories) == 0


def test_ragged_encoding_round_trips_and_tells_missing_from_values() -> None:
    values = ["ACGT", None, "", "TTGCA", None]
    data, offsets = encode_ragged(values)
    # The empty string is a value of length 0, as distinct from a missing one.
    assert offsets.tolist() == [0, 4, -1, 4, 9, -1]
    assert bytes(data) == b"ACGTTTGCA"
    assert decode_ragged(data, offsets).tolist() == ["ACGT", None, "", "TTGCA", None]
    data, offsets = encode_ragged(["αβ"])
    assert decode_ragged(data, offsets).tolist() == ["αβ"]


def _chunk(rows: dict[str, list]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: pd.Series(values, dtype=object)
            if FIELDS[name] in ("text", "ragged")
            else np.asarray(values)
            for name, values in rows.items()
        }
    )


@pytest.mark.parametrize("python_kernels", [False, True])
def test_encoder_orders_contigs_by_cell_source_and_row(
    monkeypatch: pytest.MonkeyPatch, python_kernels: bool
) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, contigs_module)
    encoder = ContigEncoder()
    encoder.add(
        _chunk(
            {
                "sequence_id": ["s1_c1", "s1_c2", "s1_c3"],
                "locus": ["TRB", "TRA", None],
                "junction_aa": ["CASSF", "CAVF", "CAVF"],
                "productive": np.asarray([1, 1, 0], dtype=np.int8),
                "umi_count": np.asarray([5, -1, 2], dtype=np.int64),
                "sequence": ["AAA", None, "GGGG"],
            }
        ),
        np.asarray([2, 0, 2]),
        source=1,
    )
    # The second chunk lacks umi_count and sequence and adds v_identity.
    encoder.add(
        _chunk(
            {
                "sequence_id": ["s0_c1"],
                "locus": ["IGH"],
                "junction_aa": ["CARW"],
                "productive": np.asarray([1], dtype=np.int8),
                "v_identity": np.asarray([0.97]),
            }
        ),
        np.asarray([2]),
        source=0,
    )
    assert encoder.n_contigs == 4
    assert encoder.nbytes > 0
    table = encoder.finish()
    # Cell 0 first, then cell 2 with source 0 before source 1, file rows kept.
    assert table.cells.tolist() == [0, 2, 2, 2]
    assert table.sources.tolist() == [1, 0, 1, 1]
    assert table.fields == (
        "sequence_id",
        "locus",
        "junction_aa",
        "productive",
        "umi_count",
        "v_identity",
        "sequence",
    )
    assert table.column("sequence_id").tolist() == ["s1_c2", "s0_c1", "s1_c1", "s1_c3"]
    assert table.column("locus").tolist() == ["TRA", "IGH", "TRB", None]
    # Codes follow the sorted distinct values.
    assert table.text["junction_aa"].categories.tolist() == ["CARW", "CASSF", "CAVF"]
    assert table.text["junction_aa"].codes.tolist() == [2, 0, 1, 2]
    assert table.column("umi_count").tolist() == [-1, -1, 5, 2]
    np.testing.assert_array_equal(
        table.column("v_identity"), [np.nan, 0.97, np.nan, np.nan]
    )
    assert table.column("sequence").tolist() == [None, None, "AAA", "GGGG"]
    assert table.column("productive").tolist() == [1, 1, 1, 0]
    assert table.n_contigs == 4
    with pytest.raises(KeyError, match="No contig field 'c_call'"):
        table.column("c_call")


def test_encoder_without_chunks_gives_an_empty_table() -> None:
    table = ContigEncoder().finish()
    assert table.n_contigs == 0
    assert table.fields == ()


def test_encoder_needs_one_cell_per_contig() -> None:
    encoder = ContigEncoder()
    with pytest.raises(ValueError, match="one cell per contig"):
        encoder.add(_chunk({"sequence_id": ["a", "b"]}), np.asarray([0]), source=0)


def test_encoder_reuses_codes_of_text_seen_in_earlier_chunks() -> None:
    encoder = ContigEncoder()
    for cells, values in (([0, 1], ["CAVF", "CASS"]), ([2, 3], ["CASS", "CAVF"])):
        encoder.add(
            _chunk({"junction_aa": values}),
            np.asarray(cells),
            source=0,
        )
    table = encoder.finish()
    assert table.text["junction_aa"].categories.tolist() == ["CASS", "CAVF"]
    assert table.text["junction_aa"].codes.tolist() == [1, 0, 0, 1]


def test_chunk_rows_follow_the_budget_within_bounds() -> None:
    assert chunk_rows_for(10**3, columns=50) == 10_000
    assert chunk_rows_for(10**12, columns=50) == 50_000
    assert chunk_rows_for(20 * 1_000 * 10 * 20_000, columns=10) == 20_000


def test_repeated_and_missing_ids_are_found_within_each_source(monkeypatch) -> None:
    encoder = ContigEncoder()
    assert encoder.repeated_ids() == []
    encoder.add(
        _chunk({"sequence_id": ["a", "b", None]}), np.asarray([0, 1, 2]), source=0
    )
    encoder.add(_chunk({"sequence_id": ["c", "a"]}), np.asarray([3, 4]), source=0)
    encoder.add(_chunk({"sequence_id": ["a", "d"]}), np.asarray([0, 1]), source=1)
    assert encoder.missing_ids == 1
    assert encoder.repeated_ids() == [(0, "a")]
    clean = ContigEncoder()
    clean.add(_chunk({"sequence_id": ["a", "b"]}), np.asarray([0, 1]), source=0)
    clean.add(_chunk({"sequence_id": ["a"]}), np.asarray([2]), source=1)
    assert clean.repeated_ids() == []


def test_hash_collisions_are_not_reported_as_repeats(monkeypatch) -> None:
    # Every ID hashes to the same value, so only the text tells them apart.
    monkeypatch.setattr(
        contigs_module.pd.util,
        "hash_array",
        lambda values: np.zeros(len(values), dtype=np.uint64),
    )
    encoder = ContigEncoder()
    encoder.add(_chunk({"sequence_id": ["a", "b", "c", "b"]}), np.arange(4), source=0)
    assert encoder.repeated_ids() == [(0, "b")]
    other = ContigEncoder()
    other.add(_chunk({"sequence_id": ["a", "b"]}), np.arange(2), source=0)
    assert other.repeated_ids() == []


def test_text_of_every_missing_value_merges_to_no_categories() -> None:
    encoder = ContigEncoder()
    encoder.add(_chunk({"c_call": [None, None]}), np.asarray([0, 1]), source=0)
    encoder.add(_chunk({"c_call": ["IGHM"]}), np.asarray([2]), source=0)
    table = encoder.finish()
    assert table.text["c_call"].codes.tolist() == [-1, -1, 0]
    assert table.text["c_call"].categories.tolist() == ["IGHM"]


def test_present_fields_follow_the_storage_order() -> None:
    assert iter_present(["junction_aa", "cell_id", "locus", "other"]) == (
        "locus",
        "junction_aa",
    )


def test_a_text_field_missing_everywhere_has_no_categories() -> None:
    encoder = ContigEncoder()
    encoder.add(_chunk({"d_call": [None, None]}), np.asarray([0, 1]), source=0)
    table = encoder.finish()
    assert table.fields == ("d_call",)
    assert table.text["d_call"].codes.tolist() == [-1, -1]
    assert len(table.text["d_call"].categories) == 0


def test_packed_text_round_trips_ascii_and_other_text() -> None:
    from cytearc.repertoire.contigs import pack_text, unpack_text

    for values in (["CASSF", "", "TRBV1*01"], ["αβ", "γ", "CAS"], []):
        data, offsets = pack_text(values)
        assert unpack_text(data, offsets).tolist() == values
    data, offsets = pack_text(["αβ", "x"])
    assert offsets.tolist() == [0, 4, 5]


def test_an_encoder_finishes_once() -> None:
    encoder = ContigEncoder()
    encoder.add(_chunk({"junction_aa": ["CASS"]}), np.asarray([0]), source=0)
    encoder.finish()
    with pytest.raises(RuntimeError, match="has finished"):
        encoder.finish()
    with pytest.raises(RuntimeError, match="has finished"):
        encoder.add(_chunk({"junction_aa": ["CASS"]}), np.asarray([0]), source=0)


def test_fields_of_some_sources_are_missing_for_the_others() -> None:
    encoder = ContigEncoder()
    encoder.add(
        _chunk(
            {
                "sequence_id": ["a"],
                "c_call": ["IGHM"],
                "sequence": ["ACGT"],
                "umi_count": np.asarray([3], dtype=np.int64),
            }
        ),
        np.asarray([1]),
        source=0,
    )
    other = _chunk({"sequence_id": ["b"]})
    other["note"] = ["a column outside the AIRR fields is ignored"]
    encoder.add(other, np.asarray([0]), source=1)
    assert encoder.nbytes > 0
    table = encoder.finish()
    assert table.column("sequence_id").tolist() == ["b", "a"]
    assert table.column("c_call").tolist() == [None, "IGHM"]
    assert table.column("sequence").tolist() == [None, "ACGT"]
    assert table.column("umi_count").tolist() == [-1, 3]


def test_some_values_of_packed_text_are_decoded_alone() -> None:
    from cytearc.repertoire.contigs import pack_text, take_text

    data, offsets = pack_text(["CASSF", "CAVF", "CARW", "αβ"])
    assert take_text(data, offsets, np.asarray([3, 0, 0])).tolist() == [
        "αβ",
        "CASSF",
        "CASSF",
    ]
    assert take_text(data, offsets, np.asarray([], dtype=np.int64)).tolist() == []
