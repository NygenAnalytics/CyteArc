"""The states of recorder sites, the prefix rule and the character matrix.

Every expectation is derived by hand from the rows each test builds.
"""

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.recorders import (
    FIELDS,
    MARK,
    MISSING,
    NOT_APPLICABLE,
    STATES,
    UNEDITED,
    UNKNOWN_MARK,
    AlleleRows,
    RefusedTokens,
    SiteRules,
    allele_chunk,
    apply_prefix_rule,
    cell_summaries,
    character_matrix,
    character_name,
    chunk_states,
    site_state_counts,
)
from cytearc.metadata.records import RecordEncoder
from cytearc.utils.memory_budget import MemoryLedger


def test_states_and_fields_are_fixed():
    assert STATES == ("unedited", "mark", "unknown_mark", "missing", "not_applicable")
    assert (UNEDITED, MARK, UNKNOWN_MARK, MISSING, NOT_APPLICABLE) == (0, 1, 2, 3, 4)
    assert list(FIELDS) == [
        "integration",
        "site",
        "allele",
        "state",
        "umis",
        "reads",
        "row",
    ]


def test_rules_check_the_order_of_their_sets():
    rules = SiteRules(
        kind="independent",
        not_applicable=frozenset({""}),
        missing=frozenset({"-"}),
        unedited=frozenset({"*"}),
        unedited_at={"EMX1": frozenset({"CTTGGG"})},
        unedited_within="[None]",
        unknown=frozenset({"!"}),
        marks_at={"EMX1": frozenset({"GGACA"})},
        pattern="[ACGT]+",
    )
    expected = {
        "": NOT_APPLICABLE,
        "-": MISSING,
        "*": UNEDITED,
        "CTTGGG": UNEDITED,
        "AC[None]GT": UNEDITED,
        "!": UNKNOWN_MARK,
        "GGACA": MARK,
        # An insertion outside the lookup is an unknown mark.
        "GTCAG": UNKNOWN_MARK,
        # Text that the pattern does not match is refused.
        "NA": None,
    }
    for token, state in expected.items():
        assert rules.state_of(token, "EMX1") == state, token
    # CTTGGG is unedited at EMX1 only.
    assert rules.state_of("CTTGGG", "RNF2") == UNKNOWN_MARK


def test_rules_without_a_lookup_take_what_matches_as_a_mark():
    rules = SiteRules(kind="indel", unedited=frozenset({"[]"}))
    assert rules.state_of("23_268del", "") == MARK
    assert rules.state_of("anything", "") == MARK
    strict = SiteRules(kind="indel", pattern=r"\d+_\d+del")
    assert strict.state_of("23_268del", "") == MARK
    assert strict.state_of("23_268", "") is None
    # A lookup without a pattern refuses every token it does not name.
    labels = SiteRules(kind="independent", marks=frozenset({"1", "2"}))
    assert labels.state_of("2", "s") == MARK
    assert labels.state_of("9", "s") is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"kind": "tree"}, "kind must be one of"),
        (
            {"kind": "indel", "missing": frozenset({"x"}), "unknown": frozenset({"x"})},
            "cannot mean both missing and unknown",
        ),
        ({"kind": "indel", "pattern": "("}, "not a regular expression"),
        ({"kind": "indel", "absent": 9}, "absent must be the code"),
    ],
)
def test_rules_refuse_what_they_cannot_mean(options, message):
    with pytest.raises(ValueError, match=message):
        SiteRules(**options)


def test_classify_reads_each_distinct_token_once_and_none_as_absent():
    rules = SiteRules(kind="indel", unedited=frozenset({"[]"}), pattern=r"\d+del")
    states, alleles, refused = rules.classify(
        np.array(["[]", "3del", None, "3del", "x"], dtype=object), ""
    )
    assert states.tolist() == [UNEDITED, MARK, MISSING, MARK, MISSING]
    assert alleles.tolist() == [None, "3del", None, "3del", None]
    assert refused.tolist() == [False, False, False, False, True]
    absent = SiteRules(kind="indel", absent=NOT_APPLICABLE)
    assert absent.classify(np.array([None], dtype=object), "")[0].tolist() == [
        NOT_APPLICABLE
    ]


def test_the_prefix_rule_marks_gaps_missing_and_slots_beyond_not_applicable():
    states = np.array(
        [
            [MARK, MARK, UNEDITED, UNEDITED, UNEDITED],
            [UNEDITED, MARK, UNEDITED, UNEDITED, UNEDITED],
            [MARK, UNEDITED, MISSING, MARK, UNEDITED],
            [MARK, MARK, MARK, MARK, MARK],
            [UNEDITED, UNEDITED, UNEDITED, UNEDITED, UNEDITED],
        ],
        dtype=np.int8,
    )
    alleles = np.array([["A", "B", None, None, None]] * 5, dtype=object)
    lengths = np.array([5, 5, 5, 4, 2])
    n_beyond = apply_prefix_rule(states, alleles, lengths)
    assert states.tolist() == [
        [MARK, MARK, UNEDITED, UNEDITED, UNEDITED],
        [MISSING, MARK, UNEDITED, UNEDITED, UNEDITED],
        [MARK, MISSING, MISSING, MARK, UNEDITED],
        [MARK, MARK, MARK, MARK, NOT_APPLICABLE],
        [UNEDITED, UNEDITED, NOT_APPLICABLE, NOT_APPLICABLE, NOT_APPLICABLE],
    ]
    # The fourth row's fifth slot held an edit beyond its four slots.
    assert n_beyond == 1
    assert alleles[3, 4] is None and alleles[4, 2] is None


def _rows(**options) -> AlleleRows:
    defaults = {
        "barcodes": np.array(["a", "b"], dtype=object),
        "rows": np.array([2, 3]),
        "columns": ((None, "1"), (None, "2")),
        "tokens": np.array([["X", ""], ["", "Y"]], dtype=object),
        "integrations": np.array(["T1", "T2"], dtype=object),
    }
    return AlleleRows(**{**defaults, **options})


def test_allele_rows_must_fit_their_columns():
    with pytest.raises(ValueError, match="one token per row and column"):
        _rows(tokens=np.array([["X"]], dtype=object))
    with pytest.raises(ValueError, match="need columns that do"):
        _rows(integrations=None)
    with pytest.raises(ValueError, match="need columns that do not"):
        _rows(columns=(("T", "1"), (None, "2")))
    with pytest.raises(ValueError, match="one value of umis per row"):
        _rows(umis=np.array([1]))
    assert len(_rows()) == 2


def test_chunk_states_refuse_tokens_and_apply_lengths_by_integration():
    rules = SiteRules(kind="ordered", unedited=frozenset({""}), pattern="[A-Z]")
    classified = chunk_states(_rows(), rules, lengths={"T2": 1}, n_slots=2)
    assert classified.states.tolist() == [[MARK, UNEDITED], [UNEDITED, NOT_APPLICABLE]]
    assert classified.n_beyond == 1
    with pytest.raises(RefusedTokens) as refused:
        chunk_states(
            _rows(tokens=np.array([["x", ""], ["", "Y"]], dtype=object)),
            rules,
            n_slots=2,
        )
    assert refused.value.rows == (0,)
    assert refused.value.tokens == (("1", "x"),)
    with pytest.raises(ValueError, match="number of slots"):
        chunk_states(_rows(), rules)
    # A wide matrix's columns name integrations, whose lengths apply per run.
    wide = AlleleRows(
        barcodes=np.array(["a"], dtype=object),
        rows=np.array([2]),
        columns=(("T1", "1"), ("T1", "2"), ("T2", "1"), ("T2", "2")),
        tokens=np.array([["X", "Y", "X", "Y"]], dtype=object),
    )
    classified = chunk_states(wide, rules, lengths={"T2": 1}, n_slots=2)
    assert classified.states.tolist() == [[MARK, MARK, MARK, NOT_APPLICABLE]]


def _table(records: list[tuple]) -> object:
    """A record table of (cell, source, integration, site, allele, state, umis, reads, row) tuples."""
    frame = pd.DataFrame(
        records,
        columns=[
            "cell",
            "source",
            "integration",
            "site",
            "allele",
            "state",
            "umis",
            "reads",
            "row",
        ],
    )
    encoder = RecordEncoder(FIELDS, ledger=MemoryLedger(None, "encoding"))
    for source, part in frame.groupby("source", sort=True):
        encoder.add(
            part.drop(columns=["cell", "source"]).astype(
                {
                    "state": np.int64,
                    "umis": np.int64,
                    "reads": np.int64,
                    "row": np.int64,
                }
            ),
            part["cell"].to_numpy(dtype=np.int64),
            int(source),
        )
    return encoder.finish()


def test_allele_chunks_keep_rows_of_selected_cells_that_name_an_integration():
    rules = SiteRules(kind="indel")
    rows = _rows(
        integrations=np.array(["T1", None], dtype=object),
        tokens=np.array([["X", None], ["Y", "Z"]], dtype=object),
        umis=np.array([4, 5]),
    )
    classified = chunk_states(rows, rules)
    frame, cells = allele_chunk(rows, np.array([7, 3]), classified)
    # The second row names no integration.
    assert cells.tolist() == [7, 7]
    assert frame["integration"].tolist() == ["T1", "T1"]
    assert frame["site"].tolist() == ["1", "2"]
    assert frame["state"].tolist() == [MARK, MISSING]
    assert frame["umis"].tolist() == [4, 4]
    assert frame["reads"].tolist() == [-1, -1]
    assert frame["row"].tolist() == [2, 2]
    frame, cells = allele_chunk(rows, np.array([-1, 3]), classified)
    assert len(frame) == len(cells) == 0


def test_each_cell_counts_its_edits_integrations_read_and_records():
    table = _table(
        [
            (0, 0, "A", "1", "x", MARK, -1, -1, 2),
            (0, 0, "A", "2", None, UNEDITED, -1, -1, 2),
            (0, 0, "B", "1", None, MISSING, -1, -1, 3),
            (2, 0, "B", "1", "?", UNKNOWN_MARK, -1, -1, 4),
            (2, 0, "C", "1", None, NOT_APPLICABLE, -1, -1, 4),
        ]
    )
    summaries = cell_summaries(table, 3)
    assert summaries["n_edits"].tolist() == [1, 0, 1]
    # Cell 0 read A only, B was missing; cell 2 read B.
    assert summaries["n_integrations"].tolist() == [1, 0, 1]
    assert summaries["n_records"].tolist() == [3, 0, 2]


def test_site_states_count_each_cell_once_in_its_first_state():
    table = _table(
        [
            # Cell 0 holds two alleles at A-1: a mark wins over unedited.
            (0, 0, "A", "1", None, UNEDITED, 5, -1, 2),
            (0, 0, "A", "1", "x", MARK, 1, -1, 3),
            (1, 0, "A", "1", None, NOT_APPLICABLE, -1, -1, 4),
            # Cell 2 has no record at A-1, so it is missing there.
            (2, 0, "B", "1", None, UNEDITED, -1, -1, 5),
        ]
    )
    counts = site_state_counts(table)
    found = {
        (row.integration, row.site, row.state): row.n_cells
        for row in counts.itertuples()
        if row.n_cells
    }
    assert found == {
        ("A", "1", "mark"): 1,
        ("A", "1", "missing"): 1,
        ("A", "1", "not_applicable"): 1,
        ("B", "1", "unedited"): 1,
        ("B", "1", "missing"): 2,
    }
    assert counts["state"].tolist()[:5] == list(STATES)


def test_characters_are_named_as_cassiopeia_names_them():
    assert character_name("intID5", "RNF2") == "intID5-RNF2"
    assert character_name("CA", "") == "CA"


def test_the_matrix_keeps_each_integrations_most_abundant_allele():
    table = _table(
        [
            # Cell 0 has two alleles of A: the second has more UMIs.
            (0, 0, "A", "1", "x", MARK, 2, 10, 2),
            (0, 0, "A", "2", None, UNEDITED, 2, 10, 2),
            (0, 0, "A", "1", "y", MARK, 3, 5, 3),
            (0, 0, "A", "2", "z", MARK, 3, 5, 3),
            # Cell 1's two alleles tie on UMIs; more reads win.
            (1, 0, "A", "1", "x", MARK, 2, 10, 4),
            (1, 0, "A", "2", None, MISSING, 2, 10, 4),
            (1, 0, "A", "1", "y", MARK, 2, 30, 5),
            (1, 0, "A", "2", None, MISSING, 2, 30, 5),
            # Cell 2's tie on both; more sites read win, then file order.
            (2, 0, "A", "1", "x", MARK, -1, -1, 6),
            (2, 0, "A", "2", None, MISSING, -1, -1, 6),
            (2, 1, "A", "1", "y", MARK, -1, -1, 2),
            (2, 1, "A", "2", "z", MARK, -1, -1, 2),
        ]
    )
    matrix = character_matrix(table)
    assert matrix.characters == ("A-1", "A-2")
    states = matrix.states
    names = {(row.character, row.allele): row.state for row in states.itertuples()}
    # Every cell keeps its allele y, so x is no exported state.
    assert names == {("A-1", "y"): 1, ("A-2", "z"): 1}
    assert matrix.codes.tolist() == [[1, 1], [1, -1], [1, 1]]
    assert matrix.n_resolved == 3
    assert states["n_cells"].tolist() == [3, 2]


def test_whole_number_marks_keep_their_numbers_and_others_follow():
    table = _table(
        [
            (0, 0, "A", "s", "3", MARK, -1, -1, 2),
            (1, 0, "A", "s", "7", MARK, -1, -1, 3),
            (2, 0, "A", "s", "!", UNKNOWN_MARK, -1, -1, 4),
            (3, 0, "A", "s", "ACGT", MARK, -1, -1, 5),
            (4, 0, "A", "s", None, UNEDITED, -1, -1, 6),
            (5, 0, "B", "s", "ZZ", MARK, -1, -1, 7),
            (5, 0, "B", "t", "0", MARK, -1, -1, 7),
        ]
    )
    matrix = character_matrix(table)
    assert matrix.characters == ("A-s", "B-s", "B-t")
    column = dict(zip(matrix.cells.tolist(), matrix.codes[:, 0].tolist(), strict=True))
    # 3 and 7 keep their numbers; ACGT follows 7, then the unknown mark.
    assert column == {0: 3, 1: 7, 2: 9, 3: 8, 4: 0, 5: -1}
    # A mark written 0 is no whole number from 1, so it takes a state.
    assert matrix.codes[matrix.cells.tolist().index(5)].tolist() == [-1, 1, 1]
    # Written missing, cell 2's one edit leaves it no read site.
    missing = character_matrix(table, unknown="missing")
    assert 2 not in missing.cells.tolist()
    assert missing.n_without_states == 1
    with pytest.raises(ValueError, match="unknown must be one of"):
        character_matrix(table, unknown="drop")


def test_placeholders_of_unknown_marks_name_no_allele_and_export_missing():
    """PEtracer's "!" and CARLIN's "X" say an edit is unknown, not which one.

    Cells that hold a placeholder share no edit, so the export writes it as
    missing; an unknown mark whose text is its allele keeps a state of its
    own.
    """
    rules = SiteRules(
        kind="independent",
        unknown=frozenset({"!"}),
        marks_at=MappingProxyType({"s": frozenset({"ACAGT"})}),
        pattern=r"[ACGT]+",
    )
    states, alleles, refused = rules.classify(["!", "GTCAG", "ACAGT", "!"], "s")
    assert states.tolist() == [UNKNOWN_MARK, UNKNOWN_MARK, MARK, UNKNOWN_MARK]
    assert alleles.tolist() == [None, "GTCAG", "ACAGT", None]
    assert not refused.any()
    table = _table(
        [
            (0, 0, "A", "s", None, UNKNOWN_MARK, -1, -1, 2),
            (1, 0, "A", "s", "GTCAG", UNKNOWN_MARK, -1, -1, 3),
            (2, 0, "A", "s", "ACAGT", MARK, -1, -1, 4),
            (3, 0, "A", "s", None, UNKNOWN_MARK, -1, -1, 5),
            (3, 0, "A", "t", None, UNEDITED, -1, -1, 5),
        ]
    )
    matrix = character_matrix(table)
    codes = {
        cell: row.tolist()
        for cell, row in zip(matrix.cells.tolist(), matrix.codes, strict=True)
    }
    # Cell 0 reads only a placeholder, so it is left out; cell 3 keeps its
    # unedited site.
    assert codes == {1: [2, -1], 2: [1, -1], 3: [-1, 0]}
    assert matrix.states["allele"].tolist() == ["ACAGT", "GTCAG"]
    assert matrix.n_without_states == 1


def test_cells_and_characters_without_a_read_site_are_left_out():
    table = _table(
        [
            (0, 0, "A", "1", None, MISSING, -1, -1, 2),
            (0, 0, "B", "1", None, NOT_APPLICABLE, -1, -1, 2),
            (1, 0, "A", "1", None, UNEDITED, -1, -1, 3),
            (1, 0, "B", "1", None, MISSING, -1, -1, 3),
        ]
    )
    matrix = character_matrix(table)
    assert matrix.characters == ("A-1",)
    assert matrix.cells.tolist() == [1]
    assert matrix.codes.tolist() == [[0]]
    assert matrix.n_without_states == 1
    assert matrix.states.empty


def test_the_matrix_refuses_codes_beyond_the_budget():
    table = _table([(0, 0, "A", "1", "x", MARK, -1, -1, 2)])
    with pytest.raises(MemoryError, match="holds about 104 bytes"):
        character_matrix(table, memory_bytes=103, held_bytes=100)
    assert character_matrix(table, memory_bytes=104, held_bytes=100).codes.tolist() == [
        [1]
    ]


def test_priors_are_each_states_share_of_its_characters_edited_cells():
    table = _table(
        [
            (0, 0, "A", "1", "x", MARK, -1, -1, 2),
            (1, 0, "A", "1", "x", MARK, -1, -1, 3),
            (2, 0, "A", "1", "y", MARK, -1, -1, 4),
            (3, 0, "A", "1", None, UNEDITED, -1, -1, 5),
            (3, 0, "B", "1", "z", MARK, -1, -1, 5),
        ]
    )
    priors = character_matrix(table).priors()
    assert priors.to_dict("list") == {
        "character": [0, 0, 1],
        "state": [1, 2, 1],
        "probability": [2 / 3, 1 / 3, 1.0],
    }


def test_integer_keys_sort_as_their_columns_and_refuse_the_int64_range():
    from cytearc.clones.recorders import _integer_keys

    rng = np.random.default_rng(23)
    first, second, third = (rng.integers(0, bound, 500) for bound in (7, 3, 1_000))
    keys = _integer_keys((first, 7), (second, 3), (third, 1_000))
    # The keys order the rows as the columns do, and the bounds recover them.
    np.testing.assert_array_equal(
        np.argsort(keys, kind="stable"), np.lexsort((third, second, first))
    )
    np.testing.assert_array_equal(keys // 3_000, first)
    np.testing.assert_array_equal(keys // 1_000 % 3, second)
    np.testing.assert_array_equal(keys % 1_000, third)
    # 2**40 integrations by 2**23 cells would wrap past the int64 range.
    with pytest.raises(ValueError, match="too many to number as one integer key"):
        _integer_keys((np.zeros(1), 2**40), (np.zeros(1), 2**23))


def test_numbers_with_leading_zeros_are_text_and_states_fit_int32():
    table = _table(
        [
            (0, 0, "A", "s", "1", MARK, -1, -1, 2),
            (1, 0, "A", "s", "01", MARK, -1, -1, 3),
            (2, 0, "A", "s", "001", MARK, -1, -1, 4),
        ]
    )
    matrix = character_matrix(table)
    # 1 keeps its number; 01 and 001 are two other alleles that follow it.
    assert matrix.codes[:, 0].tolist() == [1, 3, 2]
    assert dict(zip(matrix.states["state"], matrix.states["allele"], strict=True)) == {
        1: "1",
        2: "001",
        3: "01",
    }
    # The largest number an int32 holds leaves no state for two more alleles.
    largest = str(2**31 - 2)
    table = _table(
        [
            (0, 0, "A", "s", largest, MARK, -1, -1, 2),
            (1, 0, "A", "s", "ACGT", MARK, -1, -1, 3),
            (2, 0, "A", "s", "GTCA", MARK, -1, -1, 4),
        ]
    )
    with pytest.raises(ValueError, match="pass 2147483647, the largest that an int32"):
        character_matrix(table)


def test_characters_that_two_integrations_name_alike_are_refused():
    # "a-b" at site "c" and "a" at site "b-c" both name the character a-b-c.
    table = _table(
        [
            (0, 0, "a-b", "c", "1", MARK, -1, -1, 2),
            (1, 0, "a", "b-c", "2", MARK, -1, -1, 3),
        ]
    )
    with pytest.raises(ValueError, match="are both the character 'a-b-c'"):
        character_matrix(table)
