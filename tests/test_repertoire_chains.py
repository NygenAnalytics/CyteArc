import itertools

import numpy as np
import pandas as pd
import pytest

from cytearc.repertoire.chains import (
    CHAIN_PAIRINGS,
    RECEPTOR_SUBTYPES,
    RECEPTOR_TYPES,
    ChainIndex,
    call_locus,
    chain_arms,
    index_chains,
    infer_loci,
    receptor_quality,
)
from cytearc.repertoire.contigs import FIELDS, ContigEncoder, EncodedText, encode_text


def _table(rows: list[dict], cells: list[int]):
    """Encode contig rows of one source, given as dicts of AIRR fields."""
    columns = sorted({name for row in rows for name in row})
    frame = {}
    for name in columns:
        values = [row.get(name) for row in rows]
        kind = FIELDS[name]
        if kind in ("text", "ragged"):
            frame[name] = pd.Series(values, dtype=object)
        elif kind == "flag":
            frame[name] = np.asarray(
                [-1 if value is None else int(value) for value in values], dtype=np.int8
            )
        else:
            frame[name] = np.asarray(
                [-1 if value is None else value for value in values], dtype=np.int64
            )
    encoder = ContigEncoder()
    encoder.add(pd.DataFrame(frame), np.asarray(cells), source=0)
    return encoder.finish()


def _chain(locus: str, junction_aa: str | None = "CASSF", **fields) -> dict:
    row = {
        "sequence_id": fields.pop("sequence_id", f"{locus}-{junction_aa}"),
        "locus": locus,
        "junction_aa": junction_aa,
        "productive": fields.pop("productive", True),
    }
    row.update(fields)
    return row


@pytest.mark.parametrize(
    ("call", "locus"),
    [
        ("TRBV20-1*01", "TRB"),
        ("trac", "TRA"),
        ("IGHV3-30,IGHV3-30-5", "IGH"),
        ("TRAV29/DV5", ""),
        ("TRAV14/DV4*02", ""),
        ("TRAJ29", "TRA"),
        ("TRDV2", "TRD"),
        ("XYZ1", None),
        ("", None),
    ],
)
def test_each_call_names_the_locus_of_its_first_letters(call, locus) -> None:
    assert call_locus(call) == locus


def test_loci_are_inferred_when_every_call_agrees() -> None:
    calls = {
        "v_call": np.asarray(
            ["TRBV1", "TRAV29/DV5", "TRAV1", None, "TRBV2", "BAD1"], dtype=object
        ),
        "j_call": np.asarray(
            ["TRBJ1", "TRAJ1", "TRBJ1", None, None, "TRBJ1"], dtype=object
        ),
        "c_call": np.asarray(["TRBC1", None, None, "IGHM", None, None], dtype=object),
    }
    assert infer_loci(calls).tolist() == [
        "TRB",  # every call names TRB
        "TRA",  # the shared alpha and delta V gene takes no part
        None,  # alpha V and beta J disagree
        "IGH",  # one call is enough
        "TRB",
        None,  # a call that names no locus prevents inference
    ]
    assert infer_loci(
        {"v_call": np.asarray(["TRAV29/DV5"], dtype=object)}
    ).tolist() == [None]
    assert infer_loci({}).tolist() == []
    with pytest.raises(ValueError, match="one value per contig"):
        infer_loci({"v_call": np.asarray(["TRBV1"]), "j_call": np.asarray(["a", "b"])})


def test_arms_follow_the_loci() -> None:
    locus = encode_text(
        ["TRA", "TRB", None, "Multi", "IGL", "IGH", "TRG", "TRD", "IGK"]
    )
    assert chain_arms(locus).tolist() == [0, 1, -1, -1, 0, 1, 0, 1, 0]


def test_chains_are_ranked_by_umis_then_reads_then_sequences() -> None:
    rows = [
        _chain("TRA", "CAVA", umi_count=3, consensus_count=10, sequence_id="a1"),
        _chain("TRA", "CAVB", umi_count=5, consensus_count=1, sequence_id="a2"),
        _chain("TRB", "CASB", umi_count=2, consensus_count=50, sequence_id="b1"),
        _chain("TRB", "CASA", umi_count=2, consensus_count=60, sequence_id="b2"),
        # Equal counts: the junction later in sort order ranks first.
        _chain("TRB", "CASZ", umi_count=1, consensus_count=1, sequence_id="b3"),
        _chain("TRB", "CASC", umi_count=1, consensus_count=1, sequence_id="b4"),
    ]
    table = _table(rows, [0] * len(rows))
    index = index_chains(table, 1, locus=table.text["locus"])
    ids = table.column("sequence_id")
    assert [ids[position] for position in index.slots[0]] == ["a2", "a1", "b2", "b1"]
    assert index.n_vj.tolist() == [2] and index.n_vdj.tolist() == [4]
    assert index.multichain.tolist() == [True]
    assert index.has_receptor.tolist() == [True]


def test_missing_counts_rank_as_zero_and_duplicate_counts_follow_umis() -> None:
    rows = [
        _chain("TRB", "CASA", umi_count=None, duplicate_count=9, sequence_id="b1"),
        _chain("TRB", "CASB", umi_count=1, duplicate_count=0, sequence_id="b2"),
        _chain("TRA", "CAVA", umi_count=None, duplicate_count=2, sequence_id="a1"),
        _chain("TRA", "CAVB", umi_count=None, duplicate_count=7, sequence_id="a2"),
    ]
    table = _table(rows, [0, 0, 0, 0])
    index = index_chains(table, 1, locus=table.text["locus"])
    ids = table.column("sequence_id")
    assert [ids[position] for position in index.slots[0]] == ["a2", "a1", "b2", "b1"]


def test_full_ties_rank_by_file_order() -> None:
    rows = [
        _chain("TRB", "CASA", junction="TGT", umi_count=1, sequence_id="z"),
        _chain("TRB", "CASA", junction="TGT", umi_count=1, sequence_id="a"),
    ]
    table = _table(rows, [0, 0])
    index = index_chains(table, 1, locus=table.text["locus"])
    assert table.column("sequence_id")[index.slots[0, 2]] == "z"


def test_contigs_that_are_not_chains_fill_no_slot() -> None:
    rows = [
        _chain("TRB", "CASA", productive=False, sequence_id="nonproductive"),
        _chain("TRB", None, sequence_id="no-junction"),
        _chain("Multi", "CASB", sequence_id="no-locus"),
        _chain("TRB", "CASC", is_cell=False, sequence_id="outside"),
        _chain("TRB", "CASD", high_confidence=False, sequence_id="low"),
        _chain("TRB", "CASE", productive=None, sequence_id="unknown"),
        _chain("TRA", "CAVA", is_cell=None, high_confidence=None, sequence_id="kept"),
    ]
    table = _table(rows, [0] * len(rows))
    index = index_chains(table, 2, locus=table.text["locus"])
    ids = table.column("sequence_id")
    assert [ids[position] for position in index.eligible.nonzero()[0]] == ["kept"]
    assert index.slots.tolist() == [[ids.tolist().index("kept"), -1, -1, -1], [-1] * 4]
    assert index.n_vj.tolist() == [1, 0] and index.n_vdj.tolist() == [0, 0]
    assert index.has_receptor.tolist() == [True, False]
    # Without the productive filter, non-productive contigs and those of an
    # unknown status count; flagged contigs still do not.
    loose = index_chains(table, 2, locus=table.text["locus"], productive_only=False)
    assert sorted(ids[position] for position in loose.eligible.nonzero()[0]) == [
        "kept",
        "nonproductive",
        "unknown",
    ]


def test_without_a_productive_field_no_contig_is_productive() -> None:
    encoder = ContigEncoder()
    encoder.add(
        pd.DataFrame(
            {
                "sequence_id": pd.Series(["a"], dtype=object),
                "locus": pd.Series(["TRB"], dtype=object),
                "junction_aa": pd.Series(["CASS"], dtype=object),
            }
        ),
        np.asarray([0]),
        source=0,
    )
    table = encoder.finish()
    assert not index_chains(table, 1, locus=table.text["locus"]).eligible.any()
    assert index_chains(
        table, 1, locus=table.text["locus"], productive_only=False
    ).eligible.all()


def test_cells_without_contigs_have_empty_slots() -> None:
    table = ContigEncoder().finish()
    locus = EncodedText.from_categories(np.empty(0, np.int32), [])
    index = index_chains(table, 3, locus=locus)
    assert index.slots.shape == (3, 4) and (index.slots == -1).all()
    quality = receptor_quality(index, locus)
    assert quality.receptor_type.tolist() == ["no IR"] * 3


_VJ = ("TRA", "TRG", "IGK", "IGL")
_VDJ = ("TRB", "TRD", "IGH")


def _arm_options(loci: tuple[str, ...]) -> list[tuple[str | None, str | None]]:
    return (
        [(None, None)]
        + [(locus, None) for locus in loci]
        + list(itertools.product(loci, loci))
    )


def _expected(vj: tuple, vdj: tuple, multichain: bool) -> tuple[str, str, str]:
    """scirpy's chain_qc rules, transcribed cell by cell from its documentation."""
    present = {locus for locus in (*vj, *vdj) if locus is not None}
    if not present:
        return "no IR", "no IR", "no IR"
    if multichain:
        return "multichain", "multichain", "multichain"
    tr = present & {"TRA", "TRB", "TRG", "TRD"}
    ig = present & {"IGH", "IGK", "IGL"}
    receptor_type = "TCR" if tr and not ig else "BCR" if ig and not tr else "ambiguous"
    if present <= {"TRA", "TRB"}:
        subtype = "TRA+TRB"
    elif present <= {"TRG", "TRD"}:
        subtype = "TRG+TRD"
    elif present == {"IGH"}:
        subtype = "IGH"
    elif present == {"IGH", "IGK", "IGL"}:
        subtype = "IGH+IGK/L"
    elif "IGK" in present and present <= {"IGH", "IGK"}:
        subtype = "IGH+IGK"
    elif "IGL" in present and present <= {"IGH", "IGL"}:
        subtype = "IGH+IGL"
    else:
        subtype = "ambiguous"
    if subtype == "ambiguous":
        pairing = "ambiguous"
    elif vj[0] is not None and vdj[0] is not None:
        extra_vj, extra_vdj = vj[1] is not None, vdj[1] is not None
        pairing = (
            "two full chains"
            if extra_vj and extra_vdj
            else "extra VJ"
            if extra_vj
            else "extra VDJ"
            if extra_vdj
            else "single pair"
        )
    elif vdj[0] is not None:
        pairing = "orphan VDJ"
    else:
        pairing = "orphan VJ"
    return receptor_type, subtype, pairing


def test_every_combination_of_slot_loci_gets_scirpy_categories() -> None:
    cases = [
        (vj, vdj, multichain)
        for vj in _arm_options(_VJ)
        for vdj in _arm_options(_VDJ)
        for multichain in (False, True)
        if not (multichain and vj[0] is None and vdj[0] is None)
    ]
    loci = list(_VJ + _VDJ)
    locus = EncodedText.from_categories(np.arange(len(loci)), loci)
    slots = np.full((len(cases), 4), -1, dtype=np.int64)
    n_vj = np.zeros(len(cases), dtype=np.int32)
    n_vdj = np.zeros(len(cases), dtype=np.int32)
    for row, (vj, vdj, multichain) in enumerate(cases):
        for column, value in enumerate((*vj, *vdj)):
            if value is not None:
                slots[row, column] = loci.index(value)
        n_vj[row] = sum(value is not None for value in vj)
        n_vdj[row] = sum(value is not None for value in vdj)
        if multichain:
            if n_vj[row]:
                n_vj[row] = 3
            else:
                n_vdj[row] = 3
    index = ChainIndex(slots=slots, n_vj=n_vj, n_vdj=n_vdj, eligible=np.ones(9, bool))
    quality = receptor_quality(index, locus)
    expected = [_expected(*case) for case in cases]
    observed = list(
        zip(
            quality.receptor_type.tolist(),
            quality.receptor_subtype.tolist(),
            quality.chain_pairing.tolist(),
            strict=True,
        )
    )
    assert observed == expected
    assert set(quality.receptor_type) == set(RECEPTOR_TYPES)
    assert set(quality.receptor_subtype) == set(RECEPTOR_SUBTYPES)
    assert set(quality.chain_pairing) == set(CHAIN_PAIRINGS)
