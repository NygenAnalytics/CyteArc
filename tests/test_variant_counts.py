"""The site tables, merges and per-band rules of variant counts."""

import numpy as np
import pytest

from cytearc.variants.counts import (
    CALL_DTYPE,
    DEPTH_DTYPE,
    MAX_COUNT,
    Contig,
    SiteTable,
    count_dtype,
    finish_band,
    merge_sites,
)


def _coordinate_table(
    positions: list[int],
    alleles: list[list[tuple[str, str]]],
    *,
    contig: Contig = Contig("chrM", 100, "a" * 64),
) -> SiteTable:
    allele_site, texts, roles = [], [], []
    for site, site_alleles in enumerate(alleles):
        for text, role in site_alleles:
            allele_site.append(site)
            texts.append(text)
            roles.append(role)
    return SiteTable(
        contigs=(contig,),
        site_contig=np.zeros(len(positions)),
        site_position=np.asarray(positions),
        site_name=np.full(len(positions), "", dtype=object),
        site_channel=np.full(len(positions), "", dtype=object),
        allele_site=np.asarray(allele_site),
        allele=np.asarray(texts, dtype=object),
        allele_role=np.asarray(roles, dtype=object),
    )


def _named_table(
    names: list[str], alleles: list[list[tuple[str, str]]], channels=None
) -> SiteTable:
    allele_site, texts, roles = [], [], []
    for site, site_alleles in enumerate(alleles):
        for text, role in site_alleles:
            allele_site.append(site)
            texts.append(text)
            roles.append(role)
    return SiteTable(
        contigs=(),
        site_contig=np.full(len(names), -1),
        site_position=np.full(len(names), -1),
        site_name=np.asarray(names, dtype=object),
        site_channel=np.asarray(channels or [""] * len(names), dtype=object),
        allele_site=np.asarray(allele_site),
        allele=np.asarray(texts, dtype=object),
        allele_role=np.asarray(roles, dtype=object),
    )


REF_ALT = [("A", "reference"), ("G", "alternative")]


def test_a_site_table_names_its_sites_and_finds_their_alleles() -> None:
    table = _coordinate_table(
        [310, 3],
        [REF_ALT, [("C", "reference"), ("T", "alternative"), ("other", "other")]],
    )
    assert table.labels().tolist() == ["chrM:310", "chrM:3"]
    assert table.allele_offsets().tolist() == [0, 2, 5]
    assert table.reference_alleles().tolist() == [0, 2]
    named = _named_table(
        ["BCOR", "BCOR"], [REF_ALT, REF_ALT], channels=["geneseq", "probebc"]
    )
    assert named.labels().tolist() == ["BCOR (geneseq)", "BCOR (probebc)"]
    unnamed = _coordinate_table([5], [REF_ALT], contig=Contig(""))
    assert unnamed.label(0) == "5"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"site_name": np.asarray(["x"], dtype=object)}, "coordinates or a name"),
        (
            {"site_contig": np.asarray([3])},
            "names a contig that the table does not hold",
        ),
        ({"site_position": np.asarray([0])}, "position must be 1 or more"),
        (
            {"allele_site": np.asarray([0, 1])},
            "names a site that the table does not hold",
        ),
        (
            {"allele_role": np.asarray(["reference", "mutant"], dtype=object)},
            "Allele roles must be among",
        ),
        (
            {"allele": np.asarray(["A", "A"], dtype=object)},
            "The allele 'A' of the site chrM:7 is listed twice",
        ),
        ({"site_channel": np.asarray(["", ""], dtype=object)}, "sites must align"),
        (
            {"allele_role": np.asarray(["reference"], dtype=object)},
            "alleles must align",
        ),
    ],
)
def test_a_site_table_refuses_inconsistent_arrays(change, message) -> None:
    arrays = {
        "contigs": (Contig("chrM"),),
        "site_contig": np.asarray([0]),
        "site_position": np.asarray([7]),
        "site_name": np.asarray([""], dtype=object),
        "site_channel": np.asarray([""], dtype=object),
        "allele_site": np.asarray([0, 0]),
        "allele": np.asarray(["A", "G"], dtype=object),
        "allele_role": np.asarray(["reference", "alternative"], dtype=object),
        **change,
    }
    with pytest.raises(ValueError, match=message):
        SiteTable(**arrays)


def test_a_site_table_refuses_repeated_sites_text_types_and_two_references() -> None:
    with pytest.raises(ValueError, match="The site chrM:7 is listed twice"):
        _coordinate_table([7, 7], [REF_ALT, REF_ALT])
    with pytest.raises(TypeError, match="alleles must hold text"):
        SiteTable(
            contigs=(Contig("chrM"),),
            site_contig=np.asarray([0]),
            site_position=np.asarray([7]),
            site_name=np.asarray([""], dtype=object),
            site_channel=np.asarray([""], dtype=object),
            allele_site=np.asarray([0]),
            allele=np.asarray([1], dtype=object),
            allele_role=np.asarray(["reference"], dtype=object),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        _named_table(["x"], [REF_ALT]).__class__(
            contigs=(),
            site_contig=np.full(1, -1),
            site_position=np.full(1, -1),
            site_name=np.asarray([["x"]], dtype=object),
            site_channel=np.asarray([""], dtype=object),
            allele_site=np.asarray([], dtype=int),
            allele=np.asarray([], dtype=object),
            allele_role=np.asarray([], dtype=object),
        )
    table = _coordinate_table([7], [[("A", "reference"), ("G", "reference")]])
    with pytest.raises(ValueError, match="has two reference alleles"):
        table.reference_alleles()
    shuffled = SiteTable(
        contigs=(Contig("chrM"),),
        site_contig=np.zeros(2),
        site_position=np.asarray([7, 8]),
        site_name=np.asarray(["", ""], dtype=object),
        site_channel=np.asarray(["", ""], dtype=object),
        allele_site=np.asarray([1, 0]),
        allele=np.asarray(["A", "A"], dtype=object),
        allele_role=np.asarray(["reference", "reference"], dtype=object),
    )
    with pytest.raises(ValueError, match="not sorted by site"):
        shuffled.allele_offsets()


def test_merge_orders_contigs_naturally_then_positions_then_names() -> None:
    first = SiteTable(
        contigs=(Contig("chr10"), Contig("chr2")),
        site_contig=np.asarray([0, 1, 1]),
        site_position=np.asarray([5, 9, 3]),
        site_name=np.full(3, "", dtype=object),
        site_channel=np.full(3, "", dtype=object),
        allele_site=np.asarray([0, 1, 2]),
        allele=np.asarray(["T", "G", "C"], dtype=object),
        allele_role=np.full(3, "reference", dtype=object),
    )
    merged, site_maps, allele_maps = merge_sites([first], open_alleles=False)
    assert merged.labels().tolist() == ["chr2:3", "chr2:9", "chr10:5"]
    assert site_maps[0].tolist() == [2, 1, 0]
    assert allele_maps[0].tolist() == [2, 1, 0]
    named = _named_table(
        ["T10", "T2", "T2"], [REF_ALT] * 3, channels=["", "probebc", "geneseq"]
    )
    merged, site_maps, _ = merge_sites([named], open_alleles=False)
    assert merged.labels().tolist() == ["T2 (geneseq)", "T2 (probebc)", "T10"]


def test_merge_shares_sites_and_keeps_the_alleles_of_each_in_text_order() -> None:
    first = _coordinate_table(
        [7, 9], [[("G", "alternative"), ("A", "reference")], REF_ALT]
    )
    second = _coordinate_table([9, 11], [REF_ALT, REF_ALT])
    merged, site_maps, allele_maps = merge_sites([first, second], open_alleles=False)
    assert merged.labels().tolist() == ["chrM:7", "chrM:9", "chrM:11"]
    assert site_maps[0].tolist() == [0, 1] and site_maps[1].tolist() == [1, 2]
    assert merged.allele.tolist() == ["A", "G"] * 3
    assert allele_maps[0].tolist() == [1, 0, 2, 3]
    assert allele_maps[1].tolist() == [2, 3, 4, 5]


def test_merge_refuses_conflicting_roles_alleles_and_references() -> None:
    first = _coordinate_table([7], [REF_ALT])
    swapped = _coordinate_table([7], [[("A", "alternative"), ("G", "reference")]])
    with pytest.raises(
        ValueError,
        match="call the allele 'A' of the site chrM:7 reference and alternative",
    ):
        merge_sites([first, swapped], open_alleles=False)
    other = _coordinate_table([7], [[("A", "reference"), ("T", "alternative")]])
    with pytest.raises(
        ValueError,
        match=r"give the site chrM:7 different alleles, \{A, G\} and \{A, T\}",
    ):
        merge_sites([first, other], open_alleles=False)
    merged, _, _ = merge_sites([first, other], open_alleles=True)
    assert merged.allele.tolist() == ["A", "G", "T"]
    longer = _coordinate_table([7], [REF_ALT], contig=Contig("chrM", 101))
    with pytest.raises(ValueError, match="with different references"):
        merge_sites([first, longer], open_alleles=False)
    other_bases = _coordinate_table(
        [7], [REF_ALT], contig=Contig("chrM", 100, "b" * 64)
    )
    with pytest.raises(ValueError, match="with different references"):
        merge_sites([first, other_bases], open_alleles=False)
    unnamed = _coordinate_table([7], [REF_ALT], contig=Contig(""))
    with pytest.raises(ValueError, match="name their contig and others do not"):
        merge_sites([first, unnamed], open_alleles=False)
    with pytest.raises(ValueError, match="No site table"):
        merge_sites([], open_alleles=False)
    # A contig that one file describes and another does not combines.
    bare = _coordinate_table([7], [REF_ALT], contig=Contig("chrM"))
    merged, _, _ = merge_sites([bare, first], open_alleles=False)
    assert merged.contigs == (Contig("chrM", 100, "a" * 64),)


def test_merge_refuses_one_sequence_under_two_contig_names() -> None:
    # mgatk names the contig after the reference file, so libraries whose
    # files carry sample prefixes would count one genome as two contigs.
    first = _coordinate_table([7], [REF_ALT], contig=Contig("d2_chrM", 100, "a" * 64))
    second = _coordinate_table([7], [REF_ALT], contig=Contig("d4_chrM", 100, "a" * 64))
    with pytest.raises(
        ValueError,
        match="one reference sequence under the contig names 'd2_chrM' and 'd4_chrM'",
    ):
        merge_sites([second, first], open_alleles=False)
    # Contigs of other sequences, or without one, keep their names.
    other = _coordinate_table([7], [REF_ALT], contig=Contig("chr1", 100, "b" * 64))
    bare = _coordinate_table([7], [REF_ALT], contig=Contig("MT", 100))
    merged, site_maps, _ = merge_sites([first, other, bare], open_alleles=False)
    assert {contig.name for contig in merged.contigs} == {"chr1", "d2_chrM", "MT"}
    assert merged.n_sites == 3 and [len(sites) for sites in site_maps] == [1, 1, 1]


def test_count_rows_take_the_fields_their_source_has() -> None:
    assert count_dtype(()).names == ("cell", "key", "count")
    assert count_dtype(("reverse", "forward")).names == (
        "cell",
        "key",
        "count",
        "forward",
        "reverse",
    )
    with pytest.raises(ValueError, match="Count fields must be among"):
        count_dtype(("strand",))


def _rows(dtype, rows):
    array = np.zeros(len(rows), dtype=dtype)
    for position, row in enumerate(rows):
        for name, value in zip(dtype.names, row, strict=False):
            array[name][position] = value
    return array


def _name(cell: int) -> str:
    return f"cell{cell}"


def test_without_stated_depth_a_site_depth_sums_its_counts() -> None:
    sites = _coordinate_table([7, 9], [REF_ALT, REF_ALT])
    counts = _rows(count_dtype(()), [(0, 0, 3), (0, 1, 2), (0, 3, 0), (2, 2, 5)])
    depth, stored = finish_band(
        counts,
        np.empty(0, DEPTH_DTYPE),
        sites=sites,
        depth_roles=None,
        reference_from_depth=False,
        name_cell=_name,
    )
    assert depth.tolist() == [(0, 0, 5), (2, 1, 5)]
    # The zero count is dropped.
    assert stored[["cell", "key", "count"]].tolist() == [
        (0, 0, 3),
        (0, 1, 2),
        (2, 2, 5),
    ]
    huge = _rows(count_dtype(()), [(0, 0, MAX_COUNT), (0, 1, 1)])
    with pytest.raises(
        ValueError, match="cell 'cell0' at the site chrM:7 add up to more than"
    ):
        finish_band(
            huge,
            np.empty(0, DEPTH_DTYPE),
            sites=sites,
            depth_roles=None,
            reference_from_depth=False,
            name_cell=_name,
        )


def test_a_stated_depth_derives_the_reference_and_adds_other_bases() -> None:
    # cellsnp-lite: DP counts the reference and alternative, OTH adds to it.
    sites = _coordinate_table(
        [7, 9], [[*REF_ALT, ("other", "other")], [*REF_ALT, ("other", "other")]]
    )
    counts = _rows(
        count_dtype(()),
        [(0, 1, 2), (0, 2, 1), (0, 5, 4), (1, 4, 3)],
    )
    stated = _rows(DEPTH_DTYPE, [(0, 0, 10), (1, 1, 3)])
    depth, stored = finish_band(
        counts,
        stated,
        sites=sites,
        depth_roles=frozenset({"reference", "alternative"}),
        reference_from_depth=True,
        name_cell=_name,
    )
    # Cell 0: DP 10 at chrM:7 with AD 2 and OTH 1 gives REF 8 and depth 11;
    # at chrM:9 it has OTH 4 without DP, so depth 4. Cell 1: DP 3 = ALT 3.
    assert depth.tolist() == [(0, 0, 11), (0, 1, 4), (1, 1, 3)]
    assert stored[["cell", "key", "count"]].tolist() == [
        (0, 0, 8),
        (0, 1, 2),
        (0, 2, 1),
        (0, 5, 4),
        (1, 4, 3),
    ]


def test_a_stated_depth_refuses_counts_it_cannot_hold() -> None:
    sites = _coordinate_table([7], [REF_ALT])
    roles = frozenset({"reference", "alternative"})
    over = _rows(count_dtype(()), [(0, 1, 5)])
    with pytest.raises(ValueError, match="add up to 5, more than its depth 4"):
        finish_band(
            over,
            _rows(DEPTH_DTYPE, [(0, 0, 4)]),
            sites=sites,
            depth_roles=roles,
            reference_from_depth=False,
            name_cell=_name,
        )
    with pytest.raises(
        ValueError, match="counted where the stated depth is 0: cell 'cell0'"
    ):
        finish_band(
            over,
            np.empty(0, DEPTH_DTYPE),
            sites=sites,
            depth_roles=roles,
            reference_from_depth=False,
            name_cell=_name,
        )
    counted_reference = _rows(count_dtype(()), [(0, 0, 1)])
    with pytest.raises(ValueError, match="counted and also derived"):
        finish_band(
            counted_reference,
            _rows(DEPTH_DTYPE, [(0, 0, 4)]),
            sites=sites,
            depth_roles=roles,
            reference_from_depth=True,
            name_cell=_name,
        )
    no_reference = _named_table(["T"], [[("MUT", "alternative")]])
    with pytest.raises(ValueError, match="The site T has no reference allele"):
        finish_band(
            np.empty(0, count_dtype(())),
            _rows(DEPTH_DTYPE, [(0, 0, 4)]),
            sites=no_reference,
            depth_roles=roles,
            reference_from_depth=True,
            name_cell=_name,
        )
    with pytest.raises(ValueError, match="A depth adds up to more than"):
        finish_band(
            _rows(count_dtype(()), [(0, 1, 1)]),
            _rows(DEPTH_DTYPE, [(0, 0, MAX_COUNT)]),
            sites=_coordinate_table([7], [[("A", "reference"), ("other", "other")]]),
            depth_roles=frozenset({"reference"}),
            reference_from_depth=False,
            name_cell=_name,
        )


def test_derived_reference_counts_have_no_quality() -> None:
    sites = _coordinate_table([7], [REF_ALT])
    dtype = count_dtype(("quality_forward",))
    counts = _rows(dtype, [(0, 1, 1, 30.0)])
    _depth, stored = finish_band(
        counts,
        _rows(DEPTH_DTYPE, [(0, 0, 3)]),
        sites=sites,
        depth_roles=frozenset({"reference", "alternative"}),
        reference_from_depth=True,
        name_cell=_name,
    )
    assert stored["count"].tolist() == [2, 1]
    assert (
        np.isnan(stored["quality_forward"][0]) and stored["quality_forward"][1] == 30.0
    )


def test_calls_rows_hold_codes() -> None:
    assert CALL_DTYPE.names == ("cell", "key", "source", "value")


def test_merge_names_named_sites_in_its_errors() -> None:
    first = _named_table(
        ["BCOR"], [[("WT", "reference"), ("MUT", "alternative")]], channels=["geneseq"]
    )
    second = _named_table(
        ["BCOR"],
        [[("WT", "alternative"), ("MUT", "alternative")]],
        channels=["geneseq"],
    )
    with pytest.raises(
        ValueError, match=r"of the site BCOR \(geneseq\) reference and alternative"
    ):
        merge_sites([first, second], open_alleles=False)
    unnamed = _coordinate_table([7], [REF_ALT], contig=Contig(""))
    other = _coordinate_table(
        [7], [[("A", "alternative"), ("G", "alternative")]], contig=Contig("")
    )
    with pytest.raises(ValueError, match="of the site 7 reference and alternative"):
        merge_sites([unnamed, other], open_alleles=False)
