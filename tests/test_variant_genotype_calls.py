"""Genotype calls of imported variant counts, saved and loaded from a store.

The synthetic truth of ``tests/variant_truth.py`` fixes each cell's GoT UMIs,
GoT calls and GIFT UMIs; the expected calls are derived from those numbers
by the published rules, written out here, never by the code under test.
"""

import gzip
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.storage.artifacts import artifact_group
from cytearc.utils.logging import collect_warnings
from cytearc.variants.genotypes import CALLS, GENOTYPE_RULES
from tests.clone_helpers import every_cell, levenshtein_distance, write_cells_store
from tests.storage_helpers import copy_store, open_store
from tests.memory_helpers import smallest_budget, traced_peak
from tests.variant_helpers import write_variant_store
from tests.variant_truth import (
    PROBES,
    TARGETS,
    variant_truth,
    write_giftwrap_h5,
    write_got,
    write_got_multi,
    write_gotcha,
)

TRUTH = variant_truth()


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("variant_genotypes") / "store.zarr"
    write_variant_store(path, TRUTH)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


def _import(store, paths, **options):
    with collect_warnings(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return store.variants.import_counts(
            paths, cell_selection=every_cell(store), **options
        )


def _got(store, folder: Path, style: str):
    return _import(
        store,
        [
            write_got(TRUTH, folder / f"{name}.tsv", style=style, target=target)
            for target, name in enumerate(TARGETS)
        ],
    )


def _calls(store, genotypes) -> dict[tuple[str, str], tuple[str, str]]:
    calls = store.variants.load_genotypes(genotypes).calls
    return {
        (str(row.ids), str(row.site)): (str(row.call), str(row.label))
        for row in calls.itertuples(index=False)
    }


def test_source_calls_are_the_tools_own_with_their_labels(store, tmp_path) -> None:
    counts = _got(store, tmp_path, "geo_total")
    genotypes = store.variants.genotypes(counts, method="source")
    found = _calls(store, genotypes)
    named = {"WT": "wild_type", "MUT": "alternative_present"}
    expected = {}
    for row, cell in enumerate(TRUTH.row_ids[: TRUTH.n_cells]):
        for target, site in enumerate(TARGETS):
            label = TRUTH.got_call(row, target)
            if label != "NA":
                expected[(cell, site)] = (named[label], label)
    assert found == expected
    loaded = store.variants.load_genotypes(genotypes)
    assert loaded.method == "source"
    assert loaded.sites["source"].tolist() == ["Genotype", "Genotype"]
    assert loaded.sites.index.tolist() == list(TARGETS)
    # Each cell counts its mutant sites and its called ones.
    mutant = {cell: 0 for cell in loaded.cells.index}
    called = dict(mutant)
    for (cell, _site), (call, _label) in expected.items():
        mutant[cell] += call == "alternative_present"
        called[cell] += 1
    assert loaded.cells["n_mutant_sites"].to_dict() == mutant
    assert loaded.cells["n_called_sites"].to_dict() == called
    values = store.artifacts.load_values(genotypes)
    np.testing.assert_array_equal(values.values, list(mutant.values()))


def _source_counts(store, folder: Path, source: str):
    """Counts of each GoT tool whose calls method="source" adopts."""
    if source == "got":
        return _got(store, folder, "geo_total")
    if source == "got_multi":
        return _import(store, write_got_multi(TRUTH, folder / "ironthrone_out_pp.csv"))
    return _import(store, write_gotcha(TRUTH, folder / "gotcha.csv"))


@pytest.mark.parametrize(
    ("source", "min_quality"),
    [("got", 30), ("got", 0), ("got_multi", 30), ("gotcha", 30)],
)
def test_calls_without_a_quality_refuse_min_quality(
    store, tmp_path, source, min_quality
) -> None:
    # The GoT tools' calls carry no quality, so no call could pass any
    # min_quality: the rule is refused before anything is saved, as
    # informative refuses a test that the counts cannot make, rather than
    # filtering every call.
    counts = _source_counts(store, tmp_path, source)
    with pytest.raises(
        ValueError,
        match="min_quality tests call_quality, which these counts cannot give: "
        f"their '{source}' calls carry no quality",
    ):
        store.variants.genotypes(counts, method="source", min_quality=min_quality)
    assert store.artifacts.list(kind="variant_genotypes", scope="datastore") == []
    # Their tables count UMIs or reads, so min_depth filters their calls.
    loaded = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="source", min_depth=1)
    )
    assert "wild_type" in set(loaded.calls["call"].astype(str))


@pytest.mark.parametrize("source", ["calls_alone", "refined_alone"])
def test_calls_without_depth_refuse_min_depth(store, tmp_path, source) -> None:
    # A GoT table of calls alone, and GoT-Multi's refined table imported
    # without the stage table, give no cell a depth at their sites, so every
    # call would fail any min_depth.
    if source == "calls_alone":
        counts = _import(
            store, write_got(TRUTH, tmp_path / "CALR.tsv", style="geo_call_only")
        )
        site = "CALR"
    else:
        counts = _import(
            store,
            write_got_multi(
                TRUTH, tmp_path / "got_multi_out_refined.csv", refined=True
            ),
        )
        site = TARGETS[0]
    adopted = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="source")
    )
    assert len(adopted.calls) and set(adopted.calls["call"].astype(str)) != {"filtered"}
    with pytest.raises(
        ValueError,
        match="min_depth tests each cell's depth at the site, which these counts "
        f"cannot give at .* such as '{site}': no cell holds depth there",
    ):
        store.variants.genotypes(
            counts, method="source", min_depth=1, invalidate_cache=True
        )
    assert len(store.artifacts.list(kind="variant_genotypes", scope="datastore")) == 1


def test_got_multi_adopts_its_merged_call_and_its_refined_one(store, tmp_path) -> None:
    stage = _import(store, write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv"))
    loaded = store.variants.load_genotypes(
        store.variants.genotypes(stage, method="source")
    )
    # The target holds the merged call and the channels' calls, in the order
    # a cell's call is taken; its channels hold counts alone.
    assert loaded.sites.index.tolist() == list(TARGETS)
    assert set(loaded.sites["source"]) == {
        "genotype_merged, genotype_geneseq, genotype_probebc"
    }
    refined = _import(
        store,
        write_got_multi(TRUTH, tmp_path / "got_multi_out_refined.csv", refined=True),
    )
    found = _calls(store, store.variants.genotypes(refined, method="source"))
    # Every fourth row's mutant call the refinement rejected is filtered.
    for row, cell in enumerate(TRUTH.row_ids[: TRUTH.n_cells]):
        for target, site in enumerate(TARGETS):
            label = TRUTH.got_call(row, target)
            if label == "NA":
                assert (cell, site) not in found
                continue
            if label == "MUT" and row % 4 == 0:
                assert found[(cell, site)] == ("filtered", "FalsePositive")
            else:
                call = "wild_type" if label == "WT" else "alternative_present"
                assert found[(cell, site)] == (call, label)


def test_cells_without_a_refined_call_keep_their_merged_one(store, tmp_path) -> None:
    # The refined table holds the even rows alone, as GoT-Multi's model
    # refines the cells it can judge.
    stage = write_got_multi(TRUTH, tmp_path / "ironthrone_out_pp.csv")
    refined = write_got_multi(
        TRUTH, tmp_path / "got_multi_out_refined.csv", refined=True
    )
    rows = {cell: row for row, cell in enumerate(TRUTH.row_ids)}
    lines = refined.read_text().splitlines()
    kept = [lines[0]] + [
        line for line in lines[1:] if rows[line.split(",")[0]] % 2 == 0
    ]
    refined.write_text("\n".join(kept) + "\n")
    genotypes = store.variants.genotypes(
        _import(store, [stage, refined]), method="source"
    )
    assert set(store.variants.load_genotypes(genotypes).sites["source"]) == {
        "genotype_final, genotype_merged, genotype_geneseq, genotype_probebc"
    }
    expected = {}
    for row, cell in enumerate(TRUTH.row_ids[: TRUTH.n_cells]):
        for target, site in enumerate(TARGETS):
            label = TRUTH.got_call(row, target)
            if label == "NA":
                continue
            if row % 4 == 0 and label == "MUT":
                expected[(cell, site)] = ("filtered", "FalsePositive")
            else:
                call = "wild_type" if label == "WT" else "alternative_present"
                expected[(cell, site)] = (call, label)
    assert _calls(store, genotypes) == expected


def test_a_cell_read_without_a_call_of_the_source_is_filtered(store, tmp_path) -> None:
    # The table reads the third cell, 4 WT and 1 MUT UMI, without a call,
    # as two cells of GSE204845's MDS01 are.
    ids = TRUTH.row_ids[:3]
    path = tmp_path / "SF3B1.tsv"
    path.write_text(
        "\tWT.calls\tMUT.calls\tTotal.calls\tGenotype\tCell.Assignment\n"
        f"{ids[0]}\t2\t0\t2\tWT\tEP\n"
        f"{ids[1]}\t1\t1\t2\tMUT\tEP\n"
        f"{ids[2]}\t4\t1\t5\tNA\tIMP\n"
    )
    found = _calls(
        store, store.variants.genotypes(_import(store, path), method="source")
    )
    assert found == {
        (ids[0], "SF3B1"): ("wild_type", "WT"),
        (ids[1], "SF3B1"): ("alternative_present", "MUT"),
        (ids[2], "SF3B1"): ("filtered", ""),
    }


def test_a_call_column_without_a_vocabulary_is_refused(store, tmp_path) -> None:
    counts = _got(store, tmp_path, "geo_total")
    # A tool's call column that the vocabulary does not know, here GoT's
    # renamed in the saved counts.
    group = artifact_group(store.zw, counts)
    del group["call_source_names"]
    group.create_array("call_source_names", data=np.asarray(["call"], dtype="<U4"))
    with pytest.raises(ValueError, match="column 'call' at the site .* no vocabulary"):
        store.variants.genotypes(counts, method="source")


def test_gotchas_labels_keep_heterozygotes_apart(store, tmp_path) -> None:
    counts = _import(store, write_gotcha(TRUTH, tmp_path / "CALR.csv"))
    found = _calls(store, store.variants.genotypes(counts, method="source"))
    labels = {label for _call, label in found.values()}
    assert labels <= {"WT", "MUT", "HET"}
    for call, label in found.values():
        assert (
            call
            == {
                "WT": "wild_type",
                "MUT": "alternative_present",
                "HET": "heterozygous",
            }[label]
        )


def _expected_threshold(min_mutant_umis=1, min_fraction=None, min_umis=1):
    """GoT's rules on the truth's WT and MUT UMIs, ambiguous ones aside."""
    expected = {}
    for row, cell in enumerate(TRUTH.row_ids[: TRUTH.n_cells]):
        for target, site in enumerate(TARGETS):
            wild, mutant, ambiguous = TRUTH.got[row, target].tolist()
            if wild < 0 or wild + mutant + ambiguous == 0:
                continue
            total = wild + mutant
            call = "filtered"
            if total >= min_umis:
                if min_fraction is None:
                    if mutant >= min_mutant_umis:
                        call = "alternative_present"
                    elif mutant == 0:
                        call = "wild_type"
                else:
                    if mutant / total >= min_fraction and mutant >= (
                        min_mutant_umis or 0
                    ):
                        call = "alternative_present"
                    elif wild / total >= min_fraction:
                        call = "wild_type"
            expected[(cell, site)] = (call, "")
    return expected


@pytest.mark.parametrize(
    "rules",
    [
        {"min_mutant_umis": 1},
        {"min_mutant_umis": 2},
        {"min_mutant_umis": 1, "min_umis": 3},
        {"min_fraction": 0.9},
        {"min_fraction": 0.6, "min_mutant_umis": 2},
    ],
)
def test_threshold_calls_follow_gots_rules(store, tmp_path, rules) -> None:
    # IronThrone's tables count ambiguous UMIs too, which cover a cell but
    # support no call.
    counts = _got(store, tmp_path, "ironthrone")
    genotypes = store.variants.genotypes(counts, method="threshold", **rules)
    assert _calls(store, genotypes) == _expected_threshold(**rules)
    loaded = store.variants.load_genotypes(genotypes)
    calls = loaded.calls.set_index(["ids", "site"])
    for row, cell in enumerate(TRUTH.row_ids[: TRUTH.n_cells]):
        for target, site in enumerate(TARGETS):
            wild, mutant, ambiguous = TRUTH.got[row, target].tolist()
            if (cell, site) in calls.index:
                found = calls.loc[(cell, site)]
                assert (found["reference_umis"], found["alternative_umis"]) == (
                    wild,
                    mutant,
                )
                assert math.isnan(found["posterior_wild_type"])
    assert loaded.rules["min_umis"] == rules.get("min_umis", 1)
    # The method and its rules have one home, the provenance parameters.
    parameters = store.artifacts.inspect(genotypes).parameters
    assert loaded.method == parameters["method"] == "threshold"
    assert loaded.rules == {name: parameters[name] for name in GENOTYPE_RULES}
    assert not {"method", "rules"} & set(store.artifacts.load(genotypes).attrs)


def _gift_store(store, tmp_path):
    return _import(store, write_giftwrap_h5(TRUTH, tmp_path / "counts.1.h5"))


def _expected_gift(alleles, swap=None, min_posterior=None, min_umis=1):
    """GIFT's posterior written out per UMI of the truth, entry by entry.

    Each gapfill's likelihood under an allele is 0.01 per edit, normalized
    over the probe's two alleles; a UMI's scores are the larger of its swap
    probability and its allele's share, half of each for the heterozygote,
    and a cell's likelihood their product. A call's posterior reaches
    ``min_posterior``, or exceeds 0.6 without it, as GIFT calls cells.
    """
    expected = {}
    by_cell: dict[tuple[int, str], list] = {}
    for (row, probe, gapfill), umis in TRUTH.gift.items():
        if row >= TRUTH.n_cells:
            continue
        name = PROBES[probe][0]
        if name not in alleles:
            continue
        by_cell.setdefault((row, name), []).append((gapfill, umis))
    for (row, name), entries in by_cell.items():
        wild, mutant = alleles[name]
        logs = [0.0, 0.0, 0.0]
        confident = 0
        for gapfill, umis in entries:
            to_wild = 0.01 ** levenshtein_distance(gapfill, wild)
            to_mutant = 0.01 ** levenshtein_distance(gapfill, mutant)
            share = to_wild / (to_wild + to_mutant)
            reads = sum(read for read, _support in umis) // len(umis)
            p = 0.0
            if swap is not None:
                p = swap[max(key for key in swap if key <= reads)]
            scores = [
                max(p, share),
                max(p, 0.5 * share, 0.5 * (1 - share)),
                max(p, 1 - share),
            ]
            for genotype in range(3):
                logs[genotype] += len(umis) * math.log(scores[genotype])
            confident += len(umis) if p < 0.1 else 0
        top = max(logs)
        weights = [math.exp(value - top) for value in logs]
        posteriors = [weight / sum(weights) for weight in weights]
        best = posteriors.index(max(posteriors))
        call = (
            ("wild_type", "heterozygous", "homozygous_alternative")[best]
            if (
                max(posteriors) > 0.6
                if min_posterior is None
                else max(posteriors) >= min_posterior
            )
            and confident >= min_umis
            else "filtered"
        )
        expected[(TRUTH.row_ids[row], name)] = (call, posteriors)
    return expected


def _gift_calls(store, genotypes):
    calls = store.variants.load_genotypes(genotypes).calls
    return {
        (str(row.ids), str(row.site)): (
            str(row.call),
            [
                row.posterior_wild_type,
                row.posterior_heterozygous,
                row.posterior_homozygous_alternative,
            ],
        )
        for row in calls.itertuples(index=False)
    }


@pytest.mark.parametrize(
    ("options", "swap"),
    [
        ({}, None),
        ({"min_posterior": 0.9}, None),
        ({"swap_probabilities": {1: 0.95, 5: 0.5, 12: 0.05, 20: 0.0}}, None),
    ],
)
def test_gift_calls_follow_the_posterior_of_each_cell(
    store, tmp_path, options, swap
) -> None:
    counts = _gift_store(store, tmp_path)
    alleles = {"JAK2 c.1849G>T": ("CAC", "AAC"), "ASXL1 c.1772dup": ("GTAG", "GTAAG")}
    genotypes = store.variants.genotypes(
        counts,
        method="gift",
        alleles={"ASXL1 c.1772dup": ("GTAG", "GTAAG")},
        **options,
    )
    found = _gift_calls(store, genotypes)
    expected = _expected_gift(
        alleles,
        swap=options.get("swap_probabilities"),
        min_posterior=options.get("min_posterior"),
    )
    assert found.keys() == expected.keys()
    for key, (call, posteriors) in expected.items():
        assert found[key][0] == call, key
        np.testing.assert_allclose(found[key][1], posteriors, rtol=1e-6, atol=1e-7)
    loaded = store.variants.load_genotypes(genotypes)
    # The manifest names JAK2's alleles; alleles names ASXL1's; the control
    # probe has none and is left out.
    assert loaded.sites[["reference", "alternative"]].to_dict("index") == {
        "JAK2 c.1849G>T": {"reference": "CAC", "alternative": "AAC"},
        "ASXL1 c.1772dup": {"reference": "GTAG", "alternative": "GTAAG"},
    }
    # The rules as GIFT applied them: one UMI, the posterior given or GIFT's,
    # above 0.6, and the swap probabilities as (reads, probability) pairs by
    # reads.
    assert loaded.method == "gift"
    assert loaded.rules == {
        "min_umis": 1,
        "min_mutant_umis": None,
        "min_fraction": None,
        "min_posterior": options.get("min_posterior", 0.6000000000000001),
        "swap_probabilities": [
            [reads, probability]
            for reads, probability in options.get("swap_probabilities", {}).items()
        ],
        "min_quality": None,
        "min_depth": None,
    }


def test_gift_counts_the_umis_that_match_each_allele(store, tmp_path) -> None:
    counts = _gift_store(store, tmp_path)
    calls = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="gift")
    ).calls.set_index(["ids", "site"])
    for (row, probe, gapfill), umis in TRUTH.gift.items():
        if row >= TRUTH.n_cells or probe != 0:
            continue
        found = calls.loc[(TRUTH.row_ids[row], PROBES[0][0])]
        if gapfill == "CAC":
            assert found["reference_umis"] == len(umis)
        if gapfill == "AAC":
            assert found["alternative_umis"] == len(umis)


def test_sites_and_alleles_are_checked(store, tmp_path) -> None:
    gift = _gift_store(store, tmp_path)
    with pytest.raises(ValueError, match="has 0 wild types and 1 alternatives"):
        store.variants.genotypes(gift, method="gift", sites=["ASXL1 c.1772dup"])
    with pytest.raises(ValueError, match="no site 'KRAS'"):
        store.variants.genotypes(gift, method="gift", sites=["KRAS"])
    with pytest.raises(ValueError, match="alleles names 'KRAS'"):
        store.variants.genotypes(gift, method="gift", alleles={"KRAS": ("A", "C")})
    with pytest.raises(ValueError, match="no site 'KRAS'"):
        store.variants.genotypes(gift, method="gift", sites=["KRAS"])
    with pytest.raises(ValueError, match="sites names no site"):
        store.variants.genotypes(gift, method="gift", sites=[])
    # A named wild type that no cell shows is refused, a named alternative
    # that none shows is told of.
    with pytest.raises(ValueError, match="'GTGG', which no cell shows there"):
        store.variants.genotypes(
            gift, method="gift", alleles={"ASXL1 c.1772dup": ("GTGG", "GTAAG")}
        )
    with pytest.warns(UserWarning, match="No cell shows the alternative 'GTTAG'"):
        store.variants.genotypes(
            gift, method="gift", alleles={"ASXL1 c.1772dup": ("GTAG", "GTTAG")}
        )
    # GoT's counts hold no gapfills to weigh.
    got = _got(store, tmp_path, "xv")
    with pytest.raises(ValueError, match="weighs the gapfills of giftwrap's probes"):
        store.variants.genotypes(got, method="gift", alleles={"CALR": ("WT", "MUT")})
    # Tables without calls give the source method nothing to adopt.
    with pytest.raises(ValueError, match="no site holds calls of the source"):
        store.variants.genotypes(_got(store, tmp_path, "xv"), method="source")
    with pytest.raises(ValueError, match="holds no call of the source"):
        store.variants.genotypes(
            _got(store, tmp_path, "xv"), method="source", sites=["CALR"]
        )
    with pytest.raises(ValueError, match="variant_counts artifact"):
        store.variants.genotypes(every_cell(store), method="source")
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.variants.genotypes("counts", method="source")


def test_one_site_is_genotyped_alone_and_named_sites_follow_the_counts(
    store, tmp_path
) -> None:
    counts = _got(store, tmp_path, "geo_total")
    alone = store.variants.genotypes(counts, method="source", sites=["SF3B1"])
    loaded = store.variants.load_genotypes(alone)
    assert loaded.sites.index.tolist() == ["SF3B1"]
    both = store.variants.genotypes(counts, method="source", sites=["SF3B1", "CALR"])
    assert store.variants.load_genotypes(both).sites.index.tolist() == list(TARGETS)
    shown = store.variants.load_genotypes(both, sites=["SF3B1"])
    assert set(shown.calls["site"].astype(str)) == {"SF3B1"}
    # Sites named out of order, or twice, are loaded once each in the
    # counts' order.
    every = store.variants.load_genotypes(both)
    named = store.variants.load_genotypes(both, sites=["SF3B1", "CALR", "SF3B1"])
    pd.testing.assert_frame_equal(named.calls, every.calls)
    with pytest.raises(ValueError, match="no site 'KRAS'"):
        store.variants.load_genotypes(both, sites=["KRAS"])
    with pytest.raises(TypeError, match="a sequence of site labels"):
        store.variants.load_genotypes(both, sites="CALR")
    with pytest.raises(TypeError, match="must hold site labels or positions"):
        store.variants.load_genotypes(both, sites=[1.5])
    # GoT's targets have no coordinates, so a position names none of them.
    with pytest.raises(ValueError, match="named by position only on genotypes"):
        store.variants.load_genotypes(both, sites=[1])
    store.memoryBytes = 1_000
    with pytest.raises(MemoryError, match="Loading the genotype calls"):
        store.variants.load_genotypes(both)


def test_saved_genotypes_are_reused_by_their_rules(store, tmp_path) -> None:
    counts = _got(store, tmp_path, "ironthrone")
    first = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    assert (
        store.variants.genotypes(counts, method="threshold", min_mutant_umis=1) == first
    )
    other = store.variants.genotypes(counts, method="threshold", min_mutant_umis=2)
    assert other != first
    again = store.variants.genotypes(
        counts, method="threshold", min_mutant_umis=1, invalidate_cache=True
    )
    assert again != first
    assert _calls(store, again) == _calls(store, first)
    # Reuse takes the newest of the saved genotypes with the same rules.
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    assert (
        reader.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
        == again
    )
    with pytest.raises(PermissionError, match="call_variant_genotypes"):
        reader.variants.genotypes(counts, method="threshold", min_mutant_umis=3)


def test_saved_genotypes_that_lost_an_array_are_called_again(store, tmp_path) -> None:
    counts = _got(store, tmp_path, "ironthrone")
    first = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    del artifact_group(store.zw, first)["posterior_heterozygous"]
    again = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    assert again != first
    assert "posterior_heterozygous" in store.artifacts.load(again)
    # Offsets of another number of cells, and a long array cut short, are no
    # arrays of the calls either.
    offsets = artifact_group(store.zw, again)["call_offsets"]
    offsets.resize((offsets.shape[0] + 1,))
    assert store.variants.genotypes(counts, method="threshold", min_mutant_umis=1) != (
        again
    )
    again = store.variants.genotypes(
        counts, method="threshold", min_mutant_umis=1, invalidate_cache=True
    )
    group = artifact_group(store.zw, again)
    short = group["call_reference"][:-1]
    del group["call_reference"]
    group.create_array("call_reference", data=short)
    assert store.variants.genotypes(counts, method="threshold", min_mutant_umis=1) != (
        again
    )


@pytest.mark.parametrize(
    ("method", "step"),
    [("threshold", "for a band of"), ("source", "the sites of the source calls")],
)
def test_genotypes_refuse_a_budget_too_small_before_writing(
    store, tmp_path, method, step
) -> None:
    counts, options = _method_counts(store, tmp_path, method)
    before = len(store.artifacts.list(kind="variant_genotypes", scope="datastore"))
    store.memoryBytes = 1_000
    with pytest.raises(MemoryError, match=f"{step}.*Raise mem_budget"):
        store.variants.genotypes(counts, method=method, **options)
    assert (
        len(store.artifacts.list(kind="variant_genotypes", scope="datastore")) == before
    )


def test_bands_count_a_buffer_for_each_write_zarr_runs_at_once(store, tmp_path) -> None:
    # Appending a band's calls writes stored chunks, each through a buffered
    # file, as many at once as Zarr's async concurrency allows, and Python
    # 3.14 made each buffer 128 KiB, where it was the file system's block.
    import io
    import re

    import zarr

    counts, options = _method_counts(store, tmp_path, "threshold")

    def bytes_named(concurrency: int) -> int:
        """The bytes that the check of the first band names."""
        with zarr.config.set({"async.concurrency": concurrency}):
            store.memoryBytes = 1
            try:
                store.variants.genotypes(counts, method="threshold", **options)
            except MemoryError as error:
                found = re.search(r"holds about (\d+) bytes for a band", str(error))
                assert found is not None, str(error)
                return int(found.group(1))
            finally:
                store.memoryBytes = 1 << 30
        raise AssertionError("a budget of one byte called the genotypes")

    assert bytes_named(20) - bytes_named(10) == 10 * io.DEFAULT_BUFFER_SIZE


def _method_counts(store, tmp_path, method):
    if method == "gift":
        return _gift_store(store, tmp_path), {}
    counts = _got(store, tmp_path, "geo_total")
    return counts, {"min_mutant_umis": 1} if method == "threshold" else {}


def _recorded_bands(monkeypatch) -> list[tuple[int, int]]:
    """Record the most rows allowed and the rows held of each band read."""
    from cytearc.datastore._operations import variants

    bands: list[tuple[int, int]] = []
    read = variants._variant_bands

    def recorded(group, tables, positions, max_rows, **options):
        for band, rows in read(group, tables, positions, max_rows, **options):
            bands.append((max_rows, sum(len(cells) for cells, _ in rows.values())))
            yield band, rows

    monkeypatch.setattr(variants, "_variant_bands", recorded)
    return bands


@pytest.mark.parametrize("method", ["threshold", "gift", "source"])
def test_genotypes_read_in_bands_equal_those_read_in_one(
    store, tmp_path, monkeypatch, method
) -> None:
    from cytearc.datastore._operations import variants

    counts, options = _method_counts(store, tmp_path, method)
    whole = store.variants.load_genotypes(
        store.variants.genotypes(counts, method=method, **options)
    ).calls
    # Without a floor of rows, the smallest budget reads bands of a few cells.
    monkeypatch.setattr(variants, "_MIN_GENOTYPE_BAND_ROWS", 1)
    bands = _recorded_bands(monkeypatch)
    budget = smallest_budget(
        lambda limit: _at_budget(store, limit, counts, method, options), 1 << 26
    )
    bands.clear()
    narrow = _at_budget(store, budget, counts, method, options)
    assert len(bands) > 10
    found = store.variants.load_genotypes(narrow).calls
    pd.testing.assert_frame_equal(found, whole)


@pytest.mark.slow
def test_a_band_holds_a_floor_of_rows_and_the_cell_with_the_most(
    store, tmp_path, monkeypatch
) -> None:
    from cytearc.datastore._operations import variants

    counts = _got(store, tmp_path, "ironthrone")
    group = artifact_group(store.zw, counts)
    per_cell = np.diff(group["count_offsets"][:] + group["depth_offsets"][:])
    options = {"min_mutant_umis": 1}
    bands = _recorded_bands(monkeypatch)

    def smallest_bands() -> list[tuple[int, int]]:
        budget = smallest_budget(
            lambda limit: _at_budget(store, limit, counts, "threshold", options),
            1 << 26,
        )
        bands.clear()
        _at_budget(store, budget, counts, "threshold", options)
        return list(bands)

    # The truth's rows number fewer than the floor, so one band holds them.
    assert per_cell.sum() < variants._MIN_GENOTYPE_BAND_ROWS
    assert smallest_bands() == [(per_cell.sum(), per_cell.sum())]
    # With a floor of one row, a band still holds the cell with the most.
    monkeypatch.setattr(variants, "_MIN_GENOTYPE_BAND_ROWS", 1)
    found = smallest_bands()
    assert {allowed for allowed, _held in found} == {per_cell.max()}
    assert max(held for _allowed, held in found) == per_cell.max()


def _at_budget(store, budget, counts, method, options):
    store.memoryBytes = budget
    try:
        return store.variants.genotypes(
            counts, method=method, invalidate_cache=True, **options
        )
    finally:
        store.memoryBytes = 1 << 30


def test_loaded_calls_name_every_call_of_the_vocabulary(store, tmp_path) -> None:
    counts = _got(store, tmp_path, "geo_total")
    calls = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="source")
    ).calls
    assert list(calls["call"].cat.categories) == list(CALLS)
    assert calls["call"].cat.ordered
    assert list(calls.columns) == [
        "ids",
        "site",
        "call",
        "label",
        "reference_umis",
        "alternative_umis",
        "posterior_wild_type",
        "posterior_heterozygous",
        "posterior_homozygous_alternative",
    ]


@pytest.mark.slow
@pytest.mark.parametrize("method", ["threshold", "gift", "source"])
def test_genotypes_hold_no_more_than_the_smallest_budget_they_accept(
    store, tmp_path, method
) -> None:
    counts, options = _method_counts(store, tmp_path, method)
    budget = smallest_budget(
        lambda limit: _at_budget(store, limit, counts, method, options), 1 << 28
    )
    store.memoryBytes = budget
    _genotypes, peak = traced_peak(
        lambda: store.variants.genotypes(
            counts, method=method, invalidate_cache=True, **options
        ),
        budget,
    )
    assert peak <= budget


class _Accepted(Exception):
    """Raised where genotypes would start to read bands."""


def _many_cell_counts(
    store, ids: list[str], folder: Path, method: str, n_sites: int = 8
):
    """Counts of each cell at ``n_sites`` targets: GoT tables with their calls,
    or GEO's GIFT tables of one row per UMI."""
    rng = np.random.default_rng(16)
    clone = rng.integers(0, 200, size=len(ids))
    if method != "gift":
        paths = []
        for site in range(n_sites):
            lines = ["BC\twtUMIs\tmutUMIs\tGenotype"]
            for cell in np.flatnonzero(rng.random(len(ids)) < 0.8).tolist():
                mutant = clone[cell] % n_sites == site
                wild = int(rng.poisson(2)) + int(not mutant)
                alternative = int(rng.poisson(1)) + 1 if mutant else 0
                call = "MUT" if alternative else "WT"
                lines.append(f"{ids[cell]}\t{wild}\t{alternative}\t{call}")
            path = folder / f"T{site}.tsv"
            path.write_text("\n".join(lines) + "\n")
            paths.append(path)
        return _import(store, paths)
    lines = [
        "cell_idx\tprobe_idx\tprobe_barcode\tumi\tgapfill\t"
        "pcr_duplicate_count\tpercent_supporting"
    ]
    for cell in range(len(ids)):
        for site in np.flatnonzero(rng.random(n_sites) < 0.7).tolist():
            mutant = clone[cell] % n_sites == site
            for umi in range(int(rng.integers(1, 5))):
                gapfill = "AAC" if mutant and rng.random() < 0.5 else "CAC"
                reads = int(rng.integers(1, 9))
                lines.append(f"{cell}\t{site}\t1\tU{umi}\t{gapfill}\t{reads}\t1.0")
    tables = folder / "gift"
    tables.mkdir()
    manifest = [
        "name\tlhs_probe\trhs_probe\tgap_probe_sequence\t"
        "original_gap_probe_sequence\tgene\tindex",
        *(f"P{site}\tAAAA\tTTTT\tAAC\tCAC\t\t{site}" for site in range(n_sites)),
    ]
    for name, rows in (
        ("probe_reads", lines),
        ("barcodes", ["barcode\tplex", *(f"{cell}\t1" for cell in ids)]),
        ("manifest", manifest),
    ):
        with gzip.open(tables / f"S_GAP_{name}.tsv.gz", "wt") as handle:
            handle.write("\n".join(rows) + "\n")
    return _import(store, str(tables))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("method", "options"),
    [
        ("threshold", {"min_mutant_umis": 1}),
        ("source", {}),
        ("gift", {"swap_probabilities": {1: 0.5, 2: 0.2, 4: 0.05}}),
    ],
)
def test_genotypes_of_many_cells_hold_no_more_than_a_budget_of_wide_bands(
    tmp_path, monkeypatch, method, options
) -> None:
    from cytearc.datastore._operations import variants

    # 10,000 cells at 8 targets: bands of 60,000 rows hold a third or more of
    # the rows, so the bytes of each row outweigh those held whatever the
    # counts' size.
    ids = [f"c{cell:06d}-1" for cell in range(10_000)]
    store = write_cells_store(tmp_path / "store.zarr", ids)
    counts = _many_cell_counts(store, ids, tmp_path, method)
    read = variants._variant_bands

    def accepted(*args, **kwargs):
        raise _Accepted

    def call(budget: int) -> None:
        store.memoryBytes = budget
        store.variants.genotypes(
            counts, method=method, invalidate_cache=True, **options
        )

    def plans(budget: int) -> None:
        try:
            call(budget)
        except _Accepted:
            pass

    # The smallest budget accepted, found without reading, reads bands of
    # the floor of rows.
    monkeypatch.setattr(variants, "_variant_bands", accepted)
    high = smallest_budget(plans, 1 << 32)
    monkeypatch.setattr(variants, "_variant_bands", read)
    per_row = variants._GENOTYPE_BYTES_PER_ROW[method]
    bands = _recorded_bands(monkeypatch)
    peaks = {}
    for rows in (20_000, 60_000):
        budget = high + (rows - variants._MIN_GENOTYPE_BAND_ROWS) * per_row
        bands.clear()
        peaks[rows] = traced_peak(lambda budget=budget: call(budget), budget)[1]
        assert {allowed for allowed, _held in bands} == {rows}
        assert max(held for _allowed, held in bands) > rows // 2
        assert peaks[rows] <= budget
    # What 40,000 more rows a band add stays within their bytes per row,
    # whatever the bytes held beside the rows.
    assert peaks[60_000] - peaks[20_000] <= 40_000 * per_row


@pytest.mark.slow
def test_loaded_genotypes_of_many_cells_hold_no_more_than_the_smallest_budget(
    tmp_path,
) -> None:
    from cytearc.datastore._operations import variants

    # 64,000 and 128,000 call rows of 2,000 cells at 40 and 80 targets, which
    # outweigh what the cells hold and the read of their IDs.
    ids = [f"c{cell:06d}-1" for cell in range(2_000)]
    peaks = {}
    for n_sites in (40, 80):
        folder = tmp_path / str(n_sites)
        folder.mkdir()
        store = write_cells_store(folder / "store.zarr", ids)
        counts = _many_cell_counts(store, ids, folder, "source", n_sites=n_sites)
        genotypes = store.variants.genotypes(counts, method="source")
        n_rows = int(artifact_group(store.zw, genotypes)["calls"].shape[0])

        def load(budget: int, store=store, genotypes=genotypes) -> object:
            store.memoryBytes = budget
            try:
                return store.variants.load_genotypes(genotypes)
            finally:
                store.memoryBytes = 1 << 30

        budget = smallest_budget(load, 1 << 30)
        peaks[n_sites] = (
            n_rows,
            traced_peak(lambda load=load, budget=budget: load(budget), budget)[1],
        )
        assert peaks[n_sites][1] <= budget
    # The bytes of each call row stay within what the check counts for it.
    (rows_40, peak_40), (rows_80, peak_80) = peaks[40], peaks[80]
    assert peak_80 - peak_40 <= (rows_80 - rows_40) * (
        variants._LOADED_GENOTYPE_BYTES_PER_ROW
    )
