"""Mission Bio's published Tapestri example and a DAb-seq file of GEO, end to end.

Both files are downloaded when the tests run and are never checked in:
Mission Bio's example (Zenodo 10.5281/zenodo.20818440) is CC BY-SA 4.0, and
GEO serves GSM9567643 without a stated license. The expected numbers were
computed from the files with h5py and NumPy alone: depth rows are the
positions' DP above 0, Tapestri's alternative reads AF × DP / 100 rounded,
and the calls each NGT or GT of a cell with reads at its variant.
"""

import warnings

import h5py
import numpy as np
import pytest

from tests.clone_helpers import every_cell, write_cells_store
from tests.network_helpers import download_or_skip

_ZENODO = (
    "https://zenodo.org/records/20818440/files/"
    "4-cell-lines-AML-multiomics.dna+protein.h5?download=1"
)
_GEO = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9567nnn/GSM9567643/suppl/"
    "GSM9567643.hdf5"
)


def _call_shares(genotypes) -> dict[str, int]:
    sums = genotypes.sites[
        ["wild_type", "heterozygous", "homozygous_alternative", "filtered"]
    ].sum()
    return {name: int(value) for name, value in sums.items()}


@pytest.mark.slow
@pytest.mark.integration
def test_the_tapestri_example_reads_into_a_store_and_its_variants(tmp_path) -> None:
    from cytearc.datastore.datastore import DataStore
    from cytearc.readers import TapestriReader
    from cytearc.writers import TapestriToZarr

    file = download_or_skip(
        _ZENODO, tmp_path / "4-cell-lines-AML-multiomics.dna+protein.h5"
    )
    reader = TapestriReader(str(file))
    try:
        TapestriToZarr(reader, str(tmp_path / "store.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = DataStore(
            str(tmp_path / "store.zarr"),
            default_assay="ADT",
            min_features_per_cell=0,
            nthreads=1,
        )
    assert ds.cells.N == 1313
    assert int(np.count_nonzero(ds.cells.fetch_all("filtered"))) == 26
    amplicons = np.asarray(ds.CNV.rawData.compute())
    proteins = np.asarray(ds.ADT.rawData.compute())
    assert amplicons.shape == (1313, 127) and proteins.shape == (1313, 10)
    assert int(amplicons.sum()) == 5_417_202
    assert int(np.count_nonzero(amplicons)) == 162_041
    assert int(proteins.sum()) == 7_655_165
    assert ds.ADT.feats.fetch_all("ids").tolist()[:3] == ["CD11b", "CD19", "CD3"]
    recorded = ds.zw.attrs["tapestri"]
    assert recorded["samples"] == ["4-cell-lines-AML-multiomics"]
    dna = recorded["assays"]["dna_read_counts"]
    assert (dna["panel_name"], dna["genome_version"], dna["pipeline_version"]) == (
        "AML",
        "hg19",
        "2.0.2",
    )
    protein = recorded["assays"]["protein_read_counts"]
    assert (protein["panel_name"], protein["pipeline_version"]) == (
        "ab_barcodes_compass027",
        "2.0.1",
    )

    counts = ds.variants.import_counts(
        file, cell_selection=ds.snapshot_cell_selection()
    )
    loaded = ds.variants.load_counts(counts)
    # The 40 variants that pass lie at 40 positions.
    assert len(loaded.sites) == 40
    assert (len(loaded.depth), int(loaded.depth["depth"].sum())) == (52_236, 1_531_230)
    assert (len(loaded.counts), int(loaded.counts["count"].sum())) == (28_914, 437_059)
    assert loaded.calls["call"].value_counts().to_dict() == {
        "0": 29_973,
        "1": 12_636,
        "2": 9_476,
        "3": 151,
    }
    assert float(loaded.calls["quality"].sum()) == 3_758_700
    first = "4-cell-lines-AML-multiomics__AACAACCTACTCTAGTAA"
    site = "chr1:115256513"
    depth = loaded.depth.set_index(["ids", "site"])["depth"]
    reads = loaded.counts.set_index(["ids", "site", "allele"])["count"]
    calls = loaded.calls.set_index(["ids", "site", "allele"])
    assert int(depth[(first, site)]) == 50
    assert int(reads[(first, site, "A")]) == 1
    assert calls.loc[(first, site, "A"), "call"] == "0"
    assert calls.loc[(first, site, "A"), "quality"] == 99
    genotypes = ds.variants.load_genotypes(
        ds.variants.genotypes(counts, method="source")
    )
    assert _call_shares(genotypes) == {
        "wild_type": 29_973,
        "heterozygous": 12_636,
        "homozygous_alternative": 9_476,
        "filtered": 151,
    }
    # Under the exact rule, four cell lines give four clones at the
    # variants that some but not most called cells carry, each line's own,
    # while every variant that passes splits them wherever allelic dropout
    # turns a call.
    sites = genotypes.sites
    mutant = sites["heterozygous"] + sites["homozygous_alternative"]
    share = mutant / (mutant + sites["wild_type"])
    chosen = share[share.between(0.15, 0.4)].index.tolist()
    assert len(chosen) == 16
    genotype_ref = ds.variants.genotypes(counts, method="source")
    sizes = {}
    loaded = {}
    for name, clone_sites in (("chosen", chosen), ("every", None)):
        loaded[name] = ds.clones.load_clones(
            ds.variants.genotype_clones(genotype_ref, sites=clone_sites, method="exact")
        )
        sizes[name] = (
            loaded[name].clones["n_cells"].sort_values(ascending=False).tolist()
        )
    assert sizes["chosen"][:5] == [351, 223, 200, 194, 43]
    assert sum(sizes["chosen"][:4]) > 0.7 * 1313
    assert (len(sizes["every"]), sizes["every"][0]) == (587, 51)
    # The likelihood gives the four lines from every variant that passes:
    # each of the exact rule's four lines keeps at least 80% of its cells in
    # one of the four largest clones, beside small clones of doublets and of
    # cells that lack one variant of their line, which dropout or a lost
    # amplicon explains as well. Line 0 keeps 295 of its 351 cells.
    likely_ref = ds.variants.genotype_clones(genotype_ref)
    likely = ds.clones.load_clones(likely_ref)
    largest = likely.clones["clone"].tolist()[:4]
    labels = likely.cells["clone"]
    kept = []
    for line in loaded["chosen"].clones["clone"].tolist()[:4]:
        cells = loaded["chosen"].cells["clone"] == line
        kept.append(max(int((labels[cells] == clone).sum()) for clone in largest))
    assert kept == [295, 223, 200, 194]
    assert all(
        count >= 0.8 * size
        for count, size in zip(kept, sizes["chosen"][:4], strict=True)
    )
    diagnostics = ds.artifacts.inspect(likely_ref).diagnostics
    assert diagnostics["sites"] == 40
    assert diagnostics["dropoutProbability"] == pytest.approx(0.0714, abs=0.001)
    assert diagnostics["errorProbability"] == pytest.approx(0.0273, abs=0.0005)
    dropout = ds.artifacts.load(likely_ref)["dropout_probability"][:]
    assert dropout.shape == (1, 40)
    assert dropout.min() < 0.01 and dropout.max() > 0.4
    # Each line fitted alone, as a within group of its own beside the other
    # cells', is one clone of all its cells: no wild-type cell pins a line's
    # error rate, and the bound on the pooled error keeps the mutations that
    # every cell of the line carries from passing as errors.
    lines = loaded["chosen"].cells["clone"].astype(object)
    names = np.where(
        lines.isin(loaded["chosen"].clones["clone"].tolist()[:4]), lines, "other"
    ).astype(object)
    ds.cells.insert("line", names, overwrite=True)
    alone = ds.clones.load_clones(
        ds.variants.genotype_clones(genotype_ref, within="line")
    ).cells["clone"]
    kept = [
        int(alone[lines == line].value_counts().iloc[0])
        for line in loaded["chosen"].clones["clone"].tolist()[:4]
    ]
    assert kept == [351, 223, 200, 194]
    # Filtering calls by quality and depth leaves more cells ambiguous.
    filtered_ref = ds.variants.genotypes(
        counts, method="source", min_quality=30, min_depth=10
    )
    sites = ds.variants.load_genotypes(filtered_ref).sites
    mutant = sites["heterozygous"] + sites["homozygous_alternative"]
    share = mutant / (mutant + sites["wild_type"])
    chosen = share[share.between(0.15, 0.4)].index.tolist()
    clones = ds.variants.genotype_clones(filtered_ref, sites=chosen, method="exact")
    assert (
        len(chosen),
        ds.artifacts.inspect(clones).diagnostics["ambiguousCells"],
    ) == (17, 809)


@pytest.mark.slow
@pytest.mark.integration
def test_a_dab_seq_file_of_geo_reads_into_variant_counts(tmp_path) -> None:
    file = download_or_skip(_GEO, tmp_path / "GSM9567643.hdf5")
    with h5py.File(file, "r") as handle:
        barcodes = [value.decode() for value in handle["CELL_BARCODES"][:]]
    store = write_cells_store(tmp_path / "store.zarr", barcodes)
    with pytest.warns(
        UserWarning,
        match="43216 chosen variants with an empty ALT and 1 whose alleles are not",
    ):
        counts = store.variants.import_counts(file, cell_selection=every_cell(store))
    loaded = store.variants.load_counts(counts)
    # 4,481 variants whose alleles are bases, at 4,265 positions, 194 of
    # which hold several, and FLT3-ITD, a named site of its own.
    assert len(loaded.sites) == 4266
    assert len(loaded.alleles) == 4266 + 4481
    itd = "FLT3-ITD:chr13:28608251:T/TCTC"
    assert loaded.sites.index[-1] == itd
    shown = loaded.depth[loaded.depth["site"] == itd]
    assert (len(shown), int(shown["depth"].sum())) == (6, 595)
    shown = loaded.counts[loaded.counts["site"] == itd]
    assert (shown["allele"].astype(str).tolist(), int(shown["count"].sum())) == (
        ["TCTC"] * 6,
        365,
    )
    assert (len(loaded.depth), int(loaded.depth["depth"].sum())) == (
        265_144,
        11_883_293,
    )
    # The alternatives' 6,338 entries of 226,905 reads and the references'
    # 260,874 entries of 11,654,731 reads.
    assert (len(loaded.counts), int(loaded.counts["count"].sum())) == (
        267_212,
        11_881_636,
    )
    # Each GT of a cell with reads, so FLT3-ITD's six carriers alone.
    assert loaded.calls["call"].value_counts().to_dict() == {
        "0": 274_065,
        "2": 4_312,
        "1": 1_854,
        "3": 950,
    }
    genotypes = store.variants.load_genotypes(
        store.variants.genotypes(counts, method="source")
    )
    assert len(genotypes.sites) == 4481
    assert _call_shares(genotypes) == {
        "wild_type": 274_065,
        "heterozygous": 1_854,
        "homozygous_alternative": 4_312,
        "filtered": 950,
    }
    assert genotypes.sites.loc[
        itd, ["heterozygous", "homozygous_alternative"]
    ].tolist() == [
        5,
        1,
    ]
