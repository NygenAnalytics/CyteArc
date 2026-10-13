import gzip

import numpy as np
import pandas as pd
import pytest

from cytearc.regulons.cistromes import (
    binding_cistromes,
    cistromes_from_motifs,
    project_to_genes,
    read_peak_file,
)


def test_read_peak_file_reads_narrowpeak_and_bed(tmp_path):
    narrow = tmp_path / "spi1.narrowPeak.gz"
    with gzip.open(narrow, "wt") as handle:
        handle.write("track name=x\nchr1\t100\t300\tp1\t500\t.\t5.1\t20\t18\t110\n")
        handle.write("chr2\t50\t90\tp2\t80\t.\t2.0\t5\t4\t20\n")
    bed = tmp_path / "seacr.bed"
    bed.write_text("chr1\t10\t40\n")

    peaks = read_peak_file(narrow)
    assert peaks.to_dict("list") == {
        "chrom": ["chr1", "chr2"],
        "start": [100, 50],
        "end": [300, 90],
        "name": ["p1", "p2"],
        "score": [500.0, 80.0],
        "summit": [110, 20],
    }
    plain = read_peak_file(bed)
    assert plain["name"].tolist() == ["chr1:10-40"]
    assert np.isnan(plain["score"].iloc[0]) and plain["summit"].iloc[0] == -1


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "motif": ["m1", "m2", "m2", "m3"],
            "TF": ["SPI1", "SPI1", "SPIB", "CEBPA"],
            "category": ["direct", "orthology", "direct", "similarity"],
        }
    )


def test_cistromes_from_motifs_unions_regions_per_factor():
    matches = pd.DataFrame(
        {"source": ["m1", "m1", "m2", "m3"], "target": ["r1", "r2", "r2", "r3"]}
    )

    direct = cistromes_from_motifs(matches, _annotations())
    extended = cistromes_from_motifs(matches, _annotations(), categories=("orthology",))

    assert direct.values.tolist() == [
        ["SPI1", "r1", "m1"],
        ["SPI1", "r2", "m1"],
        ["SPIB", "r2", "m2"],
    ]
    assert extended.values.tolist() == [["SPI1", "r2", "m2"]]


def test_binding_cistromes_overlap_rules():
    regions = ["chr1:100-200", "chr1:300-500", "chr2:0-100", "chr1:1000-1100"]
    peaks = {
        "SPI1": pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1"],
                "start": [150, 480, 490],
                "end": [160, 900, 495],
            }
        ),
        "PAX5": pd.DataFrame({"chrom": ["chr3"], "start": [0], "end": [10]}),
    }

    anywhere = binding_cistromes(peaks, regions)
    strict = binding_cistromes(peaks, regions, min_overlap_fraction=0.4)

    assert anywhere.values.tolist() == [
        ["SPI1", "chr1:100-200", 1],
        ["SPI1", "chr1:300-500", 2],
    ]
    # The 10 bp peak lies wholly inside its region (fraction 1 of the peak);
    # the region at 300-500 shares 20 bp with a 420 bp peak (below 0.4 of
    # either) and 5 bp with a 5 bp peak (all of it).
    assert strict.values.tolist() == [
        ["SPI1", "chr1:100-200", 1],
        ["SPI1", "chr1:300-500", 1],
    ]
    with pytest.raises(ValueError, match="min_overlap_fraction"):
        binding_cistromes(peaks, regions, min_overlap_fraction=1.0)


def test_project_to_genes_counts_linking_regions():
    cistromes = pd.DataFrame(
        {"source": ["SPI1", "SPI1", "PAX5"], "target": ["r1", "r2", "r2"]}
    )
    links = pd.DataFrame(
        {"region": [0, 1, 1, 2], "gene_name": ["LYZ", "LYZ", "CD79A", "MS4A1"]}
    )

    genes = project_to_genes(cistromes, links, region_names=["r1", "r2", "r3"])

    assert genes.values.tolist() == [
        ["PAX5", "CD79A", 1, "r2"],
        ["PAX5", "LYZ", 1, "r2"],
        ["SPI1", "CD79A", 1, "r2"],
        ["SPI1", "LYZ", 2, "r1,r2"],
    ]
