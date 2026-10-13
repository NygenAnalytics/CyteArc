import gzip

import numpy as np
import pandas as pd
import pytest

from cytearc.features.genomic.annotation import (
    link_regions_to_genes,
    read_gene_annotation,
    transcription_start_sites,
)

_GTF = """##description: test
chr1\tHAVANA\tgene\t101\t200\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "protein_coding"; gene_name "ALPHA";
chr1\tHAVANA\ttranscript\t101\t200\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "T1";
chr1\tHAVANA\tgene\t1001\t1100\t.\t-\t.\tgene_id "ENSG2.3"; gene_type "lncRNA"; gene_name "BETA";
chr2\tHAVANA\tgene\t51\t60\t.\t+\t.\tgene_id "ENSG3.1"; gene_type "protein_coding";
"""

_GFF3 = """##gff-version 3
chr1\tensembl\tgene\t101\t200\t.\t+\t.\tID=gene:ENSG1;Name=ALPHA;biotype=protein_coding
chr1\tensembl\tmRNA\t101\t200\t.\t+\t.\tID=transcript:T1;Parent=gene:ENSG1
"""


def test_read_gtf_converts_to_bed_coordinates(tmp_path):
    path = tmp_path / "genes.gtf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(_GTF)

    genes = read_gene_annotation(path)

    assert genes.to_dict("list") == {
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [100, 1000, 50],
        "end": [200, 1100, 60],
        "strand": ["+", "-", "+"],
        "gene_id": ["ENSG1.1", "ENSG2.3", "ENSG3.1"],
        "gene_name": ["ALPHA", "BETA", "ENSG3.1"],
        "gene_type": ["protein_coding", "lncRNA", "protein_coding"],
    }
    coding = read_gene_annotation(path, gene_types={"protein_coding"})
    assert coding["gene_id"].tolist() == ["ENSG1.1", "ENSG3.1"]


def test_read_gff3_strips_the_gene_prefix(tmp_path):
    path = tmp_path / "genes.gff3"
    path.write_text(_GFF3)

    genes = read_gene_annotation(path)

    assert genes[["gene_id", "gene_name", "gene_type"]].values.tolist() == [
        ["ENSG1", "ALPHA", "protein_coding"]
    ]


def test_read_bed_annotation_requires_strand(tmp_path):
    path = tmp_path / "genes.bed"
    path.write_text("chr1\t100\t200\tALPHA\t0\t+\tENSG1\n")
    assert read_gene_annotation(path)["gene_id"].tolist() == ["ENSG1"]

    path.write_text("chr1\t100\t200\tALPHA\t0\t.\n")
    with pytest.raises(ValueError, match="strand"):
        read_gene_annotation(path)
    with pytest.raises(ValueError, match="no gene types"):
        read_gene_annotation(path, gene_types={"protein_coding"})


def test_transcription_start_sites_follow_strand():
    genes = pd.DataFrame(
        {"start": [100, 1000], "end": [200, 1100], "strand": ["+", "-"]}
    )

    np.testing.assert_array_equal(transcription_start_sites(genes), [100, 1099])


def _genes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [1000, 5000, 100],
            "end": [2000, 6000, 900],
            "strand": ["+", "-", "+"],
            "gene_id": ["G1", "G2", "G3"],
            "gene_name": ["A", "B", "C"],
            "gene_type": ["", "", ""],
        }
    )


def test_link_regions_to_genes_uses_strand_aware_windows():
    chroms = np.array(["chr1", "chr1", "chr1", "chr1", "chr3"], dtype=object)
    starts = np.array([890, 1050, 5990, 6050, 0])
    ends = np.array([900, 1060, 6000, 6060, 10])

    links = link_regions_to_genes(
        chroms, starts, ends, _genes(), upstream=110, downstream=60
    )

    # G1 (+, TSS 1000): window 890..1060. G2 (-, TSS 5999): window 5939..6109.
    assert links.to_dict("list") == {
        "region": [0, 1, 2, 3],
        "gene_id": ["G1", "G1", "G2", "G2"],
        "gene_name": ["A", "A", "B", "B"],
        "distance": [-101, 50, 0, -51],
    }


def test_link_regions_to_genes_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 300
    chroms = rng.choice(np.array(["chr1", "chr2"], dtype=object), size=n)
    starts = rng.integers(0, 20000, size=n)
    ends = starts + rng.integers(1, 800, size=n)
    genes = pd.DataFrame(
        {
            "chrom": rng.choice(["chr1", "chr2"], size=40),
            "start": rng.integers(0, 20000, size=40),
            "strand": rng.choice(["+", "-"], size=40),
            "gene_id": [f"G{i}" for i in range(40)],
            "gene_name": [f"N{i}" for i in range(40)],
            "gene_type": "",
        }
    )
    genes["end"] = genes["start"] + rng.integers(1, 3000, size=40)
    tss = transcription_start_sites(genes)

    links = link_regions_to_genes(
        chroms, starts, ends, genes, upstream=500, downstream=300
    )

    expected = set()
    for gene, row in genes.iterrows():
        plus = row["strand"] == "+"
        low = tss[gene] - (500 if plus else 300)
        high = tss[gene] + (300 if plus else 500)
        for region in range(n):
            if (
                chroms[region] == row["chrom"]
                and starts[region] <= high
                and ends[region] - 1 >= low
            ):
                expected.add((region, row["gene_id"]))
    assert set(zip(links["region"], links["gene_id"])) == expected


def test_overlapping_intervals_matches_brute_force():
    from cytearc.features.genomic.annotation import overlapping_intervals

    rng = np.random.default_rng(3)
    query_starts = rng.integers(0, 5000, size=120)
    query = (
        rng.choice(np.array(["chr1", "chr2"], dtype=object), size=120),
        query_starts,
        query_starts + rng.integers(1, 400, size=120),
    )
    subject_starts = rng.integers(0, 5000, size=90)
    subject = (
        rng.choice(np.array(["chr1", "chr3", "chr2"], dtype=object), size=90),
        subject_starts,
        subject_starts + rng.integers(1, 900, size=90),
    )

    q, s, shared = overlapping_intervals(query, subject)

    expected = []
    for i in range(120):
        for j in range(90):
            overlap = min(query[2][i], subject[2][j]) - max(query[1][i], subject[1][j])
            if query[0][i] == subject[0][j] and overlap > 0:
                expected.append((i, j, overlap))
    assert list(zip(q.tolist(), s.tolist(), shared.tolist())) == sorted(expected)


def test_link_regions_to_genes_with_gene_body_windows():
    genes = _genes()
    chroms = np.array(["chr1", "chr1", "chr1", "chr1"], dtype=object)
    # G1 (+) body 1000..2000; G2 (-) body 5000..6000.
    starts = np.array([850, 1500, 2050, 6080])
    ends = np.array([900, 1510, 2100, 6100])

    links = link_regions_to_genes(
        chroms, starts, ends, genes, upstream=200, downstream=100, anchor="gene_body"
    )

    # Region 0 is 101 bases 5' of G1; region 1 is inside it; region 2 is 51
    # bases 3' of G1 (within 100). Region 3 is 81 bases 5' of minus-strand G2.
    assert links[["region", "gene_id", "distance"]].values.tolist() == [
        [0, "G1", -101],
        [1, "G1", 0],
        [2, "G1", 51],
        [3, "G2", -81],
    ]
