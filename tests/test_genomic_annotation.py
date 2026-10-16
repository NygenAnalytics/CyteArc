import gzip

import numpy as np
import pandas as pd
import pytest

from cytearc.features.genomic import annotation
from cytearc.features.genomic.annotation import (
    link_regions_to_genes,
    overlapping_intervals,
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


def test_read_gff3_follows_ncbi_and_ensembl_conventions(tmp_path):
    path = tmp_path / "genes.gff3"
    path.write_text(
        "##gff-version 3\n"
        "1\tRefSeq\tgene\t11\t20\t.\t+\t.\t"
        "ID=gene-OR4F5;Name=OR4F5;gene_biotype=protein_coding\n"
        "1\tEnsembl\tncRNA_gene\t31\t40\t.\t-\t.\t"
        "ID=gene:ENSG2;Name=AB%2CC;biotype=lncRNA\n"
        "##FASTA\n>1\nACGT\n"
    )

    genes = read_gene_annotation(path, feature=("gene", "ncRNA_gene"))

    assert genes[["gene_id", "gene_name", "gene_type"]].values.tolist() == [
        ["OR4F5", "OR4F5", "protein_coding"],
        ["ENSG2", "AB,C", "lncRNA"],
    ]
    assert len(read_gene_annotation(path, gene_types={"protein_coding"})) == 1


def test_read_bed_annotation_requires_strand(tmp_path):
    path = tmp_path / "genes.bed"
    path.write_text("chr1\t100\t200\tALPHA\t0\t+\tENSG1\n")
    assert read_gene_annotation(path)["gene_id"].tolist() == ["ENSG1"]
    # Track lines are skipped, names stay text, and BED12 column 7 is thickStart.
    path.write_text(
        "track name=genes\nchr1\t100\t200\tNA\t0\t+\t120\t180\t0\t1\t100,\t0,\n"
    )
    assert read_gene_annotation(path)[["gene_id", "gene_name"]].values.tolist() == [
        ["NA", "NA"]
    ]

    path.write_text("chr1\t100\t200\tALPHA\t0\t.\n")
    with pytest.raises(ValueError, match="strand"):
        read_gene_annotation(path)
    with pytest.raises(ValueError, match="no gene types"):
        read_gene_annotation(path, gene_types={"protein_coding"})


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("genes.gtf", "chr1\tH\tgene\t1\t10\n", "does not have 9 columns"),
        ("genes.gtf", 'chr1\tH\tgene\t1\t10\t.\t+\t.\tgene_name "A";\n', "identifier"),
        ("genes.gtf", 'chr1\tH\tgene\t1\t10\t.\t.\t.\tgene_id "G";\n', r"\+/- strand"),
        ("genes.gtf", 'chr1\tH\tgene\t10\t5\t.\t+\t.\tgene_id "G";\n', "interval"),
        # Only gene records are read by default.
        ("genes.gtf", 'chr1\tH\texon\t1\t10\t.\t+\t.\tgene_id "G";\n', "no genes"),
        ("genes.bed", "chr1\t100\t200\tALPHA\n", "at least six columns"),
        ("genes.bed", "chr1\t200\t100\tALPHA\t0\t+\n", "invalid interval"),
        ("genes.txt", "", "must be .gtf"),
    ],
)
def test_read_gene_annotation_rejects_malformed_files(tmp_path, name, content, match):
    path = tmp_path / name
    path.write_text(content)

    with pytest.raises(ValueError, match=match):
        read_gene_annotation(path)


def test_transcription_start_sites_follow_strand():
    genes = pd.DataFrame(
        {"start": [100, 1000], "end": [200, 1100], "strand": ["+", "-"]}
    )

    np.testing.assert_array_equal(transcription_start_sites(genes), [100, 1099])
    with pytest.raises(ValueError, match="strands must be"):
        transcription_start_sites(genes.assign(strand=["+", "."]))


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
    regions = ["chr1:890-900", "chr1:1050-1060", "chr1:5990-6000", "chr1:6050-6060"]

    links = link_regions_to_genes(
        regions + ["chr3:0-10"], _genes(), upstream=110, downstream=60
    )

    # G1 (+, TSS 1000): window 890..1060. G2 (-, TSS 5999): window 5939..6109.
    assert links.to_dict("list") == {
        "region": regions,
        "gene": ["A", "A", "B", "B"],
        "gene_id": ["G1", "G1", "G2", "G2"],
        "distance": [-101, 50, 0, -51],
    }
    # Regions named without the genes' "chr" prefix share no chromosome.
    with pytest.raises(ValueError, match="'chr' prefix"):
        link_regions_to_genes(["1:0-10"], _genes(), upstream=1, downstream=1)
    # No regions at all is not a naming mismatch.
    assert link_regions_to_genes([], _genes(), upstream=1, downstream=1).empty


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

    regions = [f"{c}:{s}-{e}" for c, s, e in zip(chroms, starts, ends, strict=True)]

    links = link_regions_to_genes(regions, genes, upstream=500, downstream=300)

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
                expected.add((regions[region], row["gene_id"]))
    assert set(zip(links["region"], links["gene_id"])) == expected


@pytest.mark.parametrize("kernel", ["jit", "python"])
def test_overlapping_intervals_matches_brute_force(kernel, monkeypatch):
    if kernel == "python":
        # Line coverage cannot see compiled code, so also run the Python
        # source of the overlap kernel on the same inputs.
        python = annotation._overlap_pairs.py_func
        monkeypatch.setattr(annotation, "_overlap_pairs", python)
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


def test_overlapping_intervals_rejects_empty_intervals():
    with pytest.raises(ValueError, match="start < end"):
        overlapping_intervals((["chr1"], [5], [5]), (["chr1"], [0], [10]))


def test_overlapping_intervals_on_different_chromosomes_share_nothing():
    found = overlapping_intervals((["chr1"], [0], [10]), (["chr2"], [0], [10]))

    assert [values.tolist() for values in found] == [[], [], []]


def test_link_regions_to_genes_with_gene_body_windows():
    genes = _genes()
    # G1 (+) body 1000..2000; G2 (-) body 5000..6000.
    regions = ["chr1:850-900", "chr1:1500-1510", "chr1:2050-2100", "chr1:6080-6100"]

    links = link_regions_to_genes(
        regions, genes, upstream=200, downstream=100, anchor="gene_body"
    )

    # Region 0 is 101 bases 5' of G1; region 1 is inside it; region 2 is 51
    # bases 3' of G1 (within 100). Region 3 is 81 bases 5' of minus-strand G2.
    assert links[["region", "gene_id", "distance"]].values.tolist() == [
        [regions[0], "G1", -101],
        [regions[1], "G1", 0],
        [regions[2], "G1", 51],
        [regions[3], "G2", -81],
    ]
    with pytest.raises(ValueError, match="anchor must be"):
        link_regions_to_genes(regions, genes, upstream=1, downstream=1, anchor="body")
