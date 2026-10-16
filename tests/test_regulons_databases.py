import os
import sys

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cytearc.regulons.databases import RankingDatabase, read_motif_annotations  # noqa: E402


def _write_database(path, rankings: np.ndarray, genes: list[str], motifs: list[str]):
    columns = {
        gene: rankings[:, index].astype(np.int32) for index, gene in enumerate(genes)
    }
    columns["motifs"] = motifs
    pl.DataFrame(columns).write_ipc(path, compression="zstd")


def test_ranking_database_reads_only_requested_columns(tmp_path):
    rng = np.random.default_rng(0)
    genes = [f"G{index}" for index in range(50)]
    motifs = [f"m{index}" for index in range(8)]
    rankings = np.vstack([rng.permutation(50) for _ in motifs])
    path = tmp_path / "test.genes_vs_motifs.rankings.feather"
    _write_database(path, rankings, genes, motifs)

    database = RankingDatabase(path)
    names, ranks = database.ranks(["G7", "G2", "missing", "G2", "G40"])

    assert database.total == 50
    assert database.indexColumn == "motifs"
    assert database.rowNames.tolist() == motifs
    assert "G7" in database and "missing" not in database
    assert repr(database) == (
        "RankingDatabase('test.genes_vs_motifs.rankings.feather', rows=8, features=50)"
    )
    assert names.tolist() == ["G2", "G7", "G40"]
    np.testing.assert_array_equal(ranks, rankings[:, [2, 7, 40]])


def test_ranking_database_returns_an_empty_block_without_matches(tmp_path):
    path = tmp_path / "db.feather"
    _write_database(path, np.array([[0, 1], [1, 0]]), ["A", "B"], ["m0", "m1"])

    names, ranks = RankingDatabase(path).ranks(["Z"])

    assert names.tolist() == []
    assert ranks.shape == (2, 0)


def test_ranking_database_rejects_feather_v1_and_missing_index(tmp_path):
    legacy = tmp_path / "legacy.feather"
    legacy.write_bytes(b"FEA1" + b"\0" * 20 + b"FEA1")
    with pytest.raises(ValueError, match="Feather version 2"):
        RankingDatabase(legacy)

    no_index = tmp_path / "no_index.feather"
    pl.DataFrame({"A": [0, 1], "B": [1, 0]}).write_ipc(no_index)
    with pytest.raises(ValueError, match="index column"):
        RankingDatabase(no_index)

    # A transposed file names its rows by region, not by motif.
    transposed = tmp_path / "motifs_vs_regions.feather"
    pl.DataFrame({"m0": [0, 1], "m1": [1, 0], "regions": ["r0", "r1"]}).write_ipc(
        transposed
    )
    with pytest.raises(ValueError, match="index column"):
        RankingDatabase(transposed)

    # Scores fall inside the rank range when there are many features.
    scores = tmp_path / "genes_vs_motifs.scores.feather"
    columns = {f"G{i}": [1.5, 0.25] for i in range(10)} | {"motifs": ["m0", "m1"]}
    pl.DataFrame(columns).write_ipc(scores)
    with pytest.raises(ValueError, match="scores database"):
        RankingDatabase(scores)


def test_ranking_database_rejects_missing_and_single_motif_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        RankingDatabase(tmp_path / "missing.feather")

    single = tmp_path / "single.feather"
    _write_database(single, np.array([[0, 1]]), ["A", "B"], ["m0"])
    with pytest.raises(ValueError, match="at least two motifs"):
        RankingDatabase(single)


def test_ranking_database_validates_the_ranks_it_reads(tmp_path):
    path = tmp_path / "db.feather"
    _write_database(path, np.array([[0, 1], [1, 3]]), ["A", "B"], ["m0", "m1"])
    database = RankingDatabase(path)
    with pytest.raises(ValueError, match=r"ranks outside 0\.\.2"):
        database.ranks(["B"])

    # A file replaced after opening no longer aligns with the motifs read.
    replacement = tmp_path / "replacement.feather"
    _write_database(replacement, np.zeros((3, 2)), ["A", "B"], ["m0", "m1", "m2"])
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="misaligned"):
        database.ranks(["A"])


def test_ranking_database_explains_how_to_install_polars(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", None)

    with pytest.raises(ImportError, match=r"install cytearc\[regulons\]"):
        RankingDatabase(tmp_path / "db.feather")


_TBL = (
    "#motif_id\tmotif_name\tmotif_description\tsource_name\tsource_version\tgene_name\t"
    "motif_similarity_qvalue\tsimilar_motif_id\tsimilar_motif_description\t"
    "orthologous_identity\torthologous_gene_name\torthologous_species\tdescription\n"
    "m1\tn\td\ts\tv\tSPI1\t0.0\tNone\tNone\t1.0\tNone\tNone\tgene is directly annotated\n"
    "m2\tn\td\ts\tv\tSPI1\t0.0005\tm9\tNone\t1.0\tNone\tNone\t"
    "gene is annotated for similar motif m9 ('x'; q-value = 0.0005)\n"
    "m3\tn\td\ts\tv\tPAX5\t0.0\tNone\tNone\t0.4\tpax\tD. melanogaster\t"
    "gene is orthologous to FBgn1 in D. melanogaster (identity = 40%) which is "
    "directly annotated for motif\n"
    "m4\tn\td\ts\tv\tEBF1\t0.002\tm8\tNone\t1.0\tNone\tNone\t"
    "gene is annotated for similar motif m8 ('y'; q-value = 0.002)\n"
    "m5\tn\td\ts\tv\tEBF1\t0.0001\tm8\tNone\t0.6\tebf\tM. musculus\t"
    "motif similar to m8 ('y'; q-value = 0.0001) which is annotated for "
    "orthologous gene ebf\n"
)


def test_read_motif_annotations_filters_and_classifies(tmp_path):
    path = tmp_path / "motifs.tbl"
    path.write_text(_TBL)

    table = read_motif_annotations(path)

    assert table["motif"].tolist() == ["m1", "m2", "m3", "m5"]
    assert table["category"].tolist() == [
        "direct",
        "similarity",
        "orthology",
        "similarity_and_orthology",
    ]
    assert table.columns.tolist() == [
        "motif",
        "TF",
        "motif_similarity_qvalue",
        "orthologous_identity",
        "description",
        "category",
    ]
    direct = read_motif_annotations(path, categories=["direct"])
    assert direct["motif"].tolist() == ["m1"]
    strict = read_motif_annotations(path, orthologous_identity_threshold=0.5)
    assert strict["motif"].tolist() == ["m1", "m2", "m5"]
    with pytest.raises(ValueError, match="Unknown annotation categories"):
        read_motif_annotations(path, categories=["bogus"])
    assert isinstance(table, pd.DataFrame)
