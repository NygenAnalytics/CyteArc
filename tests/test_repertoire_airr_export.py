import gzip
import io

import numpy as np
import pandas as pd
import pytest

from cytearc.writers import write_airr_rearrangements
from cytearc.writers.airr import AIRR_REQUIRED_FIELDS
from tests.test_repertoire_clonotypes_datastore import CLONAL_ROWS, _receptors
from tests.test_repertoire_import import (
    BCR_ROWS,
    CELLRANGER_HEADER,
    TCR_ROWS,
    _store,
    _write,
)


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def test_the_writer_puts_required_fields_first_and_renders_values(tmp_path) -> None:
    table = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "sequence_id": ["s1", "s2"],
            "productive": pd.array([True, None], dtype="boolean"),
            "umi_count": pd.array([3, None], dtype="Int64"),
            "v_identity": [0.95, np.nan],
            "junction_aa": pd.array(["CASSF", None], dtype="string"),
        }
    )
    path = tmp_path / "out.tsv"
    assert write_airr_rearrangements(table, path) == 2
    written = _read(path)
    assert written.columns.tolist() == [
        *AIRR_REQUIRED_FIELDS,
        "cell_id",
        "umi_count",
        "v_identity",
    ]
    assert written["productive"].tolist() == ["T", ""]
    assert written["umi_count"].tolist() == ["3", ""]
    assert written["v_identity"].tolist() == ["0.95", ""]
    assert written["junction_aa"].tolist() == ["CASSF", ""]
    assert written["sequence"].tolist() == ["", ""]
    chosen = tmp_path / "chosen.tsv.gz"
    assert (
        write_airr_rearrangements(table, chosen, columns=["umi_count", "junction_aa"])
        == 2
    )
    with gzip.open(chosen, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    assert header == [*AIRR_REQUIRED_FIELDS, "umi_count"]
    with pytest.raises(TypeError, match="path must be a path"):
        write_airr_rearrangements(table, 3)
    with pytest.raises(ValueError, match="need a sequence_id"):
        write_airr_rearrangements(table.drop(columns="sequence_id"), path)
    with pytest.raises(ValueError, match="have no column"):
        write_airr_rearrangements(table, path, columns=["c_call"])


def test_the_writer_reads_logical_text_and_whole_counts(tmp_path) -> None:
    # A table as pandas reads an AIRR file: logical fields as text, and counts
    # as real numbers when one is missing.
    table = pd.read_csv(
        io.StringIO(
            "sequence_id\tproductive\trev_comp\tumi_count\tv_identity\n"
            "c1\tT\tFALSE\t3\t0.95\n"
            "c2\tF\ttrue\t\t1\n"
            "c3\t\t0\t12\t1\n"
        ),
        sep="\t",
    )
    path = tmp_path / "read.tsv"
    write_airr_rearrangements(table, path)
    written = _read(path)
    assert written["productive"].tolist() == ["T", "F", ""]
    assert written["rev_comp"].tolist() == ["F", "T", "F"]
    assert written["umi_count"].tolist() == ["3", "", "12"]
    assert written["v_identity"].tolist() == ["0.95", "1.0", "1.0"]
    numbers = pd.DataFrame(
        {"sequence_id": ["c1", "c2"], "productive": [1, 0], "is_cell": [1.0, np.nan]}
    )
    write_airr_rearrangements(numbers, path)
    assert _read(path)[["productive", "is_cell"]].values.tolist() == [
        ["T", "T"],
        ["F", ""],
    ]
    with pytest.raises(ValueError, match="productive must hold logical values"):
        write_airr_rearrangements(table.assign(productive=["T", "maybe", "F"]), path)
    with pytest.raises(ValueError, match="productive must hold logical values"):
        write_airr_rearrangements(numbers.assign(productive=[2, 0]), path)


def test_export_writes_each_contig_with_its_cell_and_clonotype(tmp_path) -> None:
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    clonotypes = store.repertoire.clonotypes(receptors)
    path = tmp_path / "export.tsv"
    assert store.repertoire.export_airr(clonotypes, path) == len(CLONAL_ROWS)
    written = _read(path)
    assert written.columns[: len(AIRR_REQUIRED_FIELDS)].tolist() == list(
        AIRR_REQUIRED_FIELDS
    )
    assert written.columns[
        len(AIRR_REQUIRED_FIELDS) : len(AIRR_REQUIRED_FIELDS) + 5
    ].tolist() == [
        "cell_id",
        "clone_id",
        "locus",
        "c_call",
        "umi_count",
    ]
    by_contig = written.set_index("sequence_id")
    assert by_contig.loc["cell0_contig_1", "cell_id"] == "cell0"
    assert by_contig.loc["cell0_contig_1", "clone_id"] == "clonotype_1"
    assert by_contig.loc["cell5_contig_2", "clone_id"] == "clonotype_2"
    assert by_contig.loc["cell0_contig_1", "productive"] == "T"
    assert by_contig.loc["cell0_contig_1", "junction"] == "TGTGCTGTGGCC"
    plain = store.repertoire.export_airr(
        receptors, tmp_path / "plain.tsv", fields=["junction_aa", "v_call"]
    )
    assert plain == len(CLONAL_ROWS)
    columns = _read(tmp_path / "plain.tsv").columns.tolist()
    assert "clone_id" not in columns and "umi_count" not in columns
    assert columns[len(AIRR_REQUIRED_FIELDS) :] == ["cell_id"]


def test_export_repeats_round_trip_through_an_import(tmp_path) -> None:
    store = _store(tmp_path)
    cells = store.snapshot_cell_selection()
    tcr = _write(tmp_path, "tcr.csv", CELLRANGER_HEADER, TCR_ROWS)
    bcr = _write(tmp_path, "bcr.csv", CELLRANGER_HEADER, BCR_ROWS)
    receptors = store.repertoire.import_receptors([tcr, bcr], cell_selection=cells)
    path = tmp_path / "both.tsv.gz"
    written = store.repertoire.export_airr(receptors, path)
    exported = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert written == len(exported)
    # Contig IDs that the TCR and BCR files share get a suffix, so every
    # sequence_id is unique.
    assert exported["sequence_id"].is_unique
    again = store.repertoire.import_receptors(path, cell_selection=cells)
    first = store.repertoire.load_receptors(receptors).cells
    second = store.repertoire.load_receptors(again).cells
    for column in (
        "receptor_type",
        "receptor_subtype",
        "chain_pairing",
        "VDJ_1_junction_aa",
    ):
        pd.testing.assert_series_equal(first[column], second[column], check_names=False)


def test_export_suffixes_never_repeat_an_id(tmp_path) -> None:
    store = _store(tmp_path)
    # The TCR file already names a contig X_2, the suffix that the repeated X
    # of the BCR file would take.
    tcr = _write(
        tmp_path,
        "tcr.csv",
        CELLRANGER_HEADER,
        [
            "cell0,true,X,true,500,TRA,TRAV1,,TRAJ1,TRAC,true,true,CAVA,TGTGCTGTGGCC,90,9,ct1,ct1_1",
            "cell0,true,X_2,true,500,TRB,TRBV1,,TRBJ1,TRBC1,true,true,CASSA,TGTGCCAGCAGCGCC,80,8,ct1,ct1_2",
        ],
    )
    bcr = _write(
        tmp_path,
        "bcr.csv",
        CELLRANGER_HEADER,
        [
            "cell1,true,X,true,500,IGH,IGHV1,,IGHJ1,IGHM,true,true,CARW,TGTGCGAGATGG,500,20,cb1,cb1_1",
        ],
    )
    receptors = store.repertoire.import_receptors(
        [tcr, bcr], cell_selection=store.snapshot_cell_selection()
    )
    path = tmp_path / "ids.tsv"
    assert store.repertoire.export_airr(receptors, path) == 3
    assert sorted(_read(path)["sequence_id"]) == ["X", "X_2", "X_3"]


def test_export_keeps_only_chains_on_request(tmp_path) -> None:
    rows = CLONAL_ROWS + [
        "cell7,true,cell7_contig_1,true,500,TRB,TRBV9,,TRBJ9,TRBC1,true,false,CASSQ,TGTGCCAGCAGCCAG,10,1,ct9,ct9_1",
    ]
    store, receptors = _receptors(tmp_path, rows)
    everything = store.repertoire.export_airr(receptors, tmp_path / "all.tsv")
    chains = store.repertoire.export_airr(
        receptors, tmp_path / "chains.tsv", chains_only=True
    )
    assert everything == chains + 1
    assert "cell7_contig_1" not in set(_read(tmp_path / "chains.tsv")["sequence_id"])


def test_export_refuses_other_sources_and_arguments(tmp_path) -> None:
    store, receptors = _receptors(tmp_path, CLONAL_ROWS)
    clonotypes = store.repertoire.clonotypes(receptors)
    expansion = store.repertoire.clonal_expansion(clonotypes)
    with pytest.raises(ValueError, match="immune_receptors or clonotypes artifact"):
        store.repertoire.export_airr(expansion, tmp_path / "x.tsv")
    with pytest.raises(TypeError, match="source must be an ArtifactRef"):
        store.repertoire.export_airr("receptors", tmp_path / "x.tsv")
    with pytest.raises(TypeError, match="path must be a path"):
        store.repertoire.export_airr(receptors, 5)
    with pytest.raises(TypeError, match="chains_only must be a boolean"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", chains_only=1)
    with pytest.raises(ValueError, match="store no field"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", fields=["sequence"])
    with pytest.raises(TypeError, match="fields must be a sequence"):
        store.repertoire.export_airr(receptors, tmp_path / "x.tsv", fields="v_call")
