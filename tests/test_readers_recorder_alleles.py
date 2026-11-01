"""Reading the allele files that recorder pipelines publish.

Real files guard the readers against each format's quirks. Checked in under
``tests/data/recorders/``: CARLIN's single-cell run FO864 LF and the first
lines of a bulk run (Zenodo 4817818, CC BY 4.0), DARLIN's skull library on
its three arrays (Zenodo 11929508, CC BY 4.0, a subset), darlin-core's
notation benchmark (MIT), a subset of Cassiopeia's 3432_NT_T1 allele table
(MIT), PEtracer's E7.5-R1 characters (MIT) and subsets of three of its 2025
allele tables (figshare 28473866, CC BY 4.0), a subset of DNA Typewriter's
2022 table (CC BY 4.0) and ConvexML's character matrix (BSD-3-Clause). Every
expectation is counted from the raw files by the format's documented rules,
or taken from the files' publications.
"""

import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cytearc.clones.recorders import STATES
from cytearc.readers.recorder_alleles import (
    ALLELE_FORMATS,
    allele_file,
    allele_layout,
    classified_rows,
    iter_allele_rows,
)
from tests.clone_truth import PETRACER_MARKS

DATA = Path(__file__).parent / "data" / "recorders"
_CONTRACTED = {"TGGACGAC": 4, "TTTCGTGA": 4, "TGGTTTTG": 4, "TTCACGTA": 2}


def read_all(path, **options) -> pd.DataFrame:
    """Every site of a file: its row's barcode, integration, site, state and allele."""
    layout = allele_layout(allele_file(path), **options)
    frames = []
    for rows in iter_allele_rows(layout, 1_000):
        classified = classified_rows(layout, rows)
        n_columns = len(rows.columns)
        integrations = (
            np.repeat(rows.integrations, n_columns)
            if rows.integrations is not None
            else np.tile([name for name, _site in rows.columns], len(rows))
        )
        frames.append(
            pd.DataFrame(
                {
                    "barcode": np.repeat(rows.barcodes, n_columns),
                    "row": np.repeat(rows.rows, n_columns),
                    "integration": integrations,
                    "site": np.tile([site for _name, site in rows.columns], len(rows)),
                    "state": np.asarray(STATES, dtype=object)[
                        classified.states.reshape(-1)
                    ],
                    "allele": pd.Series(classified.alleles.reshape(-1), dtype=object),
                    "umis": np.repeat(
                        rows.umis if rows.umis is not None else np.full(len(rows), -1),
                        n_columns,
                    ),
                    "reads": np.repeat(
                        rows.reads
                        if rows.reads is not None
                        else np.full(len(rows), -1),
                        n_columns,
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_carlins_single_cell_run_gives_each_cell_its_lines_allele():
    folder = DATA / "carlin_fo864_lf"
    annotations = (folder / "AlleleAnnotations.txt").read_text().split("\n")
    colonies = (folder / "AlleleColonies.txt").read_text().split("\n")
    expected = {
        barcode: allele
        for allele, line in zip(annotations, colonies, strict=True)
        for barcode in line.split(",")
    }
    read = read_all(folder, integrations="CARLIN")
    # 864 called cells, 517 of them unedited, as Results.txt and the paper's
    # deposit record.
    assert len(read) == len(expected) == 864
    assert (
        dict(zip(read["barcode"], read["allele"].fillna("[]"), strict=True)) == expected
    )
    assert (read["state"] == "unedited").sum() == 517
    assert set(read["state"]) == {"unedited", "mark"}
    assert set(read["integration"]) == {"CARLIN"}
    assert set(read["site"]) == {""}
    # Each barcode's row is the line of its allele, counting from 1.
    lines = {
        barcode: number
        for number, line in enumerate(colonies, start=1)
        for barcode in line.split(",")
    }
    assert dict(zip(read["barcode"], read["row"], strict=True)) == lines


def test_either_file_of_a_carlin_pair_or_its_folder_names_the_pair():
    folder = DATA / "carlin_fo864_lf"
    colonies = str(folder / "AlleleColonies.txt")
    assert allele_file(folder) == colonies
    assert allele_file(folder / "AlleleAnnotations.txt") == colonies
    assert allele_file(folder / "AlleleColonies.txt") == str(
        folder / "AlleleColonies.txt"
    )


def test_carlins_bulk_runs_list_umis_and_are_refused(tmp_path):
    with pytest.raises(ValueError, match="tags of 10 bases"):
        allele_layout(allele_file(DATA / "carlin_bulk_24h"), integrations="CARLIN")
    # Results.txt names the run, so a bulk run is refused by its type and a
    # single-cell run of short barcodes is read.
    for config, refused in (("BulkRNA", True), ("sc10xV3", False)):
        folder = tmp_path / config
        folder.mkdir()
        for name in ("AlleleAnnotations.txt", "AlleleColonies.txt"):
            (folder / name).write_bytes((DATA / "carlin_bulk_24h" / name).read_bytes())
        (folder / "Results.txt").write_text(f"INPUT\n\nCfgType:          {config}\n")
        if refused:
            with pytest.raises(ValueError, match="of type 'BulkRNA'"):
                allele_layout(allele_file(folder), integrations="CARLIN")
        else:
            colonies = (folder / "AlleleColonies.txt").read_text().split("\n")
            assert len(read_all(folder, integrations="CARLIN")) == sum(
                len(line.split(",")) for line in colonies
            )


def test_darlins_three_arrays_share_the_skull_librarys_cells():
    cells = {}
    for array in ("CA", "TA", "RA"):
        folder = DATA / f"darlin_ll837_skull_{array}"
        annotations = (folder / "AlleleAnnotations.txt").read_text().split("\n")
        colonies = (folder / "AlleleColonies.txt").read_text().split("\n")
        expected = {
            barcode: allele
            for allele, line in zip(annotations, colonies, strict=True)
            for barcode in line.split(",")
        }
        read = read_all(folder, integrations=array)
        assert (
            dict(zip(read["barcode"], read["allele"].fillna("[]"), strict=True))
            == expected
        )
        # Each array's Results.txt names a 10x run.
        assert (read["state"] == "unedited").sum() == sum(
            1 for allele in expected.values() if allele == "[]"
        )
        cells[array] = set(expected)
    # The subset keeps 60 cells read on all three arrays and 10 on each alone.
    assert len(cells["CA"] & cells["TA"] & cells["RA"]) == 60
    assert {len(found) for found in cells.values()} == {70}


def test_darlin_core_notation_reads_as_alleles(tmp_path):
    # darlin-core's benchmark pairs its own notation with MATLAB CARLIN's;
    # both must read as the alleles of a darlin table.
    benchmark = pd.read_csv(DATA / "darlin_core_CA_benchmark.tsv", sep="\t", dtype=str)
    alleles = pd.concat([benchmark["mutations"], benchmark["truth"]]).tolist()
    lines = ["n_UMIs\tCR\tmutation\taligned_LR\taligned_ref"] + [
        f"3\tAAAC{index:012d}\t{allele}\tA\tA" for index, allele in enumerate(alleles)
    ]
    path = tmp_path / "step4_final.tsv"
    path.write_text("\n".join(lines) + "\n")
    read = read_all(path, integrations="CA")
    assert read["allele"].tolist() == alleles
    assert set(read["state"]) == {"mark"}
    assert (read["umis"] == 3).all() and (read["reads"] == -1).all()


def test_cassiopeias_allele_table_reads_none_as_uncut_and_nc_as_missing():
    raw = pd.read_csv(
        DATA / "cassiopeia_3432_NT_T1_alleletable.txt",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    read = read_all(DATA / "cassiopeia_3432_NT_T1_alleletable.txt")
    assert len(read) == 3 * len(raw)
    for site in ("r1", "r2", "r3"):
        tokens = raw[site]
        found = read[read["site"] == site]
        assert (found["state"] == "unedited").sum() == tokens.str.contains(
            "None", regex=False
        ).sum()
        assert (found["state"] == "missing").sum() == (tokens == "NC").sum()
        marks = found[found["state"] == "mark"]
        assert sorted(marks["allele"]) == sorted(
            token for token in tokens if "None" not in token and token != "NC"
        )
    # The table's NC sites are why its cells were chosen.
    assert (
        (read["state"] == "missing").sum()
        == (raw[["r1", "r2", "r3"]] == "NC").sum().sum()
        > 0
    )
    assert sorted(read["umis"].unique()) == sorted(raw["UMI"].astype(int).unique())
    assert set(read["integration"]) == set(raw["intBC"])


def test_petracers_e75_characters_read_as_its_tokens_say():
    with gzip.open(DATA / "petracer_E7.5-R1_characters.csv.gz", "rt") as handle:
        raw = pd.read_csv(handle, dtype=str, keep_default_na=False)
    read = read_all(DATA / "petracer_E7.5-R1_characters.csv.gz")
    states = read["state"].value_counts().to_dict()
    # 1,342 cells by 33 integrations of three sites, whose tokens count
    # 51,371 '*', 32,403 '-', 49,059 marks and 25 '!'.
    assert len(raw) == 1342 and len(read) == 1342 * 99
    assert states == {
        "unedited": 51_371,
        "missing": 32_403,
        "mark": 49_059,
        "unknown_mark": 25,
    }
    assert set(read.loc[read["state"] == "mark", "allele"]) == set("12345678")
    assert read["integration"].nunique() == 33
    assert set(read["site"]) == {"EMX1", "HEK3", "RNF2"}
    # An integration's sites are missing together, as the data has them.
    missing = (
        read[read["state"] == "missing"].groupby(["barcode", "integration"]).size()
    )
    assert set(missing) == {3}


@pytest.mark.parametrize(
    "name",
    [
        "petracer_barcoded_tracing_alleles.csv",
        "petracer_B16F10_kinetics_alleles.csv",
        "petracer_preedited_10x_invitro_alleles.csv",
    ],
)
def test_petracers_2025_tables_name_insertions_outside_the_marks_unknown(name):
    raw = pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
    read = read_all(DATA / name)
    assert len(read) == 3 * len(raw)
    for site in ("RNF2", "HEK3", "EMX1"):
        tokens = raw[site]
        found = read[read["site"] == site].reset_index(drop=True)
        known = tokens.isin(PETRACER_MARKS[site])
        unedited = (tokens == "None") | ((tokens == "CTTGGG") & (site == "EMX1"))
        empty = tokens == ""
        assert (found["state"] == "mark").sum() == known.sum()
        assert (found["state"] == "unedited").sum() == unedited.sum()
        assert (found["state"] == "missing").sum() == empty.sum()
        assert (found["state"] == "unknown_mark").sum() == (
            ~known & ~unedited & ~empty
        ).sum()
    # Every row that names an integration is kept, two alleles of one cell and
    # integration included; PEtracer leaves intID empty off its whitelist.
    named = raw[raw["intID"] != ""] if "intID" in raw else raw
    assert read.dropna(subset=["integration"]).groupby(
        ["barcode", "integration", "row"]
    ).ngroups == len(named)
    assert read["integration"].isna().sum() == 3 * (len(raw) - len(named))
    assert sorted(read["umis"].unique()) == sorted(raw["UMI"].astype(int).unique())


def test_dna_typewriters_2022_table_reads_its_ragged_and_contracted_rows():
    path = DATA / "typewriter_2022_hek293t.csv"
    rows = list(csv.reader(path.open(newline="")))[1:]
    read = read_all(path, integrations=_CONTRACTED)
    assert len(read) == 5 * len(rows)
    # 16 rows hold a sixth, empty slot field, as the published table does.
    assert sum(len(row) == 9 for row in rows) == 16
    expected = Counter()
    beyond = 0
    for row in rows:
        slots = row[2:7]
        length = _CONTRACTED.get(row[1], 5)
        edited = [position for position, slot in enumerate(slots[:length]) if slot]
        last = edited[-1] if edited else -1
        for position, slot in enumerate(slots):
            if position >= length:
                expected["not_applicable"] += 1
                beyond += bool(slot)
            elif slot:
                expected["mark"] += 1
            elif position < last:
                # A blank slot before an edit was edited, but not read.
                expected["missing"] += 1
            else:
                expected["unedited"] += 1
    assert read["state"].value_counts().to_dict() == dict(expected)
    assert expected["missing"] > 0 and beyond > 0
    layout = allele_layout(path, integrations=_CONTRACTED)
    found = sum(
        classified_rows(layout, chunk).n_beyond
        for chunk in iter_allele_rows(layout, 100)
    )
    assert found == beyond
    assert (read["umis"] > 0).all()


def test_convexmls_character_matrix_names_one_character_per_column():
    raw = pd.read_csv(
        DATA / "convexml_tree_0_character_matrix.csv", dtype=str, keep_default_na=False
    )
    read = read_all(DATA / "convexml_tree_0_character_matrix.csv", format="characters")
    values = raw.iloc[:, 1:].to_numpy().reshape(-1)
    assert len(read) == len(values) == 400 * 39
    assert (read["state"] == "unedited").sum() == (values == "0").sum()
    assert (read["state"] == "missing").sum() == (values == "-1").sum()
    assert (read["state"] == "mark").sum() == ((values != "0") & (values != "-1")).sum()
    # Columns without a hyphen are integrations of one unnamed site.
    assert set(read["integration"]) == {str(column) for column in range(39)}
    assert set(read["site"]) == {""}


# The meaning of None, NA and empty text, format by format: a state, or None
# where the format refuses the token.
_TOKENS = {
    "carlin": {"None": None, "NA": None},
    "darlin": {"None": None, "NA": None, "": "unedited"},
    "cassiopeia": {"None": "unedited", "NA": "missing", "": "missing"},
    "petracer": {"None": "unedited", "NA": "missing", "": "missing"},
    "petracer_characters": {"None": None, "NA": None, "": "not_applicable"},
    "typewriter": {"None": None, "NA": "missing", "": "unedited"},
    "typewriter_chains": {"None": None, "NA": "missing", "": "missing"},
    "treedata": {"None": None, "NA": "missing"},
    "characters": {"None": None, "NA": "missing", "": None},
    "table": {"None": "missing", "NA": "missing", "": "missing"},
}


def _one_site_file(format: str, token: str, folder: Path) -> tuple[Path, dict]:
    """A file of one cell whose first site holds ``token``, and its options."""
    if format == "carlin":
        (folder / "AlleleAnnotations.txt").write_text(f"[]\n{token}")
        (folder / "AlleleColonies.txt").write_text("AAACCCAAGTGAACAT\nTTTCCCAAGTGAACAT")
        return folder, {"integrations": "CA"}
    if format == "darlin":
        path = folder / "step4_final.tsv"
        path.write_text(
            f"n_UMIs\tCR\tmutation\taligned_LR\taligned_ref\n2\tAAAC\t{token}\tA\tA\n"
        )
        return path, {"integrations": "CA"}
    if format == "cassiopeia":
        path = folder / "alleles.txt"
        path.write_text(
            f"cellBC\tintBC\tr1\tr2\tUMI\treadCount\nc1\ti1\t{token}\tAC[None]GT\t3\t30\n"
        )
        return path, {}
    if format == "petracer":
        path = folder / "alleles.csv"
        path.write_text(
            f"intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount\ni1,c1,{token},None,None,3,30\n"
        )
        return path, {}
    if format == "petracer_characters":
        path = folder / "characters.csv"
        path.write_text(f"cell_barcode,i1-RNF2,i1-HEK3\nc1,{token},*\n")
        return path, {}
    if format == "typewriter":
        path = folder / "tapes.csv"
        path.write_text(
            f"Cell,TargetBC,Site1,Site2,Site3,Site4,Site5,nUMI\nc1,T1,{token},,,,,3\n"
        )
        return path, {}
    if format == "typewriter_chains":
        path = folder / "consensus.tsv"
        path.write_text(f"cell\tACGTACGTACGT\tn_loci\nc1\t{token}\t1\n")
        return path, {}
    if format == "characters":
        path = folder / "matrix.csv"
        path.write_text(f"cell,i1-a,i1-b\nc1,{token},0\n")
        return path, {"format": "characters"}
    if format == "table":
        path = folder / "table.tsv"
        path.write_text(f"cell\tallele\nc1\t{token}\n")
        return path, {
            "columns": {"cell": "cell", "allele": "allele"},
            "integrations": "X",
        }
    raise AssertionError(format)


@pytest.mark.parametrize(
    ("format", "token", "meaning"),
    [
        (format, token, meaning)
        for format, tokens in _TOKENS.items()
        if format != "treedata"
        for token, meaning in tokens.items()
    ],
)
def test_none_na_and_empty_text_mean_what_each_format_says(
    tmp_path, format, token, meaning
):
    path, options = _one_site_file(format, token, tmp_path)
    if meaning is None:
        with pytest.raises(
            ValueError, match="tokens that|empty line|holds no|is no chain"
        ):
            read_all(path, **options)
        return
    read = read_all(path, **options)
    if format == "carlin":
        read = read[read["row"] == 2]
    first = read.iloc[0]
    assert first["state"] == meaning
    assert first["allele"] is None


@pytest.mark.parametrize(
    ("format", "token"),
    [
        # PEtracer's GEO tables of 2025 hold such reads at EMX1 and HEK3.
        ("petracer", "GTNAG"),
        ("petracer", "NTTGGG"),
        ("petracer", "CTTNGG"),
        ("typewriter", "ACNTTA"),
        ("typewriter_chains", "AAG-CNC"),
    ],
)
def test_insertions_that_hold_n_are_missing(tmp_path, format, token):
    """An N is an ambiguous base, so the insertion names no allele, edited or not."""
    path, options = _one_site_file(format, token, tmp_path)
    read = read_all(path, **options)
    if format == "typewriter_chains":
        # The chain's first symbol is read; its second, which holds N, is not.
        assert read["state"].tolist()[:3] == ["mark", "missing", "unedited"]
        assert read["allele"].tolist()[:2] == ["AAG", None]
        return
    first = read.iloc[0]
    assert first["state"] == "missing" and first["allele"] is None
    # Other text that holds N is still no insertion.
    path, options = _one_site_file(format, token.replace("N", "Q"), tmp_path)
    with pytest.raises(ValueError, match="tokens that"):
        read_all(path, **options)


_CELLS = ("AAACCCAAGTGAACAT", "CCCGCCAAGTGAACAT", "GGGTCCAAGTGAACAT")


def test_carlins_empty_line_leaves_its_cells_allele_unknown(tmp_path):
    (tmp_path / "AlleleAnnotations.txt").write_text("[]\n\n23_268del")
    (tmp_path / "AlleleColonies.txt").write_text("\n".join(_CELLS))
    with pytest.raises(ValueError, match="empty line 2"):
        read_all(tmp_path, integrations="CA")


def _treedata(path: Path, characters, **uns) -> Path:
    """An AnnData file holding ``characters`` in obsm, as treedata writes .h5td."""
    anndata = pytest.importorskip("anndata")
    cells = characters.index if hasattr(characters, "index") else None
    data = anndata.AnnData(
        obs=pd.DataFrame(index=cells if cells is not None else ["c1", "c2"])
    )
    data.obsm["characters"] = characters
    data.uns.update(uns)
    data.write_h5ad(path)
    return path


def test_treedata_reads_its_states_from_uns_or_cassiopeias_defaults(tmp_path):
    characters = pd.DataFrame(
        {"i1-a": ["*", "-", "NA", "3"], "i1-b": ["*", "*", "5", "None"]},
        index=pd.Index(["c1", "c2", "c3", "c4"]),
    )
    _treedata(tmp_path / "defaults.h5td", characters.iloc[:3])
    read = read_all(tmp_path / "defaults.h5td")
    assert read.groupby("site")["state"].apply(list).to_dict() == {
        "a": ["unedited", "missing", "missing"],
        "b": ["unedited", "unedited", "mark"],
    }
    layout = allele_layout(tmp_path / "defaults.h5td")
    assert layout.source()["states_from"] == "Cassiopeia's defaults"
    # A file that records its states reads by them, and None is no state.
    states = {"missing_state": "NA", "unmodified_state": "*"}
    _treedata(tmp_path / "none.h5td", characters, **states)
    with pytest.raises(ValueError, match="'None' at b"):
        read_all(tmp_path / "none.h5td")
    _treedata(tmp_path / "declared.h5td", characters.iloc[:, :1], **states)
    # '-' is no state that this file names, so it is a mark here.
    assert read_all(tmp_path / "declared.h5td")["state"].tolist() == [
        "unedited",
        "mark",
        "missing",
        "mark",
    ]
    assert allele_layout(tmp_path / "declared.h5td").source()["states_from"] == "uns"
    _treedata(tmp_path / "half.h5td", characters.iloc[:, :1], missing_state="NA")
    with pytest.raises(ValueError, match="names both or neither"):
        allele_layout(tmp_path / "half.h5td")


def test_treedata_reads_integer_states_and_a_plain_array(tmp_path):
    cells = pd.Index(["c1", "c2"])
    _treedata(
        tmp_path / "integers.h5td",
        pd.DataFrame(
            {"i1-a": np.array([0, 4]), "i2-a": np.array([-1, 1.0])}, index=cells
        ),
    )
    read = read_all(tmp_path / "integers.h5td")
    assert read["state"].tolist() == ["unedited", "missing", "mark", "mark"]
    assert read["allele"].tolist() == [None, None, "4", "1"]
    # An array without names gives no integrations to read.
    _treedata(tmp_path / "array.h5td", np.array([[0, 2], [-1, 0]]))
    with pytest.raises(ValueError, match="an array whose columns have no names"):
        read_all(tmp_path / "array.h5td")
    _treedata(
        tmp_path / "fractions.h5td", pd.DataFrame({"i1-a": [0.5, 1.0]}, index=cells)
    )
    with pytest.raises(ValueError, match="not whole numbers"):
        read_all(tmp_path / "fractions.h5td")
    anndata = pytest.importorskip("anndata")
    anndata.AnnData(obs=pd.DataFrame(index=cells)).write_h5ad(tmp_path / "none.h5td")
    with pytest.raises(ValueError, match=r"holds no obsm\['characters'\]"):
        allele_layout(tmp_path / "none.h5td")


def test_petracers_published_treedata_reads_its_names_and_tokens(tmp_path):
    """PEtracer's Zenodo files hold the characters of 2026 as an array of text.

    The columns' names are listed in uns['embryo_characters'], and there are
    no states in uns, so the tokens are those of the character matrices.
    """
    names = ["intID4-EMX1", "intID4-HEK3", "intID101-EMX1"]
    tokens = np.array([["*", "3", "!"], ["-", "", "8"]], dtype=object)
    _treedata(tmp_path / "embryo.h5td", tokens, embryo_characters=np.array(names))
    read = read_all(tmp_path / "embryo.h5td")
    assert read["integration"].tolist() == ["intID4", "intID4", "intID101"] * 2
    assert read["site"].tolist() == ["EMX1", "HEK3", "EMX1"] * 2
    assert read["state"].tolist() == [
        "unedited",
        "mark",
        "unknown_mark",
        "missing",
        "not_applicable",
        "mark",
    ]
    # "!" is a placeholder, which names no allele.
    assert read["allele"].tolist() == [None, "3", None, None, None, "8"]
    layout = allele_layout(tmp_path / "embryo.h5td")
    assert (layout.tool, layout.pipeline) == ("PEtracer", "PEtracer 2026")
    assert layout.source()["names_from"] == "uns['embryo_characters']"
    # The listing must name each column once.
    _treedata(tmp_path / "short.h5td", tokens, embryo_characters=np.array(names[:2]))
    with pytest.raises(ValueError, match="names 2 characters for the 3 columns"):
        read_all(tmp_path / "short.h5td")
    _treedata(
        tmp_path / "twice.h5td", tokens, embryo_characters=np.array(names[:2] * 2)[:3]
    )
    with pytest.raises(ValueError, match="names a character twice"):
        read_all(tmp_path / "twice.h5td")


def _parquet(path: Path, frame, sites=None, **attrs) -> Path:
    """A table as Cassiopeia 3.0 saves it with pandas' to_parquet.

    pandas saves a table's attrs as JSON under the schema key PANDAS_ATTRS
    and its own layout under "pandas", which holds no attrs; Cassiopeia's
    sites and the cells and intBCs are categorical.
    """
    polars = pytest.importorskip("polars")
    if sites is not None:
        attrs["sites"] = sites
    metadata = {
        "pandas": json.dumps({"index_columns": [], "columns": []}),
        **({"PANDAS_ATTRS": json.dumps(attrs)} if attrs else {}),
    }
    categorical = [*(sites or ["r1", "r2"]), "cellBC", "intBC"]
    polars.DataFrame(frame).with_columns(
        [
            polars.col(name).cast(polars.Categorical)
            for name in categorical
            if name in frame
        ]
    ).write_parquet(path, metadata=metadata)
    return path


def test_cassiopeia_parquet_reads_passing_rows_and_null_as_missing(tmp_path):
    path = _parquet(
        tmp_path / "x.alleles.parquet",
        {
            "cellBC": ["c1", "c1", "c2", "c3"],
            "intBC": ["i1", "i2", "i1", "i1"],
            "r1": ["CG[None]AA", "CG[1:2D]AA", None, "NA"],
            "r2": ["None", "None", "CG[3:1I]AA", "CG[3:1I]AA"],
            "n_umis": [5, 4, 3, 2],
            "n_reads": [50, 40, 30, 20],
            "pass": [True, True, True, False],
        },
        sites=["r1", "r2"],
    )
    layout = allele_layout(path)
    assert (
        layout.format == "cassiopeia_parquet"
        and layout.source()["sites_from"] == "attrs"
    )
    read = read_all(path)
    # The fourth row fails Cassiopeia's filters.
    assert read["barcode"].tolist() == ["c1", "c1", "c1", "c1", "c2", "c2"]
    assert read["state"].tolist() == [
        "unedited",
        "unedited",
        "mark",
        "unedited",
        "missing",
        "mark",
    ]
    assert read["row"].tolist() == [1, 1, 2, 2, 3, 3]
    assert read["reads"].tolist() == [50, 50, 40, 40, 30, 30]
    # Without its attributes, the categorical columns that no role names are
    # sites, as Cassiopeia's own rule takes them, and text columns are not.
    unnamed = _parquet(
        tmp_path / "y.alleles.parquet",
        {
            "cellBC": ["c1"],
            "intBC": ["i1"],
            "r1": ["None"],
            "r2": ["A"],
            "note": ["x"],
            "n_umis": [1],
        },
    )
    layout = allele_layout(unnamed)
    assert layout.source()["sites_from"] == "columns"
    assert layout.columns["sites"] == ["r1", "r2"]
    samples = _parquet(
        tmp_path / "z.alleles.parquet",
        {
            "sample": ["a", "b"],
            "cellBC": ["c1", "c2"],
            "intBC": ["i1", "i1"],
            "r1": ["None", "A"],
            "r2": ["A", "A"],
        },
    )
    with pytest.raises(ValueError, match="holds 2 samples"):
        allele_layout(samples)


def test_cassiopeia_parquet_reads_known_insertions_by_their_lookup(tmp_path):
    """PEtracer's sites in Cassiopeia 3.0 are insertions of known sequences.

    A label of the lookup is a mark, None unedited, an insertion outside the
    lookup its sequence and a deletion D and its length, both unknown marks
    of their own text, and an insertion that holds N or an empty site
    missing.
    """
    labels = [str(label) for label in range(1, 9)]
    path = _parquet(
        tmp_path / "p.alleles.parquet",
        {
            "cellBC": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "intBC": ["i1"] * 6,
            "RNF2": ["1", "None", "ACGTA", "D6", "D6IAC", "GTNAG"],
            "HEK3": ["8", "None", "None", "", None, "None"],
            "n_umis": [5, 4, 3, 2, 2, 1],
            "n_reads": [50, 40, 30, 20, 20, 10],
        },
        sites=["RNF2", "HEK3"],
        edit_models={"RNF2": "insertion_lookup", "HEK3": "insertion_lookup"},
        site_states={"RNF2": labels, "HEK3": labels},
    )
    layout = allele_layout(path)
    assert layout.rules.kind == "independent"
    read = read_all(path)
    assert read["state"].tolist() == [
        "mark",
        "mark",
        "unedited",
        "unedited",
        "unknown_mark",
        "unedited",
        "unknown_mark",
        "missing",
        "unknown_mark",
        "missing",
        "missing",
        "unedited",
    ]
    assert read["allele"].tolist()[::2] == ["1", None, "ACGTA", "D6", "D6IAC", None]
    # A label that the lookup does not list is refused.
    other = _parquet(
        tmp_path / "q.alleles.parquet",
        {"cellBC": ["c1"], "intBC": ["i1"], "RNF2": ["9"], "n_umis": [1]},
        sites=["RNF2"],
        edit_models={"RNF2": "insertion_lookup"},
        site_states={"RNF2": labels},
    )
    with pytest.raises(ValueError, match="'9' at RNF2"):
        read_all(other)


def test_cassiopeia_parquet_reads_tape_slots_in_order(tmp_path):
    path = _parquet(
        tmp_path / "t.alleles.parquet",
        {
            "cellBC": ["c1", "c2"],
            "intBC": ["i1", "i1"],
            "slot1": ["AAGTCA", "None"],
            "slot2": [None, "None"],
            "slot3": ["CCTAGA", None],
            "n_umis": [3, 2],
        },
        sites=["slot1", "slot2", "slot3"],
        edit_models={
            site: "sequential_insertion" for site in ("slot1", "slot2", "slot3")
        },
    )
    layout = allele_layout(path)
    assert (layout.rules.kind, layout.n_slots) == ("ordered", 3)
    # A slot not read before an edit is missing, as the prefix rule reads it.
    assert read_all(path)["state"].tolist() == [
        "mark",
        "missing",
        "mark",
        "unedited",
        "unedited",
        "missing",
    ]


@pytest.mark.parametrize(
    ("attrs", "columns", "message"),
    [
        ({"PANDAS_ATTRS": "not json"}, {}, "are not JSON"),
        ({"PANDAS_ATTRS": "[1, 2]"}, {}, "are not a JSON object"),
        (
            {"PANDAS_ATTRS": json.dumps({"sites": "r1"})},
            {},
            "not a list of column names",
        ),
        (
            {"PANDAS_ATTRS": json.dumps({"sites": ["r1"], "edit_models": ["indel"]})},
            {},
            "edit_models of .* are not a mapping",
        ),
        (
            {
                "PANDAS_ATTRS": json.dumps(
                    {
                        "sites": ["r1", "r2"],
                        "edit_models": {"r1": "indel", "r2": "substitution"},
                    }
                )
            },
            {},
            "follow several edit models, indel, substitution",
        ),
        (
            {
                "PANDAS_ATTRS": json.dumps(
                    {"sites": ["r1"], "edit_models": {"r1": "substitution"}}
                )
            },
            {},
            "'substitution' edit model, which CyteArc does not read",
        ),
        (
            {
                "PANDAS_ATTRS": json.dumps(
                    {"sites": ["r1"], "edit_models": {"r1": "insertion_lookup"}}
                )
            },
            {},
            "without the site_states that name their labels",
        ),
        ({}, {"UMI": ["AAAA"]}, "holds one row per UMI"),
    ],
)
def test_cassiopeia_parquet_refuses_attributes_it_cannot_read(
    tmp_path, attrs, columns, message
):
    polars = pytest.importorskip("polars")
    frame = {
        "cellBC": ["c1"],
        "intBC": ["i1"],
        "r1": ["None"],
        "r2": ["None"],
        **columns,
    }
    path = tmp_path / "x.alleles.parquet"
    polars.DataFrame(frame).with_columns(
        [polars.col(name).cast(polars.Categorical) for name in ("r1", "r2")]
    ).write_parquet(path, metadata=attrs or None)
    with pytest.raises(ValueError, match=message):
        allele_layout(path, format="cassiopeia_parquet")


@pytest.mark.parametrize(
    ("header", "format"),
    [
        ("n_UMIs\tCR\tmutation\taligned_LR\taligned_ref", "darlin"),
        ("cellBC\tintBC\tr1\tr2\tallele\tUMI\treadCount", "cassiopeia"),
        ("intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount", "petracer"),
        ("Cell,TargetBC,Site1,Site2,nUMI", "typewriter"),
        ("cell\tACGTACGTACGT\tn_loci", "typewriter_chains"),
        (",intID1-EMX1,intID1-HEK3", "petracer_characters"),
        ("cell_barcode,intID1-EMX1", "petracer_characters"),
    ],
)
def test_each_format_is_recognised_from_its_header(tmp_path, header, format):
    path = tmp_path / "file.txt"
    separator = "\t" if "\t" in header else ","
    path.write_text(
        header + "\n" + separator.join(["x"] * len(header.split(separator))) + "\n"
    )
    assert (
        allele_layout(path, integrations="CA" if format == "darlin" else None).format
        == format
    )


def test_unknown_and_ambiguous_tables_name_the_formats(tmp_path):
    path = tmp_path / "unknown.csv"
    path.write_text("cell,clone\nc1,1\n")
    with pytest.raises(ValueError, match="'characters' reads a character matrix"):
        allele_layout(path)
    both = tmp_path / "both.csv"
    both.write_text("intID,cellBC,intBC,r1,RNF2,HEK3,EMX1\na,b,c,d,e,f,g\n")
    with pytest.raises(ValueError, match="several formats, 'cassiopeia', 'petracer'"):
        allele_layout(both)
    assert set(ALLELE_FORMATS) >= {"carlin", "treedata", "table"}


def test_carlin_pairs_must_align_and_list_each_cell_once(tmp_path):
    def pair(annotations: str, colonies: str) -> Path:
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        (folder / "AlleleAnnotations.txt").write_text(annotations)
        (folder / "AlleleColonies.txt").write_text(colonies)
        return folder

    first, second = _CELLS[:2]
    with pytest.raises(ValueError, match="has more lines than"):
        read_all(pair("[]", f"{first}\n{second}"), integrations="CA")
    with pytest.raises(ValueError, match="has 1 lines and .* 2"):
        read_all(pair("[]\n23_268del", first), integrations="CA")
    with pytest.raises(ValueError, match="lists 1 cell barcodes on more than one line"):
        read_all(pair("[]\n23_268del", f"{first}\n{first}"), integrations="CA")
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "AlleleAnnotations.txt").write_text("[]")
    with pytest.raises(FileNotFoundError, match="lacks AlleleColonies.txt"):
        allele_file(lone / "AlleleAnnotations.txt")
    with pytest.raises(ValueError, match="neither a CARLIN pair nor a Zarr store"):
        allele_file(tmp_path / "1" / ".." / ".." / tmp_path.name)


def test_darlin_molecules_must_match_their_tables(tmp_path):
    final = tmp_path / "step4_final.tsv"
    final.write_text(
        "n_UMIs\tCR\tmutation\taligned_LR\taligned_ref\n2\tAAAC\t[]\tAC\tAC\n"
    )
    annotated = tmp_path / "step4_annotated.tsv"
    annotated.write_text(
        "CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref\nAAAC\tL\tU1\t3\t[]\tAC\tAC\n"
    )
    with pytest.raises(ValueError, match="counts other UMIs than the molecules"):
        read_all(final, integrations="CA")
    annotated.write_text(
        "CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref\nGGGT\tL\tU1\t3\t[]\tAC\tAC\n"
    )
    with pytest.raises(ValueError, match="holds no molecule of"):
        read_all(final, integrations="CA")
    annotated.write_text(
        "CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref\n"
        "AAAC\tL\tU1\t3\t[]\tAC\tAC\nAAAC\tL\tU2\t4\t[]\tAC\tAC\n"
    )
    assert read_all(final, integrations="CA")["reads"].tolist() == [7]
    final.write_text("n_UMIs\tCR\tmutation\n2\tAAAC\t[]\n")
    with pytest.raises(ValueError, match="no aligned_LR column"):
        allele_layout(final, integrations="CA")


def test_cassiopeias_molecule_tables_are_refused(tmp_path):
    path = tmp_path / "molecules.txt"
    path.write_text("cellBC\tintBC\tr1\tUMI\treadCount\nc1\ti1\tNone\tACGTACGT\t3\n")
    with pytest.raises(ValueError, match="holds sequences, as Cassiopeia's molecule"):
        read_all(path)
    path.write_text("cellBC\tintBC\tallele\tUMI\nc1\ti1\tA\t3\n")
    with pytest.raises(ValueError, match="has no site column"):
        allele_layout(path, format="cassiopeia")


def test_typewriter_rows_must_fit_its_header(tmp_path):
    path = tmp_path / "tapes.csv"
    path.write_text("Cell,TargetBC,Site1,Site2,nUMI\nc1,T1,AAAGGA\n")
    with pytest.raises(ValueError, match="cut short"):
        read_all(path)
    path.write_text("Cell,TargetBC,Site1,Site2,nUMI\nc1,T1,AAAGGA,,CCCGGA,3\n")
    with pytest.raises(ValueError, match="insertion beyond slot 2"):
        read_all(path)
    path.write_text("Cell,TargetBC,Site2,Site1,nUMI\nc1,T1,,,3\n")
    with pytest.raises(ValueError, match="Site1 to SiteN in that order"):
        allele_layout(path, format="typewriter")
    path.write_text("Cell,TargetBC,Site1,nUMI,extra\nc1,T1,,3,x\n")
    with pytest.raises(ValueError, match="after its slots"):
        allele_layout(path, format="typewriter")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("AAG-AAG-AAG-AAG-AAG-AAG-AAG", "no chain of at most 6"),
        ("AAG--AAG", "no chain of at most 6"),
        ("AAG|U|U", "3 slots where its arrays have 6"),
    ],
)
def test_typewriter_chains_must_fit_their_tapes(tmp_path, value, message):
    path = tmp_path / "consensus.tsv"
    path.write_text(f"cell\tACGTACGTACGT\nc1\t{value}\n")
    with pytest.raises(ValueError, match=message):
        read_all(path)


def test_typewriter_chains_refuse_columns_they_do_not_know(tmp_path):
    path = tmp_path / "consensus.tsv"
    path.write_text("cell\tACGTACGTACGT\tdepth\nc1\tAAG\t3\n")
    with pytest.raises(ValueError, match="neither an integration barcode"):
        allele_layout(path, format="typewriter_chains")
    path.write_text("barcode\tACGTACGTACGT\nc1\tAAG\n")
    with pytest.raises(ValueError, match="must be cell or cell_id"):
        allele_layout(path, format="typewriter_chains")


@pytest.mark.parametrize(
    ("columns", "error", "message"),
    [
        ({"allele": "a"}, ValueError, "must name the 'cell' column"),
        ({"cell": "c"}, ValueError, "must name 'sites'"),
        (
            {"cell": "c", "allele": "a", "colour": "x"},
            ValueError,
            "unknown roles 'colour'",
        ),
        ({"cell": "c", "sites": "a"}, TypeError, "sequence of column names"),
        ({"cell": "c", "sites": ["a", "a"]}, ValueError, "names a column twice"),
        ({"cell": "c", "sites": ["a"], "allele": "a"}, ValueError, "either 'sites'"),
        ({"cell": "c", "allele": "a", "separator": "--"}, ValueError, "one character"),
        (
            {"cell": "c", "allele": "a", "site": "s", "separator": "-"},
            ValueError,
            "not both",
        ),
        (
            {"cell": "c", "allele": "a", "kind": "tree"},
            ValueError,
            "kind'] must be one of",
        ),
        (
            {"cell": "c", "allele": "a", "site": "s", "kind": "ordered"},
            ValueError,
            "slots together",
        ),
        ({"cell": "c", "allele": "a", "slots": 3}, ValueError, "ordered sites only"),
        ({"cell": "c", "allele": "a", "unedited": 3}, TypeError, "text or a sequence"),
        (
            {"cell": "c", "allele": "a", "unedited": "x", "missing": "x"},
            ValueError,
            "cannot mean both",
        ),
    ],
)
def test_table_columns_are_checked(tmp_path, columns, error, message):
    path = tmp_path / "table.tsv"
    path.write_text("c\ta\ts\nc1\tx\t1\n")
    with pytest.raises(error, match=message):
        allele_layout(path, columns=columns, integrations="I")


def test_a_table_needs_an_integration_from_one_place(tmp_path):
    path = tmp_path / "table.tsv"
    path.write_text("c\ti\ta\nc1\tI1\tx\n")
    with pytest.raises(ValueError, match="give one of them"):
        allele_layout(path, columns={"cell": "c", "allele": "a"})
    with pytest.raises(ValueError, match="give one of them"):
        allele_layout(
            path,
            columns={"cell": "c", "integration": "i", "allele": "a"},
            integrations="I",
        )


def test_a_table_splits_an_allele_into_numbered_sites(tmp_path):
    path = tmp_path / "gestalt.tsv"
    path.write_text("CellBarcode\tHMID\nc1\tNONE-3D+2-NONE\nc2\tUNKNOWN-NONE-NONE\n")
    read = read_all(
        path,
        columns={
            "cell": "CellBarcode",
            "allele": "HMID",
            "separator": "-",
            "unedited": "NONE",
            "missing": "UNKNOWN",
        },
        integrations="HMID",
    )
    assert read["site"].tolist() == ["1", "2", "3"] * 2
    assert read["state"].tolist() == [
        "unedited",
        "mark",
        "unedited",
        "missing",
        "unedited",
        "unedited",
    ]
    assert read["allele"].tolist()[1] == "3D+2"


def test_a_table_of_one_site_per_row_refuses_a_site_listed_twice(tmp_path):
    path = tmp_path / "long.tsv"
    path.write_text("cell\tint\ttarget\tedit\nc1\tI1\tt1\tNONE\nc1\tI1\tt1\t3D\n")
    with pytest.raises(
        ValueError, match="1 sites of one cell and integration on more than one row"
    ):
        read_all(
            path,
            columns={
                "cell": "cell",
                "integration": "int",
                "site": "target",
                "allele": "edit",
            },
        )


def test_rows_without_a_cell_are_refused(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text("intID,cellBC,RNF2,HEK3,EMX1\ni1,,None,None,None\n")
    with pytest.raises(
        ValueError, match="without a cell barcode or integration on line 2$"
    ):
        read_all(path)


# Each table holds, before its bad row, a row, a blank line and a row whose
# quoted field holds a line break, so the bad row is the 4th row of the
# table but sits on file line 6 (5 in DNA Typewriter's table of 2022, which
# quotes nothing).
_LINE_CASES = {
    "a row without a cell": (
        "alleles.csv",
        'intID,cellBC,RNF2,HEK3,EMX1\ni1,c1,None,None,None\n\ni1,"c\n2",None,None,'
        "None\ni2,,None,None,None\n",
        {},
        "without a cell barcode or integration on line 6$",
    ),
    "a count that is no whole number": (
        "alleles.csv",
        "intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount\ni1,c1,None,None,None,2,5\n\n"
        'i1,"c\n2",None,None,None,2,5\ni2,c3,None,None,None,x,5\n',
        {},
        "UMI must be a whole number of at least 0, but holds 'x' on line 6$",
    ),
    "a token that the format does not read": (
        "alleles.csv",
        'intID,cellBC,RNF2,HEK3,EMX1\ni1,c1,None,None,None\n\ni1,"c\n2",None,None,'
        "None\ni2,c3,XYZ,None,None\n",
        {},
        "'XYZ' at RNF2, on line 6$",
    ),
    "a Cassiopeia read count": (
        "alleles.txt",
        'cellBC\tintBC\tr1\tUMI\treadCount\nc1\tI1\tNone\t2\t5\n\n"c\n2"\tI1\tNone'
        "\t2\t5\nc3\tI1\tNone\t2\tq\n",
        {},
        "readCount must be a whole number of at least 0, but holds 'q' on line 6$",
    ),
    "a matrix's state": (
        "matrix.csv",
        ',a-1,a-2\nc1,1,2\n\n"c\n2",1,2\nc3,1,\n',
        {"format": "characters"},
        "'' at 2, on line 6$",
    ),
    "a DNA Typewriter chain without a cell": (
        "chains.csv",
        'cell,ACGTACGT\nc1,ACG\n\n"c\n2",ACG\n,ACG\n',
        {},
        "without a cell barcode or integration on line 6$",
    ),
    "an allele split into other sites": (
        "table.tsv",
        'c\ta\nc1\tx;y\n\n"c\n2"\tx;y\nc3\tx;y;z\n',
        {
            "columns": {"cell": "c", "allele": "a", "separator": ";"},
            "integrations": "I",
        },
        "than the 2 of its first allele on line 6$",
    ),
    "a count of a table read through columns": (
        "table.tsv",
        'c\ta\tu\nc1\tx\t1\n\n"c\n2"\tx\t1\nc3\tx\t-1\n',
        {"columns": {"cell": "c", "allele": "a", "umis": "u"}, "integrations": "I"},
        "u must be a whole number of at least 0, but holds '-1' on line 6$",
    ),
    "a count of DNA Typewriter's table of 2022": (
        "typewriter.csv",
        "Cell,TargetBC,Site1,Site2,Site3,Site4,Site5,nUMI\n"
        "c1,AAAAAAAA,CCGGGA,ATAGGA,,,,2\nc2,AAAAAAAA,CCGGGA,,,,,2\n\n"
        "c3,AAAAAAAA,CCGGGA,,,,,x\n",
        {},
        "nUMI must be a whole number of at least 0, but holds 'x' on line 5$",
    ),
}


@pytest.mark.parametrize("case", sorted(_LINE_CASES))
def test_errors_name_the_file_line_past_blank_lines_and_quoted_breaks(tmp_path, case):
    name, body, options, expected = _LINE_CASES[case]
    path = tmp_path / name
    path.write_text(body)
    with pytest.raises(ValueError, match=expected):
        read_all(path, **options)


# Each table holds the rows of c1, c2 and c3 on file lines 2, 4 and 5 or 6:
# a blank line follows c1, and a quoted field of c2 holds a line break where
# the format quotes. A table numbers its rows from its header, line 1, so
# blank lines and line breaks within quoted fields do not count: 2, 3 and 4.
_ROW_CASES = {
    "petracer": (
        "alleles.csv",
        'intID,cellBC,RNF2,HEK3,EMX1\ni1,c1,None,None,None\n\ni1,c2,"None\n",None,'
        "None\ni2,c3,None,None,None\n",
        {},
    ),
    "cassiopeia": (
        "alleles.txt",
        'cellBC\tintBC\tr1\nc1\tI1\tNone\n\nc2\tI1\t"None\n"\nc3\tI1\tNone\n',
        {},
    ),
    "darlin": (
        "step4_final.tsv",
        "n_UMIs\tCR\tmutation\n2\tc1\t23_268del\n\n"
        '2\tc2\t"23_268del\n"\n2\tc3\t23_268del\n',
        {"integrations": "CA"},
    ),
    "characters": (
        "matrix.csv",
        ',a-1\nc1,1\n\nc2,"1\n"\nc3,1\n',
        {"format": "characters"},
    ),
    "typewriter_chains": (
        "chains.csv",
        'cell,ACGTACGT\nc1,ACG\n\nc2,"ACG\n"\nc3,ACG\n',
        {},
    ),
    "table": (
        "table.tsv",
        'c\ta\nc1\tx\n\nc2\t"x\n"\nc3\tx\n',
        {"columns": {"cell": "c", "allele": "a"}, "integrations": "I"},
    ),
    "typewriter": (
        "typewriter.csv",
        "Cell,TargetBC,Site1,Site2,Site3,Site4,Site5,nUMI\n"
        "c1,AAAAAAAA,CCGGGA,,,,,2\n\nc2,AAAAAAAA,CCGGGA,,,,,2\n"
        "c3,AAAAAAAA,CCGGGA,,,,,2\n",
        {},
    ),
}


@pytest.mark.parametrize("case", sorted(_ROW_CASES))
def test_records_keep_each_rows_line_as_its_table_numbers_it(tmp_path, case):
    name, body, options = _ROW_CASES[case]
    folder = tmp_path / case
    folder.mkdir()
    path = folder / name
    path.write_text(body)
    read = read_all(path, **options)
    rows = dict(read.groupby("barcode")["row"].first())
    assert rows == {"c1": 2, "c2": 3, "c3": 4}


def test_carlin_records_keep_the_line_of_their_allele(tmp_path):
    # Line 2 lists no cell, so the cells of line 3 keep its number, which
    # names their allele in both files.
    folder = tmp_path / "pair"
    folder.mkdir()
    (folder / "AlleleAnnotations.txt").write_text("[]\n23_268del\n24_100del\n")
    first, second, third = "AAACCTGAGAAACCAT", "AAACCTGAGAAACCGC", "AAACCTGAGAAACGAG"
    (folder / "AlleleColonies.txt").write_text(f"{first}\n\n{second},{third}\n")
    read = read_all(folder, integrations="CA")
    assert dict(read.groupby("barcode")["row"].first()) == {
        first: 1,
        second: 3,
        third: 3,
    }
    assert dict(read.groupby("barcode")["allele"].first()) == {
        first: None,
        second: "24_100del",
        third: "24_100del",
    }


def test_typewriter_tokens_refused_after_a_blank_line_name_their_file_line(tmp_path):
    path = tmp_path / "typewriter.csv"
    path.write_text(
        "Cell,TargetBC,Site1,Site2,Site3,Site4,Site5,nUMI\n"
        "c1,AAAAAAAA,CCGGGA,,,,,2\n\nc2,AAAAAAAA,XYZ,,,,,2\n"
    )
    with pytest.raises(ValueError, match="'XYZ' at 1, on line 4$"):
        read_all(path)


def _darlin_pair(tmp_path, final: str, molecules: str) -> Path:
    folder = tmp_path / "darlin"
    folder.mkdir(parents=True)
    (folder / "step4_final.tsv").write_text(final)
    (folder / "step4_annotated.tsv").write_text(molecules)
    return folder / "step4_final.tsv"


_FINAL_HEADER = "n_UMIs\tCR\tmutation\taligned_LR\taligned_ref\n"
_MOLECULE_HEADER = "CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref\n"


def test_darlin_errors_name_the_file_line_of_each_table(tmp_path):
    # The molecule of AAAC3 is missing; its row is the 3rd of step4_final.tsv,
    # on file line 6 past a blank line and a quoted line break.
    final = (
        _FINAL_HEADER
        + "2\tAAAC1\t23_268del\tLB1\tREF\n\n"
        + '2\tAAAC2\t"23_268del\n"\tLB2\tREF\n'
        + "2\tAAAC3\t23_268del\tLB3\tREF\n"
    )
    molecules = _MOLECULE_HEADER + "".join(
        f"AAAC{cell}\tx\tu{umi}\t3\tm\tLB{cell}\tREF\n"
        for cell in (1, 2)
        for umi in (1, 2)
    )
    path = _darlin_pair(tmp_path / "missing", final, molecules)
    with pytest.raises(ValueError, match="holds no molecule of, on line 6$"):
        read_all(path, integrations="CA")
    # A molecule's reads that are no count, on file line 6 of
    # step4_annotated.tsv, its 4th row.
    molecules = (
        _MOLECULE_HEADER
        + "AAAC1\tx\tu1\t3\tm\tLB1\tREF\n\n"
        + 'AAAC1\tx\t"u\n2"\t3\tm\tLB1\tREF\n'
        + "AAAC2\tx\tu1\tq\tm\tLB2\tREF\n"
    )
    path = _darlin_pair(tmp_path / "reads", final, molecules)
    with pytest.raises(
        ValueError, match="reads must be a whole number of at least 0, .* on line 6$"
    ):
        read_all(path, integrations="CA")


def test_character_matrices_without_a_cell_column_name_read_as_r_writes_them(
    tmp_path,
):
    # R's write.table leaves the row names' column unnamed, so each row holds
    # one field more than the header. PEtracer's empty state, a site of an
    # integration that the cell's embryo lacks, can end every row.
    path = tmp_path / "characters.csv"
    path.write_text("intID1-RNF2,intID1-HEK3\ncell1,1,\ncell2,2,\n")
    read = read_all(path, format="petracer_characters")
    assert read[["barcode", "site", "state", "allele"]].values.tolist() == [
        ["cell1", "RNF2", "mark", "1"],
        ["cell1", "HEK3", "not_applicable", None],
        ["cell2", "RNF2", "mark", "2"],
        ["cell2", "HEK3", "not_applicable", None],
    ]
    # A first row that ends empty, beside rows that do not: the empty state is
    # none that a character matrix writes.
    path.write_text("a-1,a-2\ncell1,0,\ncell2,2,3\n")
    with pytest.raises(ValueError, match="'' at 2, on line 2$"):
        read_all(path, format="characters")
    path.write_text("a-1,a-2\ncell1,1,-1\ncell2,2,3\n")
    read = read_all(path, format="characters")
    assert read["allele"].tolist() == ["1", None, "2", "3"]
    # A matrix whose rows end in a delimiter after a named cell column gives
    # an empty last state, or PEtracer a cell column among its characters,
    # and is refused rather than read one column over.
    path.write_text("cell,a-1,a-2\ncell1,1,2,\ncell2,2,3,\n")
    with pytest.raises(ValueError, match="'' at 2, on lines 2, 3$"):
        read_all(path, format="characters")
    path.write_text("cell_barcode,intID1-RNF2\ncell1,1,\n")
    with pytest.raises(
        ValueError,
        match="names sites ''.*its first column, 'cell_barcode', reads as a character",
    ):
        read_all(path, format="petracer_characters")
    # A header that names sites of other recorders gets no such hint.
    path.write_text("cell_barcode,intID1-SITE\ncell1,1\n")
    with pytest.raises(ValueError, match="characters are named intID<N>-<site>$"):
        read_all(path, format="petracer_characters")


def test_counts_must_be_whole_numbers_in_tool_tables(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text(
        "intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount\ni1,c1,None,None,None,NA,3\n"
    )
    with pytest.raises(ValueError, match="UMI must be a whole number"):
        read_all(path)
    table = tmp_path / "table.tsv"
    table.write_text("c\ta\tu\nc1\tx\tNA\n")
    # A table read through columns has missing counts where pandas reads them.
    read = read_all(
        table, columns={"cell": "c", "allele": "a", "umis": "u"}, integrations="I"
    )
    assert read["umis"].tolist() == [-1]


def test_character_names_split_on_their_last_hyphen(tmp_path):
    path = tmp_path / "matrix.csv"
    path.write_text("cell,E7.5-R1-int3-RNF2,-a\nc1,0,0\n")
    with pytest.raises(ValueError, match="no integration or no site"):
        allele_layout(path, format="characters")
    path.write_text("cell,E7.5-R1-int3-RNF2,int9\nc1,0,1\n")
    read = read_all(path, format="characters")
    assert list(zip(read["integration"], read["site"], strict=True)) == [
        ("E7.5-R1-int3", "RNF2"),
        ("int9", ""),
    ]
    path.write_text("cell,a-1,a-1\nc1,0,1\n")
    with pytest.raises(ValueError, match="names a character twice"):
        allele_layout(path, format="characters")


def test_gzip_files_read_as_plain_ones(tmp_path):
    plain = DATA / "petracer_B16F10_kinetics_alleles.csv"
    packed = tmp_path / "kinetics.csv.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))
    pd.testing.assert_frame_equal(read_all(plain), read_all(packed))
    cut = tmp_path / "cut.csv.gz"
    cut.write_bytes(packed.read_bytes()[:-40])
    with pytest.raises(ValueError, match="cut.csv.gz is cut short or corrupt"):
        read_all(cut)


def test_unicode_and_long_alleles_read_exactly(tmp_path):
    long_allele = ",".join(
        f"{position}_{position + 1}del" for position in range(1, 400, 3)
    )
    path = tmp_path / "table.tsv"
    path.write_text(
        f"cell\tallele\nzelle_ü\t{long_allele}\nzelle_é\tΔ7\n", encoding="utf-8"
    )
    read = read_all(
        path, columns={"cell": "cell", "allele": "allele"}, integrations="I"
    )
    assert read["barcode"].tolist() == ["zelle_ü", "zelle_é"]
    assert read["allele"].tolist() == [long_allele, "Δ7"]


def test_layouts_record_what_identifies_how_a_file_is_read():
    layout = allele_layout(
        DATA / "typewriter_2022_hek293t.csv",
        integrations={"TTCACGTA": 2, "TGGACGAC": 4},
    )
    assert layout.record() == {
        "format": "typewriter",
        "integration": None,
        "lengths": {"TGGACGAC": 4, "TTCACGTA": 2},
        "columns": {},
    }
    carlin = allele_layout(allele_file(DATA / "carlin_fo864_lf"), integrations="CARLIN")
    assert carlin.companions == (
        ("annotations", str(DATA / "carlin_fo864_lf" / "AlleleAnnotations.txt")),
    )
    assert re.fullmatch(r"\d+", str(carlin.held_bytes()))
