"""Recorder files that their readers must refuse, and the quirks they must take.

Each case writes the smallest file that shows one rule: a malformed file of
each format is refused with what is wrong, and an unusual file that a tool
can write is read as it says. TreeData stores are written element by
element with zarr, as anndata lays them out, and HDF5 files with h5py.
"""

import gzip
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc.readers._treedata import character_table, write_character_store
from cytearc.readers.recorder_alleles import (
    allele_file,
    allele_layout,
    classified_rows,
    iter_allele_rows,
)

DATA = Path(__file__).parent / "data" / "recorders"


def _rows(path, **options) -> list:
    """Every chunk of a file's rows, as the import reads them."""
    layout = allele_layout(allele_file(path), **options)
    return [classified_rows(layout, rows) for rows in iter_allele_rows(layout, 2)]


# Text tables


def test_an_empty_file_has_no_header_to_recognise(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="is empty; an allele table needs a header"):
        allele_layout(path)


@pytest.mark.parametrize(
    ("format", "text", "message"),
    [
        ("characters", "cell,i1-a,i1-b\nc1,1\n", "names 3 columns, but its first row"),
        (
            "typewriter_chains",
            "cell\tCGGGGAATTGTA\tTCCGGAAGACCC\nc1\t-\n",
            "names 3 columns, but its first row",
        ),
        ("characters", "cell\n", "holds no character column"),
        ("characters", "cell,i1-a,i1-a\nc1,1,2\n", "names a character twice"),
        ("typewriter_chains", "cell\tn_loci\nc1\t0\n", "names no integration barcode"),
        (
            "typewriter_chains",
            "cell\tCGGGGAATTGTA\tCGGGGAATTGTA\nc1\t-\t-\n",
            "names an integration twice",
        ),
    ],
)
def test_matrices_and_chains_refuse_malformed_headers_and_rows(
    tmp_path, format, text, message
):
    path = tmp_path / ("alleles.tsv" if "\t" in text else "alleles.csv")
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        _rows(path, format=format)


def test_character_matrices_read_whole_floats_as_numbers_and_refuse_na_text(
    tmp_path,
):
    """pandas writes the integers of a column with a missing value as floats."""
    path = tmp_path / "matrix.csv"
    path.write_text("cell,i1-a,i1-b,i1-c\nc1,0.0,-1.0,3.0\nc2,3,0,12.00\n")
    (chunk,) = _rows(path, format="characters")
    assert chunk.states.tolist() == [[0, 3, 1], [1, 0, 1]]
    assert chunk.alleles.tolist() == [[None, None, "3"], ["3", None, "12"]]
    for token in ("null", "NULL", "#N/A", "N/A", "<NA>", "none"):
        path.write_text(f"cell,i1-a\nc1,{token}\n")
        with pytest.raises(ValueError, match="tokens that 'characters' does not read"):
            _rows(path, format="characters")


def test_a_table_refuses_rows_of_another_width(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text("cell,allele,umis\nc1,x\n")
    with pytest.raises(ValueError, match="names 3 columns, but its first row"):
        _rows(
            path,
            format="table",
            columns={"cell": "cell", "allele": "allele"},
            integrations="i1",
        )


@pytest.mark.parametrize(
    ("columns", "integrations", "error", "message"),
    [
        (["cell"], "i1", TypeError, "columns must map roles"),
        (
            {"cell": "cell", "allele": 5},
            "i1",
            TypeError,
            r"columns\['allele'\] must be",
        ),
        (
            {"cell": "cell", "sites": ["a", "b"], "site": "site"},
            "i1",
            ValueError,
            "names 'sites' or 'site', not both",
        ),
        (
            {"cell": "cell", "sites": ["a", "b"], "slots": 2},
            "i1",
            ValueError,
            "counts the slots of ordered sites only",
        ),
        (
            {"cell": "cell", "sites": ["a", "b"], "kind": "ordered", "slots": 0},
            "i1",
            ValueError,
            "must be a whole number of at least 1",
        ),
        (
            {"cell": "cell", "integration": "integration", "sites": ["a", "b"]},
            {"i1": 1},
            ValueError,
            "gives slots of ordered sites only",
        ),
    ],
)
def test_table_roles_refuse_what_they_cannot_mean(
    tmp_path, columns, integrations, error, message
):
    path = tmp_path / "alleles.csv"
    path.write_text("cell,integration,site,a,b,allele\nc1,i1,a,x,y,x\n")
    with pytest.raises(error, match=message):
        allele_layout(path, format="table", columns=columns, integrations=integrations)


def test_a_table_takes_token_lists_and_ordered_slots_from_its_sites(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text("cell,a,b,c\nc1,AAG,-,NONE\nc2,NONE,NONE,NONE\n")
    columns = {
        "cell": "cell",
        "sites": ["a", "b", "c"],
        "kind": "ordered",
        "unedited": ["NONE", "-"],
    }
    layout = allele_layout(path, format="table", columns=columns, integrations="i1")
    # Ordered sites without slots count one slot per site.
    assert layout.n_slots == 3
    assert layout.rules.unedited == frozenset({"NONE", "-"})
    (chunk,) = _rows(path, format="table", columns=columns, integrations="i1")
    assert chunk.states.tolist() == [[1, 0, 0], [0, 0, 0]]


def test_a_split_allele_of_ordered_slots_refuses_more_parts_than_slots(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text("cell,allele\nc1,AAG|CCC\nc2,AAG|CCC|GGT\n")
    columns = {
        "cell": "cell",
        "allele": "allele",
        "separator": "|",
        "kind": "ordered",
        "slots": 2,
    }
    with pytest.raises(ValueError, match="into more than 2 sites on line 3"):
        _rows(path, format="table", columns=columns, integrations="i1")


def test_a_table_of_one_site_per_row_refuses_a_row_without_a_site(tmp_path):
    path = tmp_path / "alleles.csv"
    path.write_text("cell,site,allele\nc1,t1,x\nc1,,y\n")
    columns = {"cell": "cell", "site": "site", "allele": "allele"}
    with pytest.raises(ValueError, match="has rows without a site"):
        _rows(path, format="table", columns=columns, integrations="i1")


# CARLIN pairs and darlin tables


def _carlin_copy(tmp_path: Path) -> Path:
    folder = tmp_path / "pair"
    shutil.copytree(DATA / "carlin_fo864_lf", folder)
    return folder


def test_integrations_must_name_a_carlin_pairs_array(tmp_path):
    with pytest.raises(ValueError, match="must name an integration, not be empty"):
        allele_layout(allele_file(DATA / "carlin_fo864_lf"), integrations="")


def test_a_carlin_run_without_its_type_is_told_apart_by_its_tags(tmp_path):
    folder = _carlin_copy(tmp_path)
    results = folder / "Results.txt"
    results.write_text(
        "".join(
            line
            for line in results.read_text().splitlines(keepends=True)
            if not line.startswith("CfgType:")
        )
    )
    layout = allele_layout(allele_file(folder), integrations="CA")
    assert layout.format == "carlin"
    assert sum(len(chunk.states) for chunk in _rows(folder, integrations="CA")) == 864


def test_carlin_pairs_skip_empty_barcodes_and_take_a_final_newline(tmp_path):
    folder = tmp_path / "pair"
    folder.mkdir()
    (folder / "AlleleAnnotations.txt").write_text("[]\n23_268del\n")
    (folder / "AlleleColonies.txt").write_text(
        "AAACCTGAGAAACCAT,,AAACCTGAGAAACCGC\nAAACCTGAGAAACGAG\n"
    )
    chunks = _rows(folder, integrations="CA")
    assert [len(chunk.states) for chunk in chunks] == [2, 1]
    assert [chunk.alleles.reshape(-1).tolist() for chunk in chunks][1] == ["23_268del"]


@pytest.mark.parametrize("damaged", ["AlleleAnnotations.txt", "AlleleColonies.txt"])
def test_carlin_files_that_are_not_text_are_refused(tmp_path, damaged):
    folder = _carlin_copy(tmp_path)
    (folder / damaged).write_bytes(b"\xff\xfe\x00AAAC\n")
    with pytest.raises(ValueError, match="is not a text file that CyteArc can read"):
        _rows(folder, integrations="CA")


@pytest.mark.parametrize("damaged", ["AlleleAnnotations.txt", "AlleleColonies.txt"])
def test_carlin_files_cut_short_are_refused(tmp_path, damaged):
    folder = _carlin_copy(tmp_path)
    data = gzip.compress((folder / damaged).read_bytes())
    (folder / damaged).write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="is cut short or corrupt"):
        _rows(folder, integrations="CA")


def _darlin(tmp_path: Path, molecules: str) -> Path:
    folder = tmp_path / "darlin"
    folder.mkdir()
    (folder / "step4_final.tsv").write_text(
        "n_UMIs\tCR\tmutation\taligned_LR\taligned_ref\n"
        "2\tAAACCTGAGAAACCAT\t23_268del\tACGT-LB1-AC\tACGTAC\n"
    )
    (folder / "step4_annotated.tsv").write_text(molecules)
    return folder / "step4_final.tsv"


def test_darlin_molecules_of_another_width_are_refused(tmp_path):
    path = _darlin(
        tmp_path,
        "CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref\n"
        "AAACCTGAGAAACCAT\tLB1\tUMI1\t3\n",
    )
    with pytest.raises(ValueError, match="names 7 columns, but its first row"):
        _rows(path, integrations="CA")


def test_darlin_tables_without_molecules_name_the_rows_they_lack(tmp_path):
    path = _darlin(tmp_path, "CR\tLR\tUR\treads\tmutations\taligned_LR\taligned_ref\n")
    with pytest.raises(ValueError, match="holds no molecule of, on line 2"):
        _rows(path, integrations="CA")


@pytest.mark.parametrize("slots", [0, 6])
def test_typewriter_slots_must_fit_the_arrays_of_the_2022_table(slots):
    with pytest.raises(ValueError, match=f"has {slots} slots, .* have from 1 to 5"):
        allele_layout(
            DATA / "typewriter_2022_hek293t.csv", integrations={"TGGACGAC": slots}
        )


@pytest.mark.parametrize("slots", [0, 7])
def test_typewriter_slots_must_fit_the_arrays_of_the_2026_chains(tmp_path, slots):
    path = tmp_path / "cell_consensus.tsv"
    path.write_text("cell\tCGGGGAATTGTA\nc1\tAAG\n")
    with pytest.raises(ValueError, match=f"has {slots} slots, .* have from 1 to 6"):
        allele_layout(path, integrations={"CGGGGAATTGTA": slots})


# Cassiopeia 3.0's parquet tables


@pytest.fixture
def polars():
    return pytest.importorskip("polars")


def _parquet(polars, path: Path, **columns) -> Path:
    """A table whose site r1 is categorical, as Cassiopeia's sites are."""
    frame = polars.DataFrame(columns)
    if "r1" in columns:
        frame = frame.with_columns(polars.col("r1").cast(polars.Categorical))
    frame.write_parquet(path)
    return path


def test_parquet_tables_need_polars(tmp_path, polars, monkeypatch):
    path = _parquet(
        polars, tmp_path / "a.alleles.parquet", cellBC=["c1"], intBC=["i1"], r1=["None"]
    )
    monkeypatch.setitem(sys.modules, "polars", None)
    with pytest.raises(ImportError, match="parquet allele tables needs polars"):
        allele_layout(path, format="cassiopeia_parquet")


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"cellBC": ["c1"], "r1": ["None"]}, "has no column 'intBC'"),
        ({"cellBC": ["c1"], "intBC": ["i1"], "n_umis": [3]}, "names no site columns"),
    ],
)
def test_parquet_tables_refuse_what_cassiopeia_does_not_write(
    tmp_path, polars, columns, message
):
    path = _parquet(polars, tmp_path / "a.alleles.parquet", **columns)
    with pytest.raises(ValueError, match=message):
        allele_layout(path, format="cassiopeia_parquet")


def test_parquet_tables_read_sites_without_pandas_attributes(tmp_path, polars):
    # pandas' own layout under "pandas" holds no attrs and is not read.
    path = tmp_path / "a.alleles.parquet"
    polars.DataFrame({"cellBC": ["c1"], "intBC": ["i1"], "r1": ["None"]}).with_columns(
        polars.col("r1").cast(polars.Categorical)
    ).write_parquet(path, metadata={"pandas": "not json"})
    layout = allele_layout(path, format="cassiopeia_parquet")
    assert layout._read["sites_from"] == "columns"
    (chunk,) = _rows(path, format="cassiopeia_parquet")
    # Without n_umis and n_reads columns the counts are unknown.
    assert chunk.states.tolist() == [[0]]


def test_parquet_rows_that_do_not_pass_are_skipped_chunk_by_chunk(tmp_path, polars):
    path = _parquet(
        polars,
        tmp_path / "a.alleles.parquet",
        cellBC=["c1", "c2", "c3"],
        intBC=["i1", "i1", "i1"],
        r1=["None", "[1:2D]", "None"],
        n_umis=[3, 4, 5],
        n_reads=[30, 40, 50],
        **{"pass": [False, False, True]},
    )
    chunks = _rows(path, format="cassiopeia_parquet")
    assert [chunk.states.tolist() for chunk in chunks] == [[[0]]]


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"pass": [True, None]}, "has rows whose pass is missing"),
        ({"n_umis": [3, -1]}, "has a n_umis that is not a count of at least 0"),
        ({"n_reads": [3, None]}, "has a n_reads that is not a count of at least 0"),
    ],
)
def test_parquet_rows_refuse_missing_passes_and_counts(
    tmp_path, polars, columns, message
):
    path = _parquet(
        polars,
        tmp_path / "a.alleles.parquet",
        cellBC=["c1", "c2"],
        intBC=["i1", "i1"],
        r1=["None", "None"],
        **columns,
    )
    with pytest.raises(ValueError, match=message):
        _rows(path, format="cassiopeia_parquet")


# TreeData files and stores


def _store(path: Path, *, characters=None, index=("c1", "c2"), uns=None) -> zarr.Group:
    """A TreeData store whose ``obsm['characters']`` holds ``characters`` by name."""
    root = zarr.open_group(path, mode="w-", zarr_format=3)
    root.attrs.update({"encoding-type": "treedata", "encoding-version": "0.1.0"})
    obsm = root.create_group("obsm", attributes={"encoding-type": "dict"})
    frame = obsm.create_group(
        "characters",
        attributes={
            "encoding-type": "dataframe",
            "encoding-version": "0.2.0",
            "_index": "_index",
            "column-order": list(characters or {}),
        },
    )
    frame.create_array("_index", data=np.asarray(index, dtype=str))
    for name, values in (characters or {}).items():
        if isinstance(values, dict):
            group = frame.create_group(name)
            for key, part in values.items():
                group.create_array(key, data=np.asarray(part))
        else:
            frame.create_array(name, data=np.asarray(values))
    if uns is not None:
        group = root.create_group("uns", attributes={"encoding-type": "dict"})
        for name, value in uns.items():
            group.create_array(name, data=np.asarray(value))
    return root


def _states(path: Path) -> list[list]:
    table = character_table(path)
    return [states.tolist() for _cells, states in table.blocks(10)]


def test_treedata_reads_categorical_and_nullable_columns(tmp_path):
    path = tmp_path / "a.zarr"
    _store(
        path,
        characters={
            "i1-a": {"codes": [1, -1], "categories": [0, 3]},
            "i1-b": {"values": [2, 7], "mask": [False, True]},
            "i1-c": [1.0, np.nan],
        },
        uns={"missing_state": "-", "unmodified_state": 0.0},
    )
    table = character_table(path)
    assert (table.missing_state, table.unmodified_state) == ("-", "0")
    assert _states(path) == [[["3", "2", "1"], [None, None, None]]]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda path: _store(path, characters={"i1-a": [True, False]}),
            "holds booleans",
        ),
        (
            lambda path: _store(path, characters={"i1-a": [1.5, 2.0]}),
            "not whole numbers",
        ),
        (
            lambda path: _store(
                path, characters={"i1-a": [1, 2]}, uns={"missing_state": True}
            ),
            "cannot be the boolean True",
        ),
        (
            lambda path: _store(
                path, characters={"i1-a": [1, 2]}, uns={"missing_state": 0.5}
            ),
            "must be a whole number, not 0.5",
        ),
        (
            lambda path: _store(
                path, characters={"i1-a": [1, 2]}, uns={"missing_state": [1, 2]}
            ),
            r"uns\['missing_state'\] must hold one state",
        ),
        (
            lambda path: _store(path, characters={"i1-a": {"other": [1, 2]}}),
            "holds a column 'i1-a' that CyteArc cannot read",
        ),
        (
            lambda path: _store(path, characters={"i1-a": [[1, 2], [3, 4]]}),
            "of more than one dimension",
        ),
        (
            lambda path: _store(path, characters={"i1-a": [1, 2, 3]}),
            "holds 3 values for 2 cells",
        ),
    ],
)
def test_treedata_refuses_states_and_columns_it_cannot_read(tmp_path, build, message):
    path = tmp_path / "a.zarr"
    build(path)
    with pytest.raises(ValueError, match=message):
        _states(path)


def test_treedata_refuses_frames_without_names_or_columns(tmp_path):
    path = tmp_path / "frame.zarr"
    root = _store(path, characters={"i1-a": [1, 2]})
    frame = root["obsm/characters"]
    frame.attrs["column-order"] = ["i1-a", "i1-a"]
    with pytest.raises(ValueError, match="names a character twice"):
        character_table(path)
    frame.attrs["column-order"] = ["i1-b"]
    with pytest.raises(
        ValueError, match="lists a character 'i1-b' that it does not hold"
    ):
        character_table(path)
    del frame.attrs["column-order"]
    assert character_table(path).characters == ()
    frame.attrs["_index"] = "cells"
    with pytest.raises(ValueError, match="names no cells"):
        character_table(path)
    frame.attrs["encoding-type"] = "dict"
    with pytest.raises(ValueError, match="is no dataframe that CyteArc can read"):
        character_table(path)


def test_treedata_refuses_cells_without_names(tmp_path):
    path = tmp_path / "a.zarr"
    root = _store(path, characters={"i1-a": [1, 2]})
    frame = root["obsm/characters"]
    del frame["_index"]
    index = frame.create_group("_index")
    index.create_array("codes", data=np.asarray([0, -1]))
    index.create_array("categories", data=np.asarray(["c1"], dtype=str))
    with pytest.raises(ValueError, match="has a cell without a name"):
        _states(path)


def test_treedata_plain_matrices_need_two_dimensions_and_obs_names(tmp_path):
    path = tmp_path / "plain.h5td"
    with h5py.File(path, "w") as handle:
        handle.create_group("obsm").create_dataset("characters", data=np.zeros(3))
    with pytest.raises(ValueError, match="must be a table of cells by characters"):
        character_table(path)
    with h5py.File(path, "w") as handle:
        handle.create_group("obsm").create_dataset("characters", data=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="names no cells in obs"):
        character_table(path)


def test_files_that_are_neither_hdf5_nor_zarr_are_refused(tmp_path):
    path = tmp_path / "characters.h5td"
    path.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="is neither an HDF5 file nor a Zarr store"):
        character_table(path)
    store = tmp_path / "broken.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{not json")
    with pytest.raises(ValueError, match="cannot be opened"):
        character_table(store)


def test_the_store_writer_refuses_what_its_layout_cannot_hold(tmp_path):
    states = pd.DataFrame({"character": [], "state": [], "allele": []})
    codes = np.zeros((2, 2), dtype=np.int64)
    (tmp_path / "taken.zarr").mkdir()
    with pytest.raises(FileExistsError, match="taken.zarr exists"):
        write_character_store(
            tmp_path / "taken.zarr", ["c1", "c2"], ["a", "b"], codes, states
        )
    with pytest.raises(ValueError, match="one row per cell and a column per character"):
        write_character_store(tmp_path / "a.zarr", ["c1"], ["a", "b"], codes, states)
    with pytest.raises(ValueError, match="Each character must be named once"):
        write_character_store(
            tmp_path / "b.zarr", ["c1", "c2"], ["a", "a"], codes, states
        )


@pytest.mark.parametrize(
    ("format", "message"),
    [
        ("carlin", "is no CARLIN pair"),
        ("cassiopeia_parquet", "is no parquet file"),
        ("petracer_characters", r"names sites 'a', 'b', which are none of PEtracer's"),
    ],
)
def test_a_format_that_does_not_fit_the_file_is_refused(tmp_path, format, message):
    path = tmp_path / "matrix.csv"
    path.write_text("cell,i1-a,i1-b\nc1,1,2\n")
    with pytest.raises(ValueError, match=message):
        allele_layout(
            path, format=format, integrations="CA" if format == "carlin" else None
        )


def test_a_split_table_without_any_allele_reads_one_missing_site_per_row(tmp_path):
    path = tmp_path / "alleles.tsv"
    path.write_text("cell\tallele\nc1\t\nc2\tNA\n")
    columns = {"cell": "cell", "allele": "allele", "separator": "-"}
    layout = allele_layout(path, format="table", columns=columns, integrations="HMID")
    assert layout._read["split_sites"] == 1
    (chunk,) = _rows(path, format="table", columns=columns, integrations="HMID")
    assert chunk.states.tolist() == [[3], [3]]
