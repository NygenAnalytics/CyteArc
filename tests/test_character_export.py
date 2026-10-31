"""Character matrices exported for Cassiopeia and LAML hold the truth's states.

Expectations come from the recorder truth, from the exported files read
with pandas and anndata as treedata reads them, and from ConvexML's matrix
(BSD-3-Clause), which an import and an export must leave unchanged.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.clone_helpers import copy_clone_store, every_cell, write_truth_store
from tests.clone_truth import (
    PETRACER_INTEGRATIONS,
    PETRACER_MARKS,
    PETRACER_SITES,
    SAMPLES,
    clone_truth,
    recorder_truth,
    write_petracer_characters,
    write_petracer_long,
)
from tests.storage_helpers import write_count_store

DATA = Path(__file__).parent / "data" / "recorders"


@pytest.fixture(scope="module")
def truth():
    return clone_truth()


@pytest.fixture(scope="module")
def recorders(truth):
    return recorder_truth(truth)


@pytest.fixture(scope="module")
def template(tmp_path_factory, truth) -> Path:
    path = tmp_path_factory.mktemp("export_store") / "store.zarr"
    write_truth_store(path, truth)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_clone_store(template, tmp_path)


def read_treedata(path: Path) -> dict:
    """The elements of a TreeData store, read one by one as treedata reads them."""
    anndata = pytest.importorskip("anndata")
    import zarr

    group = zarr.open_group(path, mode="r", use_consolidated=False)
    elements = {
        key: anndata.io.read_elem(group[key])
        for key in ("obs", "var", "obsm", "uns", "allow_overlap", "alignment")
    }
    elements["attrs"] = dict(group.attrs)
    elements["obst"] = sorted(group["obst"].keys())
    return elements


def expected_characters(recorders, truth) -> pd.DataFrame:
    """The states that PEtracer's characters of the truth export as, cells by characters.

    A mark is its own state; an unknown mark follows the largest mark of its
    character; unedited is 0 and missing -1. Cells that read no site are
    left out.
    """
    table = recorders.independent.copy()
    table["character"] = table["integration"] + "-" + table["site"]
    read = table["read"] & (table["state"] != "missing")
    table["code"] = np.where(
        ~read,
        -1,
        np.where(table["state"] == "mark", table["mark"], 0),
    )
    largest = (
        table[read & (table["state"] == "mark")].groupby("character")["mark"].max()
    )
    unknown = read & (table["state"] == "unknown_mark")
    table.loc[unknown, "code"] = (
        table.loc[unknown, "character"].map(largest).fillna(0).astype(int) + 1
    )
    matrix = table.pivot(index="cell", columns="character", values="code")
    matrix = matrix.loc[(matrix >= 0).any(axis=1)]
    # Characters that no exported cell reads are left out.
    matrix = matrix.loc[:, (matrix >= 0).any(axis=0)]
    order = [
        f"{integration}-{site}"
        for integration in sorted(PETRACER_INTEGRATIONS, key=lambda name: int(name[5:]))
        for site in sorted(PETRACER_SITES)
    ]
    matrix = matrix.loc[:, [name for name in order if name in matrix.columns]]
    matrix.index = truth.cell_ids[matrix.index.to_numpy()]
    return matrix.astype(np.int64)


def _petracer_alleles(store, truth, recorders, tmp_path):
    path = write_petracer_characters(recorders, truth, tmp_path / "characters.csv.gz")
    return store.clones.import_alleles(path, cell_selection=every_cell(store))


def test_cassiopeia_stores_hold_the_truths_states(store, truth, recorders, tmp_path):
    alleles = _petracer_alleles(store, truth, recorders, tmp_path)
    expected = expected_characters(recorders, truth)
    written = store.clones.export_characters(alleles, tmp_path / "characters.zarr")
    assert written == len(expected)
    elements = read_treedata(tmp_path / "characters.zarr")
    characters = elements["obsm"]["characters"]
    pd.testing.assert_frame_equal(
        characters, expected, check_names=False, check_dtype=False
    )
    assert characters.dtypes.unique().tolist() == [np.dtype(np.int64)]
    assert elements["obs"].index.tolist() == expected.index.tolist()
    assert elements["uns"]["missing_state"] == -1
    assert elements["uns"]["unmodified_state"] == 0
    assert elements["attrs"]["encoding-type"] == "treedata"
    assert elements["allow_overlap"] is False or elements["allow_overlap"] == np.False_
    assert elements["alignment"] == "leaves"
    assert elements["obst"] == []
    # The state table names each exported edit's allele: a mark's label, or
    # '!' for an unknown mark.
    states = elements["uns"]["character_states"]
    for row in states.itertuples():
        found = set(characters[row.character][characters[row.character] == row.state])
        assert found == {row.state}
        if row.allele == "!":
            assert row.state > 8 or row.state == characters[row.character].max()
        else:
            assert row.state == int(row.allele)


def test_laml_files_hold_the_matrix_its_priors_and_states(
    store, truth, recorders, tmp_path
):
    alleles = _petracer_alleles(store, truth, recorders, tmp_path)
    expected = expected_characters(recorders, truth)
    store.clones.export_characters(alleles, tmp_path / "out.csv", format="laml")
    matrix = pd.read_csv(tmp_path / "out.csv", dtype=str, keep_default_na=False)
    assert matrix.columns.tolist() == ["cell", *expected.columns]
    assert matrix["cell"].tolist() == expected.index.tolist()
    states = matrix.iloc[:, 1:].replace("?", "-1").astype(np.int64)
    np.testing.assert_array_equal(states.to_numpy(), expected.to_numpy())
    assert "-1" not in set(matrix.iloc[:, 1:].to_numpy().reshape(-1))
    priors = pd.read_csv(
        tmp_path / "out_priors.csv",
        header=None,
        names=["character", "state", "probability"],
    )
    # Each edited state's share of its character's edited cells.
    for (character, state), probability in priors.set_index(["character", "state"])[
        "probability"
    ].items():
        column = states.iloc[:, character]
        assert probability == pytest.approx(
            (column == state).sum() / (column > 0).sum()
        )
    assert priors.groupby("character")["probability"].sum().to_numpy() == pytest.approx(
        1
    )
    assert (priors["state"] >= 1).all()
    table = pd.read_csv(tmp_path / "out_states.csv", dtype={"allele": str})
    assert table.columns.tolist() == ["character", "state", "allele", "kind", "cells"]
    counted = {(row.character, row.state): row.cells for row in table.itertuples()}
    assert counted == {
        (character, state): int((states[character] == state).sum())
        for character in states.columns
        for state in sorted(set(states[character]) - {-1, 0})
    }


def test_exported_matrices_import_back_with_their_states(
    store, truth, recorders, tmp_path
):
    alleles = _petracer_alleles(store, truth, recorders, tmp_path)
    expected = expected_characters(recorders, truth)
    store.clones.export_characters(alleles, tmp_path / "out.zarr")
    store.clones.export_characters(alleles, tmp_path / "out.csv", format="laml")
    cells = every_cell(store)
    for path, options in (
        (tmp_path / "out.zarr", {}),
        (tmp_path / "out.csv", {"format": "characters"}),
    ):
        again = store.clones.import_alleles(path, cell_selection=cells, **options)
        records = store.clones.load_alleles(again).records
        found = records.assign(
            character=records["integration"].astype(str)
            + "-"
            + records["site"].astype(str),
            code=np.where(
                records["state"] == "unedited",
                0,
                np.where(
                    records["state"] == "missing",
                    -1,
                    pd.to_numeric(records["allele"].astype(object), errors="coerce"),
                ),
            ),
        )
        matrix = found.pivot(index="cell", columns="character", values="code")
        matrix = matrix.loc[expected.index, expected.columns].astype(np.int64)
        np.testing.assert_array_equal(matrix.to_numpy(), expected.to_numpy())
    # The states survive a second export unchanged.
    reread = store.clones.import_alleles(
        tmp_path / "out.csv", cell_selection=cells, format="characters"
    )
    store.clones.export_characters(reread, tmp_path / "again.csv", format="laml")
    assert (tmp_path / "again.csv").read_text() == (tmp_path / "out.csv").read_text()


def test_the_most_abundant_allele_of_several_is_exported(
    store, truth, recorders, tmp_path
):
    paths = [
        write_petracer_long(recorders, truth, tmp_path / f"{sample}.csv", sample=sample)
        for sample in SAMPLES
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alleles = store.clones.import_alleles(
            paths,
            cell_selection=every_cell(store),
            barcode_prefix=[f"{sample}__" for sample in SAMPLES],
        )
    second = recorders.second
    pairs = second[["cell", "integration"]].drop_duplicates()
    with pytest.warns(
        UserWarning, match=f"{len(pairs)} cells and integrations have several"
    ):
        store.clones.export_characters(alleles, tmp_path / "out.csv", format="laml")
    matrix = pd.read_csv(
        tmp_path / "out.csv", dtype=str, keep_default_na=False
    ).set_index("cell")
    table = pd.read_csv(tmp_path / "out_states.csv", dtype=str, keep_default_na=False)
    allele_of = {(row.character, row.state): row.allele for row in table.itertuples()}
    first = recorders.independent.set_index(["cell", "integration", "site"])
    for row in second.itertuples():
        kept = first.loc[(row.cell, row.integration, row.site)]
        character = f"{row.integration}-{row.site}"
        written = matrix.loc[truth.cell_ids[row.cell], character]
        # The first allele's 5-mer is kept, not the weaker second's.
        if kept["state"] == "mark":
            assert (
                allele_of[(character, written)]
                == PETRACER_MARKS[row.site][kept["mark"] - 1]
            )
        elif kept["state"] == "unedited":
            assert written == "0"


def test_convexmls_matrix_survives_an_import_and_an_export(tmp_path):
    raw = pd.read_csv(
        DATA / "convexml_tree_0_character_matrix.csv", dtype=str, keep_default_na=False
    )
    cells = raw.iloc[:, 0].tolist()
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    from tests.clone_helpers import open_clone_store

    store = open_clone_store(tmp_path / "store.zarr")
    alleles = store.clones.import_alleles(
        DATA / "convexml_tree_0_character_matrix.csv",
        cell_selection=store.snapshot_cell_selection(),
        format="characters",
    )
    store.clones.export_characters(alleles, tmp_path / "out.csv", format="laml")
    written = pd.read_csv(tmp_path / "out.csv", dtype=str, keep_default_na=False)
    assert written.columns.tolist()[1:] == raw.columns.tolist()[1:]
    expected = raw.set_index(raw.columns[0]).replace("-1", "?")
    pd.testing.assert_frame_equal(
        written.set_index("cell").loc[expected.index], expected, check_names=False
    )
    store.clones.export_characters(alleles, tmp_path / "out.zarr")
    characters = read_treedata(tmp_path / "out.zarr")["obsm"]["characters"]
    np.testing.assert_array_equal(
        characters.loc[expected.index].to_numpy(),
        raw.set_index(raw.columns[0]).astype(np.int64).to_numpy(),
    )


def test_exports_refuse_what_they_cannot_write(store, truth, recorders, tmp_path):
    alleles = _petracer_alleles(store, truth, recorders, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    target = out / "taken.zarr"
    target.mkdir()
    with pytest.raises(FileExistsError, match="taken.zarr exists"):
        store.clones.export_characters(alleles, target)
    with pytest.raises(ValueError, match="format must be one of"):
        store.clones.export_characters(alleles, out / "x.nwk", format="newick")
    with pytest.raises(ValueError, match="unknown must be one of"):
        store.clones.export_characters(alleles, out / "x.zarr", unknown="drop")
    with pytest.raises(TypeError, match="path must be a path"):
        store.clones.export_characters(alleles, 3)
    with pytest.raises(TypeError, match="alleles must be an ArtifactRef"):
        store.clones.export_characters("alleles", out / "y.zarr")
    clones = store.clones.from_column("cell_type", cell_selection=every_cell(store))
    with pytest.raises(ValueError, match="must be a lineage_alleles artifact"):
        store.clones.export_characters(clones, out / "z.zarr")
    # Nothing was written by a refusal.
    assert sorted(path.name for path in out.iterdir()) == ["taken.zarr"]
    assert not any(target.iterdir())


def test_laml_refuses_names_its_files_cannot_hold(tmp_path):
    from tests.clone_helpers import open_clone_store

    cells = ["cell 1", "cell2"]
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((2, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_clone_store(tmp_path / "store.zarr")
    path = tmp_path / "matrix.csv"
    path.write_text('cell,"int:1-a"\n"cell 1",1\ncell2,0\n')
    alleles = store.clones.import_alleles(
        path, cell_selection=store.snapshot_cell_selection(), format="characters"
    )
    with pytest.raises(ValueError, match="The cell 'cell 1' holds a character"):
        store.clones.export_characters(alleles, tmp_path / "out.csv", format="laml")
    # Cassiopeia's store holds any name.
    store.clones.export_characters(alleles, tmp_path / "out.zarr")
    assert read_treedata(tmp_path / "out.zarr")["obsm"][
        "characters"
    ].columns.tolist() == ["int:1-a"]


def test_alleles_without_a_read_site_have_no_matrix(store, tmp_path, truth):
    path = tmp_path / "matrix.csv"
    path.write_text(
        "cell,i1-a\n" + "".join(f"{cell},-1\n" for cell in truth.cell_ids[:3])
    )
    alleles = store.clones.import_alleles(
        path, cell_selection=every_cell(store), format="characters"
    )
    with pytest.raises(ValueError, match="No cell reads a character"):
        store.clones.export_characters(alleles, tmp_path / "out.zarr")
    assert not (tmp_path / "out.zarr").exists()


def test_an_export_beyond_the_budget_writes_nothing(store, truth, recorders, tmp_path):
    alleles = _petracer_alleles(store, truth, recorders, tmp_path)
    store.memoryBytes = 1_000_000
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.clones.export_characters(alleles, tmp_path / "out.zarr")
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.clones.export_characters(alleles, tmp_path / "out.csv", format="laml")
    assert not (tmp_path / "out.zarr").exists() and not (tmp_path / "out.csv").exists()


def _matrix_of(codes: np.ndarray, largest: int):
    """A character matrix whose every character lists the states 1 to ``largest``."""
    from cytearc.clones.recorders import CharacterMatrix

    names = [f"intID{k // 3}-s{k % 3}" for k in range(codes.shape[1])]
    states = pd.DataFrame(
        [
            (name, state, f"allele{state}", "mark", 1)
            for name in names
            for state in range(1, largest + 1)
        ],
        columns=["character", "state", "allele", "kind", "cells"],
    )
    return CharacterMatrix(
        cells=np.arange(codes.shape[0]),
        integrations=tuple(name.split("-")[0] for name in names),
        sites=tuple(name.split("-")[1] for name in names),
        codes=codes,
        states=states,
        n_resolved=0,
        n_without_states=0,
    )


def test_laml_text_holds_what_its_rows_count(tmp_path):
    """A block of LAML's text holds at most laml_row_bytes per cell, traced.

    States of two digits each become a string of their own when every value
    is converted, which held about 85 bytes per state beside the 64 counted.
    """
    import tracemalloc

    from cytearc.writers.characters import laml_row_bytes, write_laml_characters

    rng = np.random.default_rng(5)
    n_cells, n_characters = 2_000, 300
    codes = rng.integers(-1, 41, size=(n_cells, n_characters)).astype(np.int32)
    matrix = _matrix_of(codes, 40)
    cells = np.asarray([f"cell{index:05d}" for index in range(n_cells)], dtype=object)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        write_laml_characters(
            tmp_path / "matrix.csv", cells, matrix, block_cells=n_cells
        )
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak <= n_cells * laml_row_bytes(n_characters), peak
    written = pd.read_csv(
        tmp_path / "matrix.csv", dtype=str, keep_default_na=False, index_col=0
    )
    expected = np.where(codes < 0, "?", codes.astype(str))
    np.testing.assert_array_equal(written.to_numpy(), expected)
    assert written.index.tolist() == cells.tolist()


def test_laml_text_refuses_a_state_that_its_table_does_not_list(tmp_path):
    from cytearc.writers.characters import write_laml_characters

    codes = np.asarray([[0, 1], [7, -1]], dtype=np.int32)
    cells = np.asarray(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="a state that its state table does not list"):
        write_laml_characters(tmp_path / "matrix.csv", cells, _matrix_of(codes, 2))
    # Blocks of one cell write the same text as one block of every cell.
    codes = np.asarray([[0, 2], [1, -1], [-1, -1]], dtype=np.int32)
    cells = np.asarray(["a", "b", "c"], dtype=object)
    for block_cells, name in ((1, "one.csv"), (3, "all.csv")):
        write_laml_characters(
            tmp_path / name, cells, _matrix_of(codes, 2), block_cells=block_cells
        )
    assert (tmp_path / "one.csv").read_text() == (tmp_path / "all.csv").read_text()
    assert (tmp_path / "one.csv").read_text().splitlines()[1:] == [
        "a,0,2",
        "b,1,?",
        "c,?,?",
    ]
