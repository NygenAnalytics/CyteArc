"""Recorder alleles imported from every format recover the truth's edits.

``tests/clone_truth.py`` draws each cell's edits on three kinds of recorder
along its clone and writes them as each pipeline does, with its tokens and
quirks. Each import must give exactly the records that the truth fixes for
the file, state by state.
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.clone_helpers import assert_joined, every_cell, write_truth_store
from tests.storage_helpers import copy_store
from tests.clone_truth import (
    CHAIN_TAPES,
    DARLIN_ARRAYS,
    PETRACER_INTEGRATIONS,
    SAMPLES,
    TYPEWRITER_TAPES,
    clone_truth,
    expected_cassiopeia_table,
    expected_darlin,
    expected_darlin_step4,
    expected_gestalt_table,
    expected_petracer_characters,
    expected_petracer_long,
    expected_typewriter,
    expected_typewriter_chains,
    recorder_truth,
    write_carlin_pair,
    write_cassiopeia_table,
    write_darlin_step4,
    write_gestalt_table,
    write_petracer_characters,
    write_petracer_long,
    write_typewriter,
    write_typewriter_chains,
)

_COLUMNS = ["cell", "integration", "site", "state", "allele", "umis", "reads"]
_SHORTER_TAPES = {name: slots for name, slots in TYPEWRITER_TAPES.items() if slots < 5}
_SHORTER_CHAINS = {name: slots for name, slots in CHAIN_TAPES.items() if slots < 6}


@pytest.fixture(scope="module")
def truth():
    return clone_truth()


@pytest.fixture(scope="module")
def recorders(truth):
    return recorder_truth(truth)


@pytest.fixture(scope="module")
def template(tmp_path_factory, truth) -> Path:
    path = tmp_path_factory.mktemp("allele_store") / "store.zarr"
    write_truth_store(path, truth)
    return path


@pytest.fixture
def store(template, tmp_path):
    return copy_store(template, tmp_path)


def records_of(store, ref) -> pd.DataFrame:
    """An import's records, in the columns the truth's expectations use."""
    records = store.clones.load_alleles(ref).records
    allele = records["allele"].astype(object)
    frame = pd.DataFrame(
        {
            "cell": records["ids"].astype(str),
            "integration": records["integration"].astype(str),
            "site": records["site"].astype(str),
            "state": records["state"].astype(str),
            "allele": allele.where(allele.notna(), None),
            "umis": records["umis"].fillna(-1).astype(np.int64),
            "reads": records["reads"].fillna(-1).astype(np.int64),
        }
    )
    return frame.sort_values(_COLUMNS[:5], na_position="first").reset_index(drop=True)


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    """Records with text as Python objects, None where an allele is missing."""
    found = frame.loc[:, _COLUMNS].copy()
    for name in ("cell", "integration", "site", "state", "allele"):
        values = found[name].astype(object)
        found[name] = values.where(values.notna(), None)
    found["umis"] = found["umis"].astype(np.int64)
    found["reads"] = found["reads"].astype(np.int64)
    return found.sort_values(_COLUMNS, na_position="first").reset_index(drop=True)


def assert_records(store, ref, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        _normalized(records_of(store, ref)), _normalized(expected)
    )


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def test_petracer_long_tables_recover_the_truth(store, truth, recorders, tmp_path):
    paths = [
        write_petracer_long(recorders, truth, tmp_path / f"{sample}.csv", sample=sample)
        for sample in SAMPLES
    ]
    with pytest.warns(UserWarning, match="1 of 49 barcodes of the allele files"):
        ref = store.clones.import_alleles(
            paths,
            cell_selection=every_cell(store),
            barcode_prefix=[f"{sample}__" for sample in SAMPLES],
        )
    expected = pd.concat(
        [expected_petracer_long(recorders, truth, sample=sample) for sample in SAMPLES]
    ).sort_values(_COLUMNS[:5], na_position="first")
    assert_records(store, ref, expected.reset_index(drop=True))
    # The second, weaker alleles of the first integration are kept beside
    # the first ones.
    files = store.clones.load_alleles(ref).files
    assert files["format"].tolist() == ["petracer", "petracer"]
    assert files["pipeline"].tolist() == ["PEtracer 2025"] * 2
    assert files["kind"].tolist() == ["independent"] * 2
    # Every cell of the truth is listed beside one droplet outside the store.
    droplet = [f"{SAMPLES[0]}__{truth.outside[0]}-1"]
    diagnostics = store.artifacts.inspect(ref).diagnostics
    assert_joined(diagnostics, n_file=49, unmatched=droplet)
    # The files' rows, read as tables, and those of the droplet, whose
    # prefixed barcode names no cell; every row names its integration and
    # every slot fits its integration's length.
    tables = [pd.read_csv(path, dtype=str) for path in paths]
    named = [
        (sample + "__" + table["cellBC"]).isin(truth.cell_ids.tolist())
        for sample, table in zip(SAMPLES, tables, strict=True)
    ]
    assert {
        key: diagnostics[key]
        for key in (
            "fileRows",
            "unmatchedRows",
            "slotsBeyondLength",
            "rowsWithoutIntegration",
        )
    } == {
        "fileRows": sum(map(len, tables)),
        "unmatchedRows": sum(int((~found).sum()) for found in named),
        "slotsBeyondLength": 0,
        "rowsWithoutIntegration": 0,
    }
    assert diagnostics["unmatchedRows"] > 0
    # With one sample's cells selected, the other's are cells of the store
    # outside the selection, which give no warning of their own.
    store.cells.insert(
        "petracer_first_sample", truth.samples == SAMPLES[0], overwrite=True
    )
    with pytest.warns(UserWarning, match="1 of 49 barcodes of the allele files"):
        first = store.clones.import_alleles(
            paths,
            cell_selection=store.snapshot_cell_selection("petracer_first_sample"),
            barcode_prefix=[f"{sample}__" for sample in SAMPLES],
        )
    assert_joined(
        store.artifacts.inspect(first).diagnostics,
        n_file=49,
        unmatched=droplet,
        unselected=truth.cell_ids[truth.samples != SAMPLES[0]].tolist(),
    )


@pytest.mark.parametrize("layout", ["geo", "zenodo"])
def test_petracer_character_matrices_recover_the_truth(
    store, truth, recorders, tmp_path, layout
):
    path = write_petracer_characters(
        recorders, truth, tmp_path / "characters.csv.gz", layout=layout
    )
    ref = store.clones.import_alleles(path, cell_selection=every_cell(store))
    assert_records(
        store, ref, expected_petracer_characters(recorders, truth, layout=layout)
    )


def test_typewriter_tables_follow_the_prefix_rule_and_contracted_tapes(
    store, truth, recorders, tmp_path
):
    paths = [
        write_typewriter(recorders, truth, tmp_path / f"{sample}.csv", sample=sample)
        for sample in SAMPLES
    ]
    tapes = recorders.tapes.set_index(["cell", "integration"])
    # The fifth cell's insertion beyond the contracted tape is written where
    # its tape was read.
    beyond = int(tapes.loc[(4, "TGGACGAC"), "edits"] >= 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ref = store.clones.import_alleles(
            paths,
            cell_selection=every_cell(store),
            integrations=_SHORTER_TAPES,
            barcode_prefix=[f"{sample}__" for sample in SAMPLES],
        )
    messages = [str(warning.message) for warning in caught]
    assert any(f"{beyond} slots beyond" in message for message in messages) == bool(
        beyond
    )
    expected = pd.concat(
        [expected_typewriter(recorders, truth, sample=sample) for sample in SAMPLES]
    ).sort_values(_COLUMNS[:5], na_position="first")
    assert_records(store, ref, expected.reset_index(drop=True))
    group = store.artifacts.load(ref)
    assert store.artifacts.inspect(ref).diagnostics["slotsBeyondLength"] == beyond
    assert [entry["lengths"] for entry in group.attrs["files"]] == [
        dict(sorted(_SHORTER_TAPES.items()))
    ] * 2


@pytest.mark.parametrize("form", ["chain", "slots"])
def test_typewriter_chains_recover_the_truth_in_both_forms(
    store, truth, recorders, tmp_path, form
):
    path = write_typewriter_chains(
        recorders, truth, tmp_path / "consensus.tsv.gz", form=form
    )
    ref = store.clones.import_alleles(
        path, cell_selection=every_cell(store), integrations=_SHORTER_CHAINS
    )
    assert_records(store, ref, expected_typewriter_chains(recorders, truth))


def test_carlin_pairs_of_three_arrays_recover_the_truth(
    store, truth, recorders, tmp_path
):
    paths, arrays, prefixes = [], [], []
    for sample in SAMPLES:
        for array in DARLIN_ARRAYS:
            folder = tmp_path / f"{sample}_{array}"
            write_carlin_pair(recorders, truth, folder, sample=sample, array=array)
            # A folder names its pair, as does either file of it.
            paths.append(folder if array != "TA" else folder / "AlleleAnnotations.txt")
            arrays.append(array)
            prefixes.append(f"{sample}__")
    ref = _quiet(
        lambda: store.clones.import_alleles(
            paths,
            cell_selection=every_cell(store),
            integrations=arrays,
            barcode_prefix=prefixes,
            barcode_suffix="-1",
        )
    )
    assert_records(
        store,
        ref,
        expected_darlin(recorders, truth, samples=SAMPLES, arrays=DARLIN_ARRAYS),
    )
    files = store.clones.load_alleles(ref).files
    assert set(files["integration"]) == set(DARLIN_ARRAYS)
    assert set(files["pipeline"]) == {"CARLIN (MATLAB)"}


def test_the_arrays_of_one_library_must_share_cells(store, truth, recorders, tmp_path):
    first = tmp_path / "d2_CA"
    second = tmp_path / "d4_TA"
    write_carlin_pair(recorders, truth, first, sample="d2", array="CA")
    write_carlin_pair(recorders, truth, second, sample="d4", array="TA")
    cells = every_cell(store)
    # The two samples' barcodes name different cells under one prefix each,
    # so the files cannot be the arrays of one library.
    with pytest.raises(ValueError, match="no arrays of one library"):
        _quiet(
            lambda: store.clones.import_alleles(
                [first, second],
                cell_selection=cells,
                integrations=["CA", "TA"],
                barcode_prefix=["d2__", "d2__"],
                barcode_suffix="-1",
            )
        )
    # With their own prefixes they are two libraries.
    _quiet(
        lambda: store.clones.import_alleles(
            [first, second],
            cell_selection=cells,
            integrations=["CA", "TA"],
            barcode_prefix=["d2__", "d4__"],
            barcode_suffix="-1",
        )
    )


def test_darlin_tables_take_reads_from_their_molecules(
    store, truth, recorders, tmp_path
):
    paths, prefixes = [], []
    for sample in SAMPLES:
        folder = tmp_path / sample
        paths.append(
            write_darlin_step4(recorders, truth, folder, sample=sample, array="RA")
        )
        prefixes.append(f"{sample}__")
    ref = store.clones.import_alleles(
        paths,
        cell_selection=every_cell(store),
        integrations="RA",
        barcode_prefix=prefixes,
        barcode_suffix="-1",
    )
    expected = pd.concat(
        [
            expected_darlin_step4(recorders, truth, sample=sample, array="RA")
            for sample in SAMPLES
        ]
    ).sort_values(_COLUMNS[:5], na_position="first")
    assert_records(store, ref, expected.reset_index(drop=True))
    files = store.clones.load_alleles(ref).files
    assert files["pipeline"].tolist() == ["darlin-core"] * 2
    group = store.artifacts.load(ref)
    assert [entry["reads_from"] for entry in group.attrs["files"]] == [
        "step4_annotated.tsv"
    ] * 2


def test_darlin_tables_without_their_molecules_count_no_reads(
    store, truth, recorders, tmp_path
):
    folder = tmp_path / "d2"
    path = write_darlin_step4(recorders, truth, folder, sample="d2", array="TA")
    archive = tmp_path / "archive"
    archive.mkdir()
    shutil.move(folder / "step4_annotated.tsv", archive / "step4_annotated.tsv")
    ref = store.clones.import_alleles(
        path,
        cell_selection=every_cell(store),
        integrations="TA",
        barcode_prefix="d2__",
        barcode_suffix="-1",
    )
    expected = expected_darlin_step4(recorders, truth, sample="d2", array="TA")
    expected["reads"] = -1
    assert_records(store, ref, expected)


def test_darlin_tables_refuse_carlins_unknown_allele(store, truth, recorders, tmp_path):
    # darlin-core writes no unknown allele, so CARLIN's X is no darlin token;
    # the truth gives every eleventh cell from the fourth an X on CA.
    sample = str(truth.samples[3])
    path = write_darlin_step4(recorders, truth, tmp_path, sample=sample, array="CA")
    with pytest.raises(ValueError, match="'X' at its allele"):
        store.clones.import_alleles(
            path,
            cell_selection=every_cell(store),
            integrations="CA",
            barcode_prefix=f"{sample}__",
        )


def test_cassiopeia_tables_read_none_as_uncut(store, truth, recorders, tmp_path):
    paths = [
        write_cassiopeia_table(
            recorders, truth, tmp_path / f"{sample}.txt", sample=sample
        )
        for sample in SAMPLES
    ]
    ref = store.clones.import_alleles(
        paths,
        cell_selection=every_cell(store),
        barcode_pattern=r"\.(?P<barcode>[ACGT]{16})-1$",
        barcode_prefix=[f"{sample}__" for sample in SAMPLES],
        barcode_suffix="-1",
    )
    expected = pd.concat(
        [
            expected_cassiopeia_table(recorders, truth, sample=sample)
            for sample in SAMPLES
        ]
    ).sort_values(_COLUMNS[:5], na_position="first")
    assert_records(store, ref, expected.reset_index(drop=True))


@pytest.mark.parametrize("long", [False, True])
def test_gestalt_tables_read_through_columns(store, truth, recorders, tmp_path, long):
    path = write_gestalt_table(recorders, truth, tmp_path / "gestalt.tsv", long=long)
    columns = (
        {
            "cell": "cellID",
            "integration": "intID",
            "site": "target",
            "allele": "target_edit",
            "umis": "UMIs_crispr_intID_cell",
            "unedited": "NONE",
            "missing": "UNKNOWN",
        }
        if long
        else {
            "cell": "cell",
            "integration": "integration",
            "sites": ["target1", "target2", "target3"],
            "unedited": "NONE",
            "missing": "UNKNOWN",
        }
    )
    ref = store.clones.import_alleles(
        path, cell_selection=every_cell(store), columns=columns
    )
    assert_records(store, ref, expected_gestalt_table(recorders, truth, long=long))
    group = store.artifacts.load(ref)
    assert group.attrs["files"][0]["rows_hold_sites"] is long


def test_each_cell_counts_its_edits_integrations_and_records(
    store, truth, recorders, tmp_path
):
    path = write_petracer_characters(recorders, truth, tmp_path / "characters.csv.gz")
    ref = store.clones.import_alleles(path, cell_selection=every_cell(store))
    table = recorders.independent
    read = table["read"] & (table["state"] != "missing")
    edited = read & table["state"].isin(["mark", "unknown_mark"])
    expected_edits = (
        edited.groupby(table["cell"])
        .sum()
        .reindex(range(len(truth.cell_ids)), fill_value=0)
    )
    expected_integrations = (
        table[read]
        .groupby("cell")["integration"]
        .nunique()
        .reindex(range(len(truth.cell_ids)), fill_value=0)
    )
    cells = store.clones.load_alleles(ref).cells
    assert cells.index.tolist() == truth.cell_ids.tolist()
    np.testing.assert_array_equal(cells["n_edits"], expected_edits.to_numpy())
    np.testing.assert_array_equal(
        cells["n_integrations"], expected_integrations.to_numpy()
    )
    np.testing.assert_array_equal(
        cells["n_records"], np.full(len(truth.cell_ids), 3 * len(PETRACER_INTEGRATIONS))
    )
    # The canonical value is each cell's edits.
    loaded = store.artifacts.load_values(ref)
    np.testing.assert_array_equal(loaded.values, expected_edits.to_numpy())
    # select_cells compares its bounds strictly, so low=0 keeps the cells
    # with an edit, as the clones page says.
    assert 0 < (expected_edits > 0).sum() < len(truth.cell_ids)
    selected = store.select_cells(ref, low=0)
    np.testing.assert_array_equal(
        store.artifacts.load(selected)["values"][:], expected_edits.to_numpy() > 0
    )


def test_a_saved_import_is_reused_and_warns_again(store, truth, recorders, tmp_path):
    paths = [
        write_typewriter(recorders, truth, tmp_path / f"{sample}.csv", sample=sample)
        for sample in SAMPLES
    ]
    cells = every_cell(store)
    options = {
        "cell_selection": cells,
        "integrations": _SHORTER_TAPES,
        "barcode_prefix": [f"{sample}__" for sample in SAMPLES],
    }
    first = _quiet(lambda: store.clones.import_alleles(paths, **options))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = store.clones.import_alleles(
            list(reversed(paths)),
            **{
                **options,
                "barcode_prefix": [f"{sample}__" for sample in reversed(SAMPLES)],
            },
        )
    assert again == first
    beyond = store.artifacts.inspect(first).diagnostics["slotsBeyondLength"]
    assert any("slots beyond" in str(warning.message) for warning in caught) == bool(
        beyond
    )
    fresh = _quiet(
        lambda: store.clones.import_alleles(paths, **options, invalidate_cache=True)
    )
    # A new import of the same records.
    assert fresh != first
    pd.testing.assert_frame_equal(records_of(store, fresh), records_of(store, first))
    # Other slots give another import.
    other = _quiet(
        lambda: store.clones.import_alleles(
            paths, **{**options, "integrations": {"TGGACGAC": 4}}
        )
    )
    assert other != first


def test_files_whose_rows_name_integrations_take_none(
    store, truth, recorders, tmp_path
):
    path = write_petracer_long(recorders, truth, tmp_path / "d2.csv", sample="d2")
    with pytest.raises(ValueError, match="names each row's integration"):
        store.clones.import_alleles(
            path, cell_selection=every_cell(store), integrations="CA"
        )


def test_files_of_one_array_must_have_it_named(store, truth, recorders, tmp_path):
    folder = tmp_path / "d2_CA"
    write_carlin_pair(recorders, truth, folder, sample="d2", array="CA")
    cells = every_cell(store)
    with pytest.raises(ValueError, match="pass integrations"):
        store.clones.import_alleles(folder, cell_selection=cells)
    with pytest.raises(TypeError, match="must name the array"):
        store.clones.import_alleles(
            folder, cell_selection=cells, integrations={"CA": 1}
        )
    with pytest.raises(ValueError, match="lists 2 values for 1 allele files"):
        store.clones.import_alleles(
            folder, cell_selection=cells, integrations=["CA", "TA"]
        )
    with pytest.raises(TypeError, match="Each item of integrations"):
        store.clones.import_alleles([folder], cell_selection=cells, integrations=[3])


def test_slots_of_contracted_tapes_must_fit_the_arrays(
    store, truth, recorders, tmp_path
):
    path = write_typewriter(recorders, truth, tmp_path / "d2.csv", sample="d2")
    cells = every_cell(store)
    with pytest.raises(ValueError, match="has 6 slots, but the arrays"):
        store.clones.import_alleles(
            path, cell_selection=cells, integrations={"TGGACGAC": 6}
        )
    with pytest.raises(TypeError, match="map integration names to numbers of slots"):
        store.clones.import_alleles(
            path, cell_selection=cells, integrations={"TGGACGAC": "4"}
        )
    with pytest.raises(TypeError, match="map each integration shorter"):
        store.clones.import_alleles(path, cell_selection=cells, integrations="TGGACGAC")


def test_import_arguments_are_checked(store, tmp_path, truth, recorders):
    path = write_petracer_long(recorders, truth, tmp_path / "d2.csv", sample="d2")
    cells = every_cell(store)
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.clones.import_alleles(path, cell_selection="cells")
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.import_alleles(path, cell_selection=cells, invalidate_cache=1)
    with pytest.raises(TypeError, match="format must be a format name"):
        store.clones.import_alleles(path, cell_selection=cells, format=3)
    with pytest.raises(ValueError, match="format must be one of"):
        store.clones.import_alleles(path, cell_selection=cells, format="gestalt")
    with pytest.raises(TypeError, match="paths must be a path"):
        store.clones.import_alleles(3, cell_selection=cells)
    with pytest.raises(FileNotFoundError, match="No allele file"):
        store.clones.import_alleles(tmp_path / "absent.csv", cell_selection=cells)
    with pytest.raises(ValueError, match="columns reads tables of other tools"):
        store.clones.import_alleles(
            path, cell_selection=cells, format="petracer", columns={"cell": "cellBC"}
        )
    with pytest.raises(ValueError, match="needs columns"):
        store.clones.import_alleles(path, cell_selection=cells, format="table")
    with pytest.raises(TypeError, match="paths must be a path or a sequence"):
        store.clones.import_alleles([path, 3], cell_selection=cells)
    with pytest.raises(TypeError, match="integrations must be an integration's name"):
        store.clones.import_alleles(path, cell_selection=cells, integrations=5)
    store.cells.insert(
        "no_cell", np.zeros(len(truth.cell_ids), dtype=bool), overwrite=True
    )
    with pytest.raises(ValueError, match="cell_selection selects no cell"):
        store.clones.import_alleles(
            path, cell_selection=store.snapshot_cell_selection("no_cell")
        )


def test_invalidate_cache_is_checked_before_the_store_is_read(
    store, truth, recorders, tmp_path
):
    from cytearc.storage.errors import ArtifactResolutionError
    from cytearc.storage.refs import ArtifactRef

    path = write_petracer_long(recorders, truth, tmp_path / "d2.csv", sample="d2")
    missing = ArtifactRef(
        scope="datastore", kind="cell_selection", artifact_id="f" * 64, assay=None
    )
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.import_alleles(path, cell_selection=missing, invalidate_cache=1)
    with pytest.raises(ArtifactResolutionError, match="does not exist"):
        store.clones.import_alleles(path, cell_selection=missing)
    # A NumPy boolean is a flag, so the store is read next.
    with pytest.raises(ArtifactResolutionError, match="does not exist"):
        store.clones.import_alleles(
            path, cell_selection=missing, invalidate_cache=np.True_
        )


def test_loads_name_missing_incomplete_and_other_artifacts(
    store, truth, recorders, tmp_path
):
    from cytearc.storage.artifacts import artifact_group
    from cytearc.storage.refs import ArtifactRef

    path = write_petracer_long(recorders, truth, tmp_path / "d2.csv", sample="d2")
    cells = every_cell(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = store.clones.import_alleles(path, cell_selection=cells)
    missing = ArtifactRef(
        scope="datastore", kind="lineage_alleles", artifact_id="f" * 64, assay=None
    )
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store.clones.load_alleles(missing)
    clones = store.clones.from_column("cell_type", cell_selection=cells)
    with pytest.raises(
        ValueError, match="alleles must be a lineage_alleles artifact, not 'clones'"
    ):
        store.clones.load_alleles(clones)
    # An import that stopped before it finished leaves an incomplete artifact.
    artifact_group(store.zw, ref).attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store.clones.load_alleles(ref)


def test_saved_records_keep_each_rows_line_in_its_file(tmp_path):
    """A table's rows keep the lines they start on, past blank lines and quoted breaks."""
    from tests.storage_helpers import open_store, write_count_store

    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((3, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=["c1", "c2", "c3"],
    )
    store = open_store(tmp_path / "store.zarr")
    path = tmp_path / "alleles.csv"
    # The rows of c1, c2 and c3 sit on file lines 2, 4 and 6.
    path.write_text(
        'intID,cellBC,RNF2,HEK3,EMX1\ni1,c1,None,None,None\n\ni1,c2,"None\n",None,'
        "None\ni2,c3,None,None,None\n"
    )
    ref = store.clones.import_alleles(
        path, cell_selection=store.snapshot_cell_selection()
    )
    records = store.clones.load_alleles(ref).records
    assert dict(records.groupby("ids", observed=True)["row"].first()) == {
        "c1": 2,
        "c2": 4,
        "c3": 6,
    }


def test_artifacts_without_their_counts_or_files_are_refused(
    store, truth, recorders, tmp_path
):
    """A saved import whose attributes were damaged is refused, never guessed."""
    from cytearc.storage.artifacts import artifact_group

    path = write_petracer_long(recorders, truth, tmp_path / "d2.csv", sample="d2")
    cells = every_cell(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = store.clones.import_alleles(path, cell_selection=cells)
    attributes = artifact_group(store.zw, ref).attrs
    saved = dict(attributes["diagnostics"])
    attributes["diagnostics"] = {**saved, "slotsBeyondLength": True}
    try:
        with pytest.raises(ValueError, match="call ds.clones.import_alleles again"):
            store.clones.load_alleles(ref)
    finally:
        attributes["diagnostics"] = saved
    files = attributes["files"]
    attributes["files"] = "d2.csv"
    try:
        with pytest.raises(ValueError, match="has no valid list of files"):
            store.clones.load_alleles(ref)
    finally:
        attributes["files"] = files
    assert len(store.clones.load_alleles(ref).files) == 1


def test_a_read_only_store_reuses_saved_alleles_and_refuses_new_ones(
    store, truth, recorders, tmp_path
):
    from tests.storage_helpers import open_store

    path = write_petracer_characters(recorders, truth, tmp_path / "characters.csv.gz")
    cells = every_cell(store)
    ref = store.clones.import_alleles(path, cell_selection=cells)
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.clones.import_alleles(path, cell_selection=cells) == ref
    other = write_petracer_characters(
        recorders, truth, tmp_path / "zenodo.csv.gz", layout="zenodo"
    )
    with pytest.raises(PermissionError):
        reader.clones.import_alleles(other, cell_selection=cells)


def test_rows_without_an_integration_are_left_out_counted_and_warned(tmp_path):
    from tests.storage_helpers import open_store, write_count_store

    # PEtracer leaves intID empty where an intBC is not on its whitelist.
    path = (
        Path(__file__).parent
        / "data"
        / "recorders"
        / "petracer_preedited_10x_invitro_alleles.csv"
    )
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    lacking = int((raw["intID"] == "").sum())
    assert lacking > 0
    cells = sorted(set(raw["cellBC"]))
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    selection = store.snapshot_cell_selection()
    message = f"{lacking} rows of the allele files name no integration"
    with pytest.warns(UserWarning, match=message):
        ref = store.clones.import_alleles(path, cell_selection=selection)
    with pytest.warns(UserWarning, match=message):
        assert store.clones.import_alleles(path, cell_selection=selection) == ref
    records = store.clones.load_alleles(ref).records
    assert len(records) == 3 * (len(raw) - lacking)
    assert store.artifacts.inspect(ref).diagnostics["rowsWithoutIntegration"] == lacking


@pytest.mark.parametrize(
    ("name", "text", "options"),
    [
        ("petracer.csv", "intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount\n", {}),
        ("characters.csv", "cell_barcode,intID1-EMX1,intID1-HEK3,intID1-RNF2\n", {}),
        ("typewriter.csv", "Cell,TargetBC,Site1,Site2,Site3,Site4,Site5,nUMI\n", {}),
        ("chains.tsv", "cell\tCGGGGAATTGTA\tn_loci\n", {}),
        (
            "table.csv",
            "cell,integration,allele\n",
            {
                "columns": {
                    "cell": "cell",
                    "integration": "integration",
                    "allele": "allele",
                }
            },
        ),
    ],
)
def test_files_of_a_header_alone_import_no_records(
    store, truth, tmp_path, name, text, options
):
    path = tmp_path / name
    path.write_text(text)
    ref = store.clones.import_alleles(path, cell_selection=every_cell(store), **options)
    loaded = store.clones.load_alleles(ref)
    assert len(loaded.records) == 0 and loaded.records.columns.tolist()[:3] == [
        "ids",
        "file",
        "integration",
    ]
    assert (loaded.cells["n_records"] == 0).all()
    assert loaded.files["n_rows"].tolist() == [0]
    assert loaded.site_states().empty
    with pytest.raises(ValueError, match="No cell reads a character"):
        store.clones.export_characters(ref, tmp_path / "out.zarr")


def test_an_empty_carlin_pair_imports_no_records(store, tmp_path):
    folder = tmp_path / "pair"
    folder.mkdir()
    (folder / "AlleleAnnotations.txt").write_text("")
    (folder / "AlleleColonies.txt").write_text("")
    ref = store.clones.import_alleles(
        folder, cell_selection=every_cell(store), integrations="CA"
    )
    assert len(store.clones.load_alleles(ref).records) == 0


def test_a_pair_listed_twice_keeps_each_listings_integration(tmp_path):
    """Canonical order sorts files by prefix, but options follow their listing."""
    from tests.storage_helpers import open_store, write_count_store

    pair = Path(__file__).parent / "data" / "recorders" / "darlin_ll837_skull_CA"
    barcodes = sorted(
        {
            barcode
            for line in (pair / "AlleleColonies.txt").read_text().splitlines()
            for barcode in line.split(",")
            if barcode
        }
    )
    cells = [f"z_{barcode}" for barcode in barcodes] + [
        f"a_{barcode}" for barcode in barcodes
    ]
    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((len(cells), 2), dtype=np.uint16)},
        "uint16",
        cell_ids=cells,
    )
    store = open_store(tmp_path / "store.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = store.clones.import_alleles(
            [pair, pair],
            cell_selection=store.snapshot_cell_selection(),
            barcode_prefix=["z_", "a_"],
            integrations=["ZZ", "AA"],
        )
    records = store.clones.load_alleles(ref).records
    prefixes = records["ids"].astype(str).str[:2]
    assert set(zip(prefixes, records["integration"].astype(str), strict=True)) == {
        ("z_", "ZZ"),
        ("a_", "AA"),
    }


def test_records_do_not_depend_on_the_chunks_a_file_is_read_in(
    store, truth, recorders, tmp_path, monkeypatch
):
    """Rows of one site each are read site by site within a chunk, and split
    alleles give each row the sites of the file's first allele, whatever the
    budget's chunks."""
    from cytearc.datastore._operations import recorder_alleles as operations
    from tests.clone_truth import write_gestalt_table

    long = write_gestalt_table(recorders, truth, tmp_path / "long.tsv", long=True)
    columns = {
        "cell": "cellID",
        "integration": "intID",
        "site": "target",
        "allele": "target_edit",
        "unedited": "NONE",
        "missing": "UNKNOWN",
    }
    split = tmp_path / "split.tsv"
    split.write_text(
        "cell\tallele\n"
        + "".join(
            f"{cell}\t{'NONE-3D+1-UNKNOWN' if index % 3 else ''}\n"
            for index, cell in enumerate(truth.cell_ids)
        )
    )
    split_columns = {"cell": "cell", "allele": "allele", "separator": "-"}
    cells = every_cell(store)

    def records(path, **options):
        ref = store.clones.import_alleles(
            path, cell_selection=cells, invalidate_cache=True, **options
        )
        return store.clones.load_alleles(ref).records

    expected = [
        records(long, columns=columns),
        records(split, columns=split_columns, integrations="HMID"),
    ]
    # One row per chunk.
    monkeypatch.setattr(operations, "_CHUNK_SHARE", 10**15)
    found = [
        records(long, columns=columns),
        records(split, columns=split_columns, integrations="HMID"),
    ]
    for one, other in zip(expected, found, strict=True):
        pd.testing.assert_frame_equal(one, other)
        rows = one.groupby("ids", observed=True)["row"]
        assert rows.apply(lambda values: values.is_monotonic_increasing).all()
    # A row without an allele reads each of the first allele's three sites.
    assert expected[1].groupby("row").size().unique().tolist() == [3]


def test_split_alleles_of_another_number_of_sites_are_refused(store, tmp_path):
    path = tmp_path / "split.tsv"
    path.write_text("cell\tallele\nc1\tA-B-C\nc2\tA-B\n")
    with pytest.raises(ValueError, match="another number of sites than the 3"):
        store.clones.import_alleles(
            path,
            cell_selection=every_cell(store),
            columns={"cell": "cell", "allele": "allele", "separator": "-"},
            integrations="HMID",
        )
    with pytest.raises(ValueError, match=r"columns\['slots'\] must count the slots"):
        store.clones.import_alleles(
            path,
            cell_selection=every_cell(store),
            columns={
                "cell": "cell",
                "allele": "allele",
                "separator": "-",
                "kind": "ordered",
            },
            integrations="HMID",
        )


def test_slots_of_tapes_that_no_row_names_are_refused(
    store, truth, recorders, tmp_path
):
    """A misspelled tape would otherwise keep the recorder's full length."""
    from tests.clone_truth import write_typewriter, write_typewriter_chains

    cells = every_cell(store)
    before = set(store.artifacts.list(kind="lineage_alleles", scope="datastore"))
    tapes = write_typewriter(recorders, truth, tmp_path / "d2.csv", sample="d2")
    with pytest.raises(ValueError, match=r"slots of 'TGGACGAX', which no row of"):
        store.clones.import_alleles(
            tapes, cell_selection=cells, integrations={"TGGACGAX": 4}
        )
    chains = write_typewriter_chains(recorders, truth, tmp_path / "chains.tsv.gz")
    with pytest.raises(ValueError, match=r"slots of 'GTGTAAATCGGX', which .* holds"):
        store.clones.import_alleles(
            chains, cell_selection=cells, integrations={"GTGTAAATCGGX": 4}
        )
    assert (
        set(store.artifacts.list(kind="lineage_alleles", scope="datastore")) == before
    )
