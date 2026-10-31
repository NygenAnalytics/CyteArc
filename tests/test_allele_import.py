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

from tests.clone_helpers import copy_clone_store, every_cell, write_truth_store
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
    return copy_clone_store(template, tmp_path)


def records_of(store, ref) -> pd.DataFrame:
    """An import's records, in the columns the truth's expectations use."""
    records = store.clones.load_alleles(ref).records
    allele = records["allele"].astype(object)
    frame = pd.DataFrame(
        {
            "cell": records["cell"].astype(str),
            "integration": records["integration"].astype(str),
            "site": records["site"].astype(str),
            "state": records["state"].astype(str),
            "allele": allele.where(allele.notna(), None),
            "umis": records["umis"].fillna(-1).astype(np.int64),
            "reads": records["reads"].fillna(-1).astype(np.int64),
        }
    )
    return frame.sort_values(_COLUMNS[:5], na_position="first").reset_index(drop=True)


def _normalised(frame: pd.DataFrame) -> pd.DataFrame:
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
        _normalised(records_of(store, ref)), _normalised(expected)
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
    assert group.attrs["n_beyond_length"] == beyond
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
    beyond = store.artifacts.load(first).attrs["n_beyond_length"]
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


def test_a_read_only_store_reuses_saved_alleles_and_refuses_new_ones(
    store, truth, recorders, tmp_path
):
    from tests.clone_helpers import open_clone_store

    path = write_petracer_characters(recorders, truth, tmp_path / "characters.csv.gz")
    cells = every_cell(store)
    ref = store.clones.import_alleles(path, cell_selection=cells)
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.clones.import_alleles(path, cell_selection=cells) == ref
    other = write_petracer_characters(
        recorders, truth, tmp_path / "zenodo.csv.gz", layout="zenodo"
    )
    with pytest.raises(PermissionError):
        reader.clones.import_alleles(other, cell_selection=cells)


def test_rows_without_an_integration_are_left_out_counted_and_warned(tmp_path):
    from tests.clone_helpers import open_clone_store
    from tests.storage_helpers import write_count_store

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
    store = open_clone_store(tmp_path / "store.zarr")
    selection = store.snapshot_cell_selection()
    message = f"{lacking} rows of the allele files name no integration"
    with pytest.warns(UserWarning, match=message):
        ref = store.clones.import_alleles(path, cell_selection=selection)
    with pytest.warns(UserWarning, match=message):
        assert store.clones.import_alleles(path, cell_selection=selection) == ref
    records = store.clones.load_alleles(ref).records
    assert len(records) == 3 * (len(raw) - lacking)
    assert store.artifacts.load(ref).attrs["n_rows_without_integration"] == lacking
