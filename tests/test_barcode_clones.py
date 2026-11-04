"""Clones called from imported lineage barcodes recover the truth and TREX's own clones.

Each preset reads the calling truth in the layout of its tool, with the noise
that its published rules remove, and must give back the truth's clones that
those rules call. TREX's expected outputs for 10x and Smart-seq3 (MIT) are
checked in: ``preset="trex"`` on their ``molecules.txt`` must give the clones
of their ``clones.txt``.
"""

import copy
import gzip
import io
import json
import re
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc.datastore._operations.clone_artifacts import clone_write_bytes
from cytearc.datastore.datastore import DataStore
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    every_cell,
    import_calling_truth,
    loaded_partition,
    write_truth_store,
)
from tests.memory_helpers import smallest_budget, traced_peak
from tests.clone_truth import (
    N_CELLS,
    SAMPLES,
    calling_truth,
    clone_truth,
)
from tests.storage_helpers import copy_store, open_store, write_count_store

TRUTH = clone_truth()
DATA = Path(__file__).parent / "data" / "barcodes"
# The noise each preset's rules remove, its layout, and the truth's clones
# that its rules call: those of enough barcodes and cells.
PRESET_CASES = {
    "larry": (("amplified", "ambient", "indel", "partial"), "pycashier", 1, 1),
    "trex": (("amplified", "ambient", "dropout", "partial"), "trex", 1, 1),
    "celltag": (
        (
            "amplified",
            "ambient",
            "dropout",
            "indel",
            "partial",
            "contaminant",
            "crowded",
        ),
        "celltag",
        2,
        3,
    ),
    "celltag_multi": (
        ("amplified", "ambient", "dropout", "indel", "contaminant", "crowded"),
        "celltag_multi",
        2,
        2,
    ),
}


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("barcode_clones") / "store.zarr"
    write_truth_store(path, TRUTH)
    return path


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    return copy_store(template, tmp_path)


def _imported(store: DataStore, folder: Path, preset: str, **options):
    noise, layout, _barcodes, _size = PRESET_CASES[preset]
    calling = calling_truth(TRUTH, noise=noise, **options)
    with collect_warnings():
        barcodes = import_calling_truth(store, calling, folder, layout)
    return calling, barcodes


@pytest.mark.parametrize("preset", list(PRESET_CASES))
def test_each_preset_recovers_the_truth_under_the_noise_its_rules_remove(
    store, tmp_path, preset
) -> None:
    calling, barcodes = _imported(store, tmp_path, preset)
    _noise, _layout, min_barcodes, min_size = PRESET_CASES[preset]
    allowlist = list(calling.allowlist) if preset.startswith("celltag") else None
    clones = store.clones.barcode_clones(barcodes, preset=preset, allowlist=allowlist)
    expected = calling.expected_partition(
        TRUTH, min_barcodes=min_barcodes, min_size=min_size
    )
    assert loaded_partition(store, clones) == expected
    # The largest of the truth's clones over the cells of every clone.
    diagnostics = store.clones.load_clones(clones).diagnostics
    sizes = sorted(map(len, expected))
    assert diagnostics["largestCloneShare"] == sizes[-1] / sum(sizes)
    assert diagnostics["distinctBarcodes"] >= diagnostics["mergedBarcodes"]
    # The noise is there to remove: the same barcodes without the rules call
    # other clones.
    loose = store.clones.barcode_clones(barcodes, method="sets")
    assert loaded_partition(store, loose) != expected


def test_celltag_rules_need_their_allowlist_and_cell_filter(store, tmp_path) -> None:
    calling, barcodes = _imported(store, tmp_path, "celltag")
    allowlist = list(calling.allowlist)
    expected = calling.expected_partition(TRUTH, min_barcodes=2, min_size=3)
    ids = TRUTH.cell_ids
    contaminated = set(
        ids[calling.barcodes.molecules.query("kind == 'contaminant'")["cell"].unique()]
    )
    crowded = set(
        ids[calling.barcodes.molecules.query("kind == 'crowded'")["cell"].unique()]
    )
    # Without the allowlist, the three cells of the contaminant form a clone.
    without = store.clones.barcode_clones(barcodes, preset="celltag", allowlist=False)
    assert loaded_partition(store, without) == expected | {frozenset(contaminated)}
    # Without the cell filter, so do the three crowded cells, and clone 1,
    # whose cells carry one CellTag.
    unfiltered = store.clones.barcode_clones(
        barcodes, preset="celltag", allowlist=allowlist, barcodes_per_cell=(1, None)
    )
    ones = calling.expected_partition(TRUTH, min_barcodes=1, min_size=3)
    assert loaded_partition(store, unfiltered) == ones | {frozenset(crowded)}
    with pytest.raises(ValueError, match="allowlist=False"):
        store.clones.barcode_clones(barcodes, preset="celltag")


def _trex_store(folder: Path, molecules: Path) -> DataStore:
    with gzip.open(molecules, "rt") as handle:
        cells = sorted(
            {line.split("\t")[0] for line in handle if not line.startswith("#")}
        )
    counts = np.ones((len(cells), 2), dtype=np.uint16)
    write_count_store(
        str(folder / "trex.zarr"), {"RNA": counts}, "uint16", cell_ids=cells
    )
    store = open_store(folder / "trex.zarr")
    store.cells.insert("every_cell", np.ones(len(cells), dtype=bool))
    return store


def _trex_partition(path: Path) -> set[frozenset[str]]:
    table = pd.read_csv(path, sep="\t", dtype=str)
    return {frozenset(rows["cell_id"]) for _, rows in table.groupby("clone_nr")}


@pytest.mark.parametrize("run", ["10x", "smartseq3"])
@pytest.mark.parametrize("doublets", [None, False])
def test_the_trex_preset_gives_trexs_own_clones(tmp_path, run, doublets) -> None:
    molecules = DATA / f"trex_{run}_molecules.txt.gz"
    store = _trex_store(tmp_path, molecules)
    barcodes = store.clones.import_barcodes(molecules, cell_selection=every_cell(store))
    clones = store.clones.barcode_clones(barcodes, preset="trex", doublets=doublets)
    expected = _trex_partition(DATA / f"trex_{run}_clones.txt.gz")
    assert loaded_partition(store, clones) == expected
    loaded = store.clones.load_clones(clones)
    assert not loaded.cells["doublet"].any()
    if run == "10x":
        # TREX's log: 3,639 of its 3,800 molecules remain after the
        # low-complexity filter.
        diagnostics = store.clones.load_clones(clones).diagnostics
        assert diagnostics["moleculesOfLowComplexity"] == 3_800 - 3_639
        # Most of its cloneIDs are errors of one, which correction merges, and
        # chance alone explains none of the merges.
        assert diagnostics["mergedBarcodes"] == 89
        assert diagnostics["expectedChanceMatches"] < 1
    # TREX's clones.txt lists the cells of cells_filtered.txt, one cloneID or
    # more each, and singletons are clones.
    assert int(loaded.cells["clone"].notna().sum()) == sum(map(len, expected))
    assert (loaded.cells.loc[loaded.cells["clone"].notna(), "n_barcodes"] > 0).all()


def test_the_artifact_holds_barcodes_doublets_and_what_each_rule_removed(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "trex")
    clones = store.clones.barcode_clones(barcodes, preset="trex")
    group = store.artifacts.load(clones)
    attributes = dict(group.attrs)
    assert attributes["library_units"] == [{"library": "trex", "unit": "umis"}]
    assert "rule_counts" not in attributes
    loaded = store.clones.load_clones(clones)
    assert loaded.source == "call_barcode_clones"
    assert list(loaded.cells.columns) == [
        "clone",
        "clone_size",
        "n_barcodes",
        "doublet",
    ]
    barcode_counts = store.clones.load_barcodes(barcodes).cells["n_barcodes"]
    # The rules leave each cell at most the barcodes it was read with.
    assert (loaded.cells["n_barcodes"] <= barcode_counts).all()
    record = store.artifacts.inspect(clones)
    assert record.parameters["preset"] == "trex"
    assert record.parameters["similarity"] == 0.7
    assert record.parameters["trex_rules"] is True
    assert record.operation == "call_barcode_clones"
    # The diagnostics are one flat mapping of numbers, which load_clones
    # returns as the artifact records them: a count per rule, None for a rule
    # that TREX's preset does not run.
    diagnostics = attributes["diagnostics"]
    assert dict(loaded.diagnostics) == diagnostics
    assert list(diagnostics) == [
        "cellsWithClone",
        "largestCloneShare",
        "distinctBarcodes",
        "expectedChanceMatches",
        "mergedBarcodes",
        "moleculesBelowMinReads",
        "moleculesOfLowComplexity",
        "recordsCorrected",
        "recordsOutsideAllowlist",
        "moleculesOfFewBases",
        "pairs",
        "pairsOfOneUmiShared",
        "pairsOfOneRead",
        "pairsBelowMinUmis",
        "pairsBelowMinFraction",
        "cellsWithBarcodes",
        "cellsOutsideBarcodesPerCell",
        "nodes",
        "edges",
        "edgesPruned",
        "doublets",
        "cellsInSmallClones",
    ]
    assert [name for name, value in diagnostics.items() if value is None] == [
        "moleculesBelowMinReads",
        "recordsOutsideAllowlist",
        "pairsBelowMinUmis",
        "pairsBelowMinFraction",
        "cellsOutsideBarcodesPerCell",
    ]
    # Every clonal cell carries an error; the ambient molecules are shared.
    assert diagnostics["mergedBarcodes"] > 0
    assert diagnostics["recordsCorrected"] >= diagnostics["mergedBarcodes"]
    assert diagnostics["pairsOfOneUmiShared"] > 0
    assert diagnostics["cellsWithClone"] == loaded.clones["n_cells"].sum()
    # Every cell with a barcode after the pair filters is counted, and those
    # in no clone are its doublets and the cells of clones below min_size.
    assert diagnostics["cellsWithBarcodes"] == int(
        (loaded.cells["n_barcodes"] > 0).sum()
    )
    assert diagnostics["doublets"] == int(loaded.cells["doublet"].sum())
    assert 0 < diagnostics["expectedChanceMatches"] < diagnostics["mergedBarcodes"]
    assert diagnostics["largestCloneShare"] == (
        loaded.clones["n_cells"].max() / loaded.clones["n_cells"].sum()
    )


def test_trex_rules_follow_the_preset_and_turn_on_or_off_in_any_call(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "trex")
    trex = store.clones.barcode_clones(barcodes, preset="trex")
    # True is what the preset sets, so the call is the same.
    assert store.clones.barcode_clones(barcodes, preset="trex", trex_rules=True) == trex
    without = store.clones.barcode_clones(barcodes, preset="trex", trex_rules=False)
    assert without != trex
    record = store.artifacts.inspect(without)
    assert record.parameters["trex_rules"] is False
    assert record.parameters == {
        **store.artifacts.inspect(trex).parameters,
        "trex_rules": False,
    }
    counts = store.artifacts.inspect(without).diagnostics
    assert counts["moleculesOfLowComplexity"] is None
    assert counts["pairsOfOneUmiShared"] is None
    assert store.artifacts.inspect(trex).diagnostics["moleculesOfLowComplexity"] >= 0
    # TREX's settings given one by one, with its rules, call TREX's clones.
    spelled = store.clones.barcode_clones(
        barcodes,
        max_distance=5,
        metric="hamming",
        method="jaccard",
        similarity=0.7,
        inclusive=False,
        prune=True,
        doublets=True,
        min_size=1,
        trex_rules=True,
    )
    assert spelled != trex
    pd.testing.assert_series_equal(
        store.clones.load_clones(spelled).cells["clone"],
        store.clones.load_clones(trex).cells["clone"],
    )
    with pytest.raises(TypeError, match="trex_rules must be a boolean"):
        store.clones.barcode_clones(barcodes, preset="trex", trex_rules="no")


def test_saved_clones_are_reused_only_with_their_barcodes_and_doublets(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "trex")
    clones = store.clones.barcode_clones(barcodes, preset="trex")
    group = artifact_group(store.zw, clones)
    # Each cell's barcodes and doublet flag are stored as the clones artifact
    # declares them.
    assert group["n_barcodes"].dtype == np.dtype(np.int32)
    assert group["doublet"].dtype == np.dtype(bool)
    doublet = np.asarray(group["doublet"][:])
    assert store.clones.barcode_clones(barcodes, preset="trex") == clones
    # Clones that lost one of the arrays that their source adds are called
    # again, as a new artifact, rather than reused without it.
    del group["doublet"]
    again = store.clones.barcode_clones(barcodes, preset="trex")
    assert again != clones
    np.testing.assert_array_equal(store.artifacts.load(again)["doublet"][:], doublet)
    loaded = store.clones.load_clones(again)
    np.testing.assert_array_equal(loaded.cells["doublet"], doublet)


def test_barcode_clones_store_each_array_at_its_dtype_chunks_and_codecs(
    store, tmp_path
) -> None:
    # The shared clone writer stores the arrays as the clone operations'
    # writer did: labels as fixed-width text at the width of the longest,
    # one chunk for these 48 cells, compressed by blosc.
    _calling, barcodes = _imported(store, tmp_path, "trex")
    group = artifact_group(
        store.zw, store.clones.barcode_clones(barcodes, preset="trex")
    )
    n_clones = int(group["clone_sizes"].shape[0])
    width = len(f"clone_{n_clones}")
    expected = {
        "values": (np.dtype(f"<U{width}"), N_CELLS),
        "codes": (np.dtype(np.int32), N_CELLS),
        "clone_size": (np.dtype(np.int64), N_CELLS),
        "n_barcodes": (np.dtype(np.int32), N_CELLS),
        "doublet": (np.dtype(bool), N_CELLS),
        "clone_ids": (np.dtype(f"<U{width}"), n_clones),
        "clone_sizes": (np.dtype(np.int64), n_clones),
    }
    assert set(group.array_keys()) == set(expected)
    for name, (dtype, rows) in expected.items():
        array = group[name]
        assert (array.dtype, array.shape, array.chunks) == (dtype, (rows,), (rows,))
        codecs = [codec["name"] for codec in array.metadata.to_dict()["codecs"]]
        assert codecs == ["bytes", "blosc"], (name, codecs)


def test_clones_within_groups_are_those_of_each_group_alone(store, tmp_path) -> None:
    calling, barcodes = _imported(store, tmp_path, "larry")
    within = store.clones.barcode_clones(barcodes, preset="larry", within="sample")
    expected = {
        frozenset(cell for cell in clone if cell.startswith(f"{sample}__"))
        for clone in calling.expected_partition(TRUTH)
        for sample in SAMPLES
    } - {frozenset()}
    assert loaded_partition(store, within) == expected
    for sample in SAMPLES:
        store.cells.insert(f"is_{sample}", TRUTH.samples == sample)
        # The other sample's cells are cells of the store outside the
        # selection, which give no warning.
        with collect_warnings() as messages:
            alone = import_calling_truth(
                store,
                calling,
                tmp_path,
                "pycashier",
                cell_selection=store.snapshot_cell_selection(f"is_{sample}"),
            )
        assert list(messages) == []
        listed = {
            barcode
            for other in SAMPLES
            if other != sample
            for barcode in calling.barcodes.sample_molecules(other)["cell_barcode"]
        }
        joined = store.artifacts.inspect(alone).diagnostics
        assert joined["unselectedBarcodes"] == len(listed) > 0
        assert joined["unmatchedBarcodes"] == 0
        clones = store.clones.barcode_clones(alone, preset="larry")
        assert loaded_partition(store, clones) == {
            clone for clone in expected if next(iter(clone)).startswith(sample)
        }


def test_other_labels_of_one_within_column_call_other_clones(store, tmp_path) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    store.cells.insert("batch", TRUTH.samples.astype(object))
    by_sample = store.clones.barcode_clones(barcodes, preset="larry", within="batch")
    # The column keeps its name and puts every cell in one batch: the labels
    # identify the clones, so they are called again, as without groups.
    store.cells.insert("batch", np.full(N_CELLS, "all", dtype=object), overwrite=True)
    together = store.clones.barcode_clones(barcodes, preset="larry", within="batch")
    assert together != by_sample
    alone = store.clones.barcode_clones(barcodes, preset="larry")
    assert loaded_partition(store, together) == loaded_partition(store, alone)
    assert loaded_partition(store, together) != loaded_partition(store, by_sample)


def test_numpys_false_calls_a_celltag_preset_without_its_allowlist(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "celltag")
    without = store.clones.barcode_clones(barcodes, preset="celltag", allowlist=False)
    assert (
        store.clones.barcode_clones(barcodes, preset="celltag", allowlist=np.False_)
        == without
    )
    with pytest.raises(TypeError, match="sequence of barcodes"):
        store.clones.barcode_clones(barcodes, preset="celltag", allowlist=np.True_)


def test_within_needs_a_label_for_every_cell_with_a_barcode(store, tmp_path) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    labels = TRUTH.samples.astype(object).copy()
    with_barcodes = (
        store.clones.load_barcodes(barcodes).cells["n_barcodes"].to_numpy() > 0
    )
    labels[np.flatnonzero(with_barcodes)[0]] = None
    store.cells.insert("partly", labels)
    with pytest.raises(ValueError, match="no label for 1 cells with a barcode"):
        store.clones.barcode_clones(barcodes, preset="larry", within="partly")
    with pytest.raises(ValueError, match="not a cell metadata column"):
        store.clones.barcode_clones(barcodes, preset="larry", within="mouse")


def test_within_may_name_any_column_since_its_labels_join_no_table(
    store, tmp_path
) -> None:
    # The within labels are recorded by their fingerprint and never join the
    # loaded clones, so a column named like one of their columns keeps the
    # clones apart as any other column does.
    _calling, barcodes = _imported(store, tmp_path, "larry")
    store.cells.insert("sample_copy", TRUTH.samples.astype(object))
    expected = store.clones.load_clones(
        store.clones.barcode_clones(barcodes, preset="larry", within="sample_copy")
    ).cells["clone"]
    for name in ("clone", "clone_size", "n_barcodes", "doublet"):
        store.cells.insert(name, TRUTH.samples.astype(object))
        loaded = store.clones.load_clones(
            store.clones.barcode_clones(barcodes, preset="larry", within=name)
        )
        assert list(loaded.cells.columns) == [
            "clone",
            "clone_size",
            "n_barcodes",
            "doublet",
        ]
        pd.testing.assert_series_equal(loaded.cells["clone"], expected)


def test_saved_clones_are_reused_by_their_resolved_rules(store, tmp_path) -> None:
    calling, barcodes = _imported(store, tmp_path, "celltag")
    allowlist = list(calling.allowlist)
    clones = store.clones.barcode_clones(
        barcodes, preset="celltag", allowlist=allowlist
    )
    # The same rules and an allowlist of the same barcodes, in any order.
    again = store.clones.barcode_clones(
        barcodes, preset="celltag", allowlist=allowlist[::-1] + allowlist[:3]
    )
    assert again == clones
    other = store.clones.barcode_clones(
        barcodes, preset="celltag", allowlist=allowlist[1:]
    )
    assert other != clones
    assert store.clones.barcode_clones(
        barcodes, preset="celltag", allowlist=allowlist, min_size=2
    ) not in (clones, other)
    recomputed = store.clones.barcode_clones(
        barcodes, preset="celltag", allowlist=allowlist, invalidate_cache=True
    )
    assert recomputed.kind == "clones"
    np.testing.assert_array_equal(
        store.artifacts.load_values(recomputed).values,
        store.artifacts.load_values(clones).values,
    )


def test_reused_and_recomputed_clones_give_the_same_diagnostics(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "trex")
    clones = store.clones.barcode_clones(barcodes, preset="trex")
    diagnostics = dict(store.clones.load_clones(clones).diagnostics)
    again = store.clones.barcode_clones(barcodes, preset="trex")
    assert again == clones
    assert dict(store.clones.load_clones(again).diagnostics) == diagnostics
    recomputed = store.clones.barcode_clones(
        barcodes, preset="trex", invalidate_cache=True
    )
    assert recomputed != clones
    assert dict(store.clones.load_clones(recomputed).diagnostics) == diagnostics
    # Saved clones without diagnostics are called again rather than reused.
    for saved in (clones, recomputed):
        del artifact_group(store.zw, saved).attrs["diagnostics"]
    called = store.clones.barcode_clones(barcodes, preset="trex")
    assert called not in (clones, recomputed)
    assert dict(store.clones.load_clones(called).diagnostics) == diagnostics


def test_diagnostics_other_than_those_declared_are_refused_on_loading(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    clones = store.clones.barcode_clones(barcodes, preset="larry")
    attributes = artifact_group(store.zw, clones).attrs
    recorded = dict(attributes["diagnostics"])
    for wrong, message in (
        ({**recorded, "expectedChanceMatches": float("inf")}, "must be finite"),
        ({**recorded, "mergedBarcodes": True}, "not the boolean True"),
        ({**recorded, "rate": 0.5}, "not declared rate"),
        (
            {name: recorded[name] for name in list(recorded)[:3]},
            "missing expectedChanceMatches, mergedBarcodes",
        ),
        ([1, 2], "must be a mapping"),
    ):
        attributes["diagnostics"] = wrong
        with pytest.raises(ValueError, match=message):
            store.clones.load_clones(clones)
    # Clones called from barcodes without them were saved before they were
    # recorded, and loading them names the call that saves them anew.
    del attributes["diagnostics"]
    with pytest.raises(ValueError, match="call ds.clones.barcode_clones again"):
        store.clones.load_clones(clones)
    # Clones from a column record the clone writer's alone.
    imported = store.clones.from_column("sample", cell_selection=every_cell(store))
    artifact_group(store.zw, imported).attrs["diagnostics"] = recorded
    with pytest.raises(
        ValueError, match="not declared cellsInSmallClones, .*distinctBarcodes"
    ):
        store.clones.load_clones(imported)


def test_loaded_clones_pickle_and_copy_with_their_diagnostics(store, tmp_path) -> None:
    _calling, barcodes = _imported(store, tmp_path, "trex")
    loaded = store.clones.load_clones(
        store.clones.barcode_clones(barcodes, preset="trex")
    )
    for copied in (pickle.loads(pickle.dumps(loaded)), copy.deepcopy(loaded)):
        assert copied.diagnostics == loaded.diagnostics
        assert copied.source == loaded.source
        pd.testing.assert_frame_equal(copied.cells, loaded.cells)


def test_a_read_only_store_reuses_saved_clones_and_refuses_new_ones(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    clones = store.clones.barcode_clones(barcodes, preset="larry")
    reader = open_store(Path(store.zarr_loc), zarr_mode="r")
    assert reader.clones.barcode_clones(barcodes, preset="larry") == clones
    with pytest.raises(PermissionError, match="call_barcode_clones"):
        reader.clones.barcode_clones(barcodes, preset="larry", min_umis=4)


@pytest.mark.parametrize("preset", ["trex", "celltag_multi"])
def test_calls_follow_neither_file_order_nor_chunks_nor_threads(
    tmp_path, monkeypatch, preset
) -> None:
    from cytearc.clones.barcodes import BarcodeCounter
    from cytearc.readers import lineage_barcodes as reader

    noise, layout, _barcodes, _size = PRESET_CASES[preset]
    chunks = [0]
    add = BarcodeCounter.add

    def counted(self, *args, **kwargs):
        chunks[0] += 1
        return add(self, *args, **kwargs)

    calling = calling_truth(TRUTH, noise=noise)
    allowlist = list(calling.allowlist) if preset == "celltag_multi" else None
    stored = []
    # No step of the calling runs on threads; the case of 4 keeps a later one
    # from changing the calls.
    for case in ("plain", "files reversed", "chunks of 7 rows", "4 threads"):
        folder = tmp_path / case.replace(" ", "_")
        folder.mkdir()
        write_truth_store(folder / "store.zarr", TRUTH)
        store = open_store(
            folder / "store.zarr", nthreads=4 if case == "4 threads" else 1
        )
        with monkeypatch.context() as patched:
            if case == "chunks of 7 rows":
                # Each file is read 7 rows at a time, and every chunk merges.
                patched.setattr(
                    reader.BarcodeLayout, "chunk_rows", lambda self, chunk_bytes: 7
                )
                patched.setattr(store, "_merge_rows", lambda: 1)
                patched.setattr(BarcodeCounter, "add", counted)
            with collect_warnings():
                barcodes = import_calling_truth(
                    store,
                    calling,
                    folder,
                    layout,
                    samples=SAMPLES[::-1] if case == "files reversed" else SAMPLES,
                )
        clones = store.clones.barcode_clones(
            barcodes, preset=preset, allowlist=allowlist
        )
        group = store.artifacts.load(clones)
        stored.append(
            {
                name: np.asarray(group[name][:])
                for name in ("values", "codes", "n_barcodes", "doublet", "clone_sizes")
            }
        )
        stored[-1]["diagnostics"] = dict(group.attrs)["diagnostics"]
    # The files of about 400 reads or molecules came in chunks of 7 rows.
    assert chunks[0] > 40
    assert stored[0]["diagnostics"]["expectedChanceMatches"] > 0
    for other in stored[1:]:
        for name, values in stored[0].items():
            if name == "diagnostics":
                # Equal to the last bit, the expectation's sum included.
                assert other[name] == values
            else:
                np.testing.assert_array_equal(other[name], values)


def test_edits_beyond_eight_correct_barcodes_with_no_expectation_saved(
    store, tmp_path
) -> None:
    # A diagnostic never refuses a call: correction within nine edits runs,
    # and the pairs expected by chance within them, which take too long to
    # count, are saved as not measured.
    _calling, barcodes = _imported(store, tmp_path, "larry")
    eight, nine = (
        store.artifacts.inspect(
            store.clones.barcode_clones(
                barcodes, preset="larry", metric="levenshtein", max_distance=distance
            )
        ).diagnostics
        for distance in (8, 9)
    )
    assert nine["expectedChanceMatches"] is None
    assert eight["expectedChanceMatches"] is not None
    # A wider distance merges every barcode that a narrower one merges.
    assert nine["mergedBarcodes"] >= eight["mergedBarcodes"] > 0
    assert nine["distinctBarcodes"] == eight["distinctBarcodes"]


def test_barcode_clones_refuse_other_artifacts_and_arguments(store, tmp_path) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    clones = store.clones.barcode_clones(barcodes, preset="larry")
    with pytest.raises(ValueError, match="lineage_barcodes artifact"):
        store.clones.barcode_clones(clones, preset="larry")
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.clones.barcode_clones("barcodes", preset="larry")
    with pytest.raises(TypeError, match="invalidate_cache"):
        store.clones.barcode_clones(barcodes, preset="larry", invalidate_cache=1)
    # A NumPy boolean is a flag: it computes the clones anew.
    recomputed = store.clones.barcode_clones(
        barcodes, preset="larry", invalidate_cache=np.True_
    )
    assert recomputed != clones
    assert (
        store.artifacts.inspect(recomputed).parameters
        == store.artifacts.inspect(clones).parameters
    )
    with pytest.raises(ValueError, match="needs a preset or a method"):
        store.clones.barcode_clones(barcodes)
    with pytest.raises(TypeError, match="as text"):
        store.clones.barcode_clones(barcodes, method="sets", allowlist=[1, 2])


def test_barcode_clones_name_missing_and_incomplete_barcodes(store, tmp_path) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    missing = ArtifactRef(
        scope="datastore", kind="lineage_barcodes", artifact_id="f" * 64, assay=None
    )
    with pytest.raises(KeyError, match="Artifact does not exist"):
        store.clones.barcode_clones(missing, preset="larry")
    # The arguments are checked before the store is read.
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.barcode_clones(missing, preset="larry", invalidate_cache=1)
    # An import that stopped before it finished leaves an incomplete artifact.
    artifact_group(store.zw, barcodes).attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        store.clones.barcode_clones(barcodes, preset="larry")


def test_counts_that_a_table_leaves_unknown_are_refused_by_the_rules_that_read_them(
    store, tmp_path
) -> None:
    # A pairs table whose UMIs one row leaves out: its library has no unit.
    rows = ["cell\tbarcode\tumis"] + [
        f"{TRUTH.cell_ids[cell]}\tBC{cell % 4}\t{'NA' if cell == 5 else 3}"
        for cell in range(N_CELLS)
    ]
    path = tmp_path / "pairs.tsv"
    path.write_text("\n".join(rows) + "\n")
    barcodes = store.clones.import_barcodes(
        path,
        cell_selection=every_cell(store),
        format="pairs",
        columns={"cell": "cell", "barcode": "barcode", "umis": "umis"},
    )
    loaded = store.clones.load_barcodes(barcodes)
    assert loaded.cells["n_umis"].iloc[5] == -1
    with pytest.raises(ValueError, match="counts UMIs for 47 of its 48 pairs"):
        store.clones.barcode_clones(barcodes, method="sets", min_umis=2)
    with pytest.raises(ValueError, match="Barcode correction counts each barcode"):
        store.clones.barcode_clones(
            barcodes, method="sets", max_distance=1, metric="hamming"
        )
    # Rules that read no count call clones from every pair.
    clones = store.clones.barcode_clones(barcodes, method="sets")
    assert len(loaded_partition(store, clones)) == 4
    assert dict(store.artifacts.load(clones).attrs)["library_units"] == [
        {"library": "pairs", "unit": None}
    ]


def test_calling_refuses_a_budget_too_small_before_writing(
    store, tmp_path, monkeypatch
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "trex")
    before = len(store.artifacts.list(kind="clones", scope="datastore"))
    monkeypatch.setattr(store, "memoryBytes", 50_000)
    with pytest.raises(MemoryError, match="Raise mem_budget"):
        store.clones.barcode_clones(barcodes, preset="trex")
    assert len(store.artifacts.list(kind="clones", scope="datastore")) == before


def test_the_write_check_counts_a_buffer_for_each_write_zarr_runs_at_once(
    store, tmp_path, monkeypatch
) -> None:
    # Zarr writes each chunk and each array's metadata through a buffered
    # file, as many at once as its async concurrency allows, and Python 3.14
    # made each buffer 128 KiB, where it was the file system's block.
    _calling, barcodes = _imported(store, tmp_path, "larry")

    def bytes_as_written(concurrency: int) -> int:
        """The bytes of the check of writing, which refuses after grouping's and reading's."""
        with zarr.config.set({"async.concurrency": concurrency}):
            budget = 1
            for _attempt in range(4):
                monkeypatch.setattr(store, "memoryBytes", budget)
                try:
                    store.clones.barcode_clones(
                        barcodes, preset="larry", invalidate_cache=True
                    )
                except MemoryError as error:
                    found = re.search(r"about (\d+) bytes", str(error))
                    assert found is not None, str(error)
                    if "bytes as they are written" in str(error):
                        return int(found.group(1))
                    budget = int(found.group(1))
        raise AssertionError("writing was not refused after reading")

    assert bytes_as_written(20) - bytes_as_written(10) == 10 * io.DEFAULT_BUFFER_SIZE
    # Before reading, saving is checked as if every cell were a clone of its
    # own: 24 bytes per cell for its group and the calls' flags and barcodes,
    # each clone's label as a Python string at the size of the longest, and
    # what saving that many clones holds.
    longest = f"clone_{N_CELLS}"
    with zarr.config.set({"async.concurrency": 10}):
        saving = clone_write_bytes(
            N_CELLS,
            N_CELLS,
            len(longest),
            source_arrays={"n_barcodes": np.int32, "doublet": bool},
        )
    assert bytes_as_written(10) == (24 + sys.getsizeof(longest)) * N_CELLS + saving


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["trex", "celltag_multi"])
def test_calling_holds_no_more_than_the_smallest_budget_it_accepts(
    template, tmp_path, monkeypatch, preset
) -> None:
    store = copy_store(template, tmp_path)
    calling, barcodes = _imported(store, tmp_path, preset)
    allowlist = list(calling.allowlist) if preset == "celltag_multi" else None

    def call(budget: int) -> object:
        monkeypatch.setattr(store, "memoryBytes", budget)
        return store.clones.barcode_clones(
            barcodes, preset=preset, allowlist=allowlist, invalidate_cache=True
        )

    budget = smallest_budget(call, 1 << 30)
    _, peak = traced_peak(lambda: call(budget), budget)
    assert peak <= budget


# TREX 0.4.dev412 (commit c1ade47) ran its own run10x steps on a synthetic
# molecule table with doublets, clones that share cloneIDs, dropout, errors,
# partial and low-complexity cloneIDs and ambient molecules, at several
# Jaccard thresholds, with and without doublet removal; the script stays in
# the stack's work folder.
GOLDEN = json.loads((Path(__file__).parent / "trex_0_4_dev412_golden.json").read_text())


@pytest.fixture(scope="module")
def golden_barcodes(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trex_golden")
    molecules = DATA / "trex_doublets_molecules.txt.gz"
    store = _trex_store(folder, molecules)
    return store, store.clones.import_barcodes(
        molecules, cell_selection=every_cell(store)
    )


@pytest.mark.parametrize(
    "run",
    GOLDEN["runs"],
    ids=[
        f"{run['jaccard_threshold']}-keep{run['keep_doublets']}"
        for run in GOLDEN["runs"]
    ],
)
def test_the_trex_preset_matches_trexs_clones_and_doublets(
    golden_barcodes, run
) -> None:
    store, barcodes = golden_barcodes
    clones = store.clones.barcode_clones(
        barcodes,
        preset="trex",
        similarity=run["jaccard_threshold"],
        doublets=not run["keep_doublets"],
    )
    assert loaded_partition(store, clones) == {
        frozenset(members) for members in run["clones"]
    }
    cells = store.clones.load_clones(clones).cells
    assert sorted(cells.index[cells["doublet"]]) == run["doublets"]
    counts = store.artifacts.inspect(clones).diagnostics
    # A run that keeps its doublets removes none, which is not counted.
    assert counts["doublets"] == (
        None if run["keep_doublets"] else sum(run["counts"].get("doublets", [0]))
    )
    assert counts["edgesPruned"] == sum(run["counts"]["bridges"])


def test_the_golden_holds_doublets_that_trex_removed() -> None:
    removed = {
        run["jaccard_threshold"]: len(run["doublets"])
        for run in GOLDEN["runs"]
        if not run["keep_doublets"]
    }
    assert removed[0.0] > 0 and removed[0.3] > 0


def test_an_allowlist_too_large_for_the_budget_is_refused_before_it_is_copied(
    store, tmp_path, monkeypatch
) -> None:
    calling, barcodes = _imported(store, tmp_path, "celltag")
    # A fixed-width array's barcodes become Python strings, which count too.
    allowlist = np.asarray(calling.allowlist)
    needed = 300 * len(allowlist) + len(allowlist) * (allowlist.dtype.itemsize // 4)
    monkeypatch.setattr(store, "memoryBytes", needed - 1)
    with pytest.raises(MemoryError, match=f"allowlist of {len(allowlist)} barcodes"):
        store.clones.barcode_clones(barcodes, preset="celltag", allowlist=allowlist)
    monkeypatch.setattr(store, "memoryBytes", needed)
    with pytest.raises(MemoryError, match="Raise mem_budget") as raised:
        store.clones.barcode_clones(barcodes, preset="celltag", allowlist=allowlist)
    assert "allowlist" not in str(raised.value)
    with pytest.raises(TypeError, match="sequence of barcodes"):
        store.clones.barcode_clones(
            barcodes, preset="celltag", allowlist=iter(allowlist)
        )


@pytest.mark.parametrize("n_columns", [1, 13])
def test_clones_within_a_label_of_each_cell_fit_the_smallest_budget(
    tmp_path, n_columns
) -> None:
    from tests.barcode_join_helpers import write_barcode_file
    from tests.memory_helpers import budget_that_runs, stated_bytes

    # 2,000 cells of a barcode each and a label of their own in each within
    # column, so each clone is a group, and fingerprinting the groups' labels
    # holds about as much as the groups: the clone call holds within the
    # smallest budget that its refusals name.
    n_cells = 2_000
    ids = [f"cell{index:07d}-1" for index in range(n_cells)]
    path = tmp_path / "store.zarr"
    write_count_store(
        str(path), {"RNA": np.ones((n_cells, 1), dtype=int)}, "uint16", cell_ids=ids
    )
    store = open_store(path)
    store.cells.insert("every_cell", np.ones(n_cells, dtype=bool))
    names = [f"each_{k}" for k in range(n_columns)]
    for name in names:
        store.cells.insert(
            name, np.asarray([f"c{index:06d}" for index in range(n_cells)])
        )
    with collect_warnings():
        barcodes = store.clones.import_barcodes(
            write_barcode_file(tmp_path / "counts.tsv", ids),
            cell_selection=store.snapshot_cell_selection("every_cell"),
        )

    def run(opened) -> object:
        with collect_warnings():
            return opened.clones.barcode_clones(
                barcodes,
                preset="larry",
                min_reads=0,
                min_umis=1,
                within=names,
                invalidate_cache=True,
            )

    run(store)
    budget, _refusals, _result, peak = budget_that_runs(
        store, run, refusal_budget=stated_bytes, max_refusals=100
    )
    assert peak is not None and peak <= budget, (peak, budget)
