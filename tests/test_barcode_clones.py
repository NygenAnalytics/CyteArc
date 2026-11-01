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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from cytearc.datastore.datastore import DataStore
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.logging import collect_warnings
from tests.clone_helpers import (
    copy_clone_store,
    every_cell,
    import_calling_truth,
    loaded_partition,
    open_clone_store,
    smallest_budget,
    write_truth_store,
)
from tests.perturbation_helpers import traced_peak
from tests.clone_truth import (
    N_CELLS,
    SAMPLES,
    calling_truth,
    clone_truth,
)
from tests.storage_helpers import write_count_store

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
    return copy_clone_store(template, tmp_path)


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
    store = open_clone_store(folder / "trex.zarr")
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
        counts = dict(store.artifacts.load(clones).attrs)["rule_counts"]
        assert counts["molecules_of_low_complexity"] == 3_800 - 3_639
        # Most of its cloneIDs are errors of one, which correction merges, and
        # chance alone explains none of the merges.
        diagnostics = store.clones.load_clones(clones).diagnostics
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
    counts = attributes["rule_counts"]
    for name in (
        "barcodes_corrected",
        "pairs_of_one_umi_shared",
        "molecules_of_low_complexity",
        "nodes",
        "edges",
        "doublets",
    ):
        assert name in counts
    # Every clonal cell carries an error; the ambient molecules are shared.
    assert counts["barcodes_corrected"] > 0 and counts["pairs_of_one_umi_shared"] > 0
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
    # returns as the artifact records them.
    diagnostics = attributes["diagnostics"]
    assert dict(loaded.diagnostics) == diagnostics
    assert list(diagnostics) == [
        "distinctBarcodes",
        "expectedChanceMatches",
        "mergedBarcodes",
        "largestCloneShare",
    ]
    assert diagnostics["mergedBarcodes"] == counts["barcodes_corrected"]
    assert 0 < diagnostics["expectedChanceMatches"] < diagnostics["mergedBarcodes"]
    assert diagnostics["largestCloneShare"] == (
        loaded.clones["n_cells"].max() / loaded.clones["n_cells"].sum()
    )


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
        # The other sample's cells are not selected, so their barcodes match
        # no cell.
        with pytest.warns(UserWarning, match="match no selected cell"):
            alone = import_calling_truth(
                store,
                calling,
                tmp_path,
                "pycashier",
                cell_selection=store.snapshot_cell_selection(f"is_{sample}"),
            )
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
    with pytest.raises(ValueError, match="also a column of the result"):
        store.clones.barcode_clones(barcodes, preset="larry", within="doublet")


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


def test_diagnostics_other_than_the_four_numbers_are_refused_on_loading(
    store, tmp_path
) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    clones = store.clones.barcode_clones(barcodes, preset="larry")
    attributes = artifact_group(store.zw, clones).attrs
    recorded = dict(attributes["diagnostics"])
    for wrong, message in (
        ({**recorded, "mergedBarcodes": "3"}, "must be a number"),
        ({**recorded, "mergedBarcodes": True}, "must be a number"),
        ({**recorded, "rate": 0.5}, "exactly the diagnostics"),
        (
            {name: recorded[name] for name in list(recorded)[:3]},
            "exactly the diagnostics",
        ),
        ([1, 2], "exactly the diagnostics"),
    ):
        attributes["diagnostics"] = wrong
        with pytest.raises(ValueError, match=message):
            store.clones.load_clones(clones)
    # Clones called from barcodes without them were saved before they were
    # recorded, and loading them fails as reuse refuses them.
    del attributes["diagnostics"]
    with pytest.raises(ValueError, match="exactly the diagnostics"):
        store.clones.load_clones(clones)
    # Other clones record none.
    imported = store.clones.from_column("sample", cell_selection=every_cell(store))
    artifact_group(store.zw, imported).attrs["diagnostics"] = recorded
    with pytest.raises(ValueError, match="record no diagnostics"):
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
    reader = open_clone_store(Path(store.zarr_loc), zarr_mode="r")
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
        store = open_clone_store(
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
        stored[-1]["rule_counts"] = dict(group.attrs)["rule_counts"]
        stored[-1]["diagnostics"] = dict(group.attrs)["diagnostics"]
    # The files of about 400 reads or molecules came in chunks of 7 rows.
    assert chunks[0] > 40
    assert stored[0]["diagnostics"]["expectedChanceMatches"] > 0
    for other in stored[1:]:
        for name, values in stored[0].items():
            if name in ("rule_counts", "diagnostics"):
                # Equal to the last bit, the expectation's sum included.
                assert other[name] == values
            else:
                np.testing.assert_array_equal(other[name], values)


def test_barcode_clones_refuse_other_artifacts_and_arguments(store, tmp_path) -> None:
    _calling, barcodes = _imported(store, tmp_path, "larry")
    clones = store.clones.barcode_clones(barcodes, preset="larry")
    with pytest.raises(ValueError, match="lineage_barcodes artifact"):
        store.clones.barcode_clones(clones, preset="larry")
    with pytest.raises(TypeError, match="ArtifactRef"):
        store.clones.barcode_clones("barcodes", preset="larry")
    with pytest.raises(TypeError, match="invalidate_cache"):
        store.clones.barcode_clones(barcodes, preset="larry", invalidate_cache=1)
    with pytest.raises(TypeError, match="invalidate_cache must be a boolean"):
        store.clones.barcode_clones(barcodes, preset="larry", invalidate_cache=np.True_)
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
            barcodes, method="sets", distance=1, metric="hamming"
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
        """The bytes of the check of writing, which refuses right after reading's."""
        with zarr.config.set({"async.concurrency": concurrency}):
            budget = 1
            for _attempt in range(3):
                monkeypatch.setattr(store, "memoryBytes", budget)
                try:
                    store.clones.barcode_clones(
                        barcodes, preset="larry", invalidate_cache=True
                    )
                except MemoryError as error:
                    found = re.search(
                        r"about (\d+) bytes as they are (\w+)", str(error)
                    )
                    assert found is not None, str(error)
                    if found.group(2) == "written":
                        return int(found.group(1))
                    budget = int(found.group(1))
        raise AssertionError("writing was not refused right after reading")

    assert bytes_as_written(20) - bytes_as_written(10) == 10 * io.DEFAULT_BUFFER_SIZE


@pytest.mark.slow
@pytest.mark.parametrize("preset", ["trex", "celltag_multi"])
def test_calling_holds_no_more_than_the_smallest_budget_it_accepts(
    template, tmp_path, monkeypatch, preset
) -> None:
    store = copy_clone_store(template, tmp_path)
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
    counts = dict(store.artifacts.load(clones).attrs)["rule_counts"]
    assert counts["doublets"] == sum(run["counts"].get("doublets", [0]))
    assert counts["edges_pruned"] == sum(run["counts"]["bridges"])


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
