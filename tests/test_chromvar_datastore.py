import dataclasses
import shutil

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csc_matrix
from zarr.storage import MemoryStore

from cytearc.datastore._operations.enrichment_store import _write_enrichment_slot
from cytearc.datastore.datastore import DataStore
from cytearc.features.enrichment.chromvar import (
    background_peaks,
    build_chromvar_model,
    score_chromvar_block,
)
from cytearc.features.genomic.sequence import IndexedFasta, gc_content
from cytearc.storage import layout
from cytearc.storage.artifacts import artifact_path
from cytearc.storage.budget import resolve_budget
from cytearc.storage.schema import create_cell_data, create_zarr_count_assay
from cytearc.storage.stores import load_zarr
from cytearc.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts

N_CELLS = 80
N_PEAKS = 300
MOTIF_A = list(range(0, N_PEAKS, 15))


def _peaks() -> list[str]:
    return [f"chr1:{100 * index}-{100 * index + 60}" for index in range(N_PEAKS)]


def _counts() -> np.ndarray:
    rng = np.random.default_rng(4)
    rates = rng.gamma(2.0, 0.15, size=N_PEAKS)
    scale = np.ones((N_CELLS, N_PEAKS))
    # Half the cells open motif A's peaks three times more than expected.
    scale[: N_CELLS // 2, MOTIF_A] = 3.0
    counts = rng.poisson(rates * scale)
    counts[0] = 0
    counts[1] += 1
    return counts.astype(np.uint16)


def _network() -> pd.DataFrame:
    peaks = _peaks()
    rows = [("motifA", peaks[index]) for index in MOTIF_A]
    rows += [("motifB", peak) for peak in peaks[10:40]]
    rows += [("motifC", peak) for peak in peaks[100:300:4]]
    return pd.DataFrame(rows, columns=["source", "target"])


@pytest.fixture(scope="module")
def store_path(tmp_path_factory) -> tuple[str, str]:
    directory = tmp_path_factory.mktemp("chromvar")
    path = str(directory / "atac.zarr")
    root = load_zarr(zarr_loc=path, mode="w")
    cell_ids = np.asarray([f"cell{index}" for index in range(N_CELLS)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    array = create_zarr_count_assay(
        root,
        "ATAC",
        None,
        N_CELLS,
        feat_ids=np.asarray(_peaks()),
        feat_names=np.asarray([f"peak{index}" for index in range(N_PEAKS)]),
        dtype="uint16",
    )
    array[:] = _counts()
    finalize_test_counts(array)
    finalize_writer_counts_t(root, "ATAC", None)
    # An RNA assay of the same cells, which chromVAR refuses to score.
    genes = np.asarray(["gene0", "gene1"])
    rna = create_zarr_count_assay(
        root, "RNA", None, N_CELLS, feat_ids=genes, feat_names=genes, dtype="uint16"
    )
    rna[:] = 1
    finalize_test_counts(rna)
    finalize_writer_counts_t(root, "RNA", None)
    rng = np.random.default_rng(2)
    genome = "".join(rng.choice(list("ACGT"), size=100 * N_PEAKS + 200))
    fasta = directory / "genome.fa"
    fasta.write_text(
        ">chr1\n"
        + "\n".join(genome[i : i + 60] for i in range(0, len(genome), 60))
        + "\n"
    )
    store = DataStore(path, default_assay="ATAC", min_features_per_cell=-1, nthreads=2)
    store.cells.insert("everyone", np.ones(N_CELLS, dtype=bool), overwrite=True)
    return path, str(fasta)


def _open(path: str) -> DataStore:
    return DataStore(path, default_assay="ATAC", min_features_per_cell=-1, nthreads=2)


def test_chromvar_artifact_matches_the_domain_kernel(store_path):
    path, fasta = store_path
    store = _open(path)
    cells = store.snapshot_cell_selection("everyone")
    peaks = store.features.universe()

    ref = store.scores.chromvar(
        _network(),
        cells,
        features=peaks,
        genome=fasta,
        n_backgrounds=20,
        background_seed=3,
    )
    result = store.scores.load(ref)

    counts = _counts().astype(np.float64)
    fragments = counts.sum(axis=0)
    with IndexedFasta(fasta) as genome:
        gc = gc_content(genome, _peaks())
    background = background_peaks(fragments, gc, n_iterations=20, seed=3)
    network = _network()
    position = {peak: index for index, peak in enumerate(_peaks())}
    sources = sorted(network["source"].unique())
    annotation = csc_matrix(
        (
            np.ones(len(network)),
            (
                [position[peak] for peak in network["target"]],
                [sources.index(s) for s in network["source"]],
            ),
        ),
        shape=(N_PEAKS, len(sources)),
    )
    expected_dev, expected_z = score_chromvar_block(
        counts, build_chromvar_model(annotation, fragments, background)
    )

    assert result.method == "chromvar"
    assert result.source_names.tolist() == sources
    assert result.source_sizes.tolist() == [len(MOTIF_A), 30, 50]
    z = np.asarray(result.data.compute())
    deviations = np.asarray(result.deviations.compute())
    np.testing.assert_allclose(
        z, expected_z.astype(np.float32), rtol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        deviations, expected_dev.astype(np.float32), rtol=1e-6, equal_nan=True
    )
    # The cell without fragments has no values; motif A is open in half the cells.
    assert np.isnan(z[0]).all()
    assert np.nanmean(z[2 : N_CELLS // 2, 0]) > np.nanmean(z[N_CELLS // 2 :, 0]) + 1.0
    subset = store.scores.load(ref, sources=["motifC", "motifA"])
    np.testing.assert_array_equal(
        np.asarray(subset.deviations.compute()), deviations[:, [2, 0]]
    )
    frame = subset.to_pandas()
    assert frame.columns.tolist() == ["motifC", "motifA"]
    assert frame.index.name == "cell_index"
    np.testing.assert_array_equal(frame.index, np.arange(N_CELLS))
    np.testing.assert_array_equal(frame.to_numpy(), z[:, [2, 0]])
    # Deviations belong to chromVAR results and align with their scores.
    with pytest.raises(ValueError, match="only they, carry deviations"):
        dataclasses.replace(result, deviations=None)
    with pytest.raises(ValueError, match="must align with the scores"):
        dataclasses.replace(result, deviations=subset.deviations)


def test_chromvar_scores_a_subset_of_cells_and_peaks(store_path, monkeypatch):
    path, _fasta = store_path
    store = _open(path)
    cell_mask = np.arange(N_CELLS) % 4 != 3
    counts = _counts().astype(np.float64)[cell_mask]
    peak_mask = (np.arange(N_PEAKS) % 3 != 1) & (counts.sum(axis=0) > 0)
    store.cells.insert("most", cell_mask, overwrite=True)
    cells = store.snapshot_cell_selection("most")
    peaks = store.features.snapshot(mask=peak_mask)
    gc = np.linspace(0.3, 0.7, N_PEAKS)
    # Chunks of 25 rows of the three sources leave the 60 cells a partial
    # last band to write.
    monkeypatch.setattr(layout, "DEFAULT_TARGET_CHUNK_BYTES", 25 * 3 * 4)

    ref = store.scores.chromvar(
        _network(), cells, features=peaks, gc=gc, n_backgrounds=20, background_seed=5
    )
    result = store.scores.load(ref)
    assert store.zw[artifact_path(ref)]["deviations"].chunks == (25, 3)

    selected = counts[:, peak_mask]
    fragments = selected.sum(axis=0)
    background = background_peaks(fragments, gc[peak_mask], n_iterations=20, seed=5)
    position = {
        peak: index for index, peak in enumerate(np.asarray(_peaks())[peak_mask])
    }
    network = _network()
    network = network.loc[network["target"].isin(position)]
    sources = sorted(network["source"].unique())
    annotation = csc_matrix(
        (
            np.ones(len(network)),
            (
                [position[peak] for peak in network["target"]],
                [sources.index(source) for source in network["source"]],
            ),
        ),
        shape=(len(position), len(sources)),
    )
    expected_dev, expected_z = score_chromvar_block(
        selected, build_chromvar_model(annotation, fragments, background)
    )

    assert result.source_names.tolist() == sources
    z = np.asarray(result.data.compute())
    assert z.shape == (int(cell_mask.sum()), len(sources))
    np.testing.assert_allclose(
        z, expected_z.astype(np.float32), rtol=1e-6, equal_nan=True
    )
    np.testing.assert_allclose(
        np.asarray(result.deviations.compute()),
        expected_dev.astype(np.float32),
        rtol=1e-6,
        equal_nan=True,
    )


def test_chromvar_reuses_identical_requests_and_accepts_gc_values(store_path):
    path, fasta = store_path
    store = _open(path)
    cells = store.snapshot_cell_selection("everyone")
    peaks = store.features.universe()
    with IndexedFasta(fasta) as genome:
        gc = gc_content(genome, _peaks())

    first = store.scores.chromvar(
        _network(), cells, features=peaks, genome=fasta, n_backgrounds=20
    )
    from_values = store.scores.chromvar(
        _network(), cells, features=peaks, gc=gc, n_backgrounds=20
    )
    reseeded = store.scores.chromvar(
        _network(), cells, features=peaks, gc=gc, n_backgrounds=20, background_seed=9
    )

    assert from_values == first
    assert reseeded != first
    # A stored result is returned under any budget; only new work must fit.
    small = DataStore(
        path,
        default_assay="ATAC",
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="1K",
    )
    assert (
        small.scores.chromvar(
            _network(), cells, features=peaks, gc=gc, n_backgrounds=20
        )
        == first
    )
    with pytest.raises(MemoryError, match="background model"):
        small.scores.chromvar(
            _network(),
            cells,
            features=peaks,
            gc=gc,
            n_backgrounds=20,
            background_seed=1,
        )


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"features": "peaks"}, TypeError, "features must be an ArtifactRef"),
        ({"cell_selection": "everyone"}, TypeError, "cell_selection must be an"),
        ({"gc": None}, ValueError, "exactly one of gc or genome"),
        ({"gc": np.ones(3)}, ValueError, "one value per ATAC feature"),
        ({"gc": np.full(N_PEAKS, np.nan)}, ValueError, "have no GC content"),
        ({"n_backgrounds": 1}, ValueError, "n_backgrounds"),
        ({"kernel_width": 0.0}, ValueError, "kernel_width must be positive"),
        ({"from_assay": "RNA"}, TypeError, "only be run on an ATACassay"),
        ({"cells": np.zeros(N_CELLS, dtype=bool)}, ValueError, "no active cells"),
        ({"peaks": np.arange(N_PEAKS) == 0}, ValueError, "at least two selected"),
        # Cell 0 has no fragments in any peak.
        ({"cells": np.arange(N_CELLS) == 0}, ValueError, "have no fragments"),
    ],
)
def test_chromvar_validates_its_inputs(store_path, change, error, message):
    path, _fasta = store_path
    store = _open(path)
    change = dict(change)
    store.cells.insert(
        "chosen", change.pop("cells", np.ones(N_CELLS, dtype=bool)), overwrite=True
    )
    peaks = change.pop("peaks", None)
    arguments = {
        "cell_selection": store.snapshot_cell_selection("chosen"),
        "features": (
            store.features.universe()
            if peaks is None
            else store.features.snapshot(mask=peaks)
        ),
        "gc": np.linspace(0.3, 0.7, N_PEAKS),
        **change,
    }

    with pytest.raises(error, match=message):
        store.scores.chromvar(_network(), **arguments)


def _replace(slot: zarr.Group, name: str, data: np.ndarray) -> None:
    del slot[name]
    slot.create_array(name, data=data)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda slot: slot.attrs.update(gc_digest="0" * 64),
            "GC input does not match its metadata",
        ),
        (lambda slot: slot.attrs.update(bins=1), "invalid method metadata"),
        (lambda slot: slot.__delitem__("deviations"), "missing its deviations"),
        (
            lambda slot: _replace(
                slot, "deviations", np.zeros((N_CELLS, 1), dtype=np.float32)
            ),
            "misaligned deviations",
        ),
        (
            lambda slot: _replace(slot, "background_index", np.full((N_PEAKS, 20), -1)),
            "invalid background peaks",
        ),
    ],
    ids=["gc-digest", "bins", "no-deviations", "deviation-shape", "background-range"],
)
def test_chromvar_loader_rejects_results_edited_outside_cytearc(
    store_path, tmp_path, edit, message
):
    path, _fasta = store_path
    store = _open(path)
    ref = store.scores.chromvar(
        _network(),
        store.snapshot_cell_selection("everyone"),
        features=store.features.universe(),
        gc=np.linspace(0.3, 0.7, N_PEAKS),
        n_backgrounds=20,
    )
    copy = _open(str(shutil.copytree(path, tmp_path / "copy.zarr")))
    edit(copy.zw[artifact_path(ref)])

    with pytest.raises(ValueError, match=message):
        copy.scores.load(ref)


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (np.ones((3, 2)), "lacks its extra arrays"),
        ((np.ones((3, 2)), np.ones((2, 2))), "extra batch is misaligned"),
    ],
    ids=["scores-only", "short-deviations"],
)
def test_enrichment_writer_needs_aligned_deviations_in_each_batch(batch, message):
    slot = zarr.open_group(store=MemoryStore(), mode="w").create_group("slot")

    with pytest.raises(ValueError, match=message):
        _write_enrichment_slot(
            slot,
            attrs={"method": "chromvar"},
            score_batches=iter([batch]),
            n_cells=3,
            source_names=np.array(["A", "B"]),
            source_sizes=np.array([2, 2]),
            cell_index=np.arange(3),
            matched_feature_index=np.array([1, 3]),
            rank_feature_index=None,
            resources=resolve_budget(workers=1),
            io=None,
            allow_missing=True,
            extra_dense=("deviations",),
        )


def test_profiles_name_atac_columns_by_peak_id(store_path):
    path, _fasta = store_path
    store = _open(path)
    halves = (np.arange(N_CELLS) >= N_CELLS // 2).astype(np.int64)
    store.cells.insert("halves", halves, overwrite=True)

    profiles = store.regulons.profiles("halves")

    counts = _counts().astype(np.float64)
    sums = np.vstack([counts[halves == half].sum(axis=0) for half in (0, 1)])
    expected = np.log1p(sums / sums.sum(axis=1, keepdims=True) * 1e6)
    assert profiles.columns.tolist() == _peaks()
    np.testing.assert_allclose(profiles.to_numpy(), expected)
