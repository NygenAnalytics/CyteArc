import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import pytest

from cytearc.assay import normalization
from cytearc.datastore._operations import perturbation as perturbation_operations
from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import PerturbationProfiles
from cytearc.perturbation import profiles as profiles_module
from cytearc.perturbation.profiles import (
    CONTROL_SD_DDOF,
    control_moments,
    depth_target,
    profile_values,
    z_score_sums,
)
from cytearc.storage.artifacts import ArtifactRef, inspect_artifact
from tests.perturbation_helpers import (
    fixed_labels,
    record_cell_reads,
    use_python_kernels,
    write_screen,
)
from tests.storage_helpers import insert_nullable_cell_column

# Tests that check the warning about thin perturbations catch it themselves.
pytestmark = pytest.mark.filterwarnings("ignore:.*and get no profile")


def _cells(seed: int = 0) -> tuple[np.ndarray, ...]:
    """Counts, totals, control flags, samples and groups of 200 cells.

    No cell expresses feature 3, and cell 7, which carries a perturbation,
    has no counts at all.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.gamma(2.0, 2.0, size=70), size=(200, 70))
    counts[:, 3] = 0
    counts[7] = 0
    totals = counts.sum(axis=1).astype(np.float64) + 40
    totals[7] = 0
    control = np.arange(200) % 3 == 0
    samples = rng.integers(0, 3, 200)
    groups = np.where(control, -1, rng.integers(0, 4, 200))
    return counts.astype(np.uint16), totals, control, samples, groups


def _direct(
    counts: np.ndarray,
    totals: np.ndarray,
    control: np.ndarray,
    samples: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    columns: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """The scaling target, control moments and z-score sums, by NumPy.

    The recipe that reproduces Replogle et al.'s profiles: counts scaled to
    the median total of the control cells, and z-scores against each sample's
    control mean and standard deviation with the n - 1 divisor.
    """
    target = float(np.median(totals[control]))
    scaled = np.zeros(counts.shape)
    np.divide(counts, totals[:, None], out=scaled, where=totals[:, None] > 0)
    scaled *= target
    if columns is not None:
        scaled = scaled[:, columns]
    n_samples = int(samples.max()) + 1
    means = np.stack(
        [scaled[control & (samples == s)].mean(axis=0) for s in range(n_samples)]
    )
    sds = np.stack(
        [scaled[control & (samples == s)].std(axis=0, ddof=1) for s in range(n_samples)]
    )
    z = np.zeros(scaled.shape)
    for sample in range(n_samples):
        rows = np.flatnonzero(samples == sample)
        spread = np.flatnonzero(sds[sample] > 0)
        z[np.ix_(rows, spread)] = (
            scaled[np.ix_(rows, spread)] - means[sample, spread]
        ) / sds[sample, spread]
    sums = np.stack([z[groups == group].sum(axis=0) for group in range(n_groups)])
    return target, means, sds, sums


@pytest.mark.parametrize("python_kernels", [False, True])
def test_moments_and_sums_equal_numpy(monkeypatch, python_kernels) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, profiles_module)
    counts, totals, control, samples, groups = _cells()
    target, means, sds, sums = _direct(counts, totals, control, samples, groups, 4)
    assert depth_target(totals, control) == target
    controls = np.flatnonzero(control)
    seen, got_means, got_sds = control_moments(
        [counts[controls[:13]], counts[controls[13:]]],
        totals[controls],
        target,
        samples[controls],
        3,
        70,
    )
    assert seen.tolist() == np.bincount(samples[controls], minlength=3).tolist()
    np.testing.assert_allclose(got_means, means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_sds, sds, rtol=1e-10, atol=1e-12)
    # A feature that no control cell expresses has no spread in any sample.
    assert (got_sds[:, 3] == 0).all()
    perturbed = np.flatnonzero(~control)
    got = z_score_sums(
        [counts[perturbed[:50]], counts[perturbed[50:]]],
        totals[perturbed],
        target,
        samples[perturbed],
        groups[perturbed],
        got_means,
        got_sds,
        4,
    )
    np.testing.assert_allclose(got, sums, rtol=1e-10, atol=1e-10)
    assert (got[:, 3] == 0).all()


def test_moments_and_sums_depend_neither_on_blocks_nor_threads() -> None:
    counts, totals, control, samples, groups = _cells(1)
    target = depth_target(totals, control)
    controls, perturbed = np.flatnonzero(control), np.flatnonzero(~control)

    def run(rows: int) -> tuple[np.ndarray, ...]:
        _, means, sds = control_moments(
            [
                counts[controls[start : start + rows]]
                for start in range(0, len(controls), rows)
            ],
            totals[controls],
            target,
            samples[controls],
            3,
            70,
        )
        sums = z_score_sums(
            [
                counts[perturbed[start : start + rows]]
                for start in range(0, len(perturbed), rows)
            ],
            totals[perturbed],
            target,
            samples[perturbed],
            groups[perturbed],
            means,
            sds,
            4,
        )
        return means, sds, sums

    whole = run(len(counts))
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        single = run(1)
    finally:
        numba.set_num_threads(previous)
    for expected, got in zip(whole, single, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_a_sample_with_one_or_no_control_cell_has_no_spread() -> None:
    counts = np.arange(1, 13, dtype=np.uint16).reshape(4, 3)
    totals = np.full(4, 100.0)
    # Sample 0 holds three control cells, sample 1 one and sample 2 none.
    seen, means, sds = control_moments(
        [counts], totals, 100.0, np.array([0, 0, 0, 1]), 3, 3
    )
    # The standard deviation divides by n - 1, as Replogle et al.'s does.
    assert CONTROL_SD_DDOF == 1
    assert seen.tolist() == [3, 1, 0]
    np.testing.assert_allclose(sds[0], counts[:3].std(axis=0, ddof=1))
    assert (sds[1:] == 0).all() and (means[2] == 0).all()
    np.testing.assert_array_equal(means[1], counts[3])
    sums = z_score_sums(
        [counts[:2]],
        totals[:2],
        100.0,
        np.array([1, 2]),
        np.array([0, 0]),
        means,
        sds,
        1,
    )
    np.testing.assert_array_equal(sums, np.zeros((1, 3)))


def test_profile_values_average_and_mask() -> None:
    sums = np.arange(12, dtype=np.float64).reshape(3, 4)

    values = profile_values(sums, np.array([1, 2, 4]), np.array([2, -1, 0]))

    expected = sums / np.array([1, 2, 4])[:, None]
    expected[0, 2] = expected[2, 0] = 0
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, expected.astype(np.float32))


def test_profile_values_write_a_band_of_the_profiles() -> None:
    sums = np.random.default_rng(0).normal(scale=50.0, size=(3, 4))
    n_cells = np.array([1, 7, 30])
    profiles = np.full((3, 7), np.nan, dtype=np.float32)
    band = profiles[:, 2:6]

    written = profile_values(sums, n_cells, np.array([3, -1, 0]), out=band)

    assert written is band
    expected = (sums / n_cells[:, None]).astype(np.float32)
    expected[0, 3] = expected[2, 0] = 0
    # Each quotient rounds to float32 as dividing and then converting rounds it.
    np.testing.assert_array_equal(band.view(np.uint32), expected.view(np.uint32))
    assert np.isnan(profiles[:, [0, 1, 6]]).all()


def test_profile_values_make_no_copy_of_the_sums() -> None:
    # The sums of a band of 1,000 perturbations in 1,000 features, 8 MB.
    rng = np.random.default_rng(1)
    sums = rng.normal(size=(1000, 1000))
    n_cells = rng.integers(1, 500, 1000)
    masked = np.where(np.arange(1000) % 3 == 0, np.arange(1000), -1)
    profiles = np.empty((1000, 1200), dtype=np.float32)
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    start = tracemalloc.get_traced_memory()[0]
    try:
        profile_values(sums, n_cells, masked, out=profiles[:, 100:1100])
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        if not tracing:
            tracemalloc.stop()

    # NumPy rounds the quotients into the band through buffers of about
    # 130 kB, never a float64 or float32 copy of the sums.
    assert peak < sums.nbytes / 16


# The count blocks are uint16, as the stores hold, so the kernels compile for
# one count type.
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: depth_target(np.ones(3), np.zeros(3, bool)), "no control cell"),
        (
            lambda: control_moments(
                [np.ones((2, 3), np.uint16)], np.ones(2), 1.0, np.array([0, 2]), 2, 3
            ),
            "one sample from 0 to 1 per cell",
        ),
        (
            lambda: control_moments(
                [np.ones((2, 4), np.uint16)], np.ones(2), 1.0, np.zeros(2, int), 1, 3
            ),
            "cells x 3 features",
        ),
        (
            lambda: control_moments(
                [np.ones((2, 3), np.uint16)], np.ones(3), 1.0, np.zeros(3, int), 1, 3
            ),
            "hold 2 of the 3 cells",
        ),
        (
            lambda: control_moments(
                [np.ones((2, 3), np.uint16), np.ones((2, 3), np.uint16)],
                np.ones(3),
                1.0,
                np.zeros(3, int),
                1,
                3,
            ),
            "more than the 3 cells",
        ),
        (
            lambda: z_score_sums(
                [],
                np.ones(0),
                1.0,
                np.zeros(0),
                np.zeros(0),
                np.ones((1, 3)),
                np.ones((2, 3)),
                1,
            ),
            "both be samples x features",
        ),
        (
            lambda: z_score_sums(
                [],
                np.ones(2),
                1.0,
                np.zeros(1),
                np.zeros(2),
                np.ones((1, 3)),
                np.ones((1, 3)),
                1,
            ),
            "one entry per cell",
        ),
        (
            lambda: z_score_sums(
                [],
                np.ones(1),
                1.0,
                np.zeros(1),
                np.ones(1),
                np.ones((1, 3)),
                np.ones((1, 3)),
                1,
            ),
            "out of range",
        ),
        (
            lambda: profile_values(np.ones((2, 3)), np.ones(3), np.zeros(2)),
            "one cell count and one masked column",
        ),
        (
            lambda: profile_values(np.ones((2, 3)), np.array([1, 0]), np.zeros(2)),
            "at least one cell",
        ),
        (
            lambda: profile_values(np.ones((2, 3)), np.ones(2), np.array([3, -1])),
            "masked column is out of range",
        ),
        (
            lambda: profile_values(
                np.ones((2, 3)), np.ones(2), np.full(2, -1), out=np.empty((2, 3))
            ),
            "out must be a float32 array of the shape of sums",
        ),
        (
            lambda: profile_values(
                np.ones((2, 3)),
                np.ones(2),
                np.full(2, -1),
                out=np.empty((3, 2), dtype=np.float32),
            ),
            "out must be a float32 array of the shape of sums",
        ),
        (
            lambda: profile_values(
                np.ones((2, 3)), np.ones(2), np.full(2, -1), out=[[0.0] * 3] * 2
            ),
            "out must be a float32 array of the shape of sums",
        ),
    ],
)
def test_profile_kernels_refuse_inputs_they_cannot_use(call, message) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_the_profiles_type_is_exported() -> None:
    assert PerturbationProfiles is profiles_module.PerturbationProfiles


def test_left_out_perturbations_are_named_five_at_most() -> None:
    names = np.array([f"G{k}" for k in range(7)])

    message = perturbation_operations._left_out_warning(
        names, np.arange(7), 9, 25, "guides", "cells"
    )

    assert message is not None
    assert message.startswith("7 of 9 guides have fewer than 25 cells")
    assert "'G4' (4), and more." in message and "'G5'" not in message
    assert (
        perturbation_operations._left_out_warning(
            names[:0], np.arange(0), 9, 25, "guides", "cells"
        )
        is None
    )


# GENE_A has two guides of 30 cells, GENE_B one of 50 and GENE_C one of 4;
# no cell carries GENE_D. Two non-targeting guides cover 60 cells each, two
# cells carry no guide and one cell carries the guides of GENE_A and GENE_B.
GUIDE_TARGETS = ["GENE_A", "GENE_A", "GENE_B", "GENE_C", "GENE_D", "NT", "NT"]
GUIDE_GENE_IDS = ["RNA3", "RNA3", "RNA35", "", "RNA5", "", ""]
N_GENES = 40
# The profiles hold the first 30 genes, so GENE_A's own gene is masked and
# GENE_B's lies outside them.
SELECTED = np.arange(30)
# 40 of GENE_A's 60 cells respond; Mixscape calls them knocked out.
N_RESPONDING = 40


def _screen() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(5)
    guide_of = np.r_[
        np.repeat([0, 1], 30),
        np.full(50, 2),
        np.full(4, 3),
        np.repeat([5, 6], 60),
    ]
    n_cells = len(guide_of) + 3
    library = rng.lognormal(np.log(1500), 0.3, n_cells)
    base = rng.normal(-3.0, 0.5, N_GENES)
    mu = np.exp(base[None, :] + np.log(library)[:, None])
    mu[:N_RESPONDING, :10] *= 0.05
    mu[60:110, 10:15] *= 3.0
    rna = rng.poisson(mu)
    controls = np.flatnonzero(guide_of >= 5)
    # Gene 7 has no count in the control cells of run1, gene 8 in any cell.
    runs = np.where(np.arange(n_cells) % 2 == 0, "run1", "run2")
    rna[controls[runs[controls] == "run1"], 7] = 0
    rna[:, 8] = 0
    guides = np.zeros((n_cells, len(GUIDE_TARGETS)), dtype=np.uint16)
    guides[np.arange(len(guide_of)), guide_of] = 20
    guides[-1, [0, 2]] = 20
    coordinates = rng.normal(size=(n_cells, 10))
    return guide_of, rna.astype(np.uint16), guides, runs, coordinates


GUIDE_OF, RNA, GUIDES, RUNS, COORDINATES = _screen()
N_CELLS = len(RNA)
# Each cell's row of the expected profiles, GENE_A then GENE_B, or -1.
GROUPS = np.full(N_CELLS, -1)
GROUPS[:60] = 0
GROUPS[60:110] = 1
CONTROL = np.zeros(N_CELLS, dtype=bool)
CONTROL[114:234] = True


def _store(tmp_path: Path) -> DataStore:
    return write_screen(
        tmp_path / "store.zarr",
        RNA,
        GUIDES,
        targets=GUIDE_TARGETS,
        target_ids=GUIDE_GENE_IDS,
        cell_columns={"run": np.asarray(RUNS, dtype=object)},
        nthreads=2,
    )


def _selected(store: DataStore) -> ArtifactRef:
    return store.features.snapshot(feature_indexes=SELECTED)


def _warned_profiles(
    store: DataStore, *args, **kwargs
) -> tuple[ArtifactRef, list[str]]:
    """Compute profiles, and return them with the messages they warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        profiles = store.perturbation.profiles(*args, **kwargs)
    return profiles, [
        str(entry.message) for entry in caught if entry.category is UserWarning
    ]


def _expected(
    groups: np.ndarray, samples: np.ndarray, masked: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The expected profiles, control means and standard deviations."""
    # The store's totals are the row sums of its counts.
    totals = RNA.sum(axis=1).astype(np.float64)
    _, means, sds, sums = _direct(
        RNA, totals, CONTROL, samples, groups, groups.max() + 1, SELECTED
    )
    values = sums / np.bincount(groups[groups >= 0])[:, None]
    for row, column in masked.items():
        values[row, column] = 0
    return values, means, sds


@dataclass(frozen=True)
class _Shared:
    path: Path
    store: DataStore
    labels: ArtifactRef
    features: ArtifactRef


@pytest.fixture(scope="module")
def shared(tmp_path_factory) -> _Shared:
    """One store of the screen for the tests that keep its data."""
    folder = tmp_path_factory.mktemp("profiles")
    store = _store(folder)
    return _Shared(
        folder / "store.zarr", store, fixed_labels(store, ["NT"]), _selected(store)
    )


def test_profiles_equal_a_direct_computation(shared) -> None:
    store, labels, features = shared.store, shared.labels, shared.features

    profiles, messages = _warned_profiles(store, labels, features)

    assert (profiles.kind, profiles.assay) == ("perturbation_profiles", "RNA")
    assert messages == [
        "2 of 4 targets have fewer than 25 cells and get no profile: "
        "'GENE_C' (4), 'GENE_D' (0). perturbation.load_profiles lists them all "
        "in left_out."
    ]
    result = store.perturbation.load_profiles(profiles)
    assert isinstance(result, PerturbationProfiles)
    assert (result.by, result.from_assay, result.sample_column) == (
        "target",
        "RNA",
        None,
    )
    assert result.min_cells == 25 and result.mask_targets
    assert result.labels_kind == "perturbation_labels"
    values, means, sds = _expected(GROUPS, np.zeros(N_CELLS, int), {0: 3})
    assert result.values.index.tolist() == ["GENE_A", "GENE_B"]
    assert result.values.columns.tolist() == [f"RNA{k}" for k in SELECTED]
    assert result.values.dtypes.eq(np.float32).all()
    np.testing.assert_allclose(result.values.to_numpy(), values, rtol=2e-6, atol=2e-6)
    # GENE_A lowers its first ten genes and GENE_B raises genes 10 to 14.
    assert (result.values.loc["GENE_A", ["RNA0", "RNA1", "RNA9"]] < -0.5).all()
    assert (result.values.loc["GENE_B", "RNA10":"RNA14"] > 0.5).all()
    assert result.values.loc["GENE_A", "RNA3"] == 0
    assert result.perturbations.to_dict("list") == {
        "target": ["GENE_A", "GENE_B"],
        "is_control": [False, False],
        "n_cells": [60, 50],
        "masked_feature": ["RNA3", None],
    }
    assert result.left_out.to_dict("list") == {
        "perturbation": ["GENE_C", "GENE_D"],
        "target": ["GENE_C", "GENE_D"],
        "n_cells": [4, 0],
    }
    assert result.samples.index.tolist() == [None]
    assert result.samples["n_control_cells"].tolist() == [120]
    np.testing.assert_allclose(result.control_mean.to_numpy(), means, rtol=1e-12)
    np.testing.assert_allclose(result.control_sd.to_numpy(), sds, rtol=1e-10)
    zero_sd = result.features["n_zero_sd"]
    assert zero_sd["RNA8"] == 1 and zero_sd.drop("RNA8").eq(0).all()
    assert result.features["feature_name"].tolist() == [f"RNA{k}" for k in SELECTED]
    status = inspect_artifact(store.zw, profiles)
    assert set(status.inputs) == {
        "labels",
        "feature_selection",
        "dataset_fingerprint",
        "sample_assignments_fingerprint",
        "target_features_fingerprint",
    }
    assert status.inputs["sample_assignments_fingerprint"] is None
    assert status.parameters == {
        "by": "target",
        "min_cells": 25,
        "mask_targets": True,
        "sample_columns": [],
    }


def test_profiles_of_each_guide_include_the_control_guides(shared) -> None:
    store, labels, features = shared.store, shared.labels, shared.features

    profiles, messages = _warned_profiles(store, labels, features, by="guide")

    result = store.perturbation.load_profiles(profiles)
    assert result.by == "guide"
    # Guides 0 and 1 target GENE_A, guide 2 GENE_B, and 5 and 6 are controls.
    groups = np.full(N_CELLS, -1)
    for row, (start, stop) in enumerate(
        [(0, 30), (30, 60), (60, 110), (114, 174), (174, 234)]
    ):
        groups[start:stop] = row
    values, _, _ = _expected(groups, np.zeros(N_CELLS, int), {0: 3, 1: 3})
    names = ["CRISPR0", "CRISPR1", "CRISPR2", "CRISPR5", "CRISPR6"]
    assert result.values.index.tolist() == names
    np.testing.assert_allclose(result.values.to_numpy(), values, rtol=2e-6, atol=2e-6)
    assert result.perturbations.to_dict("list") == {
        "target": ["GENE_A", "GENE_A", "GENE_B", "NT", "NT"],
        "is_control": [False, False, False, True, True],
        "n_cells": [30, 30, 50, 60, 60],
        "masked_feature": ["RNA3", "RNA3", None, None, None],
    }
    # The control cells' z-scores sum to 0, so the two control guides of
    # equal size have opposite profiles.
    np.testing.assert_allclose(
        result.values.loc["CRISPR5"], -result.values.loc["CRISPR6"], atol=1e-6
    )
    assert result.left_out.to_dict("list") == {
        "perturbation": ["CRISPR3", "CRISPR4"],
        "target": ["GENE_C", "GENE_D"],
        "n_cells": [4, 0],
    }
    assert messages == [
        "2 of 7 guides have fewer than 25 cells and get no profile: "
        "'CRISPR3' (4), 'CRISPR4' (0). perturbation.load_profiles lists them all "
        "in left_out."
    ]


def test_profiles_z_normalize_each_sample_against_its_own_controls(shared) -> None:
    store, labels, features = shared.store, shared.labels, shared.features

    profiles = store.perturbation.profiles(
        labels, features, sample_column="run", min_cells=4
    )

    result = store.perturbation.load_profiles(profiles)
    samples = (RUNS == "run2").astype(int)
    groups = GROUPS.copy()
    groups[110:114] = 2
    values, means, sds = _expected(groups, samples, {0: 3})
    assert result.sample_column == "run"
    assert result.values.index.tolist() == ["GENE_A", "GENE_B", "GENE_C"]
    np.testing.assert_allclose(result.values.to_numpy(), values, rtol=2e-6, atol=2e-6)
    assert result.samples.index.tolist() == ["run1", "run2"]
    assert result.samples["n_control_cells"].tolist() == [60, 60]
    np.testing.assert_allclose(result.control_mean.to_numpy(), means, rtol=1e-12)
    np.testing.assert_allclose(result.control_sd.to_numpy(), sds, rtol=1e-10)
    # Gene 7 has no spread over the control cells of run1, gene 8 over any.
    zero_sd = result.features["n_zero_sd"]
    assert zero_sd[["RNA7", "RNA8"]].tolist() == [1, 2]
    assert result.left_out["perturbation"].tolist() == ["GENE_D"]
    status = inspect_artifact(store.zw, profiles)
    assert status.inputs["sample_assignments_fingerprint"] is not None


def test_profiles_keep_each_target_without_masking(shared) -> None:
    store, labels, features = shared.store, shared.labels, shared.features
    masked = store.perturbation.load_profiles(
        store.perturbation.profiles(labels, features)
    )

    plain = store.perturbation.profiles(labels, features, mask_targets=False)
    # By name alone, no feature is called GENE_A, so nothing is masked.
    by_name = store.perturbation.profiles(labels, features, id_column=None)

    assert plain != by_name
    values, _, _ = _expected(GROUPS, np.zeros(N_CELLS, int), {})
    for profiles in (plain, by_name):
        result = store.perturbation.load_profiles(profiles)
        assert result.perturbations["masked_feature"].isna().all()
        np.testing.assert_allclose(
            result.values.to_numpy(), values, rtol=2e-6, atol=2e-6
        )
    unmasked = store.perturbation.load_profiles(plain)
    assert not unmasked.mask_targets
    assert unmasked.values.loc["GENE_A", "RNA3"] < -0.5
    pd.testing.assert_frame_equal(
        unmasked.values.drop(columns="RNA3"), masked.values.drop(columns="RNA3")
    )
    status = inspect_artifact(store.zw, plain)
    assert status.inputs["target_features_fingerprint"] is None


def test_profiles_of_mixscape_classes_average_knocked_out_cells(
    tmp_path, monkeypatch
) -> None:
    store = _store(tmp_path)
    labels = fixed_labels(store, ["NT"])
    coordinates = store.reduction.import_coordinates(
        COORDINATES, cell_selection=store.select_measured_cells("RNA"), name="X"
    )
    signature = store.perturbation.signature(
        labels, coordinates, features=_selected(store), dims=10, n_neighbors=10
    )
    classes = store.perturbation.mixscape(signature, perturbation_type="KD")
    reads = record_cell_reads(monkeypatch, store, {"ids"})
    called = store.perturbation.load_mixscape(classes).cells["mixscape_class"]
    knocked = called.str.endswith(" KD").to_numpy()
    # Loading the classes reads their cells' IDs once.
    assert [rows.tolist() for rows in reads["ids"]] == [list(range(N_CELLS))]
    reads["ids"].clear()

    profiles, messages = _warned_profiles(store, classes, _selected(store), min_cells=4)
    guides, guide_messages = _warned_profiles(
        store, classes, _selected(store), by="guide", min_cells=4
    )
    # The profiles read the classes of their cells, not the cells' IDs.
    assert reads["ids"] == []

    result = store.perturbation.load_profiles(profiles)
    assert result.labels_kind == "mixscape_classes"
    # The responding cells of GENE_A and every cell of GENE_B are knocked down;
    # the escapers of GENE_A do not count.
    groups = np.full(N_CELLS, -1)
    groups[called.eq("GENE_A KD").to_numpy()] = 0
    groups[called.eq("GENE_B KD").to_numpy()] = 1
    np.testing.assert_array_equal(np.flatnonzero(groups == 0), np.arange(N_RESPONDING))
    values, _, _ = _expected(groups, np.zeros(N_CELLS, int), {0: 3})
    assert result.values.index.tolist() == ["GENE_A", "GENE_B"]
    assert result.perturbations["n_cells"].tolist() == [N_RESPONDING, 50]
    np.testing.assert_allclose(result.values.to_numpy(), values, rtol=2e-6, atol=2e-6)
    # Mixscape calls none of GENE_C's 4 cells knocked down.
    assert messages == [
        "2 of 4 targets have fewer than 4 knocked-out cells and get no profile: "
        "'GENE_C' (0), 'GENE_D' (0). perturbation.load_profiles lists them all "
        "in left_out."
    ]
    assert inspect_artifact(store.zw, profiles).inputs["labels"] == classes.to_dict()
    # By guide, the knocked-out cells of each targeting guide count, and each
    # control guide keeps its cells.
    by_guide = store.perturbation.load_profiles(guides)
    groups = np.full(N_CELLS, -1)
    for row, (start, stop) in enumerate(
        [(0, 30), (30, 60), (60, 110), (114, 174), (174, 234)]
    ):
        groups[start:stop] = row
    groups[(groups < 3) & ~knocked] = -1
    values, _, _ = _expected(groups, np.zeros(N_CELLS, int), {0: 3, 1: 3})
    assert by_guide.perturbations["n_cells"].tolist() == [30, 10, 50, 60, 60]
    np.testing.assert_allclose(by_guide.values.to_numpy(), values, rtol=2e-6, atol=2e-6)
    assert guide_messages == [
        "2 of 7 guides have fewer than 4 knocked-out or control cells and get no "
        "profile: 'CRISPR3' (0), 'CRISPR4' (0). perturbation.load_profiles lists "
        "them all in left_out."
    ]


def test_profiles_are_reused_read_only_and_refused_when_new(shared) -> None:
    store, labels, features = shared.store, shared.labels, shared.features
    profiles = store.perturbation.profiles(labels, features)
    reader = DataStore(
        str(shared.path),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        zarr_mode="r",
    )

    reused, messages = _warned_profiles(reader, labels, features)

    assert reused == profiles
    # A reused result warns as the computed one did.
    assert len(messages) == 1 and messages[0].startswith("2 of 4 targets")
    with pytest.raises(PermissionError, match="compute_perturbation_profiles"):
        reader.perturbation.profiles(labels, features, min_cells=30)
    again = store.perturbation.profiles(labels, features, invalidate_cache=True)
    assert again != profiles
    pd.testing.assert_frame_equal(
        store.perturbation.load_profiles(again).values,
        store.perturbation.load_profiles(profiles).values,
    )


def test_profiles_read_bands_of_features_that_fit_the_budget(
    shared, monkeypatch
) -> None:
    store, labels, features = shared.store, shared.labels, shared.features
    whole = store.perturbation.load_profiles(
        store.perturbation.profiles(labels, features, sample_column="run")
    )
    bands: list[int] = []
    read = normalization.feature_value_blocks

    def counting(assay, physical_indices, *args, **kwargs):
        bands.append(len(physical_indices))
        return read(assay, physical_indices, *args, **kwargs)

    monkeypatch.setattr(normalization, "feature_value_blocks", counting)
    # The operation's accounting: 237 labelled cells, the 230 control and
    # profiled cells, the profiles of 2 perturbations and the moments of 2
    # samples in 30 features, and 7 features of a band.
    held = 24 * N_CELLS + 32 * 230 + 4 * 2 * 30 + 16 * 2 * 30 + 16 * 30
    per_feature = 8 * 2 + 24 * 2 + 16
    monkeypatch.setattr(store, "memoryBytes", held + 7 * per_feature + 10)

    banded = store.perturbation.profiles(
        labels, features, sample_column="run", invalidate_cache=True
    )

    # Each band reads the control cells, then the profiled cells.
    assert bands == [7, 7, 7, 7, 7, 7, 7, 7, 2, 2]
    monkeypatch.setattr(store, "memoryBytes", 1 << 30)
    result = store.perturbation.load_profiles(banded)
    pd.testing.assert_frame_equal(result.values, whole.values)
    pd.testing.assert_frame_equal(result.control_sd, whole.control_sd)
    monkeypatch.setattr(store, "memoryBytes", held + per_feature - 1)
    with pytest.raises(MemoryError, match="2 targets in 30 features"):
        store.perturbation.profiles(
            labels, features, sample_column="run", invalidate_cache=True
        )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"by": "gene"}, ValueError, "by must be 'target' or 'guide'"),
        ({"min_cells": 0}, ValueError, "min_cells must be at least 1"),
        ({"min_cells": 61}, ValueError, "No target has 61 cells or more"),
        ({"mask_targets": "yes"}, TypeError, "mask_targets must be a boolean"),
        ({"id_column": ""}, TypeError, "id_column must be a feature column"),
        ({"id_column": "gene"}, ValueError, "'gene' is not a feature column"),
        ({"sample_column": ""}, TypeError, "sample_column must be a non-empty"),
        ({"sample_column": "lane"}, ValueError, "'lane' not found in cell metadata"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
    ],
)
def test_profiles_refuse_arguments_they_cannot_use(
    shared, arguments, error, message
) -> None:
    with pytest.raises(error, match=message):
        shared.store.perturbation.profiles(shared.labels, shared.features, **arguments)


def test_profiles_refuse_inputs_they_cannot_use(tmp_path) -> None:
    store = _store(tmp_path)
    labels = fixed_labels(store, ["NT"])
    features = _selected(store)
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        store.perturbation.profiles("labels", features)
    with pytest.raises(ValueError, match="perturbation_labels or mixscape_classes"):
        store.perturbation.profiles(features, features)
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        store.perturbation.profiles(labels, "features")
    with pytest.raises(ValueError, match="not 'perturbation_labels'"):
        store.perturbation.profiles(labels, labels)
    # Batch b2 holds one control cell, too few for a standard deviation with
    # the n - 1 divisor, and batch b3 none.
    for column, last in (("batch", 1), ("plate", 0)):
        batch = np.where(CONTROL, "b1", "b2").astype(object)
        batch[np.flatnonzero(CONTROL)[:last]] = "b2"
        store.cells.insert(column, batch)
        with pytest.raises(
            ValueError,
            match=f"need 2 control cells or more, but sample 'b2' has {last};",
        ):
            store.perturbation.profiles(labels, features, sample_column=column)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[200] = True
    insert_nullable_cell_column(
        store, "lane", np.asarray(RUNS, dtype=object).astype(str), missing
    )
    with pytest.raises(ValueError, match="no label for 1 selected cells"):
        store.perturbation.profiles(labels, features, sample_column="lane")
    controls = store.perturbation.labels(
        store.perturbation.call_guides(
            store.select_measured_cells("CRISPR"), method="fixed", min_umi=10
        ),
        controls=["GENE_D"],
    )
    with pytest.raises(
        ValueError,
        match="No labelled cell carries control guides only, and each perturbation "
        "is compared with such cells; check the controls passed to "
        "perturbation.labels, or in a screen whose cells carry several guides, test "
        "associations with moi='high'",
    ):
        store.perturbation.profiles(controls, features)
    assert not store.artifacts.list(kind="perturbation_profiles")


def test_profiles_refuse_labels_with_one_control_cell(tmp_path) -> None:
    store = _store(tmp_path)
    # Of GENE_C's four cells, only cell 111 is selected, and the labels call
    # GENE_C the control, so they hold one control cell.
    store.cells.insert("cells", ~np.isin(np.arange(N_CELLS), [110, 112, 113]))
    calls = store.perturbation.call_guides(
        store.snapshot_cell_selection("cells"), method="fixed", min_umi=10
    )
    one = store.perturbation.labels(calls, controls=["GENE_C"])

    with pytest.raises(
        ValueError, match="need 2 control cells or more, but the labels have 1;"
    ):
        store.perturbation.profiles(one, _selected(store), min_cells=1)


def test_loading_profiles_refuses_other_artifacts_and_budgets(
    shared, monkeypatch
) -> None:
    store = shared.store
    profiles = store.perturbation.profiles(shared.labels, shared.features)
    with pytest.raises(TypeError, match="profiles must be an ArtifactRef"):
        store.perturbation.load_profiles("profiles")
    with pytest.raises(ValueError, match="not 'perturbation_labels'"):
        store.perturbation.load_profiles(shared.labels)
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="2 perturbations in 30 features hold 240"):
        store.perturbation.load_profiles(profiles)


def test_loading_profiles_holds_their_values_once(shared, monkeypatch) -> None:
    store = shared.store
    profiles = store.perturbation.profiles(shared.labels, shared.features)
    read: list[np.ndarray] = []
    as_zarr_array = perturbation_operations.as_zarr_array

    class Recorded:
        """The stored values, keeping each array that a read returns."""

        def __init__(self, stored) -> None:
            self.stored = stored
            self.shape = stored.shape

        def __getitem__(self, selection) -> np.ndarray:
            read.append(np.asarray(self.stored[selection]))
            return read[-1]

    def recording(node, *, name=""):
        stored = as_zarr_array(node, name=name)
        return Recorded(stored) if name == "values" else stored

    monkeypatch.setattr(perturbation_operations, "as_zarr_array", recording)

    result = store.perturbation.load_profiles(profiles)

    # The frame wraps the float32 values as read, the 4 bytes per value that
    # the memory check counts, rather than a copy of them.
    assert len(read) == 1
    assert np.shares_memory(result.values.to_numpy(), read[0])
