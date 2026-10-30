"""Mixscape classes of the cells of a screen with knocked-out cells and escapers.

The module writes the screen and its perturbation signature once, as a
template. A test that writes copies the template, and the tests of refused
calls share one copy, which they check is left as it was.
"""

import math
import warnings
from collections.abc import Iterator
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu, norm

from cytearc.datastore.datastore import DataStore
from cytearc.features.markers.rank import tie_sum
from cytearc.perturbation import MixscapeResult
from cytearc.perturbation import mixscape as mixscape_module
from cytearc.perturbation.mixscape import (
    MIXSCAPE_FEW_CELLS,
    MIXSCAPE_FEW_DE_GENES,
    MIXSCAPE_FITTED,
    MIXSCAPE_NO_FIT,
    classify_target,
    constrained_mixture,
    group_totals,
    rank_sum_p_values,
    scale_like_seurat,
    sorted_log_values,
)
from cytearc.perturbation.signature import normalize_counts
from cytearc.storage.artifacts import artifact_group, inspect_artifact
from cytearc.utils.logging import logger
from tests.perturbation_helpers import (
    copy_store,
    open_store,
    use_python_kernels,
    write_screen,
    written,
)

N_A, N_B, N_C, N_CONTROL, N_GENES = 90, 60, 2, 150, 60
# Guide 0 targets GENE_A, guide 1 GENE_B, guide 2 GENE_C, and guide 3 is
# non-targeting. The last cell carries no guide.
TARGETS = ["GENE_A", "GENE_B", "GENE_C", "NT"]


def _screen() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1)
    carried = np.r_[
        np.zeros(N_A, int), np.ones(N_B, int), np.full(N_C, 2), np.full(N_CONTROL, 3)
    ]
    n_cells = len(carried) + 1
    knocked = np.zeros(n_cells, dtype=bool)
    knocked[:60] = True  # 60 of GENE_A's 90 cells respond and 30 escape
    base = rng.gamma(2.0, 3.0, size=N_GENES)
    rates = np.tile(base, (n_cells, 1)) * rng.lognormal(0, 0.2, size=(n_cells, 1))
    rates[np.ix_(knocked, np.arange(10))] *= 0.05  # GENE_A switches off genes 0 to 9
    rna = rng.poisson(rates)
    guides = np.zeros((n_cells, 4), dtype=np.uint16)
    guides[np.arange(len(carried)), carried] = 20
    coordinates = rng.normal(size=(n_cells, 10))
    runs = np.where(np.arange(n_cells) % 2 == 0, "run1", "run2")
    return carried, knocked, rna.astype(np.uint16), guides, coordinates, runs


CARRIED, KNOCKED, RNA, GUIDES, COORDINATES, RUNS = _screen()
N_CELLS = len(RNA)


def _signature(store: DataStore, cells=None):
    """A signature of genes 5 to 59 over every cell, from labels of ``cells``."""
    calls = store.perturbation.call_guides(
        cells or store.select_measured_cells("CRISPR"), method="fixed", min_umi=10
    )
    labels = store.perturbation.labels(calls, controls=["NT"])
    rna_cells = store.select_measured_cells("RNA")
    coordinates = store.reduction.import_coordinates(
        COORDINATES, cell_selection=rna_cells, name="X"
    )
    features = store.features.snapshot(feature_indexes=np.arange(5, N_GENES))
    return store.perturbation.signature(
        labels, coordinates, features=features, dims=10, n_neighbors=10
    )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Iterator[SimpleNamespace]:
    """The screen and its signature, which every test starts from."""
    path = tmp_path_factory.mktemp("mixscape") / "store.zarr"
    store = write_screen(
        path,
        RNA,
        GUIDES,
        targets=TARGETS,
        cell_columns={"run": np.asarray(RUNS, dtype=object)},
        nthreads=2,
    )
    screen = SimpleNamespace(path=path, signature=_signature(store))
    saved = written(store)
    yield screen
    assert written(open_store(path)) == saved


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    """A copy of the template, for a test that writes."""
    return copy_store(template.path, tmp_path, nthreads=2)


@pytest.fixture(scope="module")
def shared_store(template, tmp_path_factory) -> DataStore:
    """One writable copy of the template for the calls that must be refused."""
    folder = tmp_path_factory.mktemp("mixscape_refused")
    return copy_store(template.path, folder, nthreads=2)


def test_mixscape_separates_knocked_out_cells_from_escapers(store, template) -> None:
    signature = template.signature

    classes = store.perturbation.mixscape(signature)

    assert classes.kind == "mixscape_classes"
    assert classes.assay == "RNA"
    result = store.perturbation.load_mixscape(classes)
    assert isinstance(result, MixscapeResult)
    assert (result.perturbation_type, result.split_column) == ("KO", None)
    targets = result.targets.set_index("target")
    assert targets.index.tolist() == ["GENE_A", "GENE_B", "GENE_C"]
    assert targets["status"].tolist() == ["fitted", "few_de_genes", "few_de_genes"]
    assert targets["n_cells"].tolist() == [N_A, N_B, N_C]
    assert targets.loc["GENE_A", "n_ko"] == 60
    assert targets.loc["GENE_A", "converged"]
    cells = result.cells
    assert cells["global_class"].iloc[:N_A].eq("KO").to_numpy().tolist() == (
        KNOCKED[:N_A].tolist()
    )
    assert cells["mixscape_class"].iloc[:60].eq("GENE_A KO").all()
    assert cells["mixscape_class"].iloc[60:N_A].eq("GENE_A NP").all()
    assert cells["mixscape_class"].iloc[N_A : N_A + N_B].eq("GENE_B NP").all()
    controls = cells.iloc[N_A + N_B + N_C : -1]
    assert controls["mixscape_class"].eq("control").all()
    assert controls["posterior"].isna().all()
    assert cells["mixscape_class"].iloc[-1] == "Negative"
    assert (cells["posterior"].iloc[:60] > 0.5).all()
    assert (cells["posterior"].iloc[60:N_A] < 0.5).all()
    assert cells["posterior"].iloc[N_A:].isna().all()
    assert cells["score"].iloc[:N_A].notna().all()
    # The planted genes, 5 of them outside the signature's features, are DE.
    genes = result.de_genes.query("target == 'GENE_A'")["feature"].tolist()
    assert {f"RNA{index}" for index in range(10)} <= set(genes)
    assert targets.loc["GENE_A", "n_de_genes"] == len(genes)
    knocked_out = store.select_cells(classes, include=["GENE_A KO"])
    assert int(store.artifacts.load(knocked_out)["values"][:].sum()) == 60
    assert store.perturbation.mixscape(signature) == classes
    status = inspect_artifact(store.zw, classes)
    assert set(status.inputs) == {
        "signature",
        "dataset_fingerprint",
        "split_assignments_fingerprint",
        "cell_selection",
    }
    assert status.parameters["perturbation_type"] == "KO"


def test_mixscape_classifies_each_split_apart(store, template) -> None:
    classes = store.perturbation.mixscape(
        template.signature, split_column="run", perturbation_type="KD"
    )
    result = store.perturbation.load_mixscape(classes)
    assert result.split_column == "run"
    assert result.perturbation_type == "KD"
    assert result.cells["split"].tolist() == RUNS.tolist()
    targets = result.targets
    assert targets[["target", "split"]].values.tolist() == [
        ["GENE_A", "run1"],
        ["GENE_A", "run2"],
        ["GENE_B", "run1"],
        ["GENE_B", "run2"],
        ["GENE_C", "run1"],
        ["GENE_C", "run2"],
    ]
    assert targets["n_cells"].tolist() == [45, 45, 30, 30, 1, 1]
    called = result.cells["mixscape_class"].iloc[:N_A]
    assert called.eq("GENE_A KD").to_numpy().tolist() == KNOCKED[:N_A].tolist()


def test_cells_outside_the_labels_are_unlabelled(store) -> None:
    # Labels of every cell but the last ten controls.
    keep = np.ones(N_CELLS, dtype=bool)
    keep[-11:-1] = False
    store.cells.insert("labelled", keep)
    signature = _signature(store, store.snapshot_cell_selection("labelled"))
    result = store.perturbation.load_mixscape(store.perturbation.mixscape(signature))
    labels = result.cells["mixscape_class"]
    assert labels.iloc[-11:-1].eq("unlabelled").all()
    assert labels.iloc[N_A + N_B + N_C : -11].eq("control").all()


def test_mixscape_marks_targets_with_few_knocked_out_cells(store, template) -> None:
    result = store.perturbation.load_mixscape(
        store.perturbation.mixscape(template.signature, min_cells=61)
    )
    assert result.targets["status"].iloc[0] == "few_cells"
    assert result.cells["global_class"].iloc[:N_A].eq("NP").all()
    # No gene changes 100-fold in log2, so no gene is tested.
    untested = store.perturbation.load_mixscape(
        store.perturbation.mixscape(template.signature, logfc_threshold=100)
    )
    assert untested.targets["n_de_genes"].eq(0).all()
    assert untested.de_genes.empty


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"min_de_genes": 0}, ValueError, "min_de_genes must be at least 1"),
        ({"min_cells": 0}, ValueError, "min_cells must be at least 1"),
        ({"max_iterations": 0}, ValueError, "max_iterations must be at least 1"),
        ({"logfc_threshold": -1}, ValueError, "logfc_threshold must be at least 0"),
        ({"min_fraction": 1.5}, ValueError, "min_fraction must be from 0 to 1"),
        ({"p_value_cutoff": 0}, ValueError, "p_value_cutoff must be above 0"),
        ({"scale": "yes"}, TypeError, "scale must be a boolean"),
        ({"perturbation_type": 1}, TypeError, "perturbation_type must be a string"),
        ({"perturbation_type": "K O"}, ValueError, "must be one word"),
        ({"split_column": ""}, TypeError, "split_column must be a non-empty"),
        ({"split_column": "lane"}, ValueError, "'lane' not found in cell metadata"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
    ],
)
def test_mixscape_refuses_arguments_it_cannot_use(
    shared_store, template, arguments, error, message
) -> None:
    saved = written(shared_store)
    with pytest.raises(error, match=message):
        shared_store.perturbation.mixscape(template.signature, **arguments)
    assert written(shared_store) == saved


def test_mixscape_refuses_inputs_and_budgets_it_cannot_use(
    shared_store, template, monkeypatch
) -> None:
    store = shared_store
    signature = template.signature
    saved = written(store)
    with pytest.raises(TypeError, match="signature must be an ArtifactRef"):
        store.perturbation.mixscape("signature")
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.mixscape(store.select_measured_cells("RNA"))
    reader = open_store(template.path, zarr_mode="r")
    with pytest.raises(PermissionError, match="run_mixscape"):
        reader.perturbation.mixscape(signature)
    monkeypatch.setattr(store, "memoryBytes", 1_000)
    with pytest.raises(MemoryError, match="one feature over 302 cells"):
        store.perturbation.mixscape(signature)
    # One feature at a time fits the DE tests, but not GENE_A's classification.
    monkeypatch.setattr(store, "memoryBytes", 70_000)
    with pytest.raises(
        MemoryError, match="90 cells of GENE_A and of the 150 control cells in its 12"
    ):
        store.perturbation.mixscape(signature)
    assert written(store) == saved


def test_mixscape_classifies_in_batches_that_fit_the_budget(
    store, template, monkeypatch
) -> None:
    signature = template.signature
    whole = store.perturbation.load_mixscape(
        store.perturbation.mixscape(signature, split_column="run")
    )
    # The operation's accounting: the 302 cells of a group, with 10 neighbors
    # each, and the batch of GENE_A's 45 cells of one run with the 150 control
    # cells, in its 12 DE genes, and the classification of the run's cells.
    resident = (48 + 4 * 10) * N_CELLS + 8 * (10 + 2) * 302
    one_run = 4 * (45 + 150) * 12 + 16 * (45 + 150) + 24 * (45 + 75) * 12
    budget = store.memoryBytes
    monkeypatch.setattr(store, "memoryBytes", resident + one_run + 1_000)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        classes = store.perturbation.mixscape(
            signature, split_column="run", invalidate_cache=True
        )
    finally:
        logger.remove(sink)
    assert "Classifying 2 targets and splits in 2 batches" in messages
    monkeypatch.setattr(store, "memoryBytes", budget)
    batched = store.perturbation.load_mixscape(classes)
    pd.testing.assert_frame_equal(batched.cells, whole.cells)
    pd.testing.assert_frame_equal(batched.targets, whole.targets)


def test_classes_that_name_their_statuses_otherwise_are_not_reused_or_read(
    store, template
) -> None:
    signature = template.signature
    classes = store.perturbation.mixscape(signature)
    statuses = store.perturbation.load_mixscape(classes).targets["status"].tolist()
    # Classes whose codes name the statuses in another order, as a release
    # that renamed or reordered them would save.
    stored = artifact_group(store.zw, classes)
    stored.attrs["status_names"] = list(stored.attrs["status_names"])[::-1]

    with pytest.raises(ValueError, match="names its statuses .*; compute it again"):
        store.perturbation.load_mixscape(classes)
    again = store.perturbation.mixscape(signature)

    assert again != classes
    assert store.perturbation.load_mixscape(again).targets["status"].tolist() == (
        statuses
    )


def test_loading_mixscape_refuses_other_artifacts(shared_store, template) -> None:
    saved = written(shared_store)
    with pytest.raises(TypeError, match="classes must be an ArtifactRef"):
        shared_store.perturbation.load_mixscape("classes")
    with pytest.raises(ValueError, match="not 'perturbation_signature'"):
        shared_store.perturbation.load_mixscape(template.signature)
    assert written(shared_store) == saved


def test_scaling_follows_seurat() -> None:
    values = np.asarray([[1.0, 5.0, 2.0], [3.0, 5.0, 2.0], [8.0, 5.0, -400.0]])
    scaled = scale_like_seurat(values)
    expected_first = (values[:, 0] - values[:, 0].mean()) / values[:, 0].std(ddof=1)
    np.testing.assert_allclose(scaled[:, 0], expected_first)
    assert scaled[:, 1].tolist() == [0.0, 0.0, 0.0]
    # Seurat clips the negative of these signatures above 10.
    np.testing.assert_allclose(
        scaled[:, 2],
        np.maximum(
            (values[:, 2] - values[:, 2].mean()) / values[:, 2].std(ddof=1), -10
        ),
    )
    assert scale_like_seurat(values, scale_max=0.5).min() == -0.5
    assert scale_like_seurat(values[:1]).tolist() == [[0.0, 0.0, 0.0]]


def test_constrained_mixture_maximises_the_likelihood() -> None:
    rng = np.random.default_rng(2)
    values = np.r_[rng.normal(0.0, 1.0, 300), rng.normal(4.0, 0.7, 120)]
    fit = constrained_mixture(values, 0.0, 1.0, 3.0, 1.5)
    assert not fit.collapsed
    assert 0 < fit.steps < 5000

    def negative(parameters: np.ndarray) -> float:
        weight, mean, log_sd = parameters
        density = (1 - weight) * norm.pdf(values, 0.0, 1.0) + weight * norm.pdf(
            values, mean, np.exp(log_sd)
        )
        return -float(np.log(density).sum())

    best = minimize(
        negative,
        [0.5, 3.0, np.log(1.5)],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
    )
    assert fit.log_likelihood >= -best.fun - 1e-6
    np.testing.assert_allclose(
        [fit.weight, fit.mean, fit.sd],
        [best.x[0], best.x[1], np.exp(best.x[2])],
        rtol=1e-4,
    )
    assert constrained_mixture(values, 0.0, 1.0, 3.0, 0.0).collapsed
    one_step = constrained_mixture(values, 0.0, 1.0, 3.0, 1.5, max_steps=1)
    assert one_step.steps == 1
    # Values far from the fixed component pull the whole weight to the free
    # one, which ends the fit.
    assert constrained_mixture(
        np.full(5, 50.0) + np.arange(5), 0.0, 1.0, 52.0, 1.0
    ).collapsed


def _target(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knocked = np.r_[np.ones(40, bool), np.zeros(20, bool)]
    target = rng.normal(size=(60, 8))
    target[knocked, :4] -= 3.0
    control = rng.normal(size=(100, 8))
    return target, control, knocked


def test_classify_target_finds_the_knocked_out_cells() -> None:
    target, control, knocked = _target(np.random.default_rng(5))
    settings = {
        "n_de_genes": 8,
        "min_de_genes": 5,
        "min_cells": 5,
        "max_iterations": 10,
    }
    fit = classify_target(target, control, **settings)
    assert fit.status == MIXSCAPE_FITTED
    assert fit.ko.tolist() == knocked.tolist()
    assert fit.converged
    # Every array of a target's fit holds one value per cell of the target,
    # so the fits of many targets hold nothing of the control cells.
    assert {
        field.name: len(getattr(fit, field.name))
        for field in fields(fit)
        if isinstance(getattr(fit, field.name), np.ndarray)
    } == {"ko": 60, "posterior": 60, "score": 60}
    # The classes do not depend on the sign of the signatures.
    flipped = classify_target(-target, -control, **settings)
    assert flipped.ko.tolist() == fit.ko.tolist()
    unscaled = classify_target(target, control, scale=False, **settings)
    assert unscaled.ko.tolist() == knocked.tolist()
    stopped = classify_target(target, control, **{**settings, "max_iterations": 1})
    assert stopped.iterations == 1 and not stopped.converged


def test_classify_target_statuses_without_a_fit() -> None:
    target, control, _ = _target(np.random.default_rng(6))
    settings = {"min_de_genes": 3, "min_cells": 5, "max_iterations": 10}
    few_genes = classify_target(target, control, n_de_genes=2, **settings)
    assert few_genes.status == MIXSCAPE_FEW_DE_GENES
    assert not few_genes.ko.any() and np.isnan(few_genes.posterior).all()
    few_cells = classify_target(
        target, control, n_de_genes=8, **{**settings, "min_cells": 41}
    )
    assert few_cells.status == MIXSCAPE_FEW_CELLS
    # Target cells equal to the controls' mean give no direction.
    same = classify_target(
        np.zeros((5, 3)),
        np.r_[np.ones((5, 3)), -np.ones((5, 3))],
        n_de_genes=3,
        scale=False,
        **settings,
    )
    assert same.status == MIXSCAPE_NO_FIT
    # Identical target cells project to one point, which no component fits.
    collapsed = classify_target(
        np.ones((5, 3)),
        np.random.default_rng(7).normal(size=(10, 3)),
        n_de_genes=3,
        scale=False,
        **settings,
    )
    assert collapsed.status == MIXSCAPE_NO_FIT


def test_classify_target_posterior_does_not_overflow() -> None:
    # A tight knocked-out component puts the escaper, a cell like the
    # controls, about 38 of its standard deviations from its mean, so its log
    # likelihood ratio exceeds log(DBL_MAX), where exp overflowed.
    pattern = np.linspace(-1.0, 1.0, 30)
    target = np.r_[10.0 + 0.263 * pattern / pattern.std(), 0.0][:, None]
    control = np.r_[np.ones(100), -np.ones(100)][:, None]
    with warnings.catch_warnings(), np.errstate(over="raise"):
        warnings.simplefilter("error")
        fit = classify_target(
            target,
            control,
            n_de_genes=5,
            min_de_genes=5,
            min_cells=5,
            max_iterations=10,
            scale=False,
        )
    assert fit.status == MIXSCAPE_FITTED and fit.converged
    assert fit.ko.tolist() == [True] * 30 + [False]
    log_ratio = mixscape_module._log_normal(
        fit.score, fit.control_mean, fit.control_sd
    ) - mixscape_module._log_normal(fit.score, fit.perturbed_mean, fit.perturbed_sd)
    assert np.log(np.finfo(np.float64).max) < log_ratio[-1] < 745.0
    # The escaper's posterior is exp(-log_ratio), a subnormal number, not 0.
    assert math.isclose(fit.posterior[-1], math.exp(-log_ratio[-1]), rel_tol=1e-9)
    np.testing.assert_array_equal(
        fit.posterior[:-1], 1.0 / (1.0 + np.exp(log_ratio[:-1]))
    )


def test_knocked_out_posterior_keeps_every_finite_value() -> None:
    limit = np.log(np.finfo(np.float64).max)
    finite = np.r_[np.linspace(-800.0, 709.0, 100_001), limit, -np.inf, np.nan]
    beyond = np.r_[np.nextafter(limit, np.inf), 720.0, 800.0, np.inf]
    with warnings.catch_warnings(), np.errstate(over="raise"):
        warnings.simplefilter("error")
        posterior = mixscape_module._knocked_out_posterior(np.r_[finite, beyond])
    # Where exp(log_ratio) is finite, the posterior is 1 / (1 + exp(log_ratio))
    # to the last bit, and beyond it exp(-log_ratio), subnormal or 0.
    np.testing.assert_array_equal(
        posterior[: len(finite)], 1.0 / (1.0 + np.exp(finite))
    )
    np.testing.assert_allclose(
        posterior[len(finite) :],
        [math.exp(-value) for value in beyond],
        rtol=1e-9,
        atol=0,
    )


def _ranked_values(rng: np.random.Generator) -> np.ndarray:
    """Normalized values, 12 features x 70 cells, with many tied values."""
    counts = rng.poisson(rng.uniform(0.05, 3.0, size=(1, 12)), size=(70, 12))
    totals = rng.choice([500.0, 1_000.0, 1_500.0], size=70)
    values = normalize_counts(counts, totals, 10_000.0).T
    values[11] = 0.0  # A feature that no cell expresses.
    return values


@pytest.mark.parametrize("python_kernels", [False, True])
def test_rank_sum_p_values_equal_scipys(monkeypatch, python_kernels) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, mixscape_module)
    values = _ranked_values(np.random.default_rng(8))
    controls = np.arange(30, 70)
    features = np.arange(12)
    sorted_controls, ties = sorted_log_values(values, features, controls)
    tested = features[::-1][:10]
    for cells in (np.arange(12), np.arange(12, 30), np.r_[3, 9, 15]):
        p_values = rank_sum_p_values(
            values, tested, cells, tested, sorted_controls, ties
        )
        expected = mannwhitneyu(
            np.log1p(values[tested][:, cells]),
            np.log1p(values[tested][:, controls]),
            axis=1,
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        ).pvalue
        np.testing.assert_array_equal(p_values[1:], expected[1:])
        # For one value in every cell, scipy gives 1 or NaN depending on its
        # version. Here it is 1, which makes no gene DE either way.
        assert p_values[0] == 1.0


def test_sorted_log_values_hold_each_features_ties() -> None:
    values = _ranked_values(np.random.default_rng(9))
    cells = np.arange(5, 60, 2)
    ordered, ties = sorted_log_values(values, np.r_[4, 1], cells)
    np.testing.assert_array_equal(
        ordered, np.sort(np.log1p(values[[4, 1]][:, cells]), axis=1)
    )
    assert ties.tolist() == [tie_sum(row) for row in ordered]


@pytest.mark.parametrize("python_kernels", [False, True])
def test_group_totals_add_each_groups_values(monkeypatch, python_kernels) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, mixscape_module)
    values = _ranked_values(np.random.default_rng(10))
    groups = np.random.default_rng(11).integers(-1, 3, size=values.shape[1])
    sums, detected = group_totals(values, groups, 3)
    for group in range(3):
        np.testing.assert_allclose(sums[group], values[:, groups == group].sum(axis=1))
        np.testing.assert_array_equal(
            detected[group], (values[:, groups == group] > 0).sum(axis=1)
        )


def test_de_kernels_check_their_inputs() -> None:
    values = np.ones((3, 4))
    with pytest.raises(ValueError, match="one group or -1 per cell"):
        group_totals(values, np.r_[0, 1, 2, 3], 3)
    with pytest.raises(ValueError, match="with features and cells inside it"):
        sorted_log_values(values, np.r_[3], np.r_[0])
    ordered, ties = sorted_log_values(values, np.r_[0], np.r_[0, 1])
    with pytest.raises(ValueError, match="one tie term per slot"):
        rank_sum_p_values(values, np.r_[0], np.r_[2, 3], np.r_[1], ordered, ties)
