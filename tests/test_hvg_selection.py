import inspect
import re
import shutil

import numpy as np
import pytest

from cytearc.datastore.namespaces import FeaturesAccessor
from cytearc.assay import ATACassay, RNAassay
from cytearc.datastore.datastore import DataStore
from cytearc.features.variability import HVG_OPTION_NAMES, HvgOptions, hvg_options
from cytearc.storage.artifacts import ArtifactRef, artifact_path, inspect_artifact
from cytearc.storage.errors import ArtifactResolutionError
from cytearc.storage.selections import snapshot_run_metadata


@pytest.fixture(scope="module")
def hvg_store_template(datastore_zarr_root, tmp_path_factory):
    """A PBMC store with the default HVG selection and its feature summary."""
    location = tmp_path_factory.mktemp("hvg_selection") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    store = DataStore(str(location), default_assay="RNA")
    store.features.hvgs(store.snapshot_cell_selection(), show_plot=False)
    return location


@pytest.fixture
def hvg_store(hvg_store_template, tmp_path) -> DataStore:
    """A private copy of the template, so a test may edit its records."""
    shutil.copytree(hvg_store_template, tmp_path / "data.zarr")
    return DataStore(str(tmp_path / "data.zarr"), default_assay="RNA")


def test_hvg_public_contract_removed_assay_persistence_methods() -> None:
    signature = inspect.signature(FeaturesAccessor.hvgs)
    assert "hvg_key_name" not in signature.parameters
    assert "cell_key" not in signature.parameters
    assert "label" not in signature.parameters
    assert signature.return_annotation in {ArtifactRef, "ArtifactRef"}
    assert not hasattr(DataStore, "mark_hvgs")
    assert not hasattr(DataStore, "set_hvgs")
    assert not hasattr(RNAassay, "set_hvgs")
    assert not hasattr(RNAassay, "set_summary_stats")
    assert not hasattr(RNAassay, "set_feature_stats")
    assert not hasattr(ATACassay, "set_feature_stats")
    all_features = inspect.signature(FeaturesAccessor.universe)
    assert tuple(all_features.parameters) == ("self", "from_assay")
    assert all_features.return_annotation in {ArtifactRef, "ArtifactRef"}
    assert not hasattr(DataStore, "_ensure_all_features")


def test_hvg_regex_correction_recomputes_without_rewriting_saved_selection(
    hvg_store, monkeypatch
) -> None:
    import cytearc.datastore._operations.features as operations
    import cytearc.features.variability as variability

    store = hvg_store
    indices = np.flatnonzero(store.RNA.feats.fetch_all("nCells") > 2)[:2]
    names = np.array([f"GENE_{index}" for index in range(store.RNA.feats.N)])
    names[indices] = ["RPS3", "RPSX"]
    store.RNA.feats.insert("names", names, overwrite=True)
    cells = store.snapshot_cell_selection()
    options = dict(
        min_cells=0,
        max_cells=np.inf,
        top_n=store.RNA.feats.N,
        n_bins=20,
        blacklist=r"^RPS\d+$",
        keep_bounds=True,
        show_plot=False,
    )
    plan_selection = operations._feature_selection_plan

    def plan_without_blacklist_fingerprint(*args, **kwargs):
        kwargs["parameters"] = dict(kwargs["parameters"])
        kwargs["parameters"].pop("blacklist_fingerprint", None)
        return plan_selection(*args, **kwargs)

    def uppercase_matches(values, pattern):
        expression = re.compile(pattern.upper())
        return np.array(
            [expression.match(str(value).upper()) is not None for value in values]
        )

    with monkeypatch.context() as context:
        context.setattr(
            operations, "_feature_selection_plan", plan_without_blacklist_fingerprint
        )
        context.setattr(variability, "regex_match_mask", uppercase_matches)
        original = store.features.hvgs(cells, **options)
    old_group = store.artifacts.load(original)
    old_attributes = dict(old_group.attrs)
    old_values = np.asarray(old_group["values"][:])
    np.testing.assert_array_equal(old_values[indices], [True, False])

    corrected = store.features.hvgs(cells, **options)

    assert corrected != original
    np.testing.assert_array_equal(
        store.artifacts.load(corrected)["values"][:][indices], [False, True]
    )
    assert store.features.hvgs(cells, **options) == corrected
    # A blacklisted selection without its matched-feature fingerprint fails closed.
    with pytest.raises(ArtifactResolutionError, match="blacklist fingerprint"):
        store.features.resolve("RNA", original)
    assert dict(old_group.attrs) == old_attributes
    np.testing.assert_array_equal(old_group["values"][:], old_values)


def test_select_hvgs_returns_ref_without_creating_alias(
    hvg_store,
) -> None:
    store = hvg_store
    cell_selection = store.snapshot_cell_selection()
    columns_before = set(store.RNA.feats.columns)
    ref = store.features.hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        n_bins=20,
        lowess_frac=0.2,
        blacklist="",
        keep_bounds=True,
        show_plot=False,
        max_cells=np.inf,
        bin_strategy="adaptive",
    )

    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "feature_selection"
    assert store.features.resolve("RNA", ref) == ref
    assert set(store.RNA.feats.columns) == columns_before
    status = inspect_artifact(store.zw, ref)
    assert status.operation == "select_hvgs"
    assert set(status.inputs or {}) == {"feature_summary", "feature_snapshot"}
    assert status.parameters == {
        "min_cells": 0,
        "max_cells": {"special_float": "inf"},
        "top_n": 5,
        "min_var": {"special_float": "-inf"},
        "max_var": {"special_float": "inf"},
        "min_mean": {"special_float": "-inf"},
        "max_mean": {"special_float": "inf"},
        "n_bins": 20,
        "lowess_frac": 0.2,
        "blacklist": "",
        "keep_bounds": True,
        "bin_strategy": "adaptive",
        "variance_estimator": "regularized_local_quantile",
        "variance_quantile": 0.25,
    }
    group = store.artifacts.load(ref)
    values = np.asarray(group["values"][:])
    corrected = np.asarray(group["corrected_variance"][:])
    assert values.dtype == np.dtype(bool)
    assert values.shape == corrected.shape == (store.RNA.feats.N,)

    reused = store.features.hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        n_bins=20,
        lowess_frac=0.2,
        blacklist="",
        keep_bounds=True,
        show_plot=False,
        max_cells=np.inf,
        bin_strategy="adaptive",
    )
    assert reused == ref
    assert set(store.RNA.feats.columns) == columns_before


@pytest.mark.parametrize("bin_strategy", ["adaptive", "fixed"])
def test_select_hvgs_reuse_accounts_for_variance_estimator(
    hvg_store, bin_strategy
) -> None:
    store = hvg_store
    cell_selection = store.snapshot_cell_selection()
    options = {
        "min_cells": 0,
        "top_n": 5,
        "n_bins": 20,
        "lowess_frac": 0.2,
        "blacklist": "",
        "show_plot": False,
        "max_cells": np.inf,
        "bin_strategy": bin_strategy,
    }
    existing = store.features.hvgs(cell_selection, **options)
    group = store.zw[artifact_path(existing)]
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    if bin_strategy == "adaptive":
        parameters.pop("variance_estimator")
        parameters.pop("variance_quantile")
        group.attrs["provenance"] = {**provenance, "parameters": parameters}
    else:
        assert "variance_estimator" not in parameters
    stored_attributes = dict(group.attrs)
    stored_values = np.asarray(group["values"][:])
    stored_scores = np.asarray(group["corrected_variance"][:])

    selected = store.features.hvgs(cell_selection, **options)

    assert (selected != existing) == (bin_strategy == "adaptive")
    assert store.features.hvgs(cell_selection, **options) == selected
    if bin_strategy == "adaptive":
        with pytest.raises(ArtifactResolutionError, match="variance estimator"):
            store.features.resolve("RNA", existing)
    else:
        assert store.features.resolve("RNA", existing) == existing
    preserved = store.artifacts.load(existing)
    assert dict(preserved.attrs) == stored_attributes
    np.testing.assert_array_equal(preserved["values"][:], stored_values)
    np.testing.assert_array_equal(preserved["corrected_variance"][:], stored_scores)
    assert inspect_artifact(store.zw, selected).inputs == provenance["inputs"]


@pytest.mark.parametrize("stored_quantile", [None, 0.1])
def test_select_hvgs_recomputes_when_background_quantile_changes(
    hvg_store, stored_quantile
) -> None:
    store = hvg_store
    cells = store.snapshot_cell_selection()
    existing = store.features.hvgs(cells, show_plot=False)
    group = store.zw[artifact_path(existing)]
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    if stored_quantile is None:
        parameters.pop("variance_quantile")
    else:
        parameters["variance_quantile"] = stored_quantile
    group.attrs["provenance"] = {**provenance, "parameters": parameters}
    stored_attributes = dict(group.attrs)
    stored_scores = np.asarray(group["corrected_variance"][:])

    selected = store.features.hvgs(cells, show_plot=False)

    assert selected != existing
    assert store.features.hvgs(cells, show_plot=False) == selected
    if stored_quantile is None:
        with pytest.raises(ArtifactResolutionError, match="variance quantile"):
            store.features.resolve("RNA", existing)
    else:
        assert store.features.resolve("RNA", existing) == existing
    assert dict(group.attrs) == stored_attributes
    np.testing.assert_array_equal(group["corrected_variance"][:], stored_scores)
    assert inspect_artifact(store.zw, selected).parameters["variance_quantile"] == 0.25


@pytest.mark.parametrize("quantile", [True, 0, 1, -0.1, "0.25"])
def test_hvg_artifact_rejects_invalid_background_quantile(hvg_store, quantile) -> None:
    store = hvg_store
    ref = store.features.hvgs(store.snapshot_cell_selection(), show_plot=False)
    group = store.zw[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    parameters = {**provenance["parameters"], "variance_quantile": quantile}
    group.attrs["provenance"] = {**provenance, "parameters": parameters}

    with pytest.raises(ArtifactResolutionError, match="variance quantile"):
        store.features.resolve("RNA", ref)


def test_select_hvgs_default_calibrates_pbmc_malat1(hvg_store) -> None:
    store = hvg_store
    ref = store.features.hvgs(store.snapshot_cell_selection(), show_plot=False)
    names = np.asarray(store.RNA.feats.fetch_all("names"))
    index = int(np.flatnonzero(names == "MALAT1")[0])
    corrected = np.asarray(store.artifacts.load(ref)["corrected_variance"][:])

    assert 0.9 < corrected[index] < 1.1


@pytest.mark.parametrize(
    ("estimator", "bin_strategy"),
    [("unknown", "adaptive"), ("regularized_local_quantile", "fixed")],
)
def test_hvg_artifact_rejects_incompatible_variance_estimator(
    hvg_store, estimator, bin_strategy
) -> None:
    store = hvg_store
    ref = store.features.hvgs(
        store.snapshot_cell_selection(),
        min_cells=0,
        max_cells=np.inf,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
    )
    group = store.zw[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters.update(variance_estimator=estimator, bin_strategy=bin_strategy)
    group.attrs["provenance"] = {**provenance, "parameters": parameters}

    with pytest.raises(ArtifactResolutionError, match="variance estimator"):
        store.features.resolve("RNA", ref)


def test_select_hvgs_persists_effective_default_max_cells(
    hvg_store,
) -> None:
    store = hvg_store
    n_selected = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
    expected: int | float = n_selected - 20
    if expected <= 0:
        expected = np.inf
    cell_selection = store.snapshot_cell_selection()

    implicit = store.features.hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
    )
    explicit = store.features.hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
        max_cells=expected,
    )

    assert explicit == implicit
    assert inspect_artifact(store.zw, implicit).parameters["max_cells"] == expected


def test_select_hvgs_recomputes_when_feature_names_change(
    hvg_store,
) -> None:
    store = hvg_store
    names = np.asarray(
        [f"GENE_{index}" for index in range(store.RNA.feats.N)],
    )
    store.RNA.feats.insert("names", names, overwrite=True)
    cell_selection = store.snapshot_cell_selection()
    first = store.features.hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="^MT-",
        show_plot=False,
        max_cells=np.inf,
    )

    store.RNA.feats.insert(
        "names",
        np.asarray([f"MT-{name}" for name in names]),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="No features passed HVG candidate filters"):
        store.features.hvgs(
            cell_selection,
            from_assay="RNA",
            min_cells=0,
            top_n=5,
            n_bins=20,
            blacklist="^MT-",
            show_plot=False,
            max_cells=np.inf,
        )

    assert inspect_artifact(store.zw, first).complete


def test_select_hvgs_rejects_empty_result_without_metadata_mutation(
    hvg_store,
) -> None:
    store = hvg_store
    store.features.universe(from_assay="RNA")
    cell_selection = store.snapshot_cell_selection()
    before = set(store.artifacts.list(kind="feature_selection", from_assay="RNA"))
    columns_before = set(store.RNA.feats.columns)

    with pytest.raises(ValueError, match="HVG selection contains no features"):
        store.features.hvgs(
            cell_selection,
            from_assay="RNA",
            min_cells=0,
            max_cells=np.inf,
            top_n=5,
            min_var=np.inf,
            n_bins=20,
            blacklist="",
            show_plot=False,
        )

    after = set(store.artifacts.list(kind="feature_selection", from_assay="RNA"))
    assert after == before
    assert set(store.RNA.feats.columns) == columns_before


def _all_artifacts(store: DataStore) -> set[ArtifactRef]:
    return set(store.artifacts.list()) | set(store.artifacts.list(scope="datastore"))


# hvg_options checks every option; fit_lowess and the argument helpers test each rule.
@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"min_cells": 20.0}, TypeError, "min_cells must be an integer"),
        ({"min_cells": -1}, ValueError, "min_cells must be at least 0"),
        ({"top_n": 0}, ValueError, "top_n must be at least 1"),
        ({"n_bins": 0}, ValueError, "n_bins must be at least 1"),
        ({"lowess_frac": 1.5}, ValueError, "lowess_frac must be between 0 and 1"),
        ({"keep_bounds": 1}, TypeError, "keep_bounds must be a boolean"),
        ({"bin_strategy": "loess"}, ValueError, "bin_strategy must be either"),
    ],
    ids=[
        "min_cells",
        "negative_min_cells",
        "top_n",
        "n_bins",
        "lowess_frac",
        "keep_bounds",
        "bin_strategy",
    ],
)
def test_select_hvgs_checks_counts_and_trend_options_before_writing(
    hvg_store, arguments, error, message
) -> None:
    store = hvg_store
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    subset = active.copy()
    subset[np.flatnonzero(active)[::2]] = False
    store.cells.insert("hvg_subset", subset)
    # New feature names and a selection without a feature summary, whose
    # snapshot and summary a late check would write.
    names = np.asarray([f"GENE_{index}" for index in range(store.RNA.feats.N)])
    store.RNA.feats.insert("names", names, overwrite=True)
    cells = store.snapshot_cell_selection("hvg_subset")
    before = _all_artifacts(store)

    with pytest.raises(error, match=re.escape(message)):
        store.features.hvgs(cells, show_plot=False, **arguments)

    assert _all_artifacts(store) == before


def test_hvg_options_return_the_values_selections_record() -> None:
    defaults = {
        "min_cells": 20,
        "top_n": 1000,
        "n_bins": 200,
        "lowess_frac": 0.1,
        "keep_bounds": False,
        "bin_strategy": "adaptive",
    }
    assert tuple(defaults) == HVG_OPTION_NAMES
    assert hvg_options(**defaults) == HvgOptions(**defaults)
    checked = hvg_options(
        min_cells=np.int64(0),
        top_n=np.int32(5),
        n_bins=np.uint8(20),
        lowess_frac=1,
        keep_bounds=np.bool_(True),
        bin_strategy=np.str_("fixed"),
    )
    assert checked == (0, 5, 20, 1, True, "fixed")
    # lowess_frac keeps its type, so 1 and 1.0 keep their own identities.
    assert [type(value) for value in checked] == [int, int, int, int, bool, str]
    # Zero gives each trend its smallest window.
    assert hvg_options(**{**defaults, "lowess_frac": 0}).lowess_frac == 0


def test_select_hvgs_rejects_unknown_keywords_before_saving(
    hvg_store,
) -> None:
    store = hvg_store
    cell_selection = store.snapshot_cell_selection()
    before = set(store.artifacts.list(kind="feature_selection", from_assay="RNA"))
    summaries = set(store.artifacts.list(kind="feature_summary", from_assay="RNA"))

    # A misspelled selection option must not vanish into the plot options.
    with pytest.raises(TypeError, match="'top_N'"):
        store.features.hvgs(cell_selection, top_N=5, show_plot=False)
    with pytest.raises(TypeError, match="'show'"):
        store.features.hvgs(cell_selection, show=False, show_plot=True)

    assert set(store.artifacts.list(kind="feature_selection", from_assay="RNA")) == (
        before
    )
    assert (
        set(store.artifacts.list(kind="feature_summary", from_assay="RNA")) == summaries
    )

    ref = store.features.hvgs(
        cell_selection,
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        max_cells=np.inf,
        show_plot=False,
        label_size=9,
    )
    assert inspect_artifact(store.zw, ref).execution_options["plot_kwargs"] == {
        "label_size": 9
    }


def test_select_hvgs_rejects_non_rna_assay(hvg_store) -> None:
    cell_selection = hvg_store.snapshot_cell_selection()
    with pytest.raises(TypeError, match="RNAassay"):
        hvg_store.features.hvgs(
            cell_selection,
            from_assay="assay2",
            show_plot=False,
        )


def test_select_hvgs_read_only_guard_precedes_snapshot_planning(
    hvg_store,
) -> None:
    store = hvg_store
    cell_selection = store.snapshot_cell_selection()
    store.zarr_mode = "r"

    with pytest.raises(PermissionError, match=r"zarr_mode='r\+'"):
        store.features.hvgs(cell_selection, show_plot=False)


def test_select_hvgs_requires_a_feature_name_snapshot(hvg_store) -> None:
    store = hvg_store
    cell_selection = store.snapshot_cell_selection()
    ref = store.features.hvgs(
        cell_selection,
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
        max_cells=np.inf,
    )
    unrelated_snapshot = snapshot_run_metadata(
        store.zw,
        table_path="RNA/featureData",
        id_column="ids",
        columns=("I",),
        axis="feature",
        assay="RNA",
    )
    group = store.zw[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs["feature_snapshot"] = unrelated_snapshot.to_dict()
    group.attrs["provenance"] = {**provenance, "inputs": inputs}

    with pytest.raises(ArtifactResolutionError) as caught:
        store.features.resolve("RNA", ref)
    assert caught.value.code == "snapshot_contract_mismatch"
