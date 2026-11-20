"""The normalizer of a recipe run, which the run records and applies to every stage.

A module fixture commits, in one session, an RNA run of Pearson residuals,
another whose PCA scales them, an ADT run of the first one's cells, a WNN run
of the two with markers, and an RNA run of a custom normalizer with an ADT run
of its cells, on a private copy of the PBMC store. Another DataStore of the same store stands in for a new session, whose
RNA assay normalizes by library size, as every new session's does. Tests that
write copy the store first.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from cytearc import (
    CancellationToken,
    DataStore,
    PipelineCancelled,
    PipelineExecutionError,
    PipelineRun,
)
from cytearc.assay import (
    PearsonResidualNormalizer,
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_lib_size_log,
    norm_pearson_residuals,
    norm_tf_idf,
)
from cytearc.datastore._pipeline.normalizers import (
    BUILTIN_NORMALIZERS,
    add_applied_feat_scaling,
    normalizer_setting,
)
from cytearc.datastore.pipeline_accessor import PipelineEvent
from cytearc.storage.artifacts import ArtifactRef, artifact_group, callable_identity
from cytearc.storage.pipeline_runs import pipeline_run_path, pipeline_stage_path
from tests.perturbation_helpers import write_screen
from tests.storage_helpers import open_store, write_count_store, written
from tests.test_pipeline_plan import (
    _digest,
    _dispositions,
    assert_plan_reuses_run,
    read_only_plan,
)

# The settings of the run in the issue that this module reproduces, #397: 200
# genes, 5 PCs without feature scaling, and two Leiden candidates.
SETTINGS: dict[str, Any] = {
    "cell_cycle": False,
    "hvg": {"top_n": 200},
    "pca": {"dims": 5, "feat_scaling": False},
    "neighbors": {"k": 5},
    "umap": False,
    "leiden": {"partitions": [0.5, 1.0]},
    "doublets": False,
    "markers": False,
}
ADT_SETTINGS: dict[str, Any] = {
    "neighbors": {"k": 5},
    "leiden": {"partitions": [1.0]},
    "umap": False,
    "markers": False,
}
# What provenance records for the built-in normalizers that runs here apply:
# normalized values record the normalizer as a hook with its identity, and
# feature summaries and marker tables its identity alone.
PEARSON_IDENTITY = {
    "identity": "cytearc.assay.pearson_residuals:theta=100.0:clip=sqrt-n"
}
PEARSON = {"external_hook": True, **PEARSON_IDENTITY}
CLR_IDENTITY = {"identity": "cytearc.assay.norm_clr:feature-axis"}
# The names of the built-in normalizers, in the order that messages list them.
NAMES = [
    "clr",
    "lib_size",
    "lib_size_log",
    "passthrough",
    "pearson_residuals",
    "tf_idf",
]
# The stages that a change to the Leiden partitions computes again.
LEIDEN_STAGES = {"leiden", "cluster_selection"}


def _log_library_size(assay: Any, counts: Any) -> Any:
    """A custom normalizer: the log of library-size values, by another identity."""
    return np.log1p(norm_lib_size(assay, counts))


def _half_library_size(assay: Any, counts: Any) -> Any:
    """Another custom normalizer, which a run of the first must not take for it."""
    return norm_lib_size(assay, counts) / 2


def _identity(normalizer: Callable[..., Any]) -> dict[str, str]:
    """Return the identity that a run records for a custom normalizer of this module."""
    return {
        "module": normalizer.__module__,
        "qualname": normalizer.__qualname__,
        "code": callable_identity(normalizer)["code"],
    }


def _open(location: Path) -> DataStore:
    """Open the CITE-seq store in a new session, with its antibodies typed ADT."""
    return DataStore(str(location), default_assay="RNA", assay_types={"assay2": "ADT"})


def _view(location: Path) -> DataStore:
    """Open the store read-only in a new session, for calls that write nothing."""
    return DataStore(str(location), default_assay="RNA", zarr_mode="r")


def _copy(location: Path, tmp_path: Path) -> Path:
    target = tmp_path / "data.zarr"
    shutil.copytree(location, target)
    return target


def _computed(run: PipelineRun) -> set[str]:
    return {
        stage
        for stage, disposition in _dispositions(run).items()
        if disposition == "computed"
    }


def _method(store: DataStore, ref: Any) -> dict[str, Any]:
    """Return the normalizer that an artifact records among its parameters."""
    return store.artifacts.inspect(ref).parameters["normalization_method"]


def _record_as_1_4(
    location: Path, run_id: str, *, default_scaling: bool = False
) -> None:
    """Rewrite a run record as CyteArc 1.4.0 recorded it, without the normalizer.

    With ``default_scaling``, the run's PCA scaled its features, which 1.4.0
    did for a call that set no ``feat_scaling``, so the record leaves it out.
    """
    group = zarr.open_group(str(location), mode="r+")[pipeline_run_path(run_id)]
    attrs = dict(group.attrs)
    stages = attrs["config"]["stages"]
    del stages["normalization"]["method"]
    if default_scaling:
        assert stages["pca"].pop("feat_scaling") is True
    attrs["cytearcVersion"] = "1.4.0"
    group.attrs.put(attrs)


@pytest.fixture(scope="module")
def saved(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Commit the runs of one session that normalizes RNA with Pearson residuals."""
    location = tmp_path_factory.mktemp("recipe_normalizers") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    first = _open(location)
    first.RNA.normMethod = norm_pearson_residuals
    residuals = first.pipeline.rna(label="residuals", **SETTINGS)
    scaled = first.pipeline.rna(
        label="scaled", **{**SETTINGS, "pca": {"dims": 5, "feat_scaling": True}}
    )
    adt = first.pipeline.adt(label="adt", cells=residuals, **ADT_SETTINGS)
    wnn = first.pipeline.multimodal([residuals, adt], label="wnn", umap=False)
    first.RNA.normMethod = _log_library_size
    custom = first.pipeline.rna(label="custom", **SETTINGS)
    custom_adt = first.pipeline.adt(label="custom_adt", cells=custom, **ADT_SETTINGS)
    return SimpleNamespace(
        location=location,
        residuals=residuals,
        scaled=scaled,
        adt=adt,
        wnn=wnn,
        custom=custom,
        custom_adt=custom_adt,
    )


# The run of the issue ----------------------------------------------------------


def test_a_rerun_in_a_new_session_normalizes_as_the_run_did(
    datastore_zarr_root: str, tmp_path: Path
) -> None:
    location = tmp_path / "pbmc.zarr"
    shutil.copytree(datastore_zarr_root, location)
    # The first session runs the recipe on Pearson residuals.
    first = DataStore(str(location), default_assay="RNA")
    first.RNA.normMethod = norm_pearson_residuals
    run = first.pipeline.rna(label="residuals", **SETTINGS)
    assert first.pipeline.plan(run, leiden={"partitions": [2.0]}).computed == (
        "leiden",
        "cluster_selection",
    )

    # A second session, whose RNA assay normalizes by library size, reruns the
    # run with another Leiden resolution.
    second = DataStore(str(location), default_assay="RNA")
    assert second.RNA.normMethod is norm_lib_size
    run = second.pipeline.open(label="residuals")
    plan = second.pipeline.plan(run, leiden={"partitions": [2.0]})
    fine = second.pipeline.rerun(run, label="fine", leiden={"partitions": [2.0]})

    # Every stage up to the changed one reuses the run's artifacts.
    assert plan.computed == ("leiden", "cluster_selection")
    assert _computed(fine) == LEIDEN_STAGES
    for key in ("highly_variable_features", "normalized", "pca", "connectivity_map"):
        assert fine[key] == run[key], key
    assert _method(second, fine["normalized"]) == PEARSON
    assert fine.settings()["normalization"] == {"method": "pearson_residuals"}
    assert run.settings()["normalization"] == {"method": "pearson_residuals"}
    # The session's normalizer is as it was.
    assert second.RNA.normMethod is norm_lib_size


@pytest.mark.parametrize("label", ["residuals", "scaled", "adt", "wnn"])
def test_a_plan_in_a_new_session_reuses_every_stage_of_the_run(saved, label) -> None:
    run = getattr(saved, label)

    plan = read_only_plan(saved.location, run, default_assay="RNA")

    assert_plan_reuses_run(plan, run)


def test_the_run_records_the_normalizer_that_every_stage_applied(saved) -> None:
    store = _view(saved.location)
    run = saved.residuals

    stages = run.report()["run"]["config"]["stages"]
    summary = store.artifacts.inspect(run["highly_variable_features"]).inputs

    assert stages["normalization"] == {"method": "pearson_residuals"}
    assert _method(store, run["normalized"]) == PEARSON
    # Gene selection ranked the residuals' statistics.
    assert summary is not None
    assert (
        _method(store, ArtifactRef.from_dict(summary["feature_summary"]))
        == PEARSON_IDENTITY
    )
    # The joint run ranked each assay's markers with its run's normalizer.
    assert _method(store, saved.wnn["markers_RNA"]) == PEARSON_IDENTITY
    assert _method(store, saved.wnn["markers_assay2"]) == CLR_IDENTITY


# The method setting ----------------------------------------------------------


def test_the_builtin_normalizers_are_named_after_their_functions() -> None:
    assert dict(BUILTIN_NORMALIZERS) == {
        "lib_size": norm_lib_size,
        "lib_size_log": norm_lib_size_log,
        "pearson_residuals": norm_pearson_residuals,
        "clr": norm_clr,
        "tf_idf": norm_tf_idf,
        "passthrough": norm_dummy,
    }
    # A normalizer that provenance identifies as a built-in one is recorded by
    # its name, and any other by its identity.
    assert normalizer_setting(PearsonResidualNormalizer()) == "pearson_residuals"
    assert normalizer_setting(PearsonResidualNormalizer(theta=50.0)) == {
        "identity": "cytearc.assay.pearson_residuals:theta=50.0:clip=sqrt-n"
    }
    assert normalizer_setting(_log_library_size) == _identity(_log_library_size)


def test_a_named_normalizer_plans_its_run_in_a_session_of_another(saved) -> None:
    store = _view(saved.location)

    plan = store.pipeline.plan(
        "rna", normalization={"method": "pearson_residuals"}, **SETTINGS
    )

    assert plan.computed == ()
    assert plan.config == {
        key: value
        for key, value in saved.residuals.report()["run"]["config"].items()
        if key != "shutdown"
    }
    assert store.RNA.normMethod is norm_lib_size
    # Without a method, the run normalizes with the session's normalizer, so
    # its gene selection computes.
    plan = store.pipeline.plan("rna", **SETTINGS)
    assert plan.config["stages"]["normalization"] == {"method": "lib_size"}
    assert plan.computed[0] == "highly_variable_features"
    assert plan["highly_variable_features"].changed_parameters == (
        "normalization_method",
    )


def test_pearson_residuals_reach_pca_unscaled_unless_the_call_scales_them(
    saved,
) -> None:
    store = _view(saved.location)
    residuals = {"method": "pearson_residuals"}

    # The committed run turned feature scaling off, which a run of residuals
    # does by default, so a call without it reuses the run's PCA.
    plan = store.pipeline.plan(
        "rna", normalization=residuals, **{**SETTINGS, "pca": {"dims": 5}}
    )
    scaled = store.pipeline.plan(
        "rna",
        normalization=residuals,
        **{**SETTINGS, "pca": {"dims": 5, "feat_scaling": True}},
    )

    assert plan.computed == ()
    assert plan["pca"].outputs == (("pca", saved.residuals["pca"]),)
    assert (
        store.artifacts.inspect(saved.residuals["pca"]).parameters["feat_scaling"]
        is False
    )
    # The run records the settings that the call gave.
    assert plan.config["stages"]["pca"] == {"dims": 5}
    # A value that the call gives wins: scaled residuals are another PCA, the
    # one of the committed run that scaled them.
    assert scaled.computed == ()
    assert scaled["pca"].outputs == (("pca", saved.scaled["pca"]),)
    assert saved.scaled["pca"] != saved.residuals["pca"]
    assert (
        store.artifacts.inspect(saved.scaled["pca"]).parameters["feat_scaling"] is True
    )


@pytest.mark.parametrize(
    ("method", "error", "message"),
    [
        (
            "pearson",
            ValueError,
            f"normalization['method'] must be one of {NAMES!r}, or the identity "
            "of a custom normalizer that a run recorded; got 'pearson'",
        ),
        (
            norm_pearson_residuals,
            TypeError,
            "normalization['method'] takes the name of a built-in normalizer, "
            f"one of {NAMES!r}, not the normalizer itself; a run applies a "
            "custom normalizer that is the assay's normMethod",
        ),
        (
            1,
            TypeError,
            f"normalization['method'] must be one of {NAMES!r}, or the identity "
            "of a custom normalizer that a run recorded; got 1",
        ),
        (
            {"name": "lib_size"},
            ValueError,
            f"normalization['method'] must be one of {NAMES!r}, or the identity "
            "of a custom normalizer that a run recorded: a mapping of "
            "'identity', or of 'module', 'qualname' and 'code', to text; got "
            "{'name': 'lib_size'}",
        ),
        (
            {"identity": ""},
            ValueError,
            "or of 'module', 'qualname' and 'code', to text; got {'identity': ''}",
        ),
        (
            "tf_idf",
            ValueError,
            "Assay 'RNA' opens as RNAassay, but tf_idf needs an ATACassay; pass "
            "normalization={'method': ...} with one of "
            "['clr', 'lib_size', 'lib_size_log', 'passthrough', 'pearson_residuals']",
        ),
    ],
    ids=["unknown", "normalizer", "number", "not-identity", "empty", "atac-only"],
)
def test_a_method_that_names_no_normalizer_of_the_assay_is_refused(
    saved, method, error, message
) -> None:
    store = _view(saved.location)

    # A call, and a call that changes a run's settings, check the method alike.
    with pytest.raises(error) as called:
        store.pipeline.plan("rna", normalization={"method": method})
    with pytest.raises(error) as changed:
        store.pipeline.plan(saved.residuals, normalization={"method": method})

    assert message in str(called.value)
    assert str(changed.value) == str(called.value)


# Custom normalizers ------------------------------------------------------------


def _custom_refusal(normalizer: str) -> str:
    return (
        f"normalization['method'] is the custom normalizer "
        f"{_identity(_log_library_size)!r}, which a run applies only as the "
        f"assay's normMethod, but assay 'RNA' normalizes with {normalizer}; set "
        "the assay's normMethod to that normalizer, or pass "
        f"normalization={{'method': ...}} with one of {NAMES!r}"
    )


@pytest.mark.parametrize(
    ("session_normalizer", "named"),
    [
        (norm_lib_size, "lib_size"),
        (
            _half_library_size,
            f"the custom normalizer {_identity(_half_library_size)!r}",
        ),
        (
            lambda assay, counts: counts,
            "<lambda>, which provenance cannot identify",
        ),
    ],
    ids=["missing", "other", "unidentified"],
)
def test_a_custom_run_is_refused_where_the_assay_normalizes_otherwise(
    saved, tmp_path, session_normalizer, named
) -> None:
    location = _copy(saved.location, tmp_path)
    store = _open(location)
    store.RNA.normMethod = session_normalizer
    run = store.pipeline.open(label="custom")
    before = _digest(location)

    assert run.settings()["normalization"] == {"method": _identity(_log_library_size)}
    with pytest.raises(ValueError) as planned:
        store.pipeline.plan(run, leiden={"partitions": [2.0]})
    with pytest.raises(ValueError) as rerun:
        store.pipeline.rerun(run, label="fine", leiden={"partitions": [2.0]})

    assert str(planned.value) == str(rerun.value) == _custom_refusal(named)
    # Neither call wrote a run record, an artifact, or any other file.
    assert _digest(location) == before
    assert store.RNA.normMethod is session_normalizer


def test_a_custom_run_reruns_where_the_assay_normalizes_with_it(
    saved, tmp_path
) -> None:
    store = _open(_copy(saved.location, tmp_path))
    store.RNA.normMethod = _log_library_size
    run = store.pipeline.open(label="custom")

    plan = store.pipeline.plan(run)
    fine = store.pipeline.rerun(run, label="fine", leiden={"partitions": [2.0]})

    assert plan.computed == ()
    assert _computed(fine) == LEIDEN_STAGES
    assert fine["normalized"] == run["normalized"]
    assert _method(store, fine["normalized"]) == {
        "external_hook": True,
        **_identity(_log_library_size),
    }
    assert fine.settings()["normalization"] == run.settings()["normalization"]
    # A built-in method, or None for the assay's normMethod, reruns the run
    # with another normalizer, whose genes are selected again.
    store.RNA.normMethod = norm_lib_size
    for normalization in ({"method": "lib_size"}, None, {"method": None}):
        plan = store.pipeline.plan(run, normalization=normalization)
        assert plan.config["stages"]["normalization"]["method"] == "lib_size"
        assert plan.computed[0] == "highly_variable_features"


def test_a_multimodal_run_ranks_markers_with_the_normalizer_of_each_run(
    saved, tmp_path
) -> None:
    store = _open(_copy(saved.location, tmp_path))
    wnn = store.pipeline.open(label="wnn")

    # A rerun in a new session ranks the RNA markers of the joint clusters
    # with the residuals of the RNA run.
    fine = store.pipeline.rerun(
        wnn, label="fine", leiden={"partitions": [2.0], "selected": None}
    )

    assert _computed(fine) == {"leiden", "markers"}
    assert fine["markers_RNA"] != wnn["markers_RNA"]
    assert _method(store, fine["markers_RNA"]) == PEARSON_IDENTITY
    assert store.RNA.normMethod is norm_lib_size


def test_a_multimodal_run_of_a_custom_run_needs_its_normalizer_for_markers(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    store = _open(location)
    runs = [
        store.pipeline.open(label="custom"),
        store.pipeline.open(label="custom_adt"),
    ]
    before = _digest(location)

    with pytest.raises(ValueError) as caught:
        store.pipeline.multimodal(runs, umap=False)

    assert str(caught.value) == (
        f"Run {saved.custom.run_id} (rna) normalized assay 'RNA' with the custom "
        f"normalizer {_identity(_log_library_size)!r}, with which the multimodal "
        "run ranks the assay's markers, but the assay's normMethod is lib_size; "
        "set the assay's normMethod to that normalizer, or pass markers=False"
    )
    assert _digest(location) == before
    # A joint run without markers normalizes nothing.
    plan = store.pipeline.plan("multimodal", runs, umap=False, markers=False)
    assert plan["integration"].status == "computed"


# Runs that CyteArc 1.4.0 recorded -------------------------------------------------


def test_a_run_recorded_before_the_method_reruns_with_its_saved_normalizer(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    _record_as_1_4(location, saved.residuals.run_id)
    store = _open(location)
    earlier = store.pipeline.open(label="residuals")

    settings = earlier.settings()
    plan = store.pipeline.plan(earlier)
    fine = store.pipeline.rerun(earlier, label="fine", leiden={"partitions": [2.0]})

    assert earlier.report()["run"]["config"]["stages"]["normalization"] == {}
    # The run's normalized values record the residuals, which its settings
    # take, so its plan in a new session reuses every stage.
    assert settings == saved.residuals.settings()
    assert settings["normalization"] == {"method": "pearson_residuals"}
    assert plan.computed == ()
    assert [(stage.stage, stage.outputs) for stage in plan.stages] == [
        (stage.stage, stage.outputs)
        for stage in read_only_plan(saved.location, saved.residuals).stages
    ]
    assert _computed(fine) == LEIDEN_STAGES
    assert fine["normalized"] == earlier["normalized"]
    assert store.RNA.normMethod is norm_lib_size


def test_a_scaled_run_recorded_before_the_method_reruns_with_its_scaling(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    _record_as_1_4(location, saved.scaled.run_id, default_scaling=True)
    store = _open(location)
    earlier = store.pipeline.open(label="scaled")

    settings = earlier.settings()
    plan = store.pipeline.plan(earlier)
    fine = store.pipeline.rerun(earlier, label="fine", leiden={"partitions": [2.0]})

    # The record holds what a 1.4.0 call gave, and the settings the scaling
    # that its PCA applied, where a run of residuals now defaults to none.
    assert earlier.report()["run"]["config"]["stages"]["pca"] == {"dims": 5}
    assert settings["normalization"] == {"method": "pearson_residuals"}
    assert settings["pca"] == {"dims": 5, "feat_scaling": True}
    assert settings == saved.scaled.settings()
    assert plan.computed == ()
    assert [(stage.stage, stage.outputs) for stage in plan.stages] == [
        (stage.stage, stage.outputs)
        for stage in read_only_plan(saved.location, saved.scaled).stages
    ]
    assert _computed(fine) == LEIDEN_STAGES
    assert fine["pca"] == earlier["pca"]
    assert store.artifacts.inspect(fine["pca"]).parameters["feat_scaling"] is True
    # A value that the rerun gives wins, and takes the unscaled residuals.
    unscaled = store.pipeline.plan(earlier, pca={"feat_scaling": False})
    assert unscaled.config["stages"]["pca"] == {"dims": 5, "feat_scaling": False}
    assert unscaled["pca"].outputs == (("pca", saved.residuals["pca"]),)


def test_a_scaled_run_recorded_before_the_method_reruns_without_pca(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    _record_as_1_4(location, saved.scaled.run_id, default_scaling=True)
    store = _open(location)
    earlier = store.pipeline.open(label="scaled")

    flat = store.pipeline.rerun(earlier, label="flat", pca={"dims": 0})

    # The scaling holds only while a PCA runs, so dims=0 leaves it out.
    assert flat.settings()["pca"] == {"dims": 0}
    assert "pca" not in flat
    assert flat["normalized"] == earlier["normalized"]
    assert _computed(flat) == {"ann_index", "neighbors", "connectivity", *LEIDEN_STAGES}


@pytest.mark.parametrize(
    ("method", "pca", "expected"),
    [
        ("pearson_residuals", {"dims": 5}, {"dims": 5, "feat_scaling": True}),
        (_identity(_log_library_size), {"dims": 5}, {"dims": 5, "feat_scaling": True}),
        ("lib_size", {"dims": 5}, {"dims": 5}),
        ("clr", {"dims": 5}, {"dims": 5}),
        ("pearson_residuals", {"dims": 0}, {"dims": 0}),
        (
            "pearson_residuals",
            {"dims": 5, "feat_scaling": False},
            {"dims": 5, "feat_scaling": False},
        ),
        ("pearson_residuals", {"dims": True}, {"dims": True}),
        ("pearson_residuals", None, None),
    ],
    ids=[
        "residuals",
        "custom",
        "lib-size",
        "clr",
        "no-pca",
        "given",
        "bool-dims",
        "default-pca",
    ],
)
def test_the_scaling_of_an_earlier_run_goes_where_its_default_changed(
    method, pca, expected
) -> None:
    settings: dict[str, Any] = {"pca": pca}

    add_applied_feat_scaling(settings, method)

    assert settings == {"pca": expected}
    # Settings without PCA, as an ATAC run's, stay as they are.
    without_pca: dict[str, Any] = {"lsi": {"dims": 5}}
    add_applied_feat_scaling(without_pca, method)
    assert without_pca == {"lsi": {"dims": 5}}


def test_a_custom_run_recorded_before_the_method_reruns_only_with_its_normalizer(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    _record_as_1_4(location, saved.custom.run_id)
    store = _open(location)
    earlier = store.pipeline.open(label="custom")
    before = _digest(location)

    assert earlier.settings()["normalization"] == {
        "method": _identity(_log_library_size)
    }
    with pytest.raises(ValueError) as planned:
        store.pipeline.plan(earlier)
    with pytest.raises(ValueError) as rerun:
        store.pipeline.rerun(earlier, label="fine")
    assert str(planned.value) == str(rerun.value) == _custom_refusal("lib_size")
    assert _digest(location) == before

    store.RNA.normMethod = _log_library_size
    assert store.pipeline.plan(earlier).computed == ()


def test_a_run_that_saved_no_normalized_values_takes_the_session_normalizer(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    store = _open(location)
    token = CancellationToken()

    def stop_after_filtering(event: PipelineEvent) -> None:
        if (event.kind, event.stage) == ("stage_completed", "filtering"):
            token.cancel("stopped before normalization")

    with pytest.raises(PipelineCancelled) as caught:
        store.pipeline.rna(
            normalization={"method": "pearson_residuals"},
            callback=stop_after_filtering,
            cancel=token,
            **SETTINGS,
        )
    _record_as_1_4(location, caught.value.run_id)
    earlier = _open(location).pipeline.open(run_id=caught.value.run_id)

    assert earlier.status == "interrupted"
    assert earlier.settings()["normalization"] == {}
    plan = store.pipeline.plan(earlier)
    assert plan.config["stages"]["normalization"] == {"method": "lib_size"}


def _point_normalization_at_a_missing_artifact(
    location: Path, run: PipelineRun
) -> None:
    """Point a run's normalization stage output at an artifact that the store lacks."""
    ordinal = list(run.report()["run"]["stageOrder"]).index("normalization")
    group = zarr.open_group(str(location), mode="r+")[
        pipeline_stage_path(run.run_id, ordinal)
    ]
    attrs = dict(group.attrs)
    attrs["outputs"][0]["artifact"]["artifact_id"] = "0" * 64
    group.attrs.put(attrs)


@pytest.mark.parametrize("damage", ["missing", "incomplete"])
def test_a_run_whose_normalized_values_are_gone_has_no_settings(
    saved, tmp_path, damage
) -> None:
    location = _copy(saved.location, tmp_path)
    run = saved.residuals
    _record_as_1_4(location, run.run_id)
    if damage == "missing":
        _point_normalization_at_a_missing_artifact(location, run)
        artifact = "0" * 12
    else:
        # The values are there and record their normalizer, but their write
        # never completed, so they do not say what the run applied.
        root = zarr.open_group(str(location), mode="r+")
        artifact_group(root, run["normalized"]).attrs["complete"] = False
        assert _method(_view(saved.location), run["normalized"]) == PEARSON
        artifact = run["normalized"].artifact_id[:12]
    earlier = _open(location).pipeline.open(label="residuals")

    with pytest.raises(ValueError) as caught:
        earlier.settings()

    assert str(caught.value) == (
        f"Pipeline run {run.run_id}, recorded by CyteArc 1.4.0, records no "
        f"normalization method, and its normalized values, artifact {artifact}, "
        "are missing or record no normalization_method; "
        "report()['run']['config'] holds the settings that it recorded"
    )


def test_a_run_that_records_its_method_lends_it_without_its_normalized_values(
    saved, tmp_path
) -> None:
    location = _copy(saved.location, tmp_path)
    _point_normalization_at_a_missing_artifact(location, saved.residuals)
    store = _open(location)

    # The joint run ranks the RNA markers with the residuals that the RNA run
    # records by name, so its plan reuses every stage without reading the
    # normalized values that the record no longer finds.
    plan = store.pipeline.plan(store.pipeline.open(label="wnn"))

    assert plan.computed == ()
    assert store.RNA.normMethod is norm_lib_size


def test_a_run_interrupted_while_it_normalized_takes_the_session_normalizer(
    saved, tmp_path, monkeypatch
) -> None:
    location = _copy(saved.location, tmp_path)
    store = _open(location)

    def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt("stopped while normalizing")

    monkeypatch.setattr(store, "_features_normalize", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.pipeline.rna(normalization={"method": "pearson_residuals"}, **SETTINGS)
    (run_id,) = [
        run.run_id for run in store.pipeline.list_runs(status="interrupted", limit=100)
    ]
    _record_as_1_4(location, run_id)
    earlier = _open(location).pipeline.open(run_id=run_id)

    # The normalization stage started and saved no values, so nothing says
    # which normalizer the run applied, and a call takes the session's.
    statuses = {
        stage["stage"]: (stage["status"], stage["outputs"])
        for stage in earlier.report()["stages"]
    }
    assert statuses["normalization"] == ("interrupted", [])
    assert earlier.settings()["normalization"] == {}
    plan = store.pipeline.plan(earlier)
    assert plan.config["stages"]["normalization"] == {"method": "lib_size"}


# The session's normalizer -------------------------------------------------------


def test_the_session_normalizer_returns_however_the_run_ends(saved, tmp_path) -> None:
    store = _open(_copy(saved.location, tmp_path))
    residuals = {"method": "pearson_residuals"}
    seen: list[tuple[str | None, Any]] = []

    def record(event: PipelineEvent) -> None:
        if event.kind == "stage_started":
            seen.append((event.stage, store.RNA.normMethod))

    run = store.pipeline.rna(normalization=residuals, callback=record, **SETTINGS)

    # Every stage that ran normalized with the run's normalizer.
    assert run.status == "completed"
    assert seen == [
        (stage, norm_pearson_residuals)
        for stage in (
            "input_snapshot",
            "filtering",
            "highly_variable_features",
            "normalization",
            "pca",
            "ann_index",
            "neighbors",
            "connectivity",
            "leiden",
            "cluster_selection",
        )
    ]
    assert store.RNA.normMethod is norm_lib_size
    # A stage that fails ends the run failed.
    with pytest.raises(PipelineExecutionError):
        store.pipeline.rna(
            normalization=residuals,
            **{**SETTINGS, "leiden": {"partitions": [0.5], "backend": "nope"}},
        )
    assert store.RNA.normMethod is norm_lib_size
    # A cancellation ends the run interrupted.
    token = CancellationToken()
    seen.clear()

    def stop_after_normalization(event: PipelineEvent) -> None:
        if (event.kind, event.stage) == ("stage_completed", "normalization"):
            seen.append((event.stage, store.RNA.normMethod))
            token.cancel("stopped after normalization")

    with pytest.raises(PipelineCancelled):
        store.pipeline.rna(
            normalization=residuals,
            callback=stop_after_normalization,
            cancel=token,
            **SETTINGS,
        )
    assert seen == [("normalization", norm_pearson_residuals)]
    assert store.RNA.normMethod is norm_lib_size
    # A plan that a stage's check stops, after the stages before it reused
    # their artifacts, leaves it as it was too.
    with pytest.raises(ValueError, match="backend"):
        store.pipeline.plan(run, leiden={"backend": "nope"})
    assert store.RNA.normMethod is norm_lib_size


# The other recipes ----------------------------------------------------------------


@pytest.fixture(scope="module")
def modalities(tmp_path_factory) -> Path:
    """A prepared store of 40 cells with RNA, ADT and ATAC counts."""
    location = tmp_path_factory.mktemp("normalizer_modalities") / "data.zarr"
    rng = np.random.default_rng(11)
    write_count_store(
        str(location),
        {
            "RNA": rng.poisson(1.0, size=(40, 60)),
            "ADT": rng.poisson(5.0, size=(40, 8)),
            "ATAC": rng.poisson(0.5, size=(40, 80)),
        },
        "uint16",
        feature_ids={
            "ATAC": [f"chr1:{index * 100}-{index * 100 + 50}" for index in range(80)]
        },
    )
    # A writable open prepares the assays, which a read-only open needs.
    open_store(location)
    return location


_SMALL_GRAPH: dict[str, Any] = {
    "filtering": False,
    "neighbors": {"k": 3},
    "umap": False,
    "leiden": False,
    "markers": False,
}


def test_atac_and_adt_runs_normalize_with_the_method_they_name(
    modalities, tmp_path
) -> None:
    store = open_store(_copy(modalities, tmp_path))

    atac = store.pipeline.atac(
        normalization={"method": "clr"},
        prevalent_peaks={"top_n": 30},
        lsi={"dims": 3},
        **_SMALL_GRAPH,
    )
    adt = store.pipeline.adt(
        normalization={"method": "passthrough"}, pca={"dims": 3}, **_SMALL_GRAPH
    )

    clr = {"external_hook": True, **CLR_IDENTITY}
    assert atac.settings()["normalization"] == {"method": "clr"}
    assert _method(store, atac["normalized"]) == clr
    # The prevalent peaks ranked the CLR values too.
    peaks = store.artifacts.inspect(atac["prevalent_peaks"]).inputs
    assert peaks is not None
    assert _method(store, ArtifactRef.from_dict(peaks["feature_summary"])) == (
        CLR_IDENTITY
    )
    assert adt.settings()["normalization"] == {"method": "passthrough"}
    assert _method(store, adt["normalized"]) == {
        "external_hook": True,
        "module": "cytearc.assay",
        "qualname": "norm_dummy",
    }
    # The assays normalize with their defaults again.
    assert store.ATAC.normMethod is norm_tf_idf
    assert store.ADT.normMethod is norm_clr


@pytest.mark.parametrize(
    ("recipe", "method", "message"),
    [
        (
            "atac",
            "lib_size",
            "Assay 'ATAC' opens as ATACassay, but lib_size needs an RNAassay; "
            "pass normalization={'method': ...} with one of "
            "['clr', 'passthrough', 'tf_idf']",
        ),
        (
            "adt",
            "pearson_residuals",
            "Assay 'ADT' opens as ADTassay, but pearson_residuals needs an "
            "RNAassay; pass normalization={'method': ...} with one of "
            "['clr', 'passthrough']",
        ),
    ],
)
def test_a_normalizer_of_another_assay_class_is_refused_before_the_run(
    modalities, recipe, method, message
) -> None:
    store = open_store(modalities, zarr_mode="r")

    with pytest.raises(ValueError) as caught:
        store.pipeline.plan(recipe, normalization={"method": method})

    assert str(caught.value) == message


def _normalize(store: DataStore, assay: str, **flags: Any) -> Any:
    """Normalize every cell and feature of ``assay`` with the flags."""
    features = store.features.snapshot(
        from_assay=assay, feature_indexes=np.arange(store.get_assay(assay).feats.N)
    )
    return store.features.normalize(store.snapshot_cell_selection(), features, **flags)


@pytest.mark.parametrize(
    ("recipe", "settings", "assay", "normalizer", "flag", "call"),
    [
        (
            "rna",
            {
                "normalization": {"method": "pearson_residuals", "log_transform": True},
                "doublets": False,
            },
            "RNA",
            norm_pearson_residuals,
            "log_transform",
            lambda store: _normalize(store, "RNA", log_transform=True),
        ),
        (
            "rna",
            {
                "normalization": {"method": "pearson_residuals"},
                "cell_cycle": {"log_transform": True},
                "doublets": False,
            },
            "RNA",
            norm_pearson_residuals,
            "log_transform",
            lambda store: store.scores.cell_cycle(
                store.snapshot_cell_selection(), log_transform=True
            ),
        ),
        (
            "atac",
            {"normalization": {"log_transform": True}},
            "ATAC",
            norm_tf_idf,
            "log_transform",
            lambda store: _normalize(store, "ATAC", log_transform=True),
        ),
        (
            "adt",
            {"normalization": {"renormalize_subset": True}},
            "ADT",
            norm_clr,
            "renormalize_subset",
            lambda store: _normalize(store, "ADT", renormalize_subset=True),
        ),
    ],
    ids=["rna-normalization", "rna-cell-cycle", "atac", "adt"],
)
def test_a_flag_that_the_run_normalizer_cannot_apply_is_refused_before_the_run(
    modalities, tmp_path, recipe, settings, assay, normalizer, flag, call
) -> None:
    store = open_store(_copy(modalities, tmp_path))
    before = written(store)

    with pytest.raises(ValueError) as planned:
        store.pipeline.plan(recipe, **settings)
    with pytest.raises(ValueError) as ran:
        getattr(store.pipeline, recipe)(**settings)
    assert written(store) == before
    # The stage's method refuses the flag when its assay normalizes so.
    store.get_assay(assay).normMethod = normalizer
    with pytest.raises(ValueError) as called:
        call(store)

    message = str(called.value)
    assert str(planned.value) == str(ran.value) == message
    assert message.startswith(
        f"{type(store.get_assay(assay)).__name__} '{assay}' normalizes with "
    )
    assert f"does not support {flag}=True" in message
    assert message.endswith(f"Pass {flag}=False.")


def test_a_custom_rna_normalizer_of_an_adt_assay_is_refused(modalities) -> None:
    store = open_store(modalities, zarr_mode="r")
    store.ADT.normMethod = PearsonResidualNormalizer(theta=10.0)

    with pytest.raises(ValueError) as caught:
        store.pipeline.plan("adt")

    assert str(caught.value) == (
        "Assay 'ADT' opens as ADTassay, but the custom normalizer "
        "{'identity': 'cytearc.assay.pearson_residuals:theta=10.0:clip=sqrt-n'} "
        "needs an RNAassay; pass normalization={'method': ...} with one of "
        "['clr', 'passthrough']"
    )


def test_an_unidentified_normalizer_is_refused_before_the_run(modalities) -> None:
    store = open_store(modalities, zarr_mode="r")
    store.RNA.normMethod = lambda assay, counts: counts

    with pytest.raises(ValueError) as caught:
        store.pipeline.plan("rna")

    assert str(caught.value).startswith(
        "A run records the normalizer of assay 'RNA' by its identity: Dynamic or "
        "stateful callables must define artifact_identity"
    )


def test_a_perturbation_run_normalizes_with_the_method_it_names(tmp_path) -> None:
    rng = np.random.default_rng(4)
    guides = np.zeros((80, 4), dtype=np.int64)
    guides[np.arange(80), np.arange(80) % 4] = 30
    store = write_screen(
        tmp_path / "screen.zarr",
        rng.poisson(2.0, size=(80, 40)),
        guides,
        targets=["GENE_A", "GENE_B", "NT", "NT"],
        target_ids=["RNA0", "RNA1", "", ""],
    )
    before = written(store)

    run = store.pipeline.perturbation(
        labels={"controls": ["NT"]},
        filtering=False,
        guide_calling={"method": "fixed", "min_umi": 10},
        efficiency=False,
        hvg={"top_n": 20},
        normalization={"method": "pearson_residuals"},
        pca={"dims": 3},
        edistance=False,
    )

    assert written(store) != before
    assert run.settings()["normalization"] == {"method": "pearson_residuals"}
    assert _method(store, run["normalized"]) == PEARSON
    # The PCA of residuals scales no feature by default, and the run, which
    # records its method, records only the PCA settings that the call gave.
    assert store.artifacts.inspect(run["pca"]).parameters["feat_scaling"] is False
    assert run.settings()["pca"] == {"dims": 3}
    assert store.RNA.normMethod is norm_lib_size
