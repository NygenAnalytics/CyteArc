import asyncio
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import cytearc.datastore._pipeline.ledger as ledger_module
import cytearc.datastore._pipeline.recipes.rna as rna_module
import cytearc.storage.pipeline_runs as run_storage
from cytearc.clustering.leiden import canonical_resolution
from cytearc.datastore._pipeline.events import PipelineEvent, PipelineEventEmitter
from cytearc.datastore._pipeline.filtering import (
    default_filter_columns,
    resolve_filtering,
)
from cytearc.datastore._pipeline.ledger import RunLedger
from cytearc.datastore._pipeline.recipes.common import resolve_run_filtering
from cytearc.datastore._pipeline.settings import (
    DEFAULT_LEIDEN_PARTITIONS,
    column_sequence,
    finite_real,
    keyword_default,
    leiden_config,
    resolve_leiden,
)
from cytearc.datastore._pipeline.stages import CellSource
from cytearc.datastore.pipeline_run import PipelineExecutionError, PipelineRun
from cytearc.storage.artifact_writer import ArtifactPlanReceipt
from cytearc.storage.pipeline_runs import (
    PipelineErrorRecord,
    PipelineFieldDescriptor,
    PipelineInterruptionRecord,
    PipelineOutputRecord,
    PipelinePlanRecord,
    PipelineRunRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    PipelineStageRecord,
)
from cytearc.storage.refs import ArtifactRef
from cytearc.utils.process import ProcessTreeRssMeasurement


def _ref(value: str = "a") -> ArtifactRef:
    return ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=value * 64,
    )


def _events(*callbacks: Any) -> PipelineEventEmitter:
    return PipelineEventEmitter(
        callbacks,
        run_id="run",
        recipe="rna",
        label=None,
        stage_order=("stage",),
    )


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.1,
        rss_baseline_bytes=10,
        rss_peak_bytes=20,
        rss_incremental_peak_bytes=10,
        sample_interval_seconds=0.1,
        sample_count=1,
        sampling_error_count=0,
        rss_unavailable_reason=None,
    )


def _running_stage(**changes: Any) -> PipelineStageRecord:
    values: dict[str, Any] = {
        "stage": "one",
        "ordinal": 0,
        "started_at_ns": 10,
        "finished_at_ns": None,
        "status": "running",
        "complete": False,
        "outputs": (),
        "plans": (),
        "metrics": None,
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineStageRecord(**values)


def _terminal_stage(**changes: Any) -> PipelineStageRecord:
    values: dict[str, Any] = {
        "stage": "one",
        "ordinal": 0,
        "started_at_ns": 10,
        "finished_at_ns": 20,
        "status": "completed",
        "complete": True,
        "outputs": (),
        "plans": (),
        "metrics": _metrics(),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineStageRecord(**values)


def _running_run(**changes: Any) -> PipelineRunRecord:
    values: dict[str, Any] = {
        "run_id": "1" * 64,
        "recipe": "rna",
        "requested_label": None,
        "label": None,
        "assay": "RNA",
        "started_at_ns": 10,
        "finished_at_ns": None,
        "status": "running",
        "complete": False,
        "cytearc_version": "1.0",
        "config": {},
        "stage_order": ("one",),
        "outputs": (),
        "fields": (),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineRunRecord(**values)


def _terminal_run(**changes: Any) -> PipelineRunRecord:
    values: dict[str, Any] = {
        "run_id": "1" * 64,
        "recipe": "rna",
        "requested_label": None,
        "label": None,
        "assay": "RNA",
        "started_at_ns": 10,
        "finished_at_ns": 20,
        "status": "completed",
        "complete": True,
        "cytearc_version": "1.0",
        "config": {},
        "stage_order": ("one",),
        "outputs": (),
        "fields": (),
        "error": None,
        "interruption": None,
    }
    values.update(changes)
    return PipelineRunRecord(**values)


def test_pipeline_scalar_and_json_contract_rejections() -> None:
    invalid_calls = (
        (run_storage._validate_bool, (1, "value"), TypeError, "a boolean"),
        (
            run_storage._validate_non_negative_int,
            (True, "value"),
            TypeError,
            "a non-negative integer",
        ),
        (run_storage._validate_positive_int, (0, "value"), TypeError, "a positive"),
        (
            run_storage._validate_non_negative_float,
            (object(), "value"),
            TypeError,
            "a non-negative number",
        ),
        (
            run_storage._validate_non_negative_float,
            (float("inf"), "value"),
            ValueError,
            "finite and non-negative",
        ),
        (run_storage._validate_positive_float, (0, "value"), ValueError, "positive"),
        (run_storage._mapping, ([], "value"), TypeError, "a mapping"),
    )
    for function, arguments, error_type, message in invalid_calls:
        with pytest.raises(error_type, match=f"^value must be {message}"):
            function(*arguments)

    assert run_storage._validate_nullable_non_negative_int(None, "value") is None
    assert run_storage._validate_nullable_positive_int(None, "value") is None
    with pytest.raises(ValueError, match="missing.*extra"):
        run_storage._exact_mapping({"extra": 1}, frozenset({"needed"}), "value")
    with pytest.raises(ValueError, match="non-finite"):
        run_storage._json_value(float("nan"), "value")
    with pytest.raises(TypeError, match="keys"):
        run_storage._json_value({1: "bad"}, "value")
    with pytest.raises(TypeError, match="unsupported"):
        run_storage._json_value({1, 2}, "value")
    with pytest.raises(ValueError, match="output artifact is not a valid artifact"):
        PipelineOutputRecord.from_dict({"key": "out", "artifact": {}})
    with pytest.raises(TypeError, match="raised directly"):
        run_storage._raise_type("raised directly")


def test_pipeline_leaf_records_reject_malformed_contracts() -> None:
    with pytest.raises(TypeError, match="message"):
        PipelineErrorRecord("ValueError", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="128"):
        PipelineErrorRecord("x" * 129, "message")
    with pytest.raises(ValueError, match="512"):
        PipelineErrorRecord("ValueError", "x" * 513)
    truncated = PipelineErrorRecord.from_exception(ValueError("x" * 600))
    assert len(truncated.message) == 512
    with pytest.raises(TypeError, match="message"):
        PipelineErrorRecord.from_dict({"type": "ValueError", "message": 1})

    with pytest.raises(ValueError, match="appear together"):
        PipelineInterruptionRecord("signal", "stop", 1, signal_number=2)
    with pytest.raises(ValueError, match="too long"):
        PipelineInterruptionRecord("x" * 129, "stop", 1)

    with pytest.raises(TypeError, match="output artifact"):
        PipelineOutputRecord("out", object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="stage output artifact"):
        PipelineStageOutputRecord("out", object(), False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reused"):
        PipelineStageOutputRecord("out", _ref(), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="plan ref"):
        PipelinePlanRecord("op", object(), "created")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="disposition"):
        PipelinePlanRecord("op", _ref(), "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="disposition"):
        PipelinePlanRecord.from_dict(
            {"operation": "op", "ref": _ref().to_dict(), "disposition": "invalid"}
        )


def test_pipeline_metrics_and_field_contract_rejections() -> None:
    base = _metrics()
    with pytest.raises(ValueError, match="baseline"):
        replace(base, rss_peak_bytes=None, rss_unavailable_reason="missing")
    with pytest.raises(ValueError, match="explicit reason"):
        replace(
            base,
            rss_baseline_bytes=None,
            rss_peak_bytes=None,
            rss_incremental_peak_bytes=None,
        )
    with pytest.raises(ValueError, match="unavailable reason"):
        replace(base, rss_unavailable_reason="unexpected")

    valid = PipelineFieldDescriptor(
        key="score",
        axis="cells",
        artifact=_ref(),
        source_value="values",
        value_index=None,
        dtype="float64",
        fill="nan",
        missing_mask=None,
        display={"limits": [0, 1]},
    )
    assert np.isnan(valid.fill_value)
    with pytest.raises(ValueError, match="axis"):
        replace(valid, axis="rows")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="artifact"):
        replace(valid, artifact=object())
    with pytest.raises(ValueError, match="string 'nan'"):
        replace(valid, fill=float("nan"))
    with pytest.raises(TypeError, match="JSON scalar"):
        replace(valid, fill=[])

    raw = valid.to_dict()
    raw["axis"] = "rows"
    with pytest.raises(ValueError, match="axis"):
        PipelineFieldDescriptor.from_dict(raw)
    raw = valid.to_dict()
    raw["fill"] = []
    with pytest.raises(TypeError, match="JSON scalar"):
        PipelineFieldDescriptor.from_dict(raw)


def test_pipeline_stage_record_rejects_inconsistent_states() -> None:
    output = PipelineStageOutputRecord("out", _ref(), False)
    plan = PipelinePlanRecord("op", _ref(), "created")
    interruption = PipelineInterruptionRecord("shutdown", "stop", 1)
    error = PipelineErrorRecord("ValueError", "bad")

    invalid = (
        ({"finished_at_ns": 9}, ValueError, "precede"),
        ({"status": "unknown"}, ValueError, "status"),
        ({"outputs": [output]}, TypeError, "outputs"),
        ({"plans": [plan]}, TypeError, "plans"),
        ({"complete": True}, ValueError, "running stages"),
        ({"metrics": _metrics()}, ValueError, "receipts"),
        ({"error": error}, ValueError, "terminal details"),
    )
    for changes, error_type, message in invalid:
        with pytest.raises(error_type, match=message):
            _running_stage(**changes)

    terminal_invalid = (
        ({"finished_at_ns": None}, "finish time"),
        ({"status": "skipped", "outputs": (output,)}, "skipped"),
        ({"status": "failed"}, "failed stages"),
        (
            {
                "status": "interrupted",
                "interruption": interruption,
                "outputs": (output,),
            },
            "outputs",
        ),
        ({"status": "interrupted"}, "interrupted stages"),
        ({"error": error}, "successful stages"),
        ({"outputs": (output, output)}, "unique"),
    )
    for changes, message in terminal_invalid:
        with pytest.raises(ValueError, match=message):
            _terminal_stage(**changes)

    raw = _running_stage().to_dict()
    raw["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        PipelineStageRecord.from_dict(raw)
    raw = _running_stage().to_dict()
    raw["outputs"] = "bad"
    with pytest.raises(TypeError, match="outputs"):
        PipelineStageRecord.from_dict(raw)
    raw = _running_stage().to_dict()
    raw["plans"] = "bad"
    with pytest.raises(TypeError, match="plans"):
        PipelineStageRecord.from_dict(raw)


def test_pipeline_stage_record_diagnostics_contract() -> None:
    error = PipelineErrorRecord("ValueError", "bad")
    summary = {"cells": 3, "dims": [1, 2]}

    # A running stage holds neither diagnostic, and only a completed stage
    # holds a summary; any terminal stage keeps the warnings it logged.
    for changes, error_type, message in (
        ({"summary": summary}, ValueError, "running stages cannot contain diagnostics"),
        ({"warnings": ("slow",)}, ValueError, "running stages cannot contain"),
        ({"warnings": ["slow"]}, TypeError, "tuple of non-empty strings"),
        ({"warnings": ("",)}, TypeError, "tuple of non-empty strings"),
        ({"warnings": ("slow",) * 51}, ValueError, "record limits"),
        ({"warnings": ("x" * 513,)}, ValueError, "record limits"),
        ({"summary": {"rate": float("nan")}}, ValueError, "non-finite"),
        ({"summary": ["cells"]}, TypeError, "stage summary must be a mapping"),
    ):
        with pytest.raises(error_type, match=message):
            _running_stage(**changes)
    for changes in (
        {"status": "skipped", "summary": summary},
        {"status": "failed", "error": error, "summary": summary},
    ):
        with pytest.raises(
            ValueError, match="only completed stages can hold a summary"
        ):
            _terminal_stage(**changes)
    failed = _terminal_stage(status="failed", error=error, warnings=("slow",))
    assert failed.warnings == ("slow",)

    completed = _terminal_stage(summary=summary, warnings=("slow", "late"))
    assert PipelineStageRecord.from_dict(completed.to_dict()) == completed
    # A record written before stages held diagnostics still reads.
    raw = _terminal_stage().to_dict()
    del raw["summary"], raw["warnings"]
    assert PipelineStageRecord.from_dict(raw) == _terminal_stage()
    raw = completed.to_dict()
    raw["warnings"] = "slow"
    with pytest.raises(TypeError, match="^stage warnings must be a sequence$"):
        PipelineStageRecord.from_dict(raw)
    raw = completed.to_dict()
    raw["notes"] = []
    with pytest.raises(ValueError, match=r"extra \['notes'\]"):
        PipelineStageRecord.from_dict(raw)


def test_pipeline_run_record_rejects_inconsistent_states() -> None:
    output = PipelineOutputRecord("out", _ref())
    field = PipelineFieldDescriptor(
        key="I",
        axis="cells",
        artifact=_ref(),
        source_value="values",
        value_index=None,
        dtype="bool",
        fill=False,
        missing_mask=None,
        display=None,
    )
    error = PipelineErrorRecord("ValueError", "bad")
    interruption = PipelineInterruptionRecord("shutdown", "stop", 1)

    running_invalid = (
        ({"finished_at_ns": 9}, ValueError, "precede"),
        ({"status": "unknown"}, ValueError, "status"),
        ({"stage_order": ()}, TypeError, "stage_order"),
        ({"stage_order": ("one", "one")}, ValueError, "unique"),
        ({"outputs": [output]}, TypeError, "outputs"),
        ({"fields": [field]}, TypeError, "fields"),
        ({"complete": True}, ValueError, "running runs"),
        ({"outputs": (output,)}, ValueError, "terminal results"),
        ({"error": error}, ValueError, "terminal details"),
    )
    for changes, error_type, message in running_invalid:
        with pytest.raises(error_type, match=message):
            _running_run(**changes)

    terminal_invalid = (
        ({"finished_at_ns": None}, "finished_at_ns"),
        ({"error": error}, "completed runs"),
        ({"requested_label": "wanted", "label": None}, "label"),
        ({"status": "failed", "error": error, "label": "bad"}, "expose"),
        ({"status": "failed"}, "failed runs require"),
        (
            {
                "status": "interrupted",
                "interruption": interruption,
                "outputs": (output,),
            },
            "expose",
        ),
        ({"status": "interrupted"}, "interrupted runs require"),
        ({"outputs": (output, output)}, "output keys"),
        ({"fields": (field, field)}, "field keys"),
    )
    for changes, message in terminal_invalid:
        with pytest.raises(ValueError, match=message):
            _terminal_run(**changes)

    for key in ("stageOrder", "outputs", "fields"):
        raw = _running_run().to_dict()
        raw[key] = "bad"
        with pytest.raises(TypeError, match=f"^run {key} must be a sequence$"):
            PipelineRunRecord.from_dict(raw)
    raw = _running_run().to_dict()
    raw["status"] = "unknown"
    with pytest.raises(ValueError, match="status"):
        PipelineRunRecord.from_dict(raw)
    raw = _running_run().to_dict()
    raw["runId"] = 1
    with pytest.raises(TypeError, match="runId"):
        PipelineRunRecord.from_dict(raw)


class _Cells:
    def __init__(self) -> None:
        self.columns = {
            "I",
            "RNA_nCounts",
            "RNA_nFeatures",
            "sample",
            "batch",
            "note",
            "numeric",
        }

    def get_dtype(self, column: str) -> np.dtype[Any]:
        return np.dtype(bool if column == "I" else np.int64)

    def fetch_all(self, column: str) -> np.ndarray:
        # The recipe refuses a cell column that selects no cell.
        return np.ones(4, dtype=bool)


class _FakeRNA:
    # The recipe checks PCA dimensions against the assay's features too.
    feats = SimpleNamespace(N=100)


class _RecipeStore:
    def __init__(self) -> None:
        self._defaultAssay: str | None = "RNA"
        self.assay_names: list[str] = ["RNA"]
        self.cells = _Cells()
        self.assay: Any = _FakeRNA()
        self.zw = None

    def _get_assay(self, name: str) -> Any:
        return self.assay

    def _metric_assays(self, attrs: Any, artifact_metrics: Any) -> list[str]:
        # The store's RNA assay wrote every QC column it holds.
        return ["RNA" for attr in attrs if attr.startswith("RNA_")]


def _recipe_kwargs() -> dict[str, Any]:
    return {
        "assay": "RNA",
        "label": None,
        "cells": "I",
        "cell_calling": False,
        "hashing": False,
        "filtering": False,
        "cell_cycle": False,
        "hvg": {"top_n": 10},
        "normalization": None,
        "pca": {"dims": 5},
        "harmony": None,
        "ann_index": None,
        "neighbors": {"k": 3},
        "connectivity": None,
        "embedding_initialization": None,
        "umap": False,
        "tsne": False,
        "leiden": True,
        "paris": False,
        "membership_strength": False,
        "doublets": False,
        "markers": False,
        "snapshot_columns": (),
        "species": None,
    }


def test_pipeline_recipe_helper_validation() -> None:
    for value, error_type, message in (
        ("name", TypeError, "columns must be a sequence of column names"),
        (["ok", ""], TypeError, "columns must contain non-empty strings"),
        (["same", "same"], ValueError, "columns must not contain duplicates"),
    ):
        with pytest.raises(error_type, match=f"^{message}$"):
            column_sequence(value, "columns")
    assert column_sequence(["b", "a"], "columns") == ("b", "a")
    for value, error_type, message in (
        (True, TypeError, "Leiden resolution must be a number"),
        ("one", TypeError, "Leiden resolution must be a number"),
        (float("inf"), ValueError, "Leiden resolution must be finite and positive"),
        (0, ValueError, "Leiden resolution must be finite and positive"),
    ):
        with pytest.raises(error_type, match=f"^{message}$"):
            canonical_resolution(value)
    for value, error_type, message in (
        (1, TypeError, "^leiden must be a mapping or bool$"),
        ({"resolution": 1.0}, ValueError, r"^Unknown leiden settings \['resolution'\]"),
        ({"partitions": "one"}, TypeError, "leiden partitions must be a non-empty"),
        ({"partitions": []}, ValueError, "leiden partitions must not be empty"),
        ({"partitions": [1, 1.0]}, ValueError, "contain duplicate resolutions"),
        (
            {"partitions": [0.5], "selected": 1.0},
            ValueError,
            "^leiden selected must be one of the partitions$",
        ),
        (
            {"partitions": [0.5], "selected": "0.5"},
            TypeError,
            "^Leiden resolution must be a number$",
        ),
    ):
        with pytest.raises(error_type, match=message):
            resolve_leiden(value)
    default = tuple((str(value), value) for value in DEFAULT_LEIDEN_PARTITIONS)
    # True and settings without partitions take the default candidates, which
    # a recipe may choose; False turns Leiden off.
    assert resolve_leiden(True) == (default, None, {})
    assert resolve_leiden({}) == (default, None, {})
    assert resolve_leiden({"random_seed": 1}) == (default, None, {"random_seed": 1})
    assert resolve_leiden(True, default_partitions=(1.0,)) == (
        (("1.0", 1.0),),
        None,
        {},
    )
    assert resolve_leiden(False) == ((), None, {})
    # Integer and float spellings of one resolution share one key.
    assert resolve_leiden({"partitions": [2, 0.5], "selected": 2}) == (
        (("2.0", 2.0), ("0.5", 0.5)),
        "2.0",
        {},
    )
    # The run records the candidates and the selected resolution with the
    # other settings, and False when Leiden is off.
    partitions, selected, settings = resolve_leiden(
        {"partitions": [0.5, 1], "selected": 1, "random_seed": 3}
    )
    assert leiden_config(partitions, selected, settings) == {
        "random_seed": 3,
        "partitions": [0.5, 1.0],
        "selected": 1.0,
    }
    assert leiden_config(partitions, None, {}) == {
        "partitions": [0.5, 1.0],
        "selected": None,
    }
    assert leiden_config((), None, {}) is False
    for value, error_type in ((False, TypeError), ("bad", TypeError)):
        with pytest.raises(error_type, match="^value must be a finite number$"):
            finite_real(value, "value")
    with pytest.raises(ValueError, match="^value must be a finite number$"):
        finite_real(float("nan"), "value")
    assert finite_real(3, "value") == 3.0


def test_stage_setting_defaults_require_a_keyword_default() -> None:
    def stage(top_n: int) -> int:
        return top_n

    with pytest.raises(TypeError, match=r"\.stage\(\) has no default for 'top_n'$"):
        keyword_default(stage, "top_n")


def test_pipeline_filtering_contract_errors() -> None:
    store = _RecipeStore()
    default_attrs = default_filter_columns(
        store, "RNA", ("nCounts", "nFeatures", "percentMito", "percentRibo")
    )
    # Preparation wrote counts and features, but no percentages, for this store.
    assert default_attrs == ("RNA_nCounts", "RNA_nFeatures")

    def resolve(value: Any) -> dict[str, Any]:
        return resolve_filtering(store, value, default_attrs=default_attrs)

    invalid = (
        (object(), TypeError, "mapping or bool"),
        ({"method": "other"}, ValueError, "method"),
        ({"attrs": ["missing"]}, KeyError, "not found"),
        ({"attrs": []}, ValueError, "no QC"),
        (
            {"method": "manual", "attrs": ["RNA_nCounts"], "unknown": 1},
            ValueError,
            r"^Unknown filtering settings \['unknown'\]",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0],
                "highs": [2],
                "min_p": 0.05,
            },
            ValueError,
            r"Unknown manual filtering options: \['min_p'\]",
        ),
        (
            {"method": "manual", "attrs": ["RNA_nCounts"]},
            ValueError,
            "requires lows",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0, 1],
                "highs": [2],
            },
            ValueError,
            "align",
        ),
        (
            {
                "method": "manual",
                "attrs": ["RNA_nCounts"],
                "lows": [0],
                "highs": [2],
                "keep_bounds": 1,
            },
            TypeError,
            "keep_bounds",
        ),
        *(
            (
                {
                    "method": "manual",
                    "attrs": ["RNA_nCounts"],
                    "lows": [low],
                    "highs": [high],
                },
                error_type,
                message,
            )
            for low, high, error_type, message in (
                (True, None, TypeError, "lows values"),
                ("bad", None, TypeError, "lows values"),
                (None, float("inf"), ValueError, "finite"),
                (3, 2, ValueError, "cannot exceed"),
            )
        ),
        (
            {"attrs": ["RNA_nCounts"], "unknown": 1},
            ValueError,
            r"^Unknown filtering settings \['unknown'\]",
        ),
        (
            {"attrs": ["RNA_nCounts"], "lows": [0]},
            ValueError,
            r"Unknown automatic filtering options: \['lows'\]",
        ),
        ({"attrs": ["RNA_nCounts"], "min_p": 0.999}, ValueError, "0 < min_p"),
        ({"attrs": ["RNA_nCounts"], "sample_column": ""}, TypeError, "sample_column"),
        (
            {"attrs": ["RNA_nCounts"], "sample_column": "missing"},
            KeyError,
            "Sample column",
        ),
        ({"attrs": ["RNA_nCounts"], "n_mads": 0}, ValueError, "positive"),
        ({"attrs": ["RNA_nCounts"], "min_p": 0.05}, ValueError, "method='mad'"),
        (
            {"method": "gaussian", "sample_column": "sample"},
            ValueError,
            "sample source",
        ),
        ({"method": "gaussian", "n_mads": 4.0}, ValueError, "apply only"),
        (
            {"method": "gaussian", "min_cells_per_sample": 2},
            ValueError,
            "apply only",
        ),
        (
            {"attrs": ["RNA_nCounts"], "min_cells_per_sample": 1},
            ValueError,
            "at least 2",
        ),
        (
            {
                "attrs": ["RNA_nCounts"],
                "sample_column": "sample",
                "min_p": 0.02,
            },
            ValueError,
            "cannot be changed",
        ),
    )
    for value, error_type, message in invalid:
        with pytest.raises(error_type, match=message):
            resolve(value)

    manual = resolve(
        {
            "method": "manual",
            "attrs": ["RNA_nCounts"],
            "lows": [None],
            "highs": [10],
            "keep_bounds": True,
        },
    )
    assert manual["lows"] == [None]
    assert manual["keepBounds"] is True
    automatic = resolve({"method": "auto"})
    assert automatic == resolve({"method": "mad"}) == resolve(True)
    assert automatic["attrs"] == list(default_attrs)
    assert resolve(False) == {"enabled": False}


def test_run_filtering_defaults_to_where_the_cells_come_from() -> None:
    store = _RecipeStore()
    column = CellSource(column="I", selection=None)
    selection = CellSource(column=None, selection=_ref())

    def resolve(
        cells: CellSource, value: object, *, default_on: bool = True
    ) -> dict[str, Any]:
        return resolve_run_filtering(
            store,
            value,
            cells=cells,
            assay="RNA",
            default_attrs=("RNA_nCounts",),
            operation="pipeline.rna",
            default_on=default_on,
        )

    # None filters cells that a column names and takes a given selection or
    # run's cells as already filtered; a recipe may turn the default off.
    assert resolve(column, None)["enabled"] is True
    assert resolve(selection, None) == {"enabled": False}
    assert resolve(column, None, default_on=False) == {"enabled": False}
    # A setting that the call gives applies to any cells.
    assert resolve(selection, True) == resolve(column, True)
    assert resolve(selection, True)["attrs"] == ["RNA_nCounts"]
    assert resolve(column, False) == {"enabled": False}
    with pytest.raises(TypeError, match="^filtering must be a mapping, bool, or None$"):
        resolve(column, "mad")


def test_resolve_rna_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rna_module, "RNAassay", _FakeRNA)
    store = _RecipeStore()
    kwargs = _recipe_kwargs()
    plan = rna_module.resolve_rna(store, **kwargs)
    assert (plan.recipe, plan.assay, plan.label) == ("rna", "RNA", None)
    assert plan.config["cells"] == {"column": "I"}
    assert plan.config["stages"]["hvg"] == {"top_n": 10}
    assert plan.config["stages"]["pca"] == {"dims": 5}

    failed = PipelineRun(
        store,
        _terminal_run(
            status="failed",
            error=PipelineErrorRecord("ValueError", "bad"),
        ),
    )
    other_kind = ArtifactRef(
        scope="assay", kind="connectivity_map", artifact_id="b" * 64, assay="RNA"
    )
    cases: list[tuple[dict[str, Any], type[BaseException], str]] = [
        ({"label": ""}, TypeError, "label"),
        ({"cells": ""}, TypeError, "non-empty boolean cell column"),
        ({"cells": "missing"}, KeyError, "selection column"),
        ({"cells": "numeric"}, TypeError, "boolean"),
        ({"cells": 1}, TypeError, "ArtifactRef, or a completed PipelineRun"),
        ({"cells": other_kind}, ValueError, "datastore cell-selection artifact"),
        ({"cells": failed}, ValueError, f"run {failed.run_id} is 'failed'"),
        ({"umap": 1}, TypeError, "umap"),
        ({"leiden": False, "doublets": True}, ValueError, "require"),
        ({"leiden": False, "membership_strength": True}, ValueError, "requires"),
        ({"markers": 1}, TypeError, "markers must be a boolean"),
        ({"snapshot_columns": ["I"]}, ValueError, "reserved"),
        ({"snapshot_columns": ["missing"]}, KeyError, "Snapshot columns"),
        ({"snapshot_columns": "sample"}, TypeError, "sequence of column names"),
        ({"harmony": {"batch_columns": []}}, ValueError, "must not be empty"),
        ({"harmony": {"batch_columns": ["missing"]}}, KeyError, "Harmony columns"),
        ({"hvg": {"top_n": 0}}, ValueError, "top_n must be at least 1"),
        ({"hvg": {"top_n": 2.0}}, TypeError, "top_n must be an integer"),
        ({"hvg": {"batch": "missing"}}, KeyError, "HVG batch column"),
        ({"pca": {"dims": -1}}, ValueError, "pca dims must be at least 0"),
        ({"species": "human"}, ValueError, "species must be one of"),
        ({"hashing": True}, ValueError, "needs an assay of type 'HTO'"),
        ({"cell_calling": {"sample_column": "missing"}}, KeyError, "Capture column"),
        ({"doublets": {"sample_column": "missing"}}, KeyError, "Doublet sample"),
    ]
    for changes, error_type, message in cases:
        with pytest.raises(error_type, match=message):
            rna_module.resolve_rna(store, **{**kwargs, **changes})

    # Without an assay, the recipe takes the default assay when it is an RNA
    # assay, else the store's only RNA assay.
    store._defaultAssay = None
    assert rna_module.resolve_rna(store, **{**kwargs, "assay": None}).assay == "RNA"
    store.assay_names = ["RNA", "RNA_2"]
    with pytest.raises(ValueError, match=r"several RNA assays, \['RNA', 'RNA_2'\]"):
        rna_module.resolve_rna(store, **{**kwargs, "assay": None})
    store.assay_names = []
    with pytest.raises(TypeError, match="^The store has no RNA assay for the rna"):
        rna_module.resolve_rna(store, **{**kwargs, "assay": None})
    store._defaultAssay = "RNA"
    with pytest.raises(TypeError, match="^assay must be a non-empty string or None$"):
        rna_module.resolve_rna(store, **{**kwargs, "assay": ""})
    store.assay = object()
    with pytest.raises(
        TypeError, match="^The rna recipe analyzes RNA assays; assay 'RNA' is of type"
    ):
        rna_module.resolve_rna(store, **kwargs)


@contextmanager
def _sample_rss() -> Any:
    measurement = ProcessTreeRssMeasurement(
        baseline_bytes=10,
        peak_bytes=20,
        incremental_peak_bytes=10,
        sample_interval_seconds=0.1,
        sample_count=1,
        sampling_error_count=0,
        unavailable_reason=None,
    )
    yield lambda: measurement


def test_run_ledger_records_and_interruption_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = RunLedger(object(), "run", _events())
    with pytest.raises(
        RuntimeError, match=r"not observed by artifact planning: \['out'\]"
    ):
        ledger._records((("out", _ref()),), ())
    plans = (
        ArtifactPlanReceipt("op", _ref(), "reused"),
        ArtifactPlanReceipt("op", _ref(), "created"),
    )
    # An output counts as reused only when every plan for it reused it.
    assert ledger._records((("out", _ref()),), plans)[0].reused is False
    assert ledger._records((("out", _ref()),), plans[:1])[0].reused is True
    assert [
        (plan.operation, plan.ref, plan.disposition) for plan in ledger._plans(plans)
    ] == [("op", _ref(), "reused"), ("op", _ref(), "created")]
    assert ledger_module.interruption_record(asyncio.CancelledError()).kind == (
        "asyncio_cancelled"
    )
    with pytest.raises(TypeError, match="handled"):
        ledger.interrupt_pending(ValueError("bad"), "stage")
    with pytest.raises(TypeError, match="handled"):
        ledger._finish_interrupted(
            stage="stage",
            ordinal=0,
            error=ValueError("bad"),
            metrics=_metrics(),
        )

    events: list[PipelineEvent] = []
    ledger = RunLedger(object(), "run", _events(events.append))
    monkeypatch.setattr(
        ledger_module,
        "load_pipeline_run_record",
        lambda *_args: SimpleNamespace(complete=False),
    )
    interrupted: list[Any] = []
    monkeypatch.setattr(
        ledger_module,
        "interrupt_pipeline_run_record",
        lambda *args, **kwargs: interrupted.append((args, kwargs)),
    )
    ledger.interrupt_pending(KeyboardInterrupt(), "stage")
    ((_args, kwargs),) = interrupted
    assert kwargs["run_id"] == "run"
    assert kwargs["interruption"].kind == "keyboard_interrupt"
    assert [(event.kind, event.stage, event.ordinal) for event in events] == [
        ("run_interrupted", "stage", 0)
    ]
    assert ledger._fallback_metrics(0).sample_count >= 1


def test_run_ledger_skip_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_module, "sample_process_tree_rss", _sample_rss)
    monkeypatch.setattr(
        ledger_module,
        "load_pipeline_run_record",
        lambda *_args: SimpleNamespace(complete=False),
    )
    monkeypatch.setattr(
        ledger_module, "start_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(ledger_module, "fail_pipeline_run_record", lambda *a, **k: None)

    monkeypatch.setattr(ledger_module, "shutdown_checkpoint", lambda: None)
    recorded: list[int] = []
    monkeypatch.setattr(
        RunLedger,
        "_record_interrupted_stage",
        lambda self, **kwargs: recorded.append(kwargs["ordinal"]),
    )
    pending: list[str | None] = []
    monkeypatch.setattr(
        RunLedger,
        "interrupt_pending",
        lambda self, error, stage: pending.append(stage),
    )

    # A skip emits no stage events, so an interruption while its record is
    # written marks that record and ends the run between stages.
    monkeypatch.setattr(
        ledger_module,
        "finish_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    with pytest.raises(KeyboardInterrupt):
        RunLedger(object(), "run", _events()).skip("stage")
    assert (recorded, pending) == ([0], [None])

    recorded.clear()
    pending.clear()
    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
    )
    with pytest.raises(KeyboardInterrupt):
        RunLedger(object(), "run", _events()).skip("pending")
    assert (recorded, pending) == ([], [None])

    class Fatal(BaseException):
        pass

    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(Fatal()),
    )
    with pytest.raises(Fatal):
        RunLedger(object(), "run", _events()).skip("fatal")

    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")),
    )
    failed_runs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        ledger_module,
        "fail_pipeline_run_record",
        lambda _root, **kwargs: failed_runs.append(kwargs),
    )
    with pytest.raises(
        PipelineExecutionError, match="failed during stage 'failed': bad$"
    ) as caught:
        RunLedger(object(), "run", _events()).skip("failed")
    assert isinstance(caught.value.__cause__, ValueError)
    # A stage that never started still ends its run as failed.
    assert [(kwargs["run_id"], str(kwargs["error"])) for kwargs in failed_runs] == [
        ("run", "bad")
    ]


def test_run_ledger_run_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_module, "sample_process_tree_rss", _sample_rss)
    monkeypatch.setattr(
        ledger_module,
        "load_pipeline_run_record",
        lambda *_args: SimpleNamespace(complete=False),
    )
    monkeypatch.setattr(ledger_module, "shutdown_checkpoint", lambda: None)
    monkeypatch.setattr(ledger_module, "fail_pipeline_run_record", lambda *a, **k: None)
    monkeypatch.setattr(
        ledger_module,
        "start_pipeline_stage_record",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("start")),
    )
    with pytest.raises(PipelineExecutionError, match="start"):
        RunLedger(object(), "run", _events()).run("stage", lambda: ())

    monkeypatch.setattr(
        ledger_module, "start_pipeline_stage_record", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    failures: list[BaseException] = []
    monkeypatch.setattr(
        RunLedger,
        "_finish_failed",
        lambda self, **kwargs: failures.append(kwargs["error"]),
    )
    with pytest.raises(PipelineExecutionError, match="not observed"):
        RunLedger(object(), "run", _events()).run(
            "unplanned",
            lambda: (("out", _ref()),),
        )
    assert isinstance(failures[-1], RuntimeError)

    def fail_finish(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("finish")

    monkeypatch.setattr(ledger_module, "finish_pipeline_stage_record", fail_finish)
    with pytest.raises(PipelineExecutionError, match="finish"):
        RunLedger(object(), "run", _events()).run("finish", lambda: ())

    monkeypatch.setattr(
        ledger_module, "finish_pipeline_stage_record", lambda *a, **k: None
    )
    interrupted: list[str] = []
    monkeypatch.setattr(
        RunLedger,
        "_finish_interrupted",
        lambda self, **kwargs: interrupted.append(kwargs["stage"]),
    )
    with pytest.raises(asyncio.CancelledError):
        RunLedger(object(), "run", _events()).run(
            "cancelled",
            lambda: (_ for _ in ()).throw(asyncio.CancelledError()),
        )
    assert interrupted == ["cancelled"]

    class Fatal(BaseException):
        pass

    with pytest.raises(Fatal):
        RunLedger(object(), "run", _events()).run(
            "fatal",
            lambda: (_ for _ in ()).throw(Fatal()),
        )


def test_pipeline_record_storage_rejects_torn_and_invalid_queries() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    record = run_storage.create_pipeline_run_record(
        root,
        recipe="rna",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("one",),
        cytearc_version="1.0",
        run_id="1" * 64,
        started_at_ns=10,
    )
    with pytest.raises(
        FileExistsError, match=f"^Pipeline run already exists: {record.run_id}$"
    ):
        run_storage.create_pipeline_run_record(
            root,
            recipe="rna",
            requested_label=None,
            assay="RNA",
            config={},
            stage_order=("one",),
            cytearc_version="1.0",
            run_id=record.run_id,
            started_at_ns=10,
        )
    with pytest.raises(ValueError, match="complete=True"):
        run_storage._write_terminal_attrs(root, {"complete": False})
    with pytest.raises(ValueError, match="positive integer"):
        run_storage.list_pipeline_run_records(root, limit=0)
    with pytest.raises(TypeError, match="status"):
        run_storage.list_pipeline_run_records(root, status=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        run_storage.list_pipeline_run_records(root, status=[])
    with pytest.raises(ValueError, match="Unknown"):
        run_storage.list_pipeline_run_records(root, status="other")
    with pytest.raises(ValueError, match="exactly one"):
        run_storage.open_pipeline_run_record(root)
    with pytest.raises(ValueError, match="exactly one"):
        run_storage.open_pipeline_run_record(root, run_id=record.run_id, label="x")
    with pytest.raises(KeyError, match="No completed"):
        run_storage.open_pipeline_run_record(root, label="missing")

    stages = root[run_storage.pipeline_run_path(record.run_id) + "/stages"]
    stages.create_group("bad")
    with pytest.raises(ValueError, match="child name"):
        run_storage.load_pipeline_stage_records(root, record.run_id)


def test_end_pipeline_run_records_once_and_keeps_terminal_runs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")

    def new_run() -> str:
        return run_storage.create_pipeline_run_record(
            root,
            recipe="rna",
            requested_label=None,
            assay="RNA",
            config={},
            stage_order=("one",),
            cytearc_version="1.0",
        ).run_id

    failed = new_run()
    assert ledger_module.end_pipeline_run(root, failed, ValueError("bad")) is True
    assert ledger_module.end_pipeline_run(root, failed, KeyboardInterrupt()) is False
    assert run_storage.load_pipeline_run_record(root, failed).status == "failed"

    interrupted = new_run()
    assert ledger_module.end_pipeline_run(root, interrupted, KeyboardInterrupt())
    record = run_storage.load_pipeline_run_record(root, interrupted)
    assert record.status == "interrupted"
    assert record.interruption is not None
    assert record.interruption.kind == "keyboard_interrupt"
