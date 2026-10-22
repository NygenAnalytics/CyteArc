"""Source binding, correction eligibility and QC evidence scale to large cohorts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cytearc.agent import evidence
from cytearc.agent.models import (
    AnalysisConfig,
    AnalysisInputError,
    ContextDecision,
    RuntimeConfig,
)
from tests.test_agent_evidence import study
from tests.test_agent_science_edges import Metadata
from tests.test_agent_science_edges import metadata_source as metadata_source


def _store(cells: pd.DataFrame, features: pd.DataFrame | None = None) -> Any:
    """The public metadata interface of a prepared store, without counts."""
    features = (
        pd.DataFrame({"ids": ["g1", "g2"], "names": ["A", "B"], "I": [True, True]})
        if features is None
        else features
    )
    assay = SimpleNamespace(feats=Metadata(features))
    descriptor = SimpleNamespace(
        name="RNA", dataset_fingerprint="finalized-counts", total_features=len(features)
    )
    return SimpleNamespace(
        cells=Metadata(cells),
        get_assay=lambda name: assay,
        summary=lambda: SimpleNamespace(assays=[descriptor], total_cells=len(cells)),
    )


def _reference_fingerprint() -> str:
    """The documented version 2 byte stream, written out independently."""

    def segment(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "little") + payload)

    def header(value: Any) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode()

    digest = hashlib.sha256()
    segment(header([2, "RNA", "finalized-counts", 3, 2]))
    # flag: booleans hash as one byte each behind an all-present mask.
    segment(header(["cells", "flag", "|b1", 3]))
    segment(b"\x00")
    segment(b"|u1")
    segment(bytes([1, 0, 1]))
    # label: row 1 is missing (mask bit 1), then length-prefixed UTF-8 text.
    segment(header(["cells", "label", "object", 3]))
    segment(bytes([0b010]))
    segment(b"string")
    segment(np.array([1, 2], dtype="<i8").tobytes())
    segment("aβc".encode())
    # score: the missing row hashes as zero.
    segment(header(["cells", "score", "<f8", 3]))
    segment(bytes([0b010]))
    segment(b"<f8")
    segment(np.array([1.5, 0.0, -2.0], dtype="<f8").tobytes())
    for column, values in (("ids", ["g1", "g2"]), ("names", ["A", "B"])):
        segment(header(["features", column, "object", 2]))
        segment(b"\x00")
        segment(b"string")
        segment(np.array([len(value) for value in values], dtype="<i8").tobytes())
        segment("".join(values).encode())
    segment(header(["features", "I", "|b1", 2]))
    segment(b"\x00")
    segment(b"|u1")
    segment(bytes([1, 1]))
    return digest.hexdigest()


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flag": [True, False, True],
            "label": pd.Series(["a", None, "βc"], dtype=object),
            "score": [1.5, np.nan, -2.0],
        }
    )


def test_fingerprint_hashes_whole_columns_in_the_documented_format() -> None:
    features = pd.DataFrame(
        {
            "ids": pd.Series(["g1", "g2"], dtype=object),
            "names": pd.Series(["A", "B"], dtype=object),
            "I": [True, True],
        }
    )
    store = _store(_cells(), features)
    expected = _reference_fingerprint()
    assert evidence._fingerprint(store, "RNA", ["flag", "label", "score"]) == expected
    # A different NaN payload behind the missing mask changes nothing.
    other = _cells()
    payload = np.array([0x7FF8000000000123], dtype="<u8").view("<f8")[0]
    assert np.isnan(payload)
    other.loc[1, "score"] = payload
    columns = ["flag", "label", "score"]
    assert evidence._fingerprint(_store(other, features), "RNA", columns) == expected


def test_fingerprint_reads_each_column_once_without_per_value_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 10_000
    cells = pd.DataFrame(
        {
            "count": np.arange(rows, dtype=np.int64),
            "fraction": np.linspace(0, 1, rows),
            "selected": np.arange(rows) % 2 == 0,
            "sample": [f"sample-{index % 7}" for index in range(rows)],
        }
    )
    calls = 0
    dumps = json.dumps

    def counted(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return dumps(*args, **kwargs)

    monkeypatch.setattr(evidence.json, "dumps", counted)
    evidence._fingerprint(_store(cells), "RNA", list(cells.columns))
    # One header per column and one for the dataset; never one per value.
    assert calls == 1 + len(cells.columns) + 3


@pytest.mark.parametrize(
    ("column", "values", "changed"),
    [
        ("number", [1.0, 2.0, 3.0], [1.0, 2.0, 3.5]),
        ("number", [1.0, 2.0, 3.0], [2.0, 1.0, 3.0]),
        ("number", [1.0, 2.0, 3.0], [1.0, np.nan, 3.0]),
        ("number", [0.0, 2.0, 3.0], [np.nan, 2.0, 3.0]),
        ("integer", [1, 2, 3], [1, 2, 4]),
        ("flag", [True, False, True], [True, True, True]),
        (
            "nullable",
            pd.array([True, False, True], dtype="boolean"),
            pd.array([True, None, True], dtype="boolean"),
        ),
        ("text", ["a", "b", "c"], ["a", "b", "C"]),
        ("text", ["ab", "c", ""], ["a", "bc", ""]),
        ("text", ["a", "", "c"], ["a", None, "c"]),
        ("mixed", [1, "1", 2.5], ["1", "1", 2.5]),
        ("bytes", [b"a", b"\xff", b"c"], [b"a", b"\xfe", b"c"]),
        ("category", pd.Categorical(["x", "y", "x"]), pd.Categorical(["x", "y", "y"])),
        (
            "time",
            pd.to_datetime(["2024-01-01", "2024-01-02", None]),
            pd.to_datetime(["2024-01-01", "2024-01-03", None]),
        ),
    ],
)
def test_fingerprint_is_deterministic_and_detects_each_value_change(
    column: str, values: Any, changed: Any
) -> None:
    def fingerprint(data: Any) -> str:
        cells = pd.DataFrame(
            {
                column: pd.Series(
                    data, dtype=object if column in {"mixed", "bytes"} else None
                )
            }
        )
        return evidence._fingerprint(_store(cells), "RNA", [column])

    assert fingerprint(values) == fingerprint(values)
    assert fingerprint(values) != fingerprint(changed)


def test_fingerprint_does_not_depend_on_storage_byte_order() -> None:
    little = pd.DataFrame({"score": np.array([1.5, -2.0, 3.25], dtype="<f8")})
    big = pd.DataFrame({"score": np.array([1.5, -2.0, 3.25], dtype=">f8")})
    assert evidence._fingerprint(_store(little), "RNA", ["score"]) == (
        evidence._fingerprint(_store(big), "RNA", ["score"])
    )


@pytest.mark.parametrize("version", [None, 1])
def test_earlier_fingerprint_formats_ask_for_a_new_analysis(
    tmp_path: Path, version: int | None
) -> None:
    prepared = {"fingerprint": "f" * 64, "fingerprintColumns": ["I"], "assay": "RNA"}
    if version is not None:
        prepared["fingerprintVersion"] = version
    with pytest.raises(AnalysisInputError) as raised:
        evidence.verify_source(
            tmp_path, prepared, study(), AnalysisConfig(), RuntimeConfig()
        )
    assert "start a new analysis" in str(raised.value)
    assert "remain readable" in str(raised.value)


def test_model_inferred_protected_role_never_unlocks_correction(
    metadata_source: tuple[Path, Any],
) -> None:
    path, store = metadata_source
    authorization = "Libraries were prepared independently of the conditions."
    supplied = study(
        technicalBatchColumns=["batch"], batchCorrectionEvidence=authorization
    )
    inspected = evidence.inspect_source(
        path, supplied, AnalysisConfig(), RuntimeConfig()
    )
    missing = (
        "Correction has no explicitly protected biological metadata for validation."
    )
    assert inspected["correctionEligible"] is False
    assert missing in inspected["limitations"]
    prepared = evidence.prepare_context(
        inspected,
        ContextDecision(
            rationale="Condition looks biological.",
            columnRoles={"condition": "protected"},
            evidenceIds=["column:condition"],
        ),
        supplied,
        AnalysisConfig(),
    )
    assert prepared["correctionEligible"] is False
    assert missing in prepared["limitations"]
    # The inferred column still joins the preservation checks and snapshot.
    assert prepared["protectedColumns"] == ["condition"]
    assert "condition" in prepared["snapshotColumns"]
    assert prepared["contextEvidence"]["study"]["protectedColumns"] == []
    role = next(
        row for row in prepared["resolvedRoles"] if row["column"] == "condition"
    )
    assert (role["source"], role["authority"]) == ("inferred", "diagnosticOnly")
    declared = study(
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
        batchCorrectionEvidence=authorization,
    )
    assert evidence.inspect_source(path, declared, AnalysisConfig(), RuntimeConfig())[
        "correctionEligible"
    ]


def test_model_inferred_protected_role_can_still_block_correction(
    metadata_source: tuple[Path, Any],
) -> None:
    path, store = metadata_source
    store.cells.frame["site"] = store.cells.frame["batch"]
    supplied = study(
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
        batchCorrectionEvidence="Libraries were prepared independently of biology.",
    )
    inspected = evidence.inspect_source(
        path, supplied, AnalysisConfig(), RuntimeConfig()
    )
    assert inspected["correctionEligible"] is True
    prepared = evidence.prepare_context(
        inspected,
        ContextDecision(
            rationale="Site may carry biology.",
            columnRoles={"site": "protected"},
            evidenceIds=["column:site"],
        ),
        supplied,
        AnalysisConfig(),
    )
    assert prepared["correctionEligible"] is False
    assert (
        "Technical 'batch' is not fully crossed with protected 'site'."
        in (prepared["limitations"])
    )


def test_design_requires_declared_protection_but_checks_inferred_columns() -> None:
    frame = pd.DataFrame({"batch": ["a", "b"] * 4, "condition": ["x"] * 4 + ["y"] * 4})
    eligible, design, reasons = evidence._design(
        SimpleNamespace(cells=Metadata(frame)),
        np.ones(8, dtype=bool),
        ["batch"],
        [],
        "Independent library preparation",
        inferred=["condition"],
    )
    assert eligible is False
    assert reasons == [
        "Correction has no explicitly protected biological metadata for validation."
    ]
    assert design["crossings"] == [
        {"technical": "batch", "protected": "condition", "fullyCrossed": True}
    ]


def _reference_groups(
    labels: pd.Series, active: np.ndarray, retained: np.ndarray
) -> dict[str, Any]:
    """Per-level retention, computed level by level as the earlier code did."""
    text = labels.astype(str)
    missing = (labels.isna() | text.str.strip().eq("")).to_numpy()
    counts = text.loc[active & ~missing].value_counts()
    rows = []
    for value, count in counts.head(64).items():
        mask = active & ~missing & text.eq(value).to_numpy()
        kept = int((mask & retained).sum())
        rows.append(
            {
                "value": str(value),
                "inputCells": int(count),
                "retainedCells": kept,
                "removedCells": int(count) - kept,
            }
        )
    return {
        "levels": rows,
        "missingInputCells": int((active & missing).sum()),
        "missingRetainedCells": int((retained & missing).sum()),
        "omittedLevels": max(0, len(counts) - 64),
    }


def test_qc_projections_count_group_levels_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytearc.quality_control.filtering import filter_cell_metrics

    rng = np.random.default_rng(14)
    rows = 3000
    labels = pd.Series([f"donor-{value}" for value in rng.integers(0, 70, rows)])
    labels[rng.choice(rows, 40, replace=False)] = None
    labels[rng.choice(rows, 20, replace=False)] = "  "
    counts = np.exp(rng.normal(8, 0.6, rows))
    counts[:5] = 1.0
    frame = pd.DataFrame({"donor": labels, "RNA_nCounts": counts})
    store = SimpleNamespace(cells=Metadata(frame))
    active = rng.random(rows) > 0.1
    config = AnalysisConfig(qcPolicy="manual", qcBounds={"RNA_nCounts": (500.0, None)})
    flags = evidence._qc_flags(store, active, "RNA")
    roles = [{"column": "donor", "role": "sample"}]
    conversions = 0
    astype = pd.Series.astype

    def counted(series: pd.Series, dtype: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal conversions
        conversions += dtype is str
        return astype(series, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "astype", counted)
    projections = evidence._qc_projections(store, active, "RNA", config, flags, roles)
    monkeypatch.undo()
    # The group column is converted to text once, not once per level and policy.
    assert conversions == 1
    for projection in projections:
        bounds = projection["bounds"]
        retained = (
            active
            if projection["policy"] == "retain"
            else filter_cell_metrics(
                {"RNA_nCounts": counts},
                {"RNA_nCounts": ~np.isfinite(counts)},
                active,
                method="manual",
                lows=[bounds["RNA_nCounts"][0]],
                highs=[bounds["RNA_nCounts"][1]],
                keep_bounds=True,
            ).retained
        )
        assert projection["byGroup"]["donor"] == _reference_groups(
            labels, active, retained
        )
    assert projections[0]["byGroup"]["donor"]["omittedLevels"] == 6
    assert projections[2]["removedCells"] > 0


def test_long_metadata_names_are_fingerprinted_but_not_offered_to_models(
    metadata_source: tuple[Path, Any],
) -> None:
    path, store = metadata_source
    long_name = "x" * 129
    store.cells.frame[long_name] = ["a", "b"] * 4
    store.cells.frame["y" * 128] = ["c", "d"] * 4
    prepared = evidence.inspect_source(path, study(), AnalysisConfig(), RuntimeConfig())
    assert long_name not in prepared["contextEvidence"]["columns"]
    assert long_name not in prepared["diagnosticColumns"]
    assert "y" * 128 in prepared["contextEvidence"]["columns"]
    assert long_name in prepared["fingerprintColumns"]
    assert (
        "Decision evidence omits 1 metadata columns whose names exceed 128 characters."
        in prepared["limitations"]
    )
