"""Read-only source checks and compact evidence for the RNA procedure."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import (
    AnalysisConfig,
    AnalysisInputError,
    ContextDecision,
    NeedsInput,
    RuntimeConfig,
    Study,
)
from .diagnostics import (
    design_diagnostics,
    family_audit,
    group_columns,
    resolved_roles,
)


_LABEL_PATTERN = re.compile(
    r"annotation|cell[._ ]?type|cell[._ ]?label|cluster|leiden|louvain|singler|"
    r"predicted[._]id|majority_voting|azimuth|snn_res|ann_level|broad_type|"
    r"subtype|lineage|^labels$|scanvi",
    re.I,
)
_MITO_PATTERN = r"(?i:^mt-)"
_MAX_COLUMNS = 48
# Longer metadata names are omitted from decision evidence; each offered name
# is repeated in several prompt fields that cannot be trimmed.
MAX_COLUMN_NAME = 128
# Version 2 hashes whole columns; version 1 hashed one JSON value at a time.
# Saved analyses with another version cannot verify their source.
FINGERPRINT_VERSION = 2
_NUMERICAL_ARTIFACT_KINDS = frozenset(
    {
        "feature_summary",
        "feature_selection",
        "normalized",
        "feature_scaling",
        "reduction",
        "batch_correction",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "embedding_initialization",
        "embedding",
        "cluster_labels",
        "cluster_cut",
        "cluster_selection",
        "marker_table",
        "doublet_score",
    }
)
_NONNUMERICAL_ARTIFACT_OPERATIONS = frozenset(
    {
        ("feature_selection", "create_all_features"),
        ("feature_selection", "set_feature_selection"),
        ("embedding", "import_dimreduc"),
        ("cluster_labels", "import_cluster_labels"),
        ("cluster_labels", "import_active_identity"),
        ("cluster_labels", "snapshot_cluster_labels"),
    }
)


def is_held_out_column(name: str, study: Study) -> bool:
    """Recognize annotation aliases and caller-declared held-out metadata."""
    return name in study.excludedColumns or bool(_LABEL_PATTERN.search(name))


def _local_source(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if "://" in str(source) or not path.is_dir():
        raise AnalysisInputError(
            "The RNA procedure requires an existing local CyteArc directory"
        )
    return path.resolve()


def open_store(
    source: str | Path,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
    writable: bool = False,
) -> Any:
    """Open an already prepared store without changing its live selection."""
    from cytearc import DataStore

    path = _local_source(source)
    options: dict[str, Any] = {
        "workspace": config.workspace,
        "nthreads": runtime.nthreads,
        "mem_budget": runtime.memBudget,
        "min_features_per_cell": -1,
    }
    store = DataStore(str(path), zarr_mode="r", **options)
    if not writable:
        return store
    # Writable DataStore initialization filters its *default* assay. Validate
    # that its existing active rows all survive the disabled lower threshold.
    summary = store.summary()
    frame = store.cells.to_pandas_dataframe(["I", f"{summary.default_assay}_nFeatures"])
    active = frame["I"].to_numpy(dtype=bool)
    counts = frame.iloc[:, 1].to_numpy(dtype=float)
    if not np.isfinite(counts[active]).all() or (counts[active] < 0).any():
        raise AnalysisInputError(
            "Active source cells have invalid feature counts; repair the source first"
        )
    return DataStore(str(path), zarr_mode="r+", **options)


def require_clean_analysis(store: Any, assay: str) -> None:
    """Require a source without reusable numerical results for the RNA recipe.

    The public listing includes artifacts inherited from a mounted source.
    Incomplete results cannot be reused and do not block a new analysis.
    Imported labels/embeddings and explicit feature selections are inputs, not
    computations of this recipe. Metadata snapshots and other assays also
    remain allowed. Call only before the run's first numerical invocation;
    later verification must allow that run's own completed artifacts.
    """
    for ref in store.artifacts.list(from_assay=assay, complete_only=True):
        if ref.kind not in _NUMERICAL_ARTIFACT_KINDS:
            continue
        operation = store.artifacts.inspect(ref).operation
        if (ref.kind, operation) in _NONNUMERICAL_ARTIFACT_OPERATIONS:
            continue
        raise AnalysisInputError(
            f"Prior complete numerical artifacts for RNA assay {assay!r} are "
            f"unsupported ({ref.kind}, operation {operation!r}). Prepare a clean "
            "store without prior numerical analysis. Mounting an analyzed "
            "source also inherits its artifacts and does not make it clean."
        )


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _segment(digest: Any, payload: bytes) -> None:
    """Hash one length-prefixed segment, so adjacent segments cannot shift."""
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _header(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def _dtype_name(dtype: Any) -> str:
    """Name a stored dtype the same way on little- and big-endian hosts."""
    if isinstance(dtype, np.dtype) and dtype.kind in "biufcmMU":
        return dtype.newbyteorder("<").str
    return str(dtype)


def _column_digest(digest: Any, series: pd.Series) -> None:
    """Hash one whole metadata column: its missing mask, then its values.

    Booleans, numbers and dates hash their little-endian buffers with that
    dtype, and missing rows hash as zero behind the mask. Other values hash as
    length-prefixed UTF-8 text, tagged with their Python types when a column
    mixes types, so ``1`` and ``"1"`` stay distinct. The bytes do not depend
    on the platform or on how pandas labels an equivalent text dtype.
    """
    missing = series.isna().to_numpy(dtype=bool)
    _segment(digest, np.packbits(missing, bitorder="little").tobytes())
    dtype = series.dtype
    if isinstance(dtype, pd.BooleanDtype) or (
        isinstance(dtype, np.dtype) and dtype.kind == "b"
    ):
        values = series.to_numpy(dtype=bool, na_value=False).astype("|u1")
    elif isinstance(dtype, np.dtype) and dtype.kind in "iufcmM":
        values = series.to_numpy()
        if dtype.kind in "mM":
            values = values.view(np.int64)
        values = np.where(missing, 0, values).astype(values.dtype.newbyteorder("<"))
    else:
        present = series.to_numpy(dtype=object)[~missing]
        kind = pd.api.types.infer_dtype(present, skipna=False)
        _segment(digest, kind.encode())
        if kind == "string":
            texts = present.tolist()
        else:
            if kind.startswith("mixed"):
                types = "\0".join(type(item).__name__ for item in present)
                _segment(digest, types.encode())
            texts = [
                item.decode("utf-8", "surrogateescape")
                if isinstance(item, bytes)
                else str(item)
                for item in present
            ]
        lengths = np.fromiter(map(len, texts), dtype="<i8", count=len(texts))
        _segment(digest, lengths.tobytes())
        _segment(digest, "".join(texts).encode("utf-8", "surrogatepass"))
        return
    _segment(digest, values.dtype.str.encode())
    _segment(digest, np.ascontiguousarray(values).tobytes())


def _fingerprint(store: Any, assay: str, columns: list[str]) -> str:
    """Bind finalized core count identity, ordered axes and nullable metadata.

    Each metadata column is hashed whole; the count matrix is never read for
    run bookkeeping.
    """
    summary = store.summary()
    descriptor = next(item for item in summary.assays if item.name == assay)
    if not descriptor.dataset_fingerprint:
        raise AnalysisInputError("The RNA assay has no finalized dataset fingerprint")
    digest = hashlib.sha256()
    _segment(
        digest,
        _header(
            [
                FINGERPRINT_VERSION,
                assay,
                descriptor.dataset_fingerprint,
                summary.total_cells,
                descriptor.total_features,
            ]
        ),
    )
    for axis, table, names in (
        ("cells", store.cells, columns),
        ("features", store.get_assay(assay).feats, ["ids", "names", "I"]),
    ):
        for column in names:
            series = table.to_pandas_dataframe([column])[column]
            _segment(
                digest,
                _header(
                    [axis, column, _dtype_name(table.get_dtype(column)), len(series)]
                ),
            )
            _column_digest(digest, series)
    return digest.hexdigest()


def _profile(series: pd.Series, name: str) -> dict[str, Any]:
    present = series.dropna()
    counts = present.map(lambda value: str(_json_scalar(value))).value_counts()
    result: dict[str, Any] = {
        "column": name,
        "evidenceId": f"column:{name}",
        "missing": int(series.isna().sum()),
        "unique": int(len(counts)),
        "kind": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
        "levels": [
            {"value": str(value)[:120], "count": int(count)}
            for value, count in counts.head(8).items()
        ],
    }
    if pd.api.types.is_numeric_dtype(series) and len(present):
        values = present.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            result["range"] = [
                float(values.min()),
                float(np.median(values)),
                float(values.max()),
            ]
    return result


def _qc_flags(store: Any, active: np.ndarray, assay: str) -> dict[str, Any]:
    """Record global outlier counts without turning flags into exclusions.

    Counts and bounds are saved, not the flagged rows: those follow from the
    recorded bounds and the verified source metadata.
    """
    summaries: dict[str, Any] = {}
    for suffix in ("nCounts", "nFeatures", "percentMito"):
        column = f"{assay}_{suffix}"
        if column not in store.cells.columns:
            continue
        series = store.cells.to_pandas_dataframe([column])[column]
        values = series.to_numpy(dtype=float, na_value=np.nan)
        valid = active & np.isfinite(values) & (values >= 0)
        if not valid.any():
            summaries[column] = {
                "missing": int(active.sum()),
                "low": None,
                "high": None,
                "lowFlags": 0,
                "highFlags": 0,
            }
            continue
        work = np.log1p(values[valid]) if suffix != "percentMito" else values[valid]
        median = float(np.median(work))
        deviation = float(1.4826 * np.median(np.abs(work - median)))
        low = high = None
        if deviation > 0:
            low, high = median - 5 * deviation, median + 5 * deviation
            if suffix != "percentMito":
                low, high = max(0.0, float(np.expm1(low))), float(np.expm1(high))
            else:
                low, high = None, min(100.0, high)
        summaries[column] = {
            "missing": int((active & ~valid).sum()),
            "low": low,
            "high": high,
            "lowFlags": int((valid & (values < low)).sum()) if low is not None else 0,
            "highFlags": int((valid & (values > high)).sum())
            if high is not None
            else 0,
            "zeroMad": deviation == 0,
        }
    return summaries


def _require_measured_cell_key(store: Any, assay: str, cell_key: str) -> None:
    """Ask for a cell key whose cells the RNA assay measured.

    A merged store marks the cells that an assay measured in its membership
    column ``<assay>_I``; the other cells hold zero counts, which are no
    measurement. Every stage, the QC flags included, reads the assay over the
    cells of ``cell_key``, and the pipeline refuses unmeasured cells, so
    inspection asks for a cell key before any model decision.
    """
    from cytearc.metadata.membership import (
        UnmeasuredCellsError,
        require_measured_cells,
    )

    try:
        require_measured_cells(
            store.cells,
            assay,
            cell_key,
            operation="pipeline.run",
            remedy="cell_key",
        )
    except UnmeasuredCellsError as error:
        name = f"{assay}_measured"
        raise NeedsInput(
            f"RNA assay {assay!r} did not measure {error.unmeasured} of the "
            f"{error.selected} cells of cellKey {cell_key!r}: its membership "
            f"column {error.column!r} is False for them, and their zero counts "
            "are no measurement. Supply as cellKey a boolean cell column that "
            f"is True only for the cells of {cell_key!r} that it measured, such "
            f"as {name!r} after `ds.cells.insert({name!r}, "
            f"ds.cells.fetch_all({cell_key!r}) & "
            f"ds.cells.fetch_all({error.column!r}))` on a writable DataStore.",
            field="cellKey",
        ) from error


def _filtering(
    store: Any, study: Study, config: AnalysisConfig, assay: str
) -> tuple[Any, np.ndarray, list[str], dict[str, Any]]:
    from cytearc.quality_control.filtering import filter_cell_metrics

    active_frame = store.cells.to_pandas_dataframe([config.cellKey])
    if active_frame[config.cellKey].isna().any() or store.cells.get_dtype(
        config.cellKey
    ) != np.dtype(bool):
        raise AnalysisInputError("cellKey must be a complete boolean selection")
    active = active_frame[config.cellKey].to_numpy(dtype=bool)
    if not active.any():
        raise NeedsInput(
            "The input selection has no cells. Supply a populated cellKey."
        )
    # QC flags and every stage read the assay over these cells.
    _require_measured_cell_key(store, assay, config.cellKey)
    flags = _qc_flags(store, active, assay)
    if config.qcPolicy == "retain":
        return (
            False,
            active,
            [
                "Input cells were retained with QC outlier flags; existing publication filtering is not independently verified."
            ],
            flags,
        )
    attrs = (
        sorted(config.qcBounds)
        if config.qcPolicy == "manual"
        else [
            f"{assay}_{suffix}"
            for suffix in ("nCounts", "nFeatures", "percentMito")
            if f"{assay}_{suffix}" in store.cells.columns
        ]
    )
    if not attrs:
        raise NeedsInput(
            "QC metrics are absent. Supply manual QC columns or select retain."
        )
    missing = set(attrs) - set(store.cells.columns)
    if missing:
        raise NeedsInput(f"QC columns are absent: {sorted(missing)}")
    frame = store.cells.to_pandas_dataframe(attrs)
    values = {
        name: frame[name].to_numpy(dtype=float, na_value=np.nan) for name in attrs
    }
    masks = {name: ~np.isfinite(values[name]) for name in attrs}
    if config.qcPolicy == "manual":
        options: dict[str, Any] = {
            "method": "manual",
            "attrs": attrs,
            "lows": [config.qcBounds[name][0] for name in attrs],
            "highs": [config.qcBounds[name][1] for name in attrs],
            "keep_bounds": True,
        }
    else:
        # Gentle policy removes only low complexity and high mitochondrial
        # percentages. Upper count/feature outliers remain advisory flags.
        options = {
            "method": "manual",
            "attrs": attrs,
            "keep_bounds": True,
            "lows": [
                flags[name]["low"] if not name.endswith("percentMito") else None
                for name in attrs
            ],
            "highs": [
                flags[name]["high"] if name.endswith("percentMito") else None
                for name in attrs
            ],
        }
    result = filter_cell_metrics(
        values,
        masks,
        active,
        method="manual",
        lows=options["lows"],
        highs=options["highs"],
        keep_bounds=True,
    )
    warnings: list[str] = []
    warnings.append(
        "QC thresholds may remove low-complexity biological populations; inspect retention before interpretation."
    )
    return options, result.retained, warnings, flags


def _qc_projections(
    store: Any,
    active: np.ndarray,
    assay: str,
    config: AnalysisConfig,
    flags: dict[str, Any],
    roles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project explicit global policies without changing the chosen cohort."""
    from cytearc.quality_control.filtering import filter_cell_metrics

    projections = []
    # Each group column is converted to text once; every policy then counts
    # its retained cells per level in one pass.
    group_data = {}
    for column in group_columns(roles):
        series = store.cells.to_pandas_dataframe([column])[column]
        text = series.astype(str)
        missing = (series.isna() | text.str.strip().eq("")).to_numpy()
        group_data[column] = (
            text.to_numpy(dtype=object),
            missing,
            text.loc[active & ~missing].value_counts(),
        )
    for policy in ("retain", "gentleMad5", "manual"):
        attrs = sorted(config.qcBounds) if policy == "manual" else list(flags)
        bounds: dict[str, list[float | None]] = {}
        limitations = []
        available = True
        retained = active.copy()
        if policy == "manual" and not config.qcBounds:
            available = False
            limitations.append("No manual thresholds were supplied; none are invented.")
        elif policy != "retain" and not attrs:
            available = False
            limitations.append("No available QC metrics support this projection.")
        elif policy != "retain":
            if policy == "manual":
                bounds = {column: list(config.qcBounds[column]) for column in attrs}
            else:
                bounds = {
                    column: [
                        flags[column]["low"]
                        if not column.endswith("percentMito")
                        else None,
                        flags[column]["high"]
                        if column.endswith("percentMito")
                        else None,
                    ]
                    for column in attrs
                }
            values = {
                column: store.cells.to_pandas_dataframe([column])[column].to_numpy(
                    dtype=float, na_value=np.nan
                )
                for column in attrs
            }
            retained = filter_cell_metrics(
                values,
                {column: ~np.isfinite(values[column]) for column in attrs},
                active,
                method="manual",
                lows=[bounds[column][0] for column in attrs],
                highs=[bounds[column][1] for column in attrs],
                keep_bounds=True,
            ).retained
        groups = {}
        if available:
            for column, (text, missing, counts) in group_data.items():
                kept = pd.Series(text[active & ~missing & retained]).value_counts()
                rows = []
                for value, count in counts.head(64).items():
                    retained_cells = int(kept.get(value, 0))
                    rows.append(
                        {
                            "value": str(value),
                            "inputCells": int(count),
                            "retainedCells": retained_cells,
                            "removedCells": int(count) - retained_cells,
                        }
                    )
                groups[column] = {
                    "levels": rows,
                    "missingInputCells": int((active & missing).sum()),
                    "missingRetainedCells": int((retained & missing).sum()),
                    "omittedLevels": max(0, len(counts) - 64),
                }
        projections.append(
            {
                "policy": policy,
                "available": available,
                "executed": policy == config.qcPolicy,
                "inputCells": int(active.sum()),
                "retainedCells": int(retained.sum()) if available else None,
                "removedCells": int((active & ~retained).sum()) if available else None,
                "bounds": bounds,
                "byGroup": groups,
                "limitations": limitations,
            }
        )
    return projections


def _design(
    store: Any,
    retained: np.ndarray,
    technical: list[str],
    protected: list[str],
    authorization: str | None,
    *,
    inferred: list[str] | None = None,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Decide correction eligibility from caller-declared experimental facts.

    Only caller-declared ``protected`` columns satisfy the requirement for
    protected biology. ``inferred`` protected columns proposed by a model get
    the same label and crossing checks, so they can block correction but can
    never permit it.
    """
    reasons: list[str] = []
    design: dict[str, Any] = {"evidenceId": "design:correction", "crossings": []}
    if not technical:
        return False, design, ["No technical batch correction was requested."]
    if not authorization or not authorization.strip():
        reasons.append(
            "Technical batch correction lacks caller-supplied experimental evidence."
        )
    if not protected:
        reasons.append(
            "Correction has no explicitly protected biological metadata for validation."
        )
    protected = list(dict.fromkeys([*protected, *(inferred or [])]))
    columns = list(dict.fromkeys([*technical, *protected]))
    frame = store.cells.to_pandas_dataframe(columns).loc[retained]
    for column in columns:
        series = frame[column]
        if series.isna().any() or series.astype(str).str.strip().eq("").any():
            reasons.append(f"Correction metadata {column!r} contains missing labels.")
        if series.nunique() < 2 or series.nunique() > 64:
            reasons.append(
                f"Correction metadata {column!r} needs between 2 and 64 categorical levels."
            )
    for batch in technical:
        for biology in protected:
            table = pd.crosstab(frame[batch], frame[biology])
            crossed = bool(table.size and (table.to_numpy() > 0).all())
            design["crossings"].append(
                {"technical": batch, "protected": biology, "fullyCrossed": crossed}
            )
            if not crossed or batch == biology:
                reasons.append(
                    f"Technical {batch!r} is not fully crossed with protected {biology!r}."
                )
    design["limitations"] = reasons
    return not reasons, design, reasons


def _blacklist(exclusions: list[str]) -> str:
    return "|".join(
        [
            _MITO_PATTERN,
            *(f"(?-i:^{re.escape(name)}$)" for name in sorted(set(exclusions))),
        ]
    )


def _references(study: Study) -> list[dict[str, str]]:
    references = []
    remaining = 16_384
    for index, name in enumerate(study.referenceFiles):
        path = Path(name).expanduser()
        if not path.is_file():
            raise NeedsInput("A supplied local reference file is unavailable.")
        with path.open("rb") as handle:
            raw = handle.read(remaining + 1)
        if len(raw) > remaining:
            raise NeedsInput(
                "Local reference text exceeds the 16 KiB evidence budget; supply shorter excerpts."
            )
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise NeedsInput(
                "Reference files must contain UTF-8 text excerpts."
            ) from error
        remaining -= len(raw)
        references.append(
            {
                "evidenceId": f"reference:{index}",
                "text": content,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return references


def inspect_source(
    source: str | Path,
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
    *,
    inferred_protected: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a local RNA source and summarize only non-held-out evidence.

    ``inferred_protected`` lists model-proposed protected columns. They join
    the preservation checks and frozen metadata, but correction eligibility
    still requires a caller-declared protected column.
    """
    from cytearc.assay import RNAassay

    store = open_store(source, config, runtime)
    assays = [
        name
        for name in store.assay_names
        if isinstance(store.get_assay(name), RNAassay)
    ]
    assay = config.assay or (assays[0] if len(assays) == 1 else None)
    if assay is None or assay not in assays:
        raise NeedsInput(f"Choose one RNA assay from {assays}.", field="assay")
    require_clean_analysis(store, assay)
    inferred = [
        column
        for column in dict.fromkeys(inferred_protected or [])
        if column not in study.protectedColumns
    ]
    protected = [*study.protectedColumns, *inferred]
    declared = list(
        dict.fromkeys(
            [
                *study.technicalBatchColumns,
                *protected,
                *([study.sampleColumn] if study.sampleColumn else []),
                *([study.captureColumn] if study.captureColumn else []),
            ]
        )
    )
    missing = set([config.cellKey, *declared]) - set(store.cells.columns)
    if missing:
        raise NeedsInput(f"Declared source columns are absent: {sorted(missing)}")
    forbidden = [
        name
        for name in [*declared, *config.qcBounds]
        if is_held_out_column(name, study)
    ]
    if forbidden:
        raise AnalysisInputError(
            f"Held-out annotation columns cannot define analysis roles or QC bounds: {forbidden}"
        )
    filtering, retained, limitations, flags = _filtering(store, study, config, assay)
    absent_qc = [
        f"{assay}_{suffix}"
        for suffix in ("nCounts", "nFeatures", "percentMito")
        if f"{assay}_{suffix}" not in flags
    ]
    if absent_qc:
        limitations.append(
            f"Optional QC outlier flags are unavailable for missing columns: {absent_qc}."
        )
    from zarr.storage import LocalStore

    if not isinstance(store.get_assay(assay).matrixGroup.store, LocalStore):
        limitations.append(
            "The local mount may fetch remote count blocks; remote counts are not hydrated by this procedure."
        )
    if int(retained.sum()) < 4:
        raise NeedsInput("At least four retained cells are required for RNA analysis.")
    features = store.get_assay(assay).feats.to_pandas_dataframe(["names"])["names"]
    feature_names = {str(value) for value in features.dropna()}
    unknown = set(study.featureExclusions) - feature_names
    if unknown:
        raise AnalysisInputError(
            f"Exact feature exclusions are absent from the assay: {sorted(unknown)}"
        )
    blacklist = _blacklist(study.featureExclusions)
    available = sum(not re.search(blacklist, str(value)) for value in features.dropna())
    if available < 3:
        raise NeedsInput(
            "At least three features must remain after feature exclusions."
        )
    visible = sorted(
        name
        for name in store.cells.columns
        if name not in {"I", "ids", "names", config.cellKey}
        and not is_held_out_column(name, study)
    )
    offered = [name for name in visible if len(name) <= MAX_COLUMN_NAME]
    if len(offered) < len(visible):
        limitations.append(
            f"Decision evidence omits {len(visible) - len(offered)} metadata columns "
            f"whose names exceed {MAX_COLUMN_NAME} characters."
        )
    ordered = list(dict.fromkeys([*declared, *flags, *offered]))
    shown = ordered[:_MAX_COLUMNS]
    if len(ordered) > len(shown):
        limitations.append(
            f"Context shows {len(shown)} of {len(ordered)} metadata columns; explicit study roles take priority."
        )
    profiles = [
        _profile(store.cells.to_pandas_dataframe([name])[name].loc[retained], name)
        for name in shown
    ]
    roles = resolved_roles(study)
    feature_audit = family_audit(features.fillna("").to_numpy(), blacklist)
    active = store.cells.to_pandas_dataframe([config.cellKey])[config.cellKey].to_numpy(
        dtype=bool
    )
    projections = _qc_projections(store, active, assay, config, flags, roles)
    design_summary = design_diagnostics(store.cells, retained, roles)
    eligible, design, reasons = _design(
        store,
        retained,
        study.technicalBatchColumns,
        study.protectedColumns,
        study.batchCorrectionEvidence,
        inferred=inferred,
    )
    limitations.extend(reasons)
    if study.captureColumn is None:
        limitations.append(
            "Physical capture identity is absent; capture-specific claims are unsupported."
        )
    limitations.append(
        "The procedure supports descriptive population discovery, not causal or differential inference."
    )
    excluded = sorted(
        name for name in store.cells.columns if is_held_out_column(name, study)
    )
    fingerprint_columns = sorted(set(["ids", "names", "I", config.cellKey, *visible]))
    evidence: dict[str, Any] = {
        "evidenceId": "source:summary",
        "assay": assay,
        "inputCells": int(store.cells.fetch_all(config.cellKey).sum()),
        "retainedCells": int(retained.sum()),
        "availableFeatures": available,
        "qcFlags": flags,
        "study": study.model_dump(exclude={"excludedColumns", "referenceFiles"}),
        "columns": {row["column"]: row for row in profiles},
        "resolvedRoles": roles,
        "qcProjections": projections,
        "featureAudit": feature_audit,
        "designDiagnostics": design_summary,
        "design": design,
        "offeredFeatureExclusions": sorted(study.featureExclusions),
        "mitochondrialFeatures": sorted(
            name for name in feature_names if re.search(_MITO_PATTERN, name)
        )[:64],
        "references": _references(study),
        "limitations": limitations,
    }
    return {
        "source": str(_local_source(source)),
        "fingerprint": _fingerprint(store, assay, fingerprint_columns),
        "fingerprintVersion": FINGERPRINT_VERSION,
        "fingerprintColumns": fingerprint_columns,
        "assay": assay,
        "cellKey": config.cellKey,
        "inputCells": evidence["inputCells"],
        "retainedCells": int(retained.sum()),
        "availableFeatures": available,
        "filtering": filtering,
        "blacklist": blacklist,
        "qcFlags": flags,
        "qcProjections": projections,
        "featureAudit": feature_audit,
        "resolvedRoles": roles,
        "designDiagnostics": design_summary,
        "diagnosticColumns": shown,
        "snapshotColumns": list(
            dict.fromkeys([*study.technicalBatchColumns, *declared, *shown])
        ),
        "technicalBatchColumns": list(study.technicalBatchColumns),
        "protectedColumns": protected,
        "sampleColumn": study.sampleColumn,
        "captureColumn": study.captureColumn,
        "excludedColumns": excluded,
        "contextEvidence": evidence,
        "correctionEligible": eligible,
        "limitations": limitations,
    }


def prepare_context(
    prepared: dict[str, Any],
    decision: ContextDecision,
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Resolve grounded role suggestions while preserving supplied authority."""
    evidence = prepared["contextEvidence"]
    columns = set(evidence["columns"])
    if set(decision.columnRoles) - columns:
        raise AnalysisInputError("Context roles must use offered non-held-out columns")
    offered = set(evidence["offeredFeatureExclusions"])
    if set(decision.excludeFeatures) - offered:
        raise AnalysisInputError(
            "Feature exclusions must use exact offered feature names"
        )
    inferred: list[str] = []
    for column, role in decision.columnRoles.items():
        supplied_roles = {
            entry["role"]
            for entry in resolved_roles(study)
            if entry["column"] == column
        }
        if role == "technical" and column not in study.technicalBatchColumns:
            raise AnalysisInputError(
                "A model cannot authorize a new technical batch column"
            )
        if column in study.technicalBatchColumns and role not in supplied_roles | {
            "ignore"
        }:
            raise AnalysisInputError("A model cannot change a supplied technical role")
        if column in study.protectedColumns and role not in supplied_roles:
            raise AnalysisInputError("A model cannot remove a supplied protected role")
        if role in {"sample", "capture"}:
            supplied_column = (
                study.sampleColumn if role == "sample" else study.captureColumn
            )
            if supplied_column is not None and column != supplied_column:
                raise AnalysisInputError(
                    "A model cannot replace a supplied sample or capture column"
                )
        if role == "protected" and column not in study.protectedColumns:
            inferred.append(column)
    # Caller-supplied roles are authoritative. Model guesses remain suggestions
    # in the decision record: an inferred protected column adds preservation
    # checks but cannot license correction.
    updated_study = study.model_copy(
        update={
            "featureExclusions": sorted(
                set(study.featureExclusions) | set(decision.excludeFeatures)
            ),
        }
    )
    fresh = inspect_source(
        prepared["source"],
        updated_study,
        config,
        runtime or RuntimeConfig(),
        inferred_protected=inferred,
    )
    if fresh["fingerprint"] != prepared["fingerprint"]:
        raise AnalysisInputError("Source changed while resolving context")
    roles = resolved_roles(study, decision.columnRoles, decision.evidenceIds)
    fresh["resolvedRoles"] = roles
    fresh["contextEvidence"]["resolvedRoles"] = roles
    store = open_store(prepared["source"], config, runtime or RuntimeConfig())
    _, retained, _, _ = _filtering(store, updated_study, config, fresh["assay"])
    fresh["diagnosticColumns"] = list(
        dict.fromkeys(
            [
                *group_columns(roles),
                *fresh["qcFlags"],
                *prepared["diagnosticColumns"],
                *fresh["diagnosticColumns"],
            ]
        )
    )[:_MAX_COLUMNS]
    fresh["snapshotColumns"] = list(
        dict.fromkeys(
            [
                *fresh["technicalBatchColumns"],
                *fresh["snapshotColumns"],
                *group_columns(roles),
                *fresh["diagnosticColumns"],
            ]
        )
    )
    fresh["contextEvidence"]["columns"] = {
        column: _profile(
            store.cells.to_pandas_dataframe([column])[column].loc[retained], column
        )
        for column in fresh["diagnosticColumns"]
    }
    active = store.cells.to_pandas_dataframe([config.cellKey])[config.cellKey].to_numpy(
        dtype=bool
    )
    fresh["designDiagnostics"] = design_diagnostics(store.cells, retained, roles)
    fresh["qcProjections"] = _qc_projections(
        store, active, fresh["assay"], config, fresh["qcFlags"], roles
    )
    fresh["contextEvidence"]["designDiagnostics"] = fresh["designDiagnostics"]
    fresh["contextEvidence"]["qcProjections"] = fresh["qcProjections"]
    return fresh


def verify_source(
    source: str | Path,
    prepared: dict[str, Any],
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
) -> None:
    """Reject changed axes, selections, values or nullable metadata on resume."""
    if prepared.get("fingerprintVersion") != FINGERPRINT_VERSION:
        raise AnalysisInputError(
            "This analysis recorded its source fingerprint in a format that this "
            "CyteArc release cannot verify, so its numerical results cannot be "
            "opened. Its saved report, annotations and records remain readable; "
            "start a new analysis to use the numerical results."
        )
    store = open_store(source, config, runtime)
    columns = prepared["fingerprintColumns"]
    if not set(columns).issubset(store.cells.columns):
        raise AnalysisInputError(
            "Source metadata columns changed since the saved analysis"
        )
    if _fingerprint(store, prepared["assay"], columns) != prepared["fingerprint"]:
        raise AnalysisInputError("Source fingerprint changed since the saved analysis")
    if _references(study) != prepared["contextEvidence"].get("references", []):
        raise AnalysisInputError(
            "Local reference evidence changed since the saved analysis"
        )
