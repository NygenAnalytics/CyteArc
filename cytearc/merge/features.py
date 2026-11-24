import sys
from collections import Counter
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..metadata.membership import FeatureRecord, measured_features_by_source
from ..storage.metadata_keys import (
    measured_features_column,
    measured_features_columns,
)
from ..utils.arguments import quoted_examples, quoted_value
from .row_plan import MERGED_ID_SEPARATOR

#: The most features that the warning about unshared features names per source.
UNSHARED_FEATURE_EXAMPLES = 5

#: The text between a merged source's name and the name of each of its own
#: sources, under which a merge records that source, such as ``inner__left``.
#: It separates a source's name from each merged cell ID too, so each recorded
#: name starts the IDs of its cells, such as ``inner__left__AAAC-1``.
NESTED_SOURCE_SEPARATOR = MERGED_ID_SEPARATOR

_NESTED_NAMES = (
    "A merge records each source of a merged source by the merged source's "
    f"name, {NESTED_SOURCE_SEPARATOR!r} and the source's own name."
)


@dataclass(frozen=True, slots=True)
class NestedSources:
    """The sources of a merged source, which a merge of it records in its place.

    Attributes:
        column: The merged source's cell column that names the source of
            each of its cells.
        names: Its sources: the categories of ``column`` in the order of
            their codes, or, for a column of text, the labels that it holds
            in sorted order, blank and missing labels left out.
        categorical: Whether ``column`` stores the codes of its categories.
        held: The sources of ``names`` that hold each assay of the merged
            source that the merge takes: those that its record of the assay
            names, or, for an assay without a record, those with cells that
            the assay measured.
        records: The columns of the merged source's record of each assay
            that holds one, which a refusal names.
    """

    column: str
    names: tuple[str, ...]
    categorical: bool
    held: dict[str, tuple[str, ...]]
    records: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class FeatureAlignment:
    """Resolved feature union and per-source remapping for one assay.

    ``featOrderMap`` gives, for each source, the merged position of each of
    its features, or -1 for a feature that the merged features leave out.
    ``sourcePresent`` tells, in source order, whether each source holds the
    assay, since a source without it and a source whose assay holds no
    features both map no features. ``measuredFeatures`` tells, for each
    source, whether it measured each merged feature. ``sourceRecords`` holds
    each source's own record of which features each of its sources measured,
    over the source's features, as a merge that wrote the source records it,
    or None.
    """

    mergedFeatsMap: pd.DataFrame
    featOrderMap: list[np.ndarray]
    nFeats: int
    overlapFraction: float
    sourcePresent: tuple[bool, ...]
    measuredFeatures: tuple[np.ndarray, ...]
    sourceRecords: tuple[FeatureRecord | None, ...]

    def resident_bytes(self) -> int:
        frame_bytes = int(self.mergedFeatsMap.memory_usage(index=True, deep=True).sum())
        array_bytes = sum(
            array.nbytes
            for array in {id(array): array for array in self.featOrderMap}.values()
        )
        measured_bytes = sum(vector.nbytes for vector in self.measuredFeatures)
        record_bytes = sum(
            record.measured.nbytes + record.features.nbytes
            for record in self.sourceRecords
            if record is not None
        )
        return (
            frame_bytes
            + array_bytes
            + measured_bytes
            + record_bytes
            + sys.getsizeof(self.featOrderMap)
        )


type FeatureKey = Literal["ids", "names"]
type FeatureSet = Literal["union", "intersection"]

_NO_OVERLAP = (
    "No overlapping features found! Will not merge the files. No feature {kind} "
    "are shared across the sources."
)
_NAME_HINT = (
    " If the sources use different ID schemes but comparable feature names (for "
    "example Ensembl IDs in one source and gene symbols in another), pass "
    "feature_key='names' to match features by name."
)


def _source_keys(
    assays: list[Any | None], names: list[str], key: FeatureKey
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return each source's merge keys and feature names in source order."""
    keyed: list[tuple[np.ndarray, np.ndarray]] = []
    for assay, source_name in zip(assays, names, strict=True):
        if assay is None or int(assay.feats.N) == 0:
            empty = np.asarray([], dtype=object)
            keyed.append((empty, empty))
            continue
        frame = assay.feats.to_pandas_dataframe(["names", "ids"])
        if key == "ids" and frame["ids"].duplicated().any():
            raise ValueError(
                f"Duplicate feature IDs in assay {assay.name!r} of source "
                f"{source_name!r}; assign unique feature IDs before merging"
            )
        keyed.append((frame[key].to_numpy(), frame["names"].to_numpy()))
    return keyed


def _merge_order_feats(
    keyed: list[tuple[np.ndarray, np.ndarray]],
    key: FeatureKey,
    *,
    require_overlap: bool,
) -> pd.DataFrame:
    union: dict[Any, Any] = {}
    source_presence: Counter[Any] = Counter()
    for keys, feature_names in keyed:
        source_presence.update(set(keys.tolist()))
        for feature_key, feature_name in zip(
            keys.tolist(), feature_names.tolist(), strict=True
        ):
            union.setdefault(feature_key, feature_name)
    # Object columns keep the table's representation, and so its resident
    # bytes, the same whether or not pandas stores text with pyarrow.
    ret_val = pd.DataFrame(
        {
            "idx": list(range(len(union))),
            "names": pd.Series(list(union.values()), dtype=object),
            "ids": pd.Series(list(union.keys()), dtype=object),
        }
    )
    non_empty = sum(1 for keys, _names in keyed if keys.size)
    # Sources overlap when two of them share a feature, so three sources of
    # which two overlap still merge.
    if (
        require_overlap
        and non_empty > 1
        and not any(count > 1 for count in source_presence.values())
    ):
        message = _NO_OVERLAP.format(kind="IDs" if key == "ids" else "names")
        raise ValueError(message + (_NAME_HINT if key == "ids" else ""))
    return ret_val


def _mapped_measured(
    measured: np.ndarray, order_map: np.ndarray, n_features: int
) -> np.ndarray:
    """Return whether a source measured each merged feature of the union, from ``measured``, whether it measured each of its own features.

    A merged feature that the source lacks, or of which the source holds
    several features by name and did not measure them all, is unmeasured.
    """
    merged = np.zeros(n_features, dtype=bool)
    merged[order_map] = True
    merged[order_map[~measured]] = False
    return merged


def align_features(
    assays: list[Any | None],
    names: list[str],
    *,
    key: FeatureKey = "ids",
    require_overlap: bool = True,
) -> FeatureAlignment:
    """Compute the merged feature table and remapping for one assay type.

    ``key="ids"`` matches features by exact ID. ``key="names"`` matches them by
    name, uses each name as the merged ID, and sums features that share a name
    within one source. A missing source is ``None`` and maps no features.
    """
    source_present = tuple(assay is not None for assay in assays)
    with_features = [
        assay for assay in assays if assay is not None and int(assay.feats.N) > 0
    ]
    if not with_features:
        # No source measured a feature, so no cell reads an unmeasured zero.
        empty = pd.DataFrame({"idx": [], "names": [], "ids": []})
        return FeatureAlignment(
            mergedFeatsMap=empty,
            featOrderMap=[np.asarray([], dtype=np.int64) for _ in assays],
            nFeats=0,
            overlapFraction=1.0,
            sourcePresent=source_present,
            measuredFeatures=tuple(np.zeros(0, dtype=bool) for _ in assays),
            sourceRecords=tuple(None for _ in assays),
        )

    keyed = _source_keys(assays, names, key)
    merged_feats = _merge_order_feats(keyed, key, require_overlap=require_overlap)
    positions = pd.Index(merged_feats["ids"])
    n_features = int(merged_feats.shape[0])
    order_maps = [
        np.asarray(positions.get_indexer(keys), dtype=np.int64)
        for keys, _names in keyed
    ]
    records = tuple(
        None if assay is None else measured_features_by_source(assay.feats)
        for assay in assays
    )
    # A source measured the features that it holds, except those that its own
    # record, from the merge that wrote it, marks as unmeasured by one of its
    # own sources, so that a merge of merged stores keeps their record.
    measured = tuple(
        np.zeros(n_features, dtype=bool)
        if assay is None
        else _mapped_measured(
            np.ones(order_map.size, dtype=bool)
            if record is None
            else record.measured.all(axis=0),
            order_map,
            n_features,
        )
        for assay, order_map, record in zip(assays, order_maps, records, strict=True)
    )
    return FeatureAlignment(
        mergedFeatsMap=merged_feats,
        featOrderMap=order_maps,
        nFeats=n_features,
        overlapFraction=float(_measured_by_all(measured, source_present).mean()),
        sourcePresent=source_present,
        measuredFeatures=measured,
        sourceRecords=records,
    )


def _measured_by_all(
    measured: Sequence[np.ndarray], present: Sequence[bool]
) -> np.ndarray:
    """Return whether every source that holds the assay measured each merged feature.

    At least one source holds the assay of an alignment that has features.
    """
    holders = [vector for vector, holds in zip(measured, present, strict=True) if holds]
    return np.asarray(np.logical_and.reduce(holders), dtype=bool)


def select_features(
    assay_name: str, alignment: FeatureAlignment, features: FeatureSet
) -> FeatureAlignment:
    """Return the alignment of the merged features that ``features`` keeps.

    ``"union"`` keeps every feature that a source holds, ``alignment``
    itself. ``"intersection"`` keeps the features that every source holding
    the assay measured, in merged order, maps every other source feature to
    -1, and so measures every feature that it keeps. An assay of no features
    in any source keeps its alignment either way. The merge and the composite
    check ``features`` where they take it.

    Raises:
        ValueError: If ``"intersection"`` keeps no feature of an assay whose
            sources hold some.
    """
    # An assay of no features in any source keeps none either way, and none
    # of its sources lacks one.
    if features == "union" or alignment.nFeats == 0:
        return alignment
    measured = alignment.measuredFeatures
    selected = np.flatnonzero(_measured_by_all(measured, alignment.sourcePresent))
    if selected.size == 0:
        raise ValueError(
            f"No feature of assay {assay_name!r} was measured by every source "
            "that holds it, so features='intersection' keeps none; pass "
            "features='union' to keep every feature of any source."
        )
    positions = np.full(alignment.nFeats, -1, dtype=np.int64)
    positions[selected] = np.arange(selected.size)
    frame = alignment.mergedFeatsMap.iloc[selected].copy().reset_index(drop=True)
    frame["idx"] = np.arange(selected.size)
    return FeatureAlignment(
        mergedFeatsMap=frame,
        featOrderMap=[positions[order_map] for order_map in alignment.featOrderMap],
        nFeats=int(selected.size),
        overlapFraction=1.0,
        sourcePresent=alignment.sourcePresent,
        measuredFeatures=tuple(vector[selected] for vector in measured),
        sourceRecords=alignment.sourceRecords,
    )


def nested_source_name(source: str, own: str) -> str:
    """Return the name under which a merge records ``own``, a source of the merged source ``source``, such as ``inner__left``."""
    return f"{source}{NESTED_SOURCE_SEPARATOR}{own}"


def _recorded_vectors(
    assay_name: str,
    alignment: FeatureAlignment,
    names: Sequence[str],
    nested: Sequence[NestedSources | None] | None,
) -> list[tuple[int, str, np.ndarray]]:
    """Return each recorded source of the sources that hold the assay: its source's position, its name and whether it measured each merged feature.

    A merged source of ``nested`` gives one for each of its own sources that
    hold the assay (:attr:`NestedSources.held`), named by
    :func:`nested_source_name`: what its record of the assay says that the
    source measured, or, without a record, the merged source's features,
    which each of those sources measured.
    """
    found: list[tuple[int, str, np.ndarray]] = []
    for index, (name, measured, present) in enumerate(
        zip(names, alignment.measuredFeatures, alignment.sourcePresent, strict=True)
    ):
        if not present:
            continue
        own = None if nested is None else nested[index]
        if own is None:
            found.append((index, name, measured))
            continue
        record = alignment.sourceRecords[index]
        rows = {} if record is None else dict(zip(record.sources, record.measured))
        for source in own.held[assay_name]:
            found.append(
                (
                    index,
                    nested_source_name(name, source),
                    measured
                    if record is None
                    else _mapped_measured(
                        rows[source], alignment.featOrderMap[index], alignment.nFeats
                    ),
                )
            )
    return found


def feature_record(
    alignment: FeatureAlignment,
    names: Sequence[str],
    nested: Sequence[NestedSources | None] | None = None,
    *,
    assay_name: str = "",
) -> dict[str, np.ndarray]:
    """Return whether each source that holds the assay measured each feature, by source, when the sources measured different features.

    A merge or a composite records what each source measured only then, as
    when a source lacks a feature that another holds, or a merged source's own
    record marks a feature as unmeasured. Otherwise every source measured
    every feature, and the record is empty. A merged source of ``nested``,
    in source order, is recorded as each of its own sources that hold the
    assay ``assay_name``, under :func:`nested_source_name`: with what its
    record of the assay says that the source measured, or, without one, with
    the merged source's features.

    Raises:
        ValueError: If the record's columns of two sources would be named
            alike once case is ignored
            (:func:`~cytearc.storage.metadata_keys.measured_features_columns`).
    """
    if all(
        measured.all()
        for measured, present in zip(
            alignment.measuredFeatures, alignment.sourcePresent, strict=True
        )
        if present
    ):
        return {}
    record = {
        label: measured
        for _index, label, measured in _recorded_vectors(
            assay_name, alignment, names, nested
        )
    }
    try:
        measured_features_columns(record)
    except ValueError as error:
        if nested is None or all(own is None for own in nested):
            raise
        raise ValueError(f"{error} {_NESTED_NAMES}") from error
    return record


def _combination_words(combination: str) -> tuple[str, str, str]:
    """Return the adjective of the combined features, the noun of the combination, and the verb of its zero counts."""
    if combination == "merge":
        return "merged", "merge", "writes"
    if combination == "composite":
        return "composite", "composite", "reads"
    raise ValueError(f"combination must be 'merge' or 'composite', got {combination!r}")


def unshared_features_warning(
    assay_name: str,
    alignment: FeatureAlignment,
    names: list[str],
    *,
    combination: Literal["merge", "composite"],
    assay_type: str,
    nested: Sequence[NestedSources | None] | None = None,
) -> str | None:
    """Return the warning about the merged features that a source lacks.

    A merge writes zero counts for a feature in the cells of a source that
    lacks it, and a union composite reads them. Both record which features
    each source measured in a feature column of its own
    (:func:`~cytearc.storage.metadata_keys.measured_features_column`), from
    which analyses leave the zeros out or refuse them, and the warning ends
    with those that do and those that still read them as counts, the ADT
    recipe for an assay of type ``"ADT"`` among them. The warning counts
    the merged features of ``assay_name`` that not every source holding the
    assay measured, then gives each such source's number of missing features
    and names the first ``UNSHARED_FEATURE_EXAMPLES`` of them in merged
    order. A source whose assay holds no features lacks every merged feature,
    and the warning says that it holds none. A source without the assay is
    left out, since a merge marks its cells as not measured. A merged source
    of ``nested`` counts as each of its own sources that the record holds
    (:func:`feature_record`); when it alone holds the assay, a merge's warning
    says that the merged source holds the zeros, which the merge keeps. A
    composite's warning ends with ``features='intersection'`` as the
    alternative, or says that it would keep no feature when every source
    lacks some.

    Args:
        assay_name: The merged assay, named in the warning.
        alignment: The assay's alignment from :func:`align_features`, in
            which every source feature has a merged position.
        names: The source names, in the order of ``alignment.featOrderMap``.
        combination: ``"merge"`` for a merge, which writes the zeros, or
            ``"composite"`` for a union composite, which reads them.
        assay_type: The assay's type; the warning of an ``"ADT"`` assay says
            what the ADT recipe does with the antibodies.
        nested: The sources of each merged source, which the record holds in
            its place, and None for every other source.

    Returns:
        The warning, or None when every source that holds the assay measured
        every merged feature, or when fewer than two sources hold it.
    """
    adjective, noun, verb = _combination_words(combination)
    recorded = _recorded_vectors(assay_name, alignment, names, nested)
    holding = [
        (source_name, alignment.featOrderMap[index], measured)
        for index, source_name, measured in recorded
    ]
    if len(holding) < 2:
        return None
    measured_by = np.zeros(alignment.nFeats, dtype=np.int64)
    lacking: list[str] = []
    feature_names = alignment.mergedFeatsMap["names"].to_numpy()
    for source_name, order_map, measured in holding:
        measured_by += measured
        missing = np.flatnonzero(~measured)
        if missing.size == 0:
            continue
        if order_map.size == 0:
            lacking.append(
                f"Source {source_name!r} holds no features of the assay, so it "
                f"lacks every {adjective} feature."
            )
            continue
        lacking.append(
            f"Source {source_name!r} lacks {missing.size:,}: "
            f"{quoted_examples(feature_names[missing], UNSHARED_FEATURE_EXAMPLES)}."
        )
    if not lacking:
        return None
    unshared = int(np.count_nonzero(measured_by < len(holding)))
    holders = {index for index, _source_name, _measured in recorded}
    zeros = (
        f" Merged source {names[holders.pop()]!r} holds zero counts for the "
        "features its sources lack, which the merge keeps"
        if combination == "merge" and len(holders) == 1
        else f" The {noun} {verb} zero counts for the features a source lacks"
    )
    record = (
        f"columns such as {quoted_value(measured_features_column(holding[0][0]))} "
        "record what each source measured"
    )
    warning = (
        f"Assay {assay_name!r}: {unshared:,} of the {alignment.nFeats:,} "
        f"{adjective} features {'was' if unshared == 1 else 'were'} not measured "
        "by every source that holds the assay. "
        + " ".join(lacking)
        + f"{zeros}; {record}, so "
        + (
            "analyses leave those zeros out or refuse them, except the pseudotime "
            "and lineage analyses of features, which still read them as counts."
            if assay_type == "ADT"
            else "analyses leave those zeros out or refuse them, except prepared "
            "percentage columns, potency scores, add_melded_assay and the "
            "pseudotime and lineage analyses of features, which still read them "
            "as counts."
        )
    )
    if combination == "merge":
        return warning
    if unshared < alignment.nFeats:
        return warning + (
            " Pass features='intersection' to keep only the features that every "
            "source measured."
        )
    return warning + (
        " No feature was measured by every source, so features='intersection' "
        "would keep none."
    )


def shared_features_warning(
    assay_name: str,
    alignment: FeatureAlignment,
    names: list[str],
    *,
    combination: Literal["merge", "composite"],
) -> str | None:
    """Return the warning about an intersection that keeps few features.

    An intersection keeps the features of ``alignment``, the union of the
    sources' features, that every source holding the assay measured. The
    warning is given when they are fewer than a tenth of the union, and gives
    for each source how many features the intersection would keep without it,
    so that a source sharing few features with the others stands out.

    Args:
        assay_name: The assay, named in the warning.
        alignment: The assay's alignment from :func:`align_features`, over the
            union of the sources' features.
        names: The source names, in the order of ``alignment.featOrderMap``.
        combination: ``"merge"`` or ``"composite"``, which the warning names
            as what keeps the features.

    Returns:
        The warning, or None when the intersection keeps at least a tenth of
        the union.
    """
    _adjective, noun, _verb = _combination_words(combination)
    holding = [
        (source_name, measured)
        for source_name, measured, present in zip(
            names, alignment.measuredFeatures, alignment.sourcePresent, strict=True
        )
        if present
    ]
    presence = np.zeros(alignment.nFeats, dtype=np.int64)
    for _source_name, measured in holding:
        presence += measured
    kept = int(np.count_nonzero(presence == len(holding)))
    if kept * 10 >= alignment.nFeats:
        return None
    without = []
    for source_name, measured in holding:
        kept_without = int(np.count_nonzero(presence - measured == len(holding) - 1))
        without.append(f"{kept_without:,} without {source_name!r}")
    return (
        f"Assay {assay_name!r}: the {noun} keeps {kept:,} of the "
        f"{alignment.nFeats:,} features of its sources, since "
        "features='intersection' keeps only the features that every source "
        f"measured. It would keep {', '.join(without[:-1])} and {without[-1]}."
    )


def dtype_for_integer_sum(dtype: np.dtype[Any], copies: int) -> np.dtype[Any]:
    if copies <= 1 or dtype.kind not in "biu":
        return dtype
    if dtype.kind in "bu":
        lower = 0
        upper = (1 if dtype.kind == "b" else np.iinfo(dtype).max) * copies
        for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
            candidate_info = np.iinfo(candidate)
            if lower >= candidate_info.min and upper <= candidate_info.max:
                return np.dtype(candidate)
    else:
        info = np.iinfo(dtype)
        lower = info.min * copies
        upper = info.max * copies
        for signed_candidate in (np.int8, np.int16, np.int32, np.int64):
            candidate_info = np.iinfo(signed_candidate)
            if lower >= candidate_info.min and upper <= candidate_info.max:
                return np.dtype(signed_candidate)
    return np.dtype(np.uint64 if dtype.kind in "bu" else np.int64)


def resolve_merge_dtype(
    assays: list[Any | None],
    feat_order_map: list[np.ndarray],
) -> str:
    """Return the count dtype of a merged assay.

    It is the common type of the source count dtypes, widened so that
    features summed by name cannot overflow it. An assay that has no features
    in any source holds no counts and is stored as uint8, as an empty import
    is.

    Raises:
        ValueError: If integer sources have no common integer dtype.
    """
    dtypes = [
        np.dtype(assay.rawData.dtype)
        for assay in assays
        if assay is not None and int(getattr(assay.feats, "N", 0)) > 0
    ]
    if not dtypes:
        return "uint8"
    common = np.result_type(*dtypes)
    if all(dtype.kind in "biu" for dtype in dtypes) and common.kind not in "biu":
        raise ValueError(
            "Source counts stored as "
            + ", ".join(sorted({dtype.name for dtype in dtypes}))
            + " have no common integer dtype"
        )
    # A source feature that the merged features leave out maps to -1.
    max_copies = max(
        (
            int(np.unique(order_map[order_map >= 0], return_counts=True)[1].max())
            for order_map in feat_order_map
            if np.any(order_map >= 0)
        ),
        default=1,
    )
    return str(dtype_for_integer_sum(common, max_copies))
