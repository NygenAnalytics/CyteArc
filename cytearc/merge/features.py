import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.arguments import quoted_value

#: The most features that the warning about unshared features names per source.
UNSHARED_FEATURE_EXAMPLES = 5


@dataclass(frozen=True, slots=True)
class FeatureAlignment:
    """Resolved feature union and per-source remapping for one assay.

    ``sourcePresent`` tells, in source order, whether each source holds the
    assay, since a source without it and a source whose assay holds no
    features both map no features.
    """

    mergedFeatsMap: pd.DataFrame
    featOrderMap: list[np.ndarray]
    nFeats: int
    overlapFraction: float
    sourcePresent: tuple[bool, ...]

    def resident_bytes(self) -> int:
        frame_bytes = int(self.mergedFeatsMap.memory_usage(index=True, deep=True).sum())
        array_bytes = sum(
            array.nbytes
            for array in {id(array): array for array in self.featOrderMap}.values()
        )
        return frame_bytes + array_bytes + sys.getsizeof(self.featOrderMap)


type FeatureKey = Literal["ids", "names"]

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
    holders: int,
) -> tuple[pd.DataFrame, float]:
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
    # The share of merged features that every source holding the assay measured.
    measured_by_all = sum(count == holders for count in source_presence.values())
    return ret_val, measured_by_all / len(union)


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
        )

    keyed = _source_keys(assays, names, key)
    merged_feats, overlap = _merge_order_feats(
        keyed, key, require_overlap=require_overlap, holders=sum(source_present)
    )
    positions = pd.Index(merged_feats["ids"])
    return FeatureAlignment(
        mergedFeatsMap=merged_feats,
        featOrderMap=[
            np.asarray(positions.get_indexer(keys), dtype=np.int64)
            for keys, _names in keyed
        ],
        nFeats=int(merged_feats.shape[0]),
        overlapFraction=overlap,
        sourcePresent=source_present,
    )


def unshared_features_warning(
    assay_name: str,
    alignment: FeatureAlignment,
    names: list[str],
    *,
    combination: Literal["merge", "composite"],
) -> str | None:
    """Return the warning about the merged features that a source lacks.

    A merge writes zero counts for a feature in the cells of a source that
    lacks it, and a union composite reads them, which analyses then take as
    measured counts. The warning counts the merged features of ``assay_name``
    that not every source holding the assay measured, then gives each such
    source's number of missing features and names the first
    ``UNSHARED_FEATURE_EXAMPLES`` of them in merged order. A source whose
    assay holds no features lacks every merged feature, and the warning says
    that it holds none. A source without the assay is left out, since a merge
    marks its cells as not measured.

    Args:
        assay_name: The merged assay, named in the warning.
        alignment: The assay's alignment from :func:`align_features`, in
            which every source feature has a merged position.
        names: The source names, in the order of ``alignment.featOrderMap``.
        combination: ``"merge"`` for a merge, which writes the zeros, or
            ``"composite"`` for a union composite, which reads them and whose
            warning names ``features='intersection'`` as the alternative, or
            says that it would keep no feature when every source lacks some.

    Returns:
        The warning, or None when every source that holds the assay measured
        every merged feature, or when fewer than two sources hold it.
    """
    if combination == "merge":
        adjective, zeros = "merged", "The merge writes"
    elif combination == "composite":
        adjective, zeros = "composite", "The composite reads"
    else:
        raise ValueError(
            f"combination must be 'merge' or 'composite', got {combination!r}"
        )
    holding = [
        (source_name, order_map)
        for source_name, order_map, present in zip(
            names, alignment.featOrderMap, alignment.sourcePresent, strict=True
        )
        if present
    ]
    if len(holding) < 2:
        return None
    measured_by = np.zeros(alignment.nFeats, dtype=np.int64)
    lacking: list[str] = []
    feature_names = alignment.mergedFeatsMap["names"].to_numpy()
    for source_name, order_map in holding:
        measured = np.zeros(alignment.nFeats, dtype=bool)
        measured[order_map] = True
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
        listed = ", ".join(
            quoted_value(name)
            for name in feature_names[missing[:UNSHARED_FEATURE_EXAMPLES]].tolist()
        )
        more = missing.size - UNSHARED_FEATURE_EXAMPLES
        lacking.append(
            f"Source {source_name!r} lacks {missing.size:,}: {listed}"
            + (f" and {more:,} more." if more > 0 else ".")
        )
    if not lacking:
        return None
    unshared = int(np.count_nonzero(measured_by < len(holding)))
    warning = (
        f"Assay {assay_name!r}: {unshared:,} of the {alignment.nFeats:,} "
        f"{adjective} features {'was' if unshared == 1 else 'were'} not measured "
        "by every source that holds the assay. "
        + " ".join(lacking)
        + f" {zeros} zero counts for a feature in the cells of each source that "
        "lacks it, and analyses read those zeros as measured counts."
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
    assay_name: str, alignment: FeatureAlignment, names: list[str]
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

    Returns:
        The warning, or None when the intersection keeps at least a tenth of
        the union.
    """
    holding = [
        (source_name, order_map)
        for source_name, order_map, present in zip(
            names, alignment.featOrderMap, alignment.sourcePresent, strict=True
        )
        if present
    ]
    presence = np.zeros(alignment.nFeats, dtype=np.int64)
    for _source_name, order_map in holding:
        presence[order_map] += 1
    kept = int(np.count_nonzero(presence == len(holding)))
    if kept * 10 >= alignment.nFeats:
        return None
    without = []
    for source_name, order_map in holding:
        measured = np.zeros(alignment.nFeats, dtype=np.int64)
        measured[order_map] = 1
        kept_without = int(np.count_nonzero(presence - measured == len(holding) - 1))
        without.append(f"{kept_without:,} without {source_name!r}")
    return (
        f"Assay {assay_name!r}: the composite keeps {kept:,} of the "
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
    max_copies = max(
        (
            int(np.unique(order_map, return_counts=True)[1].max())
            for order_map in feat_order_map
            if order_map.size
        ),
        default=1,
    )
    return str(dtype_for_integer_sum(common, max_copies))
