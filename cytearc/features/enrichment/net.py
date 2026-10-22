import hashlib
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from ...utils.arrays import has_duplicates, read_only_copy
from ...utils.logging import logger

__all__ = ["AmbiguousTargets", "PreparedNetwork", "prepare_network", "read_gmt"]

AmbiguousTargets = Literal["drop", "error"]


def _update_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little", signed=False))
    digest.update(encoded)


def _network_digest(
    source_names: np.ndarray,
    edge_source_index: np.ndarray,
    edge_feature_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    weighted: bool,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(b"weighted\0" if weighted else b"unweighted\0")
    digest.update(len(source_names).to_bytes(8, "little", signed=False))
    for source in source_names:
        _update_text(digest, str(source))
    for values, dtype in (
        (edge_source_index, "<i8"),
        (edge_feature_index, "<i8"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    if weighted:
        weights = np.ascontiguousarray(edge_weight, dtype="<f8")
        digest.update(np.asarray(weights.shape, dtype="<i8").tobytes())
        digest.update(weights.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedNetwork:
    """A matched, pruned, and canonically ordered gene-set network.

    ``network_digest`` hashes the retained sources, matched edges, and, for
    weighted networks, their weights. ``dropped_ambiguous_targets`` lists the
    sorted targets whose edges were dropped because each matched several
    active features.
    """

    source_names: np.ndarray
    source_sizes: np.ndarray
    matched_feature_index: np.ndarray
    edge_source_index: np.ndarray
    edge_feature_index: np.ndarray
    edge_weight: np.ndarray
    network_digest: str
    dropped_ambiguous_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        one_dimensional = (
            self.source_names,
            self.source_sizes,
            self.matched_feature_index,
            self.edge_source_index,
            self.edge_feature_index,
            self.edge_weight,
        )
        if any(values.ndim != 1 for values in one_dimensional):
            raise ValueError("Prepared network arrays must be one-dimensional")
        if len(self.source_names) == 0:
            raise ValueError("Prepared network must contain at least one source")
        if len(self.source_names) != len(self.source_sizes):
            raise ValueError("Source names and sizes must be aligned")
        n_edges = len(self.edge_source_index)
        if len(self.edge_feature_index) != n_edges or len(self.edge_weight) != n_edges:
            raise ValueError("Prepared network edge arrays must be aligned")
        if n_edges == 0 or len(self.matched_feature_index) == 0:
            raise ValueError("Prepared network must contain at least one matched edge")
        if np.any(self.source_sizes <= 0):
            raise ValueError("Prepared network source sizes must be positive")
        if has_duplicates(self.source_names):
            raise ValueError("Prepared network source names must be unique")
        if not all(isinstance(source, str) and source for source in self.source_names):
            raise ValueError("Prepared network source names must be non-empty strings")
        if not np.array_equal(self.source_names, np.sort(self.source_names)):
            raise ValueError("Prepared network source names must be sorted")
        integer_arrays = (
            self.source_sizes,
            self.matched_feature_index,
            self.edge_source_index,
            self.edge_feature_index,
        )
        if any(
            not np.issubdtype(values.dtype, np.integer) for values in integer_arrays
        ):
            raise ValueError("Prepared network index arrays must have integer dtypes")
        if np.any(self.matched_feature_index < 0) or not np.array_equal(
            self.matched_feature_index, np.unique(self.matched_feature_index)
        ):
            raise ValueError(
                "Prepared network matched features must be sorted and unique"
            )
        if np.any(self.edge_source_index < 0) or np.any(
            self.edge_source_index >= len(self.source_names)
        ):
            raise ValueError("Prepared network source indices are out of bounds")
        if (
            np.any(self.edge_feature_index < 0)
            or not np.isin(self.edge_feature_index, self.matched_feature_index).all()
        ):
            raise ValueError("Prepared network feature indices are out of bounds")
        expected_sizes = np.bincount(
            self.edge_source_index,
            minlength=len(self.source_names),
        )
        if not np.array_equal(self.source_sizes, expected_sizes):
            raise ValueError("Prepared network source sizes do not match its edges")
        edge_order = np.lexsort((self.edge_feature_index, self.edge_source_index))
        if not np.array_equal(edge_order, np.arange(n_edges)):
            raise ValueError("Prepared network edges must be canonically sorted")
        if not np.isfinite(self.edge_weight).all():
            raise ValueError("Prepared network weights must be finite")
        if not isinstance(self.network_digest, str) or not self.network_digest:
            raise ValueError("Prepared network digest must be a non-empty string")
        dropped = self.dropped_ambiguous_targets
        if (
            not isinstance(dropped, tuple)
            or not all(isinstance(target, str) and target for target in dropped)
            or list(dropped) != sorted(set(dropped))
        ):
            raise ValueError(
                "Prepared network dropped targets must be sorted unique strings"
            )


def read_gmt(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read gene sets from a tab-separated GMT file.

    The first field is used as the source name, the second description field is
    ignored, and each remaining non-empty field becomes one target row. A UTF-8
    byte order mark at the start of the file, which some editors write, is not
    part of the first source name.

    Args:
        path: Path to a GMT file with at least three fields per non-empty line.

    Returns:
        A DataFrame with ``source`` and ``target`` columns.
    """
    records: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Malformed GMT line {line_number}: expected at least 3 tab-separated fields"
                )
            source = fields[0].strip()
            targets = [target.strip() for target in fields[2:] if target.strip()]
            if not source:
                raise ValueError(f"Malformed GMT line {line_number}: source is empty")
            if not targets:
                raise ValueError(f"Malformed GMT line {line_number}: no targets found")
            records.extend((source, target) for target in targets)
    if not records:
        raise ValueError("GMT file contains no gene sets")
    return pd.DataFrame.from_records(records, columns=["source", "target"])


def _stripped_text_codes(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return the code of each value's text and the texts that the codes index.

    A value's text is ``str(value).strip()``, as the network keeps it, so
    values whose texts are equal share a code. Distinct values are stripped
    once; values of a string dtype are already their own text.
    """
    if isinstance(values.dtype, pd.StringDtype):
        codes, distinct = pd.factorize(values)
    else:
        codes, distinct = pd.factorize(
            np.asarray([str(value) for value in values.astype(object)], dtype=object)
        )
    merged, texts = pd.factorize(
        np.asarray(
            [str(text).strip() for text in np.asarray(distinct, dtype=object)],
            dtype=object,
        )
    )
    return merged[codes], np.asarray(texts, dtype=object)


def _repeats_a_pair(first: np.ndarray, second: np.ndarray, second_size: int) -> bool:
    """Whether aligned codes, ``second`` below ``second_size``, repeat a pair."""
    keys = np.sort(first.astype(np.int64) * second_size + second)
    return bool(np.any(keys[1:] == keys[:-1]))


def prepare_network(
    net: pd.DataFrame,
    *,
    active_feature_names: np.ndarray,
    active_feature_index: np.ndarray,
    tmin: int,
    weighted: bool,
    ambiguous_targets: AmbiguousTargets = "drop",
) -> PreparedNetwork:
    """Validate, match, prune, and canonically order a gene-set network.

    Targets match active feature names without case sensitivity. A target
    that matches several active features, such as a gene symbol shared by
    two feature ids, is ambiguous. With ``ambiguous_targets="drop"`` its
    edges are removed before ``tmin`` pruning, a warning names it, and the
    result lists it in ``dropped_ambiguous_targets``. With ``"error"`` it
    raises ``ValueError``. Only the retained edges enter the network digest.
    """
    if not isinstance(net, pd.DataFrame):
        raise TypeError("net must be a pandas DataFrame")
    if isinstance(tmin, bool) or not isinstance(tmin, int) or tmin < 1:
        raise ValueError("tmin must be an integer greater than or equal to 1")
    if not isinstance(weighted, bool):
        raise TypeError("weighted must be a boolean")
    if ambiguous_targets not in ("drop", "error"):
        raise ValueError("ambiguous_targets must be 'drop' or 'error'")
    if not {"source", "target"}.issubset(net.columns):
        raise ValueError("net must contain 'source' and 'target' columns")
    if any(
        list(net.columns).count(column) > 1 for column in ("source", "target", "weight")
    ):
        raise ValueError("Network columns must have unique names")

    feature_names = np.asarray(active_feature_names)
    raw_feature_index = np.asarray(active_feature_index)
    if feature_names.ndim != 1 or raw_feature_index.ndim != 1:
        raise ValueError("Active feature names and indices must be one-dimensional")
    if len(feature_names) == 0:
        raise ValueError("Feature selection contains no active features")
    if len(feature_names) != len(raw_feature_index):
        raise ValueError("Active feature names and indices must be aligned")
    if not np.issubdtype(raw_feature_index.dtype, np.integer):
        raise ValueError("Active feature indices must have an integer dtype")
    feature_index = np.asarray(raw_feature_index, dtype=np.int64)
    if np.any(feature_index < 0) or has_duplicates(feature_index):
        raise ValueError("Active feature indices must be non-negative and unique")

    if net[["source", "target"]].isna().any().any():
        raise ValueError("Network source and target values must not be missing")
    source_codes, source_texts = _stripped_text_codes(net["source"])
    target_codes, target_texts = _stripped_text_codes(net["target"])
    if (source_texts == "").any() or (target_texts == "").any():
        raise ValueError("Network source and target values must be non-empty")
    if _repeats_a_pair(source_codes, target_codes, len(target_texts)):
        raise ValueError("Network contains duplicate source-target edges")

    if "weight" in net.columns:
        raw_weights = net["weight"]
        # Values of an integer or floating dtype are never booleans.
        if raw_weights.dtype.kind not in "iuf" and (
            raw_weights.map(lambda value: isinstance(value, (bool, np.bool_))).any()
        ):
            raise ValueError("Network weights must be numeric and not boolean")
        try:
            weights = pd.to_numeric(raw_weights, errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Network weights must be numeric") from exc
        if not np.isfinite(weights).all():
            raise ValueError("Network weights must be finite")
    else:
        weights = np.ones(len(net), dtype=np.float64)
    if weighted:
        rows = np.flatnonzero(weights != 0.0)
        if len(rows) == 0:
            raise ValueError("Weighted network contains no non-zero edges")
    else:
        rows = np.arange(len(net))
        weights = np.ones(len(net), dtype=np.float64)

    # Targets match feature names as CaseInsensitiveIndex matches them, both
    # compared as str(value).upper(); a distinct text is upper-cased once.
    key_codes, keys = pd.factorize(
        np.asarray([str(name).upper() for name in feature_names], dtype=object)
    )
    key_counts = np.bincount(key_codes, minlength=len(keys))
    _, first_feature = np.unique(key_codes, return_index=True)
    target_keys = pd.Index(keys).get_indexer(
        [str(text).upper() for text in target_texts]
    )
    edge_keys = target_keys[target_codes[rows]]
    edge_matches = np.zeros(len(rows), dtype=np.int64)
    found = edge_keys >= 0
    edge_matches[found] = key_counts[edge_keys[found]]
    ambiguous = edge_matches > 1
    if ambiguous_targets == "error" and ambiguous.any():
        target = target_texts[target_codes[rows[np.argmax(ambiguous)]]]
        raise ValueError(
            f"Network target {target!r} matches multiple active assay "
            "features. Pass ambiguous_targets='drop' to drop its edges."
        )

    dropped_ambiguous_targets = tuple(
        sorted({str(target) for target in target_texts[target_codes[rows[ambiguous]]]})
    )
    if dropped_ambiguous_targets:
        logger.warning(
            f"Dropped {len(dropped_ambiguous_targets)} network targets that match "
            "multiple active assay features: " + ", ".join(dropped_ambiguous_targets)
        )
    matched = edge_matches == 1
    if not matched.any():
        raise ValueError(
            "Network has no targets overlapping the active assay features"
            + (
                f" after dropping {len(dropped_ambiguous_targets)} ambiguous targets"
                if dropped_ambiguous_targets
                else ""
            )
        )
    rows = rows[matched]
    edge_sources = source_codes[rows]
    # Active feature indices are unique, so a feature's position stands for it.
    edge_positions = first_feature[edge_keys[matched]]
    if _repeats_a_pair(edge_sources, edge_positions, len(feature_index)):
        raise ValueError(
            "Network contains duplicate edges after case-insensitive feature matching"
        )
    edge_features = feature_index[edge_positions]

    source_counts = np.bincount(edge_sources, minlength=len(source_texts))
    kept_codes = np.flatnonzero(source_counts >= tmin)
    kept_sources = sorted(str(source_texts[code]) for code in kept_codes)
    if not kept_sources:
        raise ValueError(
            "No network sources remain after feature overlap and "
            f"tmin={tmin} pruning ({len(rows)} matched edges)"
        )

    source_names = np.asarray(kept_sources, dtype="U")
    source_to_index = {
        source: index for index, source in enumerate(source_names.tolist())
    }
    source_ranks = np.full(len(source_texts), -1, dtype=np.int64)
    for code in kept_codes:
        source_ranks[code] = source_to_index[str(source_texts[code])]
    edge_ranks = source_ranks[edge_sources]
    kept = edge_ranks >= 0
    edge_source_index = edge_ranks[kept]
    edge_feature_index = edge_features[kept]
    edge_weight = weights[rows[kept]]
    # Source-feature pairs are distinct, so the order is canonical.
    order = np.lexsort((edge_feature_index, edge_source_index))
    edge_source_index = edge_source_index[order]
    edge_feature_index = edge_feature_index[order]
    edge_weight = edge_weight[order]
    matched_feature_index = np.unique(edge_feature_index).astype(np.int64)
    source_sizes = np.bincount(edge_source_index, minlength=len(source_names)).astype(
        np.int64
    )
    digest = _network_digest(
        source_names,
        edge_source_index,
        edge_feature_index,
        edge_weight,
        weighted=weighted,
    )

    return PreparedNetwork(
        source_names=read_only_copy(source_names, source_names.dtype),
        source_sizes=read_only_copy(source_sizes, np.int64),
        matched_feature_index=read_only_copy(matched_feature_index, np.int64),
        edge_source_index=read_only_copy(edge_source_index, np.int64),
        edge_feature_index=read_only_copy(edge_feature_index, np.int64),
        edge_weight=read_only_copy(edge_weight, np.float64),
        network_digest=digest,
        dropped_ambiguous_targets=dropped_ambiguous_targets,
    )
