"""Value-selection contracts shared by analysis and presentation layers."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..storage.arrays import linked_missing_mask
from ..storage.async_execution import zarr_decoding_threads
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_group,
    fingerprint_strings,
    inspect_artifact,
    parse_artifact_ref,
)
from ..storage.geometry import ArrayGeometry, array_geometry
from ..storage.partition import scan_band
from ..storage.selections import read_stored_selection_indices, validate_cell_selection
from ..storage.types import as_zarr_array
from ..storage.validation_scope import validation_scope
from ..utils.arguments import integer_argument
from ..utils.arrays import sort_categories, sort_categories_bytes
from ..utils.memory_budget import (
    MemoryLedger,
    fixed_text_bytes,
    hash_table_bytes,
    object_array_bytes,
    python_text_bytes,
)
from .rows import (
    apply_missing_mask,
    metadata_missing_mask,
    operation_concurrency,
    read_metadata_missing_rows,
    read_metadata_rows,
    read_rows,
    read_rows_bytes,
)

LookupBy = Literal["name", "id", "index"]
FeatureReduction = Literal["mean", "sum"]
CellFieldKind = Literal["auto", "categorical", "continuous"]
NormSource = Literal["assay", "raw"]
NormTransform = Literal["none", "log1p"]
Standardize = Literal["none", "feature"]

__all__ = [
    "CELL_VALUE_NAMES",
    "CellField",
    "CellFieldKind",
    "CellSelectionRead",
    "CellValueSpec",
    "CellValues",
    "FeatureReduction",
    "FeatureRef",
    "LookupBy",
    "NormalizationSpec",
    "NormSource",
    "NormTransform",
    "NamedCellArtifact",
    "ResolvedCellArtifact",
    "ResolvedGrouping",
    "Standardize",
    "StudyDesign",
    "category_codes",
    "cell_selection_indices",
    "cell_value_array",
    "cell_value_spec",
    "grouping_value_name",
    "hashed_text_bytes",
    "plan_cell_selection_read",
    "read_cell_selection",
    "resolve_cell_aligned_artifact",
    "resolve_cell_column",
    "resolve_grouping",
    "sample_codes",
    "valid_category_mask",
]


@dataclass(frozen=True, slots=True)
class CellValueSpec:
    """The per-cell arrays of one cell-aligned artifact kind."""

    name: str
    categorical: bool
    alternatives: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alternatives = dict(self.alternatives)
        if not all(
            isinstance(flag, bool)
            for flag in (self.categorical, *alternatives.values())
        ):
            raise TypeError("Cell value categorical flags must be booleans")
        if self.name in alternatives:
            raise ValueError(
                "The canonical cell value array cannot also be an alternative"
            )
        object.__setattr__(self, "alternatives", MappingProxyType(alternatives))


#: The per-cell arrays of every cell-aligned artifact kind, a read-only
#: mapping from each kind to its ``CellValueSpec``: the canonical array, whether
#: it holds labels, and the kind's other per-cell arrays. A kind is
#: cell-aligned exactly when it is listed, and readers refuse every other
#: kind, whatever array they are asked for. Reductions and Harmony corrections
#: hold a row per cell too, but their cells follow the cell selection of their
#: lineage rather than a ``cell_selection`` input, so they are not listed.
#:
#: The canonical array names are part of the identity of the results that
#: read them: ``select_cells``, plot groupings, and other consumers record
#: the artifact that they read but not that they read its canonical array;
#: ``select_cells`` names only another array that ``value`` selects. Changing
#: a kind's canonical array therefore changes what those consumers compute
#: from the same recorded inputs, and needs an operation revision for each
#: of them.
CELL_VALUE_NAMES: Mapping[str, CellValueSpec] = MappingProxyType(
    {
        "ambient_contamination": CellValueSpec(
            "values",
            False,
            {
                "observed": False,
                "expected": False,
                "p_values": False,
                "expressing": True,
                "used": True,
            },
        ),
        "cell_calling": CellValueSpec(
            "fdr",
            False,
            {
                "limited": True,
                "total": False,
                "log_probability": False,
                "p_value": False,
            },
        ),
        "cell_cycle": CellValueSpec(
            "phase", True, {"s_score": False, "g2m_score": False}
        ),
        # group_codes holds each cell's group, a position in the artifact's
        # group_keys attribute, -1 for none.
        "clonal_expansion": CellValueSpec(
            "values",
            True,
            {
                "group_codes": True,
                "group_clone_size": False,
                "group_clone_frequency": False,
            },
        ),
        # codes holds each early cell's clone and fate_codes its dominant fate,
        # a position in the artifact's group_keys attribute, -1 for none;
        # fractions holds one column per fate.
        "clone_fate_bias": CellValueSpec(
            "values",
            True,
            {
                "codes": True,
                "fate_codes": True,
                "n_late_cells": False,
                "fractions": False,
            },
        ),
        # clone_probability is stored when the clone source gives each cell's
        # posterior, such as Numbat's p_opt or genotype clones' likelihood;
        # n_barcodes when the clones were called from lineage barcodes;
        # doublet when calling removed cells as doublets, as barcode clones
        # do; ambiguous when calling left cells out, as genotype clones do
        # where no clone reaches the posterior a cell needs or, by exact
        # agreement, where a cell's calls agree with two clones that
        # disagree; and n_alleles when the clones were called from recorder
        # alleles.
        "clones": CellValueSpec(
            "values",
            True,
            {
                "codes": True,
                "clone_size": False,
                "clone_probability": False,
                "n_barcodes": False,
                "doublet": True,
                "ambiguous": True,
                "n_alleles": False,
            },
        ),
        # codes holds each cell's clonotype, a position in the artifact's
        # clonotype_ids, -1 for none.
        "clonotypes": CellValueSpec(
            "values", True, {"codes": True, "clone_size": False}
        ),
        "cluster_cut": CellValueSpec("labels", True),
        "cluster_labels": CellValueSpec("values", True),
        "doublet_score": CellValueSpec(
            "values", False, {"raw_scores": False, "predicted_doublets": True}
        ),
        "embedding": CellValueSpec("values", False),
        "enrichment_scores": CellValueSpec("scores", False),
        "fate_map": CellValueSpec("probabilities", False, {"valid": True}),
        # guide_codes and guide_umis hold one row per cell and one column per
        # called guide, padded to the cells' largest number of calls.
        "guide_calls": CellValueSpec(
            "values",
            True,
            {
                "n_guides": False,
                "top_guide": True,
                "top_umi": False,
                "top_fraction": False,
                "total_umi": False,
                "guide_codes": True,
                "guide_umis": False,
            },
        ),
        "hto_identity": CellValueSpec(
            "values", True, {"max_id": True, "second_id": True, "margin": False}
        ),
        # chain_slots holds one row per cell and one column per chain slot,
        # positions among the contigs, which are not cell-aligned.
        "immune_receptors": CellValueSpec(
            "values",
            True,
            {
                "receptor_subtype": True,
                "chain_pairing": True,
                "multichain": True,
                "n_vj": False,
                "n_vdj": False,
                "n_contigs": False,
            },
        ),
        "imported_coordinates": CellValueSpec("data", False),
        "label_transfer": CellValueSpec(
            "labels",
            True,
            {
                "abstention_reason": True,
                "candidate_codes": True,
                "vote_class_codes": True,
                "vote_class_fractions": False,
                "vote_fraction": False,
                "top_two_margin": False,
                "vote_entropy": False,
                "nearest_distance": False,
                "reference_distance_percentile": False,
            },
        ),
        # The record_ arrays hold one row per site of an allele row, or per
        # distinct integration, site or allele, which are not cell-aligned.
        "lineage_alleles": CellValueSpec(
            "n_edits",
            False,
            {"n_integrations": False, "n_records": False},
        ),
        # The record_ arrays hold one row per molecule or barcode pair, or per
        # distinct barcode or UMI, which are not cell-aligned.
        "lineage_barcodes": CellValueSpec(
            "n_barcodes",
            False,
            {"n_umis": False, "n_reads": False, "n_records": False},
        ),
        "lineage_curves": CellValueSpec(
            "progress", False, {"distance": False, "valid": True}
        ),
        "membership_strength": CellValueSpec("values", False),
        # A metadata snapshot is cell-aligned when it records a cell
        # selection, as the custom source/sink vector of pseudotime scoring
        # does. A snapshot of whole metadata columns, such as a pipeline
        # run's, records none: readers refuse it with a ValueError that names
        # the run's frozen fields, and label it corrupt only when its
        # recorded cell selection is malformed.
        "metadata_snapshot": CellValueSpec("values", False),
        # posterior and score are NaN for cells outside a fitted target.
        "mixscape_classes": CellValueSpec(
            "values",
            True,
            {"global_class": True, "posterior": False, "score": False},
        ),
        # target_codes holds one row per cell and one column per target gene,
        # padded to the cells' largest number of targets.
        "perturbation_labels": CellValueSpec(
            "values",
            True,
            {
                "guide_label": True,
                "is_control": True,
                "n_targets": False,
                "target_codes": True,
            },
        ),
        "pseudotime": CellValueSpec("pseudotime", False, {"valid": True}),
        "quality_metric": CellValueSpec("values", False),
        "smart_label": CellValueSpec("values", True),
        # mean_depth is NaN for a selected cell that no imported file names;
        # listed marks the cells that a file names.
        "variant_counts": CellValueSpec(
            "mean_depth", False, {"n_covered_sites": False, "listed": True}
        ),
        # n_mutant_sites counts the sites called heterozygous, homozygous
        # alternative or alternative present; n_called_sites those and the
        # sites called wild type.
        "variant_genotypes": CellValueSpec(
            "n_mutant_sites", False, {"n_called_sites": False}
        ),
    }
)


def _is_missing_label(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_blank_label(value: object) -> bool:
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, bytes | np.bytes_):
        return value.strip() == b""
    return False


def valid_category_mask(
    values: Any,
    *,
    missing_mask: Any | None = None,
) -> np.ndarray:
    """Mark non-missing, non-blank values that can identify categories.

    ``missing_mask`` carries the explicit missingness stored alongside typed
    metadata columns, whose placeholder values may otherwise look valid.
    A typed NumPy array is checked by its dtype: integers and booleans are
    always valid, floats and complex numbers unless NaN, and fixed-width text
    or bytes unless empty or whitespace. Other values, such as an object
    array, are checked one by one.
    """
    array = (
        values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    )
    if array.ndim != 1:
        raise ValueError("category values must be one-dimensional")
    kind = array.dtype.kind
    if kind in "biu":
        valid = np.ones(array.shape, dtype=bool)
    elif kind in "fc":
        valid = ~np.isnan(array)
    elif kind in "US":
        # Text strips the same whitespace as str.strip and bytes.strip.
        valid = ~((np.strings.str_len(array) == 0) | np.strings.isspace(array))
    else:
        valid = np.fromiter(
            (
                not _is_missing_label(value) and not _is_blank_label(value)
                for value in array.astype(object, copy=False)
            ),
            dtype=bool,
            count=array.size,
        )
    if missing_mask is not None:
        missing = np.asarray(missing_mask, dtype=bool)
        if missing.shape != array.shape:
            raise ValueError("missing mask must align with category values")
        valid &= ~missing
    return valid


#: Naming the distinct labels holds, for each label beside its sort key, its
#: slots in the lists of the distinct values, their texts and whether they
#: name a category, and its entry in the table that keeps them in order.
_NAMING_BYTES_PER_LABEL = 50
# Coding values holds, beside pandas' hash table, each value's code from
# pandas, the distinct values' references and, once they are named, whether
# each value holds a label, the codes and their two gathers.
_FACTORIZE_BYTES_PER_VALUE = 16
_CODING_BYTES_PER_VALUE = 33


def hashed_text_bytes(texts: Iterable[str]) -> int:
    """Return the bytes of the UTF-8 forms that hashing leaves in the strings ``texts`` beyond ASCII.

    pandas hashes a string by its UTF-8 form, which CPython keeps in the
    string, with a terminating NUL, for as long as the string lives; an
    ASCII string is its own UTF-8 form. Coding labels therefore leaves this
    much in each label's string beyond ASCII, which the holder of the
    strings charges with them.
    """
    return sum(len(text.encode()) + 1 for text in texts if not text.isascii())


def _naming_bytes(texts: list[str]) -> int:
    """The bytes that naming and ordering the distinct labels ``texts`` holds.

    Beside each label's slots and :func:`~cytearc.utils.arrays.sort_categories_bytes`,
    pandas keeps the UTF-8 form of each text beyond ASCII that it hashed, as
    long as the text lives, which costs about as much as the text.
    """
    sorting = sort_categories_bytes(texts)
    cached = 0
    if not all(map(str.isascii, texts)):
        characters = sum(map(len, texts))
        cached = python_text_bytes(len(texts), characters, widest=0x10FFFF)
    return _NAMING_BYTES_PER_LABEL * len(texts) + sorting + cached


def category_codes(
    values: Any,
    missing_mask: Any | None = None,
    *,
    ledger: MemoryLedger | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Code labels by their text, in the natural order of the texts.

    A value names its category by its text, ``str(value)``, so values that
    read alike share a code. Missing and blank values get -1, as
    :func:`valid_category_mask` marks them, and so do the values that
    ``missing_mask`` marks. Each distinct value is checked and named once, so
    a million labels of a few categories cost one factorization.

    With ``ledger``, what coding holds is charged to it before each step:
    pandas' hash table of the values, as
    :func:`~cytearc.utils.memory_budget.hash_table_bytes` counts it for every
    value, beside each value's code and the distinct values' references; and
    then the codes beside naming and ordering the distinct labels, from their
    texts' characters and widest character, as
    :func:`~cytearc.utils.memory_budget.python_text_bytes` counts them, and
    the tokens of their natural order. The codes and names returned are the
    caller's to charge, and so are the UTF-8 forms that hashing leaves in the
    values' strings beyond ASCII, as :func:`hashed_text_bytes` counts them.

    Returns:
        Each value's code, as int64, and the categories' texts in the order of
        :func:`~cytearc.utils.arrays.sort_categories`, so that code ``i`` names
        the ``i``-th text.

    Raises:
        ValueError: If the values are not one-dimensional or the mask does not
            align with them.
        MemoryBudgetError: If coding the values would pass the ledger's
            budget.
    """
    array = (
        values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    )
    if array.ndim != 1:
        raise ValueError("category values must be one-dimensional")
    n_values = len(array)
    if ledger is not None:
        # Distinct values of an object array are gathered as references.
        gathered = object_array_bytes(n_values) if array.dtype == object else 0
        ledger.peak(
            f"coding {n_values} labels",
            hash_table_bytes(n_values)
            + _FACTORIZE_BYTES_PER_VALUE * n_values
            + gathered,
        )
    try:
        factorized, distinct = pd.factorize(array, use_na_sentinel=True)
    except TypeError:
        # Unhashable labels, such as lists, are coded by their text.
        labelled = valid_category_mask(array)
        as_text = np.full(len(array), None, dtype=object)
        if ledger is not None:
            # Each value's text, measured one at a time, and its hashing.
            characters = sum(len(str(value)) for value in array[labelled].tolist())
            ledger.peak(
                f"coding {n_values} labels by their text",
                object_array_bytes(n_values)
                + python_text_bytes(n_values, characters, widest=0x10FFFF)
                + hash_table_bytes(n_values)
                + (_FACTORIZE_BYTES_PER_VALUE * n_values)
                + object_array_bytes(n_values),
            )
        as_text[labelled] = [str(value) for value in array[labelled].tolist()]
        factorized, distinct = pd.factorize(as_text, use_na_sentinel=True)
    distinct_values = np.asarray(distinct, dtype=object)
    texts = [str(value) for value in distinct_values.tolist()]
    if ledger is not None:
        ledger.peak(
            f"naming {len(texts)} distinct labels",
            _CODING_BYTES_PER_VALUE * n_values + _naming_bytes(texts),
        )
    present = factorized >= 0
    if missing_mask is not None:
        missing = np.asarray(missing_mask, dtype=bool)
        if missing.shape != array.shape:
            raise ValueError("missing mask must align with category values")
        present &= ~missing
    # Only the labels that some value holds name a category.
    held = np.bincount(factorized[present], minlength=len(distinct_values)) > 0
    valid = (held & valid_category_mask(distinct_values)).tolist()
    names = sort_categories(
        dict.fromkeys(text for text, ok in zip(texts, valid, strict=True) if ok)
    )
    rank = {name: position for position, name in enumerate(names)}
    lookup = np.asarray(
        [rank[text] if ok else -1 for text, ok in zip(texts, valid, strict=True)],
        dtype=np.int64,
    )
    codes = np.full(len(array), -1, dtype=np.int64)
    codes[present] = lookup[factorized[present]]
    return codes, names


def sample_codes(
    columns: tuple[str, ...],
    labels: Sequence[np.ndarray],
    n_cells: int,
    *,
    reason: str,
    ledger: MemoryLedger | None = None,
) -> tuple[np.ndarray, list[dict[str, str]], str | None]:
    """Group the selected cells into samples by their label in every column.

    Returns each cell's sample code, numbering the samples in sorted label
    order; each sample's labels by column; and the fingerprint of the cells'
    samples. Without columns every cell is in one sample without labels, and
    the fingerprint is None.

    Args:
        columns: The cell metadata columns whose labels define the samples.
        labels: Each column's label of every selected cell, with missing
            labels already masked.
        n_cells: The number of selected cells.
        reason: Why every selected cell needs a label; it ends the error for
            a cell without one.
        ledger: The caller's memory ledger, charged before each step
            allocates: numbering each column's labels, as
            :func:`label_numbering_bytes` states, and the samples and their
            fingerprint, as :func:`sample_fingerprint_bytes` states, and kept
            charged with the codes it returns. None charges nothing.

    Raises:
        ValueError: If a column does not hold one label per selected cell, or
            a selected cell has a missing or blank label.
        MemoryBudgetError: If a step would pass the ledger's budget.
    """
    if not columns:
        if ledger is not None:
            ledger.hold("the cells' sample codes", 8 * n_cells)
        return np.zeros(n_cells, dtype=np.int64), [{}], None
    names_by_column = []
    codes_by_column = []
    held = 0
    for column, values in zip(columns, labels, strict=True):
        array = np.asarray(values, dtype=object)
        if array.shape != (n_cells,):
            raise ValueError(
                f"Sample column {column!r} must hold one label per selected cell"
            )
        if ledger is not None:
            ledger.peak(
                f"numbering the cells' labels in {column!r}",
                label_numbering_bytes(array),
            )
        missing = int(np.count_nonzero(~valid_category_mask(array)))
        if missing:
            raise ValueError(
                f"Sample column {column!r} has no label for {missing} selected "
                f"cells. {reason}"
            )
        names, codes = np.unique(array.astype(str), return_inverse=True)
        if ledger is not None:
            held += ledger.hold(
                f"the cells' codes in {column!r}", codes.nbytes + names.nbytes
            )
        names_by_column.append(names)
        codes_by_column.append(codes.reshape(-1))
    if ledger is not None:
        ledger.peak(
            "the cells' samples and their fingerprint",
            sample_fingerprint_bytes(n_cells, columns, names_by_column),
        )
    combinations, codes = np.unique(
        np.column_stack(codes_by_column), axis=0, return_inverse=True
    )
    samples = [
        {
            column: str(names[code])
            for column, names, code in zip(columns, names_by_column, row, strict=True)
        }
        for row in combinations
    ]
    rendered = np.asarray([json.dumps(sample) for sample in samples])
    codes = codes.reshape(-1).astype(np.int64, copy=False)
    fingerprint = fingerprint_strings(rendered[codes])
    del codes_by_column, names_by_column, combinations, rendered
    if ledger is not None:
        # The codes of each column and their labels give way to the codes of
        # the samples and the samples' labels.
        ledger.hold(
            "the cells' sample codes and the samples' labels",
            codes.nbytes + _KEPT_BYTES_PER_SAMPLE * len(samples),
        )
        ledger.release(held)
    return codes, samples, fingerprint


#: Numbering a column's labels holds, for each label beside its text at the
#: width of the widest, whether it is valid, its position in the sorted order,
#: the flags of the distinct values and their running count, and its code.
_NUMBERING_BYTES_PER_LABEL = 40
#: Numbering the samples holds, for each cell, its codes stacked by column
#: and their copies while they are sorted, its position in the sorted order,
#: the flags of the distinct samples and their running count, and its code.
_SAMPLE_BYTES_PER_CELL = 48
#: Each sample's labels by column and their text, beside the characters.
_BYTES_PER_SAMPLE = 512
#: Each sample's labels by column, as returned: a dictionary and its texts.
_KEPT_BYTES_PER_SAMPLE = 640


def label_numbering_bytes(labels: np.ndarray) -> int:
    """Return the most bytes that :func:`sample_codes` holds to number one column's ``labels``, beside them.

    The labels become text at the width of the widest, a number as its text,
    which a sort copies twice, as NumPy's ``np.unique`` sorts them, beside
    the distinct labels, as many as the labels at most.
    """
    array = np.asarray(labels, dtype=object)
    # The labels' text, as numbering makes it: a number's text is measured
    # one at a time, as it is made and let go.
    width = max(
        (len(value) if isinstance(value, str) else len(str(value)) for value in array),
        default=1,
    )
    n_values = len(array)
    return 4 * fixed_text_bytes(n_values, max(1, width)) + (
        _NUMBERING_BYTES_PER_LABEL * n_values
    )


def sample_fingerprint_bytes(
    n_cells: int, columns: Sequence[str], names: Sequence[np.ndarray]
) -> int:
    """Return the most bytes that :func:`sample_codes` holds to number the cells' samples and fingerprint them.

    ``names`` holds each column's distinct labels, which bound the samples.
    Each cell's sample is rendered as JSON text at the width of the widest,
    which the fingerprint copies, beside each column's code and the
    samples' own labels and texts.
    """
    n_samples = min(int(n_cells), int(np.prod([max(1, len(each)) for each in names])))
    # The widest JSON text of a sample: each column's name and its widest
    # label, quoted and escaped, and the ": " between them, separated by ", "
    # between braces.
    width = 4 * len(columns) + sum(
        len(json.dumps(column))
        + max((len(json.dumps(str(name))) for name in each), default=2)
        for column, each in zip(columns, names, strict=True)
    )
    return (
        2 * fixed_text_bytes(n_cells, width)
        + (_SAMPLE_BYTES_PER_CELL + 8 * len(columns)) * int(n_cells)
        + n_samples * (_BYTES_PER_SAMPLE + 4 * width)
    )


@dataclass(frozen=True, slots=True)
class FeatureRef:
    """Reference to one assay feature.

    Parameters:
        value: Feature name, id, or physical index (see ``by``).
        assay: Assay name. Defaults to the store default assay when omitted.
        by: How to look up ``value``: ``name``, ``id``, or ``index``.
        label: Optional display label.
        reduction: Required when multiple features match; ``mean`` or ``sum``.
    """

    value: str | int
    assay: str | None = None
    by: LookupBy = "name"
    label: str | None = None
    reduction: FeatureReduction | None = None

    def __post_init__(self) -> None:
        if self.by not in ("name", "id", "index"):
            raise ValueError("by must be 'name', 'id', or 'index'")
        if self.reduction not in (None, "mean", "sum"):
            raise ValueError("reduction must be 'mean', 'sum', or None")
        if self.by == "index":
            # Resolution checks the index against the assay's feature count.
            value: str | int = integer_argument(self.value, "FeatureRef value")
        elif isinstance(self.value, str):
            value = str(self.value)
        else:
            raise TypeError(f"FeatureRef value must be a string when by is {self.by!r}")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class CellField:
    """Point ``color_by`` or similar arguments at a cell-metadata column.

    Use this when the column name alone is ambiguous. ``kind="categorical"``
    forces discrete colors (useful for integer cluster ids).
    ``kind="continuous"`` forces a colorbar. ``kind="auto"`` chooses from the
    dtype and number of unique values.
    """

    key: str
    kind: CellFieldKind = "auto"
    label: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("auto", "categorical", "continuous"):
            raise ValueError("kind must be 'auto', 'categorical', or 'continuous'")


@dataclass(frozen=True, slots=True)
class NamedCellArtifact:
    """One semantic name bound to an exact cell-aligned artifact."""

    name: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Named cell artifacts require a non-empty name")
        if self.name != self.name.strip():
            raise ValueError(
                "Named cell artifact names cannot have surrounding whitespace"
            )
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("Named cell artifacts require an ArtifactRef")


@dataclass(frozen=True, slots=True)
class ResolvedCellArtifact:
    """Artifact values aligned to one validated cell selection.

    ``missing_mask`` flags rows whose stored value is a placeholder for a
    missing entry. It is None when the artifact records no missing values.
    """

    source: ArtifactRef
    values: np.ndarray
    cell_idx: np.ndarray
    source_cell_selection: ArtifactRef
    cell_selection: ArtifactRef
    missing_mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ResolvedGrouping:
    """Categorical labels aligned to exact physical cell rows."""

    source: ArtifactRef | CellField
    labels: np.ndarray
    cell_idx: np.ndarray
    cell_selection: ArtifactRef | None
    missing_mask: np.ndarray | None


@dataclass(frozen=True, slots=True, eq=False)
class CellValues:
    """The per-cell values of one cell-aligned artifact or cell column, aligned to cells.

    Attributes:
        source: The artifact, or the ``CellField`` of the cell column, that
            holds the values.
        value: The name of the array or column that was read.
        values: One value, or one row of values, per cell.
        cell_ids: The ``ids`` of the cells, aligned to ``values``.
        cell_idx: The row of each cell in the cell table, in ascending order.
        cell_selection: The cell selection that the rows follow.
        missing: True for each cell whose value is missing, or None.
        categorical: Whether the values are labels that group cells.
    """

    source: ArtifactRef | CellField
    value: str
    values: np.ndarray
    cell_ids: np.ndarray
    cell_idx: np.ndarray
    cell_selection: ArtifactRef
    missing: np.ndarray | None
    categorical: bool

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Return the values indexed by cell id, showing missing rows as missing."""
        index = pd.Index(self.cell_ids, name="ids")
        values = np.asarray(self.values)
        if values.ndim == 1:
            return pd.Series(
                apply_missing_mask(values, self.missing, labels=self.categorical),
                index=index,
                name=self.value,
            )
        if values.ndim != 2:
            raise ValueError("to_pandas needs one or two dimensions of values")
        missing = (
            None
            if self.missing is None
            else np.broadcast_to(np.asarray(self.missing)[:, None], values.shape)
        )
        return pd.DataFrame(
            apply_missing_mask(values, missing, labels=self.categorical),
            index=index,
            columns=pd.RangeIndex(values.shape[1], name=self.value),
        )


def cell_value_spec(kind: str) -> CellValueSpec:
    """Return the per-cell arrays of a cell-aligned artifact kind."""
    try:
        return CELL_VALUE_NAMES[kind]
    except KeyError:
        raise ValueError(
            f"{kind!r} is not a cell-aligned artifact kind; the cell-aligned "
            f"kinds are {', '.join(sorted(CELL_VALUE_NAMES))}"
        ) from None


def cell_value_array(kind: str, value: str | None = None) -> tuple[str, bool]:
    """Return the per-cell array of ``kind`` to read and whether it holds labels."""
    if value is not None and not isinstance(value, str):
        raise TypeError("value must be a string or None")
    spec = cell_value_spec(kind)
    if value is None or value == spec.name:
        return spec.name, spec.categorical
    if value in spec.alternatives:
        return value, spec.alternatives[value]
    choices = ", ".join((spec.name, *spec.alternatives))
    raise ValueError(
        f"{kind!r} artifacts have no per-cell array {value!r}; their per-cell "
        f"arrays are {choices}"
    )


def grouping_value_name(kind: str) -> str:
    """Return the canonical categorical-label array for an artifact kind."""
    spec = CELL_VALUE_NAMES.get(kind)
    if spec is None or not spec.categorical:
        raise ValueError("Grouping artifacts must contain categorical cell labels")
    return spec.name


def require_complete_cluster_labels(
    group: Any,
    value_name: str,
    *,
    name: str,
    values: Any | None = None,
) -> None:
    """Reject stored cluster labels whose linked mask flags a missing label.

    Imported clusterings store a placeholder for each missing label, so
    consumers that treat every row as a member of a cluster must refuse them.
    """
    try:
        mask = linked_missing_mask(group, value_name, values=values)
    except ValueError as error:
        raise ValueError(f"{name} has a malformed missing-label mask") from error
    if mask is None:
        return
    n_rows = int(mask.shape[0])
    band = scan_band(array_geometry(mask), fallback=max(1, n_rows))
    if any(
        np.asarray(mask[start : start + band]).any() for start in range(0, n_rows, band)
    ):
        raise ValueError(f"{name} contains missing cluster labels")


def cell_selection_indices(root: Any, selection: ArtifactRef) -> np.ndarray:
    """Return the store positions of a cell selection's cells, ascending, as int64.

    The selection is checked against the cell table first.
    """
    return read_stored_selection_indices(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)


# Bytes of one int64 cell row or position.
_INDEX_BYTES = np.dtype(np.int64).itemsize


@dataclass(frozen=True, slots=True)
class CellSelectionRead:
    """A cell selection checked and counted before its cells are read.

    Attributes:
        selection: The cell selection.
        cells: The cells it selects.
        geometry: The geometry of its stored flags, or None when they have
            no chunks.
        rows: The rows of its flags, one for each cell of the store.
    """

    selection: ArtifactRef
    cells: int
    geometry: ArrayGeometry | None
    rows: int


def plan_cell_selection_read(
    root: Any, selection: ArtifactRef, *, cells: int | None = None
) -> CellSelectionRead:
    """Check a cell selection and count its cells, without reading their positions.

    The selection's identity check reads its flags and the cell table's row
    identifiers outside any budget, once in the operation's validation
    scope, so an operation plans its reads before it charges anything. A
    caller that knows the cells that the selection holds, as an artifact
    aligned to it records them, passes them as ``cells``; the identity check
    then runs as :func:`read_cell_selection` reads the positions.
    """
    n_selected = _selection_count(root, selection) if cells is None else int(cells)
    values = as_zarr_array(artifact_group(root, selection)["values"], name="values")
    return CellSelectionRead(
        selection=selection,
        cells=n_selected,
        geometry=array_geometry(values),
        rows=int(values.shape[0]),
    )


def read_cell_selection(
    ledger: MemoryLedger, root: Any, planned: CellSelectionRead, *, item: str
) -> np.ndarray:
    """Read the store positions of a planned cell selection's cells, keeping them charged as ``item``.

    :func:`cell_selection_indices` reads the flags, one for each cell of the
    store, in bands of whole chunks up to 64 MiB, so the read holds them as a
    whole read at the operation's Zarr concurrency does, as
    ``ArrayGeometry.wholeReadBytes`` counts it, beside each selected cell's
    position as found, moved and returned; a stored chunk can outweigh the
    cells it selects.

    Raises:
        MemoryBudgetError: If the read would pass the ledger's budget.
    """
    ledger.peak(item, _selection_read_bytes(planned))
    # The flags decode on the ledger's codec threads, as every read that a
    # ledger charges does.
    with zarr_decoding_threads(ledger.codec_threads):
        indices = cell_selection_indices(root, planned.selection)
    ledger.hold(item, indices.nbytes)
    return indices


def _selection_read_bytes(planned: CellSelectionRead) -> int:
    """What :func:`read_cell_selection` charges to read ``planned``'s positions."""
    positions = 3 * _INDEX_BYTES * planned.cells
    geometry = planned.geometry
    if geometry is None:
        return planned.rows + positions
    return geometry.wholeReadBytes(operation_concurrency()) + positions


def _selection_count(root: Any, selection: ArtifactRef) -> int:
    """Validate a cell selection and count its cells without reading its rows."""
    return int(validate_cell_selection(root, selection).selected_count)


def _refuse_cell_read(
    ledger: MemoryLedger, required: int, *, item: str, n_cells: int
) -> None:
    """Refuse a read of ``item`` for ``n_cells`` cells whose ``required`` bytes pass the ledger's budget.

    The refusal takes the ledger's words and remedy, so each caller names
    what it does and how its call can read less.
    """
    ledger.refuse_beyond(
        f"reading {item} for {n_cells} cells", ledger.held_bytes + required
    )


def _reads_in_turn_bytes(arrays: Sequence[Any], n_rows: int) -> int:
    """The most that reading ``n_rows`` rows of each array in turn holds.

    Each read holds what :func:`~cytearc.metadata.rows.read_rows_bytes`
    states, the codes and categories of a categorical column, beside the
    rows of the arrays read before it, which
    :func:`~cytearc.metadata.rows.read_rows` keeps charged.
    """
    kept = 0
    most = 0
    for array in arrays:
        most = max(most, kept + read_rows_bytes(array, n_rows))
        row_values = int(np.prod(array.shape[1:], dtype=np.int64))
        kept += n_rows * row_values * np.dtype(array.dtype).itemsize
    return most


def _unaligned_snapshot_error(
    artifact: ArtifactRef,
    status: ArtifactStatus,
) -> ValueError:
    """Refuse a metadata snapshot that records no cell selection.

    Such a snapshot is valid but holds whole metadata columns, one row per
    row of its table, rather than a row per cell of a recorded selection.
    """
    parameters = status.parameters or {}
    axis = parameters.get("axis")
    raw_columns = parameters.get("ordered_columns")
    columns = (
        ", ".join(repr(column) for column in raw_columns)
        if isinstance(raw_columns, list)
        else None
    )
    if status.operation == "snapshot_run_metadata" and axis == "cell":
        what = f"the cell table columns {columns}"
        remedy = (
            "Read them as the frozen fields of the pipeline run that recorded "
            "the snapshot, with run.cells.fetch(column) or "
            "run.cells.to_pandas_dataframe(columns); a plot with run=run takes "
            "a frozen field by name, such as color_by=column"
        )
    elif status.operation == "snapshot_run_metadata" and axis == "feature":
        what = f"the {artifact.assay} feature table columns {columns}"
        remedy = (
            "Read them as the frozen feature fields of the pipeline run that "
            "recorded the snapshot, with run.features.fetch(column) or "
            "run.features.to_pandas_dataframe(columns)"
        )
    else:
        what = "metadata values"
        remedy = "Open it with artifacts.load"
    return ValueError(
        f"The metadata_snapshot artifact {artifact.artifact_id} holds {what} "
        "and records no cell selection, so its rows are not aligned to a "
        "recorded cell selection and it holds no cell-aligned values. "
        f"{remedy}."
    )


def _recorded_cell_selection(
    artifact: ArtifactRef,
    status: ArtifactStatus,
) -> ArtifactRef:
    """Return the cell selection that a cell-aligned artifact records.

    A metadata snapshot without a ``cell_selection`` input is not
    cell-aligned and raises ``ValueError``. For every other kind, and for a
    malformed recorded selection, the one strict reader of recorded inputs
    raises ``ArtifactResolutionError`` with code ``corrupt_payload``.
    """
    inputs = status.inputs or {}
    if artifact.kind == "metadata_snapshot" and "cell_selection" not in inputs:
        raise _unaligned_snapshot_error(artifact, status)
    # The one strict reader of recorded inputs, as ArtifactStatus.input_ref.
    return parse_artifact_ref(
        inputs.get("cell_selection"),
        "cell_selection",
        owner=artifact,
    )


def _row_missing_mask(mask: np.ndarray, value_name: str) -> np.ndarray:
    """Reduce a linked mask to one entry per row, which must flag whole rows."""
    if mask.ndim == 1:
        return mask
    flat = mask.reshape(mask.shape[0], -1)
    flagged = flat.any(axis=1)
    if not bool(flat[flagged].all()):
        raise ValueError(
            f"Cell-aligned artifact array {value_name!r} marks only some values "
            "of a cell as missing"
        )
    return flagged


def resolve_cell_aligned_artifact(
    root: Any,
    artifact: ArtifactRef,
    *,
    cell_selection: ArtifactRef | None = None,
    value_name: str | None = None,
    expected_kind: str | None = None,
    ndim: int | None = 1,
    ledger: MemoryLedger | None = None,
    caller_reads: Sequence[Any] = (),
) -> ResolvedCellArtifact:
    """Read one artifact array in the exact requested cell order.

    With ``ledger``, what the read holds is charged to it: the cell positions
    of the selections, each read through :func:`read_cell_selection`, which
    charges the flags and positions that the read holds, and each array read
    through :func:`~cytearc.metadata.rows.read_rows` at the selected rows,
    the values, their missing mask, and ``caller_reads``, arrays that the
    caller reads at the same rows. The arrays are read in turn, so each read
    is charged beside the rows of the arrays read before it, not beside their
    reads; the largest of these is checked before anything is read, and again
    once the positions are read, before any array is. A refusal takes the
    ledger's words and remedy. The values, the mask and the positions of the
    cells stay charged when the read returns. Without a ledger, the read
    checks no budget.

    Raises:
        MemoryBudgetError: If ``ledger`` cannot hold what the read holds.
    """
    if not isinstance(artifact, ArtifactRef):
        raise TypeError("artifact must be an ArtifactRef")
    if expected_kind is not None and artifact.kind != expected_kind:
        raise ValueError(
            f"Expected a {expected_kind!r} artifact, received {artifact.kind!r}"
        )
    if value_name is not None and (not isinstance(value_name, str) or not value_name):
        raise ValueError("value_name must be a non-empty string")
    if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
        raise TypeError("cell_selection must be an ArtifactRef")
    if ndim is not None and (type(ndim) is not int or ndim < 1):
        raise ValueError("ndim must be a positive integer or None")
    value_name, _ = cell_value_array(artifact.kind, value_name)

    status = inspect_artifact(root, artifact)
    if not status.exists or not status.complete:
        raise ValueError("Cell-aligned artifact is unavailable or incomplete")
    source_selection = _recorded_cell_selection(artifact, status)
    target_selection = source_selection if cell_selection is None else cell_selection
    same_selection = target_selection == source_selection

    group = artifact_group(root, artifact)
    if value_name not in group:
        raise ValueError(f"Cell-aligned artifact has no {value_name!r} value array")
    values_array = as_zarr_array(group[value_name], name=value_name)
    shape = tuple(int(extent) for extent in values_array.shape)
    # The selections are validated once: counting their cells here and
    # reading their rows below share one validation.
    with validation_scope():
        n_source = _selection_count(root, source_selection)
        if (
            not shape
            or (ndim is not None and len(shape) != ndim)
            or shape[0] != n_source
        ):
            unit = "value" if ndim == 1 else "row"
            raise ValueError(
                f"Cell-aligned artifact must contain one {unit} per "
                "source-selected cell"
            )
        missing_array = linked_missing_mask(
            group,
            value_name,
            label=f"Cell-aligned artifact array {value_name!r}",
            values=values_array,
        )
        item = f"{value_name!r} of the {artifact.kind} artifact"
        # The order of the reads: the values, their mask, then the arrays
        # that the caller reads at the same rows.
        row_arrays = [values_array]
        if missing_array is not None:
            row_arrays.append(missing_array)
        row_arrays.extend(caller_reads)
        budgeted = ledger is not None
        planned_source = planned_target = None
        if ledger is None:
            # A read without a budget still reads through the ledger's reads.
            ledger = MemoryLedger(None, f"Reading {item}")
        else:
            # One check covers each step of the read before anything is read,
            # so a refusal names the budget that reads it all: each
            # selection's read of its positions, beside the source cells'
            # positions when the requested cells differ, then the places of
            # the requested cells among the source's as they are found and
            # checked, and the rows beside the positions kept.
            planned_source = plan_cell_selection_read(root, source_selection)
            source_read = _selection_read_bytes(planned_source)
            if same_selection:
                n_target = n_source
                steps = [source_read]
            else:
                planned_target = plan_cell_selection_read(root, target_selection)
                n_target = planned_target.cells
                steps = [
                    source_read,
                    _INDEX_BYTES * n_source + _selection_read_bytes(planned_target),
                    _INDEX_BYTES * n_source + (3 * _INDEX_BYTES + 2) * n_target,
                ]
            steps.append(
                2 * _INDEX_BYTES * n_target + _reads_in_turn_bytes(row_arrays, n_target)
            )
            _refuse_cell_read(ledger, max(steps), item=item, n_cells=n_target)

        positions = f"the cell positions of {item}"
        source_idx = _read_positions(
            ledger if budgeted else None,
            root,
            source_selection,
            planned_source,
            item=positions,
        )
        if same_selection:
            target_idx = source_idx
            if budgeted:
                ledger.hold(positions, _INDEX_BYTES * len(source_idx))
            compact_idx = np.arange(len(source_idx), dtype=np.int64)
        else:
            target_idx = _read_positions(
                ledger if budgeted else None,
                root,
                target_selection,
                planned_target,
                item=positions,
            )
            if budgeted:
                # The requested cells' places among the artifact's cells.
                ledger.hold(positions, _INDEX_BYTES * len(target_idx))
            compact_idx = np.searchsorted(source_idx, target_idx).astype(
                np.int64,
                copy=False,
            )
            if budgeted:
                # Whether each lies within them, the positions found there
                # and whether they agree, as the places are checked.
                ledger.peak(positions, (_INDEX_BYTES + 2) * len(target_idx))
            in_bounds = compact_idx < len(source_idx)
            if not bool(in_bounds.all()) or not np.array_equal(
                source_idx[compact_idx], target_idx
            ):
                raise ValueError(
                    "cell_selection must be a subset of the artifact cell selection"
                )
            # Only the positions of the requested cells are read from here.
            source_bytes = source_idx.nbytes
            del source_idx, in_bounds
            if budgeted:
                ledger.release(source_bytes)
    if budgeted:
        # Every array is read at the rows of the requested cells, in turn.
        required = _reads_in_turn_bytes(row_arrays, compact_idx.size)
        _refuse_cell_read(ledger, required, item=item, n_cells=len(target_idx))
        ledger.peak(item, required)
    values = read_rows(ledger, values_array, compact_idx, item=item)
    if values.shape != (len(target_idx), *shape[1:]):
        raise ValueError("Cell-aligned artifact values do not match the selection")
    missing = (
        None
        if missing_array is None
        else _row_missing_mask(
            np.asarray(
                read_rows(
                    ledger,
                    missing_array,
                    compact_idx,
                    item=f"the missing mask of {item}",
                ),
                dtype=bool,
            ),
            value_name,
        )
    )
    if budgeted:
        # The requested cells' positions in the store stay with the values.
        compact_bytes = compact_idx.nbytes
        del compact_idx
        ledger.release(compact_bytes)
    return ResolvedCellArtifact(
        source=artifact,
        values=values,
        cell_idx=target_idx,
        source_cell_selection=source_selection,
        cell_selection=target_selection,
        missing_mask=missing,
    )


def _read_positions(
    ledger: MemoryLedger | None,
    root: Any,
    selection: ArtifactRef,
    planned: CellSelectionRead | None,
    *,
    item: str,
) -> np.ndarray:
    """The store positions of a selection's cells, read as ``planned`` through :func:`read_cell_selection` with ``ledger``, which keeps them charged."""
    if ledger is None or planned is None:
        return cell_selection_indices(root, selection)
    return read_cell_selection(ledger, root, planned, item=item)


def resolve_cell_column(
    root: Any,
    cells: Any,
    column: str,
    *,
    cell_selection: ArtifactRef | None,
    ledger: MemoryLedger | None = None,
    caller_reads: Sequence[Any] = (),
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Read one cell metadata column and its linked missing mask at the cells of a selection.

    Without ``cell_selection``, every row of the cell table is read. With
    ``ledger``, the read is charged as :func:`resolve_cell_aligned_artifact`
    charges an artifact's: the selection's flags and positions, read through
    :func:`read_cell_selection`, then the column's rows, its mask's, and
    ``caller_reads``, arrays that the caller reads at the same rows, each
    beside the rows read before it; the largest step is checked before
    anything is read, and the values, the mask and the positions stay
    charged. Without a ledger, the read checks no budget.

    Returns:
        The values, the missing mask or None, and the rows of the cells in
        the cell table, ascending, as int64. The caller checks that the
        values and the mask hold one row per cell.

    Raises:
        MemoryBudgetError: If ``ledger`` cannot hold what the read holds.
    """
    if ledger is None:
        cell_idx = (
            np.arange(cells.N, dtype=np.int64)
            if cell_selection is None
            else cell_selection_indices(root, cell_selection)
        )
        values = np.asarray(read_metadata_rows(cells, column, cell_idx))
        missing = read_metadata_missing_rows(cells, column, cell_idx)
    else:
        values_array = cells._get_array(column)
        mask_array = metadata_missing_mask(cells, column)
        row_arrays = [values_array]
        if mask_array is not None:
            row_arrays.append(mask_array)
        row_arrays.extend(caller_reads)
        item = f"the cell column {column!r}"
        positions = f"the cell positions of {item}"
        # The selection is validated once: counting its cells here and
        # reading their rows below share one validation.
        with validation_scope():
            if cell_selection is None:
                n_cells = int(cells.N)
                selection_read = _INDEX_BYTES * n_cells
            else:
                planned = plan_cell_selection_read(root, cell_selection)
                n_cells = planned.cells
                selection_read = _selection_read_bytes(planned)
            rows_read = 2 * _INDEX_BYTES * n_cells + _reads_in_turn_bytes(
                row_arrays, n_cells
            )
            _refuse_cell_read(
                ledger, max(selection_read, rows_read), item=item, n_cells=n_cells
            )
            if cell_selection is None:
                cell_idx = np.arange(n_cells, dtype=np.int64)
                ledger.hold(positions, cell_idx.nbytes)
            else:
                cell_idx = read_cell_selection(ledger, root, planned, item=positions)
        values = read_rows(ledger, values_array, cell_idx, item=item)
        missing = (
            None
            if mask_array is None
            else read_rows(
                ledger, mask_array, cell_idx, item=f"the missing mask of {item}"
            )
        )
    missing_mask = None if missing is None else np.asarray(missing, dtype=bool)
    return values, missing_mask, cell_idx


def resolve_grouping(
    root: Any,
    cells: Any,
    grouping: ArtifactRef | CellField,
    *,
    cell_selection: ArtifactRef | None = None,
    ledger: MemoryLedger | None = None,
) -> ResolvedGrouping:
    """Resolve categorical labels from an artifact or an explicit metadata field.

    With ``ledger``, the labels are read within it, an artifact's as
    :func:`resolve_cell_aligned_artifact` charges them and a metadata
    field's as :func:`resolve_cell_column` charges them.
    """
    if isinstance(grouping, CellField):
        if grouping.kind == "continuous":
            raise ValueError("Grouping CellField must be categorical")
        labels, missing_mask, cell_idx = resolve_cell_column(
            root, cells, grouping.key, cell_selection=cell_selection, ledger=ledger
        )
        if labels.shape != (len(cell_idx),):
            raise ValueError("Grouping metadata does not align with selected cells")
        if missing_mask is not None and missing_mask.shape != labels.shape:
            raise ValueError("Grouping missing mask does not align with selected cells")
        return ResolvedGrouping(
            source=grouping,
            labels=labels,
            cell_idx=cell_idx,
            cell_selection=cell_selection,
            missing_mask=missing_mask,
        )

    if not isinstance(grouping, ArtifactRef):
        raise TypeError("grouping must be an ArtifactRef or CellField")
    value_name = grouping_value_name(grouping.kind)
    resolved = resolve_cell_aligned_artifact(
        root,
        grouping,
        cell_selection=cell_selection,
        value_name=value_name,
        expected_kind=grouping.kind,
        ledger=ledger,
    )

    return ResolvedGrouping(
        source=grouping,
        labels=resolved.values,
        cell_idx=resolved.cell_idx,
        cell_selection=resolved.cell_selection,
        missing_mask=resolved.missing_mask,
    )


def resolve_complete_labels(
    root: Any,
    labels: ArtifactRef,
    *,
    name: str,
    remedy: str | None = None,
) -> ResolvedCellArtifact:
    """Resolve artifact labels that must assign every selected cell to a group.

    Producers that persist a per-group result cannot treat a stored
    placeholder as a label, so labels whose linked mask flags a row are
    rejected before any result is reused or written. The error ends with
    ``remedy``, which says how to make labels that the caller accepts. The
    default narrows the labels to their labelled cells, which suits a
    caller that takes labels over any cell selection.
    """
    if not isinstance(labels, ArtifactRef):
        raise TypeError(f"{name} must be an ArtifactRef")
    resolved = resolve_cell_aligned_artifact(
        root,
        labels,
        value_name=grouping_value_name(labels.kind),
        expected_kind=labels.kind,
    )
    if resolved.missing_mask is not None and bool(resolved.missing_mask.any()):
        if remedy is None:
            remedy = (
                f"Select the labelled cells with select_cells({name}, include=[...]) "
                f"and freeze their labels with snapshot_cluster_labels({name}, "
                "cell_selection=...)"
            )
        raise ValueError(f"{name} contains missing labels. {remedy}")
    return resolved


@dataclass(frozen=True, slots=True)
class StudyDesign:
    """Describe samples and conditions for composition and summary plots.

    ``sample_by`` is the column that identifies biological samples.
    ``condition_by`` is the experimental condition (for example treatment).
    For paired composition plots, also set ``subject_by`` or ``pair_by`` so the
    same donor or pair can be connected across conditions.
    """

    sample_by: str
    condition_by: str | None = None
    subject_by: str | None = None
    pair_by: str | None = None
    technical_replicate_by: str | None = None
    technical_replicate_reduction: Literal["sum", "mean"] | None = None

    def __post_init__(self) -> None:
        unsupported = [
            name
            for name, value in (
                ("technical_replicate_by", self.technical_replicate_by),
                (
                    "technical_replicate_reduction",
                    self.technical_replicate_reduction,
                ),
            )
            if value is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "These StudyDesign fields are not supported yet: "
                + ", ".join(unsupported)
                + ". Collapse technical replicates into sample_by first, "
                "or omit them."
            )


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """How feature values are read for gene-colored plots.

    ``source="assay"`` uses the assay's current normalization settings.
    ``source="raw"`` reads the assay's stored values: its raw counts, or the
    values of an assay imported from normalized values.
    ``transform="log1p"`` applies log1p after that fetch, which is the usual
    choice for gene UMAPs and dotplots when you want a compressed expression
    scale. With ``source="assay"`` it needs a normalizer that supports
    ``log_transform``.
    """

    source: NormSource = "assay"
    transform: NormTransform = "none"

    def __post_init__(self) -> None:
        if self.source not in ("assay", "raw"):
            raise ValueError("source must be 'assay' or 'raw'")
        if self.transform not in ("none", "log1p"):
            raise ValueError("transform must be 'none' or 'log1p'")
