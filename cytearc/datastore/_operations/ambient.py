"""Ambient RNA operations: soup profiles, contamination fractions, and corrected counts.

The computation lives in :mod:`cytearc.quality_control.ambient`; these methods
stream counts, plan and persist the artifacts, and write the corrected assay.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import zarr

from ...assay import RNAassay
from ...features.values import spread_measured_rows
from ...metadata.artifacts import plan_cell_data_artifact
from ...metadata.membership import measured_rows, resolve_assay_membership
from ...metadata.rows import iter_metadata_column_blocks
from ...metadata.selection import resolve_cell_aligned_artifact, resolve_grouping
from ...quality_control.ambient import (
    AmbientContamination,
    ContaminationFit,
    estimate_contamination,
    estimation_cells,
    expressing_cells,
    gene_set_evidence,
    remove_ambient_counts,
    soup_fractions,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
)
from ...storage.artifacts import (
    ArtifactStatus,
    artifact_group,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    require_complete_artifact,
)
from ...storage.identity import read_dataset_fingerprint
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import float_argument, integer_argument
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _AmbientOperationsBase
else:
    _AmbientOperationsBase = object


_PROFILE_ARRAYS = ("counts", "fraction")
_EVIDENCE_ARRAYS = ("observed", "expected", "p_values", "expressing", "used")
# Bytes that ``remove_ambient_counts`` holds for each nonzero count it corrects,
# its sorted copies and running sums included.
_REMOVAL_ENTRY_BYTES = 256


def _counts_assay(store: Any, name: str | None, operation: str) -> RNAassay:
    """Return an RNA assay that stores raw UMI counts."""
    assay = store._get_assay(name)
    if not isinstance(assay, RNAassay):
        raise TypeError(
            f"{operation} needs an RNA assay; assay {assay.name!r} is a "
            f"{type(assay).__name__}"
        )
    dtype = np.dtype(assay.rawData.dtype)
    if dtype.kind != "u":
        raise TypeError(
            f"{operation} needs raw UMI counts, which CyteArc stores as unsigned "
            f"integers; assay {assay.name!r} stores {dtype} values"
        )
    return assay


def _feature_ids_fingerprint(assay: RNAassay) -> str:
    feature_data = as_zarr_group(assay.z["featureData"], name="featureData")
    return fingerprint_stored_strings(
        as_zarr_array(feature_data["ids"], name="featureData/ids")
    )


def _count_range(value: object) -> tuple[float, float]:
    """Return the inclusive total-count bounds of empty droplets."""
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise TypeError("total_counts must be a (low, high) pair of numbers")
    bounds = list(value)
    if len(bounds) != 2:
        raise ValueError("total_counts must be a (low, high) pair of numbers")
    low = float_argument(bounds[0], "total_counts")
    high = float_argument(bounds[1], "total_counts")
    if low < 0 or high < low:
        raise ValueError("total_counts needs 0 <= low <= high")
    return low, high


def _probability(value: object, name: str, *, upper_inclusive: bool) -> float:
    resolved = float_argument(value, name)
    in_range = 0.0 < resolved <= 1.0 if upper_inclusive else 0.0 < resolved < 1.0
    if not in_range:
        bound = "at most 1" if upper_inclusive else "less than 1"
        raise ValueError(f"{name} must be greater than 0 and {bound}")
    return resolved


def _resolve_gene_sets(assay: RNAassay, gene_sets: object) -> dict[str, np.ndarray]:
    """Return the sorted feature indices of each gene set, keyed in name order.

    Gene names match feature names without regard to case, and a name that
    several features share selects all of them.
    """
    if not isinstance(gene_sets, Mapping):
        raise TypeError("gene_sets must map gene-set names to gene names")
    if not gene_sets:
        raise ValueError("gene_sets must name at least one gene set")
    if not all(isinstance(name, str) and name for name in gene_sets):
        raise TypeError("gene_sets must be keyed by non-empty gene-set names")
    by_name: dict[str, list[int]] = {}
    for index, name in enumerate(np.asarray(assay.feats.fetch_all("names"))):
        by_name.setdefault(str(name).upper(), []).append(index)
    resolved: dict[str, np.ndarray] = {}
    owners: dict[int, str] = {}
    for set_name in sorted(gene_sets):
        genes = gene_sets[set_name]
        if isinstance(genes, str | bytes) or not isinstance(genes, Iterable):
            raise TypeError(f"Gene set {set_name!r} must be a collection of gene names")
        names = list(genes)
        if not all(isinstance(gene, str) for gene in names):
            raise TypeError(f"Gene set {set_name!r} must hold gene names")
        unmatched = [gene for gene in names if gene.upper() not in by_name]
        if unmatched:
            logger.warning(
                f"{len(unmatched)} of {len(names)} genes of gene set {set_name!r} "
                f"are not assay features: {', '.join(unmatched)}"
            )
        indices = sorted(
            {index for gene in names for index in by_name.get(gene.upper(), ())}
        )
        if not indices:
            raise ValueError(
                f"No gene of gene set {set_name!r} matches the assay feature names"
            )
        shared = sorted({owners[index] for index in indices if index in owners})
        if shared:
            raise ValueError(
                f"Gene set {set_name!r} shares genes with gene set {shared[0]!r}; "
                "each gene may belong to one gene set only"
            )
        owners.update(dict.fromkeys(indices, set_name))
        resolved[set_name] = np.asarray(indices, dtype=np.int64)
    return resolved


def _write_artifact(
    root: zarr.Group,
    planned: PlannedArtifact,
    arrays: Mapping[str, np.ndarray],
    attributes: Mapping[str, Any],
) -> None:
    """Write the arrays and attributes of a planned artifact and publish it."""
    with artifact_transaction(root, planned) as group:
        for name, values in arrays.items():
            chunks = (min(max(int(values.shape[0]), 1), 100_000), *values.shape[1:])
            output = create_zarr_dataset(
                group, name, chunks, values.dtype, values.shape
            )
            output[...] = values
        group.attrs.update(dict(attributes))


def _estimate_record(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


class _AmbientOperationsMixin(_AmbientOperationsBase):
    def _ambient_profile_fractions(
        self, profile: ArtifactRef, operation: str
    ) -> tuple[RNAassay, zarr.Group, np.ndarray]:
        """Return the assay, group, and soup fractions of a complete profile."""
        if not isinstance(profile, ArtifactRef):
            raise TypeError("profile must be an ArtifactRef")
        if profile.kind != "ambient_profile":
            raise ValueError(
                f"profile must be an ambient_profile artifact, not {profile.kind!r}"
            )
        require_complete_artifact(self.zw, profile)
        assay = _counts_assay(self, profile.assay, operation)
        group = artifact_group(self.zw, profile)
        if group.attrs.get("ordered_feature_ids_fingerprint") != (
            _feature_ids_fingerprint(assay)
        ):
            raise ValueError(
                f"The ambient profile does not match the features of assay "
                f"{assay.name!r}"
            )
        fractions = np.asarray(
            as_zarr_array(group["fraction"], name="fraction")[:], dtype=np.float64
        )
        return assay, group, fractions

    def _ambient_contamination_status(
        self, contamination: ArtifactRef
    ) -> ArtifactStatus:
        if not isinstance(contamination, ArtifactRef):
            raise TypeError("contamination must be an ArtifactRef")
        if contamination.kind != "ambient_contamination":
            raise ValueError(
                "contamination must be an ambient_contamination artifact, not "
                f"{contamination.kind!r}"
            )
        return require_complete_artifact(self.zw, contamination)

    @validation_scoped
    def _qc_ambient_profile(
        self,
        empty_droplets: ArtifactRef | None = None,
        *,
        total_counts: tuple[float, float] | None = None,
        from_assay: str | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Estimate the ambient RNA (soup) profile from empty droplets.

        The soup profile is the share of each gene among the summed counts of
        droplets that hold no cell, equation S1 of Young and Behjati (2020),
        "SoupX removes ambient RNA contamination from droplet-based
        single-cell RNA sequencing data", GigaScience 9:giaa151. The counts
        are summed in one bounded pass over the droplets' rows.

        Choose the droplets with exactly one of ``empty_droplets``, a cell
        selection of empty barcodes, or ``total_counts``, the inclusive range
        of total counts of the droplets to use, such as ``(2, 10)``, which the
        paper uses. A range selects among the barcodes that the assay
        measured, so the store must hold unfiltered droplets, as a raw
        Cell Ranger matrix imported with ``filtering_cutoff=0`` or ``1`` does.

        Args:
            empty_droplets: A cell-selection artifact of empty droplets.
            total_counts: Inclusive ``(low, high)`` bounds on the droplets'
                ``<assay>_nCounts``, with ``0 <= low <= high``.
            from_assay: RNA assay of raw UMI counts. Defaults to the default
                assay.
            invalidate_cache: Recompute instead of reusing a saved profile.

        Returns:
            A complete assay-scoped ``ambient_profile`` artifact. Its
            ``counts`` array holds the summed count of each feature, its
            ``fraction`` array the soup fractions, and its attributes the
            number of droplets (``n_droplets``) and their total count
            (``n_counts``). Read it with
            :meth:`~cytearc.datastore.namespaces.QcAccessor.load_ambient_profile`.

        Raises:
            TypeError: If the assay is not an RNA assay of unsigned integer
                counts, or an argument has the wrong type.
            ValueError: If both or neither of ``empty_droplets`` and
                ``total_counts`` are given, the range is invalid, or the
                droplets select no droplet or hold no counts.
            UnmeasuredCellsError: If the assay did not measure an empty droplet.
            PermissionError: If no saved profile matches and the store is
                read-only.
        """
        operation = "estimate_ambient_profile"
        if (empty_droplets is None) == (total_counts is None):
            raise ValueError("Pass exactly one of empty_droplets and total_counts")
        assay = _counts_assay(self, from_assay, operation)
        parameters: dict[str, Any] = {}
        inputs: dict[str, Any] = {
            "dataset_fingerprint": read_dataset_fingerprint(assay.z)
        }
        droplets: np.ndarray | None = None
        count_range = (0.0, 0.0)
        if empty_droplets is not None:
            if not isinstance(empty_droplets, ArtifactRef):
                raise TypeError("empty_droplets must be an ArtifactRef")
            droplets = read_stored_selection_indices(
                self.zw,
                empty_droplets,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)
            if len(droplets) == 0:
                raise ValueError("empty_droplets selects no droplet")
            self._require_measured_cells(
                assay.name, empty_droplets, operation=operation
            )
            inputs["empty_droplets"] = empty_droplets
        else:
            count_range = _count_range(total_counts)
            parameters["total_counts"] = list(count_range)
        n_features = int(assay.feats.N)
        feature_ids_fingerprint = _feature_ids_fingerprint(assay)

        def payload_matches(_ref: ArtifactRef, group: zarr.Group) -> bool:
            # Called only after the planned arrays and attributes matched.
            return set(group.array_keys()) == set(_PROFILE_ARRAYS) and (
                group.attrs["payload_fingerprint"]
                == fingerprint_stored_arrays(group, _PROFILE_ARRAYS)
            )

        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind="ambient_profile",
            operation=operation,
            parameters=parameters,
            inputs=inputs,
            execution_options={"nthreads": assay.nthreads},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("counts", shape=(n_features,), dtype=np.uint64),
                ArrayRequirement("fraction", shape=(n_features,), dtype=np.float64),
            ),
            required_attributes=(
                AttributeRequirement(
                    "ordered_feature_ids_fingerprint",
                    expected_types=(str,),
                    predicate=lambda value: value == feature_ids_fingerprint,
                ),
                AttributeRequirement("n_droplets", expected_types=(int,)),
                AttributeRequirement("n_counts", expected_types=(int,)),
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
            ),
            reuse_validator=payload_matches,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(operation)
        if droplets is None:
            droplets = self._droplets_in_count_range(assay, *count_range)
        counts = np.asarray(
            assay.rawData[droplets, :]
            .sum(axis=0, dtype=np.uint64)
            .compute(msg=f"({assay.name}) Summing the counts of empty droplets"),
            dtype=np.uint64,
        )
        fractions = soup_fractions(counts)
        with artifact_transaction(self.zw, planned) as group:
            chunks = (min(max(n_features, 1), 100_000),)
            for name, values in (("counts", counts), ("fraction", fractions)):
                output = create_zarr_dataset(
                    group, name, chunks, values.dtype, (n_features,)
                )
                output[:] = values
            group.attrs.update(
                {
                    "ordered_feature_ids_fingerprint": feature_ids_fingerprint,
                    "n_droplets": int(len(droplets)),
                    "n_counts": int(counts.sum(dtype=np.uint64)),
                    "payload_fingerprint": fingerprint_stored_arrays(
                        group, _PROFILE_ARRAYS
                    ),
                }
            )
        logger.info(
            f"Estimated the soup profile of {assay.name} from {len(droplets)} droplets"
        )
        return planned.ref

    def _droplets_in_count_range(
        self, assay: RNAassay, low: float, high: float
    ) -> np.ndarray:
        """Return the measured droplets whose total count lies in ``[low, high]``."""
        parts: list[np.ndarray] = []
        start = 0
        for block in iter_metadata_column_blocks(assay.cells, assay._totals_name):
            values = np.asarray(block, dtype=np.float64)
            parts.append(start + np.flatnonzero((values >= low) & (values <= high)))
            start += len(values)
        droplets = np.concatenate(parts).astype(np.int64, copy=False)
        measured = measured_rows(assay.cells, assay.name, droplets)
        if measured is not None:
            droplets = droplets[measured]
        if len(droplets) == 0:
            raise ValueError(
                f"No droplet of assay {assay.name!r} has between {low:g} and "
                f"{high:g} counts. Use a store of unfiltered droplets, such as a raw "
                "Cell Ranger matrix, or another total_counts range."
            )
        return droplets

    @validation_scoped
    def _qc_load_ambient_profile(self, profile: ArtifactRef) -> pd.DataFrame:
        """Read a saved soup profile as a table.

        Args:
            profile: An ``ambient_profile`` artifact from
                :meth:`~cytearc.datastore.namespaces.QcAccessor.ambient_profile`.

        Returns:
            One row per feature, in assay order and indexed by feature ``ids``,
            with the feature ``name``, the summed ``counts`` of the empty
            droplets, and the soup ``fraction``. ``attrs`` holds the number of
            droplets (``n_droplets``) and their total count (``n_counts``).

        Raises:
            TypeError: If ``profile`` is not an ArtifactRef.
            ValueError: If ``profile`` is not an ``ambient_profile`` artifact,
                or its features no longer match its assay.
        """
        assay, group, fractions = self._ambient_profile_fractions(
            profile, "load_ambient_profile"
        )
        frame = pd.DataFrame(
            {
                "name": np.asarray(assay.feats.fetch_all("names")),
                "counts": np.asarray(as_zarr_array(group["counts"], name="counts")[:]),
                "fraction": fractions,
            },
            index=pd.Index(np.asarray(assay.feats.fetch_all("ids")), name="ids"),
        )
        # The plan required both attributes to be integers.
        frame.attrs.update(
            n_droplets=cast(int, group.attrs["n_droplets"]),
            n_counts=cast(int, group.attrs["n_counts"]),
        )
        return frame

    @validation_scoped
    def _qc_ambient_contamination(
        self,
        cell_selection: ArtifactRef,
        profile: ArtifactRef,
        *,
        gene_sets: Mapping[str, Iterable[str]] | None = None,
        fraction: float | None = None,
        clusters: ArtifactRef | None = None,
        level: Literal["global", "cluster"] = "global",
        max_contamination: float = 1.0,
        fdr: float = 0.05,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Estimate the ambient RNA contamination fraction of each cell.

        The contamination fraction is the share of a cell's counts that come
        from the soup. Following Young and Behjati (2020), "SoupX removes
        ambient RNA contamination from droplet-based single-cell RNA
        sequencing data", GigaScience 9:giaa151, it is measured with
        ``gene_sets``: sets of genes that some cells do not express, such as
        haemoglobin genes outside erythroid cells or immunoglobulin genes
        outside B and plasma cells. In such a cell, every count of the set
        comes from the soup.

        A cell expresses a set when its count of the set is significantly
        larger than a droplet of pure soup with its total would hold: the
        Poisson probability of at least that count, with a mean of
        ``max_contamination`` times the pure-soup count, has a
        Benjamini-Hochberg adjusted value below ``fdr``. Each gene set leaves
        out the cells that express it and, with ``clusters``, every cell of a
        cluster that holds one, as the paper recommends. The fraction is the
        observed count of the gene sets in the remaining cells divided by the
        count of pure soup with their totals. ``level="global"`` gives every
        cell that estimate, which the paper finds sufficient in most cases;
        ``level="cluster"`` gives each cluster the estimate from its own
        cells. Alternatively, pass a known ``fraction`` for every cell. The
        paper's automatic estimate and its cell-level hierarchical model are
        not implemented.

        An estimate is exactly 0 when its cells hold no count of the gene sets
        they measure although pure soup would give them some. A count of 0
        only bounds the fraction below about 3 divided by the pure-soup count,
        with 95% confidence, so each such estimate, whether global, of a gene
        set, or of a cluster, issues a ``UserWarning`` that names it, its
        cells, the pure-soup count, and that bound: a correction with it would
        leave those cells uncorrected. The estimate is saved as computed.

        Args:
            cell_selection: Cells to estimate, such as called cells.
            profile: The soup profile from
                :meth:`~cytearc.datastore.namespaces.QcAccessor.ambient_profile`.
                Its assay is the one estimated.
            gene_sets: Gene-set names mapped to gene names. Names match feature
                names without regard to case; unmatched names are logged, a
                set without a match raises, and a gene may belong to one set
                only. The artifact stores the sets in name order.
            fraction: A contamination fraction between 0 and 1 for every cell,
                instead of ``gene_sets``.
            clusters: A cell-label artifact, such as a clustering, that labels
                every cell of ``cell_selection``.
            level: ``"global"`` or ``"cluster"``; ``"cluster"`` needs
                ``clusters``.
            max_contamination: The largest plausible contamination fraction,
                greater than 0 and at most 1 (the paper's default is 1).
            fdr: The false discovery rate of the expression test, greater than
                0 and less than 1.
            invalidate_cache: Recompute instead of reusing a saved estimate.

        Returns:
            A complete assay-scoped ``ambient_contamination`` artifact over
            ``cell_selection``. Its ``values`` hold each cell's fraction, which
            ``artifacts.load_values`` and ``select_cells`` read; read the
            estimates and their evidence with
            :meth:`~cytearc.datastore.namespaces.QcAccessor.load_ambient_contamination`.

        Raises:
            TypeError: If an argument has the wrong type, or the assay is not
                an RNA assay of unsigned integer counts.
            ValueError: If an argument is invalid, a gene set holds no soup
                counts, no cell can measure the contamination, or with
                ``level="cluster"``, a cluster has no such cell.
            UnmeasuredCellsError: If the assay did not measure a selected cell.
            PermissionError: If no saved estimate matches and the store is
                read-only.
        """
        operation = "estimate_ambient_contamination"
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if level not in ("global", "cluster"):
            raise ValueError("level must be 'global' or 'cluster'")
        if clusters is not None and not isinstance(clusters, ArtifactRef):
            raise TypeError("clusters must be an ArtifactRef or None")
        if (gene_sets is None) == (fraction is None):
            raise ValueError("Pass exactly one of gene_sets and fraction")
        rho_max = _probability(
            max_contamination, "max_contamination", upper_inclusive=True
        )
        rate = _probability(fdr, "fdr", upper_inclusive=False)
        assay, _group, soup = self._ambient_profile_fractions(profile, operation)
        parameters: dict[str, Any]
        resolved_sets: dict[str, np.ndarray] = {}
        set_soup = np.zeros(0, dtype=np.float64)
        if fraction is not None:
            if (
                clusters is not None
                or level != "global"
                or rho_max != 1.0
                or rate != 0.05
            ):
                raise ValueError(
                    "clusters, level, max_contamination, and fdr apply only to "
                    "estimates from gene_sets"
                )
            fixed = float_argument(fraction, "fraction")
            if not 0.0 <= fixed <= 1.0:
                raise ValueError("fraction must lie between 0 and 1")
            parameters = {"method": "fixed", "fraction": fixed}
        else:
            if level == "cluster" and clusters is None:
                raise ValueError("level='cluster' needs clusters")
            resolved_sets = _resolve_gene_sets(assay, gene_sets)
            set_soup = np.array(
                [float(soup[indices].sum()) for indices in resolved_sets.values()]
            )
            empty = [
                name
                for name, value in zip(resolved_sets, set_soup, strict=True)
                if value <= 0
            ]
            if empty:
                raise ValueError(
                    f"Gene sets {', '.join(map(repr, empty))} hold no counts in the "
                    "soup profile, so they cannot measure the contamination"
                )
            parameters = {
                "method": "gene_sets",
                "gene_sets": {
                    name: indices.tolist() for name, indices in resolved_sets.items()
                },
                "level": level,
                "max_contamination": rho_max,
                "fdr": rate,
            }
        cell_idx = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(cell_idx) == 0:
            raise ValueError("cell_selection must select at least one cell")
        labels: np.ndarray | None = None
        inputs: dict[str, Any] = {"profile": profile}
        if clusters is not None:
            grouping = resolve_grouping(
                self.zw, self.cells, clusters, cell_selection=cell_selection
            )
            if grouping.missing_mask is not None and grouping.missing_mask.any():
                raise ValueError(
                    "clusters has no label for some selected cells. Freeze the "
                    "labels of the selected cells with snapshot_cluster_labels("
                    "clusters, cell_selection=...) after selecting labelled cells"
                )
            labels = np.asarray(grouping.labels)
            inputs["clusters"] = clusters
        self._require_measured_cells(assay.name, cell_selection, operation=operation)
        n_cells = len(cell_idx)
        n_sets = len(resolved_sets)
        arrays: dict[str, tuple[tuple[int, ...], str | None]] = {
            "values": ((n_cells,), "f")
        }
        if resolved_sets:
            arrays.update(
                {
                    "observed": ((n_cells, n_sets), "f"),
                    "expected": ((n_cells, n_sets), "f"),
                    "p_values": ((n_cells, n_sets), "f"),
                    "expressing": ((n_cells, n_sets), "b"),
                    "used": ((n_cells, n_sets), "b"),
                }
            )
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind="ambient_contamination",
            operation=operation,
            parameters=parameters,
            inputs=inputs,
            execution_options={"nthreads": assay.nthreads},
            cell_selection=cell_selection,
            arrays=arrays,
            invalidate_cache=invalidate_cache,
            required_attributes=(
                AttributeRequirement("gene_sets", expected_types=(list,)),
                AttributeRequirement("estimate", expected_types=(float,)),
                AttributeRequirement("set_estimates", expected_types=(dict,)),
                AttributeRequirement("cluster_estimates", expected_types=(dict,)),
            ),
        )
        if planned.reused:
            return planned.ref
        self._require_writable(operation)
        if not resolved_sets:
            _write_artifact(
                self.zw,
                planned,
                {"values": np.full(n_cells, parameters["fraction"])},
                {
                    "gene_sets": [],
                    "estimate": parameters["fraction"],
                    "set_estimates": {},
                    "cluster_estimates": {},
                },
            )
            return planned.ref
        totals = assay._cell_count_totals(cell_idx)
        set_totals = assay._feature_totals(dict(resolved_sets))
        observed = np.column_stack(
            [set_totals[name][cell_idx] for name in resolved_sets]
        )
        expected, p_values = gene_set_evidence(
            observed, totals, set_soup, max_contamination=rho_max
        )
        expressing = expressing_cells(p_values, rate)
        used = estimation_cells(expressing, labels)
        fit: ContaminationFit = estimate_contamination(
            observed,
            expected,
            used,
            clusters=labels,
            level=level,
            set_names=list(resolved_sets),
        )
        cluster_estimates: dict[str, float] = {}
        if fit.cluster_labels is not None and fit.cluster_fractions is not None:
            cluster_estimates = {
                str(label): float(value)
                for label, value in zip(
                    fit.cluster_labels, fit.cluster_fractions, strict=True
                )
            }
        largest = max([fit.fraction, *cluster_estimates.values()])
        if largest > rho_max:
            logger.warning(
                f"An estimated contamination fraction, {largest:.3g}, exceeds "
                f"max_contamination={rho_max:g}: some cells used to estimate it "
                "may express a gene set. Check the gene sets and clusters."
            )
        _write_artifact(
            self.zw,
            planned,
            {
                "values": fit.cell_fractions,
                "observed": observed,
                "expected": expected,
                "p_values": p_values,
                "expressing": expressing,
                "used": used,
            },
            {
                "gene_sets": list(resolved_sets),
                "estimate": float(fit.fraction),
                "set_estimates": {
                    name: _estimate_record(value)
                    for name, value in zip(
                        resolved_sets, fit.set_fractions, strict=True
                    )
                },
                "cluster_estimates": cluster_estimates,
            },
        )
        logger.info(
            f"Estimated an ambient RNA contamination fraction of {fit.fraction:.4f} "
            f"for {n_cells} cells"
        )
        return planned.ref

    @validation_scoped
    def _qc_load_ambient_contamination(
        self, contamination: ArtifactRef
    ) -> AmbientContamination:
        """Read a saved contamination estimate with its evidence.

        Args:
            contamination: An ``ambient_contamination`` artifact from
                :meth:`~cytearc.datastore.namespaces.QcAccessor.ambient_contamination`.

        Returns:
            The per-cell fractions, the global, per-gene-set, and per-cluster
            estimates, and the per-cell evidence of each gene set.

        Raises:
            TypeError: If ``contamination`` is not an ArtifactRef.
            ValueError: If it is not an ``ambient_contamination`` artifact.
            MemoryError: If its values do not fit the datastore memory budget.
        """
        status = self._ambient_contamination_status(contamination)
        group = artifact_group(self.zw, contamination)
        parameters = status.parameters or {}
        # The plan required these attributes with these types.
        attributes = group.attrs
        names = [str(name) for name in cast(list[str], attributes["gene_sets"])]
        set_estimates = cast(dict[str, float | None], attributes["set_estimates"])
        cluster_estimates = cast(dict[str, float], attributes["cluster_estimates"])
        loaded = self._artifacts_load_values(contamination)
        fractions = pd.Series(
            np.asarray(loaded.values, dtype=np.float64),
            index=pd.Index(loaded.cell_ids, name="ids"),
            name="fraction",
        )
        evidence: pd.DataFrame | None = None
        if names:
            columns = {
                name: np.asarray(as_zarr_array(group[name], name=name)[:])
                for name in _EVIDENCE_ARRAYS
            }
            n_cells = len(loaded.cell_ids)
            evidence = pd.DataFrame(
                {
                    "ids": np.tile(loaded.cell_ids, len(names)),
                    "gene_set": np.repeat(np.asarray(names, dtype=object), n_cells),
                    "observed": columns["observed"].T.reshape(-1),
                    "expected": columns["expected"].T.reshape(-1),
                    "p_value": columns["p_values"].T.reshape(-1),
                    "expressing": columns["expressing"].T.reshape(-1),
                    "used": columns["used"].T.reshape(-1),
                }
            )
        return AmbientContamination(
            fractions=fractions,
            estimate=cast(float, attributes["estimate"]),
            method=parameters["method"],
            level=parameters.get("level", "global"),
            set_estimates=dict(set_estimates),
            cluster_estimates=dict(cluster_estimates),
            evidence=evidence,
        )

    def _qc_correct_ambient(
        self,
        contamination: ArtifactRef,
        *,
        assay_label: str,
        random_seed: int = 0,
    ) -> None:
        """Add an assay of counts with the ambient RNA removed.

        Each cell of the contamination estimate loses its contamination
        fraction of its counts, as Young and Behjati (2020), "SoupX removes
        ambient RNA contamination from droplet-based single-cell RNA
        sequencing data", GigaScience 9:giaa151, describe: the most likely
        removal under a multinomial model with the soup profile's
        probabilities that never removes more of a gene than the cell holds.
        Genes are emptied in increasing order of their count divided by their
        soup fraction, and the rest is taken from the other genes in
        proportion to their soup fraction; genes absent from the soup are
        never reduced. A cell whose soup genes hold fewer counts than its
        share loses all of them.

        As in the paper, each gene's removal is rounded to whole counts at
        random, up with a probability equal to its fractional part, so its
        expected removal is the removal itself: summed over cells, every gene
        loses its share of the soup on average, however small its share in
        each cell. The draws follow systematic sampling with probabilities
        proportional to size (Madow 1949, doi:10.1214/aoms/1177729988), one
        draw per cell, so a cell with fraction ``rho`` and total ``N`` loses
        ``rho * N`` counts rounded down or up. A cell's draw depends only on
        ``random_seed`` and the cell's row in the store, so a seed gives the
        same counts in any memory budget, and a cell gets the same counts
        whichever other cells the estimate corrects with it.

        The new assay is an RNA assay with the source's features and count
        dtype, whose counts never exceed the source's. It has a row for every
        cell but measures only the cells of the estimate, which its membership
        column ``<assay_label>_I`` records when they are not every cell; the
        other rows hold no counts. Its counts are written in bounded row bands
        with ``countsT``, so every operation can use it by name, and its
        attributes record the ``contamination`` and ``profile`` artifacts, the
        ``random_seed``, and the counts removed. The assay becomes visible only
        after its counts are complete. If the write fails or is interrupted,
        the partial assay is removed or left pending until
        :meth:`~cytearc.datastore.datastore.DataStore.discard_interrupted_assay`
        removes it.

        Args:
            contamination: An ``ambient_contamination`` artifact from
                :meth:`~cytearc.datastore.namespaces.QcAccessor.ambient_contamination`.
            assay_label: Name of the new assay.
            random_seed: Seed of the rounding draws, a non-negative integer.

        Returns:
            None

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``contamination`` is not an ``ambient_contamination``
                artifact, ``assay_label`` is taken or invalid, or
                ``random_seed`` is negative.
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
            MemoryError: If one row band of the new assay and its correction do
                not fit the datastore memory budget.
        """
        from ...storage.identity import CountSummary, finalize_counts
        from ...storage.layout import array_shard_rows
        from ...storage.schema import derived_assay_transaction
        from ...storage.sharding import (
            dense_counts_admission,
            fit_count_layout,
            write_dense_from_row_batches,
        )
        from ...storage.stores import zarr_group_root
        from ...writers.counts_t import finalize_writer_counts_t

        operation = "correct_ambient"
        status = self._ambient_contamination_status(contamination)
        if not isinstance(assay_label, str):
            raise TypeError("assay_label must be a string")
        seed = integer_argument(random_seed, "random_seed", minimum=0)
        profile = status.input_ref("profile")
        assay, _group, soup = self._ambient_profile_fractions(profile, operation)
        self._require_writable(operation)
        resolved = resolve_cell_aligned_artifact(
            self.zw,
            contamination,
            expected_kind="ambient_contamination",
            max_bytes=self.memoryBytes,
        )
        fractions = np.asarray(resolved.values, dtype=np.float64)
        cell_idx = np.asarray(resolved.cell_idx, dtype=np.int64)
        n_cells = int(assay.cells.N)
        n_genes = int(assay.feats.N)
        dtype = np.dtype(assay.rawData.dtype)
        selected = np.zeros(n_cells, dtype=bool)
        selected[cell_idx] = True
        membership: str | np.ndarray | None = (
            resolve_assay_membership(assay.cells, assay.name)
            if bool(selected.all())
            else selected
        )
        row_bytes = n_genes * dtype.itemsize
        # Each removal call takes rows holding at most this many nonzero
        # counts; a single row may hold every gene.
        entry_cap = max(n_genes, int(self.memoryBytes) // (8 * _REMOVAL_ENTRY_BYTES))
        scratch_bytes = entry_cap * _REMOVAL_ENTRY_BYTES
        source = assay.rawData[cell_idx, :]
        decode_bytes = source._max_decode_bytes()
        resident_bytes = (
            soup.nbytes
            + fractions.nbytes
            + cell_idx.nbytes
            + selected.nbytes
            + CountSummary.nbytes_for(n_cells, n_genes)
        )

        def producer_bytes(rows: int) -> int:
            # A source band and its read, its corrected copy, and the band
            # that places it among every cell, beside the removal's scratch.
            return 3 * rows * row_bytes + scratch_bytes + decode_bytes

        removed_total = 0
        root = zarr_group_root(self.z, mode="r+")
        with derived_assay_transaction(
            root,
            assay_label,
            self.workspace,
            operation=operation,
            membership=membership,
        ) as transaction:
            layout = fit_count_layout(
                {assay_label: (n_genes, dtype)},
                nCells=n_cells,
                profile=self.storageProfile,
                memoryBytes=self.memoryBytes,
                transposed=(assay_label,),
                admitCounts=dense_counts_admission(resident_bytes, producer_bytes),
            )
            counts = transaction.create_counts(
                n_cells,
                np.asarray(assay.feats.fetch_all("ids")),
                np.asarray(assay.feats.fetch_all("names")),
                dtype,
                profile=self.storageProfile,
                policy=layout,
            )
            band = max(1, array_shard_rows(counts))

            def corrected_blocks() -> Iterator[np.ndarray]:
                nonlocal removed_total
                offset = 0
                # The stream reserves its block and read beside the corrected
                # copy, the spread band, and the scratch.
                for block in source._with_block_size(band)._stream_blocks(
                    nthreads=1,
                    msg=f"({assay.name}) Removing ambient RNA counts",
                    prefetch=1,
                    row_mask=None,
                    resident_bytes=resident_bytes
                    + 2 * band * row_bytes
                    + scratch_bytes,
                ):
                    values = np.asarray(block)
                    corrected = np.empty_like(values)
                    rows = len(values)
                    held = np.cumsum(np.count_nonzero(values, axis=1))
                    start = 0
                    while start < rows:
                        # The rows of one call hold at most entry_cap counts,
                        # and each cell's row in the store keys its draw.
                        base = held[start - 1] if start else 0
                        stop = max(
                            start + 1,
                            int(np.searchsorted(held, base + entry_cap, side="right")),
                        )
                        corrected[start:stop] = remove_ambient_counts(
                            values[start:stop],
                            fractions[offset + start : offset + stop],
                            soup,
                            random_seed=seed,
                            row_keys=cell_idx[offset + start : offset + stop],
                        )
                        start = stop
                    removed_total += int(
                        values.sum(dtype=np.uint64) - corrected.sum(dtype=np.uint64)
                    )
                    offset += rows
                    yield corrected

            summary = CountSummary(counts)
            write_dense_from_row_batches(
                counts,
                (
                    piece
                    for _, piece in spread_measured_rows(
                        corrected_blocks(), selected, n_genes, band, fill=dtype.type(0)
                    )
                ),
                dtype=dtype,
                msg="Writing ambient-corrected counts",
                resources=self.resources,
                residentBytes=resident_bytes,
                producerReserveBytes=producer_bytes(band),
                io=self.storageIo,
                countSummary=summary,
            )
            finalize_counts(counts, summary=summary)
            finalize_writer_counts_t(
                root,
                assay_label,
                self.workspace,
                assay_type="RNA",
                resources=self.resources,
            )
            transaction.group.attrs.update(
                {
                    "ambient_source_assay": assay.name,
                    "ambient_contamination": contamination.to_dict(),
                    "ambient_profile": profile.to_dict(),
                    "ambient_random_seed": seed,
                    "ambient_corrected_cells": int(len(cell_idx)),
                    "ambient_removed_counts": int(removed_total),
                }
            )
        # Registered as add_grouped_assay and add_melded_assay register theirs.
        self._assayNames = tuple(self._scan_assays())
        self._load_assays(custom_assay_types={assay_label: "RNA"})
        self._ini_cell_props(mito_pattern=None, ribo_pattern=None)
        logger.info(
            f"Removed {removed_total} ambient RNA counts from {len(cell_idx)} cells "
            f"of {assay.name} into assay {assay_label!r}"
        )
