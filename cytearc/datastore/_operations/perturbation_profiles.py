from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...metadata.arguments import (
    MixscapeArguments,
    PerturbationClusterArguments,
    PerturbationLabelArguments,
    PerturbationProfileArguments,
)
from ...metadata.artifacts import write_cell_data_artifact
from ...metadata.selection import cell_selection_indices
from ...perturbation.clusters import PerturbationClusters, cluster_settings
from ...perturbation.profiles import PerturbationProfiles, profile_settings
from ...storage.artifact_writer import ArrayRequirement, AttributeRequirement
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_strings,
    require_complete_artifact,
)
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.numba import numba_threads
from ...utils.warnings import warn
from .perturbation_store import (
    _LabelRows,
    _cell_samples,
    _cell_values,
    _count_blocks,
    _label_groups,
    _label_rows,
    _matched_ids,
    _pairs_fingerprint,
    _positions_in,
    _require_budget,
    _require_cell_column,
    _require_ref,
    _row_features,
    _stored_samples,
)

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _PerturbationProfileOperationsBase,
    )
else:
    _PerturbationProfileOperationsBase = object


@dataclass(frozen=True, slots=True)
class _ProfileCells:
    """The cells of perturbation profiles, from labels or Mixscape classes.

    Attributes:
        labels: The perturbation labels that name the targets and controls.
        rows: The perturbations: each target of the labels that is not a
            control, or each guide with a target.
        row_control: Whether each row is a control guide.
        cell_selection: The selection of the cells.
        cell_index: The cells' positions in the store.
        groups: Each cell's perturbation, a row of ``rows``, or -1.
        control: Whether each cell is a control cell.
    """

    labels: ArtifactRef
    rows: _LabelRows
    row_control: np.ndarray
    cell_selection: ArtifactRef
    cell_index: np.ndarray
    groups: np.ndarray
    control: np.ndarray


def _left_out_warning(
    names: np.ndarray,
    n_cells: np.ndarray,
    n_rows: int,
    min_cells: int,
    rows: str,
    cells: str,
) -> str | None:
    """Name the perturbations without a profile, or return None when every one has one.

    Args:
        names: The perturbations with fewer than ``min_cells`` cells.
        n_cells: Their cells.
        n_rows: The perturbations of the labels.
        min_cells: The fewest cells of a profiled perturbation.
        rows: What the perturbations are, ``"targets"`` or ``"guides"``.
        cells: What their cells are, such as ``"knocked-out cells"``.
    """
    if not len(names):
        return None
    examples = ", ".join(
        f"{str(name)!r} ({int(count)})"
        for name, count in zip(names[:5], n_cells[:5], strict=True)
    )
    more = ", and more" if len(names) > 5 else ""
    return (
        f"{len(names)} of {n_rows} {rows} have fewer than {min_cells} {cells} "
        f"and get no profile: {examples}{more}. perturbation.load_profiles lists "
        "them all in left_out."
    )


class _PerturbationProfileOperationsMixin(_PerturbationProfileOperationsBase):
    def _profile_cells(self, labels: ArtifactRef, by: str) -> _ProfileCells:
        """Read the rows and cells of perturbation profiles.

        The rows are the targets of the labels that are not controls, or with
        ``by="guide"`` every guide with a target, the control guides included.
        A row's cells carry its target, or its guide, alone. With Mixscape
        classes, the cells are those of the classes, and of a row that is not
        a control guide only those that Mixscape called knocked out count.

        Raises:
            ValueError: If no labelled cell carries control guides only.
        """
        classes = None
        perturbation_labels = labels
        if labels.kind == MixscapeArguments.artifact_kind:
            classes = require_complete_artifact(self.zw, labels)
            perturbation_labels = require_complete_artifact(
                self.zw, classes.input_ref("signature")
            ).input_ref("labels")
        rows = _label_rows(self, perturbation_labels, by, control_guides=True)
        groups, control = _label_groups(self, perturbation_labels, rows, by)
        row_control = read_array(
            artifact_group(self.zw, perturbation_labels), "target_is_control"
        ).astype(bool)[rows.codes]
        selection = rows.status.input_ref("cell_selection")
        cell_index = cell_selection_indices(self.zw, selection)
        if classes is not None:
            # The cells of the classes, with the rows and flags of those that
            # the labels cover.
            selection = classes.input_ref("cell_selection")
            classified = cell_selection_indices(self.zw, selection)
            positions, present = _positions_in(classified, cell_index)
            aligned = np.full(len(classified), -1, dtype=np.int64)
            aligned[positions[present]] = groups[present]
            control_flags = np.zeros(len(classified), dtype=bool)
            control_flags[positions[present]] = control[present]
            knocked_out = _cell_values(self, labels, "global_class").astype(str) == str(
                (classes.parameters or {})["perturbation_type"]
            )
            kept = aligned >= 0
            kept[kept] = knocked_out[kept] | row_control[aligned[kept]]
            cell_index, groups, control = (
                classified,
                np.where(kept, aligned, -1),
                control_flags,
            )
        return _ProfileCells(
            labels=perturbation_labels,
            rows=rows,
            row_control=row_control,
            cell_selection=selection,
            cell_index=cell_index,
            groups=groups,
            control=control,
        )

    @validation_scoped
    def _perturbation_profiles(
        self,
        labels: ArtifactRef,
        features: ArtifactRef,
        *,
        by: str = "target",
        sample_column: str | None = None,
        min_cells: int = 25,
        mask_targets: bool = True,
        id_column: str | None = "target_gene_id",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Average each perturbation's z-normalized expression into its profile.

        Replogle et al. (2022) compare perturbations by their mean expression
        profiles, each feature measured against the control cells:

        1. Each cell's counts in ``features`` are scaled so that its total,
           the sum of its stored counts over every feature of the assay,
           equals the median total of the control cells of the labels.
        2. Within each sample of ``sample_column``, such as a GEM group, each
           feature is z-normalized against the sample's control cells,
           ``z = (x - mean) / sd``, with the standard deviation's n - 1
           divisor, for every cell, control cells included.
        3. A perturbation's profile is the mean z-score of its cells over all
           samples, computed in float64 and stored as float32.

        This recipe reproduces the published profiles of Replogle et al.'s K562
        essential screen from its cells to float32 rounding. A feature whose
        control cells in a sample hold one value has no spread there, and its
        z-scores in that sample are 0, CyteArc's choice where Replogle et al.'s
        genome-wide profiles hold infinite values; ``load_profiles`` counts
        such samples for each feature. A sample needs 2 control cells or more.

        A perturbation is a target of the labels that is not a control, or
        with ``by="guide"`` a guide, each control guide included, as Replogle
        et al.'s profiles have a row for each guide pair, control pairs
        included; with targets, the control cells get no row, since their mean
        z-score is 0. A perturbation's cells carry its target, or its guide,
        alone. With Mixscape classes as ``labels``, only the cells that
        Mixscape called knocked out count, among the cells of the classes, so
        that escapers do not dilute the profiles; a control guide keeps its
        cells. A perturbation with fewer than ``min_cells`` cells, 25 as in
        Replogle et al., gets no profile and is named in a warning, which a
        reused result repeats. With ``mask_targets``, each profile's own
        target feature is set to 0, the control mean, so that the knockdown
        of the target itself does not make two perturbations alike; a target
        is matched to a feature as
        :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.efficiency`
        matches it, and control guides mask nothing.

        The counts are read twice, the control cells and then the cells of
        the profiled perturbations, in bands of features that fit the memory
        budget, and the profiles do not depend on the bands or the threads.

        Args:
            labels: A ``perturbation_labels`` artifact, or a
                ``mixscape_classes`` artifact whose knocked-out cells alone
                count.
            features: A ``feature_selection`` artifact, such as highly
                variable genes, whose features the profiles hold; its assay is
                the expression assay.
            by: ``"target"`` for one profile per target gene, or ``"guide"``
                for one per guide, control guides included.
            sample_column: A cell metadata column whose samples are
                z-normalized against their own control cells, or None for one
                sample.
            min_cells: The fewest cells of a profiled perturbation.
            mask_targets: Set each profile's own target feature to 0.
            id_column: The feature column of the guide assay that holds each
                guide's target feature ID, or None to match targets by name
                only; it matters only with ``mask_targets``.
            invalidate_cache: Compute again instead of reusing saved
                profiles.

        Returns:
            A complete ``perturbation_profiles`` artifact scoped to the assay
            of ``features``, with one profile per perturbation. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_profiles`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` or ``features`` is not an artifact of
                its kinds, ``by`` is unknown, ``min_cells`` is below 1,
                ``sample_column`` is not a cell metadata column or leaves a
                control cell or a cell of a profiled perturbation without a
                sample, a sample has fewer than 2 control cells, ``id_column``
                is not a feature column of the guide assay, the labels have no
                control cell, or no perturbation has ``min_cells`` cells.
            PermissionError: If the store is read-only and the profiles are
                not saved yet.
            MemoryError: If the profiles and one feature's sums do not fit the
                memory budget.
        """
        from ...perturbation.profiles import (
            CONTROL_SD_DDOF,
            control_moments,
            depth_target,
            profile_values,
            z_score_sums,
        )
        from ...storage.feature_selection import read_feature_selection_indices

        _require_ref(
            labels,
            "labels",
            PerturbationLabelArguments.artifact_kind,
            MixscapeArguments.artifact_kind,
        )
        _require_ref(features, "features", "feature_selection")
        floor = profile_settings(by, min_cells, mask_targets, id_column, sample_column)
        _require_cell_column(self, sample_column, "sample_column")
        bool_argument(invalidate_cache, "invalidate_cache")
        cells = self._profile_cells(labels, by)
        expression = self._get_assay(str(features.assay))
        self._require_measured_cells(
            expression.name,
            cells.cell_selection,
            operation=PerturbationProfileArguments.operation,
        )
        feature_index = np.asarray(
            read_feature_selection_indices(self.zw, expression.name, features),
            dtype=np.int64,
        )
        n_features = len(feature_index)
        rows = cells.rows
        n_rows = len(rows.names)
        grouped = cells.groups >= 0
        sizes = np.bincount(cells.groups[grouped], minlength=n_rows)
        kept = np.flatnonzero(sizes >= floor)
        thin = np.flatnonzero(sizes < floor)
        if labels.kind == PerturbationLabelArguments.artifact_kind:
            noun = "cells"
        elif by == "target":
            noun = "knocked-out cells"
        else:
            noun = "knocked-out or control cells"
        if not len(kept):
            raise ValueError(
                f"No {by} has {floor} {noun} or more, so none gets a profile; "
                "lower min_cells"
            )
        n_kept = len(kept)
        names = rows.names[kept].astype(str)
        row_targets = rows.target_names[rows.codes].astype(str)
        feature_ids = np.asarray(expression.feats.fetch_all("ids")).astype(str)
        # Each profile's masked feature, as a position in the assay.
        masked = np.full(n_kept, -1, dtype=np.int64)
        targets_fingerprint = None
        if mask_targets:
            matched = _row_features(
                self,
                cells.labels,
                rows,
                id_column,
                feature_ids,
                np.asarray(expression.feats.fetch_all("names")).astype(str),
            )[kept]
            # A control target is no gene of the genome.
            matched[cells.row_control[kept]] = -1
            targets_fingerprint = _pairs_fingerprint(
                names, _matched_ids(matched, feature_ids)
            )
            # Only a target among the profiles' features has a value to mask.
            inside = np.isin(matched, feature_index)
            masked[inside] = matched[inside]
        slot = np.full(n_rows, -1, dtype=np.int64)
        slot[kept] = np.arange(n_kept)
        groups = np.where(grouped, slot[np.maximum(cells.groups, 0)], -1)
        # The cells read: the control cells and the cells of the profiled
        # perturbations, which by guide include the control cells of a
        # control guide, in store order.
        used = np.flatnonzero(cells.control | (groups >= 0))
        used_cells = cells.cell_index[used]
        is_control = cells.control[used]
        cell_samples = _cell_samples(
            self,
            sample_column,
            used_cells,
            reason="Each sample's control cells z-normalize its cells, so every "
            "control cell and every cell of a profiled perturbation needs one",
        )
        samples, sample_names = cell_samples.codes, cell_samples.names
        n_samples = len(sample_names)
        n_controls = np.bincount(samples[is_control], minlength=n_samples)
        # The standard deviation's divisor needs this many control cells.
        needed = CONTROL_SD_DDOF + 1
        short = np.flatnonzero(n_controls < needed)
        if len(short):
            where = (
                "the labels have"
                if sample_column is None
                else f"sample {sample_names[short[0]]!r} has"
            )
            raise ValueError(
                f"The control standard deviations of a sample need {needed} "
                f"control cells or more, but {where} "
                f"{int(n_controls[short[0]])}; pass a sample_column whose "
                f"samples each hold {needed} or more, or label more control cells"
            )
        arguments = PerturbationProfileArguments(
            labels=labels,
            feature_selection=features,
            dataset_fingerprint=self._ensure_dataset_fingerprint(expression.name),
            sample_assignments_fingerprint=cell_samples.assignments,
            target_features_fingerprint=targets_fingerprint,
            by=by,
            min_cells=floor,
            mask_targets=mask_targets,
            sample_columns=cell_samples.columns,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=expression.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("values", shape=(n_kept, n_features), dtype_kind="f"),
                *(
                    ArrayRequirement(name, shape=(n_kept,))
                    for name in ("perturbations", "targets")
                ),
                ArrayRequirement("is_control", shape=(n_kept,), dtype_kind="b"),
                *(
                    ArrayRequirement(name, shape=(n_kept,), dtype_kind="i")
                    for name in ("n_cells", "masked_features")
                ),
                *(
                    ArrayRequirement(name, shape=(len(thin),))
                    for name in ("left_out", "left_out_targets")
                ),
                ArrayRequirement(
                    "left_out_n_cells", shape=(len(thin),), dtype_kind="i"
                ),
                *(
                    ArrayRequirement(name, shape=(n_features,), dtype_kind="i")
                    for name in ("features", "n_zero_sd")
                ),
                ArrayRequirement("n_control_cells", shape=(n_samples,), dtype_kind="i"),
                *(
                    ArrayRequirement(
                        name, shape=(n_samples, n_features), dtype_kind="f"
                    )
                    for name in ("control_mean", "control_sd")
                ),
            ),
            required_attributes=(
                AttributeRequirement(
                    "sample_names", predicate=lambda value: value == sample_names
                ),
            ),
        )
        left_out = _left_out_warning(
            rows.names[thin].astype(str), sizes[thin], n_rows, floor, f"{by}s", noun
        )
        if planned.reused:
            # Saved profiles warn as they did when they were computed.
            if left_out is not None:
                warn(left_out)
            return planned.ref
        # Each labelled cell's position, perturbation and flag, each read
        # cell's position, sample and total, the float32 profiles, and each
        # sample's control mean and standard deviation of every feature.
        held = (
            24 * len(cells.cell_index)
            + 32 * len(used)
            + 4 * n_kept * n_features
            + 16 * n_samples * n_features
            + 16 * n_features
        )
        # One feature of a band: its float64 sums over the perturbations,
        # which are divided straight into the profiles counted above, its
        # running moments in each sample, and a cell's value as read and as a
        # C-ordered copy.
        per_feature = 8 * n_kept + 24 * n_samples + 16
        _require_budget(
            held + per_feature,
            self.memoryBytes,
            f"The profiles of {n_kept} {by}s in {n_features} features hold about",
            beside=" with one feature's sums",
            remedy="Raise mem_budget, or pass fewer features.",
        )
        width = min(n_features, (self.memoryBytes - held) // per_feature)
        self._require_writable(arguments.operation)
        totals = expression._cell_count_totals(used_cells)
        target = depth_target(totals, is_control)
        controls = np.flatnonzero(is_control)
        members = np.flatnonzero(groups[used] >= 0)
        member_groups = groups[used][members]
        n_cells = sizes[kept].astype(np.int64)
        masked_columns = np.full(n_kept, -1, dtype=np.int64)
        masked_columns[masked >= 0] = np.searchsorted(
            feature_index, masked[masked >= 0]
        )
        values = np.empty((n_kept, n_features), dtype=np.float32)
        means = np.empty((n_samples, n_features))
        sds = np.empty((n_samples, n_features))

        def counts(band: np.ndarray, positions: np.ndarray) -> Iterator[np.ndarray]:
            """Stream the counts of a band of features over some read cells."""
            _, blocks = _count_blocks(
                self,
                expression,
                band,
                used_cells[positions],
                resident_bytes=held + (per_feature - 16) * len(band),
                scratch_bytes_per_row=16 * len(band),
            )
            yield from blocks

        with numba_threads(self.nthreads):
            for start in range(0, n_features, width):
                stop = min(start + width, n_features)
                band = feature_index[start:stop]
                _, means[:, start:stop], sds[:, start:stop] = control_moments(
                    counts(band, controls),
                    totals[controls],
                    target,
                    samples[controls],
                    n_samples,
                    len(band),
                )
                sums = z_score_sums(
                    counts(band, members),
                    totals[members],
                    target,
                    samples[members],
                    member_groups,
                    means[:, start:stop],
                    sds[:, start:stop],
                    n_kept,
                )
                inside = (masked_columns >= start) & (masked_columns < stop)
                # The means go straight into the band of the float32 profiles.
                profile_values(
                    sums,
                    n_cells,
                    np.where(inside, masked_columns - start, -1),
                    out=values[:, start:stop],
                )
                del sums
        n_zero_sd = np.count_nonzero(sds == 0, axis=0).astype(np.int64)
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": values,
                "perturbations": names,
                "targets": row_targets[kept],
                "is_control": cells.row_control[kept],
                "n_cells": n_cells,
                "masked_features": masked,
                "left_out": rows.names[thin].astype(str),
                "left_out_targets": row_targets[thin],
                "left_out_n_cells": sizes[thin].astype(np.int64),
                "features": feature_index,
                "n_zero_sd": n_zero_sd,
                "n_control_cells": n_controls.astype(np.int64),
                "control_mean": means,
                "control_sd": sds,
            },
            attributes={"sample_names": sample_names},
        )
        if left_out is not None:
            warn(left_out)
        flat = int(np.count_nonzero(n_zero_sd))
        logger.info(
            f"Computed the profiles of {n_kept} {by}s in {n_features} features "
            f"against {len(controls)} control cells in {n_samples} samples"
            + (
                f"; {flat} features hold one value over the control cells of a "
                "sample, where their z-scores are 0"
                if flat
                else ""
            )
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_profiles(
        self, profiles: ArtifactRef
    ) -> PerturbationProfiles:
        """Read saved perturbation profiles.

        Args:
            profiles: A ``perturbation_profiles`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.profiles`.

        Returns:
            Each perturbation's profile with its cells and masked feature, the
            perturbations left out, and each sample's control cells and
            control statistics of each feature.

        Raises:
            TypeError: If ``profiles`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_profiles`` artifact.
            MemoryError: If its profiles do not fit the memory budget.
        """
        _require_ref(profiles, "profiles", PerturbationProfileArguments.artifact_kind)
        status = require_complete_artifact(self.zw, profiles)
        group = artifact_group(self.zw, profiles)
        assay = self._get_assay(str(profiles.assay))

        stored = as_zarr_array(group["values"], name="values")
        # The frame holds the float32 values as read, without a copy.
        n_bytes = 4 * int(np.prod(stored.shape))
        _require_budget(
            n_bytes,
            self.memoryBytes,
            f"The profiles of {stored.shape[0]} perturbations in {stored.shape[1]} "
            "features hold",
        )
        feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        positions = read_array(group, "features")
        columns = pd.Index(feature_ids[positions], name="feature")
        index = pd.Index(
            read_array(group, "perturbations").astype(str), name="perturbation"
        )
        masked = read_array(group, "masked_features")
        samples = pd.Index(
            np.asarray(_stored_samples(group, "sample_names"), dtype=object),
            name="sample",
        )
        parameters = status.parameters or {}
        sample_columns = list(parameters["sample_columns"])
        return PerturbationProfiles(
            values=pd.DataFrame(
                np.asarray(stored[:]), index=index, columns=columns, copy=False
            ),
            perturbations=pd.DataFrame(
                {
                    "target": read_array(group, "targets").astype(str),
                    "is_control": read_array(group, "is_control").astype(bool),
                    "n_cells": read_array(group, "n_cells"),
                    "masked_feature": pd.Series(
                        [
                            None if code < 0 else str(feature_ids[code])
                            for code in masked
                        ],
                        dtype=object,
                        index=index,
                    ),
                },
                index=index,
            ),
            left_out=pd.DataFrame(
                {
                    "perturbation": read_array(group, "left_out").astype(str),
                    "target": read_array(group, "left_out_targets").astype(str),
                    "n_cells": read_array(group, "left_out_n_cells"),
                }
            ),
            features=pd.DataFrame(
                {
                    "feature_name": np.asarray(assay.feats.fetch_all("names")).astype(
                        str
                    )[positions],
                    "n_zero_sd": read_array(group, "n_zero_sd"),
                },
                index=columns,
            ),
            samples=pd.DataFrame(
                {"n_control_cells": read_array(group, "n_control_cells")}, index=samples
            ),
            control_mean=pd.DataFrame(
                read_array(group, "control_mean"), index=samples, columns=columns
            ),
            control_sd=pd.DataFrame(
                read_array(group, "control_sd"), index=samples, columns=columns
            ),
            by=str(parameters["by"]),
            from_assay=assay.name,
            sample_column=sample_columns[0] if sample_columns else None,
            min_cells=int(parameters["min_cells"]),
            mask_targets=bool(parameters["mask_targets"]),
            labels_kind=str(status.input_ref("labels").kind),
        )

    @validation_scoped
    def _perturbation_cluster(
        self,
        profiles: ArtifactRef,
        *,
        perturbations: Sequence[str] | None = None,
        metric: str = "correlation",
        min_cluster_size: int = 4,
        min_samples: int = 2,
        selection: str = "eom",
        seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Cluster the perturbations whose profiles change expression alike.

        As Replogle et al. (2022) did, the distance between two profiles is 1
        minus their Pearson correlation over the features, which ignores how
        strongly each perturbation acts; ``metric="euclidean"`` keeps it.
        HDBSCAN finds the clusters on these distances, through scikit-learn's
        ``HDBSCAN(metric="precomputed")`` with ``alpha=1`` and without a
        single cluster, and a perturbation it cannot place gets the cluster
        -1. ``min_samples`` follows scikit-learn, which counts the point
        itself, so the ``min_samples=1`` of the ``hdbscan`` package, which
        Replogle et al. ran with ``min_cluster_size=4`` and
        ``selection="eom"``, is 2 here, the default. Average linkage of the
        same distances orders the perturbations for a heatmap. A UMAP layout
        of each perturbation's 15 nearest perturbations by the same distances
        places them in two dimensions, from the first two principal
        components of the profiles, on one thread, so that it depends only on
        ``seed``.

        Args:
            profiles: A ``perturbation_profiles`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.profiles`.
            perturbations: The names of the perturbations to cluster, in any
                order, such as those with a significant E-distance and a low
                remaining fraction, or None for every profile.
            metric: ``"correlation"`` or ``"euclidean"``.
            min_cluster_size: The fewest perturbations of a cluster, at least
                2.
            min_samples: The perturbations within a perturbation's core
                distance, itself included, at least 1.
            selection: ``"eom"`` to select clusters by excess of mass, or
                ``"leaf"`` to take the leaves of HDBSCAN's cluster tree.
            seed: The seed of the layout, from 0 to 2**32 - 1.
            invalidate_cache: Cluster again instead of reusing saved clusters.

        Returns:
            A complete ``perturbation_clusters`` artifact scoped to the assay
            of the profiles, with one row per clustered perturbation. Read it
            with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_clusters`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``profiles`` is not a ``perturbation_profiles``
                artifact, ``perturbations`` names a perturbation without a
                profile, fewer than 3 perturbations or 2 features remain,
                ``min_samples`` exceeds the perturbations, or a setting is
                unknown or outside its range.
            PermissionError: If the store is read-only and the clusters are
                not saved yet.
            MemoryError: If the profiles and the distances between them,
                with their working copies, do not fit the memory budget.
        """
        from threadpoolctl import threadpool_limits

        from ...perturbation.clusters import (
            cluster_memory_bytes,
            cluster_profiles,
        )

        _require_ref(profiles, "profiles", PerturbationProfileArguments.artifact_kind)
        cluster_size, core_size, resolved_seed, requested = cluster_settings(
            metric, min_cluster_size, min_samples, selection, seed, perturbations
        )
        bool_argument(invalidate_cache, "invalidate_cache")
        require_complete_artifact(self.zw, profiles)
        group = artifact_group(self.zw, profiles)
        names = read_array(group, "perturbations").astype(str)
        if requested is None:
            chosen = np.arange(len(names))
        else:
            positions = pd.Index(names).get_indexer(requested)
            unknown = sorted(
                {
                    name
                    for name, position in zip(requested, positions, strict=True)
                    if position < 0
                }
            )
            if unknown:
                raise ValueError(
                    "perturbations names perturbations without a profile: "
                    + ", ".join(repr(name) for name in unknown[:5])
                    + "; the profiles hold the perturbations with at least "
                    "min_cells cells"
                )
            chosen = np.unique(positions)
        stored = as_zarr_array(group["values"], name="values")
        n_chosen = len(chosen)
        n_features = int(stored.shape[1])
        if n_chosen < 3 or n_features < 2:
            raise ValueError(
                "Clustering needs at least 3 perturbations and 2 features, not "
                f"{n_chosen} and {n_features}"
            )
        if core_size > n_chosen:
            raise ValueError(
                f"min_samples={core_size} must be at most the {n_chosen} "
                "perturbations clustered"
            )
        selected = names[chosen]
        arguments = PerturbationClusterArguments(
            perturbation_profiles=profiles,
            # Every profile, named or not, is the same input.
            perturbations_fingerprint=(
                None if n_chosen == len(names) else fingerprint_strings(selected)
            ),
            metric=metric,
            min_cluster_size=cluster_size,
            min_samples=core_size,
            selection=selection,
            seed=resolved_seed,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=str(profiles.assay),
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("perturbations", shape=(n_chosen,)),
                *(
                    ArrayRequirement(name, shape=(n_chosen,), dtype_kind="i")
                    for name in ("cluster", "order")
                ),
                ArrayRequirement("probability", shape=(n_chosen,), dtype_kind="f"),
                ArrayRequirement("embedding", shape=(n_chosen, 2), dtype_kind="f"),
            ),
        )
        if planned.reused:
            return planned.ref
        # The float32 profiles as read, and the clustering's peak.
        needed = 4 * n_chosen * n_features + cluster_memory_bytes(n_chosen, n_features)
        _require_budget(
            needed,
            self.memoryBytes,
            f"Clustering {n_chosen} perturbations in {n_features} features holds about",
            remedy="Pass fewer perturbations, or raise mem_budget.",
        )
        self._require_writable(arguments.operation)
        values = np.asarray(stored.oindex[chosen, :])
        # The distances' matrix products use as many BLAS threads as the store.
        with threadpool_limits(limits=self.nthreads, user_api="blas"):
            arrays = cluster_profiles(
                values,
                metric=metric,
                min_cluster_size=cluster_size,
                min_samples=core_size,
                selection=selection,
                seed=resolved_seed,
            )
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "perturbations": selected,
                "cluster": arrays.cluster,
                "probability": arrays.probability,
                "order": arrays.order,
                "embedding": arrays.embedding,
            },
        )
        logger.info(
            f"Clustered {n_chosen} perturbations into "
            f"{int(arrays.cluster.max(initial=-1)) + 1} clusters, with "
            f"{int(np.count_nonzero(arrays.cluster < 0))} unassigned"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_clusters(
        self, clusters: ArtifactRef
    ) -> PerturbationClusters:
        """Read saved perturbation clusters with the distances between their profiles.

        The distances are computed again from the saved profiles, as the
        clustering computed them.

        Args:
            clusters: A ``perturbation_clusters`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.cluster`.

        Returns:
            Each perturbation's cluster, membership strength, cells and place
            in the layout, the leaf order and the distances.

        Raises:
            TypeError: If ``clusters`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_clusters`` artifact.
            MemoryError: If the distances between the profiles do not fit the
                memory budget.
        """
        from threadpoolctl import threadpool_limits

        from ...perturbation.clusters import profile_distances

        _require_ref(clusters, "clusters", PerturbationClusterArguments.artifact_kind)
        status = require_complete_artifact(self.zw, clusters)
        group = artifact_group(self.zw, clusters)
        source_ref = status.input_ref("perturbation_profiles")
        require_complete_artifact(self.zw, source_ref)
        source = artifact_group(self.zw, source_ref)

        names = read_array(group, "perturbations").astype(str)
        positions = pd.Index(
            read_array(source, "perturbations").astype(str)
        ).get_indexer(names)
        values = as_zarr_array(source["values"], name="values")
        n_clustered = len(names)
        n_features = int(values.shape[1])
        # The distances and their symmetric copy, and the profiles as read,
        # as float64 and centred.
        n_bytes = 16 * n_clustered * n_clustered + 20 * n_clustered * n_features
        _require_budget(
            n_bytes,
            self.memoryBytes,
            f"The distances between {n_clustered} profiles hold about",
        )
        parameters = status.parameters or {}
        metric = str(parameters["metric"])
        with threadpool_limits(limits=self.nthreads, user_api="blas"):
            distances = profile_distances(
                np.asarray(values.oindex[positions, :]), metric
            )
        index = pd.Index(names, name="perturbation")
        embedding = read_array(group, "embedding")
        return PerturbationClusters(
            table=pd.DataFrame(
                {
                    "cluster": read_array(group, "cluster"),
                    "probability": read_array(group, "probability"),
                    "n_cells": read_array(source, "n_cells")[positions],
                    "x": embedding[:, 0],
                    "y": embedding[:, 1],
                },
                index=index,
            ),
            order=names[read_array(group, "order")],
            distances=pd.DataFrame(distances, index=index, columns=index.copy()),
            metric=metric,
            min_cluster_size=int(parameters["min_cluster_size"]),
            min_samples=int(parameters["min_samples"]),
            selection=str(parameters["selection"]),
            seed=int(parameters["seed"]),
            from_assay=str(clusters.assay),
        )
