from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import (
    PerturbationAssociationArguments,
    PerturbationLabelArguments,
)
from ...metadata.artifacts import write_cell_data_artifact
from ...metadata.rows import (
    apply_missing_mask,
    read_array_rows_chunkwise,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import cell_selection_indices
from ...perturbation.associations import (
    AssociationPairs,
    AssociationTables,
    PerturbationAssociations,
    association_settings,
    covariate_columns,
)
from ...perturbation.p_values import PairResults
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
)
from ...storage.artifacts import ArtifactRef, artifact_group, require_complete_artifact
from ...storage.types import as_zarr_array, read_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.numba import numba_threads
from ...utils.warnings import warn
from .perturbation_store import (
    _BAND_BYTES,
    _LabelRows,
    _cell_samples,
    _count_blocks,
    _label_groups,
    _label_rows,
    _pairs_fingerprint,
    _require_budget,
    _require_cell_column,
    _require_ref,
    _row_features,
)

if TYPE_CHECKING:
    from ..mapping_datastore import (
        MappingDatastore as _PerturbationAssociationOperationsBase,
    )
else:
    _PerturbationAssociationOperationsBase = object

# The share of the memory left beside the tests that held counts may take, so
# that the tests need not read the counts again.
_ASSOCIATION_HELD_SHARE = 0.5
# Bytes held for each pair of the association tests: its row and feature, QC
# counts and results, and the copies that the tests, the adjustment and the
# write make of them.
_ASSOCIATION_PAIR_BYTES = 320
# The integer and floating-point columns of a perturbation association table,
# one row per pair; each pair's group of treatment cells names its target and
# guides once per group.
_ASSOCIATION_COUNTS = (
    "sets",
    "groups",
    "features",
    "n_cells",
    "n_control_cells",
    "n_nonzero_cells",
    "n_nonzero_control_cells",
    "stage",
)
_ASSOCIATION_VALUES = (
    "z",
    "p_value",
    "adjusted_p_value",
    "log2_fold_change",
    "se_log2_fold_change",
    "skew_location",
    "skew_scale",
    "skew_shape",
)


def _association_warnings(
    unmatched: np.ndarray,
    n_rows: int,
    by: str,
    assay: str,
    attributes: Mapping[str, Any],
    moi: str,
) -> list[str]:
    """Describe what association tests warn about, whether computed or reused.

    Args:
        unmatched: The rows, targets or guides, without a feature for a
            positive control.
        n_rows: The rows of the labels.
        by: ``"target"`` or ``"guide"``.
        assay: The expression assay.
        attributes: The attributes of the tests: ``calibration_status``,
            ``n_control_guides``, ``calibration_grade``,
            ``n_false_discoveries`` and ``n_negative_control_pairs``.
        moi: ``"low"`` or ``"high"``, which choose the control guides of the
            calibration groups.
    """
    messages = []
    if len(unmatched):
        messages.append(
            f"{len(unmatched)} of {n_rows} {by}s have no feature of assay "
            f"{assay!r} for a positive control, for example "
            + ", ".join(repr(str(name)) for name in unmatched[:5])
            + ". A target that is not a gene, such as an enhancer, has none; for "
            "a gene, check id_column, or the target names against the feature "
            "names."
        )
    status = attributes["calibration_status"]
    if status == "few_control_guides" and moi == "high":
        messages.append(
            "No labelled cell carries a control guide, which a calibration group "
            "needs, so no calibration check was run"
        )
    elif status == "few_control_guides":
        n_guides = int(attributes["n_control_guides"])
        guides = "1 control guide" if n_guides == 1 else f"{n_guides} control guides"
        messages.append(
            f"Control cells with one guide carry {guides}, fewer than the 2 that "
            "a calibration group and its comparison cells need, so no calibration "
            "check was run"
        )
    elif status == "no_passing_discovery":
        messages.append(
            "No discovery pair passes QC, and by default the calibration check "
            "tests as many pairs as pass, so no calibration check was run; pass "
            "n_calibration_pairs to run one"
        )
    elif status == "no_passing_calibration":
        messages.append(
            "No pair of a calibration group and a feature passes QC, so no "
            "calibration check was run"
        )
    grade = attributes["calibration_grade"]
    if grade in ("moderate", "severe"):
        messages.append(
            f"The calibration check found {attributes['n_false_discoveries']} "
            f"false discoveries among {attributes['n_negative_control_pairs']} "
            f"negative-control pairs, a {grade} miscalibration; check the "
            "covariates and the control guides before trusting the discoveries"
        )
    return messages


class _PerturbationAssociationOperationsMixin(_PerturbationAssociationOperationsBase):
    @validation_scoped
    def _perturbation_associations(
        self,
        labels: ArtifactRef,
        *,
        from_assay: str = "RNA",
        pairs: str | pd.DataFrame = "trans",
        features: ArtifactRef | None = None,
        positive_controls: str | pd.DataFrame | None = "targets",
        id_column: str | None = "target_gene_id",
        integration: str = "union",
        moi: str = "low",
        resampling: str | None = None,
        sample_column: str | None = None,
        covariates: Sequence[str] | None = None,
        side: str = "both",
        n_resamples: Sequence[int] = (499, 4999, 24999),
        stage_threshold: float = 0.02,
        approximation: str = "skew_normal",
        min_nonzero: int = 7,
        n_calibration_pairs: int | None = None,
        alpha: float = 0.1,
        seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Test whether each perturbation changes each gene, with calibrated p-values.

        SCEPTRE's low-MOI test (Barry et al. 2024), the default, compares, for
        each pair of a perturbation and a gene, the cells that carry the
        perturbation alone with the control cells of the labels;
        ``moi="high"`` runs its high-MOI test, described below. A negative
        binomial regression of the gene's counts on the cells' covariates,
        fitted on those cells together under the null hypothesis, scores each
        cell, and the pair's statistic is the perturbed cells' standardized
        score. The fit takes Poisson coefficients and the size that maximizes
        the likelihood with the Poisson means held fixed. Its p-value comes
        from the same statistic over random sets of as many cells drawn from
        the perturbed and control cells: ``n_resamples[0]`` sets first; for a
        pair whose p-value is then at most ``stage_threshold``,
        ``n_resamples[1]`` fresh sets whose statistics a skew-normal fit
        extends to far smaller p-values; and when the fit's tail beyond those
        statistics' 99th percentile holds less than half of their own share
        there, ``n_resamples[2]`` fresh sets (``approximation="none"`` keeps
        the second round's p-values). The fit does not depend on which of the
        pair's cells carry the perturbation, so the test is exact when those
        cells are exchangeable under the null hypothesis.

        The covariates are log detected features and log UMIs of each cell in
        ``from_assay``, indicators of the samples of ``sample_column``, and the
        numeric cell columns of ``covariates``. Only the tested cells, the
        control cells and the cells of the tested targets or guides, need a
        sample label and covariate values that are not missing. A pair is
        tested when at least ``min_nonzero`` perturbed and ``min_nonzero``
        control cells express the gene, and p-values are adjusted by
        Benjamini-Hochberg within each set of pairs. The fold change is log2
        of the perturbed cells' counts over their means under the pair's fit,
        the estimator of the SCEPTRE book.

        Three sets of pairs are tested:

        - Discovery pairs: with ``pairs="trans"``, every target with every
          feature of ``features``; or the rows of a DataFrame of ``target``
          and ``feature``.
        - Calibration pairs: up to 100 distinct groups of control guides,
          drawn at random with as many guides as the median target and
          possibly sharing guides, paired at random with features of the whole
          assay that pass QC; as many pairs as the discovery pairs that pass
          QC by default. A group's treatment cells are the control cells
          whose one guide is in the group, and every other control cell is
          its comparison: the cells of the other control guides, and the
          control cells with several control guides, even guides of the
          group, since under the null hypothesis no control guide has an
          effect. The pairs should give no discovery; the grade of their
          false discoveries follows the SCEPTRE book: excellent (0 or 1), mild
          (up to 6), moderate (up to 14) or severe. The grade is ``"none"``
          when no check runs: with ``n_calibration_pairs=0``; with fewer than
          2 control guides that are the one guide of a control cell; by
          default, when no discovery pair passes QC; or when no pair of a
          group and a feature passes QC. A warning says why in all but the
          first case.
        - Positive-control pairs: with ``positive_controls="targets"``, each
          target with its own gene, matched as
          :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.efficiency`
          matches them; or a DataFrame of ``target`` and ``feature``.

        With ``integration="singleton"``, each guide is tested alone instead
        of the guides of a target together, and the calibration groups hold
        one control guide each.

        In a high-MOI screen each cell carries several guides, so few cells
        carry a target alone and almost none carry control guides only. With
        ``moi="high"``, as in SCEPTRE's high-MOI test (Barry et al. 2021,
        2024), a target's treatment cells are the labelled cells whose guides
        target it, or with ``"singleton"`` the cells that carry the guide,
        whatever else they carry, and every other labelled cell is its
        comparison: cells of other targets or of sibling guides, cells with
        control guides and cells without a called guide. Every labelled cell
        is tested, so each one needs a sample label, covariate values, and
        guide UMIs, since the covariates add log detected guides and log guide
        UMIs of the guide assay. Each gene is fitted once on all labelled
        cells, and all of its pairs share that fit.
        ``resampling="conditional"``, the default at high MOI, fits a logistic
        regression of each group's cells on the covariates and draws every
        cell into a random set with its fitted probability, the conditional
        randomization test, which stays valid when sequencing depth drives
        both the detection of guides and the expression of genes;
        ``"permutation"`` draws as many cells as the group uniformly from all
        cells. The calibration groups take any control guide called in a
        labelled cell, as many guides as the median target or all of them when
        fewer, and their treatment cells, the cells that carry any guide of
        the group, are compared with all other labelled cells, so one control
        guide is enough; the grade is ``"none"`` when no labelled cell carries
        one. QC counts the expressing cells of each group against all other
        cells.

        Args:
            labels: A ``perturbation_labels`` artifact.
            from_assay: The expression assay, whose counts are tested.
            pairs: ``"trans"``, or a DataFrame with columns ``target`` and
                ``feature`` (feature IDs, or names that one feature has).
            features: A ``feature_selection`` artifact of ``from_assay`` whose
                features trans pairs test, such as highly variable genes.
            positive_controls: ``"targets"``, a DataFrame like ``pairs``, or
                None.
            id_column: The feature column of the guide assay that holds each
                guide's target feature ID, or None to match targets by name.
            integration: ``"union"`` or ``"singleton"``.
            moi: ``"low"`` for a screen of one guide per cell, or ``"high"``
                for a screen of several guides per cell.
            resampling: ``"conditional"`` or ``"permutation"`` at high MOI,
                or None for ``"conditional"`` there and ``"permutation"``,
                the only choice, at low MOI.
            sample_column: A cell metadata column whose samples get
                indicators, or None.
            covariates: Numeric cell metadata columns to adjust for, or None.
            side: ``"both"``, ``"left"`` for a decrease, or ``"right"`` for an
                increase.
            n_resamples: The random sets of the three rounds, a sequence of
                three integers such as a tuple or a list.
            stage_threshold: The p-value at or below which a pair takes the
                second round.
            approximation: ``"skew_normal"`` or ``"none"``.
            min_nonzero: The fewest perturbed and control cells that express a
                tested gene.
            n_calibration_pairs: The calibration pairs, None for as many as
                the discovery pairs that pass QC, or 0 for no calibration
                check. A positive number runs the check without discovery
                pairs too.
            alpha: The false discovery rate of ``significant``.
            seed: The seed of the random sets and the calibration pairs.
            invalidate_cache: Test again instead of reusing saved tests.

        Returns:
            A complete ``perturbation_associations`` artifact scoped to
            ``from_assay``, with one row per pair. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_associations`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``labels`` is not a ``perturbation_labels``
                artifact, an assay, column, target or feature is unknown, a
                number is out of range, ``resampling="conditional"`` goes
                with ``moi="low"``, trans pairs have no ``features``, a table
                of pairs is malformed, at low MOI no labelled cell carries
                control guides only, or a tested cell has no sample label, a
                covariate value that is missing or not finite, or at high MOI
                no guide UMIs.
            PermissionError: If the store is read-only and the tests are not
                saved yet.
            MemoryError: If one feature's counts over the tested cells, or at
                high MOI the tests of one gene, do not fit the memory budget;
                the message names the smallest budget that runs the tests.
        """
        from ...perturbation.associations import (
            ASSOCIATION_SETS,
            CALIBRATION_STATUSES,
            pair_table_rows,
        )

        from ...storage.artifacts import fingerprint_array
        from ...storage.feature_selection import read_feature_selection_indices

        _require_ref(labels, "labels", PerturbationLabelArguments.artifact_kind)
        if not isinstance(from_assay, str) or from_assay not in self.assay_names:
            raise ValueError(f"from_assay must name an assay, not {from_assay!r}")
        if isinstance(pairs, str):
            if pairs != "trans":
                raise ValueError(f"pairs must be 'trans' or a DataFrame, not {pairs!r}")
            if not isinstance(features, ArtifactRef):
                raise ValueError(
                    "Trans pairs test the features of a feature selection; pass "
                    "features, such as highly variable genes"
                )
        elif features is not None:
            raise ValueError("features goes with pairs='trans' only")
        _require_ref(features, "features", "feature_selection", optional=True)
        (
            draws,
            threshold,
            floor,
            requested,
            rate,
            resolved_seed,
            resolved_resampling,
        ) = association_settings(
            positive_controls,
            id_column,
            integration,
            side,
            approximation,
            n_resamples,
            stage_threshold,
            min_nonzero,
            n_calibration_pairs,
            alpha,
            seed,
            moi,
            resampling,
            sample_column,
            covariates,
        )
        high = moi == "high"
        _require_cell_column(self, sample_column, "sample_column")
        covariate_names = covariate_columns(covariates)
        for column in covariate_names:
            _require_cell_column(self, column, "covariate")
        bool_argument(invalidate_cache, "invalidate_cache")
        by = "target" if integration == "union" else "guide"
        rows = _label_rows(self, labels, by)
        status = rows.status
        expression = self._get_assay(from_assay)
        cell_selection = status.input_ref("cell_selection")
        self._require_measured_cells(
            expression.name,
            cell_selection,
            operation=PerturbationAssociationArguments.operation,
        )
        feature_ids = np.asarray(expression.feats.fetch_all("ids")).astype(str)
        feature_names = np.asarray(expression.feats.fetch_all("names")).astype(str)
        n_rows = len(rows.names)
        # A table of pairs names the target of each row, by target or by guide.
        row_targets = rows.names if by == "target" else rows.target_names[rows.codes]

        def pair_fingerprint(pair_rows: np.ndarray, pair_features: np.ndarray) -> str:
            return _pairs_fingerprint(rows.names[pair_rows], feature_ids[pair_features])

        pairs_fingerprint = None
        if isinstance(pairs, str):
            assert features is not None
            if str(features.assay) != expression.name:
                raise ValueError(
                    f"features must be a feature selection of assay {expression.name!r}"
                )
            selected = np.asarray(
                read_feature_selection_indices(self.zw, expression.name, features),
                dtype=np.int64,
            )
            discovery = (
                np.repeat(np.arange(n_rows, dtype=np.int64), len(selected)),
                np.tile(selected, n_rows),
            )
        else:
            discovery = pair_table_rows(
                pairs, "pairs", row_targets, feature_ids, feature_names
            )
            pairs_fingerprint = pair_fingerprint(*discovery)
        positive_fingerprint = None
        unmatched = np.empty(0, dtype=str)
        if positive_controls is None:
            positive = (np.empty(0, np.int64), np.empty(0, np.int64))
        elif isinstance(positive_controls, str):
            row_features = _row_features(
                self, labels, rows, id_column, feature_ids, feature_names
            )
            matched = np.flatnonzero(row_features >= 0)
            positive = (
                matched.astype(np.int64),
                row_features[matched].astype(np.int64),
            )
            unmatched = rows.names[row_features < 0].astype(str)
            positive_fingerprint = pair_fingerprint(*positive)
        else:
            positive = pair_table_rows(
                positive_controls,
                "positive_controls",
                row_targets,
                feature_ids,
                feature_names,
            )
            positive_fingerprint = pair_fingerprint(*positive)
        if not len(discovery[0]) and not len(positive[0]):
            raise ValueError(
                "There is no pair to test; pass pairs or positive controls"
            )
        cell_index = cell_selection_indices(self.zw, cell_selection)
        n_labelled = len(cell_index)
        used = np.unique(np.concatenate([discovery[0], positive[0]]))
        compact = np.full(n_rows, -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        if high:
            # Every labelled cell is tested, as a treatment cell of its
            # groups and a comparison cell of the others.
            tested = np.arange(n_labelled, dtype=np.int64)
            needing = "every labelled cell needs one"
        else:
            # The tested cells: the control cells, then the cells of each
            # tested row, row by row. Only their samples and covariates enter
            # the tests.
            row_of, control = _label_groups(self, labels, rows, by)
            controls = np.flatnonzero(control)
            target_row = np.where(row_of >= 0, compact[np.maximum(row_of, 0)], -1)
            targets_cells = np.flatnonzero(~control & (target_row >= 0))
            targets_cells = targets_cells[
                np.argsort(target_row[targets_cells], kind="stable")
            ]
            tested = np.concatenate([controls, targets_cells])
            needing = f"every control cell and every cell of a tested {by} needs one"
        tested_cells = cell_index[tested]
        cell_samples = _cell_samples(
            self,
            sample_column,
            tested_cells,
            reason=f"Each sample gets an indicator, so {needing}",
        )
        samples = cell_samples.codes if sample_column is not None else None
        extra: np.ndarray | None = None
        covariates_fingerprint = None
        if covariate_names:
            columns = []
            for column in covariate_names:
                values = np.asarray(
                    read_metadata_rows_chunkwise(self.cells, column, tested_cells)
                )
                if values.dtype.kind not in "biuf":
                    raise ValueError(f"covariate '{column}' must be numeric")
                # A missing value becomes NaN, which the covariates refuse.
                missing = read_metadata_missing_rows_chunkwise(
                    self.cells, column, tested_cells
                )
                columns.append(apply_missing_mask(values.astype(np.float64), missing))
            extra = np.column_stack(columns)
            covariates_fingerprint = fingerprint_array(extra)
        arguments = PerturbationAssociationArguments(
            labels=labels,
            dataset_fingerprint=self._ensure_dataset_fingerprint(expression.name),
            feature_selection=features,
            pairs_fingerprint=pairs_fingerprint,
            positive_controls_fingerprint=positive_fingerprint,
            sample_assignments_fingerprint=cell_samples.assignments,
            covariates_fingerprint=covariates_fingerprint,
            integration=integration,
            moi=moi,
            resampling=resolved_resampling,
            side=side,
            n_resamples=draws,
            stage_threshold=threshold,
            approximation=approximation,
            min_nonzero=floor,
            n_calibration_pairs=requested,
            alpha=rate,
            seed=resolved_seed,
            sample_columns=cell_samples.columns,
            covariate_columns=covariate_names,
            invalidate_cache=invalidate_cache,
        )
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=expression.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                *(
                    ArrayRequirement(name, shape=(None,), dtype_kind="i")
                    for name in _ASSOCIATION_COUNTS
                ),
                *(
                    ArrayRequirement(name, shape=(None,), dtype_kind="f")
                    for name in _ASSOCIATION_VALUES
                ),
                ArrayRequirement("group_targets", shape=(None,)),
                ArrayRequirement("group_guides", shape=(None,)),
            ),
            required_attributes=(
                AttributeRequirement(
                    "set_names",
                    predicate=lambda value: list(value) == list(ASSOCIATION_SETS),
                ),
                AttributeRequirement("calibration_grade", expected_types=(str,)),
                AttributeRequirement(
                    "calibration_status",
                    predicate=lambda value: value in CALIBRATION_STATUSES,
                ),
                *(
                    AttributeRequirement(name, expected_types=(int,))
                    for name in (
                        "n_false_discoveries",
                        "n_negative_control_pairs",
                        "n_control_guides",
                    )
                ),
            ),
        )
        if planned.reused:
            # Saved tests warn as they did when they were computed.
            for message in _association_warnings(
                unmatched,
                n_rows,
                by,
                expression.name,
                artifact_group(self.zw, planned.ref).attrs,
                moi,
            ):
                warn(message)
            return planned.ref
        if high:
            tables = self._complement_association_tables(
                labels,
                rows,
                by,
                used,
                discovery,
                positive,
                expression,
                tested_cells,
                samples,
                extra,
                covariate_names,
                n_resamples=draws,
                stage_threshold=threshold,
                approximation=approximation,
                side=side,
                min_nonzero=floor,
                n_calibration_pairs=requested,
                seed=resolved_seed,
                resampling=resolved_resampling,
                operation=arguments.operation,
            )
        else:
            tables = self._control_association_tables(
                labels,
                rows,
                by,
                used,
                discovery,
                positive,
                expression,
                cell_index,
                controls,
                targets_cells,
                target_row,
                samples,
                extra,
                covariate_names,
                n_resamples=draws,
                stage_threshold=threshold,
                approximation=approximation,
                side=side,
                min_nonzero=floor,
                n_calibration_pairs=requested,
                seed=resolved_seed,
                operation=arguments.operation,
            )
        return self._save_association_tables(
            planned,
            tables,
            rows=rows,
            by=by,
            used=used,
            unmatched=unmatched,
            assay=expression.name,
            alpha=rate,
            requested=requested,
            moi=moi,
        )

    def _control_association_tables(
        self,
        labels: ArtifactRef,
        rows: _LabelRows,
        by: str,
        used: np.ndarray,
        discovery: tuple[np.ndarray, np.ndarray],
        positive: tuple[np.ndarray, np.ndarray],
        expression: Any,
        cell_index: np.ndarray,
        controls: np.ndarray,
        targets_cells: np.ndarray,
        target_row: np.ndarray,
        samples: np.ndarray | None,
        extra: np.ndarray | None,
        covariate_names: tuple[str, ...],
        *,
        n_resamples: tuple[int, int, int],
        stage_threshold: float,
        approximation: str,
        side: str,
        min_nonzero: int,
        n_calibration_pairs: int | None,
        seed: int,
        operation: str,
    ) -> AssociationTables:
        """Test each tested row against the control cells, and each calibration group.

        ``cell_index`` holds every labelled cell's position in the store, and
        ``controls`` and ``targets_cells`` the positions among them of the
        control cells and of the cells that carry a tested row alone, row by
        row, as ``target_row`` gives each labelled cell's tested row, or -1.
        A calibration group's treatment cells are the control cells whose one
        guide is in the group, compared with the other control cells. The
        counts are read once for the QC, every feature of the assay when a
        calibration check runs, and held for the tests when they fit a share
        of the memory left. The store must be writable for ``operation`` once
        the memory budget is checked.
        """
        from ...perturbation.associations import (
            CALIBRATION_GROUPS,
            TreatmentGroups,
            association_covariates,
            association_pairs,
            calibration_group_size,
            calibration_runs,
            most_calibration_pairs,
            negative_control_groups,
            nonzero_counts,
            pair_test_bytes,
            pair_tests,
        )
        from ...perturbation.p_values import STATISTICS_BYTES

        tested = np.concatenate([controls, targets_cells])
        tested_cells = cell_index[tested]
        calls = artifact_group(self.zw, rows.status.input_ref("guide_calls"))
        guide_names = read_array(calls, "guides").astype(str)
        label_group = artifact_group(self.zw, labels)
        target_is_control = read_array(label_group, "target_is_control").astype(bool)
        guide_targets_of = rows.guide_targets
        n_controls = len(controls)
        # Each control cell's one guide, or -1 when it carries several control
        # guides; only the labels of the control cells are read.
        one_guide = pd.Index(guide_names).get_indexer(
            np.asarray(
                read_array_rows_chunkwise(
                    as_zarr_array(label_group["guide_label"], name="guide_label"),
                    controls,
                )
            ).astype(str)
        )
        # The control guides that are the one guide of a control cell make up
        # the calibration groups.
        control_guides = np.flatnonzero(
            (guide_targets_of >= 0) & target_is_control[np.maximum(guide_targets_of, 0)]
        )
        control_guides = control_guides[np.isin(control_guides, one_guide)]
        n_control_guides = len(control_guides)
        # A group leaves at least one control guide for its control cells.
        group_size = min(
            calibration_group_size(
                by, rows.guide_targets, len(rows.target_names), rows.codes[used]
            ),
            max(n_control_guides - 1, 1),
        )
        nt_groups = (
            negative_control_groups(
                n_control_guides, group_size, CALIBRATION_GROUPS, seed
            )
            if calibration_runs(n_calibration_pairs, len(discovery[0]))
            and n_control_guides >= 2
            else []
        )
        n_groups_nt = len(nt_groups)
        guide_of_control = pd.Index(control_guides).get_indexer(one_guide)
        n_tested = len(tested)
        # Each group of treatment cells: the tested rows, whose cells follow
        # the control cells, then the calibration groups among the controls.
        row_sizes = np.bincount(target_row[targets_cells], minlength=len(used))
        nt_members = [
            np.flatnonzero(np.isin(guide_of_control, members)) for members in nt_groups
        ]
        nt_sizes = np.asarray([len(cells) for cells in nt_members], dtype=np.int64)
        groups = TreatmentGroups(
            members=np.concatenate(
                [np.arange(n_controls, n_tested, dtype=np.int64), *nt_members]
            ).astype(np.int64),
            offsets=np.concatenate(
                [[0], np.cumsum(np.concatenate([row_sizes, nt_sizes]))]
            ).astype(np.int64),
            among_controls=np.concatenate(
                [np.zeros(len(used), bool), np.ones(n_groups_nt, bool)]
            ),
            codes=np.concatenate(
                [used, len(rows.names) + np.arange(n_groups_nt)]
            ).astype(np.int64),
            n_controls=n_controls,
        )
        del nt_members
        design = association_covariates(
            expression._cell_count_totals(tested_cells),
            read_metadata_rows_chunkwise(
                self.cells, f"{expression.name}_nFeatures", tested_cells
            ),
            samples,
            extra,
            names=covariate_names,
        )
        n_covariates = design.shape[1]
        # The units of the QC counts: each tested row, each control guide, and
        # the control cells without one control guide. Calibration pairs may
        # take any feature, so the QC reads every feature then.
        n_units = len(used) + n_control_guides + 1
        unit = np.concatenate(
            [
                len(used)
                + np.where(guide_of_control >= 0, guide_of_control, n_control_guides),
                target_row[targets_cells],
            ]
        ).astype(np.int64)
        target_features_checked = np.unique(np.concatenate([discovery[1], positive[1]]))
        checked = (
            np.arange(int(expression.feats.N), dtype=np.int64)
            if n_groups_nt
            else target_features_checked
        )
        largest = int(np.diff(groups.offsets).max(initial=0))
        # Every discovery and positive-control pair, and the most calibration
        # pairs that the check can draw.
        n_pairs_most = (
            len(discovery[0])
            + len(positive[0])
            + most_calibration_pairs(
                n_groups_nt,
                int(expression.feats.N),
                len(discovery[0]),
                n_calibration_pairs,
            )
        )
        # Each labelled cell's position, row and flags; each tested cell's
        # positions, sample, unit and control guide, and its covariates as read
        # and as fitted; and the treatment cells of every group.
        resident = len(cell_index) * 48 + groups.members.nbytes
        resident += n_tested * (80 + 8 * (n_covariates + len(covariate_names)))
        resident += _ASSOCIATION_PAIR_BYTES * n_pairs_most
        resident += 4 * n_units * len(checked) + 9 * n_groups_nt * len(checked)
        # Each thread's buffers of one pair's fit and the records of a batch of
        # pairs, and one group's random sets of the largest round.
        scratch = self.nthreads * pair_test_bytes(n_controls + largest, n_covariates)
        one_set = 4 * max(n_resamples) * largest
        # One feature of a band: its float32 counts over the tested cells, the
        # block they are read from, and its QC counts and fit.
        per_feature = (
            24 * n_tested + 16 * n_units + 8 * n_covariates * (n_covariates + 2)
        )
        # The least that the tests hold: one thread's random sets, the second
        # round's statistics and one working copy no larger than them, and
        # one feature's counts.
        least_statistics = 8 * n_resamples[1]
        needed = resident + scratch + one_set + 2 * least_statistics + per_feature
        _require_budget(
            needed,
            self.memoryBytes,
            f"The counts of one feature over {n_tested} tested cells hold about",
            beside=" with the tests and their random sets",
            remedy=f"Raise mem_budget to at least {needed} bytes.",
        )
        # The random sets of the other threads and the statistics may each
        # take a quarter of the rest, and the counts take what they leave.
        spare = self.memoryBytes - needed
        draw_bytes = one_set + min((self.nthreads - 1) * one_set, spare // 4)
        statistics_bytes = max(least_statistics, min(STATISTICS_BYTES, spare // 4))
        free = self.memoryBytes - resident - scratch - draw_bytes - 2 * statistics_bytes
        band_cap = max(1, _BAND_BYTES // (8 * max(n_tested, 1)))
        width = min(max(len(checked), 1), free // per_feature, band_cap)
        # The counts are read once and held for the tests when they take at
        # most a share of the memory left, so that the reads keep the rest;
        # otherwise the tests read them again.
        held_bytes = 4 * n_tested * len(checked)
        hold = held_bytes <= _ASSOCIATION_HELD_SHARE * free
        self._require_writable(operation)
        # The reads keep clear of the held counts, which fill as they go.
        counts_of = self._band_reader(
            expression,
            tested_cells,
            resident + scratch + (held_bytes if hold else 0),
            per_feature,
        )

        held: np.ndarray | None = None
        if hold:
            width = min(width, max(1, (free - held_bytes) // per_feature))
            held = np.empty((len(checked), n_tested), dtype=np.float32)
        detected = np.zeros((n_units, len(checked)), dtype=np.int32)
        with numba_threads(self.nthreads):
            for start in range(0, len(checked), width):
                band = checked[start : start + width]
                detected[:, start : start + len(band)] = nonzero_counts(
                    counts_of(
                        band,
                        np.empty((len(band), n_tested), dtype=np.float32)
                        if held is None
                        else held[start : start + len(band)],
                    ),
                    unit,
                    n_units,
                )
        guide_detected = detected[len(used) : len(used) + n_control_guides]
        compact = np.full(len(rows.names), -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        # Each calibration group's expressing cells: those of its guides.
        calibration_counts = (
            np.stack([guide_detected[members].sum(axis=0) for members in nt_groups])
            if nt_groups
            else np.zeros((0, len(checked)), dtype=np.int64)
        )
        pairs = association_pairs(
            discovery,
            positive,
            row_groups=compact,
            sizes=np.concatenate([row_sizes, nt_sizes]),
            row_counts=detected[: len(used)],
            row_features=checked,
            calibration_counts=calibration_counts,
            checked=checked,
            reference=detected[len(used) :].sum(axis=0, dtype=np.int64),
            n_reference=n_controls,
            rows_in_reference=False,
            min_nonzero=min_nonzero,
            n_calibration_pairs=n_calibration_pairs,
            seed=seed,
        )
        del calibration_counts, guide_detected, detected

        def test(
            counts: np.ndarray, chosen: np.ndarray, rows_of: np.ndarray
        ) -> PairResults:
            return pair_tests(
                counts,
                design,
                groups,
                pairs.groups[chosen],
                rows_of,
                seed=seed,
                n_resamples=n_resamples,
                stage_threshold=stage_threshold,
                approximation=approximation,
                side=side,
                statistics_bytes=statistics_bytes,
                draw_bytes=draw_bytes,
            )

        # With the counts held, every pair is tested at once; otherwise a band
        # of features at a time, read again.
        with numba_threads(self.nthreads):
            pair_values, stage = self._test_bands(
                pairs, None if hold else width, held, checked, counts_of, n_tested, test
            )
        return AssociationTables(
            pairs=pairs,
            values=pair_values,
            stage=stage,
            group_size=group_size,
            control_guides=guide_names[control_guides],
            calibration_groups=nt_groups,
        )

    def _band_reader(
        self, expression: Any, cells: np.ndarray, held_bytes: int, per_feature: int
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a reader of a band of features' counts over ``cells`` into ``out``.

        The reader writes features x cells. The cells are read in store order,
        and each read keeps clear of ``held_bytes`` and of ``per_feature``
        bytes for each feature of the band.
        """
        order = np.argsort(cells, kind="stable")
        sorted_cells = cells[order]

        def counts_of(band: np.ndarray, out: np.ndarray) -> np.ndarray:
            _, blocks = _count_blocks(
                self,
                expression,
                band,
                sorted_cells,
                resident_bytes=held_bytes + per_feature * len(band),
                scratch_bytes_per_row=8 * len(band),
            )
            offset = 0
            for block in blocks:
                out[:, order[offset : offset + len(block)]] = np.asarray(block).T
                offset += len(block)
            return out

        return counts_of

    def _test_bands(
        self,
        pairs: AssociationPairs,
        width: int | None,
        held: np.ndarray | None,
        checked: np.ndarray,
        counts_of: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_cells: int,
        test: Callable[[np.ndarray, np.ndarray, np.ndarray], PairResults],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Test the pairs that pass QC, ``width`` features at a time or all at once.

        With ``held`` counts of the ``checked`` features over the ``n_cells``
        tested cells, every band reads them; otherwise ``counts_of`` reads
        each band's counts again. ``test`` tests the chosen pairs on a band's
        counts, given the row of each pair's feature among them. Returns each
        pair's values by their names in the table, NaN for an untested pair,
        and the round that gave its p-value, 0 for an untested pair.
        """
        n_pairs = len(pairs.sets)
        values = {name: np.full(n_pairs, np.nan) for name in _ASSOCIATION_VALUES}
        stage = np.zeros(n_pairs, dtype=np.int64)
        tested = np.unique(pairs.features[pairs.pass_qc])
        bands = (
            [tested]
            if width is None
            else [
                tested[start : start + width] for start in range(0, len(tested), width)
            ]
        )
        for band in bands:
            chosen = np.flatnonzero(pairs.pass_qc & np.isin(pairs.features, band))
            if held is not None:
                counts, rows_of = held, np.searchsorted(checked, pairs.features[chosen])
            else:
                counts = counts_of(
                    band, np.empty((len(band), n_cells), dtype=np.float32)
                )
                rows_of = np.searchsorted(band, pairs.features[chosen])
            result = test(counts, chosen, rows_of)
            for name in _ASSOCIATION_VALUES:
                if name != "adjusted_p_value":
                    values[name][chosen] = getattr(result, name)
            stage[chosen] = result.stage
            del counts, result
        return values, stage

    def _complement_association_tables(
        self,
        labels: ArtifactRef,
        rows: _LabelRows,
        by: str,
        used: np.ndarray,
        discovery: tuple[np.ndarray, np.ndarray],
        positive: tuple[np.ndarray, np.ndarray],
        expression: Any,
        cells: np.ndarray,
        samples: np.ndarray | None,
        extra: np.ndarray | None,
        covariate_names: tuple[str, ...],
        *,
        n_resamples: tuple[int, int, int],
        stage_threshold: float,
        approximation: str,
        side: str,
        min_nonzero: int,
        n_calibration_pairs: int | None,
        seed: int,
        resampling: str,
        operation: str,
    ) -> AssociationTables:
        """Test each tested row and calibration group against all other labelled cells.

        A row's treatment cells are the labelled cells whose guides target it,
        or with ``by="guide"`` that carry the guide, and a calibration group's
        those that carry any of its control guides; ``cells`` are every
        labelled cell's positions in the store. The counts are read once for
        the QC, every feature of the assay when a calibration check runs, and
        held for the tests when they fit half the memory left; the tests take
        bands of genes, each gene fitted once on all cells. The store must be
        writable for ``operation`` once the memory budget is checked.
        """
        from ...perturbation.associations import (
            CALIBRATION_GROUPS,
            association_covariates,
            association_pairs,
            calibration_group_size,
            calibration_runs,
            most_calibration_pairs,
            negative_control_groups,
        )
        from ...perturbation.high_moi import (
            GATHER_BYTES,
            ComplementGroups,
            cells_by_code,
            complement_tests,
            complement_thread_bytes,
            fit_detection_models,
            nonzero_group_counts,
        )
        from ...perturbation.p_values import STATISTICS_BYTES

        calls = artifact_group(self.zw, rows.status.input_ref("guide_calls"))
        label_group = artifact_group(self.zw, labels)
        guide_names = read_array(calls, "guides").astype(str)
        target_is_control = read_array(label_group, "target_is_control").astype(bool)
        guide_codes = read_array(calls, "guide_codes")
        n_cells = len(cells)
        n_rows = len(rows.names)
        n_discovery = len(discovery[0])
        # The control guides that some labelled cell carries make up the
        # calibration groups, which hold the median target's guides, or every
        # control guide when there are fewer; one is enough, since a group is
        # compared with all other cells.
        called = np.zeros(len(guide_names), dtype=bool)
        called[guide_codes[guide_codes >= 0]] = True
        targets_of = rows.guide_targets
        control_guides = np.flatnonzero(
            called & (targets_of >= 0) & target_is_control[np.maximum(targets_of, 0)]
        )
        n_control_guides = len(control_guides)
        group_size = min(
            calibration_group_size(
                by, rows.guide_targets, len(rows.target_names), rows.codes[used]
            ),
            max(n_control_guides, 1),
        )
        nt_groups = (
            negative_control_groups(
                n_control_guides, group_size, CALIBRATION_GROUPS, seed
            )
            if calibration_runs(n_calibration_pairs, n_discovery) and n_control_guides
            else []
        )
        n_groups_nt = len(nt_groups)
        # Each tested row's cells: the cells whose target codes hold its
        # target, or whose called guides hold its guide.
        if by == "target":
            row_codes = read_array(label_group, "target_codes")
            lookup = np.full(len(rows.target_names), -1, dtype=np.int64)
            lookup[rows.codes[used]] = np.arange(len(used))
        else:
            row_codes = guide_codes
            lookup = np.full(len(guide_names), -1, dtype=np.int64)
            lookup[pd.Index(guide_names).get_indexer(rows.names[used])] = np.arange(
                len(used)
            )
        row_members, row_offsets = cells_by_code(
            np.where(row_codes >= 0, lookup[np.maximum(row_codes, 0)], -1), len(used)
        )
        # Each calibration group's cells: those that carry any of its guides.
        slot = np.full(len(guide_names), -1, dtype=np.int64)
        slot[control_guides] = np.arange(n_control_guides)
        guide_members, guide_offsets = cells_by_code(
            np.where(guide_codes >= 0, slot[np.maximum(guide_codes, 0)], -1),
            n_control_guides,
        )
        del row_codes, guide_codes
        nt_members = [
            np.unique(
                np.concatenate(
                    [
                        guide_members[guide_offsets[g] : guide_offsets[g + 1]]
                        for g in group
                    ]
                )
            )
            for group in nt_groups
        ]
        del guide_members
        members = np.concatenate([row_members, *nt_members]).astype(np.int64)
        offsets = np.concatenate(
            [
                row_offsets,
                row_offsets[-1]
                + np.cumsum([len(cells_of) for cells_of in nt_members], dtype=np.int64),
            ]
        ).astype(np.int64)
        del row_members, nt_members
        sizes = np.diff(offsets)
        codes = np.concatenate([used, n_rows + np.arange(n_groups_nt)]).astype(np.int64)
        guide_assay = self._get_assay(str(labels.assay))
        design = association_covariates(
            expression._cell_count_totals(cells),
            read_metadata_rows_chunkwise(
                self.cells, f"{expression.name}_nFeatures", cells
            ),
            samples,
            extra,
            names=covariate_names,
            guide_umis=guide_assay._cell_count_totals(cells),
            guide_detected=read_metadata_rows_chunkwise(
                self.cells, f"{guide_assay.name}_nFeatures", cells
            ),
        )
        n_covariates = design.shape[1]
        # The QC counts each row's expressing cells in the features of its
        # pairs and each calibration group's in every feature, which its pairs
        # may take.
        n_features = int(expression.feats.N)
        target_features = np.unique(np.concatenate([discovery[1], positive[1]]))
        checked = (
            np.arange(n_features, dtype=np.int64) if n_groups_nt else target_features
        )
        target_column = np.full(n_features, -1, dtype=np.int64)
        target_column[target_features] = np.arange(len(target_features))
        n_pairs_most = (
            n_discovery
            + len(positive[0])
            + most_calibration_pairs(
                n_groups_nt, n_features, n_discovery, n_calibration_pairs
            )
        )
        # Each labelled cell's positions, sample, totals and covariates as read
        # and as fitted; every group's cells, with the cells' groups that the
        # QC finds; every pair's arrays; and the QC counts.
        threads = self.nthreads
        resident = n_cells * (144 + 8 * (n_covariates + len(covariate_names)))
        resident += 32 * len(members)
        resident += _ASSOCIATION_PAIR_BYTES * n_pairs_most
        resident += 4 * len(used) * len(target_features)
        resident += 13 * n_groups_nt * len(checked) + 8 * len(checked)
        # A feature of the QC pass: its float32 counts, the block they are read
        # from and its QC counts. A gene of a band of tests: its scores and
        # weights on every cell and its fit, beside its counts when they are
        # read again.
        per_feature = 24 * n_cells + 16 * len(sizes)
        per_gene = 16 * n_cells + 8 * n_covariates * (n_covariates + 2)
        # The least that the tests hold: each thread draws a group's
        # conditional sets in 24 bytes per cell and copies one of its genes,
        # 16 bytes per cell, which also bounds its gene and logistic fits; the
        # second round holds its statistics and one working copy no larger
        # than them; and the QC reads one feature and the tests one gene.
        one_copy = 16 * n_cells
        least_statistics = 8 * n_resamples[1]
        needed = (
            resident
            + threads * complement_thread_bytes(n_cells, 0)
            + 2 * least_statistics
            + per_feature
            + per_gene
        )
        _require_budget(
            needed,
            self.memoryBytes,
            f"Testing one gene over {n_cells} labelled cells holds about",
            beside=" with the tests and their random sets",
            remedy=f"Raise mem_budget to at least {needed} bytes.",
        )
        self._require_writable(operation)
        # Each thread's copy may take more genes, up to a quarter of the rest
        # among the threads, and the statistics a quarter of the rest; the
        # counts and the tests take what they leave.
        spare = self.memoryBytes - needed
        gather_bytes = one_copy + min(
            max(GATHER_BYTES - one_copy, 0), spare // (4 * threads)
        )
        statistics_bytes = max(least_statistics, min(STATISTICS_BYTES, spare // 4))
        scratch = threads * complement_thread_bytes(n_cells, gather_bytes)
        free = self.memoryBytes - resident - scratch - 2 * statistics_bytes
        # The counts are read once and held for the tests when they take at
        # most a share of the memory left and leave room for a feature and a
        # gene; otherwise the tests read them again.
        held_bytes = 4 * n_cells * len(checked)
        hold = (
            held_bytes <= _ASSOCIATION_HELD_SHARE * free
            and free - held_bytes >= per_feature + per_gene
        )
        room = free - held_bytes if hold else free
        # A band holds at most so many bytes of float64 values, or of scores
        # and weights, whatever the budget.
        qc_width = min(
            len(checked),
            room // per_feature,
            max(1, _BAND_BYTES // (8 * max(n_cells, 1))),
        )
        test_width = min(
            room // (per_gene + (0 if hold else per_feature)),
            max(1, _BAND_BYTES // (16 * max(n_cells, 1))),
        )
        # The reads keep clear of the held counts, which fill as they go.
        counts_of = self._band_reader(
            expression,
            cells,
            resident + scratch + (held_bytes if hold else 0),
            per_feature,
        )

        proper = (sizes > 0) & (sizes < n_cells)

        def counted_groups(chosen: np.ndarray) -> ComplementGroups:
            """Return the groups of ``chosen`` that hold some cells but not all."""
            kept = chosen[proper[chosen]]
            return ComplementGroups(
                members=np.concatenate(
                    [np.empty(0, dtype=np.int64)]
                    + [members[offsets[g] : offsets[g + 1]] for g in kept]
                ),
                offsets=np.concatenate([[0], np.cumsum(sizes[kept])]).astype(np.int64),
                codes=np.arange(len(kept), dtype=np.int64),
            )

        row_groups = np.arange(len(used))
        nt_groups_of = len(used) + np.arange(n_groups_nt)
        row_counting = counted_groups(row_groups)
        nt_counting = counted_groups(nt_groups_of)

        def group_counts(
            values: np.ndarray, chosen: np.ndarray, counting: ComplementGroups
        ) -> tuple[np.ndarray, np.ndarray]:
            """Count the expressing cells of the groups ``chosen`` and of all cells.

            An empty group holds none of a feature's expressing cells and a
            group of every cell holds them all.
            """
            in_groups, totals = nonzero_group_counts(values, counting)
            counted = np.zeros((len(chosen), values.shape[0]), dtype=np.int64)
            counted[proper[chosen]] = in_groups
            counted[sizes[chosen] == n_cells] = totals
            return counted, totals

        held: np.ndarray | None = None
        if hold:
            held = np.empty((len(checked), n_cells), dtype=np.float32)
        row_detected = np.zeros((len(used), len(target_features)), dtype=np.int32)
        nt_detected = np.zeros((n_groups_nt, len(checked)), dtype=np.int32)
        expressing = np.zeros(len(checked), dtype=np.int64)
        with numba_threads(self.nthreads):
            for start in range(0, len(checked), qc_width):
                band = checked[start : start + qc_width]
                span = slice(start, start + len(band))
                values = counts_of(
                    band,
                    np.empty((len(band), n_cells), dtype=np.float32)
                    if held is None
                    else held[span],
                )
                if n_groups_nt:
                    nt_detected[:, span], expressing[span] = group_counts(
                        values, nt_groups_of, nt_counting
                    )
                targeted = np.flatnonzero(target_column[band] >= 0)
                if len(targeted) == len(band):
                    counted, expressing[span] = group_counts(
                        values, row_groups, row_counting
                    )
                    row_detected[:, target_column[band]] = counted
                elif len(targeted):
                    counted, _ = group_counts(
                        values[targeted], row_groups, row_counting
                    )
                    row_detected[:, target_column[band[targeted]]] = counted
                del values
        del row_counting, nt_counting
        compact = np.full(n_rows, -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        pairs = association_pairs(
            discovery,
            positive,
            row_groups=compact,
            sizes=sizes,
            row_counts=row_detected,
            row_features=target_features,
            calibration_counts=nt_detected,
            checked=checked,
            reference=expressing,
            n_reference=n_cells,
            rows_in_reference=True,
            min_nonzero=min_nonzero,
            n_calibration_pairs=n_calibration_pairs,
            seed=seed,
        )
        del row_detected, nt_detected
        # The groups of the tested pairs keep their codes, which key their
        # random sets, so that the sets do not depend on the other groups.
        active = np.unique(pairs.groups[pairs.pass_qc])
        engine_group = np.full(len(sizes), -1, dtype=np.int64)
        engine_group[active] = np.arange(len(active))
        engine = ComplementGroups(
            members=np.concatenate(
                [np.empty(0, dtype=np.int64)]
                + [members[offsets[g] : offsets[g + 1]] for g in active]
            ),
            offsets=np.concatenate([[0], np.cumsum(sizes[active])]).astype(np.int64),
            codes=codes[active],
        )
        with numba_threads(self.nthreads):
            models = (
                fit_detection_models(design, engine)
                if resampling == "conditional" and len(active)
                else None
            )

            def test(
                counts: np.ndarray, chosen: np.ndarray, rows_of: np.ndarray
            ) -> PairResults:
                return complement_tests(
                    counts,
                    design,
                    engine,
                    engine_group[pairs.groups[chosen]],
                    rows_of,
                    resampling=resampling,
                    detection_models=models,
                    seed=seed,
                    n_resamples=n_resamples,
                    stage_threshold=stage_threshold,
                    approximation=approximation,
                    side=side,
                    statistics_bytes=statistics_bytes,
                    gather_bytes=gather_bytes,
                )

            pair_values, stage = self._test_bands(
                pairs, test_width, held, checked, counts_of, n_cells, test
            )
        return AssociationTables(
            pairs=pairs,
            values=pair_values,
            stage=stage,
            group_size=group_size,
            control_guides=guide_names[control_guides],
            calibration_groups=nt_groups,
        )

    def _save_association_tables(
        self,
        planned: PlannedArtifact,
        tables: AssociationTables,
        *,
        rows: _LabelRows,
        by: str,
        used: np.ndarray,
        unmatched: np.ndarray,
        assay: str,
        alpha: float,
        requested: int | None,
        moi: str,
    ) -> ArtifactRef:
        """Adjust the tested pairs' p-values, grade the calibration and save the tests.

        The p-values are adjusted by Benjamini-Hochberg within each set of
        pairs. The attributes also hold what the warnings name, so that saved
        tests warn again as they did.
        """
        from ...perturbation.associations import (
            ASSOCIATION_SETS,
            calibration_grade,
            calibration_status,
        )
        from ...perturbation.p_values import adjusted_p_values

        pairs, values_out = tables.pairs, tables.values
        sets, pass_qc = pairs.sets, pairs.pass_qc
        n_calibration = int(np.count_nonzero(sets == 1))
        significant = np.zeros(len(sets), dtype=bool)
        for code in range(len(ASSOCIATION_SETS)):
            members = np.flatnonzero(sets == code)
            adjusted, found = adjusted_p_values(
                values_out["p_value"][members], pass_qc[members], alpha
            )
            values_out["adjusted_p_value"][members] = adjusted
            significant[members] = found
        n_false = int(np.count_nonzero(significant & (sets == 1)))
        grade = calibration_grade(n_false) if n_calibration else "none"
        n_control_guides = len(tables.control_guides)
        attributes = {
            "set_names": list(ASSOCIATION_SETS),
            "calibration_grade": grade,
            "n_false_discoveries": n_false,
            "calibration_group_size": tables.group_size,
            "calibration_status": calibration_status(
                n_calibration,
                requested,
                n_control_guides,
                pairs.passing_discovery,
                moi,
            ),
            "n_control_guides": n_control_guides,
            "n_negative_control_pairs": n_calibration,
        }
        # Each group of treatment cells, the tested rows and then the
        # calibration groups, names its target and guides once, and each pair
        # names its group.
        n_groups_nt = len(tables.calibration_groups)
        row_targets = rows.names if by == "target" else rows.target_names[rows.codes]
        group_targets = np.concatenate(
            [
                row_targets[used].astype(object),
                np.asarray(
                    [f"control group {group + 1}" for group in range(n_groups_nt)],
                    dtype=object,
                ),
            ]
        )
        group_guides = np.concatenate(
            [
                np.full(len(used), "", dtype=object)
                if by == "target"
                else rows.names[used].astype(object),
                np.asarray(
                    [
                        "+".join(tables.control_guides[members])
                        for members in tables.calibration_groups
                    ],
                    dtype=object,
                ),
            ]
        )
        qc = pairs.qc
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "sets": sets,
                "groups": pairs.groups,
                "group_targets": group_targets,
                "group_guides": group_guides,
                "features": pairs.features,
                "n_cells": qc[0],
                "n_control_cells": qc[1],
                "n_nonzero_cells": qc[2],
                "n_nonzero_control_cells": qc[3],
                "pass_qc": pass_qc,
                "significant": significant,
                "stage": tables.stage,
                **values_out,
            },
            attributes=attributes,
        )
        for message in _association_warnings(
            unmatched, len(rows.names), by, assay, attributes, moi
        ):
            warn(message)
        logger.info(
            f"Tested {int(np.count_nonzero(pass_qc[sets == 0]))} of "
            f"{int(np.count_nonzero(sets == 0))} discovery pairs: "
            f"{int(np.count_nonzero(significant[sets == 0]))} significant at "
            f"{alpha}; calibration {grade} with {n_false} false discoveries in "
            f"{n_calibration} pairs"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_associations(
        self, tests: ArtifactRef
    ) -> PerturbationAssociations:
        """Read saved association tests.

        Args:
            tests: A ``perturbation_associations`` artifact from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.associations`.

        Returns:
            The discovery, calibration and positive-control pairs.

        Raises:
            TypeError: If ``tests`` is not an ArtifactRef.
            ValueError: If it is not a ``perturbation_associations`` artifact.
            MemoryError: If the tables of its pairs do not fit the datastore
                memory budget.
        """
        from ...perturbation.associations import ASSOCIATION_SETS

        _require_ref(tests, "tests", PerturbationAssociationArguments.artifact_kind)
        status = require_complete_artifact(self.zw, tests)
        group = artifact_group(self.zw, tests)
        assay = self._get_assay(str(tests.assay))

        def array(name: str) -> zarr.Array:
            return as_zarr_array(group[name], name=name)

        parameters = status.parameters or {}
        integration = str(parameters["integration"])
        numbers = (
            "n_cells",
            "n_control_cells",
            "n_nonzero_cells",
            "n_nonzero_control_cells",
            "pass_qc",
            "z",
            "p_value",
            "adjusted_p_value",
            "significant",
            "log2_fold_change",
            "se_log2_fold_change",
            "stage",
            "skew_location",
            "skew_scale",
            "skew_shape",
        )
        # Each pair names its group of treatment cells, whose target and
        # guides are stored once.
        group_targets = read_array(group, "group_targets").astype(str)
        group_guides = read_array(group, "group_guides").astype(str)
        feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        feature_names = np.asarray(assay.feats.fetch_all("names")).astype(str)
        # Each pair's numbers as read, in the table and in its set's table,
        # and each of its four texts as read and as a Python string, to which
        # both tables point.
        n_pairs = int(array("sets").shape[0])
        row_bytes = sum(
            np.dtype(array(name).dtype).itemsize
            for name in ("sets", "groups", "features", *numbers)
        )
        text_bytes = sum(
            5 * (texts.dtype.itemsize // 4) + 80
            for texts in (group_targets, group_guides, feature_ids, feature_names)
        )
        n_bytes = n_pairs * (3 * row_bytes + text_bytes)
        _require_budget(
            n_bytes,
            self.memoryBytes,
            f"The association tests of {n_pairs} pairs hold about",
        )
        feature_positions = read_array(group, "features")
        groups = read_array(group, "groups")
        table: dict[str, Any] = {
            "target": group_targets[groups],
            "feature": feature_ids[feature_positions],
            "feature_name": feature_names[feature_positions],
        }
        for name in numbers:
            table[name] = read_array(group, name)
        frame = pd.DataFrame(table)
        sets = read_array(group, "sets")
        parts = {}
        for code, name in enumerate(ASSOCIATION_SETS):
            members = sets == code
            part = frame[members].reset_index(drop=True)
            if name == "calibration":
                part.insert(1, "control_guides", group_guides[groups[members]])
            elif integration == "singleton":
                part.insert(1, "guide", group_guides[groups[members]])
            parts[name] = part
        return PerturbationAssociations(
            discovery=parts["discovery"],
            calibration=parts["calibration"],
            positive=parts["positive"],
            alpha=float(parameters["alpha"]),
            side=str(parameters["side"]),
            integration=integration,
            moi=str(parameters["moi"]),
            resampling=str(parameters["resampling"]),
            from_assay=str(tests.assay),
            n_false_discoveries=int(str(group.attrs["n_false_discoveries"])),
            calibration_grade=str(group.attrs["calibration_grade"]),
        )
