from collections.abc import Callable, Iterator, Mapping, Sequence
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
    metadata_missing_mask,
    read_rows,
    read_whole,
)
from ...metadata.selection import plan_cell_selection_read, read_cell_selection
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
from ...storage.async_execution import zarr_decoding_threads
from ...storage.geometry import array_geometry
from ...storage.partition import row_band
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument
from ...utils.logging import logger
from ...utils.memory_budget import MemoryLedger, hash_table_bytes
from ...utils.numba import numba_threads
from ...utils.warnings import warn
from .perturbation_store import (
    _BAND_BYTES,
    _LabelRows,
    _call_ledger,
    _cell_samples,
    _cell_selection_check_bytes,
    _cell_totals,
    _converted,
    _count_bands,
    _count_blocks,
    _count_stream_bytes,
    _feature_text,
    _frame_bytes,
    _label_groups,
    _label_rows,
    _pairs_fingerprint,
    _require_cell_column,
    _require_ref,
    _row_features,
    _table_write_bytes,
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


def _chunk_rows(codes: zarr.Array) -> int:
    """Return the rows of a stored chunk of a cells x slots array of codes."""
    n_rows = int(codes.shape[0])
    return min(row_band(array_geometry(codes), unit="chunk", fallback=n_rows), n_rows)


def _code_blocks(
    ledger: MemoryLedger, codes: zarr.Array, *, item: str
) -> Iterator[np.ndarray]:
    """Charge ``ledger`` as ``item`` the read of the rows of a cells x slots array of codes, a stored chunk at a time, and return the reads.

    Zarr decodes a whole chunk for any of its rows, so whole chunks are read.
    Each read holds its chunk as stored and decoded and the rows it returns,
    beside what the read before can still hold: a chunk on each of the
    ledger's codec threads, which the reads decode on, and Zarr's objects for
    both reads, as ``ArrayGeometry.heldReadBytes`` counts them; before the
    first read, the read before is the ledger's last, whose leftover the
    ledger counts already. The ledger is charged that before any read, and
    once the reads end it counts what the last can still hold.
    """
    step = max(_chunk_rows(codes), 1)
    n_rows = int(codes.shape[0])
    geometry = array_geometry(codes)
    # Codes are cells x slots, so stored in chunks.
    assert geometry is not None
    n_chunks = -(-n_rows // step) * geometry.bandChunks()
    held = geometry.bandHeldBytes(ledger.holding_threads(n_chunks))
    ledger.peak(
        item,
        geometry.storedChunkBytes() * geometry.bandChunks()
        + min(step, n_rows) * geometry.rowBytes()
        + max(0, held - ledger.left),
    )
    return _read_code_blocks(ledger, codes, step, held)


def _read_code_blocks(
    ledger: MemoryLedger, codes: zarr.Array, step: int, held: int
) -> Iterator[np.ndarray]:
    """Yield the rows of ``codes`` ``step`` rows at a time on the ledger's codec threads, then record that the last read can still hold ``held`` bytes."""
    for start in range(0, int(codes.shape[0]), step):
        with zarr_decoding_threads(ledger.codec_threads):
            block = np.asarray(codes[start : start + step])
        yield block
        del block
    ledger.leave(held)


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
            ``n_false_discoveries``, ``n_negative_control_pairs``,
            ``rows_without_statistic`` and
            ``calibration_groups_without_statistic``.
        moi: ``"low"`` or ``"high"``, which choose the control guides of the
            calibration groups and the comparison cells.
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
    elif status == "no_calibration_statistic":
        messages.append(
            "No calibration pair has a statistic, so no calibration check was run"
        )
    grade = attributes["calibration_grade"]
    if grade in ("moderate", "severe"):
        messages.append(
            f"The calibration check found {attributes['n_false_discoveries']} "
            f"false discoveries among {attributes['n_negative_control_pairs']} "
            f"negative-control pairs, a {grade} miscalibration; check the "
            "covariates and the control guides before trusting the discoveries"
        )
    # Within a pair's cells, a sample or a binary covariate that holds exactly
    # the treatment cells makes the variance of their score 0.
    rows = [str(name) for name in attributes["rows_without_statistic"]]
    if rows:
        noun = f"{len(rows)} {by}" + ("s" if len(rows) > 1 else "")
        holding = (
            "samples, or of a binary covariate, that hold no control cell"
            if moi == "low"
            else "samples or of a binary covariate"
        )
        messages.append(
            f"The pairs of {noun} pass QC but have no statistic, so they have no "
            "p-value or fold change and are not significant: "
            + ", ".join(repr(name) for name in rows)
            + f". The covariates determine which of each pair's cells carry the "
            f"{by}, as they do when its cells are exactly the cells of {holding}; "
            f"test such a {by} without that sample column or covariate."
        )
    groups = [str(name) for name in attributes["calibration_groups_without_statistic"]]
    if groups:
        noun = f"{len(groups)} calibration group" + ("s" if len(groups) > 1 else "")
        messages.append(
            f"The pairs of {noun} pass QC but have no statistic, so they have no "
            "p-value or fold change and are not counted as false discoveries: "
            + ", ".join(repr(name) for name in groups)
            + ". The covariates determine which of each pair's cells carry the "
            "group's guides, as they do when those are exactly the pair's cells "
            "of samples or of a binary covariate."
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
        Benjamini-Hochberg within each set of pairs. A pair that passes QC has
        no statistic when the covariates determine which of its cells carry
        the perturbation, as they do when a target's cells are exactly the
        cells of samples, or of a binary covariate, that hold no control cell:
        it gets no p-value or fold change, ``stage`` 0, and is not significant
        or counted as tested, and a warning names its target. The p-values count only the random sets that have a
        statistic. The fold change is log2 of the perturbed cells' counts over
        their means under the pair's fit, the estimator of the SCEPTRE book.

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
          default, when no discovery pair passes QC; when no pair of a group
          and a feature passes QC; or when no calibration pair has a
          statistic. A warning says why in all but the first case.
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
        cells, and a target whose cells are exactly the cells of samples, or
        of a binary covariate, has no statistic.

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
                second round: 0, which keeps the first round's p-values, or
                at least the smallest first-round p-value, 2 /
                (``n_resamples[0]`` + 1) with ``side="both"`` and 1 /
                (``n_resamples[0]`` + 1) otherwise, so 99 or 49 first-round
                sets for the default of 0.02.
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
                number is out of range, a positive ``stage_threshold`` is
                below the smallest first-round p-value,
                ``resampling="conditional"`` goes with ``moi="low"``, trans
                pairs have no ``features``, a table
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
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
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
        by = "target" if integration == "union" else "guide"
        ledger = _call_ledger(self, f"Testing the associations of the {by}s")
        rows = _label_rows(self, ledger, labels, by)
        status = rows.status
        expression = self._get_assay(from_assay)
        cell_selection = status.input_ref("cell_selection")
        self._require_measured_cells(
            expression.name,
            cell_selection,
            operation=PerturbationAssociationArguments.operation,
        )
        feature_ids = _feature_text(ledger, expression, "ids")
        feature_names = _feature_text(ledger, expression, "names")
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
            # Each discovery pair's row and feature.
            ledger.hold("the discovery pairs", 16 * n_rows * len(selected))
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
                self, ledger, labels, rows, id_column, feature_ids, feature_names
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
        ledger.peak(
            "checking the labels' cells against the cell IDs",
            _cell_selection_check_bytes(self),
        )
        cell_index = read_cell_selection(
            ledger,
            self.zw,
            plan_cell_selection_read(self.zw, cell_selection),
            item="the labelled cells' positions",
        )
        n_labelled = len(cell_index)
        # The tested rows, and each row's place among them.
        ledger.peak(
            "finding the tested rows", 32 * (len(discovery[0]) + len(positive[0]))
        )
        used = np.unique(np.concatenate([discovery[0], positive[0]]))
        compact = np.full(n_rows, -1, dtype=np.int64)
        compact[used] = np.arange(len(used))
        if high:
            # Every labelled cell is tested, as a treatment cell of its
            # groups and a comparison cell of the others: each one's position
            # among them and in the store.
            ledger.hold("the tested cells", 16 * n_labelled)
            tested = np.arange(n_labelled, dtype=np.int64)
            needing = "every labelled cell needs one"
        else:
            # The tested cells: the control cells, then the cells of each
            # tested row, row by row. Only their samples and covariates enter
            # the tests. Each labelled cell's tested row, the control cells,
            # the cells of the tested rows as found and ordered, and each
            # tested cell's position among the labelled cells and in the store.
            row_of, control = _label_groups(self, ledger, labels, rows, by)
            ledger.hold("the tested cells", 64 * n_labelled)
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
            ledger,
            sample_column,
            tested_cells,
            reason=f"Each sample gets an indicator, so {needing}",
        )
        samples = cell_samples.codes if sample_column is not None else None
        extra: np.ndarray | None = None
        covariates_fingerprint = None
        if covariate_names:
            extra = self._covariate_values(ledger, covariate_names, tested_cells)
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
                *(
                    AttributeRequirement(name, expected_types=(list,))
                    for name in (
                        "rows_without_statistic",
                        "calibration_groups_without_statistic",
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
                ledger,
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
                ledger,
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
            ledger,
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

    def _covariate_values(
        self, ledger: MemoryLedger, columns: Sequence[str], cells: np.ndarray
    ) -> np.ndarray:
        """Read numeric cell metadata columns at ``cells`` as float64 covariates, cells x columns, keeping them charged.

        Each column and its missing mask are read through ``ledger``; a missing
        value becomes NaN, which the covariates refuse.

        Raises:
            ValueError: If a column is not numeric.
        """
        n_cells = len(cells)
        ledger.hold("the covariates", 8 * n_cells * len(columns))
        extra = np.empty((n_cells, len(columns)))
        for position, column in enumerate(columns):
            before = ledger.charged
            item = f"the cells' covariate {column!r}"
            values = read_rows(ledger, self.cells._get_array(column), cells, item=item)
            if values.dtype.kind not in "biuf":
                raise ValueError(f"covariate '{column}' must be numeric")
            mask = metadata_missing_mask(self.cells, column)
            missing = (
                None
                if mask is None
                else read_rows(ledger, mask, cells, item=item).astype(bool, copy=False)
            )
            ledger.peak(item, 16 * n_cells)
            extra[:, position] = apply_missing_mask(values.astype(np.float64), missing)
            del values, missing
            ledger.release(ledger.charged - before)
        return extra

    def _control_association_tables(
        self,
        ledger: MemoryLedger,
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
        the memory budget is checked. Each step charges ``ledger``.
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
            pair_test_work,
            pair_tests,
        )

        n_tested = len(controls) + len(targets_cells)
        ledger.hold("the tested cells' positions", 16 * n_tested)
        tested = np.concatenate([controls, targets_cells])
        tested_cells = cell_index[tested]
        calls = artifact_group(self.zw, rows.status.input_ref("guide_calls"))
        guide_names = read_whole(ledger, calls, "guides", item="the guide IDs")
        label_group = artifact_group(self.zw, labels)
        item = "the targets' control flags"
        target_is_control = _converted(
            ledger,
            read_whole(ledger, label_group, "target_is_control", item=item),
            bool,
            item,
        )
        guide_targets_of = rows.guide_targets
        n_controls = len(controls)
        # Each control cell's one guide, or -1 when it carries several control
        # guides; only the labels of the control cells are read, and matched
        # to the guides as Python strings through a hash table of the guides.
        before = ledger.charged
        control_labels = read_rows(
            ledger,
            as_zarr_array(label_group["guide_label"], name="guide_label"),
            controls,
            item="the control cells' labels",
        )
        ledger.hold("each control cell's one guide", 8 * n_controls)
        ledger.peak(
            "matching the control cells' labels to the guides",
            _frame_bytes({"guides": guide_names, "labels": control_labels})
            + hash_table_bytes(len(guide_names)),
        )
        one_guide = pd.Index(guide_names).get_indexer(
            control_labels.astype(str, copy=False)
        )
        del control_labels
        ledger.release(ledger.charged - before - one_guide.nbytes)
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
        # Each control cell's control guide, the calibration groups' control
        # cells, and the groups of treatment cells: the tested rows, whose
        # cells follow the control cells, then the calibration groups among
        # the controls.
        ledger.hold(
            "the groups of treatment cells",
            8 * n_controls * (2 + n_groups_nt) + 8 * n_tested,
        )
        guide_of_control = pd.Index(control_guides).get_indexer(one_guide)
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
        design = self._association_design(
            ledger,
            association_covariates,
            expression,
            tested_cells,
            samples,
            extra,
            covariate_names,
        )
        n_covariates = design.shape[1]
        # The units of the QC counts: each tested row, each control guide, and
        # the control cells without one control guide. Calibration pairs may
        # take any feature, so the QC reads every feature then.
        n_units = len(used) + n_control_guides + 1
        # Each tested cell's unit, and the features checked.
        ledger.hold("the QC's units", 8 * n_tested + 8 * int(expression.feats.N))
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
        # Every pair's arrays, and the QC counts of each unit and calibration
        # group in each feature checked.
        ledger.hold(
            "the pairs and their QC counts",
            _ASSOCIATION_PAIR_BYTES * n_pairs_most
            + 4 * n_units * len(checked)
            + 9 * n_groups_nt * len(checked),
        )
        # One feature of a band: its float32 counts over the tested cells, the
        # block they are read from, and its QC counts and fit.
        per_feature = (
            24 * n_tested + 16 * n_units + 8 * n_covariates * (n_covariates + 2)
        )
        # The least that the tests hold: each thread's buffers, one thread's
        # random sets, the second round's statistics and one working copy no
        # larger than them, and one feature's counts; or, while those counts
        # are read, their stream.
        stream = _count_stream_bytes(
            expression, checked[:1], tested_cells, scratch_bytes_per_row=8
        )
        room = ledger.room
        scratch, draw_bytes, statistics_bytes = pair_test_work(
            None if room is None else room - per_feature,
            groups,
            n_covariates,
            n_resamples,
            self.nthreads,
        )
        least_tests = scratch + 4 * max(n_resamples) * largest + 16 * n_resamples[1]
        ledger.doing = f"Testing the pairs of {len(used)} {by}s over {n_tested} cells"
        ledger.peak(
            "the counts of one feature with the tests and their random sets",
            per_feature + max(least_tests, scratch + stream),
        )
        # The random sets and the statistics take what pair_test_work gives
        # them, and the counts what they leave.
        room = ledger.room
        free = (
            None if room is None else room - scratch - draw_bytes - 2 * statistics_bytes
        )
        band_cap = max(1, _BAND_BYTES // (8 * max(n_tested, 1)))
        width = min(
            max(len(checked), 1),
            band_cap if free is None else free // per_feature,
            band_cap,
        )
        # The counts are read once and held for the tests when they take at
        # most a share of the memory left; otherwise the tests read them again.
        held_bytes = 4 * n_tested * len(checked)
        hold = free is None or (
            held_bytes <= _ASSOCIATION_HELD_SHARE * free
            and free + draw_bytes + 2 * statistics_bytes - held_bytes
            >= per_feature + stream
        )
        self._require_writable(operation)
        if hold:
            ledger.hold("the counts held for the tests", held_bytes)
        # The reads keep clear of the held counts, which fill as they go.
        counts_of, bands_of = self._band_reader(
            ledger, expression, tested_cells, per_feature
        )

        held: np.ndarray | None = None
        if hold:
            if free is not None:
                width = min(width, max(1, (free - held_bytes) // per_feature))
            held = np.empty((len(checked), n_tested), dtype=np.float32)
        detected = np.zeros((n_units, len(checked)), dtype=np.int32)
        with numba_threads(self.nthreads):
            for start, stop in bands_of(checked, width):
                band = checked[start:stop]
                detected[:, start:stop] = nonzero_counts(
                    counts_of(
                        band,
                        None if held is None else held[start : start + len(band)],
                    ),
                    unit,
                    n_units,
                )
                ledger.release(per_feature * len(band))
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
        ledger.release(4 * n_units * len(checked) + 9 * n_groups_nt * len(checked))

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
                ledger=ledger,
            )

        # With the counts held, every pair is tested at once; otherwise a band
        # of features at a time, read again.
        with numba_threads(self.nthreads):
            pair_values, stage = self._test_bands(
                ledger,
                pairs,
                None if hold else lambda tested: bands_of(tested, width),
                held,
                checked,
                counts_of,
                per_feature,
                test,
            )
        return AssociationTables(
            pairs=pairs,
            values=pair_values,
            stage=stage,
            group_size=group_size,
            control_guides=guide_names[control_guides],
            calibration_groups=nt_groups,
        )

    def _association_design(
        self,
        ledger: MemoryLedger,
        association_covariates: Callable[..., np.ndarray],
        expression: Any,
        cells: np.ndarray,
        samples: np.ndarray | None,
        extra: np.ndarray | None,
        covariate_names: tuple[str, ...],
        *,
        guide_assay: Any = None,
    ) -> np.ndarray:
        """Read the tested cells' counts and build the covariates of the tests, keeping them charged.

        Each cell's UMIs and detected features in ``expression``, and with a
        ``guide_assay`` its guide UMIs and detected guides, are read through
        ``ledger``; building the covariates holds each column beside their
        stacked copy, which stays, and the counts read are let go.
        """
        n_cells = len(cells)
        before = ledger.charged
        umis = _cell_totals(ledger, expression, cells, item="the tested cells' UMIs")
        detected = read_rows(
            ledger,
            self.cells._get_array(f"{expression.name}_nFeatures"),
            cells,
            item="the tested cells' detected features",
        )
        guide_umis = guide_detected = None
        if guide_assay is not None:
            guide_umis = _cell_totals(
                ledger, guide_assay, cells, item="the tested cells' guide UMIs"
            )
            guide_detected = read_rows(
                ledger,
                self.cells._get_array(f"{guide_assay.name}_nFeatures"),
                cells,
                item="the tested cells' detected guides",
            )
        n_samples = (
            1 if samples is None or not len(samples) else int(np.max(samples)) + 1
        )
        n_columns = (
            3
            + (2 if guide_assay is not None else 0)
            + n_samples
            - 1
            + (0 if extra is None else int(extra.shape[1]))
        )
        kept = ledger.hold("the covariates as fitted", 8 * n_cells * n_columns)
        ledger.peak("building the covariates", 8 * n_cells * (n_columns + 2))
        design = association_covariates(
            umis,
            detected,
            samples,
            extra,
            names=covariate_names,
            guide_umis=guide_umis,
            guide_detected=guide_detected,
        )
        del umis, detected, guide_umis, guide_detected
        ledger.release(ledger.charged - before - kept)
        return design

    def _band_reader(
        self, ledger: MemoryLedger, expression: Any, cells: np.ndarray, per_feature: int
    ) -> tuple[
        Callable[[np.ndarray, np.ndarray | None], np.ndarray],
        Callable[[np.ndarray, int], list[tuple[int, int]]],
    ]:
        """Return a reader of a band of features' counts over ``cells``, and its bands.

        The reader holds ``per_feature`` bytes of ``ledger`` for each feature
        of the band, which its caller releases once done with the band, and
        writes features x cells into ``out``, or a new array without one. The
        cells are read in store order, each read within the ledger's room. The
        second function splits features into bands of at most ``most`` whose
        reads fit the ledger's room, as start and stop positions.
        """
        # The cells' order, and the cells in store order.
        ledger.hold("the tested cells in store order", 16 * len(cells))
        order = np.argsort(cells, kind="stable")
        sorted_cells = cells[order]

        def counts_of(band: np.ndarray, out: np.ndarray | None) -> np.ndarray:
            ledger.hold("the counts of a band of features", per_feature * len(band))
            if out is None:
                out = np.empty((len(band), len(cells)), dtype=np.float32)
            offset = 0
            for block in _count_blocks(
                self,
                ledger,
                expression,
                band,
                sorted_cells,
                scratch_bytes_per_row=8 * len(band),
                item="reading the counts of a band of features",
            ):
                out[:, order[offset : offset + len(block)]] = np.asarray(block).T
                offset += len(block)
            return out

        def bands_of(features: np.ndarray, most: int) -> list[tuple[int, int]]:
            return _count_bands(
                ledger,
                expression,
                features,
                (sorted_cells,),
                per_feature=per_feature,
                scratch_per_feature=8,
                most=most,
            )

        return counts_of, bands_of

    def _test_bands(
        self,
        ledger: MemoryLedger,
        pairs: AssociationPairs,
        bands_of: Callable[[np.ndarray], list[tuple[int, int]]] | None,
        held: np.ndarray | None,
        checked: np.ndarray,
        counts_of: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
        per_feature: int,
        test: Callable[[np.ndarray, np.ndarray, np.ndarray], PairResults],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Test the pairs that pass QC, in the bands of ``bands_of`` or all at once.

        With ``held`` counts of the ``checked`` features over the tested
        cells, every band reads them; otherwise ``bands_of`` splits the tested
        features into bands, and ``counts_of`` reads each band's counts again,
        holding ``per_feature`` bytes of ``ledger`` for each feature until the
        band is tested. ``test`` tests the chosen pairs on a band's counts,
        given the row of each pair's feature among them. Returns each pair's
        values by their names in the table, NaN for an untested pair, and the
        round that gave its p-value, 0 for an untested pair or one without a
        statistic.
        """
        n_pairs = len(pairs.sets)
        values = {name: np.full(n_pairs, np.nan) for name in _ASSOCIATION_VALUES}
        stage = np.zeros(n_pairs, dtype=np.int64)
        tested = np.unique(pairs.features[pairs.pass_qc])
        bands = (
            [tested]
            if bands_of is None
            else [tested[start:stop] for start, stop in bands_of(tested)]
        )
        for band in bands:
            chosen = np.flatnonzero(pairs.pass_qc & np.isin(pairs.features, band))
            if held is not None:
                counts, rows_of = held, np.searchsorted(checked, pairs.features[chosen])
            else:
                counts = counts_of(band, None)
                rows_of = np.searchsorted(band, pairs.features[chosen])
            result = test(counts, chosen, rows_of)
            for name in _ASSOCIATION_VALUES:
                if name != "adjusted_p_value":
                    values[name][chosen] = getattr(result, name)
            # No round gives a p-value to a pair without a statistic.
            stage[chosen] = np.where(np.isnan(result.z), 0, result.stage)
            del counts, result
            if held is None:
                ledger.release(per_feature * len(band))
        return values, stage

    def _complement_association_tables(
        self,
        ledger: MemoryLedger,
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
        writable for ``operation`` once the memory budget is checked. Each step
        charges ``ledger``.
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
            ComplementGroups,
            cells_by_code,
            code_counts,
            complement_test_work,
            complement_tests,
            complement_thread_bytes,
            fit_detection_models,
            nonzero_group_counts,
        )

        calls = artifact_group(self.zw, rows.status.input_ref("guide_calls"))
        label_group = artifact_group(self.zw, labels)
        guide_names = read_whole(ledger, calls, "guides", item="the guide IDs")
        item = "the targets' control flags"
        target_is_control = _converted(
            ledger,
            read_whole(ledger, label_group, "target_is_control", item=item),
            bool,
            item,
        )
        # Each labelled cell's called guides, and by target the targets of its
        # guides, as stored: cells x the most of one cell, padded with -1. A
        # few cells with many guides widen every row, so the codes are read a
        # stored chunk of rows at a time, first to count each code's cells and,
        # once the budget is checked, to place each group's cells.
        guide_codes = as_zarr_array(calls["guide_codes"], name="guide_codes")
        row_codes = (
            as_zarr_array(label_group["target_codes"], name="target_codes")
            if by == "target"
            else guide_codes
        )
        n_cells = len(cells)
        n_rows = len(rows.names)
        n_discovery = len(discovery[0])
        # Counting each guide's cells, and by target each target's, holds the
        # counts and one stored chunk of codes read at a time. The counts size
        # the rest of the budget, so this part is charged before they are read.
        counted_codes = "guide and target" if by == "target" else "guide"
        ledger.doing = (
            f"Counting the cells of each {counted_codes} over {n_cells} labelled cells"
        )
        ledger.hold(
            "the cells of each guide and target",
            8 * (len(guide_names) + len(rows.target_names)),
        )
        guide_cells = code_counts(
            _code_blocks(ledger, guide_codes, item="counting the cells of each guide"),
            len(guide_names),
        )
        # The control guides that some labelled cell carries make up the
        # calibration groups, which hold the median target's guides, or every
        # control guide when there are fewer; one is enough, since a group is
        # compared with all other cells.
        targets_of = rows.guide_targets
        control_guides = np.flatnonzero(
            (guide_cells > 0)
            & (targets_of >= 0)
            & target_is_control[np.maximum(targets_of, 0)]
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
        # target, or whose called guides hold its guide. Each calibration
        # group's: those that carry any of its guides, at most the cells of
        # its guides together.
        if by == "target":
            row_code = rows.codes[used]
            code_cells = code_counts(
                _code_blocks(
                    ledger, row_codes, item="counting the cells of each target"
                ),
                len(rows.target_names),
            )
        else:
            row_code = pd.Index(guide_names).get_indexer(rows.names[used])
            code_cells = guide_cells
        ledger.doing = f"Testing one gene over {n_cells} labelled cells"
        row_lookup = np.full(len(code_cells), -1, dtype=np.int64)
        row_lookup[row_code] = np.arange(len(used))
        row_sizes = code_cells[row_code]
        guide_sizes = guide_cells[control_guides]
        nt_most = np.asarray(
            [guide_sizes[group].sum() for group in nt_groups], dtype=np.int64
        )
        n_groups = len(used) + n_groups_nt
        codes = np.concatenate([used, n_rows + np.arange(n_groups_nt)]).astype(np.int64)
        guide_assay = self._get_assay(str(labels.assay))
        design = self._association_design(
            ledger,
            association_covariates,
            expression,
            cells,
            samples,
            extra,
            covariate_names,
            guide_assay=guide_assay,
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
        # Every group's cells, with the cells' groups that the QC finds, every
        # pair's arrays, and the QC counts.
        ledger.hold(
            "the groups' cells, the pairs and their QC counts",
            32 * int(row_sizes.sum() + nt_most.sum())
            + _ASSOCIATION_PAIR_BYTES * n_pairs_most
            + 4 * len(used) * len(target_features)
            + 13 * n_groups_nt * len(checked)
            + 16 * n_features,
        )
        threads = self.nthreads
        # A feature of the QC pass: its float32 counts, the block they are read
        # from and its QC counts. A gene of a band of tests: its scores and
        # weights on every cell and its fit, beside its counts when they are
        # read again.
        per_feature = 24 * n_cells + 16 * n_groups
        per_gene = 16 * n_cells + 8 * n_covariates * (n_covariates + 2)
        # The least that the tests hold: each thread draws a group's
        # conditional sets in 24 bytes per cell and copies one of its genes,
        # 16 bytes per cell, which also bounds its gene and logistic fits; the
        # second round holds its statistics and one working copy no larger
        # than them; and the QC reads one feature and the tests one gene.
        least_statistics = 8 * n_resamples[1]
        # The counts of one feature stream beside the threads' buffers, while
        # the tests hold nothing else.
        stream = _count_stream_bytes(
            expression, checked[:1], cells, scratch_bytes_per_row=8
        )
        ledger.peak(
            "the tests of one gene, their random sets and one feature's counts",
            threads * complement_thread_bytes(n_cells, 0)
            + per_feature
            + max(2 * least_statistics + per_gene, stream),
        )
        self._require_writable(operation)
        row_members, row_offsets = cells_by_code(
            _code_blocks(ledger, row_codes, item="placing the groups' cells"),
            row_lookup,
            row_sizes,
        )
        nt_members: list[np.ndarray] = []
        if n_groups_nt:
            # The control guides' cells, and the cells of one group's guides as
            # merged.
            ledger.peak(
                "placing the calibration groups' cells",
                8 * int(guide_sizes.sum()) + 32 * int(nt_most.max()),
            )
            slot = np.full(len(guide_names), -1, dtype=np.int64)
            slot[control_guides] = np.arange(n_control_guides)
            guide_members, guide_offsets = cells_by_code(
                _code_blocks(
                    ledger, guide_codes, item="placing the control guides' cells"
                ),
                slot,
                guide_sizes,
            )
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
        members = np.concatenate([row_members, *nt_members])
        offsets = np.concatenate(
            [
                row_offsets,
                row_offsets[-1]
                + np.cumsum([len(cells_of) for cells_of in nt_members], dtype=np.int64),
            ]
        ).astype(np.int64)
        del row_members, nt_members
        sizes = np.diff(offsets)
        # Each thread's copy may take more genes, up to a quarter of the rest
        # among the threads, and the statistics a quarter of the rest; the
        # counts and the tests take what they leave.
        room = ledger.room
        scratch, _gather, statistics_bytes = complement_test_work(
            None if room is None else room - per_feature - per_gene,
            n_cells,
            n_resamples,
            threads,
        )
        free = None if room is None else room - scratch - 2 * statistics_bytes
        # The counts are read once and held for the tests when they take at
        # most a share of the memory left and leave room for a feature and a
        # gene; otherwise the tests read them again.
        held_bytes = 4 * n_cells * len(checked)
        hold = free is None or (
            held_bytes <= _ASSOCIATION_HELD_SHARE * free
            and free - held_bytes >= per_feature + per_gene
            and free + 2 * statistics_bytes - held_bytes >= per_feature + stream
        )
        if free is None:
            room_left = None
        else:
            room_left = free - held_bytes if hold else free
        # A band holds at most so many bytes of float64 values, or of scores
        # and weights, whatever the budget.
        qc_cap = max(1, _BAND_BYTES // (8 * max(n_cells, 1)))
        test_cap = max(1, _BAND_BYTES // (16 * max(n_cells, 1)))
        qc_width = min(
            len(checked),
            qc_cap if room_left is None else room_left // per_feature,
            qc_cap,
        )
        test_width = min(
            test_cap
            if room_left is None
            else room_left // (per_gene + (0 if hold else per_feature)),
            test_cap,
        )
        if hold:
            ledger.hold("the counts held for the tests", held_bytes)
        # The reads keep clear of the held counts, which fill as they go.
        counts_of, bands_of = self._band_reader(ledger, expression, cells, per_feature)

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
            for start, stop in bands_of(checked, qc_width):
                band = checked[start:stop]
                span = slice(start, stop)
                values = counts_of(band, None if held is None else held[span])
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
                ledger.release(per_feature * len(band))
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
            if resampling == "conditional" and len(active):
                # Each thread fits a group's detection model within the bytes
                # that its sets and copy take in the tests.
                ledger.peak(
                    "fitting the groups' detection models",
                    threads * complement_thread_bytes(n_cells, 0),
                )
            models = (
                fit_detection_models(design, engine)
                if resampling == "conditional" and len(active)
                else None
            )

            def test(
                counts: np.ndarray, chosen: np.ndarray, rows_of: np.ndarray
            ) -> PairResults:
                # The scores and weights of the band's genes on every cell.
                scoring = ledger.hold(
                    "the scores and weights of a band of genes",
                    per_gene * len(np.unique(rows_of)),
                )
                tested = complement_tests(
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
                    ledger=ledger,
                )
                ledger.release(scoring)
                return tested

            def test_bands(tested: np.ndarray) -> list[tuple[int, int]]:
                """Bands of at most test_width genes, whose reads fit the
                budget when the counts are read again."""
                if hold:
                    return [
                        (start, min(start + test_width, len(tested)))
                        for start in range(0, len(tested), test_width)
                    ]
                return bands_of(tested, test_width)

            pair_values, stage = self._test_bands(
                ledger, pairs, test_bands, held, checked, counts_of, per_feature, test
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
        ledger: MemoryLedger,
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
        pairs; a pair without a p-value, such as one that passes QC without a
        statistic, is not significant, nor a false discovery. The attributes
        also hold what the warnings name, so that saved tests warn again as
        they did.
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
        # A pair is tested when it has a statistic, not merely when it passes
        # QC, and a calibration check runs on tested pairs only.
        with_statistic = pass_qc & ~np.isnan(values_out["z"])
        n_tested_calibration = int(np.count_nonzero(with_statistic & (sets == 1)))
        grade = calibration_grade(n_false) if n_tested_calibration else "none"
        n_control_guides = len(tables.control_guides)
        attributes = {
            "set_names": list(ASSOCIATION_SETS),
            "calibration_grade": grade,
            "n_false_discoveries": n_false,
            "calibration_group_size": tables.group_size,
            "calibration_status": calibration_status(
                n_calibration,
                n_tested_calibration,
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
        # The tested rows and calibration groups whose pairs pass QC without a
        # statistic, by name, which the warning names again on reuse.
        without = np.unique(pairs.groups[pass_qc & np.isnan(values_out["z"])])
        attributes["rows_without_statistic"] = (
            rows.names[used][without[without < len(used)]].astype(str).tolist()
        )
        attributes["calibration_groups_without_statistic"] = (
            group_targets[without[without >= len(used)]].astype(str).tolist()
        )
        qc = pairs.qc
        columns = {
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
        }
        ledger.peak("saving the tests", _table_write_bytes(columns))
        write_cell_data_artifact(self.zw, planned, columns, attributes=attributes)
        for message in _association_warnings(
            unmatched, len(rows.names), by, assay, attributes, moi
        ):
            warn(message)
        discovered = sets == 0
        n_without = int(np.count_nonzero(discovered & pass_qc & ~with_statistic))
        logger.info(
            f"Tested {int(np.count_nonzero(discovered & with_statistic))} of "
            f"{int(np.count_nonzero(discovered))} discovery pairs"
            + (
                f", and {n_without} more passed QC without a statistic"
                if n_without
                else ""
            )
            + f": {int(np.count_nonzero(significant[discovered]))} significant at "
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
            KeyError: If the store holds no such artifact.
            RuntimeError: If the artifact is incomplete, as a call that
                stopped before it finished leaves it.
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
        n_pairs = int(array("sets").shape[0])
        ledger = _call_ledger(self, f"Reading the association tests of {n_pairs} pairs")
        # Each pair names its group of treatment cells, whose target and
        # guides are stored once.
        group_targets = read_whole(
            ledger, group, "group_targets", item="the groups' targets"
        )
        group_guides = read_whole(
            ledger, group, "group_guides", item="the groups' guides"
        )
        feature_ids = _feature_text(ledger, assay, "ids")
        feature_names = _feature_text(ledger, assay, "names")
        feature_positions = read_whole(
            ledger, group, "features", item="the pairs' features"
        )
        groups = read_whole(ledger, group, "groups", item="the pairs' groups")
        sets = read_whole(ledger, group, "sets", item="the pairs' sets")
        numbers_read = {
            name: read_whole(ledger, group, name, item=f"the pairs' {name}")
            for name in numbers
        }
        # Each pair's numbers in the table and in its set's table, and each of
        # its four texts as read and as a Python string, to which both tables
        # point.
        row_bytes = sum(values.dtype.itemsize for values in numbers_read.values())
        text_bytes = sum(
            5 * (texts.dtype.itemsize // 4) + 80
            for texts in (group_targets, group_guides, feature_ids, feature_names)
        )
        ledger.peak(
            "the tables of the pairs", n_pairs * (2 * row_bytes + 24 + text_bytes)
        )
        table: dict[str, Any] = {
            "target": group_targets[groups],
            "feature": feature_ids[feature_positions],
            "feature_name": feature_names[feature_positions],
        }
        table.update(numbers_read)
        del numbers_read
        frame = pd.DataFrame(table)
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
