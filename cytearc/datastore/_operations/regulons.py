import time
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import zarr

from ...assay import RNAassay
from ...metadata.arguments import GrnArguments
from ...metadata.selection import NormalizationSpec
from ...regulons.results import GrnResult
from ...storage.artifact_writer import ArrayRequirement, artifact_transaction
from ...storage.artifacts import ArtifactRef, fingerprint_strings, inspect_artifact
from ...storage.feature_selection import (
    _feature_selection_values,
    resolve_feature_selection,
)
from ...storage.selections import (
    read_stored_selection_indices,
    validate_stored_selection_integrity,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import bool_argument, float_argument, integer_argument
from ...utils.arrays import array_digest
from ...utils.logging import logger
from ...utils.memory_budget import MemoryBudgetError, MemoryLedger, hash_table_bytes

if TYPE_CHECKING:
    from ...features.enrichment.results import EnrichmentResult
    from ...matrix import ChunkedArray
    from ..mapping_datastore import MappingDatastore as _RegulonOperationsBase
else:
    _RegulonOperationsBase = object

_GRN_LAYOUT = "tf_target_edges"
# What GRN inference reads: an assay's stored values, which are its raw
# counts, or the values of an assay imported from normalized values.
_GRN_COUNTS = "raw_counts"
_GRN_NORMALIZED_VALUES = "normalized_values"
_GRN_EDGE_ARRAYS = ("tf_index", "target_index", "importance", "correlation")
_GRN_ARRAYS = (*_GRN_EDGE_ARRAYS, "regulator_index")


def _regulators_digest(names: Sequence[str]) -> str:
    """Digest of a requested regulator list, independent of order and repeats."""
    return fingerprint_strings(np.asarray(sorted({str(name) for name in names})))


def _unique_name_positions(names: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Keep features whose name occurs once; return them and the dropped names."""
    values, counts = np.unique(names.astype(str), return_counts=True)
    repeated = set(values[counts > 1].tolist())
    keep = np.asarray([name not in repeated for name in names.astype(str)], dtype=bool)
    return keep, sorted(repeated)


def _write_grn_slot(
    slot: zarr.Group,
    *,
    attrs: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    from ...storage.arrays import create_metadata_column

    for key, value in attrs.items():
        slot.attrs[key] = value
    for name in _GRN_ARRAYS:
        values = arrays[name]
        create_metadata_column(
            slot,
            name,
            data=values,
            dtype=values.dtype,
            chunkSize=1_000_000,
        )


def _grn_slot_matches(group: zarr.Group, attrs: dict[str, Any]) -> bool:
    if any(group.attrs.get(key) != value for key, value in attrs.items()):
        return False
    # Planning runs this only after it found every required array.
    shapes = {as_zarr_array(group[name], name=name).shape for name in _GRN_EDGE_ARRAYS}
    regulators = as_zarr_array(group["regulator_index"], name="regulator_index")
    return (
        len(shapes) == 1
        and len(next(iter(shapes))) == 1
        and regulators.ndim == 1
        and regulators.shape[0] > 0
    )


# Bytes of one int64 row or position.
_INDEX_BYTES = np.dtype(np.int64).itemsize


def _scored_rows(
    ledger: MemoryLedger, scored: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """Return the rows of a score matrix that hold ``cells``, positions in the store.

    ``scored`` holds the position of each row's cell. Both are ascending.
    The rows stay charged to ``ledger``.

    Raises:
        ValueError: If the scores hold no row for one of ``cells``.
    """
    item = f"the rows of the scores of {len(cells)} cells"
    # The rows found, whether each lies among the scored cells, and the cells
    # found there.
    ledger.peak(item, (2 * _INDEX_BYTES + 1) * len(cells))
    rows = np.searchsorted(scored, cells)
    if not bool((rows < len(scored)).all()) or not np.array_equal(scored[rows], cells):
        raise ValueError(
            "cell_selection holds cells that scores did not score; pass a "
            "selection of the scored cells"
        )
    ledger.hold(item, rows.nbytes)
    return rows


def _score_blocks(
    ledger: MemoryLedger,
    result: "EnrichmentResult",
    rows: np.ndarray | None,
    *,
    nthreads: int,
    msg: str,
    scratch_bytes_per_row: int = 0,
) -> Iterator[np.ndarray]:
    """Stream the scores of ``result`` at ``rows``, every row for None, one block at a time, with the block charged to ``ledger``.

    A block holds the rows of one stored chunk, which a smaller block would
    decode again for each block, or as many rows as the ledger's room holds
    beside the chunk's decode and the index of ``rows``, at least one: each
    row's scores, what Zarr's read holds for them, and the
    ``scratch_bytes_per_row`` that the caller makes of each row. No block is
    read ahead. The block's bytes are held while the stream runs, which
    refuses a budget without room for one row before anything is read, and
    released when it ends.
    """
    data: ChunkedArray = result.data
    if rows is not None:
        # Selecting rows indexes every stored row, then keeps the selected.
        data_index = _INDEX_BYTES * (len(result.cell_index) + len(rows))
        ledger.peak(f"the index of the scores of {len(rows)} cells", data_index)
        data = data[rows]
    chunk_rows = min(data._block_size, max(1, data.shape[0]))
    per_row = -(-data._block_owned_bytes() // chunk_rows) + scratch_bytes_per_row
    fixed = data._resident_bytes() + data._max_decode_bytes()
    block_rows = chunk_rows
    if ledger.room is not None:
        block_rows = max(1, min(chunk_rows, (ledger.room - fixed) // max(1, per_row)))
    charged = ledger.hold(
        "the scores read one block of cells at a time", fixed + block_rows * per_row
    )
    try:
        yield from data._with_block_size(block_rows)._stream_blocks(
            nthreads=nthreads,
            msg=msg,
            prefetch=1,
            row_mask=None,
            resident_bytes=ledger.held_bytes - charged,
            memory_bytes=ledger.budget,
            scratch_bytes_per_row=scratch_bytes_per_row,
        )
    finally:
        ledger.release(charged)


class _RegulonOperationsMixin(_RegulonOperationsBase):
    @validation_scoped
    def _regulons_grn(
        self,
        cell_selection: ArtifactRef,
        *,
        features: ArtifactRef,
        tfs: Iterable[str],
        from_assay: str | None = None,
        random_seed: int = 0,
        learning_rate: float = 0.01,
        max_stages: int = 5000,
        max_features: float = 0.1,
        subsample: float = 0.9,
        early_stop_window: int = 25,
        early_stopping: Literal["stage", "consecutive"] = "stage",
        nthreads: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Infer transcription-factor to target importances with GRNBoost2.

        For each selected gene, a stochastic gradient boosting regressor
        predicts its stored values across the selected cells from those of
        the selected transcription factors, as arboreto's GRNBoost2 does for
        pySCENIC: the same regressor settings and early stopping, the same
        seed for every target, a factor never predicting itself, and
        importances scaled by the number of fitted stages. The Pearson
        correlation of each linked factor and target is stored with it. The
        stored values are raw counts, or the values of an assay imported from
        normalized values, and the artifact's ``values`` parameter records
        which: ``"raw_counts"`` or ``"normalized_values"``.

        The values of the selected cells and genes are held in memory as
        float32, so select a representative subset of cells for large data.

        Args:
            cell_selection: Cells whose counts are modeled.
            features: Feature selection of the genes modeled as targets; the
                factors among them are the regulators. Genes whose name occurs
                more than once in the selection are left out, because regulons
                name genes.
            tfs: Candidate transcription factor names, such as a species TF
                list. Names are matched exactly to feature names.
            from_assay: RNA assay to model. The default assay when omitted.
            random_seed: Random state of every target's regressor.
            learning_rate: Shrinkage of each boosting stage.
            max_stages: Most boosting stages per target.
            max_features: Fraction of the regulators drawn at each split.
            subsample: Fraction of cells drawn for each stage, below 1.
            early_stop_window: Stop a target once the mean out-of-bag
                improvement of this many latest stages is negative.
            early_stopping: ``"stage"`` measures each stage's improvement on
                its own out-of-bag cells, as GRNBoost2 does in the pySCENIC
                0.12.1 container; ``"consecutive"`` compares consecutive
                stages, as arboreto does with scikit-learn 1.3 and later.
            nthreads: The most targets fitted at once. The store's thread
                count when omitted. Targets are fitted in this process, and
                with ``nthreads`` above 1, those left after about two seconds
                of fitting go to that many worker processes, fewer when the
                memory budget holds fewer of their copies of the regulators
                and their fits. Worker processes start as new interpreters
                that import the main module, so a script keeps its own work
                under ``if __name__ == "__main__":``. It does not change the
                result.
            invalidate_cache: Recompute even when a matching result exists.

        Returns:
            A complete ``grn_adjacencies`` artifact.

        Raises:
            MemoryError: If the counts do not fit the memory budget.
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
            UnmeasuredCellsError: If the assay did not measure a selected cell.
        """
        invalidate_cache = bool_argument(invalidate_cache, "invalidate_cache")
        from ...regulons.grn import (
            GrnParameters,
            _grn_links,
            _pair_correlations,
            grn_fit_bytes,
            grn_process_bytes,
        )

        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if not isinstance(features, ArtifactRef):
            raise TypeError("features must be an ArtifactRef")
        if isinstance(tfs, str) or not isinstance(tfs, Iterable):
            raise TypeError("tfs must be a sequence of names")
        tfs = list(tfs)
        if not all(isinstance(name, str) for name in tfs):
            raise TypeError("tfs must be a sequence of names")
        if not tfs:
            raise ValueError("tfs must name at least one transcription factor")
        # scikit-learn accepts 32-bit unsigned random states only.
        random_seed = integer_argument(
            random_seed, "random_seed", minimum=0, maximum=2**32 - 1
        )
        parameters = GrnParameters(
            learningRate=learning_rate,
            maxStages=max_stages,
            maxFeatures=max_features,
            subsample=subsample,
            earlyStopWindow=early_stop_window,
            earlyStopping=early_stopping,
        )
        threads = integer_argument(
            self.nthreads if nthreads is None else nthreads, "nthreads", minimum=1
        )
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("regulons.grn can only be run on an RNAassay")
        self._require_measured_cells(
            assay.name, cell_selection, operation="run_grn_inference"
        )
        self._require_writable("regulons.grn")
        self._ensure_dataset_fingerprint(assay.name)
        feature_selection = resolve_feature_selection(self.zw, assay.name, features)
        feature_index = np.flatnonzero(
            _feature_selection_values(self.zw, feature_selection)
        ).astype(np.int64)
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(cell_index) < 2:
            raise ValueError("GRN inference needs at least two selected cells")
        names = np.asarray(assay.feats.fetch_all("names"))[feature_index].astype(str)
        keep, dropped = _unique_name_positions(names)
        if dropped:
            logger.warning(
                f"Leaving {len(dropped)} genes with repeated names out of GRN "
                "inference: " + ", ".join(dropped[:10])
            )
        feature_index = feature_index[keep]
        names = names[keep]
        requested = set(tfs)
        regulator_mask = np.isin(names, list(requested))
        if not regulator_mask.any():
            raise ValueError("None of tfs names a selected feature")

        n_cells = len(cell_index)
        n_genes = len(feature_index)
        n_regulators = int(regulator_mask.sum())
        logger.info(
            f"{n_regulators} of {len(requested)} requested factors are selected"
        )

        arguments = GrnArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            regulators_digest=_regulators_digest(tfs),
            method="grnboost2",
            values=_GRN_COUNTS if assay.holdsCounts else _GRN_NORMALIZED_VALUES,
            learning_rate=float(learning_rate),
            max_stages=int(max_stages),
            max_features=float(max_features),
            subsample=float(subsample),
            early_stop_window=int(early_stop_window),
            early_stopping=str(early_stopping),
            random_seed=int(random_seed),
            nthreads=int(threads),
            invalidate_cache=invalidate_cache,
        )
        attrs: dict[str, Any] = {
            "cell_digest": array_digest(cell_index),
            "dropped_repeated_names": dropped,
            "feature_digest": array_digest(feature_index),
            "layout": _GRN_LAYOUT,
            "method": "grnboost2",
            "n_cells": n_cells,
            "regulator_digest": array_digest(feature_index[regulator_mask]),
        }
        planned = arguments.plan(
            self.zw,
            scope="assay",
            assay=assay.name,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("tf_index", dtype_kind="i"),
                ArrayRequirement("target_index", dtype_kind="i"),
                ArrayRequirement("importance", dtype_kind="f"),
                ArrayRequirement("correlation", dtype_kind="f"),
                ArrayRequirement(
                    "regulator_index", shape=(n_regulators,), dtype_kind="i"
                ),
            ),
            required_attributes=("method", "layout"),
            reuse_validator=lambda _ref, group: _grn_slot_matches(group, attrs),
        )
        if planned.reused:
            return planned.ref
        # The matrix, the regulators' copy, the correlation blocks, and about
        # 64 bytes per possible link while the links are sorted, correlated,
        # and written.
        held = 4 * n_cells * (n_genes + n_regulators)
        held += 8 * n_cells * 2 * max(n_regulators, 256)
        held += 64 * n_genes * n_regulators
        # One target's fit in this process, or each worker process's copy of
        # the regulators and its fit; the workers start once this process has
        # stopped fitting.
        needed = held + grn_fit_bytes(n_cells, n_regulators, max_stages)
        worker_bytes = grn_process_bytes(n_cells, n_regulators, max_stages)
        if needed > self.resources.memoryBytes:
            raise MemoryBudgetError(
                f"GRN inference over {n_cells} cells and {n_genes} genes needs "
                f"about {needed} bytes, but the memory budget is "
                f"{self.resources.memoryBytes} bytes; select fewer cells or genes",
                needed_bytes=needed,
                budget_bytes=int(self.resources.memoryBytes),
                item="the expression matrix and the links of GRN inference",
            )
        workers = threads
        while (
            workers > 1 and held + workers * worker_bytes > self.resources.memoryBytes
        ):
            workers -= 1
        if workers < threads:
            logger.info(
                f"GRN inference fits targets on {workers} of {threads} workers: "
                f"each worker process holds about {worker_bytes} bytes for its "
                "copy of the regulators and its fit, and the memory budget of "
                f"{self.resources.memoryBytes} bytes holds {workers} beside "
                f"{held} bytes of expression and links"
            )

        from ...assay.normalization import feature_value_blocks

        expression = np.empty((n_cells, n_genes), dtype=np.float32)
        _, blocks = feature_value_blocks(
            assay,
            feature_index,
            cell_index,
            NormalizationSpec(source="raw", transform="none"),
            nthreads=threads,
            resident_bytes=expression.nbytes,
            scratch_bytes_per_row=8 * n_genes,
        )
        row = 0
        for block in blocks:
            values = np.asarray(block)
            expression[row : row + values.shape[0]] = values
            row += values.shape[0]

        started = time.monotonic()
        reported = [started]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if done == total or now - reported[0] >= 60:
                reported[0] = now
                logger.info(
                    f"GRN inference: {done}/{total} targets "
                    f"({now - started:.0f} s elapsed)"
                )

        tf_codes, target_codes, importance = _grn_links(
            expression,
            names.tolist(),
            np.flatnonzero(regulator_mask),
            np.arange(n_genes),
            seed=random_seed,
            parameters=parameters,
            nthreads=workers,
            progress=progress,
        )
        correlation = _pair_correlations(expression, tf_codes, target_codes)
        import sklearn

        attrs_written = {**attrs, "sklearn_version": sklearn.__version__}
        with artifact_transaction(self.zw, planned) as slot:
            _write_grn_slot(
                slot,
                attrs=attrs_written,
                arrays={
                    "tf_index": np.asarray(feature_index[tf_codes], dtype=np.int64),
                    "target_index": np.asarray(
                        feature_index[target_codes], dtype=np.int64
                    ),
                    "importance": importance,
                    "correlation": np.asarray(correlation, dtype=np.float64),
                    "regulator_index": np.asarray(
                        feature_index[regulator_mask], dtype=np.int64
                    ),
                },
            )
        return planned.ref

    @validation_scoped
    def _regulons_load_grn(self, grn: ArtifactRef) -> GrnResult:
        """Load a GRN artifact as a table of named links.

        Args:
            grn: Artifact returned by ``regulons.grn``.

        Returns:
            The links with factor and target names, and the inputs they were
            inferred from.
        """
        if not isinstance(grn, ArtifactRef):
            raise TypeError("grn must be an ArtifactRef")
        if grn.kind != "grn_adjacencies" or grn.scope != "assay" or grn.assay is None:
            raise ValueError("grn must identify an assay grn_adjacencies artifact")
        assay = self._get_assay(grn.assay)
        status = inspect_artifact(self.zw, grn)
        label = grn.artifact_id[:12]
        if not status.complete or status.operation != GrnArguments.operation:
            raise ValueError(f"GRN artifact {label!r} is missing or incomplete")
        group = as_zarr_group(self.zw[status.path], name=status.path)
        if (
            group.attrs.get("layout") != _GRN_LAYOUT
            or group.attrs.get("method") != "grnboost2"
        ):
            raise ValueError(f"GRN artifact {label!r} has an unsupported layout")
        parameters = status.parameters or {}
        inputs = status.inputs or {}
        try:
            cell_selection = ArtifactRef.from_dict(inputs["cell_selection"])
            feature_selection = ArtifactRef.from_dict(inputs["feature_selection"])
            random_seed = int(parameters["random_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"GRN artifact {label!r} has invalid provenance") from exc
        validate_stored_selection_integrity(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        feature_selection = resolve_feature_selection(
            self.zw, assay.name, feature_selection
        )
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if array_digest(cell_index) != group.attrs.get("cell_digest"):
            raise ValueError(f"GRN artifact {label!r} cells do not match its input")
        arrays = {
            name: np.asarray(as_zarr_array(group[name], name=name)[:])
            for name in _GRN_ARRAYS
        }
        lengths = {len(arrays[name]) for name in _GRN_EDGE_ARRAYS}
        if len(lengths) != 1:
            raise ValueError(f"GRN artifact {label!r} has misaligned arrays")
        names = np.asarray(assay.feats.fetch_all("names")).astype(str)
        n_features = len(names)
        regulator_index = arrays["regulator_index"].astype(np.int64)
        if array_digest(regulator_index) != group.attrs.get("regulator_digest"):
            raise ValueError(f"GRN artifact {label!r} has mismatched regulators")
        for key in ("tf_index", "target_index", "regulator_index"):
            values = arrays[key]
            if np.any(values < 0) or np.any(values >= n_features):
                raise ValueError(f"GRN artifact {label!r} has invalid {key}")
        adjacencies = pd.DataFrame(
            {
                "TF": names[arrays["tf_index"]],
                "target": names[arrays["target_index"]],
                "importance": arrays["importance"].astype(np.float64),
                "correlation": arrays["correlation"].astype(np.float64),
            }
        )
        regulators = tuple(names[regulator_index].tolist())
        return GrnResult(
            adjacencies=adjacencies,
            regulators=regulators,
            artifact=grn,
            assay=assay.name,
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            method="grnboost2",
            random_seed=random_seed,
        )

    def _regulons_profiles(
        self,
        groups: ArtifactRef | str,
        *,
        from_assay: str | None = None,
        cell_selection: ArtifactRef | None = None,
    ) -> pd.DataFrame:
        """Log counts per million of the summed raw counts of each cell group.

        Summing small groups of similar cells, such as a fine clustering
        (metacells), averages out the sampling noise of single cells. Gene
        programs and region-to-gene links read such profiles.

        Args:
            groups: Clustering artifact or cell metadata column that groups
                the cells, as for ``compare.compute_bulk``.
            from_assay: Assay to read. The default assay when omitted.
            cell_selection: Cells to read, as for ``compare.compute_bulk``.

        Returns:
            A groups-by-features DataFrame of ``log1p`` counts per million,
            indexed by the group labels, as the columns of
            ``compare.compute_bulk`` are. Columns are peak ids
            (``chrom:start-end``) for an ATAC assay and feature names
            otherwise; names that several features share are left out.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a cell that is read.
        """
        from ...assay import ATACassay

        assay = self._get_assay(from_assay)
        bulk = cast(Any, self)._compare_compute_bulk(
            groups,
            from_assay=assay.name,
            cell_selection=cell_selection,
            aggr_type="sum",
            feature_label="id" if isinstance(assay, ATACassay) else "name",
            remove_empty_features=False,
        )
        sums = bulk.profiles
        sums = sums.loc[~sums.index.duplicated(keep=False)].T.astype(np.float64)
        profiles: pd.DataFrame = np.log1p(sums.div(sums.sum(axis=1), axis=0) * 1e6)
        return profiles

    def _regulon_scores(
        self, scores: ArtifactRef, cell_selection: ArtifactRef | None
    ) -> tuple["EnrichmentResult", MemoryLedger]:
        """Load a score artifact for a specificity call, with a ledger that holds its cells and regulons."""
        if not isinstance(scores, ArtifactRef):
            raise TypeError("scores must be an ArtifactRef")
        if scores.kind != "enrichment_scores":
            raise ValueError(
                "scores must identify an enrichment_scores artifact, such as "
                f"scores.aucell returns, not a {scores.kind} artifact"
            )
        if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef or None")
        result: EnrichmentResult = cast(Any, self)._scores_load(scores)
        ledger = MemoryLedger(
            self.memoryBytes,
            f"Scoring the specificity of {len(result.source_names)} regulons",
            codec_threads=self.nthreads,
        )
        ledger.hold(
            "the scored cells' positions and the regulons' names",
            result.cell_index.nbytes
            + result.source_names.nbytes
            + result.source_sizes.nbytes,
        )
        return result, ledger

    @validation_scoped
    def _regulons_specificity(
        self,
        scores: ArtifactRef,
        groups: ArtifactRef | str,
        *,
        cell_selection: ArtifactRef | None = None,
    ) -> pd.DataFrame:
        """Score how specific each regulon's activity is to each group of cells.

        Gives the regulon specificity scores (RSS) of
        ``regulons.regulon_specificity`` from a stored score artifact, such as
        ``scores.aucell`` writes, without reading its scores whole: one pass
        over the scores in blocks of cells sums each regulon's activity, and a
        second sums, for each group, the regulon's share of activity in the
        group's cells and their terms of the Jensen-Shannon divergence. The
        call holds one block of scores and three numbers for each group and
        regulon, beside a position, a label, and a code for each cell. A block
        holds the cells of one stored chunk of scores, or fewer when the
        memory budget leaves less room. Nothing is saved.

        Args:
            scores: An ``enrichment_scores`` artifact of finite, non-negative
                scores, such as AUCell's. chromVAR's z-scores, which can be
                negative, are refused.
            groups: Clustering artifact or user-owned cell metadata column
                that groups the cells, read for the scored cells as
                ``compare.compute_bulk`` reads it. An artifact's own cells must
                include them.
            cell_selection: The cells to score, which ``scores`` must hold.
                Every scored cell when omitted.

        Returns:
            A groups-by-regulons DataFrame of the scores, indexed by the group
            labels in their natural order, named ``group``, with a column for
            each regulon of ``scores``. A regulon without activity in the cells
            scores NaN. Up to rounding, these are the scores that
            ``regulon_specificity`` gives for the same cells' scores and labels.

        Raises:
            TypeError: If ``scores`` or ``cell_selection`` is not an
                ArtifactRef, or ``groups`` is neither an ArtifactRef nor a
                column name.
            ValueError: If ``scores`` is not an ``enrichment_scores``
                artifact, ``cell_selection`` holds a cell that ``scores`` did
                not score, ``groups`` holds no label, or a missing label, for a
                cell to score, or a score is negative or not finite.
            MemoryBudgetError: If the memory budget cannot hold the cells'
                positions, labels, and codes, the sums, and a block of one
                cell's scores beside one stored chunk's decode.
        """
        from ...clustering.connectivity import label_codes
        from ...metadata.selection import (
            CellField,
            resolve_grouping,
            valid_category_mask,
        )
        from ...regulons.specificity import specificity_bytes, specificity_scores
        from ...utils.arrays import sort_categories

        if not isinstance(groups, ArtifactRef | str):
            raise TypeError("groups must be an ArtifactRef or a cell column name")
        result, ledger = self._regulon_scores(scores, cell_selection)
        selection = result.cell_selection if cell_selection is None else cell_selection
        grouping = CellField(groups) if isinstance(groups, str) else groups
        try:
            resolved = resolve_grouping(
                self.zw, self.cells, grouping, cell_selection=selection, ledger=ledger
            )
        except ValueError as error:
            if not isinstance(groups, ArtifactRef) or (
                "must be a subset of the artifact cell selection" not in str(error)
            ):
                raise
            raise ValueError(
                f"The {groups.kind} artifact of groups holds no label for some "
                "cells to score; compute it on cells that include them, or pass "
                "a cell_selection of the cells that it labels"
            ) from error
        labels = np.asarray(resolved.labels)
        n_cells = len(labels)
        # Whether each label is missing, pandas' hash table of the labels,
        # and each label's code.
        ledger.peak(
            f"coding the labels of {n_cells} cells",
            n_cells + hash_table_bytes(n_cells) + _INDEX_BYTES * n_cells,
        )
        missing = ~valid_category_mask(labels, missing_mask=resolved.missing_mask)
        if missing.any():
            raise ValueError(
                f"groups holds no label for {int(missing.sum())} of the "
                f"{n_cells} cells to score; pass a cell_selection of the "
                "labeled cells"
            )
        categories = sort_categories(pd.unique(labels).tolist())
        codes = label_codes(labels, categories)
        ledger.hold(f"the groups of {n_cells} cells", codes.nbytes)
        rows = (
            None
            if cell_selection is None
            else _scored_rows(ledger, result.cell_index, resolved.cell_idx)
        )
        n_regulons = len(result.source_names)
        ledger.hold(
            f"the specificity sums of {n_regulons} regulons in "
            f"{len(categories)} groups",
            specificity_bytes(len(categories), n_regulons),
        )
        specificity = specificity_scores(
            lambda: _score_blocks(
                ledger,
                result,
                rows,
                nthreads=self.nthreads,
                msg=f"({result.assay}) Scoring regulon specificity",
            ),
            codes,
            len(categories),
            n_regulons,
            name=f"The scores of the {result.method} artifact",
        )
        return pd.DataFrame(
            specificity,
            index=pd.Index(categories, name="group"),
            columns=result.source_names.tolist(),
            copy=False,
        )

    @validation_scoped
    def _regulons_connection_specificity(
        self,
        scores: ArtifactRef,
        *,
        margin: float = 0.05,
        cell_selection: ArtifactRef | None = None,
    ) -> pd.DataFrame:
        """Score how specifically each pair of regulons is active in the same cells.

        Gives the connection specificity index (CSI) of
        ``regulons.connection_specificity`` from a stored score artifact, such
        as ``scores.aucell`` writes, without reading its scores whole: one pass
        over the scores in blocks of cells merges each block's sums of
        products of deviations into the regulons' correlation matrix. The call
        holds one block of scores, as float64 too, and the correlations, then
        the index, two numbers for each pair of regulons, beside a position for
        each cell. A block holds the cells of one stored chunk of scores, or
        fewer when the memory budget leaves less room. Nothing is saved.

        Args:
            scores: An ``enrichment_scores`` artifact, such as AUCell's.
            margin: Correlation margin: a regulon counts for a pair whose
                correlation is ``r`` when it correlates below ``r - margin``
                with both.
            cell_selection: The cells to read, which ``scores`` must hold.
                Every scored cell when omitted.

        Returns:
            A symmetric regulons-by-regulons DataFrame, indexed and labeled by
            the regulons of ``scores``. A regulon of constant scores correlates
            0 with every regulon. The indices equal those of
            ``connection_specificity`` for the same cells' scores, except where
            rounding moves a correlation across ``r - margin``.

        Raises:
            TypeError: If ``scores`` or ``cell_selection`` is not an
                ArtifactRef, or ``margin`` is not a real number.
            ValueError: If ``margin`` is not finite, ``scores`` is not an
                ``enrichment_scores`` artifact, or ``cell_selection`` holds a
                cell that ``scores`` did not score.
            MemoryBudgetError: If the memory budget cannot hold the cells'
                positions, the correlations of every pair of regulons twice
                over, and a block of one cell's scores beside one stored
                chunk's decode.
        """
        from ...metadata.selection import (
            plan_cell_selection_read,
            read_cell_selection,
        )
        from ...regulons.specificity import (
            activity_correlation,
            connection_specificity_from_correlation,
            correlation_bytes,
            correlation_bytes_per_row,
        )

        margin = float_argument(margin, "margin")
        result, ledger = self._regulon_scores(scores, cell_selection)
        rows = None
        if cell_selection is not None:
            cells = read_cell_selection(
                ledger,
                self.zw,
                plan_cell_selection_read(self.zw, cell_selection),
                item="the cell positions of cell_selection",
            )
            rows = _scored_rows(ledger, result.cell_index, cells)
            ledger.release(cells.nbytes)
            del cells
        n_regulons = len(result.source_names)
        remedy = "score fewer regulons"
        summing = ledger.hold(
            f"the correlations of {n_regulons} regulons",
            correlation_bytes(n_regulons),
            remedy=remedy,
        )
        correlation = activity_correlation(
            _score_blocks(
                ledger,
                result,
                rows,
                nthreads=self.nthreads,
                msg=f"({result.assay}) Correlating regulon activity",
                scratch_bytes_per_row=correlation_bytes_per_row(n_regulons),
            ),
            n_regulons,
        )
        ledger.release(summing)
        ledger.hold(f"the correlations of {n_regulons} regulons", correlation.nbytes)
        ledger.hold(
            f"the connection specificity index of {n_regulons} regulons",
            correlation.nbytes,
            remedy=remedy,
        )
        names = result.source_names.tolist()
        return pd.DataFrame(
            connection_specificity_from_correlation(correlation, margin),
            index=names,
            columns=names,
            copy=False,
        )
