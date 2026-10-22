"""Call cells in unfiltered droplet matrices with EmptyDrops."""

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from ...metadata.arguments import CellCallingArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...metadata.membership import measured_selection
from ...quality_control.cell_calling import (
    DEFAULT_LOWER,
    DEFAULT_N_ITER,
    BarcodeRanks,
    EmptyDropsResult,
    EmptyDropsSettings,
    barcode_ranks,
    run_empty_drops,
    validated_fdr,
    validated_settings,
)
from ...storage.artifacts import artifact_group, inspect_artifact
from ...storage.refs import ArtifactRef
from ...storage.selections import (
    resolve_generated_selection_artifact,
    selection_mask,
    validate_cell_selection,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.validation_scope import validation_scoped
from ...utils.arguments import integer_argument
from ...utils.logging import logger

if TYPE_CHECKING:
    from ...assay import Assay
    from .quality_control import (
        _QualityControlOperationsMixin as _CellCallingOperationsBase,
    )
else:
    _CellCallingOperationsBase = object

# The per-barcode arrays of a cell_calling artifact, and the results that its
# attributes hold beside the ambient profile.
_BARCODE_ARRAYS = ("total", "log_probability", "p_value", "fdr", "limited")
_RESULT_ATTRIBUTES = ("retain", "knee", "inflection", "alpha", "n_ambient")
_DEFAULT_FDR = 0.001


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


class _CellCallingOperationsMixin(_CellCallingOperationsBase):
    def _all_barcodes(self, assay_name: str) -> ArtifactRef:
        """Return the selection of every barcode that the assay measured.

        The live ``I`` column is not read, so the barcodes that a writable
        open removes from it with ``min_features_per_cell`` stay in the
        ambient pool.
        """
        cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        everything, _ = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=np.ones(self.cells.N, dtype=bool),
            row_ids=as_zarr_array(cell_data["ids"], name="ids"),
            operation="select_all_cells",
            parameters={},
            inputs={},
            source_column="artifact",
        )
        return self.select_measured_cells(assay_name, cell_selection=everything)

    def _empty_drops(
        self,
        assay: "Assay",
        rows: np.ndarray,
        settings: EmptyDropsSettings,
    ) -> EmptyDropsResult:
        """Run EmptyDrops over the barcodes at ``rows`` in bounded row blocks.

        Totals come from the prepared ``<assay>_nCounts`` column, so the
        barcode-rank curve needs no pass over the counts. The counts are read
        by row block beside the totals, the ambient histogram, the
        per-barcode results, and the simulations' scratch. The rows of the
        barcodes above ``lower`` are kept from that read in CSR form when,
        sized from the prepared ``<assay>_nFeatures`` column, they fit in half
        of the budget with the rest; otherwise they are read again.

        Raises:
            MemoryError: If what the computation holds exceeds the budget.
        """
        n_features = assay.feats.N
        resident = (
            8 * self.cells.N
            + 64 * len(rows)
            + 8 * n_features * (settings.lower + 3)
            + 8 * self.nthreads * (n_features + len(rows))
        )
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Cell calling over {len(rows)} barcodes and {n_features} features "
                f"holds about {resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass a smaller "
                "cell_selection or lower."
            )
        totals = np.asarray(self.cells.fetch_all(assay._totals_name))[rows]
        tested = totals > settings.lower
        detected = np.asarray(self.cells.fetch_all(f"{assay.name}_nFeatures"))[rows]
        # A kept row holds a 4-byte feature index and a count of at most 8
        # bytes per detected feature, and an 8-byte row pointer.
        kept = int(12 * detected[tested].sum()) + 8 * int(np.count_nonzero(tested))
        keep_tested = resident + kept <= self.memoryBytes // 2
        if keep_tested:
            resident += kept
        counts = assay.rawData

        def read_rows(positions: np.ndarray) -> Iterator[np.ndarray]:
            yield from counts[rows[positions], :]._stream_blocks(
                nthreads=self.nthreads,
                msg=f"({assay.name}) Calling cells",
                prefetch=None,
                row_mask=None,
                resident_bytes=resident,
                memory_bytes=self.memoryBytes,
            )

        return run_empty_drops(
            totals.astype(np.int64),
            read_rows,
            n_features,
            settings,
            nthreads=self.nthreads,
            keep_tested=keep_tested,
        )

    @validation_scoped
    def _qc_call_cells(
        self,
        from_assay: str | None = None,
        *,
        cell_selection: ArtifactRef | None = None,
        lower: int = DEFAULT_LOWER,
        retain: int | None = None,
        n_iter: int = DEFAULT_N_ITER,
        alpha: float | None = None,
        fdr: float = _DEFAULT_FDR,
        random_seed: int = 0,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Call cells in an unfiltered droplet matrix with EmptyDrops.

        EmptyDrops (Lun et al. 2019, Genome Biology 20:63) tells cells from
        empty droplets. Barcodes with at most ``lower`` counts form the ambient
        pool, whose profile is smoothed with Good-Turing estimation. Every
        other barcode is tested against that profile under a
        Dirichlet-multinomial model whose precision is estimated from the pool,
        with Monte Carlo p-values. Barcodes with at least ``retain`` counts
        are cells, and the barcodes whose Benjamini-Hochberg adjusted p-value
        is at most ``fdr`` are the called cells.

        The per-barcode statistics are saved as an assay-scoped
        ``cell_calling`` artifact, which ``qc.load_cell_calls`` reads, and the
        called cells as a ``cell_selection`` that records it as its input
        ``cell_calling``. Another ``fdr`` reuses the statistics. The counts
        are read in row blocks within the memory budget; totals come from the
        prepared ``<assay>_nCounts`` column.

        Import every barcode of the droplet run, such as Cell Ranger's raw
        matrix with ``CrH5Reader(path, is_filtered=False, filtering_cutoff=0)``:
        the ambient pool is made of the barcodes with low totals. Cell
        calling reads every barcode of the assay, never the live ``I``
        column, so barcodes that a writable open removes from ``I`` with
        ``min_features_per_cell`` stay in the pool; open raw stores with
        ``min_features_per_cell=-1`` to keep ``I`` complete as well.

        With the same ``alpha`` and ``retain``, the calls agree with
        DropletUtils' ``emptyDrops``. Two defaults differ from DropletUtils
        1.30: ``alpha`` is estimated by maximum likelihood, which is
        ``emptyDrops(alpha = NULL)``, where DropletUtils defaults to the
        multinomial ``alpha = Inf``; and the default ``retain`` is the knee
        of a smoothing spline's curvature, where DropletUtils now traces the
        curve, so the two knees differ.

        Args:
            from_assay: Assay of the counts (default: the default assay).
            cell_selection: Barcodes of one droplet run to call. None uses
                every barcode that the assay measured. Run each sample of a
                merged store separately.
            lower: Barcodes with at most this many counts form the ambient
                pool and are not tested (default 100).
            retain: Barcodes with at least this many counts are cells with
                p-value 0. None uses the total at the knee of the barcode-rank
                curve (``qc.barcode_ranks``).
            n_iter: Monte Carlo iterations (default 10,000). The smallest
                p-value is ``1 / (n_iter + 1)``; ``qc.load_cell_calls`` marks
                barcodes that more iterations could make significant as
                ``limited``.
            alpha: Dirichlet-multinomial precision of the ambient model. None
                estimates it by maximum likelihood over the ambient pool.
            fdr: Largest adjusted p-value of a called cell (default 0.001).
            random_seed: Seed of the simulations. A seed gives the same
                p-values on any number of threads.
            invalidate_cache: Compute again instead of reusing saved results.

        Returns:
            A datastore-scoped ``cell_selection`` of the called cells.

        Raises:
            TypeError: If an argument has the wrong type, or the assay stores
                counts that are not non-negative integers.
            ValueError: If an argument is out of range, ``cell_selection``
                selects no barcode, no barcode exceeds ``lower``, there is no
                ambient pool, the knee cannot be located for a default
                ``retain``, or no barcode is called.
            UnmeasuredCellsError: If the assay did not measure a barcode of
                ``cell_selection``.
            PermissionError: If the store is read-only and the results are
                not saved yet.
            MemoryError: If the computation does not fit the memory budget.
        """
        settings = validated_settings(
            lower=lower,
            retain=retain,
            n_iter=n_iter,
            alpha=alpha,
            random_seed=random_seed,
        )
        threshold = validated_fdr(fdr)
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        count_dtype = np.dtype(assay.rawData.dtype)
        if count_dtype.kind != "u":
            raise TypeError(
                "Cell calling needs non-negative integer counts, but assay "
                f"{assay.name!r} stores {count_dtype} counts"
            )
        if cell_selection is None:
            barcodes_ref = self._all_barcodes(assay.name)
        else:
            self._require_measured_cells(
                assay.name,
                cell_selection,
                operation=CellCallingArguments.operation,
                remedy="cell_selection",
            )
            barcodes_ref = cell_selection
        barcodes = validate_cell_selection(self.zw, barcodes_ref)
        rows = np.flatnonzero(selection_mask(barcodes)).astype(np.int64)
        if not len(rows):
            raise ValueError("cell_selection selects no barcode")
        n_barcodes = len(rows)
        n_features = assay.feats.N
        arguments = CellCallingArguments(
            cell_selection=barcodes_ref,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            lower=settings.lower,
            retain=settings.retain,
            n_iter=settings.n_iter,
            alpha=settings.alpha,
            random_seed=settings.random_seed,
            nthreads=self.nthreads,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()

        def has_ambient(_ref: ArtifactRef, group: zarr.Group) -> bool:
            ambient = group.get("ambient")
            return isinstance(ambient, zarr.Array) and ambient.shape == (n_features,)

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=barcodes_ref,
            arrays={
                "total": ((n_barcodes,), "i"),
                "log_probability": ((n_barcodes,), "f"),
                "p_value": ((n_barcodes,), "f"),
                "fdr": ((n_barcodes,), "f"),
                "limited": ((n_barcodes,), "b"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=_RESULT_ATTRIBUTES,
            reuse_validator=has_ambient,
        )
        if planned.reused:
            group = artifact_group(self.zw, planned.ref)
            fdr_values = np.asarray(as_zarr_array(group["fdr"], name="fdr")[:])
            limited = np.asarray(as_zarr_array(group["limited"], name="limited")[:])
        else:
            self._require_writable(arguments.operation)
            result = self._empty_drops(assay, rows, settings)
            write_cell_data_artifact(
                self.zw,
                planned,
                {
                    "total": result.total,
                    "log_probability": result.log_probability,
                    "p_value": result.p_value,
                    "fdr": result.fdr,
                    "limited": result.limited,
                    "ambient": result.ambient,
                },
                attributes={
                    "retain": result.retain,
                    "knee": _finite_or_none(result.knee),
                    "inflection": _finite_or_none(result.inflection),
                    "alpha": result.alpha,
                    "n_ambient": result.n_ambient,
                },
            )
            fdr_values = result.fdr
            limited = result.limited
        called = fdr_values <= threshold
        undecided = int(np.count_nonzero(limited & ~called))
        if undecided:
            logger.warning(
                f"{undecided} barcodes have the smallest p-value that "
                f"n_iter={settings.n_iter} iterations give, but an adjusted p-value "
                f"above fdr={threshold}; call cells again with more iterations to "
                "decide them."
            )
        n_called = int(np.count_nonzero(called))
        if not n_called:
            raise ValueError(
                f"No barcode has an adjusted p-value of at most fdr={threshold}, so "
                "no cell was called. The statistics of every barcode are saved: "
                "load them with qc.load_cell_calls on the cell_calling artifact "
                f"that artifacts.list(kind='cell_calling', from_assay={assay.name!r}) "
                "lists."
            )
        values = np.zeros(self.cells.N, dtype=bool)
        values[rows[called]] = True
        ref, _ = resolve_generated_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=values,
            row_ids=barcodes.row_ids,
            row_ids_fingerprint=barcodes.row_ids_fingerprint,
            operation="call_cells",
            parameters={"fdr": threshold},
            inputs={"cell_calling": planned.ref, "prior_cell_selection": barcodes_ref},
            source_column="artifact",
            invalidate_cache=invalidate_cache,
        )
        logger.info(f"Cell calling called {n_called} of {n_barcodes} barcodes cells")
        return ref

    @validation_scoped
    def _qc_load_cell_calls(self, calls: ArtifactRef) -> EmptyDropsResult:
        """Load the EmptyDrops statistics of every barcode of a cell calling.

        Args:
            calls: The cell selection that ``qc.call_cells`` returned, or its
                ``cell_calling`` artifact.

        Returns:
            The statistics of every barcode, with the barcode ids.

        Raises:
            TypeError: If ``calls`` is not an ArtifactRef.
            ValueError: If ``calls`` is neither a ``cell_calling`` artifact
                nor a selection of called cells, or is incomplete.
        """
        if not isinstance(calls, ArtifactRef):
            raise TypeError("calls must be an ArtifactRef")
        statistics = calls
        if calls.kind == "cell_selection":
            status = inspect_artifact(self.zw, calls)
            if status.operation != "call_cells":
                raise ValueError(
                    "calls must be a cell selection that qc.call_cells returned"
                )
            statistics = status.input_ref("cell_calling")
        elif calls.kind != CellCallingArguments.artifact_kind:
            raise ValueError(
                "calls must be a cell_calling artifact or a cell selection that "
                f"qc.call_cells returned, not a {calls.kind} artifact"
            )
        status = inspect_artifact(self.zw, statistics)
        if not status.complete:
            raise ValueError("The cell_calling artifact is unavailable or incomplete")
        parameters = status.parameters or {}
        barcodes = validate_cell_selection(self.zw, status.input_ref("cell_selection"))
        rows = np.flatnonzero(selection_mask(barcodes))
        group = artifact_group(self.zw, statistics)
        arrays: dict[str, Any] = {
            name: np.asarray(as_zarr_array(group[name], name=name)[:])
            for name in (*_BARCODE_ARRAYS, "ambient")
        }
        attributes: dict[str, Any] = {
            name: group.attrs[name] for name in _RESULT_ATTRIBUTES
        }
        return EmptyDropsResult(
            barcodes=np.asarray(self.cells.fetch_all("ids"))[rows],
            total=arrays["total"],
            log_probability=arrays["log_probability"],
            p_value=arrays["p_value"],
            fdr=arrays["fdr"],
            limited=arrays["limited"],
            ambient=arrays["ambient"],
            lower=int(parameters["lower"]),
            retain=int(attributes["retain"]),
            knee=math.nan if attributes["knee"] is None else attributes["knee"],
            inflection=(
                math.nan
                if attributes["inflection"] is None
                else attributes["inflection"]
            ),
            alpha=attributes["alpha"],
            n_iter=int(parameters["n_iter"]),
            random_seed=int(parameters["random_seed"]),
            n_ambient=int(attributes["n_ambient"]),
        )

    @validation_scoped
    def _qc_barcode_ranks(
        self,
        from_assay: str | None = None,
        *,
        cell_selection: ArtifactRef | None = None,
        lower: int = DEFAULT_LOWER,
    ) -> BarcodeRanks:
        """Locate the knee and inflection of the barcode-rank curve.

        The curve ranks barcodes by their prepared ``<assay>_nCounts`` total,
        so it reads no counts and writes nothing. ``qc.call_cells`` retains
        the barcodes with at least the knee's total by default.

        Args:
            from_assay: Assay of the totals (default: the default assay).
            cell_selection: Barcodes to rank. None ranks every barcode that
                the assay measured, whatever the live ``I`` column holds.
            lower: Totals at or below this value are left out of the fit.

        Returns:
            The curve with its knee and inflection totals, which are NaN when
            fewer than five distinct totals exceed ``lower``.

        Raises:
            TypeError: If ``cell_selection`` is not an ArtifactRef or
                ``lower`` is not an integer.
            ValueError: If ``lower`` is negative.
            UnmeasuredCellsError: If the assay did not measure a barcode of
                ``cell_selection``.
        """
        lower = integer_argument(lower, "lower", minimum=0)
        if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        assay = self._get_assay(from_assay)
        if cell_selection is None:
            mask = np.ones(self.cells.N, dtype=bool)
            measured = measured_selection(self.cells, assay.name, mask)
            if measured is not None:
                mask = measured[1]
        else:
            self._require_measured_cells(
                assay.name,
                cell_selection,
                operation="barcode_ranks",
                remedy="cell_selection",
            )
            mask = selection_mask(validate_cell_selection(self.zw, cell_selection))
        totals = np.asarray(self.cells.fetch_all(assay._totals_name))[mask]
        return barcode_ranks(totals, lower=lower)
