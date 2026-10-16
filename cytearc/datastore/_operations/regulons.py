import time
from collections.abc import Iterable, Sequence
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
from ...utils.arguments import integer_argument
from ...utils.arrays import array_digest
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _RegulonOperationsBase
else:
    _RegulonOperationsBase = object

_GRN_LAYOUT = "tf_target_edges"
_GRN_VALUES = "raw_counts"
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
        predicts its raw counts across the selected cells from the raw counts
        of the selected transcription factors, as arboreto's GRNBoost2 does
        for pySCENIC: the same regressor settings and early stopping, the same
        seed for every target, a factor never predicting itself, and
        importances scaled by the number of fitted stages. The Pearson
        correlation of each linked factor and target is stored with it.

        The counts of the selected cells and genes are held in memory as
        float32, so select a representative subset of cells for large data.

        Args:
            cell_selection: Cells whose counts are modelled.
            features: Feature selection of the genes modelled as targets; the
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
            nthreads: Targets fitted concurrently. The store's thread count
                when omitted. It does not change the result.
            invalidate_cache: Recompute even when a matching result exists.

        Returns:
            A complete ``grn_adjacencies`` artifact.

        Raises:
            MemoryError: If the counts do not fit the memory budget.
            PermissionError: If the store is not opened with ``zarr_mode='r+'``.
            UnmeasuredCellsError: If the assay did not measure a selected cell.
        """
        from ...regulons.grn import GrnParameters, _grn_links, _pair_correlations

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
            values=_GRN_VALUES,
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
        # The matrix, the regulator copy, one leave-one-out copy per thread,
        # the correlation blocks, and about 64 bytes per possible link while
        # the links are sorted, correlated, and written.
        needed = 4 * n_cells * (n_genes + n_regulators * (1 + threads))
        needed += 8 * n_cells * 2 * max(n_regulators, 256)
        needed += 64 * n_genes * n_regulators
        if needed > self.resources.memoryBytes:
            raise MemoryError(
                f"GRN inference over {n_cells} cells and {n_genes} genes needs "
                f"about {needed} bytes, but the memory budget is "
                f"{self.resources.memoryBytes} bytes; select fewer cells or genes"
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
            nthreads=threads,
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
            A groups-by-features DataFrame of ``log1p`` counts per million.
            Columns are peak ids (``chrom:start-end``) for an ATAC assay and
            feature names otherwise; names that several features share are
            left out.

        Raises:
            UnmeasuredCellsError: If the assay did not measure a cell that is read.
        """
        from ...assay import ATACassay

        assay = self._get_assay(from_assay)
        sums = cast(Any, self)._compare_compute_bulk(
            groups,
            from_assay=assay.name,
            cell_selection=cell_selection,
            aggr_type="sum",
            feature_label="id" if isinstance(assay, ATACassay) else "name",
            remove_empty_features=False,
        )
        sums = sums.loc[~sums.index.duplicated(keep=False)].T.astype(np.float64)
        profiles: pd.DataFrame = np.log1p(sums.div(sums.sum(axis=1), axis=0) * 1e6)
        return profiles
