import hashlib
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr

from ...metadata.arguments import GuideCallImportArguments
from ...metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from ...perturbation.guide_calls import (
    GuideCalls,
    guide_call_arrays,
    guide_calls_frames,
    summarize_guide_counts,
    validated_guide_ids,
)
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_strings,
    require_complete_artifact,
)
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...storage.validation_scope import validation_scoped
from ...utils.logging import logger

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _PerturbationOperationsBase
else:
    _PerturbationOperationsBase = object

_GUIDE_ASSAY = "CRISPR"
_GUIDE_CALL_ARRAYS = (
    "values",
    "n_guides",
    "top_guide",
    "top_umi",
    "top_fraction",
    "total_umi",
    "guide_codes",
    "guide_umis",
)
# Barcodes of the call file that are not selected cells, named in the
# artifact's attributes and in the warning.
_UNMATCHED_EXAMPLES = 5
# Bytes held for each selected cell beside the streamed counts: its ID and
# position, its label, and its summary and call arrays.
_BYTES_PER_CELL = 160
# Bytes held for each call of the file: its row, guide, UMIs, and order.
_BYTES_PER_CALL = 48


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _PerturbationOperationsMixin(_PerturbationOperationsBase):
    def _guide_assay(self, from_assay: str | None) -> Any:
        """Return the guide assay after checking its declared type and counts."""
        from ...assay.classification import declared_assay_type

        name = _GUIDE_ASSAY if from_assay is None else from_assay
        if not isinstance(name, str):
            raise TypeError("from_assay must be an assay name or None")
        declared = (
            declared_assay_type(self._get_assay(name))
            if name in self.assay_names
            else None
        )
        if declared != _GUIDE_ASSAY:
            raise TypeError(
                "Guide calls need an assay declared with type 'CRISPR'; "
                f"{name!r} is declared as {declared!r}"
            )
        assay = self._get_assay(name)
        count_dtype = np.dtype(assay.rawData.dtype)
        if count_dtype.kind != "u":
            raise TypeError(
                "Guide calls need non-negative integer counts, but assay "
                f"{name!r} stores {count_dtype} counts"
            )
        return assay

    @validation_scoped
    def _perturbation_import_guide_calls(
        self,
        path: str | os.PathLike[str],
        *,
        cell_selection: ArtifactRef,
        from_assay: str | None = None,
        barcode_prefix: str | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Import Cell Ranger's guide calls for a set of cells.

        Cell Ranger's CRISPR Guide Capture analysis calls the guides of each
        cell and writes them to ``crispr_analysis/protospacer_calls_per_cell.csv``.
        The import joins each row's barcode to a selected cell and saves the
        calls as a ``guide_calls`` artifact, which CyteArc's own guide calling
        writes too. A selected cell absent from the file has no call, as Cell
        Ranger leaves out cells without a confident call. The guide counts of
        the assay give each cell's top guide, its UMIs, and the cell's total
        guide UMIs, read in row blocks within the memory budget.

        A cell's ID is ``barcode_prefix`` followed by the barcode, matched
        exactly. A merged store prefixes each sample's barcodes with its name
        and two underscores, so import each sample's file with its own prefix
        and a cell selection of that sample. Barcodes that match no selected
        cell are counted, saved in the artifact's ``n_unmatched_barcodes`` and
        ``unmatched_barcode_examples`` attributes, and named in a warning.

        The artifact's ``values`` hold the called guide's ID for a cell with
        one call, ``"Negative"`` for a cell with none, and ``"Multiple"`` for a
        cell with several. Its other per-cell arrays are ``n_guides``,
        ``top_guide`` (the guide with the most UMIs, an empty string for a cell
        with no guide UMIs), ``top_umi``, ``top_fraction`` (the top guide's
        share of the cell's guide UMIs, NaN without UMIs) and ``total_umi``,
        and the two-dimensional ``guide_codes`` and ``guide_umis``, which hold
        each cell's called guides, most UMIs first, as positions among the
        assay's guides, padded with -1 and 0. ``guides`` holds the guide IDs
        in assay order. ``select_cells`` reads the one-dimensional arrays, and
        ``artifacts.load_values`` reads every per-cell array.

        Args:
            path: Cell Ranger's ``protospacer_calls_per_cell.csv``, plain or
                gzip-compressed.
            cell_selection: The cells that the calls label.
            from_assay: The guide assay, declared with type ``CRISPR``
                (default: ``'CRISPR'``).
            barcode_prefix: Text that precedes each barcode in the cell IDs,
                such as ``"lane1__"`` in a merged store. None adds nothing.
            invalidate_cache: Import again instead of reusing saved calls.

        Returns:
            A complete assay-scoped ``guide_calls`` artifact. Read it with
            :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.load_guide_calls`
            or plot it with ``ds.plots.guide_calls``.

        Raises:
            TypeError: If an argument has the wrong type, or the assay is not
                declared ``CRISPR`` or does not store unsigned integer counts.
            ValueError: If the call file is malformed, a guide ID is empty,
                repeated, or a label reserved for cells without one guide, or a
                called guide is not a feature of the assay.
            UnmeasuredCellsError: If the guide assay did not measure a selected
                cell.
            PermissionError: If the store is read-only and the calls are not
                saved yet.
            MemoryError: If the calls and the cell summaries do not fit the
                memory budget.
        """
        from ...readers.cellranger import read_protospacer_calls

        if not isinstance(path, str | os.PathLike):
            raise TypeError("path must be a path to Cell Ranger's call file")
        if not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        if barcode_prefix is not None and not isinstance(barcode_prefix, str):
            raise TypeError("barcode_prefix must be text or None")
        if not isinstance(invalidate_cache, bool):
            raise TypeError("invalidate_cache must be a boolean")
        assay = self._guide_assay(from_assay)
        guide_ids = validated_guide_ids(assay.feats.fetch_all("ids"))
        cell_index = read_stored_selection_indices(
            self.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        n_cells = len(cell_index)
        if not n_cells:
            raise ValueError("cell_selection selects no cell")
        self._require_measured_cells(
            assay.name,
            cell_selection,
            operation=GuideCallImportArguments.operation,
        )
        arguments = GuideCallImportArguments(
            cell_selection=cell_selection,
            dataset_fingerprint=self._ensure_dataset_fingerprint(assay.name),
            guide_ids_fingerprint=fingerprint_strings(guide_ids.astype(str)),
            file_sha256=_file_sha256(path),
            barcode_prefix=barcode_prefix,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        n_guide_ids = len(guide_ids)

        def has_guides(_ref: ArtifactRef, group: zarr.Group) -> bool:
            guides = group.get("guides")
            return isinstance(guides, zarr.Array) and guides.shape == (n_guide_ids,)

        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={
                "values": ((n_cells,), None),
                "n_guides": ((n_cells,), "i"),
                "top_guide": ((n_cells,), None),
                "top_umi": ((n_cells,), "u"),
                "top_fraction": ((n_cells,), "f"),
                "total_umi": ((n_cells,), "u"),
                "guide_codes": ((n_cells, None), "i"),
                "guide_umis": ((n_cells, None), "u"),
            },
            invalidate_cache=invalidate_cache,
            required_attributes=("n_file_barcodes", "n_unmatched_barcodes"),
            reuse_validator=has_guides,
        )
        if planned.reused:
            return planned.ref
        self._require_writable(arguments.operation)

        file_calls = read_protospacer_calls(path)
        n_calls = len(file_calls.features)
        resident = _BYTES_PER_CELL * n_cells + _BYTES_PER_CALL * n_calls
        if resident > self.memoryBytes:
            raise MemoryError(
                f"Importing {n_calls} guide calls for {n_cells} cells holds about "
                f"{resident} bytes, more than the memory budget of "
                f"{self.memoryBytes} bytes. Raise mem_budget, or pass a smaller "
                "cell_selection."
            )
        guide_positions = pd.Index(guide_ids.astype(str)).get_indexer(
            file_calls.features.astype(str)
        )
        unknown = np.flatnonzero(guide_positions < 0)
        if len(unknown):
            names = set(np.asarray(assay.feats.fetch_all("names")).astype(str))
            examples = pd.unique(file_calls.features[unknown])[:5].tolist()
            hint = (
                "; they match guide names, so the file names guides by name "
                "while the assay's IDs differ"
                if all(example in names for example in examples)
                else ""
            )
            raise ValueError(
                f"{len(pd.unique(file_calls.features[unknown]))} called guides are "
                f"not features of assay {assay.name!r}, for example "
                + ", ".join(repr(example) for example in examples)
                + hint
            )
        cell_ids = np.asarray(self.cells.fetch_all("ids"))[cell_index].astype(str)
        prefix = "" if barcode_prefix is None else barcode_prefix
        row_cells = pd.Index(cell_ids).get_indexer(
            np.char.add(prefix, file_calls.barcodes.astype(str))
        )
        unmatched = np.flatnonzero(row_cells < 0)
        examples = file_calls.barcodes[unmatched[:_UNMATCHED_EXAMPLES]].tolist()
        matched_calls = row_cells[file_calls.rows] >= 0
        arrays = guide_call_arrays(
            row_cells[file_calls.rows][matched_calls],
            guide_positions[matched_calls],
            file_calls.umis[matched_calls],
            n_cells=n_cells,
            guide_ids=guide_ids,
        )
        blocks = assay.rawData[cell_index, :]._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({assay.name}) Summarising guide counts",
            prefetch=None,
            row_mask=None,
            resident_bytes=resident + arrays.guide_codes.nbytes * 2,
            memory_bytes=self.memoryBytes,
        )
        summary = summarize_guide_counts(blocks, n_cells, n_guide_ids)
        top_guide = np.full(n_cells, "", dtype=object)
        counted = summary.top_code >= 0
        top_guide[counted] = guide_ids[summary.top_code[counted]]
        write_cell_data_artifact(
            self.zw,
            planned,
            {
                "values": arrays.values,
                "n_guides": arrays.n_guides,
                "top_guide": top_guide,
                "top_umi": summary.top_umi,
                "top_fraction": summary.top_fraction,
                "total_umi": summary.total_umi,
                "guide_codes": arrays.guide_codes,
                "guide_umis": arrays.guide_umis,
                "guides": guide_ids,
            },
            attributes={
                "n_file_barcodes": int(len(file_calls.barcodes)),
                "n_unmatched_barcodes": int(len(unmatched)),
                "unmatched_barcode_examples": [str(value) for value in examples],
            },
        )
        if len(unmatched):
            logger.warning(
                f"{len(unmatched)} of {len(file_calls.barcodes)} barcodes of the call "
                "file match no selected cell, for example "
                + ", ".join(repr(str(value)) for value in examples)
                + ". Check barcode_prefix and cell_selection."
            )
        logger.info(
            f"Imported guide calls of {int(np.count_nonzero(arrays.n_guides))} of "
            f"{n_cells} cells"
        )
        return planned.ref

    @validation_scoped
    def _perturbation_load_guide_calls(self, calls: ArtifactRef) -> GuideCalls:
        """Read saved guide calls as a table of cells and a table of calls.

        Args:
            calls: A ``guide_calls`` artifact, such as one from
                :meth:`~cytearc.datastore.namespaces.PerturbationAccessor.import_guide_calls`.

        Returns:
            The calls, with one row per cell and one row per called guide of a
            cell, and the guide IDs of the assay.

        Raises:
            TypeError: If ``calls`` is not an ArtifactRef.
            ValueError: If it is not a ``guide_calls`` artifact.
            MemoryError: If its values do not fit the datastore memory budget.
        """
        if not isinstance(calls, ArtifactRef):
            raise TypeError("calls must be an ArtifactRef")
        if calls.kind != GuideCallImportArguments.artifact_kind:
            raise ValueError(
                f"calls must be a guide_calls artifact, not {calls.kind!r}"
            )
        status = require_complete_artifact(self.zw, calls)
        group = artifact_group(self.zw, calls)
        guide_ids = np.asarray(as_zarr_array(group["guides"], name="guides")[:]).astype(
            str
        )
        # The canonical labels are read by default; the others by name.
        loaded = {
            name: self._artifacts_load_values(
                calls, value=None if name == "values" else name
            )
            for name in _GUIDE_CALL_ARRAYS
        }
        cells, call_table = guide_calls_frames(
            loaded["values"].cell_ids,
            **{name: np.asarray(values.values) for name, values in loaded.items()},
            guide_ids=guide_ids,
        )
        return GuideCalls(
            cells=cells,
            calls=call_table,
            guides=pd.Index(guide_ids, name="guide"),
            source=str(status.operation),
        )
