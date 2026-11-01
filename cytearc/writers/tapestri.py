"""Write a store from a Tapestri file of single-cell DNA and protein."""

from typing import Any

from ..readers.tapestri import AMPLICON_FEATURE_TYPE, TapestriReader
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.types import as_zarr_group
from .cellranger import CrToZarr

__all__ = ["TapestriToZarr"]


class TapestriToZarr(CrToZarr):
    """Convert a Tapestri file of single-cell DNA and protein to a store.

    Tapestri files hold no RNA: the store holds the amplicons' reads as the
    ``CNV`` assay, recorded as a generic ``Assay``, and the antibody tags'
    reads as the ``ADT`` assay, with each cell's ``sample_name`` and
    ``filtered`` flag as cell metadata and the amplicons' intervals as
    feature metadata, as :class:`~cytearc.readers.TapestriReader` reads
    them. The file's metadata, such as each assay's ``panel_name``,
    ``genome_version`` and ``pipeline_version``, is recorded in the store's
    ``tapestri`` attribute, of the workspace when one is given. A store of
    two assays opens with ``DataStore(path, default_assay="ADT")`` or
    ``"CNV"``. The DNA variants are imported into the store with
    ``ds.variants.import_counts(path, format="tapestri")``, whose cells are
    named as the store names them.

    Args:
        reader: A :class:`~cytearc.readers.TapestriReader` of the file.
        zarr_loc: The file name for the Zarr hierarchy or a store.
        workspace: Workspace name in the destination store. None uses the
            layout without a workspace group.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
            suffixed size (e.g. '8G'), or a fraction of total system memory
            (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None,
            auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
            None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly. When None, the
            default policy is used.
        io: Optional explicit read, compute, and write widths.
        assay_types: Preset assay types, such as ``{"CNV": "CUSTOM"}``, that
            replace the defaults: ``"Assay"`` for the amplicons' assay, and
            ``ADT``'s own type for the proteins.
        overwrite: If True, replace a CyteArc store that no ``DataStore``
            has opened.

    Raises:
        TypeError: If ``reader`` is not a ``TapestriReader``.
        ValueError: If ``assay_types`` names an assay that is not imported
            or a type that is not a preset.
        FileExistsError: If ``zarr_loc`` is not empty, unless ``overwrite``
            replaces it.
        MemoryError: If the import does not fit ``mem_budget``, before the
            destination is created.
    """

    def __init__(
        self,
        reader: TapestriReader,
        zarr_loc: ZarrLocation,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
        assay_types: dict[str, str] | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(reader, TapestriReader):
            raise TypeError(
                f"reader must be a TapestriReader, not {type(reader).__name__}"
            )
        # The amplicons' assay takes the generic type unless renamed to a
        # preset or given another type.
        amplicons = [
            str(name)
            for name, feature_type in reader.assayFeats.T["type"].items()
            if feature_type == AMPLICON_FEATURE_TYPE
        ]
        types = {name: "Assay" for name in amplicons}
        types.update(assay_types or {})
        super().__init__(
            reader,
            zarr_loc,
            workspace=workspace,
            storage_options=storage_options,
            mem_budget=mem_budget,
            nthreads=nthreads,
            profile=profile,
            policy=policy,
            io=io,
            assay_types=types,
            overwrite=overwrite,
        )
        root = (
            self.z
            if workspace is None
            else as_zarr_group(self.z[workspace], name=workspace)
        )
        root.attrs["tapestri"] = {
            "samples": list(reader.samples),
            **reader.metadata,
        }
