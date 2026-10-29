from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..storage.artifacts import ArtifactRef
from ..storage.pipeline_runs import abandon_pipeline_label_claim
from ..storage.validation_scope import validation_scoped
from ..utils.shutdown import CancellationToken
from ._pipeline.events import (
    PipelineCallback as PipelineCallback,
    PipelineEvent as PipelineEvent,
    PipelineEventKind as PipelineEventKind,
    PipelineProgress as PipelineProgress,
    pipeline_listener as pipeline_listener,
)
from ._pipeline.executor import execute_recipe
from ._pipeline.recipes.adt import DEFAULT_PANEL_EXCLUDE as DEFAULT_PANEL_EXCLUDE
from ._pipeline.recipes.adt import resolve_adt
from ._pipeline.recipes.atac import resolve_atac
from ._pipeline.recipes.multimodal import resolve_multimodal
from ._pipeline.recipes.perturbation import resolve_perturbation
from ._pipeline.recipes.rna import resolve_rna
from .pipeline_run import (
    PipelineRun,
    list_pipeline_runs,
    open_pipeline_run,
)

# Keep the documented event types' import and pickle identity here, where
# users import them, though they are implemented beside the ledger.
PipelineEvent.__module__ = __name__
PipelineProgress.__module__ = __name__
pipeline_listener.__module__ = __name__

type CellsArgument = str | ArtifactRef | PipelineRun
"""A boolean cell column, a cell-selection artifact, or a completed run."""

type StageSwitch = bool | Mapping[str, object]
"""An optional stage: False skips it, True runs it, a mapping configures it."""

type StageSettings = Mapping[str, object] | None
"""A stage that always runs: None for its defaults, or a mapping of settings."""


class PipelineAccessor:
    """Store-bound entry point for CyteArc's durable modality recipes.

    Each recipe checks its call against the store before it writes anything:
    its keywords and settings, the columns and cells it names, its stage
    combinations, and the setting values that can be checked without reading
    data. A call that fails these checks raises and writes nothing. The
    recipe then runs its stages in order and returns a durable
    :class:`~cytearc.PipelineRun`. A few values, such as ``ann_index``
    ``ann_metric`` or a Leiden ``backend``, are checked only by their stage's
    method when the stage runs, and such a failure records a failed run.

    A stage keyword takes the keywords of the stage's method, which mean
    exactly what they mean there, and an omitted setting takes that method's
    default, except for a few defaults that a recipe fits to its data, such
    as an ADT run's PCA ``dims``.
    """

    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        self._store = store

    def open(
        self,
        *,
        run_id: str | None = None,
        label: str | None = None,
    ) -> PipelineRun:
        """Open one durable run by ID or immutable label."""
        if (run_id is None) == (label is None):
            raise ValueError("Provide exactly one of run_id or label")
        return open_pipeline_run(self._store, run_id=run_id, label=label)

    def list_runs(
        self,
        *,
        status: str | Sequence[str] | None = None,
        limit: int = 20,
    ) -> tuple[PipelineRun, ...]:
        """List recent runs, newest first, optionally filtered by status."""
        return list_pipeline_runs(self._store, status=status, limit=limit)

    def abandon_label_claim(
        self,
        *,
        label: str,
        run_id: str,
        reason: str,
    ) -> PipelineRun:
        """Mark an exact stopped finalizer interrupted so its label can be retried.

        The caller must first confirm that the process executing ``run_id`` has
        stopped. CyteArc does not infer abandonment from elapsed time because a
        slow live finalizer must never lose its label claim.
        """

        if self._store.zarr_mode != "r+":
            raise PermissionError("Label-claim recovery requires zarr_mode='r+'")
        record = abandon_pipeline_label_claim(
            self._store.zw,
            label=label,
            run_id=run_id,
            reason=reason,
        )
        return PipelineRun(self._store, record)

    def _require_runnable(self, callback: PipelineCallback | None) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        if self._store.zarr_mode != "r+":
            raise PermissionError("Pipeline execution requires zarr_mode='r+'")

    # One run validates each input once across its stages; artifacts are
    # immutable and prepared cell IDs cannot change during the run.
    @validation_scoped
    def rna(
        self,
        *,
        assay: str | None = None,
        label: str | None = None,
        cells: CellsArgument = "I",
        cell_calling: StageSwitch = False,
        hashing: StageSwitch = False,
        filtering: StageSwitch | None = None,
        cell_cycle: StageSwitch = True,
        hvg: StageSettings = None,
        normalization: StageSettings = None,
        pca: StageSettings = None,
        harmony: Mapping[str, object] | Literal[False] | None = None,
        ann_index: StageSettings = None,
        neighbors: StageSettings = None,
        connectivity: StageSettings = None,
        embedding_initialization: StageSettings = None,
        umap: StageSwitch = True,
        tsne: StageSwitch = False,
        leiden: StageSwitch = True,
        paris: StageSwitch = False,
        membership_strength: bool = False,
        doublets: StageSwitch = True,
        markers: bool = True,
        snapshot_columns: Sequence[str] = (),
        species: str | None = None,
        callback: PipelineCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        """Analyze an RNA assay and return the durable run.

        The recipe optionally calls cells in raw droplets and demultiplexes
        hashtags, filters cells on their QC metrics, scores the cell cycle,
        selects highly variable genes, normalizes them by library size and
        log, reduces them with PCA, optionally corrects batches with Harmony,
        builds a neighbour graph, embeds it with UMAP, clusters it with Leiden
        at several resolutions, saves the candidate with the best silhouette,
        scores doublets, and ranks markers:

        ``input_snapshot``, ``cell_calling``, ``hashing``, ``filtering``,
        ``cell_cycle``, ``highly_variable_features``, ``normalization``,
        ``pca``, ``harmony``, ``ann_index``, ``neighbors``, ``connectivity``,
        ``embedding_initialization``, ``umap``, one ``leiden_<resolution>``
        per candidate, ``paris``, ``cluster_selection``,
        ``membership_strength``, ``tsne``, ``doublets``, ``markers``.

        Args:
            assay: The RNA assay. Defaults to the default assay when it is an
                RNA assay, else the store's only RNA assay.
            label: An immutable label for the completed run, unique in the store.
            cells: The cells to analyze: a boolean cell column, a cell-selection
                artifact, or a completed run, whose analysis cells are reused
                as they are.
            cell_calling: EmptyDrops cell calling for a raw droplet matrix,
                with the keywords of ``qc.call_cells`` and ``sample_column``,
                a cell column of captures that are called apart, which must
                label every cell of ``cells``.
            hashing: Hashtag demultiplexing, with the keywords of
                ``qc.hto_demultiplexing`` and ``assay``, the HTO assay
                (default: the store's only assay of type ``HTO``). The run
                keeps the cells of one hashtag and, unless ``filtering`` names
                a ``sample_column``, computes MAD bounds within each hashtag.
            filtering: QC filtering, with the settings of ``qc.auto_filter``
                (``method``, ``attrs``, ``n_mads``, ``sample_column``,
                ``min_cells_per_sample``, ``min_p``, ``max_p``) or, with
                ``method="manual"``, ``lows``, ``highs``, and ``keep_bounds``.
                None filters cells that ``cells`` names by column, and the
                cells that ``cell_calling`` or ``hashing`` keep whatever
                ``cells`` is; otherwise it keeps a given selection's or run's
                cells as they are.
            cell_cycle: Cell cycle scoring, with the keywords of
                ``scores.cell_cycle``.
            hvg: The keywords of ``features.hvgs``; ``batch`` names a cell
                column, and ``min_mean``, ``max_mean``, ``min_var``, and
                ``max_var`` are log2 values.
            normalization: The keywords of ``features.normalize``.
            pca: The keywords of ``reduction.pca``. ``dims`` must be below
                ``hvg`` ``top_n``. ``dims=0`` skips PCA and builds the graph
                on the normalized values, which needs ``doublets=False`` and
                no Harmony.
            harmony: ``batch_columns``, the cell columns whose batches Harmony
                corrects, with ``harmony_params`` (the options of
                ``cytearc.embeddings.harmony.fit_harmony``) and
                ``batch_size``. None or False skips Harmony. A run with
                Harmony scores doublets within each batch unless ``doublets``
                names another ``sample_column``.
            ann_index: The keywords of ``graph.ann_index``.
            neighbors: The keywords of ``graph.neighbors``, such as ``k``.
            connectivity: The keywords of ``graph.connectivity``.
            embedding_initialization: The keywords of
                ``embeddings.initialization``.
            umap: UMAP, with the keywords of ``embeddings.umap``.
            tsne: t-SNE on the UMAP graph and initialization, with the
                keywords of ``embeddings.tsne``.
            leiden: Leiden candidates: ``partitions``, the resolutions
                (default 0.5, 0.75, 1.0, 1.25), an optional ``selected``
                resolution that becomes the saved clustering in place of the
                silhouette choice, and the keywords of ``clusters.leiden``.
            paris: Paris clustering of the graph, for diagnosis, with the
                keywords of ``clusters.paris``.
            membership_strength: Score each cell's membership in its cluster.
            doublets: Doublet scoring, with the keywords of ``qc.doublets``.
            markers: Rank markers for each saved cluster.
            snapshot_columns: Cell columns to freeze into the run's cell view.
            species: A species key recorded with the run.
            callback: Called with each :class:`PipelineEvent` of the run.
            cancel: A :class:`~cytearc.CancellationToken` that stops the run.

        Raises:
            PermissionError: If the store is read-only.
            TypeError: If the call fails its checks; nothing is written.
            ValueError: If the call fails its checks; nothing is written.
            KeyError: If the call names a missing column; nothing is written.
            UnmeasuredCellsError: If the assay did not measure one of
                ``cells``; nothing is written.
            PipelineExecutionError: If a stage fails; the run is recorded failed.
            PipelineCancelled: If ``cancel`` is cancelled; the run is recorded
                interrupted.

        Example:
            >>> run = ds.pipeline.rna(label="rna", hvg={"top_n": 2000})
            >>> run["clusters"], run.cells.fetch("umap_1")
        """
        self._require_runnable(callback)
        plan = resolve_rna(
            self._store,
            assay=assay,
            label=label,
            cells=cells,
            cell_calling=cell_calling,
            hashing=hashing,
            filtering=filtering,
            cell_cycle=cell_cycle,
            hvg=hvg,
            normalization=normalization,
            pca=pca,
            harmony=harmony,
            ann_index=ann_index,
            neighbors=neighbors,
            connectivity=connectivity,
            embedding_initialization=embedding_initialization,
            umap=umap,
            tsne=tsne,
            leiden=leiden,
            paris=paris,
            membership_strength=membership_strength,
            doublets=doublets,
            markers=markers,
            snapshot_columns=snapshot_columns,
            species=species,
        )
        return execute_recipe(self._store, plan, callback=callback, cancel=cancel)

    @validation_scoped
    def atac(
        self,
        *,
        assay: str | None = None,
        label: str | None = None,
        cells: CellsArgument = "I",
        filtering: StageSwitch | None = None,
        prevalent_peaks: StageSettings = None,
        normalization: StageSettings = None,
        lsi: StageSettings = None,
        harmony: Mapping[str, object] | Literal[False] | None = None,
        ann_index: StageSettings = None,
        neighbors: StageSettings = None,
        connectivity: StageSettings = None,
        embedding_initialization: StageSettings = None,
        umap: StageSwitch = True,
        tsne: StageSwitch = False,
        leiden: StageSwitch = True,
        paris: StageSwitch = False,
        membership_strength: bool = False,
        markers: bool = True,
        snapshot_columns: Sequence[str] = (),
        species: str | None = None,
        callback: PipelineCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        """Analyze an ATAC assay of peak counts and return the durable run.

        The recipe filters cells on their counts and accessible peaks,
        selects the most prevalent peaks, normalizes them with TF-IDF, reduces
        them with LSI, optionally corrects batches with Harmony, and then
        builds, embeds, and clusters a graph as the RNA recipe does:

        ``input_snapshot``, ``filtering``, ``prevalent_peaks``,
        ``normalization``, ``lsi``, ``harmony``, ``ann_index``, ``neighbors``,
        ``connectivity``, ``embedding_initialization``, ``umap``, one
        ``leiden_<resolution>`` per candidate, ``paris``,
        ``cluster_selection``, ``membership_strength``, ``tsne``, ``markers``.

        Args:
            assay: The ATAC assay. Defaults to the default assay when it is an
                ATAC assay, else the store's only ATAC assay.
            prevalent_peaks: The keywords of ``features.prevalent_peaks``,
                such as ``top_n``, which defaults to 10,000 or one fewer than
                the assay's peaks, whichever is smaller, and must be below the
                assay's number of peaks. The run records the resolved
                ``top_n``.
            normalization: The keywords of ``features.normalize``.
            lsi: The keywords of ``reduction.lsi``, such as ``dims`` and
                ``skip_first``, a bool. ``dims``, plus the first component
                that ``skip_first`` leaves out, must not exceed
                ``prevalent_peaks`` ``top_n``. Cluster selection scores
                candidates in the LSI coordinates, or in their Harmony
                correction when the run has Harmony.

        The other arguments mean what they mean for :meth:`rna`. The run
        records the selected peaks as its ``prevalent_peaks`` feature field.
        """
        self._require_runnable(callback)
        plan = resolve_atac(
            self._store,
            assay=assay,
            label=label,
            cells=cells,
            filtering=filtering,
            prevalent_peaks=prevalent_peaks,
            normalization=normalization,
            lsi=lsi,
            harmony=harmony,
            ann_index=ann_index,
            neighbors=neighbors,
            connectivity=connectivity,
            embedding_initialization=embedding_initialization,
            umap=umap,
            tsne=tsne,
            leiden=leiden,
            paris=paris,
            membership_strength=membership_strength,
            markers=markers,
            snapshot_columns=snapshot_columns,
            species=species,
        )
        return execute_recipe(self._store, plan, callback=callback, cancel=cancel)

    @validation_scoped
    def adt(
        self,
        *,
        assay: str | None = None,
        label: str | None = None,
        cells: CellsArgument = "I",
        filtering: StageSwitch | None = None,
        panel: StageSettings = None,
        normalization: StageSettings = None,
        pca: StageSettings = None,
        harmony: Mapping[str, object] | Literal[False] | None = None,
        ann_index: StageSettings = None,
        neighbors: StageSettings = None,
        connectivity: StageSettings = None,
        embedding_initialization: StageSettings = None,
        umap: StageSwitch = True,
        tsne: StageSwitch = False,
        leiden: StageSwitch = True,
        paris: StageSwitch = False,
        membership_strength: bool = False,
        markers: bool = True,
        snapshot_columns: Sequence[str] = (),
        species: str | None = None,
        callback: PipelineCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        """Analyze an ADT (antibody) assay and return the durable run.

        The recipe leaves isotype controls out of the panel, normalizes the
        antibodies with CLR, reduces them with PCA, optionally corrects
        batches with Harmony, and then builds, embeds, and clusters a graph as
        the RNA recipe does:

        ``input_snapshot``, ``filtering``, ``panel``, ``normalization``,
        ``pca``, ``harmony``, ``ann_index``, ``neighbors``, ``connectivity``,
        ``embedding_initialization``, ``umap``, one ``leiden_<resolution>``
        per candidate, ``paris``, ``cluster_selection``,
        ``membership_strength``, ``tsne``, ``markers``.

        Analyze ADT with the cells of an RNA run, ``cells=rna_run``, to
        integrate the two with :meth:`multimodal`.

        Args:
            assay: The ADT assay, of type ``ADT``. Defaults to the default
                assay when it is one, else the store's only ADT assay.
            filtering: QC filtering, as for :meth:`rna`. The run filters only
                when ``filtering`` is True or a mapping; None, the default,
                keeps the cells as they are, since an ADT run usually takes
                the QC-passing cells of an RNA run. Without ``attrs``, the
                filter reads the assay's ``<assay>_nCounts`` column only.
            panel: ``exclude``, a regular expression of the antibody names to
                leave out (default ``DEFAULT_PANEL_EXCLUDE``, which matches
                "isotype", "control", and "ctrl" in any case), or None to keep
                every antibody. The panel must keep at least two antibodies,
                and the run records the excluded names.
            normalization: The keywords of ``features.normalize``.
            pca: The keywords of ``reduction.pca``. ``dims`` defaults to one
                fewer than the panel's antibodies, at most the PCA default,
                and must be below the panel's number of antibodies.

        The other arguments mean what they mean for :meth:`rna`. The run
        records the panel as its ``panel`` feature field.
        """
        self._require_runnable(callback)
        plan = resolve_adt(
            self._store,
            assay=assay,
            label=label,
            cells=cells,
            filtering=filtering,
            panel=panel,
            normalization=normalization,
            pca=pca,
            harmony=harmony,
            ann_index=ann_index,
            neighbors=neighbors,
            connectivity=connectivity,
            embedding_initialization=embedding_initialization,
            umap=umap,
            tsne=tsne,
            leiden=leiden,
            paris=paris,
            membership_strength=membership_strength,
            markers=markers,
            snapshot_columns=snapshot_columns,
            species=species,
        )
        return execute_recipe(self._store, plan, callback=callback, cancel=cancel)

    @validation_scoped
    def multimodal(
        self,
        runs: Sequence[PipelineRun],
        *,
        label: str | None = None,
        method: str = "wnn",
        integration: StageSettings = None,
        embedding_initialization: StageSettings = None,
        umap: StageSwitch = True,
        tsne: StageSwitch = False,
        leiden: StageSwitch = True,
        paris: StageSwitch = False,
        membership_strength: bool = False,
        markers: bool = True,
        snapshot_columns: Sequence[str] = (),
        callback: PipelineCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        """Integrate completed modality runs of the same cells into one graph.

        The recipe combines the runs' graphs with weighted nearest neighbours
        (``method="wnn"``, the default for modalities measured in the same
        cells) or shared nearest neighbours (``"snn"``), embeds and clusters
        the integrated graph, and ranks each modality's markers for the joint
        clusters:

        ``input_snapshot``, ``integration``, ``embedding_initialization``,
        ``umap``, one ``leiden_<resolution>`` per candidate, ``paris``,
        ``cluster_selection`` (always skipped), ``membership_strength``,
        ``tsne``, ``markers``.

        The runs must analyze the same cells: run each modality after the
        first with ``cells=<first run>``. WNN needs runs whose graphs were
        built on PCA, LSI, or Harmony coordinates, and SNN needs runs built
        with the same ``neighbors`` ``k``. The joint run's cell view holds the
        joint UMAP, Leiden, and ``clusters`` fields and, for WNN, each cell's
        ``weight_<assay>`` modality weights. Its ``assay`` and
        ``feature_universe`` are the first run's, and its feature view covers
        that assay's features with ``I``, ``ids``, and ``names`` only, without
        the first run's feature field.

        Args:
            runs: Two or more completed runs of different assays, such as
                ``[rna_run, adt_run]``.
            method: ``"wnn"`` or ``"snn"``.
            integration: The keywords of ``integration.modalities``:
                ``chunk_size`` and, for WNN, ``l2_normalize``.
            embedding_initialization: The keywords of
                ``embeddings.initialization``, which starts the joint
                embedding from the first run's coordinates.
            leiden: As for :meth:`rna`, but the integrated graph has no shared
                coordinates in which to compare candidates by silhouette, so
                ``partitions`` defaults to ``[1.0]`` and several partitions
                need a ``selected`` resolution.
            markers: Rank each run's assay's features for the joint clusters,
                as the ``markers_<assay>`` outputs.

        The other arguments mean what they mean for :meth:`rna`.
        """
        self._require_runnable(callback)
        plan = resolve_multimodal(
            self._store,
            runs,
            label=label,
            method=method,
            integration=integration,
            embedding_initialization=embedding_initialization,
            umap=umap,
            tsne=tsne,
            leiden=leiden,
            paris=paris,
            membership_strength=membership_strength,
            markers=markers,
            snapshot_columns=snapshot_columns,
        )
        return execute_recipe(self._store, plan, callback=callback, cancel=cancel)

    @validation_scoped
    def perturbation(
        self,
        *,
        assay: str | None = None,
        guide_assay: str | None = None,
        label: str | None = None,
        cells: CellsArgument = "I",
        filtering: StageSwitch | None = None,
        guide_calling: StageSettings = None,
        labels: Mapping[str, object],
        efficiency: StageSwitch = True,
        hvg: StageSettings = None,
        normalization: StageSettings = None,
        pca: StageSettings = None,
        harmony: Mapping[str, object] | Literal[False] | None = None,
        edistance: StageSwitch = True,
        signature: StageSettings = None,
        mixscape: StageSwitch = False,
        associations: StageSwitch = False,
        profiles: StageSettings = None,
        clusters: StageSwitch = False,
        snapshot_columns: Sequence[str] = (),
        callback: PipelineCallback | None = None,
        cancel: CancellationToken | None = None,
    ) -> PipelineRun:
        """Analyze a pooled CRISPR screen and return the durable run.

        The recipe filters cells on their RNA QC metrics, calls the guides of
        each cell from the guide counts or imports Cell Ranger's calls, labels
        each cell by the genes that its guides target, and measures each
        perturbation's efficiency on its own gene. It then selects highly
        variable genes, normalizes them, reduces them with PCA, optionally
        corrects batches with Harmony, and measures how far each perturbation
        moves its cells from the control cells. Optional stages call each
        target's knocked-out cells and escapers with Mixscape, test which
        highly variable genes each target changes, and cluster the
        perturbations whose profiles agree:

        ``input_snapshot``, ``filtering``, ``guide_calling``,
        ``perturbation_labels``, ``efficiency``, ``highly_variable_features``,
        ``normalization``, ``pca``, ``harmony``, ``edistance``,
        ``signature``, ``mixscape``, ``associations``, ``profiles``,
        ``perturbation_clusters``.

        The stages from ``filtering`` to ``harmony`` are those of
        :meth:`rna`, so a run with an RNA run's settings, or with
        ``cells=rna_run`` and the same ``hvg``, ``normalization``, ``pca``,
        and ``harmony`` settings, reuses its artifacts. The run's ``assay``
        is the RNA assay.

        The run checks its call before it writes anything. Which cells a
        stage reads depends on the data, so every sample or split column that
        a stage reads must label every cell of ``cells``, with neither a
        missing value nor empty or whitespace text, and every covariate of
        ``associations`` must hold a finite value for each of them.

        Args:
            assay: The RNA assay. Defaults to the default assay when it is an
                RNA assay, else the store's only RNA assay.
            guide_assay: The guide assay, declared ``CRISPR``. Defaults to the
                store's only assay declared ``CRISPR``.
            label: An immutable label for the completed run, unique in the store.
            cells: The cells to analyze, as for :meth:`rna`. The guide assay
                must have measured them too.
            filtering: QC filtering on the RNA assay's metrics, as for
                :meth:`rna`.
            guide_calling: The keywords of ``perturbation.call_guides``
                (``method``, ``min_umi``, ``fraction_threshold``, and
                ``sample_column``), or ``path`` and ``barcode_prefix``, which
                import Cell Ranger's ``protospacer_calls_per_cell.csv`` with
                ``perturbation.import_guide_calls`` instead. The guides of the
                run's analysis cells are called. A call file holds the
                barcodes of one sample, so with ``barcode_prefix`` every cell
                of ``cells`` must have an ID that starts with it, and without
                one, ``cells`` must not come from several samples of a merged
                store, whose cell IDs are ``<sample>__<barcode>``.
            labels: The keywords of ``perturbation.labels``: ``controls``,
                the targets of the control guides, which the recipe needs,
                and ``target_column``, ``targets``, and ``policy``.
            efficiency: The efficiency of each perturbation on its target
                gene, with the keywords of ``perturbation.efficiency`` (``by``
                and ``id_column``).
            hvg: The keywords of ``features.hvgs``, as for :meth:`rna`.
            normalization: The keywords of ``features.normalize``.
            pca: The keywords of ``reduction.pca``. ``dims`` must be at least
                1 and below ``hvg`` ``top_n``.
            harmony: Harmony settings, as for :meth:`rna`. With Harmony, the
                E-distances and the signature read the corrected coordinates.
            edistance: E-distances in the run's coordinates, with the keywords
                of ``perturbation.edistance`` (``by``, ``n_permutations``,
                and ``random_seed``).
            signature: The keywords of ``perturbation.signature`` (``dims``,
                ``n_neighbors``, ``sample_column``, and ``scale_factor``) for
                the signature that ``mixscape`` reads, computed in the highly
                variable genes. ``dims`` must be at most ``pca`` ``dims``.
                Settings without ``mixscape`` are refused.
            mixscape: Mixscape's knocked-out cells and escapers, with the
                keywords of ``perturbation.mixscape``.
            associations: Tests of every target against every highly variable
                gene, with the keywords of ``perturbation.associations`` but
                ``pairs`` and ``features``; ``positive_controls`` is
                ``"targets"`` or None.
            profiles: The keywords of ``perturbation.profiles`` (``by``,
                ``sample_column``, ``min_cells``, ``mask_targets``, and
                ``id_column``) for the profiles that ``clusters`` reads, of
                the highly variable genes. With ``mixscape``, a target's
                profile averages its knocked-out cells. Settings without
                ``clusters`` are refused.
            clusters: Clusters of the perturbations whose profiles agree, with
                the keywords of ``perturbation.cluster``.
            snapshot_columns: Cell columns to freeze into the run's cell view.
            callback: Called with each :class:`PipelineEvent` of the run.
            cancel: A :class:`~cytearc.CancellationToken` that stops the run.

        Raises:
            PermissionError: If the store is read-only.
            TypeError: If the call fails its checks; nothing is written.
            ValueError: If the call fails its checks; nothing is written.
            KeyError: If the call names a missing column; nothing is written.
            FileNotFoundError: If ``guide_calling`` ``path`` names no file.
            UnmeasuredCellsError: If the RNA or the guide assay did not
                measure one of ``cells``; nothing is written.
            PipelineExecutionError: If a stage fails; the run is recorded failed.
            PipelineCancelled: If ``cancel`` is cancelled; the run is recorded
                interrupted.

        Example:
            >>> labels = {"controls": ["Non-Targeting"]}
            >>> run = ds.pipeline.perturbation(labels=labels, label="screen")
            >>> ds.perturbation.load_efficiency(run["efficiency"]).table
        """
        self._require_runnable(callback)
        plan = resolve_perturbation(
            self._store,
            assay=assay,
            guide_assay=guide_assay,
            label=label,
            cells=cells,
            filtering=filtering,
            guide_calling=guide_calling,
            labels=labels,
            efficiency=efficiency,
            hvg=hvg,
            normalization=normalization,
            pca=pca,
            harmony=harmony,
            edistance=edistance,
            signature=signature,
            mixscape=mixscape,
            associations=associations,
            profiles=profiles,
            clusters=clusters,
            snapshot_columns=snapshot_columns,
        )
        return execute_recipe(self._store, plan, callback=callback, cancel=cancel)
