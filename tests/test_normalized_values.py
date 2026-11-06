"""Assays of values normalized before import: refusals, copies, and the RNA recipe.

Steps that need counts read only an assay's ``normalizedValues`` record, so
their refusals are tested on stores of counts that carry it, where every other
input of the step is valid. Copies and combinations must carry the record, and
the RNA recipe must give the counts' results on log-normalized values.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix

import cytearc
from cytearc import DataStore, DataStoreMerge, SparseToZarr, SubsetZarr
from cytearc.assay.normalization import PearsonResidualNormalizer, norm_clr
from cytearc.composite import create_composite
from cytearc.datastore._pipeline.filtering import default_filter_columns
from cytearc.datastore._pipeline.recipes.rna import FILTER_SUFFIXES
from cytearc.readers import H5adReader
from cytearc.tools.repack_zarr import repack_store
from cytearc.writers import H5adToZarr

from . import full_path
from .perturbation_helpers import fixed_labels, write_screen
from .storage_helpers import record_normalized_values, write_count_store

_DERIVED = {"importedAs": None, "cellTotal": None}


def _open(path: Path) -> DataStore:
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


def _grouped_counts(cells_per_group: int = 80, genes: int = 100) -> np.ndarray:
    """Seeded counts of three groups of cells, each with its own marker genes."""
    rng = np.random.default_rng(21)
    means = rng.gamma(0.6, 1.5, size=genes)
    blocks = []
    for group in range(3):
        group_means = means.copy()
        group_means[group * 10 : group * 10 + 10] *= 8
        depth = rng.lognormal(0.0, 0.3, size=(cells_per_group, 1))
        blocks.append(rng.poisson(depth * group_means))
    return np.vstack(blocks).astype(np.uint16)


@dataclass(frozen=True)
class _Stores:
    counts: DataStore
    normalized: DataStore
    cells: object
    graph: object
    clusters: object
    profile: object
    contamination: object


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> _Stores:
    """A store of counts with a graph, clusters and an ambient estimate, and its
    copy recorded as holding log-normalized values."""
    folder = tmp_path_factory.mktemp("normalized_values")
    counts_path = folder / "counts.zarr"
    values = _grouped_counts()
    write_count_store(str(counts_path), {"RNA": values}, "uint16")
    counts = _open(counts_path)
    counts.cells.insert("group", np.repeat(["a", "b", "c"], len(values) // 3))
    run = counts.pipeline.rna(
        label="graph",
        filtering=False,
        cell_cycle=False,
        doublets=False,
        markers=False,
        umap=False,
    )
    cells = counts.snapshot_cell_selection("I")
    totals = counts.cells.fetch_all("RNA_nCounts")
    profile = counts.qc.ambient_profile(total_counts=(0, float(np.median(totals))))
    contamination = counts.qc.ambient_contamination(cells, profile, fraction=0.1)
    normalized_path = folder / "normalized.zarr"
    shutil.copytree(counts_path, normalized_path)
    record_normalized_values(normalized_path, "RNA")
    return _Stores(
        counts=counts,
        normalized=_open(normalized_path),
        cells=cells,
        graph=run["connectivity_map"],
        clusters=run["clusters"],
        profile=profile,
        contamination=contamination,
    )


# Each step that needs counts, how it is called, and the start of its refusal.
_REFUSALS = {
    "cell calling": (
        lambda s, store: store.qc.call_cells("RNA"),
        "Cell calling needs counts: it ranks and models droplets",
    ),
    "barcode ranks": (
        lambda s, store: store.qc.barcode_ranks("RNA"),
        "Barcode ranks needs counts",
    ),
    "ambient profile": (
        lambda s, store: store.qc.ambient_profile(
            total_counts=(0, float(np.median(store.cells.fetch_all("RNA_nCounts"))))
        ),
        "estimate_ambient_profile needs counts: it models ambient RNA",
    ),
    "ambient contamination": (
        lambda s, store: store.qc.ambient_contamination(
            s.cells, s.profile, fraction=0.1
        ),
        "estimate_ambient_contamination needs counts",
    ),
    "ambient correction": (
        lambda s, store: store.qc.correct_ambient(s.contamination, assay_label="clean"),
        "correct_ambient needs counts",
    ),
    "doublet scoring": (
        lambda s, store: store.qc.doublets(s.clusters, s.graph),
        "Doublet scoring needs counts: it simulates each doublet by adding",
    ),
    "seurat_v3 genes": (
        lambda s, store: store.features.hvgs(s.cells, flavor="seurat_v3"),
        "flavor='seurat_v3' needs counts.*Use flavor='cytearc'",
    ),
    "Pearson residual genes": (
        lambda s, store: store.features.hvgs(s.cells, flavor="pearson_residuals"),
        "flavor='pearson_residuals' needs counts",
    ),
    "a bound on nCounts": (
        lambda s, store: store.qc.filter(attrs=["RNA_nCounts"], lows=[0], highs=[1e9]),
        "A bound on RNA_nCounts needs counts.*Bound RNA_nFeatures",
    ),
    "a MAD bound on nCounts": (
        lambda s, store: store.qc.auto_filter(attrs=["RNA_nCounts"]),
        "A bound on RNA_nCounts needs counts",
    ),
    "pseudobulk sums": (
        lambda s, store: store.compare.compute_bulk(
            "group", cell_selection=s.cells, aggr_type="sum"
        ),
        "Pseudobulk sums needs counts.*Use aggr_type='mean'",
    ),
    "Pearson residuals": (
        lambda s, store: _normed(store, PearsonResidualNormalizer()),
        "Pearson residuals needs counts",
    ),
    "CLR normalization": (
        lambda s, store: _normed(store, norm_clr),
        "CLR normalization needs counts",
    ),
}


def _normed(store: DataStore, normalizer) -> object:
    original = store.RNA.normMethod
    store.RNA.normMethod = normalizer
    try:
        return store.RNA.normed().compute()
    finally:
        store.RNA.normMethod = original


@pytest.mark.parametrize("step", list(_REFUSALS))
def test_every_step_that_needs_counts_refuses_normalized_values(stores, step):
    call, message = _REFUSALS[step]

    with pytest.raises(TypeError, match=message) as refusal:
        call(stores, stores.normalized)

    assert (
        "Assay 'RNA' holds normalized values, imported as log-normalized values"
        in str(refusal.value)
    )


@pytest.mark.parametrize(
    "step",
    [
        "barcode ranks",
        "ambient profile",
        "ambient contamination",
        "doublet scoring",
        "seurat_v3 genes",
        "Pearson residual genes",
        "a bound on nCounts",
        "a MAD bound on nCounts",
        "pseudobulk sums",
        "Pearson residuals",
        "CLR normalization",
    ],
)
def test_the_same_steps_run_on_the_counts(stores, step):
    call, _ = _REFUSALS[step]

    call(stores, stores.counts)


def test_guide_calls_and_association_tests_refuse_normalized_values(tmp_path):
    rng = np.random.default_rng(2)
    rna = rng.poisson(2.0, size=(200, 8))
    guides = np.zeros((200, 4), dtype=np.int64)
    guides[np.arange(200), np.arange(200) % 4] = 20
    path = tmp_path / "screen.zarr"
    store = write_screen(path, rna, guides, targets=["RNA0", "RNA0", "NT", "NT"])
    labels = fixed_labels(store, ["NT"])
    features = store.features.snapshot(feature_indexes=np.arange(8))

    guides_path = tmp_path / "guides.zarr"
    shutil.copytree(path, guides_path)
    record_normalized_values(guides_path, "CRISPR", "normalized")
    recorded = _open(guides_path)
    with pytest.raises(TypeError, match="Guide calls needs counts.*'CRISPR'"):
        recorded.perturbation.call_guides(
            recorded.select_measured_cells("CRISPR"), method="fixed", min_umi=10
        )

    record_normalized_values(path, "RNA")
    recorded = _open(path)
    with pytest.raises(TypeError, match="Association tests needs counts: they fit"):
        recorded.perturbation.associations(labels, features=features)


def test_the_rna_recipe_names_every_count_stage_in_one_refusal(stores):
    store = stores.normalized

    with pytest.raises(
        TypeError, match="its doublet scoring reads counts.*Pass doublets=False"
    ):
        store.pipeline.rna(label="defaults")
    with pytest.raises(
        TypeError,
        match="its cell calling and doublet scoring read counts.*Pass "
        "cell_calling=False and doublets=False",
    ):
        store.pipeline.rna(label="calling", cell_calling=True)
    with pytest.raises(TypeError, match="its bound on RNA_nCounts reads counts"):
        store.pipeline.rna(
            label="totals", doublets=False, filtering={"attrs": ["RNA_nCounts"]}
        )
    for flavor in ("seurat_v3", "pearson_residuals"):
        with pytest.raises(
            TypeError,
            match=f"its {flavor} gene selection reads counts.*Pass "
            "hvg={'flavor': 'cytearc'}",
        ):
            store.pipeline.rna(label="flavor", doublets=False, hvg={"flavor": flavor})
    with pytest.raises(
        TypeError,
        match="its cell calling, pearson_residuals gene selection and doublet "
        "scoring read counts.*Pass cell_calling=False, hvg=.* and doublets=False",
    ):
        store.pipeline.rna(
            label="three", cell_calling=True, hvg={"flavor": "pearson_residuals"}
        )


def test_the_perturbation_recipe_refuses_its_count_stages(tmp_path):
    rng = np.random.default_rng(2)
    rna = rng.poisson(2.0, size=(200, 8))
    guides = np.zeros((200, 4), dtype=np.int64)
    guides[np.arange(200), np.arange(200) % 4] = 20
    path = tmp_path / "screen.zarr"
    write_screen(
        path,
        rna,
        guides,
        targets=["RNA0", "RNA0", "NT", "NT"],
        target_ids=["RNA0", "RNA0", "NT", "NT"],
    )
    guides_path = tmp_path / "guides.zarr"
    shutil.copytree(path, guides_path)
    record_normalized_values(path, "RNA")
    record_normalized_values(guides_path, "CRISPR", "normalized")
    head = {
        "labels": {"controls": ["NT"]},
        "hvg": {"top_n": 6},
        "pca": {"dims": 3},
        "efficiency": False,
    }

    store = _open(path)
    with pytest.raises(TypeError, match="its seurat_v3 gene selection reads counts"):
        store.pipeline.perturbation(
            **{**head, "hvg": {"top_n": 6, "flavor": "seurat_v3"}}
        )
    with pytest.raises(TypeError, match="its bound on RNA_nCounts reads counts"):
        store.pipeline.perturbation(**head, filtering={"attrs": ["RNA_nCounts"]})
    with pytest.raises(
        TypeError, match="its association testing reads counts.*associations=False"
    ):
        store.pipeline.perturbation(**head, associations=True)
    # The guide assay is checked when the recipe resolves it, before the run.
    with pytest.raises(TypeError, match="Guide calls needs counts.*'CRISPR' holds"):
        _open(guides_path).pipeline.perturbation(**head)


def test_default_filtering_leaves_out_the_totals_of_normalized_values(stores):
    normalized = default_filter_columns(stores.normalized, "RNA", FILTER_SUFFIXES)
    counts = default_filter_columns(stores.counts, "RNA", FILTER_SUFFIXES)

    assert counts[0] == "RNA_nCounts"
    assert normalized == counts[1:]
    # A bound on the totals would refuse, so the defaults ran without one.
    stores.normalized.qc.auto_filter()


def test_adt_and_atac_assays_refuse_normalized_values_at_import_and_open(tmp_path):
    values = csr_matrix(_grouped_counts()[:30, :12].astype(np.float32) / 7)
    ids = [f"c{i}" for i in range(30)]
    for name, reason in (
        ("ADT", "ADT and HTO assays need counts"),
        ("ATAC", "ATAC assays need counts"),
    ):
        with pytest.raises(
            TypeError, match=f"Assay '{name}' holds normalized values, but {reason}"
        ):
            SparseToZarr(
                values,
                str(tmp_path / f"{name}.zarr"),
                cell_ids=ids,
                feature_ids=[f"f{j}" for j in range(12)],
                assay_name=name,
                values="normalized",
            )
        assert not (tmp_path / f"{name}.zarr").exists()

    path = tmp_path / "cite.zarr"
    counts = _grouped_counts()[:30, :12]
    write_count_store(str(path), {"RNA": counts, "ADT": counts}, "uint16")
    record_normalized_values(path, "ADT", "normalized")
    with pytest.raises(TypeError, match="Assay 'ADT' holds normalized values, but ADT"):
        _open(path)


def test_barcode_import_from_an_assay_refuses_normalized_values(stores):
    with pytest.raises(TypeError, match="Barcode import needs counts"):
        stores.normalized.clones.import_barcodes(
            from_assay="RNA", cell_selection=stores.cells
        )


def test_the_generic_normalizer_path_refuses_clr_for_normalized_values(tmp_path):
    path = tmp_path / "custom.zarr"
    counts = _grouped_counts()[:30, :12]
    write_count_store(str(path), {"RNA": counts, "Custom": counts}, "uint16")
    record_normalized_values(path, "Custom", "normalized")
    custom = _open(path).get_assay("Custom")
    custom.normMethod = norm_clr

    with pytest.raises(TypeError, match="CLR normalization needs counts"):
        custom.normed()


def test_values_are_checked_before_the_destination_exists(tmp_path):
    ids = [f"c{i}" for i in range(4)]
    matrix = csr_matrix(
        np.array([[1.5, 2.5], [3.5, 100.5], [2.5, 2.5], [1.5, 5.5]], dtype=np.float32)
    )
    destination = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match="values must be None or one of"):
        SparseToZarr(
            matrix,
            str(destination),
            cell_ids=ids,
            feature_ids=["a", "b"],
            values="log1p",
        )
    # Undoing the log of 100.5 overflows float32, which the scan's largest value shows.
    with pytest.raises(ValueError, match="100.5 of assay 'RNA' overflows float32"):
        SparseToZarr(
            matrix,
            str(destination),
            cell_ids=ids,
            feature_ids=["a", "b"],
            values="log_normalized",
        )
    assert not destination.exists()


def _capture_warnings():
    from contextlib import contextmanager

    from cytearc.utils.logging import logger

    @contextmanager
    def capture():
        messages: list[str] = []
        handler = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            yield messages
        finally:
            logger.remove(handler)

    return capture()


def test_an_h5ad_import_of_normalized_values_names_the_counts_beside_them(tmp_path):
    counts = csr_matrix(_grouped_counts()[:40, :20].astype(np.float64) + 1)
    logged = csr_matrix(sp.diags(1e4 / np.asarray(counts.sum(axis=1)).ravel()) @ counts)
    logged.data = np.log1p(logged.data)
    cells = [f"c{i}" for i in range(40)]
    features = [f"g{j}" for j in range(20)]

    def write(path: Path, x: csr_matrix) -> Path:
        _write_h5ad(path, x, cells, features, features)
        with h5py.File(path, "r+") as h5:
            raw = h5.create_group("raw")
            group = raw.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = counts.shape
            for name in ("data", "indices", "indptr"):
                group.create_dataset(name, data=getattr(counts, name))
            raw.create_group("var").create_dataset(
                "_index", data=np.array([name.encode() for name in features])
            )
        return path

    reader = H5adReader(str(write(tmp_path / "logged.h5ad", logged)))
    try:
        with _capture_warnings() as messages:
            writer = H5adToZarr(reader, str(tmp_path / "logged.zarr"))
    finally:
        reader.close()
    assert writer.values["RNA"].kind == "log_normalized"
    assert any(
        "X holds normalized values, and ['raw/X'] hold counts" in m for m in messages
    )

    scaled = logged.copy()
    scaled.data = scaled.data - scaled.data.mean()
    reader = H5adReader(str(write(tmp_path / "scaled.h5ad", scaled)))
    try:
        with pytest.raises(
            ValueError, match=r"negative values.*also holds counts in \['raw/X'\]"
        ):
            H5adToZarr(reader, str(tmp_path / "scaled.zarr"))
    finally:
        reader.close()


def test_melded_gene_scores_refuse_a_type_that_needs_counts(stores):
    with pytest.raises(ValueError, match="assay_type 'ATAC' needs counts"):
        stores.counts.add_melded_assay(
            from_assay="RNA", external_bed_fn=None, assay_label="g", assay_type="ATAC"
        )


@pytest.fixture
def normalized_source(tmp_path) -> DataStore:
    counts = np.random.default_rng(4).poisson(0.9, size=(60, 30)).astype(np.float64)
    counts[:, 0] = np.maximum(counts[:, 0], 1)
    values = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    SparseToZarr(
        csr_matrix(values.astype(np.float32)),
        str(tmp_path / "source.zarr"),
        cell_ids=[f"c{i}" for i in range(60)],
        feature_ids=[f"g{j}" for j in range(30)],
    ).dump()
    store = _open(tmp_path / "source.zarr")
    record = store.RNA.normalizedValues
    assert record["importedAs"] == "log_normalized"
    assert record["cellTotal"] == pytest.approx(1e4, rel=1e-5)
    return store


def _counts_source(tmp_path: Path) -> DataStore:
    counts = np.random.default_rng(5).poisson(0.9, size=(40, 30)).astype(np.uint16)
    write_count_store(
        str(tmp_path / "counts.zarr"),
        {"RNA": counts},
        "uint16",
        feature_ids={"RNA": [f"g{j}" for j in range(30)]},
        cell_ids=[f"k{i}" for i in range(40)],
    )
    return _open(tmp_path / "counts.zarr")


def _mount(source: DataStore, tmp_path: Path) -> DataStore:
    return cytearc.mount_datastore(
        source.zarr_loc,
        at=str(tmp_path / "mounted.zarr"),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
    )


def _repack(source: DataStore, tmp_path: Path, *, data_only: bool) -> DataStore:
    repack_store(source.zarr_loc, str(tmp_path / "repacked.zarr"), data_only=data_only)
    return _open(tmp_path / "repacked.zarr")


def _subset(source: DataStore, tmp_path: Path) -> DataStore:
    SubsetZarr(
        str(tmp_path / "subset.zarr"), [source.RNA], cell_idx=np.arange(20)
    ).dump()
    return _open(tmp_path / "subset.zarr")


def _merge_with_itself(source: DataStore, tmp_path: Path) -> DataStore:
    DataStoreMerge(
        [source, source], str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
    ).dump()
    return _open(tmp_path / "merged.zarr")


def _composite(source: DataStore, tmp_path: Path) -> DataStore:
    return create_composite(
        {"source": source},
        at=str(tmp_path / "joint.zarr"),
        features="intersection",
        nthreads=1,
    )


def _exported(source: DataStore, tmp_path: Path) -> DataStore:
    path = str(tmp_path / "exported.h5ad")
    cytearc.to_h5ad(source.RNA, path)
    reader = H5adReader.from_inspect(cytearc.inspect_h5ad(path))
    try:
        H5adToZarr(reader, str(tmp_path / "reimported.zarr"), nthreads=1).dump()
    finally:
        reader.close()
    return _open(tmp_path / "reimported.zarr")


@pytest.mark.parametrize(
    ("path", "same_record"),
    [
        (_mount, True),
        (lambda source, tmp: _repack(source, tmp, data_only=False), True),
        (lambda source, tmp: _repack(source, tmp, data_only=True), True),
        (_subset, True),
        # A combined assay's features can differ from a source's, so it keeps
        # the import kind but no cell total.
        (_merge_with_itself, "combined"),
        (_composite, "combined"),
        # An export holds the linear values, which a new import finds normalized.
        (_exported, False),
    ],
    ids=[
        "mount",
        "repack",
        "data-only repack",
        "subset",
        "merge",
        "composite",
        "export",
    ],
)
def test_every_copy_and_combination_carries_the_record(
    normalized_source, tmp_path, path, same_record
):
    record = normalized_source.RNA.normalizedValues

    copied = path(normalized_source, tmp_path)

    if same_record == "combined":
        assert copied.RNA.normalizedValues == {**record, "cellTotal": None}
    elif same_record:
        assert copied.RNA.normalizedValues == record
    else:
        assert copied.RNA.normalizedValues["importedAs"] == "normalized"
        assert copied.RNA.normalizedValues["cellTotal"] == pytest.approx(1e4, rel=1e-5)
    with pytest.raises(
        TypeError, match="Doublet scoring needs counts|holds normalized"
    ):
        copied.RNA.require_counts("Doublet scoring", "a reason", "a remedy")


def test_a_merge_with_counts_holds_normalized_values_of_no_single_import(
    normalized_source, tmp_path
):
    counts = _counts_source(tmp_path)
    DataStoreMerge(
        [normalized_source, counts],
        str(tmp_path / "mixed.zarr"),
        ["n", "c"],
        nthreads=1,
    ).dump()

    assert _open(tmp_path / "mixed.zarr").RNA.normalizedValues == _DERIVED


def test_grouped_assays_hold_normalized_values_even_from_counts(tmp_path):
    counts = _counts_source(tmp_path)
    counts.RNA.feats.insert("group", np.repeat(["a", "b", "c"], 10))

    counts.add_grouped_assay("group", assay_label="grouped")

    assert counts.get_assay("grouped").normalizedValues == _DERIVED
    assert counts.RNA.normalizedValues is None


def test_the_summary_reports_the_record(normalized_source, tmp_path):
    counts = _counts_source(tmp_path)

    assert normalized_source.summary().assays[0].normalized_values == (
        normalized_source.RNA.normalizedValues
    )
    assert normalized_source.summary().to_dict()["assays"][0]["normalized_values"] == (
        normalized_source.RNA.normalizedValues
    )
    assert counts.summary().to_dict()["assays"][0]["normalized_values"] is None


def _write_h5ad(path: Path, values: csr_matrix, cells, features, names) -> Path:
    with h5py.File(path, "w") as h5:
        group = h5.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = values.shape
        for name in ("data", "indices", "indptr"):
            group.create_dataset(name, data=getattr(values, name))
        h5.create_group("obs").create_dataset(
            "_index", data=np.array([str(cell).encode() for cell in cells])
        )
        var = h5.create_group("var")
        var.create_dataset(
            "_index", data=np.array([str(feature).encode() for feature in features])
        )
        var.create_dataset(
            "feature_name", data=np.array([str(name).encode() for name in names])
        )
    return path


@pytest.mark.slow
def test_the_rna_recipe_gives_the_counts_results_on_log_normalized_values(tmp_path):
    """Values normalized and logged outside CyteArc, as scanpy writes them."""
    with h5py.File(full_path("1K_pbmc_citeseq.h5"), "r") as h5:
        matrix = h5["matrix"]
        shape = tuple(matrix["shape"][:])
        source = csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape
        ).T.tocsr()
        genes = matrix["features/feature_type"][:] == b"Gene Expression"
        ids = matrix["features/id"][:].astype(str)[genes]
        names = matrix["features/name"][:].astype(str)[genes]
        barcodes = matrix["barcodes"][:].astype(str)
    counts = source[:, genes]
    keep = np.asarray(counts.sum(axis=1)).ravel() > 0
    counts = counts[keep].astype(np.float64)
    barcodes = barcodes[keep]
    # scanpy's normalize_total(target_sum=1e4) and log1p, in float32.
    totals = np.asarray(counts.sum(axis=1)).ravel()
    logged = csr_matrix(counts.multiply(1e4 / totals[:, None])).astype(np.float32)
    logged.data = np.log1p(logged.data)
    path = _write_h5ad(tmp_path / "logged.h5ad", logged, barcodes, ids, names)

    results = {}
    for name, declared in (("counts", None), ("logged", "log_normalized")):
        location = str(tmp_path / f"{name}.zarr")
        if name == "counts":
            SparseToZarr(
                counts.astype(np.uint32),
                location,
                cell_ids=barcodes,
                feature_ids=ids,
                feature_names=names,
            ).dump()
        else:
            reader = H5adReader(str(path), feature_name_key="feature_name")
            try:
                H5adToZarr(reader, location, values=declared).dump()
            finally:
                reader.close()
        store = DataStore(location, default_assay="RNA")
        run = store.pipeline.rna(
            label="check",
            filtering=False,
            doublets=False,
            markers=False,
            umap=False,
            species="homo_sapiens",
        )
        results[name] = {
            "prepared": {
                column: np.asarray(store.cells.fetch_all(column))
                for column in ("RNA_nFeatures", "RNA_percentMito", "RNA_percentRibo")
            },
            "hvgs": np.asarray(
                store.artifacts.load(run["highly_variable_features"])["values"][:]
            ),
            "pca": np.asarray(store.artifacts.load(run["pca"])["data"][:]),
            "clusters": np.asarray(run.cells.fetch("clusters")),
            "s_score": np.asarray(run.cells.fetch("s_score")),
            "record": store.RNA.normalizedValues,
        }

    counts_run, logged_run = results["counts"], results["logged"]
    assert counts_run["record"] is None
    assert logged_run["record"]["importedAs"] == "log_normalized"
    # Preparation's detected genes and percentages are ratios within a cell.
    np.testing.assert_array_equal(
        logged_run["prepared"]["RNA_nFeatures"], counts_run["prepared"]["RNA_nFeatures"]
    )
    for column in ("RNA_percentMito", "RNA_percentRibo"):
        np.testing.assert_allclose(
            logged_run["prepared"][column], counts_run["prepared"][column], atol=1e-4
        )
    np.testing.assert_array_equal(logged_run["hvgs"], counts_run["hvgs"])
    signs = np.sign(np.sum(logged_run["pca"] * counts_run["pca"], axis=0))
    np.testing.assert_allclose(
        logged_run["pca"] * signs,
        counts_run["pca"],
        atol=1e-4 * np.abs(counts_run["pca"]).max(),
    )
    np.testing.assert_array_equal(logged_run["clusters"], counts_run["clusters"])
    np.testing.assert_allclose(logged_run["s_score"], counts_run["s_score"], atol=1e-6)
    # Detection reaches the same decision as the declaration.
    reader = H5adReader(str(path), feature_name_key="feature_name")
    try:
        detected = H5adToZarr(reader, str(tmp_path / "detected.zarr"))
    finally:
        reader.close()
    assert detected.values["RNA"].kind == "log_normalized"
