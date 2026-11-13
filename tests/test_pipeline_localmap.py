"""The optional LocalMAP stage of an RNA run, on the 1K PBMC store."""

import inspect
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cytearc import DataStore, PipelineRun
from cytearc.datastore.namespaces import EmbeddingsAccessor
from cytearc.embeddings import localmap
from cytearc.graph.feature_projection import resolve_native_graph_inputs
from tests.test_pipeline_plan import assert_plan_reuses_run, read_only_plan

# Each test runs a pipeline on the 1K PBMC store.
pytestmark = pytest.mark.slow

# A quick RNA run without the stages that LocalMAP does not read.
FAST: dict[str, Any] = {
    "filtering": False,
    "cell_cycle": False,
    "hvg": {"top_n": 50},
    "pca": {"dims": 3},
    "neighbors": {"k": 3},
    "umap": False,
    "leiden": False,
    "doublets": False,
    "markers": False,
}
# Few iterations of each phase keep the layout quick.
SHORT = [4, 4, 6]
# The outputs of the quick run without LocalMAP.
DEFAULT_OUTPUTS = [
    "input_cell_selection",
    "analysis_cell_selection",
    "feature_universe",
    "highly_variable_features",
    "normalized",
    "pca",
    "ann_index",
    "neighbors",
    "connectivity_map",
]


@pytest.fixture(scope="module")
def runs(datastore_zarr_root: str, tmp_path_factory) -> SimpleNamespace:
    """Commit the quick run, and then the same run with LocalMAP.

    The calls that the LocalMAP stage makes to ``embeddings.localmap`` are
    recorded.
    """
    location = tmp_path_factory.mktemp("pipeline_localmap") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    datastore = DataStore(str(location), default_assay="RNA")
    default = datastore.pipeline.rna(label="default", **FAST)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    method = DataStore._embeddings_localmap

    def recorded(self, *args, **kwargs):
        calls.append((args, kwargs))
        return method(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DataStore, "_embeddings_localmap", recorded)
        run = datastore.pipeline.rna(
            label="localmap", localmap={"iterations": SHORT}, **FAST
        )
    return SimpleNamespace(
        location=location, datastore=datastore, default=default, run=run, calls=calls
    )


def _values(datastore: DataStore, run: PipelineRun) -> np.ndarray:
    return np.asarray(datastore.artifacts.load(run["localmap"])["values"][:])


def _kernel_layout(datastore: DataStore, run: PipelineRun) -> np.ndarray:
    """Lay out the run's graph with the LocalMAP kernels, at the method's defaults."""
    graph = run["connectivity_map"]
    # The graph was built on the run's PCA by Euclidean distance, so the
    # mid-near pairs are drawn in the PCA coordinates as they are.
    assert resolve_native_graph_inputs(datastore.zw, graph).coordinates == run["pca"]
    neighbors = datastore.artifacts.inspect(run["neighbors"]).parameters
    assert neighbors["distance_metric"] == "l2"
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            EmbeddingsAccessor.localmap
        ).parameters.items()
    }
    seed = defaults["random_seed"]
    coordinates = np.asarray(
        datastore.artifacts.load(run["pca"])["data"][:], dtype=np.float32
    )
    matrix = datastore.graph.load(graph)
    start = datastore._get_ini_embed(run["embedding_initialization"], graph, 2)
    return localmap.localmap_layout(
        localmap.initial_layout(start),
        matrix,
        localmap.mid_near_pairs(coordinates, defaults["n_mid_near"], seed),
        localmap.further_pairs(matrix, defaults["n_further"], seed),
        iterations=(SHORT[0], SHORT[1], SHORT[2]),
        learning_rate=defaults["learning_rate"],
        max_distance=defaults["max_distance"],
        random_seed=seed,
    )


def test_a_localmap_run_lays_out_its_graph_from_its_initialization(runs) -> None:
    datastore, run = runs.datastore, runs.run

    assert list(run) == [*DEFAULT_OUTPUTS, "embedding_initialization", "localmap"]
    statuses = {stage["stage"]: stage["status"] for stage in run.report()["stages"]}
    assert (
        statuses["embedding_initialization"],
        statuses["umap"],
        statuses["localmap"],
    ) == ("completed", "skipped", "completed")
    assert run.report()["run"]["config"]["stages"]["localmap"] == {"iterations": SHORT}
    # The stage passes the run's graph and initialization with the call's
    # settings only, so the layout runs on the datastore's threads.
    ((args, kwargs),) = runs.calls
    assert args == (run["connectivity_map"], run["embedding_initialization"])
    assert kwargs == {"iterations": SHORT}
    ref = run["localmap"]
    status = datastore.artifacts.inspect(ref)
    assert (ref.kind, status.operation) == ("embedding", "run_localmap")
    assert status.execution_options["nthreads"] == datastore.nthreads
    # A direct call with the same inputs and settings is the run's output, on
    # any number of threads.
    direct = datastore.embeddings.localmap(
        run["connectivity_map"],
        run["embedding_initialization"],
        iterations=(SHORT[0], SHORT[1], SHORT[2]),
        nthreads=1,
    )
    assert direct == ref
    values = _values(datastore, run)
    assert values.shape == (int(run.cells.fetch_all("I").sum()), 2)
    np.testing.assert_array_equal(values, _kernel_layout(datastore, run))


def test_a_localmap_run_freezes_its_layout_as_cell_fields(runs) -> None:
    datastore, run = runs.datastore, runs.run
    values = _values(datastore, run)

    assert run.cells.columns == ("I", "ids", "names", "localmap_1", "localmap_2")
    for index in range(2):
        np.testing.assert_array_equal(
            run.cells.fetch(f"localmap_{index + 1}"), values[:, index]
        )
    # A run plot draws the layout by its output name.
    plot = datastore.plots.embedding(run=run, layout="localmap", show=False)
    try:
        ((points,),) = [axis.collections for axis in plot.axes.values()]
        drawn = np.asarray(points.get_offsets())
    finally:
        plot.close()
    order = np.lexsort(values.T)
    np.testing.assert_allclose(drawn[np.lexsort(drawn.T)], values[order], rtol=1e-6)


def test_a_second_identical_localmap_call_reuses_the_run(runs, tmp_path) -> None:
    keywords = {**FAST, "localmap": {"iterations": SHORT}}

    # The plan of the same call reuses every stage of the run, LocalMAP too.
    plan = read_only_plan(runs.location, "rna", default_assay="RNA", **keywords)
    assert_plan_reuses_run(plan, runs.run)
    location = tmp_path / "data.zarr"
    shutil.copytree(runs.location, location)
    datastore = DataStore(str(location), default_assay="RNA")

    again = datastore.pipeline.rna(**keywords)

    assert dict(again) == dict(runs.run)
    stages = {stage["stage"]: stage for stage in again.report()["stages"]}
    assert [
        (output["outputKey"], output["reused"])
        for output in stages["localmap"]["outputs"]
    ] == [("localmap", True)]
    for field in ("localmap_1", "localmap_2"):
        np.testing.assert_array_equal(
            again.cells.fetch(field), runs.run.cells.fetch(field)
        )


def test_localmap_leaves_a_run_without_it_as_it_was(runs) -> None:
    default = runs.default
    report = default.report()

    # The run records LocalMAP as off and skipped, after UMAP, and has no
    # LocalMAP output or field.
    assert report["run"]["config"]["stages"]["localmap"] is False
    order = report["run"]["stageOrder"]
    assert order[order.index("umap") + 1] == "localmap"
    statuses = {stage["stage"]: stage["status"] for stage in report["stages"]}
    assert statuses["localmap"] == statuses["embedding_initialization"] == "skipped"
    assert list(default) == DEFAULT_OUTPUTS
    assert default.cells.columns == ("I", "ids", "names")
    # Turning LocalMAP on adds its two stages' outputs to the same artifacts.
    assert {key: runs.run[key] for key in default} == dict(default)
    # With the LocalMAP run in the store, a call without the keyword, or with
    # localmap=False, still reuses every artifact of the run without it.
    for keywords in (FAST, {**FAST, "localmap": False}):
        plan = read_only_plan(runs.location, "rna", default_assay="RNA", **keywords)
        assert_plan_reuses_run(plan, default)
