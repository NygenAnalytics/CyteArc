"""E-distances between the perturbed and control cells of a small screen.

The module writes the screen, its labels and its coordinates once, as a
template. A test that writes copies the template, and the tests of refused
calls share one copy, which they check is left as it was.
"""

import threading
import tracemalloc
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist
from scipy.stats import false_discovery_control, kstest

import cytearc.perturbation.edistance as edistance_module
from cytearc.datastore.datastore import DataStore
from cytearc.perturbation import PerturbationEDistance
from cytearc.perturbation.edistance import (
    edistance_arrays,
    edistance_from_sums,
    edistance_memory_bytes,
)
from cytearc.storage.artifacts import inspect_artifact
from cytearc.utils.logging import logger
from tests.perturbation_helpers import (
    copy_store,
    fixed_labels,
    open_store,
    use_python_kernels,
    write_screen,
    written,
)

# Guides and their targets; no cell carries GENE_D's guide.
TARGETS = ["GENE_A", "GENE_A", "GENE_B", "Non-Targeting", "GENE_C", "GENE_D"]
# The called guides of each cell.
CARRIED = [
    *[(0,)] * 3 + [(1,)],  # GENE_A, cells 0 to 3
    *[(2,)] * 3,  # GENE_B, cells 4 to 6
    *[(3,)] * 6,  # controls, cells 7 to 12
    (4,),  # GENE_C, cell 13
    (),  # no guide, cell 14
    (0, 2),  # GENE_A and GENE_B, cell 15
    (0, 3),  # GENE_A with a control guide, cell 16
]
GENE_A = [0, 1, 2, 3, 16]
GENE_B = [4, 5, 6]
CONTROLS = [7, 8, 9, 10, 11, 12]


def _coordinates() -> np.ndarray:
    values = np.random.default_rng(3).normal(size=(len(CARRIED), 3))
    values[GENE_A, 0] += 8.0  # GENE_A moves its cells far from the controls
    return values


COORDINATES = _coordinates()


def _coordinate_ref(store: DataStore, cells=None):
    """Import COORDINATES for ``cells``, every cell by default."""
    cells = cells or store.select_measured_cells("RNA")
    selected = np.asarray(store.artifacts.load(cells)["values"][:], dtype=bool)
    return store.reduction.import_coordinates(
        COORDINATES[selected], cell_selection=cells, name="X_test"
    )


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Iterator[SimpleNamespace]:
    """The screen with its labels and coordinates, and labels without controls."""
    guides = np.zeros((len(CARRIED), len(TARGETS)), dtype=np.uint16)
    for cell, carried in enumerate(CARRIED):
        guides[cell, list(carried)] = 20
    rna = np.random.default_rng(5).poisson(6.0, size=(len(CARRIED), 4))
    path = tmp_path_factory.mktemp("edistance") / "store.zarr"
    store = write_screen(path, rna, guides, targets=TARGETS)
    screen = SimpleNamespace(
        path=path,
        labels=fixed_labels(store, ["Non-Targeting"]),
        coordinates=_coordinate_ref(store),
        # Labels whose controls are GENE_D, which no cell carries.
        without_controls=fixed_labels(store, ["GENE_D"]),
    )
    saved = written(store)
    yield screen
    assert written(open_store(path)) == saved


@pytest.fixture
def store(template, tmp_path) -> DataStore:
    """A copy of the template, for a test that writes."""
    return copy_store(template.path, tmp_path)


@pytest.fixture(scope="module")
def shared_store(template, tmp_path_factory) -> DataStore:
    """One writable copy of the template for the calls that must be refused."""
    return copy_store(template.path, tmp_path_factory.mktemp("edistance_refused"))


def _brute(x: np.ndarray, y: np.ndarray) -> float:
    """The E-distance from every pair of cells."""
    return float(
        2 * cdist(x, y, "sqeuclidean").mean()
        - pdist(x, "sqeuclidean").mean()
        - pdist(y, "sqeuclidean").mean()
    )


def _replayed_p_value(
    perturbed: np.ndarray, controls: np.ndarray, n_permutations: int, seed: list[int]
) -> float:
    """The permutation p-value from brute-force distances of the same draws."""
    pool = np.concatenate([controls, perturbed]) - controls.mean(axis=0)
    n_pool, n_p = len(pool), len(perturbed)
    n_drawn = min(n_p, len(controls))
    observed = _brute(perturbed, controls)
    margin = 1e-9 * float((pool**2).sum()) / n_pool
    reached = 0
    for uniforms in np.random.default_rng(seed).random((n_permutations, n_drawn)):
        chosen: list[int] = []
        for k, uniform in enumerate(uniforms):
            last = n_pool - n_drawn + k
            position = min(int(uniform * (last + 1)), last)
            chosen.append(last if position in chosen else position)
        drawn = np.zeros(n_pool, dtype=bool)
        drawn[chosen] = True
        groups = (pool[drawn], pool[~drawn])
        permuted = _brute(*groups) if n_drawn == n_p else _brute(*groups[::-1])
        reached += permuted >= observed - margin
    return (reached + 1) / (n_permutations + 1)


def test_edistance_measures_each_target_against_the_controls(store, template) -> None:
    labels = template.labels
    coordinates = template.coordinates

    ref = store.perturbation.edistance(
        labels, coordinates, n_permutations=200, random_seed=7
    )

    assert ref.kind == "perturbation_edistance"
    assert ref.assay == "RNA"
    loaded = store.perturbation.load_edistance(ref)
    assert isinstance(loaded, PerturbationEDistance)
    assert (loaded.by, loaded.n_permutations, loaded.random_seed) == ("target", 200, 7)
    assert "guide" not in loaded.table
    table = loaded.table.set_index("target")
    assert table.index.tolist() == ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
    assert table["status"].tolist() == [
        "tested",
        "tested",
        "too_few_cells",
        "too_few_cells",
    ]
    # Cell 16 carries a control guide besides GENE_A's, so it counts for
    # GENE_A; cell 15 carries two targets and counts for neither.
    assert table["n_cells"].tolist() == [5, 3, 1, 0]
    assert table["n_control_cells"].tolist() == [6] * 4
    values = np.asarray(store.artifacts.load(coordinates)["data"][:])
    np.testing.assert_allclose(
        table.loc[["GENE_A", "GENE_B"], "edistance"],
        [
            _brute(values[GENE_A], values[CONTROLS]),
            _brute(values[GENE_B], values[CONTROLS]),
        ],
    )
    # No permutation of the pool reaches GENE_A's distance.
    assert table.loc["GENE_A", "p_value"] == 1 / 201
    groups = np.full(len(CARRIED), -1)
    groups[GENE_A], groups[GENE_B], groups[13] = 0, 1, 2
    direct = edistance_arrays(
        values,
        groups,
        np.isin(np.arange(len(CARRIED)), CONTROLS),
        4,
        n_permutations=200,
        random_seed=7,
    )
    np.testing.assert_array_equal(table["p_value"], direct.p_value)
    tested = table.loc[["GENE_A", "GENE_B"]]
    np.testing.assert_allclose(
        tested["adjusted_p_value"],
        false_discovery_control(tested["p_value"], method="bh"),
    )
    untested = table.loc[["GENE_C", "GENE_D"], ["edistance", "p_value"]]
    assert untested.isna().all().all()
    assert table.loc[["GENE_C", "GENE_D"], "adjusted_p_value"].isna().all()


def test_edistance_by_guide_takes_each_cell_with_one_guide(store, template) -> None:
    labels = template.labels
    coordinates = template.coordinates
    ref = store.perturbation.edistance(labels, coordinates, by="guide")
    loaded = store.perturbation.load_edistance(ref)
    assert (loaded.by, loaded.n_permutations, loaded.random_seed) == ("guide", 1000, 0)
    table = loaded.table.set_index("guide")
    assert table.index.tolist() == [f"CRISPR{index}" for index in (0, 1, 2, 4, 5)]
    assert table["target"].tolist() == [
        "GENE_A",
        "GENE_A",
        "GENE_B",
        "GENE_C",
        "GENE_D",
    ]
    assert table["n_cells"].tolist() == [3, 1, 3, 1, 0]
    values = np.asarray(store.artifacts.load(coordinates)["data"][:])
    np.testing.assert_allclose(
        table.loc["CRISPR0", "edistance"], _brute(values[:3], values[CONTROLS])
    )
    # Nine cells hold only 84 groups of three, so even CRISPR0's far cells
    # reach a p-value near 1 / 84.
    assert table.loc["CRISPR0", "p_value"] == _replayed_p_value(
        values[:3], values[CONTROLS], 1000, [0, 0]
    )
    assert table.loc["CRISPR0", "p_value"] < 2 / 84


def test_cells_without_coordinates_are_left_out(store, template) -> None:
    labels = template.labels
    # Coordinates of every cell but the GENE_C and unlabelled ones.
    cells = store.select_cells(
        labels, include=["GENE_A", "GENE_B", "control", "Multiple"]
    )
    coordinates = _coordinate_ref(store, cells)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        ref = store.perturbation.edistance(labels, coordinates, n_permutations=0)
    finally:
        logger.remove(sink)
    assert "1 of 15 compared cells have no coordinates and are left out" in messages
    table = store.perturbation.load_edistance(ref).table.set_index("target")
    assert table["n_cells"].tolist() == [5, 3, 0, 0]
    # Without permutations, no row has a p-value.
    assert table[["p_value", "adjusted_p_value"]].isna().all().all()
    values = np.asarray(store.artifacts.load(coordinates)["data"][:])
    # The coordinates hold cells 0 to 12, 15 and 16, so GENE_A's last cell is
    # row 14.
    np.testing.assert_allclose(
        table.loc["GENE_A", "edistance"],
        _brute(values[[0, 1, 2, 3, 14]], values[CONTROLS]),
    )
    without_controls = store.select_cells(labels, include=["GENE_A", "GENE_B"])
    with pytest.raises(ValueError, match="0 of the labels' control cells have them"):
        store.perturbation.edistance(labels, _coordinate_ref(store, without_controls))


def test_edistance_is_reused_and_follows_its_inputs(store, template) -> None:
    labels = template.labels
    coordinates = template.coordinates
    ref = store.perturbation.edistance(labels, coordinates, n_permutations=50)
    assert store.perturbation.edistance(labels, coordinates, n_permutations=50) == ref
    status = inspect_artifact(store.zw, ref)
    assert status.parameters == {
        "by": "target",
        "n_permutations": 50,
        "random_seed": 0,
    }
    assert set(status.inputs) == {"labels", "coordinates"}
    reseeded = store.perturbation.edistance(
        labels, coordinates, n_permutations=50, random_seed=1
    )
    assert reseeded != ref
    again = store.perturbation.edistance(
        labels, coordinates, n_permutations=50, invalidate_cache=True
    )
    assert again != ref
    reader = open_store(store.zarr_loc, zarr_mode="r")
    assert reader.perturbation.edistance(labels, coordinates, n_permutations=50) in (
        ref,
        again,
    )
    with pytest.raises(PermissionError, match="measure_edistance"):
        reader.perturbation.edistance(labels, coordinates, by="guide")


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"by": "gene"}, ValueError, "by must be 'target' or 'guide'"),
        ({"n_permutations": True}, TypeError, "n_permutations must be an integer"),
        ({"n_permutations": -1}, ValueError, "n_permutations must be at least 0"),
        (
            {"n_permutations": 1_000_001},
            ValueError,
            "n_permutations must be at most 1000000",
        ),
        ({"random_seed": -1}, ValueError, "random_seed must be at least 0"),
        ({"invalidate_cache": "no"}, TypeError, "invalidate_cache must be a boolean"),
    ],
)
def test_edistance_refuses_arguments_it_cannot_use(
    shared_store, template, arguments, error, message
) -> None:
    saved = written(shared_store)
    with pytest.raises(error, match=message):
        shared_store.perturbation.edistance(
            template.labels, template.coordinates, **arguments
        )
    assert written(shared_store) == saved


def test_edistance_refuses_inputs_it_cannot_measure(
    shared_store, template, monkeypatch
) -> None:
    store = shared_store
    labels = template.labels
    coordinates = template.coordinates
    saved = written(store)
    cells = store.select_measured_cells("CRISPR")
    with pytest.raises(TypeError, match="labels must be an ArtifactRef"):
        store.perturbation.edistance("labels", coordinates)
    with pytest.raises(ValueError, match="not 'cell_selection'"):
        store.perturbation.edistance(cells, coordinates)
    with pytest.raises(TypeError, match="coordinates must be an ArtifactRef"):
        store.perturbation.edistance(labels, "X_pca")
    with pytest.raises(ValueError, match="normalized artifact, not 'cell_selection'"):
        store.perturbation.edistance(labels, cells)
    # Labels whose controls are GENE_D, which no cell carries, have no control cell.
    with pytest.raises(
        ValueError, match="No labelled cell carries control guides only"
    ):
        store.perturbation.edistance(template.without_controls, coordinates)
    monkeypatch.setattr(store, "memoryBytes", 100)
    with pytest.raises(MemoryError, match="E-distances of 15 cells in 3 dimensions"):
        store.perturbation.edistance(labels, coordinates)
    assert written(store) == saved


def test_loading_edistance_refuses_other_artifacts(shared_store, template) -> None:
    saved = written(shared_store)
    with pytest.raises(TypeError, match="edistance must be an ArtifactRef"):
        shared_store.perturbation.load_edistance("edistance")
    with pytest.raises(ValueError, match="not 'perturbation_labels'"):
        shared_store.perturbation.load_edistance(template.labels)
    assert written(shared_store) == saved


def _screen(
    seed: int, sizes: list[int], n_control: int, n_dims: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of control cells and of one group per size, shuffled."""
    rng = np.random.default_rng(seed)
    groups = np.concatenate(
        [np.full(n_control, -1)]
        + [np.full(size, row) for row, size in enumerate(sizes)]
    )
    values = rng.normal(size=(len(groups), n_dims))
    order = rng.permutation(len(groups))
    return values[order], groups[order], groups[order] < 0


@pytest.mark.parametrize("python_kernels", [False, True])
def test_edistances_and_p_values_equal_brute_force(monkeypatch, python_kernels) -> None:
    if python_kernels:
        use_python_kernels(monkeypatch, edistance_module)
    # Row 2 has more cells than the controls, so its draws pick the controls.
    sizes = [5, 2, 30, 1, 0]
    values, groups, control = _screen(1, sizes, 20)
    values[groups == 0] += 1.5
    arrays = edistance_arrays(
        values, groups, control, len(sizes), n_permutations=150, random_seed=11
    )
    assert arrays.status.tolist() == [0, 0, 0, 1, 1]
    assert arrays.n_cells.tolist() == sizes
    assert arrays.n_control_cells.tolist() == [20] * 5
    for row in range(3):
        perturbed, controls = values[groups == row], values[control]
        assert arrays.edistance[row] == pytest.approx(_brute(perturbed, controls))
        assert arrays.p_value[row] == _replayed_p_value(
            perturbed, controls, 150, [11, row]
        )
    assert np.isnan(arrays.edistance[3:]).all()
    threaded = edistance_arrays(
        values,
        groups,
        control,
        len(sizes),
        n_permutations=150,
        random_seed=11,
        nthreads=2,
    )
    np.testing.assert_array_equal(threaded.p_value, arrays.p_value)
    np.testing.assert_array_equal(threaded.edistance, arrays.edistance)


def test_permutation_draws_do_not_depend_on_the_block(monkeypatch) -> None:
    values, groups, control = _screen(2, [6, 9], 12)
    expected = edistance_arrays(
        values, groups, control, 2, n_permutations=100, random_seed=3
    )
    # Blocks of one permutation draw the same numbers in more calls.
    monkeypatch.setattr(edistance_module, "_DRAW_BYTES", 8)
    blocked = edistance_arrays(
        values, groups, control, 2, n_permutations=100, random_seed=3
    )
    np.testing.assert_array_equal(blocked.p_value, expected.p_value)


def test_groups_like_the_controls_give_uniform_p_values() -> None:
    sizes = [4, 8, 16] * 40
    values, groups, control = _screen(4, sizes, 60, n_dims=3)
    arrays = edistance_arrays(
        values, groups, control, len(sizes), n_permutations=99, random_seed=5
    )
    assert kstest(arrays.p_value, "uniform").pvalue > 0.01


def test_equal_cells_tie_in_every_permutation() -> None:
    values = np.tile([[0.1, 0.3, 0.7]], (12, 1))
    groups = np.r_[np.zeros(4, dtype=int), np.full(8, -1)]
    arrays = edistance_arrays(
        values, groups, groups < 0, 1, n_permutations=40, random_seed=0
    )
    assert arrays.edistance[0] == pytest.approx(0.0, abs=1e-30)
    assert arrays.p_value.tolist() == [1.0]


def test_edistance_arrays_check_their_inputs() -> None:
    values = np.zeros((4, 2))
    controls = np.asarray([False, False, True, True])
    with pytest.raises(ValueError, match="cells x dimensions"):
        edistance_arrays(
            np.zeros(4), np.zeros(4), controls, 1, n_permutations=0, random_seed=0
        )
    with pytest.raises(ValueError, match="one control flag per cell"):
        edistance_arrays(
            values, np.zeros(3), controls, 1, n_permutations=0, random_seed=0
        )
    with pytest.raises(ValueError, match="rows below n_rows"):
        edistance_arrays(
            values, [1, 1, -1, -1], controls, 1, n_permutations=0, random_seed=0
        )
    with pytest.raises(ValueError, match="control cells carry no row"):
        edistance_arrays(
            values, [0, 0, 0, -1], controls, 1, n_permutations=0, random_seed=0
        )
    with pytest.raises(ValueError, match="at least two control cells"):
        edistance_arrays(
            values,
            [0, 0, 0, -1],
            [False, False, False, True],
            1,
            n_permutations=0,
            random_seed=0,
        )


def test_edistance_from_sums_equals_the_mean_distances() -> None:
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(9, 3)) + 1.0
    distance = edistance_from_sums.py_func(
        x.sum(axis=0), float((x**2).sum()), 7, y.sum(axis=0), float((y**2).sum()), 9
    )
    assert distance == pytest.approx(_brute(x, y))


# The Python function of the kernel, also when Numba compiles nothing.
_DISTANCE_FROM_SUMS = getattr(edistance_from_sums, "py_func", edistance_from_sums)


class _Met(Exception):
    """Every thread has noted what it holds, and the run ends."""


def _meet_in_first_permutations(
    monkeypatch, nthreads: int, *, stop: bool = False
) -> list[int]:
    """Make the threads meet in their first permutations, and note what they hold.

    The Python kernels look up ``edistance_from_sums`` when they run. A
    thread's first call measures its first row, and its second comes from
    the kernel's first permutation, once the kernel has allocated its arrays.
    There each thread waits until every thread holds its block and arrays,
    and the traced bytes are noted before any thread goes on, or, with
    ``stop``, ends the run with :class:`_Met`.
    """
    barrier = threading.Barrier(nthreads, timeout=60)
    calls = threading.local()
    held: list[int] = []

    def distance_together(*args):
        calls.n = getattr(calls, "n", 0) + 1
        if calls.n == 2:
            barrier.wait()
            held.append(tracemalloc.get_traced_memory()[0])
            barrier.wait()
            if stop:
                raise _Met
        return _DISTANCE_FROM_SUMS(*args)

    monkeypatch.setattr(edistance_module, "edistance_from_sums", distance_together)
    return held


@pytest.mark.parametrize(
    ("n_control", "sizes", "n_dims", "n_permutations", "nthreads", "draw_bytes"),
    [
        # Many cells, whose coordinates and indexes take most of the memory.
        (2_000, [3_000] * 6, 8, 1, 1, 1 << 16),
        (2_000, [3_000] * 6, 8, 1, 2, 1 << 16),
        # Few cells: each row draws its 81 permutations of 100 cells in one
        # block of uniform numbers, which takes most of the memory.
        (300, [100, 100], 4, 81, 1, 1 << 16),
        # Eight threads draw such blocks at once.
        (300, [100] * 8, 1, 81, 8, 1 << 16),
        # A row draws two blocks of 163 permutations in turn.
        (300, [100], 1, 326, 1, 1 << 17),
        # Nearly every cell is perturbed, so that sorting the cells by row
        # holds the most positions.
        (2, [199_998], 1, 3, 1, 1 << 16),
        # Thousands of rows of two cells, tested on two threads.
        (1_000, [2] * 2_000, 2, 1, 2, 1 << 16),
        # Thousands of rows without a cell.
        (1_000, [2] * 1_000 + [0] * 20_000, 2, 1, 1, 1 << 16),
    ],
)
def test_the_memory_estimate_bounds_the_edistances_peak(
    monkeypatch, n_control, sizes, n_dims, n_permutations, nthreads, draw_bytes
) -> None:
    # Blocks of at most 128 kB instead of 8 MB keep the screens small, and
    # the kernels run as Python, so that tracemalloc sees their arrays.
    monkeypatch.setattr(edistance_module, "_DRAW_BYTES", draw_bytes)
    use_python_kernels(monkeypatch, edistance_module)
    # The first call imports what the tests use.
    edistance_arrays(*_screen(5, [3, 3], 10), 2, n_permutations=5, random_seed=0)
    _meet_in_first_permutations(monkeypatch, nthreads)
    values, groups, control = _screen(6, sizes, n_control, n_dims=n_dims)
    tracemalloc.start()
    try:
        # The estimate counts the arrays that the operation passes.
        edistance_arrays(
            values.copy(),
            groups.copy(),
            control.copy(),
            len(sizes),
            n_permutations=n_permutations,
            random_seed=0,
            nthreads=nthreads,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    estimate = edistance_memory_bytes(
        len(values), n_dims, n_control, n_rows=len(sizes), nthreads=nthreads
    )
    assert peak <= estimate


def test_each_thread_holds_no_more_than_the_estimate_adds_for_it(monkeypatch) -> None:
    # Each of eight rows permutes its 10,000 cells with 10,000 control cells,
    # so that a block holds one permutation of 10,000 cells and each kernel
    # flags 20,000 cells and holds the positions of 10,000.
    monkeypatch.setattr(edistance_module, "_DRAW_BYTES", 1 << 16)
    use_python_kernels(monkeypatch, edistance_module)
    sizes = [10_000] * 8
    values, groups, control = _screen(7, sizes, 10_000, n_dims=1)
    edistance_arrays(*_screen(5, [3, 3], 10), 2, n_permutations=5, random_seed=0)
    held = {}
    for nthreads in (1, 8):
        noted = _meet_in_first_permutations(monkeypatch, nthreads, stop=True)
        tracemalloc.start()
        try:
            with pytest.raises(_Met):
                edistance_arrays(
                    values,
                    groups,
                    control,
                    len(sizes),
                    n_permutations=1,
                    random_seed=0,
                    nthreads=nthreads,
                )
        finally:
            tracemalloc.stop()
        held[nthreads] = max(noted)

    def estimate(nthreads: int) -> int:
        return edistance_memory_bytes(
            len(values), 1, 10_000, n_rows=len(sizes), nthreads=nthreads
        )

    # Seven more threads hold seven more blocks, kernel arrays and threads.
    assert held[8] - held[1] <= estimate(8) - estimate(1)
