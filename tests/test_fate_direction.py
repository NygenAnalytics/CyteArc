"""Fate maps whose walk a per-cell score directs in place of a pseudotime.

``trajectory.fate(sink_labels=..., graph=..., direction=...)`` biases the walk
on ``graph`` by a score as a pseudotime biases it. The tests compare such maps
with the maps of the pseudotime that the score equals, with absorption
probabilities worked by hand on a path and with a direct solve of the walk,
and check that a score follows the validity rules of a pseudotime: it scores
whole graph components, and sinks outside them are refused.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cytearc.graph.feature_projection import resolve_native_graph_inputs
from cytearc.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from cytearc.metadata.selection import cell_selection_indices
from cytearc.storage.artifact_writer import artifact_transaction
from cytearc.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_stored_arrays,
    list_artifacts,
)
from cytearc.storage.arrays import MISSING_MASK_PREFIX
from cytearc.storage.selections import (
    resolve_metadata_snapshot,
    resolve_stored_selection_artifact,
)
from cytearc.trajectory import FateMappingResult, SourceSinkLabelsError
from cytearc.trajectory.artifacts import FATE_PAYLOAD
from cytearc.trajectory.fate import direction_scores, fate_map_cells
from cytearc.utils.logging import collect_warnings

from .test_fate_mapping import _direct_fate_reference, _ScoredStore, _scored_y_store
from .test_trajectory_membership import _branching
from .test_trajectory_record_paths import (
    _Y_LABELS,
    _empty_graph_payload,
    _path,
    _resize_graph,
    _rewrite_inputs,
    _rewrite_parameters,
    _trajectory_store,
    _y_graph,
)

_SINKS = ["A", "B"]
# The share of an edge's weight that a drop of a quarter of the score's range
# keeps at the default beta of 10, and that a drop of the whole range keeps.
_QUARTER_DROP = 2.0 / (1.0 + np.exp(10.0 * 0.25))
_WHOLE_DROP = 2.0 / (1.0 + np.exp(10.0))
# On the path of five cells below, a score that steps up once between the
# third and fourth cells: the walk from the third cell reaches B with this
# probability (worked out in the path test).
_STEP_TO_B = 1.0 / (1.5 + 0.5 * _WHOLE_DROP)


def _direct(
    scored: _ScoredStore,
    direction: Any,
    *,
    sinks: list[str] = _SINKS,
    **options: Any,
) -> ArtifactRef:
    """Map the fates of the Y store's cells with a walk that ``direction`` directs."""
    return scored.store.trajectory.fate(
        sink_labels=scored.labels,
        graph=scored.graph,
        direction=direction,
        sinks=sinks,
        **options,
    )


def _y_with_island() -> tuple[_ScoredStore, Any]:
    """Return the Y store beside a two-cell island, and its pseudotime of the Y."""
    scored = _scored_y_store(include_disconnected_cells=True)
    return scored, scored.store.trajectory.load_pseudotime(scored.pseudotime)


def _fate_maps(store: Any) -> list[ArtifactRef]:
    return list_artifacts(store.zw, scope="assay", assay="RNA", kind="fate_map")


def _direction_snapshots(store: Any) -> list[ArtifactRef]:
    return [
        ref
        for ref in list_artifacts(store.zw, scope="datastore", kind="metadata_snapshot")
        if store.artifacts.inspect(ref).operation == "snapshot_fate_direction"
    ]


def _other_selection(scored: _ScoredStore) -> ArtifactRef:
    """Return a selection of the same cells as the graph's under another identity."""
    return resolve_stored_selection_artifact(
        scored.store.zw,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="test_other_selection",
        parameters={},
        inputs={},
    )


def _snapshot(
    scored: _ScoredStore,
    values: np.ndarray,
    selection: ArtifactRef,
) -> ArtifactRef:
    """Store one value per cell under ``selection``, as a metadata snapshot."""
    return resolve_metadata_snapshot(
        scored.store.zw,
        values=values,
        row_ids=np.array([f"c{cell}" for cell in range(len(values))]),
        operation="test_scores",
        parameters={},
        inputs={"cell_selection": selection},
        source_columns=["score"],
    )


def _cell_artifact(
    scored: _ScoredStore,
    kind: str,
    values: np.ndarray,
    missing: np.ndarray | None = None,
) -> ArtifactRef:
    """Store one value per cell of the graph's selection, with a mask of the missing ones."""
    planned = plan_cell_data_artifact(
        scored.store.zw,
        scope="datastore",
        kind=kind,
        operation="test_direction_scores",
        parameters={"case": kind},
        inputs={},
        execution_options={},
        cell_selection=scored.store.trajectory.load_pseudotime(
            scored.pseudotime
        ).cell_selection,
        arrays={"values": (values.shape, None)},
    )
    with artifact_transaction(scored.store.zw, planned) as group:
        group.create_array("values", data=values)
        if missing is not None:
            mask_name = f"{MISSING_MASK_PREFIX}values"
            group.create_array(mask_name, data=missing)
            group["values"].attrs["missing_mask"] = mask_name
    return planned.ref


def test_a_score_equal_to_the_pseudotime_directs_the_same_fate_map() -> None:
    scored, pseudotime = _y_with_island()
    store = scored.store
    by_pseudotime = store.trajectory.load_fate(
        store.trajectory.fate(scored.pseudotime, scored.labels, sinks=_SINKS)
    )
    ref = _direct(scored, pseudotime.values)
    by_score = store.trajectory.load_fate(ref)

    # The same walk of the same cells gives the same probabilities, bit for
    # bit, and the island that the pseudotime left unscored stays unmapped.
    np.testing.assert_array_equal(by_score.values, by_pseudotime.values)
    np.testing.assert_array_equal(by_score.valid, pseudotime.valid)
    np.testing.assert_array_equal(by_pseudotime.valid, pseudotime.valid)
    assert by_score.valid.tolist() == [True] * 5 + [False] * 2
    assert (by_score.graph, by_score.cell_selection) == (
        scored.graph,
        pseudotime.cell_selection,
    )
    assert (by_score.pseudotime, by_pseudotime.direction) == (None, None)

    # The map records a snapshot of the score of each of the graph's cells in
    # place of the pseudotime.
    snapshot = by_score.direction
    assert snapshot is not None and snapshot.kind == "metadata_snapshot"
    assert set(store.artifacts.inspect(ref).inputs or {}) == {
        "connectivity_map",
        "direction",
        "sink_labels",
        "cell_selection",
    }
    assert store.artifacts.inspect(snapshot).operation == "snapshot_fate_direction"
    np.testing.assert_array_equal(
        artifact_group(store.zw, snapshot)["values"][:], pseudotime.values
    )
    # The snapshot directs the same map, and the same values reuse both.
    assert _direct(scored, snapshot) == ref
    assert _direct(scored, pseudotime.values.copy()) == ref
    assert _direction_snapshots(store) == [snapshot]

    # The walk's bias rescales the score to [0, 1] over the mapped cells, so
    # an increasing affine map of it directs the same walk.
    shifted = store.trajectory.load_fate(_direct(scored, 3.0 * pseudotime.values + 7.0))
    assert shifted.ref != ref
    np.testing.assert_array_equal(shifted.valid, pseudotime.valid)
    np.testing.assert_allclose(
        shifted.values[:5], by_pseudotime.values[:5], rtol=0.0, atol=1e-6
    )


def test_a_score_directs_the_walk_that_a_direct_solve_predicts() -> None:
    # Branch B runs backward from the root: its cells score lower the farther
    # they lie, so the walk from the root turns away from B.
    scored = _scored_y_store()
    store = scored.store
    graph = store.graph.load(scored.graph, symmetric=True, upper_only=False)
    score = np.array([0.0, 1.0, 2.0, -1.0, -2.0])
    by_pseudotime = store.trajectory.load_fate(
        store.trajectory.fate(scored.pseudotime, scored.labels, sinks=_SINKS)
    )
    root_to_a = {}
    for beta in (10.0, 2.0):
        result = store.trajectory.load_fate(_direct(scored, score, beta=beta))
        np.testing.assert_allclose(
            result.values,
            _direct_fate_reference(graph, score, scored.label_values, _SINKS, beta),
            rtol=0.0,
            atol=1e-6,
        )
        root_to_a[beta] = float(result.values[0, 0])
    # The pseudotime rises along both branches and leaves the root near an
    # even split; the score sends it to A, the more so the stronger the bias.
    assert root_to_a[10.0] > 0.9 > root_to_a[2.0] > 0.6 > by_pseudotime.values[0, 0]


@pytest.mark.parametrize(
    ("score", "to_b"),
    [
        # A walk that steps toward B with probability 1 / (1 + f) and back
        # with f / (1 + f) reaches B from cell i, before A at cell 0, with the
        # gambler's ruin probability (1 - f^i) / (1 - f^4).
        pytest.param(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            (1.0 - _QUARTER_DROP ** np.arange(5)) / (1.0 - _QUARTER_DROP**4),
            id="rising-to-B",
        ),
        pytest.param(
            [4.0, 3.0, 2.0, 1.0, 0.0],
            (_QUARTER_DROP ** (4 - np.arange(5)) - _QUARTER_DROP**4)
            / (1.0 - _QUARTER_DROP**4),
            id="rising-to-A",
        ),
        # Equal scores leave an edge unbiased, so the walk steps either way
        # with probability 1/2 from cells 1 and 2, and only the edge from
        # cell 3 back to cell 2 drops: h1 = h2 / 2, h2 = (h1 + h3) / 2 and
        # h3 = (g h2 + 1) / (1 + g), so h2 = 1 / (1.5 + 0.5 g) and h3 = 1.5 h2.
        pytest.param(
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, _STEP_TO_B / 2.0, _STEP_TO_B, 1.5 * _STEP_TO_B, 1.0],
            id="one-step",
        ),
    ],
)
def test_a_score_biases_backward_edges_as_worked_by_hand(
    score: list[float],
    to_b: Any,
) -> None:
    # A path from sink A to sink B whose edges share one weight.
    trajectory = _trajectory_store(_path(5))
    labels = trajectory.labels(np.array(["A", "x", "x", "x", "B"]))
    result = trajectory.store.trajectory.load_fate(
        trajectory.store.trajectory.fate(
            sink_labels=labels,
            graph=trajectory.graph,
            direction=np.array(score),
            sinks=_SINKS,
            solver_tol=1e-9,
        )
    )
    assert result.valid.all()
    np.testing.assert_allclose(result.values[:, 1], to_b, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.values[:, 0], 1.0 - np.asarray(to_b), atol=1e-6)


@pytest.mark.parametrize("unscored", ["nan", "inf", "-inf", "masked"])
def test_cells_without_a_score_stay_out_like_cells_without_a_pseudotime(
    unscored: str,
) -> None:
    scored, pseudotime = _y_with_island()
    store = scored.store
    island = ~pseudotime.valid
    values = pseudotime.values.copy()
    if unscored == "masked":
        # The mask flags the island's cells, whose stored placeholder would
        # score them if the mask were ignored.
        values[island] = 0.5
        direction: Any = _cell_artifact(scored, "quality_metric", values, island)
    else:
        values[island] = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[unscored]
        direction = values
    by_pseudotime = store.trajectory.load_fate(
        store.trajectory.fate(scored.pseudotime, scored.labels, sinks=_SINKS)
    )
    with collect_warnings() as messages:
        by_score = store.trajectory.load_fate(_direct(scored, direction))
    np.testing.assert_array_equal(by_score.values, by_pseudotime.values)
    np.testing.assert_array_equal(by_score.valid, by_pseudotime.valid)
    # A score of the island would leave a component without a sink, which a
    # warning would name.
    assert messages == []

    # "other" labels only the island's cells, which hold no score.
    with pytest.raises(SourceSinkLabelsError) as raised:
        _direct(scored, direction, sinks=["A", "other"])
    error = raised.value
    assert (error.missing_sources, error.missing_sinks) == ((), ("other",))
    assert (error.missing_from, error.component_sizes) == ("scored component", (5, 2))
    assert str(error).startswith(
        "Sink labels were not found in the cells with a direction score: ['other']"
    )
    assert len(_fate_maps(store)) == 2


def test_a_score_of_a_component_without_a_sink_leaves_it_out_with_a_warning() -> None:
    scored, pseudotime = _y_with_island()
    store = scored.store
    everywhere = np.where(pseudotime.valid, pseudotime.values, 0.25)
    by_pseudotime = store.trajectory.load_fate(
        store.trajectory.fate(scored.pseudotime, scored.labels, sinks=_SINKS)
    )
    refs = []
    for _ in range(2):
        # The map is computed, then reused, with the same warning.
        with collect_warnings() as messages:
            refs.append(_direct(scored, everywhere))
        assert messages == [
            "Fate mapping: the direction scores 2 cells in graph components "
            "without a sink, of sizes [2], which stay out of the fate map"
        ]
    assert refs[1] == refs[0]
    result = store.trajectory.load_fate(refs[0])
    np.testing.assert_array_equal(result.values, by_pseudotime.values)
    np.testing.assert_array_equal(result.valid, pseudotime.valid)

    # With a sink in each component, the map covers both, and a walk reaches
    # only the sinks of its own component: every cell of the Y reaches A.
    both = store.trajectory.load_fate(_direct(scored, everywhere, sinks=["A", "other"]))
    assert both.valid.all()
    np.testing.assert_allclose(
        both.values, [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 2, rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize("cell", [1, 4], ids=["transient-cell", "sink-cell"])
def test_a_score_missing_inside_a_scored_component_is_refused(cell: int) -> None:
    scored, pseudotime = _y_with_island()
    values = pseudotime.values.copy()
    values[cell] = np.nan
    with pytest.raises(
        ValueError,
        match=(
            "Direction scores are missing or not finite for 1 cells of graph "
            "components whose other cells they score"
        ),
    ):
        _direct(scored, values)
    assert _fate_maps(scored.store) == []
    assert _direction_snapshots(scored.store) == []


def _score(scored: _ScoredStore) -> np.ndarray:
    return scored.store.trajectory.load_pseudotime(scored.pseudotime).values


_ARGUMENT_REFUSALS: dict[
    str,
    tuple[Callable[[_ScoredStore], dict[str, Any]], type[Exception], str],
] = {
    "pseudotime-and-direction": (
        lambda scored: {
            "pseudotime": scored.pseudotime,
            "direction": _score(scored),
            "graph": scored.graph,
        },
        ValueError,
        "either a pseudotime or a direction score, not both",
    ),
    "neither": (
        lambda scored: {},
        ValueError,
        "Provide a pseudotime, or a direction score with its graph",
    ),
    "graph-without-direction": (
        lambda scored: {"pseudotime": scored.pseudotime, "graph": scored.graph},
        ValueError,
        "graph applies only with direction",
    ),
    "direction-without-graph": (
        lambda scored: {"direction": _score(scored)},
        ValueError,
        "graph is required with direction",
    ),
    "graph-not-a-reference": (
        lambda scored: {"direction": _score(scored), "graph": "graph"},
        TypeError,
        "graph must be an ArtifactRef",
    ),
    "graph-of-another-kind": (
        lambda scored: {"direction": _score(scored), "graph": scored.labels},
        ValueError,
        "graph must be a connectivity_map or integrated_graph artifact",
    ),
    "direction-list": (
        lambda scored: {"direction": _score(scored).tolist(), "graph": scored.graph},
        TypeError,
        "direction must be an ArtifactRef or a NumPy array",
    ),
    "direction-labels": (
        lambda scored: {"direction": scored.label_values, "graph": scored.graph},
        TypeError,
        "Direction scores must be real numbers",
    ),
    "direction-booleans": (
        lambda scored: {"direction": _score(scored) > 0.5, "graph": scored.graph},
        TypeError,
        "Direction scores must be real numbers",
    ),
    "direction-too-short": (
        lambda scored: {"direction": _score(scored)[:-1], "graph": scored.graph},
        ValueError,
        "one value per cell of the graph, 7 values",
    ),
    "direction-without-a-finite-value": (
        lambda scored: {"direction": np.full(7, np.nan), "graph": scored.graph},
        ValueError,
        "Direction scores hold no finite value",
    ),
    "direction-constant-where-mapped": (
        lambda scored: {
            "direction": np.where(np.isnan(_score(scored)), np.nan, 2.0),
            "graph": scored.graph,
        },
        ValueError,
        "at least two distinct values in the graph components that hold a sink",
    ),
    "pseudotime-as-direction": (
        lambda scored: {"direction": scored.pseudotime, "graph": scored.graph},
        ValueError,
        "A pseudotime directs a fate map as pseudotime",
    ),
    "cluster-labels-as-direction": (
        lambda scored: {
            "direction": _cell_artifact(scored, "cluster_labels", np.arange(7)),
            "graph": scored.graph,
        },
        ValueError,
        "cluster_labels artifacts hold labels",
    ),
    "text-snapshot-as-direction": (
        lambda scored: {"direction": scored.labels, "graph": scored.graph},
        TypeError,
        "Direction scores must be real numbers",
    ),
    "direction-of-another-selection": (
        lambda scored: {
            "direction": _snapshot(scored, _score(scored), _other_selection(scored)),
            "graph": scored.graph,
        },
        ValueError,
        "Direction scores do not match the graph's cell selection",
    ),
    "no-sink-labels": (
        lambda scored: {
            "direction": _score(scored),
            "graph": scored.graph,
            "sink_labels": None,
        },
        ValueError,
        "sink_labels must be provided",
    ),
    "sink-labels-of-another-selection": (
        lambda scored: {
            "direction": _score(scored),
            "graph": scored.graph,
            "sink_labels": _snapshot(
                scored, scored.label_values, _other_selection(scored)
            ),
        },
        ValueError,
        "Sink labels do not match the graph's cell selection",
    ),
}


@pytest.mark.parametrize("case", sorted(_ARGUMENT_REFUSALS))
def test_fate_refuses_each_inconsistent_direction_before_saving_anything(
    case: str,
) -> None:
    scored, _ = _y_with_island()
    arguments, error, message = _ARGUMENT_REFUSALS[case]
    with pytest.raises(error, match=message):
        scored.store.trajectory.fate(
            **{"sink_labels": scored.labels, "sinks": _SINKS, **arguments(scored)}
        )
    assert _fate_maps(scored.store) == []
    assert _direction_snapshots(scored.store) == []


def test_fate_refuses_a_graph_whose_rows_do_not_match_its_cells() -> None:
    # A sixth, isolated graph cell has no row in the five-cell selection.
    larger = _trajectory_store(_y_graph())
    _resize_graph(larger, 6)
    # A graph payload emptied after its lineage was written has no cells left.
    emptied = _trajectory_store(_y_graph())
    _empty_graph_payload(emptied)
    for trajectory, message in (
        (larger, "Graph cell count does not match its stored cell selection"),
        (emptied, "No cells were selected for fate mapping"),
    ):
        with pytest.raises(ValueError, match=message):
            trajectory.store.trajectory.fate(
                sink_labels=trajectory.labels(_Y_LABELS),
                graph=trajectory.graph,
                direction=np.arange(5.0),
                sinks=_SINKS,
            )
        assert trajectory.artifacts("fate_map") == []


def test_a_fate_map_saved_with_the_record_of_a_pseudotime_is_reused() -> None:
    # A map directed by a pseudotime records what maps recorded before a score
    # could direct one, so a map saved with that record, written here by hand,
    # is the one that fate returns.
    scored, pseudotime = _y_with_island()
    store = scored.store
    probabilities = np.array(
        [[0.5, 0.5], [0.9, 0.1], [1.0, 0.0], [0.1, 0.9], [0.0, 1.0]]
        + [[np.nan, np.nan]] * 2,
        dtype=np.float32,
    )
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="fate_map",
        operation="run_fate_mapping",
        parameters={
            "sinks": ["A", "B"],
            "beta": 10.0,
            "solver_tol": 1e-6,
            "max_iterations": 1000,
        },
        inputs={
            "connectivity_map": scored.graph,
            "pseudotime": scored.pseudotime,
            "sink_labels": scored.labels,
        },
        execution_options={"invalidate_cache": False},
        cell_selection=pseudotime.cell_selection,
        arrays={"probabilities": ((7, 2), "f"), "valid": ((7,), "b")},
    )
    write_cell_data_artifact(
        store.zw,
        planned,
        {"probabilities": probabilities, "valid": pseudotime.valid},
        fingerprint_payload=True,
    )

    assert store.trajectory.fate(scored.pseudotime, scored.labels, sinks=_SINKS) == (
        planned.ref
    )
    loaded = store.trajectory.load_fate(planned.ref)
    np.testing.assert_array_equal(loaded.values, probabilities)
    assert (loaded.pseudotime, loaded.direction) == (scored.pseudotime, None)
    assert _fate_maps(store) == [planned.ref]


def _directed_map(sinks: list[str] = _SINKS) -> tuple[_ScoredStore, ArtifactRef]:
    """Return the Y store with the island and a map directed by a score of all cells."""
    scored, pseudotime = _y_with_island()
    everywhere = np.where(pseudotime.valid, pseudotime.values, 0.25)
    return scored, _direct(scored, everywhere, sinks=sinks)


def test_the_loader_returns_a_map_directed_by_a_score() -> None:
    scored, ref = _directed_map(["A", "other"])
    result = scored.store.trajectory.load_fate(ref)
    assert isinstance(result, FateMappingResult)
    assert result.ref == ref and result.graph == scored.graph
    assert result.pseudotime is None and result.direction is not None
    assert result.sink_labels == ("A", "other")
    assert result.valid.all()


def _hide_the_island(scored: _ScoredStore, ref: ArtifactRef) -> None:
    """Mark the island unmapped in a payload that stays internally consistent."""
    group = artifact_group(scored.store.zw, ref)
    probabilities = np.asarray(group["probabilities"][:])
    probabilities[5:] = np.nan
    group["probabilities"][:] = probabilities
    valid = np.asarray(group["valid"][:])
    valid[5:] = False
    group["valid"][:] = valid
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, FATE_PAYLOAD)


def _rescore(scored: _ScoredStore, ref: ArtifactRef, values: np.ndarray) -> None:
    selection = scored.store.trajectory.load_pseudotime(
        scored.pseudotime
    ).cell_selection
    _rewrite_inputs(scored.store, ref, direction=_snapshot(scored, values, selection))


_ISLAND_ONLY = np.array([np.nan] * 5 + [0.0, 1.0])
_SPLIT_Y = np.array([0.0, np.nan, 1.0, 0.5, 1.0, 0.0, 1.0])

_DIRECTED_RECORD_TAMPERS: dict[
    str,
    tuple[Callable[[_ScoredStore, ArtifactRef], None], str],
] = {
    "pseudotime-beside-the-score": (
        lambda scored, ref: _rewrite_inputs(
            scored.store, ref, pseudotime=scored.pseudotime
        ),
        "Fate-map inputs does not match its persisted contract",
    ),
    "pseudotime-as-the-score": (
        lambda scored, ref: _rewrite_inputs(
            scored.store, ref, direction=scored.pseudotime
        ),
        "A pseudotime directs a fate map as pseudotime",
    ),
    "score-of-another-selection": (
        lambda scored, ref: _rewrite_inputs(
            scored.store,
            ref,
            direction=_snapshot(scored, _score(scored), _other_selection(scored)),
        ),
        "Direction scores do not match the graph's cell selection",
    ),
    "selection-of-another-identity": (
        lambda scored, ref: _rewrite_inputs(
            scored.store, ref, cell_selection=_other_selection(scored)
        ),
        "Fate-map graph and cell selection do not match",
    ),
    "score-missing-inside-a-component": (
        lambda scored, ref: _rescore(scored, ref, _SPLIT_Y),
        "Direction scores are missing or not finite for 1 cells",
    ),
    "sinks-without-a-score": (
        lambda scored, ref: _rescore(scored, ref, _ISLAND_ONLY),
        "Fate-map sink labels are malformed",
    ),
    "unknown-sink": (
        lambda scored, ref: _rewrite_parameters(scored.store, ref, sinks=["A", "Z"]),
        "Fate-map sink labels are malformed",
    ),
    "island-left-out": (
        _hide_the_island,
        "Fate-map artifact payload is invalid",
    ),
}


@pytest.mark.parametrize("tamper", sorted(_DIRECTED_RECORD_TAMPERS))
def test_the_loader_rejects_each_tampered_map_directed_by_a_score(tamper: str) -> None:
    # Sinks in both components: the island belongs to the map.
    scored, ref = _directed_map(["A", "other"])
    change, message = _DIRECTED_RECORD_TAMPERS[tamper]
    change(scored, ref)

    with pytest.raises(ValueError, match=message):
        scored.store.trajectory.load_fate(ref)


def test_a_fate_map_names_exactly_one_direction() -> None:
    scored, pseudotime = _y_with_island()
    arguments: tuple[Any, ...] = (
        ArtifactRef(scope="assay", assay="RNA", kind="fate_map", artifact_id="8" * 64),
        scored.graph,
    )
    rest = (
        scored.labels,
        pseudotime.cell_selection,
        ("A",),
        np.ones((7, 1)),
        np.ones(7, dtype=bool),
    )
    with pytest.raises(ValueError, match="either a pseudotime or a direction score"):
        FateMappingResult(*arguments, None, *rest)
    with pytest.raises(ValueError, match="either a pseudotime or a direction score"):
        FateMappingResult(*arguments, scored.pseudotime, *rest, scored.labels)
    assert FateMappingResult(*arguments, None, *rest, scored.labels).direction == (
        scored.labels
    )


def test_lineage_curves_order_the_cells_of_a_map_by_its_score() -> None:
    scored, pseudotime = _y_with_island()
    store = scored.store
    # The coordinates that the graph was built on, laid out as a Y and an
    # island apart from it.
    coordinates = resolve_native_graph_inputs(store.zw, scored.graph).coordinates
    artifact_group(store.zw, coordinates)["data"][:] = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.0, -1.0], [2.0, -2.0]]
        + [[5.0, 5.0], [5.0, 6.0]]
    )
    by_pseudotime = store.trajectory.load_curves(
        store.trajectory.curves(
            store.trajectory.fate(scored.pseudotime, scored.labels, sinks=_SINKS),
            n_points=8,
            df=3.0,
        )
    )
    by_score = store.trajectory.load_curves(
        store.trajectory.curves(
            _direct(scored, 3.0 * pseudotime.values + 7.0), n_points=8, df=3.0
        )
    )
    # Curves follow the cells' ranks along each lineage, which an increasing
    # map of the score keeps, and the fitted ordering of each curve point is
    # in the score's units.
    np.testing.assert_array_equal(by_score.valid, by_pseudotime.valid)
    np.testing.assert_allclose(by_score.points, by_pseudotime.points, atol=1e-5)
    np.testing.assert_allclose(by_score.progress, by_pseudotime.progress, atol=1e-5)
    np.testing.assert_allclose(
        by_score.curve_pseudotime,
        3.0 * by_pseudotime.curve_pseudotime + 7.0,
        rtol=0.0,
        atol=1e-5,
    )


def test_lineage_membership_of_a_map_directed_by_a_score_parts_lineages_on_it() -> None:
    # The branching trajectory of the membership tests, its fate map directed
    # by three times its pseudotime plus seven, which lies above 7 where a
    # pseudotime lies in [0, 1].
    branching = _branching()
    store = branching.store
    score = 3.0 * store.trajectory.load_pseudotime(branching.pseudotime).values + 7.0
    fate = store.trajectory.fate(
        sink_labels=branching.label_artifact(),
        graph=branching.graph,
        direction=score,
        sinks=["a2", "b2"],
    )
    membership = store.trajectory.lineage_membership(
        store.trajectory.curves(fate, n_points=40, df=6.0)
    )
    result = store.trajectory.load_lineage_membership(membership)

    # By hand, on the score: the split with the fewest shared cells after it
    # and parted cells before it, over every score between two of the pair's
    # cells.
    (row,) = result.branch_points.itertuples(index=False)
    pair = result.members.any(axis=1)
    shared = result.members.all(axis=1)
    assert shared.any() and (pair & ~shared).any()
    candidates = np.unique(score[pair])
    best, best_errors = math.nan, math.inf
    for before, after in zip(candidates[:-1], candidates[1:], strict=True):
        split = 0.5 * (before + after)
        errors = np.sum(shared & (score > split)) + np.sum(
            pair & ~shared & (score < split)
        )
        if errors < best_errors:
            best, best_errors = split, errors
    assert row.pseudotime == pytest.approx(best)
    assert row.pseudotime > 7.0

    # The cells around the branch point are the nearest on the score.
    selection = store.trajectory.branch_cells(membership, ["a2", "b2"], window=0.25)
    rows = cell_selection_indices(store.zw, branching.selection)
    selected = np.isin(rows, cell_selection_indices(store.zw, selection))
    nearest = np.abs(score - row.pseudotime)
    assert selected.sum() == math.ceil(0.25 * pair.sum())
    assert not np.any(selected & ~pair)
    assert nearest[selected].max() <= nearest[pair & ~selected].min()


def test_direction_scores_keep_the_finite_values_of_whole_components() -> None:
    component = np.array([0, 0, 1, 1, 2])
    # Integers are real values, kept exactly as float64.
    scores = direction_scores(
        np.array([3, 1, 2, 5, 4], dtype=np.uint8), None, component
    )
    assert scores.dtype == np.float64
    assert scores.tolist() == [3.0, 1.0, 2.0, 5.0, 4.0]
    # A missing value, NaN or infinity leaves a cell unscored, and component 1
    # loses both of its cells, one to the mask and one to infinity.
    values = np.array([0.5, -1.0, 7.0, np.inf, np.nan])
    missing = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(
        direction_scores(values, missing, component),
        [0.5, -1.0, np.nan, np.nan, np.nan],
    )
    # The caller's values stay as they were.
    assert values[2] == 7.0

    with pytest.raises(ValueError, match="for 1 cells of graph components"):
        direction_scores(np.array([0.5, -1.0, 7.0, np.nan, 1.0]), None, component)
    with pytest.raises(ValueError, match="hold no finite value"):
        direction_scores(np.array([1.0] * 5), np.ones(5, dtype=bool), component)
    with pytest.raises(ValueError, match=r"5 values, not an array of shape \(5, 1\)"):
        direction_scores(np.ones((5, 1)), None, component)
    with pytest.raises(TypeError, match="not values of dtype bool"):
        direction_scores(np.ones(5, dtype=bool), None, component)
    with pytest.raises(TypeError, match="not values of dtype object"):
        direction_scores(np.array([1.0] * 5, dtype=object), None, component)


def test_a_fate_map_covers_the_scored_components_that_hold_a_sink() -> None:
    # Component 0 is scored and holds a sink; component 1 holds a sink but no
    # score; component 2 holds a score but no sink.
    scores = np.array([0.0, 1.0, np.nan, np.nan, 0.5, 0.25])
    component = np.array([0, 0, 1, 1, 2, 2])
    sink_cells = np.array([True, False, True, False, False, False])
    assert fate_map_cells(scores, component, sink_cells).tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    # A sink in component 2 adds its cells, whose scores join the range.
    sink_cells[5] = True
    assert fate_map_cells(scores, component, sink_cells).tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
    ]

    with pytest.raises(ValueError, match="No scored cell holds a sink label"):
        fate_map_cells(scores, component, np.array([False, False, True] + [False] * 3))
    # The covered cells of component 0 share one score, which the walk's
    # rescaling to [0, 1] cannot take.
    with pytest.raises(ValueError, match="at least two distinct values"):
        fate_map_cells(
            np.array([2.0, 2.0, np.nan, np.nan, 0.5, 0.25]),
            component,
            np.array([True] + [False] * 5),
        )
