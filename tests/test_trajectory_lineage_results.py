import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from cytearc.storage.refs import ArtifactRef
from cytearc.trajectory.curves import curve_values_at, project_onto_curve
from cytearc.trajectory.results import (
    CurveProjection,
    LineageAggregationResult,
    LineageCurvesResult,
    LineageDriverResult,
    LineageMarkerResult,
    LineageTreeResult,
)


def _ref(kind: str, digit: str, assay: str | None = "RNA") -> ArtifactRef:
    if assay is None:
        return ArtifactRef(scope="datastore", kind=kind, artifact_id=digit * 64)
    return ArtifactRef(scope="assay", kind=kind, artifact_id=digit * 64, assay=assay)


_FEATURES = _ref("feature_selection", "f")
_CELLS = _ref("cell_selection", "d", assay=None)
_FATE = _ref("fate_map", "8")
_COORDINATES = _ref("reduction", "c")

_FOREIGN_FEATURES = [
    pytest.param(_ref("cell_selection", "f"), id="wrong kind"),
    pytest.param(_ref("feature_selection", "f", assay=None), id="datastore scope"),
    pytest.param(_ref("feature_selection", "f", assay="ATAC"), id="other assay"),
    pytest.param(_FEATURES.to_dict(), id="not a reference"),
]


def _lineage_table(*extra: str, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    columns = {
        "feature_index": [0, 0, 3, 3],
        "feature_name": ["g0", "g0", "g3", "g3"],
        "feature_id": ["id0", "id0", "id3", "id3"],
        "lineage": ["A", "B", "A", "B"],
        "r_value": [0.4, np.nan, -0.2, 0.9],
        "p_value": [0.01, np.nan, 0.3, 1e-6],
        "p_value_adjusted": [0.02, np.nan, 0.3, 4e-6],
    }
    for name in extra:
        columns[name] = [0.1, np.nan, -0.5, 0.8]
    return pd.DataFrame(
        {name: values for name, values in columns.items() if name not in drop}
    )


def _marker_result(**overrides) -> LineageMarkerResult:
    values = {
        "ref": _ref("lineage_markers", "1"),
        "table": _lineage_table(),
        "assay": "RNA",
        "lineages": ("A", "B"),
        "effective_cells": np.array([12.5, 30.0]),
        "cell_selection": _CELLS,
        "feature_selection": _FEATURES,
        "fate": _FATE,
    }
    values.update(overrides)
    return LineageMarkerResult(**values)


def _driver_result(**overrides) -> LineageDriverResult:
    values = {
        "ref": _ref("lineage_drivers", "2"),
        "table": _lineage_table("ci_low", "ci_high"),
        "assay": "RNA",
        "lineages": ("A", "B"),
        "n_cells": 40,
        "cell_selection": _CELLS,
        "driver_cells": _ref("cell_selection", "e", assay=None),
        "feature_selection": _FEATURES,
        "fate": _FATE,
    }
    values.update(overrides)
    return LineageDriverResult(**values)


def _aggregation_arrays() -> dict[str, np.ndarray]:
    return {
        "data": np.arange(24.0).reshape(3, 2, 4),
        "feature_indices": np.array([0, 3, 5]),
        "feature_clusters": np.array([1, 0, 1]),
        "feature_names": np.array(["g0", "g3", "g5"]),
        "feature_ids": np.array(["id0", "id3", "id5"]),
        "bin_pseudotime": np.linspace(0.0, 1.0, 8).reshape(2, 4),
        "lineage_weight": np.array([40.5, 22.0]),
    }


def _aggregation_result(**overrides) -> LineageAggregationResult:
    values = {
        "ref": _ref("lineage_aggregation", "3"),
        **_aggregation_arrays(),
        "lineages": ("A", "B"),
        "assay": "RNA",
        "cell_selection": _CELLS,
        "feature_selection": _FEATURES,
        "fate": _FATE,
    }
    values.update(overrides)
    return LineageAggregationResult(**values)


def _curve_arrays() -> dict[str, np.ndarray]:
    return {
        # Lineage A turns left from the x-axis, lineage B right from the y-axis.
        "points": np.array(
            [
                [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
                [[0.0, 0.0], [0.0, 3.0], [-3.0, 3.0]],
            ]
        ),
        "curve_pseudotime": np.array([[0.0, 2.0, 10.0], [0.0, 1.0, 2.0]]),
        "progress": np.array([[0.1, 0.2], [np.nan, np.nan], [0.9, 0.4], [0.5, 0.5]]),
        "distance": np.array([[1.0, 2.0], [np.nan, np.nan], [0.3, 0.1], [0.0, 0.2]]),
        "weights": np.array([[0.7, 0.3], [np.nan, np.nan], [0.1, 0.9], [0.5, 0.5]]),
        "valid": np.array([True, False, True, True]),
    }


def _curves_result(**overrides) -> LineageCurvesResult:
    values = {
        "ref": _ref("lineage_curves", "4"),
        "fate": _FATE,
        "coordinates": _COORDINATES,
        "cell_selection": _CELLS,
        "lineages": ("A", "B"),
        **_curve_arrays(),
    }
    values.update(overrides)
    return LineageCurvesResult(**values)


def _tree_result(**overrides) -> LineageTreeResult:
    values = {
        "ref": _ref("lineage_tree", "5"),
        "labels": _ref("cluster_labels", "9"),
        "coordinates": _COORDINATES,
        "cell_selection": _CELLS,
        "clusters": ("a", "b", "c", "d"),
        "sizes": np.array([10, 25, 7, 3]),
        "means": np.arange(8.0).reshape(4, 2),
        "edges": pd.DataFrame(
            {
                "source": ["a", "b", "b"],
                "target": ["b", "c", "d"],
                "length": [1.5, 2.0, 0.5],
            }
        ),
        "root": "a",
        "ends": (),
        "leaves": ("a", "c", "d"),
        "lineages": (("a", "b", "c"), ("a", "b", "d")),
    }
    values.update(overrides)
    return LineageTreeResult(**values)


def _assert_read_only(array: np.ndarray) -> None:
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)


def test_lineage_marker_result_accepts_a_complete_table():
    result = _marker_result()

    assert result.lineages == ("A", "B")
    assert result.fate == _FATE and result.feature_selection == _FEATURES
    np.testing.assert_array_equal(result.effective_cells, [12.5, 30.0])
    _assert_read_only(result.effective_cells)
    assert set(result.table.columns) == {
        "feature_index",
        "feature_name",
        "feature_id",
        "lineage",
        "r_value",
        "p_value",
        "p_value_adjusted",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.assay = "ATAC"


@pytest.mark.parametrize("features", _FOREIGN_FEATURES)
def test_lineage_marker_result_requires_features_of_its_assay(features):
    with pytest.raises(
        ValueError, match="Lineage-marker feature selection must belong to its assay"
    ):
        _marker_result(feature_selection=features)


def test_lineage_marker_result_rejects_incomplete_tables_and_misaligned_counts():
    with pytest.raises(
        ValueError,
        match="Lineage-marker table is missing columns: feature_id, p_value_adjusted",
    ):
        _marker_result(table=_lineage_table(drop=("p_value_adjusted", "feature_id")))
    for effective_cells in (np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0]])):
        with pytest.raises(
            ValueError, match="Effective cell counts must align with the lineages"
        ):
            _marker_result(effective_cells=effective_cells)


def test_lineage_driver_result_accepts_a_table_with_confidence_intervals():
    result = _driver_result(n_cells=1)

    assert result.n_cells == 1
    assert result.driver_cells.kind == "cell_selection"
    assert {"ci_low", "ci_high"} <= set(result.table.columns)


def test_lineage_driver_result_rejects_invalid_records():
    with pytest.raises(
        ValueError, match="Lineage-driver table is missing columns: ci_high, ci_low"
    ):
        _driver_result(table=_lineage_table())
    with pytest.raises(
        ValueError, match="Lineage-driver table is missing columns: lineage"
    ):
        _driver_result(table=_lineage_table("ci_low", "ci_high", drop=("lineage",)))
    for n_cells in (0, -3):
        with pytest.raises(
            ValueError, match="Lineage drivers need at least one correlated cell"
        ):
            _driver_result(n_cells=n_cells)


@pytest.mark.parametrize("features", _FOREIGN_FEATURES)
def test_lineage_driver_result_requires_features_of_its_assay(features):
    with pytest.raises(
        ValueError, match="Lineage-driver feature selection must belong to its assay"
    ):
        _driver_result(feature_selection=features)


def test_lineage_aggregation_result_freezes_copies_of_its_arrays():
    arrays = _aggregation_arrays()

    result = _aggregation_result(**arrays)
    arrays["data"][0, 0, 0] = -1.0
    arrays["feature_names"][0] = "changed"

    for name, expected in _aggregation_arrays().items():
        stored = getattr(result, name)
        np.testing.assert_array_equal(stored, expected)
        _assert_read_only(stored)
    assert result.data.flags.c_contiguous
    assert result.lineages == ("A", "B")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "data",
            np.zeros((3, 8)),
            "Lineage profiles must be features by lineages by bins",
        ),
        (
            "data",
            np.zeros((3, 3, 4)),
            "Lineage profiles must be features by lineages by bins",
        ),
        ("feature_indices", np.array([0, 3]), "feature_indices does not align"),
        ("feature_clusters", np.zeros(4), "feature_clusters does not align"),
        ("feature_names", np.array([["g0"], ["g3"], ["g5"]]), "feature_names does not"),
        (
            "feature_ids",
            np.array(["id0"]),
            "feature_ids does not align with the profiles",
        ),
        ("bin_pseudotime", np.zeros((2, 3)), "Bin pseudotime must be lineages by bins"),
        ("lineage_weight", np.ones(3), "Lineage weights must align with the lineages"),
    ],
)
def test_lineage_aggregation_result_rejects_misaligned_arrays(name, value, message):
    with pytest.raises(ValueError, match=message):
        _aggregation_result(**{name: value})


@pytest.mark.parametrize("features", _FOREIGN_FEATURES)
def test_lineage_aggregation_result_requires_features_of_its_assay(features):
    with pytest.raises(
        ValueError,
        match="Lineage-aggregation feature selection must belong to its assay",
    ):
        _aggregation_result(feature_selection=features)


def test_lineage_curves_result_freezes_copies_and_reports_point_ranks():
    arrays = _curve_arrays()

    result = _curves_result(**arrays)
    arrays["points"][0, 0, 0] = 99.0
    arrays["valid"][1] = True

    for name, expected in _curve_arrays().items():
        stored = getattr(result, name)
        np.testing.assert_array_equal(stored, expected)
        _assert_read_only(stored)
    np.testing.assert_array_equal(result.curve_progress, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(
        _curves_result(
            points=np.zeros((2, 5, 2)), curve_pseudotime=np.zeros((2, 5))
        ).curve_progress,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"points": np.zeros((2, 3))},
            "Curve points must be lineages by points by dimensions",
        ),
        (
            {"points": np.zeros((3, 3, 2))},
            "Curve points must be lineages by points by dimensions",
        ),
        (
            {"points": np.zeros((2, 1, 2)), "curve_pseudotime": np.zeros((2, 1))},
            "Each curve needs at least two points",
        ),
        (
            {"curve_pseudotime": np.zeros((2, 4))},
            "Curve pseudotime must be lineages by points",
        ),
        (
            {"valid": np.ones((4, 1), dtype=bool)},
            "Curve validity must be one-dimensional",
        ),
        ({"progress": np.zeros((4, 3))}, "Cell positions must be cells by lineages"),
        ({"distance": np.zeros((3, 2))}, "Cell positions must be cells by lineages"),
        ({"weights": np.zeros((4, 3))}, "Cell weights must be cells by lineages"),
        ({"weights": np.zeros(4)}, "Cell weights must be cells by lineages"),
    ],
)
def test_lineage_curves_result_rejects_misaligned_arrays(overrides, message):
    with pytest.raises(ValueError, match=message):
        _curves_result(**overrides)


def test_lineage_curves_project_places_rows_on_every_curve():
    result = _curves_result()
    rows = np.array([[1.5, -1.0], [4.0, 1.5], [-1.5, 4.0]])

    projection = result.project(rows)

    assert isinstance(projection, CurveProjection)
    assert projection.lineages == ("A", "B")
    # Worked by hand: the nearest segment, the fraction along it, and the
    # pseudotime interpolated between the curve points.
    np.testing.assert_allclose(
        projection.progress, [[0.25, 0.0], [0.75, 0.25], [0.0, 0.75]], atol=1e-12
    )
    np.testing.assert_allclose(
        projection.distance,
        [[1.0, math.sqrt(3.25)], [1.0, 4.0], [math.sqrt(18.25), 1.0]],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        projection.pseudotime, [[1.0, 0.0], [6.0, 0.5], [0.0, 1.5]], atol=1e-12
    )
    for lineage in range(2):
        position, distance = project_onto_curve(
            rows, result.points[lineage], result.curve_progress
        )
        np.testing.assert_array_equal(projection.progress[:, lineage], position)
        np.testing.assert_array_equal(projection.distance[:, lineage], distance)
        np.testing.assert_array_equal(
            projection.pseudotime[:, lineage],
            curve_values_at(
                position, result.curve_progress, result.curve_pseudotime[lineage]
            ),
        )


def test_lineage_curves_project_accepts_no_rows():
    projection = _curves_result().project(np.empty((0, 2)))

    for array in (projection.progress, projection.pseudotime, projection.distance):
        assert array.shape == (0, 2)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (np.zeros((2, 3)), "coordinates must have 2 columns, one per curve dimension"),
        (np.zeros(2), "coordinates must have 2 columns, one per curve dimension"),
        (np.array([[0.0, np.nan]]), "coordinates must be finite"),
        (np.array([[np.inf, 0.0]]), "coordinates must be finite"),
    ],
)
def test_lineage_curves_project_rejects_invalid_coordinates(rows, message):
    with pytest.raises(ValueError, match=message):
        _curves_result().project(rows)


def test_lineage_curves_project_rejects_a_collapsed_curve():
    points = _curve_arrays()["points"]
    points[1] = 2.0

    with pytest.raises(ValueError, match="A curve must not collapse to one point"):
        _curves_result(points=points).project(np.zeros((1, 2)))


def test_curve_projection_is_a_frozen_record():
    projection = CurveProjection(
        lineages=("A",),
        progress=np.array([[0.5]]),
        pseudotime=np.array([[2.0]]),
        distance=np.array([[0.1]]),
    )

    assert [field.name for field in dataclasses.fields(projection)] == [
        "lineages",
        "progress",
        "pseudotime",
        "distance",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        projection.lineages = ("B",)


def test_curve_projection_freezes_copies_of_its_positions():
    progress = np.array([[0.5, 0.2], [0.1, 0.9]])

    projection = CurveProjection(
        lineages=("A", "B"),
        progress=progress,
        pseudotime=np.zeros((2, 2)),
        distance=np.ones((2, 2)),
    )
    progress[0, 0] = -1.0

    np.testing.assert_array_equal(projection.progress, [[0.5, 0.2], [0.1, 0.9]])
    for array in (projection.progress, projection.pseudotime, projection.distance):
        _assert_read_only(array)
    projected = _curves_result().project(np.zeros((3, 2)))
    for array in (projected.progress, projected.pseudotime, projected.distance):
        _assert_read_only(array)


@pytest.mark.parametrize(
    ("progress", "pseudotime"),
    [
        (np.zeros((2, 3)), np.zeros((2, 3))),
        (np.zeros(2), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros((3, 2))),
    ],
)
def test_curve_projection_rejects_misaligned_positions(progress, pseudotime):
    with pytest.raises(ValueError, match="Curve positions must be rows by lineages"):
        CurveProjection(
            lineages=("A", "B"),
            progress=progress,
            pseudotime=pseudotime,
            distance=np.zeros_like(progress),
        )


def test_lineage_tree_result_freezes_copies_and_lists_candidate_sinks():
    sizes = np.array([10, 25, 7, 3])

    result = _tree_result(sizes=sizes)
    sizes[0] = 0

    np.testing.assert_array_equal(result.sizes, [10, 25, 7, 3])
    _assert_read_only(result.sizes)
    _assert_read_only(result.means)
    assert result.candidate_sinks == ("c", "d")
    assert _tree_result(root=None, lineages=()).candidate_sinks == ("a", "c", "d")
    # An inner root leaves every leaf a candidate; labels compare by value.
    numeric = _tree_result(
        clusters=(0, 1, 2, 3),
        root=np.int64(1),
        leaves=(0, 2, 3),
        lineages=((1, 0), (1, 2), (1, 3)),
    )
    assert numeric.candidate_sinks == (0, 2, 3)
    leaf_root = _tree_result(clusters=(0, 1, 2, 3), root=np.int64(0), leaves=(0, 2, 3))
    assert leaf_root.candidate_sinks == (2, 3)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"sizes": np.array([10, 25, 7])},
            "Cluster sizes must align with the clusters",
        ),
        ({"means": np.zeros(4)}, "Cluster means must be clusters by dimensions"),
        ({"means": np.zeros((3, 2))}, "Cluster means must be clusters by dimensions"),
        (
            {"edges": pd.DataFrame({"source": ["a"] * 3, "target": ["b"] * 3})},
            "Lineage-tree table is missing columns: length",
        ),
        (
            {
                "edges": pd.DataFrame(
                    {"source": ["a", "b"], "target": ["b", "c"], "length": [1.0, 2.0]}
                )
            },
            "A tree over n clusters has n - 1 edges",
        ),
    ],
)
def test_lineage_tree_result_rejects_misaligned_records(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tree_result(**overrides)
