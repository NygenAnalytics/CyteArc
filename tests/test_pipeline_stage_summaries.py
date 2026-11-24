"""A stage summary reports what an artifact records and leaves out the rest.

Runs reuse artifacts that earlier releases saved, which may lack an array or
an attribute that a summary reads, so a summary leaves such a value out
rather than failing the run.
"""

from typing import Any

import numpy as np
import zarr
from zarr.storage import MemoryStore

from cytearc.datastore._pipeline import summaries
from cytearc.storage.artifacts import artifact_group
from cytearc.storage.refs import ArtifactRef
from cytearc.storage.selections import resolve_generated_selection_artifact
from tests.test_pipeline_run_foundation import _root

IDS = np.asarray(["c1", "c2", "c3", "c4"])


def _selection(
    root: zarr.Group,
    values: list[bool],
    operation: str = "test_selection",
    parameters: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> ArtifactRef:
    ref, _mask = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray(values),
        row_ids=IDS,
        operation=operation,
        parameters=parameters or {},
        inputs=inputs or {},
        source_column="I",
    )
    return ref


def test_a_filtering_summary_leaves_out_what_its_records_do_not_hold() -> None:
    root = _root()
    source = _selection(root, [True, True, True, False])
    # A MAD filter of one sample records no sample sizes.
    kept = _selection(
        root,
        [True, False, True, False],
        operation="filter_pipeline_cells",
        parameters={"mad": {"n_mads": 3.0}},
        inputs={"input_cell_selection": source.to_dict()},
    )

    assert summaries._summary(root, kept) == {
        "cells": 2,
        "inputCells": 3,
        "removedCells": 1,
    }
    # Without its source's values, the summary cannot count what it removed.
    del artifact_group(root, source)["values"]
    assert summaries._summary(root, kept) == {"cells": 2}


def test_a_capture_summary_counts_and_names_the_knee_of_each_capture_it_can() -> None:
    root = _root()
    with_knee = _selection(root, [True, True, True, True], operation="test_fit")
    artifact_group(root, with_knee).attrs["knee"] = 120
    without_knee = _selection(root, [True, True, True, False], operation="test_fit")
    artifact_group(root, without_knee).attrs["knee"] = None
    # Capture a's fit found no knee; capture b records no fit; capture c lost
    # its calls, and its fit found a knee.
    capture_a = _selection(
        root,
        [True, True, False, False],
        operation="call_cells",
        inputs={"cell_calling": without_knee.to_dict()},
    )
    capture_b = _selection(root, [False, False, True, False], operation="call_cells")
    capture_c = _selection(
        root,
        [False, False, False, True],
        operation="call_cells",
        inputs={"cell_calling": with_knee.to_dict()},
    )
    del artifact_group(root, capture_c)["values"]
    every = _selection(root, [True, True, True, True])
    combined = _selection(
        root,
        [True, True, True, False],
        operation="combine_capture_cell_calls",
        parameters={"captures": ["a", "b", "c"]},
        inputs={
            "input_cell_selection": every.to_dict(),
            "capture_0": capture_a.to_dict(),
            "capture_1": capture_b.to_dict(),
            "capture_2": capture_c.to_dict(),
        },
    )

    assert summaries._summary(root, combined) == {
        "cells": 3,
        "inputCells": 4,
        "removedCells": 1,
        "calledPerCapture": {"a": 2, "b": 1},
        "kneePerCapture": {"c": 120},
    }


def test_summaries_leave_out_attributes_and_arrays_that_an_artifact_lacks() -> None:
    def group() -> zarr.Group:
        return zarr.open_group(store=MemoryStore(), mode="w")

    # EmptyDrops records no knee when the barcode ranks show none.
    calls = group()
    calls.attrs.update({"retain": 50, "knee": None, "n_ambient": 3})
    assert summaries._cell_calling(calls) == {"retain": 50, "n_ambient": 3}
    graph = group()
    graph.attrs["n_cells"] = 40
    assert summaries._graph_size(graph) == {"cells": 40}
    graph.attrs["n_neighbors"] = 5
    assert summaries._neighbors(graph, {}) == {"cells": 40, "k": 5}
    assert summaries._neighbors(graph, {"selfHitRate": 0.987654}) == {
        "cells": 40,
        "k": 5,
        "selfHitRate": 0.9877,
    }
    # Doublet scores without calls or samples have nothing to summarize.
    doublets = group()
    assert summaries._doublets(doublets) == {}
    doublets.create_array("predicted_doublets", data=np.asarray([1, 0, 0, 0], bool))
    assert summaries._doublets(doublets) == {"predictedDoublets": 1, "rate": 0.25}


def test_doublet_summary_shows_each_samples_threshold_and_its_method() -> None:
    doublets = zarr.open_group(store=MemoryStore(), mode="w")
    doublets.create_array("predicted_doublets", data=np.asarray([1, 0, 0, 0], bool))
    shown = {"labels": {"sample": "A"}, "cells": 2, "predictedDoublets": 1}
    doublets.attrs["samples"] = [
        {
            **shown,
            "threshold": 0.31,
            "thresholdMethod": "minimum",
            "simulatedDoublets": 4,
            "estimatedDoubletRate": 0.5,
        },
        # A sample without a threshold shows its method beside no threshold.
        {
            **shown,
            "labels": {"sample": "B"},
            "predictedDoublets": 0,
            "threshold": None,
            "thresholdMethod": "none",
            "detectableDoubletFraction": 0.0,
        },
    ]

    summary = summaries._doublets(doublets)

    assert summary["samples"] == [
        {**shown, "threshold": 0.31, "thresholdMethod": "minimum"},
        {
            **shown,
            "labels": {"sample": "B"},
            "predictedDoublets": 0,
            "threshold": None,
            "thresholdMethod": "none",
        },
    ]
    # A record without a key shows the keys it holds.
    doublets.attrs["samples"] = [shown]
    assert summaries._doublets(doublets)["samples"] == [shown]
