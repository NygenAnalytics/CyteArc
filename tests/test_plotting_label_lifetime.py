"""Closed figures with on-data category labels are freed."""

import gc
import weakref

import cytearc.plotting as splt
from cytearc.plotting._style import (
    LEGEND_ON_DATA_MAX_CATEGORIES,
    LEGEND_SIDE_MAX_CATEGORIES,
)
from tests.test_plotting_embedding_regressions import _clustered_store


def test_closed_embeddings_with_on_data_labels_are_freed():
    # More categories than a side legend holds, so the default legend labels
    # them on the data.
    n_categories = 20
    assert LEGEND_SIDE_MAX_CATEGORIES < n_categories <= LEGEND_ON_DATA_MAX_CATEGORIES
    centres = [(10.0 * (index % 5), 10.0 * (index // 5)) for index in range(20)]
    store, _, _, _ = _clustered_store(centres, 10, spread=0.5)
    figures = []
    for _ in range(3):
        result = splt.embedding(store, layout_key="U", color_by="cl", show=False)
        assert len(result.axes["cl"].texts) == n_categories
        figures.append(weakref.ref(result.figure))
        result.close()
    del result

    gc.collect()

    # The labels refer back to their axes and figure, which nothing outside
    # the figure may keep alive.
    assert [figure() for figure in figures] == [None, None, None]
