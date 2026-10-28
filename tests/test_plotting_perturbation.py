from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytearc.perturbation.guide_calls import (
    GuideCalls,
    guide_call_arrays,
    guide_calls_frames,
)
from cytearc.plotting.perturbation import guide_calls
from cytearc.storage.refs import ArtifactRef

CALLS = ArtifactRef(
    scope="assay", assay="CRISPR", kind="guide_calls", artifact_id="a" * 64
)


@pytest.fixture(autouse=True)
def _figures_are_closed():
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for number in leaked:
        plt.close(number)
    assert not leaked, f"Figures left open: {sorted(leaked)}"


def _store(n_guides: int) -> SimpleNamespace:
    """A store whose loader returns calls of a library of ``n_guides`` guides.

    Guide g is called in g + 1 cells, so guides rank in reverse order. The
    first cell carries guide 0 and also guide 1, so it is the one cell with two
    called guides.
    """
    guide_ids = np.asarray([f"g{index}" for index in range(n_guides)], dtype=object)
    codes = np.repeat(np.arange(n_guides), np.arange(1, n_guides + 1))
    cells = np.arange(len(codes))
    arrays = guide_call_arrays(
        np.concatenate([cells, [0]]),
        np.concatenate([codes, [1]]),
        np.full(len(codes) + 1, 20),
        n_cells=len(codes),
        guide_ids=guide_ids,
    )
    frames = guide_calls_frames(
        np.asarray([f"cell{index}" for index in cells]),
        values=arrays.values,
        n_guides=arrays.n_guides,
        top_guide=guide_ids[codes],
        top_umi=np.full(len(codes), 20),
        top_fraction=np.where(arrays.n_guides > 1, 0.5, 1.0),
        total_umi=np.where(arrays.n_guides > 1, 40, 20),
        guide_codes=arrays.guide_codes,
        guide_umis=arrays.guide_umis,
        guide_ids=guide_ids,
    )
    loaded = GuideCalls(
        cells=frames[0],
        calls=frames[1],
        guides=pd.Index(guide_ids, name="guide"),
        source="import_guide_calls",
    )
    return SimpleNamespace(
        perturbation=SimpleNamespace(load_guide_calls=lambda ref: loaded)
    )


def test_a_large_library_ranks_guides_by_their_cells() -> None:
    result = guide_calls(_store(50), CALLS, show=False)

    try:
        ax = result.axes["cells_per_guide"]
        assert ax.get_xlabel() == "Guides, ranked by cells"
        lines = ax.get_lines()
        assert len(lines) == 2
        per_guide = result.tables["cells_per_guide"]
        np.testing.assert_array_equal(lines[0].get_ydata(), per_guide["cells"])
        np.testing.assert_array_equal(
            lines[1].get_ydata(), per_guide["one_guide_cells"]
        )
        assert per_guide["guide"].iloc[0] == "g49"
        # one_guide_cells, several_guide_cells, cells
        assert per_guide.set_index("guide").loc["g0"].tolist() == [0, 1, 1]
        assert per_guide.set_index("guide").loc["g1"].tolist() == [2, 1, 3]
    finally:
        result.close()


def test_a_small_library_labels_one_bar_per_guide() -> None:
    result = guide_calls(_store(5), CALLS, show=False)

    try:
        ax = result.axes["cells_per_guide"]
        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert sorted(labels) == ["g0", "g1", "g2", "g3", "g4"]
        assert ax.get_lines() == []
    finally:
        result.close()


def test_showing_the_plot_releases_its_figure() -> None:
    result = guide_calls(_store(5), CALLS)

    assert result.tables["guides_per_cell"]["cells"].sum() == 15


def test_the_plot_refuses_settings_and_stores_it_cannot_use() -> None:
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        guide_calls(_store(5), CALLS, bins=0, show=False)
    with pytest.raises(ValueError, match="max_guides_per_cell"):
        guide_calls(_store(5), CALLS, max_guides_per_cell=True, show=False)
    with pytest.raises(TypeError, match="store does not provide guide calls"):
        guide_calls(SimpleNamespace(), CALLS, show=False)
