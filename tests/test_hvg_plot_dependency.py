"""HVG selection on an installation without matplotlib, an optional dependency."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from cytearc import DataStore
from cytearc.datastore._operations.features import _matplotlib_available
from tests.storage_helpers import write_count_store

_MISSING = (
    "features.hvgs draws its mean-variance plot with matplotlib, which is not "
    "installed, so it returns the selection without the plot; install it with "
    "pip install 'cytearc[extra]', or pass show_plot=False"
)


def _store(tmp_path: Path) -> DataStore:
    rng = np.random.default_rng(7)
    counts = rng.poisson(rng.gamma(1.0, 3.0, size=24), size=(40, 24))
    zarr_loc = tmp_path / "store.zarr"
    write_count_store(str(zarr_loc), {"RNA": counts}, "uint16")
    return DataStore(
        str(zarr_loc), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


def _hvgs(store: DataStore, **options):
    cells = store.snapshot_cell_selection()
    settings = {"min_cells": 1, "top_n": 8, "n_bins": 4}
    return store.features.hvgs(cells, **(settings | options))


@pytest.fixture
def without_matplotlib(monkeypatch):
    # A None entry in sys.modules makes `import matplotlib` raise ImportError.
    for name in [name for name in sys.modules if name.split(".")[0] == "matplotlib"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)


def test_matplotlib_availability_follows_the_import():
    assert _matplotlib_available() is True


def test_matplotlib_availability_is_false_when_its_import_fails(without_matplotlib):
    assert _matplotlib_available() is False


def test_hvgs_without_matplotlib_warn_and_return_the_selection(
    tmp_path, without_matplotlib
):
    store = _store(tmp_path)

    with pytest.warns(UserWarning) as caught:
        ref = _hvgs(store)
    assert [str(warning.message) for warning in caught] == [_MISSING]
    # The warning points at the caller, and the selection was saved once.
    assert caught[0].filename == __file__
    assert (
        store.artifacts.list(kind="feature_selection", from_assay="RNA").count(ref) == 1
    )

    # A second call reuses the saved selection instead of failing again.
    with pytest.warns(UserWarning, match="returns the selection without the plot"):
        assert _hvgs(store) == ref


def test_hvgs_without_matplotlib_and_without_a_plot_do_not_warn(
    tmp_path, without_matplotlib
):
    store = _store(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = _hvgs(store, show_plot=False)
    assert ref.kind == "feature_selection"
