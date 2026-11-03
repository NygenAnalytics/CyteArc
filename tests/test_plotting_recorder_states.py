"""The recorder states plot, on the synthetic truth's PEtracer-like alleles.

Its table is checked against counts taken with pandas from the truth, whose
cells the import gives back unchanged.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cytearc import plotting
from tests.clone_helpers import every_cell, write_truth_store
from tests.clone_truth import (
    PETRACER_INTEGRATIONS,
    SAMPLES,
    clone_truth,
    recorder_truth,
    write_petracer_characters,
)

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()
RECORDERS = recorder_truth(TRUTH)


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """The truth's characters in the Zenodo layout, whose second embryo lacks an integration."""
    folder = tmp_path_factory.mktemp("recorder_states")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    path = write_petracer_characters(
        RECORDERS, TRUTH, folder / "characters.csv.gz", layout="zenodo"
    )
    return store, store.clones.import_alleles(path, cell_selection=every_cell(store))


def _expected_counts() -> dict[tuple[str, str, str], int]:
    """Each site's cells by state, taken from the truth with pandas."""
    table = RECORDERS.independent.copy()
    state = np.where(table["read"], table["state"], "missing")
    lacking = (table["integration"] == PETRACER_INTEGRATIONS[-1]) & (
        TRUTH.samples[table["cell"]] == SAMPLES[1]
    )
    table["state"] = np.where(lacking, "not_applicable", state)
    counts = table.groupby(["integration", "site", "state"]).size()
    return {key: int(value) for key, value in counts.items()}


def test_recorder_states_count_each_sites_cells(imported) -> None:
    store, ref = imported
    result = plotting.recorder_states(store, ref, show=False)
    try:
        table = result.tables["site_states"]
        assert table.columns.tolist() == [
            "integration",
            "site",
            "state",
            "cells",
            "share",
        ]
        found = {
            (row.integration, row.site, row.state): int(row.cells)
            for row in table.itertuples()
            if row.cells
        }
        assert found == _expected_counts()
        # Shares leave out the cells whose embryo lacks the integration.
        for _key, rows in table.groupby(["integration", "site"]):
            applicable = rows[rows["state"] != "not_applicable"]
            assert applicable["share"].sum() == pytest.approx(1.0)
            assert rows.loc[rows["state"] == "not_applicable", "share"].isna().all()
        # One bar per site, stacked by the four drawn states.
        ax = result.axes["site_states"]
        assert len(ax.get_xticklabels()) == 12
        assert len(ax.patches) == 12 * 4
        assert result.provenance.extras["drawn_sites"] == 12
    finally:
        plt.close(result.figure)


def test_recorder_states_draw_the_sites_read_in_the_most_cells(imported) -> None:
    store, ref = imported
    result = plotting.recorder_states(store, ref, max_sites=3, show=False)
    try:
        table = result.tables["site_states"]
        read = table[table["state"].isin(["mark", "unknown_mark", "unedited"])]
        most = read.groupby(["integration", "site"], sort=False)["cells"].sum()
        chosen = set(most.sort_values(ascending=False, kind="stable").index[:3])
        expected = [pair for pair in most.index if pair in chosen]
        labels = [
            label.get_text() for label in result.axes["site_states"].get_xticklabels()
        ]
        assert labels == [f"{integration}-{site}" for integration, site in expected]
        # The table keeps every site.
        assert table.groupby(["integration", "site"]).ngroups == 12
    finally:
        plt.close(result.figure)


def test_the_store_accessor_forwards_and_checks_its_arguments(imported) -> None:
    store, ref = imported
    result = store.plots.recorder_states(ref, show=False, theme="paper")
    plt.close(result.figure)
    assert result.provenance.notes == ("lineage_alleles",)
    with pytest.raises(ValueError, match="max_sites"):
        plotting.recorder_states(store, ref, max_sites=0, show=False)
    with pytest.raises(TypeError, match="clones.load_alleles"):
        plotting.recorder_states(object(), ref, show=False)
    figure, ax = plt.subplots()
    try:
        drawn = plotting.recorder_states(
            store, ref, target=ax, show=False, show_legend=False
        )
        assert drawn.axes["site_states"] is ax and not drawn.legends
    finally:
        plt.close(figure)


def test_alleles_without_records_draw_a_note_and_show_on_request(tmp_path):
    from tests.storage_helpers import open_store, write_count_store

    write_count_store(
        str(tmp_path / "store.zarr"),
        {"RNA": np.ones((2, 2), dtype=np.uint16)},
        "uint16",
        cell_ids=["c1", "c2"],
    )
    store = open_store(tmp_path / "store.zarr")
    path = tmp_path / "alleles.csv"
    # The table's only cell is none of the store's.
    path.write_text(
        "intID,cellBC,RNF2,HEK3,EMX1,UMI,readCount\nintID1,c9,ACAGT,None,None,3,30\n"
    )
    with pytest.warns(UserWarning, match="match no selected cell"):
        ref = store.clones.import_alleles(
            path, cell_selection=store.snapshot_cell_selection()
        )
    result = plotting.recorder_states(store, ref, show=True)
    try:
        ax = result.axes["site_states"]
        assert [text.get_text() for text in ax.texts] == ["No allele records"]
        assert result.tables["site_states"].empty
        assert not ax.patches and not result.legends
    finally:
        plt.close(result.figure)
