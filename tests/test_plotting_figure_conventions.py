"""The conventions that every clone figure keeps, on the truth's clones and clonotypes."""

import re
from dataclasses import replace
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.legend import Legend

import cytearc.plotting.clones as clone_plots
from cytearc.metadata.artifacts import categorical_display
from cytearc.plotting import compose_results
from cytearc.plotting._figure import _shared_legend_parts, is_colorbar_axes
from cytearc.plotting._style import (
    MAX_FIGURE_WIDTH_INCHES,
    RECEPTOR_TYPE_SCALE,
    category_label,
)
from tests.clone_helpers import (
    CLONE_FIGURES,
    clone_figure_calls,
    every_cell,
    import_truth_clones,
    import_truth_receptors,
    write_truth_store,
)
from tests.clone_truth import N_CELLS, clone_truth, receptor_truth
from tests.storage_helpers import (
    insert_nullable_cell_column,
    open_store,
    write_count_store,
)

pytestmark = pytest.mark.usefixtures("figures_closed")

TRUTH = clone_truth()


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("figure_conventions")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    clones = import_truth_clones(store, TRUTH, folder)
    receptors = import_truth_receptors(store, receptor_truth(TRUTH), folder)
    clonotypes = store.repertoire.clonotypes(receptors, receptor_arms="any")
    return SimpleNamespace(
        store=store,
        clones=clones,
        clonotypes=clonotypes,
        calls={
            "clones": clone_figure_calls(store, clones),
            "clonotypes": clone_figure_calls(store, clonotypes),
        },
    )


def _texts(result) -> list[str]:
    """Every text that a figure shows: titles, labels, ticks, legends and notes."""
    found = [text.get_text() for text in result.figure.findobj(matplotlib.text.Text)]
    return [text for text in found if text]


def _drawn_colors(ax) -> set[str]:
    """The colours of what a panel draws: its bars and its lines."""
    colors = {to_hex(patch.get_facecolor()) for patch in ax.patches}
    colors |= {to_hex(line.get_color()) for line in ax.get_lines()}
    return colors


def test_every_clone_figure_is_checked() -> None:
    # A new clone figure joins these conventions by joining the helper.
    assert set(CLONE_FIGURES) == set(clone_plots.__all__)


@pytest.mark.parametrize("source", ["clones", "clonotypes"])
@pytest.mark.parametrize("name", CLONE_FIGURES)
def test_legends_are_reported_with_their_scales_and_agree_with_the_drawing(
    truth, source, name
) -> None:
    result = truth.calls[source][name](show=False)
    try:
        assert result.legends, name
        kinds = [legend.kind for legend in result.legends]
        assert set(kinds) <= {"categorical", "colorbar"}
        # Each legend has its scale, in the order compose_results pairs them.
        assert len(result.scales) == len(result.legends)
        # compose_results can draw each legend again once.
        parts = _shared_legend_parts(result)
        assert parts is not None and len(parts) == len(result.legends)
        drawn = [
            artist
            for ax in result.figure.axes
            for artist in ax.get_children()
            if isinstance(artist, Legend)
        ]
        colorbars = [ax for ax in result.figure.axes if is_colorbar_axes(ax)]
        categorical = [
            (legend, scale)
            for legend, scale in zip(result.legends, result.scales, strict=True)
            if legend.kind == "categorical"
        ]
        assert len(drawn) == len(categorical)
        for (legend, scale), artist in zip(categorical, drawn, strict=True):
            # The drawn legend lists the reported scale, in its colours, and
            # each colour is one that its panel draws.
            assert artist.get_title().get_text() == legend.label
            labels = [category_label(scale, value) for value in scale.order]
            assert [text.get_text() for text in artist.get_texts()] == labels
            colors = [
                to_hex(
                    handle.get_facecolor()
                    if hasattr(handle, "get_facecolor")
                    else handle.get_color()
                )
                for handle in artist.legend_handles
            ]
            assert colors == [scale.palette[value] for value in scale.order]
            assert set(colors) <= _drawn_colors(artist.axes)
        colorbar_legends = [
            legend for legend in result.legends if legend.kind == "colorbar"
        ]
        assert len(colorbars) == len(colorbar_legends)
        for legend, colorbar in zip(colorbar_legends, colorbars, strict=True):
            assert colorbar.get_ylabel() == legend.label
            assert colorbar.get_ylim() == pytest.approx(
                (legend.extras["vmin"], legend.extras["vmax"])
            )
        assert result.provenance.notes == (name,)
    finally:
        result.close()


@pytest.mark.parametrize("name", CLONE_FIGURES)
def test_a_hidden_legend_is_still_reported(truth, name) -> None:
    shown = truth.calls["clones"][name](show=False)
    hidden = truth.calls["clones"][name](show=False, show_legend=False)
    try:
        assert hidden.legends == shown.legends
        assert not any(
            isinstance(artist, Legend)
            for ax in hidden.figure.axes
            for artist in ax.get_children()
        )
        assert not any(is_colorbar_axes(ax) for ax in hidden.figure.axes)
    finally:
        shown.close()
        hidden.close()


def test_figures_grouped_by_one_column_share_one_legend_when_composed(truth) -> None:
    calls = truth.calls["clones"]
    figure = plt.figure(figsize=(14, 9), layout="constrained")
    grid = figure.subplots(3, 4)
    abundance = calls["clone_abundance"](
        target={"sizes": grid[0, 0], "top": grid[0, 1]}, show=False
    )
    rarefaction = calls["clonal_rarefaction"](
        target={
            "order_0": grid[1, 0],
            "order_1": grid[1, 1],
            "order_2": grid[1, 2],
            "coverage": grid[1, 3],
        },
        show=False,
    )
    diversity = calls["clonal_diversity"](
        target={
            "n_clones": grid[2, 0],
            "shannon": grid[2, 1],
            "inverse_simpson": grid[2, 2],
        },
        show=False,
    )
    try:
        # Each figure draws a legend of the samples beside its own panels.
        assert all(len(result.legends) == 1 for result in (abundance, rarefaction))
        composite = compose_results(figure, [abundance, rarefaction, diversity])
        legends = list(figure.legends) + [
            artist
            for ax in figure.axes
            for artist in ax.get_children()
            if isinstance(artist, Legend)
        ]
        # The three report one legend, which the composite draws once.
        assert len(legends) == 1
        assert legends[0].get_title().get_text() == "sample"
        assert [text.get_text() for text in legends[0].get_texts()] == ["d2", "d4"]
        assert len(composite.legends) == 1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", CLONE_FIGURES)
def test_figures_of_clonotypes_name_clonotypes(truth, name) -> None:
    result = truth.calls["clonotypes"][name](show=False)
    try:
        texts = _texts(result)
        # Clonotypes are labelled by their junctions, never their IDs, and no
        # text that names them says "clone".
        assert not [text for text in texts if "clonotype_" in text]
        assert not [
            text for text in texts if re.search(r"\bclones?\b", text, re.IGNORECASE)
        ]
    finally:
        result.close()


@pytest.mark.parametrize(
    ("name", "tables"),
    [
        ("clone_abundance", ["top"]),
        ("clone_tracking", ["tracking"]),
        ("clone_fate_bias", ["clones", "fates"]),
    ],
)
def test_tables_that_list_clones_hold_their_ids_and_labels(truth, name, tables) -> None:
    result = truth.calls["clonotypes"][name](show=False)
    try:
        for key in tables:
            table = result.tables[key]
            assert {"clone", "label"} <= set(table.columns)
            assert table["clone"].str.startswith("clonotype_").all()
            assert not table["label"].str.startswith("clonotype_").any()
    finally:
        result.close()


N_MANY = 200


@pytest.fixture(scope="module")
def many(tmp_path_factory) -> SimpleNamespace:
    """200 cells, cell i in clone i // 2, each with a label of its own."""
    path = tmp_path_factory.mktemp("many_figure_groups") / "store.zarr"
    write_count_store(str(path), {"RNA": np.ones((N_MANY, 3), dtype=int)}, "uint16")
    store = open_store(path)
    store.cells.insert(
        "clone", np.asarray([f"k{i // 2:03d}" for i in range(N_MANY)], dtype=object)
    )
    store.cells.insert(
        "cell", np.asarray([f"cell{i:03d}" for i in range(N_MANY)], dtype=object)
    )
    store.cells.insert("even", np.arange(N_MANY) % 2 == 0)
    store.cells.insert("odd", np.arange(N_MANY) % 2 == 1)
    store.cells.insert("every_cell", np.ones(N_MANY, dtype=bool))
    clones = store.clones.from_column(
        "clone", cell_selection=store.snapshot_cell_selection("every_cell")
    )
    return SimpleNamespace(store=store, clones=clones)


def test_figures_of_many_groups_and_clones_stop_at_the_page_width(many) -> None:
    store, clones, plots = many.store, many.clones, many.store.plots
    results = [
        plots.clone_abundance(clones, group_by="cell", max_clones=100, show=False),
        plots.clonal_expansion(
            store.clones.expansion(clones), group_by="cell", show=False
        ),
        plots.clonal_rarefaction(
            store.clones.rarefaction(clones, group_by="cell"), show=False
        ),
        plots.clonal_diversity(
            store.clones.diversity(clones, group_by="cell"),
            indices=["n_cells", "n_clones", "shannon", "gini"],
            show=False,
        ),
        plots.clone_overlap(store.clones.overlap(clones, group_by="cell"), show=False),
        plots.clone_tracking(
            store.clones.track(clones, group_by="cell", top_n=1), show=False
        ),
        plots.clone_coupling(
            store.clones.coupling(clones, group_by="cell", n_permutations=0),
            value="coupling",
            show=False,
        ),
        plots.clone_fate_bias(
            store.clones.fate_bias(
                clones,
                group_by="cell",
                early=store.snapshot_cell_selection("even"),
                late=store.snapshot_cell_selection("odd"),
            ),
            max_clones=100,
            show=False,
        ),
    ]
    try:
        assert [result.provenance.notes[0] for result in results] == list(CLONE_FIGURES)
        for result in results:
            width, height = result.figure.get_size_inches()
            assert width <= MAX_FIGURE_WIDTH_INCHES, result.provenance.notes
            # A panel stops growing where its axis stops naming items.
            assert height <= 22.0, result.provenance.notes
    finally:
        for result in results:
            result.close()


def _empty_fate_bias(store):
    """A store whose fate bias holds no clone with late cells."""
    bias = store.clones.load_fate_bias

    def load(ref):
        loaded = bias(ref)
        return replace(loaded, clone_fates=loaded.clone_fates.iloc[:0])

    return SimpleNamespace(
        clones=SimpleNamespace(load_fate_bias=load),
        _clone_figure_nouns=store._clone_figure_nouns,
        _clone_figure_labels=store._clone_figure_labels,
    )


def test_figures_with_nothing_to_draw_raise(truth) -> None:
    store, clones = truth.store, truth.clones
    if "unlabelled" not in store.cells.columns:
        insert_nullable_cell_column(
            store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
        )
    plots = store.plots
    rarefaction = store.clones.rarefaction(clones, group_by="sample")
    diversity = store.clones.diversity(clones, group_by="sample")
    overlap = store.clones.overlap(clones, group_by="sample")
    tracking = store.clones.track(clones, group_by="sample", top_n=1)
    coupling = store.clones.coupling(clones, group_by="cell_type", n_permutations=5)
    bias = store.clones.fate_bias(
        clones,
        group_by="cell_type",
        early=store.snapshot_cell_selection("is_d2"),
        late=store.snapshot_cell_selection("is_d4"),
    )
    empty = [
        lambda: plots.clone_abundance(clones, group_by="unlabelled", show=False),
        lambda: plots.clonal_expansion(
            store.clones.expansion(clones), group_by="unlabelled", show=False
        ),
        lambda: plots.clonal_rarefaction(
            replace(rarefaction, curves=rarefaction.curves.iloc[:0]), show=False
        ),
        lambda: plots.clonal_diversity(
            replace(diversity, indices=diversity.indices.iloc[:0]), show=False
        ),
        lambda: plots.clone_overlap(
            replace(overlap, pairs=overlap.pairs.iloc[:0]), show=False
        ),
        lambda: plots.clone_tracking(replace(tracking, tracked=()), show=False),
        lambda: plots.clone_coupling(
            replace(coupling, pairs=coupling.pairs.iloc[:0]), show=False
        ),
        lambda: clone_plots.clone_fate_bias(_empty_fate_bias(store), bias, show=False),
    ]
    assert len(empty) == len(CLONE_FIGURES)
    figures = set(plt.get_fignums())
    for call in empty:
        with pytest.raises(ValueError, match="so there is nothing to plot$"):
            call()
    # A refused figure leaves no figure open.
    assert set(plt.get_fignums()) == figures


def _line_colors(ax) -> dict[str, str]:
    return {line.get_label(): to_hex(line.get_color()) for line in ax.get_lines()}


def test_group_colours_follow_a_stored_palette(truth) -> None:
    store, clones = truth.store, truth.clones
    display = categorical_display(np.asarray(["d2", "d4"], dtype=object))
    for category, color in zip(
        display["categories"], ("#cc0000", "#00aa00"), strict=True
    ):
        category["color"] = color
    column = store.zw["cellData"]["sample"]
    column.attrs["display"] = display
    try:
        calls = clone_figure_calls(store, clones)
        abundance = calls["clone_abundance"](show=False)
        rarefaction = calls["clonal_rarefaction"](show=False)
        diversity = calls["clonal_diversity"](show=False)
        try:
            expected = {"d2": "#cc0000", "d4": "#00aa00"}
            for result in (abundance, rarefaction, diversity):
                assert result.scales[0].palette == expected
            lines = _line_colors(rarefaction.axes["order_0"])
            assert lines["d2"] == "#cc0000" and lines["d4"] == "#00aa00"
            sizes = _line_colors(abundance.axes["sizes"])
            assert sizes["d2"] == "#cc0000" and sizes["d4"] == "#00aa00"
            bars = diversity.axes["n_clones"].patches
            assert [to_hex(bar.get_facecolor()) for bar in bars] == [
                "#cc0000",
                "#00aa00",
            ]
        finally:
            for result in (abundance, rarefaction, diversity):
                result.close()
    finally:
        del column.attrs["display"]


def test_a_subset_of_cells_never_recolours_a_group(truth) -> None:
    store, clones = truth.store, truth.clones
    every = store.plots.clonal_rarefaction(
        store.clones.rarefaction(clones, group_by="sample"), show=False
    )
    d4 = store.plots.clonal_rarefaction(
        store.clones.rarefaction(
            clones,
            group_by="sample",
            cell_selection=store.snapshot_cell_selection("is_d4"),
        ),
        show=False,
    )
    try:
        # d4 alone keeps the colour it takes beside d2, which colours are
        # assigned over every label of the column.
        assert d4.scales[0].palette == {"d4": every.scales[0].palette["d4"]}
        assert _line_colors(d4.axes["order_0"])["d4"] == every.scales[0].palette["d4"]
        assert every.scales[0].palette["d4"] != every.scales[0].palette["d2"]
    finally:
        every.close()
        d4.close()


def test_receptor_types_take_their_vocabularys_colours(truth) -> None:
    result = truth.store.plots.clone_abundance(truth.clonotypes, show=False)
    try:
        scale = result.scales[0]
        assert result.legends[0].label == "receptor_type"
        assert scale.palette == {
            kind: RECEPTOR_TYPE_SCALE.palette[kind] for kind in scale.order
        }
        assert list(scale.order) == [
            kind for kind in RECEPTOR_TYPE_SCALE.order if kind in scale.order
        ]
    finally:
        result.close()


def test_label_artifacts_colour_groups_over_all_their_labels(truth) -> None:
    store, clones = truth.store, truth.clones
    labels = store.snapshot_cluster_labels(
        "cell_type", cell_selection=every_cell(store)
    )
    every = store.plots.clone_abundance(clones, groups=labels, show=False)
    d4 = store.plots.clonal_rarefaction(
        store.clones.rarefaction(
            clones, groups=labels, cell_selection=store.snapshot_cell_selection("is_d4")
        ),
        show=False,
    )
    try:
        full = every.scales[0].palette
        assert d4.scales[0].palette == {name: full[name] for name in d4.scales[0].order}
    finally:
        every.close()
        d4.close()
