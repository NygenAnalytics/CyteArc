"""The conventions that every clone, repertoire and variant figure keeps, on the truth's experiment."""

import re
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.legend import Legend

import cytearc.plotting.clones as clone_plots
import cytearc.plotting.lineage_barcodes as barcode_plots
import cytearc.plotting.phylogenies as phylogeny_plots
import cytearc.plotting.recorder_alleles as recorder_plots
import cytearc.plotting.variants as variant_plots
from cytearc.metadata.artifacts import categorical_display
from cytearc.plotting import compose_results
from cytearc.plotting._figure import _shared_legend_parts, is_colorbar_axes
from cytearc.plotting._style import (
    MAX_FIGURE_WIDTH_INCHES,
    RECEPTOR_TYPE_SCALE,
    category_label,
)
from tests.clone_helpers import (
    axis_names,
    names_every_nth_item,
    names_keep_apart,
    CLONE_FIGURES,
    REPERTOIRE_FIGURES,
    VARIANT_FIGURES,
    clone_figure_calls,
    every_cell,
    import_truth_clones,
    import_truth_lineage,
    import_truth_phylogeny,
    import_truth_receptors,
    import_truth_variants,
    repertoire_figure_calls,
    variant_figure_calls,
    write_cells_store,
    write_clone_contigs,
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
        # Every figure records the clones it draws and its grouping.
        clones = truth.clones if source == "clones" else truth.clonotypes
        assert result.provenance.extras["clones"] == clones.to_dict()
        grouping = result.provenance.extras["grouping"]
        assert grouping.split(", ")[0] in ("sample", "cell_type")
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
        # The tallest: two rows of diversity panels, each of 40 groups named.
        diversity = results[3].figure.get_size_inches()
        assert tuple(diversity) == pytest.approx((7.5, 2 * (1.0 + 0.25 * 40)))
    finally:
        for result in results:
            result.close()


# Cell types named as a tutorial names them, beside their receptor type.
LONG_NAMES = (
    "CD4 memory T, TCR",
    "CD4 naive T, TCR",
    "CD8 effector memory T, TCR",
    "CD8 naive T, TCR",
    "MAIT and gamma delta T, TCR",
)


@pytest.fixture(scope="module")
def long_names(tmp_path_factory) -> SimpleNamespace:
    """300 cells of five cell types with long names, cell i in clone i // 3."""
    folder = tmp_path_factory.mktemp("long_group_names")
    ids = [f"cell{i:03d}" for i in range(300)]
    store = write_cells_store(folder / "store.zarr", ids)
    store.cells.insert(
        "cell_type", np.asarray(LONG_NAMES, dtype=object)[np.arange(300) % 5]
    )
    clone_of = np.arange(300) // 3
    store.cells.insert(
        "clone", np.asarray([f"k{clone:03d}" for clone in clone_of], dtype=object)
    )
    cells = every_cell(store)
    contigs = write_clone_contigs(folder, "contigs.csv", ids, clone_of)
    return SimpleNamespace(
        store=store,
        clones=store.clones.from_column("clone", cell_selection=cells),
        receptors=store.repertoire.import_receptors(contigs, cell_selection=cells),
    )


def _written(ax, renderer) -> list[tuple[str, object]]:
    """The extents of what a panel writes: its title, axis labels and the tick labels within its limits."""
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    for axis, at, limits in (
        (ax.xaxis, 0, ax.get_xlim()),
        (ax.yaxis, 1, ax.get_ylim()),
    ):
        low, high = sorted(limits)
        texts += [
            label
            for label in axis.get_ticklabels()
            if label.get_visible() and low <= label.get_position()[at] <= high
        ]
    return [
        (text.get_text(), text.get_window_extent(renderer))
        for text in texts
        if text.get_visible() and text.get_text()
    ]


@pytest.mark.parametrize(
    "draw",
    [
        lambda data: data.store.plots.clonal_rarefaction(
            data.store.clones.rarefaction(data.clones, group_by="cell_type"),
            show=False,
        ),
        lambda data: data.store.plots.clonal_diversity(
            data.store.clones.diversity(data.clones, group_by="cell_type"),
            show=False,
        ),
        lambda data: data.store.plots.spectratype(
            data.store.repertoire.spectratype(data.receptors, group_by="cell_type"),
            show=False,
        ),
    ],
    ids=["clonal_rarefaction", "clonal_diversity", "spectratype"],
)
def test_panels_beside_a_legend_of_long_names_keep_their_texts_apart(
    long_names, draw
) -> None:
    # The page's width holds three panels beside the legend, so a panel's
    # title, labels and tick labels must stay within its own room.
    result = draw(long_names)
    try:
        figure = result.figure
        assert figure.get_size_inches()[0] == MAX_FIGURE_WIDTH_INCHES
        figure.draw_without_rendering()
        renderer = figure.canvas.get_renderer()
        frames = {
            key: ax.get_window_extent(renderer) for key, ax in result.axes.items()
        }
        written = {key: _written(ax, renderer) for key, ax in result.axes.items()}
        collisions = [
            (key, text, other)
            for key, other in permutations(result.axes, 2)
            for text, extent in written[key]
            if extent.overlaps(frames[other])
            or any(extent.overlaps(seen) for _, seen in written[other])
        ]
        assert not collisions
        # The legend names every group beside the first row.
        assert set(LONG_NAMES) <= set(_texts(result))
    finally:
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
    store.cells.insert("not_monocyte", TRUTH.cell_types != "Monocyte", overwrite=True)
    every = store.plots.clone_abundance(clones, groups=labels, show=False)
    # Without the monocytes, the first label, the two other types keep the
    # second and third colours, which colours over the groups drawn would
    # not give them.
    rest = store.plots.clonal_rarefaction(
        store.clones.rarefaction(
            clones,
            groups=labels,
            cell_selection=store.snapshot_cell_selection("not_monocyte"),
        ),
        show=False,
    )
    try:
        full = every.scales[0].palette
        assert list(full) == ["Monocyte", "Neutrophil", "Undifferentiated"]
        assert rest.scales[0].order == ("Neutrophil", "Undifferentiated")
        assert rest.scales[0].palette == {
            "Neutrophil": full["Neutrophil"],
            "Undifferentiated": full["Undifferentiated"],
        }
        assert full["Neutrophil"] != full["Monocyte"]
    finally:
        every.close()
        rest.close()


def _insert_masked_labels(store) -> None:
    """Insert labels x and y, each monocyte's missing, with a placeholder z."""
    if "masked_xy" not in store.cells.columns:
        insert_nullable_cell_column(
            store,
            "masked_xy",
            np.where(
                TRUTH.cell_types == "Monocyte",
                "z",
                np.where(TRUTH.samples == "d2", "x", "y"),
            ),
            TRUTH.cell_types == "Monocyte",
        )


def test_missing_labels_take_no_colour(truth) -> None:
    store, clones = truth.store, truth.clones
    _insert_masked_labels(store)
    # A missing label, which would sort before x as the text of no label,
    # takes no colour.
    result = store.plots.clone_abundance(clones, group_by="masked_xy", show=False)
    try:
        assert result.scales[0].order == ("x", "y")
        assert result.scales[0].palette == {"x": "#1f77b4", "y": "#ff7f0e"}
    finally:
        result.close()


def test_a_columns_labels_are_read_in_blocks(truth, monkeypatch) -> None:
    import cytearc.plotting._groups as groups_module

    store = truth.store
    _insert_masked_labels(store)
    read = []
    whole = groups_module.iter_metadata_column_blocks

    def blocks(metadata, column, **options):
        for block in whole(metadata, column, block_rows=5):
            read.append(len(block))
            yield block

    monkeypatch.setattr(groups_module, "iter_metadata_column_blocks", blocks)
    # Every label of the column from blocks of five rows, without the
    # placeholder z of its missing rows, whose masks align with the blocks.
    assert sorted(groups_module._column_labels(store, "masked_xy")) == ["x", "y"]
    assert sum(read) == N_CELLS and max(read) == 5
    assert sorted(groups_module._column_labels(store, "cell_type")) == sorted(
        set(TRUTH.cell_types)
    )


# The panels of each clone figure, for drawing it on axes of a caller's.
_PANELS = {
    "clone_abundance": ["sizes", "top"],
    "clonal_expansion": ["categories", "rank_windows"],
    "clonal_rarefaction": ["order_0", "order_1", "order_2", "coverage"],
    "clonal_diversity": ["n_clones", "shannon", "inverse_simpson"],
    "clone_overlap": ["matrix"],
    "clone_tracking": ["tracking"],
    "clone_coupling": ["matrix"],
    "clone_fate_bias": ["early", "fates"],
}


@pytest.mark.parametrize("name", CLONE_FIGURES)
def test_each_figure_draws_in_a_composite_with_its_legends_once(truth, name) -> None:
    panels = _PANELS[name]
    figure = plt.figure(figsize=(12, 4), layout="constrained")
    grid = figure.subplots(1, len(panels), squeeze=False)[0]
    try:
        result = truth.calls["clones"][name](
            target=dict(zip(panels, grid, strict=True)), show=False
        )
        composite = compose_results(figure, [result])
        # The composite lays out and draws, a heatmap's tree included.
        figure.canvas.draw()
        legends = list(figure.legends) + [
            artist
            for ax in figure.axes
            for artist in ax.get_children()
            if isinstance(artist, Legend)
        ]
        kinds = [legend.kind for legend in result.legends]
        assert len(legends) == kinds.count("categorical")
        colorbars = [ax for ax in figure.axes if is_colorbar_axes(ax)]
        assert len(colorbars) == kinds.count("colorbar")
        assert len(composite.legends) == len(result.legends)
        # The composite names every axis that the figure still draws, a
        # matrix's tree included, and none of the colour bars it replaced.
        assert {key for _, key in composite.axes} == {
            key for key, ax in result.axes.items() if not is_colorbar_axes(ax)
        }
        if "dendrogram" in result.axes:
            # The tree stays above its matrix, which the colour bar spans.
            assert result.axes["dendrogram"] in result.axes["matrix"].child_axes
    finally:
        plt.close(figure)


@pytest.mark.parametrize("theme", ["notebook", "dark"])
@pytest.mark.parametrize("name", CLONE_FIGURES)
def test_figures_take_their_themes_background(truth, name, theme) -> None:
    result = truth.calls["clones"][name](theme=theme, show=False)
    try:
        # A light theme paints the figure white, and the dark one leaves it
        # transparent.
        alpha = 1.0 if theme == "notebook" else 0.0
        assert result.figure.patch.get_alpha() == alpha
        assert {ax.patch.get_alpha() for ax in result.figure.axes} == {alpha}
    finally:
        result.close()


# The figures of receptors, lineage barcodes and recorder alleles keep the
# same conventions, on the truth's receptors, barcodes and alleles.


@pytest.fixture(scope="module")
def repertoire(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("repertoire_figure_conventions")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    receptors = import_truth_receptors(store, receptor_truth(TRUTH), folder)
    barcodes, alleles = import_truth_lineage(store, TRUTH, folder)
    return SimpleNamespace(
        store=store,
        receptors=receptors,
        barcodes=barcodes,
        alleles=alleles,
        calls=repertoire_figure_calls(store, receptors, barcodes, alleles),
    )


# The input that each figure records under its argument's name, and whether
# it records a grouping.
_REPERTOIRE_INPUTS = {
    "receptor_qc": ("receptors", "receptors", True),
    "gene_usage": ("source", "receptors", True),
    "gene_pairing": ("source", "receptors", False),
    "spectratype": ("source", "receptors", True),
    "barcode_calls": ("barcodes", "barcodes", False),
    "recorder_states": ("alleles", "alleles", False),
}


def _drawn_legends(figure) -> list[Legend]:
    return [
        artist
        for ax in figure.axes
        for artist in ax.get_children()
        if isinstance(artist, Legend)
    ]


def test_every_repertoire_figure_is_checked() -> None:
    import cytearc.plotting.repertoire as repertoire_plots

    assert set(REPERTOIRE_FIGURES) == {
        *repertoire_plots.__all__,
        *barcode_plots.__all__,
        *recorder_plots.__all__,
    }
    assert set(_REPERTOIRE_INPUTS) == set(REPERTOIRE_FIGURES)


@pytest.mark.parametrize("name", REPERTOIRE_FIGURES)
def test_repertoire_legends_are_reported_with_their_scales_and_agree_with_the_drawing(
    repertoire, name
) -> None:
    result = repertoire.calls[name](show=False)
    try:
        assert result.legends, name
        assert {legend.kind for legend in result.legends} <= {
            "categorical",
            "colorbar",
        }
        assert len(result.scales) == len(result.legends)
        parts = _shared_legend_parts(result)
        assert parts is not None and len(parts) == len(result.legends)
        categorical = [
            (legend, scale)
            for legend, scale in zip(result.legends, result.scales, strict=True)
            if legend.kind == "categorical"
        ]
        drawn = _drawn_legends(result.figure)
        assert len(drawn) == len(categorical)
        for (legend, scale), artist in zip(categorical, drawn, strict=True):
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
        colorbars = [ax for ax in result.figure.axes if is_colorbar_axes(ax)]
        assert len(colorbars) == len(colorbar_legends)
        for legend, colorbar in zip(colorbar_legends, colorbars, strict=True):
            assert colorbar.get_ylabel() == legend.label
            assert colorbar.get_ylim() == pytest.approx(
                (legend.extras["vmin"], legend.extras["vmax"])
            )
        assert result.provenance.notes == (name,)
        key, ref, grouped = _REPERTOIRE_INPUTS[name]
        assert result.provenance.extras[key] == getattr(repertoire, ref).to_dict()
        if grouped:
            assert result.provenance.extras["grouping"] == "sample"
    finally:
        result.close()


@pytest.mark.parametrize("name", REPERTOIRE_FIGURES)
def test_a_hidden_repertoire_legend_is_still_reported(repertoire, name) -> None:
    shown = repertoire.calls[name](show=False)
    hidden = repertoire.calls[name](show=False, show_legend=False)
    try:
        assert hidden.legends == shown.legends
        assert hidden.scales == shown.scales
        assert not _drawn_legends(hidden.figure)
        assert not any(is_colorbar_axes(ax) for ax in hidden.figure.axes)
    finally:
        shown.close()
        hidden.close()


def _stacks(ax) -> dict[float, list[tuple[float, str]]]:
    """Each vertical bar's parts with a height, from its top: their heights and colours."""
    bars: dict[float, list[tuple[float, float, str]]] = {}
    for patch in ax.patches:
        if patch.get_height() > 0:
            centre = round(patch.get_x() + patch.get_width() / 2, 6)
            bars.setdefault(centre, []).append(
                (patch.get_y(), patch.get_height(), to_hex(patch.get_facecolor()))
            )
    return {
        centre: [(height, color) for _y, height, color in sorted(parts, reverse=True)]
        for centre, parts in sorted(bars.items())
    }


def test_stacked_bars_draw_their_first_part_on_top(repertoire) -> None:
    qc = repertoire.calls["receptor_qc"](show=False)
    states = repertoire.calls["recorder_states"](show=False)
    try:
        for (legend, scale), panel in zip(
            zip(qc.legends, qc.scales, strict=True),
            ("receptor_type", "chain_pairing"),
            strict=False,
        ):
            assert (
                legend.label
                == {
                    "receptor_type": "Receptor type",
                    "chain_pairing": "Chain pairing",
                }[panel]
            )
            table = qc.tables[panel]
            stacks = _stacks(qc.axes[panel])
            groups = list(dict.fromkeys(table["group"]))
            assert len(stacks) == len(groups)
            for (_centre, parts), group in zip(stacks.items(), groups, strict=True):
                # From the top, the group's categories with cells in the
                # legend's order, each its fraction of the group's cells.
                rows = table[table["group"] == group].set_index(panel)["n_cells"]
                held = [value for value in scale.order if rows[value] > 0]
                assert [color for _height, color in parts] == [
                    scale.palette[value] for value in held
                ]
                assert [height for height, _color in parts] == pytest.approx(
                    [rows[value] / rows.sum() for value in held]
                )
        site_table = states.tables["site_states"]
        scale = states.scales[0]
        assert list(scale.order) == ["mark", "unknown_mark", "unedited", "missing"]
        stacks = _stacks(states.axes["site_states"])
        sites = list(dict.fromkeys(zip(site_table["integration"], site_table["site"])))
        assert len(stacks) == len(sites)
        for parts, site in zip(stacks.values(), sites, strict=True):
            rows = site_table[
                (site_table["integration"] == site[0]) & (site_table["site"] == site[1])
            ].set_index("state")["share"]
            held = [state for state in scale.order if rows[state] > 0]
            assert [color for _height, color in parts] == [
                scale.palette[state] for state in held
            ]
            assert [height for height, _color in parts] == pytest.approx(
                [rows[state] for state in held]
            )
    finally:
        qc.close()
        states.close()


def test_locus_panels_share_one_colour_limit(repertoire) -> None:
    for name, table in (("gene_usage", "usage"), ("gene_pairing", "pairs")):
        result = repertoire.calls[name](show=False)
        try:
            fractions = result.tables[table]["fraction"]
            loci = sorted(set(result.tables[table]["locus"]))
            assert list(result.axes) == loci
            # Every locus panel colours its fractions from 0 to the largest of
            # any locus, and the one colour bar stands beside them all.
            for locus in loci:
                (image,) = result.axes[locus].images
                assert (image.norm.vmin, image.norm.vmax) == (0.0, fractions.max())
            (colorbar,) = [ax for ax in result.figure.axes if is_colorbar_axes(ax)]
            assert set(colorbar._colorbar_info["parents"]) == set(result.axes.values())
            assert colorbar.get_ylim() == (0.0, fractions.max())
            assert result.legends[0].extras == {"vmin": 0.0, "vmax": fractions.max()}
            assert colorbar.get_ylabel() == "Fraction of cells"
        finally:
            result.close()


def test_each_vocabulary_takes_one_palette_with_one_colour_for_no_value(
    repertoire,
) -> None:
    from cytearc.clones.recorders import STATES
    from cytearc.plotting._contracts import CategoricalScale
    from cytearc.plotting._style import (
        CHAIN_ARM_SCALE,
        CHAIN_PAIRING_SCALE,
        LOCUS_SCALE,
        MUTED_COLOR,
        NO_VALUE_COLOR,
        RECORDER_STATE_SCALE,
    )
    from cytearc.repertoire.chains import (
        CHAIN_PAIRINGS,
        LOCI,
        NO_RECEPTOR,
        RECEPTOR_TYPES,
    )

    vocabularies = [
        (RECEPTOR_TYPE_SCALE, RECEPTOR_TYPES),
        (CHAIN_PAIRING_SCALE, CHAIN_PAIRINGS),
        (CHAIN_ARM_SCALE, ("VJ", "VDJ")),
        (LOCUS_SCALE, LOCI),
        # The drawn states, in their stacks' order.
        (RECORDER_STATE_SCALE, ("mark", "unknown_mark", "unedited", "missing")),
    ]
    for scale, values in vocabularies:
        assert scale.order == tuple(values)
        assert set(scale.palette) == set(values)
        # Each value of a vocabulary takes a colour of its own.
        assert len(set(scale.palette.values())) == len(values)
    # Every state of a site but not applicable is drawn.
    assert set(RECORDER_STATE_SCALE.order) == set(STATES) - {"not_applicable"}
    # No value takes the missing colour of every scale, in every vocabulary.
    assert NO_VALUE_COLOR == CategoricalScale().missing_color
    assert RECEPTOR_TYPE_SCALE.palette[NO_RECEPTOR] == NO_VALUE_COLOR
    assert CHAIN_PAIRING_SCALE.palette[NO_RECEPTOR] == NO_VALUE_COLOR
    assert RECORDER_STATE_SCALE.palette["missing"] == NO_VALUE_COLOR
    assert RECORDER_STATE_SCALE.palette["unedited"] == MUTED_COLOR != NO_VALUE_COLOR
    # A pairing named like a receptor type takes the receptor type's colour.
    for value in ("ambiguous", "multichain"):
        assert CHAIN_PAIRING_SCALE.palette[value] == RECEPTOR_TYPE_SCALE.palette[value]
    # Legends name states in words, and tables keep their tokens.
    result = repertoire.calls["recorder_states"](show=False)
    try:
        (legend,) = _drawn_legends(result.figure)
        assert [text.get_text() for text in legend.get_texts()] == [
            "Mark",
            "Unknown mark",
            "Unedited",
            "Missing",
        ]
        assert "unknown_mark" in set(result.tables["site_states"]["state"])
    finally:
        result.close()
    # Ungrouped, the cells without a receptor are drawn in the grey of no value.
    qc = repertoire.store.plots.receptor_qc(repertoire.receptors, show=False)
    try:
        for panel in ("receptor_type", "chain_pairing"):
            ax = qc.axes[panel]
            names = [tick.get_text() for tick in ax.get_yticklabels()]
            colors = [to_hex(patch.get_facecolor()) for patch in ax.patches]
            assert colors[names.index(NO_RECEPTOR)] == NO_VALUE_COLOR
    finally:
        qc.close()


def test_capped_repertoire_panels_say_what_they_leave_out(repertoire) -> None:
    store, receptors = repertoire.store, repertoire.receptors
    usage = store.repertoire.gene_usage(receptors, group_by="sample")
    pairs = store.repertoire.gene_usage(receptors, genes=("v_call", "j_call"))
    capped = store.plots.gene_usage(usage, max_genes=1, show=False)
    pairing = store.plots.gene_pairing(pairs, max_genes=1, show=False)
    whole = store.plots.gene_usage(usage, show=False)
    states = store.plots.recorder_states(repertoire.alleles, max_sites=5, show=False)
    try:
        for locus, ax in capped.axes.items():
            held = usage.table.loc[usage.table["locus"] == locus, "v_call"].nunique()
            expected = "V gene" if held == 1 else f"V gene (1 of {held})"
            assert ax.get_ylabel() == expected
            assert whole.axes[locus].get_ylabel() == "V gene"
        tra = pairs.table[pairs.table["locus"] == "TRA"]
        assert pairing.axes["TRA"].get_ylabel() == (
            f"V gene (1 of {tra['v_call'].nunique()})"
        )
        assert pairing.axes["TRA"].get_xlabel() == (
            f"J gene (1 of {tra['j_call'].nunique()})"
        )
        sites = states.tables["site_states"].groupby(["integration", "site"]).ngroups
        assert states.axes["site_states"].get_title() == (
            f"The 5 of {sites} sites read in the most cells"
        )
    finally:
        for result in (capped, pairing, whole, states):
            result.close()


@pytest.mark.parametrize("name", REPERTOIRE_FIGURES)
def test_repertoire_figures_say_fraction_for_a_part_of_a_whole(
    repertoire, name
) -> None:
    result = repertoire.calls[name](show=False)
    try:
        texts = _texts(result)
        assert not [text for text in texts if re.search(r"\bshares?\b", text, re.I)]
        if name != "barcode_calls":
            assert any(text.startswith("Fraction of") for text in texts)
    finally:
        result.close()


N_GROUPS = 200


@pytest.fixture(scope="module")
def many_receptors(tmp_path_factory) -> SimpleNamespace:
    """200 cells, cell i of clone i // 2, each with a label of its own."""
    folder = tmp_path_factory.mktemp("many_receptor_groups")
    ids = [f"cell{i:03d}" for i in range(N_GROUPS)]
    store = write_cells_store(folder / "store.zarr", ids)
    contigs = write_clone_contigs(folder, "contigs.csv", ids, np.arange(N_GROUPS) // 2)
    receptors = store.repertoire.import_receptors(
        contigs, cell_selection=every_cell(store)
    )
    store.cells.insert(
        "own", np.asarray([f"g{i:03d}" for i in range(N_GROUPS)], dtype=object)
    )
    return SimpleNamespace(store=store, receptors=receptors)


def test_repertoire_figures_of_many_groups_stop_at_the_page_width(
    many_receptors, repertoire
) -> None:
    store, receptors, plots = (
        many_receptors.store,
        many_receptors.receptors,
        many_receptors.store.plots,
    )
    results = [
        plots.receptor_qc(receptors, group_by="own", show=False),
        plots.gene_usage(
            store.repertoire.gene_usage(receptors, group_by="own"), show=False
        ),
        plots.gene_pairing(
            store.repertoire.gene_usage(receptors, genes=("v_call", "j_call")),
            show=False,
        ),
        plots.spectratype(
            store.repertoire.spectratype(receptors, group_by="own"), show=False
        ),
        repertoire.calls["barcode_calls"](show=False),
        repertoire.store.plots.recorder_states(
            repertoire.alleles, max_sites=100, show=False
        ),
    ]
    try:
        assert [result.provenance.notes[0] for result in results] == list(
            REPERTOIRE_FIGURES
        )
        for result in results:
            width, height = result.figure.get_size_inches()
            assert width <= MAX_FIGURE_WIDTH_INCHES, result.provenance.notes
            assert height <= 22.0, result.provenance.notes
        # A heatmap of the 160 groups with an alpha chain names every nth, as
        # many as its axis holds once laid out, and at most 40.
        usage = results[1].tables["usage"]
        alpha = sorted(usage.loc[usage["locus"] == "TRA", "group"].unique())
        assert len(alpha) == 160
        results[1].figure.draw_without_rendering()
        for axis in (results[1].axes["TRA"].xaxis, results[1].axes["TRA"].yaxis):
            assert names_keep_apart(axis)
        step, names = axis_names(results[1].axes["TRA"].xaxis)
        assert names == (alpha[:1] if step is None else alpha[::step])
        assert len(names) <= 40
    finally:
        for result in results:
            result.close()


def test_repertoire_figures_with_nothing_to_draw_raise(repertoire) -> None:
    store, receptors = repertoire.store, repertoire.receptors
    if "unlabelled" not in store.cells.columns:
        insert_nullable_cell_column(
            store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
        )
    store.cells.insert("nobody", np.zeros(N_CELLS, dtype=bool), overwrite=True)
    nobody = store.snapshot_cell_selection("nobody")
    plots = store.plots
    barcodes = store.clones.load_barcodes(repertoire.barcodes)
    alleles = store.clones.load_alleles(repertoire.alleles)
    no_cells = SimpleNamespace(
        clones=SimpleNamespace(
            load_barcodes=lambda ref: replace(barcodes, cells=barcodes.cells.iloc[:0])
        )
    )
    no_records = SimpleNamespace(
        clones=SimpleNamespace(
            load_alleles=lambda ref: replace(alleles, records=alleles.records.iloc[:0])
        )
    )
    empty = [
        lambda: plots.receptor_qc(receptors, group_by="unlabelled", show=False),
        lambda: plots.gene_usage(
            store.repertoire.gene_usage(receptors, cell_selection=nobody), show=False
        ),
        lambda: plots.gene_pairing(
            store.repertoire.gene_usage(
                receptors, genes=("v_call", "j_call"), cell_selection=nobody
            ),
            show=False,
        ),
        lambda: plots.spectratype(
            store.repertoire.spectratype(receptors, cell_selection=nobody), show=False
        ),
        lambda: barcode_plots.barcode_calls(no_cells, repertoire.barcodes, show=False),
        lambda: recorder_plots.recorder_states(
            no_records, repertoire.alleles, show=False
        ),
    ]
    assert len(empty) == len(REPERTOIRE_FIGURES)
    figures = set(plt.get_fignums())
    for call in empty:
        with pytest.raises(ValueError, match="so there is nothing to plot$"):
            call()
    assert set(plt.get_fignums()) == figures


@pytest.mark.parametrize("name", REPERTOIRE_FIGURES)
def test_each_repertoire_figure_draws_in_a_composite_with_its_legends_once(
    repertoire, name
) -> None:
    drawn = repertoire.calls[name](show=False)
    panels = list(drawn.axes)
    drawn.close()
    figure = plt.figure(figsize=(12, 6), layout="constrained")
    grid = figure.subplots(1, len(panels), squeeze=False)[0]
    try:
        result = repertoire.calls[name](
            target=dict(zip(panels, grid, strict=True)), show=False
        )
        composite = compose_results(figure, [result])
        figure.canvas.draw()
        legends = list(figure.legends) + _drawn_legends(figure)
        kinds = [legend.kind for legend in result.legends]
        assert len(legends) == kinds.count("categorical")
        colorbars = [ax for ax in figure.axes if is_colorbar_axes(ax)]
        assert len(colorbars) == kinds.count("colorbar")
        assert len(composite.legends) == len(result.legends)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("theme", ["notebook", "dark"])
@pytest.mark.parametrize("name", REPERTOIRE_FIGURES)
def test_repertoire_figures_take_their_themes_background(
    repertoire, name, theme
) -> None:
    result = repertoire.calls[name](theme=theme, show=False)
    try:
        alpha = 1.0 if theme == "notebook" else 0.0
        assert result.figure.patch.get_alpha() == alpha
        assert {ax.patch.get_alpha() for ax in result.figure.axes} == {alpha}
    finally:
        result.close()


def test_repertoire_figure_tables_hold_their_records_values(repertoire) -> None:
    store = repertoire.store
    calls = repertoire.calls
    results = {name: calls[name](show=False) for name in REPERTOIRE_FIGURES}
    try:
        # A figure of a summary returns the summary's own table.
        usage = results["gene_usage"]
        assert usage.tables["usage"].equals(
            store.repertoire.gene_usage(repertoire.receptors, group_by="sample").table
        )
        lengths = results["spectratype"]
        assert lengths.tables["spectratype"].equals(
            store.repertoire.spectratype(repertoire.receptors, group_by="sample").table
        )
        # The receptors' cells of each sample, by receptor type and pairing,
        # and their chains of each arm, three or more together.
        cells = store.repertoire.load_receptors(repertoire.receptors, fields=()).cells
        samples = pd.Series(TRUTH.samples, index=TRUTH.cell_ids).loc[cells.index]
        qc = results["receptor_qc"]
        for column in ("receptor_type", "chain_pairing"):
            expected = pd.crosstab(samples.to_numpy(), cells[column].to_numpy())
            found = qc.tables[column].pivot(
                index="group", columns=column, values="n_cells"
            )
            for group in expected.index:
                for value in found.columns:
                    held = expected.loc[group].get(value, 0)
                    assert found.loc[group, value] == held, (column, group, value)
        chains = qc.tables["chains_per_cell"]
        for arm, column in (("VJ", "n_vj"), ("VDJ", "n_vdj")):
            capped = np.minimum(cells[column].to_numpy(), 3)
            assert chains[arm].tolist() == [int((capped == n).sum()) for n in range(4)]
        # Barcodes per cell, and pairs by UMIs, from the record's rows.
        loaded = store.clones.load_barcodes(repertoire.barcodes)
        calls_table = results["barcode_calls"].tables
        per_cell = loaded.cells["n_barcodes"].value_counts().sort_index()
        assert calls_table["barcodes_per_cell"]["n_barcodes"].tolist() == list(
            per_cell.index
        )
        assert calls_table["barcodes_per_cell"]["n_cells"].tolist() == list(per_cell)
        pairs = loaded.pairs().dropna(subset=["umis"])
        expected_pairs = pairs.groupby(["library", "umis"], observed=True).size()
        found_pairs = calls_table["umis_per_barcode"].set_index(["library", "umis"])
        assert found_pairs["n_pairs"].to_dict() == {
            (str(library), int(umis)): int(n)
            for (library, umis), n in expected_pairs.items()
        }
        # The recorder's cells of each site and state, as its record counts them.
        site_states = store.clones.load_alleles(repertoire.alleles).site_states()
        found_states = results["recorder_states"].tables["site_states"]
        assert found_states.drop(columns="share").equals(site_states)
        # Each share is the state's cells over the site's cells but those of
        # an integration without the site.
        applicable = site_states[site_states["state"] != "not_applicable"]
        totals = applicable.groupby(["integration", "site"])["n_cells"].sum()
        for row in found_states.itertuples():
            total = totals[(row.integration, row.site)]
            if row.state == "not_applicable" or not total:
                assert np.isnan(row.share)
            else:
                assert row.share == pytest.approx(row.n_cells / total)
    finally:
        for result in results.values():
            result.close()


def _patch_colors(ax) -> list[str]:
    return [to_hex(patch.get_facecolor()) for patch in ax.patches]


def test_receptor_qc_draws_the_categories_with_cells_in_their_vocabularys_colours(
    repertoire,
) -> None:
    from cytearc.plotting._style import CHAIN_ARM_SCALE, CHAIN_PAIRING_SCALE

    plain = repertoire.store.plots.receptor_qc(repertoire.receptors, show=False)
    grouped = repertoire.calls["receptor_qc"](show=False)
    try:
        for panel, vocabulary in (
            ("receptor_type", RECEPTOR_TYPE_SCALE),
            ("chain_pairing", CHAIN_PAIRING_SCALE),
        ):
            # The truth's cells hold some of the vocabulary, such as no
            # "two full chains"; the panels draw those, in its order.
            counts = plain.tables[panel].set_index(panel)["n_cells"]
            held = [value for value in vocabulary.order if counts[value] > 0]
            assert len(held) < len(vocabulary.order) or panel == "receptor_type"
            ax = plain.axes[panel]
            assert [tick.get_text() for tick in ax.get_yticklabels()] == held
            assert _patch_colors(ax) == [vocabulary.palette[value] for value in held]
            assert [patch.get_width() for patch in ax.patches] == [
                counts[value] for value in held
            ]
            # Grouped, the legend lists the same categories in the same colours.
            legend = grouped.axes[panel].get_legend()
            assert [text.get_text() for text in legend.get_texts()] == held
            assert [
                to_hex(handle.get_facecolor()) for handle in legend.legend_handles
            ] == [vocabulary.palette[value] for value in held]
        # The chains of each arm stand side by side in the arm's colour, the
        # VJ chains at the left, at the counts of the table.
        chains = plain.tables["chains_per_cell"]
        ax = plain.axes["chains_per_cell"]
        for arm in ("VJ", "VDJ"):
            bars = sorted(
                (patch.get_x(), patch.get_height())
                for patch in ax.patches
                if to_hex(patch.get_facecolor()) == CHAIN_ARM_SCALE.palette[arm]
            )
            assert [height for _x, height in bars] == chains[arm].tolist()
        vj = min(
            p.get_x()
            for p in ax.patches
            if to_hex(p.get_facecolor()) == CHAIN_ARM_SCALE.palette["VJ"]
        )
        vdj = min(
            p.get_x()
            for p in ax.patches
            if to_hex(p.get_facecolor()) == CHAIN_ARM_SCALE.palette["VDJ"]
        )
        assert vj < vdj
        # A colour of the figure means one thing: the arms take none of the
        # receptor types' or pairings' colours.
        assert not set(CHAIN_ARM_SCALE.palette.values()) & (
            set(RECEPTOR_TYPE_SCALE.palette.values())
            | set(CHAIN_PAIRING_SCALE.palette.values())
        )
        # The three panels stand in one column, the first on top.
        boxes = [grouped.axes[panel].get_position() for panel in grouped.axes]
        assert len({round(box.x0, 3) for box in boxes}) == 1
        assert [box.y0 for box in boxes] == sorted(
            (box.y0 for box in boxes), reverse=True
        )
    finally:
        plain.close()
        grouped.close()


def test_bars_of_a_locus_take_the_locus_colour(repertoire) -> None:
    from cytearc.plotting._style import LOCUS_SCALE

    store, receptors = repertoire.store, repertoire.receptors
    usage = store.plots.gene_usage(store.repertoire.gene_usage(receptors), show=False)
    lengths = store.plots.spectratype(
        store.repertoire.spectratype(receptors), show=False
    )
    try:
        for result in (usage, lengths):
            assert len(result.axes) == 5
            for locus, ax in result.axes.items():
                assert ax.patches, locus
                assert set(_patch_colors(ax)) == {LOCUS_SCALE.palette[locus]}
        # Bars of gene usage stand three panels to a row.
        rows = [round(ax.get_position().y0, 3) for ax in usage.axes.values()]
        assert rows.count(max(rows)) == 3
    finally:
        usage.close()
        lengths.close()


def test_a_spectratype_of_some_cells_keeps_each_groups_colour(repertoire) -> None:
    store, receptors = repertoire.store, repertoire.receptors
    store.cells.insert("not_monocyte", TRUTH.cell_types != "Monocyte", overwrite=True)
    full = store.plots.spectratype(
        store.repertoire.spectratype(receptors, group_by="cell_type"), show=False
    )
    rest = store.plots.spectratype(
        store.repertoire.spectratype(
            receptors,
            group_by="cell_type",
            cell_selection=store.snapshot_cell_selection("not_monocyte"),
        ),
        show=False,
    )
    try:
        palette = full.scales[0].palette
        assert list(palette) == ["Monocyte", "Neutrophil", "Undifferentiated"]
        # Without the monocytes, the first label, the other two keep the
        # colours they take beside it, which colours over the groups drawn
        # would not give them.
        assert rest.scales[0].order == ("Neutrophil", "Undifferentiated")
        assert rest.scales[0].palette == {
            "Neutrophil": palette["Neutrophil"],
            "Undifferentiated": palette["Undifferentiated"],
        }
        assert palette["Neutrophil"] != palette["Monocyte"]
        drawn = {
            line.get_label(): to_hex(line.get_color())
            for ax in rest.axes.values()
            for line in ax.get_lines()
        }
        assert drawn == rest.scales[0].palette
    finally:
        full.close()
        rest.close()


def test_barcode_libraries_take_colours_over_every_library(repertoire) -> None:
    loaded = repertoire.store.clones.load_barcodes(repertoire.barcodes)
    # A library listed first that holds no record draws in neither panel,
    # and keeps its colour from the others.
    extra = loaded.libraries.iloc[:1].assign(library="aaa", n_records=0)
    padded = replace(loaded, libraries=pd.concat([extra, loaded.libraries]))
    store = SimpleNamespace(clones=SimpleNamespace(load_barcodes=lambda ref: padded))
    every = repertoire.calls["barcode_calls"](show=False)
    result = barcode_plots.barcode_calls(store, repertoire.barcodes, show=False)
    try:
        assert every.scales[0].palette == {"bartab": "#1f77b4", "trex": "#ff7f0e"}
        assert result.scales[0].palette == {"bartab": "#ff7f0e", "trex": "#279e68"}
        for panel in ("umis_per_barcode", "reads_per_molecule"):
            for line in result.axes[panel].get_lines():
                if line.get_xdata().size:
                    assert (
                        to_hex(line.get_color())
                        == (result.scales[0].palette[line.get_label()])
                    )
        # The one legend stands beside the last panel that draws libraries.
        (legend,) = _drawn_legends(every.figure)
        assert legend.axes is every.axes["reads_per_molecule"]
    finally:
        every.close()
        result.close()


def _recorder_store(sites: list[tuple[str, str, dict[str, int]]]) -> SimpleNamespace:
    """A store whose alleles hold each site's cells by state."""
    rows = [
        {"integration": integration, "site": site, "state": state, "n_cells": n}
        for integration, site, cells in sites
        for state, n in cells.items()
    ]
    loaded = SimpleNamespace(
        site_states=lambda: pd.DataFrame(rows), cells=pd.DataFrame(index=range(10))
    )
    return SimpleNamespace(clones=SimpleNamespace(load_alleles=lambda ref: loaded))


def test_recorder_states_name_the_sites_drawn_and_rank_sites_by_reads(
    repertoire,
) -> None:
    # 60 sites, site i read in 60 - i cells: the default draws the 40 read
    # most, of which its axis names every nth, as many as its width holds.
    many = _recorder_store(
        [
            ("int1", f"s{i:02d}", {"mark": 60 - i, "missing": 3, "unedited": 0})
            for i in range(60)
        ]
    )
    result = recorder_plots.recorder_states(many, repertoire.alleles, show=False)
    try:
        ax = result.axes["site_states"]
        result.figure.draw_without_rendering()
        names_every_nth_item(ax.xaxis, [f"int1-s{i:02d}" for i in range(40)])
        assert ax.get_title() == "The 40 of 60 sites read in the most cells"
        assert result.provenance.extras["drawn_sites"] == 40
    finally:
        result.close()
    # A site of many missing cells and few reads gives way to one read more.
    ranked = _recorder_store(
        [
            ("int1", "a", {"mark": 1, "missing": 20}),
            ("int1", "b", {"mark": 3, "missing": 0}),
        ]
    )
    result = recorder_plots.recorder_states(
        ranked, repertoire.alleles, max_sites=1, show=False
    )
    try:
        ax = result.axes["site_states"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["int1-b"]
        assert ax.get_title() == "The 1 of 2 sites read in the most cells"
    finally:
        result.close()
    # A figure that draws every site keeps its title.
    every = repertoire.calls["recorder_states"](show=False)
    try:
        assert every.axes["site_states"].get_title() == "Recorder states per site"
    finally:
        every.close()


def _gene_usage(source, loci: dict[str, int], fractions: dict[str, float]):
    """A gene usage record of two samples, ``loci[locus]`` genes in each locus.

    Gene g of a locus holds the locus's fraction divided by g + 1 in each
    sample, so the locus's largest fraction is ``fractions[locus]``.
    """
    from cytearc.repertoire import GeneUsage

    rows = [
        {
            "group": group,
            "locus": locus,
            "v_call": f"{locus}V{gene + 1}",
            "n_cells": n_genes - gene,
            "fraction": fractions[locus] / (gene + 1),
        }
        for group in ("s1", "s2")
        for locus, n_genes in loci.items()
        for gene in range(n_genes)
    ]
    return GeneUsage(
        table=pd.DataFrame(rows),
        group_keys=pd.DataFrame(
            {"sample": ["s1", "s2"]}, index=pd.Index(["s1", "s2"], name="group")
        ),
        source=source,
        genes=("v_call",),
        level="gene",
        slots=("VJ_1", "VDJ_1"),
        count="cells",
        group_by=("sample",),
        groups=None,
        cell_selection=None,
        n_cells=10,
        diagnostics={},
    )


def test_locus_heatmaps_share_the_largest_fraction_of_any_locus(repertoire) -> None:
    from cytearc import plotting

    # The first locus, IGH, holds 0.4 at most, and the third, TRA, 0.9.
    usage = _gene_usage(
        repertoire.receptors,
        {"IGH": 2, "IGK": 1, "TRA": 3, "TRB": 2, "TRG": 1},
        {"IGH": 0.4, "IGK": 0.2, "TRA": 0.9, "TRB": 0.6, "TRG": 0.3},
    )
    # A figure of a record of cells reads nothing from the store.
    result = plotting.gene_usage(object(), usage, show=False)
    try:
        for locus, ax in result.axes.items():
            (image,) = ax.images
            assert (image.norm.vmin, image.norm.vmax) == (0.0, 0.9), locus
        # Heatmaps stand two to a row, in three rows.
        rows = [ax.get_subplotspec().rowspan.start for ax in result.axes.values()]
        assert rows == [0, 0, 1, 1, 2]
    finally:
        result.close()
    # A row of heatmaps grows with the genes drawn and stops where an axis
    # stops naming them, at 40: five loci of 60 genes stand in three rows.
    tall = _gene_usage(
        repertoire.receptors,
        dict.fromkeys(("IGH", "IGK", "TRA", "TRB", "TRG"), 60),
        dict.fromkeys(("IGH", "IGK", "TRA", "TRB", "TRG"), 0.5),
    )
    for max_genes, named, axis in (
        (10, 10, "V gene (10 of 60)"),
        (40, 40, "V gene (40 of 60)"),
        (60, 40, "V gene"),
    ):
        result = plotting.gene_usage(object(), tall, max_genes=max_genes, show=False)
        try:
            width, height = result.figure.get_size_inches()
            assert width == MAX_FIGURE_WIDTH_INCHES
            assert height == pytest.approx(3 * (1.2 + 0.18 * named))
            assert result.axes["TRA"].get_ylabel() == axis
        finally:
            result.close()


def _usage_of_counts(source, counts: dict[str, dict[str, int]], group_by: tuple):
    """A gene usage record of TRB V genes, ``counts[group][gene]`` cells each."""
    from cytearc.repertoire import GeneUsage

    rows = [
        {
            "group": group,
            "locus": "TRB",
            "v_call": gene,
            "n_cells": n_cells,
            "fraction": n_cells / sum(genes.values()),
        }
        for group, genes in counts.items()
        for gene, n_cells in genes.items()
    ]
    names = pd.Index(list(counts), name="group")
    return GeneUsage(
        table=pd.DataFrame(rows),
        group_keys=pd.DataFrame({"cell_type": list(counts)}, index=names),
        source=source,
        genes=("v_call",),
        level="gene",
        slots=("VDJ_1",),
        count="cells",
        group_by=group_by,
        groups=None,
        cell_selection=None,
        n_cells=sum(sum(genes.values()) for genes in counts.values()),
        diagnostics={},
    )


def test_gene_usage_draws_the_genes_that_each_group_uses_most(repertoire) -> None:
    from cytearc import plotting

    # The naive cells use TRBV1 to TRBV5 most, and the MAIT cells TRBV6-4,
    # which ranks sixth over both groups.
    common = {"TRBV1": 100, "TRBV2": 90, "TRBV3": 80, "TRBV4": 70, "TRBV5": 60}
    counts = {
        "naive": common,
        "MAIT": {**dict.fromkeys(common, 1), "TRBV6-4": 20},
    }
    usage = _usage_of_counts(repertoire.receptors, counts, ("cell_type",))
    result = plotting.gene_usage(object(), usage, max_genes=3, show=False)
    try:
        ax = result.axes["TRB"]
        # Each group's first gene, then the next best, by their chains over
        # both groups.
        genes = [tick.get_text() for tick in ax.get_yticklabels()]
        assert genes == ["TRBV1", "TRBV2", "TRBV6-4"]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["MAIT", "naive"]
        assert ax.get_ylabel() == "V gene (3 of 6)"
        (image,) = ax.images
        drawn = np.ma.filled(image.get_array(), np.nan)
        np.testing.assert_allclose(drawn[2], [20 / 25, 0.0])
        np.testing.assert_allclose(drawn[0], [1 / 25, 100 / 400])
    finally:
        result.close()
    # Without groups, the genes with the most chains are drawn.
    merged = {gene: common.get(gene, 0) + n for gene, n in counts["MAIT"].items()}
    alone = _usage_of_counts(repertoire.receptors, {"all cells": merged}, ())
    result = plotting.gene_usage(object(), alone, max_genes=3, show=False)
    try:
        ax = result.axes["TRB"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "TRBV1",
            "TRBV2",
            "TRBV3",
        ]
        assert ax.get_ylabel() == "V gene (3 of 6)"
    finally:
        result.close()


def test_gene_usage_ranks_a_groups_tied_genes_by_their_chains_overall(
    repertoire,
) -> None:
    from cytearc import plotting

    # The MAIT cells' one cell of TRBV2 and of TRBV9 tie; TRBV9 has more chains
    # over both groups, so it ranks second among the MAIT cells' genes, and
    # TRBV2, third, does not displace the naive cells' third gene.
    counts = {
        "naive": {"TRBV9": 50, "TRBV3": 40, "TRBV4": 30, "TRBV2": 10},
        "MAIT": {"TRBV6-4": 20, "TRBV2": 1, "TRBV9": 1},
    }
    usage = _usage_of_counts(repertoire.receptors, counts, ("cell_type",))
    result = plotting.gene_usage(object(), usage, max_genes=4, show=False)
    try:
        ax = result.axes["TRB"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "TRBV9",
            "TRBV3",
            "TRBV4",
            "TRBV6-4",
        ]
    finally:
        result.close()


def test_gene_usage_breaks_ties_of_first_genes_by_their_chains(repertoire) -> None:
    from cytearc import plotting

    # Three groups, each with a gene of its own first, and fewer genes drawn
    # than groups: the first genes with the most chains over all groups.
    counts = {
        "a": {"TRBV1": 5, "TRBV9": 1},
        "b": {"TRBV2": 30, "TRBV9": 1},
        "c": {"TRBV3": 20, "TRBV9": 1},
    }
    usage = _usage_of_counts(repertoire.receptors, counts, ("cell_type",))
    result = plotting.gene_usage(object(), usage, max_genes=2, show=False)
    try:
        ax = result.axes["TRB"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "TRBV2",
            "TRBV3",
        ]
        assert ax.get_ylabel() == "V gene (2 of 4)"
    finally:
        result.close()


# The figures of variant counts, informative variants and genotype calls keep
# the same conventions, on the truth's mitochondrial reads and GoT UMIs.


@pytest.fixture(scope="module")
def variants(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("variant_figure_conventions")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    artifacts = import_truth_variants(store, TRUTH, folder)
    return SimpleNamespace(
        store=store, calls=variant_figure_calls(store, artifacts), **artifacts
    )


# The input that each figure records under its argument's name, the
# fixture's artifact it is, and whether the figure records a grouping.
_VARIANT_INPUTS = {
    "variant_depth": ("counts", "counts", True),
    "variant_statistics": ("variants", "informative", False),
    "allele_frequencies": ("variants", "informative", False),
    "genotypes": ("genotypes", "genotypes", True),
}
# The panels of each variant figure, for drawing it on axes of a caller's.
_VARIANT_PANELS = {
    "variant_depth": ["sites", "cells"],
    "variant_statistics": ["variants"],
    "allele_frequencies": ["frequencies"],
    "genotypes": ["calls", "layout"],
}


def _painted_colors(ax) -> set[str]:
    """The colours that a panel paints: bars, outlines, lines, points and a strip's colours."""
    colors = {
        to_hex(patch.get_facecolor() if patch.get_fill() else patch.get_edgecolor())
        for patch in ax.patches
    }
    colors |= {to_hex(line.get_color()) for line in ax.get_lines()}
    for collection in ax.collections:
        colors |= {to_hex(color) for color in collection.get_facecolors()}
    for image in ax.images:
        colors |= {to_hex(color) for color in getattr(image.get_cmap(), "colors", ())}
    return colors


def _legends_with_insets(figure) -> list[Legend]:
    """The legends of a figure's panels and of the insets they hold, such as a clone strip."""
    found = []
    for ax in figure.axes:
        for holder in [ax, *ax.child_axes]:
            found.extend(
                artist for artist in holder.get_children() if isinstance(artist, Legend)
            )
    return found


def test_every_variant_figure_is_checked() -> None:
    assert set(VARIANT_FIGURES) == set(variant_plots.__all__)
    assert set(_VARIANT_INPUTS) == set(_VARIANT_PANELS) == set(VARIANT_FIGURES)


@pytest.mark.parametrize("name", VARIANT_FIGURES)
def test_variant_legends_are_reported_with_their_scales_and_agree_with_the_drawing(
    variants, name
) -> None:
    result = variants.calls[name](show=False)
    try:
        assert result.legends, name
        assert {legend.kind for legend in result.legends} <= {
            "categorical",
            "colorbar",
        }
        assert len(result.scales) == len(result.legends)
        parts = _shared_legend_parts(result)
        assert parts is not None and len(parts) == len(result.legends)
        categorical = [
            (legend, scale)
            for legend, scale in zip(result.legends, result.scales, strict=True)
            if legend.kind == "categorical"
        ]
        drawn = _legends_with_insets(result.figure)
        assert len(drawn) == len(categorical)
        for (legend, scale), artist in zip(categorical, drawn, strict=True):
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
            assert set(colors) <= _painted_colors(artist.axes)
        colorbar_legends = [
            legend for legend in result.legends if legend.kind == "colorbar"
        ]
        colorbars = [ax for ax in result.figure.axes if is_colorbar_axes(ax)]
        assert len(colorbars) == len(colorbar_legends)
        for legend, colorbar in zip(colorbar_legends, colorbars, strict=True):
            assert colorbar.get_ylabel() == legend.label
            assert colorbar.get_ylim() == pytest.approx(
                (legend.extras["vmin"], legend.extras["vmax"])
            )
        assert result.provenance.notes == (name,)
        key, ref, grouped = _VARIANT_INPUTS[name]
        assert result.provenance.extras[key] == getattr(variants, ref).to_dict()
        if grouped:
            assert result.provenance.extras["grouping"] == "sample"
        if name == "allele_frequencies":
            assert result.provenance.extras["clones"] == variants.clones.to_dict()
    finally:
        result.close()


@pytest.mark.parametrize("name", VARIANT_FIGURES)
def test_a_hidden_variant_legend_is_still_reported(variants, name) -> None:
    shown = variants.calls[name](show=False)
    hidden = variants.calls[name](show=False, show_legend=False)
    try:
        assert hidden.legends == shown.legends
        assert hidden.scales == shown.scales
        assert not _legends_with_insets(hidden.figure)
        assert not any(is_colorbar_axes(ax) for ax in hidden.figure.axes)
    finally:
        shown.close()
        hidden.close()


def _rows_of_parts(ax) -> list[list[tuple[float, str]]]:
    """Each horizontal bar's parts with a width, from the left: their widths and colours, top bar first."""
    bars: dict[float, list[tuple[float, float, str]]] = {}
    for patch in ax.patches:
        if patch.get_width() > 0:
            centre = round(patch.get_y() + patch.get_height() / 2, 6)
            bars.setdefault(centre, []).append(
                (patch.get_x(), patch.get_width(), to_hex(patch.get_facecolor()))
            )
    return [
        [(width, color) for _x, width, color in sorted(parts)]
        for _centre, parts in sorted(bars.items(), reverse=True)
    ]


def test_genotype_bars_draw_their_first_call_at_the_left(variants) -> None:
    result = variants.store.plots.genotypes(
        variants.genotypes, group_by="sample", show=False
    )
    try:
        table = result.tables["calls"]
        scale = result.scales[0]
        bars = list(dict.fromkeys(zip(table["site"], table["group"], strict=True)))
        rows = _rows_of_parts(result.axes["calls"])
        assert len(rows) == len(bars)
        for parts, (site, group) in zip(rows, bars, strict=True):
            # From the left, the bar's calls with cells in the legend's
            # order, each its fraction of the group's cells.
            counted = table[(table["site"] == site) & (table["group"] == group)]
            counted = counted.set_index("call")["n_cells"]
            held = [call for call in scale.order if counted[call] > 0]
            assert [color for _width, color in parts] == [
                scale.palette[call] for call in held
            ]
            assert [width for width, _color in parts] == pytest.approx(
                [counted[call] / counted.sum() for call in held]
            )
        names = [tick.get_text() for tick in result.axes["calls"].get_yticklabels()]
        assert names == [f"{site}, {group}" for site, group in bars]
    finally:
        result.close()


def test_genotype_calls_take_one_palette_with_one_grey_for_no_value(variants) -> None:
    from cytearc.plotting._contracts import CategoricalScale
    from cytearc.plotting._style import (
        GENOTYPE_CALL_SCALE,
        MUTED_COLOR,
        NO_VALUE_COLOR,
        RECORDER_STATE_SCALE,
    )
    from cytearc.variants import CALLS

    assert GENOTYPE_CALL_SCALE.order == CALLS
    assert set(GENOTYPE_CALL_SCALE.palette) == set(CALLS)
    assert len(set(GENOTYPE_CALL_SCALE.palette.values())) == len(CALLS)
    # No evidence takes the grey of no value, and evidence that a rule did
    # not call the muted grey, as the recorder's missing and unedited sites.
    assert GENOTYPE_CALL_SCALE.palette["missing"] == NO_VALUE_COLOR
    assert NO_VALUE_COLOR == CategoricalScale().missing_color
    assert GENOTYPE_CALL_SCALE.palette["filtered"] == MUTED_COLOR
    assert RECORDER_STATE_SCALE.palette["unedited"] == MUTED_COLOR
    # Legends name calls in words, and tables keep their tokens.
    assert GENOTYPE_CALL_SCALE.labels == {
        "wild_type": "Wild type",
        "heterozygous": "Heterozygous",
        "homozygous_alternative": "Homozygous alternative",
        "alternative_present": "Alternative present",
        "missing": "Missing",
        "filtered": "Filtered",
    }
    result = variants.calls["genotypes"](show=False)
    try:
        (legend,) = _legends_with_insets(result.figure)
        assert legend.get_title().get_text() == "Call"
        held = [call for call in CALLS if call in set(result.tables["cells"]["call"])]
        assert [text.get_text() for text in legend.get_texts()] == [
            GENOTYPE_CALL_SCALE.labels[call] for call in held
        ]
        assert "wild_type" in set(result.tables["calls"]["call"])
        assert "wild_type" in set(result.tables["cells"]["call"])
    finally:
        result.close()


@pytest.mark.parametrize("name", VARIANT_FIGURES)
def test_variant_figures_say_fraction_for_a_part_of_a_whole(variants, name) -> None:
    result = variants.calls[name](show=False)
    try:
        texts = _texts(result)
        assert not [text for text in texts if re.search(r"\bshares?\b", text, re.I)]
        if name == "genotypes":
            assert "Fraction of cells" in texts
    finally:
        result.close()


def test_capped_variant_panels_say_what_they_leave_out(variants) -> None:
    store = variants.store
    with_clone = int(
        store.clones.load_clones(variants.clones).cells["clone"].notna().sum()
    )
    capped = store.plots.allele_frequencies(
        variants.informative, clones=variants.clones, max_cells=10, show=False
    )
    whole = variants.calls["allele_frequencies"](show=False)
    try:
        assert capped.axes["frequencies"].get_xlabel() == (
            f"Cells (10 of {with_clone})"
        )
        assert len(capped.tables["cells"]) == 10
        assert whole.axes["frequencies"].get_xlabel() == "Cells"
    finally:
        capped.close()
        whole.close()


N_VARIANT_GROUPS = 200


@pytest.fixture(scope="module")
def many_variant_groups(tmp_path_factory) -> SimpleNamespace:
    """200 cells read at three GoT targets, each cell with a label of its own."""
    from tests.variant_truth import write_genotype_tables

    folder = tmp_path_factory.mktemp("many_variant_groups")
    ids = [f"cell{i:03d}" for i in range(N_VARIANT_GROUPS)]
    store = write_cells_store(folder / "store.zarr", ids)
    store.cells.insert(
        "own", np.asarray([f"g{i:03d}" for i in range(N_VARIANT_GROUPS)], dtype=object)
    )
    umis = np.random.default_rng(11).integers(0, 4, size=(N_VARIANT_GROUPS, 3, 2))
    counts = store.variants.import_counts(
        write_genotype_tables(umis, ids, folder / "got"),
        cell_selection=every_cell(store),
    )
    genotypes = store.variants.genotypes(counts, method="threshold", min_mutant_umis=1)
    return SimpleNamespace(store=store, counts=counts, genotypes=genotypes)


def test_variant_figures_of_many_groups_stop_at_the_page_width(
    many_variant_groups, variants
) -> None:
    plots = many_variant_groups.store.plots
    results = [
        plots.variant_depth(many_variant_groups.counts, group_by="own", show=False),
        variants.calls["variant_statistics"](show=False),
        variants.calls["allele_frequencies"](show=False),
        plots.genotypes(many_variant_groups.genotypes, group_by="own", show=False),
    ]
    try:
        assert [result.provenance.notes[0] for result in results] == list(
            VARIANT_FIGURES
        )
        for result in results:
            width, height = result.figure.get_size_inches()
            assert width <= MAX_FIGURE_WIDTH_INCHES, result.provenance.notes
            assert height <= 22.0, result.provenance.notes
        # 200 groups' histograms list 80 in their legend and say so, and
        # 600 bars of three sites and 200 groups name every nth, at most 40.
        depth = results[0]
        assert depth.axes["cells"].get_legend().get_title().get_text() == (
            "own (80 of 200)"
        )
        assert depth.legends[0].label == "own (80 of 200)"
        assert len(depth.axes["cells"].patches) == N_VARIANT_GROUPS
        calls = results[3].axes["calls"]
        assert len({round(patch.get_y(), 6) for patch in calls.patches}) == 600
        results[3].figure.draw_without_rendering()
        bars = (
            results[3]
            .tables["calls"]
            .drop_duplicates(["site", "group"])[["site", "group"]]
        )
        assert (
            names_every_nth_item(
                calls.yaxis, [f"{site}, {group}" for site, group in bars.to_numpy()]
            )
            == 15
        )
    finally:
        for result in results:
            result.close()


@pytest.mark.parametrize("n_groups", [20, 200])
def test_depth_legends_of_many_groups_fit_beside_their_panel(
    many_variant_groups, n_groups
) -> None:
    store = many_variant_groups.store
    column = f"every_{n_groups}"
    if column not in store.cells.columns:
        store.cells.insert(
            column,
            np.asarray(
                [f"g{cell % n_groups:03d}" for cell in range(N_VARIANT_GROUPS)],
                dtype=object,
            ),
        )
    result = store.plots.variant_depth(
        many_variant_groups.counts, group_by=column, show=False
    )
    try:
        figure = result.figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend = result.axes["cells"].get_legend()
        box = legend.get_window_extent(renderer)
        # The legend stands within the figure, in at most four columns, and
        # leaves both panels room to draw.
        assert box.y0 >= 0 and box.y1 <= figure.bbox.height
        assert box.x1 <= figure.bbox.width
        assert legend._ncols <= 4
        for ax in result.axes.values():
            assert ax.get_position().height * figure.get_figheight() >= 1.5
        assert figure.get_size_inches()[0] == MAX_FIGURE_WIDTH_INCHES
    finally:
        result.close()


def test_variant_statistics_count_the_variants_they_draw(variants) -> None:
    result = variants.calls["variant_statistics"](show=False)
    try:
        table = result.tables["variants"]
        shown = table["strand_correlation"].notna()
        # Variants without a strand correlation are scored but not drawn.
        assert shown.sum() < len(table)
        assert result.axes["variants"].get_title() == (
            f"{int((table['informative'] & shown).sum())} of {int(shown.sum())} "
            "variants informative"
        )
        drawn = sum(len(c.get_offsets()) for c in result.axes["variants"].collections)
        assert drawn == shown.sum()
    finally:
        result.close()


def test_variant_figures_with_nothing_to_draw_raise(variants, tmp_path) -> None:
    import warnings

    store = variants.store
    if "unlabelled" not in store.cells.columns:
        insert_nullable_cell_column(
            store, "unlabelled", TRUTH.samples, np.ones(N_CELLS, dtype=bool)
        )
    store.cells.insert("only_first_cell", np.arange(N_CELLS) == 0, overwrite=True)
    nobody = tmp_path / "nobody.tsv"
    nobody.write_text("BC\twtUMIs\tmutUMIs\nnobody\t1\t0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unnamed = store.variants.import_counts(nobody, cell_selection=every_cell(store))
    # Scored over one cell, no variant has a strand correlation; and a
    # threshold no variant passes leaves none informative.
    one_cell = store.variants.informative(
        variants.counts,
        cell_selection=store.snapshot_cell_selection("only_first_cell"),
        min_vmr=0.0,
    )
    none_informative = store.variants.informative(variants.counts, min_vmr=1e9)
    plots = store.plots
    empty = [
        lambda: plots.variant_depth(unnamed, show=False),
        lambda: plots.variant_statistics(one_cell, show=False),
        lambda: plots.allele_frequencies(none_informative, show=False),
        lambda: plots.genotypes(variants.genotypes, group_by="unlabelled", show=False),
    ]
    assert len(empty) == len(VARIANT_FIGURES)
    figures = set(plt.get_fignums())
    for call in empty:
        with pytest.raises(ValueError, match="so there is nothing to plot$"):
            call()
    assert set(plt.get_fignums()) == figures


@pytest.mark.parametrize("name", VARIANT_FIGURES)
def test_each_variant_figure_draws_in_a_composite_with_its_legends_once(
    variants, name
) -> None:
    panels = _VARIANT_PANELS[name]
    figure = plt.figure(figsize=(12, 6), layout="constrained")
    grid = figure.subplots(1, len(panels), squeeze=False)[0]
    try:
        result = variants.calls[name](
            target=dict(zip(panels, grid, strict=True)), show=False
        )
        composite = compose_results(figure, [result])
        figure.canvas.draw()
        legends = list(figure.legends) + _legends_with_insets(figure)
        kinds = [legend.kind for legend in result.legends]
        assert len(legends) == kinds.count("categorical")
        colorbars = [ax for ax in figure.axes if is_colorbar_axes(ax)]
        assert len(colorbars) == kinds.count("colorbar")
        assert len(composite.legends) == len(result.legends)
        # The composite names every axis that the figure still draws, a
        # strip of clone colours included, and none of the colour bars it
        # replaced.
        assert {key for _, key in composite.axes} == {
            key for key, ax in result.axes.items() if not is_colorbar_axes(ax)
        }
    finally:
        plt.close(figure)


@pytest.mark.parametrize("theme", ["notebook", "dark"])
@pytest.mark.parametrize("name", VARIANT_FIGURES)
def test_variant_figures_take_their_themes_background(variants, name, theme) -> None:
    result = variants.calls[name](theme=theme, show=False)
    try:
        alpha = 1.0 if theme == "notebook" else 0.0
        assert result.figure.patch.get_alpha() == alpha
        assert {ax.patch.get_alpha() for ax in result.figure.axes} == {alpha}
    finally:
        result.close()


def test_variant_figure_tables_hold_their_records_values(variants) -> None:
    store = variants.store
    results = {name: variants.calls[name](show=False) for name in VARIANT_FIGURES}
    try:
        # Each site's depth over the cells that the files name, and each
        # named cell's mean depth in its sample.
        sites = store.variants.load_counts(variants.counts, cell_selection=[]).sites
        cells = store.variants.load_counts(variants.counts, sites=[]).cells
        named = cells[cells["listed"]]
        depth = results["variant_depth"].tables
        assert depth["sites"]["site"].tolist() == sites.index.tolist()
        assert depth["sites"]["position"].tolist() == sites["position"].tolist()
        np.testing.assert_allclose(
            depth["sites"]["mean_depth"], sites["depth"] / len(named)
        )
        np.testing.assert_allclose(
            depth["sites"]["covered_fraction"], sites["n_cells"] / len(named)
        )
        samples = pd.Series(TRUTH.samples, index=TRUTH.cell_ids)
        assert depth["cells"]["ids"].tolist() == named.index.tolist()
        assert depth["cells"]["group"].tolist() == samples[named.index].tolist()
        np.testing.assert_allclose(depth["cells"]["mean_depth"], named["mean_depth"])
        assert (
            depth["cells"]["n_covered_sites"].tolist()
            == named["n_covered_sites"].tolist()
        )
        # The selection's statistics of every variant.
        scored = store.variants.load_informative(variants.informative)
        statistics = results["variant_statistics"].tables["variants"]
        assert statistics["variant"].tolist() == scored.variants.index.tolist()
        for column in ("vmr", "strand_correlation", "mean_coverage"):
            np.testing.assert_allclose(statistics[column], scored.variants[column])
        for column in ("masked", "informative"):
            assert statistics[column].tolist() == scored.variants[column].tolist()
        # Each drawn cell's alternative reads over its depth at the site of
        # each informative variant, NaN where it was not covered, the cells
        # ordered by clone.
        frequency_tables = results["allele_frequencies"].tables
        drawn = frequency_tables["cells"]
        clones = store.clones.load_clones(variants.clones)
        with_clone = clones.cells[clones.cells["clone"].notna()]
        rank = {clone: at for at, clone in enumerate(clones.clones["clone"])}
        expected_order = sorted(
            with_clone.index, key=lambda cell: rank[with_clone.loc[cell, "clone"]]
        )
        assert drawn["ids"].tolist() == expected_order
        assert (
            drawn["clone"].tolist() == with_clone.loc[expected_order, "clone"].tolist()
        )
        chosen = scored.variants[scored.variants["informative"]]
        loaded = store.variants.load_counts(
            variants.counts,
            sites=list(dict.fromkeys(chosen["site"])),
            cell_selection=expected_order,
        )
        reads = loaded.depth.astype({"ids": str, "site": str})
        alternative = loaded.counts.astype({"ids": str, "site": str, "allele": str})
        variant_of = pd.DataFrame(
            {
                "variant": chosen.index.to_numpy(dtype=object),
                "site": chosen["site"].astype(str).to_numpy(dtype=object),
                "allele": chosen["alternative"].astype(str).to_numpy(dtype=object),
            }
        )
        expected = (
            pd.MultiIndex.from_product(
                [expected_order, chosen.index.tolist()], names=["ids", "variant"]
            )
            .to_frame(index=False)
            .merge(variant_of[["variant", "site"]], on="variant")
            .merge(reads[["ids", "site", "depth"]], on=["ids", "site"], how="left")
            .merge(
                alternative.merge(variant_of, on=["site", "allele"])[
                    ["ids", "variant", "count"]
                ],
                on=["ids", "variant"],
                how="left",
            )
        )
        expected["frequency"] = np.where(
            expected["depth"].fillna(0) > 0,
            expected["count"].fillna(0) / expected["depth"],
            np.nan,
        )
        found = frequency_tables["frequencies"].merge(
            expected[["ids", "variant", "frequency"]],
            on=["ids", "variant"],
            suffixes=("", "_expected"),
        )
        assert len(found) == len(frequency_tables["frequencies"]) == len(expected)
        np.testing.assert_allclose(found["frequency"], found["frequency_expected"])
        # The first target's calls of each sample's cells, a cell that the
        # calls do not list there missing, and every cell's call on the
        # layout.
        called = store.variants.load_genotypes(variants.genotypes, sites=["T1"])
        call_of = pd.Series("missing", index=called.cells.index, dtype=object)
        listed = called.calls.astype({"ids": str, "call": str})
        call_of[listed["ids"].to_numpy()] = listed["call"].to_numpy()
        counted = pd.crosstab(samples[call_of.index].to_numpy(), call_of.to_numpy())
        tables = results["genotypes"].tables
        for row in tables["calls"].itertuples():
            held = int(counted.loc[row.group].get(row.call, 0))
            assert (row.site, row.n_cells) == ("T1", held)
            assert row.share == pytest.approx(held / counted.loc[row.group].sum())
        assert tables["cells"]["ids"].tolist() == call_of.index.tolist()
        assert tables["cells"]["call"].tolist() == call_of.tolist()
    finally:
        for result in results.values():
            result.close()


def test_the_layout_panel_draws_the_cells_where_embedding_draws_them(
    variants,
) -> None:
    store = variants.store
    figure = variants.calls["genotypes"](show=False)
    reference = store.plots.embedding(
        layout=variants.layout, color_by="sample", show=False
    )
    try:
        panel = figure.axes["layout"]
        (points,) = panel.collections
        (drawn,) = [ax for ax in reference.axes.values() if ax.collections]
        (embedded,) = drawn.collections

        def rows(collection) -> np.ndarray:
            offsets = np.asarray(collection.get_offsets(), dtype=np.float64)
            return offsets[np.lexsort(offsets.T[::-1])]

        # Each cell at the embedding's coordinates, in its window, square and
        # without ticks, as embedding draws it.
        np.testing.assert_array_equal(rows(points), rows(embedded))
        # The cells draw in embedding's seeded shuffle, those without a call
        # first: each of the two runs keeps embedding's order of its cells.
        call_of = {
            (float(x), float(y)): call
            for x, y, call in figure.tables["cells"][["x", "y", "call"]].itertuples(
                index=False
            )
        }
        drawn_order = [tuple(map(float, xy)) for xy in points.get_offsets()]
        embedded_order = [tuple(map(float, xy)) for xy in embedded.get_offsets()]
        uncalled = [
            xy for xy in embedded_order if call_of[xy] in ("missing", "filtered")
        ]
        called = [xy for xy in embedded_order if xy not in set(uncalled)]
        assert uncalled and called
        assert drawn_order == uncalled + called
        assert not points.get_rasterized()
        cells = figure.tables["cells"][["x", "y"]].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(
            rows(points), cells[np.lexsort(cells.T[::-1])], rtol=1e-6
        )
        assert panel.get_xlim() == pytest.approx(drawn.get_xlim())
        assert panel.get_ylim() == pytest.approx(drawn.get_ylim())
        assert not panel.get_xticks().size and not panel.get_yticks().size
        # Point areas follow the panel's size, as an embedding's do.
        from cytearc.plotting._style import default_point_size, panel_area_inches

        assert points.get_sizes()[0] == pytest.approx(
            default_point_size(len(cells), panel_area=panel_area_inches(panel))
        )
    finally:
        figure.close()
        reference.close()


# The figure of a phylogeny keeps the same conventions, on the truth's
# phylogeny cut at time 1 and drawn at five wedges, one of several clones.


@pytest.fixture(scope="module")
def phylogeny(tmp_path_factory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("phylogeny_figure_conventions")
    store = write_truth_store(folder / "store.zarr", TRUTH)
    tree, clones = import_truth_phylogeny(store, TRUTH, folder)
    return SimpleNamespace(
        store=store,
        tree=tree,
        clones=clones,
        draw=lambda **options: store.plots.phylogeny(clones, max_leaves=5, **options),
    )


def _tree_colors(ax) -> set[str]:
    """The colours that a tree's panel fills: its wedges and points."""
    colors = {to_hex(patch.get_facecolor()) for patch in ax.patches if patch.get_fill()}
    for collection in ax.collections:
        colors |= {to_hex(color) for color in collection.get_facecolors()}
    return colors


def test_every_phylogeny_figure_is_checked() -> None:
    assert phylogeny_plots.__all__ == ["phylogeny"]


def test_the_phylogeny_legend_is_reported_with_its_scale_and_agrees_with_the_drawing(
    phylogeny,
) -> None:
    result = phylogeny.draw(show=False)
    try:
        assert [legend.kind for legend in result.legends] == ["categorical"]
        assert len(result.scales) == 1
        parts = _shared_legend_parts(result)
        assert parts is not None and len(parts) == 1
        (legend,), (scale,) = result.legends, result.scales
        (artist,) = _drawn_legends(result.figure)
        assert artist.get_title().get_text() == legend.label == "Clone"
        labels = [category_label(scale, value) for value in scale.order]
        assert [text.get_text() for text in artist.get_texts()] == labels
        assert labels == ["Mixed", "1", "2", "3", "4"]
        colors = [to_hex(handle.get_facecolor()) for handle in artist.legend_handles]
        assert colors == [scale.palette[value] for value in scale.order]
        assert set(colors) <= _tree_colors(artist.axes)
        assert result.provenance.notes == ("phylogeny",)
        assert result.provenance.extras["clones"] == phylogeny.clones.to_dict()
        assert result.provenance.extras["phylogeny"] == phylogeny.tree.to_dict()
    finally:
        result.close()


def test_a_hidden_phylogeny_legend_is_still_reported(phylogeny) -> None:
    shown = phylogeny.draw(show=False)
    hidden = phylogeny.draw(show=False, show_legend=False)
    try:
        assert hidden.legends == shown.legends
        assert hidden.scales == shown.scales
        assert not _drawn_legends(hidden.figure)
    finally:
        shown.close()
        hidden.close()


def test_the_phylogeny_draws_in_a_composite_with_its_legend_once(phylogeny) -> None:
    figure = plt.figure(figsize=(8, 4), layout="constrained")
    try:
        result = phylogeny.draw(target={"tree": figure.subplots(1, 1)}, show=False)
        composite = compose_results(figure, [result])
        figure.canvas.draw()
        legends = list(figure.legends) + _drawn_legends(figure)
        assert len(legends) == 1
        assert len(composite.legends) == 1
    finally:
        plt.close(figure)


@pytest.mark.parametrize("theme", ["notebook", "dark"])
def test_the_phylogeny_takes_its_themes_background(phylogeny, theme) -> None:
    result = phylogeny.draw(theme=theme, show=False)
    try:
        alpha = 1.0 if theme == "notebook" else 0.0
        assert result.figure.patch.get_alpha() == alpha
        assert {ax.patch.get_alpha() for ax in result.figure.axes} == {alpha}
        # The tree's lines and the counts take the theme's foreground.
        from cytearc.plotting._style import foreground_color

        ax = result.figure.axes[0]
        lines = [patch for patch in ax.patches if not patch.get_fill()]
        assert [to_hex(patch.get_edgecolor()) for patch in lines] == [
            foreground_color(theme)
        ]
        assert {to_hex(text.get_color()) for text in ax.texts} == {
            foreground_color(theme)
        }
    finally:
        result.close()


def test_the_phylogeny_lists_its_clones_by_id_and_label_within_the_page(
    phylogeny,
) -> None:
    result = phylogeny.draw(show=False)
    try:
        nodes = result.tables["nodes"]
        assert {"clone", "label"} <= set(nodes.columns)
        listed = nodes[nodes["n_clones"] == 1]
        assert listed["clone"].tolist() == ["1", "2", "3", "4"]
        assert listed["label"].tolist() == listed["clone"].tolist()
        assert result.figure.get_size_inches()[0] <= MAX_FIGURE_WIDTH_INCHES
    finally:
        result.close()
