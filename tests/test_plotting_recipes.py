import errno
import io
import json
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import cytearc.plotting as plotting
import cytearc.plotting.recipes as recipes_module
from cytearc.plotting.recipes import (
    ALLOWED_OUTPUT_FORMATS,
    PlotOutputSettings,
    PlotPanelTarget,
    PlotRecipe,
    PlotStep,
    run_recipe,
)


class _FakeFigure:
    """A figure of a fake plot, which holds itself as a Matplotlib figure does."""

    @property
    def figure(self) -> "_FakeFigure":
        return self


class FakePlotResult:
    def __init__(self, name: str, *, owns_figure: bool = True) -> None:
        self.name = name
        # Each fake draws on a figure of its own.
        self.figure = _FakeFigure()
        self.owns_figure = owns_figure
        self.saved: list[Path] = []
        self.save_kwargs: list[dict[str, object]] = []
        self.show_calls = 0
        self.close_calls = 0
        self.usable = True

    def save(self, path: str | Path, **kwargs: object) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.name, encoding="utf-8")
        self.saved.append(output)
        self.save_kwargs.append(kwargs)
        return output

    def show(self) -> None:
        self.show_calls += 1
        # As PlotResult.show() does, showing closes a figure that the plot made.
        if self.owns_figure:
            self.close()

    def close(self) -> None:
        self.close_calls += 1
        self.usable = False


def test_runner_preserves_order_suppresses_plot_show_and_manages_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = object()
    calls: list[tuple[str, object, dict[str, object]]] = []
    plot_results: dict[str, FakePlotResult] = {}

    def fake_plot(name: str):
        def plot(store_arg: object, **kwargs: object) -> FakePlotResult:
            calls.append((name, store_arg, kwargs))
            result = FakePlotResult(name)
            plot_results[name] = result
            return result

        return plot

    monkeypatch.setattr(plotting, "embedding", fake_plot("embedding"))
    monkeypatch.setattr(plotting, "dotplot", fake_plot("dotplot"))
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                kwargs={"layout_key": "umap", "show": True},
                output_filename="figures/overview.png",
            ),
            PlotStep(
                name="markers",
                plot="dotplot",
                kwargs={"features": ["A"], "group_by": "cluster"},
                output_filename="figures/markers.svg",
            ),
        ]
    )

    result = run_recipe(store, recipe, output_dir=tmp_path, show=True)

    # Steps run in order with exactly their kwargs, and never show themselves.
    assert calls == [
        ("embedding", store, {"layout_key": "umap", "show": False}),
        ("dotplot", store, {"features": ["A"], "group_by": "cluster", "show": False}),
    ]
    expected_paths = (
        tmp_path / "figures/overview.png",
        tmp_path / "figures/markers.svg",
    )
    assert result.written_paths == expected_paths
    assert [output.name for output in result.outputs] == ["overview", "markers"]
    assert [output.result for output in result.outputs] == [
        plot_results["embedding"],
        plot_results["dotplot"],
    ]
    assert result.outputs[0].path == expected_paths[0]
    assert all(path.exists() for path in result.written_paths)
    for name, expected_path in zip(("embedding", "dotplot"), expected_paths):
        plot_result = plot_results[name]
        assert plot_result.saved == [expected_path]
        assert plot_result.save_kwargs == [
            {"dpi": None, "transparent": False, "exact_size": True}
        ]
        assert plot_result.show_calls == 1
        assert plot_result.close_calls == 1
    assert result.failures == {}


def test_no_output_mode_leaves_returned_owned_figure_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plot_result = FakePlotResult("embedding")

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        return plot_result

    monkeypatch.setattr(plotting, "embedding", embedding)
    recipe = PlotRecipe(
        [PlotStep(name="overview", plot="embedding", kwargs={"layout_key": "umap"})]
    )

    result = run_recipe(object(), recipe)

    assert [output.result for output in result.outputs] == [plot_result]
    assert result.written_paths == ()
    assert plot_result.saved == []
    assert plot_result.show_calls == 0
    assert plot_result.close_calls == 0
    assert plot_result.usable


def test_runner_does_not_close_caller_owned_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plot_result = FakePlotResult("embedding", owns_figure=False)
    monkeypatch.setattr(
        plotting,
        "embedding",
        lambda store, **kwargs: plot_result,
    )
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="overview.png",
            )
        ]
    )

    run_recipe(object(), recipe, output_dir=tmp_path, show=True)

    assert plot_result.saved == [tmp_path / "overview.png"]
    assert plot_result.show_calls == 1
    assert plot_result.close_calls == 0
    assert plot_result.usable


def test_continue_on_error_captures_compact_failures_and_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    successful = FakePlotResult("dotplot")

    def failing(store: object, **kwargs: object) -> FakePlotResult:
        calls.append("bad")
        warnings.warn("first warning", UserWarning)
        raise RuntimeError("plot\nfailed")

    def succeeding(store: object, **kwargs: object) -> FakePlotResult:
        calls.append("good")
        warnings.warn("second warning", RuntimeWarning)
        return successful

    monkeypatch.setattr(plotting, "embedding", failing)
    monkeypatch.setattr(plotting, "dotplot", succeeding)
    recipe = PlotRecipe(
        [
            PlotStep(name="bad", plot="embedding"),
            PlotStep(name="good", plot="dotplot"),
        ]
    )

    with pytest.raises(RuntimeError, match="plot"):
        run_recipe(object(), recipe)
    assert calls == ["bad"]

    calls.clear()
    result = run_recipe(object(), recipe, continue_on_error=True)

    assert calls == ["bad", "good"]
    assert result.failures == {"bad": "RuntimeError: plot failed"}
    assert [output.result for output in result.outputs] == [successful]
    assert result.warnings == (
        "bad: first warning",
        "good: second warning",
    )
    assert successful.usable


# Showing figures. The plots draw one line into the axes that a step targets,
# or into a figure of their own, and IPython's display records each figure it
# shows with the number of lines in each of the figure's axes at that moment.


def _lines_per_axes(figure: object) -> list[int]:
    return [len(axis.lines) for axis in figure.axes]  # type: ignore[attr-defined]


@pytest.fixture
def displayed(
    monkeypatch: pytest.MonkeyPatch, panels: object
) -> list[tuple[object, list[int]]]:
    # The panels come first: pyplot's first figure installs its display hook
    # through IPython, which the stub below replaces.
    shown: list[tuple[object, list[int]]] = []
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr(
        "IPython.display.display",
        lambda figure: shown.append((figure, _lines_per_axes(figure))),
    )
    return shown


def _warn_when_drawn(figure: object, message: str) -> None:
    """Warn whenever the figure is drawn, as a font or layout warning does."""
    figure.canvas.mpl_connect(  # type: ignore[attr-defined]
        "draw_event", lambda event: warnings.warn(message, UserWarning)
    )


def _line_plot(created: list[object]):
    def plot(store: object, *, show: bool, target: object = None) -> object:
        import matplotlib.pyplot as plt

        assert show is False
        owns_figure = target is None
        if owns_figure:
            figure, target = plt.subplots()
            _warn_when_drawn(figure, f"figure {len(created)} drawn")
            created.append(figure)
        target.plot([0, 1], [0, 1])  # type: ignore[attr-defined]
        return plotting.PlotResult(
            figure=target.figure,  # type: ignore[attr-defined]
            axes={"main": target},
            tables={},
            legends=(),
            scales=(),
            provenance=plotting.PlotProvenance(),
            owns_figure=owns_figure,
        )

    return plot


def _failing_plot(store: object, **kwargs: object) -> object:
    raise RuntimeError("plot failed")


def _steps(*names: str) -> PlotRecipe:
    """A step per name: "left" and "right" fill those panels, "broken" fails,
    and any other name makes a figure of its own."""
    return PlotRecipe(
        [
            PlotStep(
                name=name,
                plot="dotplot" if name.startswith("broken") else "embedding",
                target=PlotPanelTarget(name) if name in ("left", "right") else None,
            )
            for name in names
        ]
    )


@pytest.fixture
def panels(monkeypatch: pytest.MonkeyPatch):
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2)
    created: list[object] = []
    monkeypatch.setattr(plotting, "embedding", _line_plot(created))
    monkeypatch.setattr(plotting, "dotplot", _failing_plot)
    yield figure, {"left": left, "right": right}, created
    plt.close(figure)
    for own in created:
        plt.close(own)


def test_show_displays_the_figure_that_holds_the_subfigures_that_steps_fill(
    displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure()
    left, right = figure.subfigures(1, 2)
    targets = {"left": left.subplots(), "right": right.subplots()}

    try:
        run_recipe(object(), _steps("left", "right"), targets=targets, show=True)
    finally:
        plt.close(figure)

    # The steps drew into two subfigures of one figure, which is shown once.
    assert displayed == [(figure, [1, 1])]


def test_show_displays_a_figure_that_several_steps_fill_once_with_every_panel(
    displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    figure, targets, _ = panels

    result = run_recipe(object(), _steps("left", "right"), targets=targets, show=True)

    # One image with both panels drawn, where each step showed it before.
    assert displayed == [(figure, [1, 1])]
    assert [output.name for output in result.outputs] == ["left", "right"]
    # Shown in Jupyter, the figure leaves pyplot, so its cell draws it once.
    assert not plt.fignum_exists(figure.number)


def test_show_displays_a_made_figure_in_its_step_and_the_callers_after_the_last(
    monkeypatch: pytest.MonkeyPatch, displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    figure, targets, created = panels
    draw = plotting.embedding
    shown_before_each_step: list[int] = []

    def counting(store: object, **kwargs: object) -> object:
        shown_before_each_step.append(len(displayed))
        return draw(store, **kwargs)

    monkeypatch.setattr(plotting, "embedding", counting)

    run_recipe(object(), _steps("left", "own", "right"), targets=targets, show=True)

    # The figure that "own" made was shown before "right" ran, and closed;
    # the caller's figure waited for "right".
    assert shown_before_each_step == [0, 0, 1]
    assert displayed == [(created[0], [1]), (figure, [1, 1])]
    assert not plt.fignum_exists(created[0].number)


def test_the_callers_figures_are_shown_in_the_order_of_their_first_step(
    displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    first, targets, _ = panels
    second, (other_left, other_right) = plt.subplots(1, 2)
    targets = {**targets, "other_left": other_left, "other_right": other_right}
    # The steps fill the two figures in turn, the second figure first.
    recipe = PlotRecipe(
        [
            PlotStep(name=name, plot="embedding", target=PlotPanelTarget(name))
            for name in ("other_left", "left", "other_right", "right")
        ]
    )

    try:
        run_recipe(object(), recipe, targets=targets, show=True)
    finally:
        plt.close(second)

    assert displayed == [(second, [1, 1]), (first, [1, 1])]


def test_show_displays_the_figures_of_the_steps_that_succeeded(
    displayed, panels
) -> None:
    figure, targets, created = panels

    result = run_recipe(
        object(),
        _steps("left", "broken", "own", "right"),
        targets=targets,
        show=True,
        continue_on_error=True,
    )

    assert displayed == [(created[0], [1]), (figure, [1, 1])]
    assert result.failures == {"broken": "RuntimeError: plot failed"}
    assert [output.name for output in result.outputs] == ["left", "own", "right"]


def test_a_failing_step_stops_the_recipe_after_the_earlier_figures_are_shown(
    displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    figure, targets, created = panels

    with pytest.raises(RuntimeError, match="plot failed"):
        run_recipe(
            object(),
            _steps("own", "left", "broken", "right"),
            targets=targets,
            show=True,
        )

    # The made figure was shown in its step and closed, and the caller's
    # figure is shown as the steps before the failure drew it.
    assert displayed == [(created[0], [1]), (figure, [1, 0])]
    assert not plt.fignum_exists(created[0].number)


def test_without_show_the_recipe_displays_nothing_and_leaves_its_figures_open(
    displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    figure, targets, created = panels

    result = run_recipe(object(), _steps("left", "own", "right"), targets=targets)

    assert displayed == []
    assert _lines_per_axes(figure) == [1, 1]
    assert plt.fignum_exists(figure.number) and plt.fignum_exists(created[0].number)
    assert [output.result.figure for output in result.outputs] == [
        figure,
        created[0],
        figure,
    ]


def test_warnings_raised_while_a_figure_is_shown_are_recorded_under_its_steps(
    monkeypatch: pytest.MonkeyPatch, panels
) -> None:
    figure, targets, created = panels
    _warn_when_drawn(figure, "the caller's figure drawn")
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    # As matplotlib-inline's formatter does, the display renders the figure,
    # so the warnings of a draw are raised while it is shown.
    monkeypatch.setattr(
        "IPython.display.display",
        lambda shown: shown.canvas.print_figure(io.BytesIO(), format="png"),
    )

    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        result = run_recipe(
            object(), _steps("left", "own", "right"), targets=targets, show=True
        )

    assert result.warnings == (
        "own: figure 0 drawn",
        "left: the caller's figure drawn",
        "right: the caller's figure drawn",
    )
    assert [str(warning.message) for warning in escaped] == []


def test_a_recipe_keeps_one_made_figure_open_at_a_time(
    monkeypatch: pytest.MonkeyPatch, displayed, panels
) -> None:
    import matplotlib.pyplot as plt

    _, _, created = panels
    draw = plotting.embedding
    open_before_each_step: list[int] = []

    def counting(store: object, **kwargs: object) -> object:
        open_before_each_step.append(
            sum(plt.fignum_exists(own.number) for own in created)
        )
        return draw(store, **kwargs)

    monkeypatch.setattr(plotting, "embedding", counting)

    # More figures than the 20 that Matplotlib warns of when they are open.
    result = run_recipe(object(), _steps(*(f"own{i}" for i in range(22))), show=True)

    assert open_before_each_step == [0] * 22
    assert len(displayed) == 22
    assert not any("More than 20 figures" in warning for warning in result.warnings)


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_a_made_figure_that_fails_to_show_fails_its_step_and_is_closed(
    monkeypatch: pytest.MonkeyPatch, panels, continue_on_error: bool
) -> None:
    import matplotlib.pyplot as plt

    _, _, created = panels
    shown: list[object] = []

    def display(figure: object) -> None:
        if figure is created[0]:
            raise RuntimeError("display failed")
        shown.append(figure)

    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", display)

    if continue_on_error:
        result = run_recipe(
            object(), _steps("first", "second"), show=True, continue_on_error=True
        )
        # The figure that failed to show fails its step, and the next is shown.
        assert result.failures == {"first": "RuntimeError: display failed"}
        assert [output.name for output in result.outputs] == ["second"]
        assert shown == [created[1]]
    else:
        with pytest.raises(RuntimeError, match="display failed"):
            run_recipe(object(), _steps("first", "second"), show=True)
        # The error stops the recipe before the second step makes its figure.
        assert len(created) == 1
    assert not any(plt.fignum_exists(own.number) for own in created)


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_a_callers_figure_that_fails_to_show_fails_every_step_that_drew_it(
    monkeypatch: pytest.MonkeyPatch, panels, continue_on_error: bool
) -> None:
    figure, targets, created = panels
    shown: list[object] = []

    def display(displayed_figure: object) -> None:
        if displayed_figure is figure:
            raise RuntimeError("display failed")
        shown.append(displayed_figure)

    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", display)
    steps = _steps("left", "own", "right")

    if continue_on_error:
        result = run_recipe(
            object(), steps, targets=targets, show=True, continue_on_error=True
        )
        failure = "RuntimeError: display failed"
        assert result.failures == {"left": failure, "right": failure}
        assert [output.name for output in result.outputs] == ["own"]
    else:
        with pytest.raises(RuntimeError, match="display failed"):
            run_recipe(object(), steps, targets=targets, show=True)
    # The figure that "own" made was shown in its step either way.
    assert shown == [created[0]]


def test_a_failing_step_notes_an_earlier_figure_that_fails_to_show(
    monkeypatch: pytest.MonkeyPatch, panels
) -> None:
    import matplotlib.pyplot as plt

    figure, targets, _ = panels

    def display(shown: object) -> None:
        raise RuntimeError("display failed")

    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", display)

    with pytest.raises(RuntimeError, match="plot failed") as raised:
        run_recipe(object(), _steps("left", "broken"), targets=targets, show=True)

    # The step's error is raised, and the caller's figure that failed to show
    # is named in a note; in Jupyter, showing closed it in pyplot.
    assert raised.value.__notes__ == [
        "The figure of 'left' also failed to show: RuntimeError: display failed"
    ]
    assert not plt.fignum_exists(figure.number)


def test_a_written_figure_that_the_recipe_does_not_show_is_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = FakePlotResult("embedding")
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [PlotStep(name="overview", plot="embedding", output_filename="overview.png")]
    )

    recipe_result = run_recipe(object(), recipe, output_dir=tmp_path)

    assert recipe_result.written_paths == (tmp_path / "overview.png",)
    assert result.saved == [tmp_path / "overview.png"]
    assert result.show_calls == 0
    assert result.close_calls == 1


def test_json_and_toml_loaders_have_matching_camel_case_schema(
    tmp_path: Path,
) -> None:
    config = {
        "steps": [
            {
                "name": "overview",
                "plot": "embedding",
                "kwargs": {
                    "layoutKey": "umap",
                    "colorBy": "cluster",
                },
                "outputFilename": "figures/overview.png",
            },
            {
                "name": "composition",
                "plot": "composition",
                "kwargs": {"categoryBy": "cluster"},
            },
        ]
    }
    json_path = tmp_path / "recipe.json"
    json_path.write_text(json.dumps(config), encoding="utf-8")
    toml_path = tmp_path / "recipe.toml"
    toml_path.write_text(
        """
[[steps]]
name = "overview"
plot = "embedding"
outputFilename = "figures/overview.png"

[steps.kwargs]
layoutKey = "umap"
colorBy = "cluster"

[[steps]]
name = "composition"
plot = "composition"

[steps.kwargs]
categoryBy = "cluster"
""".strip(),
        encoding="utf-8",
    )

    from_dict = PlotRecipe.from_dict(config)
    from_json = PlotRecipe.from_json(json_path)
    from_toml = PlotRecipe.from_toml(toml_path)

    assert from_dict == from_json == from_toml
    assert from_json.steps[0].output_filename == "figures/overview.png"
    assert dict(from_toml.steps[1].kwargs) == {"category_by": "cluster"}


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("/tmp/plot.png", "outputFilename must be a relative path"),
        (r"C:\plots\plot.png", "outputFilename must be a relative path"),
        ("C:plot.png", "outputFilename must be a relative path"),
        (r"\\server\share\plot.png", "outputFilename must be a relative path"),
        ("../plot.png", "outputFilename must not contain parent traversal"),
        ("nested/../../plot.png", "outputFilename must not contain parent traversal"),
    ],
)
def test_output_filename_rejects_absolute_and_traversal_paths(
    filename: str, message: str
) -> None:
    with pytest.raises(ValueError) as raised:
        PlotStep(
            name="overview",
            plot="embedding",
            output_filename=filename,
        )

    assert raised.value.args == (message,)


def test_output_filename_requires_supported_format_and_output_directory() -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        PlotStep(
            name="overview",
            plot="embedding",
            output_filename="overview.txt",
        )

    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="overview.png",
            )
        ]
    )
    with pytest.raises(ValueError, match="output_dir"):
        run_recipe(object(), recipe)


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        (
            {"steps": [], "extra": True},
            ValueError,
            r"Unknown plot recipe key\(s\): 'extra'",
        ),
        (
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "output_filename": "overview.png",
                    }
                ]
            },
            ValueError,
            r"Unknown plot step 0 key\(s\): 'output_filename'",
        ),
        (
            {"steps": [{"name": "overview", "plot": "embedding", "kwargs": []}]},
            TypeError,
            "kwargs for plot step 0 must be a mapping",
        ),
        (
            {"steps": [{"name": "overview", "plot": "unknown"}]},
            ValueError,
            "Unknown plot 'unknown'; choose from",
        ),
    ],
)
def test_from_dict_rejects_unknown_keys_and_invalid_values(
    config: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        PlotRecipe.from_dict(config)


def test_recipe_validates_non_empty_unique_names_and_is_frozen() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PlotRecipe([])
    with pytest.raises(ValueError, match="non-empty"):
        PlotStep(name=" ", plot="embedding")

    step = PlotStep(name="overview", plot="embedding")
    with pytest.raises(ValueError, match="unique"):
        PlotRecipe([step, step])
    with pytest.raises(FrozenInstanceError):
        step.name = "changed"
    assert not hasattr(step, "__dict__")


def test_external_kwargs_require_camel_case_and_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="camelCase"):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {"layout_key": "umap"},
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="Unknown kwargs"):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {
                            "layoutKey": "umap",
                            "notAPlotArgument": True,
                        },
                    }
                ]
            }
        )


def test_external_recipe_coerces_plot_contracts_and_feature_refs() -> None:
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "markers",
                    "plot": "dotplot",
                    "kwargs": {
                        "features": [
                            {
                                "value": "CD3D",
                                "assay": "RNA",
                                "by": "name",
                            }
                        ],
                        "groupBy": "cluster",
                        "colorScale": {
                            "cmap": "magma",
                            "quantiles": [0.01, 0.99],
                            "missingColor": "#cccccc",
                        },
                        "sizeScale": {
                            "sizeMin": 5,
                            "sizeMax": 80,
                        },
                    },
                }
            ]
        }
    )

    kwargs = recipe.steps[0].kwargs
    assert kwargs["features"] == [
        plotting.FeatureRef(value="CD3D", assay="RNA", by="name")
    ]
    assert kwargs["color_scale"] == plotting.ColorScale(
        cmap="magma",
        quantiles=(0.01, 0.99),
        missing_color="#cccccc",
    )
    assert kwargs["size_scale"] == plotting.SizeScale(size_min=5, size_max=80)


def test_external_recipe_passes_a_single_feature_name_on_whole() -> None:
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "markers",
                    "plot": "dotplot",
                    "kwargs": {"features": "CD3D", "groupBy": "cluster"},
                }
            ]
        }
    )

    # A name is text, not a list of one-letter features: the plot reads it.
    assert recipe.steps[0].kwargs["features"] == "CD3D"


def test_runner_injects_pipeline_artifacts_without_mutating_recipe_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []
    plot_result = FakePlotResult("embedding")

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return plot_result

    monkeypatch.setattr(plotting, "embedding", embedding)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                kwargs={"layout_key": "umap"},
                artifact_kwargs={"color_by": "clusterColumn"},
            )
        ]
    )

    result = run_recipe(
        object(),
        recipe,
        artifacts={"clusterColumn": "RNA_leiden_cluster"},
    )

    assert [output.result for output in result.outputs] == [plot_result]
    assert seen == [
        {
            "layout_key": "umap",
            "color_by": "RNA_leiden_cluster",
            "show": False,
        }
    ]
    assert dict(recipe.steps[0].kwargs) == {"layout_key": "umap"}
    with pytest.raises(KeyError, match="clusterColumn"):
        run_recipe(object(), recipe, artifacts={})


def test_recipe_resolves_optional_panel_target_and_output_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []
    result = FakePlotResult("embedding", owns_figure=False)

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return result

    monkeypatch.setattr(plotting, "embedding", embedding)
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "overview",
                    "plot": "embedding",
                    "kwargs": {"layoutKey": "umap"},
                    "target": {"panel": "A"},
                    "output": {
                        "filename": "overview.png",
                        "dpi": 180,
                        "transparent": True,
                        "exactSize": False,
                    },
                }
            ]
        }
    )

    assert recipe.steps[0].target == PlotPanelTarget("A")
    assert recipe.steps[0].output == PlotOutputSettings(
        "overview.png",
        dpi=180,
        transparent=True,
        exact_size=False,
    )
    run_recipe(
        object(),
        recipe,
        targets={"A": "axes-a"},
        output_dir=tmp_path,
    )

    assert seen == [
        {
            "layout_key": "umap",
            "target": "axes-a",
            "show": False,
        }
    ]
    assert result.save_kwargs == [
        {
            "dpi": 180,
            "transparent": True,
            "exact_size": False,
        }
    ]
    with pytest.raises(KeyError, match="requires panel"):
        run_recipe(object(), recipe, targets={}, output_dir=tmp_path)


def test_json_loader_rejects_duplicate_keys_and_non_object_roots() -> None:
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        PlotRecipe.from_json(
            '{"steps": [{"name": "one", "name": "two", "plot": "embedding"}]}'
        )
    with pytest.raises(TypeError, match="mapping"):
        PlotRecipe.from_json("[]")


def test_output_formats_are_the_documented_set() -> None:
    assert ALLOWED_OUTPUT_FORMATS == frozenset({"pdf", "png", "svg", "tif", "tiff"})


@pytest.mark.parametrize(
    "filename",
    ["plot.PDF", "plot.Png", "plot.SVG", "plot.TIF", "nested/plot.tIFF"],
)
def test_output_settings_accept_supported_formats_case_insensitively(
    filename: str,
) -> None:
    step = PlotStep(name="s", plot="embedding", output_filename=filename)

    assert step.output == PlotOutputSettings(filename)
    assert step.output_filename == filename


def test_inline_json_longer_than_a_file_name_is_parsed_as_text() -> None:
    # A one-line source this long cannot name a file, so it parses as JSON.
    name = "a" * 300
    recipe = PlotRecipe.from_json(
        json.dumps({"steps": [{"name": name, "plot": "embedding"}]})
    )

    assert [step.name for step in recipe.steps] == [name]


def test_inline_json_is_parsed_as_text_when_the_file_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Python 3.13 raises for a name the filesystem rejects, where 3.14 returns
    # False, so the check raises here on every version.
    def unnamable(_path: Path, **_kwargs: object) -> bool:
        raise OSError(errno.ENAMETOOLONG, "File name too long")

    monkeypatch.setattr(Path, "is_file", unnamable)
    recipe = PlotRecipe.from_json(
        json.dumps({"steps": [{"name": "overview", "plot": "embedding"}]})
    )

    assert [step.name for step in recipe.steps] == ["overview"]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"filename": 7}, TypeError, "must be a string"),
        ({"filename": " "}, ValueError, "non-empty"),
        ({"filename": "bad\x00.png"}, ValueError, "null byte"),
        ({"filename": "plot.png", "dpi": True}, ValueError, "positive integer"),
        ({"filename": "plot.png", "dpi": 0}, ValueError, "positive integer"),
        ({"filename": "plot.png", "dpi": 1.5}, ValueError, "positive integer"),
        (
            {"filename": "plot.png", "transparent": 1},
            TypeError,
            "^transparent must be a boolean$",
        ),
        (
            {"filename": "plot.png", "exact_size": 0},
            TypeError,
            "^exact_size must be a boolean$",
        ),
    ],
)
def test_output_settings_reject_invalid_serialized_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        PlotOutputSettings(**kwargs)


def test_step_rejects_conflicting_sources_and_output_destinations() -> None:
    with pytest.raises(ValueError, match="cannot both define.*color_by"):
        PlotStep(
            name="overlap",
            plot="embedding",
            kwargs={"color_by": "cluster"},
            artifact_kwargs={"color_by": "clusterArtifact"},
        )
    with pytest.raises(ValueError, match="not both"):
        PlotStep(
            name="output",
            plot="embedding",
            output=PlotOutputSettings("plot.png"),
            output_filename="plot.svg",
        )
    with pytest.raises(ValueError, match="filenames must be unique"):
        PlotRecipe(
            [
                PlotStep(
                    name="first",
                    plot="embedding",
                    output_filename="plot.png",
                ),
                PlotStep(
                    name="second",
                    plot="embedding",
                    output_filename="plot.png",
                ),
            ]
        )


def test_step_and_recipe_reject_invalid_python_contracts() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        PlotPanelTarget(" ")
    with pytest.raises(TypeError, match="name must be a string"):
        PlotStep(name=1, plot="embedding")
    with pytest.raises(TypeError, match="plot must be a string"):
        PlotStep(name="step", plot=1)
    with pytest.raises(ValueError, match="Unknown plot"):
        PlotStep(name="step", plot="unknown")
    with pytest.raises(TypeError, match="kwargs.*mapping"):
        PlotStep(name="step", plot="embedding", kwargs=[])
    with pytest.raises(TypeError, match="kwargs keys.*strings"):
        PlotStep(name="step", plot="embedding", kwargs={1: "value"})
    with pytest.raises(TypeError, match="artifact_kwargs.*mapping"):
        PlotStep(name="step", plot="embedding", artifact_kwargs=[])
    with pytest.raises(TypeError, match="artifact_kwargs must map"):
        PlotStep(
            name="step",
            plot="embedding",
            artifact_kwargs={"color_by": 1},
        )
    with pytest.raises(TypeError, match="target must be"):
        PlotStep(name="step", plot="embedding", target="panel")
    with pytest.raises(TypeError, match="output must be"):
        PlotStep(name="step", plot="embedding", output="plot.png")
    with pytest.raises(TypeError, match="ordered sequence"):
        PlotRecipe("step")
    with pytest.raises(TypeError, match="Every recipe step"):
        PlotRecipe([object()])


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ({}, ValueError, "requires 'steps'"),
        ({"steps": {}}, TypeError, "ordered sequence"),
        ({"steps": ["step"]}, TypeError, "step 0 must be a mapping"),
        (
            {"steps": [{"plot": "embedding"}]},
            ValueError,
            "step 0 requires: name",
        ),
    ],
)
def test_serialized_recipe_requires_ordered_complete_steps(
    config: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        PlotRecipe.from_dict(config)


@pytest.mark.parametrize(
    ("step_fields", "error_type", "message"),
    [
        ({"target": []}, TypeError, "target must be a mapping"),
        ({"target": {"other": "A"}}, ValueError, "Unknown plot step 0 target"),
        ({"target": {}}, ValueError, "target requires panel"),
        ({"output": []}, TypeError, "output must be a mapping"),
        ({"output": {"dpi": 100}}, ValueError, "output requires filename"),
        (
            {
                "output": {"filename": "plot.png"},
                "outputFilename": "plot.svg",
            },
            ValueError,
            "not both",
        ),
        (
            {
                "output": {
                    "filename": "plot.png",
                    "provenanceSidecar": True,
                }
            },
            ValueError,
            "Unknown plot step 0 output",
        ),
        (
            {"artifactKwargs": {"colorBy": 7}},
            TypeError,
            "values must be artifact names",
        ),
        (
            {
                "kwargs": {"colorBy": "cluster"},
                "artifactKwargs": {"colorBy": "clusterArtifact"},
            },
            ValueError,
            "cannot both define",
        ),
    ],
)
def test_serialized_target_output_and_artifact_validation(
    step_fields: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    step = {"name": "overview", "plot": "embedding", **step_fields}

    with pytest.raises(error_type, match=message):
        PlotRecipe.from_dict({"steps": [step]})


def test_serialized_contracts_cover_color_and_grouped_feature_shapes() -> None:
    normalization = plotting.NormalizationSpec(source="raw")
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "multi-color",
                    "plot": "embedding",
                    "kwargs": {
                        "layoutKey": "umap",
                        "colorBy": [
                            {
                                "key": "cluster",
                                "kind": "categorical",
                            },
                            {"value": "CD3D"},
                        ],
                        "normalization": normalization,
                        "categoricalScale": {"order": ["B", "A"]},
                        "densityOverlay": {
                            "levels": 3,
                            "groupBy": "cluster",
                            "groups": ["A"],
                        },
                        "highlight": {"indices": [1, 2]},
                    },
                },
                {
                    "name": "single-color",
                    "plot": "embedding",
                    "kwargs": {
                        "layoutKey": "umap",
                        "colorBy": {"value": "CD4"},
                    },
                },
                {
                    "name": "grouped-features",
                    "plot": "dotplot",
                    "kwargs": {
                        "features": {
                            "T cells": [
                                {"value": "CD3D"},
                                "IL7R",
                            ]
                        },
                        "groupBy": "cluster",
                    },
                },
            ]
        }
    )

    multi_color = recipe.steps[0].kwargs
    assert multi_color["color_by"] == [
        plotting.CellField(key="cluster", kind="categorical"),
        plotting.FeatureRef(value="CD3D"),
    ]
    assert multi_color["normalization"] is normalization
    assert multi_color["categorical_scale"].order == ("B", "A")
    assert multi_color["density_overlay"] == plotting.DensityOverlay(
        levels=3,
        group_by="cluster",
        groups=("A",),
    )
    assert multi_color["highlight"] == plotting.Highlight(indices=(1, 2))
    assert recipe.steps[1].kwargs["color_by"] == plotting.FeatureRef(value="CD4")
    assert recipe.steps[2].kwargs["features"] == {
        "T cells": [
            plotting.FeatureRef(value="CD3D"),
            "IL7R",
        ]
    }


def test_serialized_contracts_cover_grouping_keys_and_nested_scales() -> None:
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "violins",
                    "plot": "distribution",
                    "kwargs": {
                        "keys": [{"key": "nCounts", "label": "Counts"}, "CD3D"],
                        "grouping": {"key": "cluster", "kind": "categorical"},
                        "splitBy": "condition",
                        "splitScale": {"order": ["ctrl", "stim"]},
                    },
                },
                {
                    "name": "matrix",
                    "plot": "matrixplot",
                    "kwargs": {
                        "features": ["CD3D"],
                        "groupBy": "cluster",
                        "annotationScales": {"panel": {"order": ["B", "A"]}},
                    },
                },
                {
                    "name": "dynamics",
                    "plot": "pseudotime_heatmap",
                    "kwargs": {
                        "featureClusterScale": {"order": [2, 1]},
                        "pseudotimeScale": {"cmap": "plasma"},
                    },
                },
            ]
        }
    )

    violins, matrix, dynamics = (step.kwargs for step in recipe.steps)
    assert violins["keys"] == [
        plotting.CellField(key="nCounts", label="Counts"),
        "CD3D",
    ]
    assert violins["grouping"] == plotting.CellField(key="cluster", kind="categorical")
    assert violins["split_scale"] == plotting.CategoricalScale(order=("ctrl", "stim"))
    assert matrix["annotation_scales"] == {
        "panel": plotting.CategoricalScale(order=("B", "A"))
    }
    assert dynamics["feature_cluster_scale"] == plotting.CategoricalScale(order=(2, 1))
    assert dynamics["pseudotime_scale"] == plotting.ColorScale(cmap="plasma")


def test_serialized_grouping_columns_read_as_cell_fields() -> None:
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "dots",
                    "plot": "dotplot",
                    "kwargs": {
                        "features": ["CD3D"],
                        "groupBy": [{"key": "cluster", "label": "Cluster"}, "sample"],
                    },
                },
                {
                    "name": "shares",
                    "plot": "composition",
                    "kwargs": {
                        "categoryBy": {"key": "cluster", "kind": "categorical"},
                        "sampleBy": {"key": "sample", "label": "Sample"},
                        "subjectBy": {"key": "donor"},
                        "pairBy": "pair",
                        "conditionBy": {"key": "condition", "label": "Treatment"},
                    },
                },
                {
                    "name": "violins",
                    "plot": "distribution",
                    "kwargs": {
                        "keys": "CD3D",
                        "splitBy": {"key": "batch", "label": "Batch"},
                    },
                },
                {
                    "name": "graph",
                    "plot": "cluster_connectivity",
                    "kwargs": {"groupBy": {"key": "cluster"}},
                },
                {
                    "name": "facets",
                    "plot": "embedding",
                    "kwargs": {
                        "layoutKey": "umap",
                        "facetBy": {"key": "stage", "label": "Stage"},
                    },
                },
            ]
        }
    )

    dots, shares, violins, graph, facets = (step.kwargs for step in recipe.steps)
    assert dots["group_by"] == [
        plotting.CellField(key="cluster", label="Cluster"),
        "sample",
    ]
    assert shares == {
        "category_by": plotting.CellField(key="cluster", kind="categorical"),
        "sample_by": plotting.CellField(key="sample", label="Sample"),
        "subject_by": plotting.CellField(key="donor"),
        "pair_by": "pair",
        "condition_by": plotting.CellField(key="condition", label="Treatment"),
    }
    assert violins["split_by"] == plotting.CellField(key="batch", label="Batch")
    assert graph["group_by"] == plotting.CellField(key="cluster")
    assert facets["facet_by"] == plotting.CellField(key="stage", label="Stage")
    # A malformed column names the keyword as the recipe spells it.
    with pytest.raises(ValueError, match="Unknown groupBy key"):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "dots",
                        "plot": "dotplot",
                        "kwargs": {"groupBy": [{"key": "cluster", "name": "x"}]},
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="Invalid sampleBy"):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "shares",
                        "plot": "composition",
                        "kwargs": {"sampleBy": {"key": "sample", "kind": "x"}},
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "filenames",
    [
        ("fig.png", "./fig.png"),
        ("panels/fig.png", "panels//fig.png"),
        ("Fig.png", "fig.png"),
    ],
)
def test_recipe_rejects_output_names_for_one_file(filenames) -> None:
    steps = [
        PlotStep(name=f"step{index}", plot="embedding", output_filename=filename)
        for index, filename in enumerate(filenames)
    ]
    with pytest.raises(ValueError, match="output filenames must be unique"):
        PlotRecipe(steps)


@pytest.mark.parametrize(
    ("color_scale", "error_type", "message"),
    [
        ("magma", TypeError, "color_scale must be a mapping"),
        ({"missing_color": "gray"}, ValueError, "lower camelCase"),
        ({"unknownField": True}, ValueError, "Unknown color_scale"),
        (
            {"quantiles": [0.9, 0.1]},
            ValueError,
            "Invalid color_scale",
        ),
    ],
)
def test_serialized_contracts_reject_invalid_shapes(
    color_scale: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {
                            "layoutKey": "umap",
                            "colorScale": color_scale,
                        },
                    }
                ]
            }
        )


def test_recipe_loaders_normalize_suffixes_and_source_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return FakePlotResult("embedding")

    monkeypatch.setattr(plotting, "embedding", embedding)
    json_path = tmp_path / "recipe.JSON"
    json_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "json",
                        "plot": "embedding",
                        "kwargs": {"layoutKey": "umap"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    toml_source = """
[[steps]]
name = "toml"
plot = "embedding"

[steps.kwargs]
layoutKey = "umap"
""".strip()
    toml_path = tmp_path / "recipe.ToMl"
    toml_path.write_text(toml_source, encoding="utf-8")

    json_recipe = PlotRecipe.from_json(str(json_path))
    toml_recipe = PlotRecipe.from_toml(str(toml_path))
    bytes_recipe = PlotRecipe.from_toml(toml_source.encode())
    json_result = run_recipe(object(), str(json_path))
    toml_result = run_recipe(object(), toml_path)

    assert json_recipe.steps[0].name == "json"
    assert toml_recipe == bytes_recipe
    assert json_result.outputs[0].result.name == "embedding"
    assert toml_result.outputs[0].result.name == "embedding"
    assert seen == [
        {"layout_key": "umap", "show": False},
        {"layout_key": "umap", "show": False},
    ]


def test_recipe_loader_rejects_invalid_source_types_and_formats(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="PlotRecipe or a JSON/TOML path"):
        run_recipe(object(), object())
    with pytest.raises(ValueError, match=r"\.json or \.toml"):
        run_recipe(object(), tmp_path / "recipe.yaml")
    with pytest.raises(ValueError, match="Invalid JSON constant"):
        PlotRecipe.from_json(
            b'{"steps": [{"name": "bad", "plot": "embedding", '
            b'"kwargs": {"pointSize": NaN}}]}'
        )


def test_output_path_rejects_symlink_escape_before_plotting(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "linked").symlink_to(outside, target_is_directory=True)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="linked/plot.png",
            )
        ]
    )

    with pytest.raises(ValueError, match="escapes output_dir"):
        run_recipe(object(), recipe, output_dir=output_root)


def test_runner_rejects_target_defined_in_kwargs_and_panel_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plotting,
        "embedding",
        lambda store, **kwargs: pytest.fail("plot should not be called"),
    )
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                kwargs={"target": "inline"},
                target=PlotPanelTarget("A"),
            )
        ]
    )

    with pytest.raises(ValueError, match="defines target in two places"):
        run_recipe(object(), recipe, targets={"A": "external"})


def test_runner_requires_artifact_mapping_before_plotting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plotting,
        "embedding",
        lambda store, **kwargs: pytest.fail("plot should not be called"),
    )
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                artifact_kwargs={"color_by": "clusterColumn"},
            )
        ]
    )

    with pytest.raises(ValueError, match="requires an artifact mapping"):
        run_recipe(object(), recipe)


@pytest.mark.parametrize(
    ("failure_point", "show", "output_filename", "error_type"),
    [
        ("save", False, "plot.png", OSError),
        ("show", True, None, RuntimeError),
    ],
)
def test_runner_closes_owned_result_after_save_or_show_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
    show: bool,
    output_filename: str | None,
    error_type: type[Exception],
) -> None:
    result = FakePlotResult("embedding")

    def fail_save(path: str | Path, **kwargs: object) -> Path:
        raise OSError("disk full")

    def fail_show() -> None:
        raise RuntimeError("display failed")

    if failure_point == "save":
        monkeypatch.setattr(result, "save", fail_save)
    else:
        monkeypatch.setattr(result, "show", fail_show)
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename=output_filename,
            )
        ]
    )

    with pytest.raises(error_type):
        run_recipe(
            object(),
            recipe,
            output_dir=tmp_path if output_filename is not None else None,
            show=show,
        )

    assert result.close_calls == 1
    assert not result.usable


def test_runner_preserves_primary_error_when_owned_result_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = FakePlotResult("embedding")

    def fail_save(path: str | Path, **kwargs: object) -> Path:
        raise OSError("disk full")

    def fail_close() -> None:
        result.close_calls += 1
        raise RuntimeError("close failed")

    monkeypatch.setattr(result, "save", fail_save)
    monkeypatch.setattr(result, "close", fail_close)
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="plot.png",
            )
        ]
    )

    with pytest.raises(OSError, match="disk full") as error:
        run_recipe(object(), recipe, output_dir=tmp_path)

    assert result.close_calls == 1
    assert error.value.__notes__ == [
        "The plot also failed to close cleanly: RuntimeError: close failed"
    ]


def test_runner_does_not_close_caller_owned_result_after_save_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = FakePlotResult("embedding", owns_figure=False)

    def fail_save(path: str | Path, **kwargs: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(result, "save", fail_save)
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="plot.png",
            )
        ]
    )

    with pytest.raises(OSError, match="disk full"):
        run_recipe(object(), recipe, output_dir=tmp_path)

    assert result.close_calls == 0
    assert result.usable


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("show", 1, "show must be a boolean"),
        ("continue_on_error", 1, "continue_on_error must be a boolean"),
        ("artifacts", [], "artifacts must be a mapping"),
        ("targets", [], "targets must be a mapping"),
    ],
)
def test_runner_validates_execution_context(
    argument: str,
    value: object,
    message: str,
) -> None:
    recipe = PlotRecipe([PlotStep(name="overview", plot="embedding")])

    with pytest.raises(TypeError, match=message):
        run_recipe(object(), recipe, **{argument: value})


def test_runner_reports_missing_or_noncallable_plot_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = PlotRecipe([PlotStep(name="overview", plot="embedding")])
    missing_module = object()
    monkeypatch.setattr(recipes_module, "import_module", lambda name: missing_module)

    with pytest.raises(RuntimeError, match="not available"):
        run_recipe(object(), recipe)

    class NonCallablePlotting:
        embedding = object()

    monkeypatch.setattr(
        recipes_module,
        "import_module",
        lambda name: NonCallablePlotting,
    )
    with pytest.raises(RuntimeError, match="not callable"):
        run_recipe(object(), recipe)
