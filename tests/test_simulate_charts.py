"""SVG geometry (`simulate/charts.py`), frontier and baseline rules, and the drawn report."""

import itertools
import re
from xml.etree import ElementTree

import pytest

from tests.simulate_fakes import make_outcome
from triage.report.builder import render_simulation_html, render_simulation_markdown
from triage.simulate import charts
from triage.simulate.summarize import summarize_simulation

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
LONG = "claude-3-7-sonnet-20250219"


def _result(*specs, logged_models=None):
    """Each spec is (model, passes, graded tasks, cost per task[, feature])."""
    outcomes = [
        make_outcome(
            f"t{i}",
            model,
            det=i < passes,
            projected=cost,
            feature=rest[0] if rest else "summarise",
        )
        for model, passes, n, cost, *rest in specs
        for i in range(n)
    ]
    targets = list(dict.fromkeys(spec[0] for spec in specs))
    return summarize_simulation(outcomes, targets=targets, logged_models=logged_models)


EXTREMES = {
    "top-and-bottom": [(HAIKU, 4, 4, "0.001"), (SONNET, 0, 4, "0.305")],
    "free": [(HAIKU, 1, 1, "0"), (SONNET, 0, 1, "0")],
    "identical": [(HAIKU, 1, 1, "0.01"), (SONNET, 1, 1, "0.01"), (OPUS, 1, 1, "0.01")],
    "all-failed-dearest": [(HAIKU, 0, 1, "0.076"), (SONNET, 0, 1, "0.305")],
    "crowded-long-ids": [(LONG, 2, 2, "0.050"), (SONNET, 2, 2, "0.049"), (OPUS, 0, 2, "0.050")],
}


def _all_charts(specs):
    built = charts.build_charts(_result(*specs))
    return [built.overall, *built.slices]


# --------------------------------------------------------------------------- geometry


@pytest.mark.parametrize("specs", EXTREMES.values(), ids=EXTREMES)
def test_marks_never_touch_the_axes_or_the_svg_edges(specs):
    for chart in _all_charts(specs):
        f = chart.frame
        reach = charts.RADIUS + 1  # circle radius plus half its stroke
        for p in chart.points:
            assert f.left + reach < p.x < f.right - reach
            for y in (p.y, p.y_low, p.y_high):
                assert f.top + reach < y < f.bottom - reach
        assert 0 < f.left and f.right < f.width
        assert 0 <= f.top and f.bottom + 40 < f.height  # ticks and axis title fit below


@pytest.mark.parametrize("specs", EXTREMES.values(), ids=EXTREMES)
def test_labels_stay_in_the_plot_and_clear_of_ticks_circles_and_each_other(specs):
    for chart in _all_charts(specs):
        f = chart.frame
        boxes = [charts.label_box(p.label, p.summary.model) for p in chart.points]
        for box in boxes:
            assert f.left <= box.x0 and box.x1 <= f.right
            assert f.top <= box.y0 and box.y1 <= f.bottom  # tick labels start below the axis
        for a, b in itertools.combinations(boxes, 2):
            assert not a.overlaps(b)
        for box in boxes:
            for p in chart.points:
                assert not (box.x0 < p.x < box.x1 and box.y0 < p.y < box.y1)


def test_a_label_too_long_for_the_plot_is_still_pinned_inside_it():
    result = _result((HAIKU, 1, 1, "0.01"))
    row = result.overall[0].model_copy(update={"model": "m" * 200})

    (point,) = charts.build_chart("t", [row], charts.SLICE).points

    assert point.label.x >= charts.SLICE.left
    assert charts.SLICE.top <= point.label.y <= charts.SLICE.bottom


def test_axes_scale_cost_rightwards_and_pass_rate_upwards():
    chart = charts.build_charts(_result((HAIKU, 1, 4, "0.01"), (OPUS, 4, 4, "0.02"))).overall
    haiku, opus = chart.points

    assert [line.percent for line in chart.grid] == [0, 25, 50, 75, 100]
    assert [line.y for line in chart.grid] == sorted((g.y for g in chart.grid), reverse=True)
    assert haiku.y == chart.grid[1].y and opus.y == chart.grid[4].y
    assert opus.x == chart.frame.x_max
    assert haiku.x == pytest.approx((chart.frame.x_min + chart.frame.x_max) / 2, abs=0.01)


@pytest.mark.parametrize(("n", "low_n"), [(4, True), (5, False)])
def test_intervals_from_few_tasks_are_marked_for_dashing(n, low_n):
    (point,) = charts.build_charts(_result((HAIKU, n, n, "0.01"))).overall.points

    assert point.low_n is low_n


# --------------------------------------------------------------------------- frontier


def test_models_without_a_pass_are_never_on_the_frontier():
    result = _result((HAIKU, 0, 4, "0.001"), (SONNET, 0, 4, "0.002"))

    assert not any(row.on_frontier for row in [*result.summaries, *result.overall])
    assert charts.build_charts(result).overall.frontier_path == ""
    assert 'class="frontier" data-model' not in render_simulation_html(result)
    assert "| yes |" not in render_simulation_markdown(result)


def test_frontier_step_runs_through_lower_bounds_to_the_right_edge():
    specs = (HAIKU, 2, 8, "0.001"), (SONNET, 8, 8, "0.01"), (OPUS, 4, 8, "0.02")

    chart = charts.build_charts(_result(*specs)).overall

    cheap, best, dominated = chart.points
    assert [p.summary.on_frontier for p in chart.points] == [True, True, False]
    assert chart.frontier_path == (
        f"M {cheap.x} {cheap.y_low} H {best.x} V {best.y_low} H {chart.frame.x_max}"
    )
    assert str(dominated.y_low) not in chart.frontier_path


# --------------------------------------------------------------------------- logged model


@pytest.mark.parametrize(
    ("logged", "shown"),
    [({("summarise", "v1"): OPUS}, OPUS), ({("summarise", "v1"): "mixed"}, "mixed"), (None, "n/a")],
    ids=["single", "mixed", "unknown"],
)
def test_both_reports_show_the_logged_model_in_place_of_baseline(logged, shown):
    result = _result((HAIKU, 1, 1, "0.07"), logged_models=logged)

    html = render_simulation_html(result)
    markdown = render_simulation_markdown(result)

    assert "<th>Logged model</th>" in html and f'<td class="model">{shown}</td></tr>' in html
    row = next(r for r in markdown.splitlines() if r.startswith("| summarise /"))
    assert row.endswith(f"| yes | {shown} |")
    assert "Baseline" not in html and "Baseline" not in markdown


def test_logged_model_text_from_the_input_file_is_escaped():
    result = _result((HAIKU, 1, 1, "0.07"), logged_models={("summarise", "v1"): "<b>a|b</b>"})

    assert "<b>a" not in render_simulation_html(result)
    markdown = render_simulation_markdown(result)
    assert "<b>a" not in markdown and "a|b" not in markdown


# --------------------------------------------------------------------------- report


def _svgs(html: str, kind: str):
    found = re.findall(rf'<svg class="{kind}".*?</svg>', html, re.DOTALL)
    return [ElementTree.fromstring(svg) for svg in found]


def test_overall_chart_comes_first_with_gridlines_caps_and_legend():
    html = render_simulation_html(_result((HAIKU, 3, 4, "0.01"), (SONNET, 3, 4, "0.05")))

    (overall,) = _svgs(html, "overall")
    assert html.index('<svg class="overall"') < html.index('<svg class="slice"')
    classes = [element.get("class", "") for element in overall.iter()]
    assert classes.count("grid") == 5
    assert classes.count("whisker low-n") == 3  # two models and the legend sample
    assert all(len(g) == 3 for g in overall.iter("g"))  # stem and two end-caps
    text = " ".join(element.text or "" for element in overall.iter("text"))
    for entry in ("Frontier", "Dominated", "95% Wilson CI", "Dashed: n < 5"):
        assert entry in text


@pytest.mark.parametrize(("n", "collapsed"), [(1, True), (2, True), (3, False)])
def test_slice_charts_collapse_only_when_every_slice_is_tiny(n, collapsed):
    result = _result((HAIKU, 1, n, "0.01", "a"), (HAIKU, 1, 1, "0.01", "b"))

    html = render_simulation_html(result)

    assert ("<details>" in html) is collapsed
    assert len(_svgs(html, "slice")) == 2
