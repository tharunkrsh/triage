"""DuckDB aggregation (`simulate/summarize.py`) and the simulate reports."""

import math
import re
from decimal import Decimal
from xml.etree import ElementTree

import duckdb
import pytest

from tests.simulate_fakes import make_outcome
from triage.report import builder
from triage.report.builder import render_simulation_html, render_simulation_markdown
from triage.simulate.stats import wilson_interval
from triage.simulate.summarize import summarize_simulation

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"


def _one(rows, model, feature=None):
    (row,) = [r for r in rows if r.model == model and r.feature == feature]
    return row


def _graded(model, passes, total, *, projected="0.01", feature="summarise"):
    return [
        make_outcome(f"t{i}", model, det=i < passes, projected=projected, feature=feature)
        for i in range(total)
    ]


# --------------------------------------------------------------------------- summarise


def test_counts_pass_rate_and_costs_per_group():
    outcomes = [
        make_outcome("t1", det=True, sim="0.001", projected="0.002"),
        make_outcome("t2", det=False, sim="0.003", projected="0.006"),
        make_outcome("t3", det=True, sim="0.002", projected="0.004"),
        make_outcome("t4", status="errored"),
        make_outcome("t5", status="expired"),
    ]

    result = summarize_simulation(outcomes, targets=[SONNET])

    row = _one(result.summaries, SONNET, "summarise")
    assert row.prompt_version == "v1"
    assert (row.n, row.passes, row.errored) == (3, 2, 2)
    assert row.pass_rate == wilson_interval(2, 3)
    assert row.sim_cost == Decimal("0.006")
    assert row.projected_cost_per_task == Decimal("0.004")  # failed replays don't dilute it
    assert row.cost_per_success == Decimal("0.006")  # 0.012 / 2


def test_ungraded_successes_count_toward_cost_but_not_pass_rate():
    outcomes = [
        make_outcome("t1", det=True, projected="0.002"),
        make_outcome("t2", projected="0.004"),
    ]

    row = summarize_simulation(outcomes, targets=[SONNET]).summaries[0]

    assert (row.n, row.passes, row.errored) == (1, 1, 0)
    assert row.projected_cost_per_task == Decimal("0.003")
    assert row.cost_per_success == Decimal("0.002")  # graded spend / passes


def test_groups_split_by_feature_and_prompt_version_including_none():
    outcomes = [
        make_outcome("t1", det=True),
        make_outcome("t2", det=True, prompt_version="v2"),
        make_outcome("t3", det=False, feature=None, prompt_version=None),
    ]

    result = summarize_simulation(outcomes, targets=[SONNET])

    assert {(r.feature, r.prompt_version, r.n) for r in result.summaries} == {
        ("summarise", "v1", 1),
        ("summarise", "v2", 1),
        (None, None, 1),
    }


@pytest.mark.parametrize(
    ("costs", "expected"),
    [
        (["0.1", "0.2"], "0.15"),
        (["0.00000005", "0.00000005"], "0.00000005"),
        (["0.000001", "0.000001", "0.000002"], "0.0000013333"),
    ],
)
def test_average_cost_converts_cleanly_to_decimal(costs, expected):
    outcomes = [make_outcome(f"t{i}", det=True, projected=c) for i, c in enumerate(costs)]

    result = summarize_simulation(outcomes, targets=[SONNET])

    for row in (result.summaries[0], result.overall[0]):
        value = row.projected_cost_per_task
        assert isinstance(value, Decimal)
        assert value.quantize(Decimal("1e-10")) == Decimal(expected)
        assert value.as_tuple().exponent >= -12  # no binary-float digit tail


def test_aggregation_runs_in_an_in_memory_duckdb(monkeypatch):
    databases: list[str] = []
    original = duckdb.connect

    def spy(database=":memory:", *args, **kwargs):
        databases.append(str(database))
        return original(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", spy)

    summarize_simulation([make_outcome(det=True)], targets=[SONNET])

    assert databases
    assert set(databases) == {":memory:"}


def test_overall_pools_groups_and_lists_every_target_in_order():
    outcomes = [
        *_graded(SONNET, 2, 2, projected="0.01", feature="a"),
        *_graded(SONNET, 0, 2, projected="0.03", feature="b"),
    ]

    result = summarize_simulation(outcomes, targets=[HAIKU, SONNET])

    assert result.targets == [HAIKU, SONNET]
    assert [r.model for r in result.overall] == [HAIKU, SONNET]
    assert all(r.feature is None and r.prompt_version is None for r in result.overall)
    pooled = _one(result.overall, SONNET)
    assert (pooled.n, pooled.passes) == (4, 2)
    assert pooled.pass_rate == wilson_interval(2, 4)
    assert pooled.projected_cost_per_task == Decimal("0.02")
    empty = _one(result.overall, HAIKU)
    assert (empty.n, empty.passes, empty.errored) == (0, 0, 0)
    assert empty.pass_rate == wilson_interval(0, 0)
    assert empty.projected_cost_per_task == Decimal(0)
    assert empty.cost_per_success is None
    assert empty.on_frontier is False


def test_cost_per_success_is_none_without_passes():
    result = summarize_simulation(_graded(SONNET, 0, 3), targets=[SONNET])

    assert result.summaries[0].cost_per_success is None


def test_all_errored_group_is_empty_evidence():
    result = summarize_simulation([make_outcome(status="errored")], targets=[SONNET])

    row = result.summaries[0]
    assert (row.n, row.errored, row.pass_rate.p_hat) == (0, 1, None)
    assert row.projected_cost_per_task == Decimal(0)
    assert row.on_frontier is False


def test_frontier_flags_and_logged_model_per_slice():
    outcomes = [
        *_graded(HAIKU, 5, 5, projected="0.01"),
        *_graded(SONNET, 3, 5, projected="0.05"),  # dearer and worse
    ]

    result = summarize_simulation(
        outcomes, targets=[HAIKU, SONNET], logged_models={("summarise", "v1"): OPUS}
    )

    assert _one(result.summaries, HAIKU, "summarise").on_frontier is True
    assert _one(result.summaries, SONNET, "summarise").on_frontier is False
    assert _one(result.overall, HAIKU).on_frontier is True
    assert [row.logged_model for row in result.summaries] == [OPUS, OPUS]
    assert [row.logged_model for row in result.overall] == [None, None]


def test_agreement_compares_judge_with_deterministic_grades():
    outcomes = [
        make_outcome("t1", det=True, judge=True),
        make_outcome("t2", det=False, judge=False),
        make_outcome("t3", det=True, judge=False),
        make_outcome("t4", det=True),  # no judge: excluded
        make_outcome("t5", judge=True),  # no deterministic check: excluded
    ]

    result = summarize_simulation(outcomes, targets=[SONNET], judge_model=OPUS)

    assert result.judge_model == OPUS
    assert result.agreement == wilson_interval(2, 3)
    assert result.summaries[0].agreement == wilson_interval(2, 3)


def test_agreement_is_none_when_nothing_is_doubly_graded():
    result = summarize_simulation([make_outcome(det=True)], targets=[SONNET])

    assert result.agreement is None
    assert result.summaries[0].agreement is None


def test_total_spend_includes_the_judge():
    outcomes = [make_outcome("t1", sim="0.001"), make_outcome("t2", sim="0.002")]

    result = summarize_simulation(
        outcomes, targets=[SONNET], judge_cost=Decimal("0.0005"), source="replay.jsonl"
    )

    assert result.total_sim_cost == Decimal("0.0035")
    assert result.source == "replay.jsonl"


def test_no_outcomes_still_summarises_targets():
    result = summarize_simulation([], targets=[SONNET])

    assert result.summaries == []
    assert [r.n for r in result.overall] == [0]
    assert result.total_sim_cost == Decimal(0)


# --------------------------------------------------------------------------- ci filter


def test_ci_shows_the_point_estimate_and_bounds():
    interval = wilson_interval(3, 4)

    text = builder.ci(interval)

    for value in (interval.p_hat, interval.low, interval.high):
        assert builder.pct(value) in text


@pytest.mark.parametrize("interval", [wilson_interval(0, 0), None])
def test_ci_renders_missing_evidence_as_na(interval):
    assert "n/a" in builder.ci(interval)


@pytest.mark.parametrize("html", [True, False])
def test_ci_is_registered_as_a_template_filter(html):
    assert builder._environment(html=html).filters["ci"] is builder.ci


# --------------------------------------------------------------------------- reports


def _render_both(result, outcomes=()):
    return (
        render_simulation_html(result, outcomes),
        render_simulation_markdown(result, outcomes),
    )


def test_reports_name_models_pass_rates_and_spend():
    outcomes = _graded(HAIKU, 3, 4)
    result = summarize_simulation(outcomes, targets=[HAIKU], judge_cost=Decimal("1.25"))

    for report in _render_both(result, outcomes):
        assert HAIKU in report
        assert builder.pct(0.75) in report
        assert builder.usd(result.total_sim_cost) in report


@pytest.mark.parametrize(
    "outcomes",
    [[], [make_outcome(status="errored")], [make_outcome()]],
    ids=["no-outcomes", "all-errored", "ungraded"],
)
def test_reports_render_rows_with_no_graded_tasks(outcomes):
    result = summarize_simulation(outcomes, targets=[SONNET])

    for report in _render_both(result, outcomes):
        assert "n/a" in report
        assert "None" not in report


def test_html_autoescapes_model_text():
    graded = make_outcome(
        "<i>task</i>",
        feature="<b>feat</b>",
        text="<script>alert('x')</script>",
        judge=False,
    )
    verdict = graded.judge.model_copy(update={"rationale": "<img src=x onerror=alert(1)>"})
    outcomes = [graded.model_copy(update={"judge": verdict})]
    result = summarize_simulation(outcomes, targets=[SONNET], source="<u>file</u>.jsonl")

    html = render_simulation_html(result, outcomes)

    for raw in ("<script>alert", "<img src=x", "<b>feat", "<i>task", "<u>file"):
        assert raw not in html
    # Escaped, not dropped: failing responses and judge rationales are shown.
    for escaped in ("&lt;script&gt;", "&lt;img src=x", "&lt;b&gt;feat", "&lt;i&gt;task"):
        assert escaped in html


def test_markdown_cells_survive_pipes_and_newlines():
    outcomes = [make_outcome(det=True, feature="a|b\nc")]
    result = summarize_simulation(outcomes, targets=[SONNET])

    markdown = render_simulation_markdown(result, outcomes)

    assert "a&#124;b c" in markdown
    assert "a|b" not in markdown


def test_html_is_self_contained():
    result = summarize_simulation(_graded(SONNET, 1, 2), targets=[SONNET])

    html = render_simulation_html(result)

    assert "<script" not in html
    assert not re.search(r"""(src|href)\s*=\s*["']https?:""", html)


# --------------------------------------------------------------------------- inline SVG


def _points(html: str) -> dict[str, tuple[float, float]]:
    """`data-model` -> (cx, cy) for the plotted circles of the overall chart."""
    (svg,) = re.findall(r'<svg class="overall".*?</svg>', html, re.DOTALL)
    found = {}
    for element in ElementTree.fromstring(svg).iter():
        if element.tag.rsplit("}", 1)[-1] == "circle" and "data-model" in element.attrib:
            found[element.attrib["data-model"]] = (
                float(element.get("cx")),
                float(element.get("cy")),
            )
    assert not re.search(r"\b(nan|inf)\b", svg, re.IGNORECASE)
    return found


@pytest.mark.parametrize(
    ("targets", "costs"),
    [
        ([SONNET], ["0.01"]),
        ([SONNET], ["0"]),
        ([SONNET, HAIKU], ["0.01", "0.01"]),
        ([SONNET, HAIKU], ["0", "0"]),
        ([SONNET, HAIKU, OPUS], ["0.002", "0.0001", "0.05"]),
    ],
    ids=["single", "single-free", "equal-cost", "all-free", "spread"],
)
def test_svg_plots_each_model_without_dividing_by_zero(targets, costs):
    outcomes = [
        o
        for model, cost in zip(targets, costs, strict=True)
        for o in _graded(model, 1, 2, projected=cost)
    ]
    result = summarize_simulation(outcomes, targets=targets)

    points = _points(render_simulation_html(result, outcomes))

    assert set(points) == set(targets)
    assert all(math.isfinite(v) and v >= 0 for xy in points.values() for v in xy)


def test_svg_places_costlier_models_further_right_and_better_models_higher():
    outcomes = [
        *_graded(HAIKU, 1, 4, projected="0.001"),
        *_graded(OPUS, 4, 4, projected="0.050"),
    ]
    result = summarize_simulation(outcomes, targets=[HAIKU, OPUS])

    points = _points(render_simulation_html(result, outcomes))

    assert points[HAIKU][0] < points[OPUS][0]
    assert points[HAIKU][1] > points[OPUS][1]  # SVG y grows downward


def test_svg_omits_models_without_evidence():
    outcomes = _graded(SONNET, 1, 1)
    result = summarize_simulation(outcomes, targets=[SONNET, HAIKU])

    assert set(_points(render_simulation_html(result, outcomes))) == {SONNET}


def test_chart_is_safe_when_no_model_has_evidence():
    result = summarize_simulation([], targets=[SONNET])

    html = render_simulation_html(result)

    assert "<circle" not in html
