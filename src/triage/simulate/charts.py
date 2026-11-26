"""SVG geometry for the simulate report's cost vs pass-rate charts.

Everything is computed here in viewBox units; `report/templates/simulate.html.j2` only draws it.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from triage.simulate.models import CandidateSummary, SimulationResult

RADIUS = 6
CAP = 6  # half-width of a whisker end-cap
LOW_N = 5  # intervals from fewer graded tasks are drawn dashed
COLLAPSE_BELOW_N = 3  # slice charts start collapsed when every slice has fewer tasks
GRID_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)
CHAR_WIDTH = 7.0  # generous estimate for the 12px label font
_ASCENT, _DESCENT, _GAP = 10, 3, 10
_LABEL_OFFSETS = (4, -12, 20, -28, 36, -44, 52, -60, 68)  # baseline relative to the point


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Frame(_Model):
    """viewBox size, the axes, and the padding that keeps marks off the axes and edges."""

    width: int
    height: int
    left: int
    right: int
    top: int
    bottom: int
    pad: int

    @property
    def x_min(self) -> int:
        return self.left + self.pad

    @property
    def x_max(self) -> int:
        return self.right - self.pad

    def x(self, share: float) -> float:
        return round(self.x_min + share * (self.x_max - self.x_min), 2)

    def y(self, rate: float) -> float:
        low, high = self.bottom - self.pad, self.top + self.pad
        return round(low - rate * (low - high), 2)


# The overall chart leaves room above the plot for its legend.
OVERALL = Frame(width=760, height=390, left=64, right=740, top=52, bottom=332, pad=18)
SLICE = Frame(width=520, height=260, left=56, right=504, top=8, bottom=206, pad=16)


class Label(_Model):
    x: float
    y: float
    anchor: Literal["start", "end"]


class Point(_Model):
    summary: CandidateSummary
    x: float
    y: float
    y_low: float
    y_high: float
    low_n: bool
    label: Label


class GridLine(_Model):
    percent: int
    y: float


class Chart(_Model):
    title: str
    frame: Frame
    max_cost: Decimal
    grid: list[GridLine]
    points: list[Point]
    frontier_path: str  # SVG path data; empty when nothing is on the frontier


class SimulationCharts(_Model):
    overall: Chart | None = None
    slices: list[Chart] = []
    collapse_slices: bool = False


def group_label(feature: str | None, prompt_version: str | None) -> str:
    return f"{feature or '(no feature)'} / {prompt_version or '(no version)'}"


class _Box(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: _Box) -> bool:
        return (
            self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1
        )


def label_box(label: Label, text: str) -> _Box:
    width = len(text) * CHAR_WIDTH
    x0 = label.x if label.anchor == "start" else label.x - width
    return _Box(x0, label.y - _ASCENT, x0 + width, label.y + _DESCENT)


def _place_label(
    frame: Frame, x: float, y: float, text: str, marks: list[_Box], whiskers: list[_Box]
) -> Label:
    """The first position beside the point that is inside the plot and clear of other marks.

    Labels never leave the plot area, so they can't run into the axis ticks. `marks` (circles
    and earlier labels) matter more than `whiskers`, which a label may cross when crowded.
    """
    sides = ("start", "end") if x <= (frame.left + frame.right) / 2 else ("end", "start")
    inside: list[tuple[Label, _Box]] = []
    for offset in _LABEL_OFFSETS:
        for anchor in sides:
            dx = _GAP if anchor == "start" else -_GAP
            label = Label(x=round(x + dx, 2), y=round(y + offset, 2), anchor=anchor)
            box = label_box(label, text)
            if (
                box.x0 >= frame.left
                and box.x1 <= frame.right
                and box.y0 >= frame.top
                and box.y1 <= frame.bottom
            ):
                inside.append((label, box))
    for obstacles in (marks + whiskers, marks):
        for label, box in inside:
            if not any(box.overlaps(other) for other in obstacles):
                return label
    if inside:
        return inside[0][0]
    # A model id too long for either side: pin it inside the plot's left edge.
    return Label(x=frame.left + 2, y=max(y - 12, frame.top + _ASCENT), anchor="start")


def _frontier_path(frame: Frame, points: Sequence[Point]) -> str:
    """Step line through the frontier at each interval's lower bound, the compared value.

    It runs right from each frontier model until a dearer one raises the bound.
    """
    steps = sorted((p.x, p.y_low) for p in points if p.summary.on_frontier)
    if not steps:
        return ""
    (x, y), *rest = steps
    path = f"M {x} {y}" + "".join(f" H {x} V {y}" for x, y in rest)
    return f"{path} H {frame.x_max}"


def build_chart(title: str, rows: Sequence[CandidateSummary], frame: Frame) -> Chart:
    """Cost (x) against pass rate (y) for rows that have graded tasks."""
    max_cost = max(row.projected_cost_per_task for row in rows)
    span = float(max_cost) or 1.0  # an all-free group plots at zero cost
    spots = [
        (
            row,
            frame.x(float(row.projected_cost_per_task) / span),
            frame.y(row.pass_rate.p_hat),
            frame.y(row.pass_rate.low),
            frame.y(row.pass_rate.high),
        )
        for row in rows
    ]
    # Every circle and whisker is an obstacle before the first label is placed.
    marks = [
        _Box(x - RADIUS - 1, y - RADIUS - 1, x + RADIUS + 1, y + RADIUS + 1)
        for _, x, y, _, _ in spots
    ]
    whiskers = [_Box(x - CAP, high, x + CAP, low) for _, x, _, low, high in spots]
    points = []
    for row, x, y, y_low, y_high in spots:
        label = _place_label(frame, x, y, row.model, marks, whiskers)
        marks.append(label_box(label, row.model))
        points.append(
            Point(
                summary=row,
                x=x,
                y=y,
                y_low=y_low,
                y_high=y_high,
                low_n=row.n < LOW_N,
                label=label,
            )
        )
    return Chart(
        title=title,
        frame=frame,
        max_cost=max_cost,
        grid=[GridLine(percent=round(rate * 100), y=frame.y(rate)) for rate in GRID_RATES],
        points=points,
        frontier_path=_frontier_path(frame, points),
    )


def build_charts(result: SimulationResult) -> SimulationCharts:
    """The overall chart plus one per (feature, prompt_version) group with graded tasks."""
    overall = [row for row in result.overall if row.n]
    groups: dict[str, list[CandidateSummary]] = {}
    for row in result.summaries:
        if row.n:
            groups.setdefault(group_label(row.feature, row.prompt_version), []).append(row)
    return SimulationCharts(
        overall=build_chart("Models overall", overall, OVERALL) if overall else None,
        slices=[build_chart(title, rows, SLICE) for title, rows in groups.items()],
        collapse_slices=all(row.n < COLLAPSE_BELOW_N for rows in groups.values() for row in rows),
    )
