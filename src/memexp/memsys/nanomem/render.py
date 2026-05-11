from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from dataclasses import dataclass

from memexp.core.contracts import PackedContext, RankedMemoryUnit
from memexp.core.time import parse_timestamp as _parse_timestamp
from memexp.memsys.nanomem.config import RenderConfig
from memexp.memsys.nanomem.utils import estimate_tokens

SUPPORTED_RENDER_POLICIES = {
    "adaptive_markdown_temporal_v1",
    "timeline_v1",
    "ranked_list_v1",
}


def _sort_units(
    units: tuple[RankedMemoryUnit, ...],
    *,
    sort_by_time: bool,
) -> list[RankedMemoryUnit]:
    if not sort_by_time:
        return sorted(units, key=lambda item: (item.rank, -item.score))

    def sort_key(item: RankedMemoryUnit) -> tuple[int, dt.datetime, int]:
        parsed = _parse_timestamp(item.unit.timestamp)
        if parsed is None:
            return (1, dt.datetime.max, item.rank)
        return (0, parsed, item.rank)

    return sorted(units, key=sort_key)


@dataclass(frozen=True)
class _RenderItem:
    ranked_unit: RankedMemoryUnit
    text: str
    timestamp: str
    parsed_time: dt.datetime | None

    @property
    def unit_id(self) -> str:
        return self.ranked_unit.unit.unit_id

    @property
    def rank(self) -> int:
        return self.ranked_unit.rank

    @property
    def score(self) -> float:
        return self.ranked_unit.score


@dataclass(frozen=True)
class _RenderCandidate:
    text: str
    token_count: int
    structure_score: int


def _to_render_items(units: list[RankedMemoryUnit]) -> list[_RenderItem]:
    items: list[_RenderItem] = []
    for item in units:
        text = item.unit.text.strip()
        if not text:
            continue
        timestamp = str(item.unit.timestamp or "unknown")
        items.append(
            _RenderItem(
                ranked_unit=item,
                text=text,
                timestamp=timestamp,
                parsed_time=_parse_timestamp(item.unit.timestamp),
            ))
    return items


def _full_date_label(item: _RenderItem) -> str:
    if item.parsed_time is not None:
        return item.parsed_time.strftime("%Y-%m-%d")
    return item.timestamp.strip() or "unknown"


def _year_label(item: _RenderItem) -> str:
    if item.parsed_time is None:
        return "Unknown"
    return item.parsed_time.strftime("%Y")


def _month_label(item: _RenderItem) -> str:
    if item.parsed_time is None:
        return "Unknown"
    return item.parsed_time.strftime("%m")


def _day_label(item: _RenderItem) -> str:
    if item.parsed_time is None:
        return "Unknown"
    return item.parsed_time.strftime("%d")


def _month_day_label(item: _RenderItem) -> str:
    if item.parsed_time is None:
        return _full_date_label(item)
    return item.parsed_time.strftime("%m-%d")


def _sort_render_items(
    items: list[_RenderItem],
    *,
    sort_by_time: bool,
) -> list[_RenderItem]:
    if not sort_by_time:
        return sorted(items,
                      key=lambda item: (item.rank, -item.score, item.unit_id))

    def sort_key(item: _RenderItem) -> tuple[int, dt.datetime, int, str]:
        if item.parsed_time is None:
            return (1, dt.datetime.max, item.rank, item.unit_id)
        return (0, item.parsed_time, item.rank, item.unit_id)

    return sorted(items, key=sort_key)


def _group_key(
    items: list[_RenderItem],
    *,
    sort_by_time: bool,
) -> tuple[int, dt.datetime, int, str]:
    parsed_times = [
        item.parsed_time for item in items if item.parsed_time is not None
    ]
    best_rank = min(item.rank for item in items)
    best_score = max(item.score for item in items)
    first_id = min(item.unit_id for item in items)
    if sort_by_time:
        if not parsed_times:
            return (1, dt.datetime.max, best_rank, first_id)
        return (0, min(parsed_times), best_rank, first_id)
    return (0, dt.datetime.min, best_rank, f"{-best_score}:{first_id}")


def _render_flat_markdown(items: list[_RenderItem], *,
                          include_timestamps: bool) -> str:
    lines: list[str] = []
    for item in items:
        if include_timestamps:
            lines.append(f"- {_full_date_label(item)}: {item.text}")
        else:
            lines.append(f"- {item.text}")
    return "\n".join(lines)


def _render_year_markdown(items: list[_RenderItem], *,
                          sort_by_time: bool) -> str:
    years: OrderedDict[str, list[_RenderItem]] = OrderedDict()
    for item in items:
        years.setdefault(_year_label(item), []).append(item)

    lines: list[str] = []
    ordered_years = sorted(
        years.items(),
        key=lambda entry: _group_key(entry[1], sort_by_time=sort_by_time),
    )
    for year, year_items in ordered_years:
        lines.append(f"# {year}")
        for item in _sort_render_items(year_items, sort_by_time=sort_by_time):
            if item.parsed_time is None:
                lines.append(f"- {item.text}")
            else:
                lines.append(f"- {_month_day_label(item)}: {item.text}")
    return "\n".join(lines)


def _render_month_markdown(items: list[_RenderItem], *,
                           sort_by_time: bool) -> str:
    years: OrderedDict[str, list[_RenderItem]] = OrderedDict()
    for item in items:
        years.setdefault(_year_label(item), []).append(item)

    lines: list[str] = []
    ordered_years = sorted(
        years.items(),
        key=lambda entry: _group_key(entry[1], sort_by_time=sort_by_time),
    )
    for year, year_items in ordered_years:
        lines.append(f"# {year}")
        months: OrderedDict[str, list[_RenderItem]] = OrderedDict()
        for item in year_items:
            months.setdefault(_month_label(item), []).append(item)
        ordered_months = sorted(
            months.items(),
            key=lambda entry: _group_key(entry[1], sort_by_time=sort_by_time),
        )
        for month, month_items in ordered_months:
            if month == "Unknown":
                for item in _sort_render_items(month_items,
                                               sort_by_time=sort_by_time):
                    lines.append(f"- {item.text}")
                continue
            lines.append(f"## {month}")
            for item in _sort_render_items(month_items,
                                           sort_by_time=sort_by_time):
                lines.append(f"- {_day_label(item)}: {item.text}")
    return "\n".join(lines)


def _render_day_markdown(
    items: list[_RenderItem],
    *,
    sort_by_time: bool,
    min_group_size: int,
) -> str:
    years: OrderedDict[str, list[_RenderItem]] = OrderedDict()
    for item in items:
        years.setdefault(_year_label(item), []).append(item)

    lines: list[str] = []
    ordered_years = sorted(
        years.items(),
        key=lambda entry: _group_key(entry[1], sort_by_time=sort_by_time),
    )
    for year, year_items in ordered_years:
        lines.append(f"# {year}")
        months: OrderedDict[str, list[_RenderItem]] = OrderedDict()
        for item in year_items:
            months.setdefault(_month_label(item), []).append(item)
        ordered_months = sorted(
            months.items(),
            key=lambda entry: _group_key(entry[1], sort_by_time=sort_by_time),
        )
        for month, month_items in ordered_months:
            if month == "Unknown":
                for item in _sort_render_items(month_items,
                                               sort_by_time=sort_by_time):
                    lines.append(f"- {item.text}")
                continue
            lines.append(f"## {month}")
            days: OrderedDict[str, list[_RenderItem]] = OrderedDict()
            for item in month_items:
                days.setdefault(_day_label(item), []).append(item)
            ordered_days = sorted(
                days.items(),
                key=lambda entry: _group_key(entry[1],
                                             sort_by_time=sort_by_time),
            )
            for day, day_items in ordered_days:
                if len(day_items) < min_group_size:
                    for item in _sort_render_items(day_items,
                                                   sort_by_time=sort_by_time):
                        lines.append(f"- {day}: {item.text}")
                    continue
                lines.append(f"### {day}")
                for item in _sort_render_items(day_items,
                                               sort_by_time=sort_by_time):
                    lines.append(f"- {item.text}")
    return "\n".join(lines)


def _merge_lines_by_timestamp(
        lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for timestamp, text in lines:
        ts = timestamp.strip() or "unknown"
        grouped.setdefault(ts, []).append(text)
    return [(timestamp, " | ".join(item for item in items if item.strip()))
            for timestamp, items in grouped.items()]


class RenderPolicy:

    def __init__(self, config: RenderConfig) -> None:
        if config.policy not in SUPPORTED_RENDER_POLICIES:
            raise ValueError(
                f"Unsupported NanoMem render policy: {config.policy}")
        if config.merge_policy != "temporal_metadata_merge_v1":
            raise ValueError(
                f"Unsupported NanoMem merge policy: {config.merge_policy}")
        self.config = config

    def render(
        self,
        ranked_units: tuple[RankedMemoryUnit, ...],
        *,
        budget_tokens: int | None = None,
    ) -> PackedContext:
        budget = budget_tokens or self.config.context_tokens
        if self.config.policy == "adaptive_markdown_temporal_v1":
            return self._render_adaptive_markdown(ranked_units,
                                                  budget_tokens=budget)

        selected = self._select_under_budget(ranked_units,
                                             budget_tokens=budget)
        if not selected:
            return PackedContext(text="",
                                 token_count=0,
                                 block_count=0,
                                 timepoint_count=0)
        return self._render_units(tuple(selected))

    def _select_under_budget(
        self,
        ranked_units: tuple[RankedMemoryUnit, ...],
        *,
        budget_tokens: int,
    ) -> list[RankedMemoryUnit]:
        ordered = sorted(ranked_units,
                         key=lambda item: (item.rank, -item.score))
        selected: list[RankedMemoryUnit] = []
        for candidate in ordered:
            next_selected = selected + [candidate]
            rendered = self._render_units(tuple(next_selected))
            if rendered.token_count > budget_tokens:
                break
            selected = next_selected
        return selected

    def _render_units(
        self,
        ranked_units: tuple[RankedMemoryUnit, ...],
    ) -> PackedContext:
        lines: list[tuple[str, str]] = []
        for item in _sort_units(ranked_units,
                                sort_by_time=self.config.sort_by_time):
            text = item.unit.text.strip()
            if not text:
                continue
            timestamp = str(item.unit.timestamp or "unknown")
            lines.append((timestamp, text))

        if self.config.merge_same_timestamp:
            lines = _merge_lines_by_timestamp(lines)

        blocks: list[str] = []
        for timestamp, text in lines:
            if self.config.include_timestamps:
                blocks.append(f"[{timestamp}] {text}")
            else:
                blocks.append(text)

        rendered_text = "\n\n".join(blocks)
        timepoints = {timestamp for timestamp, text in lines if text.strip()}
        return PackedContext(
            text=rendered_text,
            token_count=estimate_tokens(rendered_text),
            block_count=len(blocks),
            timepoint_count=len(timepoints),
        )

    def _render_adaptive_markdown(
        self,
        ranked_units: tuple[RankedMemoryUnit, ...],
        *,
        budget_tokens: int,
    ) -> PackedContext:
        selected = [
            item for item in sorted(ranked_units,
                                    key=lambda item: (item.rank, -item.score))
            if item.unit.text.strip()
        ]
        while selected:
            rendered = self._render_best_markdown(tuple(selected))
            if rendered.token_count <= budget_tokens:
                return rendered
            selected.remove(min(selected, key=self._prune_key))
        return PackedContext(text="",
                             token_count=0,
                             block_count=0,
                             timepoint_count=0)

    def _prune_key(self, item: RankedMemoryUnit) -> tuple[float, int, str]:
        text_cost = max(1, estimate_tokens(item.unit.text))
        rank_utility = 1.0 / max(1, item.rank)
        score_utility = max(0.0, item.score)
        utility_per_token = (rank_utility + score_utility) / text_cost
        return (utility_per_token, -item.rank, item.unit.unit_id)

    def _render_best_markdown(
        self,
        ranked_units: tuple[RankedMemoryUnit, ...],
    ) -> PackedContext:
        items = _sort_render_items(_to_render_items(list(ranked_units)),
                                   sort_by_time=self.config.sort_by_time)
        if not items:
            return PackedContext(text="",
                                 token_count=0,
                                 block_count=0,
                                 timepoint_count=0)

        candidates = [
            _RenderCandidate(
                text=_render_flat_markdown(
                    items,
                    include_timestamps=self.config.include_timestamps,
                ),
                token_count=0,
                structure_score=0,
            )
        ]

        min_group_size = max(1, self.config.min_group_size)
        parsed_count = sum(1 for item in items if item.parsed_time is not None)
        if self.config.include_timestamps and parsed_count >= min_group_size:
            candidates.extend([
                _RenderCandidate(
                    text=_render_year_markdown(
                        items,
                        sort_by_time=self.config.sort_by_time,
                    ),
                    token_count=0,
                    structure_score=1,
                ),
                _RenderCandidate(
                    text=_render_month_markdown(
                        items,
                        sort_by_time=self.config.sort_by_time,
                    ),
                    token_count=0,
                    structure_score=2,
                ),
                _RenderCandidate(
                    text=_render_day_markdown(
                        items,
                        sort_by_time=self.config.sort_by_time,
                        min_group_size=min_group_size,
                    ),
                    token_count=0,
                    structure_score=3,
                ),
            ])

        scored_candidates = [
            _RenderCandidate(
                text=candidate.text,
                token_count=estimate_tokens(candidate.text),
                structure_score=candidate.structure_score,
            ) for candidate in candidates if candidate.text.strip()
        ]
        best = min(
            scored_candidates,
            key=lambda candidate:
            (candidate.token_count, -candidate.structure_score),
        )
        timepoints = {
            item.timestamp
            for item in items if item.timestamp.strip()
            and item.timestamp.strip().lower() != "unknown"
        }
        return PackedContext(
            text=best.text,
            token_count=best.token_count,
            block_count=len(items),
            timepoint_count=len(timepoints),
        )
