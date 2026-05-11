from __future__ import annotations

import datetime as dt
import re


def parse_timestamp(value: str | None) -> dt.datetime | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "unknown":
        return None

    normalized = re.sub(
        r"\b(am|pm)\b",
        lambda item: item.group(1).upper(),
        raw,
        flags=re.IGNORECASE,
    )
    iso_value = normalized.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(iso_value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    formats = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return dt.datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def timestamp_lte(value: str | None, cutoff: str | None) -> bool:
    if not cutoff:
        return True
    parsed_value = parse_timestamp(value)
    parsed_cutoff = parse_timestamp(cutoff)
    if parsed_value is None or parsed_cutoff is None:
        return False
    return parsed_value <= parsed_cutoff


def min_timestamp(values: tuple[str | None, ...]) -> str | None:
    return _edge_timestamp(values, reverse=False)


def max_timestamp(values: tuple[str | None, ...]) -> str | None:
    return _edge_timestamp(values, reverse=True)


def _edge_timestamp(values: tuple[str | None, ...], *,
                    reverse: bool) -> str | None:
    parsed: list[tuple[dt.datetime, str]] = []
    fallback: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        parsed_value = parse_timestamp(text)
        if parsed_value is None:
            fallback.append(text)
        else:
            parsed.append((parsed_value, text))

    if parsed:
        return sorted(parsed, key=lambda item: item[0], reverse=reverse)[0][1]
    if fallback:
        return fallback[-1] if reverse else fallback[0]
    return None
