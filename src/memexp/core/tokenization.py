from __future__ import annotations

from functools import lru_cache

DEFAULT_TOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def tokenizer(encoding_name: str = DEFAULT_TOKEN_ENCODING):
    try:
        import tiktoken
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Token counting requires the tiktoken package.") from exc

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # pragma: no cover - cache/network guard
        raise RuntimeError(
            f"Unable to load tiktoken encoding '{encoding_name}'. "
            "Install/cache the encoding before running token-budgeted experiments."
        ) from exc


def count_tokens(text: str,
                 *,
                 encoding_name: str = DEFAULT_TOKEN_ENCODING) -> int:
    return len(tokenizer(encoding_name).encode(text or ""))
