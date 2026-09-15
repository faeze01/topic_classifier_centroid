"""Build embeddable text from post title and body."""

from __future__ import annotations

from src.preprocess import clean_text

MAX_EMBED_CHARS = 6000  # keeps text well under bge-m3's context window (some bodies run 100k+ chars)


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def build_embed_text(title: object, body: object) -> str:
    """Build the text to embed from cleaned title and cleaned body.

    Truncated to MAX_EMBED_CHARS: a handful of posts have bodies over 100k
    characters, which exceeds bge-m3's context window and errors out.
    """
    title_clean = clean_text(_cell_text(title))
    body_clean = clean_text(_cell_text(body))
    text = f"{title_clean} {body_clean}".strip()
    return text[:MAX_EMBED_CHARS]
