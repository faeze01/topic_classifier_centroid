"""Build embeddable text from post title and body."""

from __future__ import annotations

import pandas as pd

from src.preprocess import clean_text

MAX_EMBED_CHARS = 6000  # keeps text well under bge-m3's context window (some bodies run 100k+ chars)


def build_embed_text(title: object, body: object) -> str:
    """Build the text to embed from cleaned title and cleaned body.

    Truncated to MAX_EMBED_CHARS: a handful of posts have bodies over 100k
    characters, which exceeds bge-m3's context window and errors out.
    """
    title_clean = clean_text(str(title)) if pd.notna(title) else ""
    body_clean = clean_text(str(body)) if pd.notna(body) else ""
    text = f"{title_clean} {body_clean}".strip()
    return text[:MAX_EMBED_CHARS]
