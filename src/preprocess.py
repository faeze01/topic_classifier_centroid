"""Text cleaning utilities for Persian/Arabic news and social content."""

from __future__ import annotations

import re
import unicodedata

# Persian ZWNJ — must be preserved through cleaning.
ZWNJ = "\u200c"

# Arabic → Persian character map (and Tatweel removal).
_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "\u064a": "\u06cc",  # ي → ی
        "\u0643": "\u06a9",  # ك → ک
        "\u0629": "\u0647",  # ة → ه
        "\u0624": "\u0648",  # ؤ → و
        "\u0626": "\u06cc",  # ئ → ی
        "\u0640": None,  # ـ Tatweel → remove
        "\u0623": "\u0627",  # أ → ا
        "\u0625": "\u0627",  # إ → ا
        "\u0649": "\u06cc",  # ى → ی
    }
)

# Bidirectional / format controls to strip (Cf), excluding ZWNJ.
_INVISIBLE_MARKS = re.compile(
    "["
    "\u200b"  # ZERO WIDTH SPACE
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LRM
    "\u200f"  # RLM
    "\u202a-\u202e"  # LRE, RLE, PDF, LRO, RLO
    "\u2060-\u2064"  # word joiner, invisible operators
    "\u2066-\u2069"  # LRI, RLI, FSI, PDI
    "\ufeff"  # BOM / ZWNBSP
    "\u00ad"  # soft hyphen
    "]"
)

_FENCED_CODE = re.compile(r"```[\w]*\n?.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_HTML_TAG = re.compile(r"</?[^>]+>")
_WHITESPACE = re.compile(r"[\s\u00a0]+")


def _strip_markdown(text: str) -> str:
    # Drop fenced code blocks entirely (markers + body).
    text = _FENCED_CODE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _HEADER.sub("", text)
    text = _BOLD_ITALIC.sub(r"\2", text)
    return text


def clean_text(text: str) -> str:
    """Clean and normalize text for embedding / classification.

    Steps (in order): NFKC → Arabic→Persian → strip bidi marks (keep ZWNJ)
    → strip Markdown → strip HTML → collapse whitespace.
    Does not stem, lemmatize, or remove stopwords/punctuation/emojis.
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Arabic → Persian (+ remove Tatweel)
    text = text.translate(_ARABIC_TO_PERSIAN)

    # 3. Remove invisible direction/control marks; keep ZWNJ
    text = _INVISIBLE_MARKS.sub("", text)

    # 4. Strip Markdown syntax
    text = _strip_markdown(text)

    # 5. Remove raw HTML tags
    text = _HTML_TAG.sub("", text)

    # 6. Collapse whitespace (spaces, tabs, newlines) to a single space
    text = _WHITESPACE.sub(" ", text).strip()

    # 7. Punctuation, numbers, emojis, stopwords, and ZWNJ are preserved as-is.
    return text



