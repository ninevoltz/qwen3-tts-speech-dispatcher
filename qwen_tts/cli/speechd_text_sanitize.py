# coding=utf-8
from __future__ import annotations

import re

# Drop all C0 controls and DEL for Speech Dispatcher text transport.
_INVALID_SPEECHD_TEXT_RE = re.compile(r"[\x00-\x1f\x7f]")

# Normalize common Unicode punctuation that often degrades to '?' in
# non-UTF-8 Speech Dispatcher environments.
_PUNCT_TRANSLATIONS = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
    }
)


def sanitize_speechd_text(raw: str) -> str:
    text = (raw or "").translate(_PUNCT_TRANSLATIONS)
    # Normalize line breaks to spaces so words do not get merged.
    text = text.replace("\r", " ").replace("\n", " ")
    text = _INVALID_SPEECHD_TEXT_RE.sub("", text)
    # Remove common Markdown formatting artifacts that are read aloud poorly.
    text = re.sub(r"(^|\s)#{1,6}\s+", r"\1", text)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\s*-{3,}\s*", " ", text)
    # If Unicode punctuation was downgraded upstream, '?' often appears inside
    # words/phrases where a dash should be.
    text = re.sub(r"\?(?=\w)", "-", text)
    text = re.sub(r"\?(?=\s+\w)", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
