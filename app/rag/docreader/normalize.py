"""Text normalization shared by the text parser and loader."""

from __future__ import annotations

import re
import unicodedata


_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n)")
_EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize newlines and harmless Unicode noise without changing prose."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = _ZERO_WIDTH.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _EXCESSIVE_BLANK_LINES.sub("\n\n", text)
    return text.strip("\n")


def iter_nonempty_lines(text: str) -> list[tuple[int, str]]:
    """Return one-based line numbers and trimmed line content."""
    return [
        (line_number, line.strip())
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
