import re
import unicodedata

# Fenced code blocks are left as they are: in code, * ` _ and \ are real syntax
FENCED_CODE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.DOTALL | re.MULTILINE)

ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
TYPOGRAPHIC = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
})
ESCAPED_CHAR = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~|<>$])")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
# (?=\S) / (?<=\S) keep "* list item" and "2 * 3" from being read as emphasis
BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
STAR_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
UNDERSCORE_ITALIC = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
STRIKETHROUGH = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
REPEATED_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def _normalize_prose(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also turns non-breaking spaces into spaces
    text = ZERO_WIDTH.sub("", text)
    text = text.translate(TYPOGRAPHIC)
    text = INLINE_CODE.sub(r"\1", text)
    text = BOLD.sub(r"\2", text)
    text = STAR_ITALIC.sub(r"\1", text)
    text = UNDERSCORE_ITALIC.sub(r"\1", text)
    text = STRIKETHROUGH.sub(r"\1", text)
    # Unescape last, so an escaped \* is never read as an emphasis marker above
    text = ESCAPED_CHAR.sub(r"\1", text)
    return REPEATED_SPACES.sub(" ", text)


def normalize_text(text: str) -> str:
    """
    Strip inline Markdown formatting so stored chunks read as plain text:
    `code` and **bold**/*italic*/~~strike~~ markers, backslash escapes,
    curly quotes and dashes, zero-width and non-breaking spaces.
    Headings, list markers and fenced code blocks are kept.
    """
    parts = []
    last = 0
    for block in FENCED_CODE.finditer(text):
        parts.append(_normalize_prose(text[last:block.start()]))
        parts.append(block.group())
        last = block.end()
    parts.append(_normalize_prose(text[last:]))
    return EXTRA_BLANK_LINES.sub("\n\n", "".join(parts)).strip()
