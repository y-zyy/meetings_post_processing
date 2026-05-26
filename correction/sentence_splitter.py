"""
Sentence splitter for ASR output text.

ASR output typically has minimal formatting, so we split on:
  - Sentence-ending punctuation followed by whitespace
  - Newlines (paragraph breaks)

The splitter preserves trailing punctuation with each sentence so that
corrections within punctuation-adjacent tokens still work correctly.
"""

import re
from typing import List

# Matches one or more sentence-ending chars (including runs like "..."),
# followed by optional whitespace — or a bare newline sequence.
_SPLIT_RE = re.compile(
    r"(?<=[.!?。？！…])\s+|[\r\n]+"
)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentence-level chunks suitable for parallel processing.

    Empty or whitespace-only chunks are dropped.
    The original punctuation stays attached to the preceding sentence.

    >>> split_sentences("안녕하세요. 반갑습니다! 잘 부탁드립니다.")
    ['안녕하세요.', '반갑습니다!', '잘 부탁드립니다.']
    """
    chunks = _SPLIT_RE.split(text.strip())
    return [c.strip() for c in chunks if c.strip()]


def rejoin(original: str, sentences: List[str], corrected: List[str]) -> str:
    """
    Reassemble corrected sentences, preserving the original separators
    (spaces / newlines) between chunks.

    If the number of corrected sentences matches the split result, separators
    from the original text are re-inserted.  Otherwise falls back to a single
    space join.
    """
    if len(sentences) != len(corrected):
        return " ".join(corrected)

    # Find each sentence's position in the original to recover the separator.
    result_parts: List[str] = []
    search_from = 0
    for i, sentence in enumerate(sentences):
        pos = original.find(sentence, search_from)
        if pos == -1:
            # Fallback: can't locate sentence in original, use plain join.
            return " ".join(corrected)
        if i > 0:
            # The text between the previous sentence's end and this one's start
            # is the separator we want to keep.
            separator = original[search_from:pos]
            result_parts.append(separator if separator.strip() == "" else " ")
        result_parts.append(corrected[i])
        search_from = pos + len(sentence)

    return "".join(result_parts)
