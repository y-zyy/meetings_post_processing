"""
Term dictionary using Aho-Corasick automaton for O(n) multi-pattern search.

The automaton is built once and is read-only thereafter, making it safe
to share across threads and async tasks without locking.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ahocorasick


class TermDictionary:
    """
    Stores wrong→correct term mappings and applies them via Aho-Corasick.

    Build once at startup, then share the instance across all requests.
    After build() the internal automaton is immutable and thread-safe.

    Overlap resolution: greedy left-to-right with longest-match preference.
    Example: if "머신 러닝" and "머신" are both patterns, "머신 러닝" wins
    when they start at the same position.
    """

    def __init__(self) -> None:
        self._terms: Dict[str, str] = {}
        self._automaton: Optional[ahocorasick.Automaton] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def add(self, wrong: str, correct: str) -> None:
        """Add a single correction pair. Invalidates the automaton."""
        if not wrong:
            raise ValueError("wrong term must not be empty")
        self._terms[wrong] = correct
        self._automaton = None

    def load_json(self, path: str) -> None:
        """
        Load from a JSON file. Two supported formats:

        Dict:  {"wrong_term": "correct_term", ...}
        List:  [{"wrong": "wrong_term", "correct": "correct_term"}, ...]
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for wrong, correct in data.items():
                self._terms[wrong] = correct
        elif isinstance(data, list):
            for item in data:
                self._terms[item["wrong"]] = item["correct"]
        else:
            raise ValueError(f"Unsupported JSON format in {path}")

        self._automaton = None

    def load_csv(self, path: str, wrong_col: str = "wrong", correct_col: str = "correct") -> None:
        """Load from a CSV file with header row."""
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self._terms[row[wrong_col]] = row[correct_col]
        self._automaton = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Compile the Aho-Corasick automaton from loaded terms.

        Must be called before correct(). Safe to call multiple times;
        rebuilds the automaton from the current term set.
        """
        automaton = ahocorasick.Automaton()
        for wrong, correct in self._terms.items():
            # Store (pattern_length, replacement) so we can recover start index.
            automaton.add_word(wrong, (len(wrong), correct))
        automaton.make_automaton()
        self._automaton = automaton

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct(self, text: str) -> str:
        """
        Return text with all matching terms replaced.

        Automatically calls build() on first use if needed.
        This method is thread-safe once the automaton is built.
        """
        if not text:
            return text
        if self._automaton is None:
            self.build()

        matches = self._collect_matches(text)
        if not matches:
            return text
        return self._apply_matches(text, matches)

    def _collect_matches(self, text: str) -> List[Tuple[int, int, str]]:
        """Return list of (start, end, replacement) for all pattern occurrences."""
        matches: List[Tuple[int, int, str]] = []
        for end_idx, (length, replacement) in self._automaton.iter(text):
            start_idx = end_idx - length + 1
            matches.append((start_idx, end_idx + 1, replacement))
        return matches

    @staticmethod
    def _apply_matches(text: str, matches: List[Tuple[int, int, str]]) -> str:
        """
        Resolve overlapping matches (greedy left-to-right, longest wins)
        and reconstruct the corrected string.
        """
        # Sort by start position; on ties, prefer the longer match.
        matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))

        selected: List[Tuple[int, int, str]] = []
        last_end = 0
        for start, end, replacement in matches:
            if start >= last_end:
                selected.append((start, end, replacement))
                last_end = end

        parts: List[str] = []
        pos = 0
        for start, end, replacement in selected:
            parts.append(text[pos:start])
            parts.append(replacement)
            pos = end
        parts.append(text[pos:])
        return "".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, wrong: str) -> bool:
        return wrong in self._terms

    @property
    def is_built(self) -> bool:
        return self._automaton is not None
