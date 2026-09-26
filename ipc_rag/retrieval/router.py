"""Deterministic extraction of IPC section references from a user query.

Identifiers ("302", "498A", "120-B") are exactly what dense embeddings are bad
at, and exactly what a regex is perfect at. If the query names a section we
fetch it by ID - that path cannot miss. Everything else goes to hybrid search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "section 302", "sec. 302", "s. 302", "u/s 302", "ss. 299 and 300", "IPC 420", "§ 34"
_TRIGGER = r"(?:sections?|secs?\.?|ss?\.|u/s\.?|§+|ipc|i\.p\.c\.?)"
_ID = r"\d{1,3}(?:\s?-?\s?[A-Za-z]{1,2}\b)?"
_LIST = rf"{_ID}(?:\s*(?:,|/|&|and|or|to|vs\.?|versus)\s*{_ID})*"

_after_trigger = re.compile(rf"\b{_TRIGGER}\s*({_LIST})", re.I)
_before_ipc = re.compile(rf"\b({_LIST})\s*(?:of\s+(?:the\s+)?)?(?:ipc|i\.p\.c|indian penal code)\b", re.I)
_bare_id = re.compile(rf"(?<![\w.])({_ID})(?![\w.])", re.I)
_one_id = re.compile(_ID, re.I)
# "between 299 and 300", "302 vs 304" - two+ ids joined, NOT followed by a unit ("between 2 and 5 years")
_UNITS = r"(?:years?|months?|days?|hours?|rupees|rs|persons?|people|times|members?|or\s+more)"
_joined = re.compile(rf"(?<![\w.])({_ID}(?:\s*(?:,|&|and|vs\.?|versus)\s*{_ID})+)(?![\w.])(?!\s*{_UNITS}\b)", re.I)
_FILLER = {"what", "whats", "is", "explain", "define", "tell", "me", "about", "the", "say", "says",
           "does", "do", "describe", "show", "meaning", "of", "please", "details", "text"}


def _normalise(raw: str) -> str:
    return re.sub(r"[\s-]", "", raw).upper()


@dataclass
class SectionRefs:
    ids: list[str] = field(default_factory=list)  # valid sections, in order of mention
    unknown: list[str] = field(default_factory=list)  # e.g. "376DB" - not in this edition of the Code


def extract_section_ids(query: str, known_ids: set[str]) -> list[str]:
    return parse_section_refs(query, known_ids).ids


def parse_section_refs(query: str, known_ids: set[str]) -> SectionRefs:
    """Find section references in `query`, validated against `known_ids`.

    Validation against the real ID list is what keeps "10 years" or "2 persons"
    from being treated as sections.
    """
    spans: list[tuple[int, str]] = []
    for pattern in (_after_trigger, _before_ipc, _joined):
        for m in pattern.finditer(query):
            for idm in _one_id.finditer(m.group(1)):
                spans.append((m.start(1) + idm.start(), idm.group()))

    if not spans:
        # Short queries that are basically just an ID: "302", "what is 498A?", "explain 120B"
        rest = [w for w in re.findall(r"[\w-]+", query.lower()) if w not in _FILLER]
        if len(rest) == 1 and _bare_id.fullmatch(rest[0]):
            spans = [(0, rest[0])]

    out = SectionRefs()
    for _, raw in sorted(spans):
        sid = _normalise(raw)
        if sid not in known_ids:
            if re.fullmatch(r"\d+[A-Za-z]+", raw.strip()):
                out.unknown.append(sid)  # letters glued to the number: a real (but absent) id
                continue
            sid = re.match(r"\d+", sid).group()  # "302 A person" -> the "A" is not part of the id
        if sid in known_ids and sid not in out.ids:
            out.ids.append(sid)
        elif sid not in known_ids and sid not in out.unknown:
            out.unknown.append(sid)
    return out
