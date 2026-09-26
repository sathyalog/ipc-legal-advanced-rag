"""Structure-aware parser for the India Code IPC PDF.

Why not a generic text splitter: in this PDF each section is a bold heading
(`302. Punishment for murder.—Whoever ...`), amendment markers are 7pt
superscripts (`9[4. Extension ...`), and every page ends with 9pt legislative
footnotes (`1. Subs. by Act 4 of 1898 ...`) that *look* like section headings.
A character splitter mixes all of that together. Here we use layout signals
(font size, bold, footnote separator line) to cut the document exactly at
section boundaries, and use the "Arrangement of Sections" (the TOC) as a
checklist so ingestion can prove it found every section.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

SUPERSCRIPT_MAX_SIZE = 7.5  # amendment/footnote markers are 6-7pt
PAGE_FOOTER_Y = 770  # page numbers sit at y≈782
SECTION_ID = r"\d{1,3}[A-Z]{0,3}"

_toc_entry = re.compile(rf"^({SECTION_ID})\.\s*(.*)$")
_heading = re.compile(rf"^[\[\s*]*({SECTION_ID})\.?\s+(\S.*)$")
_chapter = re.compile(r"^\[?\s*CHAPTER\s+([IVXL]+\s*[A-Z]?)\s*$")
_repealed = re.compile(r"\b(Repealed|Omitted|Rep\.)", re.I)
_ws = re.compile(r"\s+")


@dataclass
class TocEntry:
    section_id: str
    title: str
    chapter_no: str
    chapter_title: str
    repealed: bool


@dataclass
class Section:
    section_id: str
    title: str
    chapter_no: str
    chapter_title: str
    page_start: int
    page_end: int
    text: str
    repealed: bool = False
    has_illustrations: bool = False
    has_explanation: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseReport:
    toc_count: int
    parsed_count: int
    missing: list[str] = field(default_factory=list)  # in TOC, not found in body (non-repealed)
    missing_repealed: list[str] = field(default_factory=list)  # in TOC, repealed, no body text

    @property
    def ok(self) -> bool:
        return not self.missing


def _clean(text: str) -> str:
    return _ws.sub(" ", text).strip()


def _join_spans(spans: list[dict]) -> str:
    """Join spans, inserting a space where the PDF leaves a visual gap.

    Small-caps headings are rendered as separate spans ("O" + "F OFFENCES")
    with no space character between words, so plain concatenation yields
    "OF OFFENCESAFFECTINGTHE HUMAN BODY".
    """
    out = ""
    prev_x1 = None
    for s in sorted(spans, key=lambda s: s["bbox"][0]):
        t = s["text"]
        if prev_x1 is not None and s["bbox"][0] - prev_x1 > 1.5 and out and not out.endswith(" ") and not t.startswith(" "):
            out += " "
        out += t
        prev_x1 = s["bbox"][2]
    return out


def _page_lines(page: pymupdf.Page) -> list[dict]:
    raw = [ln for b in page.get_text("dict")["blocks"] for ln in b.get("lines", [])]
    raw.sort(key=lambda ln: (round(ln["bbox"][1]), ln["bbox"][0]))
    return raw


def section_sort_key(section_id: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-Z]*)", section_id)
    return (int(m.group(1)), m.group(2)) if m else (10**6, section_id)


def _split_glued(token: str, vocab: set[str]) -> list[str] | None:
    """Split 'OFFENCESAFFECTINGTHE' -> ['OFFENCES', 'AFFECTING', 'THE'] using a vocabulary."""
    if token in vocab:
        return [token]
    for i in range(len(token) - 1, 1, -1):
        head = token[:i]
        if head in vocab and (rest := _split_glued(token[i:], vocab)):
            return [head, *rest]
    return None


def _normalise_chapter_titles(titles: list[str]) -> dict[str, str]:
    """Repair words the PDF renders without spaces, using words seen elsewhere as vocabulary."""
    vocab = {w for t in titles for w in re.findall(r"[A-Z]{2,}", t)}
    vocab |= {"OF", "TO", "BY", "OR", "THE", "AND", "EVIDENCE", "DECENCY"}
    # drop vocabulary entries that are themselves glued ("OFFENCESRELATING")
    for w in sorted(vocab, key=len):
        if len(w) > 4 and len(_split_glued(w, vocab - {w}) or []) > 1:
            vocab.discard(w)
    out = {}
    for t in titles:
        clean = re.sub(r"[\[\]*]", " ", t)
        words = []
        for w in clean.split():
            core = w.rstrip(",")
            parts = _split_glued(core, vocab) if len(core) > 4 else None
            words += (parts or [core]) + ([","] if w.endswith(",") else [])
        out[t] = _clean(" ".join(words).replace(" ,", ","))
    return out


# ---------------------------------------------------------------- TOC
def find_body_start(doc: pymupdf.Document) -> int:
    """Index of the first body page (the one with 'ACT NO. 45 OF 1860')."""
    for i, page in enumerate(doc):
        if "OF 1860" in page.get_text().upper().replace("\n", " ") and i > 0:
            return i
    raise ValueError("Could not locate start of the Act body")


def parse_toc(doc: pymupdf.Document, body_start: int) -> list[TocEntry]:
    entries: list[TocEntry] = []
    chapter_no, chapter_title = "", ""
    pending_chapter = False
    lines = []
    for pno in range(body_start):
        rows: dict[int, list[dict]] = {}
        for ln in _page_lines(doc[pno]):
            rows.setdefault(round(ln["bbox"][1]), []).extend(ln["spans"])
        lines += [_join_spans(sp).strip() for _, sp in sorted(rows.items())]

    for ln in lines:
        if not ln or ln in {"SECTIONS", "PREAMBLE"} or set(ln) <= set("_") or ln.isdigit():
            continue
        if m := _chapter.match(ln):
            chapter_no, chapter_title, pending_chapter = _clean(m.group(1)), "", True
            continue
        if pending_chapter and ln.upper() == ln and not _toc_entry.match(ln):
            chapter_title = _clean(f"{chapter_title} {ln}")
            continue
        pending_chapter = False
        if m := _toc_entry.match(ln):
            title = m.group(2)
            entries.append(TocEntry(m.group(1), title, chapter_no, chapter_title, False))
        elif entries:  # continuation of previous title
            entries[-1].title += " " + ln

    for e in entries:
        e.title = _clean(e.title)
        e.repealed = bool(re.fullmatch(r"\[?(Repealed|Omitted)\.?\]?\.?", e.title, re.I))
    return entries


# ---------------------------------------------------------------- body
@dataclass
class _Line:
    page: int
    text: str
    bold_start: bool


def _footnote_separator_y(page: pymupdf.Page, lines: list[dict]) -> float:
    """y of the short rule above the footnotes (drawn line or a run of spaces)."""
    ys = [
        d["rect"].y0
        for d in page.get_drawings()
        if d["rect"].height < 3 and 80 < d["rect"].width < 250 and d["rect"].x0 < 100
    ]
    for ln in lines:
        txt = "".join(s["text"] for s in ln["spans"])
        if len(txt) >= 20 and not txt.strip():
            ys.append(ln["bbox"][1])
    return min(ys) if ys else float("inf")


def _body_lines(doc: pymupdf.Document, body_start: int) -> list[_Line]:
    out: list[_Line] = []
    for pno in range(body_start, len(doc)):
        page = doc[pno]
        raw = _page_lines(page)
        sep_y = _footnote_separator_y(page, raw)
        # merge spans that share a baseline (PyMuPDF sometimes splits one visual line)
        merged: dict[int, list[dict]] = {}
        for ln in raw:
            y = ln["bbox"][1]
            if y >= sep_y - 1 or y > PAGE_FOOTER_Y:
                continue
            spans = [s for s in ln["spans"] if s["size"] > SUPERSCRIPT_MAX_SIZE]
            if spans:
                merged.setdefault(round(y), []).extend(spans)
        for _, spans in sorted(merged.items()):
            text = _join_spans(spans)
            if not text.strip() or re.fullmatch(r"[\s*]+", text):
                continue
            first = next((s for s in spans if s["text"].strip(" [*")), spans[0])
            out.append(_Line(pno + 1, text.rstrip(), "Bold" in first["font"]))
    return out


def parse_sections(pdf_path: Path | str) -> tuple[list[Section], list[TocEntry], ParseReport]:
    doc = pymupdf.open(pdf_path)
    body_start = find_body_start(doc)
    toc = parse_toc(doc, body_start)
    toc_index = {e.section_id: i for i, e in enumerate(toc)}
    lines = _body_lines(doc, body_start)

    sections: list[Section] = []
    chapter_no, chapter_title = "", ""
    last_idx = -1
    i = 0
    buf: list[str] = []

    def flush(end_page: int) -> None:
        if sections and buf:
            sections[-1].text = _clean(" ".join(buf))
            sections[-1].page_end = end_page

    while i < len(lines):
        ln = lines[i]
        text = ln.text.strip()

        if m := _chapter.match(text):  # chapter heading + upper-case title lines
            chapter_no, title_parts = _clean(m.group(1)), []
            j = i + 1
            while j < len(lines) and lines[j].text.strip().upper() == lines[j].text.strip() and not _heading.match(lines[j].text.strip()):
                title_parts.append(lines[j].text.strip())
                j += 1
            chapter_title = _clean(" ".join(title_parts))
            i = j
            continue

        m = _heading.match(text)
        # A heading must (a) be a TOC section id, (b) come after the previous one
        # (monotonic order rules out cross-references like "see section 34"), and
        # (c) be bold -- or, if the PDF forgot the bold (s.77), be the very next
        # TOC entry and contain the heading dash.
        is_heading = False
        if m and m.group(1) in toc_index and toc_index[m.group(1)] > last_idx:
            is_next = toc_index[m.group(1)] == last_idx + 1
            is_heading = ln.bold_start or (is_next and "—" in text)
        if is_heading:
            flush(lines[i - 1].page if i else ln.page)
            buf = [text.lstrip("[* ")]
            sid = m.group(1)
            last_idx = toc_index[sid]
            sections.append(Section(sid, "", chapter_no or toc[last_idx].chapter_no,
                                    chapter_title or toc[last_idx].chapter_title, ln.page, ln.page, ""))
        elif sections:
            buf.append(text)
        i += 1
    flush(lines[-1].page)

    for s in sections:
        body = s.text
        head = body.split("—", 1)[0]
        head = re.sub(rf"^{re.escape(s.section_id)}\.?\s*", "", head).strip()
        s.title = head[:200].rstrip(".") if len(head) < 250 else toc[toc_index[s.section_id]].title
        s.repealed = bool(_repealed.search(body[:160])) and len(body) < 300
        s.has_illustrations = "Illustration" in body
        s.has_explanation = "Explanation" in body

    fixed = _normalise_chapter_titles([s.chapter_title for s in sections] + [e.chapter_title for e in toc])
    for s in sections:
        s.chapter_title = fixed[s.chapter_title]

    found = {s.section_id for s in sections}
    report = ParseReport(toc_count=len(toc), parsed_count=len(sections))
    for e in toc:
        if e.section_id not in found:
            (report.missing_repealed if e.repealed else report.missing).append(e.section_id)
    return sections, toc, report
