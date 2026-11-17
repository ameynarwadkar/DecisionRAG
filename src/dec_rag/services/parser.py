from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import io
from pathlib import Path
import re

import pypdf

PART_SUFFIX_RE = re.compile(
    r"(?:\s*(?:#\d+|\(\s*part\s+\d+\s*\)|-\s*part\s+\d+|\bpart\s+\d+))\s*$",
    re.IGNORECASE,
)
MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.\)]?\s+[A-Z0-9].*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ParsedHeading:
    heading_index: int
    level: int
    title: str
    heading_path: str


@dataclass(frozen=True)
class ParsedPassage:
    passage_index: int
    heading_index: int | None
    heading_path: str
    content: str
    norm_hash: str
    token_estimate: int


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    first_sentence: str
    headings_summary: list[str]
    headings: list[ParsedHeading]
    passages: list[ParsedPassage]


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def compute_norm_hash(text: str) -> str:
    canonical = " ".join(text.lower().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def strip_part_suffix(title: str) -> str:
    cleaned = PART_SUFFIX_RE.sub("", title.strip()).strip()
    return cleaned or title.strip()


def humanize_stem(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return normalize_whitespace(stem) or "Untitled Document"


def extract_first_sentence(text: str, max_chars: int = 240) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    parts = SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)
    first = parts[0].strip()
    if len(first) > max_chars:
        return first[: max_chars - 3].rstrip() + "..."
    return first


def split_overlong_text(text: str, max_chars: int = 1200) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = SENTENCE_SPLIT_RE.split(cleaned)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            words = sent.split()
            w_buf = ""
            for w in words:
                candidate = f"{w_buf} {w}".strip() if w_buf else w
                if len(candidate) <= max_chars:
                    w_buf = candidate
                else:
                    if w_buf:
                        chunks.append(w_buf)
                    w_buf = w[:max_chars]
            if w_buf:
                chunks.append(w_buf)
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)
    return chunks


class _StructureBuilder:
    def __init__(self, max_passage_chars: int) -> None:
        self.max_passage_chars = max_passage_chars
        self.headings: list[ParsedHeading] = []
        self.passages: list[ParsedPassage] = []
        self._stack: dict[int, tuple[int, str]] = {}

    def push_heading(self, level: int, raw_title: str) -> None:
        title = normalize_whitespace(raw_title)
        if not title:
            return
        clamped_level = max(1, min(3, level))
        for deeper in [k for k in self._stack if k >= clamped_level]:
            del self._stack[deeper]

        idx = len(self.headings)
        self._stack[clamped_level] = (idx, title)
        ordered_titles = [self._stack[k][1] for k in sorted(self._stack)]
        heading_path = " > ".join(ordered_titles)
        self.headings.append(
            ParsedHeading(
                heading_index=idx,
                level=clamped_level,
                title=title,
                heading_path=heading_path,
            )
        )

    def current_heading_state(self) -> tuple[int | None, str]:
        if not self._stack:
            return None, ""
        deepest_level = max(self._stack)
        active_idx = self._stack[deepest_level][0]
        ordered_titles = [self._stack[k][1] for k in sorted(self._stack)]
        return active_idx, " > ".join(ordered_titles)

    def add_block(self, raw_text: str) -> None:
        heading_idx, heading_path = self.current_heading_state()
        for piece in split_overlong_text(raw_text, self.max_passage_chars):
            p_idx = len(self.passages)
            self.passages.append(
                ParsedPassage(
                    passage_index=p_idx,
                    heading_index=heading_idx,
                    heading_path=heading_path,
                    content=piece,
                    norm_hash=compute_norm_hash(piece),
                    token_estimate=estimate_tokens(piece),
                )
            )

    def finalize(self, doc_title: str) -> ParsedDocument:
        clean_title = normalize_whitespace(doc_title) or "Untitled Document"
        if not self.passages:
            raise ValueError("Document contains no extractable passage text.")
        first_sent = extract_first_sentence(self.passages[0].content)
        top_headings: list[str] = []
        min_level = min((h.level for h in self.headings), default=1)
        for h in self.headings:
            if h.level == min_level and h.title not in top_headings:
                top_headings.append(h.title)
        return ParsedDocument(
            title=clean_title,
            first_sentence=first_sent,
            headings_summary=top_headings,
            headings=self.headings,
            passages=self.passages,
        )


class _HTMLStructureParser(HTMLParser):
    IGNORED_TAGS = {"script", "style", "noscript"}

    def __init__(self, builder: _StructureBuilder) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = builder
        self.page_title: str = ""
        self.first_h1: str = ""
        self._ignored_depth = 0
        self._in_title = False
        self._title_buf: list[str] = []
        self._active_heading_level: int | None = None
        self._heading_buf: list[str] = []
        self._block_tag: str | None = None
        self._block_buf: list[str] = []
        self._in_tr = False
        self._tr_cells: list[str] = []
        self._in_cell = False
        self._cell_buf: list[str] = []
        self._tr_emitted_inner = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        t = tag.lower()
        if t in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth > 0:
            return
        if t == "title":
            self._in_title = True
            self._title_buf = []
        elif t in {"h1", "h2", "h3"}:
            self._active_heading_level = int(t[1])
            self._heading_buf = []
        elif t in {"p", "li"}:
            self._block_tag = t
            self._block_buf = []
        elif t == "tr":
            self._in_tr = True
            self._tr_cells = []
            self._tr_emitted_inner = False
        elif t in {"td", "th"} and self._in_tr:
            self._in_cell = True
            self._cell_buf = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth > 0:
            return
        if t == "title" and self._in_title:
            self._in_title = False
            text = normalize_whitespace("".join(self._title_buf))
            if text and not self.page_title:
                self.page_title = text
        elif t in {"h1", "h2", "h3"} and self._active_heading_level is not None:
            level = self._active_heading_level
            self._active_heading_level = None
            h_text = normalize_whitespace("".join(self._heading_buf))
            if h_text:
                if level == 1 and not self.first_h1:
                    self.first_h1 = h_text
                self.builder.push_heading(level, h_text)
        elif t in {"p", "li"} and self._block_tag == t:
            self._block_tag = None
            b_text = normalize_whitespace("".join(self._block_buf))
            if b_text:
                self.builder.add_block(b_text)
                if self._in_tr:
                    self._tr_emitted_inner = True
        elif t in {"td", "th"} and self._in_cell:
            self._in_cell = False
            c_text = normalize_whitespace("".join(self._cell_buf))
            if c_text:
                self._tr_cells.append(c_text)
        elif t == "tr" and self._in_tr:
            self._in_tr = False
            if self._tr_cells and not self._tr_emitted_inner:
                self.builder.add_block(" | ".join(self._tr_cells))
            self._tr_cells = []

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if self._in_title:
            self._title_buf.append(data)
        if self._active_heading_level is not None:
            self._heading_buf.append(data)
        if self._block_tag is not None:
            self._block_buf.append(data)
        elif self._in_cell:
            self._cell_buf.append(data)


def _parse_html(text: str, fallback_title: str, max_passage_chars: int) -> ParsedDocument:
    builder = _StructureBuilder(max_passage_chars=max_passage_chars)
    parser = _HTMLStructureParser(builder)
    parser.feed(text)
    parser.close()
    title = parser.page_title or parser.first_h1 or fallback_title
    return builder.finalize(title)


def _parse_markdown(text: str, fallback_title: str, max_passage_chars: int) -> ParsedDocument:
    lines = text.splitlines()
    h1_matches = [
        MD_HEADING_RE.match(line.strip())
        for line in lines
        if line.strip().startswith("# ")
    ]
    single_top_h1 = len([m for m in h1_matches if m is not None]) == 1

    builder = _StructureBuilder(max_passage_chars=max_passage_chars)
    doc_title: str | None = None
    para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal para_lines
        if para_lines:
            joined = " ".join(s.strip() for s in para_lines if s.strip())
            para_lines = []
            if joined:
                builder.add_block(joined)

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            flush_para()
            continue

        m = MD_HEADING_RE.match(stripped)
        if m:
            flush_para()
            hashes, h_title = m.group(1), m.group(2).strip()
            raw_level = len(hashes)
            if raw_level == 1 and doc_title is None:
                doc_title = h_title
                if single_top_h1:
                    continue
            adj_level = max(1, raw_level - 1) if single_top_h1 else raw_level
            builder.push_heading(adj_level, h_title)
            continue

        para_lines.append(stripped)

    flush_para()
    return builder.finalize(doc_title or fallback_title)


def _detect_layout_heading(line: str) -> tuple[bool, int, str]:
    s = line.strip()
    if len(s) < 2 or len(s) > 80:
        return False, 1, ""
    if not any(ch.isalpha() for ch in s):
        return False, 1, ""
    if s[-1] in ".?!,;:":
        return False, 1, ""

    num_match = NUMBERED_HEADING_RE.match(s)
    if num_match:
        dots = num_match.group(1).count(".")
        return True, min(3, dots + 1), s

    alpha_chars = [ch for ch in s if ch.isalpha()]
    if alpha_chars and all(ch.isupper() for ch in alpha_chars):
        return True, 1, s

    if s[0].isupper() and len(s.split()) <= 10:
        return True, 1, s

    return False, 1, ""


def _parse_plain_text(
    text: str,
    fallback_title: str,
    max_passage_chars: int,
    *,
    is_pdf: bool = False,
) -> ParsedDocument:
    builder = _StructureBuilder(max_passage_chars=max_passage_chars)
    blocks = re.split(r"\n\s*\n", text.strip())
    doc_title: str | None = None

    for idx, block in enumerate(blocks):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        if len(lines) == 1:
            is_h, level, h_title = _detect_layout_heading(lines[0])
            if is_h:
                if idx == 0 and doc_title is None and len(blocks) > 1:
                    doc_title = h_title
                else:
                    builder.push_heading(level, h_title)
                continue
            builder.add_block(lines[0])
            continue

        # Multi-line block: handle numbered/all-caps or PDF single-newline section headers
        pos = 0
        while pos < len(lines):
            ln = lines[pos]
            is_h, level, h_title = _detect_layout_heading(ln)
            is_explicit_header = bool(NUMBERED_HEADING_RE.match(ln)) or (
                any(c.isalpha() for c in ln) and all(c.isupper() for c in ln if c.isalpha())
            )
            pdf_header = (
                is_pdf
                and is_h
                and pos + 1 < len(lines)
                and lines[pos + 1][0].isupper()
                and lines[pos + 1][-1] in ".?!"
            )
            if is_h and (is_explicit_header or pdf_header) and pos + 1 < len(lines):
                builder.push_heading(level, h_title)
                pos += 1
                continue
            break

        if pos < len(lines):
            builder.add_block(" ".join(lines[pos:]))

    return builder.finalize(doc_title or fallback_title)


def _extract_pdf_text(raw_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages_text.append(extracted.strip())
    return "\n\n".join(pages_text)


def parse_document(
    raw_bytes: bytes,
    filename: str,
    content_type: str | None = None,
    *,
    max_passage_chars: int = 1200,
) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    ctype = (content_type or "").lower()
    fallback_title = humanize_stem(filename)

    if ext == ".pdf" or "application/pdf" in ctype:
        pdf_text = _extract_pdf_text(raw_bytes)
        return _parse_plain_text(pdf_text, fallback_title, max_passage_chars, is_pdf=True)

    decoded = raw_bytes.decode("utf-8", errors="replace")

    if ext in {".html", ".htm"} or "text/html" in ctype:
        return _parse_html(decoded, fallback_title, max_passage_chars)

    if ext in {".md", ".markdown"} or "markdown" in ctype:
        return _parse_markdown(decoded, fallback_title, max_passage_chars)

    return _parse_plain_text(decoded, fallback_title, max_passage_chars, is_pdf=False)