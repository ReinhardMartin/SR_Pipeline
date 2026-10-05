import re
from html.parser import HTMLParser

import pyromark
import tiktoken
from blingfire import text_to_sentences


def is_ref_section(name: str) -> bool:
    lower = name.lower()
    return "reference" in lower or "bibliography" in lower or "works cited" in lower


def _is_numeric_cell(text: str) -> bool:
    return len(re.sub(r"[^a-zA-Z]", "", text)) < 2


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.headers: list[str] = []
        self.rows: list[str] = []
        self._buf: list[str] = []
        self._row_cells: list[str] = []
        self._in_cell = False
        self._is_header = False

    def handle_starttag(self, tag, _):
        tag = tag.lower()
        if tag in ("th", "td"):
            self._in_cell = True
            self._is_header = tag == "th"
            self._buf = []
        elif tag == "tr":
            self._row_cells = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("th", "td"):
            text = "".join(self._buf).strip()
            if self._is_header:
                if text:
                    self.headers.append(text)
            else:
                self._row_cells.append(text)
            self._in_cell = False
        elif tag == "tr":
            text_cells = [c for c in self._row_cells if c and not _is_numeric_cell(c)]
            if text_cells:
                self.rows.append(" | ".join(text_cells))

    def handle_data(self, data):
        if self._in_cell:
            self._buf.append(data)


_CHUNK_LEVELS = {"sentence", "paragraph"}


class Chunker:
    def __init__(
        self,
        max_tokens: int,
        min_tokens: int,
        overlap_tokens: int,
        caption_max_tokens: int = 45,
        chunk_level: str = "paragraph",
        tokenizer: str = "cl100k_base",
    ):
        if chunk_level not in _CHUNK_LEVELS:
            raise ValueError(f"chunk_level must be one of {_CHUNK_LEVELS}, got {chunk_level!r}")
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.caption_max_tokens = caption_max_tokens
        self.chunk_level = chunk_level
        self._enc = tiktoken.get_encoding(tokenizer)

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _push_heading(self, stack: list, level: int, text: str, allow_same_level: bool = False) -> list:
        if allow_same_level:
            stack.append((level, text))
            return stack
        stack = [(l, t) for l, t in stack if l < level and l >= 2]
        stack.append((level, text))
        return stack

    def _section_path(self, stack: list) -> str:
        return " > ".join(t for _, t in stack) if stack else "Preamble"

    def _chunk(self, text: str, section: str, kind: str, **metadata) -> dict:
        chunk = {"section": section, "kind": kind, "text": text.strip(), "tokens": self._count(text)}
        chunk.update(metadata)
        return chunk

    def _flush(self, buf: list[str], section: str, kind: str) -> list[dict]:
        if is_ref_section(section):
            return []
        text = " ".join(buf).strip()
        if not text or self._count(text) < self.min_tokens:
            return []
        if kind != "paragraph" or self._count(text) <= self.max_tokens:
            return [self._chunk(text, section, kind)]
        return self._split_sentences(text, section)

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in text_to_sentences(text).splitlines() if s.strip()]

    def _split_sentences(self, text: str, section: str) -> list[dict]:
        sentences = self._sentences(text)
        if len(sentences) <= 1:
            return [self._chunk(text, section, "paragraph")]

        token_counts = {s: self._count(s) for s in sentences}

        chunks, window, window_tokens = [], [], 0
        for sent in sentences:
            sent_tokens = token_counts[sent]
            if window and window_tokens + sent_tokens > self.max_tokens:
                chunks.append(self._chunk(" ".join(window), section, "paragraph"))
                overlap, overlap_tokens = [], 0
                for s in reversed(window):
                    t = token_counts[s]
                    if overlap and overlap_tokens + t > self.overlap_tokens:
                        break
                    overlap = [s] + overlap
                    overlap_tokens += t
                    if overlap_tokens >= self.overlap_tokens:
                        break
                if overlap_tokens + sent_tokens > self.max_tokens:
                    overlap, overlap_tokens = [], 0
                window, window_tokens = overlap, overlap_tokens
            window.append(sent)
            window_tokens += sent_tokens

        if window:
            chunks.append(self._chunk(" ".join(window), section, "paragraph"))
        return chunks

    def _flush_sentences(self, buf: list[str], section: str, paragraph_id: str = "") -> list[dict]:
        if is_ref_section(section):
            return []
        text = " ".join(buf).strip()
        sentences = self._sentences(text)
        if not sentences:
            return []

        meta = {"paragraph_id": paragraph_id} if paragraph_id else {}
        chunks: list[dict] = []
        carry = ""
        for sent in sentences:
            combined = f"{carry} {sent}".strip() if carry else sent
            if self._count(combined) < self.min_tokens:
                carry = combined
                continue
            chunks.append(self._chunk(combined, section, "paragraph", **meta))
            carry = ""
        if carry:
            if chunks:
                chunks[-1] = self._chunk(chunks[-1]["text"] + " " + carry, section, "paragraph", **meta)
            else:
                chunks.append(self._chunk(carry, section, "paragraph", **meta))
        return chunks

    def _flush_pending(self, chunks: list[dict], section: str, pending_prefix: str, paragraph_index: int) -> int:
        if self.chunk_level == "sentence":
            chunks.extend(self._flush_sentences([pending_prefix], section, paragraph_id=f"p{paragraph_index}"))
            return paragraph_index + 1
        chunks.append(self._chunk(pending_prefix, section, "paragraph"))
        return paragraph_index

    def _extract_table_text(self, html: str) -> str:
        parser = _TableParser()
        parser.feed(html)
        parts: list[str] = []
        if parser.headers:
            parts.append(" | ".join(parser.headers))
        parts.extend(parser.rows)
        return "\n".join(parts)

    def create_chunks(self, document: str) -> list[dict]:
        chunks: list[dict] = []
        heading_stack: list[tuple[int, str]] = []

        heading_level = 0
        list_depth = 0
        in_html = False
        in_image = False

        image_url = ""
        image_alt_buf: list[str] = []
        buf: list[str] = []
        last_heading_chunk_count = 0
        pending_prefix = ""
        paragraph_index = 0

        for event in pyromark.events(document):
            match event:
                case {"Start": {"Heading": {"level": level}}}:
                    if pending_prefix:
                        section = self._section_path(heading_stack)
                        if not is_ref_section(section):
                            paragraph_index = self._flush_pending(chunks, section, pending_prefix, paragraph_index)
                        pending_prefix = ""
                    heading_level = int(level[1])
                    buf = []

                case {"End": {"Heading": _}}:
                    is_same_level_empty = (
                        bool(heading_stack)
                        and heading_stack[-1][0] == heading_level
                        and len(chunks) == last_heading_chunk_count
                    )
                    heading_stack = self._push_heading(heading_stack, heading_level, " ".join(buf), is_same_level_empty)
                    last_heading_chunk_count = len(chunks)
                    heading_level = 0
                    buf = []

                case {"Start": "HtmlBlock"}:
                    in_html = True
                    buf = []

                case {"End": "HtmlBlock"}:
                    raw = " ".join(buf).strip()
                    section = self._section_path(heading_stack)
                    if "<table" in raw.lower() and not is_ref_section(section):
                        index_text = self._extract_table_text(raw)
                        if pending_prefix:
                            index_text = f"{pending_prefix}\n{index_text}" if index_text else pending_prefix
                            pending_prefix = ""
                        if index_text:
                            chunks.append(self._chunk(index_text, section, "table", raw=raw))
                    else:
                        if pending_prefix:
                            paragraph_index = self._flush_pending(chunks, section, pending_prefix, paragraph_index)
                            pending_prefix = ""
                        chunks.extend(self._flush(buf, section, "html"))
                    in_html = False
                    buf = []

                case {"Start": {"CodeBlock": _}}:
                    buf = []

                case {"End": "CodeBlock"}:
                    chunks.extend(self._flush(buf, self._section_path(heading_stack), "code"))
                    buf = []

                case {"Start": {"List": _}}:
                    list_depth += 1

                case {"Start": "Item"}:
                    buf = []

                case {"End": "Item"}:
                    chunks.extend(self._flush(buf, self._section_path(heading_stack), "list_item"))
                    buf = []

                case {"End": {"List": _}}:
                    list_depth -= 1

                case {"Start": {"Image": {"dest_url": url}}}:
                    in_image = True
                    image_url = url
                    image_alt_buf = []

                case {"End": "Image"}:
                    in_image = False

                case {"Start": "Paragraph"} if list_depth == 0 and not in_html:
                    image_url = ""
                    buf = []

                case {"End": "Paragraph"} if list_depth == 0 and not in_html:
                    text = " ".join(buf).strip()
                    section = self._section_path(heading_stack)

                    if not is_ref_section(section):
                        if image_url:
                            alt = " ".join(image_alt_buf).strip()
                            index_text = " ".join(filter(None, [alt, text]))
                            if index_text:
                                chunks.append(self._chunk(index_text, section, "image", image_url=image_url))

                        elif text.startswith("$$") and text.endswith("$$"):
                            formula_text = text[2:-2].strip()
                            if pending_prefix:
                                formula_text = f"{pending_prefix}\n{formula_text}"
                                pending_prefix = ""
                            chunks.append(self._chunk(formula_text, section, "formula"))

                        else:
                            if text and self._count(text) < self.caption_max_tokens:
                                if pending_prefix:
                                    paragraph_index = self._flush_pending(chunks, section, pending_prefix, paragraph_index)
                                pending_prefix = text
                            else:
                                if pending_prefix:
                                    paragraph_index = self._flush_pending(chunks, section, pending_prefix, paragraph_index)
                                    pending_prefix = ""
                                if self.chunk_level == "sentence":
                                    chunks.extend(self._flush_sentences(buf, section, paragraph_id=f"p{paragraph_index}"))
                                    paragraph_index += 1
                                else:
                                    chunks.extend(self._flush(buf, section, "paragraph"))

                    buf = []

                case {"Text": text} | {"Html": text} | {"Code": text}:
                    if in_image:
                        image_alt_buf.append(text)
                    else:
                        buf.append(text)

                case "SoftBreak" | "HardBreak":
                    if not in_image:
                        buf.append(" ")

        if pending_prefix:
            section = self._section_path(heading_stack)
            if not is_ref_section(section):
                paragraph_index = self._flush_pending(chunks, section, pending_prefix, paragraph_index)

        for i, chunk in enumerate(chunks):
            chunk["id"] = f"chunk_{i}"

        return chunks
