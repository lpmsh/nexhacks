import re
from typing import List, Optional, Sequence

from .types import Span


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
CODE_PATTERN = re.compile(r"```|\b(class|def|function|public|private|console\.log|import\s+\w+)\b", re.IGNORECASE)
ROLE_PATTERN = re.compile(r"^(user|assistant|system)\s*:", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if stripped.endswith(":"):
        return True
    words = stripped.split()
    if len(words) <= 8 and stripped.isupper():
        return True
    return False


def is_code_block(text: str) -> bool:
    return bool(CODE_PATTERN.search(text))


def has_role_marker(text: str) -> bool:
    return bool(ROLE_PATTERN.match(text.strip()))


def sentence_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Split on sentence enders while keeping structure simple.
    parts = re.split(r"(?<=[\.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        sentences = [text]
    return sentences


def chunk_text(
    text: str,
    query: Optional[str] = None,
    target_span_tokens: int = 120,
    min_span_tokens: int = 60,
    max_span_tokens: int = 220,
    keep_headings: bool = True,
    keep_code_blocks: bool = True,
    keep_role_markers: bool = True,
    must_keep_keywords: Optional[Sequence[str]] = None,
    keep_last_n: int = 1,
) -> List[Span]:
    must_keep_keywords = must_keep_keywords or [
        "must",
        "requirement",
        "deadline",
        "constraint",
        "always",
        "never",
        "critical",
        "important",
    ]
    spans: List[Span] = []
    span_id = 0

    if query:
        query_text = normalize_whitespace(query)
        spans.append(
            Span(
                id=span_id,
                text=f"Query: {query_text}",
                token_count=count_tokens(query_text),
                is_question=True,
                must_keep=True,
                weight=1.4,
            )
        )
        span_id += 1

    sections = re.split(r"\n\s*\n", text.strip())
    for section_idx, raw_section in enumerate(sections):
        raw_section = raw_section.strip()
        if not raw_section:
            continue

        lines = raw_section.splitlines()
        heading_line = None
        body_lines = lines

        if keep_headings and lines and is_heading(lines[0]):
            heading_line = lines[0].strip()
            body_lines = lines[1:] or []

        if heading_line:
            spans.append(
                Span(
                    id=span_id,
                    text=heading_line,
                    token_count=count_tokens(heading_line),
                    is_heading=True,
                    must_keep=True,
                    weight=1.25,
                    metadata={"section": section_idx},
                )
            )
            span_id += 1

        paragraph = normalize_whitespace(" ".join(body_lines))
        if not paragraph:
            continue

        sentences = sentence_split(paragraph)
        buffer: List[str] = []
        for sentence in sentences:
            if not sentence:
                continue

            prospective = " ".join(buffer + [sentence]).strip()
            token_len = count_tokens(prospective)

            if token_len >= max_span_tokens and buffer:
                # Flush buffer before adding oversized sentence.
                merged = " ".join(buffer).strip()
                spans.append(
                    Span(
                        id=span_id,
                        text=merged,
                        token_count=count_tokens(merged),
                        metadata={"section": section_idx},
                    )
                )
                span_id += 1
                buffer = [sentence]
                continue

            buffer.append(sentence)
            token_len = count_tokens(" ".join(buffer))

            if token_len >= target_span_tokens and token_len >= min_span_tokens:
                merged = " ".join(buffer).strip()
                spans.append(
                    Span(
                        id=span_id,
                        text=merged,
                        token_count=token_len,
                        metadata={"section": section_idx},
                    )
                )
                span_id += 1
                buffer = []

        if buffer:
            merged = " ".join(buffer).strip()
            spans.append(
                Span(
                    id=span_id,
                    text=merged,
                    token_count=count_tokens(merged),
                    metadata={"section": section_idx},
                )
            )
            span_id += 1

    # Mark must-keep spans based on constraints.
    for span in spans:
        if keep_code_blocks and is_code_block(span.text):
            span.must_keep = True
            span.weight = max(span.weight, 1.25)
            span.metadata["contains_code"] = True
        if keep_role_markers and has_role_marker(span.text):
            span.must_keep = True
            span.weight = max(span.weight, 1.15)
            span.metadata["contains_role_marker"] = True

    # Mark must-keep spans based on keywords and recency.
    for span in spans:
        lowered = span.text.lower()
        if any(keyword in lowered for keyword in must_keep_keywords):
            span.must_keep = True
            span.weight = max(span.weight, 1.2)

    for span in spans[-keep_last_n:]:
        span.must_keep = True
        span.weight = max(span.weight, 1.1)

    return spans
