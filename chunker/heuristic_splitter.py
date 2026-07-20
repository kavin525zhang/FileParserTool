import re
import string
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from splitter import (
    Chunk, 
    SplitterConfig, 
    split_text,
    protected_spans,
    protected_spans_rune
)

from profiler import DocProfile


class boundary:
    def __init__(self, rune_start: int, priority: int):
        self.rune_start = rune_start
        self.priority = priority

def split_by_heuristics_impl(text: str, cfg: SplitterConfig, _profile: DocProfile = None) -> List[Chunk]:
    """
    splitByHeuristicsImpl is the Tier-2 implementation. Falls through to the
    legacy splitter when no heuristic boundaries are found.

    profile is currently unused (this tier scans for boundaries directly) but
    is accepted to keep the splitByHeadings / splitByHeuristics signatures
    uniform — see strategy.runTier.
    """
    if not text:
        return []

    runes = list(text)
    total_runes = len(runes)
    if total_runes <= cfg.chunk_size:
        return split_text(text, cfg)

    bounds = find_heuristic_boundaries(text, cfg.languages)
    # Drop any boundary that falls strictly inside a protected region (table,
    # fenced code block, LaTeX block, etc.) — splitting there would cut
    # through atomic content. Boundaries on a span edge are kept since they
    # align with the span edge and don't split protected content.
    prot = protected_spans_rune(text, protected_spans(text))
    if prot:
        bounds = drop_bounds_inside_spans(bounds, prot)

    if not bounds:
        return split_text(text, cfg)

    # Append a sentinel at end-of-document so the bin-packer can flush.
    bounds.append(boundary(rune_start=total_runes, priority=0))
    # Always start with a boundary at offset 0 if not already there.
    if not bounds or bounds[0].rune_start != 0:
        bounds.insert(0, boundary(rune_start=0, priority=0))

    # Sort bounds by position
    bounds.sort(key=lambda x: x.rune_start)

    # Greedy bin-packing.
    out = []
    seq = 0
    chunk_start = bounds[0].rune_start
    cur_end = chunk_start
    min_chunk_size = cfg.chunk_size // 4
    if min_chunk_size < 50:
        min_chunk_size = 50

    for i in range(1, len(bounds)):
        next_end = bounds[i].rune_start
        block_len = next_end - cur_end

        if block_len > cfg.chunk_size:
            # The block between the previous and this boundary is itself too
            # large to fit in any chunk. Flush current accumulation, then
            # recursively chunk the oversize block via the legacy splitter.
            if cur_end - chunk_start > 0:
                out = append_chunk(out, runes, chunk_start, cur_end, seq)
                seq += 1
                chunk_start = cur_end
            out = append_oversize_block(out, runes, cur_end, next_end, cfg, seq)
            seq += len(out)  # Adjust seq based on number of chunks added
            cur_end = next_end
            chunk_start = next_end
            continue

        # Would adding this block exceed the budget?
        accumulated = next_end - chunk_start
        if accumulated > cfg.chunk_size and cur_end - chunk_start >= min_chunk_size:
            # Flush accumulated content as a chunk, restart at curEnd.
            out = append_chunk(out, runes, chunk_start, cur_end, seq)
            seq += 1
            # Snap overlap start to the nearest semantic boundary or line
            # break instead of slicing mid-line / mid-word.
            chunk_start = apply_overlap_aligned(runes, cur_end, cfg.chunk_overlap, bounds)

        cur_end = next_end

    # Flush remaining content.
    if cur_end > chunk_start:
        out = append_chunk(out, runes, chunk_start, cur_end, seq)

    return out

def find_heuristic_boundaries(text: str, langs: List[str]) -> List[boundary]:
    """
    findHeuristicBoundaries scans text and returns boundary positions in
    ascending order. Lower-priority duplicates at the same offset are dropped.
    """
    bounds = []

    # Form feeds — strongest single-character boundary.
    for i, c in enumerate(text):
        if c == '\f':
            bounds.append(boundary(rune_start=i, priority=100))  # PrioFormFeed

    # Per-line patterns walk the text once, line by line.
    lines = text.split('\n')
    chapter_patterns = chapter_patterns_for_langs(langs)
    pos = 0
    in_fence = False

    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            rune_start = pos
            added = False

            # Check chapter patterns
            for pattern in chapter_patterns:
                if re.search(pattern.pattern, line):
                    bounds.append(boundary(rune_start=rune_start, priority=85))  # PrioChapterMarker
                    added = True
                    break

            # Check numbered sections
            if not added and re.search(r'^[ \t]*(?:\d+(?:\.\d+){1,3\.?|(?:\d+|[IVX]{1,5})\.)[ \t]+\S.{0,200}$', line, re.MULTILINE):
                bounds.append(boundary(rune_start=rune_start, priority=90))  # PrioNumberedHead
                added = True

            # Check all-caps headings
            if not added and re.search(r'^[ \t]*([A-ZÄÖÜ][A-ZÄÖÜ \-]{3,80}):?\s*$', line, re.MULTILINE):
                bounds.append(boundary(rune_start=rune_start, priority=70))  # PrioAllCapsHeading
                added = True

            # Check visual separators
            if not added and re.search(r'^[ \t]*(?:-{3,}|={3,}|\*{3,}|_{3,})[ \t]*$', line, re.MULTILINE):
                bounds.append(boundary(rune_start=rune_start, priority=60))  # PrioVisualSep
                added = True

            # Check page footers
            if not added and re.search(r'^[ \t]*(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?[ \t]*$', line, re.MULTILINE):
                bounds.append(boundary(rune_start=rune_start, priority=50))  # PrioPageFooter
                added = True

        pos += len(line)
        if i < len(lines) - 1:
            pos += 1  # \n

    # Excessive blank blocks (\n{3,}). Match at the *start* of the run so we
    # drop into the next paragraph cleanly.
    for match in re.finditer(r'\n{3,}', text):
        rune_start = len(text[:match.start()])
        bounds.append(boundary(rune_start=rune_start, priority=40))  # PrioBlankBlock

    if not bounds:
        return []

    # Sort by position; drop near-duplicate offsets keeping the highest priority.
    bounds.sort(key=lambda x: (x.rune_start, -x.priority))
    deduped = []
    prev = -1
    for b in bounds:
        if b.rune_start != prev:
            deduped.append(b)
            prev = b.rune_start

    return deduped

def chapter_patterns_for_langs(langs: List[str]) -> List[re.Pattern]:
    """Return chapter-marker regexes that apply for the given language hints."""
    if not langs:
        return [
            re.compile(r'^[ \t]*(?:Kapitel|Abschnitt|Teil)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$', re.MULTILINE),
            re.compile(r'^[ \t]*(?:Chapter|Section|Part)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$', re.MULTILINE),
            re.compile(r'^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$', re.MULTILINE)
        ]

    patterns = []
    for l in langs:
        if l.lower() == "german":
            patterns.append(re.compile(r'^[ \t]*(?:Kapitel|Abschnitt|Teil)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$', re.MULTILINE))
        elif l.lower() == "english":
            patterns.append(re.compile(r'^[ \t]*(?:Chapter|Section|Part)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$', re.MULTILINE))
        elif l.lower() == "chinese":
            patterns.append(re.compile(r'^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$', re.MULTILINE))

    if not patterns:
        return [
            re.compile(r'^[ \t]*(?:Kapitel|Abschnitt|Teil)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$', re.MULTILINE),
            re.compile(r'^[ \t]*(?:Chapter|Section|Part)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$', re.MULTILINE),
            re.compile(r'^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$', re.MULTILINE)
        ]

    return patterns

def drop_bounds_inside_spans(bounds: List[boundary], spans: List[Dict[str, int]]) -> List[boundary]:
    """
    dropBoundsInsideSpans returns bounds with entries that fall strictly
    inside any of the (rune-offset) protected spans removed. Bounds at a
    span's start or end are kept — they align with the span edge and don't
    split protected content. spans must be sorted by start.
    """
    if not spans:
        return bounds

    out = []
    for b in bounds:
        inside = False
        for s in spans:
            if s["start"] < b.rune_start < s["end"]:
                inside = True
                break
        if not inside:
            out.append(b)

    return out

def append_chunk(out: List[Chunk], runes: List[str], start: int, end: int, seq: int) -> List[Chunk]:
    """
    appendChunk slices runes[start:end] into a Chunk and appends it to out.
    Pure-whitespace slices are skipped (boundary clustering can occasionally
    produce them). The Content stored is the raw slice — Start/End rune
    offsets must match utf8.RuneCountInString(Content) for downstream
    reconstruction code; whitespace stripping for embedding happens in
    Chunk.EmbeddingContent.
    """
    if end <= start:
        return out

    raw = ''.join(runes[start:end])
    if not raw.strip():
        return out

    c = Chunk(content=raw, seq=seq, start=start, end=end)
    return out + [c]

def append_oversize_block(out: List[Chunk], runes: List[str], start: int, end: int, cfg: SplitterConfig, seq: int) -> List[Chunk]:
    """
    appendOversizeBlock recursively chunks a region that is itself larger than
    cfg.chunk_size, using the legacy splitter so internal length budgets and
    protected patterns are still respected.
    """
    if end <= start:
        return out

    sub_text = ''.join(runes[start:end])
    subs = split_text(sub_text, cfg)
    for s in subs:
        out.append(Chunk(
            content=s.content,
            seq=seq,
            start=start + s.start,
            end=start + s.end
        ))
        seq += 1

    return out

def apply_overlap_aligned(runes: List[str], cur_end: int, overlap: int, bounds: List[boundary]) -> int:
    """
    applyOverlapAligned returns the rune offset where the next chunk should
    start. The target is `curEnd - overlap`, but we snap to the nearest
    preceding boundary (within 2x overlap) or, failing that, the previous
    newline so chunks don't begin mid-line / mid-word. Falls back to the raw
    target only if neither option is available.

    curEnd itself is always a boundary (the bin-packer flushes at boundary
    positions), so we exclude it from the search — picking it would yield
    zero overlap, defeating the purpose of this function.
    """
    if overlap <= 0:
        return cur_end

    target = cur_end - overlap
    if target < 0:
        target = 0

    # Allowed search window: [curEnd - 2*overlap, curEnd)
    window_start = cur_end - 2 * overlap
    if window_start < 0:
        window_start = 0

    # Prefer a semantic boundary strictly inside the window.
    best_bound = -1
    for b in bounds:
        if window_start <= b.rune_start < cur_end and b.rune_start > best_bound:
            best_bound = b.rune_start

    if best_bound >= 0:
        return best_bound

    # Fallback: scan backwards from `target` to the previous newline, but
    # not past windowStart so we keep the overlap roughly the right size.
    i = target
    while i > window_start and i < len(runes):
        if runes[i] == '\n':
            return i + 1
        i -= 1

    return target

def build_units_with_protection(text: str, protected: List[Dict[str, int]], separators: List[str], chunk_size: int) -> List[Dict[str, Any]]:
    """Split text into units, preserving protected spans as atomic."""
    MAX_PROTECTED_SIZE = 7500

    units = []
    byte_pos = 0
    rune_pos = 0

    for span in protected:
        if span["start"] > byte_pos:
            pre = text[byte_pos:span["start"]]
            parts = split_by_separators(pre, separators, chunk_size)
            rune_offset = rune_pos
            for part in parts:
                part_len = len(part)
                units.append({
                    'text': part,
                    'start': rune_offset,
                    'end': rune_offset + part_len
                })
                rune_offset += part_len
            rune_pos += len(pre)

        prot_text = text[span["start"]:span["end"]]
        prot_len = len(prot_text)

        # If protected content is too large, forcibly split it
        if prot_len > MAX_PROTECTED_SIZE:
            offset = 0
            while offset < len(prot_text):
                chunk_end = offset + MAX_PROTECTED_SIZE
                if chunk_end > len(prot_text):
                    chunk_end = len(prot_text)
                else:
                    # Try to break at a newline or space
                    for i in range(chunk_end - 1, max(offset, chunk_end - 200), -1):
                        if prot_text[i] in ['\n', ' ']:
                            chunk_end = i + 1
                            break

                chunk_text = prot_text[offset:chunk_end]
                chunk_len = chunk_end - offset
                units.append({
                    'text': chunk_text,
                    'start': rune_pos + offset,
                    'end': rune_pos + offset + chunk_len
                })
                offset = chunk_end
        else:
            # Normal case: keep protected content as a single unit
            units.append({
                'text': prot_text,
                'start': rune_pos,
                'end': rune_pos + prot_len
            })

        rune_pos += prot_len
        byte_pos = span["end"]

    if byte_pos < len(text):
        remaining = text[byte_pos:]
        parts = split_by_separators(remaining, separators, chunk_size)
        rune_offset = rune_pos
        for part in parts:
            part_len = len(part)
            units.append({
                'text': part,
                'start': rune_offset,
                'end': rune_offset + part_len
            })
            rune_offset += part_len

    return units

def split_by_separators(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Split text by separators in priority order, recursively applying the next separator to any piece that is still larger than chunkSize."""
    if not text or not separators:
        return [text]

    if chunk_size > 0 and len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if not sep:
            continue

        # Escape special regex characters in separator
        escaped_sep = re.escape(sep)
        pattern = f'({escaped_sep})'

        # Split while keeping the separator
        parts = re.split(pattern, text)

        # Combine text and separators back
        pieces = []
        for j in range(0, len(parts), 2):
            if j < len(parts) and parts[j]:
                pieces.append(parts[j])
            if j + 1 < len(parts) and parts[j + 1]:
                pieces.append(parts[j + 1])

        if len(pieces) <= 1:
            continue

        # Recursively split any piece that is still too large with the remaining (lower-priority) separators.
        out = []
        remaining = separators[i + 1:]
        for p in pieces:
            if chunk_size > 0 and len(p) > chunk_size and remaining:
                out.extend(split_by_separators(p, remaining, chunk_size))
            else:
                out.append(p)
        return out

    return [text]

def units_text(units: List[Dict[str, Any]]) -> str:
    """Concatenate the text of all units."""
    return ''.join(u['text'] for u in units)

def header_already_present(headers: str, overlap_text: str, unit_text: str) -> bool:
    """Return True if the column-name row from the header is already present in the overlap or the next unit."""
    if headers in overlap_text or headers in unit_text:
        return True

    col_row = header_column_row(headers)
    if not col_row:
        return False

    return col_row in overlap_text or col_row in unit_text

def header_column_row(header: str) -> str:
    """Extract the column-name line from a header string."""
    for line in header.split('\n'):
        line = line.strip()
        if not line or '---' in line:
            continue
        # Skip lines that are only pipes/whitespace
        if all(c in '| \t\r\n' for c in line):
            continue
        return line
    return ""

def build_chunk(units: List[Dict[str, Any]], seq: int) -> Chunk:
    """Build a chunk from units."""
    content = ''.join(u['text'] for u in units)
    return Chunk(
        Content=content,
        Seq=seq,
        Start=units[0]['start'],
        End=units[-1]['end']
    )

def compute_overlap(current: List[Dict[str, Any]], chunk_overlap: int, chunk_size: int, next_len: int) -> Tuple[List[Dict[str, Any]], int]:
    """Compute the units to keep for overlap and their total length."""
    if chunk_overlap <= 0:
        return [], 0

    overlap_len = 0
    start_idx = len(current)
    for i in range(len(current) - 1, -1, -1):
        u_len = len(current[i]['text'])
        if overlap_len + u_len > chunk_overlap:
            break
        # Check that overlap + next unit fits in chunk
        if overlap_len + u_len + next_len > chunk_size:
            break
        overlap_len += u_len
        start_idx = i

    # Skip leading separator-only and header-marker units in the overlap
    while start_idx < len(current):
        u = current[start_idx]
        is_header_marker = u['start'] == u['end']
        trimmed = u['text'].strip()
        if is_header_marker or not trimmed or is_separator_only(u['text']):
            overlap_len -= len(u['text'])
            start_idx += 1
        else:
            break

    if start_idx >= len(current):
        return [], 0

    overlap = current[start_idx:]
    return overlap, overlap_len

def is_separator_only(s: str) -> bool:
    """Check if string contains only separator characters."""
    for c in s:
        if c not in '\n\r \t。':
            return False
    return True

