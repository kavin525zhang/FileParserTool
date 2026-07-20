import re
from typing import List, Dict, Any, Tuple

from chunker.heading_hierarchy import HeadingHierarchy
from chunker.profiler import profile_document, DocProfile
from chunker.splitter import Chunk, SplitterConfig, split_text
from chunker.patterns import MarkdownHeadingPattern



def split_by_headings_impl(text: str, cfg: SplitterConfig, profile: DocProfile = None) -> List[Chunk]:
    """
    split_by_headings_impl is the Tier-1 implementation. It falls through to the
    legacy splitter when the document has no usable heading structure or when
    the heading split would produce a single section anyway.

    profile may be None; we compute one on demand. When the strategy resolver
    already ran the profiler (auto strategy), the same profile is threaded
    through here so we don't re-scan the entire document.
    """
    if not text:
        return []

    if profile is None:
        profile = profile_document(text)

    primary_level = profile.dominant_heading_level()
    if primary_level == 0:
        return split_text(text, cfg)

    bounds = find_heading_boundaries(text, primary_level)
    if len(bounds) <= 1:
        return split_text(text, cfg)

    runes = list(text)
    hierarchy = HeadingHierarchy()

    # Pre-walk every heading (not just primary-level) so the hierarchy
    # reflects the full nesting context for each section's start. We only
    # snapshot the breadcrumb at section boundaries; deeper sub-headings
    # inside a section update the hierarchy but do not change the chunk's
    # breadcrumb (chunks within a section share one breadcrumb).
    out = []
    seq = 0

    for i, b in enumerate(bounds):
        end_rune = len(runes)
        if i + 1 < len(bounds):
            end_rune = bounds[i + 1].rune_start

        if b.line:
            hierarchy.observe(b.line)

        # Catch sub-headings that occur between this primary boundary and
        # the next so the hierarchy stays in sync for subsequent sections.
        # We intentionally do this after observing the section header so
        # the breadcrumb reflects the section-leading heading.
        breadcrumb = hierarchy.breadcrumb_with_hashes()
        section_start = HeadingHierarchy()
        # Copy the current hierarchy state
        section_start._stack = hierarchy._stack.copy()
        section_start._depth = hierarchy._depth
        observe_sub_headings(runes[b.rune_start:end_rune], primary_level, hierarchy)

        section_runes = runes[b.rune_start:end_rune]
        section_content = ''.join(section_runes)
        sec_len = len(section_runes)
        if sec_len == 0:
            continue

        bc_len = len(breadcrumb)
        # Single-chunk section: emit as-is, breadcrumb tracked separately.
        # The breadcrumb is delivered via Chunk.ContextHeader (not Content)
        # to preserve End-Start == len(Content) invariants relied on by
        # document reconstruction (knowledge.go:2278+).
        if bc_len + 2 + sec_len <= cfg.chunk_size:
            out.append(Chunk(
                content=section_content,
                context_header=breadcrumb,
                seq=seq,
                start=b.rune_start,
                end=end_rune
            ))
            seq += 1
            continue

        # Section too large: defer to the legacy splitter for inner
        # segmentation. We do NOT shrink the inner ChunkSize budget here
        # because the breadcrumb no longer counts against Content size.
        # Each sub-chunk gets a breadcrumb reflecting the deepest heading
        # active at its start, so deep `###`/`####` sub-headings inside a
        # long section aren't collapsed to the section-level header.
        sub_breadcrumbs = section_breadcrumbs(section_runes, primary_level, section_start)
        sub_chunks = split_text(section_content, cfg)
        for sub in sub_chunks:
            out.append(Chunk(
                content=sub.content,
                context_header=breadcrumb_at_offset(sub_breadcrumbs, sub.start, breadcrumb),
                seq=seq,
                start=b.rune_start + sub.start,
                end=b.rune_start + sub.end
            ))
            seq += 1

    return coalesce_tiny_chunks(out, cfg.chunk_size)

class headingBoundary:
    def __init__(self, rune_start: int, line: str = ""):
        self.rune_start = rune_start
        self.line = line

def coalesce_tiny_chunks(in_chunks: List[Chunk], chunk_size: int) -> List[Chunk]:
    """
    coalesce_tiny_chunks merges adjacent small chunks under their shared heading
    context so that documents whose primary sections are mostly short (FAQs,
    install logs, change-lists) don't trip the validator's "too many tiny
    chunks" rule and fall through all the way to legacy. The merged breadcrumb
    is the line-prefix shared by both inputs; the original sub-headings remain
    visible because heading_splitter includes the heading line in each
    section's Content.

    Safety:
      - We only merge when cur.End == next.Start. That preserves the
        End-Start == len([]rune(Content)) invariant that document
        reconstruction relies on, and naturally skips legacy sub-chunks (which
        may overlap due to ChunkOverlap).
      - We stop accumulating once the running chunk reaches the merge target
        (≈ ChunkSize/2) so we don't aggressively pack chunks beyond what the
        validator considers comfortable.
    """
    if len(in_chunks) <= 1 or chunk_size <= 0:
        return in_chunks

    target = chunk_size // 2
    if target < 200:
        target = 200

    out = []
    cur = in_chunks[0]
    cur_len = len(cur.content)

    for i in range(1, len(in_chunks)):
        next_chunk = in_chunks[i]
        next_len = len(next_chunk.content)
        shared_header = common_heading_prefix(cur.context_header, next_chunk.context_header)
        # Adjacent + still-small + would not blow the size budget → merge.
        if (shared_header and
            cur.end == next_chunk.start and
            cur_len < target and
            cur_len + next_len <= chunk_size):
            cur.content += next_chunk.content
            cur.context_header = shared_header
            cur.end = next_chunk.end
            cur_len += next_len
            continue

        out.append(cur)
        cur = next_chunk
        cur_len = next_len

    out.append(cur)

    # Re-sequence — downstream code (knowledge.go) expects Seq to be a dense
    # 0..N-1 range over the returned slice.
    for i, chunk in enumerate(out):
        chunk.Seq = i

    return out

def common_heading_prefix(a: str, b: str) -> str:
    """
    common_heading_prefix returns the longest line-aligned prefix shared by two
    breadcrumb strings. Heading hierarchies are emitted as
    "# Top\n## Section\n### Sub", so a line-by-line comparison is sufficient
    and avoids partial-line truncation that would corrupt the breadcrumb.
    """
    if a == b:
        return a

    la = a.split('\n') if a else []
    lb = b.split('\n') if b else []
    n = min(len(la), len(lb))
    common = 0
    for i in range(n):
        if la[i] != lb[i]:
            break
        common = i + 1

    if common == 0:
        return ""
    return '\n'.join(la[:common])

def find_heading_boundaries(text: str, primary_level: int) -> List[headingBoundary]:
    """
    find_heading_boundaries returns one boundary at offset 0 plus one per
    Markdown heading at level <= primary_level that sits outside fenced code
    blocks. Heading detection is line-oriented — a heading must occupy a
    whole line to be recognized.
    """
    if not text:
        return [headingBoundary(0)]

    bounds = [headingBoundary(0)]
    lines = text.split('\n')
    pos = 0
    in_fence = False

    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            pos += len(line)
            if i < len(lines) - 1:
                pos += 1  # newline
            continue

        if not in_fence:
            m = MarkdownHeadingPattern.match(line)
            if m:
                level = len(m.group(1))
                if 1 <= level <= primary_level:
                    if pos > 0:
                        bounds.append(headingBoundary(
                            rune_start=pos,
                            line=line
                        ))
                    elif pos == 0:
                        # First line is a heading — replace the leading boundary
                        bounds[0].line = line

        pos += len(line)
        if i < len(lines) - 1:
            pos += 1  # account for the \n that split removed

    return bounds

def observe_sub_headings(runes: List[str], primary_level: int, h: HeadingHierarchy) -> None:
    """
    observe_sub_headings walks the section's lines and feeds every Markdown
    heading deeper than primary_level into the hierarchy. This keeps the
    hierarchy state correct so the breadcrumb at the next primary section
    reflects the truly active stack.
    """
    if not runes:
        return

    text = ''.join(runes)
    in_fence = False
    for line in text.split('\n'):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            continue

        if in_fence:
            continue

        m = re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$', line)
        if not m:
            continue

        level = len(m.group(1))
        if level > primary_level:
            h.observe(line)

class sectionBreadcrumb:
    def __init__(self, rune_start: int, breadcrumb: str):
        self.rune_start = rune_start
        self.breadcrumb = breadcrumb

def section_breadcrumbs(section_runes: List[str], primary_level: int, seed: HeadingHierarchy) -> List[sectionBreadcrumb]:
    """
    sectionBreadcrumbs walks a section's deeper sub-headings (level >
    primaryLevel) and records, for each, the rune offset where it takes effect
    and the resulting breadcrumb. seed is the hierarchy state at the section's
    start (already including the section heading and its ancestors). The
    returned slice is ordered by runeStart and always begins with the seed
    breadcrumb at offset 0, so a sub-chunk sitting far below a deep heading
    still resolves to that heading's path rather than the section header.
    """
    h = HeadingHierarchy()
    # Copy the seed state
    h._stack = seed._stack.copy()
    h._depth = seed._depth

    result = [sectionBreadcrumb(rune_start=0, breadcrumb=h.breadcrumb_with_hashes())]
    pos = 0
    in_fence = False
    lines = ''.join(section_runes).split('\n')

    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            pos += len(line)
            if i < len(lines) - 1:
                pos += 1
            continue

        if not in_fence:
            m = re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$', line)
            if m:
                level = len(m.group(1))
                if level > primary_level:
                    h.observe(line)
                    result.append(sectionBreadcrumb(
                        rune_start=pos,
                        breadcrumb=h.breadcrumb_with_hashes()
                    ))

        pos += len(line)
        if i < len(lines) - 1:
            pos += 1

    return result

def breadcrumb_at_offset(bcs: List[sectionBreadcrumb], offset: int, fallback: str) -> str:
    """
    breadcrumbAtOffset returns the breadcrumb in effect at the given rune offset
    — the last entry whose runeStart <= offset. fallback covers the (unreachable
    in practice) empty-slice case.
    """
    bc = fallback
    for e in bcs:
        if e.rune_start > offset:
            break
        bc = e.breadcrumb
    return bc

def match_heading(line: str, counts: Dict[int, int]) -> bool:
    """Check whether line is an ATX heading and increment the appropriate level counter."""
    m = re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$', line)
    if not m:
        return False
    level = len(m.group(1))
    if level < 1 or level > 6:
        return False
    counts[level] = counts.get(level, 0) + 1
    return True

def detect_language(text: str) -> str:
    """Detect language of the text. Simplified version."""
    # This is a simplified version - in practice you'd use a proper language detection library
    german_indicators = ["der", "die", "das", "und", "ist", "Kapitel", "Abschnitt"]
    english_indicators = ["the", "and", "of", "to", "in", "Chapter", "Section"]
    chinese_indicators = ["第", "章", "节", "的", "是", "在", "和"]

    german_count = sum(1 for word in german_indicators if word in text)
    english_count = sum(1 for word in english_indicators if word in text)
    chinese_count = sum(1 for word in chinese_indicators if word in text)

    if chinese_count > english_count and chinese_count > german_count:
        return "chinese"
    elif english_count > german_count:
        return "english"
    elif german_count > 0:
        return "german"
    else:
        return "english"  # default

def build_units_with_protection(text: str, protected: List[Tuple[int, int]], separators: List[str], chunk_size: int) -> List[Dict[str, Any]]:
    """Split text into units, preserving protected spans as atomic."""
    MAX_PROTECTED_SIZE = 7500

    units = []
    byte_pos = 0
    rune_pos = 0

    for start, end in protected:
        if start > byte_pos:
            pre = text[byte_pos:start]
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

        prot_text = text[start:end]
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
        byte_pos = end

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
        content=content,
        seq=seq,
        start=units[0]['start'],
        end=units[-1]['end']
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
