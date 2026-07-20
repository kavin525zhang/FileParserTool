from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import re
import unicodedata

from chunker.header_tracker import HeaderTracker, header_column_mismatch


def normalize_text(s: str) -> str:
    """Normalize text for comparison by removing extra whitespace and normalizing unicode."""
    s = re.sub(r'\s+', ' ', s).strip()
    return unicodedata.normalize('NFKC', s)

def rune_len(s: str) -> int:
    """Return the number of Unicode code points (runes) in a string."""
    return len(s)

def is_separator_only(s: str) -> bool:
    """Check if a string contains only separator characters."""
    for c in s:
        if c not in '\n\r \t。':
            return False
    return True

def header_column_row(header: str) -> str:
    """Extract the column-name line from a header string.

    Returns empty string if the header has no meaningful column names.
    """
    for line in header.split('\n'):
        line = line.strip()
        if not line or '---' in line:
            continue
        # Skip lines that are only pipes/whitespace (empty header rows)
        only_pipes = True
        for c in line:
            if c not in '| \t':
                only_pipes = False
                break
        if not only_pipes:
            return line
    return ''

def header_already_present(headers: str, overlap_text: str, unit_text: str) -> bool:
    """Check if the column-name row from the header is already present in the overlap or unit.

    Prevents duplication of header content.
    """
    # Fast path: full header already in overlap or unit
    if headers in overlap_text or headers in unit_text:
        return True

    # Extract the column-name row (first meaningful non-separator line)
    col_row = header_column_row(headers)
    if not col_row:
        return False

    return col_row in overlap_text or col_row in unit_text

def units_text(units: List[SplitUnit]) -> str:
    """Concatenate the text of all units."""
    return ''.join(unit.text for unit in units)

def compute_overlap(
    current: List[SplitUnit],
    chunk_overlap: int,
    chunk_size: int,
    next_len: int
) -> Tuple[List[SplitUnit], int]:
    """Compute the units to keep for overlap and their total rune length.

    Args:
        current: Current list of units in the chunk
        chunk_overlap: Desired overlap size in runes
        chunk_size: Maximum chunk size in runes
        next_len: Length of the next unit to be added

    Returns:
        Tuple of (overlap units, total overlap length)
    """
    if chunk_overlap <= 0:
        return [], 0

    # Walk backward from end, accumulating overlap
    overlap_len = 0
    start_idx = len(current)
    for i in range(len(current) - 1, -1, -1):
        u_len = rune_len(current[i].text)
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
        is_header_marker = u.start == u.end
        trimmed = u.text.strip()
        if is_header_marker or not trimmed or is_separator_only(u.text):
            overlap_len -= rune_len(u.text)
            start_idx += 1
        else:
            break

    if start_idx >= len(current):
        return [], 0

    overlap = current[start_idx:].copy()
    return overlap, overlap_len

def build_chunk(units: List[SplitUnit], seq: int) -> Chunk:
    """Build a chunk from a list of units."""
    content = ''.join(unit.text for unit in units)
    return Chunk(
        content=content,
        seq=seq,
        start=units[0].start,
        end=units[-1].end
    )

def split_by_separators(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Split text by separators in priority order, recursively applying the next
    separator to any piece that is still larger than chunk_size.

    Mirrors the recursive priority semantics of the Python reference splitter:
    if '\n\n' produces a piece that's still too big, '\n' (and subsequent separators)
    are applied within that piece — not to the whole text.

    chunk_size == 0 disables the recursion guard; callers that don't care
    about size budget (e.g. a final mergeUnits-style pass) pass 0.

    Args:
        text: Text to split
        separators: List of separators in priority order
        chunk_size: Maximum size for pieces (0 to disable)

    Returns:
        List of text pieces
    """
    if not text or not separators:
        return [text]

    if chunk_size > 0 and rune_len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if not sep:
            continue

        # Escape special regex characters in separator
        escaped_sep = re.escape(sep)
        pattern = f'({escaped_sep})'

        # Split text and capture separators
        parts = re.split(pattern, text)

        # Reconstruct pieces with separators
        pieces = []
        for j in range(0, len(parts), 2):
            if j < len(parts) and parts[j]:
                pieces.append(parts[j])
            if j + 1 < len(parts) and parts[j + 1]:
                pieces.append(parts[j + 1])

        if len(pieces) <= 1:
            continue

        # Recursively split any piece that is still too large with the
        # remaining (lower-priority) separators.
        out = []
        remaining = separators[i + 1:]
        for p in pieces:
            if chunk_size > 0 and rune_len(p) > chunk_size and remaining:
                out.extend(split_by_separators(p, remaining, chunk_size))
            else:
                out.append(p)
        return out

    return [text]

def protected_spans(text: str) -> List[Tuple[int, int]]:
    """Find all non-overlapping protected regions in text.

    Protected regions include:
    - LaTeX block math ($$...$$)
    - Markdown images (![alt](url))
    - Markdown links ([text](url))
    - Markdown tables (header + separator, rows)
    - Fenced code blocks (```...```)

    Returns:
        List of (start, end) byte offsets for protected regions
    """
    patterns = [
        re.compile(r'(?s)\$\$.*?\$\$'),  # LaTeX block math
        re.compile(r'!\[[^\]]*\]\([^)]+\)'),  # Markdown images
        re.compile(r'\[[^\]]*\]\([^)]+\)'),  # Markdown links
        # Table header + separator
        re.compile(r'(?m)[ ]*(?:\|[^|\n]*)+\|[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|[\r\n]+'),
        # Table rows
        re.compile(r'(?m)[ ]*(?:\|[^|\n]*)+\|[\r\n]+'),
        # Fenced code blocks
        re.compile(r'(?s)```(?:\w+)?[\r\n].*?```'),
    ]

    all_matches = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if end > start:
                all_matches.append((start, end))

    if not all_matches:
        return []

    # Sort by start, then by length descending
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Remove overlaps
    result = []
    last_end = 0
    for start, end in all_matches:
        if start >= last_end:
            result.append((start, end))
            last_end = end

    return result

def protected_spans_rune(text: str, byte_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Convert byte-offset protected spans to rune offsets in a single forward pass.

    Used by callers that work in rune space to avoid choosing chunk boundaries
    that cut through protected content. byte_spans must be sorted by start.

    Args:
        text: Original text
        byte_spans: List of (start, end) byte offsets

    Returns:
        List of (start, end) rune offsets
    """
    if not byte_spans:
        return []

    out = []
    rune_idx = 0
    byte_idx = 0

    for start_byte, end_byte in byte_spans:
        # Advance to start of span
        while byte_idx < start_byte and byte_idx < len(text):
            char = text[byte_idx]
            size = len(char.encode('utf-8'))
            byte_idx += size
            rune_idx += 1

        start_rune = rune_idx

        # Advance to end of span
        while byte_idx < end_byte and byte_idx < len(text):
            char = text[byte_idx]
            size = len(char.encode('utf-8'))
            byte_idx += size
            rune_idx += 1

        out.append((start_rune, rune_idx))

    return out

def extract_image_refs(text: str) -> List[ImageRef]:
    """Extract markdown image references from text.

    The URL group supports one level of balanced parentheses so that URLs
    like https://example.com/item_(abc)/123 are captured in full.

    Args:
        text: Text to search for image references

    Returns:
        List of ImageRef objects
    """
    # This pattern captures URLs with balanced parentheses
    pattern = r'!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)'
    matches = re.finditer(pattern, text)
    refs = []
    for match in matches:
        refs.append(ImageRef(
            original_ref=match.group(2),
            alt_text=match.group(1),
            start=match.start(),
            end=match.end()
        ))
    return refs


def new_header_tracker() -> HeaderTracker:
    """Create a new header tracker."""
    return HeaderTracker()


def build_units_with_protection(
    text: str,
    protected: List[Tuple[int, int]],
    separators: List[str],
    chunk_size: int
) -> List[SplitUnit]:
    """Split text into units, preserving protected spans as atomic.

    Start/End positions in the returned units are rune offsets (not byte offsets),
    because downstream merge logic indexes content via string slicing.
    If a protected span exceeds max_protected_size, it will be forcibly split to prevent
    creating chunks that are too large for downstream processing (e.g., embedding APIs).
    chunk_size is forwarded to split_by_separators so recursive splitting can keep pieces
    under the budget when one separator alone leaves a piece oversize.

    Args:
        text: Text to split
        protected: List of (start, end) byte offsets for protected regions
        separators: List of separators in priority order
        chunk_size: Maximum size for pieces

    Returns:
        List of split units with position tracking
    """
    MAX_PROTECTED_SIZE = 7500  # Maximum size for a protected unit

    units = []
    byte_pos = 0
    rune_pos = 0

    if protected:
        for i, (p_start, p_end) in enumerate(protected):
            # Add text before protected span
            if p_start > byte_pos:
                pre = text[byte_pos:p_start]
                parts = split_by_separators(pre, separators, chunk_size)
                rune_offset = rune_pos
                for part in parts:
                    part_rune_len = rune_len(part)
                    units.append(SplitUnit(
                        text=part,
                        start=rune_offset,
                        end=rune_offset + part_rune_len
                    ))
                    rune_offset += part_rune_len
                rune_pos += rune_len(pre)

            # Process protected text
            prot_text = text[p_start:p_end]
            prot_rune_len = rune_len(prot_text)

            # If protected content is too large, forcibly split it
            if prot_rune_len > MAX_PROTECTED_SIZE:
                # Split into smaller chunks at line breaks or spaces
                runes = list(prot_text)
                offset = 0
                while offset < len(runes):
                    chunk_end = offset + MAX_PROTECTED_SIZE
                    if chunk_end > len(runes):
                        chunk_end = len(runes)
                    else:
                        # Try to break at a newline or space
                        for i in range(chunk_end - 1, max(offset, chunk_end - 200), -1):
                            if runes[i] in ['\n', ' ']:
                                chunk_end = i + 1
                                break

                    chunk_text = ''.join(runes[offset:chunk_end])
                    chunk_len = chunk_end - offset
                    units.append(SplitUnit(
                        text=chunk_text,
                        start=rune_pos + offset,
                        end=rune_pos + offset + chunk_len
                    ))
                    offset = chunk_end
            else:
                # Normal case: keep protected content as a single unit
                units.append(SplitUnit(
                    text=prot_text,
                    start=rune_pos,
                    end=rune_pos + prot_rune_len
                ))

            rune_pos += prot_rune_len
            byte_pos = p_end

    # Add remaining text after last protected span
    if byte_pos < len(text):
        remaining = text[byte_pos:]
        parts = split_by_separators(remaining, separators, chunk_size)
        rune_offset = rune_pos
        for part in parts:
            part_rune_len = rune_len(part)
            units.append(SplitUnit(
                text=part,
                start=rune_offset,
                end=rune_offset + part_rune_len
            ))
            rune_offset += part_rune_len

    return units

def merge_units(
    units: List[SplitUnit],
    chunk_size: int,
    chunk_overlap: int
) -> List[Chunk]:
    """Merge split units into chunks with overlap tracking.

    Enforces an absolute maximum chunk size to prevent exceeding downstream limits.
    Active contextual headers (e.g., Markdown table headers) are prepended to new
    chunks so that every chunk carries its own header context.

    Args:
        units: List of split units to merge
        chunk_size: Maximum size for chunks in runes
        chunk_overlap: Overlap size between chunks in runes

    Returns:
        List of chunks
    """
    if not units:
        return []

    ABSOLUTE_MAX_SIZE = 7500

    ht = new_header_tracker()
    chunks = []
    current = []
    cur_len = 0

    for u in units:
        u_len = rune_len(u.text)

        # If this single unit exceeds absolute max, force split it further
        if u_len > ABSOLUTE_MAX_SIZE:
            # Flush current chunk if any
            if current:
                chunks.append(build_chunk(current, len(chunks)))
                current = []
                cur_len = 0

            # Update header state even for oversized units
            ht.update(u.text)

            # Split this oversized unit into smaller chunks
            runes = list(u.text)
            offset = 0
            while offset < len(runes):
                chunk_end = offset + ABSOLUTE_MAX_SIZE
                if chunk_end > len(runes):
                    chunk_end = len(runes)
                else:
                    # Try to break at a newline or space
                    for i in range(chunk_end - 1, max(offset, chunk_end - 200), -1):
                        if runes[i] in ['\n', ' ']:
                            chunk_end = i + 1
                            break

                chunk_text = ''.join(runes[offset:chunk_end])
                chunks.append(Chunk(
                    content=chunk_text,
                    seq=len(chunks),
                    start=u.start + offset,
                    end=u.start + chunk_end
                ))
                offset = chunk_end
            continue

        # Update header tracking
        ht.update(u.text)
        # Flush at table boundary so the next table is not merged into a chunk
        # that still carries the previous table's prepended header context.
        if ht.header_ended_this_unit and current:
            chunks.append(build_chunk(current, len(chunks)))
            current = []
            cur_len = 0

        headers = ht.get_headers()
        headers_len = rune_len(headers)
        if headers_len > chunk_size:
            headers = ""
            headers_len = 0

        # If adding this unit (plus reserving space for headers in a potential
        # next chunk) would exceed chunk size, flush the current chunk.
        if cur_len + u_len + headers_len > chunk_size and current:
            chunks.append(build_chunk(current, len(chunks)))

            # Keep overlap from the end of current
            current, cur_len = compute_overlap(current, chunk_overlap, chunk_size, u_len)

            # Shrink overlap further if needed to fit headers + next unit
            if headers and headers_len + u_len <= chunk_size:
                while current and cur_len + u_len + headers_len > chunk_size:
                    cur_len -= rune_len(current[0].text)
                    current = current[1:]

                # Prepend headers if the column-name context is not already present
                # in the overlap or the next unit being added.
                overlap_text = units_text(current)
                if (not header_already_present(headers, overlap_text, u.text) and
                    not header_column_mismatch(headers, u.text)):
                    start_pos = u.start
                    if current:
                        start_pos = current[0].start
                    h_unit = SplitUnit(text=headers, start=start_pos, end=start_pos)
                    current = [h_unit] + current
                    cur_len += headers_len

        # Check if adding this unit would exceed absolute max
        if cur_len + u_len > ABSOLUTE_MAX_SIZE:
            if current:
                chunks.append(build_chunk(current, len(chunks)))
                current = []
                cur_len = 0

        current.append(u)
        cur_len += u_len

    # Flush remaining
    if current:
        chunks.append(build_chunk(current, len(chunks)))

    return chunks


def split_text(text: str, cfg: SplitterConfig) -> List[Chunk]:
    """Split text into chunks with overlap, respecting protected patterns.

    Args:
        text: Text to split
        cfg: Splitter configuration

    Returns:
        List of chunks
    """
    if not text:
        return []

    chunk_size = cfg.chunk_size
    chunk_overlap = cfg.chunk_overlap
    separators = cfg.separators

    if chunk_size <= 0:
        chunk_size = 512
    if chunk_overlap < 0:
        chunk_overlap = 0

    # Step 1: Find protected spans
    protected = protected_spans(text)

    # Step 2: Split non-protected regions by separators, keep protected as atomic units.
    # chunk_size is forwarded so split_by_separators can recursively apply lower-priority
    # separators to oversize pieces (Python-parity recursive split).
    units = build_units_with_protection(text, protected, separators, chunk_size)

    # Step 3: Merge units into chunks with overlap
    return merge_units(units, chunk_size, chunk_overlap)

def split_text_parent_child(
    text: str,
    parent_cfg: SplitterConfig,
    child_cfg: SplitterConfig
) -> ParentChildResult:
    """Perform two-level chunking:

    1. Split text into large parent chunks (parent_cfg).
    2. Split each parent into smaller child chunks (child_cfg) for embedding.

    The child Seq is globally unique across the entire document.

    Args:
        text: Text to split
        parent_cfg: Configuration for parent chunks
        child_cfg: Configuration for child chunks

    Returns:
        ParentChildResult containing parents and children
    """
    parents = split_text(text, parent_cfg)
    if not parents:
        return ParentChildResult(parents=[], children=[])

    new_parents = []
    children = []
    child_seq = 0
    for parent in parents:
        subs = split_text(parent.content, child_cfg)

        parent_index = -1
        if len(subs) > 1 or (len(subs) == 1 and subs[0].content != parent.content):
            parent_index = len(new_parents)
            new_parents.append(parent)

        for sub in subs:
            # Adjust offsets: sub positions are relative to parent content,
            # shift to document-level offsets.
            # Use additive shift (not Content-length based) so that chunks with
            # prepended context headers keep correct positional tracking.
            # sub.seq = child_seq
            # sub.start += parent.start
            # sub.end += parent.start
            children.append(ChildChunk(
                content=sub.content,
                seq = child_seq,
                start = sub.start + parent.start,
                end = sub.end + parent.start,
                context_header=sub.context_header,
                parent_index=parent_index
            ))
            child_seq += 1

    return ParentChildResult(parents=new_parents, children=children)


def default_config() -> SplitterConfig:
    """Return sensible defaults for splitter configuration."""
    return SplitterConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=['\n\n', '\n', '。']
    )


def unwrap_linked_images(text: str) -> str:
    """Unwrap linked images from markdown text.

    This is a placeholder function that should be implemented based on the
    actual requirements of the docparser.UnwrapLinkedImages function.
    """
    # This is a simplified implementation - the actual implementation
    # would depend on the specific requirements of the original function
    # which is imported from another package.
    return text


@dataclass
class Chunk:
    """Represents a piece of split text with position tracking.

    Content holds exactly the text from the original document between Start
    and End (rune offsets), so End-Start == len(Content) in runes.
    This invariant is relied on by document-reconstruction code paths
    (knowledge.go:2278+ for summary generation, UI highlighting, etc.).

    ContextHeader is a separately-tracked context string (e.g. a Markdown
    heading breadcrumb) that should be prepended at embedding/retrieval time
    but is NOT part of Content. Keeping the two apart preserves the
    position invariant while still letting embedding pipelines see the
    section context.
    """
    content: str
    context_header: str = ''
    seq: int = 0
    start: int = 0
    end: int = 0

    def embedding_content(self) -> str:
        """Return the text that should be fed to the embedding model.

        The ContextHeader is prepended (when set) plus the chunk content.
        Use this where Content alone would lose semantic context (Tier-1 chunks).

        Content is returned verbatim from the source document (the End-Start
        rune-count invariant requires that), but for embedding we trim the
        surrounding whitespace so leading/trailing newlines from boundary slices
        don't dilute the embedded vector or waste tokens. Inner whitespace is
        preserved.
        """
        body = self.content.strip()
        if not self.context_header:
            return body
        return f'{self.context_header}\n\n{body}'

@dataclass
class ImageRef:
    """An image reference found within a chunk's content."""
    original_ref: str
    alt_text: str
    start: int  # offset within the chunk content
    end: int

@dataclass
class SplitUnit:
    """A piece of text with its original position."""
    text: str
    start: int
    end: int

@dataclass
class SplitterConfig:
    """Configuration for the text splitter.

    Strategy and TokenLimit are honored by the strategy entry point in strategy.py;
    the legacy split_text path uses only ChunkSize/Overlap/Separators.
    """
    chunk_size: int = 512
    chunk_overlap: int = 80
    separators: List[str] = field(default_factory=lambda: ['\n\n', '\n', '。'])
    strategy: str = ''  # Empty = legacy (backwards-compatible)
    token_limit: int = 0  # 0 = use ChunkSize chars
    languages: List[str] = field(default_factory=list)  # Empty = auto-detect

@dataclass
class ChildChunk(Chunk):
    """A child chunk with a reference to its parent."""
    parent_index: int = -1  # index into ParentChildResult.Parents

@dataclass
class ParentChildResult:
    """Holds the two-level chunking output.

    Parent chunks provide context (large window), child chunks are used for
    embedding/retrieval (small window). Each child carries its ParentIndex so
    the caller can wire up ParentChunkID after DB insertion.
    """
    parents: List[Chunk]
    children: List[ChildChunk]


# Default chunk sizing constants
# Single source of truth for the entire chunker package
# DEFAULT_CHUNK_SIZE = 512 chars: ~100–130 English tokens / ~300 Chinese tokens.
# Validated as a strong baseline by the Vecta Feb-2026 benchmark across 50 academic papers.
# Use 200–400 for FAQ-style atomic content, 1000–2000 for narrative/argumentative documents.
#
# DefaultChunkOverlap = 80 chars (≈15% of DEFAULT_CHUNK_SIZE): community-
# recommended sweet spot between recall (an answer split across a
# boundary needs overlap to be retrievable) and storage cost.
# Use 0 for strictly atomic data (FAQ, JSON records), 150–200 for long narratives
# where reasoning crosses chunks.
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80