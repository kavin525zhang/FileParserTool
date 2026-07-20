from typing import List
from chunker.splitter import Chunk

class ValidationResult:
    """Captures the verdict and reason for a chunk-set."""
    def __init__(self, ok: bool = False, reason: str = ""):
        self.ok = ok
        self.reason = reason


def validate_chunks(chunks: List[Chunk], total_chars: int, chunk_size: int) -> ValidationResult:
    """Validate whether the given chunks form a usable result for a document.

    Returns a ValidationResult with ok=True when no broken-output indicator triggers.
    """
    if not chunks:
        return ValidationResult(reason="no chunks produced")

    # A single chunk for a document much larger than chunk_size means the
    # strategy did not actually split — fail so the next tier runs.
    if len(chunks) == 1 and total_chars > 2 * chunk_size:
        return ValidationResult(reason="single chunk for large document")

    # Compute size statistics.
    sum_len = 0
    sum_sq = 0
    max_len = 0
    min_len = float('inf')

    for chunk in chunks:
        # In Python, len() on a string returns the number of characters (Unicode code points)
        l = len(chunk.content)
        sum_len += l
        sum_sq += l * l
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    # All but the last chunk should carry meaningful content. We allow the
    # last chunk to be tiny because tail residue is normal.
    tiny_count = 0
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1:
            continue
        if len(chunk.content) < 50:
            tiny_count += 1

    if tiny_count > len(chunks) // 4 and tiny_count > 2:
        return ValidationResult(reason="too many tiny chunks")

    # Reject when no chunk reached at least 25% of the target — the splitter
    # is fragmenting too aggressively to be useful.
    if max_len < chunk_size // 4 and total_chars > chunk_size:
        return ValidationResult(reason="all chunks far below target size")

    # Sanity check on absolute upper bound. Anything past 2x chunk_size is a
    # red flag — the splitter ignored its size budget.
    if max_len > 2 * chunk_size and chunk_size > 0:
        return ValidationResult(reason="chunk exceeds 2x target size")

    return ValidationResult(ok=True)