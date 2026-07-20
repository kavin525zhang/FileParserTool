import math
import re
from typing import Dict, List, Optional
from patterns import (
    MarkdownHeadingPattern,
    NumberedSectionPattern,
    GermanChapterPattern,
    EnglishChapterPattern,
    ChineseChapterPattern,
    AllCapsHeadingPattern,
    VisualSeparatorPattern,
    PageFooterPattern
)

from tokens import detect_language

class DocProfile:
    """Holds document-level signals used to choose a chunking strategy.

    The attributes correspond to the JSON shape (snake_case) used in the
    public preview endpoint API. Keep field names stable.
    """

    def __init__(self):
        self.total_chars: int = 0
        self.total_lines: int = 0
        self.avg_line_len: float = 0.0
        self.std_line_len: float = 0.0

        # Markdown structure
        self.md_heading_counts: Dict[int, int] = {i: 0 for i in range(1, 7)}
        self.md_heading_total: int = 0

        # Heuristic indicators
        self.numbered_section_count: int = 0
        self.all_caps_short_line_count: int = 0
        self.blank_paragraph_breaks: int = 0
        self.form_feed_count: int = 0   # 换页符
        self.visual_sep_count: int = 0
        self.german_chapter_count: int = 0
        self.english_chapter_count: int = 0
        self.chinese_chapter_count: int = 0
        self.repeated_footer_count: int = 0

        # Content characteristics
        self.has_tables: bool = False
        self.has_code: bool = False
        self.code_ratio: float = 0.0

        # Detected language hints (best-effort)
        self.detected_langs: List[str] = []

    def heading_density(self) -> float:
        """Returns the share of lines that are Markdown headings."""
        if self.total_lines == 0:
            return 0.0
        return self.md_heading_total / self.total_lines

    def dominant_heading_level(self) -> int:
        """Returns the heading level (1-6) that should drive section splitting.

        Preference order:
        1. The lowest level (closest to root) that has at least 3 occurrences -
           a "real" structural backbone of the document.
        2. Otherwise the deepest level present at least once - gives finer-grained
           boundaries for small documents that just have an H1 + a few H2s.

        Returns 0 when no Markdown headings exist.
        """
        if self.md_heading_total == 0:
            return 0

        # Look for the lowest level with at least 3 occurrences
        for level in range(1, 7):
            if self.md_heading_counts[level] >= 3:
                return level

        # Otherwise return the deepest level present
        for level in range(6, 0, -1):
            if self.md_heading_counts[level] > 0:
                return level

        return 0

    def heuristic_marker_total(self) -> int:
        """Sums the non-Markdown structural markers."""
        return (
            self.numbered_section_count +
            self.german_chapter_count +
            self.english_chapter_count +
            self.chinese_chapter_count +
            self.all_caps_short_line_count +
            self.visual_sep_count +
            self.form_feed_count
        )


def profile_document(text: str) -> DocProfile:
    """profile_document runs a single pass over text and returns its profile."""
    p = DocProfile()
    if not text:
        return p

    p.total_chars = len(text)
    p.form_feed_count = text.count("\f")

    lines = text.split('\n')
    p.total_lines = len(lines)

    # First pass: per-line markers and length stats
    lengths = []
    in_fence = False
    code_chars = 0

    for line in lines:
        trimmed = line.strip()

        # Toggle fenced-code state
        if trimmed.startswith("```"):
            in_fence = not in_fence
            p.has_code = True
            continue

        if in_fence:
            code_chars += len(line)
            continue

        lengths.append(float(len(line)))

        if match_heading(line, p.md_heading_counts):
            p.md_heading_total += 1
            continue

        if NumberedSectionPattern.match(line):
            p.numbered_section_count += 1

        if GermanChapterPattern.match(line):
            p.german_chapter_count += 1

        if EnglishChapterPattern.match(line):
            p.english_chapter_count += 1

        if ChineseChapterPattern.match(line):
            p.chinese_chapter_count += 1

        if AllCapsHeadingPattern.match(line):
            p.all_caps_short_line_count += 1

        if VisualSeparatorPattern.match(line):
            p.visual_sep_count += 1

        if re.search(r'^[ \t]*(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?[ \t]*$', line, re.MULTILINE):
            p.repeated_footer_count += 1

        if trimmed.startswith("|") and trimmed.endswith("|"):
            p.has_tables = True

    if lengths:
        avg = sum(lengths) / len(lengths)
        p.avg_line_len = avg
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        p.std_line_len = variance ** 0.5

    if p.total_chars > 0:
        p.code_ratio = code_chars / p.total_chars

    p.blank_paragraph_breaks = text.count("\n\n\n")

    # Sample a slice of the document for language detection
    sample = text
    if len(sample) > 4096:
        sample = sample[:4096]
    lang = detect_language(sample)
    p.detected_langs = [lang]
    if lang == "mixed":
        p.detected_langs = ["english", "german", "chinese"]

    return p

def profile_document(text: str) -> DocProfile:
    """Runs a single pass over text and returns its profile."""
    p = DocProfile()

    # 如果文本为空， 返回默认对象
    if not text:
        return p

    # Convert string to list of runes (Unicode code points)
    p.total_chars = len(text)
    p.form_feed_count = text.count("\f")  # 换页符

    lines = text.split("\n")
    p.total_lines = len(lines)

    # First pass: per-line markers and length stats
    lengths = []
    in_fence = False
    code_chars = 0

    for line in lines:
        trimmed = line.strip()

        # Toggle fenced-code state
        # 估计是用来合并代码模块
        if trimmed.startswith("```"):
            in_fence = not in_fence
            p.has_code = True
            continue

        if in_fence:
            code_chars += len(line)
            continue

        # Record line length for statistics
        rune_len = len(line)
        lengths.append(float(rune_len))

        # Check for Markdown headings
        if match_heading(line, p.md_heading_counts):
            p.md_heading_total += 1
            continue

        # Check for various structural patterns
        if NumberedSectionPattern.match(line):
            p.numbered_section_count += 1
        if GermanChapterPattern.match(line):
            p.german_chapter_count += 1
        if EnglishChapterPattern.match(line):
            p.english_chapter_count += 1
        if ChineseChapterPattern.match(line):
            p.chinese_chapter_count += 1
        if AllCapsHeadingPattern.match(line) and len(trimmed) < 50:
            p.all_caps_short_line_count += 1
        if VisualSeparatorPattern.match(line):
            p.visual_sep_count += 1
        if PageFooterPattern.match(line):
            p.repeated_footer_count += 1

        # Check for tables (simple detection)
        if trimmed.startswith("|") and trimmed.endswith("|"):
            p.has_tables = True

    # Calculate line length statistics
    if lengths:
        p.avg_line_len = sum(lengths) / len(lengths)
        variance = sum((l - p.avg_line_len) ** 2 for l in lengths) / len(lengths)
        p.std_line_len = math.sqrt(variance)

    # Calculate code ratio
    if p.total_chars > 0:
        p.code_ratio = code_chars / p.total_chars

    # Count blank paragraph breaks (three consecutive newlines)
    p.blank_paragraph_breaks = text.count("\n\n\n")

    # Sample for language detection
    sample = text
    if len(sample) > 4096:
        sample = sample[:4096]

    lang = detect_language(sample)
    p.detected_langs = [lang]
    if lang == "mixed":
        # Provide all three for downstream pattern selection
        p.detected_langs = ["english", "german", "chinese"]

    return p

def match_heading(line: str, counts: Dict[int, int]) -> bool:
    """Checks whether line is an ATX heading and increments the appropriate
    level counter when so. Returns True on match."""
    match = MarkdownHeadingPattern.match(line)
    if not match:
        return False

    level = len(match.group(1))
    if level < 1 or level > 6:
        return False

    counts[level] += 1
    return True

class StrategyTier:
    """Identifies which chunking implementation should run."""
    HEADING = "heading"
    HEURISTIC = "heuristic"
    LEGACY = "legacy"

def select_strategy(p: Optional[DocProfile]) -> List[str]:
    """Returns the ordered tier chain to attempt for this document.

    The first tier is the primary choice; subsequent tiers are fallbacks if
    validation rejects the previous output. The "legacy" tier is appended as
    a final safety net so callers always receive at least one chunk-set.
    """
    if p is None:
        return [StrategyTier.LEGACY]

    chain = []

    # Tier 1 candidate: Markdown heading-aware
    if (p.md_heading_total >= 3 and
        p.heading_density() > 0.005 and
        p.dominant_heading_level() > 0):
        chain.append(StrategyTier.HEADING)

    # Tier 2 candidate: heuristic boundary detection
    if (p.heuristic_marker_total() >= 5 or
        p.form_feed_count > 0 or
        p.german_chapter_count + p.english_chapter_count + p.chinese_chapter_count > 0):
        chain.append(StrategyTier.HEURISTIC)

    # Legacy is the ultimate fallback
    chain.append(StrategyTier.LEGACY)
    return chain