import unittest
import re
from typing import List, Dict, Any
from splitter import Chunk, SplitterConfig
from profiler import (
    DocProfile,
    select_strategy,
    profile_document
)

class TestProfileDocument(unittest.TestCase):

    def test_empty(self):
        p = profile_document("")
        self.assertEqual(p.total_chars, 0, f"empty doc should have zero stats, got {p}")
        self.assertEqual(p.total_lines, 0, f"empty doc should have zero stats, got {p}")

    def test_markdown_headings(self):
        doc = """# Title
Some intro text here.

## Section 1
Body of section 1.

## Section 2
Body of section 2.

### Subsection 2.1
Detail.

## Section 3
More body."""
        p = profile_document(doc)

        self.assertEqual(p.md_heading_counts[1], 1, f"expected 1 H1, got {p.md_heading_counts[1]}")
        self.assertEqual(p.md_heading_counts[2], 3, f"expected 3 H2, got {p.md_heading_counts[2]}")
        self.assertEqual(p.md_heading_counts[3], 1, f"expected 1 H3, got {p.md_heading_counts[3]}")
        self.assertEqual(p.md_heading_total, 5, f"expected 5 headings total, got {p.md_heading_total}")
        self.assertEqual(p.dominant_heading_level(), 2, f"dominant level should be 2 (≥3 occurrences), got {p.dominant_heading_level()}")

    def test_dominant_level_fallback(self):
        # No level reaches 3 occurrences — should fall back to most frequent.
        doc = "# Single H1\n## H2 a\n## H2 b"
        p = profile_document(doc)
        self.assertEqual(p.dominant_heading_level(), 2, f"expected fallback to level 2 (most frequent), got {p.dominant_heading_level()}")

    def test_numbered_sections(self):
        doc = """1. Introduction
text

2. Methodology
text

3. Results
text"""
        p = profile_document(doc)
        self.assertGreaterEqual(p.numbered_section_count, 3, f"expected ≥3 numbered sections, got {p.numbered_section_count}")

    def test_german_chapters(self):
        doc = "Kapitel 1: Einführung\n\nText\n\nKapitel 2: Hauptteil\n\nText"
        p = profile_document(doc)
        self.assertEqual(p.german_chapter_count, 2, f"expected 2 German chapters, got {p.german_chapter_count}")

    def test_chinese_chapters(self):
        doc = "第一章 引言\n\n内容\n\n第二章 方法\n\n内容"
        p = profile_document(doc)
        self.assertEqual(p.chinese_chapter_count, 2, f"expected 2 Chinese chapters, got {p.chinese_chapter_count}")

    def test_form_feed(self):
        doc = "page 1 content\f\npage 2 content\f\npage 3 content"
        p = profile_document(doc)
        self.assertEqual(p.form_feed_count, 2, f"expected 2 form feeds, got {p.form_feed_count}")

    def test_detects_code_block(self):
        doc = "Some prose.\n\n```go\nfunc main() {\n```\n\nMore prose."
        p = profile_document(doc)
        self.assertTrue(p.has_code, "expected has_code=true for fenced block")

    def test_detects_table(self):
        doc = "Intro.\n\n| col a | col b |\n| --- | --- |\n| 1 | 2 |\n"
        p = profile_document(doc)
        self.assertTrue(p.has_tables, "expected has_tables=true")

    def test_line_statistics(self):
        doc = "short\nthis is a longer line of text\nanother line here"
        p = profile_document(doc)
        self.assertEqual(p.total_lines, 3, f"expected 3 lines, got {p.total_lines}")
        self.assertGreater(p.avg_line_len, 0, "expected positive avg line len")

class TestSelectStrategy(unittest.TestCase):

    def test_heading_doc(self):
        doc = "# A\nbody\n## B\nbody\n## C\nbody\n## D\nbody"
        p = profile_document(doc)
        chain = select_strategy(p)
        self.assertEqual(chain[0], "heading", f"expected heading tier first, got {chain}")

    def test_heuristic_doc(self):
        doc = ("Kapitel 1: Foo\nbody body body\n\n" * 1) + \
              ("Kapitel 2: Bar\nbody body body\n\n" * 1)
        p = profile_document(doc)
        chain = select_strategy(p)
        # no markdown headings → heuristic must come first (heading tier skipped)
        self.assertEqual(chain[0], "heuristic", f"expected heuristic tier first, got {chain}")

    def test_plain_doc(self):
        doc = "just a paragraph of plain text without any structure indicators at all here"
        p = profile_document(doc)
        chain = select_strategy(p)
        self.assertEqual(chain[0], "legacy", f"expected legacy tier first for unstructured doc, got {chain}")

    def test_always_falls_back_to_legacy(self):
        for doc in ["", "simple", "# H1\nbody"]:
            p = profile_document(doc)
            chain = select_strategy(p)
            self.assertEqual(chain[-1], "legacy", f"chain must end with legacy, got {chain} for doc={repr(doc)}")

if __name__ == '__main__':
    unittest.main()