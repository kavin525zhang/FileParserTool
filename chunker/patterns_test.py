import unittest
import re
from typing import List, Dict, Any
from patterns import (
    MarkdownHeadingPattern,
    NumberedSectionPattern,
    GermanChapterPattern,
    EnglishChapterPattern,
    ChineseChapterPattern,
    VisualSeparatorPattern,
    PageFooterPattern,
    AllCapsHeadingPattern,
    sentence_separators,
    chapter_patterns_for_langs
)

class TestMarkdownHeadingPattern(unittest.TestCase):

    def test_basic_levels(self):
        cases = [
            {"in": "# Heading 1", "match": True},
            {"in": "## Heading 2", "match": True},
            {"in": "###### Heading 6", "match": True},
            {"in": "####### Too many", "match": False},
            {"in": "#NoSpace", "match": False},
            {"in": "  # Indented", "match": False},
            {"in": "plain text", "match": False},
        ]

        for c in cases:
            got = bool(MarkdownHeadingPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"MarkdownHeadingPattern({c['in']}): got {got} want {c['match']}")


class TestNumberedSectionPattern(unittest.TestCase):

    def test_numbered_section_pattern(self):
        cases = [
            {"in": "1. Introduction", "match": True},
            {"in": "2.3 Methodology", "match": True},
            {"in": "2.3. Methodology", "match": True},
            {"in": "2.2.1 用户与权限", "match": True},
            {"in": "3.2.1 单机 Docker Compose", "match": True},
            {"in": "IV. Results", "match": True},
            {"in": "1.Introduction", "match": False},
            {"in": "1.", "match": False},
            {"in": "1.1", "match": False},
            {"in": "1 NoDotSingleLevel", "match": False},
            {"in": "1.2.3.4.5 TooDeep", "match": False},
            {"in": "plain text", "match": False},
        ]

        for c in cases:
            got = bool(NumberedSectionPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"NumberedSectionPattern({c['in']}): got {got} want {c['match']}")


class TestGermanChapterPattern(unittest.TestCase):

    def test_german_chapter_pattern(self):
        cases = [
            {"in": "Kapitel 1: Einführung", "match": True},
            {"in": "Abschnitt 2.3 Methodik", "match": True},
            {"in": "Abschnitt 3 Methodik", "match": True},
            {"in": "Teil II Ergebnisse", "match": True},
            {"in": "chapter 1", "match": False},
        ]

        for c in cases:
            got = bool(GermanChapterPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"GermanChapterPattern({c['in']}): got {got} want {c['match']}")


class TestEnglishChapterPattern(unittest.TestCase):

    def test_english_chapter_pattern(self):
        cases = [
            {"in": "Chapter 1: Intro", "match": True},
            {"in": "Section 5 Methods", "match": True},
            {"in": "Part IV Results", "match": True},
            {"in": "Kapitel 1", "match": False},
        ]

        for c in cases:
            got = bool(EnglishChapterPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"EnglishChapterPattern({c['in']}): got {got} want {c['match']}")


class TestChineseChapterPattern(unittest.TestCase):

    def test_chinese_chapter_pattern(self):
        cases = [
            {"in": "第一章 引言", "match": True},
            {"in": "第3节 方法论", "match": True},
            {"in": "第二部分 结果", "match": True},
            {"in": "第 1 章 引言", "match": True},
            {"in": "第 一 章 引言", "match": True},
            {"in": "第1章 引言", "match": True},
            {"in": "Chapter 1", "match": False},
            {"in": "第 章 空数字", "match": False},
        ]

        # Define the regex pattern
        pattern = re.compile(r'^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$')

        for c in cases:
            got = bool(ChineseChapterPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"ChineseChapterPattern({c['in']}): got {got} want {c['match']}")


class TestVisualSeparatorPattern(unittest.TestCase):

    def test_visual_separator_pattern(self):
        cases = [
            {"in": "---", "match": True},
            {"in": "========", "match": True},
            {"in": "***", "match": True},
            {"in": "____", "match": True},
            {"in": "--", "match": False},
            {"in": "-- text", "match": False},
        ]

        for c in cases:
            got = bool(VisualSeparatorPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"VisualSeparatorPattern({c['in']}): got {got} want {c['match']}")


class TestPageFooterPattern(unittest.TestCase):

    def test_page_footer_pattern(self):
        cases = [
            {"in": "Seite 3 von 24", "match": True},
            {"in": "Page 5 of 12", "match": True},
            {"in": "page 7", "match": True},
            {"in": "Seite 9", "match": True},
            {"in": "页 3", "match": True},
            {"in": "页码 3 / 12", "match": True},
            {"in": "Some text", "match": False},
        ]

        # Define the regex pattern
        pattern = re.compile(r'^[ \t]*(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?[ \t]*$')

        for c in cases:
            got = bool(PageFooterPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"PageFooterPattern({c['in']}): got {got} want {c['match']}")


class TestAllCapsHeadingPattern(unittest.TestCase):

    def test_all_caps_heading_pattern(self):
        cases = [
            {"in": "INTRODUCTION", "match": True},
            {"in": "METHODS AND MATERIALS", "match": True},
            {"in": "Mixed Case Heading", "match": False},
            {"in": "ABC", "match": False},
        ]

        for c in cases:
            got = bool(AllCapsHeadingPattern.match(c["in"]))
            self.assertEqual(got, c["match"],
                           f"AllCapsHeadingPattern({c['in']}): got {got} want {c['match']}")

class TestSentenceSeparators(unittest.TestCase):

    def test_sentence_separators(self):
        got = sentence_separators("chinese")
        self.assertEqual(got[0], "。",
                       f"Chinese should start with 。, got {got}")

        got = sentence_separators("english")
        self.assertEqual(got[0], ". ",
                       f"English should start with '. ', got {got}")

        got = sentence_separators("xx")
        self.assertGreaterEqual(len(got), 5,
                              f"Unknown lang should return mixed (>=5 separators), got {got}")

class TestChapterPatternsForLangs(unittest.TestCase):

    def test_chapter_patterns_for_langs(self):
        got = chapter_patterns_for_langs(None)
        self.assertEqual(len(got), 3,
                       f"nil langs should return all 3, got {len(got)}")

        got = chapter_patterns_for_langs(["german"])
        self.assertEqual(len(got), 1,
                       f"only DE requested should return 1, got {len(got)}")

        got = chapter_patterns_for_langs(["german", "chinese"])
        self.assertEqual(len(got), 2,
                       f"DE+ZH requested should return 2, got {len(got)}")

        got = chapter_patterns_for_langs(["xx"])
        self.assertEqual(len(got), 3,
                       f"unknown lang should fall back to all 3, got {len(got)}")

if __name__ == '__main__':
    unittest.main()