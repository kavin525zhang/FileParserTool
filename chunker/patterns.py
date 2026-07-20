import re
from typing import List

# BoundaryPriority levels for heuristic chunk boundaries. Higher = stronger.
PrioFormFeed = 100
PrioNumberedHead = 90
PrioChapterMarker = 85
PrioAllCapsHeading = 70
PrioVisualSep = 60
PrioPageFooter = 50
PrioBlankBlock = 40

# MarkdownHeadingPattern matches an ATX-style Markdown heading at line start.
# Capture groups: (1) hashes, (2) heading text.
MarkdownHeadingPattern = re.compile(r'(?m)^(#{1,6})\s+(.+?)\s*#*\s*$')

# FormFeedPattern matches the form-feed control character used by some PDF
# converters as a page break marker.
FormFeedPattern = re.compile(r'\f')

# NumberedSectionPattern matches lines starting with numeric or roman numbering
# followed by a non-empty title, e.g. "1. Intro", "2.3 Methods", "IV. Results",
# "2.2.1 用户与权限". The trailing dot after a multi-level numeral is optional
# because many technical documents write "1.1 Foo" without a closing dot.
NumberedSectionPattern = re.compile(r'(?m)^[ \t]*(?:\d+(?:\.\d+){1,3}\.?|(?:\d+|[IVX]{1,5})\.)[ \t]+\S.{0,200}$')

# AllCapsHeadingPattern matches short all-caps lines (likely section titles
# rendered without Markdown headings). It requires at least 4 letters and
# up to ~10 words. Trailing colons are tolerated.
AllCapsHeadingPattern = re.compile(r'(?m)^[ \t]*([A-ZÄÖÜ][A-ZÄÖÜ \-]{3,80}):?\s*$')

# VisualSeparatorPattern matches horizontal rules / divider lines used as
# section separators in plain text or pre-Markdown documents.
VisualSeparatorPattern = re.compile(r'(?m)^[ \t]*(?:-{3,}|={3,}|\*{3,}|_{3,})[ \t]*$')

# ExcessiveBlanksPattern matches three or more consecutive newlines, which
# usually denote a hard section break.
ExcessiveBlanksPattern = re.compile(r'\n{3,}')

# PageFooterPattern matches typical "Seite X von Y" / "Page X of Y" lines.
PageFooterPattern = re.compile(r'(?mi)^[ \t]*(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?[ \t]*$')

# GermanChapterPattern matches German chapter / section markers.
GermanChapterPattern = re.compile(r'(?m)^[ \t]*(?:Kapitel|Abschnitt|Teil)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$')

# EnglishChapterPattern matches English chapter / section markers.
EnglishChapterPattern = re.compile(r'(?m)^[ \t]*(?:Chapter|Section|Part)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$')

# ChineseChapterPattern matches CJK chapter / section markers like 第一章,
# 第3节, 第 1 章 (whitespace between 第 / numeral / unit is tolerated).
ChineseChapterPattern = re.compile(r'(?m)^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$')

# SentenceSeparators returns sentence-level separators tuned for the language.
# Used for fine-grained sub-splitting when a section is still too large.
def sentence_separators(lang: str) -> List[str]:
    """Return sentence-level separators tuned for the language."""
    if lang == "chinese":
        return ["。", "！", "？", "；", "\n"]
    elif lang in ["german", "english"]:
        return [". ", "! ", "? ", "; ", "\n"]
    else:
        return ["。", "！", "？", "；", ". ", "! ", "? ", "; ", "\n"]

def chapter_patterns_for_langs(langs: List[str]) -> List[re.Pattern]:
    """Return the chapter-marker regexes that apply for the given language hints. An empty / unknown list returns all of them so that auto-detected documents still match."""
    if not langs:
        return [GermanChapterPattern, EnglishChapterPattern, ChineseChapterPattern]

    out = []
    for l in langs:
        if l == "german":
            out.append(GermanChapterPattern)
        elif l == "english":
            out.append(EnglishChapterPattern)
        elif l == "chinese":
            out.append(ChineseChapterPattern)

    if not out:
        out = [GermanChapterPattern, EnglishChapterPattern, ChineseChapterPattern]

    return out