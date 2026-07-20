import unittest
import re
from typing import List, Dict, Any
from heading_splitter import split_by_headings_impl, common_heading_prefix
from splitter import Chunk, SplitterConfig


class TestSplitByHeadings(unittest.TestCase):

    def test_basic_sections(self):
        # Each section is intentionally larger than the merge-target (≈
        # ChunkSize/2) so the post-split coalesce pass leaves them as distinct
        # chunks. We're testing per-section emission + breadcrumb here, not
        # merging.
        body = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 4
        doc = "# Top\n" + body + "\n\n## Section A\n" + body + "\n\n### Section 1\n" + body + "\n\n## Section B\n" + body + "\n\n## Section C\n" + body
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 3,
                               f"expected ≥3 chunks (one per section), got {len(chunks)}")

        # Breadcrumb is delivered via ContextHeader, not Content.
        for i, c in enumerate(chunks):
            self.assertIn("# Top", c.context_header,
                         f"chunk {i} missing H1 in ContextHeader:\n{c.context_header}")
            # EmbeddingContent merges header + content for the embedder.
            self.assertIn("# Top", c.embedding_content(),
                         f"chunk {i} EmbeddingContent missing H1")

        found = False
        for c in chunks:
            if "## Section B" in c.content and "Lorem ipsum" in c.content:
                found = True

        self.assertTrue(found, "no chunk contains Section B with its body")

    def test_falls_through_for_unstructured_doc(self):
        doc = "Just a plain paragraph without any headings at all in this text."
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        # no headings → falls through to split_text, which keeps the whole thing
        self.assertEqual(len(chunks), 1,
                        f"expected fallthrough single chunk, got {len(chunks)}")

    def test_large_section_recurses_into_legacy(self):
        body = "This is a long sentence repeated many times. " * 50
        doc = "# Top\n## Big\n" + body
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=30, separators=[". "])
        chunks = split_by_headings_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 2,
                               f"large section should be sub-split, got {len(chunks)} chunks")

        # Every sub-chunk should carry the breadcrumb via ContextHeader.
        for i, c in enumerate(chunks):
            self.assertIn("# Top", c.context_header,
                         f"sub-chunk {i} missing H1 in context_header")

    def test_breadcrumb_reflects_latest_path(self):
        # Sized so each section stays its own chunk after the tiny-section
        # coalesce pass — we're verifying breadcrumb assignment per section,
        # not the merge behavior.
        body = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 4
        doc = "# Chapter 1\n" + body + "\n\n## Section A\n" + body + "\n\n## Section B\n" + body
        cfg = SplitterConfig(chunk_size=300, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 3,
                               f"expected ≥3 chunks, got {len(chunks)}")

        for c in chunks:
            if "text B" in c.content:
                self.assertNotIn("## Section A", c.context_header,
                               f"Section B chunk should not include Section A in breadcrumb:\n{c.context_header}")
                self.assertIn("## Section B", c.context_header,
                            f"Section B chunk should include its own heading in breadcrumb:\n{c.context_header}")

    def test_ignores_headings_inside_code_fence(self):
        doc = '''# Real\n\n
        # Fake heading inside code\n
        \nbody'''
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        found = False
        for c in chunks:
            if "# Real" in c.context_header or "# Real" in c.content:
                found = True
                break

        self.assertTrue(found, "expected real H1 breadcrumb on some chunk")

    def test_preserves_position_relative_to_original(self):
        doc = "# Top\nintro\n\n## A\nbody A\n\n## B\nbody B"
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        for i, c in enumerate(chunks):
            self.assertGreaterEqual(c.start, 0, f"chunk {i} has negative start")
            self.assertGreaterEqual(c.end, c.start, f"chunk {i} end < start")

    def test_position_invariant(self):
        # TestSplitByHeadings_PositionInvariant ensures End-Start == len(Content)
        # and runes[Start:End] == Content for every emitted chunk. This invariant
        # is required by knowledge.go:2278+ document reconstruction logic.
        doc = '''# Top
intro paragraph here.

## Section A
content of A here, several sentences.

## Section B
content of B here.

## Section C
content of C here.'''
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20)
        chunks = split_by_headings_impl(doc, cfg)

        self.assertGreater(len(chunks), 0, "expected chunks")

        for i, c in enumerate(chunks):
            content_rune_len = len(c.content)
            span = c.end - c.start

            self.assertEqual(span, content_rune_len,
                           f"chunk {i}: span({span}) != content_runes({content_rune_len})\nContent:\n{c.content}")

            if c.start >= 0 and c.end <= len(doc):
                self.assertEqual(doc[c.start:c.end], c.content,
                               f"chunk {i}: runes[Start:End] != Content")

    def test_coalesces_tiny_adjacent_sections(self):
        # TestSplitByHeadings_CoalescesTinyAdjacentSections 涵盖了 FAQ 或
        # install-log 这类场景，即父级标题下包含许多简短的子章节。 
        # 若不进行合并，每个 `##` 标题都会形成一个长度不足 50 字符的独立片段，
        # 导致验证器因“微小片段过多”而拒绝该层级；合并后，
        # 它们被整合为少量大小适中的片段，同时仍能保留共享的父级面包屑导航信息。
        doc = '''# Install Log

## Docker镜像
使用 daocloud 部署 v0.3.1。

## 前端老版本
浏览器缓存了旧前端资源。

## 登录报错
ERROR: column missing.

## 解析失败
embedding 表缺列。'''
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)
        self.assertGreater(len(chunks), 0, "expected at least one chunk")
        self.assertLess(len(chunks), 5,
                       f"expected coalesce to produce <5 chunks, got {len(chunks)}")

        # Every merged chunk must carry the shared parent in its breadcrumb so
        # retrieval can still answer "what document is this from".
        for i, c in enumerate(chunks):
            self.assertIn("# Install Log", c.context_header,
                         f"chunk {i} missing parent H1 in breadcrumb: {c.context_header}")

        # All four sub-section headings must remain visible somewhere in the
        # merged content (heading_splitter keeps the heading line as part of
        # each section's Content).
        for h in ["## Docker镜像", "## 前端老版本", "## 登录报错", "## 解析失败"]:
            seen = False
            for c in chunks:
                if h in c.content:
                    seen = True
                    break

            self.assertTrue(seen, f"merged chunks should still contain heading {h} somewhere")

    def test_does_not_coalesce_distinct_top_level_headings(self):
        doc = '''# Intro
short intro.

# Usage
short usage.

# FAQ
short faq.'''
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        self.assertEqual(len(chunks), 3,
                        f"expected one chunk per top-level heading, got {len(chunks)}:\n{chunks}")

        for i, heading in enumerate(["# Intro", "# Usage", "# FAQ"]):
            self.assertIn(heading, chunks[i].content,
                         f"chunk {i} should contain heading {heading}, got:\n{chunks[i].content}")

    def test_coalesce_preserves_position_invariant(self):
        # TestSplitByHeadings_CoalescePreservesPositionInvariant guards the
        # End-Start == len([]rune(Content)) invariant after merging. Adjacent
        # chunks (cur.End == next.Start) must concatenate cleanly; the merge must
        # refuse to combine non-adjacent chunks (e.g. legacy sub-chunks from an
        # oversized section that overlap).
        doc = '''# Top

## A
short A.

## B
short B.

## C
short C.'''
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        for i, c in enumerate(chunks):
            content_rune_len = len(c.content)
            if c.end - c.start != content_rune_len:
                self.assertEqual(c.end - c.start, content_rune_len,
                               f"chunk {i}: End-Start({c.end-c.start}) != content_runes({content_rune_len}) after merge")

            if c.start >= 0 and c.end <= len(doc):
                self.assertEqual(doc[c.start:c.end], c.content,
                               f"chunk {i}: source[Start:End] != Content after merge")

    def test_coalesce_respects_chunk_size(self):
        # TestSplitByHeadings_CoalesceRespectsChunkSize ensures the merge target
        # stays within the ChunkSize budget — the validator caps oversize chunks
        # at 2x and we should never approach that line via merging.
        sections = 30
        sb = "# Doc\n"
        for i in range(sections):
            sb += f"\n## Section {i}\nshort body line.\n"

        cfg = SplitterConfig(chunk_size=200, chunk_overlap=0)
        chunks = split_by_headings_impl(sb, cfg)

        for i, c in enumerate(chunks):
            content_len = len(c.content)
            if content_len > cfg.chunk_size:
                self.assertLessEqual(content_len, cfg.chunk_size,
                                   f"chunk {i} exceeds ChunkSize: {content_len > {cfg.chunk_size}}")

    def test_common_heading_prefix(self):
        # TestCommonHeadingPrefix exercises the breadcrumb-prefix helper directly.
        test_cases = [
            ("# Top\n## A", "# Top\n## B", "# Top"),
            ("# Top", "# Top", "# Top"),
            ("# X", "# Y", ""),
            ("# Top\n## A\n### x", "# Top\n## A\n### y", "# Top\n## A"),
            ("", "# Top", "")
        ]

        for a, b, expected in test_cases:
            result = common_heading_prefix(a, b)
            self.assertEqual(result, expected,
                           f"commonHeadingPrefix({a!r}, {b!r}) = {result!r}, want {expected!r}")

    def test_no_breadcrumb_duplication(self):
        # TestSplitByHeadings_NoBreadcrumbDuplication ensures the section's own
        # heading line does not appear twice in the chunk content (once as part of
        # the breadcrumb, once as the section's first line).
        doc = '''# Chapter 1
intro.

## Section A
body A.

## Section B
body B.'''
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=0)
        chunks = split_by_headings_impl(doc, cfg)

        for i, c in enumerate(chunks):
            # Count occurrences of "## Section A" / "## Section B"
            for heading in ["## Section A", "## Section B"]:
                count = c.content.count(heading)
                if count > 1:
                    self.assertLess(count, 2,
                                 f"chunk {i} contains {heading} count times — duplicated by breadcrumb prepend:\n{c.Content}")

    def test_deep_sub_heading_in_large_section(self):
        # TestSplitByHeadings_DeepSubHeadingInLargeSection covers issue #1674: when a
        # shallow heading level repeats and therefore becomes the dominant split
        # backbone, a long section split into sub-chunks must still carry the deep
        # `###`/`####` heading that defines a sub-chunk's meaning, not just the
        # section-level header. The marker line sits far below its deep heading, in a
        # later sub-chunk that has no heading of its own.
        filler = "clause body sentence that pads the section out. " * 20
        doc = "# Standard XYZ\n" +\
            "## Preface\n" + filler + "\n\n" +\
            "## 5 Classification\n" + filler + "\n\n" +\
            "### 5.9 Grade Nine\n" + filler + "\n\n" +\
            "#### 5.9.2 Clause Series\n" + filler + "\n\n" +\
            "the item users search for is MARKER_ITEM_23 graded here.\n\n" +\
            "## Appendix A\n" + filler + "\n\n" +\
            "## Appendix B\n" + filler + "\n\n" +\
            "## Appendix C\n" + filler

        cfg = SplitterConfig(chunk_size=300, chunk_overlap=0, separators=[". "])
        chunks = split_by_headings_impl(doc, cfg)

        marker = None
        for i in range(len(chunks)):
            if "MARKER_ITEM_23" in chunks[i].content:
                marker = chunks[i]
                break

        self.assertIsNotNone(marker, "no chunk contains the marker line")

        if marker:
            self.assertIn("#### 5.9.2 Clause Series", marker.context_header,
                         f"marker chunk lost its deep sub-heading in the breadcrumb:\nContextHeader={marker.context_header}")
            self.assertIn("## 5 Classification", marker.context_header,
                         f"marker chunk lost its section heading in the breadcrumb:\nContextHeader={marker.context_header}")

if __name__ == '__main__':
    unittest.main()