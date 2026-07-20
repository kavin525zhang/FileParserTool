import unittest
import re
from typing import List, Dict, Any
from splitter import (
    Chunk, 
    SplitterConfig, 
    default_config,
    protected_spans,
    protected_spans_rune
)
from profiler import DocProfile
from heuristic_splitter import (
    split_by_heuristics_impl,
    find_heuristic_boundaries,
    drop_bounds_inside_spans
)

class TestSplitByHeuristics(unittest.TestCase):

    def test_form_feed_boundary(self):
        doc = "page one body text. " * 30 + "\f" + "page two body. " * 30
        cfg = SplitterConfig(chunk_size=400, chunk_overlap=20, separators=[". "])
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 2,
                               f"form feed should produce ≥2 chunks, got {len(chunks)}")

    def test_numbered_sections(self):
        body = "body sentence. " * 8
        doc = "1. Introduction\n" + body + "\n\n2. Methods\n" + body + "\n\n3. Results\n" + body
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, separators=[". "])
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 2,
                               f"numbered sections should split: got {len(chunks)} chunks")

    def test_german_chapter_markers(self):
        body = "Beispieltext. " * 10
        doc = "Kapitel 1: Einführung\n" + body + "\n\nKapitel 2: Hauptteil\n" + body
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, separators=[". "])
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 2,
                                f"German chapter markers should split: got {len(chunks)}")

    def test_chinese_chapter_markers(self):
        body = "内容内容内容。" * 60
        doc = "第一章 引言\n" + body + "\n\n第二章 方法\n" + body
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, separators=["。"])
        cfg.Languages = ["chinese"]
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 2,
                               f"Chinese chapter markers should split: got {len(chunks)}")

    def test_falls_through_for_unstructured_doc(self):
        doc = "plain prose without structure. " * 5
        cfg = SplitterConfig(chunk_size=1000, chunk_overlap=20)
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertEqual(len(chunks), 1,
                        f"unstructured short doc should be one chunk, got {len(chunks)}")

    def test_oversize_block_recurses_into_legacy(self):
        huge = "This is a long sentence. " * 200  # ~5000 chars
        doc = "1. Intro\n" + huge
        cfg = SplitterConfig(chunk_size=500, chunk_overlap=50, separators=[". "])
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 5,
                               f"oversize block should produce many sub-chunks, got {len(chunks)}")

        # No single chunk should massively exceed the budget.
        for i, c in enumerate(chunks):
            content_len = len(c.Content)
            if content_len > 2 * cfg.ChunkSize:
                self.assertLessEqual(content_len, 2 * cfg.ChunkSize,
                                   f"chunk {i} exceeds 2x size: {content_len} runes")

    def test_boundaries_are_ordered(self):
        doc = "Kapitel 1: A\nbody\n\n---\n\n2. Section B\nbody\n\nPage 3 of 10\n\n第三章 C\nbody"
        bounds = find_heuristic_boundaries(doc)

        self.assertGreaterEqual(len(bounds), 2,
                               f"expected multiple boundaries, got {len(bounds)}")

        # Check that boundaries are sorted by position
        for i in range(1, len(bounds)):
            self.assertGreaterEqual(bounds[i]["rune_start"], bounds[i-1]["rune_start"],
                                  f"bounds not sorted: {bounds[i]["rune_start"]} before {bounds[i-1]["rune_start"]}")

    def test_empty_text(self):
        result = split_by_heuristics_impl("", default_config())
        self.assertIsNone(result, f"empty doc should be None, got {result}")

    def test_overlap_actually_overlaps(self):
        # Build many small numbered sections so the bin-packer flushes mid-doc
        # with at least one earlier boundary inside the overlap window.
        sb = ""
        for i in range(1, 13):
            sb += "\n\n"
            sb += str(i % 10)
            sb += ". "
            sb += "alpha beta gamma. " * 4  # ~72 chars / section

        doc = sb
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=80, separators=[". "])
        chunks = split_by_heuristics_impl(doc, cfg)

        self.assertGreaterEqual(len(chunks), 2,
                               f"need >=2 chunks to test overlap, got {len(chunks)}")

        # At least one consecutive chunk pair must share a non-trivial suffix /
        # prefix. We don't require *every* pair to overlap (oversize blocks
        # short-circuit through legacy and reset chunkStart), but at least one
        # regular flush boundary should produce real overlap.
        saw = False
        for i in range(1, len(chunks)):
            prev = chunks[i-1].Content.strip()
            cur = chunks[i].Content.strip()
            # Walk back the longest suffix of prev that prefixes cur.
            match = 0
            max_scan = min(len(prev), len(cur))
            for n in range(1, max_scan + 1):
                if cur.startswith(prev[-n:]):
                    match = n

            if match >= 20:
                saw = True
                break

        self.assertTrue(saw,
                       f"expected at least one chunk pair to overlap by >=20 chars, none did. chunk sizes: {self.chunk_lengths(chunks)}")

    def test_drops_boundaries_inside_protected_spans(self):
        body = "filler. " * 30
        # LaTeX block whose middle line matches NumberedSectionPattern. The
        # filter should drop that boundary so the math block stays intact.
        doc = body + "\n\n$$\nx = 1\n1. equation step one\ny = 2\n$$\n\n" + body

        bounds = find_heuristic_boundaries(doc)
        prot = protected_spans_rune(doc, protected_spans(doc))

        self.assertGreater(len(prot), 0, "expected protected spans for doc, got none")

        filtered = drop_bounds_inside_spans(bounds, prot)

        # Check that no boundary falls inside a protected span
        for b in filtered:
            for s in prot:
                self.assertFalse(s["start"] < b["rune_start"] < s["end"],
                               f"boundary {b["rune_start"]} still inside protected span [{s["start"]},{s["end"]})")

        # And it should actually have removed at least one boundary.
        self.assertLess(len(filtered), len(bounds),
                       f"filter removed nothing: before={len(bounds)} after={len(filtered)}")

    def chunk_lengths(self, chunks: List[Chunk]) -> List[int]:
        return [len(c.Content) for c in chunks]

if __name__ == '__main__':
    unittest.main()