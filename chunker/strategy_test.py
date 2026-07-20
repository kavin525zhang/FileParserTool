import unittest
from chunker.splitter import (
    Chunk, 
    SplitterConfig, 
    split_text, 
    default_config,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from chunker.strategy import (
    split_parent_child, 
    split, 
    merge_breadcrumbs, 
    ensure_defaults,
    STRATEGY_HEADING
)
from chunker.validator import validate_chunks

class TestSplit(unittest.TestCase):

    def test_empty_text(self):
        """Test that empty text returns None"""
        result = split("", default_config())
        self.assertEqual(result, [], "empty text should return None")

    def test_legacy_strategy_matches_split_text(self):
        """Test that legacy strategy matches split_text output"""
        text = "Hello world.\n\n" * 30
        cfg = SplitterConfig(chunk_size=100, chunk_overlap=20, separators=["\n\n"], strategy="legacy")
        a = split(text, cfg)
        b = split_text(text, cfg)

        self.assertEqual(len(a), len(b),
                        f"legacy strategy should match split_text: got {len(a)} vs {len(b)} chunks")

        for i in range(len(a)):
            self.assertEqual(a[i].content, b[i].content, f"chunk {i} differs")

    def test_empty_strategy_equals_legacy(self):
        """Test that empty Strategy equals legacy strategy"""
        text = "Sentence one. Sentence two.\n" * 20
        cfg = SplitterConfig(chunk_size=80, chunk_overlap=10)
        a = split(text, cfg)
        cfg.strategy = "legacy"
        b = split(text, cfg)

        self.assertEqual(len(a), len(b),
                        f"empty Strategy should equal legacy: {len(a)} vs {len(b)}")

    def test_auto_strategy_picks_heading_for_markdown_doc(self):
        """Test that auto strategy produces chunks for markdown document"""
        doc = "# A\nbody\n## B\nbody\n## C\nbody\n## D\nbody\n" * 1
        cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, strategy="auto")
        chunks = split(doc, cfg)

        self.assertGreater(len(chunks), 0, "auto strategy should produce chunks")

    def test_heading_strategy_keeps_distinct_top_level_headings(self):
        """Test that heading strategy creates one chunk per top-level heading"""
        doc = """# Intro
short intro.

# Usage
short usage.

# FAQ
short faq."""
        cfg = SplitterConfig(chunk_size=500, 
                             chunk_overlap=0, 
                             strategy=STRATEGY_HEADING)
        chunks = split(doc, cfg)

        self.assertEqual(len(chunks), 3,
                        f"expected one chunk per top-level heading, got {len(chunks)}")

        expected_headings = ["# Intro", "# Usage", "# FAQ"]
        for i, heading in enumerate(expected_headings):
            self.assertIn(heading, chunks[i].content,
                         f"chunk {i} should contain heading '{heading}', got:\n{chunks[i].content}")

    def test_preserves_position_invariant_across_tiers(self):
        """Test that chunk positions are consistent with content"""
        test_cases = {
            "heading-tier": "# Top\nintro paragraph here.\n\n## Section A\nbody A here.\n\n## Section B\nbody B here.\n\n## Section C\nbody C.",
            "heuristic-tier": "Kapitel 1: Einleitung\n" + "Beispieltext. " * 50 + "\n\n" + "Kapitel 2: Hauptteil\n" + "Mehr Text. " * 50,
            "recursive-tier": "plain prose without structure. " * 100
        }

        cfg = SplitterConfig(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", ". "],
            strategy="auto"
        )

        for name, doc in test_cases.items():
            with self.subTest(name=name):
                runes = list(doc)
                chunks = split(doc, cfg)

                self.assertGreater(len(chunks), 0, "expected chunks")

                for i, c in enumerate(chunks):
                    content_rune_len = len(list(c.content))
                    span_len = c.end - c.start

                    self.assertEqual(span_len, content_rune_len,
                                   f"chunk {i}: End({c.end})-Start({c.start})={span_len} but Content has {content_rune_len} runes:\n{c.content}")

                    self.assertGreaterEqual(c.start, 0, f"chunk {i}: Start should be >= 0")
                    self.assertLessEqual(c.end, len(runes), f"chunk {i}: End should be <= {len(runes)}")

                    if 0 <= c.start <= c.end <= len(runes):
                        sliced = ''.join(runes[c.start:c.end])
                        self.assertEqual(sliced, c.content,
                                       f"chunk {i}: runes[Start:End] differs from Content")

    def test_split_parent_child_auto_strategy_enriches_child_breadcrumbs(self):
        """Test that auto strategy enriches child breadcrumbs with sub-headings"""
        body = "Lorem ipsum dolor sit amet. " * 40
        doc = f'''# Chapter\n" {body}
          \n\n## Section A\n {body}
          \n\n## Section B\n {body}
          \n\n## Section C\n {body}'''
        seps = ["\n\n", "\n", ". "]
        parent_cfg = SplitterConfig(chunk_size=800, chunk_overlap=80, strategy="auto", separators=seps)
        child_cfg = SplitterConfig(chunk_size=200, chunk_overlap=20, strategy="auto", separators=seps)

        #parents, children = split_parent_child(doc, parent_cfg, child_cfg)
        res = split_parent_child(doc, parent_cfg, child_cfg)

        self.assertGreater(len(res.children), 0, "expected children")

        # Check if at least one child carries a sub-heading breadcrumb
        saw_sub_heading = any("## Section" in c.context_header for c in res.children)

        if not saw_sub_heading:
            # Get first and last headers for error message
            first_hdr = res.children[0].context_header if res.children else ""
            last_hdr = res.children[-1].context_header if res.children else ""
            self.fail(f"no child carries a sub-heading breadcrumb. samples:\n  {first_hdr}\n  {last_hdr}")

        # Check that no breadcrumb contains the same line twice in a row
        for i, c in enumerate(res.children):
            lines = [line.strip() for line in c.context_header.split('\n') if line.strip()]
            for j in range(1, len(lines)):
                if lines[j] == lines[j-1]:
                    self.fail(f"child[{i}] has duplicate breadcrumb lines: {c.context_header}")

    def test_merge_breadcrumbs(self):
        """Test merge_breadcrumbs function with various cases"""
        test_cases = [
            ("", "## Sub", "## Sub", "empty parent"),
            ("# Top", "", "# Top", "empty child"),
            ("", "", "", "both empty"),
            ("# Top", "## Other", "# Top\n## Other", "disjoint"),
            ("# Top\n## A", "## A\n### A1", "# Top\n## A\n### A1", "duplicate seam"),
            ("# Top", "# Top", "# Top", "deep duplicate"),
            ("# Top\n## A", "  ## A  \n### A1", "# Top\n## A\n### A1", "only whitespace differs")
        ]

        for parent, child, expected, name in test_cases:
            with self.subTest(name=name):
                result = merge_breadcrumbs(parent, child)
                self.assertEqual(result, expected,
                               f"merge_breadcrumbs({parent!r}, {child!r}) = {result!r}, want {expected!r}")

    def test_split_parent_child_legacy_strategy(self):
        """Test split_parent_child with legacy strategy"""
        text = "This is a sentence. Another one.\n\n" * 50
        parent_cfg = SplitterConfig(chunk_size=400, chunk_overlap=40, strategy="legacy")
        child_cfg = SplitterConfig(chunk_size=100, chunk_overlap=20, strategy="legacy")

        res = split_parent_child(text, parent_cfg, child_cfg)

        self.assertGreater(len(res.children), 0, "expected children chunks")

        for i, c in enumerate(res.children):
            self.assertGreaterEqual(c.parent_index, 0, f"child[{i}] should have valid ParentIndex")
            if res.parents:  # Only check if parents exist
                self.assertLess(c.parent_index, len(res.parents),
                             f"child[{i}] has invalid ParentIndex {c.parent_index} (parents={len(res.parents)})")

    def test_ensure_defaults(self):
        """Test that ensure_defaults sets default values correctly"""
        cfg = ensure_defaults(SplitterConfig())

        self.assertEqual(cfg.chunk_size, DEFAULT_CHUNK_SIZE,
                        f"expected default chunk_size 512, got {cfg.chunk_size}")
        self.assertEqual(cfg.chunk_overlap, DEFAULT_CHUNK_OVERLAP,
                        f"expected default chunk_overlap 50, got {cfg.chunk_overlap}")
        self.assertGreater(len(cfg.separators), 0, "expected default separators")

    def test_validate_chunks_empty(self):
        """Test that validate_chunks rejects empty chunks"""
        result = validate_chunks(None, 1000, 500)
        self.assertFalse(result.ok, "nil chunks should be invalid")

    def test_validate_chunks_single_chunk_large_doc(self):
        """Test that validate_chunks rejects a single chunk that is too large"""
        chunks = [Chunk(content="a" * 5000)]
        result = validate_chunks(chunks, 5000, 500)
        self.assertFalse(result.ok, "single 10x-too-large chunk should be invalid")

    def test_validate_chunks_accepts_reasonable_output(self):
        """Test that validate_chunks accepts reasonable chunk sizes"""
        chunks = [
            Chunk(content="a" * 480),
            Chunk(content="b" * 510),
            Chunk(content="c" * 460)
        ]
        result = validate_chunks(chunks, 1500, 512)
        self.assertTrue(result.ok, f"reasonable chunks should validate, got: {result.reason}")

    def test_validate_chunks_rejects_oversized(self):
        """Test that validate_chunks rejects chunks that are too large"""
        chunks = [
            Chunk(content="a" * 100),
            Chunk(content="b" * 5000)  # > 2x chunkSize
        ]
        result = validate_chunks(chunks, 5100, 1000)
        self.assertFalse(result.ok, "chunk >2x size should be invalid")

    def test_validate_chunks_tolerant_tiny_tail(self):
        """Test that validate_chunks tolerates a small final chunk"""
        chunks = [
            Chunk(content="a" * 480),
            Chunk(content="b" * 510),
            Chunk(content="tail")
        ]
        result = validate_chunks(chunks, 994, 512)
        self.assertTrue(result.ok, f"tiny last chunk should be tolerated, got: {result.reason}")

if __name__ == '__main__':
    unittest.main()