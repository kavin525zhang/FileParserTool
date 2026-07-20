import unittest
from unittest.mock import patch

# Assuming these are the functions and classes that would be imported from the chunker package
# In a real scenario, these would be imported from the actual module
from strategy import split
from strategy import (
    split_with_diagnostics, 
    STRATEGY_LEGACY, 
    STRATEGY_HEADING,
    STRATEGY_HEURISTIC,
    STRATEGY_RECURSIVE,
    TIER_LEGACY,
    STRATEGY_AUTO,
    TIER_HEADING
)
from splitter import (
    SplitterConfig,
    default_config
)

class TestSplitWithDiagnostics(unittest.TestCase):

    def test_split_with_diagnostics_legacy_strategy_reports_legacy_tier(self):
        # Splittable input so the validator accepts the legacy output cleanly.
        text = "Hello world.\n\nNext paragraph here.\n\n" * 50
        cfg = SplitterConfig(chunk_size=200, 
                             chunk_overlap=20, 
                             separators=["\n\n", "\n"], 
                             strategy=STRATEGY_LEGACY)
        chunks, diag = split_with_diagnostics(text, cfg)

        self.assertGreater(len(chunks), 0, "expected chunks")
        self.assertEqual(diag.selected_tier, TIER_LEGACY, f"expected SelectedTier=legacy, got {diag.selected_tier}")

        self.assertEqual(len(diag.tier_chain), 1, f"expected single-tier chain [legacy], got {diag.tier_chain}")
        self.assertEqual(diag.tier_chain[0], TIER_LEGACY, f"expected single-tier chain [legacy], got {diag.tier_chain}")

        self.assertEqual(len(diag.rejected), 0, f"expected no rejections for splittable input, got {diag.rejected}")

    def test_split_with_diagnostics_auto_on_heading_doc_picks_heading(self):
        doc = "# Top\nintro paragraph here.\n\n## Section A\nbody A here.\n\n## Section B\nbody B here.\n\n## Section C\nbody C here.\n\n" * 1
        cfg = SplitterConfig(chunk_size=300, 
                             chunk_overlap=30, 
                             strategy=STRATEGY_AUTO)
        _, diag = split_with_diagnostics(doc, cfg)

        self.assertGreater(len(diag.tier_chain), 0, "expected non-empty tier chain")

        # Heading tier should be tried first for this doc.
        self.assertEqual(diag.tier_chain[0], TIER_HEADING, f"expected heading tier first, got chain {diag.tier_chain}")

    def test_split_with_diagnostics_empty_text(self):
        chunks, diag = split_with_diagnostics("", default_config())
        self.assertEqual(chunks, [], f"expected None chunks for empty text, got {chunks}")
        self.assertIsNotNone(diag, "diag must never be None")

    def test_split_and_diagnostics_agree_on_chunks(self):
        """TestSplit_AndDiagnostics_AgreeOnChunks ensures Split (no diagnostics)
        and SplitWithDiagnostics produce the same chunk set for a given input.
        They run independent loops as of the post-audit refactor — this test
        is the regression wall against them drifting."""

        text = "para one.\n\npara two.\n\npara three."
        cfg = SplitterConfig(chunk_size=100, 
                             chunk_overlap=10)
        a = split(text, cfg)
        b, diag = split_with_diagnostics(text, cfg)

        self.assertEqual(len(a), len(b), f"chunk count disagrees: Split={len(a)} Diagnostics={len(b)}")

        for i in range(len(a)):
            self.assertEqual(a[i].content, b[i].content, f"chunk {i} content differs")
            self.assertEqual(a[i].start, b[i].start, f"chunk {i} start differs")
            self.assertEqual(a[i].end, b[i].end, f"chunk {i} end differs")

        self.assertIsNotNone(diag, "diagnostics must not be None")

    def test_split_with_diagnostics_profile_set_for_auto(self):
        """TestSplitWithDiagnostics_ProfileSetForAuto verifies that auto-strategy
        returns the DocProfile that drove tier selection — required by the
        preview endpoint to avoid double-profiling."""

        doc = "# Top\nintro.\n\n## A\nbody A.\n\n## B\nbody B."
        _, diag = split_with_diagnostics(doc, SplitterConfig(chunk_size=200, 
                                                             strategy=STRATEGY_AUTO))

        self.assertIsNotNone(diag.profile, "auto strategy must populate diag.profile")

        self.assertGreater(diag.profile.md_heading_total, 0, f"profile should have detected headings, got {diag.profile}")

    def test_split_with_diagnostics_profile_nil_for_explicit(self):
        """TestSplitWithDiagnostics_ProfileNilForExplicit verifies the inverse:
        explicit strategies bypass profiling and leave Profile None so the
        preview handler knows to materialize one if it needs stats."""

        strategies = [STRATEGY_HEADING, STRATEGY_HEURISTIC, STRATEGY_RECURSIVE, STRATEGY_LEGACY]

        for strat in strategies:
            with self.subTest(strategy=strat):
                _, diag = split_with_diagnostics("plain text", SplitterConfig(chunk_size=200, 
                                                                              strategy=strat))

                self.assertIsNone(diag.profile, f"strategy '{strat}' should leave Profile None, got {diag.profile}")

if __name__ == '__main__':
    unittest.main()