import unittest
from typing import Tuple, Optional

from heading_hierarchy import HeadingHierarchy

class TestHeadingHierarchy(unittest.TestCase):

    def test_linear_nesting(self):
        h = HeadingHierarchy()
        h.observe("# Chapter 1")
        h.observe("## Section 1.1")
        h.observe("### Subsection 1.1.1")
        self.assertEqual(h.breadcrumb(), "Chapter 1 > Section 1.1 > Subsection 1.1.1")

    def test_pops_deeper_on_sibling_heading(self):
        h = HeadingHierarchy()
        h.observe("# Chapter 1")
        h.observe("## Section 1.1")
        h.observe("### Subsection 1.1.1")
        h.observe("## Section 1.2")  # pops the H3
        self.assertEqual(h.breadcrumb(), "Chapter 1 > Section 1.2")

    def test_pops_all_on_new_top_level(self):
        h = HeadingHierarchy()
        h.observe("# Chapter 1")
        h.observe("## Section A")
        h.observe("### Sub")
        h.observe("# Chapter 2")
        self.assertEqual(h.breadcrumb(), "Chapter 2")
        self.assertEqual(h.depth(), 1)

    def test_non_heading_ignored(self):
        h = HeadingHierarchy()
        h.observe("# Title")
        level, txt = h.observe("just a paragraph")
        self.assertEqual(level, 0)
        self.assertEqual(txt, "")
        self.assertEqual(h.breadcrumb(), "Title")

    def test_breadcrumb_with_hashes(self):
        h = HeadingHierarchy()
        h.observe("# A")
        h.observe("## B")
        got = h.breadcrumb_with_hashes()
        want = "# A\n## B"
        self.assertEqual(got, want)

    def test_empty_state(self):
        h = HeadingHierarchy()
        self.assertEqual(h.breadcrumb(), "")
        self.assertEqual(h.breadcrumb_with_hashes(), "")
        self.assertEqual(h.depth(), 0)

    def test_reset(self):
        h = HeadingHierarchy()
        h.observe("# A")
        h.observe("## B")
        h.reset()
        self.assertEqual(h.breadcrumb(), "")
        self.assertEqual(h.depth(), 0)

    def test_skip_levels(self):
        h = HeadingHierarchy()
        # Document jumps from H1 directly to H3, skipping H2.
        h.observe("# Top")
        h.observe("### Deep")
        self.assertEqual(h.breadcrumb(), "Top > Deep")

if __name__ == '__main__':
    unittest.main()