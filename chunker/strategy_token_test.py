import unittest
from splitter import SplitterConfig
from strategy import ensure_defaults
from tokens import LANG_ENGLISH, LANG_CHINESE

class TestEnsureDefaults(unittest.TestCase):

    def test_token_limit_clamps_chunk_size(self):
        cfg = SplitterConfig(chunk_size=10000, 
                             token_limit=100, 
                             languages=[LANG_ENGLISH])
        out = ensure_defaults(cfg)
        # 100 tokens * 4 chars/token * 0.9 ≈ 360 chars
        self.assertLess(out.chunk_size, 1000,
                       f"expected ChunkSize clamped by TokenLimit, got {out.chunk_size}")
        self.assertLess(out.chunk_overlap, out.chunk_size,
                       f"overlap should be smaller than clamped chunk size: overlap={out.chunk_overlap} size={out.chunk_size}")

    def test_token_limit_chinese_tighter(self):
        cfg_en = ensure_defaults(SplitterConfig(token_limit=200, languages=[LANG_ENGLISH]))
        cfg_zh = ensure_defaults(SplitterConfig(token_limit=200, languages=[LANG_CHINESE]))
        self.assertLess(cfg_zh.chunk_size, cfg_en.chunk_size,
                       f"Chinese char budget should be tighter than English: zh={cfg_zh.chunk_size} en={cfg_en.chunk_size}")

    def test_no_token_limit_keeps_chunk_size(self):
        cfg = SplitterConfig(chunk_size=800)
        out = ensure_defaults(cfg)
        self.assertEqual(out.chunk_size, 800,
                        f"ChunkSize should stay 800, got {out.chunk_size}")

if __name__ == '__main__':
    unittest.main()