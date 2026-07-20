import unittest
from chunker.tokens import (
    approx_token_count, 
    detect_language, 
    chars_for_token_limit, 
    LANG_ENGLISH, 
    LANG_GERMAN, 
    LANG_CHINESE, 
    LANG_MIXED
)

class TestApproxTokenCount(unittest.TestCase):

    def test_english(self):
        """Test English token estimation"""
        got = approx_token_count("The quick brown fox jumps over the lazy dog.", LANG_ENGLISH)
        # 44 chars / 4 ≈ 11 tokens
        self.assertGreaterEqual(got, 9)
        self.assertLessEqual(got, 13)

    def test_chinese(self):
        """Test Chinese token estimation"""
        got = approx_token_count("这是一段中文测试内容用于检验分词估算", LANG_CHINESE)
        # 18 characters / 1.7 ≈ 10
        self.assertGreaterEqual(got, 9)
        self.assertLessEqual(got, 12)

    def test_empty(self):
        """Test empty string returns 0 tokens"""
        got = approx_token_count("")
        self.assertEqual(got, 0)

    def test_unknown_lang(self):
        """Test unknown language falls back to mixed"""
        got = approx_token_count("Hello world hello world", "xx")
        self.assertGreater(got, 0)

class TestDetectLanguage(unittest.TestCase):

    def test_english(self):
        """Test English language detection"""
        got = detect_language("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(got, LANG_ENGLISH)

    def test_german(self):
        """Test German language detection with umlauts"""
        got = detect_language("Der schnelle braune Fuchs springt über den faulen Hund.")
        self.assertEqual(got, LANG_GERMAN)

    def test_german_by_stopwords(self):
        """Test German detection via stopwords without umlauts"""
        got = detect_language("Das ist ein Test und nicht mit Umlauten.")
        self.assertEqual(got, LANG_GERMAN)

    def test_chinese(self):
        """Test Chinese language detection"""
        got = detect_language("这是一段中文测试内容")
        self.assertEqual(got, LANG_CHINESE)

    def test_mixed(self):
        """Test mixed language detection"""
        got = detect_language("This 这是 mixed 测试 content with 多语言 inside")
        self.assertEqual(got, LANG_MIXED)

class TestCharsForTokenLimit(unittest.TestCase):

    def test_applies_safety_margin(self):
        """Test character limit calculation with safety margin"""
        got = chars_for_token_limit(1000, LANG_ENGLISH)
        # 1000 * 4 * 0.9 = 3600
        self.assertGreaterEqual(got, 3500)
        self.assertLessEqual(got, 3700)

    def test_zero_tokens(self):
        """Test zero tokens returns zero characters"""
        got = chars_for_token_limit(0, LANG_ENGLISH)
        self.assertEqual(got, 0)

if __name__ == '__main__':
    unittest.main()