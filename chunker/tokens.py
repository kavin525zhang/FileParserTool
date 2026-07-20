# tokens.py provides language-aware token count approximation.
#
# We avoid pulling in a tokenizer dependency (e.g. tiktoken) and instead use
# per-language chars-per-token ratios derived from common embedding model
# vocabularies. The numbers are conservative — they tend to slightly
# over-estimate token counts so that chunks stay safely under model limits.

import unicodedata

# Language identifiers used by the token estimator and the heuristic splitter.
LANG_ENGLISH = "en"
LANG_GERMAN = "de"
LANG_CHINESE = "zh"
LANG_MIXED = "mixed"

# chars_per_token holds approximate chars/token ratios per language.
# Numbers err on the conservative side so estimates over-shoot a little.
chars_per_token = {
    LANG_ENGLISH: 4.0,
    LANG_GERMAN: 4.5,
    LANG_CHINESE: 1.7,
    LANG_MIXED: 3.0,
}


def approx_token_count(s: str, lang: str = LANG_MIXED) -> int:
    """Returns a conservative token estimate for s in the given language.
    An empty or unknown lang falls back to "mixed".
    """
    if not s:
        return 0
    return approx_token_count_from_rune_len(len(s), lang)


def approx_token_count_from_rune_len(rune_len: int, lang: str = LANG_MIXED) -> int:
    """The allocation-free variant of approx_token_count when the caller has
    already computed the rune length. Use this in hot loops where the same
    content's rune count would otherwise be recomputed multiple times
    (e.g. preview endpoint emitting per-chunk stats).
    """
    if rune_len <= 0:
        return 0

    ratio = chars_per_token.get(lang, chars_per_token[LANG_MIXED])
    approx = rune_len / ratio
    if approx < 1:
        return 1
    return int(approx + 0.5)


def detect_language(s: str) -> str:
    """Returns a coarse language label by counting CJK runes vs. Latin runes.
    The result is one of LANG_CHINESE, LANG_GERMAN, LANG_ENGLISH or LANG_MIXED.
    Detection is cheap and meant only for heuristic dispatch — it is NOT a
    replacement for proper language identification.
    """
    if not s:
        return LANG_MIXED

    cjk = 0
    latin = 0
    umlaut = 0

    for r in s:
        if (unicodedata.name(r, "").startswith("CJK") or
            unicodedata.name(r, "").startswith("HANGUL") or
            unicodedata.name(r, "").startswith("HIRAGANA") or
            unicodedata.name(r, "").startswith("KATAKANA")):
            cjk += 1
        elif is_german_umlaut(r):
            umlaut += 1
            latin += 1
        elif ('a' <= r <= 'z') or ('A' <= r <= 'Z'):
            latin += 1

    total = cjk + latin
    if total == 0:
        return LANG_MIXED

    cjk_ratio = cjk / total
    latin_ratio = latin / total

    # Mixed: meaningful presence of both scripts (>=15% each).
    if cjk_ratio >= 0.15 and latin_ratio >= 0.15:
        return LANG_MIXED

    if cjk_ratio > 0.3:
        return LANG_CHINESE

    if umlaut > 0 or has_german_words(s):
        return LANG_GERMAN

    return LANG_ENGLISH


def is_german_umlaut(r: str) -> bool:
    """Check if a character is a German umlaut or sharp s."""
    return r in ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']


def has_german_words(s: str) -> bool:
    """Does a tiny stop-word check to bias towards "de" when the text uses
    common German function words. Cheap heuristic — false positives on borrowed
    terms are acceptable.
    """
    sample = 512
    if len(s) > sample:
        s = s[:sample]

    german_words = [" der ", " die ", " das ", " und ", " ist ", " nicht ", " mit ", " auf "]
    return any(contains_lower(s, w) for w in german_words)


def contains_lower(haystack: str, needle: str) -> bool:
    """Case-insensitive substring search."""
    if len(haystack) < len(needle):
        return False

    haystack_lower = haystack.lower()
    needle_lower = needle.lower()
    return needle_lower in haystack_lower


def chars_for_token_limit(tokens: int, lang: str = LANG_MIXED) -> int:
    """Converts a token limit into an approximate character budget for a given
    language. Used to size chunks so they fit within an embedding model's
    max-token window with a small safety margin.
    """
    if tokens <= 0:
        return 0

    ratio = chars_per_token.get(lang, chars_per_token[LANG_MIXED])
    # 0.9 safety factor so we under-shoot the model limit instead of overshooting.
    return int(tokens * ratio * 0.9)