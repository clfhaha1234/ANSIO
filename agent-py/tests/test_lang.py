"""Offline unit tests for lang.py — no network, no credentials, no livekit.

Covers:
  - normalize_language: all documented aliases + illegal-value fallback to "auto"
  - language_directive: mutual-exclusion keywords per branch
  - greeting_suffix: three branches, all non-empty and distinct
  - tts_language_boost: en/zh/auto mapping against TTSLanguageBoost enum values
"""

from __future__ import annotations

from lang import (
    LANGS,
    greeting_suffix,
    language_directive,
    normalize_language,
    stt_language,
    tts_language_boost,
)

# ---------------------------------------------------------------------------
# normalize_language
# ---------------------------------------------------------------------------


class TestNormalizeLanguage:
    # -- English aliases -------------------------------------------------------

    def test_en_lower(self):
        assert normalize_language("en") == "en"

    def test_en_mixed_case(self):
        assert normalize_language("EN") == "en"

    def test_english_word(self):
        assert normalize_language("english") == "en"

    def test_english_word_capitalized(self):
        assert normalize_language("English") == "en"

    def test_chinese_zh_kanji(self):
        assert normalize_language("英文") == "en"

    # -- Chinese aliases -------------------------------------------------------

    def test_zh_lower(self):
        assert normalize_language("zh") == "zh"

    def test_zh_upper(self):
        assert normalize_language("ZH") == "zh"

    def test_chinese_word(self):
        assert normalize_language("chinese") == "zh"

    def test_chinese_word_capitalized(self):
        assert normalize_language("Chinese") == "zh"

    def test_zh_hanzi(self):
        assert normalize_language("中文") == "zh"

    def test_zh_cn_hyphen(self):
        assert normalize_language("zh-cn") == "zh"

    def test_zh_cn_upper(self):
        assert normalize_language("ZH-CN") == "zh"

    # -- Fallback to "auto" ----------------------------------------------------

    def test_none_returns_auto(self):
        assert normalize_language(None) == "auto"

    def test_empty_string_returns_auto(self):
        assert normalize_language("") == "auto"

    def test_unknown_string_returns_auto(self):
        assert normalize_language("fr") == "auto"

    def test_numeric_value_returns_auto(self):
        assert normalize_language(42) == "auto"

    def test_gibberish_returns_auto(self):
        assert normalize_language("xyz_bogus") == "auto"

    # -- LANGS tuple -----------------------------------------------------------

    def test_langs_tuple_contents(self):
        assert set(LANGS) == {"en", "zh", "auto"}


# ---------------------------------------------------------------------------
# language_directive — mutual-exclusion keyword checks
# ---------------------------------------------------------------------------


class TestLanguageDirective:
    def test_en_contains_only_in_english(self):
        directive = language_directive("en")
        assert "ONLY in English" in directive

    def test_en_does_not_contain_chinese_keyword(self):
        directive = language_directive("en")
        assert "只用中文" not in directive
        assert "NEVER mix" not in directive

    def test_zh_contains_zh_keyword(self):
        directive = language_directive("zh")
        assert "只用中文" in directive

    def test_zh_does_not_contain_english_keyword(self):
        directive = language_directive("zh")
        assert "ONLY in English" not in directive
        assert "NEVER mix" not in directive

    def test_auto_contains_never_mix(self):
        directive = language_directive("auto")
        assert "NEVER mix" in directive

    def test_auto_does_not_contain_exclusive_keywords(self):
        directive = language_directive("auto")
        assert "ONLY in English" not in directive
        assert "只用中文" not in directive

    def test_directives_all_start_with_double_newline(self):
        for lang in LANGS:
            assert language_directive(lang).startswith("\n\n"), (
                f"failed for lang={lang!r}"
            )

    def test_unknown_lang_behaves_like_auto(self):
        assert language_directive("fr") == language_directive("auto")


# ---------------------------------------------------------------------------
# greeting_suffix — three distinct non-empty branches
# ---------------------------------------------------------------------------


class TestGreetingSuffix:
    def test_en_non_empty(self):
        assert greeting_suffix("en").strip()

    def test_zh_non_empty(self):
        assert greeting_suffix("zh").strip()

    def test_auto_non_empty(self):
        assert greeting_suffix("auto").strip()

    def test_all_branches_distinct(self):
        results = {
            greeting_suffix("en"),
            greeting_suffix("zh"),
            greeting_suffix("auto"),
        }
        assert len(results) == 3, "greeting_suffix branches must all be distinct"

    def test_en_suffix_content(self):
        assert "English" in greeting_suffix("en")

    def test_zh_suffix_content(self):
        assert "中文" in greeting_suffix("zh")

    def test_auto_suffix_content(self):
        assert "English" in greeting_suffix("auto")

    def test_unknown_lang_behaves_like_auto(self):
        assert greeting_suffix("bogus") == greeting_suffix("auto")


# ---------------------------------------------------------------------------
# tts_language_boost — maps to TTSLanguageBoost Literal values
# Evidence: tts.py line 94-98:
#   TTSLanguageBoost = Literal["auto", "Chinese", "Chinese,Yue", "English", ...]
# ---------------------------------------------------------------------------


class TestTtsLanguageBoost:
    def test_en_maps_to_english(self):
        assert tts_language_boost("en") == "English"

    def test_zh_maps_to_chinese(self):
        assert tts_language_boost("zh") == "Chinese"

    def test_auto_maps_to_auto(self):
        assert tts_language_boost("auto") == "auto"

    def test_unknown_maps_to_auto(self):
        assert tts_language_boost("fr") == "auto"

    def test_all_values_are_valid_literal_members(self):
        # The three values we return must be exact members of TTSLanguageBoost.
        # Verified against the Literal sourced from tts.py lines 94-98.
        valid = {"auto", "Chinese", "Chinese,Yue", "English"}
        for lang in LANGS:
            result = tts_language_boost(lang)
            assert result in valid, (
                f"tts_language_boost({lang!r}) -> {result!r} not in TTSLanguageBoost"
            )


# ---------------------------------------------------------------------------
# stt_language — Deepgram nova-3 `language` arg per toggle.
# nova-3 "multi" EXCLUDES Chinese, so zh must map to the dedicated "zh" model.
# ---------------------------------------------------------------------------


class TestSttLanguage:
    def test_zh_maps_to_zh(self):
        # The crux: Chinese must NOT use "multi" (which can't transcribe it).
        assert stt_language("zh") == "zh"

    def test_en_maps_to_en(self):
        assert stt_language("en") == "en"

    def test_auto_maps_to_multi(self):
        assert stt_language("auto") == "multi"

    def test_unknown_maps_to_multi(self):
        assert stt_language("fr") == "multi"

    def test_zh_is_never_multi(self):
        # Regression guard for the original bug (language hardcoded to "multi").
        assert stt_language("zh") != "multi"

    def test_all_langs_resolve(self):
        for lang in LANGS:
            assert stt_language(lang) in {"zh", "en", "multi"}
