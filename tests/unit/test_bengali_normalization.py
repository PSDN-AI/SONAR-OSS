"""Tests for canonical Bengali text normalization and loanword replacement.

Pins the exact behavior of the canonical WER normalization pipeline
(loanword replacement, suffix splitting, nasal normalization, number
variant canonicalization) so scoring stays comparable across releases.
"""

# ---------------------------------------------------------------------------
# Loanword normalizer tests
# ---------------------------------------------------------------------------


class TestLoanwordNormalizer:
    """Tests for psdn_sonar.utils.loanword module."""

    def test_is_latin(self):
        from psdn_sonar.utils.loanword import is_latin

        assert is_latin("insurance")
        assert is_latin("hello")
        assert is_latin("COVID-19")
        assert is_latin("don't")
        assert is_latin("123")
        assert not is_latin("ইনস্যুরেন্স")
        assert not is_latin("hello world")
        assert not is_latin("")
        assert not is_latin("hello!")

    def test_extract_latin_tokens(self):
        from psdn_sonar.utils.loanword import extract_latin_tokens

        texts = ["আমার insurance পলিসি", "একটি facebook পেজ", "insurance আবার"]
        tokens = extract_latin_tokens(texts)
        assert tokens == {"insurance", "facebook"}

    def test_extract_latin_tokens_skips_digits(self):
        from psdn_sonar.utils.loanword import extract_latin_tokens

        tokens = extract_latin_tokens(["আমার 24 insurance 100 পলিসি"])
        assert "insurance" in tokens
        assert "24" not in tokens
        assert "100" not in tokens

    def test_extract_latin_tokens_empty(self):
        from psdn_sonar.utils.loanword import extract_latin_tokens

        assert extract_latin_tokens([]) == set()
        assert extract_latin_tokens(["আমার বাংলা টেক্সট"]) == set()

    def test_replace_latin_tokens_basic(self):
        from psdn_sonar.utils.loanword import replace_latin_tokens

        cache = {"insurance": "ইনস্যুরেন্স", "facebook": "ফেসবুক"}
        result, replaced, uncached = replace_latin_tokens("আমার insurance এবং facebook", cache)
        assert result == "আমার ইনস্যুরেন্স এবং ফেসবুক"
        assert replaced == 2
        assert uncached == 0

    def test_replace_latin_tokens_case_insensitive(self):
        from psdn_sonar.utils.loanword import replace_latin_tokens

        cache = {"insurance": "ইনস্যুরেন্স"}
        result, replaced, _ = replace_latin_tokens("Insurance INSURANCE insurance", cache)
        assert replaced == 3

    def test_replace_latin_tokens_uncached(self):
        from psdn_sonar.utils.loanword import replace_latin_tokens

        cache = {"insurance": "ইনস্যুরেন্স"}
        result, replaced, uncached = replace_latin_tokens("আমার insurance এবং facebook", cache)
        assert replaced == 1
        assert uncached == 1
        assert "facebook" in result

    def test_replace_latin_tokens_no_latin(self):
        from psdn_sonar.utils.loanword import replace_latin_tokens

        cache = {"insurance": "ইনস্যুরেন্স"}
        result, replaced, uncached = replace_latin_tokens("আমার বাংলা টেক্সট", cache)
        assert result == "আমার বাংলা টেক্সট"
        assert replaced == 0
        assert uncached == 0

    def test_replace_latin_tokens_skips_digits(self):
        from psdn_sonar.utils.loanword import replace_latin_tokens

        cache = {"insurance": "ইনস্যুরেন্স", "24": "টোয়েন্টি ফোর"}
        result, replaced, uncached = replace_latin_tokens("আমার 24 insurance পলিসি", cache)
        assert "24" in result
        assert "ইনস্যুরেন্স" in result
        assert replaced == 1
        assert uncached == 0

    def test_replace_mixed_script_token(self):
        from psdn_sonar.utils.loanword import replace_latin_tokens

        cache = {"helpline": "হেল্পলাইন"}
        result, replaced, _ = replace_latin_tokens("helpline-এ কল করুন", cache)
        assert "হেল্পলাইন" in result
        assert "helpline" not in result
        assert replaced == 1

    def test_load_cache(self):
        from psdn_sonar.utils.loanword import get_cache_path, load_cache

        cache_path = get_cache_path("bn")
        cache = load_cache(cache_path)
        assert len(cache) > 1000
        assert "customer" in cache
        assert cache["customer"] == "কাস্টমার"

    def test_load_cache_nonexistent(self):
        from psdn_sonar.utils.loanword import load_cache

        assert load_cache("/tmp/nonexistent_sonar_cache.json") == {}


# ---------------------------------------------------------------------------
# canonical Bengali normalization tests
# ---------------------------------------------------------------------------


class TestBengaliForWerNormalization:
    """Tests for normalize_bengali_for_wer (canonical pipeline)."""

    def test_basic(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("  হ্যালো   ওয়ার্ল্ড  ")
        assert "হ্যালো" in result
        assert "ওয়ার্ল্ড" in result
        assert "  " not in result

    def test_removes_punctuation(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("হ্যালো। ওয়ার্ল্ড!")
        assert "।" not in result
        assert "!" not in result

    def test_removes_zwj(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("হ্যা\u200dলো\u200cওয়ার্ল্ড")
        assert "\u200d" not in result
        assert "\u200c" not in result

    def test_lowercases_latin(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("Hello WORLD বাংলা")
        # "Hello" should be replaced by loanword cache (হ্যালো)
        assert "Hello" not in result
        assert "WORLD" not in result

    def test_bengali_numerals_to_words(self):
        """১২৩ → 123 → একশো তেইশ."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("১২৩ টেক্সট")
        assert "একশো" in result
        assert "তেইশ" in result

    def test_digits_to_words(self):
        """24 → চব্বিশ."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("আমি 24 টাকা দিয়েছি")
        assert "চব্বিশ" in result
        assert "24" not in result

    def test_hundred_sot_to_sho(self):
        """100 → একশত → একশো."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("আমি 100 টাকা দিয়েছি")
        assert "একশো" in result
        assert "একশত" not in result

    def test_number_variant_normalization(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        assert "পনেরো" in normalize_bengali_for_wer("পনের দিন")
        assert "দুইশো" in normalize_bengali_for_wer("দুশো টাকা")
        assert "ত্রিশ" in normalize_bengali_for_wer("তিরিশ মিনিট")

    def test_shotangsho_not_affected(self):
        """শতাংশ should NOT become শোংশ."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("শতাংশ হারে")
        assert "শতাংশ" in result
        assert "শোংশ" not in result

    def test_percent_symbol_verbalized(self):
        """৫০% → পঞ্চাশ শতাংশ (issue #136: % used to survive as a literal)."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("৫০%")
        assert result == "পঞ্চাশ শতাংশ"
        assert "%" not in result

    def test_percent_symbol_and_word_form_match(self):
        """The two common written forms of the same quantity normalize identically."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        assert normalize_bengali_for_wer("৫০%") == normalize_bengali_for_wer("৫০ শতাংশ")
        assert normalize_bengali_for_wer("১০০%") == normalize_bengali_for_wer("১০০ শতাংশ")

    def test_no_symbol_survives_normalization(self):
        """The issue-#136 follow-up corpus: nothing whose spacing could vary
        by bnlp availability may survive normalization."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        for text in ("৫০%", "১০০% ভালো", "A+B", "test@mail.com"):
            result = normalize_bengali_for_wer(text)
            for symbol in "%+@":
                assert symbol not in result, f"{symbol!r} survived in {result!r} (input {text!r})"

    def test_plus_and_at_verbalized(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        assert "যোগ" in normalize_bengali_for_wer("A+B")
        assert "অ্যাট" in normalize_bengali_for_wer("test@mail.com")

    def test_phone_number_not_converted(self):
        """5+ digit sequences stay as digits."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("নম্বর 98765 দিন")
        assert "98765" in result

    def test_leading_zero_not_converted(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("কল করুন 0176 নম্বরে")
        assert "0176" in result

    def test_suffix_splitting_ta(self):
        """প্যাকেটটা → প্যাকেট টা."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("প্যাকেটটা")
        assert "প্যাকেট" in result
        assert "টা" in result

    def test_suffix_splitting_er(self):
        """রিফান্ডের → রিফান্ড এর."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("রিফান্ডের")
        assert "রিফান্ড" in result
        assert "এর" in result

    def test_suffix_splitting_e_locative(self):
        """অর্ডারে → অর্ডার এ."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("অর্ডারে")
        assert "অর্ডার" in result
        assert "এ" in result

    def test_whole_words_not_split(self):
        """Issue #142: a bare endswith match used to cut whole words in two
        (মাটি → মা টি, ছেলে → ছেল এ). Structural stem guards and the
        protected-word lexicon must keep them intact."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        for word in ["মাটি", "বাটি", "রুটি", "ছুটি", "কাকে", "তাকে", "ছেলে", "মেয়ে", "কমিটি", "এটি"]:
            assert normalize_bengali_for_wer(word) == word, f"{word} must not be split"

    def test_no_virama_terminated_fragment(self):
        """Issue #142: ঘণ্টা is ঘ ণ ্ ট া — matching the টা suffix used to
        cut inside the conjunct and leave the fragment ঘণ্. A split must
        never land after a virama."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("ঘণ্টা বাজে")
        assert "ঘণ্টা" in result
        assert not any(tok.endswith("\u09cd") for tok in result.split())

    def test_untouched_words_stay_untouched(self):
        """The issue's control set — no suffix-lookalike ending, must pass
        through whole."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        for word in ["ঘটনা", "কণ্টক", "ঠাণ্ডা", "চিন্তা", "গল্প", "পাখি", "মাথা"]:
            assert normalize_bengali_for_wer(word) == word

    def test_real_suffixes_still_split(self):
        """The intended splits from the issue's 'correctly split' rows must
        keep working with the new guards in place."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        assert normalize_bengali_for_wer("প্যাকেটটা") == "প্যাকেট টা"
        assert normalize_bengali_for_wer("বইগুলো") == "বই গুলো"
        assert normalize_bengali_for_wer("ছেলেটি") == "ছেলে টি"
        assert normalize_bengali_for_wer("দেশে") == "দেশ এ"
        assert normalize_bengali_for_wer("একটি") == "এক টি"

    def test_locative_ekar_splits_at_correct_point(self):
        """হাতে used to hit the তে suffix first and split as হা তে (a
        nonsense stem). With the single-cluster guard it now falls through
        to the ekar rule and splits at the morpheme boundary, consistent
        with the দেশে → দেশ এ precedent."""
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        assert normalize_bengali_for_wer("হাতে") == "হাত এ"
        assert normalize_bengali_for_wer("রাতে") == "রাত এ"

    def test_nasal_normalization(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("ব্যাঙ্কের")
        assert "ব্যাং" in result

    def test_loanword_replacement(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        result = normalize_bengali_for_wer("Hello বাংলা")
        assert "hello" not in result
        assert "Hello" not in result
        assert "হ্যালো" in result

    def test_empty_input(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer

        assert normalize_bengali_for_wer("") == ""
        assert normalize_bengali_for_wer("   ") == ""

    def test_loanword_wer_reduction(self):
        """Loanword normalization should dramatically reduce WER on code-mixed text."""
        from jiwer import wer

        from psdn_sonar.utils.text_processing import normalize_text_unified

        hyp = "আমি customer care-এ try করছি"
        ref = "আমি কাস্টমার কেয়ারে ট্রাই করছি"

        raw_wer = wer(ref, hyp) * 100
        norm_wer = (
            wer(
                normalize_text_unified(ref, "bn"),
                normalize_text_unified(hyp, "bn"),
            )
            * 100
        )

        assert raw_wer > 30, "Raw WER should be high due to script mismatch"
        assert norm_wer < 5, "Normalized WER should be near zero"


class TestNormalizeTextUnifiedBengali:
    """Test that normalize_text_unified delegates to canonical normalization for Bengali."""

    def test_unified_uses_canonical_bengali_pipeline(self):
        from psdn_sonar.utils.text_processing import normalize_bengali_for_wer, normalize_text_unified

        text = "Hello, হ্যাঁ নমস্কার। customer care-এ try করছি।"
        unified = normalize_text_unified(text, language="bn")
        direct = normalize_bengali_for_wer(text)
        assert unified == direct

    def test_unified_removes_punctuation(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("এটি, একটি! পরীক্ষা?", language="bn")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert result.strip() != ""

    def test_unified_english_unchanged(self):
        """English normalization should not be affected by Bengali changes."""
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Hello, World!", language="en")
        assert "hello" in result
        assert "," not in result

    def test_unified_empty(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        assert normalize_text_unified("", language="bn") == ""
        assert normalize_text_unified("   ", language="bn") == ""
