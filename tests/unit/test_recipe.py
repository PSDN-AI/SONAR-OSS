"""Tests for language evaluation recipes."""

import json

import pytest

from psdn_sonar.models.registry import get_language_defaults, get_model_config
from psdn_sonar.recipe import Recipe, RecipeFactory, get_recipe


class TestRecipeCreation:
    def test_bengali_recipe(self):
        recipe = get_recipe("bengali")
        assert recipe.language == "bengali"
        assert {d["name"] for d in recipe.datasets} == {
            "common_voice",
            "fleurs",
            "openslr37_bd",
            "openslr37_in",
            "openslr53",
        }

    def test_models_come_from_registry_defaults(self):
        for language in ("bengali", "hindi", "english", "korean"):
            recipe = get_recipe(language)
            assert [m["name"] for m in recipe.models] == get_language_defaults(language)

    def test_model_entries_have_type_and_provider(self):
        recipe = get_recipe("english")
        by_name = {m["name"]: m for m in recipe.models}
        assert by_name["whisper_api"] == {"name": "whisper_api", "type": "api", "provider": "openai"}
        assert by_name["whisper_base_en"]["provider"] == "huggingface"
        assert by_name["whisper_base_en"]["model_id"] == "openai/whisper-base"
        assert get_model_config("whisper_base_en") is not None

    def test_language_aliases(self):
        for alias, name in [("bn", "bengali"), ("hi", "hindi"), ("en", "english"), ("ko", "korean")]:
            assert get_recipe(alias).language == name
            assert get_recipe(name.title()).language == name

    def test_user_dataset_appended(self):
        recipe = get_recipe("hindi", "path/to/my/data.tsv")
        assert recipe.datasets[-1] == {"name": "user_dataset", "path": "path/to/my/data.tsv"}

    def test_korean_includes_zeroth(self):
        names = {d["name"] for d in get_recipe("korean").datasets}
        assert names == {"common_voice", "fleurs", "zeroth"}

    def test_common_voice_points_to_live_source(self):
        # Mozilla emptied the HF common_voice_* repos in Oct 2025.
        for language in ("bengali", "hindi", "english", "korean"):
            cv = next(d for d in get_recipe(language).datasets if d["name"] == "common_voice")
            assert "mozilla-foundation" not in cv["path"]
            assert "datacollective" in cv["path"]

    def test_dataset_lists_are_not_shared_between_recipes(self):
        first = get_recipe("hindi")
        first.datasets[0]["path"] = "mutated"
        assert get_recipe("hindi").datasets[0]["path"] != "mutated"

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_recipe("klingon")

    def test_factory_and_helper_agree(self):
        assert RecipeFactory.create("ko").language == get_recipe("ko").language


class TestNormalization:
    def test_bengali_punctuation_and_numbers(self):
        recipe = get_recipe("bengali")
        normalized = recipe.normalize("এটি একটি পরীক্ষা, ১২৩ নম্বর!")
        assert "," not in normalized
        assert "!" not in normalized
        assert "১২৩" not in normalized  # digits verbalized
        assert len(recipe.tokenize(normalized)) > 0

    def test_hindi_danda_removed(self):
        normalized = get_recipe("hindi").normalize("यह एक परीक्षण वाक्य है।")
        assert "।" not in normalized
        assert len(normalized) > 0

    def test_english_lowercased_and_stripped(self):
        recipe = get_recipe("english")
        normalized = recipe.normalize("This is a Test Sentence!")
        assert normalized == normalized.lower()
        assert "!" not in normalized
        assert len(recipe.tokenize(normalized)) == 5

    def test_korean_punctuation_removed(self):
        normalized = get_recipe("korean").normalize("안녕하세요, 테스트!")
        assert "," not in normalized
        assert "안녕하세요" in normalized


class TestSerialization:
    def test_to_json_round_trip(self):
        recipe = get_recipe("bengali", "my/data.tsv")
        payload = json.loads(recipe.to_json())
        assert payload["language"] == "bengali"
        assert payload["datasets"][-1]["name"] == "user_dataset"
        assert len(payload["models"]) == len(recipe.models)

    def test_display_helpers(self):
        recipe = get_recipe("english")
        assert json.loads(recipe.display_models()) == recipe.models
        assert json.loads(recipe.display_datasets()) == recipe.datasets

    def test_tokenize_is_whitespace_split(self):
        assert Recipe.tokenize("a b  c") == ["a", "b", "c"]
