"""Tests for pyannote_utils helpers that do not require pyannote.audio.

The gated-model error wrapping (issue #171) is pure exception inspection, so
it is exercised here without the [pyannote] extra installed.
"""

import pytest

from psdn_sonar.preprocessing.pyannote_utils import GATED_MODEL_HINT, _raise_load_error, _refused_repo


class TestRaiseLoadError:
    @pytest.mark.parametrize(
        "message",
        [
            "403 Client Error: Forbidden. Your token is not in the authorized list",
            "401 Client Error: Unauthorized for url",
            "Cannot access gated repo for url",
            "Access to model pyannote/segmentation-3.0 is restricted",
        ],
    )
    def test_auth_failures_get_gated_model_guidance(self, message):
        original = OSError(message)
        with pytest.raises(RuntimeError) as excinfo:
            _raise_load_error("pyannote/segmentation-3.0", original)

        text = str(excinfo.value)
        assert "pyannote/segmentation-3.0" in text
        assert message in text  # the raw HuggingFace error is preserved
        assert "accept each model's user conditions" in text
        assert "https://huggingface.co/pyannote/segmentation-3.0" in text
        assert "https://huggingface.co/pyannote/speaker-diarization-3.1" in text
        assert excinfo.value.__cause__ is original

    def test_unrelated_failures_are_reraised_unchanged(self):
        original = ConnectionError("Connection refused by proxy")
        with pytest.raises(ConnectionError) as excinfo:
            _raise_load_error("pyannote/segmentation-3.0", original)
        assert excinfo.value is original

    def test_hint_names_the_required_step(self):
        # The hint must say the token alone is insufficient — that is the exact
        # gap issue #171 is about.
        assert "HF_TOKEN alone is not" in GATED_MODEL_HINT
        assert "user conditions" in GATED_MODEL_HINT

    def test_hint_names_all_three_gated_models(self):
        # Issue #190: pyannote.audio 4.x's diarization pipeline pulls a third
        # gated repo no command names; the instructions must list it too.
        assert "https://huggingface.co/pyannote/segmentation-3.0" in GATED_MODEL_HINT
        assert "https://huggingface.co/pyannote/speaker-diarization-3.1" in GATED_MODEL_HINT
        assert "https://huggingface.co/pyannote/speaker-diarization-community-1" in GATED_MODEL_HINT


class TestRefusedDependencyInHeadline:
    """Issue #190: when the 403 belongs to a dependency repo, the headline must
    name it — not just the requested pipeline, whose authorization may already
    be granted."""

    # The exact shape reported in the issue: 3.1 was requested and authorized,
    # community-1 was refused.
    _DEPENDENCY_403 = (
        "Cannot access gated repo for url "
        "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/plda/xvec_transform.npz. "
        "Access to model pyannote/speaker-diarization-community-1 is restricted "
        "and you are not in the authorized list."
    )

    def test_headline_names_the_repo_the_403_refused(self):
        original = OSError(self._DEPENDENCY_403)
        with pytest.raises(RuntimeError) as excinfo:
            _raise_load_error("pyannote/speaker-diarization-3.1", original)

        text = str(excinfo.value)
        assert (
            "access was refused for 'pyannote/speaker-diarization-community-1', "
            "a gated repo this pipeline depends on" in text
        )
        assert "Could not load 'pyannote/speaker-diarization-3.1'" in text  # what the caller asked for
        assert self._DEPENDENCY_403 in text  # the raw error is preserved
        assert excinfo.value.__cause__ is original

    def test_refusal_of_the_requested_model_itself_keeps_the_plain_headline(self):
        original = OSError("Access to model pyannote/speaker-diarization-3.1 is restricted")
        with pytest.raises(RuntimeError) as excinfo:
            _raise_load_error("pyannote/speaker-diarization-3.1", original)
        assert "a gated repo this pipeline depends on" not in str(excinfo.value)

    def test_extracts_repo_from_url_form(self):
        text = "403 for url https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml"
        assert _refused_repo(text, "pyannote/speaker-diarization-3.1") == "pyannote/speaker-diarization-community-1"

    def test_extracts_repo_from_prose_form(self):
        text = "Access to model pyannote/speaker-diarization-community-1 is restricted."
        assert _refused_repo(text, "pyannote/speaker-diarization-3.1") == "pyannote/speaker-diarization-community-1"

    def test_requested_model_is_never_reported_as_its_own_dependency(self):
        text = "Access to model pyannote/segmentation-3.0 is restricted"
        assert _refused_repo(text, "pyannote/segmentation-3.0") is None

    def test_non_repo_hub_links_are_ignored(self):
        # HuggingFace auth errors link documentation and settings pages; those
        # must not be mistaken for a refused dependency repo.
        text = (
            "401 Client Error. Make sure you are authenticated: "
            "https://huggingface.co/docs/huggingface_hub/quick-start, "
            "get a token at https://huggingface.co/settings/tokens"
        )
        assert _refused_repo(text, "pyannote/segmentation-3.0") is None

    def test_trailing_sentence_punctuation_is_stripped(self):
        text = "Cannot access repo pyannote/speaker-diarization-community-1."
        assert _refused_repo(text, "pyannote/speaker-diarization-3.1") == "pyannote/speaker-diarization-community-1"
