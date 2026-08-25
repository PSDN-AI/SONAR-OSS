"""Tests for pyannote_utils helpers that do not require pyannote.audio.

The gated-model error wrapping (issue #171) is pure exception inspection, so
it is exercised here without the [pyannote] extra installed.
"""

import pytest

from psdn_sonar.preprocessing.pyannote_utils import GATED_MODEL_HINT, _raise_load_error


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
