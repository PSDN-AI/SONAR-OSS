"""Tests for the HuggingFace ASR adapters.

The registry's core promise — importing :mod:`psdn_sonar.models` never pulls
in torch/transformers — is checked in a subprocess so it runs in every
environment, including CI (which installs only ``[dev]``).

The adapter tests require the ``[ml]`` extra (``torch`` is a module-level
import of :mod:`psdn_sonar.models.huggingface`) and skip cleanly without it.
They are hermetic: no model weights are downloaded — instances are built via
``__new__`` with mocked internals.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
except ImportError:  # pragma: no cover — exercised only in [dev]-only installs
    torch = None

requires_ml = pytest.mark.skipif(torch is None, reason="HuggingFace adapter tests require the [ml] extra")


class TestLazyImportContract:
    """Runs in every environment — must NOT be skipped when torch is absent."""

    def test_models_package_does_not_import_ml_libraries(self):
        code = (
            "import sys; import psdn_sonar.models; "
            "banned = {'torch', 'transformers', 'librosa', 'peft'} & set(sys.modules); "
            "sys.exit(f'eagerly imported: {banned}' if banned else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


@requires_ml
class TestCreateModelCustomHF:
    def test_custom_hf_model_bypasses_registry(self):
        from psdn_sonar.models.registry import create_model

        with patch("psdn_sonar.models.huggingface.CustomHuggingFaceModel") as mock_cls:
            create_model("ignored", custom_hf_model="my-org/my-model", language="bn")
            mock_cls.assert_called_once_with(model_id="my-org/my-model", language="bn")


@requires_ml
class TestTranscribeErrorHandling:
    """Every adapter's ``transcribe`` returns None on failure instead of raising."""

    def test_standard_pipeline_adapter_returns_none(self):
        from psdn_sonar.models.huggingface import StandardHuggingFaceASR

        model = StandardHuggingFaceASR.__new__(StandardHuggingFaceASR)
        model.pipe = MagicMock(side_effect=RuntimeError("decode failed"))
        assert model.transcribe("missing.wav") is None

    def test_whisper_adapter_returns_none(self):
        from psdn_sonar.models.huggingface import WhisperASRModel

        model = WhisperASRModel.__new__(WhisperASRModel)
        model._librosa = MagicMock()
        model._librosa.load.side_effect = RuntimeError("corrupt audio")
        assert model.transcribe("corrupt.wav") is None

    def test_standard_pipeline_adapter_strips_text(self):
        from psdn_sonar.models.huggingface import StandardHuggingFaceASR

        model = StandardHuggingFaceASR.__new__(StandardHuggingFaceASR)
        model.pipe = MagicMock(return_value={"text": "  hello world  "})
        assert model.transcribe("clip.wav") == "hello world"


@requires_ml
class TestCustomHuggingFaceModelDispatch:
    """``CustomHuggingFaceModel`` routes transcription by detected model type."""

    def _bare_instance(self):
        from psdn_sonar.models.huggingface import CustomHuggingFaceModel

        model = CustomHuggingFaceModel.__new__(CustomHuggingFaceModel)
        model.device = torch.device("cpu")
        model.language = None
        return model

    def test_pipeline_type_uses_pipe(self):
        model = self._bare_instance()
        model.model_type = "pipeline"
        model.pipe = MagicMock(return_value={"text": " out "})
        assert model.transcribe("clip.wav") == "out"

    def test_unknown_failure_returns_none(self):
        model = self._bare_instance()
        model.model_type = "pipeline"
        model.pipe = MagicMock(side_effect=RuntimeError("boom"))
        assert model.transcribe("clip.wav") is None

    def test_whisper_type_uses_processor_and_generate(self):
        model = self._bare_instance()
        model.model_type = "whisper"
        model.librosa = MagicMock()
        model.librosa.load.return_value = ([0.0], 16000)

        inputs = MagicMock()
        inputs.input_features.to.return_value = "features"
        model.processor = MagicMock()
        model.processor.return_value = inputs
        model.processor.batch_decode.return_value = [" decoded text "]
        model.model = MagicMock()

        assert model.transcribe("clip.wav") == "decoded text"
        model.model.generate.assert_called_once_with("features", return_timestamps=False)


@requires_ml
class TestRegistryHuggingFacePaths:
    def test_all_hf_class_paths_resolve(self):
        """Every registry entry pointing at this module must name a real class."""
        from psdn_sonar.models import huggingface
        from psdn_sonar.models.registry import _MODEL_CONFIGS

        prefix = "psdn_sonar.models.huggingface."
        hf_entries = {name: path for name, (path, _) in _MODEL_CONFIGS.items() if path.startswith(prefix)}
        assert hf_entries, "expected HuggingFace-backed entries in the registry"
        for name, path in hf_entries.items():
            cls_name = path.rsplit(".", 1)[1]
            assert hasattr(huggingface, cls_name), f"registry entry '{name}' points at missing class {cls_name}"
