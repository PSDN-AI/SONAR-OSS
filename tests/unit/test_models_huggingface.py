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
class TestFfmpegPreflight:
    """Issue #109: adapters that hand file paths to the transformers pipeline
    require ffmpeg for ALL input (WAV included) and must fail once at model
    load with an actionable message, not once per utterance at transcribe
    time."""

    def _hide_ffmpeg(self, monkeypatch):
        import psdn_sonar.models.huggingface as hf

        monkeypatch.setattr(hf.shutil, "which", lambda name: None)

    def test_require_ffmpeg_raises_actionable_error(self, monkeypatch):
        from psdn_sonar.models.base import MissingFfmpegError
        from psdn_sonar.models.huggingface import _require_ffmpeg

        self._hide_ffmpeg(monkeypatch)
        with pytest.raises(MissingFfmpegError) as exc_info:
            _require_ffmpeg("StandardHuggingFaceASR (openai/whisper-base)")
        message = str(exc_info.value)
        assert "ffmpeg" in message
        assert "WAV" in message
        assert "install ffmpeg" in message.lower()

    def test_require_ffmpeg_noop_when_present(self, monkeypatch):
        import psdn_sonar.models.huggingface as hf

        monkeypatch.setattr(hf.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        hf._require_ffmpeg("whatever")  # must not raise

    def test_standard_adapter_fails_before_any_download(self, monkeypatch):
        from psdn_sonar.models.base import MissingFfmpegError
        from psdn_sonar.models.huggingface import StandardHuggingFaceASR

        self._hide_ffmpeg(monkeypatch)
        # The preflight is the constructor's first statement, so nothing from
        # transformers may be touched: poison from_pretrained to prove it.
        with patch("transformers.AutoModelForSpeechSeq2Seq") as mock_model:
            mock_model.from_pretrained.side_effect = AssertionError("checkpoint download attempted")
            with pytest.raises(MissingFfmpegError):
                StandardHuggingFaceASR("openai/whisper-base")
            mock_model.from_pretrained.assert_not_called()

    def test_khushids_adapter_fails_fast(self, monkeypatch):
        from psdn_sonar.models.base import MissingFfmpegError
        from psdn_sonar.models.huggingface import KhushiDSBengaliModel

        self._hide_ffmpeg(monkeypatch)
        with pytest.raises(MissingFfmpegError):
            KhushiDSBengaliModel()

    def test_custom_model_generic_pipeline_branch_not_wrapped(self, monkeypatch):
        from types import SimpleNamespace

        from psdn_sonar.models.base import MissingFfmpegError
        from psdn_sonar.models.huggingface import CustomHuggingFaceModel

        self._hide_ffmpeg(monkeypatch)
        with patch("transformers.AutoConfig") as mock_config:
            mock_config.from_pretrained.return_value = SimpleNamespace(model_type="hubert")
            # Must surface as MissingFfmpegError, not be swallowed into the
            # generic "Failed to load custom HuggingFace model" RuntimeError.
            with pytest.raises(MissingFfmpegError):
                CustomHuggingFaceModel("org/some-hubert-model")

    def test_custom_model_librosa_branches_skip_preflight(self, monkeypatch):
        from types import SimpleNamespace

        from psdn_sonar.models.huggingface import CustomHuggingFaceModel

        self._hide_ffmpeg(monkeypatch)
        # whisper branch decodes via librosa itself — must NOT demand ffmpeg.
        with (
            patch("transformers.AutoConfig") as mock_config,
            patch("transformers.WhisperProcessor") as mock_proc,
            patch("transformers.WhisperForConditionalGeneration") as mock_model,
        ):
            mock_config.from_pretrained.return_value = SimpleNamespace(model_type="whisper")
            mock_proc.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            model = CustomHuggingFaceModel("org/whisper-fine-tune")
            assert model.model_type == "whisper"


@requires_ml
class TestDeviceResolution:
    """Issue #111: adapters checked CUDA only, so on Apple Silicon diarization
    used the GPU while ASR inference silently ran on CPU — and the device
    recorded in scores.json (mps) misstated where inference happened. Auto-
    detection must follow the diarization order: CUDA, then MPS, then CPU."""

    def _force(self, monkeypatch, *, cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    def test_cuda_preferred_over_mps(self, monkeypatch):
        from psdn_sonar.models.huggingface import resolve_device

        self._force(monkeypatch, cuda=True, mps=True)
        assert resolve_device() == torch.device("cuda")

    def test_mps_when_cuda_absent(self, monkeypatch):
        from psdn_sonar.models.huggingface import resolve_device

        self._force(monkeypatch, cuda=False, mps=True)
        assert resolve_device() == torch.device("mps")

    def test_cpu_fallback(self, monkeypatch):
        from psdn_sonar.models.huggingface import resolve_device

        self._force(monkeypatch, cuda=False, mps=False)
        assert resolve_device() == torch.device("cpu")

    def test_explicit_device_passes_through_unchanged(self, monkeypatch):
        """Every historically accepted form stays valid and untouched."""
        from psdn_sonar.models.huggingface import resolve_device

        self._force(monkeypatch, cuda=True, mps=True)  # must NOT override explicit values
        for explicit in (torch.device("cpu"), -1, 0, "mps", "cuda:1"):
            assert resolve_device(explicit) is explicit or resolve_device(explicit) == explicit

    def test_is_cuda_device_across_accepted_forms(self):
        from psdn_sonar.models.huggingface import _is_cuda_device

        assert _is_cuda_device(torch.device("cuda"))
        assert _is_cuda_device(0)
        assert _is_cuda_device("cuda:1")
        assert not _is_cuda_device(torch.device("cpu"))
        assert not _is_cuda_device(torch.device("mps"))
        assert not _is_cuda_device(-1)
        assert not _is_cuda_device("mps")

    def test_whisper_adapter_lands_on_mps(self, monkeypatch):
        """Direct-model adapters must move the model to the resolved device."""
        from psdn_sonar.models.huggingface import WhisperASRModel

        self._force(monkeypatch, cuda=False, mps=True)
        with (
            patch("transformers.WhisperProcessor") as mock_proc,
            patch("transformers.WhisperForConditionalGeneration") as mock_model,
        ):
            mock_proc.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            adapter = WhisperASRModel("org/whisper-fine-tune")
        assert adapter.device == torch.device("mps")
        mock_model.from_pretrained.return_value.to.assert_called_once_with(torch.device("mps"))

    def test_pipeline_adapter_passes_mps_and_stays_fp32(self, monkeypatch):
        """Pipeline adapters hand the resolved device to the pipeline; fp16
        remains CUDA-only (fp16 Whisper generation on MPS is unreliable)."""
        from psdn_sonar.models.huggingface import StandardHuggingFaceASR

        self._force(monkeypatch, cuda=False, mps=True)
        import psdn_sonar.models.huggingface as hf

        monkeypatch.setattr(hf.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        with (
            patch("transformers.AutoModelForSpeechSeq2Seq") as mock_model,
            patch("transformers.AutoProcessor") as mock_proc,
            patch("transformers.pipeline") as mock_pipeline,
        ):
            mock_model.from_pretrained.return_value = MagicMock()
            mock_proc.from_pretrained.return_value = MagicMock()
            adapter = StandardHuggingFaceASR("openai/whisper-base")
        assert adapter.device == torch.device("mps")
        assert mock_model.from_pretrained.call_args.kwargs["dtype"] is torch.float32
        assert mock_pipeline.call_args.kwargs["device"] == torch.device("mps")

    def test_pipeline_adapter_uses_fp16_on_cuda(self, monkeypatch):
        from psdn_sonar.models.huggingface import StandardHuggingFaceASR

        self._force(monkeypatch, cuda=True, mps=False)
        import psdn_sonar.models.huggingface as hf

        monkeypatch.setattr(hf.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        with (
            patch("transformers.AutoModelForSpeechSeq2Seq") as mock_model,
            patch("transformers.AutoProcessor") as mock_proc,
            patch("transformers.pipeline") as mock_pipeline,
        ):
            mock_model.from_pretrained.return_value = MagicMock()
            mock_proc.from_pretrained.return_value = MagicMock()
            adapter = StandardHuggingFaceASR("openai/whisper-base")
        assert adapter.device == torch.device("cuda")
        assert mock_model.from_pretrained.call_args.kwargs["dtype"] is torch.float16
        assert mock_pipeline.call_args.kwargs["device"] == torch.device("cuda")

    def test_custom_model_generic_pipeline_receives_resolved_device(self, monkeypatch):
        from types import SimpleNamespace

        from psdn_sonar.models.huggingface import CustomHuggingFaceModel

        self._force(monkeypatch, cuda=False, mps=True)
        import psdn_sonar.models.huggingface as hf

        monkeypatch.setattr(hf.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        with (
            patch("transformers.AutoConfig") as mock_config,
            patch("transformers.pipeline") as mock_pipeline,
        ):
            mock_config.from_pretrained.return_value = SimpleNamespace(model_type="hubert")
            adapter = CustomHuggingFaceModel("org/some-hubert-model")
        assert adapter.device == torch.device("mps")
        assert mock_pipeline.call_args.kwargs["device"] == torch.device("mps")


@requires_ml
class TestMissingPeftIsActionable:
    """Issue #108: khushids_bengali needs peft ([bengali] extra), which the
    documented [ml] environment lacks. The bare ModuleNotFoundError must
    become an error naming the extra, raised before any download."""

    def test_missing_peft_names_bengali_extra(self, monkeypatch):
        import psdn_sonar.models.huggingface as hf
        from psdn_sonar.models.base import MissingDependencyError

        monkeypatch.setattr(hf.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setitem(sys.modules, "peft", None)  # forces ImportError on 'from peft import ...'
        with patch("transformers.AutoModelForSpeechSeq2Seq") as mock_model:
            mock_model.from_pretrained.side_effect = AssertionError("checkpoint download attempted")
            with pytest.raises(MissingDependencyError, match=r"psdn-sonar\[bengali\]") as exc_info:
                hf.KhushiDSBengaliModel()
            mock_model.from_pretrained.assert_not_called()
        assert "peft" in str(exc_info.value)


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
