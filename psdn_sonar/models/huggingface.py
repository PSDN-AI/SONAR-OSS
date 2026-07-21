"""HuggingFace-based ASR model adapters.

This module imports ``torch`` (and, per adapter, ``transformers`` /
``librosa`` / ``peft``) at module scope, so it requires the ``[ml]`` extra:

    pip install "psdn-sonar[ml]"

It is never imported by the model registry until a HuggingFace-backed model
is actually instantiated (``psdn_sonar.models.registry`` stores dotted
class-path strings), so registry listing and non-HF adapters work without
these dependencies installed.

All ``transcribe`` implementations return ``None`` on failure rather than
raising, so a single corrupt clip does not abort a long evaluation run.
"""

import logging
from typing import Optional

import torch

from .base import ASRModel

logger = logging.getLogger(__name__)


def _pipeline_text(result) -> str:
    """Extract stripped text from an ASR pipeline result.

    For a single audio input the pipeline returns ``{"text": ...}``; the
    isinstance check narrows the union return type (batched calls return a
    list, which these adapters never issue).
    """
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return ""


class StandardHuggingFaceASR(ASRModel):
    """Generic seq2seq ASR adapter built on the ``transformers`` pipeline.

    Works for any ``AutoModelForSpeechSeq2Seq``-compatible checkpoint
    (Whisper fine-tunes and similar). Pass ``language`` to force decoding
    into a specific language for multilingual checkpoints.
    """

    def __init__(self, model_id, device=None, chunk_length_s=30, language=None):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = 0 if device is None and torch.cuda.is_available() else (-1 if device is None else device)
        dtype = torch.float16 if device >= 0 and torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
        if device >= 0:
            model = model.to(f"cuda:{device}")
        processor = AutoProcessor.from_pretrained(model_id)

        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=chunk_length_s,
            generate_kwargs=generate_kwargs,
        )

    def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            return _pipeline_text(self.pipe(audio_path))
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None


class WhisperASRModel(ASRModel):
    """Base class for Whisper-based ASR models loaded without the pipeline.

    Uses ``WhisperProcessor`` + ``WhisperForConditionalGeneration`` directly
    with ``librosa`` decoding at 16 kHz. Subclasses only pin a default
    ``model_id``.
    """

    def __init__(self, model_id: str, device=None):
        import librosa
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(
            self.device
        )
        self._librosa = librosa

    def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            audio, sr = self._librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, return_timestamps=False)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None


class BanglaSpeech2TextModel(WhisperASRModel):
    def __init__(self, model_id="anuragshas/whisper-large-v2-bn", device=None):
        super().__init__(model_id, device)


class BanglaASRModel(WhisperASRModel):
    def __init__(self, model_id="bangla-speech-processing/BanglaASR", device=None):
        super().__init__(model_id, device)


class TugstugiBengaliWhisperModel(WhisperASRModel):
    def __init__(self, model_id="bengaliAI/tugstugi_bengaliai-asr_whisper-medium", device=None):
        super().__init__(model_id, device)


class TugstugiRegionalWhisperModel(WhisperASRModel):
    def __init__(self, model_id="bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium", device=None):
        super().__init__(model_id, device)


class BanglaASRV5Model(WhisperASRModel):
    def __init__(self, model_id="arif11/bangla-ASR-v5", device=None):
        super().__init__(model_id, device)


class KhushiDSBengaliModel(ASRModel):
    """Bengali Whisper fine-tune published as a PEFT/LoRA adapter.

    Loads ``openai/whisper-large-v3`` as the base model and merges the
    adapter weights (``merge_and_unload``), so inference runs as a plain
    Whisper pipeline. Requires ``peft`` (part of the ``[bengali]`` extra).
    """

    def __init__(self, model_id="KhushiDS/whisper-large-v3-Bengali", device=None, chunk_length_s=30):
        from peft import PeftModel
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = 0 if device is None and torch.cuda.is_available() else (-1 if device is None else device)
        dtype = torch.float16 if device >= 0 and torch.cuda.is_available() else torch.float32
        base = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3", dtype=dtype, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, model_id).merge_and_unload()
        if device >= 0:
            model = model.to(f"cuda:{device}")
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=chunk_length_s,
        )

    def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            return _pipeline_text(self.pipe(audio_path))
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None


class Wav2Vec2BengaliModel(ASRModel):
    """Bengali Wav2Vec2 CTC adapter.

    Some published checkpoints ship a ``tokenizer_config.json`` whose
    ``extra_special_tokens`` field has a shape newer ``transformers``
    versions reject; the fallback path in ``__init__`` rewrites that field
    and rebuilds the processor from its parts before giving up.
    """

    def __init__(self, model_id="arijitx/wav2vec2-xls-r-300m-bengali", device=None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        except Exception as e:
            logger.warning("Standard processor loading failed, trying alternative approach...")
            import json
            import tempfile

            from huggingface_hub import hf_hub_download
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

            try:
                config_file = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
                with open(config_file, "r") as f:
                    config = json.load(f)
                if "extra_special_tokens" in config and not isinstance(config["extra_special_tokens"], list):
                    config["extra_special_tokens"] = []

                with tempfile.TemporaryDirectory() as temp_dir:
                    import os

                    fixed_config_path = os.path.join(temp_dir, "tokenizer_config.json")
                    with open(fixed_config_path, "w") as f:
                        json.dump(config, f)
                    vocab_file = hf_hub_download(repo_id=model_id, filename="vocab.json")
                    tokenizer = Wav2Vec2CTCTokenizer(
                        vocab_file, **{k: v for k, v in config.items() if k != "extra_special_tokens"}
                    )
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
                    self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            except Exception as e2:
                logger.error("Error fixing tokenizer: %s", e2)
                raise e
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)

    def transcribe(self, audio_path: str) -> Optional[str]:
        import librosa

        try:
            sa, _ = librosa.load(audio_path, sr=16000)
            inputs = self.processor(sa, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            return self.processor.decode(torch.argmax(logits, dim=-1)[0]).strip()
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None


class Wav2Vec2KoreanModel(ASRModel):
    """Korean Wav2Vec2 model handler."""

    def __init__(self, model_id="kresnik/wav2vec2-large-xlsr-korean", device=None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Korean Wav2Vec2 model {model_id}: {e}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        import librosa

        try:
            speech_array, _ = librosa.load(audio_path, sr=16000)
            inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription.strip()
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None


class WhisperKoreanModel(ASRModel):
    """Korean Whisper fine-tuned model handler."""

    def __init__(self, model_id="SungBeom/whisper-small-ko", device=None):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        try:
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
            self.model.config.forced_decoder_ids = None
        except Exception as e:
            raise RuntimeError(f"Failed to load Korean Whisper model {model_id}: {e}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        import librosa

        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                predicted_ids = self.model.generate(**inputs)

            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None


class CustomHuggingFaceModel(ASRModel):
    """
    Generic HuggingFace model loader for Bengali ASR.
    Automatically detects model architecture and loads appropriate processor.

    Supports:
    - Whisper-based models (most Bengali models)
    - Wav2Vec2-based models
    - Generic ASR pipeline models

    Usage:
        model = CustomHuggingFaceModel("username/model-name")
        transcription = model.transcribe("audio.wav")
    """

    def __init__(self, model_id: str, device=None, chunk_length_s=30, language=None, trust_remote_code: bool = False):
        """
        Initialize custom HuggingFace model.

        Args:
            model_id: HuggingFace model ID (e.g., "username/model-name")
            device: Device to use (None for auto-detect, 0 for cuda:0, -1 for CPU)
            chunk_length_s: Chunk length for pipeline models (default: 30)
            language: Language code for forced decoding (e.g., "ko", "bn")
            trust_remote_code: Whether to trust remote code when loading model config/weights
        """
        import librosa
        from transformers import AutoConfig
        from transformers import pipeline as hf_pipeline

        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.librosa = librosa
        self.language = language

        logger.info("Loading custom HuggingFace model: %s", model_id)

        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            model_type = config.model_type.lower()

            logger.info("Detected model type: %s", model_type)

            if "whisper" in model_type:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                self.processor = WhisperProcessor.from_pretrained(model_id)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32).to(
                    self.device
                )
                self.model_type = "whisper"

            elif "wav2vec2" in model_type:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                self.processor = Wav2Vec2Processor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
                self.model_type = "wav2vec2"

            else:
                logger.info("Using generic ASR pipeline for model type: %s", model_type)
                device_id = 0 if self.device.type == "cuda" else -1
                self.pipe = hf_pipeline(
                    "automatic-speech-recognition", model=model_id, device=device_id, chunk_length_s=chunk_length_s
                )
                self.model_type = "pipeline"

            logger.info("Model loaded successfully (%s)", self.model_type)

        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise RuntimeError(f"Failed to load custom HuggingFace model '{model_id}': {e}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file."""
        try:
            if self.model_type == "whisper":
                audio, sr = self.librosa.load(audio_path, sr=16000)
                inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(self.device)

                generate_kwargs: dict = {"return_timestamps": False}
                if self.language:
                    from psdn_sonar.language_codes import to_long_name

                    generate_kwargs["language"] = to_long_name(self.language)

                with torch.no_grad():
                    predicted_ids = self.model.generate(input_features, **generate_kwargs)

                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return transcription.strip()

            elif self.model_type == "wav2vec2":
                audio, sr = self.librosa.load(audio_path, sr=16000)
                inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.decode(predicted_ids[0])
                return transcription.strip()

            else:
                return _pipeline_text(self.pipe(audio_path))

        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return None
