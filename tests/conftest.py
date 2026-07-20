import numpy as np
import pandas as pd
import pytest
import soundfile as sf


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory path."""
    return tmp_path


@pytest.fixture
def sine_wave_audio():
    """Generate a 1-second 440 Hz sine wave at 16 kHz."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silent_audio():
    """Generate 1 second of near-silence."""
    return np.zeros(16_000, dtype=np.float32)


@pytest.fixture
def clipped_audio():
    """Generate audio where every sample is at the clipping boundary."""
    return np.ones(16_000, dtype=np.float32)


@pytest.fixture
def wav_file(tmp_path, sine_wave_audio):
    """Write a sine wave to a temporary WAV file and return its path."""
    path = tmp_path / "test.wav"
    sf.write(str(path), sine_wave_audio, 16_000)
    return str(path)


@pytest.fixture
def single_speaker_csv(tmp_path):
    """Create a minimal single-speaker results CSV."""
    df = pd.DataFrame(
        {
            "audio_path": ["a.wav", "b.wav", "c.wav"],
            "ground_truth": ["hello world", "foo bar", "test text"],
            "prediction": ["hello world", "foo baz", "tset text"],
            "wer": [0.0, 0.5, 0.5],
            "cer": [0.0, 0.167, 0.25],
            "semantic_similarity": [1.0, 0.85, 0.9],
            "poseidon_score": [1.0, 0.7, 0.75],
            "snr_db": [25.0, 15.0, 5.0],
            "clipping_ratio": [0.0, 0.005, 0.02],
            "silence_ratio": [0.1, 0.3, 0.9],
            "snr_tier": ["High", "Medium", "Low"],
        }
    )
    path = tmp_path / "asr_detailed_whisper_api.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multi_speaker_csv(tmp_path):
    """Create a minimal multi-speaker results CSV with _conv columns."""
    df = pd.DataFrame(
        {
            "audio_id": ["clip1", "clip1", "clip2"],
            "speaker": ["A", "B", "A"],
            "best_method": ["method1", "method1", "method2"],
            "transcription": ["hello", "world", "test"],
            "asr_transcription": ["hello", "word", "tset"],
            "wer_conv": [0.0, 1.0, 1.0],
            "cer_conv": [0.0, 0.4, 0.5],
            "semantic_similarity_conv": [1.0, 0.6, 0.5],
            "poseidon_score_conv": [1.0, 0.5, 0.4],
            "snr_db": [22.0, 18.0, 8.0],
            "clipping_ratio": [0.0, 0.001, 0.03],
            "silence_ratio": [0.05, 0.2, 0.85],
            "snr_tier": ["High", "Medium", "Low"],
        }
    )
    path = tmp_path / "asr_eval_results_elevenlabs_api_20240101.csv"
    df.to_csv(path, index=False)
    return path
