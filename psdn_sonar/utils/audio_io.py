"""Shared ffmpeg-by-path audio decoding.

Two halves of a run decode audio independently: the pipeline ASR adapters
hand ffmpeg the file path (issue #182), while the audio-quality metrics went
through ``librosa.load``, whose libsndfile build reads neither AAC nor ALAC.
The same M4A file was readable to transcription and unreadable to the
quality metrics (issue #206). Both paths now share this decoder.
"""

import subprocess

import numpy as np


def decode_audio_ffmpeg(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Decode *audio_path* to a mono float32 array at *sampling_rate* by
    handing ffmpeg the file path.

    The path matters: pushed bytes on stdin, an MP4-family container
    (m4a/mp4 — the default recording format on iOS) cannot be demuxed from
    a non-seekable pipe (issue #182). A path input is seekable, so
    everything the installed ffmpeg can read, this can. When decoding
    genuinely fails, the error names ffmpeg and carries its stderr instead
    of claiming the file is malformed.
    """
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode '{audio_path}': {stderr or f'exit code {proc.returncode}'}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(
            f"ffmpeg decoded no audio from '{audio_path}': {stderr or 'the file has no decodable audio stream'}"
        )
    return audio
