"""Pure audio helpers shared by the realtime pipeline and the WS ingest path.

Kept dependency-free (numpy only) so PCM decode/resample can run without any
extra audio codec library on either the server or the ESP32/phone-facing
test client (scripts/ws_mic_client.py).
"""
import numpy as np


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample.

    Good enough for speech-band PCM from a cheap mic (ESP32 I2S, phone mic
    at an odd sample rate); this is NOT a substitute for a proper
    polyphase/sinc resampler if a project ever needs broadcast-quality
    audio. No scipy/soxr/librosa dependency exists in this project yet, so
    this mirrors the same trade-off realtime_transcribe.py already made for
    --wav file resampling.
    """
    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    dst_n = int(round(duration * dst_rate))
    if dst_n <= 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.arange(len(samples)) / src_rate
    dst_x = np.arange(dst_n) / dst_rate
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def decode_pcm16(data: bytes, channels: int = 1) -> np.ndarray:
    """Raw little-endian s16 PCM bytes -> float32 mono samples in [-1, 1].

    Drops a trailing odd byte instead of raising: a flaky WiFi client
    (ESP32) can hand us a chunk boundary that lands mid-sample, and losing
    one sample is much better than killing the connection over it.
    """
    n = len(data) - (len(data) % 2)
    samples = np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1 and len(samples) >= channels:
        usable = len(samples) - (len(samples) % channels)
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples
