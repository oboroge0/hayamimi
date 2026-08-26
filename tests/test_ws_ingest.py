"""Unit tests for the WS ingest protocol's pure functions: handshake parsing,
frame encode/decode, PCM decode, and resampling. No network, no models.

Run with: .venv/Scripts/python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from audio_utils import decode_pcm16, resample_linear
from ws_protocol import (OP_BINARY, OP_CLOSE, OP_PING, OP_TEXT, build_handshake_request,
                         build_handshake_response, compute_accept_key, decode_frame,
                         encode_frame, parse_handshake_request)

# ---- handshake --------------------------------------------------------------

# RFC 6455 section 1.3 worked example.
RFC_KEY = "dGhlIHNhbXBsZSBub25jZQ=="
RFC_ACCEPT = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_compute_accept_key_matches_rfc_example():
    assert compute_accept_key(RFC_KEY) == RFC_ACCEPT


def test_parse_handshake_request_extracts_path_and_key():
    req = (b"GET /ingest HTTP/1.1\r\n"
           b"Host: localhost:8766\r\n"
           b"Upgrade: websocket\r\n"
           b"Connection: Upgrade\r\n" +
           f"Sec-WebSocket-Key: {RFC_KEY}\r\n".encode("ascii") +
           b"Sec-WebSocket-Version: 13\r\n")
    parsed = parse_handshake_request(req)
    assert parsed is not None
    assert parsed["path"] == "/ingest"
    assert parsed["key"] == RFC_KEY


def test_parse_handshake_request_rejects_non_websocket():
    req = b"GET /ingest HTTP/1.1\r\nHost: localhost\r\n"
    assert parse_handshake_request(req) is None


def test_parse_handshake_request_rejects_missing_key():
    req = b"GET /ingest HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
    assert parse_handshake_request(req) is None


def test_build_handshake_response_contains_correct_accept():
    resp = build_handshake_response(RFC_KEY)
    assert b"101" in resp
    assert RFC_ACCEPT.encode("ascii") in resp


def test_build_handshake_request_roundtrips_through_parse():
    req = build_handshake_request("127.0.0.1", 8766, "/ingest", RFC_KEY)
    parsed = parse_handshake_request(req)
    assert parsed["path"] == "/ingest"
    assert parsed["key"] == RFC_KEY


# ---- frame encode/decode ----------------------------------------------------

def test_frame_roundtrip_unmasked_binary():
    payload = bytes(range(256)) * 4  # > 126 bytes, exercises 16-bit length
    frame = encode_frame(payload, opcode=OP_BINARY, mask=False)
    result = decode_frame(frame)
    assert result is not None
    opcode, out, consumed = result
    assert opcode == OP_BINARY
    assert out == payload
    assert consumed == len(frame)


def test_frame_roundtrip_masked_text():
    payload = '{"sr": 16000, "format": "pcm_s16le", "channels": 1}'.encode("utf-8")
    frame = encode_frame(payload, opcode=OP_TEXT, mask=True)
    opcode, out, consumed = decode_frame(frame)
    assert opcode == OP_TEXT
    assert out == payload
    assert consumed == len(frame)


def test_frame_masked_bytes_differ_from_plaintext_on_wire():
    payload = b"\x00" * 32  # all-zero payload would be trivially unmasked otherwise
    frame = encode_frame(payload, opcode=OP_BINARY, mask=True)
    header_len = 2 + 4  # short length + 4-byte mask key
    assert frame[header_len:] != payload  # masked on the wire
    assert decode_frame(frame)[1] == payload  # but decodes back correctly


def test_decode_frame_incomplete_returns_none():
    payload = b"x" * 50
    frame = encode_frame(payload, opcode=OP_BINARY)
    assert decode_frame(frame[:5]) is None


def test_decode_frame_two_frames_in_one_buffer():
    f1 = encode_frame(b"one", opcode=OP_TEXT)
    f2 = encode_frame(b"two", opcode=OP_TEXT)
    buf = f1 + f2
    opcode1, payload1, consumed1 = decode_frame(buf)
    assert (opcode1, payload1) == (OP_TEXT, b"one")
    opcode2, payload2, consumed2 = decode_frame(buf[consumed1:])
    assert (opcode2, payload2) == (OP_TEXT, b"two")
    assert consumed1 + consumed2 == len(buf)


def test_decode_frame_large_payload_uses_64bit_length():
    payload = b"z" * 70000  # forces the 127 / 64-bit extended length form
    frame = encode_frame(payload, opcode=OP_BINARY)
    opcode, out, consumed = decode_frame(frame)
    assert opcode == OP_BINARY
    assert len(out) == 70000
    assert consumed == len(frame)


def test_frame_close_and_ping_opcodes_roundtrip():
    for op in (OP_CLOSE, OP_PING):
        frame = encode_frame(b"", opcode=op)
        opcode, payload, _ = decode_frame(frame)
        assert opcode == op
        assert payload == b""


# ---- PCM decode --------------------------------------------------------------

def test_decode_pcm16_mono_roundtrip():
    src = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    out = decode_pcm16(src.tobytes(), channels=1)
    expected = src.astype(np.float32) / 32768.0
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_decode_pcm16_drops_trailing_odd_byte():
    src = np.array([100, 200], dtype=np.int16).tobytes() + b"\x7f"  # one stray byte
    out = decode_pcm16(src, channels=1)
    assert len(out) == 2  # stray trailing byte silently dropped, no crash


def test_decode_pcm16_averages_stereo_to_mono():
    # interleaved L/R: (1000, -1000) pairs -> average 0 for every frame
    src = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
    out = decode_pcm16(src.tobytes(), channels=2)
    assert len(out) == 2
    np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)


def test_decode_pcm16_empty_input():
    assert len(decode_pcm16(b"", channels=1)) == 0


# ---- resampling ---------------------------------------------------------------

def test_resample_linear_identity_when_rates_match():
    samples = np.arange(10, dtype=np.float32)
    out = resample_linear(samples, 16000, 16000)
    np.testing.assert_array_equal(out, samples)


def test_resample_linear_upsamples_to_expected_length():
    samples = np.zeros(8000, dtype=np.float32)  # 1s @ 8kHz
    out = resample_linear(samples, 8000, 16000)
    assert len(out) == 16000  # 1s @ 16kHz


def test_resample_linear_downsamples_to_expected_length():
    samples = np.zeros(48000, dtype=np.float32)  # 1s @ 48kHz
    out = resample_linear(samples, 48000, 16000)
    assert len(out) == 16000


def test_resample_linear_preserves_a_ramp_shape():
    # a linear ramp resampled linearly should still be (close to) linear
    samples = np.linspace(0.0, 1.0, 1600, dtype=np.float32)  # 0.1s @ 16kHz
    out = resample_linear(samples, 16000, 8000)
    assert len(out) == 800
    np.testing.assert_allclose(out[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(out[-1], 1.0, atol=1e-2)
