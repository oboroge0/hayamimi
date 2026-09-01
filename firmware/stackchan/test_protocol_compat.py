"""Protocol compatibility check for the stackchan firmware's framing.

The firmware itself doesn't run here (no ESP32 hardware in this environment
-- see README.md's "known limitations" for what still needs a real board).
What we *can* verify on the host is that the exact frame sequence the
firmware's main.cpp builds -- one masked WS text frame carrying
{"sr":16000,"format":"pcm_s16le","channels":1}, followed by masked binary
frames of 100ms / 3200-sample s16le PCM chunks (see CHUNK_SAMPLES in
main.cpp) -- is accepted by hayamimi's real ws_ingest.IngestServer and
lands in its audio_q unchanged. This is scripts/ws_mic_client.py's
sequence with the chunk size pinned to match the firmware instead of
CHUNK_S, so any framing drift between the two clients would show up here.

This does not exercise the ASR pipeline (no model load) -- IngestServer is
used standalone, exactly as realtime_transcribe.py wires it up before
handing audio_q to the VAD/ASR stage.
"""
import base64
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # firmware/stackchan
FIRMWARE = os.path.dirname(HERE)                           # firmware
ROOT = os.path.dirname(FIRMWARE)                            # repo root
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from ws_ingest import FLUSH, IngestServer  # noqa: E402
from ws_protocol import (OP_BINARY, OP_TEXT, build_handshake_request,  # noqa: E402
                          decode_frame, encode_frame)

SAMPLE_RATE = 16000
# Mirrors main.cpp: CHUNK_SAMPLES = SAMPLE_RATE / 10 (100ms chunks).
CHUNK_SAMPLES = SAMPLE_RATE // 10
CHUNK_BYTES = CHUNK_SAMPLES * 2  # s16le


def ws_connect(host: str, port: int, path: str) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    sock.sendall(build_handshake_request(host, port, path, key))
    buf = b""
    while b"\r\n\r\n" not in buf:
        data = sock.recv(4096)
        if not data:
            raise ConnectionError("server closed during handshake")
        buf += data
    status_line = buf.split(b"\r\n", 1)[0]
    assert b"101" in status_line, f"handshake failed: {status_line!r}"
    return sock


def recv_one_text_frame(sock: socket.socket, timeout: float = 5.0) -> dict:
    sock.settimeout(timeout)
    buf = bytearray()
    while True:
        data = sock.recv(4096)
        if not data:
            raise ConnectionError("server closed while waiting for a reply")
        buf += data
        result = decode_frame(bytes(buf))
        if result is None:
            continue
        opcode, payload, consumed = result
        del buf[:consumed]
        assert opcode == OP_TEXT
        return json.loads(payload.decode("utf-8"))


def test_firmware_framing_is_accepted_by_ingest_server():
    # IngestServer doesn't expose its bound port, so pick a free one up
    # front rather than binding to port 0.
    import contextlib
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    ingest = IngestServer("127.0.0.1", free_port, sample_rate=SAMPLE_RATE).start()
    time.sleep(0.2)  # let the asyncio server actually start listening

    sock = ws_connect("127.0.0.1", free_port, "/ingest")

    # 1. handshake JSON text frame, masked (client->server), exactly what
    #    main.cpp's sendHandshake() sends.
    handshake = json.dumps({"sr": SAMPLE_RATE, "format": "pcm_s16le", "channels": 1})
    sock.sendall(encode_frame(handshake.encode("utf-8"), opcode=OP_TEXT, mask=True))

    reply = recv_one_text_frame(sock)
    assert reply == {"type": "ready", "sr": SAMPLE_RATE}

    # 2. binary PCM frames, one CHUNK_SAMPLES chunk at a time -- same shape
    #    main.cpp's micTask()/sendBIN() calls produce.
    rng = np.random.default_rng(0)
    n_chunks = 5
    chunks = [rng.integers(-30000, 30000, size=CHUNK_SAMPLES, dtype=np.int16) for _ in range(n_chunks)]
    for chunk in chunks:
        payload = chunk.tobytes()
        assert len(payload) == CHUNK_BYTES
        sock.sendall(encode_frame(payload, opcode=OP_BINARY, mask=True))

    # 3. the server decodes each binary frame into audio_q as float32
    #    mono at the target sample rate (no resampling needed here since
    #    src_rate == server rate) -- verify every chunk arrived intact and
    #    in order.
    deadline = time.time() + 5
    received = []
    total_expected = sum(len(c) for c in chunks)
    total_received = 0
    while total_received < total_expected and time.time() < deadline:
        try:
            item = ingest.audio_q.get(timeout=0.5)
        except Exception:
            continue
        if item is FLUSH:
            continue
        received.append(item)
        total_received += len(item)

    assert total_received == total_expected, (
        f"expected {total_expected} samples total, got {total_received}"
    )

    got_f32 = np.concatenate(received) if received else np.array([], dtype=np.float32)
    # IngestServer's decode_pcm16 -> float32 in [-1, 1]; scale back to
    # int16 to compare against the chunks we sent.
    got_i16 = np.clip(np.round(got_f32.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)

    sock.close()

    sent_all = np.concatenate([c.astype(np.int64) for c in chunks])
    # allow the float32 round-trip's quantization error of +-1 LSB
    assert len(sent_all) == len(got_i16)
    diff = np.abs(sent_all - got_i16.astype(np.int64))
    assert diff.max() <= 1, f"PCM round-tripped through the server with drift > 1 LSB: {diff.max()}"
