"""Reference client for hayamimi's --input ws /ingest endpoint.

Streams a local wav file at real-time pace over a raw WebSocket connection
and prints whatever subtitle events come back. This is both the smoke-test
client used against realtime_transcribe.py --input ws, and a template for a
future phone or stackchan/ESP32 client: the wire protocol (ws_protocol.py)
needs no websocket library, just a plain TCP socket.

Usage:
    python scripts/ws_mic_client.py --wav testdata/eval_real/ja_01.wav \
        --host 127.0.0.1 --port 8766
"""
import argparse
import base64
import json
import os
import socket
import sys
import threading
import time
import wave

sys.stdout.reconfigure(encoding="utf-8")

from ws_protocol import (OP_BINARY, OP_CLOSE, OP_TEXT, build_handshake_request,
                         decode_frame, encode_frame)

CHUNK_S = 0.1  # seconds of audio per network frame


def ws_connect(host: str, port: int, path: str, timeout: float = 10.0) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    sock.sendall(build_handshake_request(host, port, path, key))
    buf = b""
    while b"\r\n\r\n" not in buf:
        data = sock.recv(4096)
        if not data:
            raise ConnectionError("server closed the connection during handshake")
        buf += data
    status_line = buf.split(b"\r\n", 1)[0]
    if b"101" not in status_line:
        raise ConnectionError(f"handshake failed: {status_line!r}")
    return sock


def recv_events(sock: socket.socket, stop: threading.Event, events: list):
    buf = bytearray()
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            data = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        if not data:
            break
        buf += data
        while True:
            result = decode_frame(bytes(buf))
            if result is None:
                break
            opcode, payload, consumed = result
            del buf[:consumed]
            if opcode == OP_TEXT:
                text = payload.decode("utf-8")
                events.append(text)
                print("<-", text)
            elif opcode == OP_CLOSE:
                stop.set()
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True, help="16-bit PCM wav file to stream")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--path", default="/ingest")
    ap.add_argument("--no-realtime", action="store_true",
                    help="send the whole file as fast as possible instead of at playback pace")
    ap.add_argument("--wait-final", type=float, default=8.0,
                    help="seconds to keep listening for events after the file finishes sending")
    ap.add_argument("--tail-silence", type=float, default=1.5, metavar="SEC",
                    help="seconds of silence to stream after the file, so the server's VAD sees "
                         "an actual pause and finalizes while we're still connected -- a real "
                         "mic-always-on client (phone/ESP32) does this implicitly; a one-shot "
                         "clip needs it spelled out, or the last utterance only finalizes once "
                         "we disconnect and it's too late for us to hear the result")
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as f:
        sr = f.getframerate()
        channels = f.getnchannels()
        assert f.getsampwidth() == 2, "expected 16-bit PCM wav"
        pcm = f.readframes(f.getnframes())

    sock = ws_connect(args.host, args.port, args.path)
    print(f"connected to ws://{args.host}:{args.port}{args.path}", file=sys.stderr)

    events: list = []
    stop = threading.Event()
    listener = threading.Thread(target=recv_events, args=(sock, stop, events), daemon=True)
    listener.start()

    handshake = json.dumps({"sr": sr, "format": "pcm_s16le", "channels": channels})
    sock.sendall(encode_frame(handshake.encode("utf-8"), opcode=OP_TEXT, mask=True))

    bytes_per_s = sr * channels * 2
    chunk_bytes = max(2, int(bytes_per_s * CHUNK_S))
    chunk_bytes -= chunk_bytes % 2
    # a real mic-always-on client (phone/ESP32) keeps streaming through
    # silence too, which is what lets the server's VAD ever see a pause and
    # finalize; append some so a one-shot clip behaves the same way instead
    # of just stopping mid-utterance from the VAD's point of view.
    payload = pcm + b"\x00" * (int(bytes_per_s * args.tail_silence) // 2 * 2)
    start = time.perf_counter()
    sent = 0
    pos = 0
    while pos < len(payload):
        chunk = payload[pos:pos + chunk_bytes]
        sock.sendall(encode_frame(chunk, opcode=OP_BINARY, mask=True))
        pos += len(chunk)
        sent += len(chunk)
        if not args.no_realtime:
            target = start + sent / bytes_per_s
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    print(f"sent {len(pcm)} bytes of PCM ({len(pcm) / bytes_per_s:.1f}s audio) "
          f"+ {args.tail_silence:.1f}s trailing silence", file=sys.stderr)

    time.sleep(args.wait_final)
    stop.set()
    try:
        sock.sendall(encode_frame(b"", opcode=OP_CLOSE, mask=True))
    except OSError:
        pass
    sock.close()

    parsed = [json.loads(e) for e in events]
    finals = [e for e in parsed if e.get("type") == "final"]
    refines = [e for e in parsed if e.get("type") == "refine"]
    print(f"received {len(events)} events: {len(finals)} final(s), {len(refines)} refine(s)",
          file=sys.stderr)
    for e in finals:
        print(f"  final [{e.get('lang')}] {e.get('text')}")
    for e in refines:
        print(f"  refine [{e.get('lang')}] {e.get('text')}")


if __name__ == "__main__":
    main()
