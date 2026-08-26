"""WebSocket audio ingest for realtime_transcribe.py's --input ws mode.

Protocol:
  1. HTTP upgrade to ws://host:port/ingest (RFC 6455 handshake).
  2. First WS message: one JSON text frame,
     {"sr": 16000, "format": "pcm_s16le", "channels": 1}.
  3. After that: binary frames of raw PCM (little-endian s16), one message
     per chunk, sent continuously for the life of the connection.
  4. Server replies with JSON text frames carrying the same subtitle events
     the SSE dashboard sees (partial/final/translation/refine), so a
     stackchan-class client can show its own results if it wants to. This
     also means audio sent to /ingest shows up on the existing /dashboard
     and OBS overlay for free -- both read off the same SubtitleServer.

Only one audio-producing client is accepted at a time; a second connection
gets a JSON {"type": "error", ...} frame and is closed immediately, keeping
the server dead simple (no need to mix multiple mic streams). A client that
disconnects just leaves the ingest queue idle -- the pipeline and server
keep running, and the next connection resumes feeding it.
"""
import asyncio
import json
import queue
import threading

import numpy as np

from audio_utils import decode_pcm16, resample_linear
from ws_protocol import (OP_BINARY, OP_CLOSE, OP_PING, OP_PONG, OP_TEXT,
                         build_handshake_response, decode_frame, encode_frame,
                         parse_handshake_request)

READ_CHUNK = 4096
MAX_HEADER_BYTES = 16384
INGEST_PATH = "/ingest"

# Sentinel pushed onto audio_q when a streaming client disconnects, so the
# pipeline can flush its in-progress VAD segment instead of waiting on
# silence that may never come once the client is gone (see
# realtime_transcribe.run_stream(), which treats any non-ndarray item on the
# queue as "flush now"). Any object works; identity is never compared.
FLUSH = object()


class IngestServer:
    """Accepts PCM over a WS connection, exposes it as a chunk queue.

    `audio_q` yields whatever-sized float32 mono arrays the network handed
    us, already resampled to `sample_rate`; realtime_transcribe.py's
    ws_chunks() re-chunks that into fixed VAD windows.
    """

    def __init__(self, host: str, port: int, sample_rate: int = 16000, subtitle_server=None):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.subtitle_server = subtitle_server
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._active = False
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> "IngestServer":
        ready = threading.Event()
        error: list[Exception] = []

        def run():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.create_task(self._serve(ready))
                self._loop.run_forever()
            except Exception as exc:  # surfaced to the caller via ready.wait
                error.append(exc)
                ready.set()

        threading.Thread(target=run, daemon=True, name="ws-ingest").start()
        ready.wait(timeout=5)
        if error:
            raise error[0]
        return self

    async def _serve(self, ready: threading.Event):
        server = await asyncio.start_server(self._handle, self.host, self.port)
        ready.set()
        async with server:
            await server.serve_forever()

    async def _read_handshake_head(self, reader: asyncio.StreamReader) -> bytes | None:
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = await reader.read(READ_CHUNK)
            if not chunk:
                return None
            buf += chunk
            if len(buf) > MAX_HEADER_BYTES:  # runaway header: bail
                return None
        return buf.split(b"\r\n\r\n", 1)[0]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            head = await self._read_handshake_head(reader)
            if head is None:
                writer.close()
                return
            req = parse_handshake_request(head)
            if req is None or req["path"] != INGEST_PATH:
                writer.write(b"HTTP/1.1 404 Not Found\r\n\r\n")
                await writer.drain()
                writer.close()
                return
            writer.write(build_handshake_response(req["key"]))
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            writer.close()
            return

        with self._lock:
            if self._active:
                try:
                    await self._send_json(writer, {"type": "error",
                                                    "message": "already streaming; only one audio "
                                                               "client is accepted at a time"})
                except (ConnectionError, OSError):
                    pass
                writer.close()
                return
            self._active = True

        forward_task = None
        got_handshake = False
        try:
            if self.subtitle_server is not None:
                forward_task = asyncio.ensure_future(self._forward_events(writer))

            buf = bytearray()
            channels = 1
            src_rate = self.sample_rate

            while True:
                data = await reader.read(READ_CHUNK)
                if not data:
                    break
                buf += data
                while True:
                    result = decode_frame(bytes(buf))
                    if result is None:
                        break
                    opcode, payload, consumed = result
                    del buf[:consumed]

                    if opcode == OP_CLOSE:
                        return
                    elif opcode == OP_PING:
                        writer.write(encode_frame(payload, opcode=OP_PONG))
                        await writer.drain()
                    elif opcode == OP_PONG:
                        pass
                    elif opcode == OP_TEXT and not got_handshake:
                        try:
                            cfg = json.loads(payload.decode("utf-8"))
                            src_rate = int(cfg.get("sr", self.sample_rate))
                            channels = max(1, int(cfg.get("channels", 1)))
                            fmt = cfg.get("format", "pcm_s16le")
                        except (ValueError, UnicodeDecodeError, TypeError):
                            await self._send_json(writer, {"type": "error",
                                                            "message": "bad handshake JSON"})
                            return
                        if fmt != "pcm_s16le":
                            await self._send_json(writer, {"type": "error",
                                                            "message": f"unsupported format "
                                                                       f"{fmt!r}, only pcm_s16le"})
                            return
                        got_handshake = True
                        await self._send_json(writer, {"type": "ready", "sr": self.sample_rate})
                    elif opcode == OP_BINARY:
                        if not got_handshake:
                            continue  # audio before the JSON handshake: drop and wait
                        samples = decode_pcm16(payload, channels)
                        if len(samples) == 0:
                            continue
                        if src_rate != self.sample_rate:
                            samples = resample_linear(samples, src_rate, self.sample_rate)
                        self.audio_q.put(samples)
                    # OP_TEXT after handshake / OP_CONT: ignored, not part of this protocol
        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            pass
        finally:
            if forward_task is not None:
                forward_task.cancel()
            with self._lock:
                self._active = False
            if got_handshake:
                self.audio_q.put(FLUSH)
            writer.close()

    async def _send_json(self, writer: asyncio.StreamWriter, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        writer.write(encode_frame(data, opcode=OP_TEXT))
        await writer.drain()

    async def _forward_events(self, writer: asyncio.StreamWriter):
        """Mirror the SubtitleServer's broadcast (partial/final/...) onto
        this WS connection, the same events the SSE dashboard receives."""
        q = self.subtitle_server.subscribe()
        loop = asyncio.get_event_loop()
        try:
            while True:
                data = await loop.run_in_executor(None, q.get)
                writer.write(encode_frame(data.encode("utf-8"), opcode=OP_TEXT))
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError, OSError):
            pass
        finally:
            self.subtitle_server.unsubscribe(q)
