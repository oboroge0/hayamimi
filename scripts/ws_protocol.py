"""Minimal RFC 6455 WebSocket framing/handshake -- no external ws library.

hayamimi's ingest endpoint targets bare-metal clients (ESP32 firmware) that
won't carry a full websocket stack, so both the server (ws_ingest.py) and
the reference test client (ws_mic_client.py) implement only the subset that
matters here: single, unfragmented text/binary/close/ping/pong frames, no
compression, no extensions. This is intentionally not a general-purpose
WebSocket implementation.
"""
import base64
import hashlib
import os
import struct

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def compute_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def parse_handshake_request(head: bytes) -> dict | None:
    """Parse an HTTP upgrade request's head (everything before the blank
    line that ends the headers) into {"method", "path", "key"}, or None if
    it isn't a usable WebSocket upgrade request.
    """
    try:
        text = head.decode("iso-8859-1")
    except UnicodeDecodeError:
        return None
    lines = text.split("\r\n")
    if not lines or not lines[0]:
        return None
    parts = lines[0].split()
    if len(parts) < 2:
        return None
    method, path = parts[0], parts[1]
    headers = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        headers[k.strip().lower()] = v.strip()
    if headers.get("upgrade", "").lower() != "websocket":
        return None
    key = headers.get("sec-websocket-key")
    if not key:
        return None
    return {"method": method, "path": path, "key": key}


def build_handshake_response(client_key: str) -> bytes:
    accept = compute_accept_key(client_key)
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode("ascii")


def build_handshake_request(host: str, port: int, path: str, key: str) -> bytes:
    return (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")


def encode_frame(payload: bytes, opcode: int = OP_BINARY, mask: bool = False) -> bytes:
    """Build one complete, unfragmented WS frame.

    `mask=True` for client->server frames -- RFC 6455 requires clients to
    mask every frame they send; servers must not mask theirs.
    """
    fin_opcode = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = bytes([fin_opcode, length | (0x80 if mask else 0)])
    elif length < 65536:
        header = bytes([fin_opcode, 126 | (0x80 if mask else 0)]) + struct.pack(">H", length)
    else:
        header = bytes([fin_opcode, 127 | (0x80 if mask else 0)]) + struct.pack(">Q", length)
    if not mask:
        return header + payload
    mask_key = os.urandom(4)
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return header + mask_key + masked


def decode_frame(buf: bytes):
    """Try to decode one frame from the front of `buf`.

    Returns (opcode, payload, consumed_bytes) on success, or None if `buf`
    doesn't yet hold a complete frame. The caller drops `consumed_bytes`
    from the front of its buffer and calls again in case more than one
    frame arrived in the same read.
    """
    if len(buf) < 2:
        return None
    b0, b1 = buf[0], buf[1]
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    pos = 2
    if length == 126:
        if len(buf) < pos + 2:
            return None
        length = struct.unpack(">H", buf[pos:pos + 2])[0]
        pos += 2
    elif length == 127:
        if len(buf) < pos + 8:
            return None
        length = struct.unpack(">Q", buf[pos:pos + 8])[0]
        pos += 8
    mask_key = b""
    if masked:
        if len(buf) < pos + 4:
            return None
        mask_key = buf[pos:pos + 4]
        pos += 4
    if len(buf) < pos + length:
        return None
    payload = buf[pos:pos + length]
    if masked:
        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return opcode, bytes(payload), pos + length
