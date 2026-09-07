# hayamimi stackchan client (M5Stack CoreS3)

Streams the CoreS3's built-in mic to hayamimi's `/ingest` WebSocket and
shows the returned subtitles on screen. Single board, no wiring required.

## Build / flash

Requires [PlatformIO](https://platformio.org/) (`pip install platformio`).

```
cd firmware/stackchan
cp include/config.h.example include/config.h   # then edit config.h
pio run                 # build
pio run -t upload       # build + flash (board connected over USB)
pio device monitor       # serial log, 115200 baud
```

`config.h` is gitignored -- it holds your Wi-Fi credentials and the
hayamimi server address, so it's never committed.

## Configuration (`include/config.h`)

| Macro | Meaning |
|---|---|
| `WIFI_SSID` / `WIFI_PASSWORD` | Wi-Fi the CoreS3 joins |
| `HAYAMIMI_HOST` | IP/hostname of the machine running hayamimi |
| `HAYAMIMI_PORT` | hayamimi's `/ingest` WS port -- **default 8766**, not the dashboard port (see below) |
| `HAYAMIMI_PATH` | always `/ingest` |

## Running hayamimi

On the machine hayamimi runs on:

```
python scripts/realtime_transcribe.py --input ws --serve
```

- `--input ws` opens the `/ingest` WebSocket the firmware streams PCM to.
  It binds `--ws-host` (default `0.0.0.0`) : `--ws-port` (default
  **8766**) -- point `HAYAMIMI_PORT` at this port.
- `--serve` (bare, defaults to port 8833) additionally starts the
  SSE dashboard/OBS-overlay HTTP server on a *separate* port. It's
  optional for the firmware itself -- audio sent to `/ingest` is
  broadcast to both the WS reply frames the firmware reads and the
  dashboard, off the same `SubtitleServer`, so `--serve` is only useful
  if you also want to watch the dashboard in a browser at the same time.

If both ports collide with something else on your network, pass
`--ws-port <port>` and update `HAYAMIMI_PORT` to match.

## Protocol

Implemented from `../../scripts/ws_protocol.py` and
`../../scripts/ws_ingest.py`, which are the source of truth -- do not
change the framing here without checking those first.

1. RFC 6455 WebSocket handshake to `ws://HAYAMIMI_HOST:HAYAMIMI_PORT/ingest`.
2. First frame: one JSON text frame `{"sr":16000,"format":"pcm_s16le","channels":1}`.
3. After that: binary frames of raw little-endian s16 PCM, sent
   continuously (including silence) for the life of the connection --
   the server's VAD needs to see actual pauses to finalize utterances.
4. Server replies with JSON text frames: `partial`, `final`,
   `translation`, `refine`, `ready`, `error`. The firmware shows `final`
   text as the main subtitle, `partial` dimmed underneath it, `refine`
   text replaces the last `final`, and `translation`/`error` show briefly
   in the status line.

## WebSocket library

[`links2004/WebSockets`](https://github.com/Links2004/arduinoWebSockets)
(PlatformIO registry name `links2004/WebSockets`), not
`gilmaimon/ArduinoWebsockets`. Reasons:

- Built-in reconnect handling (`setReconnectInterval`) -- this is an
  always-on mic client, so losing and regaining Wi-Fi/AP needs to be a
  non-event, not something the app has to hand-roll.
- Callback shape (`WStype_t type, uint8_t* payload, size_t length`)
  cleanly separates `WStype_TEXT` (server's JSON events) from
  `WStype_BINARY`/outgoing `sendBIN` for PCM, matching this protocol's
  text-then-binary-frames shape without extra plumbing.
- Long, stable maintenance history and wide usage on ESP32 Arduino
  projects, including other M5Stack builds.

`links2004/WebSockets` masks outgoing client frames automatically, which
is what `ws_protocol.py`'s server-side `decode_frame` requires (RFC 6455:
clients must mask, servers must not).

## Architecture

- `micTask` (FreeRTOS task pinned to core 0): blocking
  `M5.Mic.record()` calls, one 100ms/3200-sample chunk at a time, sent
  via `webSocket.sendBIN()` once the WS is connected and the JSON
  handshake has been sent.
- `loop()` (core 1, Arduino default): `webSocket.loop()` (drives
  reconnect + incoming frame parsing) and Wi-Fi reconnect checks.
- A mutex (`wsMutex`) guards every `webSocket.*` call since it's touched
  from both tasks; `WebSocketsClient` itself isn't documented as
  thread-safe.
- Display: an `M5Canvas` off-screen sprite is redrawn and pushed on every
  status/subtitle change (avoids flicker vs. drawing straight to the
  panel). Japanese text uses the bundled `lgfxJapanGothic` fonts (M5GFX).

## Known limitations / what still needs real hardware

This was built and build-verified on Windows without a physical CoreS3 --
no on-device testing was possible in this environment. Before trusting it
end-to-end, still verify on real hardware:

- Actual mic quality/gain -- `M5.Mic.config()` defaults were used as-is;
  levels may need tuning against a real room.
- Screen must stay on for the firmware to be useful (no attempt is made
  to keep the display awake beyond M5Unified's defaults) -- if CoreS3
  auto-dims/sleeps the panel, subtitles stop being visible even though
  streaming keeps running.
- Real Wi-Fi drop/reconnect behavior (`setReconnectInterval` + the
  `loop()`-level Wi-Fi status check are untested against actual AP
  roaming/dropouts).
- Long-run stability (memory, task scheduling) hasn't been soak-tested.
- `M5Canvas`/`lgfxJapanGothic` font rendering was only checked by reading
  M5GFX's API, not by seeing actual glyphs on a panel.
