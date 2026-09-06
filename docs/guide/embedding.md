# Embedding hayamimi's Python engine in another app

Moved out of the repository root `README.md` on 2026-09-03; the two sections
below are verbatim, so "above" in them still means "earlier in the README"
([`../../README.md`](../../README.md)).

For the Flutter/Dart side, the embedding guide is
[`mobile/hayamimi_core/README.md`](../../mobile/hayamimi_core/README.md) --
this file is the Python engine only.

## Embedding in another app

`scripts/realtime_transcribe.py`'s pieces (`RoutedASR`, `build_vad`,
`run_stream`) are importable, not just CLI-only: `RoutedASR(...)` and
`build_vad(...)` raise a catchable `asr_engine.ModelUnavailable` instead of
letting sherpa-onnx's C++ layer call `exit()` on a missing model path, and
`run_stream(..., stop_event=some_threading_Event)` accepts a
`threading.Event` cancellation token so a host app running the pipeline on
its own thread can stop it cleanly (`stop_event.set()`) without relying on
`KeyboardInterrupt`, which only works for the CLI's own process.

## Embedding: runtime control and structured events

Two gaps remained even with the pieces above importable. First, session
settings -- which language to force, whether to translate, how sensitive the
VAD (voice activity detector, the component that decides where an utterance
starts and ends) is -- could only be set once, at construction: a host app
whose user changed their mind mid-session had to tear down the whole engine
and rebuild it. Second, structured session events only existed at all behind
`--serve`: without it, the only way to learn that a model failed to load, or
that a segment's language just switched, was to scrape stderr and the
console's free-text diagnostic line, which a program can't parse reliably.

Both are addressed by the same mechanism. `realtime_transcribe.main()` (and
any app that constructs `subtitle_server.EventHub` directly) now always
creates an event hub, and every stage of the pipeline publishes to it
whether or not `--serve`'s HTTP server exists. `--serve` just adds that HTTP
server as an optional front end over the same hub, plus three JSON
endpoints backed by a `subtitle_server.RuntimeControls` bundle.

**Consuming events without HTTP.** An app importing this module directly
doesn't need `--serve`, or even a `SubtitleServer`, to see these events:
`hub.add_listener(callback)` registers a synchronous, in-process callback
invoked with the raw event dict on every publish. `hub.subscribe()` -- what
the dashboard/overlay pages above consume internally, over SSE at
`/events`, and what `scripts/ws_ingest.py` mirrors onto a WebSocket client
-- hands back a `queue.Queue` of the same events as JSON strings instead,
for a consumer that wants to poll rather than be called back.

**Event types.** Every event is a JSON object with a `type` key:

| `type` | Shape | When |
|---|---|---|
| `session_start` | `{"type":"session_start"}` | once, at the top of a session |
| `partial` | `{"type":"partial","text":str}` | an in-progress draft updates (about every 0.5s while speaking) |
| `final` | `{"type":"final","text":str,"lang":str,"speaker":str,"latency_ms":float\|null,"tier":str,"audio_s":float,"lid_ms":float\|null,"decode_ms":float\|null,"switched":bool}` | a VAD segment finalizes; `switched` is true only when `lang` differs from the previous final's (false for the session's first) |
| `translation` | `{"type":"translation","lang":str,"text":str}` | a `--translate` target finishes translating a ja final or refine line |
| `refine` | `{"type":"refine","text":str,"lang":str,"speaker":str,"audio_s":float}` | the second-pass re-decode of an utterance group lands |
| `model_load` | `{"type":"model_load","model":str,"phase":"start"\|"done","ms":float\|null}` | a recognizer/LID/punctuation/translator model starts or finishes loading; `model` is the engine's own short name (`rz`/`pz`/`sv`/`v3`/`omni`/`pja`/`lid`/`punct`, or `translator:<lang>`) |
| `model_fallback` | `{"type":"model_fallback","requested":str,"used":str,"reason":str}` | routing substituted a different tier because the one asked for is missing (e.g. a `--minimal` install), reported once per requested model per session |
| `warning` | `{"type":"warning","code":str,"message":str}` | a degraded-but-non-fatal condition; `code` is one of `hotwords_unencodable`, `segmentation_vad_unavailable`, `second_opinion_unavailable`, `diarization_failed` |
| `session_summary` | `{"type":"session_summary","stats":{...},"speakers":{...}\|null}` | at process shutdown and again right before a session reset; `stats` mirrors the `=== session summary: ... ===` console line's numbers, `speakers` mirrors the speaker-diagnostic console lines when `--speakers` is on (`null` otherwise) |
| `recluster` | `{"type":"recluster","time_s":float,"n_entries":int,"n_clusters":int,"mapping":{...}}` | `--speaker-global-recluster`'s end-of-session diagnostic actually runs |
| `session_reset` | `{"type":"session_reset"}` | a `POST /reset` (or a direct `reset_live_session()` call) finishes |

**Runtime control over HTTP (`--serve` only).** Three endpoints besides the
`/replacements`/`/itn_overrides` pair described above:

```bash
# read the live session's current configuration
curl http://localhost:8833/config

# force the session onto English, and stop requiring SenseVoice's own LID
# to agree before a language switch is accepted
curl -X POST http://localhost:8833/config \
  -d '{"lang": "en", "dual_confirm": false}'

# add Korean as a live translation target (any target NOT listed here is
# removed -- this replaces the whole set, like the other config keys)
curl -X POST http://localhost:8833/config -d '{"translate": ["en", "ko"]}'

# loosen VAD sensitivity: wait longer for silence before a segment finalizes
curl -X POST http://localhost:8833/config -d '{"vad": {"min_silence": 0.6}}'

# start a fresh conversation: forget every speaker and language-switch
# state, but keep every model resident (no reload cost)
curl -X POST http://localhost:8833/reset
```

`GET /config` returns `{"lang": null|str, "dual_confirm": bool, "punctuate":
bool, "lid_switch_confirm": int, "min_switch_s": float, "translate":
[str, ...], "vad": {"threshold": float, "min_silence": float, "max_speech":
float}}`. `POST /config` accepts any subset of those keys and applies each
through the matching `RoutedASR`/`TranslatorPool`/VAD setter; an invalid
value (an unroutable language code, a negative `min_switch_s`, an
unsupported translation target) answers `400` with that setter's own error
message rather than applying a partial or silently-clamped change. VAD
sensitivity is the one key that can't take effect immediately: sherpa-onnx
has no in-place setter for it, so a `vad` change is deferred until the
detector isn't in the middle of a speech segment, then it rebuilds -- a
change requested while someone is talking takes effect once they pause, not
at the moment of the request.

`POST /reset` clears the running session's speaker centroids, sticky/pending
language state, and refine/session statistics, without reloading any model
-- the next segment starts as if the process had just launched. Useful
between unrelated recordings inside one long-running process (a stream host
switching guests, a kiosk resetting between visitors), where paying the
model-load cost again would be wasted time.

The reset itself doesn't run on the HTTP handler thread: it has to run on
the same thread that's decoding audio, since it touches speaker/language
state that thread reads and writes with no locking of its own (running it
directly from a second thread was found, in review, to be able to raise an
exception on the decode thread mid-segment). It's queued instead, and picked
up the next time that thread reaches a safe point between audio chunks --
normally near-instant, since a chunk is only tens of milliseconds. `POST
/reset` answers `200 {"ok": true}` once the reset has actually run, or, if
ten seconds pass without a safe point to apply it at, `202 {"ok": false,
"pending": true}` -- it's still queued and will still apply, just later.
This matters mainly for `--input ws`: with no client currently sending
audio, nothing is reaching those chunk boundaries at all, so a reset
requested then won't apply until audio resumes.
