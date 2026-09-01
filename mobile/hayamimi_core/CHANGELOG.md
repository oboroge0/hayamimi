## 0.1.0

First release: `hayamimi_core` extracted out of `mobile/`'s app-only codebase
into a standalone package a third-party Flutter app can embed.

* **On-device live transcription.** `HayamimiLive` (and the lower-level
  `LiveTranscriber` it wraps) run mic → Silero VAD → a
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) offline ASR model,
  emitting a draft ("発話中の暫定字幕") line while a VAD segment is still in
  progress and a finalized line once it closes. An optional two-pass
  "refine" (清書) step re-decodes a buffered group of finals for a cleaner
  transcript, on a silence gap, a buffered-duration ceiling, or a manual
  `refineNow()` call. Passing `routingProfile: RoutingProfile.jaSenseVoice`
  switches from the single fixed-language model to per-segment routing
  between ReazonSpeech (ja) and SenseVoice (en/zh/ko/yue), arbitrated by a
  whisper-tiny language-ID probe and SenseVoice's own dual-confirmation
  policy.
* **Remote client.** `HayamimiRemote` (and `RemoteTranscriber`) stream this
  device's mic to a hayamimi server running on a PC (`--input ws --serve`)
  over its `/ingest` WebSocket and surface the subtitle events streamed
  back, so the PC's full pipeline (multi-language routing, refine,
  translation) does the actual recognition. `RemoteFinalEvent`/
  `FinalSubtitleEvent` and `RemoteRefineEvent`/`RefineSubtitleEvent` carry
  the same `audioSeconds`/`switched`/`latencyMs`/`punctuated` fields the
  on-device path reports (`latencyMs` was previously parsed for `final` but
  not `refine`), and `RemoteModelLoadEvent`/`RemoteSessionResetEvent` map
  onto `ModelLoadSubtitleEvent`/`SessionResetSubtitleEvent` the same way.
  `model_fallback`/`warning`/`session_summary`/`recluster` frames from the
  desktop pipeline still fall back to `RemoteUnknownEvent` — this package
  doesn't have typed events for them yet.
* **LAN subtitle broadcast server.** `SubtitleBroadcastServer` re-broadcasts
  either facade's events as SSE plus a transparent overlay page, speaking
  the same protocol `scripts/subtitle_server.py` does, so an OBS browser
  source or web client that already works against the desktop app works
  against an embedding app too. It gained `bindAddress` (still defaults to
  `InternetAddress.anyIPv4` — the point of this server is LAN reachability)
  and `allowOrigin` (still defaults to `'*'`) constructor parameters.
* **Runtime configuration, events, and session reset**
  ([#29](https://github.com/oboroge0/hayamimi/issues/29)). Every pacing
  knob that used to be a hardcoded `default*` constant
  (`draftIntervalSeconds`, `draftWindowSeconds`, `minDraftAudioSeconds`,
  `autoRefineSilenceSeconds`, `autoRefineMaxBufferedSeconds`,
  `refineBufferMaxSeconds`) is now a `LiveTranscriber`/`HayamimiLive`
  constructor parameter and a mid-session-settable property, validated
  (positive, finite) with `ArgumentError`. `start`/`startDebugWavStream`
  also gained `decodingMethod` (recognizer search algorithm),
  `vadSensitivity` (a new `VadSensitivity` class covering Silero VAD's
  threshold/min-silence/min-speech/max-speech knobs, with a runtime
  `setVadSensitivity` that rebuilds and safely swaps the VAD without
  reloading any other model or touching an in-progress segment), and
  `hotwordsFile`/`hotwordsScore` (sherpa-onnx hotword biasing). Added
  `resetSession()`, which clears the refine buffer, draft state, and (for a
  routed session) the language lock, without reloading any native model —
  useful for a host app's "start a new conversation" action. New event
  types: `LiveTranscriptEntry`/`FinalSubtitleEvent`/`RefineSubtitleEvent`
  gained `audioSeconds` (wire: `audio_s`), `FinalSubtitleEvent` gained
  `switched` (whether this segment is why a routed session's language
  changed), and `ModelLoadSubtitleEvent` (wire: `model_load`) /
  `SessionResetSubtitleEvent` (wire: `session_reset`) were added — see
  "Runtime configuration and session control" and "Events" in the README
  for the full parameter and JSON-shape reference. Three bugs surfaced
  while building this were fixed alongside it: a native-handle leak where
  `start()`/`startDebugWavStream()` left a partially built native state
  (e.g. a good recognizer load followed by a bad VAD model file) allocated
  with `isRunning` still `false` instead of freeing it before rethrowing;
  `setVadSensitivity()` losing its effect when called while `start()` was
  still loading, instead of being queued and applied once loading finished;
  and a race where a `setVadSensitivity()` rebuild still in flight when a
  session was stopped and immediately restarted could silently replace the
  *new* session's freshly built VAD, now caught by a monotonically
  increasing generation counter (`isVadBuildStale`) that discards the stale
  build.
* **Decoding off the UI isolate**
  ([#24](https://github.com/oboroge0/hayamimi/issues/24)). Every sherpa-onnx
  call is synchronous FFI, so decoding on the isolate driving the
  microphone stream — for a host app, Flutter's UI isolate — stopped the
  app painting for the length of each decode, and much longer during a
  refine pass over a whole buffered group. Each session now spawns one
  persistent decode worker isolate that owns that session's recognizer
  handles from first load to last free; requests reach it as messages
  carrying the audio, and it serves them one at a time, the same
  single-flight guarantee the old synchronous code got for free. Because
  results now come back later rather than inline, the queue in front of the
  worker follows explicit rules: finals are always queued, never dropped,
  and emitted in the order their segments closed; drafts are skipped while
  anything is outstanding and discarded on arrival if their segment has
  since produced a final; refine passes coalesce, and claim their audio
  when the request is actually sent rather than when it was asked for (so a
  segment still decoding when you tap 清書 joins that group). `refineNow()`'s
  future now completes once that pass's result has been emitted, and a
  second call while one is still queued awaits the same pass;
  `resetSession()` completes on the worker's acknowledgement; and
  `decoding` reports whether *anything* is outstanding, so it emits one
  `true`/`false` pair per burst rather than one per decode. If the decode
  worker dies mid-session, a `LiveTranscriberException` is emitted on
  `LiveTranscriber.errors` (an `ErrorSubtitleEvent` on
  `HayamimiLive.events`), microphone capture is torn down, and the object
  is left ready to `start` again — the same handling a revoked microphone
  gets; a single decode failing inside the worker is reported the same way
  but doesn't stop the session. `stop`/`dispose` wait for the worker's
  shutdown acknowledgement and kill the isolate if it doesn't answer; see
  "Threading / known limitations" in the README for why nothing is freed by
  force in that case. New exported building blocks, none of which a caller
  of `HayamimiLive`/`LiveTranscriber` has to touch, back this: the wire
  protocol between the caller's isolate and the worker
  (`decode_protocol.dart`), the pure queue policy (`decode_scheduler.dart`,
  `DecodeScheduler`), the caller's-isolate half (`decode_session.dart`,
  `DecodeSession`), the worker itself (`decode_worker.dart`,
  `DecodeWorker`/`IsolateDecodeWorker`), and a `LiveVad` interface plus its
  Silero-via-sherpa-onnx implementation (`live_vad.dart`) — all three exist
  so a whole session can be exercised without a device, and all default to
  production behavior via `LiveTranscriber`'s new `decodeWorkerFactory`/
  `vadFactory` constructor parameters and `RoutedRecognizerSet.build`'s new
  `loadOffIsolate` one. Two smaller pieces of decode-path plumbing moved
  off the FFI-backed sherpa-onnx package as part of this: `isSegmentWorthDecoding`
  now takes the segment's `Float32List` samples rather than a
  `sherpa_onnx.SpeechSegment`, and `startDebugWavStream` reads its wav file
  with this package's own pure-Dart PCM16 parser instead of sherpa-onnx's
  `readWave` (same accepted format — 16-bit PCM — but a wrong channel count
  is now reported explicitly rather than as a generic read failure).
* **Japanese punctuation restoration**
  ([#15](https://github.com/oboroge0/hayamimi/issues/15), refine only). The
  desktop pipeline punctuates its refine (清書) output with
  `scripts/punct_ja.py` before anything sees it, so desktop captions read
  as sentences; this package's refine output previously had no 、 and no
  。 at all. `PunctuatorJa.restore()` is a Dart port of that script — the
  same character-level BERT model, the same thresholds, the same
  rule-based ？ pass — pinned to the reference implementation by
  `test/punct/punct_ja_parity_test.dart`, which replays 40 FLEURS ja
  sentences plus 11 edge cases and asserts the two produce identical text.
  Passing a `JaPunctuation` to `LiveTranscriber.start`/
  `HayamimiLive.start` (or their `startDebugWavStream` counterparts) turns
  it on for that session's refine output only — that is where the desktop
  pipeline punctuates, and the pass with the context for it, since a final
  covers one speech segment and a draft part of one still being spoken, so
  sentence boundaries there would fall wherever the speaker paused. A
  `RoutingProfile.jaSenseVoice` session punctuates the refines that came
  back as Japanese and leaves the rest alone; a plain single-model session
  has no language tag to test, so passing this to one is itself the
  statement that its model transcribes Japanese. `LiveTranscriptEntry` and
  `RefineSubtitleEvent` gained `punctuated` (and `"punctuated"` in that
  event's JSON) so a consumer can tell text the punctuation model wrote
  from text the recognizer produced; the refine pass's "too short, fall
  back to the fast finals" guard now compares lengths with the restored
  marks stripped, since they are characters nobody said and would
  otherwise excuse a truncated re-decode. It runs the model through ONNX
  Runtime's C API over `dart:ffi`, borrowing the runtime `sherpa_onnx`
  already loads rather than adding a second `libonnxruntime.so` to the app
  (two in one Android process is a known crash,
  [k2-fsa/sherpa-onnx#3261](https://github.com/k2-fsa/sherpa-onnx/issues/3261)),
  and loads in the decode worker isolate rather than the caller's, for the
  same reason every other decode moved there. New dependencies: `unorm_dart`
  (NFKC normalization, which `dart:core` has no equivalent of) and `ffi`.
  The float16 model file (181.8 MB) is still not downloadable —
  `ModelDownloader`/`downloadProfile` has no entry for it — so a host app
  has to place it and its `vocab.txt` on the device itself; see "Japanese
  punctuation restoration" in the README for the model-placement and
  platform-status details, and "Known limitations" below.
* **Bench and eval utilities.** `BenchRunner` measures offline WAV→RTF for a
  single zipformer-transducer model; `ManifestEvalRunner` batch-decodes a
  manifest of labeled clips (plain or through `RoutingProfile.jaSenseVoice`
  routing) for CER/accuracy comparison against the PC pipeline's own
  manifest runs. Both are debug/dev tools, not part of the embedding
  surface a host app needs — see the `kDebugMode`-gated call sites in the
  `mobile/` reference app's Bench tab.

### Known limitations

* The punctuation model has not been run on a device in this release: its
  wiring was verified on a Windows host (`flutter analyze`/`flutter test`,
  including the parity test above), but what a phone's ONNX Runtime does,
  and what `restore()` costs on an ARM CPU, is unverified. It's also
  refused by default on iOS — `sherpa_onnx` links ONNX Runtime as an
  xcframework and it hasn't been confirmed that `OrtGetApiBase` stays
  exported from the app binary, so `PunctuatorJa.load` throws rather than
  guess unless a caller passes `libraryPath: OrtLibrary.processSymbols`.
* This package does not observe the app lifecycle: a host app that wants
  `stop()` on background and `start()` on resume has to wire that up
  itself from its own `WidgetsBindingObserver`.
* `textTransform` is an insertion point, not an implementation: CJK
  inverse-text-normalization (kanji numerals → arabic digits) and user
  find/replace dictionaries, both present on the desktop pipeline, are not
  ported to Dart — a host app that wants them supplies its own callback.
* `FinalSubtitleEvent.speaker` is always empty; there is no on-device
  speaker diarization yet.
* `translation` is reserved for wire compatibility with the desktop
  pipeline's machine translation frames but is never emitted by this
  package.
* No on-device frame-timing or jank measurement was taken for the
  decode-worker migration above — the claim is about where the work runs,
  not a measured improvement in any particular app.
