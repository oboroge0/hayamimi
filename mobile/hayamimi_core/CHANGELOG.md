## 0.1.0

* Added `PunctuatorJa`: Japanese punctuation restoration for the mobile
  pipeline. The desktop pipeline punctuates its refine ("清書") output with
  `scripts/punct_ja.py`, so desktop captions read as sentences while this
  package's refine output arrived with no 、 and no 。 at all. `restore()`
  is a Dart port of that script — the same character-level BERT model, the
  same thresholds, the same rule-based ？ pass — pinned to the reference
  implementation by `test/punct/punct_ja_parity_test.dart`, which replays
  40 FLEURS ja sentences plus 11 edge cases and asserts the two produce
  identical text. It runs the model through ONNX Runtime's C API over
  `dart:ffi`, borrowing the runtime `sherpa_onnx` already loads rather than
  adding a second `libonnxruntime.so` to the app (two in one Android
  process is a known crash, k2-fsa/sherpa-onnx#3261). New dependencies:
  `unorm_dart` (NFKC, which `dart:core` has no equivalent of) and `ffi`.
  See "Japanese punctuation restoration" in the README for the platform
  status and what is still unverified.
* `LiveTranscriber.start`/`startDebugWavStream` and `HayamimiLive.start`/
  `startDebugWavStream` take an optional `JaPunctuation`, which turns the
  punctuation model above on for that session's refine ("清書") output —
  the second half of
  [#15](https://github.com/oboroge0/hayamimi/issues/15). Passing nothing
  keeps the previous behaviour exactly. Only refine results are
  punctuated: that is where the desktop pipeline punctuates, and a final
  covers one speech segment while a draft covers part of one still being
  spoken, so sentence boundaries there would fall wherever the speaker
  paused. A `RoutingProfile.jaSenseVoice` session punctuates the refines
  that came back as Japanese and leaves the rest alone; a plain
  single-model session has no language tag to test, so passing this to one
  is itself the statement that its model transcribes Japanese (do not pass
  it to a plain session with a non-Japanese model). The model loads in the
  decode worker isolate, after the recognizers, and is reported on
  `modelLoads`/`ModelLoadSubtitleEvent` as `model: "punct"`: `restore()` is
  a synchronous native call, and running it on the caller's isolate would
  hand back the pause that moving decoding off it removed. Failing to load
  it fails `start()` with a message naming the file, rather than starting a
  session that quietly produces unpunctuated text. `LiveTranscriptEntry`
  and `RefineSubtitleEvent` gained `punctuated` (and `"punctuated"` in that
  event's JSON, beside every key that was already there) so a consumer can
  tell text the punctuation model wrote from text the recognizer produced.
  The refine pass's "too short, fall back to the fast finals" guard now
  compares lengths with the restored marks stripped, since they are
  characters nobody said and would otherwise excuse a truncated re-decode.
  The float16 model file (181.8 MB) is still not downloadable —
  `ModelDownloader` has no entry for it — so a host app has to place it and
  its `vocab.txt` on the device itself. No device or emulator run happened
  in this change: what a phone's ONNX Runtime does, and what `restore()`
  costs on an ARM CPU, is still unverified.
* Initial extraction from `mobile/`'s app-only codebase: on-device live
  transcription (`HayamimiLive`/`LiveTranscriber`), a remote-server client
  (`HayamimiRemote`/`RemoteTranscriber`), a LAN subtitle broadcast server
  (`SubtitleBroadcastServer`), and offline WAV benchmarking (`BenchRunner`).
* Every pacing knob that used to be a hardcoded `default*` constant
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
  `hotwordsFile`/`hotwordsScore` (sherpa-onnx hotword biasing). See
  "Runtime configuration and session control" in the README.
* Added `resetSession()` on `LiveTranscriber`/`HayamimiLive`: clears the
  refine buffer, draft state, and (for a `RoutingProfile.jaSenseVoice`
  session) the routed language lock, without reloading any native model —
  useful for a host app's "start a new conversation" action.
* Decoding no longer runs on the caller's isolate. Every per-utterance
  decode used to be a synchronous FFI call made on whichever isolate drove
  the microphone stream — for a host app, Flutter's UI isolate — so the app
  stopped painting for the length of each decode, and for much longer
  during a refine ("清書") pass over a whole buffered group. Each session
  now spawns one persistent decode worker isolate that owns that session's
  recognizers (for `RoutingProfile.jaSenseVoice`, all three of them,
  including the whisper-tiny language identification and the sticky-language
  state) from first load to last free; requests reach it as messages
  carrying the audio. What is left on the caller's isolate is microphone
  capture, one Silero VAD `acceptWaveform` per 32 ms frame, and event
  dispatch. Decode latency itself is unchanged — the work moved, it did not
  shrink — and no on-device jank measurement was taken for this change.
* Because decodes are now answered later rather than inline, the queue in
  front of the worker follows explicit rules: finals are always queued,
  never dropped, and emitted in the order their segments closed; drafts are
  skipped while anything is outstanding and discarded on arrival if their
  segment has since produced a final; refine passes coalesce, and claim
  their audio when the request is actually sent rather than when it was
  asked for (so a segment still decoding when you tap 清書 joins that
  group). `refineNow()`'s future now completes once that pass's result has
  been emitted, and a second call while one is still queued awaits the same
  pass. `resetSession()` now round-trips the clear through the worker and
  completes on its acknowledgement. `decoding` reports whether *anything*
  is outstanding, so it emits one `true`/`false` pair per burst rather than
  one per decode.
* Added handling for the decode worker isolate dying mid-session: a
  `LiveTranscriberException` on `LiveTranscriber.errors` (an
  `ErrorSubtitleEvent` on `HayamimiLive.events`), microphone capture torn
  down, and the object left ready to `start` again — the same handling a
  revoked microphone gets. A single decode failing inside the worker is
  reported the same way but does not stop the session. `stop`/`dispose`
  wait for the worker's shutdown acknowledgement and kill the isolate if it
  does not answer; see "Threading / known limitations" in the README for
  why nothing is freed by force in that case.
* New exported building blocks for the above, none of which a caller of
  `HayamimiLive`/`LiveTranscriber` has to touch: `decode_protocol.dart`
  (the worker message types and their encoding), `decode_scheduler.dart`
  (`DecodeScheduler`, the pure queue policy), `decode_session.dart`
  (`DecodeSession`, the caller's-isolate half) and `decode_worker.dart`
  (`DecodeWorker`/`IsolateDecodeWorker`), and `live_vad.dart` (`LiveVad`,
  the voice-activity detector a session feeds, plus the
  Silero-via-sherpa-onnx implementation). `LiveTranscriber`'s constructor
  gained `decodeWorkerFactory` and `vadFactory` parameters, and
  `RoutedRecognizerSet.build` a `loadOffIsolate` one — all three exist so a
  whole session can be exercised without a device, and all three default to
  production behaviour.
* `isSegmentWorthDecoding` now takes the segment's `Float32List` samples
  rather than a `sherpa_onnx.SpeechSegment`, so neither it nor
  `LiveTranscriber`'s segment handling depends on the FFI-backed package's
  types any more.
* `startDebugWavStream` reads its wav file with this package's own
  pure-Dart PCM16 parser instead of sherpa-onnx's `readWave`. The accepted
  format is unchanged (16-bit PCM), a wrong channel count is now reported
  explicitly rather than as a generic read failure, and the debug streaming
  path no longer touches FFI outside the models themselves.
* `LiveTranscriptEntry`/`FinalSubtitleEvent`/`RefineSubtitleEvent` gained
  `audioSeconds` (wire: `audio_s`) — the segment's or refined group's audio
  duration — and `FinalSubtitleEvent` gained `switched` — whether this
  segment is why a routed session's language changed. Added
  `ModelLoadSubtitleEvent` (wire: `model_load`, one per native model's
  load start/done, surfaced on `HayamimiLive.events` and
  `LiveTranscriber.modelLoads`) and `SessionResetSubtitleEvent` (wire:
  `session_reset`, emitted by `resetSession()`). See "Events" in the
  README for the full JSON shape table.
* `SubtitleBroadcastServer` gained `bindAddress` (still defaults to
  `InternetAddress.anyIPv4` — the point of this server is LAN reachability)
  and `allowOrigin` (still defaults to `'*'`) constructor parameters.
* `RemoteFinalEvent`/`FinalSubtitleEvent` (via `HayamimiRemote`) now carry
  `audioSeconds`/`switched` from a remote server the same way the on-device
  path does; `RemoteRefineEvent`/`RefineSubtitleEvent` gained
  `audioSeconds`, and `RemoteRefineEvent` also gained `latencyMs` (it was
  parsed for `final` but never for `refine`). Added `RemoteModelLoadEvent`/
  `RemoteSessionResetEvent`, mapped onto `ModelLoadSubtitleEvent`/
  `SessionResetSubtitleEvent` on `HayamimiRemote.events` the same way the
  on-device versions are. `model_fallback`/`warning`/`session_summary`/
  `recluster` frames from the desktop pipeline still fall back to
  `RemoteUnknownEvent` — this package doesn't have typed events for them
  yet.
* Fixed a native-handle leak: `LiveTranscriber.start()`/
  `startDebugWavStream()` now free whatever `_buildNativeState` managed to
  build (e.g. a good recognizer load followed by a bad VAD model file)
  before rethrowing, instead of leaking it with `isRunning` still `false`.
* Fixed `setVadSensitivity()` losing its effect when called while `start()`
  was still loading: it's now queued and applied once loading finishes,
  instead of being silently overwritten by the in-progress `start()` call's
  own sensitivity.
* Fixed a race where a `setVadSensitivity()` rebuild still in flight when a
  session was stopped and immediately restarted could land on, and
  silently replace, the *new* session's freshly built VAD. A monotonically
  increasing generation counter (`isVadBuildStale`) now detects this and
  discards the stale build instead of installing it.
