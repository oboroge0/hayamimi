## 0.1.0

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
