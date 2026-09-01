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
