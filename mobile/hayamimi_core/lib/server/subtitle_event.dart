import 'dart:convert';

/// One subtitle event, wire-compatible with the desktop hayamimi subtitle
/// server (`scripts/subtitle_server.py`): same `type` discriminator and
/// field names, so the same OBS browser source / browser client works
/// against either the desktop app or this phone.
sealed class SubtitleEvent {
  const SubtitleEvent();

  Map<String, dynamic> toJson();

  /// Formats this event as one SSE `data:` frame, ready to write straight
  /// into a `text/event-stream` response body.
  String toSseFrame() => 'data: ${jsonEncode(toJson())}\n\n';
}

/// An in-progress (not yet finalized) recognition result.
///
/// The current mobile pipeline is VAD-segment-then-decode (see
/// `LiveTranscriber`), so it has no true incremental partials the way the
/// desktop streaming pipeline does. This type exists so the wire format is
/// ready for that if/when it lands, without shipping a fake stream today.
class PartialSubtitleEvent extends SubtitleEvent {
  const PartialSubtitleEvent(this.text);

  final String text;

  @override
  Map<String, dynamic> toJson() => {'type': 'partial', 'text': text};
}

/// A finalized transcript line for one VAD-bounded speech segment.
class FinalSubtitleEvent extends SubtitleEvent {
  const FinalSubtitleEvent({
    required this.text,
    this.lang = '',
    this.speaker = '',
    this.latencyMs,
    this.audioSeconds,
    this.switched = false,
    this.punctuated = false,
  });

  final String text;

  /// BCP-47-ish language tag. The mobile app currently runs a single model
  /// per session rather than per-utterance language routing (unlike the
  /// desktop's multi-language pipeline), so this is a fixed value chosen by
  /// the caller — see `SubtitleBroadcastServer`'s `lang` parameter.
  final String lang;

  /// Empty unless/until the mobile app gains speaker diarization.
  final String speaker;

  /// Wall-clock decode time for this segment, in milliseconds.
  final double? latencyMs;

  /// How much audio this segment covers, in seconds (its VAD-detected
  /// sample count / 16000). `null` when the producing session doesn't
  /// report it.
  final double? audioSeconds;

  /// `true` when this segment is the reason a multilingual routed session
  /// changed language (see `RoutedRecognizerSet`'s dual-LID switch policy).
  /// Always `false` for a single-model session.
  final bool switched;

  /// Whether [text] had Japanese punctuation (、 。 ？) restored into it —
  /// see `LiveTranscriptEntry.punctuated`. `false` from any producer that
  /// does not punctuate its fast line, which includes every remote (desktop)
  /// session, any on-device session started without a `JaPunctuation`
  /// model, and one whose `JaPunctuation` set `applyToFinals: false`.
  final bool punctuated;

  @override
  Map<String, dynamic> toJson() => {
    'type': 'final',
    'text': text,
    'lang': lang,
    'speaker': speaker,
    'latency_ms': latencyMs,
    'audio_s': audioSeconds,
    'switched': switched,
    'punctuated': punctuated,
  };
}

/// A machine translation of the most recent final line.
class TranslationSubtitleEvent extends SubtitleEvent {
  const TranslationSubtitleEvent({required this.lang, required this.text});

  final String lang;
  final String text;

  @override
  Map<String, dynamic> toJson() => {
    'type': 'translation',
    'lang': lang,
    'text': text,
  };
}

/// A second-pass, re-decoded and cleaned-up version of a group of finals
/// (the mobile app's two-pass "清書" — see `HayamimiLive.refineNow`).
class RefineSubtitleEvent extends SubtitleEvent {
  const RefineSubtitleEvent({
    required this.text,
    this.lang = '',
    this.speaker = '',
    this.latencyMs,
    this.audioSeconds,
    this.punctuated = false,
  });

  final String text;
  final String lang;
  final String speaker;
  final double? latencyMs;

  /// Total duration (seconds) of the buffered group this refine re-decoded.
  /// `null` when the producing session doesn't report it.
  final double? audioSeconds;

  /// Whether [text] had Japanese punctuation (、 。 ？) restored into it —
  /// see `LiveTranscriptEntry.punctuated`. `false` from any producer that
  /// does not punctuate, which includes every remote (desktop) session and
  /// any on-device session started without a `JaPunctuation` model.
  final bool punctuated;

  @override
  Map<String, dynamic> toJson() => {
    'type': 'refine',
    'text': text,
    'lang': lang,
    'speaker': speaker,
    'latency_ms': latencyMs,
    'audio_s': audioSeconds,
    'punctuated': punctuated,
  };
}

/// A reported error, e.g. a rejected handshake or a connection failure.
class ErrorSubtitleEvent extends SubtitleEvent {
  const ErrorSubtitleEvent({required this.message});

  final String message;

  @override
  Map<String, dynamic> toJson() => {'type': 'error', 'message': message};
}

/// One phase of loading a native model on the producing session — see
/// `LiveTranscriber.modelLoads`/`ModelLoadEvent` for the mobile-side
/// producer. Lets a host UI show e.g. "loading SenseVoice..." instead of
/// treating model loading as one opaque wait.
class ModelLoadSubtitleEvent extends SubtitleEvent {
  const ModelLoadSubtitleEvent({
    required this.model,
    required this.phase,
    this.ms,
  });

  /// Which native model this phase is about (e.g. `"vad"`, `"recognizer"`,
  /// `"ja"`, `"sensevoice"`, `"lid"` — see `ModelLoadEvent.model`).
  final String model;

  /// `"start"` or `"done"`.
  final String phase;

  /// Wall-clock milliseconds the build took. `null` on `"start"`.
  final double? ms;

  @override
  Map<String, dynamic> toJson() => {
    'type': 'model_load',
    'model': model,
    'phase': phase,
    'ms': ms,
  };
}

/// The producing session's "conversation" state (refine buffer, draft
/// state, routed-session language) was just cleared by a
/// `resetSession()`/`HayamimiLive.resetSession()` call, without reloading
/// any native model.
class SessionResetSubtitleEvent extends SubtitleEvent {
  const SessionResetSubtitleEvent();

  @override
  Map<String, dynamic> toJson() => {'type': 'session_reset'};
}
