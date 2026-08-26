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

  @override
  Map<String, dynamic> toJson() => {
    'type': 'final',
    'text': text,
    'lang': lang,
    'speaker': speaker,
    'latency_ms': latencyMs,
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
  });

  final String text;
  final String lang;
  final String speaker;
  final double? latencyMs;

  @override
  Map<String, dynamic> toJson() => {
    'type': 'refine',
    'text': text,
    'lang': lang,
    'speaker': speaker,
    'latency_ms': latencyMs,
  };
}

/// A reported error, e.g. a rejected handshake or a connection failure.
class ErrorSubtitleEvent extends SubtitleEvent {
  const ErrorSubtitleEvent({required this.message});

  final String message;

  @override
  Map<String, dynamic> toJson() => {'type': 'error', 'message': message};
}
