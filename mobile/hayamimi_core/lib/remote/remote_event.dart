import 'dart:convert';

/// One subtitle event received over the `/ingest` WebSocket from a hayamimi
/// server (`--input ws --serve`), wire-compatible with
/// `scripts/subtitle_server.py`'s `publish()` schema — the same JSON the
/// desktop SSE dashboard and OBS overlay consume.
sealed class RemoteEvent {
  const RemoteEvent();
}

/// Sent once right after the ingest handshake is accepted, confirming the
/// sample rate the server will resample to.
class RemoteReadyEvent extends RemoteEvent {
  const RemoteReadyEvent({required this.sampleRate});
  final int sampleRate;
}

/// Sent once per pipeline session start (`{"type": "session_start"}`).
class RemoteSessionStartEvent extends RemoteEvent {
  const RemoteSessionStartEvent();
}

/// An in-progress (not yet finalized) recognition result.
class RemotePartialEvent extends RemoteEvent {
  const RemotePartialEvent({required this.text});
  final String text;
}

/// A finalized transcript line for one detected speech segment.
class RemoteFinalEvent extends RemoteEvent {
  const RemoteFinalEvent({
    required this.text,
    this.lang = '',
    this.speaker = '',
    this.latencyMs,
    this.tier = '',
    this.audioSeconds,
    this.switched = false,
    this.punctuated = false,
  });

  final String text;
  final String lang;
  final String speaker;
  final double? latencyMs;
  final String tier;

  /// The segment's audio duration in seconds (wire: `audio_s`), mirroring
  /// `LiveTranscriptEntry.audioSeconds`/`FinalSubtitleEvent.audioSeconds`
  /// on the on-device path. `null` if the server didn't send it.
  final double? audioSeconds;

  /// Whether this segment is why the server's multi-language routing
  /// switched languages, mirroring `FinalSubtitleEvent.switched`. `false`
  /// if the server didn't send it (including servers running a version of
  /// `subtitle_server.py` from before this field existed).
  final bool switched;

  /// Whether the server had already punctuated this line, mirroring
  /// `FinalSubtitleEvent.punctuated`. `false` if it didn't send the field.
  ///
  /// The desktop pipeline does punctuate its Japanese finals, but
  /// `subtitle_server.py` has no `punctuated` key on `final` frames to say
  /// so, so today this is `false` for every real server and exists so that a
  /// consumer reading `FinalSubtitleEvent.punctuated` gets the same field
  /// from both paths rather than a field that only some producers have.
  final bool punctuated;
}

/// A machine translation of the most recent final line.
class RemoteTranslationEvent extends RemoteEvent {
  const RemoteTranslationEvent({required this.lang, required this.text});
  final String lang;
  final String text;
}

/// A second-pass, re-decoded and cleaned-up version of a group of finals.
class RemoteRefineEvent extends RemoteEvent {
  const RemoteRefineEvent({
    required this.text,
    this.lang = '',
    this.speaker = '',
    this.latencyMs,
    this.audioSeconds,
  });
  final String text;
  final String lang;
  final String speaker;

  /// Wall-clock decode time for this refine pass, in milliseconds. `null`
  /// if the server didn't send it.
  final double? latencyMs;

  /// Total duration (seconds) of the buffered group this refine
  /// re-decoded (wire: `audio_s`), mirroring
  /// `RefineSubtitleEvent.audioSeconds`. `null` if the server didn't send
  /// it.
  final double? audioSeconds;
}

/// A server-reported error, e.g. a rejected handshake or a second
/// audio-producing client trying to connect while one is already active.
class RemoteErrorEvent extends RemoteEvent {
  const RemoteErrorEvent({required this.message});
  final String message;
}

/// One phase of the server loading a native model, mirroring
/// `LiveTranscriber.modelLoads`/`ModelLoadSubtitleEvent` on the on-device
/// path — see those for the exact model-name/phase vocabulary.
class RemoteModelLoadEvent extends RemoteEvent {
  const RemoteModelLoadEvent({
    required this.model,
    required this.phase,
    this.ms,
  });
  final String model;
  final String phase;
  final double? ms;
}

/// The server's session state (refine buffer, draft state, routed
/// language) was just cleared by a reset, mirroring
/// `LiveTranscriber.resetSession`/`SessionResetSubtitleEvent` on the
/// on-device path.
class RemoteSessionResetEvent extends RemoteEvent {
  const RemoteSessionResetEvent();
}

/// Any event whose `type` isn't recognized. Kept instead of dropped so a
/// protocol change on the server side degrades gracefully in the UI rather
/// than silently vanishing.
class RemoteUnknownEvent extends RemoteEvent {
  const RemoteUnknownEvent({required this.type, required this.raw});
  final String? type;
  final Map<String, dynamic> raw;
}

/// Thrown while decoding a server frame into a [RemoteEvent]: malformed
/// JSON, or JSON that isn't an object.
class RemoteEventParseException implements Exception {
  RemoteEventParseException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Parses one JSON text frame from the `/ingest` WebSocket into a
/// [RemoteEvent]. Throws [RemoteEventParseException] if `raw` isn't valid
/// JSON or isn't a JSON object.
///
/// An unrecognized `type` (including `model_fallback`/`warning`/
/// `session_summary`/`recluster`, which `subtitle_server.py` also emits
/// but this package doesn't yet have a typed [RemoteEvent] for) falls back
/// to [RemoteUnknownEvent] rather than throwing, so a protocol change or an
/// event this package hasn't caught up to yet degrades gracefully instead
/// of breaking the connection.
RemoteEvent parseRemoteEvent(String raw) {
  final Object? decoded;
  try {
    decoded = jsonDecode(raw);
  } on FormatException catch (e) {
    throw RemoteEventParseException('invalid JSON: $e');
  }
  if (decoded is! Map<String, dynamic>) {
    throw RemoteEventParseException('expected a JSON object, got: $raw');
  }
  final map = decoded;
  final type = map['type'] as String?;
  switch (type) {
    case 'ready':
      return RemoteReadyEvent(sampleRate: (map['sr'] as num?)?.toInt() ?? 0);
    case 'session_start':
      return const RemoteSessionStartEvent();
    case 'partial':
      return RemotePartialEvent(text: map['text'] as String? ?? '');
    case 'final':
      return RemoteFinalEvent(
        text: map['text'] as String? ?? '',
        lang: map['lang'] as String? ?? '',
        speaker: map['speaker'] as String? ?? '',
        latencyMs: (map['latency_ms'] as num?)?.toDouble(),
        tier: map['tier'] as String? ?? '',
        audioSeconds: (map['audio_s'] as num?)?.toDouble(),
        switched: map['switched'] as bool? ?? false,
        punctuated: map['punctuated'] as bool? ?? false,
      );
    case 'translation':
      return RemoteTranslationEvent(
        lang: map['lang'] as String? ?? '',
        text: map['text'] as String? ?? '',
      );
    case 'refine':
      return RemoteRefineEvent(
        text: map['text'] as String? ?? '',
        lang: map['lang'] as String? ?? '',
        speaker: map['speaker'] as String? ?? '',
        latencyMs: (map['latency_ms'] as num?)?.toDouble(),
        audioSeconds: (map['audio_s'] as num?)?.toDouble(),
      );
    case 'error':
      return RemoteErrorEvent(message: map['message'] as String? ?? 'unknown error');
    case 'model_load':
      return RemoteModelLoadEvent(
        model: map['model'] as String? ?? '',
        phase: map['phase'] as String? ?? '',
        ms: (map['ms'] as num?)?.toDouble(),
      );
    case 'session_reset':
      return const RemoteSessionResetEvent();
    default:
      return RemoteUnknownEvent(type: type, raw: map);
  }
}
