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
  });

  final String text;
  final String lang;
  final String speaker;
  final double? latencyMs;
  final String tier;
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
  });
  final String text;
  final String lang;
  final String speaker;
}

/// A server-reported error, e.g. a rejected handshake or a second
/// audio-producing client trying to connect while one is already active.
class RemoteErrorEvent extends RemoteEvent {
  const RemoteErrorEvent({required this.message});
  final String message;
}

/// Any event whose `type` isn't recognized. Kept instead of dropped so a
/// protocol change on the server side degrades gracefully in the UI rather
/// than silently vanishing.
class RemoteUnknownEvent extends RemoteEvent {
  const RemoteUnknownEvent({required this.type, required this.raw});
  final String? type;
  final Map<String, dynamic> raw;
}

class RemoteEventParseException implements Exception {
  RemoteEventParseException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Parses one JSON text frame from the `/ingest` WebSocket into a
/// [RemoteEvent]. Throws [RemoteEventParseException] if `raw` isn't valid
/// JSON or isn't a JSON object.
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
      );
    case 'error':
      return RemoteErrorEvent(message: map['message'] as String? ?? 'unknown error');
    default:
      return RemoteUnknownEvent(type: type, raw: map);
  }
}
