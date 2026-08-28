import 'dart:async';

import 'remote/remote_connection_state.dart';
import 'remote/remote_event.dart';
import 'remote/remote_transcriber.dart';
import 'server/subtitle_event.dart';

/// Thin client for a hayamimi server running `--input ws --serve`, packaged
/// as a single [SubtitleEvent] stream.
///
/// This is the embedding-friendly facade over [RemoteTranscriber]: it maps
/// the server's [RemoteEvent] protocol frames onto the same [SubtitleEvent]
/// types [HayamimiLive] and `SubtitleBroadcastServer` use, so a host app
/// can treat on-device and remote-PC transcription interchangeably.
/// Protocol-only frames ([RemoteReadyEvent], [RemoteSessionStartEvent],
/// [RemoteUnknownEvent]) carry no subtitle content and are dropped from
/// [events]; use [rawEvents] if you need them.
///
/// ```dart
/// final remote = HayamimiRemote();
/// remote.events.listen((event) {
///   if (event is FinalSubtitleEvent) print(event.text);
/// });
/// await remote.connect('ws://192.168.1.10:8766/ingest');
/// // ... later
/// await remote.disconnect();
/// await remote.dispose();
/// ```
class HayamimiRemote {
  HayamimiRemote({RemoteTranscriber? transcriber})
    : _transcriber = transcriber ?? RemoteTranscriber() {
    _rawSubscription = _transcriber.events.listen(_onRawEvent);
  }

  final RemoteTranscriber _transcriber;
  final _eventsController = StreamController<SubtitleEvent>.broadcast();
  late final StreamSubscription<RemoteEvent> _rawSubscription;
  bool _disposed = false;

  /// Subtitle content received from the server: partials, finals,
  /// translations, refines, and errors, normalized to [SubtitleEvent].
  Stream<SubtitleEvent> get events => _eventsController.stream;

  /// The unmodified protocol stream, including handshake/session frames
  /// [events] drops. Most callers want [events] instead.
  Stream<RemoteEvent> get rawEvents => _transcriber.events;

  Stream<RemoteConnectionState> get connectionState =>
      _transcriber.connectionState;
  RemoteConnectionState get state => _transcriber.state;
  bool get isConnected => _transcriber.isConnected;

  /// Connects to [url] (e.g. `ws://192.168.1.10:8766/ingest`), sends the
  /// ingest handshake, and starts streaming mic audio. Reconnects
  /// automatically with a fixed backoff if the connection drops, until
  /// [disconnect] is called. Throws [StateError] if already connected,
  /// connecting, or reconnecting — call [disconnect] first.
  Future<void> connect(String url) => _transcriber.connect(url);

  /// Stops mic streaming and closes the connection.
  Future<void> disconnect() => _transcriber.disconnect();

  /// Debug helper: streams a 16-bit PCM `.wav` file at real-time playback
  /// pace over its own one-shot `/ingest` connection — useful for
  /// exercising the pipeline on an emulator, which has no usable
  /// microphone. Events received on this connection also arrive on
  /// [events].
  Future<void> sendTestWavFile(String url, String wavPath) {
    return _transcriber.sendTestWavFile(url, wavPath);
  }

  void _onRawEvent(RemoteEvent event) {
    final mapped = switch (event) {
      RemotePartialEvent(:final text) => PartialSubtitleEvent(text),
      RemoteFinalEvent(:final text, :final lang, :final speaker, :final latencyMs) =>
        FinalSubtitleEvent(
          text: text,
          lang: lang,
          speaker: speaker,
          latencyMs: latencyMs,
        ),
      RemoteTranslationEvent(:final lang, :final text) =>
        TranslationSubtitleEvent(lang: lang, text: text),
      RemoteRefineEvent(:final text, :final lang, :final speaker) =>
        RefineSubtitleEvent(text: text, lang: lang, speaker: speaker),
      RemoteErrorEvent(:final message) => ErrorSubtitleEvent(
        message: message,
      ),
      RemoteReadyEvent() || RemoteSessionStartEvent() || RemoteUnknownEvent() =>
        null,
    };
    if (mapped != null) {
      _eventsController.add(mapped);
    }
  }

  /// Releases everything, including the mic recorder. Call once when the
  /// owner is done with this instance. Idempotent: a second call is a
  /// no-op.
  Future<void> dispose() async {
    if (_disposed) {
      return;
    }
    _disposed = true;
    await _rawSubscription.cancel();
    await _transcriber.dispose();
    await _eventsController.close();
  }
}
