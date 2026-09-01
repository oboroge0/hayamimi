import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';

import 'remote_connection_state.dart';
import 'remote_event.dart';
import 'remote_handshake.dart';
import 'wav_pcm_reader.dart';

/// Thrown by [RemoteTranscriber.connect]/[RemoteTranscriber.sendTestWavFile]
/// for a connection or microphone-permission failure -- a rejected
/// handshake, an unreachable server, or a denied `RECORD_AUDIO` request.
class RemoteTranscriberException implements Exception {
  RemoteTranscriberException(this.message);
  final String message;

  @override
  String toString() => message;
}

const _reconnectDelay = Duration(seconds: 2);

/// Remote-mode client: streams this phone's mic audio to a hayamimi server
/// running `--input ws --serve` over its `/ingest` WebSocket, and surfaces
/// the subtitle events it streams back.
///
/// This is deliberately a thin client — all recognition happens on the PC.
/// The wire protocol is documented in `scripts/ws_ingest.py`: one JSON
/// handshake frame, then continuous binary PCM16 frames; the server replies
/// with the same partial/final/translation/refine JSON events the desktop
/// SSE dashboard sees.
///
/// Also provides [sendTestWavFile], a debug helper that streams a `.wav`
/// file at real-time pace over its own one-shot connection — used to
/// exercise the full pipeline from an emulator, which has no usable
/// microphone (see `scripts/ws_mic_client.py`, the desktop-side reference
/// client this mirrors).
class RemoteTranscriber {
  /// Creates a client with no active connection. Pass [recorder] only to
  /// inject a fake `AudioRecorder` in tests; a real host app should leave
  /// it unset.
  RemoteTranscriber({AudioRecorder? recorder}) : _recorder = recorder ?? AudioRecorder();

  /// The mic capture rate this client records at and streams to the
  /// server, in Hz. Fixed -- not configurable per session.
  static const int sampleRate = 16000;

  final AudioRecorder _recorder;

  WebSocket? _socket;
  StreamSubscription<Uint8List>? _micSubscription;
  StreamSubscription<RecordState>? _recorderStateSubscription;
  StreamSubscription? _socketSubscription;
  Timer? _reconnectTimer;

  String? _url;
  // Doubles as the connect() re-entrancy guard: set synchronously by
  // connect() and only cleared by disconnect() (or by a failed connect
  // attempt), so it covers connecting/connected/reconnecting alike.
  bool _desiredConnected = false;
  bool _disposed = false;
  RemoteConnectionState _state = RemoteConnectionState.disconnected;

  final _eventsController = StreamController<RemoteEvent>.broadcast();
  final _stateController = StreamController<RemoteConnectionState>.broadcast();

  /// Subtitle events received from the server, on either the mic connection
  /// or a [sendTestWavFile] debug send.
  Stream<RemoteEvent> get events => _eventsController.stream;

  /// Fires whenever [state] changes, e.g. connecting -> connected, or a
  /// dropped socket -> reconnecting.
  Stream<RemoteConnectionState> get connectionState => _stateController.stream;

  /// This client's current connection lifecycle state.
  RemoteConnectionState get state => _state;

  /// Shorthand for `state == RemoteConnectionState.connected`.
  bool get isConnected => _state == RemoteConnectionState.connected;

  /// Connects to [url] (e.g. `ws://10.0.2.2:8833/ingest`), sends the
  /// ingest handshake, and starts streaming mic audio. Reconnects
  /// automatically with a fixed backoff if the connection drops, until
  /// [disconnect] is called.
  ///
  /// Throws [StateError] if this client is already connected, connecting,
  /// or waiting to reconnect — connecting twice would orphan the first
  /// socket and stream the mic into both. Call [disconnect] first to move
  /// to a different server.
  Future<void> connect(String url) async {
    if (_desiredConnected) {
      throw StateError(
        'RemoteTranscriber is already connected or connecting to $_url; '
        'call disconnect() before connecting again.',
      );
    }
    _url = url;
    _desiredConnected = true;
    await _connectOnce();
  }

  Future<void> _connectOnce() async {
    _reconnectTimer?.cancel();
    _setState(RemoteConnectionState.connecting);

    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      _setState(RemoteConnectionState.disconnected);
      _desiredConnected = false;
      throw RemoteTranscriberException('Microphone permission was not granted.');
    }

    final WebSocket socket;
    try {
      socket = await WebSocket.connect(_url!);
    } catch (e) {
      _setState(RemoteConnectionState.disconnected);
      _desiredConnected = false;
      throw RemoteTranscriberException('Could not connect to $_url: $e');
    }

    _socket = socket;
    socket.add(
      buildIngestHandshakeJson(sampleRate: sampleRate, channels: 1),
    );
    _socketSubscription = socket.listen(
      _onSocketData,
      onDone: _onSocketDisconnected,
      onError: (_) => _onSocketDisconnected(),
      cancelOnError: true,
    );

    try {
      final micStream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: sampleRate,
          numChannels: 1,
        ),
      );
      _micSubscription = micStream.listen(
        _onMicChunk,
        onError: (Object error) =>
            _onCaptureFailure('Microphone capture failed: $error'),
        onDone: () => _onCaptureFailure(
          'Microphone capture ended unexpectedly (the OS may have revoked '
          'mic access, e.g. on backgrounding or another app taking the '
          'microphone).',
        ),
        cancelOnError: true,
      );
      // record 6.2.1's stream wrapper forwards data only (see
      // _startRecordStream in package:record), so the source stream's
      // error/done never reach the handlers above. Its state channel does
      // report an OS-side stop -- watch both.
      _recorderStateSubscription = _recorder.onStateChanged().listen(
        (state) {
          if (state == RecordState.stop) {
            _onCaptureFailure(
              'Microphone capture was stopped by the system (mic access '
              'revoked, e.g. on backgrounding or another app taking the '
              'microphone).',
            );
          }
        },
        onError: (Object error) =>
            _onCaptureFailure('Microphone capture failed: $error'),
      );
    } catch (e) {
      await socket.close();
      _socket = null;
      _setState(RemoteConnectionState.disconnected);
      _desiredConnected = false;
      throw RemoteTranscriberException('Could not start mic capture: $e');
    }

    _setState(RemoteConnectionState.connected);
  }

  void _onMicChunk(Uint8List bytes) {
    _socket?.add(bytes);
  }

  /// Mic capture died under a live connection. The socket is fine, so the
  /// automatic reconnect path would just re-establish a connection that
  /// still has no audio to send and leave [isConnected] true forever —
  /// report it and disconnect for real instead.
  void _onCaptureFailure(String message) {
    if (_micSubscription == null) {
      return;
    }
    if (!_eventsController.isClosed) {
      _eventsController.add(RemoteErrorEvent(message: message));
    }
    unawaited(disconnect());
  }

  void _onSocketData(dynamic data) {
    if (data is! String) {
      return; // server never sends binary frames to the client
    }
    try {
      _eventsController.add(parseRemoteEvent(data));
    } on RemoteEventParseException {
      // malformed/unrecognized frame: drop it, keep the connection alive
    }
  }

  void _onSocketDisconnected() {
    _socketSubscription?.cancel();
    _socketSubscription = null;
    _socket = null;
    unawaited(_micSubscription?.cancel());
    _micSubscription = null;
    unawaited(_recorderStateSubscription?.cancel());
    _recorderStateSubscription = null;
    unawaited(_recorder.stop());

    if (!_desiredConnected) {
      _setState(RemoteConnectionState.disconnected);
      return;
    }
    _setState(RemoteConnectionState.reconnecting);
    _reconnectTimer = Timer(_reconnectDelay, () {
      if (_desiredConnected) {
        unawaited(_connectOnce());
      }
    });
  }

  /// Stops mic streaming and closes the connection. Cancels any pending
  /// reconnect attempt.
  Future<void> disconnect() async {
    _desiredConnected = false;
    _reconnectTimer?.cancel();
    _reconnectTimer = null;

    await _micSubscription?.cancel();
    _micSubscription = null;
    // Cancelled before _recorder.stop() so our own shutdown doesn't get
    // reported back to us as a capture failure.
    await _recorderStateSubscription?.cancel();
    _recorderStateSubscription = null;
    await _recorder.stop();

    await _socketSubscription?.cancel();
    _socketSubscription = null;
    await _socket?.close();
    _socket = null;

    _setState(RemoteConnectionState.disconnected);
  }

  void _setState(RemoteConnectionState state) {
    _state = state;
    _stateController.add(state);
  }

  /// Debug helper: streams a 16-bit PCM `.wav` file at real-time playback
  /// pace over its own one-shot `/ingest` connection (independent of any
  /// mic connection — the server only accepts one audio-producing client
  /// at a time), appending a little trailing silence so the server's VAD
  /// finalizes the last utterance before the connection closes. Mirrors
  /// `scripts/ws_mic_client.py`.
  ///
  /// Events received on this connection are forwarded onto [events] like
  /// any other. Intended for exercising the full pipeline on an emulator,
  /// which has no usable microphone.
  Future<void> sendTestWavFile(
    String url,
    String wavPath, {
    Duration tailSilence = const Duration(milliseconds: 1500),
    Duration chunkDuration = const Duration(milliseconds: 100),
  }) async {
    final file = File(wavPath);
    if (!await file.exists()) {
      throw RemoteTranscriberException('Test wav file not found: $wavPath');
    }
    final wav = parseWavPcm16(await file.readAsBytes());

    final WebSocket socket;
    try {
      socket = await WebSocket.connect(url);
    } catch (e) {
      throw RemoteTranscriberException('Could not connect to $url: $e');
    }

    final done = Completer<void>();
    late final StreamSubscription sub;
    sub = socket.listen(
      (data) {
        if (data is String) {
          try {
            _eventsController.add(parseRemoteEvent(data));
          } on RemoteEventParseException {
            // ignore malformed frames
          }
        }
      },
      onDone: () {
        if (!done.isCompleted) done.complete();
      },
      onError: (Object e) {
        if (!done.isCompleted) done.completeError(e);
      },
      cancelOnError: true,
    );

    try {
      socket.add(
        buildIngestHandshakeJson(
          sampleRate: wav.sampleRate,
          channels: wav.channels,
        ),
      );

      final bytesPerSecond = wav.sampleRate * wav.channels * 2;
      var chunkBytes = (bytesPerSecond * chunkDuration.inMilliseconds / 1000).round();
      chunkBytes -= chunkBytes % 2;
      chunkBytes = chunkBytes < 2 ? 2 : chunkBytes;

      final tailSamples = (bytesPerSecond * tailSilence.inMilliseconds / 1000).round() ~/ 2 * 2;
      final payload = Uint8List(wav.pcmBytes.length + tailSamples)
        ..setRange(0, wav.pcmBytes.length, wav.pcmBytes);

      final stopwatch = Stopwatch()..start();
      var sent = 0;
      var pos = 0;
      while (pos < payload.length) {
        final end = (pos + chunkBytes).clamp(0, payload.length);
        socket.add(payload.sublist(pos, end));
        sent += end - pos;
        pos = end;
        final targetMicros = (sent / bytesPerSecond * Duration.microsecondsPerSecond).round();
        final delayMicros = targetMicros - stopwatch.elapsedMicroseconds;
        if (delayMicros > 0) {
          await Future<void>.delayed(Duration(microseconds: delayMicros));
        }
      }

      // give the server a moment to finalize + reply before we close
      await Future.any([
        done.future.catchError((_) {}),
        Future<void>.delayed(const Duration(seconds: 8)),
      ]);
    } finally {
      await sub.cancel();
      await socket.close();
    }
  }

  /// Releases everything, including the mic recorder itself. Call once
  /// when the owning widget is disposed. Idempotent: a second call is a
  /// no-op rather than a "Bad state: Stream has already been closed".
  Future<void> dispose() async {
    if (_disposed) {
      return;
    }
    _disposed = true;
    await disconnect();
    await _eventsController.close();
    await _stateController.close();
    await _recorder.dispose();
  }
}
