import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';

/// In-memory stand-in for the `record` plugin's platform channel, so the
/// lifecycle of [AudioRecorder]-owning classes ([LiveTranscriber],
/// [RemoteTranscriber]) can be exercised in a plain `flutter test` run —
/// no device, no mic, no method channel.
///
/// Installed by assigning it to `RecordPlatform.instance`; every
/// `AudioRecorder` created afterwards talks to this instead.
class FakeRecordPlatform extends RecordPlatform {
  final micController = StreamController<Uint8List>.broadcast();
  final stateController = StreamController<RecordState>.broadcast();

  /// What [hasPermission] answers.
  bool permission = true;

  Future<void> close() async {
    await micController.close();
    await stateController.close();
  }

  @override
  Future<void> create(String recorderId) async {}

  @override
  Future<void> start(
    String recorderId,
    RecordConfig config, {
    required String path,
  }) async {}

  @override
  Future<Stream<Uint8List>> startStream(
    String recorderId,
    RecordConfig config,
  ) async => micController.stream;

  @override
  Future<String?> stop(String recorderId) async => null;

  @override
  Future<void> pause(String recorderId) async {}

  @override
  Future<void> resume(String recorderId) async {}

  @override
  Future<bool> isRecording(String recorderId) async => false;

  @override
  Future<bool> isPaused(String recorderId) async => false;

  @override
  Future<bool> hasPermission(String recorderId, {bool request = true}) async =>
      permission;

  @override
  Future<void> dispose(String recorderId) async {}

  @override
  Future<Amplitude> getAmplitude(String recorderId) async =>
      Amplitude(current: 0, max: 0);

  @override
  Future<bool> isEncoderSupported(String recorderId, AudioEncoder encoder) async =>
      true;

  @override
  Future<List<InputDevice>> listInputDevices(String recorderId) async => [];

  @override
  Future<void> cancel(String recorderId) async {}

  @override
  Stream<RecordState> onStateChanged(String recorderId) =>
      stateController.stream;
}
