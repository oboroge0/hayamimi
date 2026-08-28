import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_record_platform.dart';

/// Lifecycle behaviour that doesn't need the sherpa-onnx native libs or a
/// real microphone: dispose idempotency across all four public lifecycle
/// owners, and the connect/start re-entrancy guards.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late RecordPlatform originalPlatform;
  late FakeRecordPlatform fakePlatform;

  setUp(() {
    originalPlatform = RecordPlatform.instance;
    fakePlatform = FakeRecordPlatform();
    RecordPlatform.instance = fakePlatform;
  });

  tearDown(() async {
    await fakePlatform.close();
    RecordPlatform.instance = originalPlatform;
  });

  group('dispose is idempotent', () {
    // A second dispose() used to throw "Bad state: Stream has already been
    // closed" on the already-closed StreamControllers, which is easy to
    // hit when a host app disposes on both a route pop and a widget
    // dispose.
    test('LiveTranscriber', () async {
      final transcriber = LiveTranscriber();
      await transcriber.dispose();
      await transcriber.dispose();
    });

    test('HayamimiLive', () async {
      final live = HayamimiLive();
      await live.dispose();
      await live.dispose();
    });

    test('RemoteTranscriber', () async {
      final remote = RemoteTranscriber();
      await remote.dispose();
      await remote.dispose();
    });

    test('HayamimiRemote', () async {
      final remote = HayamimiRemote();
      await remote.dispose();
      await remote.dispose();
    });
  });

  group('RemoteTranscriber.connect guard', () {
    late HttpServer server;
    late String url;

    setUp(() async {
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((request) async {
        final socket = await WebSocketTransformer.upgrade(request);
        socket.listen((_) {}, onError: (_) {});
      });
      url = 'ws://127.0.0.1:${server.port}/ingest';
    });

    tearDown(() async {
      await server.close(force: true);
    });

    test('a second connect() while connected throws StateError', () async {
      final remote = RemoteTranscriber();
      addTearDown(remote.dispose);

      await remote.connect(url);
      expect(remote.isConnected, isTrue);

      await expectLater(remote.connect(url), throwsStateError);

      // The rejected second call must not have torn down the first one.
      expect(remote.isConnected, isTrue);
    });

    test('connect() is allowed again after disconnect()', () async {
      final remote = RemoteTranscriber();
      addTearDown(remote.dispose);

      await remote.connect(url);
      await remote.disconnect();
      expect(remote.state, RemoteConnectionState.disconnected);

      await remote.connect(url);
      expect(remote.isConnected, isTrue);
    });

    test('a failed connect() leaves the client connectable again', () async {
      final remote = RemoteTranscriber();
      addTearDown(remote.dispose);

      // Nothing is listening on this port, so the connect fails.
      await expectLater(
        remote.connect('ws://127.0.0.1:1/ingest'),
        throwsA(isA<RemoteTranscriberException>()),
      );
      expect(remote.state, RemoteConnectionState.disconnected);

      await remote.connect(url);
      expect(remote.isConnected, isTrue);
    });
  });

  group('RemoteTranscriber capture watchdog', () {
    late HttpServer server;
    late String url;

    setUp(() async {
      server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      server.listen((request) async {
        final socket = await WebSocketTransformer.upgrade(request);
        socket.listen((_) {}, onError: (_) {});
      });
      url = 'ws://127.0.0.1:${server.port}/ingest';
    });

    tearDown(() async {
      await server.close(force: true);
    });

    test(
      'an OS-side recorder stop surfaces an error event and disconnects',
      () async {
        final remote = RemoteTranscriber();
        addTearDown(remote.dispose);

        final errors = <RemoteErrorEvent>[];
        remote.events.listen((event) {
          if (event is RemoteErrorEvent) errors.add(event);
        });

        await remote.connect(url);
        expect(remote.isConnected, isTrue);

        // What the OS revoking the mic looks like through package:record.
        fakePlatform.stateController.add(RecordState.stop);
        await Future<void>.delayed(const Duration(milliseconds: 50));

        expect(errors, hasLength(1));
        expect(errors.single.message, contains('stopped by the system'));
        expect(remote.isConnected, isFalse);
      },
    );
  });
}
