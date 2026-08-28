import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/server/subtitle_broadcast_server.dart';
import 'package:hayamimi_core/server/subtitle_event.dart';

void main() {
  group('SubtitleBroadcastServer', () {
    late SubtitleBroadcastServer server;

    setUp(() {
      // Port 0 asks the OS for an ephemeral port so tests don't collide
      // with each other or a real running instance.
      server = SubtitleBroadcastServer(port: 0);
    });

    tearDown(() async {
      await server.stop();
    });

    test('is not running until start() is called', () {
      expect(server.isRunning, isFalse);
      expect(server.boundPort, isNull);
    });

    test('serves the overlay page at /', () async {
      await server.start();
      final client = HttpClient();
      try {
        final request = await client.get(
          '127.0.0.1',
          server.boundPort!,
          '/',
        );
        final response = await request.close();
        final body = await response.transform(utf8.decoder).join();

        expect(response.statusCode, 200);
        expect(
          response.headers.contentType?.mimeType,
          'text/html',
        );
        expect(body, contains('EventSource("/events")'));
      } finally {
        client.close(force: true);
      }
    });

    test('streams broadcast events to an /events subscriber as SSE', () async {
      await server.start();

      // A raw socket, rather than HttpClient, so this test controls exactly
      // when the connection is torn down instead of racing HttpClient's
      // own (asynchronous, exception-prone) force-close teardown against a
      // still-open streaming response.
      final socket = await Socket.connect('127.0.0.1', server.boundPort!);
      socket.write(
        'GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n',
      );

      final chunks = <int>[];
      final dataFrameSeen = Completer<void>();
      final sub = socket.listen((data) {
        chunks.addAll(data);
        if (utf8.decode(chunks, allowMalformed: true).contains('data: ') &&
            !dataFrameSeen.isCompleted) {
          dataFrameSeen.complete();
        }
      });

      try {
        // Give the server a moment to register the subscriber before
        // broadcasting, then send one event.
        await Future.delayed(const Duration(milliseconds: 50));
        server.broadcast(
          const FinalSubtitleEvent(text: 'テスト', lang: 'ja', latencyMs: 42),
        );

        await dataFrameSeen.future.timeout(const Duration(seconds: 5));

        final text = utf8.decode(chunks, allowMalformed: true);
        expect(text, contains('HTTP/1.1 200'));
        expect(text, contains('text/event-stream'));
        // The server sends a leading ": connected" SSE comment to force
        // headers out immediately; the real event is the "data: " frame
        // after it.
        final dataLine = text
            .split('\n')
            .firstWhere((line) => line.startsWith('data: '));
        final decoded =
            jsonDecode(dataLine.substring('data: '.length))
                as Map<String, dynamic>;
        expect(decoded['type'], 'final');
        expect(decoded['text'], 'テスト');
        expect(decoded['lang'], 'ja');
        expect(decoded['latency_ms'], 42);
      } finally {
        await sub.cancel();
        socket.destroy();
      }
    });

    test('a second start() while running keeps the same bound port', () async {
      await server.start();
      final port = server.boundPort;

      await server.start();

      expect(server.boundPort, port);
    });

    test('concurrent start() calls bind exactly one server', () async {
      // The isRunning guard alone doesn't cover this: it only flips once
      // the bind completes, so two overlapping starts both used to bind,
      // and the first server was dropped on the floor still listening.
      await Future.wait([server.start(), server.start(), server.start()]);
      final port = server.boundPort;
      expect(port, isNotNull);

      await server.stop();

      // If a second, unreferenced server had also bound, it would still be
      // holding the port here.
      final rebound = await ServerSocket.bind(InternetAddress.anyIPv4, port!);
      await rebound.close();
    });

    test('stop() closes the port so a new server can bind a fresh one', () async {
      await server.start();
      expect(server.isRunning, isTrue);

      await server.stop();

      expect(server.isRunning, isFalse);
      expect(server.boundPort, isNull);
    });
  });
}
