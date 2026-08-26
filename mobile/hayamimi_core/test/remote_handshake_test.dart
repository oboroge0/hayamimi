import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/remote/remote_handshake.dart';

void main() {
  group('buildIngestHandshakeJson', () {
    test('defaults to mono pcm_s16le', () {
      final json = buildIngestHandshakeJson(sampleRate: 16000);
      final decoded = jsonDecode(json) as Map<String, dynamic>;
      expect(decoded, {'sr': 16000, 'format': 'pcm_s16le', 'channels': 1});
    });

    test('honors an explicit channel count and format', () {
      final json = buildIngestHandshakeJson(
        sampleRate: 44100,
        channels: 2,
        format: 'pcm_s16le',
      );
      final decoded = jsonDecode(json) as Map<String, dynamic>;
      expect(decoded, {'sr': 44100, 'format': 'pcm_s16le', 'channels': 2});
    });
  });
}
