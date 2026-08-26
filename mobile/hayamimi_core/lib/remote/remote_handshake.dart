import 'dart:convert';

/// Builds the JSON text frame that must be the first message sent on a
/// hayamimi `/ingest` WebSocket connection, per `scripts/ws_ingest.py`'s
/// protocol: `{"sr": <rate>, "format": "pcm_s16le", "channels": <n>}`.
///
/// Everything sent after this handshake is raw little-endian PCM16 binary
/// frames until the connection closes.
String buildIngestHandshakeJson({
  required int sampleRate,
  int channels = 1,
  String format = 'pcm_s16le',
}) {
  return jsonEncode({'sr': sampleRate, 'format': format, 'channels': channels});
}
