import 'dart:typed_data';

/// Decoded contents of a 16-bit PCM `.wav` file, split out from the file
/// I/O so the parsing logic is unit testable without touching disk.
class WavPcm16 {
  const WavPcm16({
    required this.sampleRate,
    required this.channels,
    required this.pcmBytes,
  });

  final int sampleRate;
  final int channels;

  /// Raw little-endian PCM16 sample bytes (the `data` chunk contents), in
  /// the exact format the `/ingest` WebSocket expects on the wire.
  final Uint8List pcmBytes;
}

class WavParseException implements Exception {
  WavParseException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Parses a canonical (uncompressed, 16-bit PCM) `.wav` file's bytes into
/// its sample rate, channel count, and raw sample data.
///
/// Used by the Remote tab's debug "send test wav" button (see
/// `remote_transcriber.dart`) to stream a pushed test file over the
/// `/ingest` WebSocket exactly the way `scripts/ws_mic_client.py` does,
/// for testing on an emulator with no real microphone.
WavPcm16 parseWavPcm16(Uint8List bytes) {
  if (bytes.length < 44) {
    throw WavParseException('file too short to be a wav file (${bytes.length} bytes)');
  }
  final data = ByteData.sublistView(bytes);
  if (_ascii(bytes, 0, 4) != 'RIFF' || _ascii(bytes, 8, 4) != 'WAVE') {
    throw WavParseException('not a RIFF/WAVE file');
  }

  int? sampleRate;
  int? channels;
  int? bitsPerSample;
  int? audioFormat;
  Uint8List? pcmBytes;

  var offset = 12;
  while (offset + 8 <= bytes.length) {
    final chunkId = _ascii(bytes, offset, 4);
    final chunkSize = data.getUint32(offset + 4, Endian.little);
    final bodyStart = offset + 8;
    if (bodyStart + chunkSize > bytes.length) {
      break; // truncated/garbage trailing chunk: stop, use what we have
    }

    if (chunkId == 'fmt ') {
      if (chunkSize < 16) {
        throw WavParseException('fmt chunk too small ($chunkSize bytes)');
      }
      audioFormat = data.getUint16(bodyStart, Endian.little);
      channels = data.getUint16(bodyStart + 2, Endian.little);
      sampleRate = data.getUint32(bodyStart + 4, Endian.little);
      bitsPerSample = data.getUint16(bodyStart + 14, Endian.little);
    } else if (chunkId == 'data') {
      pcmBytes = bytes.sublist(bodyStart, bodyStart + chunkSize);
    }

    offset = bodyStart + chunkSize + (chunkSize.isOdd ? 1 : 0); // chunks are word-aligned
  }

  if (sampleRate == null || channels == null || bitsPerSample == null) {
    throw WavParseException('missing fmt chunk');
  }
  if (audioFormat != 1) {
    throw WavParseException('unsupported wav audio format $audioFormat (only PCM=1)');
  }
  if (bitsPerSample != 16) {
    throw WavParseException('unsupported bit depth $bitsPerSample (only 16-bit PCM)');
  }
  if (pcmBytes == null) {
    throw WavParseException('missing data chunk');
  }

  return WavPcm16(sampleRate: sampleRate, channels: channels, pcmBytes: pcmBytes);
}

String _ascii(Uint8List bytes, int start, int length) {
  return String.fromCharCodes(bytes.sublist(start, start + length));
}
