import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/remote/wav_pcm_reader.dart';

/// Builds a minimal canonical 16-bit PCM WAV file's bytes, for testing
/// [parseWavPcm16] without touching disk.
Uint8List _buildWav({
  required int sampleRate,
  required int channels,
  required Uint8List pcmData,
}) {
  const bitsPerSample = 16;
  final blockAlign = channels * bitsPerSample ~/ 8;
  final byteRate = sampleRate * blockAlign;
  final dataSize = pcmData.length;
  final riffSize = 36 + dataSize;

  final builder = BytesBuilder();
  void writeAscii(String s) => builder.add(s.codeUnits);
  void writeU32(int v) {
    final b = ByteData(4)..setUint32(0, v, Endian.little);
    builder.add(b.buffer.asUint8List());
  }

  void writeU16(int v) {
    final b = ByteData(2)..setUint16(0, v, Endian.little);
    builder.add(b.buffer.asUint8List());
  }

  writeAscii('RIFF');
  writeU32(riffSize);
  writeAscii('WAVE');

  writeAscii('fmt ');
  writeU32(16); // fmt chunk size
  writeU16(1); // PCM
  writeU16(channels);
  writeU32(sampleRate);
  writeU32(byteRate);
  writeU16(blockAlign);
  writeU16(bitsPerSample);

  writeAscii('data');
  writeU32(dataSize);
  builder.add(pcmData);

  return builder.toBytes();
}

void main() {
  group('parseWavPcm16', () {
    test('parses sample rate, channels, and pcm bytes', () {
      final pcm = Uint8List.fromList([0x01, 0x02, 0x03, 0x04]);
      final wav = _buildWav(sampleRate: 16000, channels: 1, pcmData: pcm);

      final result = parseWavPcm16(wav);

      expect(result.sampleRate, 16000);
      expect(result.channels, 1);
      expect(result.pcmBytes, pcm);
    });

    test('handles stereo files', () {
      final pcm = Uint8List.fromList(List.generate(16, (i) => i));
      final wav = _buildWav(sampleRate: 44100, channels: 2, pcmData: pcm);

      final result = parseWavPcm16(wav);

      expect(result.sampleRate, 44100);
      expect(result.channels, 2);
      expect(result.pcmBytes, pcm);
    });

    test('skips extra chunks before locating fmt/data', () {
      final pcm = Uint8List.fromList([0xAA, 0xBB]);
      final base = _buildWav(sampleRate: 8000, channels: 1, pcmData: pcm);
      // Splice a fake "LIST" chunk in right after the RIFF/WAVE header.
      final extraBody = Uint8List.fromList([0x01, 0x02, 0x03, 0x04]);
      final builder = BytesBuilder();
      builder.add(base.sublist(0, 12)); // RIFF....WAVE
      builder.add('LIST'.codeUnits);
      final sizeBytes = ByteData(4)..setUint32(0, extraBody.length, Endian.little);
      builder.add(sizeBytes.buffer.asUint8List());
      builder.add(extraBody);
      builder.add(base.sublist(12)); // fmt + data chunks
      final spliced = Uint8List.fromList(builder.toBytes());

      final result = parseWavPcm16(spliced);

      expect(result.sampleRate, 8000);
      expect(result.pcmBytes, pcm);
    });

    test('throws when the file is too short', () {
      expect(
        () => parseWavPcm16(Uint8List(10)),
        throwsA(isA<WavParseException>()),
      );
    });

    test('throws when RIFF/WAVE magic is missing', () {
      final bytes = Uint8List(44);
      expect(() => parseWavPcm16(bytes), throwsA(isA<WavParseException>()));
    });

    test('throws on non-PCM audio format', () {
      final pcm = Uint8List.fromList([0, 0]);
      final wav = _buildWav(sampleRate: 16000, channels: 1, pcmData: pcm);
      // audioFormat is at byte offset 20 (12 header + 8 chunk header + 0)
      wav[20] = 3; // IEEE float, not PCM
      wav[21] = 0;

      expect(() => parseWavPcm16(wav), throwsA(isA<WavParseException>()));
    });

    test('throws when data chunk is missing', () {
      final builder = BytesBuilder();
      void writeAscii(String s) => builder.add(s.codeUnits);
      void writeU32(int v) {
        final b = ByteData(4)..setUint32(0, v, Endian.little);
        builder.add(b.buffer.asUint8List());
      }

      void writeU16(int v) {
        final b = ByteData(2)..setUint16(0, v, Endian.little);
        builder.add(b.buffer.asUint8List());
      }

      writeAscii('RIFF');
      writeU32(36);
      writeAscii('WAVE');
      writeAscii('fmt ');
      writeU32(16);
      writeU16(1);
      writeU16(1);
      writeU32(16000);
      writeU32(32000);
      writeU16(2);
      writeU16(16);

      expect(
        () => parseWavPcm16(Uint8List.fromList(builder.toBytes())),
        throwsA(isA<WavParseException>()),
      );
    });
  });
}
