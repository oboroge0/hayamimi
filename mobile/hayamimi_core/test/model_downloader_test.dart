import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';

/// In-memory stand-in for [ModelDownloadTransport], the same pattern
/// `test/fake_record_platform.dart` uses for the `record` plugin: no real
/// network access happens in this file.
class FakeTransport implements ModelDownloadTransport {
  FakeTransport(this.body, {this.contentLength, this.chunkSize = 7});

  /// Responses keyed by URL. `null` for a given call count means "throw".
  Uint8List body;
  int? contentLength;
  final int chunkSize;

  int callCount = 0;
  final List<Uri> requestedUrls = [];

  @override
  Future<ModelDownloadResponse> get(Uri url) async {
    callCount++;
    requestedUrls.add(url);
    final chunks = <List<int>>[];
    for (var i = 0; i < body.length; i += chunkSize) {
      chunks.add(
        body.sublist(i, i + chunkSize > body.length ? body.length : i + chunkSize),
      );
    }
    return ModelDownloadResponse(
      contentLength: contentLength,
      stream: Stream.fromIterable(chunks),
    );
  }
}

class ThrowingTransport implements ModelDownloadTransport {
  @override
  Future<ModelDownloadResponse> get(Uri url) async {
    throw StateError('FakeTransport.get should not have been called for $url');
  }
}

String _sha256Hex(List<int> bytes) => sha256.convert(bytes).toString();

/// Builds a `.tar.bz2` fixture in memory containing [files] (archive member
/// path -> file content), mirroring the real sherpa-onnx release assets
/// this package downloads (a tar archive, bzip2-compressed) at a scale
/// that's actually fast to build/decode in a test.
Uint8List _buildTarBz2(Map<String, String> files) {
  final archive = Archive();
  for (final entry in files.entries) {
    archive.addFile(ArchiveFile.string(entry.key, entry.value));
  }
  final tarBytes = TarEncoder().encodeBytes(archive);
  return BZip2Encoder().encodeBytes(tarBytes);
}

void main() {
  group('modelManifest integrity', () {
    test('jaOnly "model" placements are loadable by '
        'resolveZipformerTransducerFiles', () {
      final reazon = modelManifest[ModelProfile.jaOnly]!.firstWhere(
        (s) => s.placements.any((p) => p.subdir == 'model'),
      );
      final filenames = reazon.placements
          .where((p) => p.subdir == 'model')
          .map((p) => p.filename)
          .toList();

      final resolved = resolveZipformerTransducerFiles(filenames);
      expect(resolved.encoder, contains('int8'));
      expect(resolved.decoder, contains('int8'));
      expect(resolved.joiner, contains('int8'));
      expect(resolved.tokens, 'tokens.txt');
    });

    test('jaOnly includes silero_vad.onnx under vad/', () {
      final placements = modelManifest[ModelProfile.jaOnly]!
          .expand((s) => s.placements)
          .toList();
      final vad = placements.singleWhere((p) => p.subdir == 'vad');
      expect(vad.filename, 'silero_vad.onnx');
    });

    test('jaSenseVoice "sense_voice" placements are loadable by '
        'resolveOnnxFile', () {
      final placements = modelManifest[ModelProfile.jaSenseVoice]!
          .expand((s) => s.placements)
          .where((p) => p.subdir == 'sense_voice')
          .toList();
      final filenames = placements.map((p) => p.filename).toList();

      expect(filenames, contains('tokens.txt'));
      final model = resolveOnnxFile(filenames);
      expect(model, 'model.int8.onnx');
    });

    test('jaSenseVoice "lid" placements are loadable by '
        'resolveOnnxFilePair(requireInt8: true)', () {
      final placements = modelManifest[ModelProfile.jaSenseVoice]!
          .expand((s) => s.placements)
          .where((p) => p.subdir == 'lid')
          .toList();
      final filenames = placements.map((p) => p.filename).toList();

      final pair = resolveOnnxFilePair(
        filenames,
        role1: 'encoder',
        role2: 'decoder',
        requireInt8: true,
      );
      expect(pair.first, contains('int8'));
      expect(pair.second, contains('int8'));
    });

    test('jaSenseVoice is a superset of jaOnly\'s sources', () {
      final jaOnly = modelManifest[ModelProfile.jaOnly]!;
      final jaSenseVoice = modelManifest[ModelProfile.jaSenseVoice]!;
      for (final source in jaOnly) {
        expect(jaSenseVoice, contains(source));
      }
    });

    test('every declared sha256 is a 64-char lowercase hex string', () {
      for (final sources in modelManifest.values) {
        for (final source in sources) {
          if (source.sha256 != null) {
            expect(source.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
          }
          for (final placement in source.placements) {
            if (placement.sha256 != null) {
              expect(placement.sha256, matches(RegExp(r'^[0-9a-f]{64}$')));
            }
          }
        }
      }
    });

    test('tarBz2 sources give every placement an archiveMemberPath, and '
        'rawFile sources have exactly one placement with none', () {
      for (final sources in modelManifest.values) {
        for (final source in sources) {
          switch (source.archiveType) {
            case ModelArchiveType.tarBz2:
              for (final p in source.placements) {
                expect(p.archiveMemberPath, isNotNull, reason: source.url.toString());
              }
            case ModelArchiveType.rawFile:
              expect(source.placements, hasLength(1));
              expect(source.placements.single.archiveMemberPath, isNull);
          }
        }
      }
    });
  });

  group('downloadModelSource', () {
    late Directory tempDir;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('model_downloader_test_');
    });

    tearDown(() async {
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test('extracts a tar.bz2 fixture and verifies both the archive and '
        'per-file checksums', () async {
      const memberA = 'fixture/a.onnx';
      const memberB = 'fixture/tokens.txt';
      final archiveBytes = _buildTarBz2({
        memberA: 'fake onnx bytes',
        memberB: 'fake tokens',
      });

      final source = ModelSource(
        url: Uri.parse('https://example.invalid/fixture.tar.bz2'),
        archiveType: ModelArchiveType.tarBz2,
        sha256: _sha256Hex(archiveBytes),
        placements: [
          ModelPlacement(
            subdir: 'model',
            filename: 'a.onnx',
            archiveMemberPath: memberA,
            sha256: _sha256Hex(utf8.encode('fake onnx bytes')),
          ),
          const ModelPlacement(
            subdir: 'model',
            filename: 'tokens.txt',
            archiveMemberPath: memberB,
            // Deliberately unverified at the placement level -- only the
            // whole-archive sha256 above covers this one.
          ),
        ],
      );

      final events = <ModelDownloadEvent>[];
      await downloadModelSource(
        source,
        tempDir.path,
        transport: FakeTransport(archiveBytes, contentLength: archiveBytes.length),
        onProgress: events.add,
      );

      final aFile = File('${tempDir.path}${Platform.pathSeparator}model'
          '${Platform.pathSeparator}a.onnx');
      final tokensFile = File('${tempDir.path}${Platform.pathSeparator}model'
          '${Platform.pathSeparator}tokens.txt');
      expect(await aFile.readAsString(), 'fake onnx bytes');
      expect(await tokensFile.readAsString(), 'fake tokens');

      // One or more `downloading` events (one per fake-transport chunk,
      // starting from an initial 0-byte event), then exactly one each of
      // verifyingDownload/extracting/done, in that order.
      final tail = events.sublist(events.length - 3).map((e) => e.phase);
      expect(tail, [
        ModelDownloadPhase.verifyingDownload,
        ModelDownloadPhase.extracting,
        ModelDownloadPhase.done,
      ]);
      final downloadingEvents = events.sublist(0, events.length - 3);
      expect(downloadingEvents, isNotEmpty);
      expect(
        downloadingEvents.every((e) => e.phase == ModelDownloadPhase.downloading),
        isTrue,
      );
      expect(downloadingEvents.first.bytesReceived, 0);

      // The temp download (the archive itself) must not be left behind.
      final downloadDir = Directory(
        '${tempDir.path}${Platform.pathSeparator}.hayamimi_model_downloads',
      );
      expect(await downloadDir.list().toList(), isEmpty);
    });

    test('a rawFile source is written straight to its placement', () async {
      final bytes = Uint8List.fromList(utf8.encode('silero vad bytes'));
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/silero_vad.onnx'),
        archiveType: ModelArchiveType.rawFile,
        sha256: _sha256Hex(bytes),
        placements: [
          ModelPlacement(
            subdir: 'vad',
            filename: 'silero_vad.onnx',
            sha256: _sha256Hex(bytes),
          ),
        ],
      );

      await downloadModelSource(
        source,
        tempDir.path,
        transport: FakeTransport(bytes, contentLength: bytes.length),
      );

      final target = File(
        '${tempDir.path}${Platform.pathSeparator}vad${Platform.pathSeparator}'
        'silero_vad.onnx',
      );
      expect(await target.readAsString(), 'silero vad bytes');
    });

    test('is idempotent: a second call makes no network request once '
        'placements already match their checksums', () async {
      final bytes = Uint8List.fromList(utf8.encode('silero vad bytes'));
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/silero_vad.onnx'),
        archiveType: ModelArchiveType.rawFile,
        sha256: _sha256Hex(bytes),
        placements: [
          ModelPlacement(
            subdir: 'vad',
            filename: 'silero_vad.onnx',
            sha256: _sha256Hex(bytes),
          ),
        ],
      );

      final fake = FakeTransport(bytes, contentLength: bytes.length);
      await downloadModelSource(source, tempDir.path, transport: fake);
      expect(fake.callCount, 1);

      final events = <ModelDownloadEvent>[];
      // A transport that throws if it's ever called -- proves the second
      // run does no network I/O at all.
      await downloadModelSource(
        source,
        tempDir.path,
        transport: ThrowingTransport(),
        onProgress: events.add,
      );
      expect(events, [
        isA<ModelDownloadEvent>().having(
          (e) => e.phase,
          'phase',
          ModelDownloadPhase.skipped,
        ),
      ]);
    });

    test('a corrupted on-disk placement is treated as missing and '
        're-fetched', () async {
      final bytes = Uint8List.fromList(utf8.encode('silero vad bytes'));
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/silero_vad.onnx'),
        archiveType: ModelArchiveType.rawFile,
        sha256: _sha256Hex(bytes),
        placements: [
          ModelPlacement(
            subdir: 'vad',
            filename: 'silero_vad.onnx',
            sha256: _sha256Hex(bytes),
          ),
        ],
      );

      final fake = FakeTransport(bytes, contentLength: bytes.length);
      await downloadModelSource(source, tempDir.path, transport: fake);
      expect(fake.callCount, 1);

      final target = File(
        '${tempDir.path}${Platform.pathSeparator}vad${Platform.pathSeparator}'
        'silero_vad.onnx',
      );
      await target.writeAsString('corrupted!');

      await downloadModelSource(source, tempDir.path, transport: fake);
      expect(fake.callCount, 2);
      expect(await target.readAsString(), 'silero vad bytes');
    });

    test('throws ModelDownloadException on a download sha256 mismatch, '
        'and does not place any file', () async {
      final bytes = Uint8List.fromList(utf8.encode('silero vad bytes'));
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/silero_vad.onnx'),
        archiveType: ModelArchiveType.rawFile,
        sha256: '0' * 64, // deliberately wrong
        placements: [
          ModelPlacement(subdir: 'vad', filename: 'silero_vad.onnx'),
        ],
      );

      await expectLater(
        downloadModelSource(
          source,
          tempDir.path,
          transport: FakeTransport(bytes, contentLength: bytes.length),
        ),
        throwsA(isA<ModelDownloadException>()),
      );

      final target = File(
        '${tempDir.path}${Platform.pathSeparator}vad${Platform.pathSeparator}'
        'silero_vad.onnx',
      );
      expect(await target.exists(), isFalse);
    });

    test('throws ModelDownloadException when a declared archive member is '
        'missing from the actual archive', () async {
      final archiveBytes = _buildTarBz2({'fixture/present.onnx': 'x'});
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/fixture.tar.bz2'),
        archiveType: ModelArchiveType.tarBz2,
        sha256: _sha256Hex(archiveBytes),
        placements: const [
          ModelPlacement(
            subdir: 'model',
            filename: 'missing.onnx',
            archiveMemberPath: 'fixture/missing.onnx',
          ),
        ],
      );

      await expectLater(
        downloadModelSource(
          source,
          tempDir.path,
          transport: FakeTransport(archiveBytes, contentLength: archiveBytes.length),
        ),
        throwsA(
          isA<ModelDownloadException>().having(
            (e) => e.message,
            'message',
            contains('missing'),
          ),
        ),
      );
    });

    test('progress events accumulate bytesReceived up to the full length',
        () async {
      final bytes = Uint8List.fromList(List.generate(101, (i) => i % 256));
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/blob.onnx'),
        archiveType: ModelArchiveType.rawFile,
        sha256: _sha256Hex(bytes),
        placements: [
          ModelPlacement(
            subdir: 'vad',
            filename: 'blob.onnx',
            sha256: _sha256Hex(bytes),
          ),
        ],
      );

      final events = <ModelDownloadEvent>[];
      await downloadModelSource(
        source,
        tempDir.path,
        transport: FakeTransport(bytes, contentLength: bytes.length, chunkSize: 10),
        onProgress: events.add,
      );

      final downloading = events
          .where((e) => e.phase == ModelDownloadPhase.downloading)
          .toList();
      expect(downloading.last.bytesReceived, bytes.length);
      expect(downloading.last.totalBytes, bytes.length);
      // Monotonically non-decreasing.
      for (var i = 1; i < downloading.length; i++) {
        expect(
          downloading[i].bytesReceived,
          greaterThanOrEqualTo(downloading[i - 1].bytesReceived),
        );
      }
    });

    test('works with no Content-Length (totalBytes stays null while '
        'downloading)', () async {
      final bytes = Uint8List.fromList(utf8.encode('no length header'));
      final source = ModelSource(
        url: Uri.parse('https://example.invalid/blob.onnx'),
        archiveType: ModelArchiveType.rawFile,
        placements: const [ModelPlacement(subdir: 'vad', filename: 'blob.onnx')],
      );

      final events = <ModelDownloadEvent>[];
      await downloadModelSource(
        source,
        tempDir.path,
        transport: FakeTransport(bytes, contentLength: null),
        onProgress: events.add,
      );

      final downloading = events.where(
        (e) => e.phase == ModelDownloadPhase.downloading,
      );
      expect(downloading, isNotEmpty);
      for (final e in downloading) {
        expect(e.totalBytes, isNull);
      }
    });
  });
}
