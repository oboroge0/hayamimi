import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:archive/archive.dart';
import 'package:crypto/crypto.dart';

/// Thrown by [downloadProfile] on an HTTP failure, a sha256 mismatch (of a
/// download or of an already-on-disk file being verified), a missing
/// archive member, or any other step of getting a [ModelProfile]'s files
/// onto disk.
class ModelDownloadException implements Exception {
  ModelDownloadException(this.message);
  final String message;

  @override
  String toString() => 'ModelDownloadException: $message';
}

/// Which bundle of on-device models to fetch. Mirrors the two profiles
/// documented in the package README's "Recommended configurations" table
/// (`RoutingProfile.jaOnly`/`RoutingProfile.jaSenseVoice` in
/// `lib/routing/routing_profile.dart`), but is kept as its own enum so this
/// file has no dependency on the recognizer/routing code -- a host app can
/// download models without pulling in `HayamimiLive`.
enum ModelProfile {
  /// ReazonSpeech ja zipformer transducer (int8) + Silero VAD.
  /// ~72 MB once extracted.
  jaOnly,

  /// [jaOnly]'s files plus SenseVoice small (int8) and whisper-tiny (int8,
  /// used as a language-ID probe). ~396 MB once extracted.
  jaSenseVoice,
}

/// How a [ModelSource]'s downloaded bytes become on-disk files.
enum ModelArchiveType {
  /// The download itself *is* the one target file (e.g. `silero_vad.onnx`,
  /// which sherpa-onnx ships as a bare file, not an archive).
  rawFile,

  /// The download is a `.tar.bz2` archive; each [ModelPlacement] names a
  /// member path inside it to extract.
  tarBz2,
}

/// One file this package's [ModelSource] must have on disk afterward, at
/// `<targetDir>/<subdir>/<filename>`.
///
/// The subdir values used by the manifest below (`'model'`, `'vad'`,
/// `'sense_voice'`, `'lid'`) match the layout `example/lib/main.dart` and
/// `mobile/lib/live/live_page.dart` resolve their `HayamimiLive.start`
/// arguments from (`Documents/model`, `Documents/vad/silero_vad.onnx`,
/// `Documents/sense_voice`, `Documents/lid`), and `'model'`/`'sense_voice'`
/// are loadable as-is by `resolveZipformerTransducerFiles`/
/// `resolveOnnxFile` (`lib/bench/model_file_resolver.dart`), which match
/// files by role substring, not exact name -- so extracting the archive's
/// own filenames (rather than renaming them) is sufficient.
class ModelPlacement {
  const ModelPlacement({
    required this.subdir,
    required this.filename,
    this.archiveMemberPath,
    this.sha256,
  });

  final String subdir;
  final String filename;

  /// Path of this file inside the [ModelSource]'s archive. `null` for a
  /// [ModelArchiveType.rawFile] source, where the whole download becomes
  /// this one placement.
  final String? archiveMemberPath;

  /// sha256 of this exact placed file, when known. Lets [downloadProfile]
  /// treat a [ModelSource] as already-done (skipping the download and any
  /// extraction) once every one of its placements is on disk and passes
  /// this check, and lets it tell a corrupt file apart from a merely
  /// missing one on re-run. `null` means "don't verify this file
  /// individually" -- verification still happens once for the whole
  /// download via [ModelSource.sha256] when that is set.
  final String? sha256;
}

/// One HTTP download (an archive, or a bare file) plus the files it must
/// yield on disk.
class ModelSource {
  const ModelSource({
    required this.url,
    required this.archiveType,
    required this.placements,
    this.sha256,
  });

  final Uri url;
  final ModelArchiveType archiveType;
  final List<ModelPlacement> placements;

  /// sha256 of the raw download (the archive bytes, or for
  /// [ModelArchiveType.rawFile] the file's own bytes), checked right after
  /// the transfer completes and before anything is extracted/placed.
  /// `null` means verification of the download itself is skipped for this
  /// source (see the doc comment on [modelManifest] for which entries, if
  /// any, that applies to and why); per-placement [ModelPlacement.sha256]
  /// checks, where present, still run independently of this.
  final String? sha256;

  String get _downloadFileName => url.pathSegments.last;
}

/// The two profiles' sources, with real sherpa-onnx GitHub release asset
/// URLs (tag `asr-models`) and sha256 checksums computed from an actual
/// download of every asset below -- see this package's PR description /
/// commit message for the verification session; none of these are
/// invented. `sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2`
/// and `sherpa-onnx-whisper-tiny.tar.bz2` each bundle fp32/fp16/int8
/// variants together (there is no int8-only release asset for either),
/// so only the `int8` members are extracted -- matching the README's
/// "Recommended configurations" guidance and
/// `resolveOnnxFile`'s int8-preference rule.
final Map<ModelProfile, List<ModelSource>> modelManifest = {
  ModelProfile.jaOnly: [_reazonspeechJa, _sileroVad],
  ModelProfile.jaSenseVoice: [
    _reazonspeechJa,
    _sileroVad,
    _senseVoice,
    _whisperTinyLid,
  ],
};

const _reazonspeechDir =
    'sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17';

final _reazonspeechJa = ModelSource(
  url: Uri.parse(
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
    '$_reazonspeechDir.tar.bz2',
  ),
  archiveType: ModelArchiveType.tarBz2,
  sha256: 'dc03758608c0280e2cbcaac4597467ffcf846ae0b06436f1706738a11da86f5d',
  placements: [
    ModelPlacement(
      subdir: 'model',
      filename: 'encoder-epoch-35-avg-1.int8.onnx',
      archiveMemberPath: '$_reazonspeechDir/encoder-epoch-35-avg-1.int8.onnx',
      sha256: 'ead1579e118b821a767242a8eb9272634b0e63ba16f8dfc4d126732406eae268',
    ),
    ModelPlacement(
      subdir: 'model',
      filename: 'decoder-epoch-35-avg-1.int8.onnx',
      archiveMemberPath: '$_reazonspeechDir/decoder-epoch-35-avg-1.int8.onnx',
      sha256: 'd0179db78a2e65445c5c3dc41e94c62068fc539fe4e45060e32f438cca76432f',
    ),
    ModelPlacement(
      subdir: 'model',
      filename: 'joiner-epoch-35-avg-1.int8.onnx',
      archiveMemberPath: '$_reazonspeechDir/joiner-epoch-35-avg-1.int8.onnx',
      sha256: 'c7f4ba40a8ae307a6c30b5c06e2570add04466bcb45bab62699f0ec5d00ed495',
    ),
    ModelPlacement(
      subdir: 'model',
      filename: 'tokens.txt',
      archiveMemberPath: '$_reazonspeechDir/tokens.txt',
      sha256: '144f8a4f639373a1bdf7eabb2437482ef64b0cc5db24ad27cce65f293e4faa24',
    ),
  ],
);

final _sileroVad = ModelSource(
  url: Uri.parse(
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
    'silero_vad.onnx',
  ),
  archiveType: ModelArchiveType.rawFile,
  sha256: '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6',
  placements: const [
    ModelPlacement(
      subdir: 'vad',
      filename: 'silero_vad.onnx',
      sha256:
          '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6',
    ),
  ],
);

const _senseVoiceDir = 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17';

final _senseVoice = ModelSource(
  url: Uri.parse(
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
    '$_senseVoiceDir.tar.bz2',
  ),
  archiveType: ModelArchiveType.tarBz2,
  sha256: '7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e',
  placements: [
    ModelPlacement(
      subdir: 'sense_voice',
      filename: 'model.int8.onnx',
      archiveMemberPath: '$_senseVoiceDir/model.int8.onnx',
      sha256: 'c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51',
    ),
    ModelPlacement(
      subdir: 'sense_voice',
      filename: 'tokens.txt',
      archiveMemberPath: '$_senseVoiceDir/tokens.txt',
      sha256: 'f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc',
    ),
  ],
);

const _whisperTinyDir = 'sherpa-onnx-whisper-tiny';

final _whisperTinyLid = ModelSource(
  url: Uri.parse(
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
    '$_whisperTinyDir.tar.bz2',
  ),
  archiveType: ModelArchiveType.tarBz2,
  sha256: 'c46116994e539aa165266d96b325252728429c12535eb9d8b6a2b10f129e66b1',
  placements: [
    ModelPlacement(
      subdir: 'lid',
      filename: 'tiny-encoder.int8.onnx',
      archiveMemberPath: '$_whisperTinyDir/tiny-encoder.int8.onnx',
      sha256: 'd24fb083ae3b1041fc24e97971d60e280c9342201fbb67b0ab428a8b4a51a434',
    ),
    ModelPlacement(
      subdir: 'lid',
      filename: 'tiny-decoder.int8.onnx',
      archiveMemberPath: '$_whisperTinyDir/tiny-decoder.int8.onnx',
      sha256: 'd2fece8dd42771f1df975c6c0445770d0c292bf7547c2cae04a6c0cc57540925',
    ),
    ModelPlacement(
      subdir: 'lid',
      filename: 'tiny-tokens.txt',
      archiveMemberPath: '$_whisperTinyDir/tiny-tokens.txt',
      sha256: 'b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126',
    ),
  ],
);

/// What step of a [ModelSource] a [ModelDownloadEvent] is reporting on.
enum ModelDownloadPhase {
  /// Every placement for this source was already on disk and passed its
  /// checksum -- nothing was downloaded.
  skipped,

  /// Streaming the source's bytes from the network.
  downloading,

  /// Verifying the completed download's sha256 against
  /// [ModelSource.sha256].
  verifyingDownload,

  /// Extracting a `.tar.bz2` source's members to their placements (a
  /// no-op phase for [ModelArchiveType.rawFile] sources).
  extracting,

  /// This source's placements are all on disk and verified.
  done,
}

/// One progress update from [downloadProfile], for one [ModelSource].
class ModelDownloadEvent {
  const ModelDownloadEvent({
    required this.sourceUrl,
    required this.sourceIndex,
    required this.sourceCount,
    required this.phase,
    this.bytesReceived = 0,
    this.totalBytes,
  });

  final Uri sourceUrl;

  /// 0-based index of this source within the profile's source list.
  final int sourceIndex;
  final int sourceCount;
  final ModelDownloadPhase phase;

  /// Bytes received so far, meaningful only during
  /// [ModelDownloadPhase.downloading].
  final int bytesReceived;

  /// Total bytes expected, when the server sent a `Content-Length`.
  /// `null` otherwise (progress UIs should fall back to an indeterminate
  /// spinner/counter in that case).
  final int? totalBytes;
}

/// [downloadProfile]/[downloadModelSource]'s `onProgress` callback type --
/// called once per [ModelDownloadEvent], e.g. to drive a progress bar.
typedef ModelDownloadProgress = void Function(ModelDownloadEvent event);

/// Transport [downloadProfile] uses to fetch a URL, factored out so tests
/// can inject a fake instead of hitting the network -- mirrors how
/// `test/fake_record_platform.dart` fakes the `record` plugin for
/// [LiveTranscriber] lifecycle tests. The default implementation (below)
/// uses `dart:io`'s [HttpClient], which follows redirects automatically;
/// GitHub release asset URLs always redirect once to
/// `objects.githubusercontent.com`.
abstract class ModelDownloadTransport {
  Future<ModelDownloadResponse> get(Uri url);
}

/// A [ModelDownloadTransport.get] result: the response body as a byte
/// stream, plus its declared length when the server sent one.
class ModelDownloadResponse {
  const ModelDownloadResponse({required this.contentLength, required this.stream});

  /// `null` when the server didn't send a `Content-Length` header.
  final int? contentLength;

  /// The response body, in whatever chunks the transport delivers them.
  final Stream<List<int>> stream;
}

class _HttpClientTransport implements ModelDownloadTransport {
  const _HttpClientTransport();

  @override
  Future<ModelDownloadResponse> get(Uri url) async {
    final client = HttpClient();
    HttpClientResponse response;
    try {
      final request = await client.getUrl(url);
      response = await request.close();
    } catch (e) {
      client.close(force: true);
      throw ModelDownloadException('GET $url failed: $e');
    }
    if (response.statusCode != 200) {
      await response.drain<void>();
      client.close(force: true);
      throw ModelDownloadException(
        'GET $url returned HTTP ${response.statusCode}',
      );
    }
    final contentLength = response.contentLength >= 0
        ? response.contentLength
        : null;
    // Close the underlying HttpClient once the body stream finishes or
    // errors, since nothing else owns that lifecycle once this returns.
    final controller = StreamController<List<int>>();
    unawaited(
      controller.addStream(response).then(
        (_) => controller.close(),
        onError: (Object e, StackTrace st) {
          controller.addError(e, st);
          controller.close();
        },
      ).whenComplete(() => client.close(force: true)),
    );
    return ModelDownloadResponse(
      contentLength: contentLength,
      stream: controller.stream,
    );
  }
}

/// Downloads, verifies, and places every file [profile] needs under
/// `<targetDir>/<subdir>/<filename>` (see [ModelPlacement] for the subdir
/// layout), ready for [HayamimiLive.start] to load unchanged.
///
/// Idempotent: sources whose every placement is already on disk and
/// passes its [ModelPlacement.sha256] check are skipped entirely (no
/// network request). A source with a missing or corrupt placement is
/// re-downloaded and re-extracted/placed in full -- placements aren't
/// patched individually, since they come from one archive download.
///
/// Throws [ModelDownloadException] on any HTTP failure, checksum
/// mismatch, or missing archive member; files partially written for the
/// failing source are left in place for inspection rather than deleted
/// (a caller that wants a clean retry can just call this again -- the
/// idempotency check above will re-verify or re-fetch them).
Future<void> downloadProfile(
  ModelProfile profile,
  String targetDir, {
  ModelDownloadProgress? onProgress,
  ModelDownloadTransport transport = const _HttpClientTransport(),
}) async {
  final sources = modelManifest[profile];
  if (sources == null) {
    throw ModelDownloadException('Unknown model profile: $profile');
  }
  for (var i = 0; i < sources.length; i++) {
    await downloadModelSource(
      sources[i],
      targetDir,
      sourceIndex: i,
      sourceCount: sources.length,
      onProgress: onProgress,
      transport: transport,
    );
  }
}

/// Downloads, verifies, and places one [ModelSource]'s files under
/// `<targetDir>/<subdir>/<filename>`. This is what [downloadProfile] calls
/// once per source in a profile's manifest entry; it's exposed directly so
/// tests (and callers who want just one asset) can drive it against a
/// [ModelSource] built by hand and an injected [transport], without going
/// through a whole [ModelProfile]. [sourceIndex]/[sourceCount] are only
/// used to stamp [ModelDownloadEvent]s -- pass the defaults for a
/// standalone call.
///
/// See [downloadProfile] for the idempotency and error-handling behavior;
/// this is where both are actually implemented.
Future<void> downloadModelSource(
  ModelSource source,
  String targetDir, {
  int sourceIndex = 0,
  int sourceCount = 1,
  ModelDownloadProgress? onProgress,
  ModelDownloadTransport transport = const _HttpClientTransport(),
}) async {
  final sep = Platform.pathSeparator;
  File placementFile(ModelPlacement p) =>
      File('$targetDir$sep${p.subdir}$sep${p.filename}');

  void emit(
    ModelDownloadPhase phase, {
    int bytesReceived = 0,
    int? totalBytes,
  }) {
    onProgress?.call(
      ModelDownloadEvent(
        sourceUrl: source.url,
        sourceIndex: sourceIndex,
        sourceCount: sourceCount,
        phase: phase,
        bytesReceived: bytesReceived,
        totalBytes: totalBytes,
      ),
    );
  }

  if (await _allPlacementsVerified(source, placementFile)) {
    emit(ModelDownloadPhase.skipped);
    return;
  }

  final downloadDir = Directory('$targetDir$sep.hayamimi_model_downloads');
  await downloadDir.create(recursive: true);
  final tempFile = File('${downloadDir.path}$sep${source._downloadFileName}');

  emit(ModelDownloadPhase.downloading, bytesReceived: 0, totalBytes: null);
  try {
    final response = await transport.get(source.url);
    final sink = tempFile.openWrite();
    var received = 0;
    try {
      await for (final chunk in response.stream) {
        sink.add(chunk);
        received += chunk.length;
        emit(
          ModelDownloadPhase.downloading,
          bytesReceived: received,
          totalBytes: response.contentLength,
        );
      }
      await sink.flush();
    } finally {
      await sink.close();
    }

    if (source.sha256 != null) {
      emit(ModelDownloadPhase.verifyingDownload, totalBytes: received);
      final actual = await _fileSha256(tempFile.path);
      if (actual != source.sha256) {
        throw ModelDownloadException(
          'sha256 mismatch for ${source.url}: expected ${source.sha256}, '
          'got $actual',
        );
      }
    }

    emit(ModelDownloadPhase.extracting);
    switch (source.archiveType) {
      case ModelArchiveType.rawFile:
        final placement = source.placements.single;
        final target = placementFile(placement);
        await target.parent.create(recursive: true);
        await tempFile.copy(target.path);
        if (placement.sha256 != null) {
          final actual = await _fileSha256(target.path);
          if (actual != placement.sha256) {
            throw ModelDownloadException(
              'sha256 mismatch for ${target.path}: expected '
              '${placement.sha256}, got $actual',
            );
          }
        }
      case ModelArchiveType.tarBz2:
        final members = <String, (String, String?)>{
          for (final p in source.placements)
            p.archiveMemberPath!: (placementFile(p).path, p.sha256),
        };
        try {
          await Isolate.run(() => _extractTarBz2Sync(tempFile.path, members));
        } catch (e) {
          throw ModelDownloadException(
            'Failed extracting ${source.url}: $e',
          );
        }
    }
  } finally {
    if (await tempFile.exists()) {
      await tempFile.delete();
    }
  }

  emit(ModelDownloadPhase.done);
}

/// Whether every one of [source]'s placements already exists on disk and
/// (when it declares a checksum) matches it.
Future<bool> _allPlacementsVerified(
  ModelSource source,
  File Function(ModelPlacement) placementFile,
) async {
  for (final placement in source.placements) {
    final file = placementFile(placement);
    if (!await file.exists()) {
      return false;
    }
    if (placement.sha256 != null) {
      final actual = await _fileSha256(file.path);
      if (actual != placement.sha256) {
        return false;
      }
    }
  }
  return true;
}

/// Hashes a file off the main isolate -- these can be tens to hundreds of
/// MB (see `docs/MOBILE.md`'s size table), and this runs both on every
/// idempotency check and after every download, so keeping it off whatever
/// isolate the caller is on matters the same way model loading does (see
/// `lib/live/native_model_loader.dart`).
Future<String> _fileSha256(String path) =>
    Isolate.run(() => sha256.convert(File(path).readAsBytesSync()).toString());

/// Runs on a background isolate (via [Isolate.run] in [_downloadSource]):
/// decompresses [archivePath] (a `.tar.bz2`), and writes each member in
/// [members] (archive member path -> (target file path, expected sha256))
/// to its target, verifying the checksum first when one is given. Throws
/// a plain [Exception] (not [ModelDownloadException] -- kept a built-in
/// type so it survives the isolate round trip without any custom-class
/// serialization assumptions) if a member is missing or fails its check.
void _extractTarBz2Sync(
  String archivePath,
  Map<String, (String, String?)> members,
) {
  final compressed = File(archivePath).readAsBytesSync();
  final decompressed = BZip2Decoder().decodeBytes(compressed);
  final archive = TarDecoder().decodeBytes(decompressed);

  final remaining = Set<String>.from(members.keys);
  for (final entry in archive) {
    final spec = members[entry.name];
    if (spec == null || !entry.isFile) {
      continue;
    }
    final (targetPath, expectedSha256) = spec;
    final content = entry.content;
    if (expectedSha256 != null) {
      final actual = sha256.convert(content).toString();
      if (actual != expectedSha256) {
        throw Exception(
          'sha256 mismatch for archive member ${entry.name}: expected '
          '$expectedSha256, got $actual',
        );
      }
    }
    final targetFile = File(targetPath);
    targetFile.parent.createSync(recursive: true);
    targetFile.writeAsBytesSync(content);
    remaining.remove(entry.name);
  }

  if (remaining.isNotEmpty) {
    throw Exception(
      'Archive $archivePath is missing expected member(s): '
      '${remaining.join(', ')}',
    );
  }
}
