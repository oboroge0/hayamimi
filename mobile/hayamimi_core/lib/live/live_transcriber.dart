import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../bench/model_file_resolver.dart';
import '../bench/model_kind.dart';
import '../routing/routed_recognizer.dart';
import '../routing/routing_profile.dart';
import 'draft_pass.dart';
import 'live_transcript_entry.dart';
import 'native_model_loader.dart';
import 'pcm_frame_buffer.dart';
import 'refine_pass.dart';
import 'speech_segment_filter.dart';

/// How often the auto-refine due check ([isAutoRefineDue]) runs while a
/// session is live and auto mode is on. Cheap (a few comparisons), so a 1s
/// tick is plenty responsive without meaningfully affecting battery.
const _autoRefineCheckInterval = Duration(seconds: 1);

class LiveTranscriberException implements Exception {
  LiveTranscriberException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Streams mic audio through Silero VAD to find speech segments, decodes
/// each finalized segment with the same [sherpa_onnx.OfflineRecognizer]
/// `BenchRunner` uses, and emits one [LiveTranscriptEntry] per segment.
///
/// This class owns the mic stream, the native VAD/recognizer handles, and
/// the glue between them, so it isn't unit-testable on its own (it needs a
/// real device/emulator mic and the sherpa-onnx native libs). The pure
/// pieces it's built from — [pcm16BytesToFloat32], [PcmFrameBuffer], and
/// [isSegmentWorthDecoding] — are unit tested separately.
///
/// Threading: model *loading* happens on a background isolate (see
/// `native_model_loader.dart`), so [start] doesn't freeze the UI. Per-segment
/// *decoding* still runs synchronously on whichever isolate drives the mic
/// stream — see the README's "Threading / known limitations".
class LiveTranscriber {
  LiveTranscriber({AudioRecorder? recorder})
    : _recorder = recorder ?? AudioRecorder();

  static const int sampleRate = 16000;

  final AudioRecorder _recorder;
  final _entriesController = StreamController<LiveTranscriptEntry>.broadcast();
  final _decodingController = StreamController<bool>.broadcast();
  final _refineEntriesController =
      StreamController<LiveTranscriptEntry>.broadcast();
  final _draftController = StreamController<LiveTranscriptEntry>.broadcast();
  final _errorsController =
      StreamController<LiveTranscriberException>.broadcast();

  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.OfflineRecognizer? _recognizer;
  RoutedRecognizerSet? _routed;
  StreamSubscription<Uint8List>? _micSubscription;
  StreamSubscription<RecordState>? _recorderStateSubscription;
  PcmFrameBuffer? _frameBuffer;

  // --- Draft ("発話中の暫定字幕"): while a VAD segment is still in progress,
  // periodically re-decode what's been captured of it so far and emit a
  // provisional partial -- see `draft_pass.dart` for the timing/skip logic
  // and why it's a coarser, cheaper pass than the fast-final/refine ones.
  bool _busy = false; // true while ANY decode (final/refine/draft) is running
  // Kept trimmed to defaultDraftWindowSeconds as frames arrive (see
  // slideDraftFrames in draft_pass.dart) -- without this, a long
  // uninterrupted utterance would grow this list without bound and
  // _runDraftDecode's concatFloat32Lists over the whole thing, once per
  // draft tick, made the total work across the utterance O(n^2) in its
  // length even though only the trailing defaultDraftWindowSeconds is ever
  // actually decoded.
  List<Float32List> _draftFrames = [];
  bool _draftSegmentActive = false;
  DateTime? _lastDraftAt;
  bool _debugStreaming = false;
  bool _debugStreamCancelRequested = false;
  // Set synchronously by [start]/[startDebugWavStream] before their first
  // await. [isRunning] only flips once the mic subscription exists, several
  // awaits in, so without this a second concurrent start() would sail past
  // the isRunning guard and build a *second* full model set (up to 396 MB),
  // orphaning the first one's handles.
  bool _starting = false;
  bool _disposed = false;

  // --- Two-pass "refine" (清書): buffer finalized segments' audio and, on
  // demand, re-decode the whole group together for a cleaner result than
  // any single segment got on its own (same effect the desktop pipeline's
  // `Refiner` gets from re-decoding across segment boundaries — see
  // `scripts/realtime_transcribe.py`). No punctuation-restoration model is
  // layered on top of this; it's re-decode only.
  final RefineBuffer _refineBuffer = RefineBuffer(sampleRate: sampleRate);
  DateTime? _lastSegmentAt;
  Timer? _autoRefineTimer;
  bool _autoRefineEnabled = false;

  /// Finalized transcript lines, one per detected speech segment.
  Stream<LiveTranscriptEntry> get entries => _entriesController.stream;

  /// Emits `true` right before a segment starts decoding and `false` right
  /// after, so the UI can show a busy indicator. Shared by both the fast
  /// per-segment decode and the refine pass, since only one decode ever
  /// runs at a time on this recognizer.
  Stream<bool> get decoding => _decodingController.stream;

  /// Refine ("清書") results: one entry per manual or automatic refine
  /// pass, each covering everything buffered since the previous refine (or
  /// since the session started).
  Stream<LiveTranscriptEntry> get refineEntries => _refineEntriesController.stream;

  /// In-progress ("draft") decodes: one per draft re-decode while a VAD
  /// segment is still open, replaced by the next draft or cleared by the
  /// segment's eventual [entries] final. Never buffered/stored by this
  /// class -- purely a live "typewriter" signal for the UI/broadcast.
  Stream<LiveTranscriptEntry> get drafts => _draftController.stream;

  /// Asynchronous session failures that aren't attributable to a call the
  /// caller made — today, mic capture dying under a running session (the OS
  /// revoking the mic on backgrounding, another app taking it). The session
  /// is already stopped by the time the error is emitted, so [isRunning]
  /// reflects reality; errors raised *by* [start] itself are thrown, not
  /// emitted here.
  Stream<LiveTranscriberException> get errors => _errorsController.stream;

  bool get isRunning => _micSubscription != null;

  /// Whether [startDebugWavStream] is currently paced-streaming a wav file
  /// through the same VAD/draft/final/refine pipeline a live mic session
  /// uses. See that method's doc for why this exists (emulators have no
  /// usable microphone).
  bool get isDebugStreaming => _debugStreaming;

  /// The session's current language when running with
  /// [RoutingProfile.jaSenseVoice] (`null` before the first segment
  /// resolves one, and always `null` for a plain single-model session).
  String? get currentLang => _routed?.currentLang;

  /// Total audio currently buffered for the next refine pass, in seconds.
  double get refineBufferedSeconds => _refineBuffer.totalDurationSeconds;

  /// Whether auto-refine (silence-triggered, see [refine_pass.dart]) is on.
  /// Defaults to off: on a phone every refine is a full re-decode burning
  /// battery and generating heat, so firing it automatically is opt-in —
  /// the manual "清書" button always works regardless of this setting.
  bool get autoRefineEnabled => _autoRefineEnabled;

  set autoRefineEnabled(bool value) {
    _autoRefineEnabled = value;
    if (!value) {
      _autoRefineTimer?.cancel();
      _autoRefineTimer = null;
    } else if (isRunning) {
      _startAutoRefineTimer();
    }
  }

  /// Starts capturing mic audio and transcribing it.
  ///
  /// [modelDir] must contain a zipformer transducer model (see
  /// `resolveZipformerTransducerFiles`) -- this is the ja tier model dir
  /// even when [routingProfile] is [RoutingProfile.jaSenseVoice].
  /// [vadModelPath] must point at a Silero VAD onnx model file (e.g.
  /// `silero_vad.onnx`).
  ///
  /// When [routingProfile] is [RoutingProfile.jaSenseVoice],
  /// [senseVoiceModelDir] and [lidModelDir] are also required: each VAD
  /// segment is then routed to ReazonSpeech (ja) or SenseVoice
  /// (en/zh/ko/yue) per the dual-LID policy in
  /// `../routing/lang_routing.dart`, and emitted [LiveTranscriptEntry]s
  /// carry [LiveTranscriptEntry.lang].
  ///
  /// Model loading runs on a background isolate (see
  /// `native_model_loader.dart`), so awaiting this does not block the UI.
  /// A call made while a session is already running — or while an earlier
  /// [start]/[startDebugWavStream] is still loading — is a no-op.
  Future<void> start({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
  }) async {
    if (isRunning || _debugStreaming || _starting) {
      return;
    }
    _starting = true;
    try {
      final hasPermission = await _recorder.hasPermission();
      if (!hasPermission) {
        throw LiveTranscriberException(
          'Microphone permission was not granted.',
        );
      }

      await _buildNativeState(
        modelKind: modelKind,
        modelDir: modelDir,
        vadModelPath: vadModelPath,
        routingProfile: routingProfile,
        senseVoiceModelDir: senseVoiceModelDir,
        lidModelDir: lidModelDir,
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
          onError: (Object error) {
            unawaited(_handleCaptureFailure('Microphone capture failed: $error'));
          },
          onDone: () {
            unawaited(
              _handleCaptureFailure(
                'Microphone capture ended unexpectedly (the OS may have '
                'revoked mic access, e.g. on backgrounding or another app '
                'taking the microphone).',
              ),
            );
          },
          cancelOnError: true,
        );
        // record 6.2.1 wraps the platform's audio stream in a broadcast
        // controller that forwards data only (see _startRecordStream in
        // package:record), so the source stream's error/done never reach
        // the handlers above. Its state channel does report an OS-side
        // stop, so watch that as well -- between the two, a revoked mic
        // can't leave this session silently "running".
        _recorderStateSubscription = _recorder.onStateChanged().listen(
          (state) {
            if (state == RecordState.stop) {
              unawaited(
                _handleCaptureFailure(
                  'Microphone capture was stopped by the system (mic access '
                  'revoked, e.g. on backgrounding or another app taking the '
                  'microphone).',
                ),
              );
            }
          },
          onError: (Object error) {
            unawaited(_handleCaptureFailure('Microphone capture failed: $error'));
          },
        );
      } catch (_) {
        await _teardownNativeState();
        rethrow;
      }

      _resetSessionState();
    } finally {
      _starting = false;
    }
  }

  /// The mic stream ended or errored out from under a live session. Without
  /// this the session would keep reporting [isRunning] while silently
  /// receiving no audio forever, so report it on [errors] and tear the
  /// session down so the state matches what's actually happening.
  Future<void> _handleCaptureFailure(String message) async {
    if (!isRunning) {
      return;
    }
    if (!_errorsController.isClosed) {
      _errorsController.add(LiveTranscriberException(message));
    }
    await stop();
  }

  /// Shared "load VAD + recognizer(s) from disk" step behind both [start]
  /// (mic input) and [startDebugWavStream] (paced wav input) -- everything
  /// except the mic-specific permission check and stream subscription.
  Future<void> _buildNativeState({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
    required RoutingProfile routingProfile,
    String? senseVoiceModelDir,
    String? lidModelDir,
  }) async {
    if (!modelKind.isImplemented) {
      throw LiveTranscriberException(
        '${modelKind.label} is not implemented yet. Only Zipformer '
        '(transducer) works today.',
      );
    }

    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      throw LiveTranscriberException('Model directory not found: $modelDir');
    }

    final vadModelFile = File(vadModelPath);
    if (!await vadModelFile.exists()) {
      throw LiveTranscriberException('VAD model not found: $vadModelPath');
    }

    if (routingProfile.dualConfirmed) {
      if (senseVoiceModelDir == null || lidModelDir == null) {
        throw LiveTranscriberException(
          '${routingProfile.label} requires senseVoiceModelDir and '
          'lidModelDir.',
        );
      }
      try {
        _routed = await RoutedRecognizerSet.build(
          reazonModelDir: modelDir,
          senseVoiceModelDir: senseVoiceModelDir,
          lidModelDir: lidModelDir,
        );
      } on RoutedRecognizerException catch (e) {
        throw LiveTranscriberException(e.message);
      }
    } else {
      final filenames = await dir
          .list()
          .where((e) => e is File)
          .map((e) => e.uri.pathSegments.last)
          .toList();

      final ResolvedModelFiles resolved;
      try {
        resolved = resolveZipformerTransducerFiles(filenames);
      } on ModelFileResolutionException catch (e) {
        throw LiveTranscriberException(e.message);
      }

      final sep = Platform.pathSeparator;
      _recognizer = await buildOfflineRecognizerOffIsolate(
        sherpa_onnx.OfflineRecognizerConfig(
          model: sherpa_onnx.OfflineModelConfig(
            transducer: sherpa_onnx.OfflineTransducerModelConfig(
              encoder: '$modelDir$sep${resolved.encoder}',
              decoder: '$modelDir$sep${resolved.decoder}',
              joiner: '$modelDir$sep${resolved.joiner}',
            ),
            tokens: '$modelDir$sep${resolved.tokens}',
            numThreads: 2,
            debug: false,
            provider: 'cpu',
          ),
        ),
      );
    }

    final vadConfig = sherpa_onnx.VadModelConfig(
      sileroVad: sherpa_onnx.SileroVadModelConfig(model: vadModelPath),
      sampleRate: sampleRate,
    );
    _vad = await buildVadOffIsolate(
      config: vadConfig,
      bufferSizeInSeconds: 30,
    );
    _frameBuffer = PcmFrameBuffer(frameSize: vadConfig.sileroVad.windowSize);
  }

  /// Resets per-session bookkeeping (refine buffer, draft state, auto-refine
  /// timer) once native state is up and audio is about to start flowing --
  /// shared by [start] and [startDebugWavStream].
  void _resetSessionState() {
    _refineBuffer.clear();
    _lastSegmentAt = null;
    _clearDraftFrames();
    _draftSegmentActive = false;
    _lastDraftAt = null;
    if (_autoRefineEnabled) {
      _startAutoRefineTimer();
    }
  }

  void _startAutoRefineTimer() {
    _autoRefineTimer?.cancel();
    _autoRefineTimer = Timer.periodic(_autoRefineCheckInterval, (_) {
      final lastSegmentAt = _lastSegmentAt;
      if (lastSegmentAt == null) {
        return;
      }
      final due = isAutoRefineDue(
        sinceLastSegment: DateTime.now().difference(lastSegmentAt),
        bufferedDurationSeconds: _refineBuffer.totalDurationSeconds,
      );
      if (due) {
        unawaited(refineNow());
      }
    });
  }

  void _onMicChunk(Uint8List bytes) {
    final frameBuffer = _frameBuffer;
    if (frameBuffer == null) {
      return;
    }
    frameBuffer.add(pcm16BytesToFloat32(bytes));
    for (final frame in frameBuffer.drainFrames()) {
      _processFrame(frame);
    }
  }

  /// Runs one VAD window of audio through the pipeline: feeds the VAD,
  /// tracks it for the draft accumulator, drains any now-finalized
  /// segments, and (if due) fires a draft decode. Shared by the mic path
  /// ([_onMicChunk]) and the paced wav path ([startDebugWavStream]) so both
  /// exercise identical logic.
  void _processFrame(Float32List frame) {
    final vad = _vad;
    if (vad == null) {
      return;
    }
    vad.acceptWaveform(frame);

    // Accumulate this VAD window for the draft pass while a segment is in
    // progress. `isDetected()` has no "current segment audio" accessor in
    // the Dart bindings (unlike the desktop's `vad.current_segment.samples`
    // -- see draft_pass.dart's doc comment), so this buffer is built by
    // hand from the same frames already being fed to the VAD.
    final detected = vad.isDetected();
    if (detected && !_draftSegmentActive) {
      // Just started a new segment: drop anything left over from before
      // (shouldn't normally happen, since a finalized segment clears this
      // in _drainReadySegments, but guards against drift either way).
      _clearDraftFrames();
    }
    if (detected) {
      _draftFrames.add(frame);
      _draftFrames = slideDraftFrames(_draftFrames, sampleRate: sampleRate);
    }
    _draftSegmentActive = detected;

    _drainReadySegments();
    _maybeEmitDraft();
  }

  void _clearDraftFrames() {
    _draftFrames = [];
  }

  void _drainReadySegments() {
    final vad = _vad;
    if (vad == null) {
      return;
    }
    while (!vad.isEmpty()) {
      final segment = vad.front();
      vad.pop();
      // The segment just popped supersedes whatever the draft pass had
      // accumulated for it -- the real (properly routed/decoded) final is
      // about to replace any draft the UI was showing.
      _clearDraftFrames();
      _draftSegmentActive = false;
      _decodeSegment(segment);
    }
  }

  void _setBusy(bool value) {
    _busy = value;
    _decodingController.add(value);
  }

  void _decodeSegment(sherpa_onnx.SpeechSegment segment) {
    if (!isSegmentWorthDecoding(segment, sampleRate: sampleRate)) {
      return;
    }
    final routed = _routed;
    if (routed != null) {
      _decodeSegmentRouted(routed, segment);
      return;
    }

    final recognizer = _recognizer;
    if (recognizer == null) {
      return;
    }

    _setBusy(true);
    final stopwatch = Stopwatch()..start();
    final stream = recognizer.createStream();
    try {
      stream.acceptWaveform(samples: segment.samples, sampleRate: sampleRate);
      recognizer.decode(stream);
      final result = recognizer.getResult(stream);
      stopwatch.stop();
      final text = result.text.trim();
      if (text.isNotEmpty) {
        final now = DateTime.now();
        _entriesController.add(
          LiveTranscriptEntry(
            text: result.text,
            timestamp: now,
            latencyMs: stopwatch.elapsedMicroseconds / 1000,
          ),
        );
        _lastSegmentAt = now;
        _refineBuffer.add(
          RefineSegment(samples: segment.samples, text: text, capturedAt: now),
        );
      }
    } finally {
      stream.free();
      _setBusy(false);
    }
  }

  void _decodeSegmentRouted(
    RoutedRecognizerSet routed,
    sherpa_onnx.SpeechSegment segment,
  ) {
    _setBusy(true);
    final stopwatch = Stopwatch()..start();
    try {
      final result = routed.decode(segment.samples);
      stopwatch.stop();
      final text = result.text.trim();
      if (text.isNotEmpty) {
        final now = DateTime.now();
        _entriesController.add(
          LiveTranscriptEntry(
            text: text,
            timestamp: now,
            latencyMs: stopwatch.elapsedMicroseconds / 1000,
            lang: result.lang,
          ),
        );
        _lastSegmentAt = now;
        _refineBuffer.add(
          RefineSegment(samples: segment.samples, text: text, capturedAt: now),
        );
      }
    } finally {
      _setBusy(false);
    }
  }

  /// Fires a draft decode if one is due (see [isDraftDue]): enough draft
  /// audio has accumulated, the interval has elapsed, and nothing else is
  /// currently decoding. A no-op otherwise -- the next mic/wav frame will
  /// just check again.
  void _maybeEmitDraft() {
    if (_draftFrames.isEmpty) {
      return;
    }
    final now = DateTime.now();
    final sinceLastDraft = _lastDraftAt == null
        ? const Duration(days: 1)
        : now.difference(_lastDraftAt!);
    if (!isDraftDue(isDecoding: _busy, sinceLastDraft: sinceLastDraft)) {
      return;
    }
    _lastDraftAt = now;
    unawaited(_runDraftDecode());
  }

  Future<void> _runDraftDecode() async {
    // Snapshot now: more frames may arrive (and even a final may pop) while
    // this decode is in flight, since decoding is async but frame handling
    // isn't -- decoding off a snapshot avoids racing the live buffer.
    final frames = List<Float32List>.of(_draftFrames);
    if (frames.isEmpty) {
      return;
    }
    final combined = concatFloat32Lists(frames);
    final windowed = capDraftWindow(
      combined,
      sampleRate: sampleRate,
      maxSeconds: defaultDraftWindowSeconds,
    );
    if (windowed.length < sampleRate * defaultMinDraftAudioSeconds) {
      return;
    }

    _setBusy(true);
    final stopwatch = Stopwatch()..start();
    try {
      String text;
      String? lang;
      final routed = _routed;
      if (routed != null) {
        // Current-language-only decode, no LID/routing judgment -- see
        // RoutedRecognizerSet.decodeCurrentLangOnly's doc for why.
        final result = routed.decodeCurrentLangOnly(windowed);
        text = result.text.trim();
        lang = result.lang;
      } else {
        final recognizer = _recognizer;
        if (recognizer == null) {
          return;
        }
        // The plain (non-routed) recognizer already defaults to
        // `greedy_search` (sherpa_onnx's OfflineRecognizerConfig default) --
        // unlike the routed set's ReazonSpeech tier, which is explicitly
        // `modified_beam_search` for fast-final quality. Reusing it here
        // rather than building a second, lighter recognizer avoids doubling
        // this session's model memory footprint just for drafts.
        final stream = recognizer.createStream();
        try {
          stream.acceptWaveform(samples: windowed, sampleRate: sampleRate);
          recognizer.decode(stream);
          text = recognizer.getResult(stream).text.trim();
        } finally {
          stream.free();
        }
      }
      stopwatch.stop();
      if (text.isNotEmpty) {
        _draftController.add(
          LiveTranscriptEntry(
            text: text,
            timestamp: DateTime.now(),
            latencyMs: stopwatch.elapsedMicroseconds / 1000,
            lang: lang,
          ),
        );
      }
    } finally {
      _setBusy(false);
    }
  }

  /// Runs a refine pass over everything currently buffered: combines the
  /// buffered segments' audio and re-decodes it as one utterance, using the
  /// same recognizer the fast per-segment path uses. Emits nothing if the
  /// buffer is empty, the combined audio is under half a second (mirrors
  /// the desktop `Refiner`'s `len(buf) < sr // 2` guard), or the result is
  /// empty after the too-short fallback below.
  ///
  /// Safe to call whether or not [autoRefineEnabled] is on — this is also
  /// what the manual "清書" button calls directly, and what the debug wav
  /// test path (see `runDebugWavRefineTest`) exercises indirectly via its
  /// own recognizer.
  Future<void> refineNow() async {
    final recognizer = _recognizer;
    final routed = _routed;
    if ((recognizer == null && routed == null) || _refineBuffer.isEmpty) {
      return;
    }
    final segments = _refineBuffer.takeAll();
    final combined = combineSegmentSamples(segments);
    if (combined.length < sampleRate ~/ 2) {
      return;
    }
    final fastJoined = combineSegmentFastText(segments);

    _setBusy(true);
    final stopwatch = Stopwatch()..start();
    try {
      var text = '';
      String? lang;
      if (routed != null) {
        // Re-runs LID on the merged group audio too, same as the desktop
        // refine pass's re-judgment (docs/LID.md's REFINE_MIN_REGROUP_S
        // rationale: a longer merged group is a better LID input than any
        // single segment).
        final result = routed.decode(combined);
        text = result.text.trim();
        lang = result.lang;
      } else if (recognizer != null) {
        final stream = recognizer.createStream();
        try {
          stream.acceptWaveform(samples: combined, sampleRate: sampleRate);
          recognizer.decode(stream);
          text = recognizer.getResult(stream).text.trim();
        } finally {
          stream.free();
        }
      }
      stopwatch.stop();
      // A merged re-decode must never LOSE content: if it comes back much
      // shorter than the fast finals combined, trust those instead (mirrors
      // scripts/realtime_transcribe.py's Refiner).
      if (isRefineTextTooShort(text, fastJoined)) {
        text = fastJoined;
      }
      if (text.isNotEmpty) {
        _refineEntriesController.add(
          LiveTranscriptEntry(
            text: text,
            timestamp: DateTime.now(),
            latencyMs: stopwatch.elapsedMicroseconds / 1000,
            lang: lang,
          ),
        );
      }
    } finally {
      _setBusy(false);
    }
  }

  /// Stops capturing, flushes any speech still buffered in the VAD, and
  /// releases native resources. Anything still buffered for refine is
  /// dropped unrefined — this mirrors the fast path (an in-progress VAD
  /// segment gets flushed and decoded, but there's no "finish the refine
  /// pass on shutdown" step the way the desktop's `force=True` shutdown
  /// path has, since stopping is a deliberate user action here rather than
  /// a process exit).
  Future<void> stop() async {
    if (!isRunning) {
      return;
    }
    _autoRefineTimer?.cancel();
    _autoRefineTimer = null;

    await _micSubscription?.cancel();
    _micSubscription = null;
    // Cancelled before _recorder.stop() so our own shutdown doesn't get
    // reported back to us as a capture failure.
    await _recorderStateSubscription?.cancel();
    _recorderStateSubscription = null;
    await _recorder.stop();

    _vad?.flush();
    _drainReadySegments();

    _refineBuffer.clear();
    _lastSegmentAt = null;
    _clearDraftFrames();
    _draftSegmentActive = false;
    _lastDraftAt = null;

    await _teardownNativeState();
  }

  Future<void> _teardownNativeState() async {
    _vad?.free();
    _vad = null;
    _recognizer?.free();
    _recognizer = null;
    _routed?.free();
    _routed = null;
    _frameBuffer = null;
  }

  /// Releases everything, including the mic recorder itself. Call once when
  /// the owning widget is disposed.
  Future<void> dispose() async {
    await stop();
    await stopDebugWavStream();
    await _entriesController.close();
    await _decodingController.close();
    await _refineEntriesController.close();
    await _draftController.close();
    await _recorder.dispose();
  }

  /// Debug-only (see `kDebugMode` at the call site in `live_page.dart`):
  /// streams [wavPath] through the exact same VAD/draft/final/refine
  /// pipeline a live mic session uses ([_processFrame]), instead of the
  /// static two-halves comparison [runDebugWavRefineTest] does. This is how
  /// the draft pass gets verified on an emulator, which has no usable
  /// microphone: push a recording through at (roughly) real-time pace and
  /// watch [drafts]/[entries]/[refineEntries] the same way a live session
  /// would produce them.
  ///
  /// [realtime] paces delivery to roughly the wav's own duration (one VAD
  /// window's worth of audio every ~32ms at 16kHz); pass `false` to stream
  /// as fast as decoding allows instead, e.g. for a quick smoke test.
  /// Awaits until the whole file has been fed through and any in-progress
  /// segment has been flushed and decoded -- same shutdown behavior [stop]
  /// gives a live session.
  Future<void> startDebugWavStream({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
    required String wavPath,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
    bool realtime = true,
  }) async {
    if (isRunning || _debugStreaming) {
      return;
    }

    final wavFile = File(wavPath);
    if (!await wavFile.exists()) {
      throw LiveTranscriberException('WAV file not found: $wavPath');
    }

    await _buildNativeState(
      modelKind: modelKind,
      modelDir: modelDir,
      vadModelPath: vadModelPath,
      routingProfile: routingProfile,
      senseVoiceModelDir: senseVoiceModelDir,
      lidModelDir: lidModelDir,
    );
    _resetSessionState();

    final frameBuffer = _frameBuffer;
    if (frameBuffer == null) {
      await _teardownNativeState();
      return;
    }

    _debugStreaming = true;
    _debugStreamCancelRequested = false;
    try {
      final wave = sherpa_onnx.readWave(wavPath);
      if (wave.samples.isEmpty) {
        throw LiveTranscriberException(
          'Failed to read WAV file (unsupported format or empty): $wavPath',
        );
      }
      if (wave.sampleRate != sampleRate) {
        // Unlike runDebugWavRefineTest (which just decodes at the wav's own
        // rate, no VAD involved), this path feeds the wav through the VAD,
        // which is configured for a fixed 16kHz -- a mismatched wav would
        // silently desync the VAD's speech/silence timing from real time.
        throw LiveTranscriberException(
          'WAV sample rate ${wave.sampleRate}Hz != required ${sampleRate}Hz '
          '(resample the test wav to 16kHz mono first): $wavPath',
        );
      }
      // wave.samples is already mono float32 (sherpa-onnx's reader decodes
      // straight to that), so no pcm16BytesToFloat32 step is needed here
      // (unlike the mic path) -- feed it straight into the same
      // PcmFrameBuffer to get fixed-size VAD windows.
      final frameSize = frameBuffer.frameSize;
      final samplesPerFrameDuration = Duration(
        microseconds: (frameSize / sampleRate * 1000000).round(),
      );
      var offset = 0;
      while (offset < wave.samples.length) {
        if (_debugStreamCancelRequested) {
          break;
        }
        final end = (offset + frameSize).clamp(0, wave.samples.length);
        final chunk = Float32List.sublistView(wave.samples, offset, end);
        frameBuffer.add(chunk);
        for (final frame in frameBuffer.drainFrames()) {
          _processFrame(frame);
        }
        offset = end;
        if (realtime) {
          await Future.delayed(samplesPerFrameDuration);
        }
      }

      if (!_debugStreamCancelRequested) {
        _vad?.flush();
        _drainReadySegments();
      }
    } finally {
      _debugStreaming = false;
      await _teardownNativeState();
    }
  }

  /// Stops an in-progress [startDebugWavStream] early, mid-file. A no-op if
  /// no debug stream is running.
  Future<void> stopDebugWavStream() async {
    if (!_debugStreaming) {
      return;
    }
    _debugStreamCancelRequested = true;
    // startDebugWavStream's loop checks the flag once per frame delay
    // (~32ms of wav audio in realtime mode); give it a moment to notice
    // and tear down before returning.
    while (_debugStreaming) {
      await Future.delayed(const Duration(milliseconds: 20));
    }
  }

  /// Debug-only helper (see `kDebugMode` at the call site in `live_page.dart`):
  /// exercises the refine pass's audio-combining logic end to end against a
  /// real WAV file, without needing a live mic session or an emulator's
  /// (nonexistent) microphone.
  ///
  /// Loads its own short-lived recognizer(s) from [modelDir] (independent of
  /// any running live session), splits [wavPath]'s samples into two halves
  /// to stand in for two VAD segments, decodes each half individually (the
  /// "fast path" comparison), then runs them back through
  /// [combineSegmentSamples] and decodes the combined audio (the "refine"
  /// result) — so a caller can see whether the refine decode differs from
  /// the two individual ones.
  ///
  /// When [routingProfile] is [RoutingProfile.jaSenseVoice] (mirrors
  /// [start]'s routing parameters), each decode goes through a short-lived
  /// [RoutedRecognizerSet] instead of a plain ja-only recognizer, so this
  /// path can also exercise the routing badge shown on the Live screen
  /// without a live mic session — the only way to do so on an emulator,
  /// which has no usable microphone.
  static Future<DebugRefineTestResult> runDebugWavRefineTest({
    required String modelDir,
    required String wavPath,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
  }) async {
    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      throw LiveTranscriberException('Model directory not found: $modelDir');
    }
    final wavFile = File(wavPath);
    if (!await wavFile.exists()) {
      throw LiveTranscriberException('WAV file not found: $wavPath');
    }

    if (routingProfile.dualConfirmed &&
        (senseVoiceModelDir == null || lidModelDir == null)) {
      throw LiveTranscriberException(
        '${routingProfile.label} requires senseVoiceModelDir and '
        'lidModelDir.',
      );
    }

    sherpa_onnx.OfflineRecognizer? recognizer;
    RoutedRecognizerSet? routed;
    if (routingProfile.dualConfirmed) {
      try {
        routed = await RoutedRecognizerSet.build(
          reazonModelDir: modelDir,
          senseVoiceModelDir: senseVoiceModelDir!,
          lidModelDir: lidModelDir!,
        );
      } on RoutedRecognizerException catch (e) {
        throw LiveTranscriberException(e.message);
      }
    } else {
      final filenames = await dir
          .list()
          .where((e) => e is File)
          .map((e) => e.uri.pathSegments.last)
          .toList();

      final ResolvedModelFiles resolved;
      try {
        resolved = resolveZipformerTransducerFiles(filenames);
      } on ModelFileResolutionException catch (e) {
        throw LiveTranscriberException(e.message);
      }

      final sep = Platform.pathSeparator;
      recognizer = await buildOfflineRecognizerOffIsolate(
        sherpa_onnx.OfflineRecognizerConfig(
          model: sherpa_onnx.OfflineModelConfig(
            transducer: sherpa_onnx.OfflineTransducerModelConfig(
              encoder: '$modelDir$sep${resolved.encoder}',
              decoder: '$modelDir$sep${resolved.decoder}',
              joiner: '$modelDir$sep${resolved.joiner}',
            ),
            tokens: '$modelDir$sep${resolved.tokens}',
            numThreads: 2,
            debug: false,
            provider: 'cpu',
          ),
        ),
      );
    }

    try {
      final wave = sherpa_onnx.readWave(wavPath);
      if (wave.samples.isEmpty) {
        throw LiveTranscriberException(
          'Failed to read WAV file (unsupported format or empty): $wavPath',
        );
      }

      (String, String?) decode(Float32List samples) {
        if (routed != null) {
          final result = routed.decode(samples);
          return (result.text, result.lang);
        }
        final stream = recognizer!.createStream();
        try {
          stream.acceptWaveform(samples: samples, sampleRate: wave.sampleRate);
          recognizer.decode(stream);
          return (recognizer.getResult(stream).text.trim(), null);
        } finally {
          stream.free();
        }
      }

      final mid = wave.samples.length ~/ 2;
      final segment1 = RefineSegment(
        samples: Float32List.sublistView(wave.samples, 0, mid),
        text: '',
        capturedAt: DateTime.now(),
      );
      final segment2 = RefineSegment(
        samples: Float32List.sublistView(wave.samples, mid),
        text: '',
        capturedAt: DateTime.now(),
      );

      final (text1, lang1) = decode(segment1.samples);
      final (text2, lang2) = decode(segment2.samples);
      final combined = combineSegmentSamples([segment1, segment2]);
      final (refineText, refineLang) = decode(combined);

      return DebugRefineTestResult(
        segment1Text: text1,
        segment2Text: text2,
        refineText: refineText,
        segment1Lang: lang1,
        segment2Lang: lang2,
        refineLang: refineLang,
      );
    } finally {
      recognizer?.free();
      routed?.free();
    }
  }
}

/// Result of [LiveTranscriber.runDebugWavRefineTest]: the two individually
/// decoded halves of the test wav, and the combined ("清書") re-decode, for
/// visually comparing whether the refine pass changed anything.
class DebugRefineTestResult {
  const DebugRefineTestResult({
    required this.segment1Text,
    required this.segment2Text,
    required this.refineText,
    this.segment1Lang,
    this.segment2Lang,
    this.refineLang,
  });

  final String segment1Text;
  final String segment2Text;
  final String refineText;

  /// The language each decode resolved to, when [LiveTranscriber.start]'s
  /// `routingProfile` was [RoutingProfile.jaSenseVoice] — `null` for a
  /// plain ja-only test run.
  final String? segment1Lang;
  final String? segment2Lang;
  final String? refineLang;
}
