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
import 'model_load_event.dart';
import 'native_model_loader.dart';
import 'pcm_frame_buffer.dart';
import 'refine_pass.dart';
import 'speech_segment_filter.dart';
import 'vad_sensitivity.dart';

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
  /// [draftIntervalSeconds]/[draftWindowSeconds]/[minDraftAudioSeconds]
  /// (see `draft_pass.dart`) and [autoRefineSilenceSeconds]/
  /// [autoRefineMaxBufferedSeconds]/[refineBufferMaxSeconds] (see
  /// `refine_pass.dart`) seed the identically-named runtime-settable
  /// properties below, each defaulting to that file's `default*` constant
  /// so a caller who doesn't pass any of them gets today's pacing
  /// unchanged. All six must be positive and finite; an out-of-range value
  /// throws [ArgumentError] here (and from the property setters) rather
  /// than silently misbehaving mid-session.
  ///
  /// Validation runs before [recorder] is defaulted to a fresh
  /// `AudioRecorder()` (Dart evaluates an initializer list left to right):
  /// `AudioRecorder()`'s constructor kicks off a real, asynchronous
  /// platform-channel call under the hood, so a caller passing an invalid
  /// knob value gets a clean, synchronous [ArgumentError] instead of also
  /// leaking that half-started platform call.
  LiveTranscriber({
    AudioRecorder? recorder,
    double draftIntervalSeconds = defaultDraftIntervalSeconds,
    double draftWindowSeconds = defaultDraftWindowSeconds,
    double minDraftAudioSeconds = defaultMinDraftAudioSeconds,
    double autoRefineSilenceSeconds = defaultAutoRefineSilenceSeconds,
    double autoRefineMaxBufferedSeconds = defaultAutoRefineMaxBufferedSeconds,
    double refineBufferMaxSeconds = defaultRefineBufferMaxSeconds,
  }) : _draftIntervalSeconds = _requirePositive(
         draftIntervalSeconds,
         'draftIntervalSeconds',
       ),
       _draftWindowSeconds = _requirePositive(
         draftWindowSeconds,
         'draftWindowSeconds',
       ),
       _minDraftAudioSeconds = _requirePositive(
         minDraftAudioSeconds,
         'minDraftAudioSeconds',
       ),
       _autoRefineSilenceSeconds = _requirePositive(
         autoRefineSilenceSeconds,
         'autoRefineSilenceSeconds',
       ),
       _autoRefineMaxBufferedSeconds = _requirePositive(
         autoRefineMaxBufferedSeconds,
         'autoRefineMaxBufferedSeconds',
       ),
       _refineBuffer = RefineBuffer(
         sampleRate: sampleRate,
         maxDurationSeconds: _requirePositive(
           refineBufferMaxSeconds,
           'refineBufferMaxSeconds',
         ),
       ),
       _recorder = recorder ?? AudioRecorder();

  static const int sampleRate = 16000;

  static double _requirePositive(double value, String name) {
    if (!value.isFinite || value <= 0) {
      throw ArgumentError.value(value, name, 'must be a positive, finite number');
    }
    return value;
  }

  final AudioRecorder _recorder;
  final _entriesController = StreamController<LiveTranscriptEntry>.broadcast();
  final _decodingController = StreamController<bool>.broadcast();
  final _refineEntriesController =
      StreamController<LiveTranscriptEntry>.broadcast();
  final _draftController = StreamController<LiveTranscriptEntry>.broadcast();
  final _errorsController =
      StreamController<LiveTranscriberException>.broadcast();
  final _modelLoadController = StreamController<ModelLoadEvent>.broadcast();
  final _sessionResetController = StreamController<void>.broadcast();

  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.OfflineRecognizer? _recognizer;
  RoutedRecognizerSet? _routed;
  StreamSubscription<Uint8List>? _micSubscription;
  StreamSubscription<RecordState>? _recorderStateSubscription;
  PcmFrameBuffer? _frameBuffer;
  // Remembered from the last _buildNativeState call so setVadSensitivity
  // can rebuild just the VAD later without the caller having to repeat the
  // model path.
  String? _vadModelPath;
  VadSensitivity _vadSensitivity = VadSensitivity();
  // The sensitivity setVadSensitivity should (re)build for once it's safe
  // to; always the latest call wins (see setVadSensitivity's doc for the
  // "overlapping calls" guard this implements).
  VadSensitivity? _pendingVadSensitivity;
  bool _vadRebuildInFlight = false;
  // A freshly built replacement VAD, waiting for shouldSwapVadNow to allow
  // installing it -- checked once per processed frame (_maybeSwapPendingVad,
  // called from _processFrame) so the swap happens as soon as it's safe
  // without polling on a timer.
  sherpa_onnx.VoiceActivityDetector? _pendingVadInstall;
  // Bumped by _buildNativeState and _teardownNativeState -- lets a
  // long-running background rebuild (setVadSensitivity's VAD rebuild) tell,
  // once its await resolves, whether the native state it was building
  // against is still the current one or whether the session was torn down
  // (and possibly rebuilt from scratch by a fresh start()) in the meantime.
  // See isVadBuildStale in vad_sensitivity.dart for the comparison itself.
  int _sessionGeneration = 0;

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
  // Pacing knobs, set from the constructor and re-validated on every
  // runtime assignment through their public setters below -- see those
  // setters' doc comments and `draft_pass.dart`/`refine_pass.dart` for what
  // each one controls.
  double _draftIntervalSeconds;
  double _draftWindowSeconds;
  double _minDraftAudioSeconds;
  double _autoRefineSilenceSeconds;
  double _autoRefineMaxBufferedSeconds;
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
  // layered on top of this; it's re-decode only. Built in the constructor
  // initializer list above (its cap comes from the refineBufferMaxSeconds
  // constructor parameter).
  final RefineBuffer _refineBuffer;
  DateTime? _lastSegmentAt;
  Timer? _autoRefineTimer;
  bool _autoRefineEnabled = false;

  /// How often (wall-clock seconds) a draft re-decode fires while a VAD
  /// segment is in progress. Defaults to [defaultDraftIntervalSeconds].
  /// Runtime-settable: a new value applies from the next due-check
  /// ([isDraftDue]) onward, without touching any native model. Throws
  /// [ArgumentError] for a non-positive or non-finite value.
  double get draftIntervalSeconds => _draftIntervalSeconds;
  set draftIntervalSeconds(double value) =>
      _draftIntervalSeconds = _requirePositive(value, 'draftIntervalSeconds');

  /// Trailing-audio cap (seconds) a draft decode re-processes. Defaults to
  /// [defaultDraftWindowSeconds]. Runtime-settable, same caveats as
  /// [draftIntervalSeconds].
  double get draftWindowSeconds => _draftWindowSeconds;
  set draftWindowSeconds(double value) =>
      _draftWindowSeconds = _requirePositive(value, 'draftWindowSeconds');

  /// Minimum accumulated audio (seconds) before a draft decode runs at all.
  /// Defaults to [defaultMinDraftAudioSeconds]. Runtime-settable, same
  /// caveats as [draftIntervalSeconds].
  double get minDraftAudioSeconds => _minDraftAudioSeconds;
  set minDraftAudioSeconds(double value) =>
      _minDraftAudioSeconds = _requirePositive(value, 'minDraftAudioSeconds');

  /// Silence gap (seconds) that fires an auto-refine when [autoRefineEnabled]
  /// is on. Defaults to [defaultAutoRefineSilenceSeconds]. Runtime-settable:
  /// a new value applies from the next per-second due-check onward (see
  /// [_startAutoRefineTimer]).
  double get autoRefineSilenceSeconds => _autoRefineSilenceSeconds;
  set autoRefineSilenceSeconds(double value) => _autoRefineSilenceSeconds =
      _requirePositive(value, 'autoRefineSilenceSeconds');

  /// Buffered-duration ceiling (seconds) that fires an auto-refine even
  /// without a silence gap. Defaults to [defaultAutoRefineMaxBufferedSeconds].
  /// Runtime-settable, same caveats as [autoRefineSilenceSeconds].
  double get autoRefineMaxBufferedSeconds => _autoRefineMaxBufferedSeconds;
  set autoRefineMaxBufferedSeconds(double value) =>
      _autoRefineMaxBufferedSeconds = _requirePositive(
        value,
        'autoRefineMaxBufferedSeconds',
      );

  /// Hard cap (seconds) on how much audio the refine buffer holds before it
  /// starts dropping the oldest segment. Defaults to
  /// [defaultRefineBufferMaxSeconds]. Runtime-settable: applies from the
  /// next [RefineBuffer.add] onward, without touching any native model.
  double get refineBufferMaxSeconds => _refineBuffer.maxDurationSeconds;
  set refineBufferMaxSeconds(double value) => _refineBuffer.maxDurationSeconds =
      _requirePositive(value, 'refineBufferMaxSeconds');

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

  /// One event right before and one right after each native model finishes
  /// loading -- see [ModelLoadEvent] for the exact model names/phases. Lets
  /// a host UI show which model is loading instead of treating [start] as
  /// one opaque `Future`, which matters most for
  /// [RoutingProfile.jaSenseVoice]'s three-model, multi-second load.
  Stream<ModelLoadEvent> get modelLoads => _modelLoadController.stream;

  /// Fires once per completed [resetSession] call that actually did
  /// something (i.e. the session was running). Carries no data -- the
  /// session state [resetSession] clears isn't otherwise observable on this
  /// class, so this is purely a "something changed" signal for a listener
  /// (`HayamimiLive` turns it into a `SessionResetSubtitleEvent` on its
  /// `events` stream).
  Stream<void> get sessionResets => _sessionResetController.stream;

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
  ///
  /// [decodingMethod] picks the sherpa-onnx offline-recognizer search
  /// algorithm (e.g. `'greedy_search'`, `'modified_beam_search'`). Leaving
  /// it `null` (the default) reproduces this method's behavior before this
  /// parameter existed: the plain (non-routed) path used sherpa-onnx's own
  /// `'greedy_search'` default, and the [RoutingProfile.jaSenseVoice] ja
  /// tier used `'modified_beam_search'` (matching desktop production, see
  /// `bench_runner.dart`) — passing a value here overrides whichever of
  /// those applies. SenseVoice's own decode doesn't use this parameter.
  ///
  /// [vadSensitivity] configures Silero VAD's speech/silence detection
  /// (threshold, minimum silence/speech durations, max segment duration —
  /// see [VadSensitivity]); `null` (the default) uses sherpa-onnx's own
  /// defaults, same as before this parameter existed. Change it after
  /// [start] via [setVadSensitivity] instead of stopping and restarting the
  /// session.
  ///
  /// [hotwordsFile]/[hotwordsScore] bias the recognizer toward a wordlist
  /// (sherpa-onnx's own hotwords feature) on the plain path and the routed
  /// ja tier; `null` (the default) leaves hotwords off. Unlike the pacing
  /// knobs and [vadSensitivity], hotwords have no runtime setter — they're
  /// baked into the recognizer at build time, so changing them requires a
  /// fresh [start] (or [stop] + [start]).
  Future<void> start({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
    String? decodingMethod,
    VadSensitivity? vadSensitivity,
    String? hotwordsFile,
    double hotwordsScore = 1.5,
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

      try {
        await _buildNativeState(
          modelKind: modelKind,
          modelDir: modelDir,
          vadModelPath: vadModelPath,
          routingProfile: routingProfile,
          senseVoiceModelDir: senseVoiceModelDir,
          lidModelDir: lidModelDir,
          decodingMethod: decodingMethod,
          vadSensitivity: vadSensitivity ?? _vadSensitivity,
          hotwordsFile: hotwordsFile,
          hotwordsScore: hotwordsScore,
        );
      } catch (_) {
        // _buildNativeState can throw after already building the
        // recognizer/routed set (e.g. a good recognizer load followed by a
        // truncated VAD model file) -- at this point isRunning is still
        // false (no mic subscription yet), so without this, stop()/dispose()
        // would treat the session as never having started and leak whatever
        // was built.
        await _teardownNativeState();
        rethrow;
      }
      // A setVadSensitivity() call that arrived while _buildNativeState was
      // still loading (isRunning/_debugStreaming were both false, so it
      // couldn't rebuild against anything yet) queued its target instead --
      // apply it now that native state, including _vadModelPath, is up.
      _applyPendingVadSensitivityIfAny();

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
    String? decodingMethod,
    required VadSensitivity vadSensitivity,
    String? hotwordsFile,
    double hotwordsScore = 1.5,
  }) async {
    // A new generation starts the moment this build begins, not once it
    // succeeds: any setVadSensitivity rebuild already in flight for the
    // PREVIOUS generation must be treated as stale from this point on, even
    // if this build itself goes on to fail (see start()'s/
    // _runDebugWavStream's catch, which tears down -- and bumps the
    // generation again -- on failure).
    _sessionGeneration++;
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
          decodingMethod: decodingMethod ?? 'modified_beam_search',
          hotwordsFile: hotwordsFile,
          hotwordsScore: hotwordsScore,
          onModelLoad: (model, phase, ms) =>
              _emitModelLoad(model: model, phase: phase, ms: ms),
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
      _emitModelLoad(model: 'recognizer', phase: 'start');
      final recognizerStopwatch = Stopwatch()..start();
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
          decodingMethod: decodingMethod ?? 'greedy_search',
          hotwordsFile: hotwordsFile ?? '',
          hotwordsScore: hotwordsScore,
        ),
      );
      recognizerStopwatch.stop();
      _emitModelLoad(
        model: 'recognizer',
        phase: 'done',
        ms: recognizerStopwatch.elapsedMicroseconds / 1000,
      );
    }

    _vadModelPath = vadModelPath;
    _vadSensitivity = vadSensitivity;
    final vadConfig = _vadConfigFor(vadSensitivity, vadModelPath);
    _emitModelLoad(model: 'vad', phase: 'start');
    final vadStopwatch = Stopwatch()..start();
    _vad = await buildVadOffIsolate(
      config: vadConfig,
      bufferSizeInSeconds: 30,
    );
    vadStopwatch.stop();
    _emitModelLoad(
      model: 'vad',
      phase: 'done',
      ms: vadStopwatch.elapsedMicroseconds / 1000,
    );
    _frameBuffer = PcmFrameBuffer(frameSize: vadConfig.sileroVad.windowSize);
  }

  sherpa_onnx.VadModelConfig _vadConfigFor(
    VadSensitivity sensitivity,
    String vadModelPath,
  ) {
    return sherpa_onnx.VadModelConfig(
      sileroVad: sherpa_onnx.SileroVadModelConfig(
        model: vadModelPath,
        threshold: sensitivity.threshold,
        minSilenceDuration: sensitivity.minSilenceSeconds,
        minSpeechDuration: sensitivity.minSpeechSeconds,
        maxSpeechDuration: sensitivity.maxSpeechSeconds,
      ),
      sampleRate: sampleRate,
    );
  }

  void _emitModelLoad({
    required String model,
    required String phase,
    double? ms,
  }) {
    if (!_modelLoadController.isClosed) {
      _modelLoadController.add(
        ModelLoadEvent(model: model, phase: phase, ms: ms),
      );
    }
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
        silenceSeconds: _autoRefineSilenceSeconds,
        maxBufferedSeconds: _autoRefineMaxBufferedSeconds,
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
      _draftFrames = slideDraftFrames(
        _draftFrames,
        sampleRate: sampleRate,
        maxSeconds: _draftWindowSeconds,
      );
    }
    _draftSegmentActive = detected;

    _drainReadySegments();
    _maybeEmitDraft();
    _maybeSwapPendingVad();
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
            audioSeconds: segment.samples.length / sampleRate,
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
            audioSeconds: segment.samples.length / sampleRate,
            switched: result.switched,
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
    if (!isDraftDue(
      isDecoding: _busy,
      sinceLastDraft: sinceLastDraft,
      intervalSeconds: _draftIntervalSeconds,
    )) {
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
      maxSeconds: _draftWindowSeconds,
    );
    if (windowed.length < sampleRate * _minDraftAudioSeconds) {
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
            audioSeconds: combined.length / sampleRate,
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
    // Bumped here too (not just in _buildNativeState): a setVadSensitivity
    // rebuild that's still in flight when this teardown happens must be
    // treated as stale even if no new start() ever follows -- see
    // isVadBuildStale and _rebuildVadUntilCaughtUp's use of it.
    _sessionGeneration++;
    _vad?.free();
    _vad = null;
    // A rebuild that finished (or is still in flight) around the same time
    // as this teardown has nothing left to install into -- free it rather
    // than leaking the native handle. A rebuild still in flight notices via
    // the generation bump just above (isVadBuildStale) and won't try to
    // install its result at all; this here covers a rebuild that had
    // already finished and was sitting in _pendingVadInstall waiting for a
    // safe swap point.
    _pendingVadInstall?.free();
    _pendingVadInstall = null;
    _vadModelPath = null;
    _recognizer?.free();
    _recognizer = null;
    _routed?.free();
    _routed = null;
    _frameBuffer = null;
  }

  /// Releases everything, including the mic recorder itself. Call once when
  /// the owning widget is disposed. Idempotent: a second call is a no-op
  /// rather than a "Bad state: Stream has already been closed".
  Future<void> dispose() async {
    if (_disposed) {
      return;
    }
    _disposed = true;
    await stop();
    await stopDebugWavStream();
    await _entriesController.close();
    await _decodingController.close();
    await _refineEntriesController.close();
    await _draftController.close();
    await _errorsController.close();
    await _modelLoadController.close();
    await _sessionResetController.close();
    await _recorder.dispose();
  }

  /// Rebuilds Silero VAD off-isolate with [sensitivity] and swaps it into a
  /// running session the moment it's safe to (see [shouldSwapVadNow]:
  /// never mid-segment, never while another decode is in flight) --
  /// [_maybeSwapPendingVad], called once per processed frame, is what
  /// actually performs the swap once that moment arrives.
  ///
  /// If no session is running or loading (and no debug wav stream is),
  /// [sensitivity] is just remembered for the next [start]/
  /// [startDebugWavStream] -- there is nothing native to rebuild against
  /// yet. If [start]/[startDebugWavStream] is *currently* loading (between
  /// the call and [isRunning]/[isDebugStreaming] actually flipping true),
  /// [sensitivity] is queued the same way and applied the moment that
  /// load's own [_buildNativeState] finishes -- calling this mid-load used
  /// to silently lose the request, since [_buildNativeState] overwrites
  /// [_vadSensitivity] with whatever it was given once it completes.
  ///
  /// Calling this again before an earlier call has finished replaces the
  /// pending target rather than queueing both: only the most recently
  /// requested [sensitivity] ever gets installed, and a build already in
  /// flight for a now-superseded target is discarded (freed) once it
  /// completes instead of being installed. The same discard-instead-of-
  /// install applies if the session is stopped (and possibly restarted)
  /// while a rebuild's own model load is in flight -- see
  /// [isVadBuildStale].
  Future<void> setVadSensitivity(VadSensitivity sensitivity) async {
    _pendingVadSensitivity = sensitivity;
    if (_starting) {
      // start()/startDebugWavStream() is currently loading: its own
      // _buildNativeState call hasn't set _vadModelPath yet, so there is
      // nothing to rebuild against. Once it finishes, it calls
      // _applyPendingVadSensitivityIfAny, which picks this target up.
      return;
    }
    if (!isRunning && !_debugStreaming) {
      _vadSensitivity = sensitivity;
      _pendingVadSensitivity = null;
      return;
    }
    if (_vadRebuildInFlight) {
      // The in-flight rebuild loop below re-reads _pendingVadSensitivity
      // once it finishes its current build, so it will pick this newer
      // target up on its own -- nothing more to do here.
      return;
    }
    await _rebuildVadUntilCaughtUp();
  }

  /// Called right after a successful [_buildNativeState], from both [start]
  /// and [_runDebugWavStream]: if a [setVadSensitivity] call arrived while
  /// this load was still in progress (queued into [_pendingVadSensitivity]
  /// because [_starting] was true), kick off the rebuild for it now that
  /// native state -- including [_vadModelPath] -- actually exists. Runs in
  /// the background rather than being awaited: a rebuild here is a
  /// follow-up request from the caller, not part of loading itself, so
  /// [start]/[startDebugWavStream] shouldn't block their own return on it.
  void _applyPendingVadSensitivityIfAny() {
    if (_pendingVadSensitivity != null && !_vadRebuildInFlight) {
      unawaited(_rebuildVadUntilCaughtUp());
    }
  }

  Future<void> _rebuildVadUntilCaughtUp() async {
    _vadRebuildInFlight = true;
    try {
      while (_pendingVadSensitivity != null) {
        final target = _pendingVadSensitivity!;
        _pendingVadSensitivity = null;

        final vadModelPath = _vadModelPath;
        if (vadModelPath == null) {
          // No native state to rebuild against right now (e.g. stop() ran
          // concurrently) -- just remember the target for next time.
          _vadSensitivity = target;
          continue;
        }

        // Captured *before* the await below: this is the native-state
        // "epoch" this rebuild is being built for. Compared against
        // _sessionGeneration once the build resolves (see isVadBuildStale)
        // to detect a stop()+start() (or any other teardown/rebuild) that
        // happened while buildVadOffIsolate was in flight.
        final buildGeneration = _sessionGeneration;

        _emitModelLoad(model: 'vad', phase: 'start');
        final stopwatch = Stopwatch()..start();
        final built = await buildVadOffIsolate(
          config: _vadConfigFor(target, vadModelPath),
          bufferSizeInSeconds: 30,
        );
        stopwatch.stop();
        _emitModelLoad(
          model: 'vad',
          phase: 'done',
          ms: stopwatch.elapsedMicroseconds / 1000,
        );

        final stale = isVadBuildStale(
          buildGeneration: buildGeneration,
          currentGeneration: _sessionGeneration,
        );
        if (_pendingVadSensitivity != null || stale) {
          // Superseded by a newer call while this was building, or the
          // native state it was built against has moved on (torn down,
          // and/or rebuilt by a fresh start() -- see isVadBuildStale) --
          // either way this build doesn't belong to the current session,
          // so free it instead of installing it.
          built.free();
          continue;
        }

        _vadSensitivity = target;
        _pendingVadInstall = built;
        _maybeSwapPendingVad();
      }
    } finally {
      _vadRebuildInFlight = false;
    }
  }

  /// Installs [_pendingVadInstall] in place of [_vad] the moment
  /// [shouldSwapVadNow] says it's safe to -- called once per processed
  /// frame ([_processFrame]) so the swap happens as soon as possible
  /// without polling on a timer.
  void _maybeSwapPendingVad() {
    final pending = _pendingVadInstall;
    if (pending == null) {
      return;
    }
    if (!shouldSwapVadNow(speechActive: _draftSegmentActive, busy: _busy)) {
      return;
    }
    _pendingVadInstall = null;
    final old = _vad;
    _vad = pending;
    old?.free();
  }

  /// Clears everything about the current "conversation" without touching
  /// any loaded native model: the refine buffer, in-progress draft state,
  /// silence timing, and (for a [RoutingProfile.jaSenseVoice] session)
  /// which language it's currently locked to (via
  /// [RoutedRecognizerSet.reset]). Lets a host app start a fresh
  /// conversation -- e.g. the user tapped "new session" -- without paying
  /// to reload up to 396 MB of model weights.
  ///
  /// If a decode is in flight, waits for it to finish first (so the reset
  /// can't race a final/refine/draft write into the buffers it's about to
  /// clear) before actually clearing anything.
  ///
  /// A no-op when no session ([isRunning]) or debug wav stream is active --
  /// there is nothing to reset, and [sessionResets] does not fire.
  Future<void> resetSession() async {
    if (!isRunning && !_debugStreaming) {
      return;
    }
    while (_busy) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    _refineBuffer.clear();
    _lastSegmentAt = null;
    _clearDraftFrames();
    _draftSegmentActive = false;
    _lastDraftAt = null;
    _routed?.reset();
    if (!_sessionResetController.isClosed) {
      _sessionResetController.add(null);
    }
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
  /// See [start] for what [decodingMethod]/[vadSensitivity]/[hotwordsFile]/
  /// [hotwordsScore] do -- identical meaning and defaults here.
  Future<void> startDebugWavStream({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
    required String wavPath,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
    bool realtime = true,
    String? decodingMethod,
    VadSensitivity? vadSensitivity,
    String? hotwordsFile,
    double hotwordsScore = 1.5,
  }) async {
    if (isRunning || _debugStreaming || _starting) {
      return;
    }
    _starting = true;
    try {
      await _runDebugWavStream(
        modelKind: modelKind,
        modelDir: modelDir,
        vadModelPath: vadModelPath,
        wavPath: wavPath,
        routingProfile: routingProfile,
        senseVoiceModelDir: senseVoiceModelDir,
        lidModelDir: lidModelDir,
        realtime: realtime,
        decodingMethod: decodingMethod,
        vadSensitivity: vadSensitivity ?? _vadSensitivity,
        hotwordsFile: hotwordsFile,
        hotwordsScore: hotwordsScore,
      );
    } finally {
      _starting = false;
    }
  }

  Future<void> _runDebugWavStream({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
    required String wavPath,
    required RoutingProfile routingProfile,
    String? senseVoiceModelDir,
    String? lidModelDir,
    required bool realtime,
    String? decodingMethod,
    required VadSensitivity vadSensitivity,
    String? hotwordsFile,
    double hotwordsScore = 1.5,
  }) async {
    final wavFile = File(wavPath);
    if (!await wavFile.exists()) {
      throw LiveTranscriberException('WAV file not found: $wavPath');
    }

    try {
      await _buildNativeState(
        modelKind: modelKind,
        modelDir: modelDir,
        vadModelPath: vadModelPath,
        routingProfile: routingProfile,
        senseVoiceModelDir: senseVoiceModelDir,
        lidModelDir: lidModelDir,
        decodingMethod: decodingMethod,
        vadSensitivity: vadSensitivity,
        hotwordsFile: hotwordsFile,
        hotwordsScore: hotwordsScore,
      );
    } catch (_) {
      // See start()'s identical catch: _buildNativeState can throw after
      // already building the recognizer/routed set, and _debugStreaming is
      // still false here, so without this the partially-built native state
      // would leak.
      await _teardownNativeState();
      rethrow;
    }
    _applyPendingVadSensitivityIfAny();
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
