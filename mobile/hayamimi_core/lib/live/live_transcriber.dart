import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../bench/model_file_resolver.dart';
import '../bench/model_kind.dart';
import '../punct/punct_ja_text.dart';
import '../remote/wav_pcm_reader.dart';
import '../routing/routed_recognizer.dart';
import '../routing/routing_profile.dart';
import 'decode_protocol.dart';
import 'decode_session.dart';
import 'decode_worker.dart';
import 'draft_pass.dart';
import 'ja_punctuation.dart';
import 'live_transcript_entry.dart';
import 'live_vad.dart';
import 'model_load_event.dart';
import 'native_model_loader.dart';
import 'pcm_frame_buffer.dart';
import 'preroll.dart';
import 'refine_pass.dart';
import 'speech_segment_filter.dart';
import 'vad_sensitivity.dart';

/// How often the auto-refine due check ([isAutoRefineDue]) runs while a
/// session is live and auto mode is on. Cheap (a few comparisons), so a 1s
/// tick is plenty responsive without meaningfully affecting battery.
const _autoRefineCheckInterval = Duration(seconds: 1);

/// How long [LiveTranscriber.stop] waits for whatever is still queued at
/// the decode worker before giving up on it.
///
/// Stopping flushes the VAD, which usually produces one last segment; that
/// segment's transcript is worth waiting for, and used to arrive for free
/// when decoding was synchronous. The bound is here so a worker that has
/// wedged inside a decode cannot wedge the host app's stop()/dispose()
/// with it. Ten seconds is generous next to the fraction of a second a
/// segment takes, and still short next to a user waiting for a screen to
/// close.
const _decodeDrainTimeout = Duration(seconds: 10);

/// The least audio a refine ("清書") pass is worth running over, in
/// seconds. Mirrors the desktop `Refiner`'s own `len(buf) < sr // 2`
/// guard: re-decoding a fraction of a second of speech as a "group" costs
/// a full decode and cannot produce a better result than the final that
/// already covered it.
const _minRefineAudioSeconds = 0.5;

/// Builds the real, isolate-backed decode worker. Swapped out only by
/// tests (see [LiveTranscriber]'s `decodeWorkerFactory`).
DecodeWorker _spawnDecodeWorker() => IsolateDecodeWorker();

/// A [LiveTranscriber] failure: thrown by `start`/`startDebugWavStream`
/// for a bad configuration (e.g. a missing model file), and emitted on
/// [LiveTranscriber.errors] for a failure mid-session -- e.g. the OS
/// revoking microphone access, or the decode worker isolate dying.
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
/// pieces it's built from — [pcm16BytesToFloat32], [PcmFrameBuffer],
/// [isSegmentWorthDecoding] and [PrerollHistory] — are unit tested
/// separately.
///
/// Threading: neither model loading nor decoding runs on the caller's
/// isolate. Loading happens on short-lived background isolates (see
/// `native_model_loader.dart`); every decode — per-segment finals, drafts
/// and refine passes alike — is sent to a decode worker isolate that owns
/// the recognizers for the session (`decode_worker.dart`). What is left on
/// the caller's isolate is the microphone stream, one Silero VAD
/// `acceptWaveform` call per 32 ms frame, and dispatching the events below.
/// See the README's "Threading / known limitations" for what that does and
/// does not buy.
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
  /// [prerollSeconds] (see `preroll.dart`) is the seventh knob and the one
  /// exception to that rule: it may be 0, which turns pre-roll off, and
  /// only a negative or non-finite value throws.
  ///
  /// Validation runs before [recorder] is defaulted to a fresh
  /// `AudioRecorder()` (Dart evaluates an initializer list left to right):
  /// `AudioRecorder()`'s constructor kicks off a real, asynchronous
  /// platform-channel call under the hood, so a caller passing an invalid
  /// knob value gets a clean, synchronous [ArgumentError] instead of also
  /// leaking that half-started platform call.
  ///
  /// [decodeWorkerFactory] decides what a session's decodes are sent to,
  /// and [vadFactory] what its speech detection runs on. Leave both unset
  /// in an app: the defaults spawn the real decode worker isolate and load
  /// Silero VAD through sherpa-onnx. They exist because the sherpa-onnx
  /// native libraries cannot be loaded in a plain `flutter test` run, and
  /// between them they are what lets a whole session -- microphone to
  /// transcript line -- be driven without a device.
  LiveTranscriber({
    AudioRecorder? recorder,
    DecodeWorker Function()? decodeWorkerFactory,
    LiveVadFactory? vadFactory,
    double draftIntervalSeconds = defaultDraftIntervalSeconds,
    double draftWindowSeconds = defaultDraftWindowSeconds,
    double minDraftAudioSeconds = defaultMinDraftAudioSeconds,
    double autoRefineSilenceSeconds = defaultAutoRefineSilenceSeconds,
    double autoRefineMaxBufferedSeconds = defaultAutoRefineMaxBufferedSeconds,
    double refineBufferMaxSeconds = defaultRefineBufferMaxSeconds,
    double prerollSeconds = defaultPrerollSeconds,
  }) : _prerollSeconds = _requireNonNegative(
         prerollSeconds,
         'prerollSeconds',
       ),
       _draftIntervalSeconds = _requirePositive(
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
       _decodeWorkerFactory = decodeWorkerFactory ?? _spawnDecodeWorker,
       _vadFactory = vadFactory ?? buildSileroLiveVad,
       _recorder = recorder ?? AudioRecorder();

  /// The mic capture / VAD / recognizer sample rate this transcriber uses
  /// throughout, in Hz. Fixed -- not configurable per session.
  static const int sampleRate = 16000;

  static double _requirePositive(double value, String name) {
    if (!value.isFinite || value <= 0) {
      throw ArgumentError.value(value, name, 'must be a positive, finite number');
    }
    return value;
  }

  /// Same as [_requirePositive] but allows zero, for [prerollSeconds]: zero
  /// is the meaningful "no pre-roll at all" setting, not a mistake.
  static double _requireNonNegative(double value, String name) {
    if (!value.isFinite || value < 0) {
      throw ArgumentError.value(
        value,
        name,
        'must be a non-negative, finite number',
      );
    }
    return value;
  }

  final AudioRecorder _recorder;
  final DecodeWorker Function() _decodeWorkerFactory;
  final LiveVadFactory _vadFactory;
  final _entriesController = StreamController<LiveTranscriptEntry>.broadcast();
  final _decodingController = StreamController<bool>.broadcast();
  final _refineEntriesController =
      StreamController<LiveTranscriptEntry>.broadcast();
  final _draftController = StreamController<LiveTranscriptEntry>.broadcast();
  final _errorsController =
      StreamController<LiveTranscriberException>.broadcast();
  final _modelLoadController = StreamController<ModelLoadEvent>.broadcast();
  final _sessionResetController = StreamController<void>.broadcast();

  LiveVad? _vad;
  // The audio a segment's pre-roll is taken from: every frame fed to the
  // VAD is also pushed here, so a segment can be extended backwards past
  // the onset the VAD reported (see preroll.dart for why that is needed).
  // Built alongside the VAD, dropped with it.
  PrerollHistory? _prerollHistory;
  // How much audio this session has fed its VAD(s) in total, and how much
  // it had fed when the CURRENT VAD instance was installed. A VAD counts
  // samples from its own first frame, so a mid-session sensitivity swap
  // resets that count to zero; adding the origin back is what keeps
  // segment positions on one continuous session clock.
  int _samplesFedToVad = 0;
  int _vadSampleOrigin = 0;
  // The worker isolate that owns this session's recognizer(s), and the
  // queue in front of it. Null between sessions; built by
  // _buildNativeState, shut down by _teardownNativeState.
  DecodeSession? _decode;
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
  LiveVad? _pendingVadInstall;
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
  // True while ANY request (final, draft, refine, session reset) is
  // outstanding at the decode worker -- in flight or still waiting its
  // turn on this side. This is what [decoding] reports, and what the draft
  // and VAD-swap guards read. It used to be a plain flag set around a
  // synchronous decode call; now it is the decode queue's own state, which
  // means it stays true across the whole burst rather than flickering once
  // per decode.
  bool get _busy => _decode?.isBusy ?? false;
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
  double _prerollSeconds;
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
  // `scripts/realtime_transcribe.py`). When start() was given a
  // `punctuation` model, the worker also restores 、 and 。 into the result
  // before sending it back; without one this stays re-decode only. Built in
  // the constructor initializer list above (its cap comes from the
  // refineBufferMaxSeconds constructor parameter).
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

  /// How much audio from before a segment's detected onset is prepended to
  /// it before decoding, in seconds. Defaults to [defaultPrerollSeconds].
  ///
  /// Silero VAD reports a speech onset slightly late, so the samples it
  /// hands over can start partway into the first word and the recognizer
  /// then transcribes that word wrong (`資料は` came back as `昨日は` on the
  /// Android emulator). Prepending the audio recorded just before the
  /// onset gives the recognizer the run-up it needs. That audio counts as
  /// part of the segment: it is what gets decoded, what
  /// [LiveTranscriptEntry.audioSeconds] reports, and what the refine
  /// ("清書") buffer stores.
  ///
  /// Runtime-settable, applying from the next segment onward, without
  /// touching any native model. Unlike the other knobs, `0` is allowed and
  /// means "no pre-roll", i.e. decode exactly what the VAD delimited;
  /// negative or non-finite values throw [ArgumentError].
  double get prerollSeconds => _prerollSeconds;
  set prerollSeconds(double value) =>
      _prerollSeconds = _requireNonNegative(value, 'prerollSeconds');

  /// Hard cap (seconds) on how much audio the refine buffer holds before it
  /// starts dropping the oldest segment. Defaults to
  /// [defaultRefineBufferMaxSeconds]. Runtime-settable: applies from the
  /// next [RefineBuffer.add] onward, without touching any native model.
  double get refineBufferMaxSeconds => _refineBuffer.maxDurationSeconds;
  set refineBufferMaxSeconds(double value) => _refineBuffer.maxDurationSeconds =
      _requirePositive(value, 'refineBufferMaxSeconds');

  /// Finalized transcript lines, one per detected speech segment.
  Stream<LiveTranscriptEntry> get entries => _entriesController.stream;

  /// Emits `true` when the decode worker has work outstanding and `false`
  /// when it runs out, so the UI can show a busy indicator. Covers every
  /// kind of request — per-segment finals, drafts, refine passes and
  /// session resets — because the worker serves them one at a time.
  ///
  /// One event per transition, not one per decode: a burst of segments
  /// queued back to back reads as a single `true` … `false` pair.
  Stream<bool> get decoding => _decodingController.stream;

  /// Refine ("清書") results: one entry per manual or automatic refine
  /// pass, each covering everything buffered since the previous refine (or
  /// since the session started).
  ///
  /// This is the only stream Japanese punctuation restoration touches: when
  /// [start] was given a `punctuation` model, an entry here can carry 、 。
  /// ？ that the recognizer never produced, and says so through
  /// [LiveTranscriptEntry.punctuated].
  Stream<LiveTranscriptEntry> get refineEntries => _refineEntriesController.stream;

  /// In-progress ("draft") decodes: one per draft re-decode while a VAD
  /// segment is still open, replaced by the next draft or cleared by the
  /// segment's eventual [entries] final. Never buffered/stored by this
  /// class -- purely a live "typewriter" signal for the UI/broadcast.
  Stream<LiveTranscriptEntry> get drafts => _draftController.stream;

  /// Asynchronous session failures that aren't attributable to a call the
  /// caller made. Three things reach here:
  ///
  ///  * mic capture dying under a running session (the OS revoking the mic
  ///    on backgrounding, another app taking it),
  ///  * the decode worker isolate dying mid-session,
  ///  * a single decode throwing inside the worker.
  ///
  /// The first two stop the session, so [isRunning] reflects reality by the
  /// time the error is emitted; the third loses only that one utterance and
  /// leaves the session running. Errors raised *by* [start] itself are
  /// thrown, not emitted here.
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

  /// Whether a session is currently active: microphone capture is running
  /// and the decode worker is loaded.
  bool get isRunning => _micSubscription != null;

  /// Whether [startDebugWavStream] is currently paced-streaming a wav file
  /// through the same VAD/draft/final/refine pipeline a live mic session
  /// uses. See that method's doc for why this exists (emulators have no
  /// usable microphone).
  bool get isDebugStreaming => _debugStreaming;

  /// The session's current language when running with
  /// [RoutingProfile.jaSenseVoice] (`null` before the first segment
  /// resolves one, and always `null` for a plain single-model session).
  String? get currentLang => _decode?.currentLang;

  /// Total audio currently buffered for the next refine pass, in seconds.
  double get refineBufferedSeconds => _refineBuffer.totalDurationSeconds;

  /// Whether auto-refine (silence-triggered, see `refine_pass.dart`) is on.
  /// Defaults to off: on a phone every refine is a full re-decode burning
  /// battery and generating heat, so firing it automatically is opt-in —
  /// the manual "清書" button always works regardless of this setting.
  bool get autoRefineEnabled => _autoRefineEnabled;

  set autoRefineEnabled(bool value) {
    _autoRefineEnabled = value;
    if (!value) {
      _autoRefineTimer?.cancel();
      _autoRefineTimer = null;
    } else if (isRunning || _debugStreaming) {
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
  /// see [VadSensitivity], whose defaults are the desktop pipeline's tuned
  /// values rather than sherpa-onnx's stock ones, and which explains why).
  /// `null` (the default) uses those defaults. Change it after [start] via
  /// [setVadSensitivity] instead of stopping and restarting the session.
  ///
  /// [hotwordsFile]/[hotwordsScore] bias the recognizer toward a wordlist
  /// (sherpa-onnx's own hotwords feature) on the plain path and the routed
  /// ja tier; `null` (the default) leaves hotwords off. Unlike the pacing
  /// knobs and [vadSensitivity], hotwords have no runtime setter — they're
  /// baked into the recognizer at build time, so changing them requires a
  /// fresh [start] (or [stop] + [start]).
  ///
  /// [punctuation] turns on Japanese punctuation restoration for this
  /// session's refine ("清書") results: the recognizer emits a bare run of
  /// characters, and the model named here puts 、 and 。 (and ？ after a
  /// question ending) back into it, the way the desktop pipeline's
  /// `scripts/punct_ja.py` does. `null` (the
  /// default) leaves it off, which is what this method did before the
  /// parameter existed. What changes when it is set:
  ///
  ///  * only [refineEntries] is affected — [entries] and [drafts] carry the
  ///    recognizer's text unchanged (see `decode_worker.dart` for why);
  ///  * a routed session punctuates the refines that came back as Japanese
  ///    and leaves the rest alone; a plain session punctuates every refine,
  ///    so **only pass this to a plain session whose model is Japanese**;
  ///  * each affected entry carries [LiveTranscriptEntry.punctuated];
  ///  * the model is loaded in the decode worker, after the recognizers,
  ///    and reported on [modelLoads] as `model: 'punct'`. It is another
  ///    181.8 MB of memory for the life of the session, and a failure to
  ///    load it fails this call rather than starting a session that
  ///    quietly produces unpunctuated text.
  ///
  /// Like hotwords, it is fixed for the session: changing it means a fresh
  /// [start].
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
    JaPunctuation? punctuation,
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
          punctuation: punctuation,
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
    JaPunctuation? punctuation,
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

    if (routingProfile.dualConfirmed &&
        (senseVoiceModelDir == null || lidModelDir == null)) {
      throw LiveTranscriberException(
        '${routingProfile.label} requires senseVoiceModelDir and '
        'lidModelDir.',
      );
    }

    if (punctuation != null) {
      // Checked here, alongside the recognizer and VAD paths, so a missing
      // 182 MB model is reported before an isolate is spawned and up to
      // 396 MB of recognizer weights are loaded for a session that is about
      // to fail anyway. The worker checks again -- it is the one that can
      // tell a file that exists from a file it can actually load.
      for (final (label, path) in <(String, String)>[
        ('Japanese punctuation model', punctuation.modelPath),
        ('Japanese punctuation vocabulary', punctuation.vocabPath),
      ]) {
        if (!await File(path).exists()) {
          throw LiveTranscriberException('$label not found: $path');
        }
      }
    }

    // Captured before the worker is started and stamped on every callback
    // below (see [_ifCurrent]): a worker replies on its own schedule, so
    // this is what tells a reply whether the session it belongs to is
    // still the one running -- the same generation check
    // [isVadBuildStale] applies to a VAD rebuild.
    final generation = _sessionGeneration;
    final DecodeSession session;
    try {
      session = await DecodeSession.start(
        worker: _decodeWorkerFactory(),
        config: DecodeWorkerConfig(
          routed: routingProfile.dualConfirmed,
          modelDir: modelDir,
          senseVoiceModelDir: senseVoiceModelDir,
          lidModelDir: lidModelDir,
          decodingMethod: decodingMethod,
          hotwordsFile: hotwordsFile,
          hotwordsScore: hotwordsScore,
          sampleRate: sampleRate,
          punctModelPath: punctuation?.modelPath,
          punctVocabPath: punctuation?.vocabPath,
          punctLibraryPath: punctuation?.libraryPath,
          punctNumThreads: punctuation?.numThreads ?? defaultPunctNumThreads,
          // Left at the default (true): for a plain single-model session
          // there is no language tag to test, and passing a Japanese
          // punctuation model to one is itself the statement that its model
          // transcribes Japanese. See
          // DecodeWorkerConfig.punctuatePlainSession.
        ),
        buildRefinePayload: () => _buildRefinePayload(generation),
        onFinal: (samples, result) =>
            _ifCurrent(generation, () => _onFinalDecoded(samples, result)),
        onDraft: (result) =>
            _ifCurrent(generation, () => _onDraftDecoded(result)),
        onRefine: (payload, result) =>
            _ifCurrent(generation, () => _onRefineDecoded(payload, result)),
        onReset: () => _ifCurrent(generation, _onSessionReset),
        onModelLoad: (model, phase, ms) => _ifCurrent(
          generation,
          () => _emitModelLoad(model: model, phase: phase, ms: ms),
        ),
        onBusyChanged: (busy) => _ifCurrent(generation, () {
          if (!_decodingController.isClosed) {
            _decodingController.add(busy);
          }
        }),
        onDecodeError: (message) =>
            _ifCurrent(generation, () => _onDecodeError(message)),
        onWorkerDied: (message) => _ifCurrent(
          generation,
          () => unawaited(_handleWorkerFailure(message)),
        ),
      );
    } on DecodeWorkerException catch (e) {
      // The worker frees whatever it had already built and exits before
      // this throw, so there is nothing left over here -- only the message,
      // which is verbatim what the in-process loader used to throw.
      throw LiveTranscriberException(e.message);
    }
    if (generation != _sessionGeneration) {
      // The session was torn down (or restarted) while these models were
      // loading. Installing this worker would orphan whichever one the new
      // session built, so shut it down instead of keeping it.
      await session.shutdown();
      throw LiveTranscriberException(
        'The session was stopped while its models were still loading.',
      );
    }
    _decode = session;

    _vadModelPath = vadModelPath;
    _vadSensitivity = vadSensitivity;
    _emitModelLoad(model: 'vad', phase: 'start');
    final vadStopwatch = Stopwatch()..start();
    final vad = await _vadFactory(
      modelPath: vadModelPath,
      sensitivity: vadSensitivity,
      sampleRate: sampleRate,
    );
    vadStopwatch.stop();
    _vad = vad;
    _emitModelLoad(
      model: 'vad',
      phase: 'done',
      ms: vadStopwatch.elapsedMicroseconds / 1000,
    );
    _frameBuffer = PcmFrameBuffer(frameSize: vad.frameSize);
    _prerollHistory = PrerollHistory(sampleRate: sampleRate);
    _samplesFedToVad = 0;
    _vadSampleOrigin = 0;
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
    // Recorded before the VAD sees it, on the same clock the VAD counts
    // on, so a segment the VAD reports can be extended backwards into the
    // audio that preceded it.
    _prerollHistory?.push(frame);
    _samplesFedToVad += frame.length;
    vad.acceptWaveform(frame);

    // Accumulate this VAD window for the draft pass while a segment is in
    // progress. `isDetected()` has no "current segment audio" accessor in
    // the Dart bindings (unlike the desktop's `vad.current_segment.samples`
    // -- see draft_pass.dart's doc comment), so this buffer is built by
    // hand from the same frames already being fed to the VAD.
    final detected = vad.isSpeechDetected;
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
    // Whichever segment those frames belonged to is over (it closed, or a
    // new one just started, or the session was reset). A draft still in
    // flight for it would come back describing audio the UI has already
    // moved past, so bump the id it was stamped with -- that is what lets
    // its result be recognized as stale and dropped instead of
    // overwriting the final that replaced it.
    _decode?.beginDraftSegment();
  }

  void _drainReadySegments() {
    final vad = _vad;
    if (vad == null) {
      return;
    }
    while (vad.hasSegment) {
      final segment = vad.takeSegment();
      // The segment just taken supersedes whatever the draft pass had
      // accumulated for it -- the real (properly routed/decoded) final is
      // about to replace any draft the UI was showing.
      _clearDraftFrames();
      _draftSegmentActive = false;
      _decodeSegment(segment);
    }
  }

  void _decodeSegment(LiveSpeechSegment segment) {
    // Judged on what the VAD actually delimited, before any pre-roll is
    // added: pre-roll is context, not speech, and letting a second of it
    // pad a 0.1s blip past the minimum would send near-silence to the
    // recognizer for no reason.
    final worthDecoding = isSegmentWorthDecoding(
      segment.samples,
      sampleRate: sampleRate,
    );
    // Extended even when the segment is about to be dropped, because this
    // call is also what moves the history's "do not reach back past here"
    // marker: skipping it would let the NEXT segment's pre-roll cover this
    // one's audio. Same order the desktop pipeline's drain_segments uses.
    final samples = _withPreroll(segment);
    if (!worthDecoding) {
      return;
    }
    // Queued at the worker, never dropped, and answered in the order the
    // segments closed -- so the transcript keeps the order the speaker
    // said things in even when several segments finish back to back.
    _decode?.submitFinal(samples);
  }

  /// Gives [segment] the audio recorded just before the VAD noticed it,
  /// which is what stops an utterance-initial word being clipped (see
  /// `preroll.dart`). Returns the segment's own samples unchanged when
  /// [prerollSeconds] is 0, or when there is no history to draw on.
  Float32List _withPreroll(LiveSpeechSegment segment) {
    final history = _prerollHistory;
    if (history == null) {
      return segment.samples;
    }
    return history.withPreroll(
      segmentStartSample: _vadSampleOrigin + segment.startSample,
      samples: segment.samples,
      prerollSeconds: _prerollSeconds,
    );
  }

  /// Turns one finished segment's decode into a transcript line and files
  /// its audio for the next refine pass.
  ///
  /// The empty check is on the trimmed text but the emitted text is not
  /// trimmed. That asymmetry is what the in-process version did, and it is
  /// what a caller diffing output across this change would notice if it
  /// were quietly cleaned up here.
  void _onFinalDecoded(Float32List samples, DecodeWorkerResult result) {
    final text = result.text.trim();
    if (text.isEmpty) {
      return;
    }
    final now = DateTime.now();
    if (!_entriesController.isClosed) {
      _entriesController.add(
        LiveTranscriptEntry(
          text: result.text,
          timestamp: now,
          latencyMs: result.latencyMs,
          lang: result.lang,
          audioSeconds: samples.length / sampleRate,
          switched: result.switched,
        ),
      );
    }
    _lastSegmentAt = now;
    _refineBuffer.add(
      RefineSegment(samples: samples, text: text, capturedAt: now),
    );
  }

  void _onDraftDecoded(DecodeWorkerResult result) {
    final text = result.text.trim();
    if (text.isEmpty) {
      return;
    }
    if (!_draftController.isClosed) {
      _draftController.add(
        LiveTranscriptEntry(
          text: text,
          timestamp: DateTime.now(),
          latencyMs: result.latencyMs,
          lang: result.lang,
        ),
      );
    }
  }

  void _onRefineDecoded(
    RefineRequestPayload payload,
    DecodeWorkerResult result,
  ) {
    var text = result.text.trim();
    var punctuated = result.punctuated;
    // A merged re-decode must never LOSE content: if it comes back much
    // shorter than the fast finals combined, trust those instead (mirrors
    // scripts/realtime_transcribe.py's Refiner).
    //
    // The length compared is the text WITHOUT the marks the punctuation
    // model wrote: it runs in the worker, before this, so by here a
    // punctuated refine is several characters longer than what was actually
    // said, and crediting it for those would let a truncated re-decode pass
    // a check it should fail. The fast text it is compared against is never
    // punctuated (only refines are), so only this side is stripped.
    if (isRefineTextTooShort(
      punctuated ? withoutRestoredMarks(text) : text,
      payload.fastText,
    )) {
      text = payload.fastText;
      // The fallback is the finals' own text, which no model punctuated.
      punctuated = false;
    }
    if (text.isEmpty) {
      return;
    }
    if (!_refineEntriesController.isClosed) {
      _refineEntriesController.add(
        LiveTranscriptEntry(
          text: text,
          timestamp: DateTime.now(),
          latencyMs: result.latencyMs,
          lang: result.lang,
          audioSeconds: payload.samples.length / sampleRate,
          punctuated: punctuated,
        ),
      );
    }
  }

  /// Claims the buffered segments for a refine pass, at the moment that
  /// pass is actually about to be sent rather than when it was asked for.
  ///
  /// The difference matters when the user taps the refine button while the
  /// last thing they said is still decoding: claiming the buffer late
  /// means that segment's final lands first and is part of the group being
  /// refined, instead of being pushed into the next one.
  ///
  /// Returns null when there is nothing worth decoding -- an empty buffer,
  /// or under half a second of audio (mirrors the desktop `Refiner`'s
  /// `len(buf) < sr // 2` guard) -- and the refine is dropped.
  RefineRequestPayload? _buildRefinePayload(int generation) {
    if (generation != _sessionGeneration || _refineBuffer.isEmpty) {
      return null;
    }
    final segments = _refineBuffer.takeAll();
    final combined = combineSegmentSamples(segments);
    if (combined.length < sampleRate * _minRefineAudioSeconds) {
      return null;
    }
    return RefineRequestPayload(
      samples: combined,
      fastText: combineSegmentFastText(segments),
    );
  }

  /// Clears the "conversation" state this class owns, once the worker has
  /// confirmed it cleared its own (see [resetSession]).
  void _onSessionReset() {
    _refineBuffer.clear();
    _lastSegmentAt = null;
    _clearDraftFrames();
    _draftSegmentActive = false;
    _lastDraftAt = null;
    if (!_sessionResetController.isClosed) {
      _sessionResetController.add(null);
    }
  }

  /// One decode threw inside the worker. That utterance is lost; the
  /// worker and the session are both still fine, so this is reported and
  /// nothing is torn down.
  void _onDecodeError(String message) {
    if (!_errorsController.isClosed) {
      _errorsController.add(
        LiveTranscriberException('Decode failed: $message'),
      );
    }
  }

  /// The decode worker isolate died without being asked to.
  ///
  /// Nothing can be transcribed any more, and the queue that was waiting
  /// on it is gone, so this is handled exactly like the microphone being
  /// revoked: report it on [errors] and tear the session down, leaving the
  /// object ready to [start] again.
  Future<void> _handleWorkerFailure(String message) async {
    if (_decode == null) {
      return;
    }
    if (!_errorsController.isClosed) {
      _errorsController.add(
        LiveTranscriberException(
          'The decode worker stopped unexpectedly ($message); this session '
          'has been stopped.',
        ),
      );
    }
    if (isRunning) {
      await stop();
      return;
    }
    if (_debugStreaming) {
      // The wav loop notices this once per frame and tears down in its own
      // finally, the same way stopDebugWavStream() ends it.
      _debugStreamCancelRequested = true;
      return;
    }
    await _teardownNativeState();
  }

  /// Runs [action] only if the session it was registered for is still the
  /// one running. Every decode-worker callback goes through this: a
  /// worker replies on its own schedule, so a reply that was already in
  /// flight when the session stopped must not write into the streams of
  /// whatever came after it.
  void _ifCurrent(int generation, void Function() action) {
    if (generation != _sessionGeneration) {
      return;
    }
    action();
  }

  /// Waits, with a bound, for everything queued at the decode worker to
  /// finish. See [_decodeDrainTimeout] for why the bound is there.
  Future<void> _waitForDecodesToDrain() async {
    final decode = _decode;
    if (decode == null) {
      return;
    }
    await decode.waitForIdle().timeout(_decodeDrainTimeout, onTimeout: () {});
  }

  /// Fires a draft decode if one is due (see [isDraftDue]): enough draft
  /// audio has accumulated, the interval has elapsed, and the decode
  /// worker has nothing outstanding. A no-op otherwise -- the next mic/wav
  /// frame will just check again.
  void _maybeEmitDraft() {
    final decode = _decode;
    if (decode == null || _draftFrames.isEmpty) {
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
    // Snapshot now: more frames arrive (and a final may pop) while this
    // decode is in flight at the worker, so decoding off a copy avoids
    // racing the live accumulator.
    final windowed = capDraftWindow(
      concatFloat32Lists(_draftFrames),
      sampleRate: sampleRate,
      maxSeconds: _draftWindowSeconds,
    );
    if (windowed.length < sampleRate * _minDraftAudioSeconds) {
      return;
    }
    decode.submitDraft(windowed);
  }

  /// Runs a refine pass over everything currently buffered: combines the
  /// buffered segments' audio and re-decodes it as one utterance in the
  /// decode worker. Emits nothing if the buffer is empty, the combined
  /// audio is under half a second, or the result is empty after the
  /// too-short fallback.
  ///
  /// The returned future completes once that pass's result has been
  /// emitted on [refineEntries] (or once it has been decided there was
  /// nothing to emit). Calling this again while a refine is still waiting
  /// its turn behind other work returns *that* pass's future rather than
  /// starting a second one over overlapping audio; a refine that has
  /// already been sent to the worker is past that point, so a call made
  /// then does queue a second pass over whatever has been buffered since.
  ///
  /// Safe to call whether or not [autoRefineEnabled] is on -- this is also
  /// what the manual refine button calls directly.
  Future<void> refineNow() async {
    await _decode?.refine();
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
    // The flush above usually produces one last segment. Its transcript
    // used to arrive before stop() returned simply because decoding was
    // synchronous; now it has to be waited for on purpose.
    await _waitForDecodesToDrain();

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
    _prerollHistory = null;
    _vadModelPath = null;
    // The decode worker owns this session's recognizer handles and is the
    // only isolate allowed to free them, so ending the session means
    // asking it to (and, if it will not answer, killing it -- see
    // IsolateDecodeWorker.shutdown for why nothing is freed by force).
    final decode = _decode;
    _decode = null;
    await decode?.shutdown();
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
        final built = await _vadFactory(
          modelPath: vadModelPath,
          sensitivity: target,
          sampleRate: sampleRate,
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
    // The replacement has been fed nothing yet, so its own sample counter
    // starts at zero here. Remembering where "here" falls on the session
    // clock keeps segment positions -- and therefore pre-roll -- correct
    // across the swap.
    _vadSampleOrigin = _samplesFedToVad;
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
  /// The clear is queued behind whatever the decode worker already has,
  /// so a segment that was still decoding when this was called still lands
  /// in the transcript first, rather than racing a write into the buffers
  /// this is about to empty. The returned future completes once the worker
  /// has confirmed it cleared its own state and [sessionResets] has fired.
  ///
  /// A no-op when no session ([isRunning]) or debug wav stream is active --
  /// there is nothing to reset, and [sessionResets] does not fire.
  Future<void> resetSession() async {
    if (!isRunning && !_debugStreaming) {
      return;
    }
    await _decode?.reset();
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
  ///
  /// **What "awaits until done" covers.** The whole file is fed through,
  /// any in-progress segment is flushed and decoded, and then -- unlike
  /// [stop], which drops whatever was buffered -- one last refine ("清書")
  /// pass runs over the finals this stream produced, provided they add up
  /// to at least half a second of audio. That refine is awaited before the
  /// session is torn down, so by the time this future completes,
  /// [refineEntries] has already carried it.
  ///
  /// That closing refine is deliberate, and it is the difference between
  /// this method and [start]. The session is gone before the future
  /// resolves (the same teardown [stop] performs), so a caller who awaits
  /// this and *then* calls [refineNow] gets nothing: there is no longer a
  /// worker to run it. Without the closing pass, a default-configured debug
  /// stream would produce no refine at all -- which is the one thing this
  /// method exists to demonstrate on a device with no usable microphone.
  ///
  /// [autoRefineEnabled] also works here, not only for microphone
  /// sessions: set it to `true` before calling this and refines fire on
  /// silence gaps during the stream, the same way they would live. Setting
  /// it while a stream is already running works too.
  ///
  /// See [start] for what [decodingMethod]/[vadSensitivity]/[hotwordsFile]/
  /// [hotwordsScore]/[punctuation] do -- identical meaning and defaults
  /// here, which is what makes this the way to see punctuated refine output
  /// on an emulator.
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
    JaPunctuation? punctuation,
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
        punctuation: punctuation,
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
    JaPunctuation? punctuation,
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
        punctuation: punctuation,
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
      final samples = await _readDebugWavSamples(wavFile, wavPath);
      // Feed it into the same PcmFrameBuffer the mic path uses, to get the
      // fixed-size windows the VAD requires.
      final frameSize = frameBuffer.frameSize;
      final samplesPerFrameDuration = Duration(
        microseconds: (frameSize / sampleRate * 1000000).round(),
      );
      var offset = 0;
      while (offset < samples.length) {
        if (_debugStreamCancelRequested) {
          break;
        }
        final end = (offset + frameSize).clamp(0, samples.length);
        final chunk = Float32List.sublistView(samples, offset, end);
        frameBuffer.add(chunk);
        for (final frame in frameBuffer.drainFrames()) {
          _processFrame(frame);
        }
        offset = end;
        if (realtime) {
          await Future.delayed(samplesPerFrameDuration);
        } else {
          // Even when not pacing, yield once per frame: decodes now come
          // back from the worker as messages, and a loop that never
          // returns to the event loop would hold every result until the
          // whole file had been fed through.
          await Future<void>.delayed(Duration.zero);
        }
      }

      if (!_debugStreamCancelRequested) {
        _vad?.flush();
        _drainReadySegments();
        await _waitForDecodesToDrain();
        // The whole point of this method is to show what a live session
        // produces, and on a device without a microphone it is the only
        // way to see a punctuated refine at all. Teardown below ends the
        // session before this future completes, so a caller cannot ask for
        // one afterwards -- refineNow() would find no worker and silently
        // do nothing. So run the last one here, while there is still a
        // session to run it in, over everything the finals just buffered.
        if (_refineBuffer.totalDurationSeconds >= _minRefineAudioSeconds) {
          await refineNow();
        }
      } else {
        await _waitForDecodesToDrain();
      }
    } finally {
      _debugStreaming = false;
      // Started by _resetSessionState above when autoRefineEnabled was
      // already on. Nothing else cancels it on this path -- stop() only
      // ends a microphone session -- so without this it would keep ticking
      // against a torn-down session for the life of the object.
      _autoRefineTimer?.cancel();
      _autoRefineTimer = null;
      await _teardownNativeState();
    }
  }

  /// Reads [wavFile] as the 16 kHz mono 16-bit PCM audio this path
  /// requires.
  ///
  /// Uses this package's own pure-Dart wav parser rather than sherpa-onnx's
  /// `readWave`. The accepted format is the same -- both handle 16-bit PCM
  /// only -- but parsing in Dart keeps the streaming debug path free of FFI
  /// entirely, which is what lets a test push a wav through the whole
  /// VAD/draft/final/refine pipeline with no native library present.
  Future<Float32List> _readDebugWavSamples(File wavFile, String wavPath) async {
    final WavPcm16 wav;
    try {
      wav = parseWavPcm16(await wavFile.readAsBytes());
    } on WavParseException catch (e) {
      throw LiveTranscriberException(
        'Failed to read WAV file (${e.message}): $wavPath',
      );
    }
    if (wav.channels != 1) {
      throw LiveTranscriberException(
        'WAV has ${wav.channels} channels != required mono '
        '(resample the test wav to 16kHz mono first): $wavPath',
      );
    }
    if (wav.sampleRate != sampleRate) {
      // Unlike runDebugWavRefineTest (which just decodes at the wav's own
      // rate, no VAD involved), this path feeds the wav through the VAD,
      // which is configured for a fixed 16kHz -- a mismatched wav would
      // silently desync the VAD's speech/silence timing from real time.
      throw LiveTranscriberException(
        'WAV sample rate ${wav.sampleRate}Hz != required ${sampleRate}Hz '
        '(resample the test wav to 16kHz mono first): $wavPath',
      );
    }
    final samples = pcm16BytesToFloat32(wav.pcmBytes);
    if (samples.isEmpty) {
      throw LiveTranscriberException(
        'Failed to read WAV file (unsupported format or empty): $wavPath',
      );
    }
    return samples;
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
  ///
  /// Unlike a live session, this helper's three decodes run on the isolate
  /// that calls it, not on a decode worker. It is a debug-only,
  /// one-shot, explicitly-awaited call with its own short-lived models and
  /// no session to keep responsive, so a worker would buy nothing here and
  /// cost a spawn and a full model load per invocation. If you call it
  /// from a UI isolate, expect it to block for as long as the three
  /// decodes take. [startDebugWavStream] is the path that exercises the
  /// real pipeline, worker included.
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
