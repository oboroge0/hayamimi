import 'dart:async';

import 'bench/model_kind.dart';
import 'live/draft_pass.dart';
import 'live/ja_punctuation.dart';
import 'live/live_transcriber.dart';
import 'live/refine_pass.dart';
import 'live/vad_sensitivity.dart';
import 'routing/routing_profile.dart';
import 'server/subtitle_event.dart';

/// On-device live transcription, packaged as a single [SubtitleEvent]
/// stream.
///
/// This is the embedding-friendly facade over [LiveTranscriber]: it hides
/// the two separate `entries`/`refineEntries` streams behind one
/// [events] stream of [FinalSubtitleEvent]/[RefineSubtitleEvent], the same
/// event types [HayamimiRemote] and `SubtitleBroadcastServer` use, so a
/// host app can treat on-device and remote-PC transcription
/// interchangeably.
///
/// ```dart
/// final live = HayamimiLive();
/// live.events.listen((event) {
///   if (event is FinalSubtitleEvent) print(event.text);
/// });
/// await live.start(modelDir: modelDir, vadModelPath: vadPath);
/// // ... later
/// await live.stop();
/// await live.dispose();
/// ```
class HayamimiLive {
  /// If [transcriber] is omitted, the six pacing-knob parameters
  /// ([draftIntervalSeconds] through [refineBufferMaxSeconds]) seed the
  /// [LiveTranscriber] this facade creates for itself -- see that class's
  /// constructor for what each one does and its `default*` constant
  /// (`draft_pass.dart`/`refine_pass.dart`) for the value a caller who
  /// passes nothing gets. They're ignored if [transcriber] is given
  /// instead: configure an injected transcriber directly (its own
  /// constructor, or its runtime setters, both mirrored on this class).
  HayamimiLive({
    LiveTranscriber? transcriber,
    this.textTransform,
    double draftIntervalSeconds = defaultDraftIntervalSeconds,
    double draftWindowSeconds = defaultDraftWindowSeconds,
    double minDraftAudioSeconds = defaultMinDraftAudioSeconds,
    double autoRefineSilenceSeconds = defaultAutoRefineSilenceSeconds,
    double autoRefineMaxBufferedSeconds = defaultAutoRefineMaxBufferedSeconds,
    double refineBufferMaxSeconds = defaultRefineBufferMaxSeconds,
  }) : _transcriber =
           transcriber ??
           LiveTranscriber(
             draftIntervalSeconds: draftIntervalSeconds,
             draftWindowSeconds: draftWindowSeconds,
             minDraftAudioSeconds: minDraftAudioSeconds,
             autoRefineSilenceSeconds: autoRefineSilenceSeconds,
             autoRefineMaxBufferedSeconds: autoRefineMaxBufferedSeconds,
             refineBufferMaxSeconds: refineBufferMaxSeconds,
           ) {
    _entriesSubscription = _transcriber.entries.listen((entry) {
      final entryLang = entry.lang ?? lang;
      _eventsController.add(
        FinalSubtitleEvent(
          text: _transform(entry.text, entryLang),
          lang: entryLang,
          latencyMs: entry.latencyMs,
          audioSeconds: entry.audioSeconds,
          switched: entry.switched,
        ),
      );
    });
    _refineEntriesSubscription = _transcriber.refineEntries.listen((entry) {
      final entryLang = entry.lang ?? lang;
      _eventsController.add(
        RefineSubtitleEvent(
          text: _transform(entry.text, entryLang),
          lang: entryLang,
          latencyMs: entry.latencyMs,
          audioSeconds: entry.audioSeconds,
          punctuated: entry.punctuated,
        ),
      );
    });
    _decodingSubscription = _transcriber.decoding.listen(
      _decodingController.add,
    );
    _draftsSubscription = _transcriber.drafts.listen((entry) {
      final entryLang = entry.lang ?? lang;
      _eventsController.add(
        PartialSubtitleEvent(_transform(entry.text, entryLang)),
      );
    });
    _errorsSubscription = _transcriber.errors.listen((error) {
      _eventsController.add(ErrorSubtitleEvent(message: error.message));
    });
    _modelLoadsSubscription = _transcriber.modelLoads.listen((event) {
      _eventsController.add(
        ModelLoadSubtitleEvent(model: event.model, phase: event.phase, ms: event.ms),
      );
    });
    _sessionResetsSubscription = _transcriber.sessionResets.listen((_) {
      _eventsController.add(const SessionResetSubtitleEvent());
    });
  }

  final LiveTranscriber _transcriber;
  final _eventsController = StreamController<SubtitleEvent>.broadcast();
  final _decodingController = StreamController<bool>.broadcast();
  late final StreamSubscription _entriesSubscription;
  late final StreamSubscription _refineEntriesSubscription;
  late final StreamSubscription _decodingSubscription;
  late final StreamSubscription _draftsSubscription;
  late final StreamSubscription _errorsSubscription;
  late final StreamSubscription _modelLoadsSubscription;
  late final StreamSubscription _sessionResetsSubscription;
  bool _disposed = false;

  /// Optional text-postprocessing hook applied to every draft/final/refine
  /// entry's text, right here where [LiveTranscriber] entries become
  /// [SubtitleEvent]s -- before they reach [events] and therefore before
  /// any consumer (including `SubtitleBroadcastServer`) sees the text. Takes
  /// the raw decoded text and the entry's language tag (`entry.lang`, or
  /// this facade's [lang] when the entry doesn't carry its own) and returns
  /// the text to actually publish.
  ///
  /// Settable at any time, including mid-session: the next entry to arrive
  /// picks up the new transform. It runs last, after Japanese punctuation
  /// restoration if the session has it on, so a refine's text reaches it
  /// with 、 and 。 already in place. `null` (the default) is a no-op. This is
  /// deliberately just an insertion point -- e.g. for CJK ITN
  /// (`scripts/itn_cjk.py` on the desktop side) or a user find/replace
  /// dictionary -- and does not itself implement any postprocessing.
  String Function(String text, String lang)? textTransform;

  String _transform(String text, String entryLang) {
    final transform = textTransform;
    if (transform == null) {
      return text;
    }
    return transform(text, entryLang);
  }

  /// BCP-47-ish language tag stamped on every emitted event that doesn't
  /// carry its own (i.e. a plain single-model session, or a routed
  /// session's very first segment before any language has resolved yet —
  /// see [currentLang]). With [RoutingProfile.jaSenseVoice], each event's
  /// language instead comes from the segment that produced it.
  String lang = '';

  /// The routed session's current language (`null` for a plain
  /// single-model session, or before a routed session's first segment
  /// resolves one). Mirrors [LiveTranscriber.currentLang].
  String? get currentLang => _transcriber.currentLang;

  /// Finalized transcript lines ([FinalSubtitleEvent]), two-pass "refine"
  /// results ([RefineSubtitleEvent]), in-progress drafts
  /// ([PartialSubtitleEvent]), session failures ([ErrorSubtitleEvent] —
  /// e.g. the OS killing mic capture mid-session, after which [isRunning]
  /// is false again), native model loading progress
  /// ([ModelLoadSubtitleEvent], around each [start]/[startDebugWavStream]
  /// build and each [setVadSensitivity] rebuild), and [resetSession]
  /// completions ([SessionResetSubtitleEvent]), in arrival order.
  ///
  /// [ErrorSubtitleEvent] also covers the decode worker isolate dying
  /// mid-session (which stops the session, same as a revoked microphone)
  /// and a single decode failing inside it (which loses that utterance and
  /// leaves the session running) — see [LiveTranscriber.errors].
  Stream<SubtitleEvent> get events => _eventsController.stream;

  /// Emits `true` when the decode worker has work outstanding and `false`
  /// when it runs out — useful for a busy indicator. One event per
  /// transition, not one per decode; see [LiveTranscriber.decoding].
  Stream<bool> get decoding => _decodingController.stream;

  bool get isRunning => _transcriber.isRunning;

  /// Total audio currently buffered for the next refine pass, in seconds.
  double get refineBufferedSeconds => _transcriber.refineBufferedSeconds;

  /// Whether silence-triggered auto-refine is on. Off by default: every
  /// refine pass is a full re-decode, which costs battery and heat on a
  /// phone. [refineNow] always works regardless of this setting.
  bool get autoRefineEnabled => _transcriber.autoRefineEnabled;
  set autoRefineEnabled(bool value) => _transcriber.autoRefineEnabled = value;

  /// How often (seconds) a draft re-decode fires while a VAD segment is in
  /// progress. See [LiveTranscriber.draftIntervalSeconds] -- same default,
  /// same runtime-settable, no-native-rebuild behavior.
  double get draftIntervalSeconds => _transcriber.draftIntervalSeconds;
  set draftIntervalSeconds(double value) =>
      _transcriber.draftIntervalSeconds = value;

  /// Trailing-audio cap (seconds) a draft decode re-processes. See
  /// [LiveTranscriber.draftWindowSeconds].
  double get draftWindowSeconds => _transcriber.draftWindowSeconds;
  set draftWindowSeconds(double value) =>
      _transcriber.draftWindowSeconds = value;

  /// Minimum accumulated audio (seconds) before a draft decode runs. See
  /// [LiveTranscriber.minDraftAudioSeconds].
  double get minDraftAudioSeconds => _transcriber.minDraftAudioSeconds;
  set minDraftAudioSeconds(double value) =>
      _transcriber.minDraftAudioSeconds = value;

  /// Silence gap (seconds) that fires an auto-refine. See
  /// [LiveTranscriber.autoRefineSilenceSeconds].
  double get autoRefineSilenceSeconds => _transcriber.autoRefineSilenceSeconds;
  set autoRefineSilenceSeconds(double value) =>
      _transcriber.autoRefineSilenceSeconds = value;

  /// Buffered-duration ceiling (seconds) that fires an auto-refine even
  /// without a silence gap. See
  /// [LiveTranscriber.autoRefineMaxBufferedSeconds].
  double get autoRefineMaxBufferedSeconds =>
      _transcriber.autoRefineMaxBufferedSeconds;
  set autoRefineMaxBufferedSeconds(double value) =>
      _transcriber.autoRefineMaxBufferedSeconds = value;

  /// Hard cap (seconds) on how much audio the refine buffer holds. See
  /// [LiveTranscriber.refineBufferMaxSeconds].
  double get refineBufferMaxSeconds => _transcriber.refineBufferMaxSeconds;
  set refineBufferMaxSeconds(double value) =>
      _transcriber.refineBufferMaxSeconds = value;

  /// Starts capturing mic audio and transcribing it.
  ///
  /// [modelDir] must contain a zipformer transducer model.
  /// [vadModelPath] must point at a Silero VAD onnx model file (e.g.
  /// `silero_vad.onnx`).
  ///
  /// See [LiveTranscriber.start] for what [decodingMethod]/
  /// [vadSensitivity]/[hotwordsFile]/[hotwordsScore] do and their defaults
  /// -- forwarded through unchanged.
  ///
  /// [punctuation] turns on Japanese punctuation restoration: refine
  /// ("清書") results then arrive with 、 and 。 in them instead of as an
  /// unbroken run of characters, and their [RefineSubtitleEvent] says so
  /// through `punctuated` (also `"punctuated"` in `toJson`, so a LAN
  /// consumer can see it too). Nothing else changes -- [FinalSubtitleEvent]
  /// and [PartialSubtitleEvent] carry the recognizer's text as before. The
  /// model is loaded in the decode worker and costs 181.8 MB of memory for
  /// the session; failing to load it fails this call. [textTransform], if
  /// set, still runs last, on the punctuated text.
  Future<void> start({
    required String modelDir,
    required String vadModelPath,
    ModelKind modelKind = ModelKind.zipformerTransducer,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
    String? decodingMethod,
    VadSensitivity? vadSensitivity,
    String? hotwordsFile,
    double hotwordsScore = 1.5,
    JaPunctuation? punctuation,
  }) {
    return _transcriber.start(
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
  }

  /// Stops capturing and releases native VAD/recognizer resources.
  Future<void> stop() => _transcriber.stop();

  /// Runs a refine ("清書") pass over everything buffered since the last
  /// refine (or session start): re-decodes it as one utterance and emits a
  /// [RefineSubtitleEvent] on [events]. Safe to call at any time; emits
  /// nothing if there's too little buffered audio. The returned future
  /// completes once that pass's event has been emitted — see
  /// [LiveTranscriber.refineNow] for what a second call during one does.
  Future<void> refineNow() => _transcriber.refineNow();

  /// Rebuilds Silero VAD off-isolate with [sensitivity] and swaps it in at
  /// the next safe point (never mid-segment) -- see
  /// [LiveTranscriber.setVadSensitivity]. If no session is running, just
  /// remembers [sensitivity] for the next [start].
  Future<void> setVadSensitivity(VadSensitivity sensitivity) =>
      _transcriber.setVadSensitivity(sensitivity);

  /// Clears the current "conversation" -- refine buffer, draft state, and
  /// (for a routed session) which language it's locked to -- without
  /// reloading any native model, then emits a [SessionResetSubtitleEvent]
  /// on [events]. See [LiveTranscriber.resetSession] for the full
  /// behavior, including what happens if a decode is in flight. A no-op
  /// (no event emitted) when no session is running.
  Future<void> resetSession() => _transcriber.resetSession();

  /// Whether [startDebugWavStream] is currently paced-streaming a wav file
  /// through the pipeline instead of the mic.
  bool get isDebugStreaming => _transcriber.isDebugStreaming;

  /// Streams [wavPath] (16kHz mono) through the exact same VAD/draft/final/
  /// refine pipeline [start] uses, at (roughly) real-time pace, instead of
  /// mic input. Emits the same [events] a live mic session would.
  ///
  /// Exists so this facade can be exercised end to end without a real
  /// microphone -- e.g. on an Android emulator, which has none. Awaits
  /// until the whole file has been fed through and any in-progress segment
  /// has been flushed and decoded. See [start] for [decodingMethod]/
  /// [vadSensitivity]/[hotwordsFile]/[hotwordsScore]/[punctuation].
  Future<void> startDebugWavStream({
    required String modelDir,
    required String vadModelPath,
    required String wavPath,
    ModelKind modelKind = ModelKind.zipformerTransducer,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
    bool realtime = true,
    String? decodingMethod,
    VadSensitivity? vadSensitivity,
    String? hotwordsFile,
    double hotwordsScore = 1.5,
    JaPunctuation? punctuation,
  }) {
    return _transcriber.startDebugWavStream(
      modelKind: modelKind,
      modelDir: modelDir,
      vadModelPath: vadModelPath,
      wavPath: wavPath,
      routingProfile: routingProfile,
      senseVoiceModelDir: senseVoiceModelDir,
      lidModelDir: lidModelDir,
      realtime: realtime,
      decodingMethod: decodingMethod,
      vadSensitivity: vadSensitivity,
      hotwordsFile: hotwordsFile,
      hotwordsScore: hotwordsScore,
      punctuation: punctuation,
    );
  }

  /// Stops an in-progress [startDebugWavStream] early, mid-file. A no-op if
  /// no debug stream is running.
  Future<void> stopDebugWavStream() => _transcriber.stopDebugWavStream();

  /// Releases everything, including the mic recorder. Call once when the
  /// owner is done with this instance. Idempotent: a second call is a
  /// no-op.
  Future<void> dispose() async {
    if (_disposed) {
      return;
    }
    _disposed = true;
    await _entriesSubscription.cancel();
    await _refineEntriesSubscription.cancel();
    await _decodingSubscription.cancel();
    await _draftsSubscription.cancel();
    await _errorsSubscription.cancel();
    await _modelLoadsSubscription.cancel();
    await _sessionResetsSubscription.cancel();
    await _transcriber.dispose();
    await _eventsController.close();
    await _decodingController.close();
  }
}
