import 'dart:async';

import 'bench/model_kind.dart';
import 'live/live_transcriber.dart';
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
  HayamimiLive({LiveTranscriber? transcriber, this.textTransform})
    : _transcriber = transcriber ?? LiveTranscriber() {
    _entriesSubscription = _transcriber.entries.listen((entry) {
      final entryLang = entry.lang ?? lang;
      _eventsController.add(
        FinalSubtitleEvent(
          text: _transform(entry.text, entryLang),
          lang: entryLang,
          latencyMs: entry.latencyMs,
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
  }

  final LiveTranscriber _transcriber;
  final _eventsController = StreamController<SubtitleEvent>.broadcast();
  final _decodingController = StreamController<bool>.broadcast();
  late final StreamSubscription _entriesSubscription;
  late final StreamSubscription _refineEntriesSubscription;
  late final StreamSubscription _decodingSubscription;
  late final StreamSubscription _draftsSubscription;
  late final StreamSubscription _errorsSubscription;
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
  /// picks up the new transform. `null` (the default) is a no-op. This is
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
  /// ([PartialSubtitleEvent]), and session failures
  /// ([ErrorSubtitleEvent] — e.g. the OS killing mic capture mid-session,
  /// after which [isRunning] is false again), in arrival order.
  Stream<SubtitleEvent> get events => _eventsController.stream;

  /// Emits `true` right before a segment starts decoding and `false` right
  /// after — useful for a busy indicator.
  Stream<bool> get decoding => _decodingController.stream;

  bool get isRunning => _transcriber.isRunning;

  /// Total audio currently buffered for the next refine pass, in seconds.
  double get refineBufferedSeconds => _transcriber.refineBufferedSeconds;

  /// Whether silence-triggered auto-refine is on. Off by default: every
  /// refine pass is a full re-decode, which costs battery and heat on a
  /// phone. [refineNow] always works regardless of this setting.
  bool get autoRefineEnabled => _transcriber.autoRefineEnabled;
  set autoRefineEnabled(bool value) => _transcriber.autoRefineEnabled = value;

  /// Starts capturing mic audio and transcribing it.
  ///
  /// [modelDir] must contain a zipformer transducer model.
  /// [vadModelPath] must point at a Silero VAD onnx model file (e.g.
  /// `silero_vad.onnx`).
  Future<void> start({
    required String modelDir,
    required String vadModelPath,
    ModelKind modelKind = ModelKind.zipformerTransducer,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
  }) {
    return _transcriber.start(
      modelKind: modelKind,
      modelDir: modelDir,
      vadModelPath: vadModelPath,
      routingProfile: routingProfile,
      senseVoiceModelDir: senseVoiceModelDir,
      lidModelDir: lidModelDir,
    );
  }

  /// Stops capturing and releases native VAD/recognizer resources.
  Future<void> stop() => _transcriber.stop();

  /// Runs a refine ("清書") pass over everything buffered since the last
  /// refine (or session start): re-decodes it as one utterance and emits a
  /// [RefineSubtitleEvent] on [events]. Safe to call at any time; emits
  /// nothing if there's too little buffered audio.
  Future<void> refineNow() => _transcriber.refineNow();

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
  /// has been flushed and decoded.
  Future<void> startDebugWavStream({
    required String modelDir,
    required String vadModelPath,
    required String wavPath,
    ModelKind modelKind = ModelKind.zipformerTransducer,
    RoutingProfile routingProfile = RoutingProfile.jaOnly,
    String? senseVoiceModelDir,
    String? lidModelDir,
    bool realtime = true,
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
    await _transcriber.dispose();
    await _eventsController.close();
    await _decodingController.close();
  }
}
