/// The messages [LiveTranscriber] (on the caller's isolate) and its decode
/// worker isolate exchange, plus their wire encoding.
///
/// Why there is a protocol at all: every per-utterance decode used to be a
/// *synchronous* FFI call made on whichever isolate drives the mic stream —
/// for a host app, Flutter's UI isolate — so the app stopped painting for
/// the length of each decode. Moving the recognizers onto a long-lived
/// worker isolate means the two sides can no longer share objects, so
/// requests and results have to be described as data. That is what this
/// file is: plain, sendable value types with `toMessage()`/`fromMessage()`
/// pairs, and no FFI import anywhere, so the encoding is unit-testable
/// without the sherpa-onnx native libraries.
///
/// Audio travels as [TransferableTypedData]. A refine ("清書") request can
/// carry 60 seconds of 16 kHz `Float32` samples — about 960 thousand
/// floats, 3.8 MB — and a plain `SendPort.send` of a typed list copies it
/// into the receiving isolate's garbage-collected heap; the transferable
/// form moves it through external memory instead, so the samples are
/// materialized in the worker without a second copy and without adding
/// that much collector pressure on either side.
library;

import 'dart:isolate';
import 'dart:typed_data';

/// A failure raised while starting, or by, the decode worker.
///
/// `LiveTranscriber` re-wraps this as a `LiveTranscriberException` carrying
/// the same [message], so a model path that doesn't resolve produces the
/// exact same text a caller saw when loading happened in-process.
class DecodeWorkerException implements Exception {
  DecodeWorkerException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// What one request asks the decode worker to do.
enum DecodeRequestKind {
  /// A closed VAD segment. Always queued, never dropped, and emitted in
  /// the order the segments closed.
  finalSegment,

  /// The in-progress ("draft", 発話中の暫定字幕) window of a segment that
  /// hasn't closed yet. Dropped rather than queued while anything else is
  /// outstanding, and discarded on arrival if its segment has since closed.
  draft,

  /// The whole buffered group, re-decoded together (the "refine"/清書 pass).
  refine,

  /// Control: clear the worker's per-conversation state (for a routed
  /// session, which language it is locked to). Acked.
  resetSession,

  /// Control: free the native handles the worker owns, ack, then exit.
  shutdown,
}

/// Everything the worker needs to build its recognizer(s) itself.
///
/// The worker loads the models rather than being handed them: the objects
/// it builds never leave that isolate, so there is no pointer handoff to
/// justify (contrast `native_model_loader.dart`, which does hand pointers
/// back for the VAD that stays on the caller's isolate).
class DecodeWorkerConfig {
  const DecodeWorkerConfig({
    required this.routed,
    required this.modelDir,
    this.senseVoiceModelDir,
    this.lidModelDir,
    this.decodingMethod,
    this.hotwordsFile,
    this.hotwordsScore = 1.5,
    this.numThreads = 2,
    this.sampleRate = 16000,
  });

  /// Build the three-model `RoutedRecognizerSet` (ReazonSpeech ja +
  /// SenseVoice + whisper-tiny LID) rather than one plain recognizer —
  /// i.e. `RoutingProfile.jaSenseVoice`.
  final bool routed;

  /// The zipformer transducer directory; the ja tier when [routed].
  final String modelDir;
  final String? senseVoiceModelDir;
  final String? lidModelDir;

  /// sherpa-onnx search algorithm; `null` keeps each path's own default
  /// (`greedy_search` plain, `modified_beam_search` for the routed ja tier).
  final String? decodingMethod;

  final String? hotwordsFile;
  final double hotwordsScore;
  final int numThreads;
  final int sampleRate;

  List<Object?> toMessage() => <Object?>[
    routed,
    modelDir,
    senseVoiceModelDir,
    lidModelDir,
    decodingMethod,
    hotwordsFile,
    hotwordsScore,
    numThreads,
    sampleRate,
  ];

  static DecodeWorkerConfig fromMessage(Object? message) {
    final m = message! as List<Object?>;
    return DecodeWorkerConfig(
      routed: m[0]! as bool,
      modelDir: m[1]! as String,
      senseVoiceModelDir: m[2] as String?,
      lidModelDir: m[3] as String?,
      decodingMethod: m[4] as String?,
      hotwordsFile: m[5] as String?,
      hotwordsScore: m[6]! as double,
      numThreads: m[7]! as int,
      sampleRate: m[8]! as int,
    );
  }
}

/// A request travelling from the caller's isolate to the worker.
sealed class DecodeWorkerCommand {
  const DecodeWorkerCommand({required this.id, required this.kind});

  /// Monotonic per-session request id. Echoed back on the response so a
  /// reply belonging to a request that has since been abandoned can be
  /// recognized and dropped.
  final int id;

  final DecodeRequestKind kind;

  List<Object?> toMessage();

  static DecodeWorkerCommand fromMessage(Object? message) {
    final m = message! as List<Object?>;
    final kind = DecodeRequestKind.values.byName(m[1]! as String);
    switch (kind) {
      case DecodeRequestKind.finalSegment:
      case DecodeRequestKind.draft:
      case DecodeRequestKind.refine:
        final transferable = m[2]! as TransferableTypedData;
        return DecodeRequest(
          id: m[0]! as int,
          kind: kind,
          samples: transferable.materialize().asFloat32List(),
          segmentId: m[3]! as int,
        );
      case DecodeRequestKind.resetSession:
      case DecodeRequestKind.shutdown:
        return ControlRequest(id: m[0]! as int, kind: kind);
    }
  }
}

/// "Decode this audio" — the [DecodeRequestKind.finalSegment],
/// [DecodeRequestKind.draft] and [DecodeRequestKind.refine] requests.
class DecodeRequest extends DecodeWorkerCommand {
  const DecodeRequest({
    required super.id,
    required super.kind,
    required this.samples,
    this.segmentId = 0,
  });

  /// 16 kHz mono float samples.
  final Float32List samples;

  /// Which in-progress VAD segment a [DecodeRequestKind.draft] belongs to.
  /// The caller bumps its own counter whenever a segment opens or closes,
  /// so a draft result that comes back after its segment already produced a
  /// final can be recognized as stale and thrown away instead of
  /// overwriting that final on screen. Meaningless (and `0`) for the other
  /// kinds.
  final int segmentId;

  @override
  List<Object?> toMessage() => <Object?>[
    id,
    kind.name,
    TransferableTypedData.fromList(<TypedData>[samples]),
    segmentId,
  ];
}

/// "Do this, then tell me you did" — the [DecodeRequestKind.resetSession]
/// and [DecodeRequestKind.shutdown] requests. Both are answered with a
/// [DecodeWorkerAck] rather than a result.
class ControlRequest extends DecodeWorkerCommand {
  const ControlRequest({required super.id, required super.kind});

  @override
  List<Object?> toMessage() => <Object?>[id, kind.name];
}

/// Anything travelling from the worker back to the caller's isolate.
sealed class DecodeWorkerMessage {
  const DecodeWorkerMessage();

  List<Object?> toMessage();

  static DecodeWorkerMessage fromMessage(Object? message) {
    final m = message! as List<Object?>;
    switch (m[0]! as String) {
      case 'ready':
        return const DecodeWorkerReady();
      case 'build_failed':
        return DecodeWorkerBuildFailed(m[1]! as String);
      case 'model_load':
        return DecodeWorkerModelLoad(
          model: m[1]! as String,
          phase: m[2]! as String,
          ms: m[3] as double?,
        );
      case 'result':
        return DecodeWorkerResult(
          id: m[1]! as int,
          kind: DecodeRequestKind.values.byName(m[2]! as String),
          segmentId: m[3]! as int,
          text: m[4]! as String,
          lang: m[5] as String?,
          switched: m[6]! as bool,
          latencyMs: m[7]! as double,
        );
      case 'failure':
        return DecodeWorkerFailure(
          id: m[1]! as int,
          kind: DecodeRequestKind.values.byName(m[2]! as String),
          message: m[3]! as String,
        );
      case 'ack':
        return DecodeWorkerAck(
          id: m[1]! as int,
          kind: DecodeRequestKind.values.byName(m[2]! as String),
        );
      case 'died':
        return DecodeWorkerDied(m[1]! as String);
      default:
        throw ArgumentError.value(message, 'message', 'unknown worker message');
    }
  }
}

/// Every model is loaded and the worker is accepting requests. Consumed by
/// the worker handle itself (it is what resolves `DecodeWorker.start`), so
/// it never reaches `LiveTranscriber`.
class DecodeWorkerReady extends DecodeWorkerMessage {
  const DecodeWorkerReady();

  @override
  List<Object?> toMessage() => <Object?>['ready'];
}

/// Building a model failed inside the worker (a directory with no matching
/// files, a truncated onnx file). Carries the same text the in-process
/// loader used to throw, so `LiveTranscriber.start` can rethrow it
/// unchanged. Also consumed by the worker handle.
class DecodeWorkerBuildFailed extends DecodeWorkerMessage {
  const DecodeWorkerBuildFailed(this.message);

  final String message;

  @override
  List<Object?> toMessage() => <Object?>['build_failed', message];
}

/// One model's load start/finish, forwarded onto
/// `LiveTranscriber.modelLoads`. Same `model` names and measured `ms` the
/// in-process loader reported.
class DecodeWorkerModelLoad extends DecodeWorkerMessage {
  const DecodeWorkerModelLoad({
    required this.model,
    required this.phase,
    this.ms,
  });

  final String model;
  final String phase;
  final double? ms;

  @override
  List<Object?> toMessage() => <Object?>['model_load', model, phase, ms];
}

/// One finished decode.
class DecodeWorkerResult extends DecodeWorkerMessage {
  const DecodeWorkerResult({
    required this.id,
    required this.kind,
    required this.text,
    required this.latencyMs,
    this.segmentId = 0,
    this.lang,
    this.switched = false,
  });

  /// Echoes [DecodeWorkerCommand.id].
  final int id;

  final DecodeRequestKind kind;

  /// Echoes [DecodeRequest.segmentId] for drafts.
  final int segmentId;

  final String text;

  /// The language this segment was decoded with, or `null` for a plain
  /// (non-routed) session, which has only one model and no language to
  /// report.
  final String? lang;

  /// Whether this segment is what made a routed session change language.
  final bool switched;

  /// Wall-clock milliseconds the decode itself took inside the worker.
  /// Deliberately excludes the message round trip, so it stays the same
  /// quantity `LiveTranscriptEntry.latencyMs` reported when decoding
  /// happened in-process.
  final double latencyMs;

  @override
  List<Object?> toMessage() => <Object?>[
    'result',
    id,
    kind.name,
    segmentId,
    text,
    lang,
    switched,
    latencyMs,
  ];
}

/// One request threw inside the worker. The worker stays up and keeps
/// serving the rest of the queue; only this request is lost.
class DecodeWorkerFailure extends DecodeWorkerMessage {
  const DecodeWorkerFailure({
    required this.id,
    required this.kind,
    required this.message,
  });

  final int id;
  final DecodeRequestKind kind;
  final String message;

  @override
  List<Object?> toMessage() => <Object?>['failure', id, kind.name, message];
}

/// A [DecodeRequestKind.resetSession] or [DecodeRequestKind.shutdown]
/// finished.
class DecodeWorkerAck extends DecodeWorkerMessage {
  const DecodeWorkerAck({required this.id, required this.kind});

  final int id;
  final DecodeRequestKind kind;

  @override
  List<Object?> toMessage() => <Object?>['ack', id, kind.name];
}

/// The worker isolate died without being asked to (an uncaught error, or an
/// exit nobody requested). Synthesized on the caller's isolate from the
/// isolate's `onError`/`onExit` ports — the worker itself is in no position
/// to report this — and turned into a `LiveTranscriberException` on
/// `LiveTranscriber.errors`.
class DecodeWorkerDied extends DecodeWorkerMessage {
  const DecodeWorkerDied(this.message);

  final String message;

  @override
  List<Object?> toMessage() => <Object?>['died', message];
}
