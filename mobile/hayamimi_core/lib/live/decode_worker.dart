/// The decode worker: a long-lived isolate that owns this session's
/// sherpa-onnx recognizer(s) and answers decode requests, plus the handle
/// the caller's isolate drives it through.
///
/// Why it exists: an `OfflineRecognizer.decode` call is synchronous FFI.
/// Made from the isolate that drives the mic stream — for a host app,
/// Flutter's UI isolate — it stops the app from painting or handling input
/// for as long as the decode takes, which is a sizeable fraction of a
/// second per utterance and much longer for a refine ("清書") pass over a
/// whole buffered group. Nothing about that work is faster here; it simply
/// happens somewhere the UI does not have to wait for it.
///
/// Why the worker *builds* the models rather than being handed them:
/// `native_model_loader.dart` hands raw pointers back from a short-lived
/// loading isolate because the VAD it builds has to live on the caller's
/// isolate afterwards. The recognizers here never leave the worker — it
/// creates them, decodes with them, and frees them — so there is no handoff
/// to justify. The worker calls `sherpa_onnx.initBindings()` itself, as
/// every isolate that touches the FFI bindings must.
///
/// The Japanese punctuation model lives here for the same reason. Its
/// `restore()` is another synchronous native call, tens of milliseconds per
/// refine on the Windows x86 host the parity test runs on (see the README's
/// punctuation section for the measurement and its conditions; no phone was
/// measured), so running it on the caller's isolate would hand back the
/// pause that moving decoding here removed.
///
/// Only refine ("清書") results are punctuated. That is where the desktop
/// pipeline punctuates, and it is the pass with the context to do it well:
/// a refine re-decodes a whole buffered group, while a final covers one
/// speech segment and a draft covers part of one still being spoken, so
/// sentence boundaries there fall wherever the speaker paused. Punctuating
/// every final would also add a full model run to every utterance, on a
/// phone, for a line the next refine replaces. Finals and drafts therefore
/// come back exactly as the recognizer produced them, with
/// [DecodeWorkerResult.punctuated] false.
///
/// Ownership, stated once: from `initBindings` to the [DecodeRequestKind.shutdown]
/// ack, exactly one isolate — this one — creates, uses and frees the
/// recognizer handles and the punctuation session. The caller's isolate
/// never holds a pointer to them.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../bench/model_file_resolver.dart';
import '../punct/punctuator_ja.dart';
import '../routing/routed_recognizer.dart';
import 'decode_protocol.dart';

/// How long [IsolateDecodeWorker.shutdown] waits for the worker's shutdown
/// ack before giving up on an orderly exit.
///
/// A worker that has not answered by then is stuck inside a decode that is
/// not going to return (or has wedged in native code), and there is nothing
/// useful left to wait for — the caller is typically on its way out of a
/// screen. See [IsolateDecodeWorker.shutdown] for what happens next.
const Duration defaultDecodeWorkerShutdownTimeout = Duration(seconds: 3);

/// The caller's-isolate handle on a decode worker.
///
/// Split out as an interface for one reason: the sherpa-onnx native
/// libraries cannot be loaded in a plain `flutter test` run, so without a
/// seam here nothing about the queueing, staleness and lifecycle rules
/// around decoding could be tested at all. A test supplies a stand-in that
/// answers requests from a script; production uses [IsolateDecodeWorker].
abstract class DecodeWorker {
  /// Everything the worker reports: model-load progress, results, per-request
  /// failures, acks, and — synthesized locally — [DecodeWorkerDied].
  ///
  /// [DecodeWorkerReady] and [DecodeWorkerBuildFailed] are consumed by
  /// [start] and are not republished here.
  Stream<DecodeWorkerMessage> get messages;

  /// False once the worker has exited or been killed. Requests sent to a
  /// dead worker go nowhere, so callers check this before queueing more.
  bool get isAlive;

  /// Spawns the worker and waits until every model is loaded.
  ///
  /// Model-load progress arrives on [messages] while this runs. Throws
  /// [DecodeWorkerException] if a model cannot be built, leaving nothing
  /// behind to clean up: the worker frees whatever it had already built and
  /// exits before the throw reaches the caller.
  Future<void> start(DecodeWorkerConfig config);

  /// Queues one request. Requests are served in the order they are sent.
  void send(DecodeWorkerCommand command);

  /// Frees the worker's native handles and stops it. Safe to call twice,
  /// and safe to call on a worker that already died.
  Future<void> shutdown();
}

/// The real worker: an [Isolate] running [decodeWorkerMain].
class IsolateDecodeWorker implements DecodeWorker {
  IsolateDecodeWorker({
    this.shutdownTimeout = defaultDecodeWorkerShutdownTimeout,
  });

  /// How long [shutdown] waits for the shutdown ack.
  final Duration shutdownTimeout;

  final StreamController<DecodeWorkerMessage> _messages =
      StreamController<DecodeWorkerMessage>.broadcast();

  Isolate? _isolate;
  ReceivePort? _fromWorker;
  ReceivePort? _errors;
  ReceivePort? _exits;
  SendPort? _toWorker;
  Completer<void>? _ready;
  Completer<void>? _shutdownAck;
  bool _alive = false;
  bool _stopping = false;
  // An isolate that dies of an uncaught error reports itself twice -- once
  // on the error port, then again on the exit port -- and a clean shutdown
  // reports on the exit port too, after its ack. This makes "the worker is
  // gone" a one-time event whichever way it arrives.
  bool _gone = false;

  @override
  Stream<DecodeWorkerMessage> get messages => _messages.stream;

  @override
  bool get isAlive => _alive;

  @override
  Future<void> start(DecodeWorkerConfig config) async {
    final fromWorker = ReceivePort('hayamimi:decode-worker');
    final errors = ReceivePort('hayamimi:decode-worker-errors');
    final exits = ReceivePort('hayamimi:decode-worker-exit');
    _fromWorker = fromWorker;
    _errors = errors;
    _exits = exits;
    final ready = Completer<void>();
    _ready = ready;

    fromWorker.listen(_onRaw);
    errors.listen((Object? error) {
      // onError delivers [errorString, stackTraceString].
      final described = error is List && error.isNotEmpty
          ? '${error.first}'
          : '$error';
      _onDeath(described);
    });
    exits.listen((Object? _) => _onDeath('the isolate exited'));

    try {
      _isolate = await Isolate.spawn<List<Object?>>(
        decodeWorkerMain,
        <Object?>[fromWorker.sendPort, config.toMessage()],
        onError: errors.sendPort,
        onExit: exits.sendPort,
        debugName: 'hayamimi:decode-worker',
      );
    } catch (e) {
      _closePorts();
      throw DecodeWorkerException('Could not start the decode worker: $e');
    }
    _alive = true;

    try {
      await ready.future;
    } catch (_) {
      // The worker already freed whatever it built and is on its way out;
      // kill it anyway so a build that failed *after* spawning can't leave
      // an isolate behind, then let the caller see the failure.
      await _kill();
      rethrow;
    } finally {
      _ready = null;
    }
  }

  @override
  void send(DecodeWorkerCommand command) {
    _toWorker?.send(command.toMessage());
  }

  /// Asks the worker to free its handles and exit, and waits up to
  /// [shutdownTimeout] for it to say it did.
  ///
  /// If that ack does not arrive, the isolate is killed and **nothing
  /// further is freed**. That is deliberate: the recognizer handles belong
  /// to the worker, and freeing a `Pointer` from another isolate while the
  /// worker might still be inside a decode on it is a use-after-free — a
  /// crash, not a leak. Losing the C-heap allocation of a session that was
  /// ending anyway is the cheaper of the two failures.
  @override
  Future<void> shutdown() async {
    if (_stopping) {
      return;
    }
    _stopping = true;
    final port = _toWorker;
    if (_alive && port != null) {
      final ack = Completer<void>();
      _shutdownAck = ack;
      port.send(
        const ControlRequest(
          id: 0,
          kind: DecodeRequestKind.shutdown,
        ).toMessage(),
      );
      try {
        await ack.future.timeout(shutdownTimeout);
      } on TimeoutException {
        // Fall through to the kill below.
      } finally {
        _shutdownAck = null;
      }
    }
    await _kill();
  }

  void _onRaw(Object? raw) {
    if (raw is List && raw.isNotEmpty && raw.first == 'port') {
      _toWorker = raw[1]! as SendPort;
      return;
    }
    final message = DecodeWorkerMessage.fromMessage(raw);
    switch (message) {
      case DecodeWorkerReady():
        _settle(_ready);
      case DecodeWorkerBuildFailed(:final message):
        _settleError(_ready, DecodeWorkerException(message));
      case DecodeWorkerAck(kind: DecodeRequestKind.shutdown):
        _alive = false;
        _settle(_shutdownAck);
      default:
        if (!_messages.isClosed) {
          _messages.add(message);
        }
    }
  }

  void _onDeath(String reason) {
    if (_gone) {
      return;
    }
    _gone = true;
    _alive = false;
    final ready = _ready;
    if (ready != null && !ready.isCompleted) {
      _settleError(
        ready,
        DecodeWorkerException(
          'The decode worker stopped while loading models ($reason).',
        ),
      );
      return;
    }
    _settle(_shutdownAck);
    if (!_stopping && !_messages.isClosed) {
      _messages.add(DecodeWorkerDied(reason));
    }
  }

  // Every completer here can be settled from two directions -- a frame the
  // worker sent, and the error/exit ports reporting that it is gone -- and
  // the two race. A build failure is the sharpest case: the worker sends
  // `build_failed` and then exits, so if the exit lands first, _onDeath has
  // already settled `_ready` by the time the frame arrives. Completing an
  // already-completed Completer throws a StateError, and inside a port
  // listener that surfaces as an unhandled async error rather than as
  // anything a caller could catch. So neither direction ever completes
  // blind.
  static void _settle(Completer<void>? completer) {
    if (completer != null && !completer.isCompleted) {
      completer.complete();
    }
  }

  static void _settleError(Completer<void>? completer, Object error) {
    if (completer != null && !completer.isCompleted) {
      completer.completeError(error);
    }
  }

  Future<void> _kill() async {
    _gone = true;
    _alive = false;
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _closePorts();
    if (!_messages.isClosed) {
      await _messages.close();
    }
  }

  void _closePorts() {
    _fromWorker?.close();
    _fromWorker = null;
    _errors?.close();
    _errors = null;
    _exits?.close();
    _exits = null;
    _toWorker = null;
  }
}

/// The worker isolate's entry point.
///
/// [bootstrap] is `[SendPort back to the caller, DecodeWorkerConfig.toMessage()]`.
/// The worker replies with its own command port, builds the models,
/// announces [DecodeWorkerReady] (or [DecodeWorkerBuildFailed]), then
/// serves requests one at a time until it is asked to shut down.
///
/// Serving strictly one at a time is the point: each decode is a
/// synchronous call and the loop below never awaits between requests, so
/// the mailbox is drained in order and no two decodes can overlap on the
/// same recognizer. That is exactly the mutual exclusion the old
/// synchronous, on-the-caller's-isolate code got for free.
Future<void> decodeWorkerMain(List<Object?> bootstrap) async {
  final toMain = bootstrap[0]! as SendPort;
  final config = DecodeWorkerConfig.fromMessage(bootstrap[1]);
  final commands = ReceivePort('hayamimi:decode-worker-commands');
  toMain.send(<Object?>['port', commands.sendPort]);

  sherpa_onnx.initBindings();

  sherpa_onnx.OfflineRecognizer? recognizer;
  RoutedRecognizerSet? routed;
  PunctuatorJa? punctuator;

  // The punctuation model loads *after* the recognizers, so a build that
  // gets that far has native handles to release before reporting the
  // failure -- the caller is told the session did not start, and nothing
  // may be left allocated in an isolate that is about to exit.
  void reportBuildFailure(String message) {
    recognizer?.free();
    routed?.free();
    toMain.send(DecodeWorkerBuildFailed(message).toMessage());
    commands.close();
  }

  try {
    if (config.routed) {
      final senseVoiceModelDir = config.senseVoiceModelDir;
      final lidModelDir = config.lidModelDir;
      if (senseVoiceModelDir == null || lidModelDir == null) {
        // Unreachable from LiveTranscriber, which checks this before
        // spawning anything. Said out loud anyway, because the alternative
        // for someone driving this worker directly is a build failure that
        // reads "Null check operator used on a null value".
        throw RoutedRecognizerException(
          'A routed decode worker needs both senseVoiceModelDir and '
          'lidModelDir.',
        );
      }
      routed = await RoutedRecognizerSet.build(
        reazonModelDir: config.modelDir,
        senseVoiceModelDir: senseVoiceModelDir,
        lidModelDir: lidModelDir,
        numThreads: config.numThreads,
        sampleRate: config.sampleRate,
        decodingMethod: config.decodingMethod ?? 'modified_beam_search',
        hotwordsFile: config.hotwordsFile,
        hotwordsScore: config.hotwordsScore,
        // This isolate *is* the background isolate: building in place skips
        // a pointless nested spawn and pointer handoff per model.
        loadOffIsolate: false,
        onModelLoad: (model, phase, ms) => toMain.send(
          DecodeWorkerModelLoad(model: model, phase: phase, ms: ms).toMessage(),
        ),
      );
    } else {
      recognizer = await _buildPlainRecognizer(config, toMain);
    }
    punctuator = await loadWorkerPunctuator(
      config,
      onModelLoad: (event) => toMain.send(event.toMessage()),
    );
  } on RoutedRecognizerException catch (e) {
    reportBuildFailure(e.message);
    return;
  } on ModelFileResolutionException catch (e) {
    reportBuildFailure(e.message);
    return;
  } on DecodeWorkerException catch (e) {
    // What loadWorkerPunctuator throws: its message already names the
    // punctuation model, so it travels as-is.
    reportBuildFailure(e.message);
    return;
  } catch (e) {
    reportBuildFailure('$e');
    return;
  }
  toMain.send(const DecodeWorkerReady().toMessage());

  await for (final raw in commands) {
    final command = DecodeWorkerCommand.fromMessage(raw);
    if (command.kind == DecodeRequestKind.shutdown) {
      recognizer?.free();
      routed?.free();
      punctuator?.dispose();
      toMain.send(
        DecodeWorkerAck(id: command.id, kind: command.kind).toMessage(),
      );
      break;
    }
    if (command.kind == DecodeRequestKind.resetSession) {
      routed?.reset();
      toMain.send(
        DecodeWorkerAck(id: command.id, kind: command.kind).toMessage(),
      );
      continue;
    }

    final request = command as DecodeRequest;
    final stopwatch = Stopwatch()..start();
    try {
      final (text, lang, switched) = _decode(
        request: request,
        config: config,
        recognizer: recognizer,
        routed: routed,
      );
      // Punctuation is part of producing this result, so it is inside the
      // timed section and inside the same try: a restore() that throws
      // loses this one refine the way a decode that throws does, and leaves
      // the session running.
      final punct = punctuator;
      var outText = text;
      var punctuated = false;
      if (punct != null &&
          request.kind == DecodeRequestKind.refine &&
          config.shouldPunctuateRefine(lang: lang)) {
        outText = punct.restore(text);
        punctuated = true;
      }
      stopwatch.stop();
      toMain.send(
        DecodeWorkerResult(
          id: request.id,
          kind: request.kind,
          segmentId: request.segmentId,
          text: outText,
          lang: lang,
          switched: switched,
          latencyMs: stopwatch.elapsedMicroseconds / 1000,
          punctuated: punctuated,
        ).toMessage(),
      );
    } catch (e) {
      // One request failed; the recognizer is still there and the queue
      // behind it still deserves to be served, so report just this one and
      // keep going.
      toMain.send(
        DecodeWorkerFailure(
          id: request.id,
          kind: request.kind,
          message: '$e',
        ).toMessage(),
      );
    }
  }

  commands.close();
}

/// Decodes one request with whichever recognizer this session has.
///
/// A draft goes through `decodeCurrentLangOnly` on a routed session — the
/// deliberately cheap path that skips language identification, because the
/// segment's own final will re-decide the language in a moment anyway. A
/// plain session has one recognizer and every kind of request goes through
/// it; building a second, lighter one just for drafts would double this
/// session's model memory for a line that is about to be replaced.
(String, String?, bool) _decode({
  required DecodeRequest request,
  required DecodeWorkerConfig config,
  required sherpa_onnx.OfflineRecognizer? recognizer,
  required RoutedRecognizerSet? routed,
}) {
  if (routed != null) {
    final result = request.kind == DecodeRequestKind.draft
        ? routed.decodeCurrentLangOnly(request.samples)
        : routed.decode(request.samples);
    return (result.text, result.lang, result.switched);
  }
  if (recognizer == null) {
    return ('', null, false);
  }
  final stream = recognizer.createStream();
  try {
    stream.acceptWaveform(
      samples: request.samples,
      sampleRate: config.sampleRate,
    );
    recognizer.decode(stream);
    return (recognizer.getResult(stream).text, null, false);
  } finally {
    stream.free();
  }
}

/// Builds the single-model (non-routed) recognizer, reporting its load the
/// same way the routed set reports each of its three.
Future<sherpa_onnx.OfflineRecognizer> _buildPlainRecognizer(
  DecodeWorkerConfig config,
  SendPort toMain,
) async {
  final dir = Directory(config.modelDir);
  final filenames = await dir
      .list()
      .where((e) => e is File)
      .map((e) => e.uri.pathSegments.last)
      .toList();
  final resolved = resolveZipformerTransducerFiles(filenames);

  final sep = Platform.pathSeparator;
  final modelDir = config.modelDir;
  toMain.send(
    const DecodeWorkerModelLoad(
      model: 'recognizer',
      phase: 'start',
    ).toMessage(),
  );
  final stopwatch = Stopwatch()..start();
  final recognizer = sherpa_onnx.OfflineRecognizer(
    sherpa_onnx.OfflineRecognizerConfig(
      model: sherpa_onnx.OfflineModelConfig(
        transducer: sherpa_onnx.OfflineTransducerModelConfig(
          encoder: '$modelDir$sep${resolved.encoder}',
          decoder: '$modelDir$sep${resolved.decoder}',
          joiner: '$modelDir$sep${resolved.joiner}',
        ),
        tokens: '$modelDir$sep${resolved.tokens}',
        numThreads: config.numThreads,
        debug: false,
        provider: 'cpu',
      ),
      decodingMethod: config.decodingMethod ?? 'greedy_search',
      hotwordsFile: config.hotwordsFile ?? '',
      hotwordsScore: config.hotwordsScore,
    ),
  );
  stopwatch.stop();
  toMain.send(
    DecodeWorkerModelLoad(
      model: 'recognizer',
      phase: 'done',
      ms: stopwatch.elapsedMicroseconds / 1000,
    ).toMessage(),
  );
  return recognizer;
}

/// Builds the Japanese punctuation model [config] asks for, or returns
/// `null` when it asks for none.
///
/// Loading it last is deliberate: the recognizers are what the session
/// cannot run without, and the punctuation model is 181.8 MB on top of
/// them, so a device that is going to run out of memory does so after the
/// part that matters has had its chance. Progress is reported as
/// `model: 'punct'` on the same `model_load` `start`/`done` events every
/// other model uses, with the measured elapsed milliseconds on `done`.
///
/// A failure here throws [DecodeWorkerException] with a message naming the
/// files, which the worker turns into a build failure and
/// `LiveTranscriber.start` rethrows: a missing 182 MB file on a phone is a
/// configuration mistake, and starting a session that silently produces
/// unpunctuated text would hide it.
///
/// Called by the worker isolate. It is public, and takes a callback rather
/// than the worker's `SendPort`, so a test can build exactly what a given
/// config would build without spawning an isolate — which is as close to
/// the real worker as a `flutter test` run can get, since the recognizers
/// beside it need native libraries no test VM can load.
Future<PunctuatorJa?> loadWorkerPunctuator(
  DecodeWorkerConfig config, {
  void Function(DecodeWorkerModelLoad event)? onModelLoad,
}) async {
  final modelPath = config.punctModelPath;
  final vocabPath = config.punctVocabPath;
  if (modelPath == null && vocabPath == null) {
    return null;
  }
  if (modelPath == null || vocabPath == null) {
    // Unreachable through JaPunctuation, which requires both, and caught by
    // an assert in DecodeWorkerConfig in debug builds. Said out loud for
    // anyone driving the worker directly in a release build.
    throw DecodeWorkerException(
      'The Japanese punctuation model needs both punctModelPath and '
      'punctVocabPath; only '
      '${modelPath == null ? 'the vocabulary' : 'the model'} was given.',
    );
  }

  onModelLoad?.call(
    const DecodeWorkerModelLoad(model: 'punct', phase: 'start'),
  );
  final stopwatch = Stopwatch()..start();
  final PunctuatorJa punctuator;
  try {
    punctuator = await PunctuatorJa.load(
      modelPath: modelPath,
      vocabPath: vocabPath,
      libraryPath: config.punctLibraryPath,
      intraOpNumThreads: config.punctNumThreads,
    );
  } catch (e) {
    throw DecodeWorkerException(
      'Could not load the Japanese punctuation model "$modelPath" '
      '(vocabulary "$vocabPath"): $e',
    );
  }
  stopwatch.stop();
  onModelLoad?.call(
    DecodeWorkerModelLoad(
      model: 'punct',
      phase: 'done',
      ms: stopwatch.elapsedMicroseconds / 1000,
    ),
  );
  return punctuator;
}
