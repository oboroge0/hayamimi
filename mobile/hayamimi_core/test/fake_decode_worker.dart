import 'dart:async';

import 'package:hayamimi_core/hayamimi_core.dart';

/// A scripted stand-in for the decode worker isolate.
///
/// The real worker loads sherpa-onnx models over FFI, which a plain
/// `flutter test` run cannot do at all. This one records what it was asked
/// and answers only when the test says to, which is what makes the
/// interesting behaviour — ordering, dropping, coalescing, staleness,
/// what happens when the worker dies — observable without a device.
class FakeDecodeWorker implements DecodeWorker {
  /// When set, [start] reports this as a model-build failure instead of
  /// coming up.
  String? buildFailure;

  /// Emitted (in order) while [start] runs, standing in for the real
  /// worker's per-model load progress.
  final List<DecodeWorkerModelLoad> modelLoads = <DecodeWorkerModelLoad>[];

  /// Every command this worker was sent, oldest first.
  final List<DecodeWorkerCommand> commands = <DecodeWorkerCommand>[];

  /// The config [start] was called with, or null if it never was.
  DecodeWorkerConfig? config;

  int startCount = 0;
  int shutdownCount = 0;

  /// When set, every request is answered with this text on the next turn
  /// of the event loop, and every control request is acked. For tests that
  /// care about what came out rather than about when.
  String? autoReplyText;

  /// Whether [shutdown] closes the reply stream, as the real worker's does.
  /// Set false to keep pushing frames at a session that has already
  /// stopped, which is how "a reply that arrives too late" is staged.
  bool closeOnShutdown = true;

  final StreamController<DecodeWorkerMessage> _controller =
      StreamController<DecodeWorkerMessage>.broadcast();
  bool _alive = false;

  @override
  Stream<DecodeWorkerMessage> get messages => _controller.stream;

  @override
  bool get isAlive => _alive;

  @override
  Future<void> start(DecodeWorkerConfig config) async {
    startCount++;
    this.config = config;
    for (final event in modelLoads) {
      _emit(event);
    }
    // One turn of the event loop, so a listener sees the load events
    // before this future resolves — the real worker's arrive the same way.
    await Future<void>.delayed(Duration.zero);
    final failure = buildFailure;
    if (failure != null) {
      throw DecodeWorkerException(failure);
    }
    _alive = true;
  }

  @override
  void send(DecodeWorkerCommand command) {
    commands.add(command);
    final text = autoReplyText;
    if (text != null) {
      Future<void>.delayed(Duration.zero, () => _autoAnswer(command, text));
    }
  }

  void _autoAnswer(DecodeWorkerCommand command, String text) {
    if (command is DecodeRequest) {
      replyTo(command, text);
    } else {
      _emit(DecodeWorkerAck(id: command.id, kind: command.kind));
    }
  }

  @override
  Future<void> shutdown() async {
    shutdownCount++;
    _alive = false;
    if (closeOnShutdown && !_controller.isClosed) {
      await _controller.close();
    }
  }

  /// The most recent command, as an audio request.
  DecodeRequest get lastRequest => commands.last as DecodeRequest;

  /// Every audio request this worker was sent.
  List<DecodeRequest> get decodeRequests =>
      commands.whereType<DecodeRequest>().toList();

  /// Answers the outstanding request with a result.
  void reply(String text, {String? lang, bool switched = false}) =>
      replyTo(lastRequest, text, lang: lang, switched: switched);

  /// Answers a specific request -- including one that was answered already,
  /// which is how a reply arriving after its session ended is staged.
  void replyTo(
    DecodeRequest request,
    String text, {
    String? lang,
    bool switched = false,
  }) {
    _emit(
      DecodeWorkerResult(
        id: request.id,
        kind: request.kind,
        segmentId: request.segmentId,
        text: text,
        lang: lang,
        switched: switched,
        latencyMs: 12.5,
      ),
    );
  }

  /// Answers the outstanding request with a failure.
  void failRequest(String message) {
    final request = commands.last;
    _emit(
      DecodeWorkerFailure(
        id: request.id,
        kind: request.kind,
        message: message,
      ),
    );
  }

  /// Answers every request this worker has ever been sent. Only the one
  /// actually outstanding takes effect -- the rest no longer match an id
  /// the session is waiting on -- so this is a safe way to let a session
  /// drain on its way out of a test.
  void answerAll([String text = 'teardown']) {
    for (final command in List<DecodeWorkerCommand>.of(commands)) {
      _autoAnswer(command, text);
    }
  }

  /// Acknowledges the outstanding control request.
  void ack() {
    final request = commands.last;
    _emit(DecodeWorkerAck(id: request.id, kind: request.kind));
  }

  /// Reports one model's load progress after the session is already up.
  void emitModelLoad(DecodeWorkerModelLoad event) => _emit(event);

  /// The isolate went away without being asked to.
  void die(String message) {
    _alive = false;
    _emit(DecodeWorkerDied(message));
  }

  void _emit(DecodeWorkerMessage message) {
    if (!_controller.isClosed) {
      _controller.add(message);
    }
  }
}

/// Lets queued stream events be delivered. Every worker reply travels over
/// a stream, so a test has to give the event loop a turn before asserting
/// on what the reply caused.
Future<void> settle() => Future<void>.delayed(Duration.zero);
