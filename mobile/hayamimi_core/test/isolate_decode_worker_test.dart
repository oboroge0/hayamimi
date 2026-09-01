import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';

/// The real, isolate-backed worker — as far as a `flutter test` run can go
/// with it.
///
/// It cannot go far: the worker's first act is to bind the sherpa-onnx
/// native library, which no test VM can load. What that does cover is the
/// plumbing around the failure — the spawn, the error and exit ports, the
/// promise that a worker which never came up is not left running — which is
/// the part a fake worker cannot check.
void main() {
  test('a worker that cannot come up reports it and leaves nothing running', () async {
    final worker = IsolateDecodeWorker();

    await expectLater(
      worker.start(
        const DecodeWorkerConfig(
          routed: false,
          modelDir: '/definitely/does/not/exist/model',
        ),
      ),
      throwsA(isA<DecodeWorkerException>()),
    );

    expect(worker.isAlive, isFalse);

    // Tidying up after a failed start must stay a safe no-op rather than
    // throwing on the ports the failure already closed.
    await worker.shutdown();
    await worker.shutdown();
  });
}
