/// The caller's-isolate half of the decode worker: it turns "decode this"
/// into a worker request, applies the queue policy in
/// `decode_scheduler.dart`, and hands each answer back to whoever asked.
///
/// This sits between `LiveTranscriber` (which owns the mic, the VAD and the
/// transcript streams) and [DecodeWorker] (which owns the recognizers).
/// Keeping it separate is what makes the interesting part testable: it has
/// no FFI, no microphone and no platform channel, so a test can drive a
/// whole session — finals in order, drafts dropped and discarded, refines
/// folded together, a worker dying mid-session — against a scripted
/// stand-in worker in a plain `flutter test` run.
library;

import 'dart:async';
import 'dart:typed_data';

import 'decode_protocol.dart';
import 'decode_scheduler.dart';
import 'decode_worker.dart';

/// The audio a refine pass is about to decode, plus the fast per-segment
/// text it is trying to improve on.
///
/// Both are built at the moment the request is actually sent rather than
/// when the refine was asked for, so a final that was still in flight when
/// the user tapped 清書 is included in the group instead of being pushed
/// into the next one.
class RefineRequestPayload {
  const RefineRequestPayload({required this.samples, required this.fastText});

  /// The buffered segments' audio, joined.
  final Float32List samples;

  /// Those segments' fast finals, joined — the fallback if the merged
  /// re-decode comes back suspiciously short.
  final String fastText;
}

sealed class _Pending {
  const _Pending();
}

final class _PendingFinal extends _Pending {
  const _PendingFinal(this.samples);

  final Float32List samples;
}

final class _PendingDraft extends _Pending {
  const _PendingDraft(this.samples, this.segmentId);

  final Float32List samples;
  final int segmentId;
}

final class _PendingRefine extends _Pending {
  final Completer<void> completer = Completer<void>();
  RefineRequestPayload? payload;
}

final class _PendingReset extends _Pending {
  final Completer<void> completer = Completer<void>();
}

/// Drives one decode worker for the life of one transcription session.
class DecodeSession {
  DecodeSession._(
    this._worker, {
    required this.buildRefinePayload,
    required this.onFinal,
    required this.onDraft,
    required this.onRefine,
    required this.onReset,
    required this.onModelLoad,
    required this.onBusyChanged,
    required this.onDecodeError,
    required this.onWorkerDied,
  });

  /// Spawns [worker], waits for its models to load, and returns the session
  /// wrapped around it.
  ///
  /// Throws [DecodeWorkerException] if a model cannot be built, with the
  /// worker already cleaned up — nothing is left running and no native
  /// handle is left allocated.
  ///
  /// [buildRefinePayload] is called on the caller's isolate at the moment a
  /// refine is about to be sent; returning `null` cancels that refine
  /// (nothing buffered, or too little audio to be worth a pass) and the
  /// future handed out for it completes without a result.
  static Future<DecodeSession> start({
    required DecodeWorker worker,
    required DecodeWorkerConfig config,
    required RefineRequestPayload? Function() buildRefinePayload,
    required void Function(Float32List samples, DecodeWorkerResult result)
    onFinal,
    required void Function(DecodeWorkerResult result) onDraft,
    required void Function(RefineRequestPayload payload, DecodeWorkerResult result)
    onRefine,
    required void Function() onReset,
    required void Function(String model, String phase, double? ms) onModelLoad,
    required void Function(bool busy) onBusyChanged,
    required void Function(String message) onDecodeError,
    required void Function(String message) onWorkerDied,
  }) async {
    final session = DecodeSession._(
      worker,
      buildRefinePayload: buildRefinePayload,
      onFinal: onFinal,
      onDraft: onDraft,
      onRefine: onRefine,
      onReset: onReset,
      onModelLoad: onModelLoad,
      onBusyChanged: onBusyChanged,
      onDecodeError: onDecodeError,
      onWorkerDied: onWorkerDied,
    );
    // Subscribe before start(), so the model-load events that arrive while
    // the models are still loading are the caller's to show.
    session._messages = worker.messages.listen(session._onMessage);
    try {
      await worker.start(config);
    } catch (_) {
      await session._messages?.cancel();
      session._messages = null;
      // A [DecodeWorker] is supposed to clean up after a failed start on
      // its own; asking anyway costs nothing (shutdown is documented safe
      // to call twice, and on a worker that already died) and makes "this
      // never leaves a worker running" true of this method rather than of
      // every implementation of that one.
      await worker.shutdown();
      rethrow;
    }
    return session;
  }

  final DecodeWorker _worker;
  final RefineRequestPayload? Function() buildRefinePayload;
  final void Function(Float32List samples, DecodeWorkerResult result) onFinal;
  final void Function(DecodeWorkerResult result) onDraft;
  final void Function(RefineRequestPayload payload, DecodeWorkerResult result)
  onRefine;
  final void Function() onReset;
  final void Function(String model, String phase, double? ms) onModelLoad;
  final void Function(bool busy) onBusyChanged;
  final void Function(String message) onDecodeError;
  final void Function(String message) onWorkerDied;

  final DecodeScheduler<_Pending> _scheduler = DecodeScheduler<_Pending>();
  StreamSubscription<DecodeWorkerMessage>? _messages;
  int _nextRequestId = 1;
  int _inFlightId = 0;
  int _draftSegmentId = 0;
  String? _currentLang;
  bool _lastBusy = false;
  bool _closed = false;
  Completer<void>? _idle;

  /// Whether any request is outstanding — in flight or queued.
  bool get isBusy => _scheduler.isBusy;

  /// The routed session's current language, mirrored from the worker (which
  /// is where the real state lives). `null` for a plain single-model
  /// session, before a routed session's first segment resolves one, and
  /// after [reset].
  String? get currentLang => _currentLang;

  /// The id every draft request is stamped with right now.
  int get draftSegmentId => _draftSegmentId;

  /// Call whenever a VAD segment opens or closes: any draft still in flight
  /// for the previous segment becomes stale and is discarded when it comes
  /// back.
  void beginDraftSegment() {
    _draftSegmentId++;
  }

  /// Queues a closed segment for decoding. Never dropped.
  void submitFinal(Float32List samples) {
    if (_closed) {
      return;
    }
    _scheduler.offer(DecodeRequestKind.finalSegment, _PendingFinal(samples));
    _pump();
  }

  /// Offers an in-progress window for a draft decode. Returns false if it
  /// was dropped because something else is already outstanding.
  bool submitDraft(Float32List samples) {
    if (_closed) {
      return false;
    }
    final result = _scheduler.offer(
      DecodeRequestKind.draft,
      _PendingDraft(samples, _draftSegmentId),
    );
    if (result.admission == DecodeAdmission.skipped) {
      return false;
    }
    _pump();
    return true;
  }

  /// Runs a refine pass, completing when its result has been handed to
  /// [onRefine] (or when there was nothing to refine).
  ///
  /// Calling this again while a refine is still waiting its turn returns
  /// *that* refine's future rather than starting a second one: the two
  /// would decode overlapping audio for a single visible result. A refine
  /// that has already been sent is past coalescing, so a call made then
  /// does queue a second pass over whatever has been buffered since.
  Future<void> refine() {
    if (_closed || !_worker.isAlive) {
      return Future<void>.value();
    }
    final pending = _PendingRefine();
    final result = _scheduler.offer(DecodeRequestKind.refine, pending);
    if (result.admission == DecodeAdmission.coalesced) {
      return (result.existing! as _PendingRefine).completer.future;
    }
    _pump();
    return pending.completer.future;
  }

  /// Clears the worker's per-conversation state, completing once the worker
  /// has acknowledged it. Queued behind anything already outstanding, so a
  /// final still decoding when this was called still lands before the
  /// clear rather than after it.
  Future<void> reset() {
    if (_closed || !_worker.isAlive) {
      return Future<void>.value();
    }
    final pending = _PendingReset();
    _scheduler.offer(DecodeRequestKind.resetSession, pending);
    _pump();
    return pending.completer.future;
  }

  /// Completes once nothing is outstanding.
  Future<void> waitForIdle() {
    if (!isBusy) {
      return Future<void>.value();
    }
    return (_idle ??= Completer<void>()).future;
  }

  /// Ends the session: abandons anything still queued, then shuts the
  /// worker down (which is what frees the native handles).
  Future<void> shutdown() async {
    if (_closed) {
      return;
    }
    _closed = true;
    _abandonQueued();
    _syncBusy();
    await _worker.shutdown();
    await _messages?.cancel();
    _messages = null;
  }

  void _pump() {
    if (!_closed && _worker.isAlive) {
      while (!_scheduler.hasInFlight) {
        final slot = _scheduler.takeNext();
        if (slot == null) {
          break;
        }
        if (_dispatch(slot)) {
          break;
        }
        // Nothing was actually sent (a refine with nothing to refine), so
        // free the slot and look at what is behind it.
        _scheduler.finishInFlight();
      }
    } else {
      _abandonQueued();
    }
    _syncBusy();
  }

  /// Sends [slot]'s request. Returns false when there turned out to be
  /// nothing to send.
  bool _dispatch(DecodeSlot<_Pending> slot) {
    final id = _nextRequestId++;
    switch (slot.payload) {
      case _PendingFinal(:final samples):
        _inFlightId = id;
        _worker.send(
          DecodeRequest(
            id: id,
            kind: DecodeRequestKind.finalSegment,
            samples: samples,
          ),
        );
        return true;
      case _PendingDraft(:final samples, :final segmentId):
        _inFlightId = id;
        _worker.send(
          DecodeRequest(
            id: id,
            kind: DecodeRequestKind.draft,
            samples: samples,
            segmentId: segmentId,
          ),
        );
        return true;
      case final _PendingRefine pending:
        final payload = buildRefinePayload();
        if (payload == null) {
          _complete(pending.completer);
          return false;
        }
        pending.payload = payload;
        _inFlightId = id;
        _worker.send(
          DecodeRequest(
            id: id,
            kind: DecodeRequestKind.refine,
            samples: payload.samples,
          ),
        );
        return true;
      case _PendingReset():
        _inFlightId = id;
        _worker.send(
          ControlRequest(id: id, kind: DecodeRequestKind.resetSession),
        );
        return true;
    }
  }

  void _onMessage(DecodeWorkerMessage message) {
    if (_closed) {
      return;
    }
    switch (message) {
      case DecodeWorkerModelLoad(:final model, :final phase, :final ms):
        onModelLoad(model, phase, ms);
      case final DecodeWorkerResult result:
        _onResult(result);
      case final DecodeWorkerFailure failure:
        _onFailure(failure);
      case final DecodeWorkerAck ack:
        _onAck(ack);
      case DecodeWorkerDied(:final message):
        _onDied(message);
      case DecodeWorkerReady():
      case DecodeWorkerBuildFailed():
        break;
    }
  }

  void _onResult(DecodeWorkerResult result) {
    final slot = _takeAnsweredSlot(result.id);
    if (slot == null) {
      return;
    }
    switch (slot.payload) {
      case _PendingFinal(:final samples):
        _currentLang = result.lang;
        onFinal(samples, result);
      case _PendingDraft(:final segmentId):
        if (!isDraftResponseStale(
          responseSegmentId: segmentId,
          currentSegmentId: _draftSegmentId,
        )) {
          onDraft(result);
        }
      case final _PendingRefine pending:
        _currentLang = result.lang;
        final payload = pending.payload;
        if (payload != null) {
          onRefine(payload, result);
        }
        _complete(pending.completer);
      case _PendingReset():
        break;
    }
    _pump();
  }

  void _onAck(DecodeWorkerAck ack) {
    final slot = _takeAnsweredSlot(ack.id);
    if (slot == null) {
      return;
    }
    if (slot.payload case final _PendingReset pending) {
      _currentLang = null;
      onReset();
      _complete(pending.completer);
    }
    _pump();
  }

  void _onFailure(DecodeWorkerFailure failure) {
    final slot = _takeAnsweredSlot(failure.id);
    if (slot == null) {
      return;
    }
    switch (slot.payload) {
      case final _PendingRefine pending:
        _complete(pending.completer);
      case final _PendingReset pending:
        _complete(pending.completer);
      case _PendingFinal():
      case _PendingDraft():
        break;
    }
    onDecodeError(failure.message);
    _pump();
  }

  /// Matches a response to the request that is actually outstanding, or
  /// `null` if it belongs to one that was abandoned in the meantime.
  DecodeSlot<_Pending>? _takeAnsweredSlot(int id) {
    if (id != _inFlightId) {
      return null;
    }
    final slot = _scheduler.finishInFlight();
    _inFlightId = 0;
    return slot;
  }

  void _onDied(String message) {
    _abandonQueued();
    _syncBusy();
    onWorkerDied(message);
  }

  /// Drops everything outstanding and settles the futures handed out for
  /// it. They complete normally rather than with an error: "the session
  /// ended before your refine ran" is the same non-event as "there was
  /// nothing buffered to refine", and a caller who never awaited the future
  /// should not get an unhandled error out of a session that simply
  /// stopped.
  void _abandonQueued() {
    _inFlightId = 0;
    for (final slot in _scheduler.clear()) {
      switch (slot.payload) {
        case final _PendingRefine pending:
          _complete(pending.completer);
        case final _PendingReset pending:
          _complete(pending.completer);
        case _PendingFinal():
        case _PendingDraft():
          break;
      }
    }
  }

  void _syncBusy() {
    final busy = _scheduler.isBusy;
    if (busy != _lastBusy) {
      _lastBusy = busy;
      onBusyChanged(busy);
    }
    if (!busy) {
      final idle = _idle;
      _idle = null;
      if (idle != null && !idle.isCompleted) {
        idle.complete();
      }
    }
  }

  void _complete(Completer<void> completer) {
    if (!completer.isCompleted) {
      completer.complete();
    }
  }
}
