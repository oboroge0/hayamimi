/// Pure queue policy for the decode worker: which requests wait, which are
/// dropped, and which fold into one another.
///
/// Why this needs a policy at all: decoding used to be a synchronous FFI
/// call, and being synchronous *was* the mutual exclusion — one decode ran
/// to completion before the next line of code executed, so "is a decode
/// running?" was answered by a plain `bool` set around the call. With the
/// decode on a worker isolate, a request is outstanding across an
/// arbitrary number of mic frames, and the caller has to decide explicitly
/// what to do with everything that arrives meanwhile. The three answers
/// differ per kind:
///
///  * A **final** is the transcript. Never drop one, and emit them in the
///    order the segments closed — so finals queue.
///  * A **draft** is a provisional line that the segment's own final will
///    replace in a moment. Queueing one only delays the final it is about
///    to be superseded by, so a draft offered while anything is
///    outstanding is dropped instead (the same rule `isDraftDue`'s
///    `isDecoding` guard has always applied, now against a queue rather
///    than a single flag).
///  * A **refine** re-decodes everything buffered. Two of them back to back
///    would decode overlapping audio twice for one result, so a refine
///    offered while another is already waiting folds into that one.
///
/// Everything here is plain data with no FFI and no isolate dependency, so
/// the policy is unit tested directly; the glue that turns a decision into
/// an actual `SendPort.send` lives in `decode_session.dart`.
library;

import 'dart:collection';

import 'decode_protocol.dart';

/// What [DecodeScheduler.offer] did with a request.
enum DecodeAdmission {
  /// Accepted and waiting its turn.
  queued,

  /// Dropped without being sent: a draft offered while something else was
  /// already outstanding.
  skipped,

  /// Folded into a refine that is already waiting. The caller should
  /// attach to that one (see [DecodeAdmissionResult.existing]) rather than
  /// starting a second pass.
  coalesced,
}

/// One accepted request: what to ask for, and the caller's own bookkeeping
/// for it.
class DecodeSlot<T> {
  const DecodeSlot({required this.kind, required this.payload});

  final DecodeRequestKind kind;

  /// Whatever the caller needs when the response arrives — for a final,
  /// the segment's audio so it can go into the refine buffer; for a draft,
  /// the segment id to check staleness against; for a refine or a reset,
  /// the `Completer` whose future the caller handed out.
  final T payload;
}

/// The outcome of [DecodeScheduler.offer].
class DecodeAdmissionResult<T> {
  const DecodeAdmissionResult(this.admission, {this.existing});

  final DecodeAdmission admission;

  /// Set only for [DecodeAdmission.coalesced]: the payload of the queued
  /// refine this offer folded into.
  final T? existing;
}

/// Decides what happens to each offered request, and hands them back one at
/// a time.
///
/// Exactly one request is outstanding at the worker at a time. The worker
/// is single-threaded and serves its mailbox first-in-first-out, so it
/// would serialize a burst of requests by itself — but holding the rest on
/// this side buys two things worth one round trip of idle worker time per
/// decode: a refine can read the buffer at the moment it is actually sent
/// rather than the moment it was asked for (so a final still in flight when
/// the user taps 清書 still makes it into that refine), and requests
/// belonging to a session that has since stopped can be dropped before they
/// are ever sent.
class DecodeScheduler<T> {
  final Queue<DecodeSlot<T>> _queue = Queue<DecodeSlot<T>>();
  DecodeSlot<T>? _inFlight;

  /// The request currently outstanding at the worker, if any.
  DecodeSlot<T>? get inFlight => _inFlight;

  bool get hasInFlight => _inFlight != null;

  /// Whether anything is outstanding — in flight or still queued. This is
  /// what `LiveTranscriber.decoding` reports and what the draft and
  /// VAD-swap guards read: true from the moment a request is accepted
  /// until the queue drains.
  bool get isBusy => _inFlight != null || _queue.isNotEmpty;

  int get queueLength => _queue.length;

  /// Offers a request of [kind], carrying [payload].
  DecodeAdmissionResult<T> offer(DecodeRequestKind kind, T payload) {
    if (kind == DecodeRequestKind.draft && isBusy) {
      return const DecodeAdmissionResult(DecodeAdmission.skipped);
    }
    if (kind == DecodeRequestKind.refine) {
      for (final slot in _queue) {
        if (slot.kind == DecodeRequestKind.refine) {
          return DecodeAdmissionResult(
            DecodeAdmission.coalesced,
            existing: slot.payload,
          );
        }
      }
    }
    _queue.add(DecodeSlot<T>(kind: kind, payload: payload));
    return const DecodeAdmissionResult(DecodeAdmission.queued);
  }

  /// Promotes the next queued request to in-flight and returns it, or
  /// `null` when something is already in flight or the queue is empty.
  DecodeSlot<T>? takeNext() {
    if (_inFlight != null || _queue.isEmpty) {
      return null;
    }
    return _inFlight = _queue.removeFirst();
  }

  /// Marks the in-flight request finished (its response arrived, or the
  /// caller decided not to send it after all) and returns it.
  DecodeSlot<T>? finishInFlight() {
    final slot = _inFlight;
    _inFlight = null;
    return slot;
  }

  /// Drops everything — the in-flight request and the whole queue — and
  /// returns what was dropped, in-flight first. Used when the session ends
  /// or the worker dies, so the caller can settle any futures it handed
  /// out for those requests.
  List<DecodeSlot<T>> clear() {
    final dropped = <DecodeSlot<T>>[
      if (_inFlight != null) _inFlight!,
      ..._queue,
    ];
    _inFlight = null;
    _queue.clear();
    return dropped;
  }
}

/// Whether a draft result that just came back is about a segment that has
/// since ended (or been abandoned), and so must be thrown away rather than
/// shown.
///
/// A draft is a guess at a segment that is still being spoken. By the time
/// its result crosses back from the worker, the speaker may already have
/// paused: the segment closed, its properly routed final was decoded, and
/// the UI is showing that final. Publishing the older, coarser draft on top
/// of it would visibly replace a finished line with a worse one. The caller
/// bumps a counter every time a segment opens or closes and stamps each
/// draft request with the value at the time; anything that comes back
/// stamped with an older value is stale.
bool isDraftResponseStale({
  required int responseSegmentId,
  required int currentSegmentId,
}) {
  return responseSegmentId != currentSegmentId;
}
