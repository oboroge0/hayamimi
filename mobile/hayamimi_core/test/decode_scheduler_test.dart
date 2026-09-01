import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';

/// The queue policy in front of the decode worker: finals wait their turn,
/// drafts are dropped rather than queued, and refines fold together.
void main() {
  group('DecodeScheduler', () {
    test('starts idle', () {
      final scheduler = DecodeScheduler<String>();

      expect(scheduler.isBusy, isFalse);
      expect(scheduler.hasInFlight, isFalse);
      expect(scheduler.queueLength, 0);
      expect(scheduler.takeNext(), isNull);
    });

    test('finals queue and come back out in the order they were offered', () {
      final scheduler = DecodeScheduler<String>();

      for (final text in <String>['a', 'b', 'c']) {
        expect(
          scheduler.offer(DecodeRequestKind.finalSegment, text).admission,
          DecodeAdmission.queued,
        );
      }
      expect(scheduler.queueLength, 3);

      final order = <String>[];
      for (var i = 0; i < 3; i++) {
        order.add(scheduler.takeNext()!.payload);
        scheduler.finishInFlight();
      }

      expect(order, <String>['a', 'b', 'c']);
      expect(scheduler.isBusy, isFalse);
    });

    test('only one request is in flight at a time', () {
      final scheduler = DecodeScheduler<String>();
      scheduler.offer(DecodeRequestKind.finalSegment, 'a');
      scheduler.offer(DecodeRequestKind.finalSegment, 'b');

      expect(scheduler.takeNext()!.payload, 'a');
      // Nothing else goes out until the one in flight is finished.
      expect(scheduler.takeNext(), isNull);

      scheduler.finishInFlight();
      expect(scheduler.takeNext()!.payload, 'b');
    });

    test('a draft offered while idle is accepted', () {
      final scheduler = DecodeScheduler<String>();

      expect(
        scheduler.offer(DecodeRequestKind.draft, 'draft').admission,
        DecodeAdmission.queued,
      );
    });

    test('a draft offered while anything is outstanding is dropped', () {
      final scheduler = DecodeScheduler<String>();

      // Queued but not yet sent still counts: a draft behind a final would
      // only delay the final it is about to be replaced by.
      scheduler.offer(DecodeRequestKind.finalSegment, 'final');
      expect(
        scheduler.offer(DecodeRequestKind.draft, 'draft').admission,
        DecodeAdmission.skipped,
      );

      scheduler.takeNext();
      expect(
        scheduler.offer(DecodeRequestKind.draft, 'draft').admission,
        DecodeAdmission.skipped,
      );
      expect(scheduler.queueLength, 0);
    });

    test('a refine offered while another is queued folds into it', () {
      final scheduler = DecodeScheduler<String>();
      scheduler.offer(DecodeRequestKind.finalSegment, 'final');
      scheduler.takeNext();

      expect(
        scheduler.offer(DecodeRequestKind.refine, 'first').admission,
        DecodeAdmission.queued,
      );
      final second = scheduler.offer(DecodeRequestKind.refine, 'second');

      expect(second.admission, DecodeAdmission.coalesced);
      expect(second.existing, 'first');
      expect(scheduler.queueLength, 1);
    });

    test('a refine offered while one is already in flight is queued, not folded', () {
      final scheduler = DecodeScheduler<String>();
      scheduler.offer(DecodeRequestKind.refine, 'first');
      scheduler.takeNext();

      // The first pass has already claimed its audio, so a second pass has
      // real work to do over whatever has been buffered since.
      expect(
        scheduler.offer(DecodeRequestKind.refine, 'second').admission,
        DecodeAdmission.queued,
      );
    });

    test('resets queue behind outstanding work like anything else', () {
      final scheduler = DecodeScheduler<String>();
      scheduler.offer(DecodeRequestKind.finalSegment, 'final');
      scheduler.offer(DecodeRequestKind.resetSession, 'reset');

      expect(scheduler.takeNext()!.kind, DecodeRequestKind.finalSegment);
      scheduler.finishInFlight();
      expect(scheduler.takeNext()!.kind, DecodeRequestKind.resetSession);
    });

    test('isBusy is true from the first offer until the queue drains', () {
      final scheduler = DecodeScheduler<String>();
      expect(scheduler.isBusy, isFalse);

      scheduler.offer(DecodeRequestKind.finalSegment, 'a');
      expect(scheduler.isBusy, isTrue);

      scheduler.takeNext();
      expect(scheduler.isBusy, isTrue);

      scheduler.finishInFlight();
      expect(scheduler.isBusy, isFalse);
    });

    test('clear drops the in-flight request first, then the queue', () {
      final scheduler = DecodeScheduler<String>();
      scheduler.offer(DecodeRequestKind.finalSegment, 'a');
      scheduler.offer(DecodeRequestKind.finalSegment, 'b');
      scheduler.offer(DecodeRequestKind.refine, 'c');
      scheduler.takeNext();

      final dropped = scheduler.clear();

      expect(dropped.map((s) => s.payload), <String>['a', 'b', 'c']);
      expect(scheduler.isBusy, isFalse);
      expect(scheduler.hasInFlight, isFalse);
    });
  });

  group('isDraftResponseStale', () {
    test('a draft for the segment still in progress is kept', () {
      expect(
        isDraftResponseStale(responseSegmentId: 4, currentSegmentId: 4),
        isFalse,
      );
    });

    test('a draft whose segment has since ended is stale', () {
      expect(
        isDraftResponseStale(responseSegmentId: 4, currentSegmentId: 5),
        isTrue,
      );
    });
  });
}
