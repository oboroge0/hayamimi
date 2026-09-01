import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_record_platform.dart';

/// A [RemoteTranscriber] whose `events` stream is injectable, so
/// `HayamimiRemote._onRawEvent`'s [RemoteEvent] -> [SubtitleEvent] mapping
/// can be exercised without a real WebSocket server -- same
/// fake-injection approach `hayamimi_live_events_test.dart` uses for
/// `LiveTranscriber`. `RemoteTranscriber()`'s own constructor touches
/// `AudioRecorder` like `LiveTranscriber`'s does, so this still needs
/// `FakeRecordPlatform` installed (see `lifecycle_test.dart`).
class _FakeRemoteTranscriber extends RemoteTranscriber {
  final _eventsCtrl = StreamController<RemoteEvent>.broadcast();

  @override
  Stream<RemoteEvent> get events => _eventsCtrl.stream;

  void emit(RemoteEvent event) => _eventsCtrl.add(event);

  @override
  Future<void> dispose() async {
    await _eventsCtrl.close();
    await super.dispose();
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late RecordPlatform originalPlatform;

  setUp(() {
    originalPlatform = RecordPlatform.instance;
    RecordPlatform.instance = FakeRecordPlatform();
  });

  tearDown(() {
    RecordPlatform.instance = originalPlatform;
  });

  test(
    'a RemoteFinalEvent\'s audioSeconds/switched propagate onto '
    'FinalSubtitleEvent',
    () async {
      final fake = _FakeRemoteTranscriber();
      final remote = HayamimiRemote(transcriber: fake);
      addTearDown(remote.dispose);

      final events = <SubtitleEvent>[];
      remote.events.listen(events.add);

      fake.emit(
        const RemoteFinalEvent(
          text: 'hello',
          lang: 'en',
          speaker: 'S1',
          latencyMs: 42.0,
          audioSeconds: 1.5,
          switched: true,
        ),
      );
      await Future<void>.delayed(Duration.zero);

      final finalEvent = events.whereType<FinalSubtitleEvent>().single;
      expect(finalEvent.text, 'hello');
      expect(finalEvent.lang, 'en');
      expect(finalEvent.speaker, 'S1');
      expect(finalEvent.latencyMs, 42.0);
      expect(finalEvent.audioSeconds, 1.5);
      expect(finalEvent.switched, isTrue);
    },
  );

  test(
    'a RemoteFinalEvent with no audioSeconds/switched keeps the defaults',
    () async {
      final fake = _FakeRemoteTranscriber();
      final remote = HayamimiRemote(transcriber: fake);
      addTearDown(remote.dispose);

      final events = <SubtitleEvent>[];
      remote.events.listen(events.add);

      fake.emit(const RemoteFinalEvent(text: 'plain'));
      await Future<void>.delayed(Duration.zero);

      final finalEvent = events.whereType<FinalSubtitleEvent>().single;
      expect(finalEvent.audioSeconds, isNull);
      expect(finalEvent.switched, isFalse);
    },
  );

  test(
    'a RemoteRefineEvent\'s latencyMs/audioSeconds propagate onto '
    'RefineSubtitleEvent',
    () async {
      final fake = _FakeRemoteTranscriber();
      final remote = HayamimiRemote(transcriber: fake);
      addTearDown(remote.dispose);

      final events = <SubtitleEvent>[];
      remote.events.listen(events.add);

      fake.emit(
        const RemoteRefineEvent(
          text: 'refined',
          lang: 'ja',
          speaker: 'S2',
          latencyMs: 200.0,
          audioSeconds: 8.0,
        ),
      );
      await Future<void>.delayed(Duration.zero);

      final refineEvent = events.whereType<RefineSubtitleEvent>().single;
      expect(refineEvent.text, 'refined');
      expect(refineEvent.latencyMs, 200.0);
      expect(refineEvent.audioSeconds, 8.0);
    },
  );

  test('a RemoteModelLoadEvent maps to ModelLoadSubtitleEvent', () async {
    final fake = _FakeRemoteTranscriber();
    final remote = HayamimiRemote(transcriber: fake);
    addTearDown(remote.dispose);

    final events = <SubtitleEvent>[];
    remote.events.listen(events.add);

    fake.emit(
      const RemoteModelLoadEvent(model: 'ja', phase: 'done', ms: 900.0),
    );
    await Future<void>.delayed(Duration.zero);

    final loadEvent = events.whereType<ModelLoadSubtitleEvent>().single;
    expect(loadEvent.model, 'ja');
    expect(loadEvent.phase, 'done');
    expect(loadEvent.ms, 900.0);
  });

  test('a RemoteSessionResetEvent maps to SessionResetSubtitleEvent', () async {
    final fake = _FakeRemoteTranscriber();
    final remote = HayamimiRemote(transcriber: fake);
    addTearDown(remote.dispose);

    final events = <SubtitleEvent>[];
    remote.events.listen(events.add);

    fake.emit(const RemoteSessionResetEvent());
    await Future<void>.delayed(Duration.zero);

    expect(events.whereType<SessionResetSubtitleEvent>(), hasLength(1));
  });

  test(
    'a RemoteUnknownEvent (e.g. model_fallback/warning/session_summary/'
    'recluster) is dropped from events, not surfaced as a SubtitleEvent',
    () async {
      final fake = _FakeRemoteTranscriber();
      final remote = HayamimiRemote(transcriber: fake);
      addTearDown(remote.dispose);

      final events = <SubtitleEvent>[];
      remote.events.listen(events.add);

      fake.emit(
        const RemoteUnknownEvent(
          type: 'model_fallback',
          raw: {'type': 'model_fallback'},
        ),
      );
      await Future<void>.delayed(Duration.zero);

      expect(events, isEmpty);
    },
  );
}
