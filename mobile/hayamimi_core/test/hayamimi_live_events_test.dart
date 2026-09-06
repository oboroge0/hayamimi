import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_record_platform.dart';

/// A [LiveTranscriber] whose entries/refineEntries/drafts/modelLoads/
/// sessionResets streams are injectable, so `HayamimiLive`'s event-mapping
/// logic (the new `audioSeconds`/`switched` fields, the `modelLoads` ->
/// [ModelLoadSubtitleEvent] and `sessionResets` -> [SessionResetSubtitleEvent]
/// wiring, and [HayamimiLive.resetSession] delegating to
/// [LiveTranscriber.resetSession]) can be exercised without a real mic or
/// native VAD/recognizer -- same fake-injection approach
/// `hayamimi_live_text_transform_test.dart` uses for the original three
/// streams, extended here to the two added by issue #29.
class _FakeLiveTranscriber extends LiveTranscriber {
  final _entriesCtrl = StreamController<LiveTranscriptEntry>.broadcast();
  final _refineCtrl = StreamController<LiveTranscriptEntry>.broadcast();
  final _draftCtrl = StreamController<LiveTranscriptEntry>.broadcast();
  final _modelLoadCtrl = StreamController<ModelLoadEvent>.broadcast();
  final _sessionResetCtrl = StreamController<void>.broadcast();

  bool resetSessionCalled = false;

  @override
  Stream<LiveTranscriptEntry> get entries => _entriesCtrl.stream;

  @override
  Stream<LiveTranscriptEntry> get refineEntries => _refineCtrl.stream;

  @override
  Stream<LiveTranscriptEntry> get drafts => _draftCtrl.stream;

  @override
  Stream<ModelLoadEvent> get modelLoads => _modelLoadCtrl.stream;

  @override
  Stream<void> get sessionResets => _sessionResetCtrl.stream;

  void emitFinal(LiveTranscriptEntry entry) => _entriesCtrl.add(entry);
  void emitRefine(LiveTranscriptEntry entry) => _refineCtrl.add(entry);
  void emitModelLoad(ModelLoadEvent event) => _modelLoadCtrl.add(event);

  @override
  Future<void> resetSession() async {
    // Mirrors the real method's contract (clear state, then emit on
    // sessionResets) without touching any native model -- there's nothing
    // native for this fake to clear.
    resetSessionCalled = true;
    _sessionResetCtrl.add(null);
  }

  @override
  Future<void> dispose() async {
    await _entriesCtrl.close();
    await _refineCtrl.close();
    await _draftCtrl.close();
    await _modelLoadCtrl.close();
    await _sessionResetCtrl.close();
    await super.dispose();
  }
}

LiveTranscriptEntry _entry(
  String text, {
  String? lang,
  double? audioSeconds,
  bool switched = false,
}) => LiveTranscriptEntry(
  text: text,
  timestamp: DateTime.now(),
  lang: lang,
  audioSeconds: audioSeconds,
  switched: switched,
);

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
    'a final entry\'s audioSeconds/switched propagate onto FinalSubtitleEvent',
    () async {
      final fake = _FakeLiveTranscriber();
      final live = HayamimiLive(transcriber: fake);
      addTearDown(live.dispose);

      final events = <SubtitleEvent>[];
      live.events.listen(events.add);

      fake.emitFinal(
        _entry('hello', lang: 'en', audioSeconds: 1.75, switched: true),
      );
      await Future<void>.delayed(Duration.zero);

      final finalEvent = events.whereType<FinalSubtitleEvent>().single;
      expect(finalEvent.audioSeconds, 1.75);
      expect(finalEvent.switched, isTrue);
    },
  );

  test(
    'a refine entry\'s audioSeconds propagates onto RefineSubtitleEvent',
    () async {
      final fake = _FakeLiveTranscriber();
      final live = HayamimiLive(transcriber: fake);
      addTearDown(live.dispose);

      final events = <SubtitleEvent>[];
      live.events.listen(events.add);

      fake.emitRefine(_entry('refined', lang: 'ja', audioSeconds: 8.0));
      await Future<void>.delayed(Duration.zero);

      final refineEvent = events.whereType<RefineSubtitleEvent>().single;
      expect(refineEvent.audioSeconds, 8.0);
    },
  );

  test('a final entry with no audioSeconds/switched keeps the defaults', () async {
    final fake = _FakeLiveTranscriber();
    final live = HayamimiLive(transcriber: fake);
    addTearDown(live.dispose);

    final events = <SubtitleEvent>[];
    live.events.listen(events.add);

    fake.emitFinal(_entry('plain'));
    await Future<void>.delayed(Duration.zero);

    final finalEvent = events.whereType<FinalSubtitleEvent>().single;
    expect(finalEvent.audioSeconds, isNull);
    expect(finalEvent.switched, isFalse);
  });

  test('modelLoads events surface as ModelLoadSubtitleEvent on events', () async {
    final fake = _FakeLiveTranscriber();
    final live = HayamimiLive(transcriber: fake);
    addTearDown(live.dispose);

    final events = <SubtitleEvent>[];
    live.events.listen(events.add);

    fake.emitModelLoad(const ModelLoadEvent(model: 'ja', phase: 'start'));
    fake.emitModelLoad(
      const ModelLoadEvent(model: 'ja', phase: 'done', ms: 123.4),
    );
    await Future<void>.delayed(Duration.zero);

    final loadEvents = events.whereType<ModelLoadSubtitleEvent>().toList();
    expect(loadEvents, hasLength(2));
    expect(loadEvents[0].model, 'ja');
    expect(loadEvents[0].phase, 'start');
    expect(loadEvents[0].ms, isNull);
    expect(loadEvents[1].phase, 'done');
    expect(loadEvents[1].ms, 123.4);
  });

  test(
    'resetSession() delegates to the transcriber and its sessionResets '
    'signal surfaces as SessionResetSubtitleEvent on events',
    () async {
      final fake = _FakeLiveTranscriber();
      final live = HayamimiLive(transcriber: fake);
      addTearDown(live.dispose);

      final events = <SubtitleEvent>[];
      live.events.listen(events.add);

      await live.resetSession();
      await Future<void>.delayed(Duration.zero);

      expect(fake.resetSessionCalled, isTrue);
      expect(events.whereType<SessionResetSubtitleEvent>(), hasLength(1));
    },
  );

  test(
    'constructor forwards the pacing knobs to the transcriber it creates '
    'when none is injected',
    () async {
      final live = HayamimiLive(
        draftIntervalSeconds: 2.5,
        draftWindowSeconds: 9.0,
        minDraftAudioSeconds: 0.6,
        autoRefineSilenceSeconds: 5.5,
        autoRefineMaxBufferedSeconds: 18.0,
        refineBufferMaxSeconds: 50.0,
      );
      addTearDown(live.dispose);

      expect(live.draftIntervalSeconds, 2.5);
      expect(live.draftWindowSeconds, 9.0);
      expect(live.minDraftAudioSeconds, 0.6);
      expect(live.autoRefineSilenceSeconds, 5.5);
      expect(live.autoRefineMaxBufferedSeconds, 18.0);
      expect(live.refineBufferMaxSeconds, 50.0);
    },
  );

  test('pacing-knob runtime setters forward to the underlying transcriber', () async {
    final fake = _FakeLiveTranscriber();
    final live = HayamimiLive(transcriber: fake);
    addTearDown(live.dispose);

    live.draftIntervalSeconds = 4.0;
    expect(fake.draftIntervalSeconds, 4.0);
    expect(live.draftIntervalSeconds, 4.0);
  });
}
