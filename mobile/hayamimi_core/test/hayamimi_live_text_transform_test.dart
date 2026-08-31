import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_record_platform.dart';

/// A [LiveTranscriber] whose entries/refineEntries/drafts streams are
/// injectable, so `HayamimiLive`'s `textTransform` hook can be exercised
/// without a real mic or native VAD/recognizer (same fake-injection
/// approach `lifecycle_test.dart` uses for `RecordPlatform`, applied here
/// to the transcript streams instead).
class _FakeLiveTranscriber extends LiveTranscriber {
  final _entriesCtrl = StreamController<LiveTranscriptEntry>.broadcast();
  final _refineCtrl = StreamController<LiveTranscriptEntry>.broadcast();
  final _draftCtrl = StreamController<LiveTranscriptEntry>.broadcast();

  @override
  Stream<LiveTranscriptEntry> get entries => _entriesCtrl.stream;

  @override
  Stream<LiveTranscriptEntry> get refineEntries => _refineCtrl.stream;

  @override
  Stream<LiveTranscriptEntry> get drafts => _draftCtrl.stream;

  void emitFinal(LiveTranscriptEntry entry) => _entriesCtrl.add(entry);
  void emitRefine(LiveTranscriptEntry entry) => _refineCtrl.add(entry);
  void emitDraft(LiveTranscriptEntry entry) => _draftCtrl.add(entry);

  @override
  Future<void> dispose() async {
    await _entriesCtrl.close();
    await _refineCtrl.close();
    await _draftCtrl.close();
    await super.dispose();
  }
}

LiveTranscriptEntry _entry(String text, {String? lang}) =>
    LiveTranscriptEntry(text: text, timestamp: DateTime.now(), lang: lang);

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
    'textTransform is applied to final/refine/draft text before broadcasting',
    () async {
      final fake = _FakeLiveTranscriber();
      final live = HayamimiLive(
        transcriber: fake,
        textTransform: (text, lang) => '[$lang]$text',
      );
      addTearDown(live.dispose);

      final events = <SubtitleEvent>[];
      live.events.listen(events.add);

      fake.emitFinal(_entry('hello', lang: 'en'));
      fake.emitRefine(_entry('world', lang: 'ja'));
      fake.emitDraft(_entry('draft', lang: 'ko'));
      await Future<void>.delayed(Duration.zero);

      expect(events.whereType<FinalSubtitleEvent>().single.text, '[en]hello');
      expect(events.whereType<RefineSubtitleEvent>().single.text, '[ja]world');
      expect(events.whereType<PartialSubtitleEvent>().single.text, '[ko]draft');
    },
  );

  test('textTransform is a no-op by default', () async {
    final fake = _FakeLiveTranscriber();
    final live = HayamimiLive(transcriber: fake);
    addTearDown(live.dispose);

    final events = <SubtitleEvent>[];
    live.events.listen(events.add);

    fake.emitFinal(_entry('plain text', lang: 'en'));
    await Future<void>.delayed(Duration.zero);

    expect(events.whereType<FinalSubtitleEvent>().single.text, 'plain text');
  });

  test('textTransform can be swapped at runtime mid-session', () async {
    final fake = _FakeLiveTranscriber();
    final live = HayamimiLive(transcriber: fake);
    addTearDown(live.dispose);

    final events = <SubtitleEvent>[];
    live.events.listen(events.add);

    fake.emitFinal(_entry('one', lang: 'en'));
    await Future<void>.delayed(Duration.zero);
    expect(events.whereType<FinalSubtitleEvent>().single.text, 'one');

    live.textTransform = (text, lang) => text.toUpperCase();
    fake.emitFinal(_entry('two', lang: 'en'));
    await Future<void>.delayed(Duration.zero);
    expect(events.whereType<FinalSubtitleEvent>().last.text, 'TWO');
  });

  test(
    'entries with no lang of their own fall back to HayamimiLive.lang',
    () async {
      final fake = _FakeLiveTranscriber();
      final live = HayamimiLive(
        transcriber: fake,
        textTransform: (text, lang) => '$lang:$text',
      );
      live.lang = 'ja';
      addTearDown(live.dispose);

      final events = <SubtitleEvent>[];
      live.events.listen(events.add);

      fake.emitFinal(_entry('x'));
      await Future<void>.delayed(Duration.zero);

      expect(events.whereType<FinalSubtitleEvent>().single.text, 'ja:x');
    },
  );
}
