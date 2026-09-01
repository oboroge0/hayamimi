import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_decode_worker.dart';
import 'fake_live_vad.dart';
import 'fake_record_platform.dart';

/// A whole running session, microphone to transcript line, with no native
/// library present.
///
/// Both FFI edges are stood in for — the decode worker and the VAD — so
/// `isRunning` actually becomes true here. That is what makes the parts
/// which only exist while a session is live observable: segments becoming
/// transcript lines in order, drafts being skipped and discarded, a refine
/// pass folding, a reset landing behind the work it must not race, the
/// worker dying with the microphone open, and a reply arriving after
/// `stop()`.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late RecordPlatform originalPlatform;
  late _Harness h;

  setUp(() {
    originalPlatform = RecordPlatform.instance;
    h = _Harness()..install();
  });

  tearDown(() async {
    await h.dispose();
    RecordPlatform.instance = originalPlatform;
  });

  group('finals', () {
    test('two segments become two transcript lines, in order and fully populated', () async {
      await h.start();
      expect(h.transcriber.isRunning, isTrue);

      await h.endSegment(seconds: 0.5);
      await h.endSegment(seconds: 1.0);

      // Only one is at the worker; the second waits its turn.
      expect(h.worker.decodeRequests, hasLength(1));

      h.worker.reply('first line', lang: 'ja');
      await settle();
      h.worker.reply('second line', lang: 'en', switched: true);
      await settle();

      expect(h.entries.map((e) => e.text), <String>[
        'first line',
        'second line',
      ]);
      expect(h.entries[0].audioSeconds, closeTo(0.5, 1e-9));
      expect(h.entries[1].audioSeconds, closeTo(1.0, 1e-9));
      expect(h.entries[0].lang, 'ja');
      expect(h.entries[0].switched, isFalse);
      expect(h.entries[1].lang, 'en');
      expect(h.entries[1].switched, isTrue);
      expect(h.entries[0].latencyMs, 12.5);

      expect(h.transcriber.currentLang, 'en');
      // Both segments' audio is buffered for the next refine pass.
      expect(h.transcriber.refineBufferedSeconds, closeTo(1.5, 1e-9));
      // One busy transition for the burst, not one per decode.
      expect(h.decoding, <bool>[true, false]);
    });

    test('a segment that decodes to nothing produces no line and buffers no audio', () async {
      await h.start();
      await h.endSegment(seconds: 1.0);

      h.worker.reply('   ');
      await settle();

      expect(h.entries, isEmpty);
      expect(h.transcriber.refineBufferedSeconds, 0);
    });

    test('a segment too short to be worth decoding never reaches the worker', () async {
      await h.start();

      await h.endSegment(seconds: 0.05);

      expect(h.worker.decodeRequests, isEmpty);
    });
  });

  group('drafts', () {
    test('one goes out while the speaker is still talking', () async {
      await h.start();

      h.vad.speechActive = true;
      await h.pushFrames(1);

      expect(h.worker.lastRequest.kind, DecodeRequestKind.draft);
      h.worker.reply('in progr');
      await settle();

      expect(h.drafts.single.text, 'in progr');
      // A draft is provisional: it is never measured for duration.
      expect(h.drafts.single.audioSeconds, isNull);
    });

    test('none is even queued while a final is still outstanding', () async {
      await h.start();
      await h.endSegment(seconds: 1.0);

      h.vad.speechActive = true;
      await h.pushFrames(3);

      // Nothing but the final has been sent -- but that alone would also be
      // true of a draft merely waiting its turn, so let the final finish
      // and check that nothing follows it out. A draft is dropped, not
      // queued: queued, it would only delay the line about to replace it.
      expect(h.worker.decodeRequests, hasLength(1));
      expect(h.worker.lastRequest.kind, DecodeRequestKind.finalSegment);

      h.worker.reply('the whole utterance');
      await settle();

      expect(h.worker.decodeRequests, hasLength(1));
      expect(h.drafts, isEmpty);
      expect(h.entries.single.text, 'the whole utterance');
    });

    test('one whose segment ended while it was decoding is discarded', () async {
      await h.start();

      h.vad.speechActive = true;
      await h.pushFrames(1);
      expect(h.worker.lastRequest.kind, DecodeRequestKind.draft);

      // The speaker paused: the segment closed and queued its own final.
      await h.endSegment(seconds: 1.0);

      // Only now does the draft come back, describing audio the session has
      // already moved past.
      h.worker.reply('in progr');
      await settle();
      expect(h.drafts, isEmpty);

      // ...and the final that superseded it goes out and is emitted.
      expect(h.worker.lastRequest.kind, DecodeRequestKind.finalSegment);
      h.worker.reply('in progress, finished');
      await settle();
      expect(h.entries.single.text, 'in progress, finished');
    });
  });

  group('refineNow', () {
    // The "second call returns the *same* future" half of coalescing is
    // only visible on DecodeSession (see decode_session_test.dart);
    // LiveTranscriber.refineNow is an async wrapper, so it necessarily
    // hands out a fresh future. What is observable here is the effect:
    // one pass, over both segments, and the buffer emptied once.
    test('two calls yield a single pass covering everything buffered', () async {
      await h.start();

      await h.endSegment(seconds: 1.0);
      h.worker.reply('hello');
      await settle();
      expect(h.transcriber.refineBufferedSeconds, closeTo(1.0, 1e-9));

      // A final in flight, so both refine calls queue behind it.
      await h.endSegment(seconds: 1.0);
      final first = h.transcriber.refineNow();
      final second = h.transcriber.refineNow();

      h.worker.reply('second');
      await settle();

      // The refine claimed the buffer only now, so it covers both segments
      // — including the one that was still decoding when the caller asked.
      expect(h.worker.lastRequest.kind, DecodeRequestKind.refine);
      expect(h.worker.lastRequest.samples.length, 32000);

      h.worker.reply('hello second, refined');
      await settle();
      await first;
      await second;

      expect(h.refines, hasLength(1));
      expect(
        h.worker.decodeRequests
            .where((r) => r.kind == DecodeRequestKind.refine),
        hasLength(1),
      );
      expect(h.refines.single.text, 'hello second, refined');
      expect(h.refines.single.audioSeconds, closeTo(2.0, 1e-9));
      expect(h.transcriber.refineBufferedSeconds, 0);
    });

    test('with nothing buffered emits nothing and reaches no worker', () async {
      await h.start();

      await h.transcriber.refineNow();
      await settle();

      expect(h.worker.decodeRequests, isEmpty);
      expect(h.refines, isEmpty);
    });
  });

  group('Japanese punctuation', () {
    // The punctuation model runs inside the worker, so what a session can be
    // checked for here is what it does with a punctuated reply: it is passed
    // through untouched, it is reported, it reaches the refine buffer, and
    // it does not quietly excuse a refine that lost content.

    test('a ja refine is emitted as the worker punctuated it', () async {
      await h.start(punctuation: h.writePunctuationFiles());

      await h.endSegment(seconds: 1.0);
      h.worker.reply('今日めっちゃ疲れたわ', lang: 'ja');
      await settle();

      final refine = h.transcriber.refineNow();
      await settle();
      h.worker.reply(
        '今日めっちゃ疲れたわ。もう寝る。',
        lang: 'ja',
        punctuated: true,
      );
      await settle();
      await refine;

      expect(h.refines.single.text, '今日めっちゃ疲れたわ。もう寝る。');
      expect(h.refines.single.punctuated, isTrue);
      // The final the worker sent unpunctuated stays unpunctuated and says
      // so -- the flag reports what the worker did, it is not inferred from
      // the session's configuration.
      expect(h.entries.single.text, '今日めっちゃ疲れたわ');
      expect(h.entries.single.punctuated, isFalse);
    });

    test('a ja final is emitted as the worker punctuated it', () async {
      // The default now: a session with a punctuation model gets its fast
      // line punctuated too, not only its refines.
      await h.start(punctuation: h.writePunctuationFiles());
      expect(h.worker.config!.punctuateFinals, isTrue);

      await h.endSegment(seconds: 1.0);
      h.worker.reply('東京の天気は晴れです。', lang: 'ja', punctuated: true);
      await settle();

      expect(h.entries.single.text, '東京の天気は晴れです。');
      expect(h.entries.single.punctuated, isTrue);
    });

    test('an unpunctuated refine still reports itself as such', () async {
      // What a routed session's non-Japanese refine, or any session started
      // without a punctuation model, looks like from here.
      await h.start();

      await h.endSegment(seconds: 1.0);
      h.worker.reply('hello there', lang: 'en');
      await settle();

      final refine = h.transcriber.refineNow();
      await settle();
      h.worker.reply('hello there, refined', lang: 'en');
      await settle();
      await refine;

      expect(h.refines.single.punctuated, isFalse);
    });

    test('restored marks do not save a refine that lost content', () async {
      // The fallback guard compares lengths, and punctuation adds
      // characters nobody said. Here the marks alone are what would push a
      // truncated re-decode over the 0.7 threshold, so this is the case
      // that fails if the comparison forgets to strip them.
      await h.start(punctuation: h.writePunctuationFiles());

      await h.endSegment(seconds: 1.0);
      h.worker.reply('あいうえおかきくけこ', lang: 'ja');
      await settle();
      await h.endSegment(seconds: 1.0);
      h.worker.reply('さしすせそたちつてと', lang: 'ja');
      await settle();

      // 21 characters of fast text; the guard fires below 14.7 of them.
      const fastText = 'あいうえおかきくけこ さしすせそたちつてと';
      expect(fastText.length, 21);

      final refine = h.transcriber.refineNow();
      await settle();
      // 13 characters of speech, 16 with the marks: over the threshold as
      // sent, under it once the marks come off.
      h.worker.reply(
        'あい、うえお。かきくけこさしす。',
        lang: 'ja',
        punctuated: true,
      );
      await settle();
      await refine;

      expect(h.refines.single.text, fastText);
      // The text that survived is the finals', which nothing punctuated.
      expect(h.refines.single.punctuated, isFalse);
    });

    test('a long enough punctuated refine keeps its own text', () async {
      // The other side of the same guard: a refine that did not lose
      // anything is emitted as the worker sent it, marks included.
      await h.start(punctuation: h.writePunctuationFiles());

      await h.endSegment(seconds: 1.0);
      h.worker.reply('あいうえおかきくけこ', lang: 'ja');
      await settle();
      await h.endSegment(seconds: 1.0);
      h.worker.reply('さしすせそたちつてと', lang: 'ja');
      await settle();

      final refine = h.transcriber.refineNow();
      await settle();
      h.worker.reply(
        'あいうえおかきくけこ、さしすせそたちつてと。',
        lang: 'ja',
        punctuated: true,
      );
      await settle();
      await refine;

      expect(h.refines.single.text, 'あいうえおかきくけこ、さしすせそたちつてと。');
      expect(h.refines.single.punctuated, isTrue);
    });

    test('a refine that lost content falls back to the punctuated finals', () async {
      // The case the Android emulator hit: three sentences, each finalized
      // fine, and a merged re-decode that comes back holding only the last
      // of them, because this ja transducer keeps the last utterance of
      // multi-utterance audio. The guard rejects that re-decode -- and the
      // text it falls back to is punctuated, because the finals were.
      await h.start(punctuation: h.writePunctuationFiles());

      for (final sentence in <String>[
        '東京の天気は晴れです。',
        'あしたの会議は十時からです。',
        '資料は昨日送りました。',
      ]) {
        await h.endSegment(seconds: 1.0);
        h.worker.reply(sentence, lang: 'ja', punctuated: true);
        await settle();
      }

      final refine = h.transcriber.refineNow();
      await settle();
      h.worker.reply('資料は昨日送りました。', lang: 'ja', punctuated: true);
      await settle();
      await refine;

      expect(
        h.refines.single.text,
        '東京の天気は晴れです。あしたの会議は十時からです。資料は昨日送りました。',
      );
      expect(h.refines.single.text, contains('。'));
      expect(h.refines.single.punctuated, isTrue);
    });

    test('the shrink comparison strips the marks off the fast text too', () async {
      // Both sides can be punctuated now, so both are measured without
      // their marks. Here the fast finals are mark-heavy: counted raw they
      // are 21 characters and the 12-character re-decode looks like a loss;
      // counted as words -- 11 against 12 -- it lost nothing.
      await h.start(punctuation: h.writePunctuationFiles());

      for (final sentence in <String>[
        'あ、い、う、え、お。',
        'か、き、く、け、こ。',
      ]) {
        await h.endSegment(seconds: 1.0);
        h.worker.reply(sentence, lang: 'ja', punctuated: true);
        await settle();
      }
      expect('あ、い、う、え、お。 か、き、く、け、こ。'.length, 21);

      final refine = h.transcriber.refineNow();
      await settle();
      h.worker.reply('あいうえお、かきくけこ。', lang: 'ja', punctuated: true);
      await settle();
      await refine;

      expect(h.refines.single.text, 'あいうえお、かきくけこ。');
      expect(h.refines.single.punctuated, isTrue);
    });

    test('applyToFinals: false leaves the finals -- and the fallback -- unpunctuated', () async {
      // The switch is honoured in the worker, which this test stands in
      // for, so it is checked in two places: the config the session sent,
      // and what the session does with the unpunctuated finals that follow.
      await h.start(
        punctuation: h.writePunctuationFiles(applyToFinals: false),
      );
      expect(h.worker.config!.punctuateFinals, isFalse);

      for (final sentence in <String>[
        '東京の天気は晴れです',
        'あしたの会議は十時からです',
        '資料は昨日送りました',
      ]) {
        await h.endSegment(seconds: 1.0);
        h.worker.reply(sentence, lang: 'ja');
        await settle();
      }
      expect(h.entries.every((e) => !e.punctuated), isTrue);

      final refine = h.transcriber.refineNow();
      await settle();
      h.worker.reply('資料は昨日送りました。', lang: 'ja', punctuated: true);
      await settle();
      await refine;

      expect(
        h.refines.single.text,
        '東京の天気は晴れです あしたの会議は十時からです 資料は昨日送りました',
      );
      expect(h.refines.single.text, isNot(contains('。')));
      expect(h.refines.single.punctuated, isFalse);
    });
  });

  group('resetSession', () {
    test('lands behind the segment still decoding, then clears and reports', () async {
      await h.start();
      await h.endSegment(seconds: 1.0);

      final reset = h.transcriber.resetSession();
      await settle();
      // Still queued: the transcript line comes first.
      expect(h.worker.commands, hasLength(1));

      h.worker.reply('last words', lang: 'ja');
      await settle();
      expect(h.entries.single.text, 'last words');
      expect(h.transcriber.currentLang, 'ja');

      h.worker.ack();
      await settle();
      await reset;

      expect(h.resets, hasLength(1));
      expect(h.transcriber.currentLang, isNull);
      expect(h.transcriber.refineBufferedSeconds, 0);
    });
  });

  group('the decode worker dying', () {
    test('reports once, stops the session, frees the VAD, and stays restartable', () async {
      await h.start();
      expect(h.transcriber.isRunning, isTrue);
      final vad = h.vad;
      final worker = h.worker;

      worker.die('the isolate exited');
      await h.pump();

      expect(h.errors, hasLength(1));
      expect(
        h.errors.single.message,
        contains('decode worker stopped unexpectedly'),
      );
      expect(h.transcriber.isRunning, isFalse);
      expect(vad.freed, isTrue);
      expect(worker.shutdownCount, 1);

      // A second death report from the same, already-torn-down session must
      // not produce a second error.
      worker.die('and again');
      await h.pump();
      expect(h.errors, hasLength(1));

      // Still usable: a fresh start builds a new worker and a new VAD.
      await h.start();
      expect(h.transcriber.isRunning, isTrue);
      expect(h.workers, hasLength(2));
      expect(h.vads.built, hasLength(2));
    });

    test('a decode that fails is reported without stopping the session', () async {
      await h.start();
      await h.endSegment(seconds: 1.0);

      h.worker.failRequest('sherpa-onnx blew up');
      await settle();

      expect(h.errors.single.message, contains('sherpa-onnx blew up'));
      expect(h.transcriber.isRunning, isTrue);

      // And the next segment is still transcribed.
      await h.endSegment(seconds: 1.0);
      h.worker.reply('still here');
      await settle();
      expect(h.entries.single.text, 'still here');
    });
  });

  group('stop', () {
    test('emits the segment its VAD flush produces before returning', () async {
      h.vads.setUp = (vad) => vad.flushSegment = FakeLiveVad.segment(1.0);
      await h.start();

      final stopping = h.transcriber.stop();
      await h.pump();

      expect(h.vad.flushCount, 1);
      expect(h.worker.decodeRequests, hasLength(1));
      h.worker.reply('last words');
      await stopping;

      expect(h.entries.map((e) => e.text), <String>['last words']);
      expect(h.transcriber.isRunning, isFalse);
    });

    test('a reply that arrives afterwards emits nothing and throws nothing', () async {
      // Keep the worker's reply stream open past shutdown, so a message
      // that crossed the session's teardown can actually be delivered.
      h.workerSetUp = (worker) => worker.closeOnShutdown = false;
      await h.start();

      await h.endSegment(seconds: 1.0);
      final request = h.worker.lastRequest;
      h.worker.reply('in time');
      await settle();
      expect(h.entries, hasLength(1));

      await h.transcriber.stop();
      expect(h.transcriber.isRunning, isFalse);

      h.worker.replyTo(request, 'too late');
      await h.pump();

      expect(h.entries, hasLength(1));
    });
  });

  group('pre-roll', () {
    // Silero VAD reports a speech onset slightly late, so a session decodes
    // the second before the onset along with the segment (see
    // preroll.dart). What is checked here is the wiring: that the audio
    // actually sent to the worker grew, that the reported duration grew
    // with it, and that two segments in a row do not both claim the gap
    // between them.

    test('a final is decoded with the second of audio before its onset', () async {
      await h.start();
      // 1.28s of session audio before the segment the VAD reports at 1.28s.
      await h.pushFrames(40);

      await h.endSegment(seconds: 1.0, startSample: 40 * 512);
      expect(h.worker.lastRequest.samples.length, 16000 + 16000);

      h.worker.reply('one line');
      await settle();

      // The pre-roll is part of the segment, so it is part of what the
      // entry says it covered.
      expect(h.entries.single.audioSeconds, closeTo(2.0, 1e-9));
      expect(h.transcriber.refineBufferedSeconds, closeTo(2.0, 1e-9));
    });

    test('two segments in a row do not both include the gap between them', () async {
      await h.start();
      await h.pushFrames(60);

      // 0.5s of speech starting half a second in: nothing before it has
      // been claimed, so it takes its full half second of run-up.
      await h.endSegment(seconds: 0.5, startSample: 8000);
      expect(h.worker.lastRequest.samples.length, 8000 + 8000);
      h.worker.reply('first');
      await settle();

      // The next starts 0.25s after the first ended. A full second of
      // pre-roll would reach back into it; it gets the 0.25s gap instead.
      await h.endSegment(seconds: 0.5, startSample: 20000);
      expect(h.worker.lastRequest.samples.length, 4000 + 8000);
      h.worker.reply('second');
      await settle();

      // 1.75s of session audio, decoded as 1.75s across two lines.
      expect(h.entries[0].audioSeconds, closeTo(1.0, 1e-9));
      expect(h.entries[1].audioSeconds, closeTo(0.75, 1e-9));
    });

    test('prerollSeconds = 0 decodes exactly what the VAD delimited', () async {
      h.transcriber.prerollSeconds = 0;
      await h.start();
      await h.pushFrames(40);

      await h.endSegment(seconds: 1.0, startSample: 40 * 512);

      expect(h.worker.lastRequest.samples.length, 16000);
      h.worker.reply('one line');
      await settle();
      expect(h.entries.single.audioSeconds, closeTo(1.0, 1e-9));
    });

    test('a segment dropped as too short still blocks the next pre-roll', () async {
      // A blip the app-level guard throws away never reaches the worker,
      // but the audio it covered has still been accounted for -- the next
      // segment must not pick it up as context.
      await h.start();
      await h.pushFrames(60);

      await h.endSegment(seconds: 0.05, startSample: 16000);
      expect(h.worker.decodeRequests, isEmpty);

      await h.endSegment(seconds: 0.5, startSample: 20000);
      // 16000 + 800 = 16800, so only 3200 samples of gap are available.
      expect(h.worker.lastRequest.samples.length, 3200 + 8000);
    });
  });

  group('startDebugWavStream', () {
    test('feeds the whole file through and flushes its last segment', () async {
      h.workerSetUp = (worker) => worker.autoReplyText = 'flushed line';
      h.vads.setUp = (vad) => vad.flushSegment = FakeLiveVad.segment(1.0);
      final wavPath = h.writeWav(sampleCount: 4096);

      await h.transcriber.startDebugWavStream(
        modelKind: ModelKind.zipformerTransducer,
        modelDir: h.modelDir.path,
        vadModelPath: h.vadModelPath,
        wavPath: wavPath,
        realtime: false,
      );

      expect(h.vad.frames, hasLength(4096 ~/ 512));
      expect(h.vad.flushCount, 1);
      expect(h.entries.map((e) => e.text), <String>['flushed line']);
      expect(h.transcriber.isDebugStreaming, isFalse);
      expect(h.worker.shutdownCount, 1);
      expect(h.vad.freed, isTrue);
    });

    test('ends with a refine over what it transcribed', () async {
      // The session is torn down before this future completes, so a caller
      // cannot ask for a refine afterwards -- refineNow() would find no
      // worker. Without a closing pass, the method that exists to show
      // punctuated refine output on a device with no microphone would
      // never produce one.
      h.workerSetUp = (worker) => worker.autoReplyText = 'flushed line';
      h.vads.setUp = (vad) => vad.flushSegment = FakeLiveVad.segment(1.0);
      final wavPath = h.writeWav(sampleCount: 4096);

      await h.transcriber.startDebugWavStream(
        modelKind: ModelKind.zipformerTransducer,
        modelDir: h.modelDir.path,
        vadModelPath: h.vadModelPath,
        wavPath: wavPath,
        realtime: false,
      );

      expect(
        h.worker.decodeRequests.map((r) => r.kind),
        contains(DecodeRequestKind.refine),
      );
      expect(h.refines.single.text, 'flushed line');
      expect(h.refines.single.audioSeconds, closeTo(1.0, 1e-9));
      // And it landed before the session went away, not after.
      expect(h.transcriber.isDebugStreaming, isFalse);
      expect(h.worker.shutdownCount, 1);
    });

    test('a stream that transcribed nothing ends with no refine', () async {
      // Nothing buffered means nothing for a refine to cover; asking for
      // one anyway would cost a decode of silence.
      h.workerSetUp = (worker) => worker.autoReplyText = 'never asked for';
      final wavPath = h.writeWav(sampleCount: 4096);

      await h.transcriber.startDebugWavStream(
        modelKind: ModelKind.zipformerTransducer,
        modelDir: h.modelDir.path,
        vadModelPath: h.vadModelPath,
        wavPath: wavPath,
        realtime: false,
      );

      expect(h.worker.decodeRequests, isEmpty);
      expect(h.refines, isEmpty);
    });

    test('auto-refine set before the stream fires during it', () async {
      // The autoRefineEnabled setter only started its timer for a
      // microphone session; a debug stream got one only because
      // _resetSessionState starts it when the flag is already set. Both
      // paths now work, and this pins the documented one: set it, stream,
      // and refines arrive as the audio goes by rather than only at the
      // end.
      h.workerSetUp = (worker) => worker.autoReplyText = 'a line';
      h.vads.setUp = (vad) =>
          vad.segmentsPerFrame.add(FakeLiveVad.segment(1.0));
      // Any silence at all is enough; the due check itself runs once a
      // second, which is what paces this test.
      h.transcriber.autoRefineSilenceSeconds = 0.001;
      h.transcriber.autoRefineEnabled = true;
      // ~2.6s of audio at 16kHz, streamed at its own pace, so the
      // once-a-second due check runs at least twice while it plays.
      final wavPath = h.writeWav(sampleCount: 41600);

      // A second segment, timed to arrive after the first due check has
      // already refined the first one.
      Timer(const Duration(milliseconds: 1700), () {
        h.vad.segmentsPerFrame.add(FakeLiveVad.segment(1.0));
      });

      await h.transcriber.startDebugWavStream(
        modelKind: ModelKind.zipformerTransducer,
        modelDir: h.modelDir.path,
        vadModelPath: h.vadModelPath,
        wavPath: wavPath,
      );

      // Two refines, not one: a single closing pass would have covered both
      // segments at once, so the first one covering 1.0s on its own is the
      // evidence that a refine fired while the stream was still playing.
      expect(h.refines.length, greaterThanOrEqualTo(2));
      expect(h.refines.first.audioSeconds, closeTo(1.0, 1e-9));
    });

    test('auto-refine turned on mid-stream starts firing too', () async {
      h.workerSetUp = (worker) => worker.autoReplyText = 'a line';
      h.vads.setUp = (vad) =>
          vad.segmentsPerFrame.add(FakeLiveVad.segment(1.0));
      h.transcriber.autoRefineSilenceSeconds = 0.001;
      final wavPath = h.writeWav(sampleCount: 41600);

      Timer(const Duration(milliseconds: 300), () {
        h.transcriber.autoRefineEnabled = true;
      });
      Timer(const Duration(milliseconds: 1700), () {
        h.vad.segmentsPerFrame.add(FakeLiveVad.segment(1.0));
      });

      await h.transcriber.startDebugWavStream(
        modelKind: ModelKind.zipformerTransducer,
        modelDir: h.modelDir.path,
        vadModelPath: h.vadModelPath,
        wavPath: wavPath,
      );

      expect(h.refines.length, greaterThanOrEqualTo(2));
      expect(h.refines.first.audioSeconds, closeTo(1.0, 1e-9));
    });

    test('a wav at the wrong sample rate is rejected and leaves nothing running', () async {
      final wavPath = h.writeWav(sampleCount: 1024, sampleRate: 8000);

      await expectLater(
        h.transcriber.startDebugWavStream(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: h.modelDir.path,
          vadModelPath: h.vadModelPath,
          wavPath: wavPath,
          realtime: false,
        ),
        throwsA(
          isA<LiveTranscriberException>().having(
            (e) => e.message,
            'message',
            contains('8000Hz != required 16000Hz'),
          ),
        ),
      );

      expect(h.transcriber.isDebugStreaming, isFalse);
      expect(h.worker.shutdownCount, 1);
      expect(h.vad.freed, isTrue);
    });
  });
}

/// One `LiveTranscriber` with both of its native edges replaced, plus the
/// plumbing to feed it audio and read what came out.
class _Harness {
  late FakeRecordPlatform platform;
  late Directory modelDir;
  late String vadModelPath;
  late LiveTranscriber transcriber;

  final FakeVadSource vads = FakeVadSource();
  final List<FakeDecodeWorker> workers = <FakeDecodeWorker>[];

  /// Applied to each worker as it is built, so a test can script one before
  /// the session that will use it exists.
  void Function(FakeDecodeWorker worker)? workerSetUp;

  final List<LiveTranscriptEntry> entries = <LiveTranscriptEntry>[];
  final List<LiveTranscriptEntry> drafts = <LiveTranscriptEntry>[];
  final List<LiveTranscriptEntry> refines = <LiveTranscriptEntry>[];
  final List<LiveTranscriberException> errors = <LiveTranscriberException>[];
  final List<bool> decoding = <bool>[];
  final List<void> resets = <void>[];

  FakeDecodeWorker get worker => workers.last;
  FakeLiveVad get vad => vads.last;

  void install() {
    platform = FakeRecordPlatform();
    RecordPlatform.instance = platform;
    modelDir = Directory.systemTemp.createTempSync('hayamimi_session');
    vadModelPath = '${modelDir.path}${Platform.pathSeparator}vad.onnx';
    File(vadModelPath).writeAsBytesSync(<int>[0]);

    transcriber = LiveTranscriber(
      decodeWorkerFactory: _newWorker,
      vadFactory: vads.build,
      // A draft on every frame, so the draft rules can be exercised without
      // waiting out the production pacing.
      draftIntervalSeconds: 0.001,
      minDraftAudioSeconds: 0.01,
    );
    transcriber.entries.listen(entries.add);
    transcriber.drafts.listen(drafts.add);
    transcriber.refineEntries.listen(refines.add);
    transcriber.errors.listen(errors.add);
    transcriber.decoding.listen(decoding.add);
    transcriber.sessionResets.listen(resets.add);
  }

  DecodeWorker _newWorker() {
    final worker = FakeDecodeWorker();
    workerSetUp?.call(worker);
    workers.add(worker);
    return worker;
  }

  Future<void> start({JaPunctuation? punctuation}) => transcriber.start(
    modelKind: ModelKind.zipformerTransducer,
    modelDir: modelDir.path,
    vadModelPath: vadModelPath,
    punctuation: punctuation,
  );

  /// Stand-ins for the punctuation model and its vocabulary. `start()`
  /// checks both files exist before it spawns anything; the fake worker
  /// never loads them, so their contents do not matter.
  JaPunctuation writePunctuationFiles({bool applyToFinals = true}) {
    final sep = Platform.pathSeparator;
    final model = '${modelDir.path}${sep}punct.onnx';
    final vocab = '${modelDir.path}${sep}vocab.txt';
    File(model).writeAsBytesSync(<int>[0]);
    File(vocab).writeAsStringSync('[PAD]\n');
    return JaPunctuation(
      modelPath: model,
      vocabPath: vocab,
      applyToFinals: applyToFinals,
    );
  }

  /// Delivers [count] frames' worth of microphone audio.
  Future<void> pushFrames(int count) async {
    platform.micController.add(Uint8List(count * vad.frameSize * 2));
    await settle();
  }

  /// Ends a speech segment carrying [seconds] of audio, the way a speaker
  /// pausing does.
  ///
  /// [startSample] is where the VAD claims the segment began, counted in
  /// samples from the session's first frame — what the pre-roll is measured
  /// back from. It defaults to 0, i.e. "at the very start of the session",
  /// which leaves nothing in front of the segment to prepend and so keeps
  /// the audio a test sees equal to what it queued.
  Future<void> endSegment({
    required double seconds,
    int startSample = 0,
  }) async {
    vad.segmentsPerFrame.add(
      FakeLiveVad.segment(seconds, startSample: startSample),
    );
    await pushFrames(1);
  }

  /// Writes a 16-bit PCM wav file and returns its path.
  String writeWav({required int sampleCount, int sampleRate = 16000}) {
    final path = '${modelDir.path}${Platform.pathSeparator}test_$sampleRate.wav';
    File(path).writeAsBytesSync(
      _wavBytes(sampleCount: sampleCount, sampleRate: sampleRate),
    );
    return path;
  }

  /// Several turns of the event loop — enough for the teardown paths, which
  /// await a handful of things in sequence.
  Future<void> pump() => Future<void>.delayed(const Duration(milliseconds: 20));

  Future<void> dispose() async {
    // Anything a test left outstanding gets answered now, so dispose()'s
    // drain finishes promptly instead of waiting out its timeout. Answers
    // for requests that were already settled no longer match anything the
    // session is waiting on, so they are ignored.
    for (final worker in workers) {
      worker.autoReplyText ??= 'teardown';
      worker.answerAll();
    }
    await transcriber.dispose();
    await platform.close();
    if (modelDir.existsSync()) {
      modelDir.deleteSync(recursive: true);
    }
  }
}


/// A canonical 44-byte-header, 16-bit PCM wav of silence.
Uint8List _wavBytes({
  required int sampleCount,
  int sampleRate = 16000,
  int channels = 1,
}) {
  final dataSize = sampleCount * channels * 2;
  final bytes = Uint8List(44 + dataSize);
  final header = ByteData.sublistView(bytes);
  void ascii(int offset, String text) {
    for (var i = 0; i < text.length; i++) {
      header.setUint8(offset + i, text.codeUnitAt(i));
    }
  }

  ascii(0, 'RIFF');
  header.setUint32(4, 36 + dataSize, Endian.little);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  header.setUint32(16, 16, Endian.little);
  header.setUint16(20, 1, Endian.little); // PCM
  header.setUint16(22, channels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, sampleRate * channels * 2, Endian.little);
  header.setUint16(32, channels * 2, Endian.little);
  header.setUint16(34, 16, Endian.little);
  ascii(36, 'data');
  header.setUint32(40, dataSize, Endian.little);
  return bytes;
}
