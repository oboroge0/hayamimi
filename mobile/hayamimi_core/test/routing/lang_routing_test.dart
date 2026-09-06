import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/routing/lang_routing.dart';

/// Ported test cases from `tests/test_units.py` (desktop `asr_engine.py`),
/// so the Dart routing logic stays behaviorally identical to the PC
/// pipeline's dual-LID policy documented in `docs/eval/lid.md`.
void main() {
  group('svLidTag', () {
    test('normalizes bracketed tag', () {
      expect(svLidTag('<|ja|>'), 'ja');
      expect(svLidTag('<|yue|>'), 'yue');
      expect(svLidTag('<|unk|>'), '');
    });
  });

  group('resolveDualConfirm', () {
    test('same lang is a noop', () {
      final r = resolveDualConfirm(
        lang: 'ja',
        lastLang: 'ja',
        speechSeconds: 3.0,
        svLang: 'ja',
      );
      expect(r.lang, 'ja');
      expect(r.switched, false);
    });

    test('bootstrap trusts probe over whisper misfire', () {
      // LID.md real-mic accident scenario: session start, whisper-tiny
      // says "zh" (wrong), SenseVoice's own probe says "ja" (right) — the
      // very first segment must decode as "ja", not "zh".
      final r = resolveDualConfirm(
        lang: 'zh',
        lastLang: null,
        speechSeconds: 3.0,
        svLang: 'ja',
      );
      expect(r.lang, 'ja');
      expect(r.switched, false); // the two LIDs disagreed
    });

    test('bootstrap agreement marks switched', () {
      final r = resolveDualConfirm(
        lang: 'en',
        lastLang: null,
        speechSeconds: 3.0,
        svLang: 'en',
      );
      expect(r.lang, 'en');
      expect(r.switched, true);
    });

    test('holds current lang on disagreement', () {
      // ja-only session hit by a whisper-tiny "en" misfire; SenseVoice's
      // probe still says "ja" -> stays on "ja", no switch.
      final r = resolveDualConfirm(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 1.5,
        svLang: 'ja',
      );
      expect(r.lang, 'ja');
      expect(r.switched, false);
    });

    test('switches immediately on agreement', () {
      // Both LIDs agree -> switch immediately, no length/repeat-count gate.
      final r = resolveDualConfirm(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 1.0,
        svLang: 'en',
      );
      expect(r.lang, 'en');
      expect(r.switched, true);
    });

    test('ignores sub-probe-length even on agreement', () {
      final r = resolveDualConfirm(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 0.3,
        svLang: 'en',
      );
      expect(r.lang, 'ja');
      expect(r.switched, false);
    });

    test('bootstrap with an out-of-coverage whisper guess still resolves '
        'via the SenseVoice probe', () {
      // Mobile-specific scenario (routed_recognizer.dart): whisper-tiny's
      // very first guess is a language with no loaded tier at all (e.g.
      // "fr" -- outside jaSenseVoiceLangs), and SenseVoice's own probe on
      // the same audio says "ja". Must resolve to "ja", not blindly default
      // there without ever consulting the probe, and not get stuck decoding
      // "fr" forever (same semantics as the desktop's resolve_sticky_lang
      // bootstrap_probe_lang fix, ported here as of the routed_recognizer.dart
      // bootstrap fix).
      final r = resolveDualConfirm(
        lang: 'fr',
        lastLang: null,
        speechSeconds: 3.0,
        svLang: 'ja',
      );
      expect(r.lang, 'ja');
      expect(r.switched, false); // the two LIDs disagreed, not a confirmed match
    });

    test('mismatch at bootstrap falls back to whisper guess', () {
      // No SenseVoice tag at all (svLang empty) -- bootstrap must still
      // resolve to something.
      final r = resolveDualConfirm(
        lang: 'zh',
        lastLang: null,
        speechSeconds: 3.0,
        svLang: '',
      );
      expect(r.lang, 'zh');
      expect(r.switched, false);
    });
  });

  group('resolveStickyLang', () {
    test('first utterance has no last_lang yet', () {
      final r = resolveStickyLang(
        lang: 'en',
        lastLang: null,
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r.lang, 'en');
      expect(r.suppressFallback, false);
      expect(r.pendingLang, 'en');
      expect(r.pendingCount, 1);
    });

    test('bootstrap prefers probe language over unarbitrable guess', () {
      final r = resolveStickyLang(
        lang: 'ru',
        lastLang: null,
        speechSeconds: 1.9,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
        bootstrapProbeLang: 'ja',
      );
      expect(r.lang, 'ja');
      expect(r.suppressFallback, true);
      expect(r.pendingLang, null);
      expect(r.pendingCount, 0);
    });

    test('bootstrap prefers probe language for long unarbitrable guess', () {
      final r = resolveStickyLang(
        lang: 'ru',
        lastLang: null,
        speechSeconds: 2.5,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
        bootstrapProbeLang: 'ja',
      );
      expect(r.lang, 'ja');
      expect(r.suppressFallback, false);
      expect(r.pendingLang, 'ru');
      expect(r.pendingCount, 1);
    });

    test('bootstrap non-SV lang confirms after repeats', () {
      final r1 = resolveStickyLang(
        lang: 'fr',
        lastLang: null,
        speechSeconds: 5.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
        bootstrapProbeLang: 'ja',
      );
      expect(r1.lang, 'ja');
      expect(r1.pendingLang, 'fr');
      expect(r1.pendingCount, 1);

      final r2 = resolveStickyLang(
        lang: 'fr',
        lastLang: null,
        speechSeconds: 5.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: r1.pendingLang,
        pendingCount: r1.pendingCount,
        bootstrapProbeLang: 'ja',
      );
      expect(r2.lang, 'fr');
      expect(r2.suppressFallback, false);
      expect(r2.pendingLang, null);
      expect(r2.pendingCount, 0);
    });

    test('same lang resets pending', () {
      final r = resolveStickyLang(
        lang: 'ja',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: 'en',
        pendingCount: 1,
      );
      expect(r.lang, 'ja');
      expect(r.suppressFallback, false);
      expect(r.pendingLang, null);
      expect(r.pendingCount, 0);
    });

    test('single misfire is held not switched', () {
      final r = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r.lang, 'ja');
      expect(r.suppressFallback, false);
      expect(r.pendingLang, 'en');
      expect(r.pendingCount, 1);
    });

    test('short misfire suppresses fallback', () {
      final r = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 1.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r.lang, 'ja');
      expect(r.suppressFallback, true);
    });

    test('confirmed switch after two consecutive detections', () {
      final r1 = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r1.lang, 'ja');
      expect(r1.pendingLang, 'en');
      expect(r1.pendingCount, 1);

      final r2 = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: r1.pendingLang,
        pendingCount: r1.pendingCount,
      );
      expect(r2.lang, 'en');
      expect(r2.suppressFallback, false);
      expect(r2.pendingLang, null);
      expect(r2.pendingCount, 0);
    });

    test('alternating misfires never accumulate', () {
      final r1 = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r1.pendingLang, 'en');
      expect(r1.pendingCount, 1);

      final r2 = resolveStickyLang(
        lang: 'zh',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: r1.pendingLang,
        pendingCount: r1.pendingCount,
      );
      expect(r2.lang, 'ja'); // still held at the session language
      expect(r2.pendingLang, 'zh'); // candidate reset to the new guess
      expect(r2.pendingCount, 1);
    });

    test('switch_confirm=1 disables hysteresis', () {
      final r = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 1,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r.lang, 'en');
      expect(r.pendingLang, null);
      expect(r.pendingCount, 0);
    });

    test('switch_confirm=1 and guard=0 fully disable the lock', () {
      final r1 = resolveStickyLang(
        lang: 'zh',
        lastLang: 'ja',
        speechSeconds: 0.3,
        minSwitchSeconds: 0.0,
        switchConfirm: 1,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r1.lang, 'zh');
      expect(r1.suppressFallback, false);
      expect(r1.pendingLang, null);
      expect(r1.pendingCount, 0);

      final r2 = resolveStickyLang(
        lang: 'ja',
        lastLang: 'zh',
        speechSeconds: 0.3,
        minSwitchSeconds: 0.0,
        switchConfirm: 1,
        pendingLang: r1.pendingLang,
        pendingCount: r1.pendingCount,
      );
      expect(r2.lang, 'ja');
      expect(r2.suppressFallback, false);
      expect(r2.pendingLang, null);
      expect(r2.pendingCount, 0);
    });

    test('short detection never advances pending count', () {
      final r = resolveStickyLang(
        lang: 'zh',
        lastLang: 'ja',
        speechSeconds: 1.9,
        minSwitchSeconds: 10.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r.lang, 'ja');
      expect(r.suppressFallback, true);
      expect(r.pendingLang, null);
      expect(r.pendingCount, 0);
    });

    test('short detections never confirm a switch', () {
      String? pendingLang;
      int pendingCount = 0;
      for (final speechSeconds in [9.1, 1.9, 1.5, 1.5]) {
        final r = resolveStickyLang(
          lang: 'zh',
          lastLang: 'ja',
          speechSeconds: speechSeconds,
          minSwitchSeconds: 10.0,
          switchConfirm: 2,
          pendingLang: pendingLang,
          pendingCount: pendingCount,
        );
        expect(r.lang, 'ja');
        expect(r.suppressFallback, true);
        expect(r.pendingLang, null);
        expect(r.pendingCount, 0);
        pendingLang = r.pendingLang;
        pendingCount = r.pendingCount;
      }
    });

    test('short detection does not reset a real candidate', () {
      final r1 = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r1.pendingLang, 'en');
      expect(r1.pendingCount, 1);

      final r2 = resolveStickyLang(
        lang: 'zh',
        lastLang: 'ja',
        speechSeconds: 0.5,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: r1.pendingLang,
        pendingCount: r1.pendingCount,
      );
      expect(r2.lang, 'ja');
      expect(r2.suppressFallback, true);
      expect(r2.pendingLang, 'en'); // unchanged
      expect(r2.pendingCount, 1);

      final r3 = resolveStickyLang(
        lang: 'en',
        lastLang: 'ja',
        speechSeconds: 3.0,
        minSwitchSeconds: 2.0,
        switchConfirm: 2,
        pendingLang: r2.pendingLang,
        pendingCount: r2.pendingCount,
      );
      expect(r3.lang, 'en');
      expect(r3.pendingLang, null);
      expect(r3.pendingCount, 0);
    });

    test('long detections still confirm a switch under large guard', () {
      final r1 = resolveStickyLang(
        lang: 'zh',
        lastLang: 'ja',
        speechSeconds: 10.0,
        minSwitchSeconds: 10.0,
        switchConfirm: 2,
        pendingLang: null,
        pendingCount: 0,
      );
      expect(r1.lang, 'ja');
      expect(r1.pendingLang, 'zh');
      expect(r1.pendingCount, 1);
    });
  });
}
