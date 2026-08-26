/// Multilingual routing decisions, ported from the desktop pipeline's
/// dual-LID policy (see `docs/LID.md` and `scripts/asr_engine.py`'s
/// `resolve_dual_confirm` / `resolve_sticky_lang` / `sv_lid_tag`).
///
/// These are pure functions: no model calls, no I/O. Callers own running
/// whisper-tiny's spoken-language identifier and SenseVoice's own decode
/// to get the two LID signals these functions arbitrate between.
library;

/// The 5 languages SenseVoice's own internal LID can arbitrate (its model
/// directory name: zh-en-ja-ko-yue). whisper-tiny can never actually emit
/// "yue" as a candidate, but it's listed here for documentation symmetry
/// with `docs/LID.md` and the desktop `DUAL_CONFIRM_LANGS`.
const Set<String> dualConfirmLangs = {'ja', 'en', 'zh', 'ko', 'yue'};

/// A whisper-tiny candidate shorter than this is presumed non-speech noise
/// (jingle/SFX/misfire) and never confirms a switch even if SenseVoice
/// agrees. Mirrors the desktop `MIN_PROBE_S`.
const double minProbeSeconds = 0.5;

const List<String> _svLidCodes = ['ja', 'en', 'zh', 'ko', 'yue'];

/// Normalizes SenseVoice's raw `'<|xx|>'`-style language tag (or its bare
/// `result.lang` field) to one of `"ja"/"en"/"zh"/"ko"/"yue"`, or `""` if
/// none of the 5 codes appear. Mirrors the desktop `sv_lid_tag`.
String svLidTag(String rawTag) {
  for (final code in _svLidCodes) {
    if (rawTag.contains(code)) {
      return code;
    }
  }
  return '';
}

/// Result of [resolveDualConfirm]: the language to decode this segment
/// with, and whether this call is the reason the session's language
/// changed.
class DualConfirmResult {
  const DualConfirmResult(this.lang, this.switched);

  final String lang;
  final bool switched;
}

/// Dual-LID switch confirmation for the 5 SenseVoice-covered languages.
///
/// `docs/LID.md` measured whisper-tiny alone at only 59-65% LID accuracy at
/// 2 seconds (far worse under babble noise), but whisper-tiny AND
/// SenseVoice's own internal LID AGREEING on the same language hits
/// 85-98% accuracy at the same length. So instead of gating a switch on
/// segment length or repeat-count, this gates it on the two independent
/// LID signals agreeing: agreement is available from the very first
/// segment.
///
/// [svLang] is the caller's already-computed SenseVoice LID tag for this
/// exact audio (via [svLidTag] on its decode's `.lang` field) — this
/// function is pure and makes no model calls itself.
///
/// Session bootstrap ([lastLang] is `null`) has no current language to
/// hold at while waiting for agreement, so it resolves directly to
/// [svLang]: SenseVoice's own LID is already more accurate alone than
/// whisper-tiny alone.
///
/// A candidate shorter than [minProbeSeconds] is presumed non-speech noise
/// and never confirms a switch, even on agreement.
DualConfirmResult resolveDualConfirm({
  required String lang,
  required String? lastLang,
  required double? speechSeconds,
  required String svLang,
}) {
  if (lang == lastLang) {
    return DualConfirmResult(lang, false);
  }
  final tooShort = speechSeconds != null && speechSeconds < minProbeSeconds;
  if (lastLang == null) {
    // no current language to hold at: trust the probe's own judgment
    final resolved = svLang.isNotEmpty ? svLang : lang;
    return DualConfirmResult(resolved, !tooShort && svLang == lang);
  }
  if (tooShort) {
    return DualConfirmResult(lastLang, false);
  }
  if (svLang == lang) {
    return DualConfirmResult(lang, true);
  }
  return DualConfirmResult(lastLang, false);
}

/// Result of [resolveStickyLang].
class StickyLangResult {
  const StickyLangResult(
    this.lang,
    this.suppressFallback,
    this.pendingLang,
    this.pendingCount,
  );

  final String lang;
  final bool suppressFallback;
  final String? pendingLang;
  final int pendingCount;
}

/// Sticky-LID hysteresis: decide whether to accept a new LID detection as
/// a real language switch, or hold the session's current language.
///
/// Ported from the desktop `resolve_sticky_lang` for languages outside
/// SenseVoice's 5-language coverage (mobile currently has no tier for
/// those — see `docs/MOBILE.md` — but this is kept as a faithful, tested
/// port so a future non-SenseVoice tier can reuse it without re-deriving
/// the hysteresis rules).
///
/// A single new-language detection can be a babble-noise misfire or a
/// jingle/SFX blip rather than a genuine switch. [switchConfirm]
/// CONSECUTIVE detections of one new language are required before
/// switching; staying on the current language needs no confirmation.
///
/// [minSwitchSeconds] is the noise filter on each individual candidate
/// detection: a new-language segment shorter than this is presumed
/// non-speech and does NOT advance [pendingCount] at all.
///
/// [bootstrapProbeLang] is the caller's SenseVoice probe result for THIS
/// exact audio (only meaningful at bootstrap): while no candidate has
/// accumulated [switchConfirm] detections, segments decode using
/// [bootstrapProbeLang] instead of blindly trusting the LID's possibly
/// wrong candidate.
StickyLangResult resolveStickyLang({
  required String lang,
  required String? lastLang,
  required double? speechSeconds,
  required double minSwitchSeconds,
  required int switchConfirm,
  required String? pendingLang,
  required int pendingCount,
  String? bootstrapProbeLang,
}) {
  if (lastLang != null && lang == lastLang) {
    return StickyLangResult(lang, false, null, 0);
  }

  // while nothing has been confirmed yet, decode with the best available
  // guess: the established session language, or (at bootstrap) the
  // SenseVoice probe's own judgment if the caller has one
  final fallback =
      lastLang ??
      (bootstrapProbeLang != null && bootstrapProbeLang.isNotEmpty
          ? bootstrapProbeLang
          : lang);

  final isShort = speechSeconds != null && speechSeconds < minSwitchSeconds;
  if (isShort) {
    return StickyLangResult(fallback, true, pendingLang, pendingCount);
  }

  String newPendingLang;
  int newPendingCount;
  if (lang == pendingLang) {
    newPendingLang = pendingLang!;
    newPendingCount = pendingCount + 1;
  } else {
    newPendingLang = lang;
    newPendingCount = 1;
  }

  if (newPendingCount < switchConfirm) {
    // Hold the session language for this segment; it's a genuine-speech
    // candidate (>= minSwitchSeconds) merely decoded under the wrong
    // tier's model, so let a fallback specialist have a shot if that
    // draws a blank.
    return StickyLangResult(fallback, false, newPendingLang, newPendingCount);
  }

  return StickyLangResult(lang, false, null, 0);
}
