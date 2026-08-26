/// Which model tiers a routed live session loads.
///
/// The desktop pipeline (`scripts/asr_engine.py`) has 4 tiers (ja, zh, en +
/// EU langs via Parakeet-TDT-v3, everything else via Omnilingual). Mobile
/// starts with a deliberately smaller catalog: Parakeet-TDT-0.6B-v3-int8 is
/// ~640MB on disk (see `docs/MOBILE.md`'s routing note), well over a
/// phone-friendly budget, so English is served by SenseVoice's own
/// multilingual coverage instead of a dedicated v3 tier. There is
/// currently no third profile distinct from [jaSenseVoice]: SenseVoice
/// already covers en/zh/ko/yue in one model, so a hypothetical "ja+en+SV"
/// profile would load the exact same two models as "ja+SV" — see
/// `docs/MOBILE.md`.
enum RoutingProfile {
  /// Single model: ReazonSpeech ja only, no language routing. This is the
  /// pre-existing mobile behavior (`LiveTranscriber.start` without a
  /// routing profile).
  jaOnly('ja only', dualConfirmed: false),

  /// ReazonSpeech ja (tier 0) + SenseVoice small (tier 1: en/zh/ko/yue),
  /// arbitrated per segment by whisper-tiny LID + SenseVoice's own LID
  /// dual-confirmation (see `lib/routing/lang_routing.dart`,
  /// `docs/LID.md`).
  jaSenseVoice('ja + SenseVoice (en/zh/ko/yue)', dualConfirmed: true);

  const RoutingProfile(this.label, {required this.dualConfirmed});

  final String label;

  /// Whether this profile routes between languages at all (and therefore
  /// needs the LID + SenseVoice models loaded).
  final bool dualConfirmed;
}

/// Languages [RoutingProfile.jaSenseVoice] can resolve a segment to.
/// ReazonSpeech decodes "ja"; SenseVoice decodes everything else in this
/// set. Mirrors the desktop `SV_LANGS ∪ RZ_LANGS` restricted to what
/// mobile actually loads (no zh gets its own Paraformer tier here — see
/// `docs/MOBILE.md` for why that's an acceptable simplification).
const Set<String> jaSenseVoiceLangs = {'ja', 'en', 'zh', 'ko', 'yue'};
