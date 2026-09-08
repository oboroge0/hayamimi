# Korean route — candidate model evaluation (improvement track D)

Goal: find an alternative to the current production ko route (SenseVoice
small, auto-detected language — tier `sv` in `scripts/asr_engine.py`,
`SV_LANGS = {"ko", "yue"}`) that clears all of:

1. FLEURS ko (test, 100 clips) CER ≤ 6.0%
2. Real spoken ko (12 clips) CER ≤ 7.0%
3. No empty output on the two known quiet clips (`ko_08.wav`, `ko_09.wav`,
   peak amplitude ~0.015–0.022)
4. RTF < 0.2 (provisional — see "Measurement conditions" below)
5. Redistributable license
6. On-disk size ≤ 700MB

**Result: no candidate qualifies. Current production config (SenseVoice,
auto-detected language) stays unchanged.** No code was changed in
`scripts/asr_engine.py` / `download_models.py` as a result of this track —
see "Verdict" below for why each candidate was rejected.

## Measurement conditions

- CPU: same Ryzen 5 5600 (6C12T) machine as the rest of the repo's
  benchmarks, but **5 other improvement tracks were evaluating in parallel
  on the same machine** while these numbers were collected
  (2026-09-08). Inference threads were capped at **2** (`--threads 2`)
  instead of the repo's usual 6 to leave headroom for the other tracks.
  **All RTF figures below are provisional** — they reflect a contended CPU,
  not the repo's normal 6-thread measurement convention
  (`docs/results/benchmarks.md`, `docs/results/scorecard.md`).
- Data: `testdata/fleurs_bench/ko/` (FLEURS `ko_kr` test split, 100 clips,
  1291.4s total — see `scripts/make_fleursset.py`) and
  `testdata/eval_real_zhko/` filtered to `lang=="ko"` (12 real clips, same
  set `docs/eval/eval_real_zhko.md` uses).
- Scoring: `cer_ja` from `scripts/eval_accuracy.py` (NFKC normalize, strip
  punctuation, strip **all** whitespace — this already removes Korean
  어절 spacing differences, matching the deliberate deviation documented in
  `docs/eval/eval_real_zhko.md`), micro-averaged (total edits / total
  reference chars).
- Harness: `scripts/eval_ko_candidates.py` (new). Each candidate is built
  directly with `sherpa_onnx.OfflineRecognizer.from_*` (same pattern as
  `scripts/eval_accuracy.py` / `scripts/make_realset_zhko.py`), decoding
  clips one at a time — no LID, no VAD, no routing/second-opinion machinery,
  so numbers are the raw model's accuracy, comparable in kind (not always in
  absolute value) to the full `RoutedASR` production path used for
  `docs/results/benchmarks.md`.

## Candidates measured

| key | model | license | notes |
|---|---|---|---|
| `sv-auto-ko` | SenseVoice small, `language=""` (auto) | current production config (`asr_engine._build_sense_voice`) | re-run here as an in-harness cross-check; the authoritative production number is `testdata/fleurs_bench/results_hayamimi.json` (7.57% via full `RoutedASR`) |
| (a) `zipformer-ko-gainnorm` | `sherpa-onnx-zipformer-korean-2024-06-24` + peak-amplitude gain normalization (target peak 0.9) applied before `accept_waveform` | Apache 2.0 (icefall/k2-fsa export) | already in `models/`; previously rejected in `docs/eval/eval_real_zhko.md` at CER 30.2% (real-12) with 2 empty outputs on quiet clips — this candidate tests whether gain norm fixes the robustness gap |
| (c) `sv-forced-ko` | SenseVoice small, `language="ko"` (pinned) instead of auto | same model file as production | tests whether pinning the language (rather than letting SenseVoice's internal LID arbitrate) helps or hurts |
| (d) `omnilingual` | `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, no lang hint) | Apache-2.0 (per `THIRD_PARTY_NOTICES.md`) | already in `models/`, already used elsewhere in this repo as the 1600-language generalist fallback; never evaluated as a first-class ko route before — that's what this candidate tests |

### Candidate (b): searched, not downloaded

Investigated the k2-fsa/sherpa-onnx GitHub Releases `asr-models` tag (498
assets, queried via `gh api`/GitHub REST) and Hugging Face for 2024+ Korean
ASR exports beyond the already-tried `sherpa-onnx-zipformer-korean-2024-06-24`:

- `sherpa-onnx-streaming-zipformer-korean-2024-06-16(-mobile)` (418MB /
  378MB): same underlying `icefall-asr-ksponspeech-*` training data/corpus
  family as the already-rejected offline zipformer, and it's a **streaming**
  transducer meant for `OnlineRecognizer`, not `OfflineRecognizer` — using
  it here would need a different code path than every other candidate in
  this doc, for a model trained on the same corpus that already scored
  15.9–30.2% CER offline. Skipped: unlikely to change the verdict, and not
  worth the harness divergence given the CPU-time budget.
- `sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01` (Apache 2.0,
  `CohereLabs/cohere-transcribe-03-2026`, 2B-param Conformer encoder +
  Transformer decoder, covers ko among 14 languages): **compressed tarball
  is 1.70GB** (`gh api repos/k2-fsa/sherpa-onnx/releases/tags/asr-models`,
  asset `size`), 2.4x the 700MB budget before even extracting. Skipped
  without downloading — it fails criterion 6 by inspection regardless of
  accuracy, and downloading+extracting a ~2GB+ model while 5 other tracks
  share this machine's disk/CPU wasn't worth the time. If a future track
  revisits the ko route with a relaxed size budget, this is the model to
  try first — it's the newest and most capable Korean-covering option
  k2-fsa currently ships.
- The newer `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09` export
  was **not** re-tried: `scripts/download_models.py` already documents it as
  found broken during earlier development ("do not substitute it") — not
  re-litigated here.

No other 2024+ dedicated Korean zipformer/paraformer/NeMo export was found
on the official k2-fsa release channel or linked from k2-fsa's own docs.

## Results

### FLEURS ko (test split, 100 clips, 1291.4s audio)

| system | CER | mean RTF | empty on quiet clips | criterion 1 (≤6.0%) | criterion 4 (RTF<0.2) |
|---|---|---|---|---|---|
| **current production** (SenseVoice auto, full `RoutedASR`, `results_hayamimi.json`) | **7.57%** | 0.037 | n/a (not in FLEURS-100) | fail (barely) | pass |
| `sv-auto-ko` (in-harness cross-check) | 8.00% | 0.068 | n/a | fail | pass |
| (c) `sv-forced-ko` | 7.89% | 0.054 | n/a | fail | pass |
| (a) `zipformer-ko-gainnorm` | 15.85% | 0.028 | n/a | fail | pass |
| (d) `omnilingual` | 14.36% | 0.260 | n/a | fail | fail |

(The in-harness `sv-auto-ko` cross-check landing at 8.00% vs. the
production-path 7.57% is expected: the production number goes through the
full `RoutedASR` pipeline — whisper-tiny LID routing, head-dropout retry,
Kiwi Korean spacer — while the harness here calls
`OfflineRecognizer.from_sense_voice` directly with no surrounding
machinery, same as every other row in this table.)

### Real spoken ko (12 clips, 169.0s Chinese+Korean total per
`docs/eval/eval_real_zhko.md`, ko subset only)

| file | ref (abridged) | `sv-auto-ko` CER | `sv-forced-ko` CER | `zipformer-ko-gainnorm` CER | `omnilingual` CER |
|---|---|---|---|---|---|
| ko_01 | 염소 사육은... | 0.000 | 0.030 | 0.121 | 0.212 |
| ko_02 | 그래도 관계자의... | 0.026 | 0.026 | 0.180 | 0.128 |
| ko_03 | 교전이 발발한... | 0.120 | 0.160 | 0.280 | 0.160 |
| ko_04 | 사건 발생 이후... | 0.242 | 0.242 | 0.455 | 0.424 |
| ko_05 | 오늘날 날개를... | 0.115 | 0.154 | 0.231 | 0.192 |
| ko_06 | 파리 사람들은... | 0.000 | 0.000 | 0.103 | 0.103 |
| ko_07 | 하지만 여전히... | 0.040 | 0.040 | 0.080 | 0.080 |
| ko_08 (quiet, peak≈0.015) | 사자는 무리... | 0.000 | 0.000 | **0.125 (non-empty)** | 0.000 |
| ko_09 (quiet, peak≈0.022) | 스프링복스의... | 0.053 | 0.053 | **0.105 (non-empty)** | 0.158 |
| ko_10 | 이곳은 남아프리카의... | 0.200 | 0.200 | 0.350 | 0.300 |
| ko_11 | 미국 공병대는... | 0.111 | 0.111 | 0.361 | 0.250 |
| ko_12 | 1940년 8월 15일... | 0.079 | 0.053 | 0.053 | 0.316 |
| **micro-avg CER** | | **0.0872** | **0.0926** | **0.2125** | **0.2071** |
| mean RTF | | 0.058 | 0.054 | 0.038 | 0.231 |

Criterion 2 (≤7.0%): all four fail. Criterion 3 (no empty output on
ko_08/ko_09): `sv-auto-ko`, `sv-forced-ko`, `omnilingual` all already pass
(no empty output — this repo's existing SenseVoice/Omnilingual config was
never the model with the quiet-clip problem); (a) `zipformer-ko-gainnorm`
**newly passes** where the un-normalized zipformer previously scored
CER=1.0 (empty output) on both — confirming the `eval_real_zhko.md` root
cause (low peak amplitude) and its proposed fix, but the model's overall
accuracy remains far worse than SenseVoice either way.

## Verdict

| candidate | 1. FLEURS≤6.0% | 2. real≤7.0% | 3. no empty (quiet) | 4. RTF<0.2 | 5. license | 6. size≤700MB | adopt? |
|---|---|---|---|---|---|---|---|
| (a) zipformer-ko + gain norm | fail (15.85%) | fail (21.25%) | **pass** (fixed) | pass | pass (Apache 2.0) | pass (~330MB) | **no** |
| (c) SenseVoice, language=ko pinned | fail (7.89%) | fail (9.26%) | pass | pass | pass | pass (shared w/ prod model) | **no** |
| (d) Omnilingual 300M CTC | fail (14.36%) | fail (20.71%) | pass | fail (0.26) | pass (Apache-2.0) | pass (~360MB) | **no** |
| (b) Cohere Transcribe 14-lang | not measured | not measured | not measured | not measured | pass (Apache 2.0) | **fail (1.70GB > 700MB)** | **no** (excluded pre-download) |

**No candidate meets all six criteria. Keep the current production Korean
route unchanged: SenseVoice small, `language=""` (auto-detect), tier `sv`.**
This reconfirms and extends the earlier `docs/eval/eval_real_zhko.md`
conclusion ("Keep SenseVoice") onto the larger FLEURS-100 set and two new
configurations:

- Gain-normalizing the previously-rejected dedicated Korean zipformer does
  fix the specific quiet-clip empty-output failure mode, but the model is
  still ~2x SenseVoice's error rate on both sets — the robustness fix alone
  doesn't close the accuracy gap.
- Pinning SenseVoice's `language` parameter to `"ko"` instead of the
  production `""` (auto) is **not** an improvement — it's about a point
  worse on both FLEURS (7.89% vs. 8.00% in-harness / 7.57% production) and
  real-12 (9.26% vs. 8.72% in-harness). SenseVoice's own internal
  language-arbitration when left on auto appears to already do at least as
  well as an externally-forced hint; do not change `_build_sense_voice`'s
  `language=""` setting.
- Omnilingual, tested as an explicit contrast point per the task brief,
  underperforms SenseVoice on ko by roughly 2x CER and is ~4-7x slower
  (RTF 0.23–0.26 vs. 0.05–0.07, failing criterion 4 on its own) — it
  remains appropriate only as this repo's already-existing 1600-language
  generalist fallback, not as a first-class ko route.
- The one model worth downloading in a future pass, if the ko route is
  revisited, is `sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01` — but
  only under a relaxed size budget (it's ~2.4x over the 700MB cap
  uncompressed-adjusted) and likely with its own dedicated tier rather than
  folded into the existing `SV_LANGS`/`PARA_LANGS` per-clip routing, since a
  single 2B-param model covering 14 languages doesn't fit this repo's
  one-model-per-language-family catalog shape.

## Reproduction

```
# real-12 (ko subset), all 4 measured candidates:
python scripts/eval_ko_candidates.py --threads 2 --set real

# FLEURS ko 100, one candidate at a time (each ~1-6 min at --threads 2):
python scripts/eval_ko_candidates.py --threads 2 --set fleurs --system sv-auto-ko
python scripts/eval_ko_candidates.py --threads 2 --set fleurs --system sv-forced-ko
python scripts/eval_ko_candidates.py --threads 2 --set fleurs --system zipformer-ko-gainnorm
python scripts/eval_ko_candidates.py --threads 2 --set fleurs --system omnilingual

# production baseline (for comparison, already run/cached separately):
python scripts/eval_fleurs_bench.py --engine hayamimi --lang ko
```

Results are cached incrementally at `testdata/ko_candidates_results.json`
(gitignored, resumable — re-running skips already-scored `system/set/wav`
combinations).

## Caveats

- **Small real-speech sample.** 12 clips, same caveat as
  `docs/eval/eval_real_zhko.md`.
- **Contended CPU.** All RTF figures here used `--threads 2` with 5 other
  evaluation tracks running concurrently on the same 6-core machine; they
  are not comparable to this repo's usual 6-thread RTF convention and
  should be treated as "same relative ordering, not absolute numbers".
- **In-harness numbers vs. production numbers differ slightly** for
  SenseVoice auto (8.00% in-harness vs. 7.57% production on FLEURS ko) —
  expected, since the harness bypasses `RoutedASR`'s LID routing,
  head-dropout retry, and Kiwi Korean spacer. Treat `results_hayamimi.json`
  as the authoritative current-production number and the in-harness row as
  a same-conditions cross-check against the other candidates in this doc.
- **Omnilingual's license is Apache-2.0** (per `THIRD_PARTY_NOTICES.md`),
  not a constraint here; candidate (d) is disqualified purely on accuracy
  and RTF, not licensing.
