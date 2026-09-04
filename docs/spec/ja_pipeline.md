# Japanese route implementation specification (JA_PIPELINE)

日本語版: [ja_pipeline.ja.md](ja_pipeline.ja.md)

## Purpose and scope

This document specifies hayamimi's Japanese route precisely enough that
another project can rebuild it in a different language. The intended
reimplementation target is a **C++ Godot GDExtension**: it links the
binaries sherpa-onnx ships for speech recognition and VAD as-is, and drives
the punctuation model directly through the ONNX Runtime that sherpa-onnx
bundles.

Every number and procedure in this document is taken from either the
implementation under `scripts/` or a measurement already recorded in this
repository. Nothing here is a guess. Anything not yet measured is listed
separately under "Not yet verified."

The scope is only the `--mode single --lang ja` route. hayamimi itself has
5-layer language routing (`docs/eval/lid.md`), but a reimplementation that
only handles Japanese needs none of it. When `RoutedASR.forced_lang` is set
to `"ja"`, language identification (LID), language-switch confirmation,
zh/yue arbitration, and script-based re-decoding are all skipped wholesale
inside `transcribe()` (see the top of `transcribe()` in
`scripts/asr_engine.py` and the `if self.forced_lang is not None:` branch
inside it). So a Japanese-only implementation only needs to provide three
models: the recognizer, the VAD, and the punctuation restorer.

Terms are defined once here:

- **VAD** (voice activity detection): the process of cutting speech
  segments out of audio. This route uses Silero VAD through sherpa-onnx.
- **Pre-roll**: prepending **real audio** from before the onset the VAD
  detected to the front of the buffer handed to the recognizer. The main
  countermeasure for the head-dropout problem described below.
- **ITN** (inverse text normalization): converting spoken-form numerals
  back to written form. Here that means turning kanji numerals into Arabic
  digits (`千九百四十年` [*sen kyūhyaku yonjū nen*, "the year 1940"] →
  `1940年`).
- **final / refine**: `final` is the immediate captioned text emitted for
  each VAD segment; `refine` is a clean rewrite produced by re-decoding the
  whole utterance group afterward.
- **CER** (character error rate): edit distance to the reference text,
  divided by the reference text's length.

## Model inventory

The Japanese route loads 7 files. The sha256 and byte counts are computed
from the actual files by `scripts/dump_ja_config.py --with-models`; the
machine-readable form is the `models` block of
`docs/spec/ja_pipeline_spec.json`.

| Role | File (under `models/`) | Bytes | sha256 | License / source |
|---|---|---:|---|---|
| Recognizer encoder | `sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/encoder-epoch-35-avg-1.int8.onnx` | 70,876,409 | `ead1579e118b821a767242a8eb9272634b0e63ba16f8dfc4d126732406eae268` | Apache-2.0 / Reazon Human Interaction Lab, [reazon-research/reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2), packaged by k2-fsa/sherpa-onnx |
| Recognizer decoder | `.../decoder-epoch-35-avg-1.int8.onnx` | 1,308,690 | `d0179db78a2e65445c5c3dc41e94c62068fc539fe4e45060e32f438cca76432f` | same as above |
| Recognizer joiner | `.../joiner-epoch-35-avg-1.int8.onnx` | 1,033,417 | `c7f4ba40a8ae307a6c30b5c06e2570add04466bcb45bab62699f0ec5d00ed495` | same as above |
| Recognizer tokens | `.../tokens.txt` | 26,631 | `144f8a4f639373a1bdf7eabb2437482ef64b0cc5db24ad27cce65f293e4faa24` | same as above |
| VAD | `silero_vad.onnx` | 643,854 | `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6` | MIT / [snakers4/silero-vad](https://github.com/snakers4/silero-vad), packaged by k2-fsa/sherpa-onnx |
| Punctuation model (fp32) | `mojicast-punct-onnx/punct_bert.onnx` | 363,501,157 | `4ed3c28ede4792526c6abab9101a3b6c304ab09fd0bda4e318e0acb2b7008e63` | Apache-2.0 / base model [tohoku-nlp/bert-base-japanese-char-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v3) + head [bobfromjapan/bert_japanese_punctuation](https://huggingface.co/bobfromjapan/bert_japanese_punctuation), ONNX export [ishiki-emo/mojicast-punct-onnx](https://huggingface.co/ishiki-emo/mojicast-punct-onnx) |
| Punctuation vocab | `mojicast-punct-onnx/vocab.txt` | 27,928 | `57411bcac5e9559f2aa4d316a2217289048cb40fe23187b02a81aeb3e5d61cf3` | same as above |

**Total 437,418,086 bytes (417.2 MiB).** The punctuation model alone is 83%
of that.

### Do not use the upstream INT8 punctuation model

Hugging Face's `ishiki-emo/mojicast-punct-onnx` lists `punct_bert.int8.onnx`
(dynamic INT8) as its recommended default. hayamimi tried it and **confirmed
it is broken**. On onnxruntime 1.29.0's CPU EP, feeding either random token
sequences or real sentences produces logits that stay nearly constant (the
comma/period probabilities sit in the 0.31–0.36 range everywhere), i.e. the
output does not respond to the input at all — no punctuation is restored.
The cause has not been identified (a broken export, an upload mistake, or
incompatibility with a specific onnxruntime build are all possible). Details
are in `docs/design/punct_ja.md`.

**Do not use this file in a reimplementation.** Use either fp32
(`punct_bert.onnx`) or the fp16 variant below.

### Choosing fp32 vs. fp16

hayamimi re-quantized the model itself into three variants and scored them
on FLEURS ja, 250 sentences (seed 0), with `scripts/quantize_punct.py`. The
pre-declared pass condition was "F1 within −0.02 of fp32."

| Variant | Size | P | R | F1 | Verdict |
|---|---:|---:|---:|---:|---|
| fp32 (`punct_bert.onnx`) | 363.5 MB | 0.8724 | 0.4831 | 0.6218 | baseline |
| Dynamic INT8 | 91.4 MB | 0.9638 | 0.3757 | 0.5407 | **fail** (−0.0812) |
| Static INT8 (QDQ, calibrated) | 91.3 MB | 0.9275 | 0.3432 | 0.5010 | **fail** (−0.1208) |
| **fp16** | **181.8 MB** | 0.8724 | 0.4831 | **0.6218** | **pass (diff ±0.0000)** |

fp16 is a full-graph conversion via
`onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`,
so the inputs and outputs stay int64/float32. On both this 250-sentence set
and `eval_real`, its predictions did not differ from fp32's **on a single
example**. Both INT8 variants broke the same way — precision goes up while
recall collapses — which is read as a property of quantizing this BERT-char
head's threshold-adjacent decision boundary, not a problem with the
quantization recipe.

Decision:

- **Use fp32 on PC (desktop).** There's no size constraint, so there's no
  reason to convert. This is `scripts/punct_ja.py`'s default.
- **Use fp16 wherever memory or package size is constrained.** Accuracy is
  identical to fp32, size is halved. That brings the total down to
  255,720,140 bytes (243.9 MiB). The Android emulator verification below
  runs on fp16.
- There's a caveat on fp16 speed: on this Windows/x86 machine, fp16 is
  **10x slower** (`restore()` averages 532 ms vs. fp32's 52–78 ms). x86's
  CPU EP has no native fp16 compute path and casts to fp32 for every
  operator and back; ARM chips with NEON fp16 are a different story. Do not
  extrapolate this slowdown to ARM until it's measured on real hardware.

## Recognizer configuration

`_build_reazon()` in `scripts/asr_engine.py` is the single construction
point. Every value passed to sherpa-onnx's
`OfflineRecognizer.from_transducer` is listed here, and maps one-to-one to
C++'s `OfflineRecognizerConfig`.

| Field | Value | Source |
|---|---|---|
| `model_type` | `"zipformer"` | `asr_engine.RZ_MODEL_TYPE` |
| `decoding_method` | `"modified_beam_search"` | `asr_engine.RZ_DECODING_METHOD` |
| `modeling_unit` | `"cjkchar"` | `asr_engine.RZ_MODELING_UNIT` |
| `num_threads` | `4` | CLI `--threads` default |
| `hotwords_file` | `""` (unused) | `_build_reazon`'s default |
| `hotwords_score` | `2.0` | `asr_engine.RZ_HOTWORDS_SCORE` |
| feature `sampling_rate` | `16000` | sherpa-onnx `FeatureExtractorConfig` default |
| feature `feature_dim` | `80` | same (80-dim fbank) |

Feature extraction is left at sherpa-onnx's defaults, so the Python side
specifies nothing for it. But if you're assembling `OfflineRecognizerConfig`
by hand in C++, you'll need to state it explicitly, so it's recorded here:
**16 kHz, 80-dimensional log-mel filterbank.** Input audio must also be
16 kHz mono (`realtime_transcribe.read_wave()` linearly resamples anything
at a different rate).

### Why `modified_beam_search`

The switch from `greedy_search` is measurement-based. On real broadcast
Japanese, CER went from **8.6% to 5.8%**, at a decode-time cost of +25%
(still 37x real-time). The English v3 tier showed no improvement, so it's
left on greedy (see the comment in `_build_reazon`).

### Hotwords are effectively unusable

`--hotwords` is accepted as an API parameter, but it has no effect with
this recognizer. ReazonSpeech's `tokens.txt` is byte-level BPE, and
`bpe.model` is not bundled with it. So hotwords cannot be encoded under any
`modeling_unit`, `cjkchar` included. sherpa-onnx only reports the encoding
failure as an stderr warning and exits 0 — captions still come out normally
— so this can go unnoticed while doing nothing at all (see sherpa-onnx
GitHub issue #1). hayamimi checks encodability against `tokens.txt` at
startup and, when it can't, emits a prominent warning plus a `warning`
event (`code: "hotwords_unencodable"`)
(`RoutedASR._warn_hotwords_encodability`).

**Guidance for a reimplementation**: handle proper nouns with the
downstream replacement dictionary (`--replace`; step 10 under "The
`forced_lang="ja"` route" below), not hotwords. If hotwords are exposed in
a UI, either document this limitation or don't expose them at all.

## VAD configuration

`build_vad()` in `scripts/realtime_transcribe.py` is the single
construction point.

| Field | Value | Constant |
|---|---|---|
| Model | `silero_vad.onnx` | `VAD_MODEL` |
| `threshold` | `0.5` | `build_vad(vad_threshold=...)`'s default |
| `min_silence_duration` | `0.35` s | `build_vad(min_silence=...)`'s default |
| `min_speech_duration` | `0.25` s | `VAD_MIN_SPEECH_S` |
| `max_speech_duration` | `12.0` s | `build_vad(max_speech=...)`'s default |
| `window_size` | `512` samples (≈32 ms at 16 kHz) | `WINDOW_SIZE` |
| `sample_rate` | `16000` | `SAMPLE_RATE` |
| `buffer_size_in_seconds` | `30.0` | `VAD_BUFFER_S` |
| `num_threads` | `1` | `VAD_NUM_THREADS` |

Audio is fed to `accept_waveform()` in chunks of exactly 512 samples; after
each chunk, `empty()` is checked and any accumulated segment is drained via
`front` / `pop`.

Rationale for each value:

- **`min_silence` 0.35 s**: swept 0.5 → 0.35 → 0.30 on 15 real-broadcast ja
  clips; CER did not change (segment count rose 17 → 19). Since the
  accuracy cost is zero, 0.35 was adopted because it makes finals appear a
  uniform 150 ms sooner (`docs/results/benchmarks.md`, improvement
  iteration #9).
- **`threshold` 0.5**: swept 0.40 / 0.30 / 0.20 on the AMI evaluation set.
  Misses dropped as expected, but confusion rose more than that, and
  average DER got worse at every value tried (14.1% → 15.6–16.5%). Lowering
  it was rejected; Silero's own default of 0.5 is kept
  (`docs/design/diarization.md`, section 13).
- **`max_speech` 12 s**: added because unbroken commentary produced 21-second
  segments that delayed finals. However, an Android emulator measurement
  found that sherpa-onnx treats this as a "hint" rather than a hard split —
  with it set to 5.0 s, a 6.134-second segment still came out. It cannot be
  relied on as a latency ceiling.

**sherpa-onnx 1.13.6's `SileroVadModelConfig` has no field for padding
around a speech segment** (there's no equivalent to `speech_pad_ms`). So
"pad the onset inside the VAD" is not an option with this stack, which is
why the countermeasure in the next section is needed.

## Head dropout and mitigations

The rates in this section are transcribed from the measurements in
`docs/eval/head_dropout.md`; see that document for the measurement
procedure, per-clip breakdown, and the determinism check.

### The symptom

When an utterance starts right at the beginning of the buffer (sample 0),
this ReazonSpeech zipformer sometimes drops its leading tokens. It's not
just dropping — the leading word can also get misrecognized as another
word entirely.

Concrete cases observed inside hayamimi:

- While building the offline-splitting code, a fragment that started
  exactly at an utterance's onset lost `東京の` (*Tōkyō no*, "Tokyo's")
  entirely from `東京の天気は晴れです` (*Tōkyō no tenki wa hare desu*,
  "Tokyo's weather is sunny") — see the comment in `_speech_pieces()`.
- During live-route verification, setting pre-roll to 0 turned
  `資料は昨日送りました` (*shiryō wa kinō okurimashita*, "I sent the
  materials yesterday") into `昨日は昨日送りました` (*kinō wa kinō
  okurimashita*, "yesterday sent yesterday" — nonsensical). The same
  observation recurs independently in three places: the desktop Python
  route (`docs/results/benchmarks.md`, 2026-09-01 "measured a suspected
  live-route head-drop"), the Android emulator (see below), and the
  condition grid in `docs/eval/head_dropout.md` (both `greedy`/no-mitigation
  and `beam`/no-mitigation produce `昨日は昨日送りました`; every condition
  with a prepended segment produces `資料は昨日送りました`).
- The same thing reproduced on the Android emulator. With zero pre-roll:
  `昨日は昨日送りました` / `会議は十時からです` (*kaigi wa jūji kara desu*,
  "the meeting is from 10 o'clock" — with `あしたの`, "tomorrow's", missing).
  With 1.0-second pre-roll, both matched the desktop output:
  `資料は昨日送りました` / `あしたの会議は十時からです`
  (`docs/verify/android_emulator.md`, branch `agent/feature/core-release`,
  "Android emulator verification, run 2"). That run was a controlled
  experiment: a pass that fixed only the VAD values (`r2-preroll0`) and a
  pass that fixed both (`r1-defaults`) ran separately, isolating which fix
  produced which effect.

Silero's speech-onset detection is also measured to lag behind the true
onset: **198 ms late** on the first sentence of the fixture above, within
±25 ms on the second and third. But the breakage doesn't scale with the
lag — a segment with only 18 ms of lag still had its leading word garbled.
In other words, "compensating for VAD lag" is not what's working here —
"putting something in front of the onset" is.

### Mitigation 1 (primary): 1.0-second real-audio pre-roll

hayamimi's standard mitigation. `PREROLL_S = 1.0` seconds.

`realtime_transcribe.AudioHistory` keeps the last 30 seconds of input audio
in a ring buffer, and `with_preroll(seg_start, seg_samples)` prepends up to
1.0 seconds of **real audio** to a VAD segment before it reaches the
recognizer. The amount added is clamped three ways:

```
want = max(seg_start - PREROLL_S * sr,   # never reach back more than 1 s
           self.last_seg_end,            # never overlap the previous segment
           self.offset)                  # never exceed what history still holds
```

The `last_seg_end` clamp matters: without it, the tail of the previous
utterance would be pulled in and decoded twice. The amount actually
prepended is situational — the Android measurements saw 0.198 / 0.356 /
0.420 seconds (not the full 1.0 second).

The effect is large. On 15 real-broadcast ja clips, VAD-path CER went from
**40.2% with no pre-roll to 15.5% with 0.8-second pre-roll**
(`docs/results/benchmarks.md`, improvement iteration #9 — at that time pre-roll
was 0.8 s; the current implementation uses 1.0 s). Decoding an entire clip
offline in one pass (the reference ceiling) gives 8.6%, so the remaining gap
of 15.5% − 8.6% is context loss from streaming segmentation itself — a
separate problem.

This invariant is pinned by
`tests/test_asr_segment.py::test_live_path_preroll_keeps_utterance_initial_words`,
which runs a 3-sentence fixture through the real VAD → pre-roll → decode
path and requires every sentence's leading word to survive.

### Mitigation 2 (secondary): retry-on-suspicion split

Added in v0.3.1. `_looks_truncated()` and `_split_retry()`
(`scripts/asr_engine.py`).

Splitting a long buffer at its internal silences and re-decoding it one
utterance at a time fixes clips whose head was dropped (FLEURS ja clip 15:
CER 0.67 → 0.11). But applying it **unconditionally is a net loss overall**:
on an external FLEURS 5×100 A/B test it made things worse — ja 8.6% → 9.9%,
en 9.4% → 10.2%, ko 8.1% → 9.1% (for ja, 16 clips improved and 26 got worse;
words that straddle a fragment boundary are dropped, and shorter fragments
give the decoder worse conditions to work with).

So it's applied only **on retry**. The whole buffer is decoded as before,
and splitting is only attempted when that result looks like its head was
dropped, and the split result is only adopted when it's clearly better.

- Suspicion test (`_looks_truncated`): a segment is suspected when its
  alphanumeric-character density per second of speech (punctuation not
  counted) is below `DENSITY_FLOOR_CJK = 2.4`. Measured over 60 FLEURS ja
  clips, density ranged 3.46–14.22 (median 6.54); a known head-dropped clip
  measured 1.70. 2.4 sits at the log-midpoint of that gap. Non-CJK uses
  `DENSITY_FLOOR_LATIN = 6.0`. A buffer of `SEGMENT_MIN_S = 4.0` seconds or
  less is never suspected. An empty string is never suspected either.
- Splitting (`_speech_pieces`): uses the same Silero VAD, cutting at
  silences of `SEGMENT_MIN_SILENCE_S = 0.35` seconds or more —
  **deliberately the same value** as the live VAD's `min_silence`. Because
  the value matches, a segment the live VAD already produced can never be
  split further, which automatically disables retry for it.
- Building a fragment: each fragment is `SEGMENT_PAD_S = 0.35` seconds of
  **zero-valued samples**, plus the VAD span extended by 0.35 seconds of
  **real audio** on each side, plus another 0.35 seconds of zero-valued
  samples. Synthesizing the leading silence is needed because real audio
  alone can't fully absorb the VAD's lag, and a fragment could still start
  right at an utterance's onset.
- Adoption test (`_retry_is_better`): the retry result is adopted only when
  it is (a) longer than the original with density back in the healthy
  range, **and** (b) at least `RETRY_TAIL_MATCH = 0.6` of the original
  text's last `RETRY_TAIL_CHARS = 12` characters appear in the retry result
  (as a longest common substring). The reasoning: the original decode is
  the utterance's **surviving tail**, so a genuine recovery should still
  contain it. When in doubt, keep the original.

**This doesn't help the live route** (and doesn't need to). The live route
decodes one VAD segment at a time, so there's no internal silence to split
on, and segments are short and dense enough that they never pass the
suspicion gate. It's a safety net for the offline case, where a whole file
is fed through in one pass.

### Alternative: prepending 300 ms of silence

Another project observed the same phenomenon with `greedy_search` and works
around it by **prepending 300 ms of silence** to the buffer's start. Their
VAD doesn't retain audio from before the speech onset, so real-audio
pre-roll wasn't structurally possible for them, and synthetic silence was
the only option.

**hayamimi treats pre-roll (1.0 s of real audio) as the standard.** The
measurements in the next section show that prepending silence also
prevents most dropouts, but doesn't reach real audio's effectiveness. If an
implementation has real audio available, use pre-roll; if it doesn't, use
silence prepending. Note that hayamimi also uses synthetic silence for the
offline-split fragments above (the `SEGMENT_PAD_S` described earlier) —
real audio and zero samples aren't mutually exclusive, and some situations
use both.

### Measured incidence

The full write-up is in `docs/eval/head_dropout.md`; only the numbers
needed for the spec decision are transcribed here.

**Measurement conditions (common to every number below)**: FLEURS ja test
split, 100 clips (read speech, 16 kHz mono, CC BY 4.0). For each clip, only
**the first VAD span of 1.0 second or longer** is measured. VAD is
identical to the live route (Silero, threshold 0.5, min_silence 0.35 s,
512-sample window). CPU is an AMD Ryzen 5 5600 / Windows 11, `--threads 4`,
sherpa-onnx 1.13.6. Scoring is character-level Levenshtein after
`eval_accuracy.normalize_ja` (NFKC → strip punctuation → strip whitespace),
with the reference's tail left as a free end.

**Head-dropout definition**: in the alignment, **2 or more reference
characters were deleted** before the first matching character, **and** at
least **60% of the hypothesis's characters** match the reference. The
second condition exists so a wholesale bad hypothesis doesn't get counted
as a dropout — note the denominator is the hypothesis side, not the
reference side (using the reference side would make bigger dropouts look
like a *higher* match rate, turning the worst dropouts into "ordinary
errors"). `strict` uses the same alignment with 1 or more leading deletions.

**Definition of the onset-resolvable subset (n=90)**: only clips where **at
least one condition** reached the reference's first character. The 10
excluded clips are ones where Silero missed the speech onset itself under
every condition, so the same leading character is missing regardless of
condition, which would uniformly inflate every raw number. The subset is
defined as the **union** across all conditions, so it doesn't favor any one
of them.

#### Onset-resolvable subset (n=90) — read the condition comparison here

| Condition | Head dropouts | strict | general errors | mean CER | mean ms |
|---|---:|---:|---:|---:|---:|
| `greedy` / no mitigation | 29 | 36 | 5 | 0.1944 | 142 |
| `greedy` / pre-roll 1.0 s | 1 | 4 | 1 | 0.0923 | 184 |
| `greedy` / silence 300 ms | 3 | 6 | 1 | 0.0986 | 170 |
| `greedy` / silence 1.0 s | 2 | 6 | 1 | 0.0920 | 179 |
| `beam` / no mitigation | 27 | 34 | 5 | 0.1879 | 207 |
| **`beam` / pre-roll 1.0 s** | **0** | 5 | 1 | 0.0860 | 228 |
| `beam` / silence 300 ms | 4 | 7 | 1 | 0.1028 | 204 |
| `beam` / silence 1.0 s | 2 | 6 | 1 | 0.0989 | 226 |
| **production** (beam + pre-roll + split-retry + ITN, punctuation stripped for scoring) | **0** | 5 | 0 | 0.0554 | 272 |

#### All 100 clips (raw counts run higher due to the 10 clips above)

| Condition | Head dropouts | strict | general errors | mean CER | mean ms |
|---|---:|---:|---:|---:|---:|
| `greedy` / no mitigation | 32 | 40 | 11 | 0.2015 | 135 |
| `greedy` / pre-roll 1.0 s | 7 | 12 | 3 | 0.1100 | 177 |
| `greedy` / silence 300 ms | 8 | 12 | 5 | 0.1094 | 163 |
| `greedy` / silence 1.0 s | 8 | 13 | 4 | 0.1088 | 172 |
| `beam` / no mitigation | 30 | 38 | 11 | 0.1956 | 198 |
| `beam` / pre-roll 1.0 s | 6 | 13 | 3 | 0.1042 | 218 |
| `beam` / silence 300 ms | 9 | 13 | 5 | 0.1128 | 195 |
| `beam` / silence 1.0 s | 8 | 13 | 4 | 0.1150 | 217 |
| `production` | 6 | 13 | 2 | 0.0761 | 259 |

#### Takeaways

1. **The decoding method doesn't protect the onset.** With no mitigation,
   greedy drops 29/90, beam drops 27/90. Switching to
   `modified_beam_search` doesn't change the dropout rate. The phenomenon
   the other project observed with `greedy_search` shows up the same way
   under beam search — this should be specified as **onset handling, not a
   decoder choice.**
2. **Prepending anything helps, and by a lot.** 27–29/90 (about 30%) drops
   to 0–4/90. Mean CER also roughly halves, from ~0.19 to ~0.09.
3. **Real-audio pre-roll beats silence prepending.** For beam: pre-roll
   0/90, silence 300 ms 4/90, silence 1.0 s 2/90. But this gap is 2–4 clips
   out of 90 — not large enough to claim statistical significance.
4. **The cost is about +10%.** For beam: 207 ms with no pre-roll, 228 ms
   with it (per span). The extra time is consistent with just the longer
   segment length, nothing more. Note that **run-to-run variance for this
   measurement is the same order of magnitude**, so a ms difference under
   10% shouldn't be read as a real effect.
5. **Split-retry never fired under this condition**
   (`split_retry_called` = 0/100). A 2–5 character head drop doesn't push character density below
   `_looks_truncated`'s floor, so **the production row's 0/90 is pre-roll
   working alone.** Split-retry cannot be specified as a head-dropout
   mitigation.
6. `production`'s lower CER vs. raw `beam/pre-roll` (0.0554 vs. 0.0860) is
   from CJK ITN, a separate effect from head dropout.

The same pattern shows up in `testdata/multi_sentence_ja.wav`. With `beam` /
no mitigation, the second sentence comes out `会議は十時からです`
(`あしたの` missing), the third `昨日は昨日送りました`
(`資料は` misrecognized) — and **any of** pre-roll, silence 300 ms, or
silence 1.0 s restores `あしたの会議は十時からです` /
`資料は昨日送りました`.

#### Conclusion as spec

- **The Japanese route's onset handling standard is real-audio pre-roll**
  (up to 1.0 second of real audio prepended before the VAD's detected
  onset).
- **Prepending silence (300 ms or 1 s) is a weaker but workable
  alternative** for an implementation that doesn't retain audio from before
  the VAD onset. It is not equivalent to pre-roll (0 vs. 2–4 out of 90),
  but that difference is not claimed to be significant.
- **The decoder choice doesn't protect the onset.** Choosing
  `greedy_search` vs. `modified_beam_search` is an accuracy/speed decision,
  not an onset mitigation.

### Reproduction procedure

Uses `testdata/multi_sentence_ja.wav` (3 sentences synthesized with
edge-tts, 0.5 s apart, 6.26 s total, 16 kHz mono, speech starts with no
lead-in silence).

Correct output with the current configuration:

```
python scripts/realtime_transcribe.py --wav testdata/multi_sentence_ja.wav \
    --no-realtime --mode single --lang ja --threads 4
```

```
[ja/rz] 東京の天気は晴れです。       (seg=1.8s)
[ja/rz] あしたの会議は十時からです。  (seg=2.4s)
[ja/rz] 資料は昨日送りました。        (seg=2.1s)
```

Setting `realtime_transcribe.PREROLL_S` to 0 and running the same command
garbles the third sentence into `昨日は昨日送りました`. This is the proof
that mitigation 1 is doing its job.

Feeding the same file through in a single pass, with no VAD, returns only
`資料は昨日送りました` — one sentence. This holds on both Windows x86 and
Android x86_64: **this recognizer, given audio with multiple utterances,
only returns the last one.** VAD-based splitting is required for
correctness here, not merely an optimization.

To re-run the incidence table above:

```
python scripts/eval_head_dropout.py --limit 100 --threads 4        # the table above
python scripts/eval_head_dropout.py --limit 20 --threads 4 --determinism
python scripts/eval_head_dropout.py --multi-sentence --threads 4   # 3 spans x every condition
```

Per-clip hypotheses, leading-character deletions, CER, and timing are in
`docs/eval/head_dropout_results.json` (same PR as `docs/eval/head_dropout.md`).

## The `forced_lang="ja"` route

The call order from audio input to a finalized text. The location in the
implementation is in parentheses.

1. **Convert to 16 kHz mono float32** (`read_wave` / mic input).
2. **Feed the VAD 512 samples at a time** (`run_stream`). Simultaneously,
   `AudioHistory.push()` accumulates it into the ring buffer.
3. **Drain any accumulated segment** (`drain_segments`: `vad.empty()` →
   `vad.front` → `vad.pop()`).
4. **Prepend the pre-roll** (`AudioHistory.with_preroll`). This is where the
   audio actually handed to the recognizer is decided.
5. **Decode with zipformer** (`RoutedASR.transcribe` → `_route("ja")` →
   `_decode`). Because `forced_lang` is set, neither LID nor language
   switching runs.
6. **Suspicion check and split retry** (`_looks_truncated` →
   `_split_retry`). Effectively a no-op on the live route. **Optional.**
7. **Second-opinion gate** (`_maybe_second_opinion`). Re-decodes with
   parakeet-ja and adopts its result when the two decoders' mutual CER is
   0.25 or less. **Disabled by default** (`ja_second_opinion=False`), and
   applies only to the refine pass even when enabled. Adds one model and
   about 250 MB RSS. **Optional.**
8. **CJK ITN** (`itn_cjk.convert(text, "ja")`). Kanji numerals → Arabic
   digits.
9. **Punctuation restoration** (`punct_ja.PunctuatorJa.restore`).
10. **User replacement dictionary** (`RoutedASR._replace`). Applies
    `wrong=right` pairs supplied via `--replace`. **Always last**, so it can
    override anything ITN or punctuation produced. **Optional.**

The 8 → 9 → 10 order is fixed; the module docstring of `scripts/itn_cjk.py`
gives the reasoning. ITN runs first so punctuation restoration sees
already-normalized digits (this matters for decimal points). The
replacement dictionary runs last so the user's own instruction always wins.

### Which stages are required vs. optional

| Stage | Required? | What breaks if dropped |
|---|---|---|
| VAD splitting | **Required** | Given multi-utterance audio, only the last sentence is returned |
| Pre-roll | **Effectively required** | On real broadcast ja, CER goes 15.5% → 40.2% |
| zipformer decode | **Required** | — |
| Split retry | Optional | A long offline buffer can lose its opening sentence |
| Second opinion | Optional (off by default) | Slightly lower accuracy, one fewer model's memory |
| ITN | Optional | Kanji numerals stay as-is (`千九百四十年`) |
| Punctuation restoration | Optional | Plain text with no punctuation |
| Replacement dictionary | Optional | No way to fix proper-noun errors |

### Minimal C++-side composition

What a Japanese-only reimplementation needs at minimum:

- One sherpa-onnx `OfflineRecognizer` with the configuration above.
- One sherpa-onnx `VoiceActivityDetector` with the configuration above.
- An audio ring buffer that can hold 1.0 second (16000 samples at 16 kHz),
  plus one variable remembering "the previous segment's end." The
  implementation of `AudioHistory` is about 30 lines, so it can be ported
  as-is.
- One ONNX Runtime session for the punctuation model (next section).
- ITN is pure string processing with no dependencies. `scripts/itn_cjk.py`
  is 239 lines, zero external dependencies, and every regex quantifier is
  bounded (no catastrophic backtracking), so it can be ported directly.

Split retry and second opinion don't need to be included from the start.

## C++ reimplementation spec for punctuation restoration

The complete specification of `PunctuatorJa` in `scripts/punct_ja.py`. A
Dart port exists on the `agent/feature/core-release` branch, so reading it
before writing C++ is worthwhile (see "Reference implementation" below).

### Model input/output

Char-level BERT token classification.

| | Name | Type | Shape |
|---|---|---|---|
| Input 1 | `input_ids` | int64 | `[1, seq]` |
| Input 2 | `attention_mask` | int64 | `[1, seq]` (all 1s) |
| Output | `logits` | float32 | `[1, seq, 2]` |

`token_type_ids` is not needed (the model doesn't require it). No batching
either — one string, one call.

`logits[0][i]` has column 0 for the **comma (、)** logit and column 1 for
the **period (。)** logit. Position `i` means "**immediately after**
`input_ids[i]`." Since `input_ids[0]` is `[CLS]`, the logit corresponding
to character `chars[i]` is `logits[0][i+1]`. Getting this off by one
produces the subtle failure mode of punctuation shifting by one character.

The fp16 model is also converted with `keep_io_types=True`, so its
input/output types don't change. If the output's element type isn't
float32, you've grabbed **a different export** — treat that as an error
instead of reading it as float32 anyway (the Dart port does this).

### Vocabulary

`vocab.txt` is a standard BERT vocab where **the line number is the token
ID** (0-indexed). Strip only the trailing newline; don't process it
otherwise. The 4 special tokens needed are `[PAD]` `[UNK]` `[CLS]` `[SEP]`,
all looked up from the vocab (never hardcode their IDs). Characters not in
the vocabulary fall back to `[UNK]`.

### Tokenization (MeCab is not needed)

The original `BertJapaneseTokenizer` pipeline is "NFKC normalize → split
into morphemes with MeCab → split each morpheme into individual
characters." `scripts/punct_ja.py` reproduces that with fugashi +
unidic-lite. But since **every morpheme gets split back into individual
characters** right afterward, MeCab's only observable effect is **dropping
whitespace**.

This has been verified. `scripts/make_punct_fixture.py` cross-checks the
rule

```
NFKC normalize → strip whitespace characters → split into code points
```

against the MeCab-based version, and found **0 disagreements across 102
texts**: FLEURS ja reference sentences (46, with and without their
punctuation stripped), plus 10 synthetic cases (empty string,
whitespace-only, punctuation-only, half-width katakana, full-width ASCII,
strings containing `モーニング娘。` [a band name ending in a literal
period], etc.). The 15 ja sentences of `eval_real` also match.

**So a C++ implementation does not need MeCab.** The 3-step rule above is
sufficient. After porting, run the same cross-check once and confirm 0
disagreements yourself.

`[CLS]` + the string + `[SEP]` becomes the final `input_ids`. The string is
truncated at **500 characters** (`PUNCT_MAX_CHARS`). The model's position
embeddings go up to 512, and 500 leaves room for the 2 special tokens.
Anything beyond that is **silently discarded**, with no chunking — this is
fine in practice since calls are made per 1–2 sentence utterance (FLEURS
ja's longest reference sentence is 133 characters).

### What happens if you skip NFKC

Don't skip it. Full-width alphanumerics (`１５`) or half-width katakana
(`ｱｼﾀ`) would go straight to a vocabulary lookup as-is and mostly resolve to
`[UNK]`, breaking the model's predictions. Conversely, applying NFKC means
you have to accept that **the output text is also the normalized form** —
full-width alphanumerics in the input come back half-width.

### Post-processing order

The body of `restore()`, in order:

1. `strip()` the input. If empty, **return it unchanged** (don't call the
   model).
2. Tokenize (above). If the result is empty, return it unchanged. Truncate
   at 500 characters.
3. Build `input_ids` / `attention_mask` and run the model.
4. `probs = sigmoid(logits)`.
5. Emit characters `chars[i]` in order, deciding after each one whether to
   insert a mark. With `comma_p, period_p = probs[i+1]`:
   - If `chars[i]` itself is in the punctuation set, **do nothing** (don't
     stack marks).
   - If the next character `chars[i+1]` is in the punctuation set, **do
     nothing** (don't double up right before one).
   - If `period_p >= 0.5`, insert `。`. Except at the very last character,
     where it's only inserted if `force_final_period` is true.
   - Otherwise, if `comma_p >= 0.5`, insert `、`. When both thresholds are
     exceeded, **the period wins**.
6. If `force_final_period` is true and the result doesn't end in a
   punctuation-set character, append `。`.
7. Apply the question-mark rule (below).

Both the comma and period thresholds are **0.5**. `force_final_period`
defaults to **true**.

The punctuation set (`_JA_PUNCT_CHARS`) is these 21 characters:

```
。 、 ！ ？ ! ? … 「 」 『 』 （ ） ( ) 【 】 ・ , . \n
```

Both half-width and full-width forms are included because NFKC folds
`！？（）` down to `!?()` before this set is checked; checking only one form
would miss the other.

### The question-mark rule

The model only has two classes: comma and period. `？` isn't a model
output — it comes from a hand-written rule that looks at the sentence
ending (`_apply_question_marks`).

Implementation: split the output on `。`, and for each non-empty segment,
reattach `？` if it ends in one of the suffixes below, otherwise reattach
`。`, then join.

The suffixes are exactly these **10**, in this order (`_QUESTION_SUFFIXES`):

```
ですか  ますか  でしょうか  かな  かしら  かい  の  だろうか  でしたか  ましたか
```

The limits are worth stating explicitly. This rule misses in both
directions: intonation-only questions aren't caught, and a plain statement
ending in the nominalizer `の` gets a false positive. `！` isn't handled at
all. A port should still carry **these exact 10 suffixes, verbatim**,
because producing identical text to the desktop is the only way to confirm
the port is correct.

### Known quirk: input ending in a full-width `？`

If the input ends in a full-width `？`, the output ends up `?。`. NFKC folds
`？` to half-width `?`, and since `?` is in the punctuation set, the
`_apply_question_marks` "append `。`" step fires through the other code
path. This is a bug, but since it's how desktop behaves, **a port should
reproduce it too** (fix both at once if you fix it at all).

### ONNX Runtime call order

Uses sherpa-onnx's bundled `libonnxruntime.so` / `onnxruntime.dll` as-is.
Loading a second ONNX Runtime into the same process crashes
([sherpa-onnx#3261](https://github.com/k2-fsa/sherpa-onnx/issues/3261)).

The entry point is `OrtGetApiBase()`, from which `GetApi(version)` gets the
`OrtApi` function table. The Dart port actually calls 27 members of
`OrtApi` (plus 2 of `OrtApiBase` and 3 of `OrtAllocator`):

- Lifecycle: `CreateEnv`, `CreateSessionOptions`, `SetIntraOpNumThreads`,
  `SetInterOpNumThreads`, `CreateSession`, `CreateCpuMemoryInfo`,
  `GetAllocatorWithDefaultOptions`
- Graph inspection: `SessionGetInputCount`, `SessionGetOutputCount`,
  `SessionGetInputName`, `SessionGetOutputName`
- Inference: `CreateTensorWithDataAsOrtValue`, `Run`, `GetTensorTypeAndShape`,
  `GetTensorElementType`, `GetDimensionsCount`, `GetDimensions`,
  `GetTensorMutableData`
- Errors: `GetErrorCode`, `GetErrorMessage`
- Release: `ReleaseStatus`, `ReleaseSessionOptions`, `ReleaseValue`,
  `ReleaseTensorTypeAndShapeInfo`, `ReleaseMemoryInfo`, `ReleaseSession`,
  `ReleaseEnv`
- `OrtApiBase`: `GetApi`, `GetVersionString` / `OrtAllocator`: `Alloc`,
  `Free`, `Info`

Order and gotchas:

1. `CreateEnv(ORT_LOGGING_LEVEL_WARNING, "hayamimi_punct", &env)`.
2. `CreateSessionOptions(&options)` → `SetIntraOpNumThreads(options, 2)` →
   `SetInterOpNumThreads(options, 1)`. Threads are kept low because this
   runs alongside the recognizer (the Python side uses intra-op 4, but
   that's a separate process from the recognizer — pick what fits your own
   process layout). **Do not register** an execution provider — with none
   registered, it falls back to the CPU provider.
3. **`CreateSession(env, model_path, options, &session)` must be called
   with a file path.** Reading from a byte buffer would load 182–364 MB
   onto the heap first and then copy it again into the runtime. Passing a
   path lets the runtime mmap it itself.
   **`model_path`'s type is `ORTCHAR_T*`, which is `wchar_t*` on Windows and
   `char*` elsewhere.** Getting this wrong produces a confusing "file not
   found" error.
4. `CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info)`.
5. Graph validation (optional but recommended): use `SessionGetInputCount` /
   `SessionGetInputName` to confirm the inputs start with `input_ids`,
   `attention_mask`, in that order, and `SessionGetOutputName` to confirm
   `logits` is present. Names are allocated by ORT's allocator, so **free
   them with the same allocator's `Free`** (don't use your own free). This
   turns a mismatched-model error into a readable message instead of a
   confusing failure inside `Run`.
6. Per inference: `CreateTensorWithDataAsOrtValue` builds two int64 tensors
   for `input_ids` and `attention_mask` (shape `{1, seq}`, element type
   `ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64`). →
   `Run(session, nullptr, input_names, inputs, 2, output_names, 1, outputs)`.
7. Output: `GetTensorTypeAndShape` → confirm `GetTensorElementType` is
   float32 → confirm `GetDimensionsCount` is 3 → confirm `GetDimensions` is
   `{1, seq, 2}` → `GetTensorMutableData` gets the pointer. **The buffer is
   owned by the `OrtValue`**, so don't `ReleaseValue` it before you're done
   reading it — copy first, then release.
8. Every call: `ReleaseValue` (the 2 inputs + 1 output),
   `ReleaseTensorTypeAndShapeInfo`. At shutdown: `ReleaseMemoryInfo` →
   `ReleaseSession` → `ReleaseEnv`. The allocator returned by
   `GetAllocatorWithDefaultOptions` is process-owned — **do not release
   it**.
9. Every API call returns an `OrtStatus*`; non-NULL means error. Read
   `GetErrorCode` / `GetErrorMessage`, then **always call `ReleaseStatus`**.

### What's known about ONNX Runtime compatibility

- **Android x86_64 emulator (API 35)**: opened sherpa_onnx's bundled
  `lib/x86_64/libonnxruntime.so` (25,000,408 bytes) via `dlopen`-equivalent
  loading, resolved `OrtGetApiBase`, requested C API version 11
  successfully, and the runtime reported itself as **ONNX Runtime 1.27.1**.
  Only one `libonnxruntime.so` ships in the APK — no second copy was added.
  With that, the 181.8 MB fp16 model was loaded and `restore()` was run
  against it.
- **Windows**: confirmed the same with `sherpa_onnx_windows`'s bundled
  `onnxruntime.dll` (also 1.27.1).
- **iOS: unverified.** The Dart port assumes an
  `OrtLibrary.processSymbols` path (looking up symbols already loaded into
  the process) on iOS, but this has not been run on either a real device or
  a simulator. The Dart port refuses to load on iOS by default.

That sherpa-onnx 1.13.6 bundles ONNX Runtime 1.27.1 is confirmed by having
the runtime itself report its own version, on both Android and Windows.

### Reference implementation and parity fixtures

A Dart port exists on the `agent/feature/core-release` branch — worth
reading before writing C++.

| File | Contents |
|---|---|
| `mobile/hayamimi_core/lib/punct/ort_bindings.dart` | The `OrtApi` function table definition. Only the members actually used are typed concretely; the rest are `Pointer<Void>` placeholders. `OrtApi` has an append-only ABI guarantee, which is why a table built from an old header still works against a newer runtime — the reasoning is documented there. |
| `mobile/hayamimi_core/lib/punct/punct_ort_session.dart` | The call-order implementation above, including `ORTCHAR_T` handling. |
| `mobile/hayamimi_core/lib/punct/punct_ja_tokenizer.dart` | Vocab loading and MeCab-free tokenization. |
| `mobile/hayamimi_core/lib/punct/punct_ja_text.dart` | Thresholds, mark insertion, question-mark rule. The punctuation set and suffix list constants are here too. |
| `mobile/hayamimi_core/test/fixtures/punct_ja_parity.json` | **The parity fixture.** |
| `scripts/make_punct_fixture.py` | The script that generates it (also on this branch). |

The fixture format has a header (generator, source, model used,
onnxruntime version, `unicodedata` version, thresholds) and a `cases`
array. Each case looks like:

```json
{"name":"fleurs_00","source":"fleurs",
 "input":"海の下は薄く高地の下は厚くなっています",
 "input_ids":[2,3348,464,...,3],
 "expected":"海の下は薄く高地の下は厚くなっています。"}
```

Recording `input_ids` separately is the key detail: it lets **tokenization
be verified without a model at all** — the tokenizer is the part of a port
most prone to breaking, so covering half of it without the 182 MB model is
significant. A C++ port should read the same file and cross-check both
stages (ids, then final string).

FLEURS-derived cases are CC BY 4.0, with `、。？` stripped to make an
ASR-style unpunctuated input. FLEURS ja has no question sentences at all,
so the question-mark rule, the empty string, punctuation-only input,
half-width katakana, full-width ASCII, and the 500-character overflow are
all covered by synthetic cases instead.

## Golden tests

`tests/golden/ja/` holds 8 FLEURS ja clips and the text the Japanese route
currently produces for them, as `golden.json`. Details are in
`tests/golden/ja/README.md`.

Summary:

- Generated by `scripts/make_ja_golden.py`. It builds the same objects
  `realtime_transcribe.main()` assembles for
  `--wav X --no-realtime --mode single --lang ja --threads 4`
  (`RoutedASR(forced_lang="ja")`, `LiveVad`, `AudioHistory`, `Refiner`), and
  reads results off `EventHub.add_listener`. stdout is not read.
- Checked by `tests/test_ja_golden.py`, with a **two-level judgment**: (1)
  the count of exact matches is reported but not itself the pass condition;
  (2) pass/fail is decided by whether the **CER against the golden text is
  1.0% or less**. int8 ONNX kernels aren't bit-reproducible across CPU
  microarchitectures (onnxruntime picks a different code path depending on
  vector width), so requiring an exact match would fail on another machine
  for reasons unrelated to a regression — hence the two levels.
- **At this set's lengths, though, 1.0% admits zero characters of drift.**
  Normalized length is 21–44 characters, so a single differing character is
  already a CER of 2.3–4.8%. In other words, the current threshold behaves
  identically to exact match. It's kept anyway because a failure reports a
  number instead of a diff, it becomes meaningful once longer clips are
  added, and it communicates the intended contract to a reimplementation:
  "pin the text, tolerate sub-percent drift." The test also reports the CER
  of all 8 clips concatenated (291 normalized characters, so 1.0% there is
  about 2.9 characters), but that figure is display-only, not a pass
  condition.
- On the recording machine, all 8 clips matched exactly across 3
  independent runs.
- Skipped in an environment without `models/` or `testdata/`. CI has
  neither.

Reusing this golden set as-is in a reimplementation isn't recommended
(decoder bit-reproducibility can't be counted on to that degree). If you do
use it, re-derive the CER threshold from your own measurements.

## Configuration dump and CI check

Left alone, documentation drifts away from code. So the numbers here are
generated mechanically from one place.

- `scripts/dump_ja_config.py` prints the Japanese route's effective
  configuration as JSON **without loading any model**. Every value comes
  from a module-level constant or a function's declared default (some read
  via `inspect.signature`), so it works even without a `models/` directory.
  `--with-models` additionally reads `models/` to add sha256 and byte
  counts.
- That output is pinned in `docs/spec/ja_pipeline_spec.json`. Regenerate
  with:

  ```
  python scripts/dump_ja_config.py --with-models --out docs/spec/ja_pipeline_spec.json
  ```

- `tests/test_ja_pipeline_spec.py` re-dumps the `config` block every run
  and compares it against the committed one — this is the **drift
  detector**. The `models` block is only compared where `models/` exists,
  and is skipped otherwise.

Building this required pulling the code's bare literals out into module
constants (`asr_engine.RZ_MODEL_TYPE` / `RZ_DECODING_METHOD` /
`RZ_MODELING_UNIT` / `RZ_HOTWORDS_SCORE` / `RZ_MODEL_FILES`,
`realtime_transcribe.VAD_MIN_SPEECH_S` / `VAD_BUFFER_S` / `VAD_NUM_THREADS`,
`punct_ja.PUNCT_*`). No value was changed in the process. `punct_ja.py` now
imports numpy / onnxruntime / fugashi inside its methods, so none of the
three is needed just to read the constants.

## Not yet verified

Things stated in this document that haven't been measured or confirmed yet.

- **How far the head-dropout measurements reach**
  (`docs/eval/head_dropout.md`). What was measured is one dataset (FLEURS
  ja, **read speech**), 100 clips, one language, one machine — and only
  **the first speech span of each clip**. There's no guarantee the same
  rate holds for broadcast, live commentary, or casual conversation, and
  nothing is said about the onset of the second or later spans (except the
  3 spans in `multi_sentence_ja.wav`). Every count is two digits or fewer,
  so treat a difference of a few clips as noise. In particular, **do not
  claim the pre-roll-vs-silence gap (0 vs. 2–4 out of 90) is significant.**
- **The optimal amount to prepend.** Only 300 ms and 1.0 s were tried — no
  search was done.
- **Pre-roll's effect on latency.** The ms figures above are offline
  wall-clock decode time, not live-route latency (which also includes
  waiting for the VAD to detect the segment's end).
- **iOS.** The path that runs the punctuation model through ONNX Runtime
  has never been run on iOS. The Dart port refuses it by default.
- **Real ARM devices.** All Android verification has been on an x86_64
  emulator (host: Ryzen 5 5600); fp16 speed is the number least safe to
  extrapolate from it. x86's CPU EP has no fp16 compute path and casts to
  fp32 per operator; ARM chips with NEON fp16 would behave differently.
- **fp16 punctuation model latency measured on ARM.** Same caveat as above.
  On this PC, fp16 is about 10x slower than fp32.
- **Punctuation restoration's standalone latency (Android).** Only derived
  as the difference between refine's total time and final's time (25–35 ms
  for 11–14 characters) — not a direct measurement.
- **The effect of the `--replace` dictionary on this route.** The mechanism
  is implemented, but its effect hasn't been measured for the Japanese
  route specifically.
- **Whether the golden test's 1.0% CER threshold is right for a different
  CPU.** Not run on another machine, so whether the threshold is too tight
  or too loose is unknown.
- **Truncation past 500 characters.** Left unaddressed on the assumption
  that 1–2 sentence utterances never hit it. An implementation feeding
  longer text through needs to add its own chunking.
