# Japanese route golden set

Eight audio clips and the exact text hayamimi's Japanese route produces for
them. `docs/spec/ja_pipeline.md` (Japanese: `docs/spec/ja_pipeline.ja.md`)
describes that route in prose so another project can re-implement it; this
directory is the part a machine can check.

## Attribution

The clips come from **FLEURS** (`google/fleurs`, `ja_jp`), Copyright Google,
licensed **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>). They
are copied unmodified from `testdata/fleurs_bench/ja/`, which is the
benchmark set `docs/results/benchmarks.md` scores against; the reference text in
`golden.json` is that set's `manifest.json`, verbatim.

| file | FLEURS id | seconds |
|---|---:|---:|
| `ja_014.wav` | 1669 | 9.60 |
| `ja_024.wav` | 1674 | 8.70 |
| `ja_033.wav` | 1677 | 7.80 |
| `ja_036.wav` | 1679 | 6.48 |
| `ja_044.wav` | 1684 | 9.06 |
| `ja_079.wav` | 1702 | 9.90 |
| `ja_085.wav` | 1705 | 12.48 |
| `ja_090.wav` | 1710 | 11.22 |

## Why these eight

- **Lengths 6.5 s to 12.5 s.** FLEURS ja's shortest clip is 6.48 s, so there
  is no shorter end to sample; the top of the range is chosen to cross the
  12 s `max_speech` VAD limit's neighbourhood without becoming a long-form
  test.
- **`ja_085` and `ja_090` exercise the CJK ITN pass** (`scripts/itn_cjk.py`),
  which is the stage most easily left out of a re-implementation. The
  recognizer emits `千九百四十年八月十五日` and `三十四件`; ITN rewrites them to
  `1940年八月15日` and `34件`. Note `八月` staying in kanji: a single bare
  numeral is never converted, which is the rule that keeps `一番`/`九州` intact.
- **`ja_044` records a real failure, on purpose.** Its first sentence
  (`アピアはサモアの首都です`) survives only as `あ。` — the head-dropout
  `docs/spec/ja_pipeline.md` describes, in the form the pre-roll cannot fix
  because the whole utterance, not just its head, is what the VAD cut short.
  Pinning it means a change that fixes it is visible as a change.
- **No clip's reference ends in `？`.** All 100 FLEURS ja references were
  checked and none contains a question mark at all — it is news and
  encyclopedia prose. So the question-mark rule in `scripts/punct_ja.py` is
  not covered here; the parity fixture on the `agent/feature/core-release`
  branch (`mobile/hayamimi_core/test/fixtures/punct_ja_parity.json`) covers
  it with synthetic cases instead.

## How the expected text was produced

`scripts/make_ja_golden.py` builds the same objects
`scripts/realtime_transcribe.py`'s `main()` builds for

```
python scripts/realtime_transcribe.py --wav <clip> --no-realtime \
    --mode single --lang ja --threads 4
```

and reads the results off the `EventHub` with `add_listener`, so what is
recorded is the pipeline's own structured output, not scraped stdout.
`finals` is one entry per `final` event (fast path, one per VAD segment, after
ITN and punctuation) and `refine` is the text of the last `refine` event (the
second pass over the whole utterance group).

Regenerate after an intentional change:

```
python scripts/make_ja_golden.py            # rewrite golden.json
python scripts/make_ja_golden.py --check    # run and report, write nothing
```

`golden.json` also records the CER of each recorded text against the FLEURS
reference. Those numbers are **transparency, not a threshold** — nothing
asserts them. They are there so a reader can see at a glance which clips the
pipeline handles well and which it does not.

## The two-level comparison rule

`tests/test_ja_golden.py` re-runs the pipeline and compares, at two levels:

1. **Exact match** — the `finals` list, character for character. Counted and
   reported, but *not* the pass condition.
2. **CER against the golden text** — the pass condition. A clip fails when
   the character error rate between the recorded text and the fresh run
   exceeds **1.0%** (`make_ja_golden.CER_TOLERANCE`).

The reason for the second level is that the recognizer runs int8 ONNX
kernels, and int8 kernels are not bit-reproducible across CPU
microarchitectures: onnxruntime dispatches to different vector code paths
depending on the instruction sets available, and the accumulated rounding can
render a character differently while transcribing the same words. Demanding
exact text on every machine would turn that into a failing test for a reason
that is not a regression.

### What 1.0% actually admits here, measured

Be honest about the size of that allowance. Normalized (NFKC, punctuation and
whitespace removed), the recorded texts are 21 to 44 characters long, so one
differing character is a CER of 2.3% to 4.8% on every clip in this set. **At
these lengths the 1.0% per-clip gate admits zero differing characters**, which
makes it, today, the same test as exact match.

It is kept anyway, for three reasons. It fails with a graded number instead of
a diff, so a report says how far off a run is. It stays meaningful if longer
clips are added later. And it states the intended contract, which is the thing
a re-implementation should copy: *pin the text, tolerate sub-percent drift*.

The test additionally prints a **set-level CER** over all eight clips
concatenated (291 normalized characters, so 1.0% there is about 2.9
characters). That figure is reported, never asserted.

If a different CPU turns out to produce benign single-character drift, the
threshold is what should be revisited — with the measurement written down —
not the golden text.

On the machine the file was recorded on, all eight clips match exactly, three
independent full runs in a row.
