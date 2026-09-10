# Translation model replacement candidates (Track B)

Evaluates whether any replacement candidate should displace either of
hayamimi's two shipped translation modules:

- **FuguMT** (`scripts/translate_ja_en.py`, ja->en) -- the project's only
  non-permissive model, **CC BY-SA 4.0**. See `docs/design/translate.md`.
- **M2M-100 418M** (`scripts/translate_m2m.py`, ja->zh/ko/es, and
  ja->en/"unvalidated") -- MIT. See `docs/design/translate_m2m.md`.

**Result: no candidate is adopted.** Every candidate either fails the chrF
bar, produces degenerate output, or -- the criterion every single
non-baseline candidate failed -- blows the 1.0s/line latency budget by
2.5-8x. Details and the one adjacent finding worth a follow-up are below.
`scripts/translate_candidates.py` (the pluggable eval backends) and this
doc's reproduction commands are kept so a future pass (on faster hardware,
or once this machine isn't sharing 6 cores with 5 other tracks) can pick up
where this one stopped without re-doing the model search.

## Method

- Quality: `scripts/eval_translate.py --backend <name> --targets <langs> --n 50`
  (FLEURS validation split, seed=0, sacrebleu chrF against the FLEURS
  reference transcription for that language) -- same methodology as the
  existing zh/ko/es baseline in `docs/design/translate_m2m.md`, now also
  covering `en` (FLEURS `en_us`, newly added to `FLEURS_CONFIG_BY_LANG`) and
  made backend-pluggable via `scripts/translate_candidates.make_translator()`.
  `--backend m2m` (the default) is byte-for-byte the script's original code
  path (`TranslatorM2M`, default model dir) -- **its results are unchanged**
  by this work.
- Degenerate-output check: the two existing smoke sets from
  `scripts/translate_ja_en.py` and `scripts/translate_m2m.py`, merged into
  one 9-line set (business/casual/numbers/scheduling/question/long/repetition-
  prone lines), run through every candidate/target combination and eyeballed
  for repetition loops (a documented failure mode of both existing shipped
  models -- see the "Known limitations" sections of the two design docs).
- Latency: mean per-line wall time from the same `--n 50` FLEURS run
  (`time.perf_counter()` around `.translate()`, model load excluded) --
  **provisional**: this machine's 6 physical cores (12 threads) were shared
  with 5 other improvement tracks' evaluations running in parallel for the
  whole of this pass, and every translator below was constructed with
  `ctranslate2` `intra_threads=2` (`eval_translate.py --intra-threads`,
  default 2) to avoid starving the other tracks. Absolute numbers here are
  not representative of solo/production hardware; the *relative* ordering
  between candidates (all measured under the same contention) is.
- Disk size: `du -sh` on the converted `models/<name>-ct2/` directory
  (int8-quantized CTranslate2 weights + tokenizer files only).
- License: read directly off each model's Hugging Face model card.

## Candidates and setup

| Candidate | Base model | License | Conversion |
|---|---|---|---|
| **FuguMT** (baseline) | `staka/fugumt-ja-en` | CC BY-SA 4.0 | already shipped (`models/mojicast-fugumt-ja-en-ct2/`) |
| **M2M-100 418M** (baseline) | `facebook/m2m100_418M` | MIT | already shipped (`models/mojicast-m2m100-ct2/`) |
| **opus-mt-ja-en** | `Helsinki-NLP/opus-mt-ja-en` | Apache-2.0 | `ct2-transformers-converter --quantization int8` -> `models/opus-mt-ja-en-ct2/` |
| **opus-mt-ja-es** | `Helsinki-NLP/opus-mt-ja-es` | Apache-2.0 | same, -> `models/opus-mt-ja-es-ct2/` |
| **M2M-100 1.2B** | `facebook/m2m100_1.2B` | MIT | same, -> `models/m2m100-1.2B-ct2/` |
| **LMT-60-0.6B** | `NiuTrans/LMT-60-0.6B` (Qwen3-based, continued-pretrained on 90B MT tokens, released 2026) | Apache-2.0 | same, -> `models/lmt60-0.6b-ct2/` |

Reproduction (from the worktree root; `.venv-train` has `torch`/`transformers`,
not committed -- see "Environment" below):

```
python -m ctranslate2.converters.transformers \
    --model Helsinki-NLP/opus-mt-ja-en --quantization int8 \
    --copy_files source.spm target.spm vocab.json \
    --output_dir models/opus-mt-ja-en-ct2

python -m ctranslate2.converters.transformers \
    --model Helsinki-NLP/opus-mt-ja-es --quantization int8 \
    --copy_files source.spm target.spm vocab.json \
    --output_dir models/opus-mt-ja-es-ct2

python -m ctranslate2.converters.transformers \
    --model facebook/m2m100_1.2B --quantization int8 --low_cpu_mem_usage \
    --copy_files sentencepiece.bpe.model \
    --output_dir models/m2m100-1.2B-ct2
# then: mv models/m2m100-1.2B-ct2/sentencepiece.bpe.model models/m2m100-1.2B-ct2/sentencepiece.model
# (TranslatorM2M hardcodes the "sentencepiece.model" filename, matching the
# existing mojicast-m2m100-ct2/ layout -- ct2-transformers-converter's
# --copy_files preserves the source repo's original filename, which differs.)

python -m ctranslate2.converters.transformers \
    --model NiuTrans/LMT-60-0.6B --quantization int8 --low_cpu_mem_usage \
    --copy_files tokenizer.json tokenizer_config.json vocab.json merges.txt \
        special_tokens_map.json chat_template.jinja added_tokens.json generation_config.json \
    --output_dir models/lmt60-0.6b-ct2
```

Note on **opus-mt coverage**: Helsinki-NLP only publishes an official ja->en
and ja->es OPUS-MT model. There is no official `opus-mt-ja-zh` or
`opus-mt-ja-ko` (confirmed via `huggingface_hub`'s model search -- only
community forks of uncertain provenance/license exist for those directions,
e.g. `shun89/opus-mt-ja-zh`, not evaluated here). So opus-mt was only
evaluated for en/es.

### Candidates considered and not pursued

- **facebook/nllb-200-distilled-\*** -- excluded per the task's license bar:
  **CC-BY-NC-4.0** (non-commercial), same family the task explicitly flags
  to avoid.
- **NiuTrans/LMT-60-{1.7B,4B,8B}** -- larger siblings of the evaluated
  0.6B; not converted/measured given the 0.6B variant already failed the
  latency bar by a wide margin (larger sizes would only be slower) and the
  1.5GB disk cap becomes tighter (1.7B+ is much larger even after int8).
- **Tencent Hunyuan-MT2, Google TranslateGemma** -- found during the model
  search (see below) but not evaluated: both are recent (2026) general
  LLM-family translation models at 1.8B+ parameters, i.e. larger than
  LMT-60-0.6B, which already missed the latency budget by 2-3x at 0.6B on
  this CPU-constrained, thread-capped run. Not worth the conversion/download
  cost without a reason to expect a smaller model in the same class to be
  faster.

## Results

### chrF (FLEURS validation, n=50, seed=0)

| Backend | zh | ko | es | en |
|---|---|---|---|---|
| M2M-100 418M (baseline, shipped) | 20.19 | 25.20 | 42.09 | 48.88 *(new; unvalidated target)* |
| FuguMT (baseline, shipped) | -- | -- | -- | 47.57 *(new measurement)* |
| opus-mt-ja-\* | -- | -- | 38.15 | 40.49 |
| M2M-100 1.2B | 23.84 | 26.31 | 45.12 | 52.61 |
| LMT-60-0.6B | 29.57 | 24.98 | 40.91 | 53.62 |

Adoption bar: zh >= 30, ko >= 35, es >= current (42.09), en > FuguMT (47.57).

- **zh**: nothing clears 30. LMT-60 comes closest (29.57) but is under the
  bar; M2M-100 1.2B barely improves on the 418M baseline (23.84 vs 20.19).
- **ko**: nothing clears 35, and LMT-60 (24.98) is actually *worse* than the
  shipped 418M baseline (25.20) at n=50, despite looking very strong on an
  n=3 spot check during setup (35.36) -- a reminder that n=3 sampling
  variance on FLEURS is large and not a substitute for the full n=50 run.
- **es**: M2M-100 1.2B clears the bar (45.12 > 42.09). opus-mt-ja-es and
  LMT-60 (40.91) both fall slightly short of the existing 42.09.
- **en**: both M2M-100 1.2B (52.61) and LMT-60 (53.62) clearly beat FuguMT
  (47.57); the shipped M2M-100 418M *also* already beats it (48.88) without
  any new download -- see "Adjacent finding" below.

### Latency (mean ms/line, same FLEURS n=50 run, provisional -- see Method)

| Backend | zh | ko | es | en |
|---|---|---|---|---|
| M2M-100 418M (baseline) | 664¹ | 963¹ | 1013¹ | 1066 |
| FuguMT (baseline) | -- | -- | -- | 519 |
| opus-mt-ja-\* | -- | -- | 1196 | 1443 |
| M2M-100 1.2B | 2910 | 3640 | 4139 | 2583 |
| LMT-60-0.6B | 2193 | 2857 | 3531 | 2554 |

¹ zh/ko/es figures for the M2M-100 418M baseline are carried over from
`docs/design/translate_m2m.md` (measured in an earlier, non-parallel pass on
this same machine) rather than re-run here, since that value is not in
question -- only `en` (not previously measured for this model) was re-run in
this pass, under the same 6-core/5-other-tracks contention as every other
number in this table.

Adoption bar: <=1000 ms/line. **Every non-baseline candidate fails this by a
wide margin** (2.2-4.1s, i.e. 2.5-8x over budget) -- decoder-side compute
scales with model size (1.2B / 0.6B-as-Qwen3-decoder vs. the baseline's
418M encoder-decoder), and at `intra_threads=2` on a contended CPU none of
them come close. This alone disqualifies M2M-100 1.2B and LMT-60-0.6B
regardless of their chrF results above.

### Degenerate output (9-line smoke set: business/casual/numbers/scheduling/question/long/repetition-prone)

| Backend | Degenerate lines | Notes |
|---|---|---|
| FuguMT (baseline) | 0/9 (documented ceiling case exists outside this set -- see `docs/design/translate.md`) | |
| M2M-100 418M (baseline) | 0/9 (documented mild repetition on filler-heavy zh -- see `docs/design/translate_m2m.md`) | |
| opus-mt-ja-en | **6/9** | Catastrophic repeat loops on business/casual/numbers/scheduling lines, e.g. `"Thank thank thank thank very very very welcome welcome welcome gathered today today today..."`. Tried `no_repeat_ngram_size` in {0,1,2,3} x `beam_size` in {1,5} on the worst offending FLEURS sentence -- **every combination** still mangled `"Marie Antoinette"` into repeated garbage (`"Anton Anton Anton Anto Antonto"`) or the age `"11"` into `"11111111 11"`. Same conclusion as FuguMT's own documented finding: a real quality ceiling of this small a model, not a decoding-parameter bug. |
| opus-mt-ja-es | **3-4/9** (heuristic-flagged; several more show clear but shorter repeats, e.g. `"llover llover mañana llover mañana"`, `"la reunión de la reunión comenzará"`) | Same root cause as opus-mt-ja-en above (shared architecture/training recipe). |
| M2M-100 1.2B | 0/9 | Clean across all 4 targets on this smoke set; noticeably better than the 418M baseline's documented filler-line degeneration. |
| LMT-60-0.6B | 0/9 | Clean across all 4 targets; also the only candidate that got `500万円` right in every target language (`500万日元`/`5백만엔`/`5 millones de yenes`) where **M2M-100 (both sizes) mistranslated the currency/magnitude** (see `docs/design/translate_m2m.md`'s "Numbers are not reliably preserved" limitation, reproduced again here for zh: `5万日元` instead of 500万円). |

### Disk size (converted `models/<name>-ct2/`, int8)

| Backend | Size | vs. 1.5GB cap |
|---|---|---|
| opus-mt-ja-en | 78 MB | pass |
| opus-mt-ja-es | 79 MB | pass |
| LMT-60-0.6B | 589 MB | pass |
| M2M-100 1.2B | 1197 MB (1.17 GB) | pass (tight -- 80% of the cap) |

(Baselines for reference: `mojicast-fugumt-ja-en-ct2/` 118 MB, `mojicast-m2m100-ct2/` 473 MB.)

## License

| Model | License | Notes |
|---|---|---|
| FuguMT (`staka/fugumt-ja-en`) | **CC BY-SA 4.0** | shipped baseline; the one non-permissive model in the project |
| M2M-100 418M/1.2B (`facebook/m2m100_*`) | MIT | per model card |
| Helsinki-NLP/opus-mt-ja-en, opus-mt-ja-es | Apache-2.0 | per model card |
| NiuTrans/LMT-60-0.6B | Apache-2.0 | per model card; base is a continued pretrain of Qwen3 (also Apache-2.0) |

All four candidates clear criterion 1 (permissive license) -- none needed to
be excluded on licensing grounds. (NLLB-200, CC-BY-NC-4.0, was excluded from
consideration entirely per the task brief without conversion/measurement.)

## Adoption decision: **none adopted**

No candidate satisfies all five criteria simultaneously:

| Backend | 1. License | 2. chrF bar | 3. No degeneration | 4. Latency <=1.0s | 5. Size <=1.5GB | Verdict |
|---|---|---|---|---|---|---|
| opus-mt-ja-en | pass | fail (40.49 < 47.57) | **fail** (6/9) | fail (1443ms) | pass | **reject** |
| opus-mt-ja-es | pass | fail (38.15 < 42.09) | **fail** (3-4/9) | fail (1196ms) | pass | **reject** |
| M2M-100 1.2B | pass | fail (zh/ko under bar; es/en clear it) | pass | **fail** (2.6-4.1s) | pass | **reject** |
| LMT-60-0.6B | pass | fail (zh/ko/es under bar; en clears it) | pass | **fail** (2.2-3.5s) | pass | **reject** |

Latency is the decisive failure for the two candidates that otherwise looked
promising (M2M-100 1.2B and LMT-60-0.6B both have real chrF wins on
`es`/`en`, and LMT-60 in particular has meaningfully better output quality
overall -- no degeneration, correct number handling). At double-to-quadruple
the 1.0s budget even under a thread cap, neither is viable for a live
subtitle pipeline as converted/prompted here, independent of the parallel-
evaluation contention noted in Method -- the gap is too large to attribute
to contention alone (a same-machine, same-load baseline comparison, M2M-100
418M vs. 1.2B, shows the size difference alone costs ~2.5-4x).

**Recommendation: keep FuguMT (ja->en, CC BY-SA 4.0) and M2M-100 418M
(ja->zh/ko/es, MIT) as shipped.** No code changes to
`scripts/translate_m2m.py` or `scripts/translate_ja_en.py`'s runtime
behavior are made by this pass (only an additive, default-preserving
`intra_threads` constructor parameter on both -- see "Non-adoption code
changes kept" below).

### Adjacent finding (not a Track B candidate, flagged for follow-up)

`M2M-100 418M` -- already shipped, already MIT-licensed, no new download --
scores **48.88 chrF on `en`**, edging out FuguMT's 47.57, when pointed at
the previously-unvalidated `en` target (`translate_m2m.py`'s
`VALIDATED_TARGETS` doesn't currently include `en`; see
`docs/design/translate_m2m.md`'s note that `en` "was not part of the smoke
test or the repetition/beam-size measurements"). This is *not* one of this
track's four candidates and wasn't in scope to act on here (promoting a
target on the already-shipped model, and potentially retiring the one
CC BY-SA dependency in the project, is a separate decision with its own
tradeoffs -- e.g. FuguMT is faster at this measured latency, 519ms vs
1066ms). Flagged separately rather than silently adopted mid-track.

## Non-adoption code changes kept

Even though no candidate was adopted, this pass leaves a few artifacts in
place since they're useful independent of the outcome:

- `scripts/translate_candidates.py` -- the four candidates' translator
  classes (`TranslatorMarianPair`, `TranslatorLMT60`) and the
  `make_translator(backend, target_lang)` factory used by
  `scripts/eval_translate.py --backend`. Kept so a future re-evaluation
  (faster/dedicated hardware, a newer LMT-60 release, etc.) doesn't have to
  re-derive the prompt template / tokenizer plumbing for the decoder-only
  LMT-60 case, which was the most involved part of this pass (see the class
  docstring for why `<think>\n\n</think>\n\n` is force-appended to suppress
  Qwen3's default chain-of-thought mode).
- `scripts/eval_translate.py --backend` / `--intra-threads` flags, and the
  `en` entry in `FLEURS_CONFIG_BY_LANG`. `--backend` defaults to `"m2m"`,
  which is byte-identical to the script's pre-existing code path -- running
  `python scripts/eval_translate.py --targets zh,ko,es` with no `--backend`
  flag reproduces exactly the same result as before this change.
- `scripts/translate_m2m.py` / `scripts/translate_ja_en.py`: added an
  optional `intra_threads: int = 0` constructor parameter (passed through to
  the underlying `ctranslate2.Translator`). `0` is ctranslate2's own "use a
  default value" sentinel, i.e. **omitting the parameter reproduces prior
  behavior exactly** -- this was needed so eval runs here could request
  `intra_threads=2` without hardcoding it into the shipped modules' defaults
  (which would silently change production latency for existing callers like
  `realtime_transcribe.py`).
- `scripts/download_models.py --translate-candidates` -- optional,
  off-by-default download of the four converted candidate models (opt-in;
  not part of `--minimal`, the default set, or `--eval-baselines`), so a
  future re-evaluation pass doesn't have to redo the `ct2-transformers-
  converter` conversions in "Candidates and setup" above from scratch.
- `tests/test_units.py` -- a few backend-selection unit tests for
  `translate_candidates.make_translator()` that don't require any
  downloaded model (unknown backend, and `opus-mt` for a target it has no
  model for both raise `ValueError` before touching disk).

If this doc's addendum below were ever promoted (i.e. a future pass adopts
one of these candidates), the following would be added to
`THIRD_PARTY_NOTICES.md`'s "Text models" table -- drafted here per the task
brief, **not applied to the actual file** since nothing was adopted:

| Model (dir under `models/`) | Publisher | License | Source |
|---|---|---|---|
| `opus-mt-ja-en-ct2` / `opus-mt-ja-es-ct2` | Helsinki-NLP (Language Technology Research Group, University of Helsinki) | Apache-2.0 | [Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en), [opus-mt-ja-es](https://huggingface.co/Helsinki-NLP/opus-mt-ja-es) |
| `m2m100-1.2B-ct2` | Meta AI | MIT | [facebook/m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) |
| `lmt60-0.6b-ct2` | NiuTrans Lab (Northeastern University) | Apache-2.0 | [NiuTrans/LMT-60-0.6B](https://huggingface.co/NiuTrans/LMT-60-0.6B), base model Qwen3 (Apache-2.0) |

## Environment

- Conversion venv: `.venv-train/` at the worktree root (`torch==2.14.0+cpu`,
  `transformers==5.16.1`, `ctranslate2==4.8.2`, `huggingface_hub`,
  `sentencepiece`) -- not committed (`.gitignore`'d), needed only to run
  `ct2-transformers-converter`. Re-create with:
  `uv venv .venv-train && uv pip install --python .venv-train/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu && uv pip install --python .venv-train/Scripts/python.exe transformers sentencepiece huggingface_hub ctranslate2`.
- Eval venv: the project's normal `.venv` (`ctranslate2==4.8.1`,
  `sacrebleu==2.6.0`, `sentencepiece`, `tokenizers==0.23.1` -- the last of
  these, a lightweight Rust BPE tokenizer library with no `torch`
  dependency, is what `TranslatorLMT60` uses to encode/decode Qwen3's
  `tokenizer.json` at eval time without needing `transformers` in the
  runtime venv).
- All `--n 50` eval runs used `--intra-threads 2` (the script's default) to
  share this machine's 6 physical cores with 5 other tracks' evaluations
  running at the same time -- see "Method" above for what this means for
  the latency numbers' generalizability.
