# Contributing to hayamimi

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # includes requirements.txt
.venv/Scripts/python scripts/download_models.py
```

## Tests

```bash
.venv/Scripts/python -m pytest tests
```

`tests/test_units.py` covers unit-level logic (character-set arbitration,
fallback behavior, etc.) and doesn't require the full model set.

## Accuracy evaluation

Model or routing changes must be validated against real speech, not just
unit tests -- this project's whole design history (`docs/results/benchmarks.md`) is
built on real-speech regressions that unit tests alone would have missed
(e.g. iteration #27's Cantonese regression from a routing change that passed
every existing test).

- `scripts/eval_accuracy.py` -- ja accuracy vs. `faster-whisper large-v3-turbo`
  as reference.
- `scripts/eval_engine.py` -- end-to-end scorecard across all 5 language
  routes (what `docs/results/scorecard.md` is generated from).
- `scripts/bench_offline.py` -- offline RTF (speed) benchmarking for a given
  model directory.

**Policy: any change to model routing, decoding parameters, or
language-detection logic should be re-validated by re-running the relevant
eval script against real speech clips before merging, not just against
synthetic/TTS test data.** TTS audio is too clean to catch the failure modes
that matter here (see `docs/eval/eval.md` vs. `docs/eval/eval_real.md` for why this
project moved from synthetic to real-speech evaluation early on). If you
touch language-detection or routing logic specifically, re-run the eval
across *all* languages, not just the one you changed -- several regressions
in this project's history were caught only because of a full re-score
(see `docs/results/benchmarks.md` iterations #12 and #27).

## Documentation

`docs/` is grouped by what a reader wants from it. The index is
[`docs/README.md`](docs/README.md) (Japanese: `docs/README.ja.md`) and every
new document belongs in it.

```
docs/README.md, docs/README.ja.md   the index
docs/guide/    how to embed and how to tune it
docs/spec/     the Japanese route, specified for re-implementation
docs/results/  current numbers -- kept up to date
docs/eval/     experiment records -- dated, never rewritten afterwards
docs/design/   design investigations behind a decision
docs/verify/   procedures for checking something on real hardware
docs/research/ the 2026-08-23 pre-code research snapshot
docs/images/   images the READMEs use
```

Where a new document goes: a number you intend to keep current goes in
`results/`; the run that produced it goes in `eval/` with its date in the
title. An investigation you did to reach a decision goes in `design/`, even
if the decision was "no". A procedure someone will follow goes in `verify/`.
If it tells a user how to use or configure hayamimi, it is `guide/`, not
`design/`.

**Language.** English is primary and Japanese sits alongside it, in the
`README.md` + `README.ja.md` pattern: an English file `x.md` has its Japanese
counterpart at `x.ja.md`. Documents written in Japanese before this policy
stay Japanese; the index tags each file's language so nobody has to open one
to find out.

**How to write it.** The house style, used everywhere in this repo:

- Problem first, then the change, then the reasoning, then the effect. Not a
  bare list of what changed -- a reader should be able to tell why it was
  worth doing.
- Explain a term once, the first time it appears (VAD, refine, LID, RTF).
- A number only ever appears with the conditions that produced it: which
  audio, which model, which machine, how many samples.
- Negative results are recorded too. "Measured, not adopted" with the numbers
  is worth as much as a win, and this project's history is full of them.

Run `python scripts/check_doc_links.py` after moving or renaming anything
under `docs/`; `tests/test_doc_links.py` runs the same check in CI.

## Pull requests

- Keep unrelated changes out of one PR.
- If you change decode parameters, routing thresholds, or add/replace a
  model, include the before/after eval numbers in the PR description.
- If you add a new model dependency, update `THIRD_PARTY_NOTICES.md` and
  `scripts/download_models.py` in the same PR.
