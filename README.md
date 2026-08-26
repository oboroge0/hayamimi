# hayamimi (早耳)

[![tests](https://github.com/oboroge0/hayamimi/actions/workflows/test.yml/badge.svg)](https://github.com/oboroge0/hayamimi/actions/workflows/test.yml) [![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/oboroge0/hayamimi)](https://github.com/oboroge0/hayamimi/releases)

**Real-time, multilingual speech-to-text on CPU only.** Live subtitles, a
browser dashboard, speaker labels, and on-the-fly translation -- no GPU, no
cloud API, under 2GB RAM.

日本語版 README は [README.ja.md](README.ja.md) にあります。

"早耳" (hayamimi) is Japanese for "quick ear" -- someone who picks up on
things fast. That's the design goal: partial subtitles appear while you're
still talking, and a finalized line lands roughly **100ms after you stop**.

## Why

Most CPU-only real-time transcription setups fall back to a single
general-purpose model (Whisper) and accept its accuracy ceiling. hayamimi
instead routes each utterance to whichever specialist model is best for its
language, all running as quantized (INT8) ONNX models via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) -- no PyTorch, no CUDA.

On real broadcast Japanese audio (see `docs/SCORECARD.md`), that routing
gets **5.8% CER**, less than half of `whisper-large-v3-turbo`'s 13.8% on the
same clips, while running at 10-50x realtime on a 6-core desktop CPU.

## Features

| Feature | What it does |
|---|---|
| 5-route language catalog | ja/zh/ko/yue/en+24 EU languages each go to a dedicated best-in-class model; everything else (~1600 languages) falls back to Meta's Omnilingual ASR |
| Partial subtitles | in-progress draft text updates every ~0.5s while you're still speaking |
| Fast finals | a finalized line typically lands ~100ms after you stop talking (ja; see `docs/GOALS.md` for other languages) |
| Two-pass refinement | after 2s of silence, recent utterances are batch re-decoded for a higher-accuracy "clean" transcript (ja real-broadcast CER 15.5% -> 12.0%) |
| Speaker labels | `--speakers` tags each utterance S1/S2/... using CAM++ speaker embeddings (turn-taking, not full diarization) |
| Translation | `--translate en,zh,ko,es,...` translates Japanese lines live (en via FuguMT; any other M2M-100 target code is accepted if the model's vocabulary supports it -- zh/ko/es have measured quality, see docs/TRANSLATE_M2M.md) |
| Hotwords / user dictionary | `--hotwords` biases decoding toward proper nouns (currently has no effect on the ja tier -- see Limitations); `--replace` does post-hoc find/replace and works everywhere |
| OBS overlay + dashboard | `--serve` starts a local HTTP server with a browser-source overlay and a live dashboard |
| Network audio input | `--input ws` accepts mic audio over a WebSocket (phone, ESP32/stackchan) and feeds it through the same pipeline, including `--serve`'s dashboard/overlay |
| Memory-bounded | LRU model eviction keeps resident models under a configurable cap (default: <2GB total) |
| CPU-only | every model runs as quantized ONNX via sherpa-onnx; no GPU or PyTorch required |

## Demo UI

`--serve` starts a local server exposing three views:

- **`http://localhost:8833/dashboard`** -- the live dashboard: a partial-text
  strip for in-progress speech, a finals feed with language badges, speaker
  chips, and per-line latency, inline translations under each line, and a
  second column with the refined (two-pass) transcript as it lands.
- **`http://localhost:8833/`** -- a minimal OBS browser-source overlay
  (add this URL as a Browser Source in OBS for stream captions). The
  confirmed line and the in-progress line are separate rows; append
  `?show=final` or `?show=partial` to render only one of them, so each
  can be placed and styled as its own OBS source.
- **`http://localhost:8833/transcript`** -- plain scrolling transcript
  history.

![dashboard](docs/images/dashboard.png)

🎬 **[Watch the demo video](https://github.com/oboroge0/hayamimi/releases/download/v0.1.0/hayamimi_demo.mp4)** — real 4-language audio (ja/en/ko/zh) transcribed live, replayed frame-accurately from a captured session.

## Network audio input

`--input ws` runs a WebSocket ingest endpoint instead of reading the local
microphone, so a phone or a stackchan-class ESP32 board can stream mic audio
over the LAN and get it transcribed through hayamimi's normal pipeline:

```bash
.venv/Scripts/python scripts/realtime_transcribe.py --input ws --serve
# -> ws://<host>:8766/ingest accepts audio; http://localhost:8833/dashboard shows the results
```

Protocol: connect to `/ingest`, send one JSON text frame
(`{"sr": 16000, "format": "pcm_s16le", "channels": 1}`), then stream raw
`pcm_s16le` audio as binary frames. The server resamples non-16kHz audio and
replies with the same partial/final/translation/refine JSON events the
dashboard's SSE stream carries, so a client can show its own subtitles too.
Only one audio-producing client is accepted at a time; `scripts/ws_mic_client.py`
is a dependency-free reference client (streams a wav file at real-time pace)
that doubles as a template for a phone/ESP32 implementation.

## Requirements

Python 3.10+ and ffmpeg on PATH. Developed and tested on **Windows 11**;
macOS/Linux are expected to work (all runtimes are cross-platform) but are
not yet CI-tested end to end — reports welcome.

## Quickstart

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py

# macOS / Linux
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py

# Real-time transcription from your microphone
.venv/Scripts/python scripts/realtime_transcribe.py     # Windows
.venv/bin/python scripts/realtime_transcribe.py          # macOS/Linux

# With the dashboard + OBS overlay
.venv/Scripts/python scripts/realtime_transcribe.py --serve
# -> open http://localhost:8833/dashboard in a browser
```

`scripts/download_models.py` pulls ~3.1GB of pretrained models into
`models/` (git-ignored). Pass `--minimal` for a ~1.1GB ja/en-only install
(ReazonSpeech, whisper-tiny, Silero VAD, Japanese punctuation). See
`THIRD_PARTY_NOTICES.md` for what each model's license commits you to.

## CLI reference

All flags are on `scripts/realtime_transcribe.py`:

| Flag | Default | Description |
|---|---|---|
| `--wav PATH` | mic input | simulate streaming from a 16kHz mono WAV file instead of the microphone |
| `--no-realtime` | off | with `--wav`, don't sleep between chunks (fast batch processing) |
| `--input {mic,wav,ws}` | mic, or wav if `--wav` is given | audio source; `ws` accepts audio over the network (see below) |
| `--ws-host HOST` | `0.0.0.0` | bind host for `--input ws`'s `/ingest` endpoint |
| `--ws-port PORT` | 8766 | port for `--input ws`'s `/ingest` endpoint |
| `--threads N` | 4 | inference threads per model |
| `--no-partial` | off | disable in-progress draft subtitles |
| `--min-silence SEC` | 0.35 | silence duration that ends an utterance; lower = snappier finals, more splits |
| `--max-speech SEC` | 12.0 | force-finalize an utterance after this many seconds of continuous speech |
| `--max-resident N` | 3 | max non-tier0 models kept resident (LRU eviction); `<=0` = unlimited |
| `--serve [PORT]` | off, 8833 | serve the dashboard + OBS overlay at `http://localhost:PORT` |
| `--no-refine` | off | disable the second-pass re-decode of utterance groups |
| `--transcript PATH` | none | append refined transcript lines to this file |
| `--hotwords PATH` | none | hotword list (one per line) to bias decoding toward proper nouns -- **currently has no effect on the ja tier** (ReazonSpeech's byte-level BPE tokens.txt can't encode them; a startup warning tells you how many failed). Use `--replace` for ja proper nouns instead |
| `--replace PATH` | none | user dictionary: `wrong=right` per line, applied to all output |
| `--lang-switch-guard SEC` | 2.0 | treat a new-language detection shorter than this as noise: it can never count toward confirming a switch (see `--lid-switch-confirm`) and it suppresses the omnilingual fallback on an empty decode (`0` disables) |
| `--lid-switch-confirm N` | 2 | consecutive new-language detections (each >= `--lang-switch-guard` long) required before the session actually switches language; raise for stickier single-language sessions |
| `--speakers` | off | label utterances with speaker ids (S1, S2, ...) |
| `--translate [LANGS]` | off, `en` | translate Japanese lines to these comma-separated languages. `en` uses the dedicated FuguMT module; any other M2M-100 target code (`zh`, `ko`, `es`, `fr`, ...) is accepted if the model's vocabulary supports it -- unvalidated targets (anything outside `zh`/`ko`/`es`) print a quality-not-measured note, see docs/TRANSLATE_M2M.md |

## Architecture

```
                          ┌─────────────┐
  mic / wav ───────────▶ │  Silero VAD │  0.35s end-of-speech + 0.8s preroll
                          └──────┬──────┘
                                 │ speech segment
                                 ▼
                   ┌───────────────────────────┐
                   │  whisper-tiny spoken-LID   │  runs on first ~4s while
                   │  (+ char-set arbitration)  │  the segment is still coming in
                   └─────────────┬─────────────┘
                                 │ language tag
                 ┌───────────────┼────────────────┬─────────────┬──────────────┐
                 ▼               ▼                ▼             ▼              ▼
             ┌───────┐      ┌─────────┐      ┌──────────┐  ┌─────────┐   ┌──────────┐
             │  ja   │      │   zh    │      │  ko/yue  │  │ en + 24 │   │  ~1600   │
             │ Reazon│      │Paraformer│      │SenseVoice│  │EU langs │   │  other   │
             │Speech │      │   -zh   │      │  small   │  │Parakeet │   │Omnilingual│
             │Zipform│      │         │      │          │  │TDT v3   │   │  ASR     │
             └───┬───┘      └────┬────┘      └────┬─────┘  └────┬────┘   └────┬─────┘
                 └───────────────┴────────────────┴─────────────┴─────────────┘
                                                │
                     partial (every ~0.5s)      │      final (~0.1s after end-of-speech)
                     ◀───────────────────────────┴───────────────────────▶
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                 ┌────────────────┐   ┌──────────────────┐   ┌────────────────┐
                 │ ja punctuation  │   │ speaker labeling  │   │  translation    │
                 │ (BERT restore)  │   │ (CAM++, --speakers)│   │ (FuguMT/M2M-100)│
                 └────────────────┘   └──────────────────┘   └────────────────┘
                                                │
                     2s silence: batch re-decode recent utterances (two-pass refine)
                                                │
                                                ▼
                              dashboard / OBS overlay / transcript file
```

Models are lazy-loaded on first use; an LRU cache evicts the
least-recently-used non-Japanese models (`--max-resident`) so memory stays
bounded no matter how many languages a session wanders through.

## Measured performance

End-to-end (LID -> routing -> decode -> ja punctuation), real speech, no
preroll/two-pass (single clips). `en` uses WER, all others use CER (`yue`
t2s-normalized). Full methodology in `docs/SCORECARD.md`.

| Language | Clips | LID accuracy | Route | Mean error | Mean RTF |
|---|---|---|---|---|---|
| ja | 15 | 15/15 | ReazonSpeech | 7.5% | 0.071 |
| en | 15 | 15/15 | Parakeet v3 | 2.3% | 0.109 |
| zh | 12 | 12/12 | Paraformer-zh | 5.3% | 0.102 |
| ko | 12 | 12/12 | SenseVoice | 8.1% | 0.062 |
| yue | 12 | 12/12 | SenseVoice | 6.1% | 0.061 |

RTF (real-time factor) well under 0.2 across every route means each route
runs 9-16x faster than realtime on CPU alone -- see `docs/GOALS.md` for the
full target table and `docs/BENCHMARKS.md` for the complete iteration log
(30+ measured changes, latency/memory/accuracy tradeoffs and why each one was
made or rejected).

Headline numbers from that log:

- **Japanese CER 5.8%** (beam search) on real broadcast audio, vs. 13.8% for
  `whisper-large-v3-turbo` on the same clips -- less than half the error rate.
- **~100ms mean final latency** (ja, punctuated); ~236ms mean / 552ms max
  across a 5-language soak test with every feature enabled.
- **<2GB RAM** with `--max-resident 3` (1.35GB at `--max-resident 2`).

## Limitations (honest list)

- **Code-switching mid-sentence is not supported.** The router picks one
  language per utterance; a sentence that mixes Japanese and English within
  itself will have the minority-language portion mangled or dropped.
  Utterance-level switching (e.g. an interpreter alternating full sentences)
  works well; word-level switching within one sentence does not.
- **Very short utterances after a jingle/sting/BGM burst can misroute.**
  The language-switch guard (`--lang-switch-guard`, paired with
  `--lid-switch-confirm`) mitigates this but a session's very first
  utterance (before any session language is established) and
  confidently-wrong LID+decode combinations (where the garbled text happens
  to match the wrong language's character set) are known blind spots -- see
  `docs/BENCHMARKS.md`'s iteration #29 for a quantified before/after.
- **`--hotwords` currently has no effect on the ja (ReazonSpeech) tier.**
  ReazonSpeech's `tokens.txt` is byte-level BPE, incompatible with the
  `modeling_unit=cjkchar` encoding hayamimi uses for hotwords, so every
  hotword fails to encode (sherpa-onnx only reports this as stderr warnings
  and still exits 0 -- see GitHub issue #1). hayamimi now prints a startup
  warning telling you how many hotwords failed to encode; use `--replace`
  for ja proper nouns instead. A real fix needs either a matching
  `bpe.model` for the ReazonSpeech release (not currently shipped) or a
  from-scratch byte-BPE hotword encoder -- tracked as future work.
- **Two overlapping speakers are not separated.** `--speakers` does
  turn-taking speaker labeling (one embedding per finalized VAD segment,
  nearest-centroid assignment), not true diarization -- simultaneous speech
  gets one label.
- **Translation quality has a real ceiling**, not just a tuning one.
  FuguMT (ja->en) and M2M-100 (ja->zh/ko) are small models; repetition loops
  are suppressed but not eliminated, and numeric values are not reliably
  preserved in ja->zh/ko translation (see `docs/TRANSLATE.md` and
  `docs/TRANSLATE_M2M.md` for measured failure cases before you rely on this
  for anything numeric or financial).
- **The end-to-end mic pipeline has not been independently verified beyond
  this project's own testing** -- see `docs/GOALS.md`'s remaining-work
  section. File an issue if your results differ from the numbers above.

## License

Source code is MIT (`LICENSE`, copyright oboroge0). No model weights are
committed to this repository -- `scripts/download_models.py` fetches them
from their original publishers at install time, and each carries its own
license (`THIRD_PARTY_NOTICES.md` has the full table).

**One model is not permissive:** the ja->en translation model
(`mojicast-fugumt-ja-en-ct2`, used by `--translate en`) is
**CC BY-SA 4.0 (share-alike)**. If you redistribute that model's weights,
you must keep attribution and license any redistribution under CC BY-SA 4.0
too. This does not affect hayamimi's own code license, and does not affect
any other `--translate` target (M2M-100, MIT).

## Credits

hayamimi exists on top of, and would not exist without:

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) -- the ONNX
  Runtime inference engine every model here runs through.
- [ReazonSpeech](https://research.reazon.jp/) (Reazon Human Interaction Lab)
  -- the Japanese ASR model that anchors this project's accuracy claim.
- [NVIDIA NeMo / Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  -- English + 24 European languages.
- [Meta AI Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
  -- the ~1600-language fallback that makes "multilingual" not a lie.
- [FunASR / SenseVoice](https://github.com/FunAudioLLM/SenseVoice) (Alibaba
  DAMO Academy) -- Chinese, Korean, and Cantonese ASR.
- [Mojicast](https://github.com/ishiki-emo/mojicast) (ishiki-emo) -- design
  inspiration for the live-captioning pipeline, and the source of the
  converted punctuation/translation model artifacts this project uses.
  Mojicast is itself a full offline real-time captioning app worth checking
  out.
- [Silero VAD](https://github.com/snakers4/silero-vad) -- voice activity
  detection.
- [3D-Speaker](https://github.com/modelscope/3D-Speaker) (Alibaba DAMO
  Academy) -- the CAM++ speaker embedding model behind `--speakers`.
- [Kiwi](https://github.com/bab2min/kiwipiepy) -- Korean morphological
  tokenizer, used to fix SenseVoice's token-spaced Korean output.

## Contributing

See `CONTRIBUTING.md`.
