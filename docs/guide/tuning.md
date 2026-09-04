# Tuning reference

Every user-facing knob on both sides of hayamimi, with the value it ships
at, where you change it, and the record that set that value. Nothing here is
an estimate: each default is read out of the code named in the row, and each
"evidence" link is the measurement (or the design note) the default came
from. A blank evidence cell means the value is inherited from an upstream
library's own default and this project has not measured an alternative.

Two independent implementations are covered, and they do not share a config
file:

- **Python engine** — `scripts/realtime_transcribe.py` (CLI),
  `scripts/asr_engine.py` (`RoutedASR`), `scripts/subtitle_server.py`
  (`POST /config`).
- **`hayamimi_core`** (Flutter/Dart) — `mobile/hayamimi_core/lib/live/*.dart`.
  Its own guide is
  [`mobile/hayamimi_core/README.md`](../../mobile/hayamimi_core/README.md).

To dump the ja route's fully-resolved configuration as JSON, run
`python scripts/dump_ja_config.py`; the committed copy is
[`../spec/ja_pipeline_spec.json`](../spec/ja_pipeline_spec.json) and the prose
spec it describes is [`../spec/ja_pipeline.ja.md`](../spec/ja_pipeline.ja.md)
(Japanese).

## Python engine

"`RoutedASR` setter" is a method on the live `asr_engine.RoutedASR` instance,
callable from an embedding app (see [`embedding.md`](embedding.md)).
"`POST /config` key" is a key accepted by the `--serve` HTTP endpoint. A knob
with all three is changeable mid-session; a CLI-only knob is fixed for the
life of the process.

### Recognition and routing

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| `--threads` | `4` | CLI flag; `RoutedASR(threads=...)` | intra-op thread count handed to every sherpa-onnx recognizer | |
| `--max-resident` | `3` | CLI flag; `RoutedASR(max_resident=...)` | how many recognizers besides the always-resident ja/en tier stay in memory; the least recently used one is dropped past the cap | [benchmarks](../results/benchmarks.md) iteration #3 (RAM), #7 (LRU unload) |
| `--mode {single,balanced,fast}` | `balanced` | CLI flag | language-switch policy preset; sets the two flags below and `dual_confirm` | [lid.md](../eval/lid.md) |
| `--lang CODE` | `None` | CLI flag; `RoutedASR.set_forced_lang()`; `POST /config` `lang` | pins every segment to one language and skips LID entirely (required by `--mode single`) | |
| `--lang-switch-guard SEC` | `2.0` (`balanced`), `0.0` (`fast`) via `mode_defaults` in `realtime_transcribe.py`; `RoutedASR.min_switch_s` itself is `2.0` | CLI flag; `RoutedASR.set_min_switch_s()`; `POST /config` `min_switch_s` | a new-language detection shorter than this never counts toward a switch | [benchmarks](../results/benchmarks.md) iteration #29; [lid.md](../eval/lid.md) |
| `--lid-switch-confirm N` | `2` (`balanced`), `1` (`fast`) | CLI flag; `RoutedASR.set_lid_switch_confirm()`; `POST /config` `lid_switch_confirm` | consecutive same-language detections needed before the session switches | [benchmarks](../results/benchmarks.md) iteration #29; [lid.md](../eval/lid.md) |
| dual-LID confirmation | on (`RoutedASR(dual_confirm=True)`; off under `--mode fast`) | `RoutedASR.set_dual_confirm()`; `POST /config` `dual_confirm` | whether SenseVoice's own LID must agree with whisper-tiny before a switch into one of ja/en/zh/ko/yue is accepted | [noise.md](../eval/noise.md); [lid.md](../eval/lid.md) |
| `asr_engine.LID_MAX_SECONDS` | `4.0` s | module constant | how much of a segment's head is fed to the LID model | [lid.md](../eval/lid.md) |
| `--hotwords PATH` | `""` | CLI flag; `RoutedASR(hotwords_file=...)` | sherpa-onnx recognizer-level hotword biasing. **No effect on the ja tier** (byte-BPE `tokens.txt` vs. the `cjkchar` modeling unit) | [README](../../README.md), "Limitations" |
| `asr_engine.RZ_HOTWORDS_SCORE` | `2.0` | module constant | hotword boost score where hotwords do encode | |

### Segmentation (VAD)

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| `--min-silence SEC` | `0.35` s | CLI flag; `POST /config` `vad.min_silence` | how long a pause must last before a segment finalizes | [benchmarks](../results/benchmarks.md) iteration #9: 0.5→0.35 cost no CER and finalized 150 ms sooner |
| `--max-speech SEC` | `12.0` s | CLI flag; `POST /config` `vad.max_speech` | when the VAD force-closes a segment that never pauses | [benchmarks](../results/benchmarks.md) iteration #23 |
| VAD `threshold` | `0.5` | `POST /config` `vad.threshold` only (no CLI flag) | Silero's speech-probability cutoff; higher = less sensitive | [diarization.md](../design/diarization.md) section 13 (Round 3 sweep) |
| `realtime_transcribe.VAD_MIN_SPEECH_S` | `0.25` s | module constant | speech runs shorter than this are discarded | |
| `realtime_transcribe.WINDOW_SIZE` | `512` samples (~32 ms @ 16 kHz) | module constant | frame size Silero is fed | |
| `realtime_transcribe.VAD_BUFFER_S` | `30.0` s | module constant | the detector's internal ring buffer | |
| `realtime_transcribe.PREROLL_S` | `1.0` s | module constant | audio prepended before the VAD's detected onset, clamped to the previous segment's end | [head_dropout.md](../eval/head_dropout.md) |

A `vad` change over `POST /config` is the one key that cannot apply
immediately: sherpa-onnx has no in-place setter, so the detector is rebuilt
at the next moment it is not mid-segment. See [`embedding.md`](embedding.md).

### Drafts and the refine ("清書") pass

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| `--no-partial` | off (drafts on) | CLI flag | in-progress draft subtitles | [benchmarks](../results/benchmarks.md) iteration #2 |
| `realtime_transcribe.PARTIAL_EVERY_S` | `0.5` s | module constant | how often a draft re-decodes during speech | |
| `realtime_transcribe.PARTIAL_WINDOW_S` | `8.0` s | module constant | trailing window a draft decode re-processes | |
| `--no-refine` | off (refine on) | CLI flag | the second-pass re-decode of an utterance group | [benchmarks](../results/benchmarks.md) iteration #10 |
| `realtime_transcribe.GROUP_GAP_S` | `2.0` s | module constant | silence that closes an utterance group | |
| `realtime_transcribe.GROUP_MAX_S` | `25.0` s | module constant | length that closes a group early | |
| `asr_engine.REFINE_MIN_REGROUP_S` | `2.5` s | module constant | shortest group worth re-grouping | |
| `--refine-ja-second-opinion` | off | CLI flag; `RoutedASR(ja_second_opinion=...)` | run parakeet-ja as a second opinion on ja refines | [eval_real.md](../eval/eval_real.md) |
| `--refine-agree-threshold CER` | `0.25` (`asr_engine.SECOND_OPINION_THRESHOLD`) | CLI flag; `RoutedASR(agree_threshold=...)` | CER distance below which the two ja recognizers count as agreeing | |
| head-dropout retry | on; `asr_engine.SEGMENT_MIN_S` `4.0` s, `SEGMENT_MIN_SILENCE_S` `0.35` s, `SEGMENT_MIN_SPEECH_S` `0.25` s, `SEGMENT_PAD_S` `0.35` s | module constants | when a suspicious decode is retried, split at internal silences | [head_dropout.md](../eval/head_dropout.md); [benchmarks](../results/benchmarks.md), 2026-08-31 |

### Text post-processing

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| Japanese punctuation | on | `RoutedASR(punctuate=...)`; `RoutedASR.set_punctuate()`; `POST /config` `punctuate` | 、。？ insertion on ja output | [punct_ja.md](../design/punct_ja.md) |
| `--replace PATH` | `""` | CLI flag; `RoutedASR.set_replacements()`; `POST /replacements` | literal string substitutions applied last (the ja proper-noun workaround for hotwords) | [benchmarks](../results/benchmarks.md) iteration #14 |
| ITN overrides | empty | `RoutedASR.set_itn_overrides()`; `POST /itn_overrides` | exceptions to CJK inverse text normalization (spelled-out numerals → digits) | [benchmarks](../results/benchmarks.md) iteration #17 |
| `--translate [LANGS]` | off; `en` when the flag is bare | CLI flag; `POST /config` `translate` (replaces the whole target set) | live translation of ja lines; `en` uses FuguMT, other targets M2M-100 | [translate.md](../design/translate.md); [translate_m2m.md](../design/translate_m2m.md) |

Post-processing order is fixed at ITN → punctuation → replacements
(`dump_ja_config.py`'s `postprocessing_order`).

### Speaker labelling (`--speakers`)

Every threshold here was swept against 5 AMI meetings (50 min total,
CC BY 4.0, collar 0.25 s); the section numbers are in
[`diarization.md`](../design/diarization.md), which is Japanese.

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| `--speakers` | off | CLI flag | speaker labels (`S1`, `S2`, …) on finals and refines | [diarization.md](../design/diarization.md) sections 6-8 |
| `speaker_id.SIM_THRESHOLD` | `0.45` | module constant | cosine similarity to join an existing speaker on the live path | [diarization.md](../design/diarization.md) section 8 |
| `--speaker-remap-threshold T` | `speaker_id.REMAP_THRESHOLD` = `0.35` | CLI flag | the refine path's local-cluster-to-global remap threshold, tuned separately from the live path | [diarization.md](../design/diarization.md) section 8 |
| `diarize.DEFAULT_THRESHOLD` | `0.5` | module constant | clustering threshold inside a refine group | [diarization.md](../design/diarization.md) section 13 |
| `diarize.DEFAULT_MIN_DURATION_ON` | `0.3` s | module constant | shortest speech region the segmentation model reports | [diarization.md](../design/diarization.md) section 12 (Round 2) |
| `diarize.DEFAULT_MIN_DURATION_OFF` | `0.5` s | module constant | shortest silence that separates two regions | [diarization.md](../design/diarization.md) section 12 |
| `--speaker-merge` / `--speaker-merge-threshold T` | off / `speaker_id.MERGE_THRESHOLD` = `0.80` | CLI flags | centroid merging of over-split speakers — **measured and not adopted** | [diarization.md](../design/diarization.md) section 9 |
| `--speaker-hysteresis` / `--speaker-hysteresis-min-hits N` | off / `speaker_id.HYSTERESIS_MIN_HITS` = `2` | CLI flags | new-speaker hysteresis — **measured and not adopted** | [diarization.md](../design/diarization.md) section 9 |
| `speaker_id.PROVISIONAL_CONFIRM_HITS` | `2` | module constant | how many appearances resolve a provisional `S5?` label to `S5` | [diarization.md](../design/diarization.md) section 11 |
| `--speaker-min-remap-update-s S` | `0.0` (no-op) | CLI flag | read-only remap for clusters shorter than S | [diarization.md](../design/diarization.md) section 14 (Round 4, T2) |
| `--speaker-joint-remap` | off | CLI flag | Hungarian assignment for a group's remap | [diarization.md](../design/diarization.md) section 15 (Round 5, T1) |
| `--speaker-exclude-provisional-remap` | off | CLI flag | never remap onto a still-provisional centroid | [diarization.md](../design/diarization.md) section 15 (T3) |
| `--speaker-global-recluster` | off | CLI flag | end-of-session re-clustering diagnostic — **measured and not adopted** (confusion +22.5 pt) | [diarization.md](../design/diarization.md) section 17 |
| `--speaker-global-recluster-threshold T` | `0.65` | CLI flag | that diagnostic's agglomerative merge threshold | [diarization.md](../design/diarization.md) section 17 |
| `--speaker-num-clusters-hint` / `--speaker-num-clusters-hint-min-s` | `off` / `0.0` | CLI flags | pass a cluster-count hint into `FastClustering` — **measured and not adopted** | [diarization.md](../design/diarization.md) section 19 |
| `speaker_id.MAX_EMBED_SECONDS` | `6.0` s | module constant | input-length cap for one speaker embedding | |

### Server and I/O

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| `--serve [PORT]` | off; `8833` when the flag is bare | CLI flag | the HTTP dashboard, SSE `/events`, and the JSON control endpoints | [embedding.md](embedding.md) |
| `--input {mic,wav,ws}` | inferred (`wav` when `--wav` is given, else `mic`) | CLI flag | audio source | |
| `--ws-host` | `127.0.0.1` | CLI flag | bind address for `--input ws`; `/ingest` has no authentication | |
| `--ws-port` | `8766` | CLI flag | port for `--input ws` | |
| `--transcript PATH` | off | CLI flag | append finals and refines to a file | |
| `realtime_transcribe.RESET_TIMEOUT_S` | `10.0` s | module constant | how long `POST /reset` waits for a chunk boundary before answering `202` | [embedding.md](embedding.md) |

## `hayamimi_core` (Flutter/Dart)

The package's own reference is
[`mobile/hayamimi_core/README.md`](../../mobile/hayamimi_core/README.md);
this table is the same knobs in one place, with what set each default.
"constructor param" means `LiveTranscriber`/`HayamimiLive`'s constructor;
"runtime setter" means a same-named property that can be reassigned
mid-session — it takes effect at the next due-check or buffer write, and an
invalid (non-positive or non-finite) value throws `ArgumentError` at once.

| Knob | Default | Where to change it | What it affects | Evidence |
|---|---|---|---|---|
| `draftIntervalSeconds` | `1.0` s (`defaultDraftIntervalSeconds`, `live/draft_pass.dart`) | constructor param; runtime setter | how often a draft re-decode fires during a segment | [android_emulator.md](../verify/android_emulator.md) |
| `draftWindowSeconds` | `8.0` s (`defaultDraftWindowSeconds`) | constructor param; runtime setter | trailing audio window a draft re-processes | [android_emulator.md](../verify/android_emulator.md) |
| `minDraftAudioSeconds` | `0.25` s (`defaultMinDraftAudioSeconds`) | constructor param; runtime setter | least audio worth a draft decode | |
| `autoRefineSilenceSeconds` | `4.0` s (`defaultAutoRefineSilenceSeconds`, `live/refine_pass.dart`) | constructor param; runtime setter | silence gap that fires an auto-refine | [android_emulator.md](../verify/android_emulator.md), run 3 |
| `autoRefineMaxBufferedSeconds` | `20.0` s (`defaultAutoRefineMaxBufferedSeconds`) | constructor param; runtime setter | buffered-duration ceiling that fires an auto-refine without a gap | |
| `refineBufferMaxSeconds` | `60.0` s (`defaultRefineBufferMaxSeconds`) | constructor param; runtime setter | hard cap on the refine buffer before the oldest segment is dropped | |
| `prerollSeconds` | `1.0` s (`defaultPrerollSeconds`, `live/preroll.dart`) | constructor param; runtime setter | audio prepended before a segment's detected onset; `0` disables it | [android_emulator.md](../verify/android_emulator.md), run 1 (`資料は昨日送りました` came back as `昨日は昨日送りました`); [head_dropout.md](../eval/head_dropout.md) for the desktop equivalent |
| `defaultPrerollKeepSeconds` | `30.0` s (`live/preroll.dart`) | module constant | how much past audio the pre-roll history retains | |
| `VadSensitivity.threshold` | `0.5` | `start(vadSensitivity:)`; `setVadSensitivity()` | Silero speech-probability cutoff (sherpa-onnx's own value, which the desktop also leaves alone) | [diarization.md](../design/diarization.md) section 13 |
| `VadSensitivity.minSilenceSeconds` | `0.35` s | `start(vadSensitivity:)`; `setVadSensitivity()` | pause before a segment finalizes. **Not sherpa-onnx's 0.5**: at 0.5 an emulator run merged three Japanese sentences into one 6.13 s segment and the recognizer returned only the last one | [android_emulator.md](../verify/android_emulator.md), run 1; [benchmarks](../results/benchmarks.md) iteration #9 |
| `VadSensitivity.minSpeechSeconds` | `0.25` s | `start(vadSensitivity:)`; `setVadSensitivity()` | shorter blips are discarded before decoding (sherpa-onnx's own value) | |
| `VadSensitivity.maxSpeechSeconds` | `12.0` s | `start(vadSensitivity:)`; `setVadSensitivity()` | a nudge, not a bound: a session configured with 5.0 s still emitted a 6.134 s segment, so do not size a buffer or a timeout on it | [android_emulator.md](../verify/android_emulator.md), run 1 |
| `defaultMinDecodeDurationSeconds` | `0.2` s (`live/speech_segment_filter.dart`) | module constant | segments shorter than this are never decoded | |
| `decodingMethod` | `null`, meaning `greedy_search` on the plain path and `modified_beam_search` on `RoutingProfile.jaSenseVoice`'s ja tier | `start()` / `startDebugWavStream()` param | search algorithm; `modified_beam_search` is the desktop production value | [mobile_quantization.md](../design/mobile_quantization.md) |
| `hotwordsFile` / `hotwordsScore` | `null` / `1.5` | `start()` / `startDebugWavStream()` params — **no runtime setter**, they are compiled into the recognizer | recognizer-level hotword biasing on the plain path and the routed ja tier | |
| `defaultPunctNumThreads` | `2` (`live/ja_punctuation.dart`) | `JaPunctuation` constructor | ONNX intra-op threads for the punctuation model | [android_emulator.md](../verify/android_emulator.md), run 3: about 38-49 ms per 11-14 character line, warm |
| `JaPunctuation(applyToFinals:)` | `true` | `JaPunctuation` constructor | whether fast finals are punctuated as well as refines; `false` restores refine-only behaviour | [android_emulator.md](../verify/android_emulator.md), run 3 |
| `SubtitleBroadcastServer.defaultPort` | `8833` (`lib/server/subtitle_broadcast_server.dart`) | constructor param | LAN broadcast port, matching the Python `--serve` default | |

### Not exposed, by design

Internal timeouts and guards with no constructor parameter and no setter.
They are listed so you know they exist and stop looking for a knob; changing
one means editing the package.

| Constant | Value | Where | Why it is not a knob |
|---|---|---|---|
| `_decodeDrainTimeout` | 10 s | `live/live_transcriber.dart` | how long `stop()` waits for work still queued at the decode worker. Generous next to the fraction of a second a segment takes, short next to a user waiting for a screen to close — a bound against a wedged worker, not a tuning dial. |
| `defaultDecodeWorkerShutdownTimeout` | 3 s | `live/decode_worker.dart` | wait for the worker isolate's shutdown ack. A worker that has not answered by then is stuck inside a decode that is not going to return, and there is nothing useful left to wait for. It is a named `const` so tests can reason about it, not a public knob. |
| `_autoRefineCheckInterval` | 1 s | `live/live_transcriber.dart` | how often the auto-refine due check runs. The check is a few comparisons, so a 1 s tick is responsive without meaningfully affecting battery. |
| `_minRefineAudioSeconds` | 0.5 s | `live/live_transcriber.dart` | the least audio a refine pass is worth running over. Mirrors the desktop `Refiner`'s own `len(buf) < sr // 2` guard: re-decoding a fraction of a second costs a full decode and cannot beat the final that already covered it. |
