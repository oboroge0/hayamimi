# Demo video audio credits

## v0.3.0 speaker-diarization video (`hayamimi_v030_diar_demo.mp4`)

The only speech heard in this video is an unmodified 85-second excerpt of a
real four-person meeting from the **AMI Meeting Corpus**, used under
**CC BY 4.0**:

> J. Carletta, S. Ashby, S. Bourban, M. Flynn, M. Guillemot, T. Hain,
> J. Kadlec, V. Karaiskos, W. Kraaij, M. Kronenthal, G. Lathoud, M. Lincoln,
> A. Lisowska, I. McCowan, W. Post, D. Reidsma and P. Wellner.
> "The AMI Meeting Corpus: A Pre-Announcement." In *Machine Learning for
> Multimodal Interaction* (MLMI 2005), LNCS 3869, pp. 28–39. Springer, 2006.
> Corpus home: https://groups.inf.ed.ac.uk/ami/corpus/ — licensed CC BY 4.0.

Specific material: meeting **IS1008a**, the segment at meeting time
60.0s–145.0s (the opening round of self-introductions). Cut without
re-encoding from the 16 kHz mono dev-split copy in `testdata/eval_diar/`;
see `demo/diar/NOTES.md` for the exact cut and checksums. Original speaker
identities in the corpus reference are MIO086, FIE038, FIE073 and MIE085;
the `S1`–`S4` labels on screen are hayamimi's own cluster labels, not the
corpus's.

Every subtitle line, speaker label and `?` marker shown on screen is
hayamimi's actual output on that audio, replayed at the real timestamps
from a timestamped capture of the production pipeline
(`demo/diar/events.json`, 146 events) without alteration — including the
mistake at t=38.5s and its correction at t=58.1s. Nothing was retyped,
re-timed or staged.

### Background music

The instrumental bed under the video is **not licensed from anyone** — it is
synthesized from scratch by `demo/diar/make_bgm.py` (numpy + scipy: additive
sine pads over a four-chord diatonic loop in D major, a sine sub-bass and a
low-passed noise layer; no samples, no external audio, no melody). Output:
`demo/diar/bgm.wav`. It is an original work of this repository and carries the
repository's own licence, so the finished video has no third-party music
rights attached at all.

## v0.1.0 / v0.2.0 videos

The demo video's speech clips are unmodified samples from these datasets
(all redistribution-compatible, attribution required):

- Japanese: FLEURS (google/fleurs, ja_jp) — CC-BY-4.0, (c) Google.
- English: LibriSpeech dev-clean (openslr/12) — CC-BY-4.0, read by
  LibriVox volunteers from public-domain texts.
- Korean / Chinese: FLEURS (ko_kr, cmn_hans_cn) — CC-BY-4.0, (c) Google.

Transcriptions shown in the video are hayamimi's actual output on these
clips, replayed from a timestamped capture (demo/events.json) without
alteration. ReazonSpeech audio is NOT used in the video (its corpus terms
restrict use to text-and-data-mining under Art. 30-4 of the Japanese
Copyright Act; we use it only for offline evaluation, never redistributed).
