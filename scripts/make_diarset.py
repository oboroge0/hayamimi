"""Build a small speaker-diarization evaluation set from the AMI Meeting Corpus.

docs/DIARIZATION_PLAN.md iteration (1): DER scoring needs reference audio +
reference speaker-turn labels. AMI ("headset-mix", CC BY 4.0) is the
recommended starting point there -- a clean multi-speaker meeting-room
recording with a standard reference RTTM maintained by BUT (the
AMI-diarization-setup repo), widely used as a diarization benchmark.

For each selected meeting this script:
  1. downloads the official "Mix-Headset" (all-headsets-summed) WAV from the
     AMI corpus mirror at Edinburgh (groups.inf.ed.ac.uk);
  2. downloads the matching reference RTTM (word-level force-aligned speaker
     turns) from BUTSpeechFIT/AMI-diarization-setup on GitHub;
  3. cuts a WINDOW_S-second slice out of the full meeting (skipping the
     opening minute, which in AMI meetings is mostly silence/settling-in) so
     the eval set stays small;
  4. shifts the RTTM turns to the cut's local timeline and clips them to
     [0, WINDOW_S].

Output goes to testdata/eval_diar/ (git-ignored, like the rest of testdata/):
  - <meeting>.wav   16kHz mono s16 PCM, WINDOW_S seconds
  - <meeting>.rttm  reference speaker turns, local timeline
  - manifest.json   [{"meeting", "wav", "rttm", "split", "duration_s",
                       "start_offset_s", "n_speakers"}, ...]
  - README.txt      provenance + license note (also written here, see below)

Usage:
    python scripts/make_diarset.py                # download + cut everything
    python scripts/make_diarset.py --skip-existing # skip meetings already cut
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIAR_DIR = os.path.join(ROOT, "testdata", "eval_diar")
MANIFEST_PATH = os.path.join(EVAL_DIAR_DIR, "manifest.json")
README_PATH = os.path.join(EVAL_DIAR_DIR, "README.txt")

AMI_AUDIO_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_RTTM_BASE = (
    "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms"
)

# 5 meetings across AMI's official dev/test split (see BUTSpeechFIT's
# lists/{dev,test}.meetings.txt), picked to keep total download small (~35MB
# WAV each) while covering both splits. All have 4 headset-mic speakers.
MEETINGS = [
    ("ES2011a", "dev"),
    ("IS1008a", "dev"),
    ("ES2004a", "test"),
    ("IS1009a", "test"),
    ("TS3003a", "test"),
]

START_OFFSET_S = 60.0   # skip the opening minute (setup/silence in AMI recordings)
WINDOW_S = 600.0        # 10-minute slice per meeting


def download(url: str, dest_path: str):
    req = urllib.request.Request(url, headers={"User-Agent": "hayamimi-diarset/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)


def parse_rttm_lines(text: str) -> list[tuple[str, float, float]]:
    """Parse RTTM `SPEAKER` lines into (speaker, start, end) tuples.

    Standard NIST RTTM SPEAKER line:
        SPEAKER <file> <chnl> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    Non-SPEAKER lines (rare in AMI's RTTMs) are ignored.
    """
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]
        out.append((speaker, start, start + dur))
    return out


def shift_and_clip_rttm(turns: list[tuple[str, float, float]], offset: float,
                        window: float) -> list[tuple[str, float, float]]:
    """Shift turns so `offset` becomes t=0, then clip to [0, window].

    Turns entirely outside the window are dropped; turns straddling a window
    edge are truncated to it.
    """
    out = []
    for speaker, start, end in turns:
        s, e = start - offset, end - offset
        s, e = max(s, 0.0), min(e, window)
        if e > s:
            out.append((speaker, s, e))
    return out


def write_rttm(path: str, meeting: str, turns: list[tuple[str, float, float]]):
    with open(path, "w", encoding="utf-8") as f:
        for speaker, start, end in turns:
            f.write(
                f"SPEAKER {meeting} 1 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )


def ffmpeg_cut_to_wav16k(src_path: str, dst_path: str, start: float, duration: float):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-t", str(duration),
            "-i", src_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav",
            dst_path,
        ],
        check=True,
    )


def build_one(meeting: str, split: str, skip_existing: bool) -> dict:
    wav_out = os.path.join(EVAL_DIAR_DIR, f"{meeting}.wav")
    rttm_out = os.path.join(EVAL_DIAR_DIR, f"{meeting}.rttm")

    rttm_url = f"{AMI_RTTM_BASE}/{split}/{meeting}.rttm"
    req = urllib.request.Request(rttm_url, headers={"User-Agent": "hayamimi-diarset/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        rttm_text = resp.read().decode("utf-8")
    full_turns = parse_rttm_lines(rttm_text)
    n_speakers = len({s for s, _, _ in full_turns})

    if skip_existing and os.path.exists(wav_out) and os.path.exists(rttm_out):
        print(f"  [{meeting}] already present, skipping download/cut")
    else:
        os.makedirs(EVAL_DIAR_DIR, exist_ok=True)
        audio_url = f"{AMI_AUDIO_BASE}/{meeting}/audio/{meeting}.Mix-Headset.wav"
        with tempfile.TemporaryDirectory(prefix="amidl_") as tmpdir:
            raw_path = os.path.join(tmpdir, f"{meeting}.wav")
            print(f"  [{meeting}] downloading {audio_url} ...")
            download(audio_url, raw_path)
            print(f"  [{meeting}] cutting [{START_OFFSET_S:.0f}s, "
                  f"{START_OFFSET_S + WINDOW_S:.0f}s) -> {wav_out}")
            ffmpeg_cut_to_wav16k(raw_path, wav_out, START_OFFSET_S, WINDOW_S)

        cut_turns = shift_and_clip_rttm(full_turns, START_OFFSET_S, WINDOW_S)
        write_rttm(rttm_out, meeting, cut_turns)
        print(f"  [{meeting}] wrote {rttm_out} ({len(cut_turns)} turns, "
              f"{n_speakers} speakers)")

    return {
        "meeting": meeting,
        "split": split,
        "wav": f"{meeting}.wav",
        "rttm": f"{meeting}.rttm",
        "duration_s": WINDOW_S,
        "start_offset_s": START_OFFSET_S,
        "n_speakers": n_speakers,
    }


README_TEXT = """testdata/eval_diar/ -- speaker diarization evaluation subset
=============================================================

Generated by scripts/make_diarset.py. Not committed to git (see .gitignore's
testdata/ exclusion) -- re-run the script to regenerate.

Source
------
Audio: AMI Meeting Corpus (http://groups.inf.ed.ac.uk/ami/corpus/), the
"Mix-Headset" per-meeting mono mix (all close-talk headset mics summed),
downloaded from the official Edinburgh mirror
(groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/).

Reference labels: RTTM speaker-turn annotations from
BUTSpeechFIT/AMI-diarization-setup
(https://github.com/BUTSpeechFIT/AMI-diarization-setup), the "only_words"
variant (word-level force-aligned speech regions; excludes vocal sounds like
laughs/coughs that the "word_and_vocalsounds" variant includes). This is the
de facto standard reference RTTM used across diarization papers benchmarking
on AMI, rather than the raw NXT XML annotations.

License
-------
AMI Meeting Corpus: CC BY 4.0
(https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).
AMI-diarization-setup (BUT's RTTM derivation): MIT license, see the repo's
LICENSE file.

Redistribution note: this directory is git-ignored specifically so hayamimi
does not redistribute AMI audio; only the download script (which fetches
from the original sources) is committed.

What's in each meeting
-----------------------
Each meeting is a %(window)ss slice starting %(offset)ss into the full
recording (skipping the opening minute, which in AMI recordings is mostly
room noise / participants settling in before the meeting proper starts).
RTTM turn times are shifted so the slice's local t=0 lines up with the WAV.

Meetings (dev/test per AMI's official split, lists/{dev,test}.meetings.txt
in AMI-diarization-setup):
%(meetings)s

manifest.json holds the same info in machine-readable form.
"""


def write_readme(entries: list[dict]):
    lines = "\n".join(
        f"  - {e['meeting']} ({e['split']}, {e['n_speakers']} speakers)"
        for e in entries
    )
    text = README_TEXT % {
        "window": f"{WINDOW_S:.0f}",
        "offset": f"{START_OFFSET_S:.0f}",
        "meetings": lines,
    }
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't re-download/cut a meeting whose wav+rttm are already present")
    args = ap.parse_args()

    os.makedirs(EVAL_DIAR_DIR, exist_ok=True)
    entries = []
    for meeting, split in MEETINGS:
        print(f"[{meeting}] ({split})")
        entries.append(build_one(meeting, split, args.skip_existing))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    write_readme(entries)

    total = sum(e["duration_s"] for e in entries)
    print(f"\nWrote {MANIFEST_PATH}: {len(entries)} meetings, "
          f"{total:.0f}s ({total / 60:.1f} min) total audio")
    print(f"Wrote {README_PATH}")


if __name__ == "__main__":
    main()
