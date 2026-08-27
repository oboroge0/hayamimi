"""Lightweight per-utterance speaker labeling (S1, S2, ...).

Not full diarization: each finalized VAD segment gets one speaker embedding
(CAM++ zh-en, 28MB) and is assigned to the nearest running centroid by
cosine similarity, or opens a new speaker when nothing is close enough.
Good for turn-taking conversations; overlapping speech stays one label.
"""
import os

import numpy as np
import sherpa_onnx

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EMBED_MODEL = os.path.join(MODELS_DIR, "campplus_sv.onnx")

SIM_THRESHOLD = 0.45  # cosine similarity to join an existing speaker (fast path)
MAX_EMBED_SECONDS = 6.0  # embeddings saturate; cap input length

# Threshold for the refine-path local-cluster-to-global remap only (see
# match_embedding()'s threshold= override, and __init__'s docstring below).
# docs/DIARIZATION_PLAN.md section 8 (iteration 5) swept this independently
# of SIM_THRESHOLD on the AMI dev meetings (ES2011a, IS1008a) and confirmed
# on the 3 test meetings: 0.35 gave the best DER/speaker-count trade-off
# (mean DER ~13.9% over 5 meetings vs ~14.1-14.3% at the old single
# threshold=0.45, and hypothesized speaker counts moved closer to the
# reference 4, e.g. ES2004a 10-11->8, IS1009a 8-9->6, TS3003a 8->7 --
# still overestimating, but the single biggest lever found so far).
REMAP_THRESHOLD = 0.35


class SpeakerLabeler:
    def __init__(self, threads: int = 2, threshold: float = SIM_THRESHOLD,
                 remap_threshold: float | None = REMAP_THRESHOLD):
        """threshold governs the fast path (label()/match_embedding() calls
        that don't pass their own threshold -- one embedding per VAD
        segment, called often). remap_threshold governs calls that pass
        threshold=None to match_embedding() explicitly for the refine-path
        local-cluster-to-global remap (scripts/diarize.py's GroupDiarizer
        output going through realtime_transcribe.Refiner._emit_turns()):
        defaults to REMAP_THRESHOLD (pass None explicitly to fall back to
        `threshold` instead, e.g. for a caller that wants the old
        single-threshold behavior).

        docs/DIARIZATION_PLAN.md section 7 found DER improved a lot under
        the refine path but global speaker count got *worse* (more S{n}
        splitting) -- the remap call re-matches embeddings far more often
        than the fast path does (once per local diarization cluster per
        refine group, instead of once per VAD segment), so the same 0.45
        threshold trips into "new speaker" more often there. Splitting the
        threshold in two lets a stricter (lower) remap_threshold curb that
        over-splitting without touching the fast path's own behavior.
        Section 8 found lowering remap_threshold alone (fast path left at
        SIM_THRESHOLD=0.45) beat lowering both thresholds together, so that
        is the default here -- see REMAP_THRESHOLD's comment above.
        """
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=EMBED_MODEL, num_threads=threads
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._threshold = threshold
        self._remap_threshold = threshold if remap_threshold is None else remap_threshold
        self._centroids: list[np.ndarray] = []  # running mean per speaker
        self._counts: list[int] = []

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute an L2-normalized CAM++ embedding for one audio buffer.

        Split out of label() so callers that already have their own
        clustering (e.g. scripts/diarize.py's GroupDiarizer, whose local
        speaker clusters need remapping onto this labeler's global
        centroids -- docs/DIARIZATION_PLAN.md iteration 4) can get the same
        embedding label() would compute without going through its
        assign-or-open-new-speaker side effects.
        """
        max_len = int(MAX_EMBED_SECONDS * sample_rate)
        if len(samples) > max_len:
            samples = samples[:max_len]
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        return emb

    @property
    def remap_threshold(self) -> float:
        return self._remap_threshold

    def match_embedding(self, emb: np.ndarray, update: bool = True,
                         threshold: float | None = None) -> str:
        """Assign a precomputed embedding to the nearest global centroid
        (or open a new speaker), same policy as label() but for a caller
        that already has an embedding (see embed()).

        update=False looks up the nearest speaker without folding the
        embedding into the running centroid mean and without opening a new
        speaker on a miss (returns "" instead) -- used for read-only
        lookups where mutating session state would be wrong (e.g. probing
        which existing global speaker a diarization cluster most resembles
        before deciding whether it deserves a brand-new global label).

        threshold overrides self._threshold for this one call -- pass
        self.remap_threshold from the refine-path local-cluster-to-global
        remap call sites (realtime_transcribe.Refiner._emit_turns(),
        eval_diar.py's --method refine_diarize) so that path can use a
        different (independently tuned) threshold than the fast path's
        label() calls, which always use self._threshold. See __init__'s
        docstring for why the two calls warrant separate thresholds.
        """
        thr = self._threshold if threshold is None else threshold
        best, best_sim = -1, -1.0
        for i, c in enumerate(self._centroids):
            sim = float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9))
            if sim > best_sim:
                best, best_sim = i, sim

        if best >= 0 and best_sim >= thr:
            if update:
                n = self._counts[best]
                self._centroids[best] = (self._centroids[best] * n + emb) / (n + 1)
                self._counts[best] = n + 1
            return f"S{best + 1}"

        if not update:
            return ""
        self._centroids.append(emb)
        self._counts.append(1)
        return f"S{len(self._centroids)}"

    def label(self, samples: np.ndarray, sample_rate: int) -> str:
        emb = self.embed(samples, sample_rate)
        return self.match_embedding(emb, update=True)
