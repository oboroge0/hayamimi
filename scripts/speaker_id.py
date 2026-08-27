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

SIM_THRESHOLD = 0.45  # cosine similarity to join an existing speaker
MAX_EMBED_SECONDS = 6.0  # embeddings saturate; cap input length


class SpeakerLabeler:
    def __init__(self, threads: int = 2, threshold: float = SIM_THRESHOLD):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=EMBED_MODEL, num_threads=threads
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._threshold = threshold
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

    def match_embedding(self, emb: np.ndarray, update: bool = True) -> str:
        """Assign a precomputed embedding to the nearest global centroid
        (or open a new speaker), same policy as label() but for a caller
        that already has an embedding (see embed()).

        update=False looks up the nearest speaker without folding the
        embedding into the running centroid mean and without opening a new
        speaker on a miss (returns "" instead) -- used for read-only
        lookups where mutating session state would be wrong (e.g. probing
        which existing global speaker a diarization cluster most resembles
        before deciding whether it deserves a brand-new global label).
        """
        best, best_sim = -1, -1.0
        for i, c in enumerate(self._centroids):
            sim = float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9))
            if sim > best_sim:
                best, best_sim = i, sim

        if best >= 0 and best_sim >= self._threshold:
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
