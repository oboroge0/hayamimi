"""Offline speaker diarization for one bounded audio buffer (a Refiner group).

Wraps sherpa_onnx.OfflineSpeakerDiarization: pyannote segmentation-3.0 for
speaker-change/voice-activity detection, the same CAM++ embedding model
scripts/speaker_id.py already uses, and FastClustering for the actual
speaker assignment. See docs/DIARIZATION_PLAN.md sections 2-3 for the
design rationale (this is the "clustering diarizer" that iteration (3)
plugs into Refiner.maybe_refine()).

Not a streaming API: process() takes the whole buffer at once and is meant
to be called on bounded, already-finalized audio (a Refiner group, at most
GROUP_MAX_S=25s), not on unbounded live audio.
"""
import os

import numpy as np
import sherpa_onnx

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EMBED_MODEL = os.path.join(MODELS_DIR, "campplus_sv.onnx")
SEGMENTATION_MODEL = os.path.join(
    MODELS_DIR, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"
)

# FastClusteringConfig.threshold: lower = more willing to merge two segments
# into the same speaker (fewer, larger clusters); higher = more willing to
# split (more clusters). This is the diarizer's analogue of speaker_id.py's
# SIM_THRESHOLD=0.45, but the two are not directly comparable -- FastClustering
# clusters embeddings with (1 - cosine_similarity) distance internally, and
# operates on pyannote-segmentation-bounded speech turns rather than raw VAD
# segments, so it tends to need different tuning. See docs/DIARIZATION_PLAN.md
# section 6 (iteration 3 sweep) for the values tried.
DEFAULT_THRESHOLD = 0.5

# OfflineSpeakerDiarizationConfig's VAD-adjacent knobs (same sherpa-onnx
# semantics as the segmentation model's underlying VAD post-processing):
# min_duration_on drops speech turns shorter than this after segmentation,
# min_duration_off bridges (merges across) silence gaps shorter than this.
# docs/DIARIZATION_PLAN.md Round 2 (section 12) sweeps these -- see that
# section for the min-silence-0.5-equivalent confusion-vs-latency tradeoff
# that motivated exposing them instead of only tuning production's
# min_silence_duration.
DEFAULT_MIN_DURATION_ON = 0.3
DEFAULT_MIN_DURATION_OFF = 0.5


class GroupDiarizer:
    """Offline diarizer for a single bounded audio buffer.

    Usage:
        diarizer = GroupDiarizer()
        segments = diarizer.process(samples, sample_rate)
        # segments: list of (local_speaker_id: int, start_s: float, end_s: float)
        # sorted by start time, local_speaker_id is 0-based and only
        # meaningful within this one process() call (no session identity).
    """

    def __init__(self, threads: int = 2, threshold: float = DEFAULT_THRESHOLD,
                 num_clusters: int = -1,
                 min_duration_on: float = DEFAULT_MIN_DURATION_ON,
                 min_duration_off: float = DEFAULT_MIN_DURATION_OFF):
        if not os.path.exists(SEGMENTATION_MODEL):
            raise FileNotFoundError(
                f"pyannote segmentation model missing: {SEGMENTATION_MODEL}\n"
                "download it: see docs/DIARIZATION_PLAN.md section 2, or run "
                "`curl -L -o /tmp/seg.tar.bz2 "
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
                " && tar xjf /tmp/seg.tar.bz2 -C models/`"
            )
        self._threads = threads
        self._threshold = threshold
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off
        self._default_num_clusters = num_clusters
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(self._build_config(num_clusters))
        self.sample_rate = self._sd.sample_rate
        # Round 9 Experiment A (docs/DIARIZATION_PLAN.md section 19): tracks
        # whichever num_clusters value is CURRENTLY live in self._sd's
        # config, so process() only calls the (cheap but not free) set_config
        # when a call's requested hint actually differs from what's already
        # applied -- most calls in a session repeat the same value (or None,
        # i.e. this constructor's default) back-to-back.
        self._live_num_clusters = num_clusters

    def _build_config(self, num_clusters: int) -> "sherpa_onnx.OfflineSpeakerDiarizationConfig":
        return sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=SEGMENTATION_MODEL
                ),
                num_threads=self._threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=EMBED_MODEL, num_threads=self._threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters, threshold=self._threshold,
            ),
            min_duration_on=self._min_duration_on,
            min_duration_off=self._min_duration_off,
        )

    def process(self, samples: np.ndarray, sample_rate: int,
                num_clusters: int | None = None
               ) -> list[tuple[int, float, float]]:
        """Diarize one buffer. Resamples to the model's expected rate if needed.

        Returns [] for audio too short/quiet for the segmentation model to
        find any speech turn (callers should fall back to their existing
        single-label behavior in that case).

        num_clusters (Round 9 Experiment A, docs/DIARIZATION_PLAN.md section
        19): None (default) is a no-op -- this call uses whatever
        FastClusteringConfig.num_clusters the constructor was given (-1 =
        "let FastClustering pick the count itself", sherpa-onnx's own
        default and every prior round's behavior). When an int is given, it
        OVERRIDES the live config for this call via sherpa_onnx's
        OfflineSpeakerDiarization.set_config() -- a cheap in-place config
        swap (confirmed via direct inspection of the installed sherpa_onnx
        1.13.6 API: no model reload) -- so a caller can pass a per-group
        hint (e.g. the session's confirmed global speaker count) without
        paying GroupDiarizer's model-loading cost per call. The requested
        value is only applied when it differs from what's already live, so
        back-to-back calls with the same hint (or no hint) cost nothing
        extra.
        """
        if sample_rate != self.sample_rate:
            from audio_utils import resample_linear

            samples = resample_linear(samples, sample_rate, self.sample_rate)
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) < self.sample_rate // 2:  # <0.5s: not worth calling
            return []
        effective = self._default_num_clusters if num_clusters is None else num_clusters
        if effective != self._live_num_clusters:
            self._sd.set_config(self._build_config(effective))
            self._live_num_clusters = effective
        result = self._sd.process(samples)
        segments = result.sort_by_start_time()
        return [(seg.speaker, seg.start, seg.end) for seg in segments]
