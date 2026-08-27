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

# docs/DIARIZATION_PLAN.md section 8's "残課題" identified the fast path
# (label(), SIM_THRESHOLD=0.45) as the root cause of the remaining
# speaker-count overestimation: it keeps opening a fresh global centroid
# any time one VAD segment's embedding falls short of 0.45 against every
# existing centroid, with no mechanism to notice later that two of its
# centroids actually belong to the same speaker (voice drift across a
# meeting, a noisy single segment, etc.). Section 9 (iteration 6) added
# two independent, off-by-default mitigations for that:
#
#   A. Periodic centroid merging (merge_enabled): at each refine-path
#      "clean copy" boundary (Refiner._emit_turns() / eval_diar.py's
#      generate_diarize_hypothesis(), once per closed utterance group),
#      call maybe_merge_centroids() to fold together any two global
#      centroids whose cosine similarity has drifted above
#      merge_threshold. Past emitted labels are not rewritten (that would
#      need re-sending already-printed lines -- out of scope, see
#      docs/DIARIZATION_PLAN.md section 9), but every later label for
#      either speaker converges on the surviving (lower-numbered) one.
#      merge_history() exposes the old->new label map for a session
#      summary.
#   B. New-speaker hysteresis (hysteresis_enabled): a centroid opened by
#      a fast-path miss starts "provisional" and is displayed under its
#      nearest already-confirmed speaker's label instead of a brand-new
#      S{n} until it has itself been the best match
#      hysteresis_min_hits times (a real recurring voice, not a one-off
#      embedding outlier). The very first speaker of a session has no
#      confirmed speaker to fall back to, so it is confirmed immediately.
#
# Both are independently toggleable, and BOTH default to off as of section
# 9 -- neither survived the full evaluation (AMI sweep + a real short
# two-speaker recording) cleanly enough to flip on by default:
#
#   A rejected: docs/DIARIZATION_PLAN.md section 9's dev sweep (ES2011a,
#   IS1008a) found merging fragile -- its effective threshold range is a
#   knife-edge (0.55-0.60 catastrophically merged *unrelated* speakers on
#   ES2011a, DER 49-50%; 0.65-0.75 either matched baseline exactly or
#   barely nudged the count; IS1008a never merged anything at any
#   threshold tried). No threshold gave a safe, reliable win.
#
#   B rejected: hysteresis (min_hits=2) looked promising on the AMI sweep
#   -- it cut speaker-count overestimation on 2 of 5 meetings (IS1008a dev
#   7->5, TS3003a test 7->5) with no DER regression anywhere in the
#   dev+test sweep (min_hits=3 was tried too and caused a real regression
#   on ES2004a, +6.9pp DER, for no count improvement there). But testing
#   it against testdata/two_speakers.wav (a real, short two-speaker
#   recording -- tests/test_diarize.py's fixture) surfaced a worse
#   failure mode than the one it fixes: that recording's second speaker
#   only speaks once (one ~3s segment among four), so with min_hits=2 the
#   fast path never accumulates enough hits to confirm them -- they get
#   permanently displayed under the *other* speaker's label instead of
#   getting their own. For the module's stated use case ("turn-taking
#   conversations", commonly just 1-2 speakers -- see the module
#   docstring), silently merging a real second speaker into the first one
#   is a worse user-facing bug than overcounting extra S{n} labels in a
#   many-speaker meeting. See tests/test_speaker_id.py's
#   test_hysteresis_can_swallow_a_rare_real_speaker for the reproduction.
#
# Both mitigations stay implemented and available (a caller who mainly
# runs multi-speaker meetings and doesn't mind this trade-off can still
# opt in), but neither is recommended, and speaker-count overestimation
# in meetings remains open for a future iteration -- see
# docs/DIARIZATION_PLAN.md section 9's "残課題".
MERGE_THRESHOLD = 0.80
HYSTERESIS_MIN_HITS = 2

# GitHub issue #11 / docs/DIARIZATION_PLAN.md section 10.8's option B: a
# DISPLAY-ONLY mitigation for the same speaker-count-overestimation problem
# section 9 tried (and rejected) two assignment-changing fixes for. Unlike
# those, this one never touches which centroid an embedding gets assigned to
# -- match_embedding()/label() keep returning the exact same canonical
# "S{n}" they always have, and every caller that groups or votes on those
# labels (Refiner's majority vote, eval_diar.py's DER hypothesis) keeps
# working on the unchanged canonical value. is_provisional()/display_label()
# are a pure read of _counts, applied by a caller only at the moment it
# actually prints/publishes a label, to decide whether to show it as
# provisional. A centroid is provisional from the moment it opens
# (final_match_count == 1, section 10.8's "one-off outlier" case) until it
# is matched a second time (final_match_count >= 2, a real recurring
# voice) -- at which point it's confirmed permanently (_counts never
# decreases). Rows already printed while a label was provisional are not
# retroactively rewritten (no mechanism to revise console/SSE output
# already sent -- same constraint section 9 documented for merge_history());
# they stay as printed, which is why the session summary also reports how
# many labels never made it past provisional by the time the session ended.
PROVISIONAL_CONFIRM_HITS = 2


class SpeakerLabeler:
    def __init__(self, threads: int = 2, threshold: float = SIM_THRESHOLD,
                 remap_threshold: float | None = REMAP_THRESHOLD,
                 merge_enabled: bool = False, merge_threshold: float = MERGE_THRESHOLD,
                 hysteresis_enabled: bool = False, hysteresis_min_hits: int = HYSTERESIS_MIN_HITS):
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

        merge_enabled/merge_threshold and hysteresis_enabled/
        hysteresis_min_hits are the two section-9 (iteration 6) mitigations
        for the speaker-count overestimation left after section 8 -- see
        the module-level comment above MERGE_THRESHOLD for what each does.
        Both default to off: merging had a catastrophic failure mode on
        the AMI sweep, and hysteresis -- though clean on that sweep --
        turned out able to permanently swallow a real speaker who only
        speaks once in a short conversation (testdata/two_speakers.wav
        regression). Available as an opt-in for callers mainly running
        multi-speaker meetings who accept that trade-off.
        """
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=EMBED_MODEL, num_threads=threads
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._threshold = threshold
        self._remap_threshold = threshold if remap_threshold is None else remap_threshold
        self._centroids: list[np.ndarray] = []  # running mean per speaker
        self._counts: list[int] = []

        # --- iteration 6 (docs/DIARIZATION_PLAN.md section 9) ---
        self._merge_enabled = merge_enabled
        self._merge_threshold = merge_threshold
        # index -> canonical (surviving) index, for centroids merged away by
        # merge_centroids(). Never removed from _centroids/_counts (that
        # would shift every later index and break already-emitted labels);
        # instead skipped whenever match_embedding() searches for the
        # nearest centroid, so its slot just becomes permanently unused.
        self._alias: dict[int, int] = {}
        self._merge_history: dict[str, str] = {}  # old S{n} -> surviving S{n}, cumulative

        self._hysteresis_enabled = hysteresis_enabled
        self._hysteresis_min_hits = hysteresis_min_hits
        # index-aligned with _centroids/_counts; always maintained (even
        # with hysteresis off, where every centroid is confirmed on
        # creation) so merge_centroids() can fold this flag unconditionally.
        self._confirmed: list[bool] = []

        # Diagnostics for docs/DIARIZATION_PLAN.md section 10.6's open
        # question: does the production full pipeline open more global
        # centroids via the fast path (label(), called once per VAD
        # segment) or via the refine-path remap (match_embedding() with an
        # explicit threshold=, called once per local diarization cluster
        # per closed refine group)? match_embedding()'s `source` kwarg (a
        # free-form string the caller supplies -- realtime_transcribe.py
        # and eval_diar.py pass "fast"/"remap") is appended here every time
        # a NEW centroid is opened (not on an existing-centroid match), so
        # centroid_open_counts() can report the breakdown. Purely
        # observational: never read back to influence matching.
        self._open_log: list[str] = []

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
                         threshold: float | None = None, source: str = "") -> str:
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

        source is a free-form diagnostic tag ("fast"/"remap") recorded in
        centroid_open_counts() whenever this call opens a brand-new
        centroid -- see __init__'s _open_log comment. Purely observational.
        """
        thr = self._threshold if threshold is None else threshold
        best, best_sim = -1, -1.0
        for i, c in enumerate(self._centroids):
            if i in self._alias:  # merged away (iteration 6); dead slot
                continue
            sim = float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9))
            if sim > best_sim:
                best, best_sim = i, sim

        if best >= 0 and best_sim >= thr:
            if update:
                n = self._counts[best]
                self._centroids[best] = (self._centroids[best] * n + emb) / (n + 1)
                self._counts[best] = n + 1
                if self._hysteresis_enabled and not self._confirmed[best]:
                    if self._counts[best] >= self._hysteresis_min_hits:
                        self._confirmed[best] = True
            return self._label_for(best)

        if not update:
            return ""
        self._centroids.append(emb)
        self._counts.append(1)
        self._open_log.append(source)
        new_idx = len(self._centroids) - 1
        if self._hysteresis_enabled:
            has_confirmed = any(
                self._confirmed[i] for i in range(new_idx) if i not in self._alias
            )
            # No confirmed speaker yet to fall back to (typically the very
            # first speaker of the session) -- confirm immediately instead
            # of displaying a label that doesn't exist.
            self._confirmed.append(not has_confirmed)
        else:
            self._confirmed.append(True)
        return self._label_for(new_idx)

    def label(self, samples: np.ndarray, sample_rate: int, source: str = "fast") -> str:
        emb = self.embed(samples, sample_rate)
        return self.match_embedding(emb, update=True, source=source)

    def centroid_open_counts(self) -> dict[str, int]:
        """{source: count} of how many currently-open centroids were first
        opened by each caller-supplied `source` tag (see match_embedding()),
        for diagnosing docs/DIARIZATION_PLAN.md section 10.6."""
        counts: dict[str, int] = {}
        for src in self._open_log:
            counts[src] = counts.get(src, 0) + 1
        return counts

    def centroid_summary(self) -> list[tuple[str, str, int]]:
        """[(label, opened_by_source, final_match_count)] for every centroid
        ever opened (including merged-away ones), in opening order.

        final_match_count is self._counts[i] -- how many embeddings (its own
        opening one plus every later match_embedding() hit) ever folded into
        that centroid's running mean. A count of 1 means the centroid was
        opened and never matched again: exactly the "one-off embedding
        outlier" case docs/DIARIZATION_PLAN.md section 10.6 asks about --
        such a centroid still prints its own S{n} line the moment it opens
        (production's console has no concept of "too rare to show"), while
        a DER hypothesis built from per-group majority votes or remap
        results (eval_diar.py's generate_diarize_hypothesis) will often
        never surface it at all if it's never the mode of any group.
        """
        return [
            (self._label_for(i), self._open_log[i] if i < len(self._open_log) else "",
             self._counts[i])
            for i in range(len(self._centroids))
        ]

    # --- issue #11 (docs/DIARIZATION_PLAN.md section 10.8, option B):
    # display-only provisional labeling ---

    def _index_for_label(self, label: str) -> int | None:
        """Parse an "S{n}" label back into its centroid index, or None for
        anything that isn't one (e.g. the empty string when no speaker was
        assigned)."""
        if not label.startswith("S"):
            return None
        try:
            idx = int(label[1:]) - 1
        except ValueError:
            return None
        if idx < 0 or idx >= len(self._counts):
            return None
        return idx

    def is_provisional(self, label: str) -> bool:
        """True if `label`'s centroid has matched fewer than
        PROVISIONAL_CONFIRM_HITS times so far (still just its opening hit,
        i.e. exactly the "one-off outlier" case section 10.8 diagnosed).
        A label this returns False for stays False forever (_counts only
        grows), so a caller doesn't need to remember past decisions."""
        idx = self._index_for_label(label)
        if idx is None:
            return False
        return self._counts[idx] < PROVISIONAL_CONFIRM_HITS

    def display_label(self, label: str) -> str:
        """`label` as it should be shown to a user right now: unchanged once
        confirmed, or suffixed "?" while provisional (e.g. "S5?"). Assignment
        is untouched by this -- callers should keep using the plain `label`
        (from label()/match_embedding()) for grouping, majority votes, and
        any other internal bookkeeping, and only call display_label() at the
        point where a line is actually printed or published."""
        return f"{label}?" if self.is_provisional(label) else label

    def provisional_label_count(self) -> int:
        """How many centroids are still provisional (never matched a second
        time) -- for a session-end summary line reporting how many
        displayed labels never got past their tentative "S{n}?" form.
        Skips centroids merged away by merge_centroids() (merge_enabled
        default False, so normally a no-op filter): once merged, matches
        resolve to the surviving centroid's label, so the merged-away
        index's own label is never displayed again and its provisional
        status is moot."""
        return sum(1 for i in range(len(self._centroids))
                   if i not in self._alias and self._counts[i] < PROVISIONAL_CONFIRM_HITS)

    # --- iteration 6: new-speaker hysteresis (docs/DIARIZATION_PLAN.md section 9) ---

    def _label_for(self, idx: int) -> str:
        """Display label for global centroid `idx`.

        Confirmed centroids (the normal case, and always the case with
        hysteresis_enabled=False) get their own S{idx+1}. A provisional
        centroid displays under its nearest already-confirmed speaker
        instead, until enough hits promote it (see match_embedding()).
        """
        if not self._hysteresis_enabled or self._confirmed[idx]:
            return f"S{idx + 1}"
        fallback = self._nearest_confirmed(idx)
        return f"S{fallback + 1}" if fallback is not None else f"S{idx + 1}"

    def _nearest_confirmed(self, idx: int) -> int | None:
        emb = self._centroids[idx]
        best, best_sim = None, -1.0
        for i, c in enumerate(self._centroids):
            if i == idx or i in self._alias or not self._confirmed[i]:
                continue
            sim = float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9))
            if sim > best_sim:
                best, best_sim = i, sim
        return best

    # --- iteration 6: periodic centroid merging (docs/DIARIZATION_PLAN.md section 9) ---

    def maybe_merge_centroids(self) -> dict[str, str]:
        """No-op unless merge_enabled; see merge_centroids()."""
        if not self._merge_enabled:
            return {}
        return self.merge_centroids()

    def merge_centroids(self, threshold: float | None = None) -> dict[str, str]:
        """Fold together any two (non-aliased) global centroids whose
        cosine similarity is >= threshold (default self._merge_threshold).

        Meant to be called at a natural "clean copy" boundary -- once per
        closed refine group (realtime_transcribe.Refiner._emit_turns()) or
        eval group (eval_diar.py's generate_diarize_hypothesis) -- not per
        VAD segment, so it stays cheap (O(n^2) over the *global* speaker
        count, which is small, not over segments).

        Merging is transitive within one call (if a merges into b and the
        resulting b then qualifies to merge into c, that happens too) and
        always keeps the lower-numbered (earlier-created) centroid as the
        survivor, so S1 never disappears mid-session. Returns this call's
        {old_label: surviving_label} map; the cumulative map across all
        calls is available via merge_history().
        """
        thr = self._merge_threshold if threshold is None else threshold
        merged_map: dict[str, str] = {}
        changed = True
        while changed:
            changed = False
            roots = [i for i in range(len(self._centroids)) if i not in self._alias]
            for a in range(len(roots)):
                i = roots[a]
                for b in range(a + 1, len(roots)):
                    j = roots[b]
                    sim = float(
                        np.dot(self._centroids[i], self._centroids[j])
                        / (np.linalg.norm(self._centroids[i]) * np.linalg.norm(self._centroids[j]) + 1e-9)
                    )
                    if sim >= thr:
                        self._merge_into(i, j)
                        merged_map[f"S{j + 1}"] = f"S{i + 1}"
                        changed = True
                        break
                if changed:
                    break
        self._merge_history.update(merged_map)
        return merged_map

    def _merge_into(self, root: int, other: int) -> None:
        n_r, n_o = self._counts[root], self._counts[other]
        total = max(n_r + n_o, 1)
        merged = (self._centroids[root] * n_r + self._centroids[other] * n_o) / total
        merged /= np.linalg.norm(merged) + 1e-9
        self._centroids[root] = merged
        self._counts[root] = n_r + n_o
        self._alias[other] = root
        self._confirmed[root] = self._confirmed[root] or self._confirmed[other]
        # Any label already pointing at `other` (via an earlier merge)
        # should now point at `root` too, so merge_history() stays flat
        # (old_label -> current survivor) instead of a chain callers would
        # have to walk themselves.
        for old, survivor in self._merge_history.items():
            if survivor == f"S{other + 1}":
                self._merge_history[old] = f"S{root + 1}"

    def merge_history(self) -> dict[str, str]:
        """Cumulative {old_label: surviving_label} map from every
        merge_centroids() call so far, for a session summary."""
        return dict(self._merge_history)
