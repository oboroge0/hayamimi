"""Two-stage constrained agglomerative re-clustering -- shared pure logic.

Originally written inline in eval_diar_overlap.py (Round 6,
docs/design/diarization.md section 16) to solve a very concrete problem: naive
single-stage agglomerative clustering of window-local speaker embeddings
produced a "dendrogram cliff" (threshold 0.7 -> 14 speakers, 0.9 -> 2
speakers, nothing usable in between) because embeddings built from under a
second of speech are mostly noise and bridge otherwise well-separated
speakers. The fix: only embeddings with enough *reliable* (clean, minimum-
duration) speech take part in the agglomerative merge; everything else is
assigned afterward to whichever resulting centroid it is closest to.

Round 7 (docs/design/diarization.md section 17) reuses this exact algorithm for
a different accumulation granularity -- refine-group local clusters instead
of sliding-window local speakers -- via eval_diar.py's --global-recluster and
realtime_transcribe.py's --speaker-global-recluster. This module is the
single place the two-stage algorithm lives; eval_diar_overlap.py imports it
rather than keeping its own copy.

Round 7 shipped rejected (+22.5pt simpleder regression, 100% confusion):
the root cause was pool INCOMPLETENESS, not the clustering algorithm above.
A refine group the local diarizer judges single-speaker (or declines
entirely) never built a per-cluster embedding in the first place, so it kept
its fast-path label forever and never entered the re-cluster pool -- the
session split into two disconnected label spaces (re-clustered vs.
untouched) and the same real speaker could land in either one depending on
which space their groups happened to fall into. Round 8 (docs/design/
diarization.md section 18) T1 closes that gap: every accumulation site
now also embeds a *group-level* representative for a single-speaker/declined
group (pool_audio_for_group() below picks the audio), so every refine
group -- not just the multi-cluster ones -- contributes exactly one entry to
the pool and every turn's label is eligible for the post-hoc rewrite.
"""
import numpy as np


def pool_audio_for_group(buf: np.ndarray, sr: int, turns: list,
                         g_start: int | None = None, g_end: int | None = None):
    """Pick the audio + duration to embed for a refine group's re-cluster
    pool contribution, when the group is single-speaker or the local
    diarizer declined (Round 8, docs/design/diarization.md section 18 T1).

    turns: list of (local_id, start, end) tuples ALREADY FILTERED to the
    diarizer's own >=0.3s (or equivalent MIN_TURN_S) speech turns, in the
    same sample-index units as buf (eval_diar.py) or (start, end) already in
    sample units (realtime_transcribe.py) -- either way, slice-and-concat
    semantics only, no timestamp math happens in here.

    When turns is non-empty (the diarizer found real speech, just judged it
    one speaker), the pool entry is built from exactly that speech -- the
    diarizer's own speech-only view, cleaner than the raw group buffer
    (which can include VAD gaps the diarizer didn't call speech). When turns
    is empty (diarizer declined entirely, or every candidate turn was below
    the duration floor), falls back to the group's full buffer -- something
    is better than nothing, and this group's turn still needs a pool entry
    for its label to ever be eligible for rewrite.

    g_start/g_end (sample indices into the *original* audio, eval_diar.py's
    convention) are only used for the duration fallback when turns is empty
    and buf itself isn't reliable for a sample count (kept optional so
    realtime_transcribe.py, which always passes a self-contained buf, can
    omit them and use len(buf) directly).

    Returns (audio: np.ndarray, duration_s: float). audio can be a
    zero-length array (e.g. an entirely empty group) -- callers should skip
    adding a pool entry when that happens, same as they already skip an
    empty embedding for a real local cluster.
    """
    if turns:
        pieces = [buf[s:e] for _, s, e in turns]
        duration_s = sum((e - s) / sr for _, s, e in turns)
    else:
        pieces = [buf] if len(buf) else []
        if g_start is not None and g_end is not None:
            duration_s = (g_end - g_start) / sr
        else:
            duration_s = len(buf) / sr
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return audio, duration_s


def cluster_reliable(embeddings: np.ndarray, group_ids: list, threshold: float,
                     num_clusters: int | None, method: str = "average") -> np.ndarray:
    """Constrained agglomerative clustering of a set of "reliable" embeddings.

    Cosine distance, `method` linkage. Two embeddings that share the same
    `group_ids` entry are, by construction, different people (they were
    observed active at the same time -- the same window in Round 6, the same
    refine group's distinct local diarization cluster in Round 7), so their
    pairwise distance is forced to a value above any achievable merge height
    -- the standard cannot-link trick for hierarchical clustering. Without it
    a single group where two co-present speakers' embeddings happen to be
    close can collapse two real speakers into one.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    dist = squareform(pdist(embeddings, metric="cosine"))
    ids = np.array(group_ids)
    cannot_link = ids[:, None] == ids[None, :]
    np.fill_diagonal(cannot_link, False)
    dist[cannot_link] = 10.0
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method=method)
    if num_clusters:
        labels = fcluster(z, t=min(num_clusters, n), criterion="maxclust")
    else:
        labels = fcluster(z, t=threshold, criterion="distance")
    return labels - 1


def assign_by_centroid(embeddings: np.ndarray, reliable: np.ndarray,
                       labels_reliable: np.ndarray) -> np.ndarray:
    """Two-stage stitching: cluster only the trustworthy embeddings (already
    done by the caller via cluster_reliable(), passed in as labels_reliable),
    then pull the rest in by nearest centroid.

    A cluster with only a fraction of a second of clean speech produces an
    embedding that is mostly noise. Letting those points take part in the
    agglomerative merge is what produced the original dendrogram cliff (see
    module docstring): the noisy points bridge otherwise well-separated
    speakers. Here they are excluded from the merge entirely and afterward
    assigned to whichever reliable cluster centroid they are closest to --
    they still contribute their speech time to the hypothesis, they just do
    not get a vote on the speaker inventory.
    """
    n_clusters = int(labels_reliable.max()) + 1 if len(labels_reliable) else 0
    if n_clusters == 0:
        return np.zeros(len(embeddings), dtype=int)
    rel_emb = embeddings[reliable]
    centroids = np.stack([rel_emb[labels_reliable == g].mean(axis=0)
                          for g in range(n_clusters)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    out = np.empty(len(embeddings), dtype=int)
    out[reliable] = labels_reliable
    other = ~reliable
    if other.any():
        out[other] = np.argmax(embeddings[other] @ centroids.T, axis=1)
    return out


def two_stage_cluster(embeddings: np.ndarray, group_ids: list, durations: list,
                      threshold: float, reliable_s: float,
                      num_clusters: int | None = None,
                      method: str = "average") -> np.ndarray:
    """High-level entry point combining both stages.

    embeddings: (n, d) array of L2-normalized speaker embeddings.
    group_ids: length-n list/array -- cannot-link key, two entries sharing a
        group id can never end up merged during the agglomerative stage
        (they may still be later assigned to the same centroid via nearest-
        centroid assignment if one of them isn't reliable, since that stage
        doesn't see the cannot-link constraint -- same behavior as Round 6's
        original two-stage design).
    durations: length-n list/array of seconds of (exclusive) speech behind
        each embedding -- entries with durations[i] >= reliable_s take part
        in the agglomerative merge; the rest are assigned afterward by
        nearest centroid.
    threshold: cosine-distance cut for the agglomerative stage (ignored when
        num_clusters is given).
    reliable_s: minimum duration for an embedding to be "reliable" (see
        durations above). If fewer than 2 embeddings clear this bar, the bar
        is dropped for this call (everything becomes reliable) rather than
        clustering on 0-1 points and producing a degenerate answer.

    Returns an (n,) int array of 0-based cluster labels, same order as
    `embeddings`.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    durations = np.asarray(durations, dtype=float)
    reliable = durations >= reliable_s
    if reliable.sum() < 2:  # nothing trustworthy to cluster: use everything
        reliable = np.ones(n, dtype=bool)
    reliable_group_ids = [g for g, keep in zip(group_ids, reliable) if keep]
    labels_reliable = cluster_reliable(embeddings[reliable], reliable_group_ids,
                                       threshold, num_clusters, method)
    return assign_by_centroid(embeddings, reliable, labels_reliable)
