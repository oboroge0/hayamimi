"""Two-stage constrained agglomerative re-clustering -- shared pure logic.

Originally written inline in eval_diar_overlap.py (Round 6,
docs/DIARIZATION_PLAN.md section 16) to solve a very concrete problem: naive
single-stage agglomerative clustering of window-local speaker embeddings
produced a "dendrogram cliff" (threshold 0.7 -> 14 speakers, 0.9 -> 2
speakers, nothing usable in between) because embeddings built from under a
second of speech are mostly noise and bridge otherwise well-separated
speakers. The fix: only embeddings with enough *reliable* (clean, minimum-
duration) speech take part in the agglomerative merge; everything else is
assigned afterward to whichever resulting centroid it is closest to.

Round 7 (docs/DIARIZATION_PLAN.md section 17) reuses this exact algorithm for
a different accumulation granularity -- refine-group local clusters instead
of sliding-window local speakers -- via eval_diar.py's --global-recluster and
realtime_transcribe.py's --speaker-global-recluster. This module is the
single place the two-stage algorithm lives; eval_diar_overlap.py imports it
rather than keeping its own copy.
"""
import numpy as np


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
