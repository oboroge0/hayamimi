"""Build a training corpus for the 4-class (+ none) Japanese punctuation
retraining task (docs/eval/punct_retrain.md).

Two corpus recipes, selected with --corpus:

  `--corpus fineweb` (**the permissive recipe, used by the shipped model**)
      HuggingFaceFW/fineweb-2, config `jpn_Jpan` (**ODC-By 1.0** -- a
      permissive, non-share-alike license). Streamed, not downloaded in
      full. Japanese web text, so unlike Wikipedia it contains natural
      ？ and ！ in useful quantities and no separate question/exclamation
      source is needed. Web text is noisy, so it goes through a much
      stricter quality gate than the wiki path (see `_web_sentence_ok`).

  `--corpus wikipedia` (**the original, share-alike recipe -- retained
      only to reproduce the superseded first-round model**)
      wikimedia/wikipedia 20231101.ja (CC-BY-SA-3.0 + GFDL) for bulk
      declarative text, plus sbintuitions/JSQuAD (CC-BY-SA-4.0) questions
      for ？ coverage and a small self-authored ！ template set. Both text
      sources are share-alike, which is why this recipe was dropped: see
      "Round 1 vs round 2" in docs/eval/punct_retrain.md.

No FLEURS text is used by either recipe -- FLEURS ja is the held-out
accuracy/latency eval set for this task and must stay unseen during
training.

Output: JSONL files of *punctuated* text chunks (the training script
strips punctuation itself and derives labels at train time, using the same
convention as scripts/quantize_punct.py's strip_marks/marks_from_restored,
extended to also track "！"), under --out-dir/data/:
    <out-dir>/data/train.jsonl
    <out-dir>/data/val.jsonl

Usage:
    # permissive (shipped) recipe -- the defaults
    python scripts/make_punct_trainset.py --corpus fineweb
    # superseded share-alike recipe
    python scripts/make_punct_trainset.py --corpus wikipedia --out-dir models/punct-ja-4class
"""
import argparse
import json
import os
import random
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(ROOT, "models", "punct-ja-4class-permissive")

TARGET_MARKS = ("、", "。", "？", "！")

# Sentence-boundary split: keep the terminal mark attached to the sentence
# that precedes it.
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？])")
# Drop obvious wiki-extraction leftovers: section headers wikiextractor
# sometimes leaves as their own line, bullet/table markers, bracketed
# citation markers like [1], and stray whitespace runs.
_CITATION_RE = re.compile(r"\[\d+\]")
_MULTI_WS_RE = re.compile(r"[ \t　]+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
# Web text uses halfwidth "?"/"!" at least as often as the fullwidth forms.
# Fold them to fullwidth *before* _safe_nfkc so they become real training
# signal for the ？/！ classes instead of being dropped as non-target chars.
_HALFWIDTH_QE = str.maketrans({"?": "？", "!": "！"})
# "。。。" / "！！！" / "？！" runs: keep only the first mark. Marks are a
# per-character label here, so a run would be unlearnable noise.
_MARK_RUN_RE = re.compile(r"([、。？！])[、。？！]+")




from ja_text_norm import safe_nfkc as _safe_nfkc  # noqa: E402


def _clean_line(line: str) -> str:
    line = line.translate(_HALFWIDTH_QE)
    line = _URL_RE.sub("", line)
    line = _safe_nfkc(line).strip()
    line = _CITATION_RE.sub("", line)
    line = _MARK_RUN_RE.sub(r"\1", line)
    line = _MULTI_WS_RE.sub(" ", line)
    return line.strip()


def _looks_like_prose(sent: str) -> bool:
    if len(sent) < 8 or len(sent) > 400:
        return False
    if not any(m in sent for m in TARGET_MARKS):
        return False
    # skip lines that are mostly non-Japanese (tables/refs/formulas that
    # slipped through wikiextractor) -- count CJK-ish chars vs length.
    cjk = sum(1 for c in sent if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    if cjk / len(sent) < 0.3:
        return False
    # skip list/table remnants
    if sent.count("|") > 1 or sent.startswith(("*", "#", "!")):
        return False
    return True


def iter_wikipedia_windows(n_articles: int, seed: int = 0, min_len: int = 30, max_len: int = 320):
    """Stream wikimedia/wikipedia ja articles, split into sentences, and
    yield windows of 1-4 consecutive sentences (target length sampled per
    window) that look like clean prose."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)

    n_seen = 0
    for article in ds:
        if n_seen >= n_articles:
            break
        n_seen += 1
        text = article.get("text", "")
        if not text:
            continue
        for para in text.split("\n"):
            para = _clean_line(para)
            if len(para) < min_len:
                continue
            sents = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
            sents = [s for s in sents if _looks_like_prose(s)]
            if not sents:
                continue
            i = 0
            while i < len(sents):
                target = rng.randint(min_len, max_len)
                chunk = []
                length = 0
                while i < len(sents) and (length < target or not chunk):
                    chunk.append(sents[i])
                    length += len(sents[i])
                    i += 1
                    if len(chunk) >= 4:
                        break
                joined = "".join(chunk)
                if _looks_like_prose(joined) or (len(joined) <= max_len and len(joined) >= min_len):
                    yield joined


_TERMINAL_MARKS = ("。", "？", "！")
# Boilerplate/navigation furniture that survives the prose filters: bullet
# glyphs, table pipes, decorative separators, bracketed labels.
_JUNK_CHARS = set("|｜★☆※◆◇■□●○→⇒【】≪≫…・~～_=+*#@/<>")
_LATIN_DIGIT = set("0123456789abcdefghijklmnopqrstuvwxyz"
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# A "." that is not a decimal point. NFKC folds fullwidth "．" to "." too, so
# this catches both.
_ASCII_PERIOD_RE = re.compile(r"(?<![0-9])[.](?![0-9])")


def _web_sentence_ok(sent: str) -> bool:
    """Quality gate for a single FineWeb-2 sentence.

    FineWeb-2 is raw-ish web text: menus, breadcrumbs, price tables, SEO
    keyword soup, and half-sentences with no terminal mark. Unterminated
    fragments are actively harmful here -- "no mark follows this character"
    is the majority class already, and every fragment that *should* have
    ended in 。 but didn't teaches the model to withhold a mark. So this
    gate is deliberately strict, biased toward throwing away good text
    rather than keeping bad text.
    """
    n = len(sent)
    if n < 10 or n > 200:
        return False
    if not sent.endswith(_TERMINAL_MARKS):
        return False
    cjk = sum(1 for c in sent if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    if cjk / n < 0.4:
        return False
    if sum(1 for c in sent if c in _LATIN_DIGIT) / n > 0.3:
        return False
    if sum(1 for c in sent if c in _JUNK_CHARS) > 1:
        return False
    if sent[0] in _JUNK_CHARS:
        return False
    # Machine-translated pages punctuate Japanese with ASCII ","/"." instead
    # of 、/。. Those are not target marks, so every one of them becomes a
    # position labelled "no mark follows" sitting exactly where a mark
    # belongs -- precisely the wrong lesson. Drop the sentence.
    if "," in sent or _ASCII_PERIOD_RE.search(sent):
        return False
    # mark spam ("あ、い、う、え、") is as useless as no marks at all
    if sum(sent.count(m) for m in TARGET_MARKS) / n > 0.2:
        return False
    if sent[0] in TARGET_MARKS:
        return False
    return True


def iter_fineweb_windows(n_docs: int, seed: int = 0, min_len: int = 30, max_len: int = 320,
                         min_lang_score: float = 0.9):
    """Stream HuggingFaceFW/fineweb-2 `jpn_Jpan` (ODC-By 1.0) and yield
    windows of 1-4 consecutive quality-gated sentences.

    Web text is the whole point of this recipe: it carries natural ？ and
    ！, which the encyclopedic Wikipedia recipe did not (the first-round
    model had to bolt on a share-alike QA corpus for ？ and ~600
    self-authored sentences for ！ -- see docs/eval/punct_retrain.md).
    The price is noise, paid for by `_web_sentence_ok` above.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceFW/fineweb-2", "jpn_Jpan", split="train", streaming=True)

    n_seen = 0
    for doc in ds:
        if n_seen >= n_docs:
            break
        n_seen += 1
        score = doc.get("language_score")
        if score is not None and score < min_lang_score:
            continue
        text = doc.get("text", "")
        if not text:
            continue
        for para in text.split("\n"):
            para = _clean_line(para)
            if len(para) < min_len:
                continue
            sents = [x.strip() for x in _SENT_SPLIT_RE.split(para) if x.strip()]
            sents = [x for x in sents if _web_sentence_ok(x)]
            if not sents:
                continue
            i = 0
            while i < len(sents):
                target = rng.randint(min_len, max_len)
                chunk = []
                length = 0
                while i < len(sents) and (length < target or not chunk):
                    chunk.append(sents[i])
                    length += len(sents[i])
                    i += 1
                    if len(chunk) >= 4:
                        break
                joined = "".join(chunk)
                if min_len <= len(joined) <= max_len:
                    yield joined


def load_jsquad_questions(n: int, seed: int = 0):
    """sbintuitions/JSQuAD question column, deduplicated. These are natural
    ？-terminated Japanese questions (Wikipedia-derived SQuAD-style QA,
    CC-BY-SA-4.0) -- the counterweight source for ？ coverage."""
    from datasets import load_dataset

    seen = set()
    out = []
    for split in ("train", "validation"):
        ds = load_dataset("sbintuitions/JSQuAD", split=split)
        for ex in ds:
            q = _clean_line(ex["question"])
            if not q or q in seen:
                continue
            if len(q) < 6 or len(q) > 200:
                continue
            if not q.endswith(("？", "?")):
                q = q + "？"
            q = q.replace("?", "？")
            seen.add(q)
            out.append(q)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


_EXCLAIM_TEMPLATES = [
    # Short standalone exclamations (common interjection patterns).
    "すごい！", "やった！", "危ない！", "頑張って！", "待って！", "うそでしょ！",
    "最高だ！", "やめて！", "見て！", "早く！", "本当に驚いた！", "信じられない！",
    "ありがとう！", "おめでとう！", "気をつけて！", "静かにして！", "行くぞ！",
    "やればできる！", "諦めるな！", "よくやった！", "素晴らしい！", "大変だ！",
    "火事だ！", "助けて！", "無理だ！", "最悪だ！", "最高の気分だ！",
]
_EXCLAIM_PREFIXES = [
    "彼は驚いて言った", "彼女は叫んだ", "観客は歓声を上げた", "選手たちは喜んだ",
    "先生は大きな声で言った", "友人が興奮して言った", "子供たちは声をそろえて言った",
    "監督はベンチから叫んだ", "群衆は一斉に叫んだ", "彼は思わず口にした",
]


def synthetic_exclamation_examples(n: int, seed: int = 0):
    """A small, self-authored set of ！-terminated Japanese sentences.

    No public corpus of comparable size/quality/license was readily
    available for this task's time budget (Japanese ！-heavy text tends to
    live in social-media or dialogue corpora with unclear or ND/NC
    licensing). Wikipedia (this script's main source) essentially never
    uses ！, so without *some* exposure the "！" label would have zero
    training examples and the model could not learn it at all. These
    templates give the model a minimal, clearly-synthetic grounding for
    the class; see docs/eval/punct_retrain.md for the resulting (weak,
    honestly-reported) ！ behavior -- it is not a task acceptance
    criterion, unlike ？ recall.
    """
    rng = random.Random(seed)
    out = []
    # bare exclamations
    out.extend(_EXCLAIM_TEMPLATES)
    # "<prefix>、<exclamation>" -- exercises ！ appearing after a 、-joined
    # clause, not just as a whole-sentence match
    for _ in range(n - len(out)):
        prefix = rng.choice(_EXCLAIM_PREFIXES)
        tail = rng.choice(_EXCLAIM_TEMPLATES)
        out.append(f"{prefix}、「{tail}」")
    rng.shuffle(out)
    return out[:n]


def _collect_unique(it, cap, log_every=20000):
    """Drain `it` into a deduplicated list of at most `cap` windows."""
    out = []
    seen = set()
    for w in it:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= cap:
            break
        if len(out) % log_every == 0:
            print(f"  ... {len(out)} windows so far")
    return out


def _collect_two_bucket(it, plain_cap, q_cap, log_every=20000):
    """Drain `it` into two deduplicated buckets: windows containing ？ or ！,
    and everything else, each with its own cap.

    Japanese web prose is overwhelmingly 、/。; ？ and ！ sit around 0.07% of
    characters, an order of magnitude below what the first-round corpus got
    by importing a share-alike QA dataset. Rather than reintroduce a
    license-incompatible source (or duplicate examples, which adds no new
    signal), this keeps streaming and *retains every* ？/！-bearing window it
    sees while sampling the plentiful declarative ones only up to
    `plain_cap`. The result is more real, diverse ？/！ evidence out of the
    same ODC-By corpus.
    """
    plain, q = [], []
    seen = set()
    for w in it:
        if w in seen:
            continue
        seen.add(w)
        bucket = q if ("？" in w or "！" in w) else plain
        cap = q_cap if bucket is q else plain_cap
        if len(bucket) >= cap:
            if len(plain) >= plain_cap and len(q) >= q_cap:
                break
            continue
        bucket.append(w)
        if (len(plain) + len(q)) % log_every == 0:
            print(f"  ... {len(plain)} plain + {len(q)} ？/！ windows so far")
    return plain, q


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["fineweb", "wikipedia"], default="fineweb",
                    help="fineweb: HuggingFaceFW/fineweb-2 jpn_Jpan, ODC-By 1.0 "
                         "(permissive; the shipped recipe). wikipedia: the superseded "
                         "share-alike recipe (wikipedia ja + JSQuAD + synthetic ！).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="model directory; the corpus is written to <out-dir>/data/")
    ap.add_argument("--n-docs", type=int, default=120000,
                    help="fineweb: number of documents to stream (most are dropped "
                         "by the quality gate, so this is much larger than the "
                         "resulting example count)")
    ap.add_argument("--max-windows", type=int, default=110000,
                    help="cap on the number of unique ？/！-free windows kept")
    ap.add_argument("--max-q-windows", type=int, default=45000,
                    help="cap on the number of unique ？/！-bearing windows kept "
                         "(see _collect_two_bucket -- these are rare in web prose "
                         "and are retained preferentially)")
    ap.add_argument("--n-wiki-articles", type=int, default=20000,
                    help="wikipedia recipe only: number of ja articles to stream")
    ap.add_argument("--n-jsquad", type=int, default=40000,
                    help="wikipedia recipe only")
    ap.add_argument("--n-exclaim-synth", type=int, default=600,
                    help="wikipedia recipe only: small self-authored ！-terminated "
                         "sentence set (see synthetic_exclamation_examples). The "
                         "fineweb recipe needs none -- web text has real ！.")
    ap.add_argument("--max-wiki-windows", type=int, default=250000,
                    help="wikipedia recipe only (kept for reproducibility of the "
                         "first-round corpus; --max-windows is the fineweb knob)")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = os.path.join(os.path.abspath(args.out_dir), "data")
    os.makedirs(data_dir, exist_ok=True)

    all_examples = []
    if args.corpus == "fineweb":
        print(f"[fineweb-2 jpn_Jpan] streaming up to {args.n_docs} documents "
              f"(caps: {args.max_windows} plain / {args.max_q_windows} ？！)...")
        plain, q = _collect_two_bucket(
            iter_fineweb_windows(args.n_docs, seed=args.seed),
            args.max_windows, args.max_q_windows)
        print(f"[fineweb-2 jpn_Jpan] {len(plain)} plain + {len(q)} ？/！ windows")
        all_examples += [{"text": t, "source": "fineweb-2/jpn_Jpan"} for t in plain + q]
    else:
        print(f"[wikipedia] streaming up to {args.n_wiki_articles} articles "
              f"(cap {args.max_wiki_windows} windows)...")
        windows = _collect_unique(
            iter_wikipedia_windows(args.n_wiki_articles, seed=args.seed),
            args.max_wiki_windows)
        print(f"[wikipedia] {len(windows)} unique windows")
        all_examples += [{"text": t, "source": "wikipedia"} for t in windows]

        print(f"[jsquad] loading up to {args.n_jsquad} unique questions...")
        jsquad_questions = load_jsquad_questions(args.n_jsquad, seed=args.seed)
        print(f"[jsquad] {len(jsquad_questions)} unique questions")
        all_examples += [{"text": t, "source": "jsquad"} for t in jsquad_questions]

        exclaim_synth = synthetic_exclamation_examples(args.n_exclaim_synth, seed=args.seed)
        print(f"[exclaim-synth] {len(exclaim_synth)} self-authored ！ sentences")
        all_examples += [{"text": t, "source": "exclaim_synth"} for t in exclaim_synth]

    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    n_val = max(200, int(len(all_examples) * args.val_frac))
    val = all_examples[:n_val]
    train = all_examples[n_val:]

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)

    print(f"\nwrote {len(train)} train examples -> {train_path}")
    print(f"wrote {len(val)} val examples -> {val_path}")

    # quick class-balance sanity check on the raw text (count of chars
    # immediately followed by each mark, normalized as a rate)
    from collections import Counter
    mark_counts = Counter()
    total_chars = 0
    for ex in all_examples:
        t = ex["text"]
        total_chars += len(t)
        for m in TARGET_MARKS:
            mark_counts[m] += t.count(m)
    print("[mark frequency over all examples]")
    for m in TARGET_MARKS:
        print(f"  {m}: {mark_counts[m]} ({mark_counts[m] / total_chars * 100:.3f}% of chars)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
