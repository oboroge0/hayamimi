"""Train the 4-class (+ "none") Japanese punctuation-restoration token
classifier for docs/eval/punct_retrain.md (improvement track C).

Base model (default): sbintuitions/modernbert-ja-30m -- **MIT**, 10 layers,
hidden=256, ~36.8M params with the 5-way head, 37MB as a dynamic-INT8 ONNX.

The first round of this work used ku-nlp/deberta-v2-tiny-japanese, which is
**CC-BY-SA-4.0**; a share-alike base model would have made the derived
punctuation weights share-alike too, a regression against the Apache-2.0
model this repo ships today (THIRD_PARTY_NOTICES.md). That round was
rejected on licensing and this one replaces it. Pass --base-model to
reproduce it. See docs/eval/punct_retrain.md for the full writeup.

Task: token classification over 5 labels -- for each subword token, does a
punctuation mark immediately follow it in the original (punctuated) text?
    0 = O (no mark)      1 = 、      2 = 。      3 = ？      4 = ！
Labels are derived at train time from plain punctuated text (see
scripts/make_punct_trainset.py) using the tokenizer's fast offset mapping:
strip target marks to get the raw (unpunctuated) string + a per-character
"mark that followed this character" list, tokenize the raw string with
return_offsets_mapping=True, and assign each token the mark (if any) that
followed its last character.

Resumable: saves a checkpoint (model + optimizer + scheduler + step) to
--ckpt-dir after every --save-every steps and at each epoch end; --resume
picks up from the latest checkpoint automatically. Designed to be run in
multiple <=10-minute chunks.

Usage (fresh run):
    .venv-train/Scripts/python scripts/train_punct_ja.py --epochs 4
Resume:
    .venv-train/Scripts/python scripts/train_punct_ja.py --epochs 4 --resume
Export the final model to HF format only (no training) once done:
    (happens automatically at the end of the last epoch)
"""
import argparse
import json
import os
import sys
import time
import unicodedata

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(ROOT, "models", "punct-ja-4class-permissive")
DEFAULT_BASE_MODEL = "sbintuitions/modernbert-ja-30m"

# Set from --model-dir in main(); module-level so find_latest_ckpt()/save_ckpt()
# can stay the simple helpers they were.
MODEL_DIR = DEFAULT_MODEL_DIR
DATA_DIR = os.path.join(MODEL_DIR, "data")
CKPT_DIR = os.path.join(MODEL_DIR, "ckpt")
FINAL_DIR = os.path.join(MODEL_DIR, "hf")  # trained HF-format model (for ONNX export)


def _set_model_dir(model_dir):
    global MODEL_DIR, DATA_DIR, CKPT_DIR, FINAL_DIR
    MODEL_DIR = os.path.abspath(model_dir)
    DATA_DIR = os.path.join(MODEL_DIR, "data")
    CKPT_DIR = os.path.join(MODEL_DIR, "ckpt")
    FINAL_DIR = os.path.join(MODEL_DIR, "hf")

LABELS = ["O", "、", "。", "？", "！"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
TARGET_MARKS = ("、", "。", "？", "！")
IGNORE_INDEX = -100


_Q_SENTINEL = ""  # private-use-area placeholder for "？" during NFKC
_E_SENTINEL = ""  # private-use-area placeholder for "！" during NFKC


def _safe_nfkc(text: str) -> str:
    """unicodedata.normalize("NFKC", ...) folds fullwidth "？"/"！" (U+FF1F/
    U+FF01) to their ASCII halfwidth forms "?"/"!" -- this is standard NFKC
    behavior (fullwidth Latin-range compatibility folding), but it silently
    breaks any code that NFKC-normalizes a string and *then* checks
    membership in a fullwidth-only mark set, which is exactly what
    scripts/quantize_punct.py::strip_marks does (TARGET_MARKS = ("、", "。",
    "？")). "、"/"。" are ideographic punctuation, not fullwidth-Latin, so
    they're unaffected -- only ？/！ silently vanish from `marks_after`
    while a stray ASCII "?"/"!" is left behind in the "stripped" text
    instead of being removed. Found while building this task's training
    labels (？/！ label counts came out exactly 0 despite the corpus having
    tens of thousands of them) -- see docs/eval/punct_retrain.md. Protect
    fullwidth ？/！ with private-use sentinels around the NFKC call so they
    survive normalization intact."""
    text = text.replace("？", _Q_SENTINEL).replace("！", _E_SENTINEL)
    text = unicodedata.normalize("NFKC", text)
    return text.replace(_Q_SENTINEL, "？").replace(_E_SENTINEL, "！")


def strip_marks(text: str):
    """Same convention as scripts/quantize_punct.py::strip_marks, extended
    to also track "！" (the existing script only tracks 、/。/？, since the
    shipped fp32 model never predicts ！) and fixed to not lose ？/！ to
    NFKC width-folding (see _safe_nfkc)."""
    norm = _safe_nfkc(text)
    stripped = []
    marks_after = []
    for ch in norm:
        if ch in TARGET_MARKS:
            if stripped and not marks_after[-1]:
                marks_after[-1] = ch
            continue
        stripped.append(ch)
        marks_after.append("")
    return "".join(stripped), marks_after


class PunctDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=320):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f]

        raw_texts = []
        marks_list = []
        for t in texts:
            raw, marks = strip_marks(t)
            if not raw:
                continue
            raw_texts.append(raw)
            marks_list.append(marks)

        print(f"  tokenizing {len(raw_texts)} examples...")
        enc = tokenizer(
            raw_texts,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        for i in range(len(raw_texts)):
            input_ids = enc["input_ids"][i]
            offsets = enc["offset_mapping"][i]
            marks = marks_list[i]
            labels = []
            for (start, end) in offsets:
                if end <= start:
                    labels.append(IGNORE_INDEX)  # special tokens (CLS/SEP/PAD)
                    continue
                mark = marks[end - 1] if end - 1 < len(marks) else ""
                labels.append(LABEL2ID[mark] if mark else LABEL2ID["O"])
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long).fill_(pad_id)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_class_weights(dataset, num_labels=5, max_weight=15.0, power=0.5):
    """Per-class loss weights of the form freq**-power, normalized to mean 1.

    `power` trades precision against recall: 0.5 (inverse sqrt frequency) is
    the aggressive setting the first round used, and it pushed the model to
    over-insert 、 (precision 0.63). Lowering it toward 0 flattens the
    weights back toward unweighted cross-entropy, which raises precision at
    the cost of recall on the rare marks. See the class-weight sweep in
    docs/eval/punct_retrain.md.
    """
    counts = np.zeros(num_labels, dtype=np.int64)
    for ex in dataset.examples:
        for l in ex["labels"]:
            if l != IGNORE_INDEX:
                counts[l] += 1
    total = counts.sum()
    freq = counts / total
    weights = np.clip(freq, 1e-8, None) ** (-power)
    weights = weights / weights.mean()
    weights = np.clip(weights, None, max_weight)
    print(f"  label counts: {dict(zip(LABELS, counts.tolist()))}")
    print(f"  class weights: {dict(zip(LABELS, np.round(weights, 3).tolist()))}")
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = np.zeros(len(LABELS))
    fp = np.zeros(len(LABELS))
    fn = np.zeros(len(LABELS))
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits
        preds = logits.argmax(-1)
        labels = batch["labels"]
        mask = labels != IGNORE_INDEX
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1))[mask.view(-1)],
            labels.view(-1)[mask.view(-1)],
        )
        total_loss += loss.item()
        n_batches += 1
        p = preds[mask].cpu().numpy()
        y = labels[mask].cpu().numpy()
        for c in range(len(LABELS)):
            tp[c] += ((p == c) & (y == c)).sum()
            fp[c] += ((p == c) & (y != c)).sum()
            fn[c] += ((p != c) & (y == c)).sum()
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / np.clip(tp + fn, 1, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
    model.train()
    return {
        "loss": total_loss / max(n_batches, 1),
        "per_class": {
            LABELS[c]: {"precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])}
            for c in range(len(LABELS))
        },
        "macro_f1_punct": float(np.mean(f1[1:])),  # exclude "O" from the macro average
    }


def find_latest_ckpt():
    if not os.path.isdir(CKPT_DIR):
        return None
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.startswith("step_") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f[len("step_"):-len(".pt")]))
    return os.path.join(CKPT_DIR, ckpts[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="HF id of the pretrained encoder to fine-tune. The default "
                         "is MIT-licensed; see the module docstring for why the "
                         "first-round CC-BY-SA base was dropped.")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="output directory: reads <dir>/data/{train,val}.jsonl, "
                         "writes <dir>/ckpt/ and <dir>/hf/")
    ap.add_argument("--class-weight-power", type=float, default=0.5,
                    help="exponent for the freq**-power class weights (0.5 = inverse "
                         "sqrt frequency, 0 = unweighted); see compute_class_weights")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--epochs-this-run", type=int, default=None,
                    help="stop cleanly after this many epochs in this invocation, at "
                         "an epoch boundary. Preferred over --max-minutes for chunked "
                         "training: resuming from an epoch-boundary checkpoint replays "
                         "the data loader exactly, whereas resuming mid-epoch skips a "
                         "differently-shuffled prefix.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=9.0,
                     help="wall-clock budget for this invocation; saves and exits "
                          "cleanly (without finalizing) if exceeded, for resumable "
                          "training under the 10-minute Bash-call limit")
    args = ap.parse_args()

    _set_model_dir(args.model_dir)
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[base] {args.base_model}")
    print(f"[out]  {MODEL_DIR}")

    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("[data] loading + tokenizing train set...")
    train_ds = PunctDataset(os.path.join(DATA_DIR, "train.jsonl"), tokenizer, args.max_length)
    print("[data] loading + tokenizing val set...")
    val_ds = PunctDataset(os.path.join(DATA_DIR, "val.jsonl"), tokenizer, args.max_length)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    class_weights = compute_class_weights(
        train_ds, power=args.class_weight_power).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    # attn_implementation="eager": ModernBERT defaults to an sdpa/flash path
    # that torch.onnx.export cannot trace. Training with the same attention
    # implementation the export uses keeps train and inference numerics
    # identical rather than merely close.
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        attn_implementation="eager",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.1,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt_path = find_latest_ckpt()
        if ckpt_path:
            print(f"[resume] loading {ckpt_path}")
            # weights_only=False: these checkpoints are written by this same
            # script into our own <model-dir>/ckpt/ (never
            # downloaded/untrusted), and carry optimizer/scheduler state
            # dicts (not just tensors), which weights_only=True rejects.
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_total_steps = ckpt.get("total_steps")
            if ckpt_total_steps is not None and ckpt_total_steps != total_steps:
                raise SystemExit(
                    f"[resume] --epochs={args.epochs} implies total_steps={total_steps}, "
                    f"but the checkpoint was trained with total_steps={ckpt_total_steps} "
                    f"(a different --epochs). The OneCycleLR schedule is baked into the "
                    f"checkpoint's optimizer/scheduler state; resuming with a different "
                    f"--epochs corrupts the LR schedule. Re-run with the same --epochs "
                    f"used to produce this checkpoint, or delete {CKPT_DIR} to start over."
                )
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            print(f"[resume] resuming at epoch={start_epoch} global_step={global_step}")
        else:
            print("[resume] no checkpoint found, starting fresh")

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    t_start = time.time()
    stopped_early = False
    epochs_done_this_run = 0
    log_path = os.path.join(MODEL_DIR, "train_log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    def save_ckpt(epoch, step):
        path = os.path.join(CKPT_DIR, f"step_{step}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "total_steps": total_steps,
        }, path)
        print(f"  [ckpt] saved {path}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== epoch {epoch + 1}/{args.epochs} ===")
        running_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            # skip batches already processed in a previous (resumed) run of
            # this same epoch, by step count within the epoch
            if epoch == start_epoch and global_step > 0 and n_seen < (global_step % len(train_loader)):
                n_seen += 1
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            n_seen += 1

            if global_step % 200 == 0:
                avg_loss = running_loss / 200
                print(f"  step={global_step} epoch={epoch + 1} loss={avg_loss:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time() - t_start:.0f}s")
                log_f.write(json.dumps({"step": global_step, "epoch": epoch + 1, "loss": avg_loss}) + "\n")
                log_f.flush()
                running_loss = 0.0

            if global_step % args.save_every == 0:
                save_ckpt(epoch, global_step)

            if (time.time() - t_start) / 60.0 > args.max_minutes:
                print(f"[time budget] {args.max_minutes} min reached, saving and exiting "
                      "(rerun with --resume to continue)")
                save_ckpt(epoch, global_step)
                stopped_early = True
                break
        if stopped_early:
            break

        metrics = evaluate(model, val_loader, device)
        print(f"[val] epoch={epoch + 1} loss={metrics['loss']:.4f} "
              f"macro_f1(punct)={metrics['macro_f1_punct']:.4f}")
        for label, m in metrics["per_class"].items():
            print(f"    {label}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        log_f.write(json.dumps({"epoch_end": epoch + 1, "val": metrics}) + "\n")
        log_f.flush()
        save_ckpt(epoch + 1, global_step)

        epochs_done_this_run += 1
        if args.epochs_this_run and epochs_done_this_run >= args.epochs_this_run \
                and epoch + 1 < args.epochs:
            print(f"[chunk] {epochs_done_this_run} epoch(s) done this run; stopping at "
                  f"the epoch boundary (rerun with --resume to continue)")
            stopped_early = True
            break

    log_f.close()

    if stopped_early:
        print("\n[incomplete] training stopped early on the time budget; rerun with --resume")
        return

    print(f"\n[done] saving final HF-format model to {FINAL_DIR}")
    os.makedirs(FINAL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print("[done] training complete")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
