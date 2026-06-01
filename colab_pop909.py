"""Self-contained Colab trainer for the melody->chord accompanist (POP909).

Runs entirely in Colab (or any GPU box) with NO local install and NO access to the
private app repo. Downloads POP909, parses each song into per-beat (melody pitch-class,
chord), trains the streaming ChordGPT, and saves a checkpoint.

v2 improvements (higher accuracy):
  * Transposition augmentation: each example is randomly transposed to one of 12 keys
    on the fly -> the model becomes key-invariant and effectively sees ~12x the data.
    (Validation uses the original key, so the metric stays honest.)
  * Longer context (more beats), a bigger model, more epochs, and a one-cycle LR schedule.

USAGE IN COLAB (Runtime -> change runtime type -> T4 GPU):
    !pip -q install pretty_midi
    !git clone -q https://github.com/music-x-lab/POP909-Dataset.git
    !wget -q https://raw.githubusercontent.com/420755/AI-Accompanist-releases/main/colab_pop909.py
    !python colab_pop909.py --pop909 POP909-Dataset/POP909 --epochs 40

Self-test without data:  python colab_pop909.py --synthetic
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------- vocabulary
PAD, BOS, MEL_REST = 0, 1, 2
MEL_BASE = 3            # melody pc 0..11 -> 3..14
CH_NC = 15
CH_BASE = 16           # root(0..11) x {maj,min} -> 16..39
QUAL = ("maj", "min")
VOCAB = CH_BASE + 12 * len(QUAL)
IGNORE = -100
_NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4, "F": 5,
          "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11, "CB": 11}


def mel_tok(pc): return MEL_REST if pc is None else MEL_BASE + pc % 12
def chord_tok(root, q): return CH_NC if root is None else CH_BASE + (root % 12) * 2 + QUAL.index(q)
def is_chord(t): return t == CH_NC or t >= CH_BASE


def encode(melody, chords, shift=0):
    """Interleave [BOS,m1,c1,...] transposed by `shift` semitones; score chord targets."""
    seq = [BOS]
    for pc, (r, q) in zip(melody, chords):
        seq.append(mel_tok(None if pc is None else (pc + shift) % 12))
        seq.append(chord_tok(None if r is None else (r + shift) % 12, q))
    inp = seq[:-1]
    tgt = [t if is_chord(t) else IGNORE for t in seq[1:]]
    return inp, tgt


# ---------------------------------------------------------------- POP909 parsing
def _parse_chord_label(label: str):
    label = label.strip()
    if not label or label.upper().startswith("N"):
        return (None, "maj")
    root = label.split(":")[0].strip().upper()
    qual = label.split(":")[1] if ":" in label else "maj"
    if root not in _NAMES:
        return (None, "maj")
    q = "min" if ("min" in qual or "dim" in qual) else "maj"
    return (_NAMES[root], q)


def _read_chord_spans(path: Path):
    spans = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            spans.append((start, end, *_parse_chord_label(parts[2])))
    return spans


def parse_song(folder: Path):
    import pretty_midi
    mids = list(folder.glob("*.mid"))
    chord_file = folder / "chord_midi.txt"
    if not mids or not chord_file.exists():
        return None
    try:
        pm = pretty_midi.PrettyMIDI(str(mids[0]))
    except Exception:
        return None
    melody = next((i for i in pm.instruments if i.name.strip().upper() == "MELODY"), None)
    if melody is None or not melody.notes:
        return None
    beats = pm.get_beats()
    if len(beats) < 8:
        return None
    spans = _read_chord_spans(chord_file)
    melody_pcs, chords = [], []
    for bt in beats:
        active = [n for n in melody.notes if n.start <= bt + 1e-3 < n.end]
        melody_pcs.append(max(active, key=lambda n: n.pitch).pitch % 12 if active else None)
        chords.append(next(((r, q) for (s, e, r, q) in spans if s <= bt < e), (None, "maj")))
    return melody_pcs, chords


def raw_chunks(root, seq_beats, hop):
    """All per-song beat sequences -> list of raw (melody, chords) chunks."""
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir())
    chunks = []
    for i, folder in enumerate(folders):
        song = parse_song(folder)
        if song:
            mel, ch = song
            for s in range(0, max(1, len(mel) - seq_beats + 1), hop):
                m, c = mel[s:s + seq_beats], ch[s:s + seq_beats]
                if len(m) >= 8 and any(pc is not None for pc in m):
                    chunks.append((m, c))
        if (i + 1) % 150 == 0:
            print(f"  parsed {i+1}/{len(folders)} songs, {len(chunks)} chunks")
    print(f"POP909: {len(chunks)} chunks from {len(folders)} songs")
    return chunks


class ChunkData(Dataset):
    """Raw (melody, chords) chunks; transposed to a random key when augment=True."""
    def __init__(self, chunks, augment=True):
        self.chunks = chunks
        self.augment = augment

    def __len__(self): return len(self.chunks)

    def __getitem__(self, i):
        mel, ch = self.chunks[i]
        shift = random.randint(0, 11) if self.augment else 0
        inp, tgt = encode(mel, ch, shift)
        return torch.tensor(inp), torch.tensor(tgt)


class Synthetic(Dataset):
    def __init__(self, n=1500, seq_beats=24):
        scale = [0, 2, 4, 5, 7, 9, 11]
        rule = {0: (0, "maj"), 2: (7, "maj"), 4: (0, "maj"), 5: (5, "maj"),
                7: (0, "maj"), 9: (5, "min"), 11: (7, "maj")}
        g = random.Random(0)
        self.items = []
        for _ in range(n):
            m = [g.choice(scale) for _ in range(seq_beats)]
            self.items.append(encode(m, [rule[pc] for pc in m]))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        inp, tgt = self.items[i]
        return torch.tensor(inp), torch.tensor(tgt)


def collate(batch):
    L = max(x.size(0) for x, _ in batch)
    xs = [torch.cat([x, torch.full((L - x.size(0),), PAD)]) for x, _ in batch]
    ts = [torch.cat([t, torch.full((L - t.size(0),), IGNORE)]) for _, t in batch]
    return torch.stack(xs), torch.stack(ts)


# ---------------------------------------------------------------- model
class ChordGPT(nn.Module):
    def __init__(self, vocab=VOCAB, block=96, n_layer=6, n_head=8, n_embd=320, drop=0.1):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.drop = nn.Dropout(drop)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            n_embd, n_head, 4 * n_embd, drop, batch_first=True, norm_first=True)
            for _ in range(n_layer))
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        mask = torch.triu(torch.ones(T, T, device=idx.device), 1).bool()
        for lyr in self.layers:
            x = lyr(x, src_mask=mask)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=IGNORE)
        return logits, loss


@torch.no_grad()
def chord_acc(model, dl, dev):
    model.eval(); ok = tot = 0
    for x, t in dl:
        x, t = x.to(dev), t.to(dev)
        pred = model(x)[0].argmax(-1)
        m = t != IGNORE
        ok += (pred[m] == t[m]).sum().item(); tot += int(m.sum())
    return ok / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seq-beats", type=int, default=24)
    ap.add_argument("--hop", type=int, default=12)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=320)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", default="training_chordgpt.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.synthetic or not a.pop909:
        ds = Synthetic(seq_beats=a.seq_beats)
        nval = max(1, len(ds) // 10)
        tr, va = torch.utils.data.random_split(ds, [len(ds) - nval, nval])
    else:
        chunks = raw_chunks(a.pop909, a.seq_beats, a.hop)
        random.Random(0).shuffle(chunks)
        nval = max(1, len(chunks) // 10)
        va = ChunkData(chunks[:nval], augment=False)             # honest val: original key
        tr = ChunkData(chunks[nval:], augment=not a.no_augment)  # key-augmented train

    tdl = DataLoader(tr, a.batch, shuffle=True, collate_fn=collate, drop_last=True)
    vdl = DataLoader(va, a.batch, collate_fn=collate)

    model = ChordGPT(block=2 * a.seq_beats + 2, n_layer=a.n_layer, n_embd=a.n_embd).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(tdl), pct_start=0.1)
    print(f"device={dev} params={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(tr)} val={len(va)} augment={not a.no_augment}")

    best = 0.0
    for ep in range(a.epochs):
        model.train(); run = 0.0
        for x, t in tdl:
            x, t = x.to(dev), t.to(dev)
            _, loss = model(x, t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        acc = chord_acc(model, vdl, dev)
        best = max(best, acc)
        print(f"epoch {ep+1}/{a.epochs} loss={run/len(tdl):.3f} val_chord_acc={acc:.3f}")
    torch.save({"model": model.state_dict(), "vocab": VOCAB, "seq_beats": a.seq_beats,
                "n_layer": a.n_layer, "n_embd": a.n_embd}, a.out)
    print(f"best_val_chord_acc={best:.3f}  saved -> {a.out}")


if __name__ == "__main__":
    main()
