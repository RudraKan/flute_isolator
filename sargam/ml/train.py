"""Train NoteNet on synthetic performances of the notation corpus; report real-song accuracy.

    python -m sargam.ml.train --corpus data/corpus/songs.jsonl --out runs/notenet-v1 --steps 20000

Runs on CUDA (PACE), Apple MPS or CPU. Writes <out>/config.json, metrics.jsonl, last.pt, best.pt.
best.pt is chosen on the held-out synthetic songs (not on the real songs, which are too few to select on).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .decode import DecodeParams, decode, symbols
from .features import N_CLASSES
from .model import build_model
from .synth_data import load_melodies, make_window


class SynthStream(IterableDataset):
    def __init__(self, melodies: list[dict], n_frames: int, seed: int):
        self.melodies, self.n_frames, self.seed = melodies, n_frames, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid, int(time.time() * 1000) % 100000])
        while True:
            w = make_window(self.melodies, rng, self.n_frames)
            yield w.rel_cents, w.energy, w.cls, w.onset


class RealPairs:
    """Real songs with a reference notation. Labels come from forced alignment with the current model and are
    refreshed as it improves; frames the alignment is unsure about are ignored by the loss."""

    def __init__(self, files: list[Path]):
        from .evalset import load

        self.songs = [s for s in map(load, files) if s["truth_rels"]]
        self.labels: list[tuple[np.ndarray, np.ndarray]] = []

    def realign(self, model, device) -> dict:
        from .align import IGNORE, align
        from .infer import predict_logits

        model.eval()
        self.labels = []
        stats = {}
        for s in self.songs:
            cl, _ = predict_logits(model, s["rel_cents"], s["energy"], device)
            lab, on = align(cl, s["truth_rels"])
            self.labels.append((lab, on))
            stats[s["name"]] = round(float((lab == IGNORE).mean()), 3)
        model.train()
        return stats

    def sample(self, rng: np.random.Generator, n_frames: int) -> tuple:
        k = int(rng.integers(len(self.songs)))
        s, (lab, on) = self.songs[k], self.labels[k]
        T = len(lab)
        a = int(rng.integers(0, max(1, T - n_frames)))
        sl = slice(a, a + n_frames)
        pad = n_frames - len(lab[sl])
        rel = np.concatenate([s["rel_cents"][sl], np.full(pad, np.nan, np.float32)])
        en = np.concatenate([s["energy"][sl], np.zeros(pad, np.float32)])
        cls = np.concatenate([lab[sl], np.zeros(pad, np.int64)])
        ons = np.concatenate([on[sl], np.zeros(pad, np.float32)])
        return rel, en, cls, ons


def fixed_set(melodies: list[dict], n: int, n_frames: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = make_window(melodies, rng, n_frames)
        out.append((w.rel_cents, w.energy, w.cls, w.onset))
    return out


def batches(items: list[tuple], bs: int):
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        yield [torch.from_numpy(np.stack([c[k] for c in chunk])) for k in range(4)]


def soften_onsets(onset: torch.Tensor) -> torch.Tensor:
    """Onset targets 1 at the written start, 0.5 one frame either side (the boundary is fuzzy by a frame)."""
    side = F.max_pool1d(onset.unsqueeze(1), 3, 1, 1).squeeze(1)
    return torch.maximum(onset, 0.5 * side)


def losses(model, rel, energy, cls, onset, onset_pos_weight: float):
    cl, ol = model(rel, energy)
    l_cls = F.cross_entropy(cl.reshape(-1, N_CLASSES).float(), cls.reshape(-1), label_smoothing=0.02)
    pw = torch.tensor(onset_pos_weight, device=ol.device)
    l_on = F.binary_cross_entropy_with_logits(ol.float(), soften_onsets(onset), pos_weight=pw)
    return l_cls, l_on, cl, ol


def _levenshtein(a, b):
    from ..metrics import levenshtein

    return levenshtein(a, b)


def label_symbols(cls: np.ndarray, onset: np.ndarray) -> list[str]:
    from ..swara import rel_to_symbol
    from .features import class_to_rel

    out: list[str] = []
    prev = 0
    for t, c in enumerate(cls):
        if c and (c != prev or onset[t] > 0):
            s = rel_to_symbol(int(class_to_rel(int(c))))
            if not out or out[-1] != s:
                out.append(s)
        prev = c
    return out


@torch.no_grad()
def evaluate_synth(model, items, device, bs, dp: DecodeParams) -> dict:
    model.eval()
    tot = {"loss_cls": 0.0, "loss_on": 0.0, "frame_acc": 0.0, "n": 0}
    edits = ref = 0
    for rel, energy, cls, onset in batches(items, bs):
        rel, energy, cls, onset = rel.to(device), energy.to(device), cls.to(device), onset.to(device)
        l_cls, l_on, cl, ol = losses(model, rel, energy, cls, onset, 4.0)
        tot["loss_cls"] += float(l_cls) * len(rel)
        tot["loss_on"] += float(l_on) * len(rel)
        pred = cl.argmax(-1)
        m = cls > 0
        tot["frame_acc"] += float(((pred == cls) & m).sum() / m.sum().clamp(min=1)) * len(rel)
        tot["n"] += len(rel)
        cln, oln, clsn, onn = cl.float().cpu().numpy(), ol.float().cpu().numpy(), cls.cpu().numpy(), onset.cpu().numpy()
        for k in range(len(rel)):
            hyp = symbols(decode(cln[k], oln[k], dp))
            tru = label_symbols(clsn[k], onn[k])
            edits += _levenshtein(hyp, tru)
            ref += len(tru)
    model.train()
    n = max(1, tot.pop("n"))
    out = {k: v / n for k, v in tot.items()}
    out["note_ser"] = edits / max(1, ref)
    return out


@torch.no_grad()
def evaluate_real(model, eval_files: list[Path], device, dp: DecodeParams) -> dict:
    from ..metrics import aligned_f1
    from .evalset import load
    from .infer import predict_logits

    model.eval()
    out = {}
    for f in eval_files:
        if not f.exists():
            continue
        e = load(f)
        cl, ol = predict_logits(model, e["rel_cents"], e["energy"], device)
        hyp = symbols(decode(cl, ol, dp))
        ser = _levenshtein(hyp, e["truth"]) / max(1, len(e["truth"]))
        p, r, f1 = aligned_f1(hyp, e["truth"])
        out[e["name"]] = {"ser": round(ser, 4), "p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3), "n_pred": len(hyp)}
    model.train()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus/songs.jsonl")
    ap.add_argument("--eval-dir", default="data/eval", help="real-song .npz files from `sargam model export-eval`")
    ap.add_argument("--out", default="runs/notenet")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--onset-weight", type=float, default=1.0, help="weight of the onset loss")
    ap.add_argument("--onset-pos-weight", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-windows", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (weights + optimiser)")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from (weights only)")
    ap.add_argument("--model", default="{}", help='JSON model hparams, e.g. \'{"layers": 6, "d_model": 320, "heads": 5}\'')
    ap.add_argument("--kinds", default="sargam,piano")
    ap.add_argument("--real-train", default=None, help="glob of real .npz pairs to mix into training (forced alignment)")
    ap.add_argument("--real-holdout", default="", help="comma-separated song names kept out of real training (eval only)")
    ap.add_argument("--real-frac", type=float, default=0.25, help="share of each batch drawn from real pairs")
    ap.add_argument("--real-start", type=int, default=1000, help="first step that uses real pairs (alignment needs a trained model)")
    ap.add_argument("--realign-every", type=int, default=2000)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = tuple(args.kinds.split(","))
    train_mel = load_melodies(args.corpus, kinds, "train")
    val_mel = load_melodies(args.corpus, kinds, "val")
    eval_files = sorted(Path(args.eval_dir).glob("*.npz")) if Path(args.eval_dir).exists() else []
    print(f"device {device}; {len(train_mel)} train songs, {len(val_mel)} val songs, {len(eval_files)} real eval files", flush=True)

    real = None
    if args.real_train:
        import glob

        hold = [h for h in args.real_holdout.split(",") if h]
        files = [Path(f) for f in sorted(glob.glob(args.real_train)) if not any(Path(f).name.startswith(h) for h in hold)]
        real = RealPairs(files) if files else None
        trained_on = {s["name"] for s in real.songs} if real else set()
        overlap = [f.stem for f in eval_files if f.stem in trained_on]
        print(f"real pairs: {len(trained_on)} songs; eval songs also trained on: {overlap or 'none'}", flush=True)
    hp = json.loads(args.model)
    model = build_model(hp).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model = build_model(ck["hparams"]).to(device)
        model.load_state_dict(ck["model"])
    n_params = sum(p.numel() for p in model.parameters())
    cfg = {**vars(args), "device": device, "hparams": model.hparams, "n_params": n_params}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"model {model.hparams} ({n_params / 1e6:.2f} M params)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    )
    step, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck.get("best", best)

    val_items = fixed_set(val_mel or train_mel, args.val_windows, args.frames, seed=1234)
    loader = DataLoader(SynthStream(train_mel, args.frames, args.seed + step), batch_size=args.batch,
                        num_workers=args.workers, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    amp = device == "cuda"
    real_rng = np.random.default_rng(args.seed + 99)
    dp = DecodeParams()
    log = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    run_cls = run_on = 0.0

    def save(path, extra=None):
        torch.save({"model": model.state_dict(), "hparams": model.hparams, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "decode": vars(dp), **(extra or {})}, path)

    model.train()
    for rel, energy, cls, onset in loader:
        if step >= args.steps:
            break
        if real is not None and step >= args.real_start:
            if not real.labels or (step - args.real_start) % args.realign_every == 0:
                print(f"step {step} realigned real pairs (ignored frame share: {real.realign(model, device)})", flush=True)
            n_real = int(round(args.real_frac * len(rel)))
            for k in range(n_real):
                r_, e_, c_, o_ = real.sample(real_rng, args.frames)
                rel[k], energy[k], cls[k], onset[k] = map(torch.from_numpy, (r_, e_, c_, o_))
        rel, energy, cls, onset = (x.to(device, non_blocking=True) for x in (rel, energy, cls, onset))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            l_cls, l_on, _, _ = losses(model, rel, energy, cls, onset, args.onset_pos_weight)
            loss = l_cls + args.onset_weight * l_on
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run_cls = 0.98 * run_cls + 0.02 * l_cls.item() if step > 1 else l_cls.item()
        run_on = 0.98 * run_on + 0.02 * l_on.item() if step > 1 else l_on.item()
        if step % 50 == 0:
            rate = step / (time.time() - t0)
            print(f"step {step} loss_cls {run_cls:.4f} loss_on {run_on:.4f} lr {sched.get_last_lr()[0]:.2e} {rate:.2f} it/s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            vm = evaluate_synth(model, val_items, device, args.batch, dp)
            rm = evaluate_real(model, eval_files, device, dp) if eval_files else {}
            score = vm["loss_cls"] + args.onset_weight * vm["loss_on"]
            rec = {"step": step, "time_s": round(time.time() - t0), "val": vm, "real": rm}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if score < best:
                best = score
                save(out / "best.pt", {"val": vm, "real": rm})
            save(out / "last.pt")
    log.close()
    print(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min; best val score {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
