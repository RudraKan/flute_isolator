"""Print the last and best evaluation of each run:  python pace/summarize.py runs/*"""
import json
import sys
from pathlib import Path

for d in map(Path, sys.argv[1:]):
    f = d / "metrics.jsonl"
    if not f.exists():
        continue
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not recs:
        continue
    best = min(recs, key=lambda r: r["val"]["loss_cls"] + r["val"]["loss_on"])
    for tag, r in (("last", recs[-1]), ("best", best)):
        real = "  ".join(f"{k} SER {v['ser']:.3f} F1 {v['f1']:.3f}" for k, v in r["real"].items())
        print(f"{d.name:24} {tag} step {r['step']:6}  val note-SER {r['val']['note_ser']:.3f} frame-acc {r['val']['frame_acc']:.3f}  {real}")
