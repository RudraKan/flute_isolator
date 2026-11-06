# Training NoteNet on Georgia Tech PACE

NoteNet turns a Sa-relative pitch contour into the plain sargam a notation writer would write. It trains on
synthetic performances of the notationsworld corpus (infinitely many, generated on the fly by CPU workers) and
reports accuracy on the real songs you have references for (`data/eval/*.npz`). The bundle holds no audio.

## 1. Build the bundle (on the Mac)

```
uv run sargam data build             # refresh data/corpus/songs.jsonl from the crawled pages
uv run sargam model export-eval      # real-song features + references -> data/eval/*.npz
uv run sargam model bundle           # -> dist/sargam-pace-YYYYMMDD-HHMM.zip (a few MB)
```

## 2. Upload and set up (once per bundle)

Upload the zip (OnDemand file browser, or `scp dist/sargam-pace-*.zip <gt-username>@login-ice.pace.gatech.edu:~/scratch/`,
use `login-phoenix.pace.gatech.edu` on Phoenix), then on a login node:

```
cd ~/scratch && unzip sargam-pace-*.zip && cd sargam_pace
bash pace/setup_env.sh               # creates .venv with torch (CUDA), numpy, scipy; ~5 min
```

## 3. Train

```
sbatch pace/train.sbatch             # one GPU, 60k steps, writes runs/notenet-<date>-<job>/
squeue -u $USER                      # check it is running
tail -f notenet-<jobid>.out          # live log: loss, then a JSON line every 2000 steps
```

Settings are environment variables: `STEPS=100000 BATCH=96 MODEL='{"layers": 6, "d_model": 320, "heads": 5}' sbatch pace/train.sbatch`.
To fine-tune the model you already have locally (it rides along in the bundle when installed), add
`EXTRA="--init data/models/notenet.pt --lr 3e-4 --warmup 200"`.

On **Phoenix**, uncomment the `-A` (charge account) and `-q` (inferno, or embers for free preemptible time) lines
in the `.sbatch` files. On **ICE** they are not needed. To ask for a GPU type, change `--gres=gpu:1` to e.g.
`--gres=gpu:H100:1`. In an interactive session (OnDemand Jupyter terminal or `salloc --gres=gpu:1`) run
`bash pace/run_interactive.sh` instead of sbatch.

Model-size sweep (4 GPUs in parallel as a job array): `sbatch pace/sweep.sbatch`, then
`.venv/bin/python pace/summarize.py runs/*`.

## 4. Read the log

Every evaluation line has
- `val.note_ser`: note error rate on held-out synthetic songs (selects `best.pt`);
- `val.frame_acc`: frame accuracy on notes;
- `real.<song>.<backend>`: SER / precision / recall / F1 against your reference notation, the same metric as
  `sargam compare`. The rule-based pipeline scores SER 0.32 / F1 0.82 on Jeena Jeena with rmvpe; that is the
  number to beat. One song is too few to select checkpoints on, so `best.pt` is chosen on synthetic validation.

## 5. Bring the model back

Download `runs/<run>/best.pt` (and `metrics.jsonl`), then on the Mac:

```
uv run sargam model eval path/to/best.pt          # real-song scores next to the rule-based baseline
uv run sargam model install path/to/best.pt       # -> data/models/notenet.pt
uv run sargam transcribe song.mp3 --notes notenet # or pick "NoteNet" in the web UI
```
