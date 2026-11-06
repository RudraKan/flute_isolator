# sargam — song to bansuri sargam notation

Takes any song audio file and outputs playable Indian sargam notation for bansuri, in the
style of notationsworld-type pages ("SCALE OF THE FLUTE IS F#", then lines of sargam).

Pipeline: ffmpeg decode → Demucs vocal isolation (MLX on Apple Silicon, torch fallback) →
pitch tracking (RMVPE by default; pyin, FCPE, CREPE) → tuning estimate → note segmentation (rule-based Viterbi, or
the learned NoteNet model) → tonic (Sa) + scale detection (melodic evidence + key profile of the separated
accompaniment) → sargam mapping → bansuri range fit → text / JSON / MIDI output.

Accuracy on the reference song (Jeena Jeena vs its notationsworld notation, plain notes):

| engine | SER | precision | recall | F1 |
|---|---|---|---|---|
| rules + pyin | 0.400 | 0.79 | 0.74 | 0.76 |
| rules + RMVPE (default) | 0.323 | 0.82 | 0.82 | 0.82 |
| NoteNet + RMVPE (`--notes notenet`, first local model) | 0.288 | 0.85 | 0.83 | 0.84 |

One song is a thin test set: add references through the web UI to grow it (see "Learned note model" below).

## Setup

```
uv sync --group dev
uv run sargam transcribe song.mp3 --flutes "C,G bass,F#" -o out/
```

Outputs in `out/`: `song.sargam.txt` (notation), `song.notes.json` (timed notes, tonic
diagnostics), `song.melody.mid` (audition the transcription). Stems are cached under
`data/cache/<hash>/stems`.

Useful options:

```
--sa F#          force the tonic (or a frequency, e.g. --sa 370)
--diag           print the tonic candidate table and the degree histogram
--times          prefix each phrase with its start time
--detail 0|1|2   plain notes / with ornaments (default) / also mark low-confidence notes
--fold-octaves   shift out-of-range notes by octaves into .P .. S''
--no-separate    input is already an isolated vocal or solo instrument
--pitch rmvpe    pitch tracker: rmvpe (default, downloads 181 MB once), pyin, fcpe, crepe
--notes notenet  learned note model instead of the rules (needs data/models/notenet.pt)
--simplify 120   merge notes shorter than 120 ms (default per tracker: rmvpe 100, pyin off; 0 = off)
--style nw       notationsworld legend (n = lower Ni, D(k) = komal); dots / dashes use our symbols
```

Other commands: `sargam audition song.mp3` (wav with vocal stem left, transcription right),
`sargam compare out/song.notes.json truth.txt` (score against a reference notation),
`sargam stems song.mp3` (separation only).

## Web UI

```
uv run sargam serve --open        # http://127.0.0.1:8765
```

Drop a song (mp3/wav/m4a/flac/mp4) and it is analysed in the background (one at a time; progress shows each
stage). The result page shows the scale of the flute, Sa confidence, scale/raga, tempo and flute advice, then the
notation phrase by phrase in notationsworld style (or our dots/dashes styles). Play the song, the vocal stem or a
flute-like voice playing the transcription (or both with a balance slider); the current note is highlighted, click a
note to jump there, click a phrase number to loop it, slow down to 0.5x for practice. Keys: space play/pause,
←/→ previous/next phrase, L loop, Esc stop looping.

"Re-analyse" re-runs with a different Sa, pitch tracker or simplify setting (stems and pitch tracks are cached, so it
takes seconds). "Compare with a reference" takes pasted notationsworld notation or the page URL, shows SER /
precision / recall / F1 and a per-line diff (click a wrong note to hear that spot), and saves the reference with the
song. Everything lives in `data/library/<id>/` (input, outputs, pickled result, `truth.txt`); `GET /api/pairs` lists
songs that have both audio and a reference, for evaluation and training. API docs at `/api/docs`.

## Learned note model (NoteNet)

The rule-based segmenter is tuned on one song. NoteNet learns the step that matters most, turning a Sa-relative
pitch contour into the plain notes a notation writer would write. The data comes from notationsworld.

```
uv run sargam data crawl      # polite crawler (one page every ~1-3 s, resumable) -> data/corpus/raw/
uv run sargam data build      # -> data/corpus/songs.jsonl (~1800 sargam + ~1600 piano pages; tonic, lyric, notes)
uv run sargam data show jeena # check a parsed page
```

Pages have no audio, so `sargam/ml/synth_data.py` performs each melody the way a singer and a pitch tracker would.
It adds vibrato, meend, scoops, kann/murki, unwritten passing notes, consonant gaps, octave glitches, reverb tails
and separation bleed, then keeps the written notes as labels. The model (`sargam/ml/model.py`, about 3.7M params)
reads a 20-cent pitch-salience image plus loudness and predicts a note class and an onset per 10 ms frame. A
Viterbi decoder turns those into notes.

```
uv run sargam model export-eval              # labelled songs -> data/eval/*.npz (features + reference)
uv run sargam model train --out runs/x       # local (MPS) training; see pace/README.md for the GPU cluster
uv run sargam model eval runs/x/best.pt      # real-song scores next to the rule-based baseline
uv run sargam model install runs/x/best.pt   # -> data/models/notenet.pt; enables --notes notenet and the UI option
uv run sargam model bundle                   # zip for PACE (code + corpus + eval set, no audio)
```

Real pairs make it better: every song you give a reference in the web UI becomes an eval song. `sargam/ml/align.py`
force-aligns the written notes to the audio, and `--real-train 'data/eval/*.npz' --real-holdout jeena` mixes those
frame labels into training while keeping held-out songs for evaluation.

## Notation conventions

- `S R G m P D N` shuddha swaras; `r g d n` komal; `M` tivra Ma, `m` shuddha Ma
- `.P` lower octave, `S'` upper octave, `..`/`''` two octaves
- `{RSn}S` murki, `(R)S` kann swar, `S~G` meend, `-` sustain (one per extra beat)
- One phrase per line (split at pauses), blank line every four phrases

## Tests

```
uv run pytest            # unit tests + synthetic end-to-end (no dataset needed)
uv run pytest -m slow -s # real songs under data/songs/<slug>/{audio.mp3, truth.txt}
```

See `FOLLOWUPS.md` for ideas deferred to future versions.
