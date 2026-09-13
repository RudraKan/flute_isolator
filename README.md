# sargam — song to bansuri sargam notation

Takes any song audio file and outputs playable Indian sargam notation for bansuri, in the
style of notationsworld-type pages ("SCALE OF THE FLUTE IS F#", then lines of sargam).

Pipeline: ffmpeg decode → Demucs vocal isolation (MLX on Apple Silicon, torch fallback) →
pitch tracking (pyin) → tuning estimate → Viterbi note segmentation → tonic (Sa) + scale
detection (melodic evidence + key profile of the separated accompaniment) → sargam mapping →
bansuri range fit → text / JSON / MIDI output.

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
--pitch crepe    use torchcrepe instead of pyin (needs `uv add torchcrepe`)
```

Other commands: `sargam audition song.mp3` (wav with vocal stem left, transcription right),
`sargam compare out/song.notes.json truth.txt` (score against a reference notation),
`sargam stems song.mp3` (separation only).

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
