# sargam — song to bansuri sargam notation

Takes any song audio file and outputs playable Indian sargam notation for bansuri, in the
style of notationsworld-type pages ("SCALE OF THE FLUTE IS F#", then lines of sargam).

Pipeline: ffmpeg decode → Demucs vocal isolation (MLX on Apple Silicon) → pitch tracking →
tuning estimate → Viterbi note segmentation → tonic (Sa) + scale detection → sargam mapping →
bansuri range fit → text / JSON / MIDI output.

## Setup

```
uv sync --group dev
uv run sargam transcribe song.mp3 --flutes C,G -o out/
```

## Notation conventions

- `S R G m P D N` shuddha swaras; `r g d n` komal; `M` tivra Ma, `m` shuddha Ma
- `.P` lower octave, `S'` upper octave, `..`/`''` two octaves
- `{RSn}S` murki, `(R)S` kann swar, `S~G` meend, `-` sustain
- One phrase per line

See `FOLLOWUPS.md` for ideas deferred to future versions.
