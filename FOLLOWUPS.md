# Follow-ups: ideas for future versions

Running list of ideas deferred from v1. Append here whenever something comes up during
implementation rather than building it immediately.

## Transcription quality
- Pitch tracker upgrade: torchcrepe / RMVPE / SwiftF0 behind the existing `backend=` switch in `sargam/pitch.py`;
  narrow fmin/fmax per song from a first pass to cut octave errors.
- Energy/onset-based repeated-note detection from spectral flux instead of RMS dips.
- Shruti-aware komal/tivra decisions and an intonation report from the 120-bin histogram
  (e.g. "komal Ga sits 20 cents sharp").
- Richer raga templates (Todi, Purvi, Marwa, Bageshri, Jog with mixed G/g); vadi/samvadi-weighted tonic scoring.
- Learn tonic-scoring weights from labelled songs once there are more than ~20 with ground truth.
- Key-change segmentation: re-run tonic per segment and emit "SCALE CHANGES TO G#" instead of only warning.
- Transcribe instrumental interludes (flute/other stems), not just the vocal line.

## Rhythm and layout
- Beat-quantised durations (1/2, 1/4 beat), bar-aligned line breaks, taal detection.
- Lyrics alignment: Whisper (mlx-whisper) word timestamps on the vocal stem, lyric line above each sargam line,
  notationsworld-style layout.

## Bansuri playability
- Half-hole / fingering feasibility for tivra Ma and komal notes; breath-phrase feasibility.
- Automatic octave folding suggestions per phrase when the melody exceeds .P .. S''.

## Inputs, interface, data
- YouTube URL input via yt-dlp.
- Web UI (FastAPI/Streamlit) wrapping the CLI, rendering pages like notationsworld.
- Fine-tuning on a sargam notation dataset (dataset TBD). NoteEvents + timings are exported in JSON so they
  can be paired with any dataset later.

## Noticed during v1 implementation (2026-09-13)
- Ornament fidelity: 50 ms murki/kann notes are only partially recovered through pyin (main melody is exact);
  a better tracker (crepe/RMVPE) or onset-aware segmentation should fix this.
- Scale-aware snapping of glide targets and short notes (e.g. a stray tivra Ma in a Bilawal song is almost
  certainly a pitch-tracking artefact); use the detected scale as a prior in the Viterbi emissions.
- Separated vocals carry backing-vocal / reverb residue that shows up as tiny one-note phrases; gate phrases by
  energy relative to the song's vocal level, or drop phrases shorter than ~2 notes and 0.3 s.
- Key-change segmentation: the windowed check now uses the accompaniment key profile; next step is to re-run
  tonic per segment and emit "SCALE CHANGES TO X" in the notation.
- Tempo: beat tracking gives a bpm but sustain marks from beats are coarse; a per-phrase local tempo would be better.
