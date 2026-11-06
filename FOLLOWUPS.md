# Follow-ups: ideas for future versions

Running list of ideas deferred from v1. Append here whenever something comes up during
implementation rather than building it immediately.

## Transcription quality
- ~~Pitch tracker upgrade~~ done 2026-11-06: RMVPE is the default (SER 0.40 -> 0.32), FCPE/CREPE available. Still open:
  SwiftF0; narrow fmin/fmax per song from a first pass to cut octave errors.
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
- ~~Web UI~~ done 2026-11-06 (`sargam serve`).
- ~~Fine-tuning on a sargam notation dataset~~ v1 done 2026-11-06: notationsworld corpus + NoteNet (see below).

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
- Pitch backend comparison on the reference song (Jeena Jeena, pyin vs torchcrepe-full on MPS): pyin SER 0.40 / F1 0.75;
  crepe needs a periodicity threshold ~0.75-0.85 to match (SER 0.40, recall 0.76 vs 0.69) and takes ~110 s vs ~10 s.
  Next candidates: RMVPE (trained for vocals in polyphonic mixes) and SwiftF0; tune segmentation per backend.

## After comparing against the user's reference notation for Jeena Jeena (2026-09-14)
- Remaining error budget (pyin, plain notes): precision 0.78, recall 0.73, SER 0.41. Missed notes are quick passing
  tones in stepwise runs; extra notes are vibrato/approach fragments in the hummed intro. torchcrepe reaches recall
  0.85 but adds fragments; `--pitch crepe --simplify 120` gives the best F1 so far (0.77).
- Biggest next lever is a note-level model rather than more rules: notation-site pages (audio + plain sargam, no timing)
  are exactly the weakly-aligned dataset for the fine-tuning phase (CTC-style training of a frame->swara model on the
  separated vocal, tonic-normalised). Start with RMVPE / SwiftF0 as the frame model.
- Snap ornament/grace pitches to the scale too (only host notes are snapped today), or expand ornaments before snapping.
- Phrase splitting by lyric line (Whisper) would make side-by-side comparison with references far easier than gap-based phrases.

## After the web UI + learned note model session (2026-11-06)
- **More labelled songs is the top priority.** Every accuracy number rests on Jeena Jeena alone, and the rules and the
  NoteNet decode setting were both tuned on it. Add 5-10 songs with references through the UI ("Compare with a
  reference"), keep 2-3 as a fixed test set (`--real-holdout`), and only then compare model variants.
- Audio for the scraped notations: pairing corpus pages with the songs' audio (e.g. YouTube) would be the biggest
  data lever, but bulk-downloading copyrighted audio is a licensing/ToS decision for the user. Not built.
  Cleaner options: public-domain recordings of songs that have pages (Jana Gana Mana, Vande Mataram, Happy Birthday,
  Jingle Bells, Western classical piano pages) from Wikimedia Commons / Musopen; open singing datasets with note labels
  (GTSinger, Vocadito) to make the synthetic contours more realistic.
- RMVPE + NoteNet end to end: feed RMVPE's 360-bin salience rolled to Sa (`sargam.rmvpe.salience`) as NoteNet's
  `salience=` input instead of the rendered contour, then unfreeze RMVPE on real pairs (PACE job).
- Audio-level synthetic data: WORLD (pyworld) resynthesis of real vocal stems with synthetic contours -> realistic
  timbre with exact labels, for the end-to-end model.
- Real-pair fine-tuning test (2026-11-06): training on the aligned first half of Jeena Jeena moved SER on the second
  half 0.373 -> 0.354, where the same amount of extra synthetic training did nothing. That is from about 200 notes, and
  the repeated chorus makes it optimistic. The path works; measurable gains need many labelled songs.
- Writer-style simplification: NoteNet still writes some sung-but-unwritten embellishments (passing notes in runs).
  The generator now simulates them; real-pair fine-tuning (`--real-train`) is the proper fix; `--simplify` is the
  stopgap.
- Rhythm: add a duration head (the corpus has dot counts) so the rendered dots come from the model instead of beat
  tracking / median note length.
- Tonic detection still uses the rule-based notes; a second pass with NoteNet notes could firm it up.
- Corpus checks: songs with both a sargam and a piano page give a free check of the piano key estimate (Krumhansl);
  minor-key piano songs currently take Sa = minor tonic.
- Lyrics: corpus lines carry the lyric line above each notation line, so we can align lyrics with Whisper and show them
  in the UI like notationsworld does.
- UI: YouTube/URL input (same licensing question), upload progress bar, NoteNet progress stage, show NoteNet confidence.
- The RMVPE network was rebuilt from memory (fetching the RVC source was blocked); weights load with strict matching
  and decode test tones exactly, but a side-by-side check against the reference implementation is still worth doing.
