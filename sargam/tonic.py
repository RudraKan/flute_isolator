"""Tonic (Sa) and scale-type detection from note events, frame histogram and mix/bass chroma."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .model import F0Track, KeyChangeWarning, NoteEvent, Phrase, ScaleEstimate, TonicEstimate, hz_to_midi, parse_pitch_name, pitch_name

# what makes Sa "Sa": equal weight on both variants of each degree -> scale-agnostic
K_TONIC = np.array([1.0, 0.05, 0.30, 0.30, 0.30, 0.40, 0.05, 0.70, 0.20, 0.20, 0.20, 0.20])

SCALE_TEMPLATES: dict[str, set[int]] = {
    "Bilawal": {0, 2, 4, 5, 7, 9, 11},
    "Khamaj": {0, 2, 4, 5, 7, 9, 10},
    "Kafi": {0, 2, 3, 5, 7, 9, 10},
    "Bhairavi": {0, 1, 3, 5, 7, 8, 10},
    "Kalyan": {0, 2, 4, 6, 7, 9, 11},
    "Asavari": {0, 2, 3, 5, 7, 8, 10},
    "Bhairav": {0, 1, 4, 5, 7, 8, 11},
    "Kirwani": {0, 2, 3, 5, 7, 8, 11},
    "Charukeshi": {0, 2, 4, 5, 7, 8, 10},
    "Todi": {0, 1, 3, 6, 7, 8, 11},
    "Purvi": {0, 1, 4, 6, 7, 8, 11},
    "Marwa": {0, 1, 4, 6, 7, 9, 11},
}
PENTATONIC_TEMPLATES: dict[str, set[int]] = {
    "Bhupali": {0, 2, 4, 7, 9},
    "Durga": {0, 2, 5, 7, 9},
    "Malkauns": {0, 3, 5, 8, 10},
    "Shivranjani": {0, 2, 3, 7, 9},
    "Bhimpalasi (penta)": {0, 3, 5, 7, 10},
    "Megh": {0, 2, 5, 7, 10},
}
DEGREE_PAIRS = {"R": (1, 2), "G": (3, 4), "M": (5, 6), "D": (8, 9), "N": (10, 11)}


@dataclass
class Profiles:
    dur: np.ndarray
    frame: np.ndarray
    fine120: np.ndarray
    onset: np.ndarray
    end: np.ndarray
    mix: np.ndarray | None
    bass: np.ndarray | None


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def _rot(p: np.ndarray, t: int) -> np.ndarray:
    return np.roll(p, -t)


# ---------------------------------------------------------------- profiles


def frame_histograms(track: F0Track, tuning_cents: float, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    conf = np.asarray(track.conf, dtype=np.float64)
    ok = np.isfinite(f0) & (conf >= cfg.voicing_thr)
    fine = np.zeros(120)
    if ok.sum() == 0:
        return np.zeros(12), fine
    cents = 1200.0 * np.log2(f0[ok] / 440.0) - tuning_cents
    stable = np.ones(len(cents))
    stable[np.abs(np.diff(cents, prepend=cents[0])) > 30] = 0.2
    w = conf[ok] * stable
    bins = np.mod(np.round((cents / 100.0 + 69.0) * 10.0), 120).astype(int)
    np.add.at(fine, bins, w)
    coarse = np.zeros(12)
    for b in range(120):
        coarse[int(round(b / 10.0)) % 12] += fine[b]
    return coarse, fine


def chroma_profile(y: np.ndarray, sr: int, tuning_cents: float, active_mask: np.ndarray | None = None, fmin_note: str = "C2") -> tuple[np.ndarray, np.ndarray]:
    """(mean chroma weighted by RMS, per-frame chroma (12, T))."""
    import librosa

    hop = 2048
    chroma = librosa.feature.chroma_cqt(
        y=y.astype(np.float32), sr=sr, hop_length=hop, tuning=tuning_cents / 100.0, fmin=librosa.note_to_hz(fmin_note), n_octaves=5
    )
    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=hop * 2, hop_length=hop)[0]
    T = min(chroma.shape[1], len(rms))
    chroma, rms = chroma[:, :T], rms[:T]
    w = rms.copy()
    if active_mask is not None:
        w = w * active_mask[:T]
    prof = (chroma * w[None, :]).sum(axis=1)
    return _norm(prof), chroma


def bass_activity(y: np.ndarray, sr: int, hop: int = 2048) -> tuple[np.ndarray, float]:
    import librosa

    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=hop * 2, hop_length=hop)[0]
    thr = max(np.percentile(rms, 20), 1e-3)
    mask = (rms > thr).astype(np.float64)
    active_frac = float((rms > 1e-3).mean())
    return mask, active_frac


def build_profiles(
    notes: list[NoteEvent], phrases: list[Phrase], track: F0Track, tuning_cents: float,
    mix: np.ndarray | None, bass: np.ndarray | None, sr: int, cfg: Config,
) -> Profiles:
    dur, onset, end = np.zeros(12), np.zeros(12), np.zeros(12)
    for n in notes:
        pc = n.midi % 12
        dur[pc] += n.duration
        onset[pc] += 1
    for p in phrases:
        if p.notes:
            last = p.notes[-1]
            end[last.midi % 12] += last.duration
    for n in notes[-3:]:
        end[n.midi % 12] += 3.0 * n.duration
    frame, fine = frame_histograms(track, tuning_cents, cfg)
    mixp = chroma_profile(mix, sr, tuning_cents)[0] if mix is not None and len(mix) > sr else None
    bassp = None
    if bass is not None and len(bass) > sr:
        mask, active = bass_activity(bass, sr)
        if active > 0.2:
            bassp = chroma_profile(bass, sr, tuning_cents, mask, fmin_note="C1")[0]
    return Profiles(dur=_norm(dur), frame=_norm(frame), fine120=fine, onset=_norm(onset), end=_norm(end), mix=mixp, bass=bassp)


# ---------------------------------------------------------------- scoring


def score_tonics(p: Profiles, cfg: Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    comps: dict[str, np.ndarray] = {k: np.zeros(12) for k in ("dur", "end", "bass", "mix", "frame", "onset")}
    for t in range(12):
        d, e, o, f = _rot(p.dur, t), _rot(p.end, t), _rot(p.onset, t), _rot(p.frame, t)
        comps["dur"][t] = d[0] + 0.6 * d[7]
        comps["end"][t] = e[0] + 0.5 * e[7] + 0.15 * e[5]
        comps["onset"][t] = o[0] + 0.4 * o[7]
        comps["frame"][t] = float(np.dot(f, K_TONIC)) / K_TONIC.sum()
        if p.bass is not None:
            b = _rot(p.bass, t)
            comps["bass"][t] = b[0] + 0.5 * b[7] + 0.3 * b[5]
        if p.mix is not None:
            comps["mix"][t] = float(np.dot(_rot(p.mix, t), K_TONIC)) / K_TONIC.sum()
    weights = {"dur": cfg.w_dur, "end": cfg.w_end, "bass": cfg.w_bass, "mix": cfg.w_mix, "frame": cfg.w_frame, "onset": cfg.w_onset}
    if p.bass is None:
        weights["bass"] = 0.0
    if p.mix is None:
        weights["mix"] = 0.0
    elif p.mix.max() > cfg.drone_peak_thr:  # drone / shruti box: trust harmony more
        weights["mix"], weights["dur"] = 0.25, max(0.0, cfg.w_dur - 0.10)
    total_w = sum(weights.values())
    S = np.zeros(12)
    for k, w in weights.items():
        comps[k] = _norm(comps[k])
        S += (w / total_w) * comps[k]
    return S, comps


def _label(margin: float, cfg: Config) -> str:
    if margin > cfg.margin_high:
        return "high"
    if margin > cfg.margin_medium:
        return "medium"
    return "low"


def parse_sa_override(value: str) -> int:
    v = value.strip()
    try:
        hz = float(v)
        return int(round(hz_to_midi(hz))) % 12
    except ValueError:
        return parse_pitch_name(v)


def estimate_tonic(p: Profiles, cfg: Config, override: str | None = None) -> TonicEstimate:
    S, comps = score_tonics(p, cfg)
    order = np.argsort(-S)
    best, second = int(order[0]), int(order[1])
    margin = float(S[best] - S[second])
    probs = np.exp((S - S.max()) / cfg.softmax_temp)
    probs /= probs.sum()
    rel = (second - best) % 12
    ambiguity = "fifth" if rel in (5, 7) else "relative" if rel in (3, 9) else None
    pc, overridden = best, False
    if override is not None:
        pc, overridden = parse_sa_override(override), True
    return TonicEstimate(
        pc=pc, name=pitch_name(pc), sa_hz=0.0, sa_midi=0, confidence=float(probs[best]), margin=margin,
        label=_label(margin, cfg), ambiguity=ambiguity, scores=S, components=comps, overridden=overridden,
    )


# ---------------------------------------------------------------- scale


def estimate_scale(p: Profiles, tonic_pc: int, cfg: Config) -> ScaleEstimate:
    D = 0.7 * _rot(p.dur, tonic_pc) + 0.3 * _rot(p.frame, tonic_pc)
    D = _norm(D)
    present = {0}
    variants: dict[str, int] = {}
    mixed: dict[str, bool] = {}
    for name, (lo, hi) in DEGREE_PAIRS.items():
        tot = D[lo] + D[hi]
        if tot < cfg.degree_absent_thr:
            continue
        main = lo if D[lo] >= D[hi] else hi
        variants[name] = main
        mixed[name] = bool((tot - D[main]) / tot > cfg.mixed_thr)
        present.add(main)
        if mixed[name]:
            present.add(lo if main == hi else hi)
    if D[7] >= cfg.degree_absent_thr:
        present.add(7)
    core = {0} | {variants[k] for k in variants} | ({7} if 7 in present else set())
    templates = dict(SCALE_TEMPLATES)
    if len(core) == 5:
        templates = {**PENTATONIC_TEMPLATES, **templates}
    best_name, best_d = "", 99
    for name, tpl in templates.items():
        d = len(core ^ tpl)
        # prefer templates of matching size on ties
        if d < best_d or (d == best_d and len(tpl) == len(core)):
            best_name, best_d = name, d
    label = best_name
    if best_d > 0:
        tpl = templates[best_name]
        extra = sorted(core - tpl)
        missing = sorted(tpl - core)
        from .swara import DEGREE_SYMBOLS

        parts = []
        if extra:
            parts.append("+" + " ".join(DEGREE_SYMBOLS[x] for x in extra))
        if missing:
            parts.append("-" + " ".join(DEGREE_SYMBOLS[x] for x in missing))
        label = f"closest {best_name} ({', '.join(parts)})"
    return ScaleEstimate(name=label, distance=best_d, degrees_present=sorted(present), variants=variants, mixed=mixed, degree_mass=D)


# ---------------------------------------------------------------- key changes


def detect_key_changes(notes: list[NoteEvent], phrases: list[Phrase], global_pc: int, cfg: Config) -> list[KeyChangeWarning]:
    if not notes:
        return []
    t_end = max(n.end for n in notes)
    warnings: list[KeyChangeWarning] = []
    run: list[tuple[float, float, int, float]] = []
    t0 = 0.0
    while t0 < t_end:
        t1 = t0 + cfg.keychange_window_s
        win = [n for n in notes if t0 <= n.start < t1]
        if sum(n.duration for n in win) < 5.0:
            t0 += cfg.keychange_hop_s
            continue
        dur, end = np.zeros(12), np.zeros(12)
        for n in win:
            dur[n.midi % 12] += n.duration
        for p in phrases:
            if p.notes and t0 <= p.notes[-1].start < t1:
                end[p.notes[-1].midi % 12] += p.notes[-1].duration
        prof = Profiles(dur=_norm(dur), frame=_norm(dur), fine120=np.zeros(120), onset=_norm(dur), end=_norm(end), mix=None, bass=None)
        S, _ = score_tonics(prof, cfg)
        order = np.argsort(-S)
        best, margin = int(order[0]), float(S[order[0]] - S[order[1]])
        if best != global_pc and margin > cfg.keychange_margin:
            if run and run[-1][2] == best:
                run.append((t0, t1, best, margin))
            else:
                run = [(t0, t1, best, margin)]
            if len(run) >= 2 and (not warnings or warnings[-1].local_tonic_pc != best or warnings[-1].end_s < run[0][0]):
                warnings.append(KeyChangeWarning(run[0][0], min(t1, t_end), best, margin))
            elif len(run) >= 2:
                warnings[-1].end_s = min(t1, t_end)
        else:
            run = []
        t0 += cfg.keychange_hop_s
    return warnings


def describe_scores(est: TonicEstimate, top: int = 3) -> list[str]:
    """Human-readable lines for the top candidates and their relationship to the winner."""
    order = np.argsort(-est.scores)
    best = int(order[0])
    rel_names = {7: "Pa", 5: "ma", 9: "Dha", 3: "ga", 4: "Ga", 8: "dha", 2: "Re", 10: "ni", 11: "Ni", 1: "re", 6: "Ma"}
    lines = []
    for k in order[:top]:
        k = int(k)
        rel = (k - best) % 12
        tag = "" if k == best else f"  ({pitch_name(k)} is {rel_names.get(rel, '?')} of {pitch_name(best)})"
        comps = " ".join(f"{name}={est.components[name][k]:.2f}" for name in ("dur", "end", "bass", "mix", "frame", "onset"))
        lines.append(f"{pitch_name(k):>2}  score={est.scores[k]:.3f}  {comps}{tag}")
    return lines
