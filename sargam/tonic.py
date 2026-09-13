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
    acc: np.ndarray | None = None  # chroma of the harmonic accompaniment (other + bass stems)
    acc_separated: bool = False  # True when acc comes from separated stems rather than the mix
    acc_frames: np.ndarray | None = None  # (12, T) per-frame accompaniment chroma
    acc_hop_s: float = 0.0


# Temperley key profiles (flatter than Krumhansl-Kessler; better margin on pop accompaniment)
KEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
KEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])


def key_correlations(chroma12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(major[12], minor[12]) Pearson correlations of the chroma with each key's profile."""
    c = np.asarray(chroma12, dtype=np.float64)
    if c.std() == 0:
        return np.zeros(12), np.zeros(12)
    maj = np.array([np.corrcoef(np.roll(c, -t), KEY_MAJOR)[0, 1] for t in range(12)])
    mn = np.array([np.corrcoef(np.roll(c, -t), KEY_MINOR)[0, 1] for t in range(12)])
    return np.nan_to_num(maj), np.nan_to_num(mn)


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
    mix: np.ndarray | None, bass: np.ndarray | None, sr: int, cfg: Config, other: np.ndarray | None = None,
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
    acc, acc_sep, acc_frames = None, False, None
    if other is not None and len(other) > sr:
        acc_y = other + bass if (bass is not None and len(bass) == len(other)) else other
        if float(np.sqrt(np.mean(acc_y**2))) > 1e-3:
            acc, acc_frames = chroma_profile(acc_y, sr, tuning_cents)
            acc_sep = True
    if acc is None and mix is not None and len(mix) > sr:
        acc, acc_frames = chroma_profile(mix, sr, tuning_cents)
    return Profiles(
        dur=_norm(dur), frame=_norm(frame), fine120=fine, onset=_norm(onset), end=_norm(end), mix=mixp, bass=bassp,
        acc=acc, acc_separated=acc_sep, acc_frames=acc_frames, acc_hop_s=2048 / sr,
    )


# ---------------------------------------------------------------- scoring


def score_tonics(p: Profiles, cfg: Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    comps: dict[str, np.ndarray] = {k: np.zeros(12) for k in ("dur", "end", "key", "bass", "mix", "frame", "onset")}
    if p.acc is not None:
        maj, mn = key_correlations(p.acc)
        best = np.maximum(maj, mn)
        comps["key"] = np.exp((best - best.max()) / cfg.key_temp)
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
    weights = {"dur": cfg.w_dur, "end": cfg.w_end, "key": cfg.w_key, "bass": cfg.w_bass, "mix": cfg.w_mix, "frame": cfg.w_frame, "onset": cfg.w_onset}
    if p.acc is None:
        weights["key"] = 0.0
    elif not p.acc_separated:
        weights["key"] = cfg.w_key_unseparated
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
    mode = None
    if p.acc is not None:
        maj, mn = key_correlations(p.acc)
        mode = "major" if maj[pc] >= mn[pc] else "minor"
    return TonicEstimate(
        pc=pc, name=pitch_name(pc), sa_hz=0.0, sa_midi=0, confidence=float(probs[best]), margin=margin,
        label=_label(margin, cfg), ambiguity=ambiguity, scores=S, components=comps, overridden=overridden, mode=mode,
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


def detect_key_changes(notes: list[NoteEvent], phrases: list[Phrase], global_pc: int, cfg: Config, p: Profiles | None = None) -> list[KeyChangeWarning]:
    """Windowed tonic scoring (melodic components + windowed accompaniment key profile).
    Two consecutive windows disagreeing with the global tonic by a clear margin raise a warning."""
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
            run = []
            continue
        dur, end = np.zeros(12), np.zeros(12)
        for n in win:
            dur[n.midi % 12] += n.duration
        for ph in phrases:
            if ph.notes and t0 <= ph.notes[-1].start < t1:
                end[ph.notes[-1].midi % 12] += ph.notes[-1].duration
        acc = None
        if p is not None and p.acc_frames is not None and p.acc_hop_s > 0:
            i0, i1 = int(t0 / p.acc_hop_s), int(t1 / p.acc_hop_s)
            seg = p.acc_frames[:, i0:i1]
            if seg.shape[1] > 0:
                acc = _norm(seg.mean(axis=1))
        prof = Profiles(dur=_norm(dur), frame=_norm(dur), fine120=np.zeros(120), onset=_norm(dur), end=_norm(end), mix=None, bass=None,
                        acc=acc, acc_separated=bool(p.acc_separated) if p is not None else False)
        S, _ = score_tonics(prof, cfg)
        order = np.argsort(-S)
        best, margin = int(order[0]), float(S[order[0]] - S[order[1]])
        if best != global_pc and margin > cfg.keychange_margin:
            if run and run[-1][2] == best:
                run.append((t0, t1, best, margin))
            else:
                run = [(t0, t1, best, margin)]
            if len(run) == 2:
                warnings.append(KeyChangeWarning(run[0][0], min(t1, t_end), best, margin))
            elif len(run) > 2:
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
        comps = " ".join(f"{name}={est.components[name][k]:.2f}" for name in ("dur", "end", "key", "bass", "mix", "frame", "onset"))
        lines.append(f"{pitch_name(k):>2}  score={est.scores[k]:.3f}  {comps}{tag}")
    return lines


def snap_rare_degrees(notes: list[NoteEvent], scale: ScaleEstimate, tonic_pc: int, cfg: Config) -> int:
    """Conservative cleanup: notes on a degree the song barely uses *and* sings with low confidence are
    almost always pitch-tracking artefacts (slides, breathy onsets, bleed); move each such note one
    semitone to the neighbouring degree the song does use. Returns the number of notes snapped."""
    D = scale.degree_mass
    by_deg: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_deg.setdefault((n.midi - tonic_pc) % 12, []).append(n)
    artefact: set[int] = set()
    for deg, ns in by_deg.items():
        if deg == 0 or D[deg] >= cfg.snap_degree_max_mass:
            continue
        mean_conf = float(np.mean([n.confidence for n in ns]))
        if mean_conf < cfg.snap_degree_max_conf:
            artefact.add(deg)
    n_snapped = 0
    for deg in artefact:
        for n in by_deg[deg]:
            if n.duration >= cfg.snap_max_ms / 1000 or n.confidence >= cfg.snap_max_conf:
                continue
            up, down = (deg + 1) % 12, (deg - 1) % 12
            cands = [(D[d], sh) for d, sh in ((up, +1), (down, -1)) if d not in artefact and D[d] >= cfg.degree_absent_thr]
            if not cands:
                continue
            if len(cands) == 2:
                cands.sort(key=lambda c: (-(c[1] * n.cents_dev), -c[0]))
            else:
                cands.sort(key=lambda c: -c[0])
            shift = cands[0][1]
            n.midi += shift
            if n.glide_to_midi is not None:
                n.glide_to_midi += shift
            n.flags.add("snapped")
            n_snapped += 1
    return n_snapped
