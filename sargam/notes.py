"""f0 frames -> discrete NoteEvents (segmentation in absolute semitones, tonic-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .config import Config
from .model import F0Track, NoteEvent, Phrase

SIL = -10_000  # sentinel state


@dataclass
class RawSegment:
    i0: int  # first frame (inclusive)
    i1: int  # last frame (exclusive)
    semitone: int  # relative to A4 (midi = 69 + semitone)
    mean_cents: float
    std_cents: float
    slope: float  # cents per second
    resid_std: float  # std after removing a linear fit
    mean_conf: float
    n_voiced: int

    @property
    def n(self) -> int:
        return self.i1 - self.i0

    @property
    def midi(self) -> int:
        return 69 + self.semitone


# ---------------------------------------------------------------- preprocessing


def to_cents(track: F0Track, tuning_cents: float) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(f0 / 440.0) - tuning_cents
    voiced = np.isfinite(cents) & (np.asarray(track.conf) >= 0.0)
    cents = np.where(voiced, cents, np.nan)
    return cents, voiced


def bridge_dropouts(cents: np.ndarray, conf: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate short unvoiced gaps whose neighbours are close in pitch."""
    cents = cents.copy()
    conf = conf.copy()
    n = len(cents)
    isnan = ~np.isfinite(cents)
    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j < n and isnan[j]:
            j += 1
        gap = j - i
        if 0 < i and j < n and gap <= cfg.bridge_gap_frames:
            if abs(cents[i - 1] - cents[j]) <= cfg.bridge_max_cents:
                cents[i:j] = np.linspace(cents[i - 1], cents[j], gap + 2)[1:-1]
                conf[i:j] = min(conf[i - 1], conf[j], 0.5)
        i = j
    return cents, conf


def smooth_cents(cents: np.ndarray, cfg: Config) -> np.ndarray:
    """Median filter inside voiced runs only (NaNs preserved)."""
    out = cents.copy()
    k = max(1, cfg.medfilt_frames | 1)
    n = len(cents)
    i = 0
    while i < n:
        if not np.isfinite(cents[i]):
            i += 1
            continue
        j = i
        while j < n and np.isfinite(cents[j]):
            j += 1
        if j - i >= k:
            out[i:j] = median_filter(cents[i:j], size=k, mode="nearest")
        i = j
    return out


def fix_octave_errors(cents: np.ndarray, hop_s: float, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Shift frames that sit ~1200 cents away from a local running median. Returns (cents, fixed_mask)."""
    cents = cents.copy()
    fixed = np.zeros(len(cents), dtype=bool)
    win = max(3, int(cfg.octave_med_window_ms / 1000 / hop_s) | 1)
    n = len(cents)
    i = 0
    while i < n:
        if not np.isfinite(cents[i]):
            i += 1
            continue
        j = i
        while j < n and np.isfinite(cents[j]):
            j += 1
        seg = cents[i:j]
        if len(seg) >= 3:
            med = median_filter(seg, size=min(win, len(seg) | 1), mode="nearest")
            for _ in range(2):
                dev = seg - med
                up = (dev < -900) & (np.abs(dev + 1200 - 0) < 300) | (dev < -900) & (np.abs(dev + 1200) < 300)
                down = (dev > 900) & (np.abs(dev - 1200) < 300)
                seg = np.where(up, seg + 1200, seg)
                seg = np.where(down, seg - 1200, seg)
                fixed[i:j] |= up | down
            cents[i:j] = seg
        i = j
    # global range prior: short excursions far outside the song's range
    v = cents[np.isfinite(cents)]
    if len(v) > 50:
        lo, hi = np.percentile(v, 5) - 300, np.percentile(v, 95) + 300
        max_len = int(0.4 / hop_s)
        i = 0
        while i < n:
            if not np.isfinite(cents[i]) or lo <= cents[i] <= hi:
                i += 1
                continue
            j = i
            while j < n and np.isfinite(cents[j]) and not (lo <= cents[j] <= hi):
                j += 1
            if j - i <= max_len:
                seg = cents[i:j]
                far = np.abs(seg - np.clip(seg, lo, hi)).max() > 1000
                if far:
                    shift = 1200 if seg.mean() < lo else -1200
                    if np.all(lo <= seg + shift) and np.all(seg + shift <= hi):
                        cents[i:j] = seg + shift
                        fixed[i:j] = True
                    else:
                        cents[i:j] = np.nan
            i = j
    return cents, fixed


# ---------------------------------------------------------------- viterbi


def viterbi_states(cents: np.ndarray, conf: np.ndarray, cfg: Config) -> np.ndarray:
    """Most likely semitone state per frame (SIL for silence)."""
    n = len(cents)
    voiced = np.isfinite(cents) & (conf >= cfg.voicing_thr)
    if not voiced.any():
        return np.full(n, SIL, dtype=np.int64)
    v = cents[voiced]
    s_lo = int(np.floor(v.min() / 100.0)) - 1
    s_hi = int(np.ceil(v.max() / 100.0)) + 1
    states = np.arange(s_lo, s_hi + 1)
    S = len(states) + 1  # + SIL
    sil = S - 1

    # emissions (n, S)
    centres = 100.0 * states
    c = np.where(voiced, cents, 0.0)[:, None]
    e = -((c - centres[None, :]) ** 2) / (2 * cfg.sigma_cents**2)
    e = np.where(voiced[:, None], e, -cfg.unvoiced_in_note_cost)
    e_sil = np.where(voiced, -cfg.voiced_in_sil_cost, 0.0)
    E = np.concatenate([e, e_sil[:, None]], axis=1)

    # transitions T[prev, next]
    d = np.abs(states[:, None] - states[None, :])
    T = -(cfg.switch_penalty + cfg.per_semitone_penalty * d)
    T[d == 0] = 0.0
    T[d > cfg.max_jump_semitones] = -1e9
    Tf = np.full((S, S), -cfg.silence_penalty)
    Tf[:sil, :sil] = T
    Tf[sil, sil] = 0.0

    delta = E[0].copy()
    back = np.zeros((n, S), dtype=np.int32)
    for i in range(1, n):
        cand = delta[:, None] + Tf  # (prev, next)
        back[i] = np.argmax(cand, axis=0)
        delta = cand[back[i], np.arange(S)] + E[i]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    out = np.where(path == sil, SIL, states[np.minimum(path, len(states) - 1)])
    return out


def _make_segment(i0: int, i1: int, semitone: int, cents: np.ndarray, conf: np.ndarray, hop_s: float) -> RawSegment | None:
    c = cents[i0:i1]
    ok = np.isfinite(c)
    cv = c[ok]
    if len(cv) == 0:
        return None
    t = np.arange(len(c))[ok] * hop_s
    if len(cv) >= 3:
        slope, icpt = np.polyfit(t, cv, 1)
        resid = float(np.std(cv - (slope * t + icpt)))
    else:
        slope, resid = 0.0, float(np.std(cv))
    return RawSegment(
        i0=i0, i1=i1, semitone=int(semitone), mean_cents=float(cv.mean()), std_cents=float(cv.std()),
        slope=float(slope), resid_std=resid, mean_conf=float(np.mean(conf[i0:i1][ok])), n_voiced=int(ok.sum()),
    )


def segments_from_states(states: np.ndarray, cents: np.ndarray, conf: np.ndarray, hop_s: float) -> list[RawSegment]:
    segs: list[RawSegment] = []
    n = len(states)
    i = 0
    while i < n:
        if states[i] == SIL:
            i += 1
            continue
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        seg = _make_segment(i, j, states[i], cents, conf, hop_s)
        if seg is not None:
            segs.append(seg)
        i = j
    return segs


# ---------------------------------------------------------------- post-processing


def _contig(a: RawSegment, b: RawSegment) -> bool:
    return a.i1 == b.i0


def _rms_db(rms: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(rms, 1e-6))


def _has_dip(rms_db: np.ndarray, i0: int, i1: int, cfg: Config, hop_s: float) -> bool:
    """Is there an RMS dip of >= rms_dip_db inside [i0, i1) relative to 100 ms on either side?"""
    if i1 <= i0:
        return False
    ctx = int(0.1 / hop_s)
    left = rms_db[max(0, i0 - ctx) : i0]
    right = rms_db[i1 : i1 + ctx]
    if len(left) == 0 or len(right) == 0:
        return False
    ref = min(left.max(), right.max())
    return bool((ref - rms_db[i0:i1].min()) >= cfg.rms_dip_db)


def split_repeats(segs: list[RawSegment], rms: np.ndarray | None, cents: np.ndarray, conf: np.ndarray, hop_s: float, cfg: Config) -> list[RawSegment]:
    if rms is None:
        return segs
    db = _rms_db(np.asarray(rms, dtype=np.float64))
    min_len = int(cfg.repeat_split_min_ms / 1000 / hop_s)
    dmin, dmax = int(cfg.rms_dip_min_ms / 1000 / hop_s), int(cfg.rms_dip_max_ms / 1000 / hop_s)
    ctx = int(0.1 / hop_s)
    out: list[RawSegment] = []
    for s in segs:
        if s.n < min_len:
            out.append(s)
            continue
        cut_points: list[int] = []
        k = s.i0 + ctx
        while k < s.i1 - ctx:
            left_ref = db[k - ctx : k].max()
            # find a dip run starting at k
            if db[k] <= left_ref - cfg.rms_dip_db:
                j = k
                while j < s.i1 - 1 and db[j] <= left_ref - cfg.rms_dip_db:
                    j += 1
                if dmin <= j - k <= dmax and db[j : j + ctx].max() >= left_ref - 3:
                    cut_points.append((k + j) // 2)
                    k = j + ctx
                    continue
                k = j + 1
            else:
                k += 1
        if not cut_points:
            out.append(s)
            continue
        bounds = [s.i0] + cut_points + [s.i1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            sub = _make_segment(a, b, s.semitone, cents, conf, hop_s)
            if sub is not None:
                out.append(sub)
    return out


def merge_gaps(segs: list[RawSegment], rms: np.ndarray | None, hop_s: float, cfg: Config) -> list[RawSegment]:
    """Merge same-semitone segments separated by a short gap that has no RMS dip."""
    if not segs:
        return segs
    db = _rms_db(np.asarray(rms, dtype=np.float64)) if rms is not None else None
    max_gap = int(cfg.merge_gap_ms / 1000 / hop_s)
    out = [segs[0]]
    for s in segs[1:]:
        p = out[-1]
        gap = s.i0 - p.i1
        if s.semitone == p.semitone and 0 < gap <= max_gap:
            dip = _has_dip(db, p.i1, s.i0, cfg, hop_s) if db is not None else False
            if not dip:
                out[-1] = _merge(p, s)
                continue
        out.append(s)
    return out


def _merge(a: RawSegment, b: RawSegment, semitone: int | None = None) -> RawSegment:
    na, nb = max(a.n_voiced, 1), max(b.n_voiced, 1)
    return RawSegment(
        i0=a.i0, i1=b.i1, semitone=a.semitone if semitone is None else semitone,
        mean_cents=(a.mean_cents * na + b.mean_cents * nb) / (na + nb),
        std_cents=max(a.std_cents, b.std_cents), slope=0.0, resid_std=max(a.resid_std, b.resid_std),
        mean_conf=(a.mean_conf * na + b.mean_conf * nb) / (na + nb), n_voiced=a.n_voiced + b.n_voiced,
    )


def collapse_vibrato(segs: list[RawSegment], hop_s: float, cfg: Config) -> list[RawSegment]:
    max_elem = int(cfg.vibrato_elem_max_ms / 1000 / hop_s)
    out: list[RawSegment] = []
    i = 0
    while i < len(segs):
        j = i
        # extend run: alternating between two semitones 1 apart, contiguous, each short
        while (
            j + 1 < len(segs)
            and _contig(segs[j], segs[j + 1])
            and abs(segs[j + 1].semitone - segs[j].semitone) == 1
            and segs[j + 1].n < max_elem
            and segs[j].n < max_elem
            and (j == i or segs[j + 1].semitone == segs[j - 1].semitone)
        ):
            j += 1
        if j - i + 1 >= 3:
            run = segs[i : j + 1]
            dur = {}
            for s in run:
                dur[s.semitone] = dur.get(s.semitone, 0) + s.n
            major = max(dur, key=dur.get)
            m = run[0]
            for s in run[1:]:
                m = _merge(m, s, semitone=major)
            m.semitone = major
            m.slope = 0.0
            out.append(m)
            out[-1].std_cents = -1.0  # marker: vibrato (converted to a flag later)
            i = j + 1
        else:
            out.append(segs[i])
            i += 1
    return out


def detect_glides(segs: list[RawSegment], hop_s: float, cfg: Config) -> list[tuple[RawSegment, int | None, bool]]:
    """Return (segment, glide_to_semitone or None, is_ramp_only).

    A meend shows up after Viterbi as a run of steep and/or very short segments stepping 1-3 semitones
    in one direction. The run is merged into the preceding stable note (S~G) or, if the note starts
    sliding immediately, forms a glide event on its own.
    """
    short = int(cfg.vibrato_elem_max_ms / 1000 / hop_s)
    min_ramp = int(cfg.glide_min_ms / 1000 / hop_s)
    gap_tol = 2

    def steep(r: RawSegment) -> bool:
        return abs(r.slope) > cfg.glide_slope_cents_per_s and r.resid_std < 40 and r.std_cents >= 0

    def end_semitone(r: RawSegment, direction: int) -> int:
        e = int(round((r.mean_cents + r.slope * r.n * hop_s / 2) / 100.0))
        return e if e != r.semitone else r.semitone + direction

    result: list[tuple[RawSegment, int | None, bool]] = []
    i = 0
    while i < len(segs):
        s = segs[i]
        s_steep = steep(s)
        start = i if s_steep else i + 1
        j = start
        direction = 0
        while j < len(segs):
            r = segs[j]
            if j > i:
                prev = segs[j - 1]
                if r.i0 - prev.i1 > gap_tol:
                    break
                step = r.semitone - prev.semitone
                if step == 0 or abs(step) > 3 or (direction and np.sign(step) != direction):
                    break
                if not (steep(r) or (r.n < short and abs(step) == 1)):
                    break
                direction = int(np.sign(step))
            j += 1
        ramp = segs[start:j]
        n_ramp = len(ramp)
        if s_steep:
            if n_ramp == 1 and s.n < min_ramp:
                result.append((s, None, False))
                i += 1
                continue
            if direction == 0:
                direction = 1 if s.slope > 0 else -1
            merged = s
            for r in ramp[1:]:
                merged = _merge(merged, r, semitone=s.semitone)
            merged.std_cents = s.std_cents
            merged.mean_cents = s.mean_cents
            last = ramp[-1]
            target = end_semitone(last, direction)
            if j < len(segs) and segs[j].i0 - last.i1 <= gap_tol and (segs[j].semitone - last.semitone) * direction >= 0:
                target = segs[j].semitone
            result.append((merged, target, True))
            i = j
            continue
        if n_ramp >= 2 or (n_ramp == 1 and steep(ramp[0]) and ramp[0].i0 - s.i1 <= gap_tol):
            merged = s
            for r in ramp:
                merged = _merge(merged, r, semitone=s.semitone)
            merged.std_cents = s.std_cents
            merged.mean_cents = s.mean_cents
            last = ramp[-1]
            target = last.semitone if not steep(last) else end_semitone(last, direction)
            if j < len(segs) and segs[j].i0 - last.i1 <= gap_tol and (segs[j].semitone - last.semitone) * direction >= 0:
                target = segs[j].semitone
            result.append((merged, target, False))
            i = j
            continue
        result.append((s, None, False))
        i += 1
    return result


def to_events(segs: list[RawSegment], hop_s: float, glide_to: dict[int, int] | None = None) -> list[NoteEvent]:
    ev: list[NoteEvent] = []
    for k, s in enumerate(segs):
        flags: set[str] = set()
        std = s.std_cents
        if std < 0:
            flags.add("vibrato")
            std = 0.0
        e = NoteEvent(
            start=s.i0 * hop_s, end=s.i1 * hop_s, midi=s.midi, cents_dev=s.mean_cents - 100 * s.semitone,
            confidence=float(s.mean_conf * np.clip(1 - std / 100.0, 0, 1)), flags=flags,
        )
        if glide_to and k in glide_to:
            e.kind = "glide"
            e.glide_to_midi = 69 + glide_to[k]
        ev.append(e)
    return ev


def attach_ornaments(events: list[NoteEvent], hop_s: float, cfg: Config) -> list[NoteEvent]:
    """Murki clusters and single graces are attached to the following note; leftovers < min_note merge/drop."""
    short_s = cfg.grace_max_ms / 1000
    win_s = cfg.ornament_window_ms / 1000
    next_min = cfg.grace_next_min_ms / 1000
    eps = hop_s / 2
    out: list[NoteEvent] = []
    i = 0
    n = len(events)
    while i < n:
        # collect run of contiguous short notes starting at i
        j = i
        while j < n and events[j].duration < short_s and (j == i or events[j].start - events[j - 1].end <= eps):
            j += 1
        run = events[i:j]
        nxt = events[j] if j < n else None
        if run and nxt is not None and nxt.start - run[-1].end <= eps:
            span = max(e.midi for e in run + [nxt]) - min(e.midi for e in run + [nxt])
            total = run[-1].end - run[0].start
            if len(run) >= 2 and total <= win_s and span * 100 <= cfg.ornament_max_span_cents and nxt.duration >= 0.1:
                nxt.ornament_before = [e.midi for e in run]
                i = j
                continue
            if len(run) == 1 and nxt.duration >= next_min and 1 <= abs(run[0].midi - nxt.midi) <= 5:
                nxt.grace_before = run[0].midi
                i = j
                continue
        if run:
            out.extend(run)
            i = j
        else:
            out.append(events[i])
            i += 1
    return out


def absorb_slide_ins(events: list[NoteEvent], hop_s: float, cfg: Config) -> list[NoteEvent]:
    """A short glide that lands exactly on the following note is an approach slide (portamento into
    the note), not a meend worth notating: fold it into that note's onset."""
    eps = hop_s / 2
    max_s = cfg.slide_in_max_ms / 1000
    out: list[NoteEvent] = []
    for k, e in enumerate(events):
        nxt = events[k + 1] if k + 1 < len(events) else None
        if (
            e.kind == "glide" and e.duration < max_s and nxt is not None and nxt.start - e.end <= eps
            and e.glide_to_midi == nxt.midi and e.grace_before is None and not e.ornament_before
        ):
            nxt.start = e.start
            nxt.flags.add("slide_in")
            continue
        out.append(e)
    return out


def enforce_min_duration(events: list[NoteEvent], hop_s: float, cfg: Config) -> list[NoteEvent]:
    min_s = cfg.min_note_ms / 1000
    eps = hop_s / 2
    out: list[NoteEvent] = []
    for k, e in enumerate(events):
        if e.duration >= min_s - 1e-9:
            out.append(e)
            continue
        prev = out[-1] if out and abs(out[-1].end - e.start) <= eps else None
        nxt = events[k + 1] if k + 1 < len(events) and abs(events[k + 1].start - e.end) <= eps else None
        if prev is None and nxt is None:
            continue
        cand = [c for c in (prev, nxt) if c is not None]
        target = min(cand, key=lambda c: abs(c.midi - e.midi))
        if target is prev:
            prev.end = e.end
            prev.flags.add("merged_short")
        else:
            nxt.start = e.start
            nxt.flags.add("merged_short")
            if e.grace_before is not None and nxt.grace_before is None:
                nxt.grace_before = e.grace_before
    return out


def postprocess(segs: list[RawSegment], rms: np.ndarray | None, cents: np.ndarray, conf: np.ndarray, hop_s: float, cfg: Config) -> list[NoteEvent]:
    segs = split_repeats(segs, rms, cents, conf, hop_s, cfg)
    segs = merge_gaps(segs, rms, hop_s, cfg)
    segs = collapse_vibrato(segs, hop_s, cfg)
    glides = detect_glides(segs, hop_s, cfg)
    segs = [g[0] for g in glides]
    glide_to = {k: g[1] for k, g in enumerate(glides) if g[1] is not None}
    events = to_events(segs, hop_s, glide_to)
    events = attach_ornaments(events, hop_s, cfg)
    events = absorb_slide_ins(events, hop_s, cfg)
    events = enforce_min_duration(events, hop_s, cfg)
    events = [e for e in events if e.confidence >= cfg.min_note_conf or e.duration >= 0.3]
    return events


# ---------------------------------------------------------------- entry points


def prepare_cents(track: F0Track, tuning_cents: float, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cents, _ = to_cents(track, tuning_cents)
    conf = np.asarray(track.conf, dtype=np.float64).copy()
    cents = np.where(conf >= cfg.voicing_thr, cents, np.nan)
    cents, conf = bridge_dropouts(cents, conf, cfg)
    cents = smooth_cents(cents, cfg)
    cents, fixed = fix_octave_errors(cents, track.hop_s, cfg)
    return cents, conf, fixed


def segment_notes(track: F0Track, tuning_cents: float, cfg: Config) -> list[NoteEvent]:
    cents, conf, fixed = prepare_cents(track, tuning_cents, cfg)
    states = viterbi_states(cents, conf, cfg)
    segs = segments_from_states(states, cents, conf, track.hop_s)
    events = postprocess(segs, track.rms, cents, conf, track.hop_s, cfg)
    if fixed.any():
        for e in events:
            i0, i1 = int(e.start / track.hop_s), int(e.end / track.hop_s)
            if fixed[i0:i1].any():
                e.flags.add("octave_fixed")
    return events


def segment_naive(track: F0Track, tuning_cents: float, cfg: Config) -> list[NoteEvent]:
    """Baseline: round to semitone, median filter, run-length encode, drop short notes."""
    cents, conf, _ = prepare_cents(track, tuning_cents, cfg)
    semis = np.where(np.isfinite(cents), np.round(cents / 100.0), SIL).astype(np.int64)
    k = 7
    v = semis != SIL
    sm = semis.copy()
    if v.any():
        sm = median_filter(semis, size=k, mode="nearest")
        sm = np.where(v, sm, SIL)
    segs = segments_from_states(sm, cents, conf, track.hop_s)
    min_n = int(cfg.min_note_ms / 1000 / track.hop_s)
    return to_events([s for s in segs if s.n >= min_n], track.hop_s)


def split_phrases(notes: list[NoteEvent], gap_s: float) -> list[Phrase]:
    phrases: list[Phrase] = []
    cur: list[NoteEvent] = []
    for n in notes:
        if cur and n.start - cur[-1].end > gap_s:
            phrases.append(Phrase(cur[0].start, cur[-1].end, cur))
            cur = []
        cur.append(n)
    if cur:
        phrases.append(Phrase(cur[0].start, cur[-1].end, cur))
    for pid, p in enumerate(phrases):
        for n in p.notes:
            n.phrase_id = pid
    return phrases


def choose_sa_octave(notes: list[NoteEvent], tonic_pc: int) -> int:
    """Absolute midi of 'middle' Sa: maximise duration mass in [sa-5, sa+12)."""
    best, best_mass = None, -1.0
    for k in range(2, 8):
        sa = tonic_pc + 12 * k
        mass = sum(n.duration for n in notes if sa - 5 <= n.midi < sa + 12)
        if mass > best_mass:
            best, best_mass = sa, mass
    return best if best is not None else tonic_pc + 60


def assign_degrees(notes: list[NoteEvent], sa_midi: int) -> list[NoteEvent]:
    for n in notes:
        rel = n.midi - sa_midi
        n.degree = rel % 12
        n.octave = rel // 12
    return notes
