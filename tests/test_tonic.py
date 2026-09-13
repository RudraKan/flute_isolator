import numpy as np

from sargam.config import Config
from sargam.tonic import Profiles, estimate_scale, estimate_tonic, score_tonics

cfg = Config()


def prof(dur=None, end=None, mix=None, bass=None, frame=None, onset=None):
    z = np.zeros(12)
    d = np.array(dur if dur is not None else z, dtype=float)
    return Profiles(
        dur=d / d.sum() if d.sum() else d,
        frame=np.array(frame if frame is not None else d, dtype=float),
        fine120=np.zeros(120),
        onset=np.array(onset if onset is not None else d, dtype=float),
        end=np.array(end if end is not None else d, dtype=float),
        mix=None if mix is None else np.array(mix, dtype=float),
        bass=None if bass is None else np.array(bass, dtype=float),
    )


def vec(**kw):
    v = np.zeros(12)
    for k, val in kw.items():
        v[int(k[1:])] = val
    return v


def test_fifth_trap_pa_held_longer_but_phrases_end_on_sa():
    # Sa = C (0). Pa (7) is held more than Sa, but every phrase resolves to Sa.
    dur = vec(p0=0.25, p2=0.1, p4=0.1, p5=0.1, p7=0.3, p9=0.1, p11=0.05)
    end = vec(p0=0.8, p7=0.2)
    est = estimate_tonic(prof(dur=dur, end=end), cfg)
    assert est.pc == 0
    order = np.argsort(-est.scores)
    assert est.ambiguity == "fifth" or int(order[1]) == 7


def test_relative_minor_trap_with_major_pad():
    # Melody in Asavari on A (9): pitch set == C major. A held and phrases end on A; pad plays C-E-G.
    dur = vec(p9=0.3, p4=0.2, p2=0.12, p0=0.12, p7=0.12, p5=0.07, p11=0.07)
    end = vec(p9=0.85, p4=0.15)
    mix = vec(p0=0.4, p4=0.3, p7=0.3)
    est = estimate_tonic(prof(dur=dur, end=end, mix=mix), cfg)
    assert est.pc == 9


def test_bass_breaks_tie():
    dur = vec(p0=0.2, p7=0.2, p4=0.15, p2=0.15, p9=0.15, p5=0.15)
    end = vec(p0=0.5, p7=0.5)
    est_c = estimate_tonic(prof(dur=dur, end=end, bass=vec(p0=0.7, p7=0.3)), cfg)
    est_g = estimate_tonic(prof(dur=dur, end=end, bass=vec(p7=0.7, p2=0.3)), cfg)
    assert est_c.pc == 0 and est_g.pc == 7


def test_drone_dominates():
    # melody spends equal time on every note of the F# major set and ends anywhere: only the drone decides
    dur = vec(p6=1, p8=1, p10=1, p11=1, p1=1, p3=1, p5=1)
    end = dur
    mix = vec(p6=0.6, p1=0.3, p0=0.1)  # shruti box on F# with C#
    est = estimate_tonic(prof(dur=dur, end=end, mix=mix), cfg)
    assert est.pc == 6


def test_override_and_labels():
    dur = vec(p0=0.5, p7=0.5)
    est = estimate_tonic(prof(dur=dur, end=vec(p0=1)), cfg, override="F#")
    assert est.pc == 6 and est.overridden
    est = estimate_tonic(prof(dur=dur, end=vec(p0=1)), cfg, override="370")
    assert est.pc == 6
    assert estimate_tonic(prof(dur=vec(p3=1), end=vec(p3=1)), cfg).label == "high"


def scale_prof(tonic, degrees, weights=None):
    v = np.zeros(12)
    for i, d in enumerate(degrees):
        v[(tonic + d) % 12] = weights[i] if weights else 1.0
    return prof(dur=v, end=vec(**{f"p{tonic}": 1}))


def test_scale_templates():
    assert estimate_scale(scale_prof(6, [0, 2, 3, 5, 7, 9, 10]), 6, cfg).name == "Kafi"
    assert estimate_scale(scale_prof(2, [0, 2, 4, 6, 7, 9, 11]), 2, cfg).name == "Kalyan"
    assert estimate_scale(scale_prof(0, [0, 2, 4, 7, 9]), 0, cfg).name == "Bhupali"
    assert estimate_scale(scale_prof(0, [0, 3, 5, 8, 10]), 0, cfg).name == "Malkauns"


def test_scale_mixed_variant_reported():
    # Khamaj-style: both N and n used
    s = estimate_scale(scale_prof(0, [0, 2, 4, 5, 7, 9, 10, 11], [1, 1, 1, 1, 1, 1, 0.6, 0.4]), 0, cfg)
    assert s.mixed["N"] is True
    assert s.name.startswith("Khamaj") or s.name.startswith("closest")


def test_score_components_normalised():
    S, comps = score_tonics(prof(dur=vec(p0=1), end=vec(p0=1)), cfg)
    assert abs(S.sum() - 1) < 1e-6
    for v in comps.values():
        assert abs(v.sum() - 1) < 1e-6 or v.sum() == 0


def test_snap_rare_degrees_moves_only_artefacts():
    from sargam.model import NoteEvent
    from sargam.tonic import snap_rare_degrees

    tonic = 5  # F
    # Bilawal-ish song: lots of S R G P D, one short unconfident komal Ga blip, one long confident komal Ga
    notes = []
    for i, deg in enumerate([0, 2, 4, 7, 9, 0, 2, 4, 7, 9, 0, 4]):
        notes.append(NoteEvent(start=i, end=i + 1.0, midi=60 + tonic + deg, confidence=0.9))
    blip = NoteEvent(start=20, end=20.1, midi=60 + tonic + 3, confidence=0.4, cents_dev=+30)
    real = NoteEvent(start=21, end=22.5, midi=60 + tonic + 3, confidence=0.9)
    notes += [blip, real]
    p = scale_prof(tonic, [0, 2, 4, 7, 9, 3], [12, 2, 3, 2, 2, 1.6])
    scale = estimate_scale(p, tonic, cfg)
    # komal Ga mass ~ 7% here -> above snap_degree_max_mass, nothing snapped
    assert snap_rare_degrees(notes, scale, tonic, cfg) == 0
    p = scale_prof(tonic, [0, 2, 4, 7, 9, 3], [12, 2, 3, 2, 2, 0.5])
    scale = estimate_scale(p, tonic, cfg)
    # now rare; the long confident note keeps the degree's mean confidence high -> still not an artefact degree
    assert snap_rare_degrees(notes, scale, tonic, cfg) == 0
    real.confidence = 0.3
    n = snap_rare_degrees(notes, scale, tonic, cfg)
    assert n == 1  # only the short blip moves (the long note exceeds snap_max_ms)
    assert blip.midi == 60 + tonic + 4 and "snapped" in blip.flags  # leaned sharp -> shuddha Ga
    assert real.midi == 60 + tonic + 3
