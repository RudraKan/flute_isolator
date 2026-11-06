"""Web UI API: upload -> background analysis -> render / audio / reference, on a short synthesized clip.

The real pipeline runs with `no_separate` (no Demucs), so these stay fast and need no dataset."""
import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sargam.audio import write_wav  # noqa: E402
from sargam.metrics import compare_files  # noqa: E402
from sargam.model import midi_to_hz  # noqa: E402
from sargam.synth import synthesize  # noqa: E402
from sargam.web.server import create_app  # noqa: E402

MELODY = "S R G - P G R S -\nG P D P G R S - -\nS R G R S - -\n"
REFERENCE = "SCALE OF THE SONG IS F# MAJOR\nSome lyrics line\nS.R.G..P.G.R.S..\nG.P.D.P.G.R.S...\nS.R.G.R.S...\n"
NO_SEP = {"no_separate": True, "pitch": "pyin"}


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    r = synthesize(MELODY, sa_hz=midi_to_hz(66), bpm=150, seed=3)
    p = tmp_path_factory.mktemp("audio") / "my_song.wav"
    write_wav(p, r.vocal, r.sr)
    return p


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "library")
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def upload(client, song, options=NO_SEP):
    with open(song, "rb") as f:
        r = client.post("/api/jobs", files={"file": (song.name, f, "audio/wav")}, data={"options": json.dumps(options)})
    assert r.status_code == 200, r.text
    return r.json()


def wait_done(client, item_id, timeout=60):
    assert client.app_state.worker.join_idle(timeout)
    meta = client.get(f"/api/jobs/{item_id}").json()
    assert meta["state"] in ("done", "error"), meta
    return meta


def test_options_schema(client):
    o = client.get("/api/options").json()
    keys = {f["key"]: f for f in o["schema"]}
    assert keys["style"]["default"] == "nw" and keys["pitch"]["default"] == "rmvpe"
    assert keys["simplify"]["default"] == "auto" and "rules" in keys["notes"]["choices"]
    assert {f["affects"] for f in o["schema"]} == {"analysis", "render"}
    assert [s["key"] for s in o["stages"]][0] == "load"
    assert client.get("/").status_code == 200 and "app.js" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_upload_render_audio_reference_pairs(client, song, tmp_path):
    meta = upload(client, song)
    assert meta["title"] == "my song" and meta["state"] in ("queued", "running", "done")
    meta = wait_done(client, meta["id"])
    assert meta["state"] == "done", meta.get("error")
    s = meta["summary"]
    assert s["sa"] == "F#" and s["n_notes"] > 10 and meta["revision"] == 1
    item = client.app_state.library.dir(meta["id"])
    for name in ("result.pkl", "transcription.sargam.txt", "transcription.notes.json", "transcription.melody.mid"):
        assert (item / name).exists(), name

    lib = client.get("/api/jobs").json()["items"]
    assert lib[0]["id"] == meta["id"] and lib[0]["sa"] == "F#"

    # structured rendering joins back to exactly the text output, for every style
    for style in ("nw", "dots", "dashes"):
        r = client.get(f"/api/jobs/{meta['id']}/render", params={"style": style, "detail": 1}).json()
        lines = ["".join(sg["t"] for sg in p["segs"]) for p in r["phrases"]]
        body = [l for l in r["text"].split("\n\n", 1)[1].splitlines() if l.strip()]
        assert lines == body, style
        timed = [sg for p in r["phrases"] for sg in p["segs"] if "s" in sg]
        assert timed and all(a["s"] <= b["s"] for a, b in zip(timed, timed[1:]))
        assert r["notes"] and len(r["notes"][0]) == 3
    assert client.get(f"/api/jobs/{meta['id']}/render", params={"style": "bogus"}).status_code == 400

    # audio with HTTP range (seeking)
    full = client.get(f"/api/jobs/{meta['id']}/audio/song")
    assert full.status_code == 200 and len(full.content) == song.stat().st_size
    part = client.get(f"/api/jobs/{meta['id']}/audio/song", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and len(part.content) == 100
    assert client.get(f"/api/jobs/{meta['id']}/audio/vocals").status_code == 404  # no separation for this one

    # downloads
    txt = client.get(f"/api/jobs/{meta['id']}/download/txt", params={"style": "dashes"})
    assert txt.status_code == 200 and "attachment" in txt.headers["content-disposition"] and "SCALE OF THE FLUTE IS F#" in txt.text
    assert client.get(f"/api/jobs/{meta['id']}/download/mid").content[:4] == b"MThd"

    # reference: scores identical to `sargam compare`, diff covers every reference symbol, saved as a pair
    assert client.get(f"/api/jobs/{meta['id']}/reference").json() == {"saved": None}
    r = client.post(f"/api/jobs/{meta['id']}/reference", json={"text": REFERENCE, "save": True})
    assert r.status_code == 200, r.text
    cmp = r.json()["comparison"]
    truth = tmp_path / "truth.txt"
    truth.write_text(REFERENCE)
    rep = compare_files(item / "transcription.notes.json", truth)
    assert cmp["f1"] == round(rep.f1, 4) and cmp["ser"] == round(rep.ser, 4) and cmp["tonic"]["ok"] is True
    assert cmp["f1"] > 0.8
    n_truth = sum(1 for l in cmp["lines"] for o in l["ops"] if o["op"] != "ins")
    assert n_truth == cmp["n_truth"] and len(cmp["lines"]) == 3
    assert all("t" in o for l in cmp["lines"] for o in l["ops"] if o["op"] in ("match", "sub", "ins"))
    assert (item / "truth.txt").read_text().startswith("SCALE OF THE FLUTE IS F#\nS.R.G..P")
    saved = client.get(f"/api/jobs/{meta['id']}/reference").json()
    assert saved["saved"]["source"] == "pasted" and saved["comparison"]["f1"] == cmp["f1"]
    pairs = client.get("/api/pairs").json()["pairs"]
    assert [p["id"] for p in pairs] == [meta["id"]] and pairs[0]["sa_truth"] == "F#"
    bad = client.post(f"/api/jobs/{meta['id']}/reference", json={"text": "just some words here"})
    assert bad.status_code == 400 and "No notation" in bad.json()["error"]
    assert client.post(f"/api/jobs/{meta['id']}/reference", json={"url": "file:///etc/passwd"}).status_code == 400

    # re-run with a forced Sa makes revision 2 and keeps the saved reference's score current
    r = client.post(f"/api/jobs/{meta['id']}/rerun", json={"options": {"sa": "C#"}})
    assert r.status_code == 200
    meta2 = wait_done(client, meta["id"])
    assert meta2["revision"] == 2 and meta2["summary"]["sa"] == "C#" and meta2["summary"]["overridden"]
    assert meta2["options"]["no_separate"] is True and len(meta2["revisions"]) == 2
    assert meta2["truth"]["f1"] is not None

    # rename and delete
    assert client.patch(f"/api/jobs/{meta['id']}", json={"title": "Renamed"}).json()["title"] == "Renamed"
    assert client.delete(f"/api/jobs/{meta['id']}").status_code == 200
    assert client.get(f"/api/jobs/{meta['id']}").status_code == 404
    assert client.get("/api/jobs").json()["items"] == []


def test_validation_errors(client, song, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with open(bad, "rb") as f:
        r = client.post("/api/jobs", files={"file": ("notes.txt", f, "text/plain")}, data={"options": "{}"})
    assert r.status_code == 400 and "Unsupported" in r.json()["error"]
    with open(song, "rb") as f:
        r = client.post("/api/jobs", files={"file": (song.name, f)}, data={"options": json.dumps({"pitch": "nope"})})
    assert r.status_code == 400
    with open(song, "rb") as f:
        r = client.post("/api/jobs", files={"file": (song.name, f)}, data={"options": json.dumps({"flutes": "zz"})})
    assert r.status_code == 400 and "flute" in r.json()["error"].lower()
    assert client.get("/api/jobs/nope").status_code == 404


def test_errors_are_reported_and_interrupted_jobs_recover(tmp_path, song):
    def broken(path, options, progress):
        progress("separating vocals (Demucs)")
        raise RuntimeError("demucs exploded")

    app = create_app(tmp_path / "library", runner=broken)
    with TestClient(app) as c:
        c.app_state = app.state
        meta = upload(c, song)
        meta = wait_done(c, meta["id"])
        assert meta["state"] == "error" and "demucs exploded" in meta["error"]
        assert (app.state.library.dir(meta["id"]) / "error.log").exists()
        assert c.get(f"/api/jobs/{meta['id']}/render").status_code == 409

    # a job left 'running' by a crashed server is marked as interrupted on the next start
    lib = app.state.library
    lib.update(meta["id"], state="running", started=time.time())
    app2 = create_app(tmp_path / "library", runner=broken)
    m = app2.state.library.load(meta["id"])
    assert m["state"] == "error" and "Interrupted" in m["error"]


def test_stage_keys():
    from sargam.web.jobs import stage_key

    assert stage_key("loading audio.mp3") == "load"
    assert stage_key("separating vocals (Demucs)") == "separate"
    assert stage_key("separated with demucs-mlx") == "separate"
    assert stage_key("tracking pitch (pyin)") == "pitch"
    assert stage_key("finding notes") == "notes"
    assert stage_key("detecting Sa and scale") == "tonic"
    assert stage_key("writing notation") == "write"
    assert stage_key("something else") is None


def test_simplify_and_engine_mapping():
    from sargam.web.server import clean_options, pipeline_args

    cfg, _ = pipeline_args(clean_options({"pitch": "rmvpe"}))
    assert cfg.resolved().simplify_min_ms == 100.0  # auto = the tracker's default
    cfg, _ = pipeline_args(clean_options({"pitch": "rmvpe", "simplify": "off"}))
    assert cfg.resolved().simplify_min_ms == -1.0
    cfg, _ = pipeline_args(clean_options({"simplify": "150"}))
    assert cfg.simplify_min_ms == 150.0 and cfg.notes_engine == "rules"
