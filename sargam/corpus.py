"""Notation corpus from notationsworld: polite crawler, page parser, JSONL export.

Everything lands under DATA_DIR/corpus (gitignored): raw/<slug>.html.gz page cache, urls.json, songs.jsonl.
robots.txt allows crawling; we fetch one page at a time with a delay and never refetch a cached page.

A song record:
  {"url", "slug", "kind": "sargam"|"piano", "title", "tonic": "F#"|None, "tonic_source": "header"|"key-profile",
   "mode", "lines": [{"lyric": str|None, "raw": str, "notes": [[rel, units, chunk_start], ...]}]}
rel = semitones relative to Sa (degree + 12*octave), units = length in the page's dot units (piano pages: 1),
chunk_start = 1 when the note opens a whitespace-separated group (a short breath on the page).
"""
from __future__ import annotations

import gzip
import html as html_mod
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from .audio import DATA_DIR
from .model import parse_pitch_name, pitch_name
from .swara import HEADER_RE, TIVRA_MA, Token, parse_nw_line

SITE = "https://www.notationsworld.com"
SITEMAPS = [f"{SITE}/post-sitemap{s}.xml" for s in ("", "2", "3", "4")]
USER_AGENT = "Mozilla/5.0 (compatible; sargam-corpus/0.1; personal non-commercial research)"
_SCALE_RE = re.compile(r"SCALE\s+OF\s+THE\s+(?:FLUTE|SONG)\s+IS\s+([A-G][#b♯♭]?)\s*(MAJOR|MINOR)?", re.IGNORECASE)
_PIANO_NOTE_RE = re.compile(r"^([A-G])([#b]?)(-?\d)$")
_BOILERPLATE = re.compile(r"flute notes|harmonium|convert|undo changes|remove numbers|notationsworld|also read|you may also like|sargam notes|piano notes", re.I)


def corpus_dir() -> Path:
    d = DATA_DIR / "corpus"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def http_get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore")


def page_kind(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    if "sargam" in slug:
        return "sargam"
    if "piano-notes" in slug or "paino-notes" in slug:
        return "piano"
    return "other"


def slug_of(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")


def list_urls(refresh: bool = False) -> list[str]:
    path = corpus_dir() / "urls.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    urls: list[str] = []
    for sm in SITEMAPS:
        urls += [u for u in re.findall(r"<loc>([^<]+)</loc>", http_get(sm)) if u.endswith(".html")]
    urls = sorted(set(urls))
    path.write_text(json.dumps(urls, indent=0))
    return urls


def cached_html(url: str) -> str | None:
    p = corpus_dir() / "raw" / f"{slug_of(url)}.html.gz"
    return gzip.decompress(p.read_bytes()).decode("utf-8") if p.exists() else None


def fetch(url: str) -> tuple[str, bool]:
    """(html, was_cached)."""
    got = cached_html(url)
    if got is not None:
        return got, True
    text = http_get(url)
    (corpus_dir() / "raw" / f"{slug_of(url)}.html.gz").write_bytes(gzip.compress(text.encode("utf-8")))
    return text, False


def crawl(kinds: tuple[str, ...] = ("sargam", "piano"), delay_s: float = 1.2, limit: int | None = None, progress=None) -> int:
    """Fetch every page of the given kinds not yet cached. Returns the number of new pages."""
    say = progress or (lambda *_: None)
    todo = [u for u in list_urls() if page_kind(u) in kinds and cached_html(u) is None]
    if limit is not None:
        todo = todo[:limit]
    new = 0
    for i, url in enumerate(todo):
        try:
            fetch(url)
            new += 1
        except Exception as e:  # noqa: BLE001 - keep crawling past one bad page
            say(f"[{i + 1}/{len(todo)}] failed {url}: {e}")
            time.sleep(delay_s * 5)
            continue
        if (i + 1) % 25 == 0:
            say(f"[{i + 1}/{len(todo)}] {slug_of(url)}")
        time.sleep(delay_s)
    return new


# ---------------------------------------------------------------- page parsing


def page_lines(html: str) -> tuple[str, list[str]]:
    """(title, text lines of the post body). Stops before the related-posts footer."""
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    title = html_mod.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""
    m = re.search(r'class="entry-content[^"]*"[^>]*>(.*?)(?:Also Read This|You May Also Like|</article>)', html, re.S)
    body = m.group(1) if m else html
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
    body = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", body)
    body = html_mod.unescape(re.sub(r"<[^>]+>", " ", body))
    lines = [re.sub(r"\s+", " ", l).strip() for l in body.splitlines()]
    return title, [l for l in lines if l]


def _notes_from_tokens(tokens: list[Token], chunk_starts: list[bool]) -> list[list[int]]:
    out: list[list[int]] = []
    k = 0
    for t in tokens:
        if t.kind == "note":
            out.append([t.degree + 12 * t.octave, 1, int(chunk_starts[k]) if k < len(chunk_starts) else 0])
            k += 1
        elif t.kind == "sustain" and out:
            out[-1][1] += 1
    return out


def _chunk_starts(raw: str) -> list[bool]:
    """Which notes open a whitespace group, by re-parsing each chunk on its own."""
    from .swara import _nw_normalise, _parse_nw_word, ParseError

    starts: list[bool] = []
    for w in _nw_normalise(raw).split():
        try:
            toks, _ = _parse_nw_word(w)
        except ParseError:
            continue
        n = sum(t.kind == "note" for t in toks)
        starts += [True] + [False] * (n - 1) if n else []
    return starts


def _is_lyric(line: str) -> bool:
    return bool(re.search(r"[A-Za-zऀ-ॿ]{2,}", line)) and not _BOILERPLATE.search(line) and len(line) < 120


def parse_sargam_page(url: str, html: str) -> dict | None:
    title, lines = page_lines(html)
    tonic, mode = None, None
    out_lines: list[dict] = []
    lyric: str | None = None
    used_m = False
    for line in lines:
        hm = _SCALE_RE.search(line)
        if hm:
            if tonic is None:
                tonic = pitch_name(parse_pitch_name(hm.group(1)))
                mode = (hm.group(2) or "").lower() or None
            continue
        if HEADER_RE.search(line):
            continue
        parsed = parse_nw_line(line)
        if parsed:
            toks, um = parsed
            used_m |= um
            out_lines.append({"lyric": lyric, "raw": line, "notes": _notes_from_tokens(toks, _chunk_starts(line))})
            lyric = None
        elif _is_lyric(line):
            lyric = line
    if not out_lines:
        return None
    if not used_m:  # page writes shuddh Ma as M
        for ln in out_lines:
            for n in ln["notes"]:
                if n[0] % 12 == TIVRA_MA:
                    n[0] -= 1
    return {"url": url, "slug": slug_of(url), "kind": "sargam", "title": title, "tonic": tonic,
            "tonic_source": "header" if tonic else None, "mode": mode, "lines": out_lines}


# Krumhansl-Kessler key profiles, used to pick Sa for piano pages that only list absolute notes.
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(midis: list[int]) -> tuple[int, str]:
    """(tonic pitch class, 'major'|'minor') from a note list. Minor keys are reported on their own tonic."""
    hist = np.bincount(np.asarray(midis) % 12, minlength=12).astype(float)
    best = (-2.0, 0, "major")
    for mode, prof in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
        for t in range(12):
            r = float(np.corrcoef(hist, np.roll(prof, t))[0, 1])
            if r > best[0]:
                best = (r, t, mode)
    return best[1], best[2]


def _parse_piano_line(line: str) -> list[int] | None:
    words = line.replace(",", " ").replace("-", " ").split()
    if len(words) < 2:
        return None
    midis = []
    for w in words:
        m = _PIANO_NOTE_RE.match(w)
        if not m:
            return None
        pc = parse_pitch_name(m.group(1) + m.group(2))
        midis.append(12 * (int(m.group(3)) + 1) + pc)
    return midis


def parse_piano_page(url: str, html: str) -> dict | None:
    title, lines = page_lines(html)
    raw_lines: list[tuple[str | None, str, list[int]]] = []
    lyric: str | None = None
    for line in lines:
        midis = _parse_piano_line(line)
        if midis:
            raw_lines.append((lyric, line, midis))
            lyric = None
        elif _is_lyric(line):
            lyric = line
    allm = [m for _, _, ms in raw_lines for m in ms]
    if len(allm) < 8:
        return None
    pc, mode = estimate_key(allm)
    # Sa in the octave that keeps the median note within S..N of the middle octave
    med = int(np.median(allm))
    sa = med - ((med - pc) % 12)
    out_lines = [{"lyric": ly, "raw": raw, "notes": [[m - sa, 1, int(i == 0)] for i, m in enumerate(ms)]} for ly, raw, ms in raw_lines]
    return {"url": url, "slug": slug_of(url), "kind": "piano", "title": title, "tonic": pitch_name(pc),
            "tonic_source": "key-profile", "mode": mode, "lines": out_lines}


def parse_page(url: str, html: str) -> dict | None:
    kind = page_kind(url)
    if kind == "sargam":
        return parse_sargam_page(url, html)
    if kind == "piano":
        return parse_piano_page(url, html)
    return None


def build(out: Path | None = None) -> dict:
    """Parse every cached page into songs.jsonl. Returns summary counts."""
    out = out or corpus_dir() / "songs.jsonl"
    counts = {"pages": 0, "songs": 0, "sargam": 0, "piano": 0, "lines": 0, "notes": 0, "no_tonic": 0}
    with open(out, "w") as f:
        for p in sorted((corpus_dir() / "raw").glob("*.html.gz")):
            counts["pages"] += 1
            slug = p.name.removesuffix(".html.gz")
            rec = parse_page(f"{SITE}/{slug}.html", gzip.decompress(p.read_bytes()).decode("utf-8"))
            if rec is None:
                continue
            counts["songs"] += 1
            counts[rec["kind"]] += 1
            counts["lines"] += len(rec["lines"])
            counts["notes"] += sum(len(l["notes"]) for l in rec["lines"])
            counts["no_tonic"] += rec["tonic"] is None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return counts


def load_songs(path: Path | None = None, kinds: tuple[str, ...] = ("sargam", "piano")) -> list[dict]:
    path = path or corpus_dir() / "songs.jsonl"
    with open(path) as f:
        return [r for r in map(json.loads, f) if r["kind"] in kinds]


def line_text(notes: list[list[int]]) -> str:
    """Render a corpus line back to notationsworld notation (for inspection)."""
    from .swara import nw_symbol

    parts: list[str] = []
    for rel, units, start in notes:
        sym = nw_symbol(rel % 12, rel // 12) + "." * units
        parts.append((" " if start and parts else "") + sym)
    return "".join(parts)
