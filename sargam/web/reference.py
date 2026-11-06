"""Compare a transcription with a reference notation (notationsworld paste or URL) and build the diff view."""
from __future__ import annotations

import html as html_mod
import re
import urllib.parse
import urllib.request

from ..metrics import align_ops, compare_tokens, tokens_from_json
from ..swara import Token, nw_symbol, parse_notation, parse_nw_line, swara_symbol

_ARTICLE_RE = re.compile(r'class="entry-content[^"]*"[^>]*>(.*?)(?:Also Read This|You May Also Like|</article>)', re.S)


class ReferenceError(ValueError):
    pass


def html_to_text(raw: str) -> str:
    """Same stripping as the `sargam truth` command, limited to the article body when the page has one."""
    m = _ARTICLE_RE.search(raw)
    body = m.group(1) if m else raw
    txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
    txt = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", txt)
    txt = html_mod.unescape(re.sub(r"<[^>]+>", " ", txt))
    if m:  # the scale header sits above the article body on some pages
        head = re.search(r"SCALE\s+OF\s+THE\s+(?:FLUTE|SONG)\s+IS\s+[A-G][#b]?[^<\n]*", html_mod.unescape(raw), re.I)
        if head and head.group(0) not in txt:
            txt = head.group(0) + "\n" + txt
    return txt


def fetch_page_text(url: str, timeout: float = 20.0) -> str:
    parts = urllib.parse.urlparse(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ReferenceError("Enter a full http(s) URL of a notation page.")
    req = urllib.request.Request(url.strip(), headers={"User-Agent": "Mozilla/5.0 (sargam evaluation helper)"})
    try:
        raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        raise ReferenceError(f"Couldn't fetch that page: {e}") from e
    return html_to_text(raw)


def notation_lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip() and parse_nw_line(l)]


def truth_file_text(sa: str | None, text: str) -> str:
    """truth.txt in the same shape `sargam truth` writes."""
    return (f"SCALE OF THE FLUTE IS {sa}\n" if sa else "") + "\n".join(notation_lines(text)) + "\n"


def _collapse(tokens: list[Token], tags: list) -> list[dict]:
    """collapse_tokens() that remembers degree/octave and a tag (time or line) of the first token of each run."""
    out: list[dict] = []
    for t, tag in zip(tokens, tags):
        if t.kind != "note":
            continue
        sym = swara_symbol(t.degree, t.octave)
        if not out or out[-1]["sym"] != sym:
            out.append({"sym": sym, "nw": nw_symbol(t.degree, t.octave), "tag": tag})
    return out


def pred_tokens_with_times(res) -> tuple[list[Token], list[float]]:
    from ..render import to_json

    data = to_json(res)
    tokens = tokens_from_json(data)
    times: list[float] = []
    for n in data["notes"]:
        k = len(n.get("ornament_before") or []) + (1 if n.get("grace_before") is not None else 0)
        times.extend([float(n["start"])] * (k + 1))
    assert len(times) == len(tokens)
    return tokens, times


def compare(res, text: str) -> dict:
    """Scores (identical to `sargam compare`) plus the aligned diff grouped by reference line."""
    truth_sa, phrases = parse_notation(text, "nw")
    if not phrases:
        raise ReferenceError("No notation lines found. Paste lines like  G..m.. R..G..R.S.R..")
    pred_tokens, times = pred_tokens_with_times(res)
    truth_tokens = [t for p in phrases for t in p]
    rep = compare_tokens(pred_tokens, truth_tokens, res.tonic.name, truth_sa)
    pred = _collapse(pred_tokens, times)
    truth = _collapse(truth_tokens, [i for i, p in enumerate(phrases) for _ in p])
    ops = align_ops([p["sym"] for p in pred], [t["sym"] for t in truth])
    lines: list[list[dict]] = [[] for _ in phrases]
    pending: list[dict] = []  # extras before the first truth symbol of a line attach to that line
    for op, i, j in ops:
        item = {"op": op}
        if i is not None:
            item.update(pred=pred[i]["sym"], pred_nw=pred[i]["nw"], t=round(pred[i]["tag"], 3))
        if j is not None:
            item.update(truth=truth[j]["sym"], truth_nw=truth[j]["nw"])
            lines[truth[j]["tag"]].extend(pending)
            pending = []
            lines[truth[j]["tag"]].append(item)
        else:
            pending.append(item)
    if pending:
        lines[-1].extend(pending)
    counts = {k: sum(1 for op, _, _ in ops if op == k) for k in ("match", "octave", "sub", "ins", "del")}
    return {
        "tonic": {"pred": rep.tonic_pred, "truth": rep.tonic_truth, "ok": rep.tonic_ok, "relation": rep.tonic_relation},
        "ser": round(rep.ser, 3), "ser_no_octave": round(rep.ser_no_octave, 3),
        "precision": round(rep.precision, 3), "recall": round(rep.recall, 3), "f1": round(rep.f1, 3),
        "scale_jaccard": round(rep.scale_jaccard, 3), "n_pred": rep.n_pred, "n_truth": rep.n_truth,
        "counts": counts, "lines": [{"i": k, "ops": ops_k} for k, ops_k in enumerate(lines)],
        "table": rep.table(),
    }
