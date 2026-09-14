"""Sargam symbols, notation parser and renderer.

Grammar (notationsworld style):
  S r R g G m M P D d n N   (komal lowercase, tivra Ma = M, shuddha Ma = m)
  .P / ,P  lower octave     S' upper octave      ..P / S'' two octaves
  {RSn}S   murki cluster glued before a note
  (R)S     kann swar (grace) glued before a note
  S~G      meend (glide)
  -        sustain (one per extra beat)
  _        rest
The parser is shared by the synth (tests), the truth loader and the metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .model import NoteEvent, Phrase, parse_pitch_name, pitch_name

DEGREE_SYMBOLS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]
SYMBOL_TO_DEGREE = {s: i for i, s in enumerate(DEGREE_SYMBOLS)}
_LETTERS = set(DEGREE_SYMBOLS)

HEADER_RE = re.compile(
    r"SCALE\s+OF\s+THE\s+(?:FLUTE|SONG)\s+IS\s+([A-G][#b♯♭]?)|^\s*(?:SA|SCALE|TONIC)\s*[:=]\s*([A-G][#b♯♭]?)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Token:
    kind: Literal["note", "sustain", "rest"]
    degree: int = -1
    octave: int = 0
    glide_to: tuple[int, int] | None = None
    grace: tuple[int, int] | None = None
    ornament: list[tuple[int, int]] | None = None

    @property
    def pitch(self) -> tuple[int, int]:
        return (self.degree, self.octave)

    def __str__(self) -> str:
        if self.kind == "sustain":
            return "-"
        if self.kind == "rest":
            return "_"
        out = ""
        if self.ornament:
            out += "{" + "".join(swara_symbol(d, o) for d, o in self.ornament) + "}"
        if self.grace:
            out += "(" + swara_symbol(*self.grace) + ")"
        out += swara_symbol(self.degree, self.octave)
        if self.glide_to:
            out += "~" + swara_symbol(*self.glide_to)
        return out


# ---------------------------------------------------------------- symbols


def swara_symbol(degree: int, octave: int = 0) -> str:
    s = DEGREE_SYMBOLS[degree % 12]
    if octave < 0:
        return "." * (-octave) + s
    return s + "'" * octave


def rel_to_symbol(rel: int) -> str:
    """Semitones relative to Sa -> symbol (e.g. -5 -> '.P', 12 -> \"S'\")."""
    return swara_symbol(rel % 12, rel // 12)


def midi_to_symbol(midi: int, sa_midi: int) -> str:
    return rel_to_symbol(midi - sa_midi)


# ---------------------------------------------------------------- parsing


class ParseError(ValueError):
    pass


def _parse_atom(s: str, i: int) -> tuple[tuple[int, int], int]:
    """Parse prefix dots, letter, suffix quotes at s[i:]. Returns ((degree, octave), next_i)."""
    n = len(s)
    octave = 0
    while i < n and s[i] in ".,":
        octave -= 1
        i += 1
    if i >= n or s[i] not in _LETTERS:
        raise ParseError(f"expected swara letter at {i} in {s!r}")
    degree = SYMBOL_TO_DEGREE[s[i]]
    i += 1
    if octave < 0 and i < n and s[i] == "'":
        raise ParseError("both lower and upper octave marks")
    while i < n and s[i] == "'":
        octave += 1
        i += 1
    return (degree, octave), i


def parse_word(word: str) -> list[Token]:
    """Parse one whitespace-free chunk into tokens. Raises ParseError if not notation."""
    tokens: list[Token] = []
    s = word
    i, n = 0, len(s)
    ornament: list[tuple[int, int]] | None = None
    grace: tuple[int, int] | None = None
    while i < n:
        c = s[i]
        if c == "-":
            tokens.append(Token("sustain"))
            i += 1
        elif c == "_":
            tokens.append(Token("rest"))
            i += 1
        elif c == "{":
            j = s.find("}", i)
            if j < 0:
                raise ParseError("unclosed {")
            inner, k, orn = s[i + 1 : j], 0, []
            while k < len(inner):
                if inner[k] in " ,":
                    k += 1
                    continue
                atom, k = _parse_atom(inner, k)
                orn.append(atom)
            if not orn:
                raise ParseError("empty {}")
            ornament = orn
            i = j + 1
        elif c == "(":
            j = s.find(")", i)
            if j < 0:
                raise ParseError("unclosed (")
            inner = s[i + 1 : j].strip()
            atom, k = _parse_atom(inner, 0)
            if k != len(inner):
                raise ParseError("bad grace")
            grace = atom
            i = j + 1
        elif c in ".,'" or c in _LETTERS:
            atom, i = _parse_atom(s, i)
            tok = Token("note", atom[0], atom[1], ornament=ornament, grace=grace)
            ornament, grace = None, None
            if i < n and s[i] == "~":
                target, i = _parse_atom(s, i + 1)
                tok.glide_to = target
            tokens.append(tok)
        else:
            raise ParseError(f"unexpected {c!r} in {word!r}")
    if ornament is not None or grace is not None:
        raise ParseError("dangling ornament/grace")
    return tokens


_DOT_DIALECT_RE = re.compile(r"[SRGMPDNsrgmpdn]['\u2019]*\.+")
_DOT_ATOM_RE = re.compile(r"(\.?)([SRGMPDNsrgmpdn])(['\u2019]*)(\.*)")


def _normalise(line: str) -> str:
    return line.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")


def _dot_dialect_to_ours(line: str) -> str | None:
    """Convert 'G..m.. R.S.R..' (dots after a note = duration units) into 'G - m - R S R -'.
    A dot at the start of a chunk is still a lower-octave mark. Returns None if the line doesn't convert."""
    words = []
    for w in line.split():
        pos, out = 0, []
        while pos < len(w):
            m = _DOT_ATOM_RE.match(w, pos)
            if not m:
                return None
            pre, letter, quotes, dots = m.groups()
            out.append(pre + letter + quotes)
            out.extend(["-"] * max(0, len(dots) - 1))
            pos = m.end()
        words.append(" ".join(out))
    return " ".join(words)


def parse_line(line: str) -> list[Token] | None:
    """Parse a line of notation. Returns None if the line looks like lyrics/prose.

    Accepts our grammar and the common 'dot-duration' dialect used on notation sites."""
    line = _normalise(line)
    words = line.split()
    if not words:
        return None
    if len(_DOT_DIALECT_RE.findall(line)) >= 3:
        converted = _dot_dialect_to_ours(line)
        if converted is not None:
            line, words = converted, converted.split()
    parsed: list[list[Token]] = []
    ok = 0
    for w in words:
        try:
            toks = parse_word(w)
            ok += 1
            parsed.append(toks)
        except ParseError:
            parsed.append([])
    if ok / len(words) < 0.7:
        return None
    return [t for toks in parsed for t in toks]


def parse_notation(text: str) -> tuple[str | None, list[list[Token]]]:
    """Parse a notationsworld-style page or our own output.

    Returns (sa pitch name or None, phrases) where each phrase is a list of tokens.
    Lines that don't parse as notation (lyrics, headers, prose) are skipped.
    """
    sa: str | None = None
    m = HEADER_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        sa = pitch_name(parse_pitch_name(raw))
    phrases: list[list[Token]] = []
    for line in text.splitlines():
        if HEADER_RE.search(line):
            continue
        toks = parse_line(line)
        if toks:
            phrases.append(toks)
    return sa, phrases


def collapse_tokens(tokens: list[Token], keep_octave: bool = True) -> list[str]:
    """Sequence of plain note symbols: sustains/rests dropped, ornaments/graces/glides stripped,
    consecutive repeats collapsed. Used by the metrics."""
    out: list[str] = []
    for t in tokens:
        if t.kind != "note":
            continue
        sym = swara_symbol(t.degree, t.octave if keep_octave else 0)
        if not out or out[-1] != sym:
            out.append(sym)
    return out


# ---------------------------------------------------------------- rendering


def _sustain_dashes(duration: float, beat_period: float | None, median_dur: float) -> int:
    if beat_period and beat_period > 0:
        held = int(round(duration / beat_period))
        return max(0, min(8, held - 1))
    if median_dur > 0 and duration >= 2 * median_dur:
        return max(0, min(8, int(duration / median_dur) - 1))
    return 0


def note_to_token(n: NoteEvent, sa_midi: int, detail: int = 1) -> Token:
    rel = n.midi - sa_midi
    tok = Token("note", rel % 12, rel // 12)
    if detail >= 1:
        if n.kind == "glide" and n.glide_to_midi is not None and n.glide_to_midi != n.midi:
            r2 = n.glide_to_midi - sa_midi
            tok.glide_to = (r2 % 12, r2 // 12)
        if n.grace_before is not None:
            r3 = n.grace_before - sa_midi
            tok.grace = (r3 % 12, r3 // 12)
        if n.ornament_before:
            tok.ornament = [((m - sa_midi) % 12, (m - sa_midi) // 12) for m in n.ornament_before]
    return tok


def render_phrase(
    notes: list[NoteEvent],
    sa_midi: int,
    beat_period: float | None = None,
    median_dur: float = 0.0,
    detail: int = 1,
    style: str = "dashes",
) -> str:
    """style 'dashes': `S R - G` (one '-' per extra unit). style 'dots': `S.R..G.` as on notation sites
    (each note followed by one dot per unit it lasts; notes of one breath group are glued, groups separated by a space)."""
    parts: list[str] = []
    if detail == 0:
        from .notes import expand_ornaments

        notes = expand_ornaments(notes)
    if style == "dots":
        unit = (beat_period / 2 if beat_period else median_dur) or 0.0
        chunks: list[str] = []
        cur = ""
        last_end = None
        for n in notes:
            if last_end is not None and n.start - last_end > 0.12 and cur:
                chunks.append(cur)
                cur = ""
            units = max(1, int(round(n.duration / unit))) if unit > 0 else 1
            cur += str(note_to_token(n, sa_midi, detail)) + "." * min(units, 6)
            last_end = n.end
        if cur:
            chunks.append(cur)
        return " ".join(chunks)
    for n in notes:
        s = str(note_to_token(n, sa_midi, detail))
        if detail >= 2 and n.confidence < 0.5:
            s += "?"
        dashes = _sustain_dashes(n.duration, beat_period, median_dur)
        parts.append(s)
        parts.extend(["-"] * dashes)
    return " ".join(parts)


def render_notation(
    phrases: list[Phrase],
    sa_midi: int,
    beat_period: float | None = None,
    detail: int = 1,
    phrases_per_block: int = 4,
    header_lines: list[str] | None = None,
    style: str = "dashes",
) -> str:
    durs = sorted(n.duration for p in phrases for n in p.notes)
    median_dur = durs[len(durs) // 2] if durs else 0.0
    lines: list[str] = list(header_lines or [])
    if lines:
        lines.append("")
    for i, p in enumerate(phrases):
        if i and i % phrases_per_block == 0:
            lines.append("")
        lines.append(render_phrase(p.notes, sa_midi, beat_period, median_dur, detail, style))
    return "\n".join(lines) + "\n"


def tokens_to_symbols(tokens: list[Token]) -> str:
    return " ".join(str(t) for t in tokens)
