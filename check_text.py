"""Check a transcription of a handwritten page — no model needed.

    python check_text.py ocr.txt                           # list suspicious spots
    python check_text.py ocr.txt --ref gt/page001.gt.tsv   # + accuracy against a reference

`ocr.txt` may be:
  * a .gt.tsv / .ocr.tsv file (as saved by proof_page.py / recognize.py),
  * chatbot-style text with lines "* Text: ..." and "* Gloss: ..."
    (Russian "Текст:/Глоссы:" and French "Texte:/Gloses:" also work),
  * plain text, one line of the language per line.

Without a reference, it compares the text with ALL checked pages in --gt-dir
(default: gt/) and flags:
  1. NEW CHARACTER - a letter+diacritic combination never seen in the checked
     pages (e.g. "ő" when the writer only ever puts a double acute on "u");
  2. NEW WORD      - a word never seen before, with the closest known words;
  3. COUNT         - the number of words differs from the number of gloss cells.
The more corrected pages there are in gt/, the more useful 1 and 2 become.

With --ref (a corrected version of the same page) it prints:
  CER all      share of wrong characters (edit distance / length);
  letters      the same with all diacritics removed (base letters only);
  tones        the same over the diacritics only;
and the word-by-word differences.
"""
import argparse
import csv
import difflib
import pathlib
import re
import sys
import unicodedata as ud

PUNCT = set(".,;:!?«»\"'()[]…-–—")


def nfd(s):
    """Decompose: 'ó' -> 'o' + combining acute. Makes tones separate characters."""
    return ud.normalize("NFD", s)


def graphemes(word):
    """Split a word into letters, each together with its diacritics (NFC)."""
    out = []
    for ch in nfd(word):
        if ud.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return [ud.normalize("NFC", g) for g in out]


def strip_marks(s):
    return "".join(c for c in nfd(s) if not ud.combining(c))


def only_marks(s):
    return "".join(c for c in nfd(s) if ud.combining(c))


def words(line):
    return [w for w in line.split() if not all(c in PUNCT for c in w)]


def gloss_cells(line):
    return [c.strip() for c in line.split("|") if c.strip() and c.strip() not in PUNCT]


def load(path: pathlib.Path):
    """Read any supported format -> list of [text_line, gloss_line or None]."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".tsv" or text.startswith("file\tkind\ttext"):
        pairs = []
        for r in csv.DictReader(text.splitlines(), delimiter="\t"):
            if r["kind"] in ("text", "ngen"):          # "ngen" = older name for "text"
                pairs.append([r["text"], None])
            elif r["kind"] == "gloss" and pairs and pairs[-1][1] is None:
                pairs[-1][1] = r["text"]
        return pairs
    m_text = re.findall(r"^\s*\*?\s*(?:Text|Texte|Текст)\s*:\s*(.*)$", text, re.M)
    if m_text:
        m_gl = re.findall(r"^\s*\*?\s*(?:Gloss(?:es)?|Gloses|Глоссы)\s*:\s*(.*)$", text, re.M)
        return [[t, m_gl[i] if i < len(m_gl) else None] for i, t in enumerate(m_text)]
    return [[ln, None] for ln in text.splitlines() if ln.strip()]


def inventory(gt_dir: pathlib.Path, exclude=None):
    """All letters+diacritics and all words found in the checked pages."""
    g_set, w_set = set(), set()
    for p in sorted(gt_dir.glob("*.gt.tsv")):
        if exclude and p.resolve() == exclude.resolve():
            continue
        for line, _ in load(p):
            for w in words(line):
                w_set.add(ud.normalize("NFC", w.lower()))
                g_set.update(g.lower() for g in graphemes(w))
    return g_set, w_set


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    return edit_distance(hyp, ref) / max(1, len(ref))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ocr", type=pathlib.Path, help="transcription to check")
    ap.add_argument("--ref", type=pathlib.Path, help="corrected .gt.tsv of the same page")
    ap.add_argument("--gt-dir", type=pathlib.Path, default=pathlib.Path(__file__).parent / "gt",
                    help="folder with checked pages (*.gt.tsv)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    hyp = load(a.ocr)
    # the reference page itself is left out of the word list, otherwise the check "knows the answer"
    g_known, w_known = inventory(a.gt_dir, exclude=a.ref)
    print(f"checked pages in {a.gt_dir}: {len(list(a.gt_dir.glob('*.gt.tsv')))}, "
          f"known words {len(w_known)}, known characters {len(g_known)}\n")

    n_flags = 0
    for i, (line, gl) in enumerate(hyp, 1):
        flags = []
        for w in words(line):
            wl = ud.normalize("NFC", w.lower())
            new_g = [g for g in graphemes(wl) if g not in g_known]
            if new_g and g_known:
                flags.append(f"NEW CHARACTER {' '.join(new_g)} in '{w}'")
            if w_known and wl not in w_known:
                near = difflib.get_close_matches(wl, w_known, n=3, cutoff=0.6)
                flags.append(f"NEW WORD '{w}'" + (f" - similar known: {', '.join(near)}" if near else ""))
        if gl is not None:
            nw, ng = len(words(line)), len(gloss_cells(gl))
            if nw != ng:
                flags.append(f"COUNT: {nw} words but {ng} gloss cells")
        if flags:
            n_flags += len(flags)
            print(f"{i}. {line}")
            for f in flags:
                print(f"     ! {f}")
    if not n_flags:
        print("nothing suspicious found" + ("" if w_known else " (no checked pages yet - nothing to compare with)"))

    if a.ref:
        ref = load(a.ref)
        H = " / ".join(nfd(p[0]) for p in hyp)
        R = " / ".join(nfd(p[0]) for p in ref)
        print(f"\nagainst reference {a.ref.name}:")
        print(f"  CER all   {cer(H, R):6.1%}")
        print(f"  letters   {cer(strip_marks(H), strip_marks(R)):6.1%}   (diacritics removed)")
        print(f"  tones     {cer(only_marks(H), only_marks(R)):6.1%}   (diacritics only)")
        print("  differences (line: transcription -> reference):")
        for k, (h, r) in enumerate(zip(hyp, ref), 1):
            hw, rw = words(h[0]), words(r[0])
            sm = difflib.SequenceMatcher(a=[nfd(x) for x in hw], b=[nfd(x) for x in rw], autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "equal":
                    print(f"    {k}: {' '.join(hw[i1:i2]) or '-'}  ->  {' '.join(rw[j1:j2]) or '-'}")
        if len(hyp) != len(ref):
            print(f"  (text lines: {len(hyp)} in transcription, {len(ref)} in reference)")


if __name__ == "__main__":
    main()
