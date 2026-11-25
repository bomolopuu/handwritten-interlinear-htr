"""Turn checked pages (gt/*.gt.tsv) into training data for Kraken.

    python gt_export.py                       # defaults: --gt gt --lines lines --out train
    python gt_export.py --kind text           # only lines of the language, no glosses

For every non-empty line of every gt/*.gt.tsv it puts into train/:
    page001_01.png      the line image (copied from lines/)
    page001_01.gt.txt   its text (Unicode NFC)
and writes three lists of image names:
    manifest.txt        all lines
    split_train.txt     lines to learn from
    split_val.txt       every 10th line, held out to measure the error rate

--kind all (default) keeps the gloss lines too: they are in another language,
but they teach the model the handwriting, which helps when data is scarce.

At the end it lists characters seen fewer than 5 times: the model cannot learn
them yet - transcribe more pages that contain them.
"""
import argparse
import csv
import pathlib
import shutil
import unicodedata as ud


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=pathlib.Path, default=pathlib.Path("gt"), help="folder with *.gt.tsv")
    ap.add_argument("--lines", type=pathlib.Path, default=pathlib.Path("lines"), help="folder with line images")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("train"))
    ap.add_argument("--kind", choices=["text", "gloss", "all"], default="all")
    ap.add_argument("--val-every", type=int, default=10, help="every N-th line goes to validation")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    names, chars = [], {}
    for tsv in sorted(a.gt.glob("*.gt.tsv")):
        for r in csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"):
            kind = {"ngen": "text"}.get(r["kind"], r["kind"])
            text = ud.normalize("NFC", r["text"].strip())
            if kind == "skip" or not text or (a.kind != "all" and kind != a.kind):
                continue
            img = a.lines / r["file"]
            if not img.exists():
                print(f"missing image {img} - run segment_lines.py on that page first; skipped")
                continue
            shutil.copy2(img, a.out / r["file"])
            (a.out / f"{img.stem}.gt.txt").write_text(text + "\n", encoding="utf-8")
            names.append(r["file"])
            for c in ud.normalize("NFD", text):   # count tones separately from letters
                chars[c] = chars.get(c, 0) + 1

    val = names[::a.val_every]
    trn = [n for n in names if n not in val]
    for fname, lst in (("manifest.txt", names), ("split_train.txt", trn), ("split_val.txt", val)):
        (a.out / fname).write_text("".join(n + "\n" for n in lst), encoding="utf-8")
    print(f"{len(names)} lines ({len(trn)} train, {len(val)} validation), {len(chars)} distinct characters")
    rare = sorted((n, c) for c, n in chars.items() if n < 5)
    if rare:
        print("rare characters (<5 times, the model will not learn them yet):",
              " ".join(f"{'◌' + c if ud.combining(c) else c}x{n}" for n, c in rare))


if __name__ == "__main__":
    main()
