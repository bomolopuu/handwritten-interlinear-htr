"""Recognise new pages with a trained Kraken model and open them for proofreading.

    python recognize.py scans/page003.jpg [more scans] --model models/htr_best.mlmodel

For each scan:
  1. segment_lines.py cuts the page into lines   -> lines/page003_NN.png
  2. kraken reads every line image               -> lines/page003_NN.ocr.txt
     (--no-segmentation: the lines are already cut, kraken must not re-cut them)
  3. the lines are collected in one table        -> ocr/page003.ocr.tsv
  4. proof_page.py builds the proofreading page  -> proof/page003.html
Correct the page in the browser, save, put the file into gt/ - it then
improves the next model (gt_export.py + train_kraken.sh).

Needs Kraken (Linux/macOS, or WSL on Windows). If the kraken program is not on
your PATH, give it with --kraken ~/venvs/kraken/bin/kraken.
"""
import argparse
import csv
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans", nargs="+", type=pathlib.Path)
    ap.add_argument("--model", required=True, type=pathlib.Path, help="trained .mlmodel file")
    ap.add_argument("--kraken", default=shutil.which("kraken") or "kraken", help="path to the kraken program")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--lines", type=pathlib.Path, default=HERE / "lines")
    ap.add_argument("--ocr", type=pathlib.Path, default=HERE / "ocr")
    ap.add_argument("--first", choices=["text", "gloss"], default="text", help="type of the first line on a page")
    a = ap.parse_args()

    a.ocr.mkdir(parents=True, exist_ok=True)
    order = ["text", "gloss"] if a.first == "text" else ["gloss", "text"]
    for scan in a.scans:
        page = scan.stem
        # 1. cut lines (same defaults as running segment_lines.py by hand)
        subprocess.run([sys.executable, str(HERE / "segment_lines.py"), str(scan), "--out", str(a.lines)], check=True)
        rows = list(csv.DictReader((a.lines / f"{page}.lines.tsv").open(encoding="utf-8"), delimiter="\t"))

        # 2. one kraken call for the whole page: "-i input output" pairs, then the ocr step
        cmd = [a.kraken, "-d", a.device]
        for r in rows:
            img = a.lines / r["file"]
            cmd += ["-i", str(img), str(img.with_suffix(".ocr.txt"))]
        cmd += ["ocr", "-m", str(a.model), "--no-segmentation"]
        subprocess.run(cmd, check=True)

        # 3. collect into a table in the same format as gt/*.gt.tsv
        out = a.ocr / f"{page}.ocr.tsv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["file", "kind", "text"])
            for i, r in enumerate(rows):
                txt = (a.lines / r["file"]).with_suffix(".ocr.txt")
                text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
                w.writerow([r["file"], order[i % 2], text.replace("\t", " ")])

        # 4. proofreading page, pre-filled with the model's reading
        subprocess.run([sys.executable, str(HERE / "proof_page.py"), str(a.lines / f"{page}.lines.tsv"),
                        "--text", str(out), "--out", str(HERE / "proof" / f"{page}.html")], check=True)
        print(f"{page}: recognised -> {out}; proofread -> proof/{page}.html")


if __name__ == "__main__":
    main()
