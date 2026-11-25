"""Cut a scanned notebook page into line images (no neural network needed).

    python segment_lines.py page.png [page2.jpg ...] --out lines/

For every page it writes:
    lines/<page>_01.png, _02.png, ...   one image per handwritten line
    lines/<page>.lines.tsv              file name + box (x0 y0 x1 y1) on the scan

How it works
------------
1. The page is turned to grey; pixels darker than a threshold count as ink
   (the threshold is chosen automatically with Otsu's method).
2. For every pixel row we count the ink -> a "profile" of the page.
3. Empty rows separate lines. Where two lines touch (the gloss line's "|"
   strokes often reach up into the text line), we cut at the deepest dip
   of the profile instead.
4. Tone marks sit in a thin, faint strip ABOVE the letters. Such a strip is
   glued to the line below it, and every line gets extra margin on top
   (--pad-top). Losing tone marks here would make them impossible to learn.

If lines stick together or vanish on a pale scan, change --ink (lower = only
darker pixels count) or --valley. Check the result with --overlay.
"""
import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def otsu(gray: np.ndarray) -> int:
    """Classic Otsu threshold: the grey level that best separates ink from paper."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = (np.arange(256) * hist).sum()
    w0 = s0 = 0.0
    best, best_t = -1.0, 128
    for t in range(256):
        w0 += hist[t]
        if w0 == 0 or w0 == total:
            continue
        s0 += t * hist[t]
        m0, m1 = s0 / w0, (sum_all - s0) / (total - w0)
        between = w0 * (total - w0) * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, t
    return best_t


def find_bands(ink: np.ndarray, min_px: int) -> list[list[int]]:
    """Runs of pixel rows that contain more than `min_px` ink pixels."""
    rows = ink.sum(axis=1) > min_px
    bands, start = [], None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, len(rows)])
    return bands


def merge_gaps(bands, gap_merge: int):
    """Join bands separated by a gap of only a few pixels (a break inside one line)."""
    if not bands:
        return []
    merged = [bands[0][:]]
    for b in bands[1:]:
        if b[0] - merged[-1][1] <= gap_merge:
            merged[-1][1] = b[1]
        else:
            merged.append(b[:])
    return merged


def split_at_valleys(bands, profile: np.ndarray, valley_frac: float, min_piece: int):
    """Split a band where the (smoothed) ink profile drops below
    `valley_frac` x its peak. This separates a text line from the gloss line
    under it when the two are connected by "|" strokes."""
    smooth = np.convolve(profile, np.ones(5) / 5, mode="same")
    out = []
    for y0, y1 in bands:
        seg = smooth[y0:y1]
        if len(seg) < 2 * min_piece:
            out.append([y0, y1])
            continue
        low = seg < valley_frac * seg.max()
        cuts, y = [], 0
        while y < len(seg):
            if low[y]:
                z = y
                while z < len(seg) and low[z]:
                    z += 1
                if y > 0 and z < len(seg):  # a dip inside the band, not at its edge
                    cuts.append(y + int(np.argmin(seg[y:z])))
                y = z
            else:
                y += 1
        edges = [0] + [c for c in cuts if min_piece // 2 <= c <= len(seg) - min_piece // 2] + [len(seg)]
        for a, b in zip(edges, edges[1:]):
            out.append([y0 + a, y0 + b])
    return out


def merge_weak(bands, profile: np.ndarray, weak_frac: float, touch: int = 4):
    """A strip of tone marks has much less ink than the letters under it.
    If a band's peak is below `weak_frac` x the peak of the band directly
    below, glue it to that band. A weak band at the very bottom (an
    underline, a one-word note) is glued to the band above."""
    bands = [b[:] for b in bands]
    peak = lambda b: float(profile[b[0]:b[1]].max()) if b[1] > b[0] else 0.0
    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, b in enumerate(bands):
            nxt = bands[i + 1] if i + 1 < len(bands) else None
            prv = bands[i - 1] if i > 0 else None
            if nxt and nxt[0] - b[1] <= touch and peak(b) < weak_frac * peak(nxt):
                nxt[0] = b[0]
            elif nxt is None and prv and b[0] - prv[1] <= touch and peak(b) < weak_frac * peak(prv):
                prv[1] = b[1]
            else:
                continue
            del bands[i]
            changed = True
            break
    return bands


def segment(path: pathlib.Path, out_dir: pathlib.Path, ink_thr=None, pad_top=12, pad_bottom=6,
            gap_merge=3, weak_frac=0.5, min_px=3, valley_frac=0.2, overlay=False):
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")  # respect phone rotation
    gray = np.asarray(img)
    thr = ink_thr if ink_thr is not None else min(otsu(gray), 200)
    ink = gray < thr
    # ignore 1 % of the width at each side: scanner frame, page shadow
    edge = max(2, gray.shape[1] // 100)
    ink[:, :edge] = False
    ink[:, -edge:] = False
    # rows inked across most of the width are ruled lines or the frame, not text
    ink[ink.mean(axis=1) > 0.6] = False
    profile = ink.sum(axis=1).astype(float)

    bands = merge_gaps(find_bands(ink, min_px), gap_merge)
    bands = split_at_valleys(bands, profile, valley_frac, min_piece=8)
    bands = merge_weak(bands, profile, weak_frac)

    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = gray.shape
    rows = []
    for i, (y0, y1) in enumerate(bands, 1):
        cols = np.where(ink[y0:y1].any(axis=0))[0]
        x0, x1 = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, W)
        box = (max(0, x0 - 10), max(0, y0 - pad_top), min(W, x1 + 10), min(H, y1 + pad_bottom))
        name = f"{path.stem}_{i:02d}.png"
        img.crop(box).save(out_dir / name)
        rows.append((name, *box))
    with (out_dir / f"{path.stem}.lines.tsv").open("w", encoding="utf-8") as f:
        f.write("file\tx0\ty0\tx1\ty1\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

    if overlay:  # red/blue boxes drawn on the page, to check the cut by eye
        vis = img.convert("RGB")
        d = ImageDraw.Draw(vis)
        for k, (_, X0, Y0, X1, Y1) in enumerate(rows):
            d.rectangle([X0 + k % 2 * 4, Y0 + k % 2 * 2, X1 - k % 2 * 4, Y1 - k % 2 * 2],
                        outline=[(220, 0, 0), (0, 110, 220)][k % 2], width=2)
        vis.save(out_dir / f"{path.stem}.overlay.png")
    print(f"{path.name}: ink threshold {thr}, {len(rows)} lines -> {out_dir}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", type=pathlib.Path, help="scans: png / jpg / tif")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("lines"), help="output folder")
    ap.add_argument("--ink", type=int, default=None, help="ink threshold 0-255 (default: automatic)")
    ap.add_argument("--pad-top", type=int, default=12, help="extra pixels above each line, for tone marks")
    ap.add_argument("--pad-bottom", type=int, default=6, help="extra pixels below each line")
    ap.add_argument("--gap-merge", type=int, default=3, help="gaps up to this many px stay inside one line")
    ap.add_argument("--weak", type=float, default=0.5,
                    help="a band with less than this share of the ink of the band below = tone marks, glue down")
    ap.add_argument("--valley", type=float, default=0.2,
                    help="cut touching lines where ink falls below this share of the peak")
    ap.add_argument("--min-px", type=int, default=3, help="rows with fewer ink pixels count as empty")
    ap.add_argument("--overlay", action="store_true", help="also save <page>.overlay.png with the boxes")
    a = ap.parse_args()
    for p in a.pages:
        segment(p, a.out, a.ink, a.pad_top, a.pad_bottom, a.gap_merge, a.weak, a.min_px, a.valley, a.overlay)


if __name__ == "__main__":
    main()
