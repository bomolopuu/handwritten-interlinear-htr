"""Build an offline proofreading page: each line image with an editable text field.

    python proof_page.py lines/page001.lines.tsv
    python proof_page.py lines/page001.lines.tsv --text ocr/page001.ocr.tsv   # pre-filled

Open the resulting proof/<page>.html in any browser (double-click). No internet
or server is needed: the line images are embedded in the file.

* Buttons at the top insert tone marks and special letters at the cursor.
  Put the cursor right AFTER the letter, then press the tone
  (the mark is a separate Unicode character, as typed in Praat).
* Every line has a type: text (the language), gloss, or skip
  (crossed-out lines, page numbers, noise). By default lines alternate
  text / gloss; fix the type where the layout differs.
* "Save" downloads <page>.gt.tsv (columns: file, kind, text; Unicode NFC).
  Put it into gt/ - it is now a checked page ("ground truth"):
  gt_export.py makes training data from it, check_text.py uses it to check.

--text pre-fills the fields with an existing transcription: a .gt.tsv/.ocr.tsv
file, chatbot-style "* Text: / * Gloss:" output, or plain lines.
--letters sets the special-letter buttons for your language.
"""
import argparse
import base64
import csv
import html
import json
import pathlib

# (button label, inserted combining character)
MARKS = [
    ("◌́ high", "́"), ("◌̀ low", "̀"), ("◌̄ mid", "̄"),
    ("◌̂ falling", "̂"), ("◌̌ rising", "̌"), ("◌̋ extra high", "̋"),
    ("◌̏ extra low", "̏"), ("◌̃ nasal", "̃"), ("◌̰ creaky", "̰"),
]
DEFAULT_LETTERS = "əɛɔŋɲɓɗʊɩ"


def read_text(path: pathlib.Path):
    """-> ({file: (kind, text)}, [] ) for tsv input, or ({}, [line, line, ...]) otherwise."""
    if path is None:
        return {}, []
    if path.suffix == ".tsv":
        rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
        return {r["file"]: (r.get("kind", ""), r.get("text", "")) for r in rows}, []
    from check_text import load  # chatbot "* Text: / * Gloss:" or plain lines
    flat = []
    for line, gloss in load(path):
        flat.append(line)
        if gloss is not None:
            flat.append(gloss)
    return {}, flat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines_tsv", type=pathlib.Path, help="<page>.lines.tsv from segment_lines.py")
    ap.add_argument("--text", type=pathlib.Path, help="existing transcription to pre-fill")
    ap.add_argument("--out", type=pathlib.Path, help="output html (default: proof/<page>.html)")
    ap.add_argument("--letters", default=DEFAULT_LETTERS, help="special letters for the buttons")
    ap.add_argument("--first", choices=["text", "gloss"], default="text", help="type of the first line")
    a = ap.parse_args()

    folder = a.lines_tsv.parent
    page = a.lines_tsv.name.replace(".lines.tsv", "")
    rows = list(csv.DictReader(a.lines_tsv.open(encoding="utf-8"), delimiter="\t"))
    by_file, plain = read_text(a.text)
    order = ["text", "gloss"] if a.first == "text" else ["gloss", "text"]

    items = []
    for i, r in enumerate(rows):
        kind, text = by_file.get(r["file"], ("", plain[i] if i < len(plain) else ""))
        kind = {"ngen": "text"}.get(kind, kind)
        items.append({"file": r["file"], "kind": kind or order[i % 2], "text": text,
                      "img": base64.b64encode((folder / r["file"]).read_bytes()).decode()})

    marks = "".join(f'<button type="button" data-ins="{m}">{html.escape(lbl)}</button>' for lbl, m in MARKS)
    letters = "".join(f'<button type="button" data-ins="{html.escape(c)}">{html.escape(c)}</button>'
                      for c in a.letters)
    out = a.out or folder.parent / "proof" / f"{page}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__PAGE__", html.escape(page))
                   .replace("__MARKS__", marks).replace("__LETTERS__", letters)
                   .replace("__DATA__", json.dumps(items, ensure_ascii=False)), encoding="utf-8")
    print(f"proofreading page: {out}")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Proofread __PAGE__</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#d9d8d0;--text:#fff;--gloss:#f2f1ea;--acc:#2a5db0}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--fg:#ecebe4;--mut:#a3a29b;--line:#3a3935;--text:#252522;--gloss:#2d2d29;--acc:#8fb3ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
header h1{font-size:17px;margin:0 0 6px}
.pal button{font:15px "Charis SIL","Doulos SIL","Gentium Plus",serif;margin:2px;padding:3px 8px;border:1px solid var(--line);background:var(--text);color:var(--fg);border-radius:4px;cursor:pointer}
.pal button:hover{border-color:var(--acc)}
main{padding:12px 16px;max-width:1500px}
.row{border:1px solid var(--line);border-radius:6px;margin:0 0 10px;padding:8px}
.row.text{background:var(--text)}.row.gloss{background:var(--gloss)}.row.skip{opacity:.5}
.row img{max-width:100%;display:block;background:#fff}
.row .meta{font-size:12px;color:var(--mut);display:flex;gap:10px;align-items:center;margin:4px 0}
.row input[type=text]{width:100%;box-sizing:border-box;font:20px "Charis SIL","Doulos SIL","Gentium Plus",serif;padding:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:4px}
.row.changed input[type=text]{border-color:var(--acc)}
#save{font-size:15px;padding:6px 14px;background:var(--acc);color:#fff;border:0;border-radius:4px;cursor:pointer;margin-left:8px}
.hint{font-size:12px;color:var(--mut)}
</style></head><body>
<header><h1>Proofread: __PAGE__ <button id="save">Save .gt.tsv</button></h1>
<div class="pal">__MARKS__ &nbsp; __LETTERS__</div>
<div class="hint">Put the cursor right after a letter, then click a tone. Changed lines get a blue frame. Save downloads the file; put it into the gt/ folder.</div></header>
<main id="m"></main>
<script>
const DATA=__DATA__, PAGE="__PAGE__";
const m=document.getElementById('m'); let last=null;
DATA.forEach((d,i)=>{
  const r=document.createElement('div'); r.className='row '+d.kind;
  r.innerHTML=`<img alt="line ${i+1}"><div class="meta"><span>${i+1}. ${d.file}</span>
  <select><option value="text">text</option><option value="gloss">gloss</option><option value="skip">skip</option></select></div>
  <input type="text" spellcheck="false">`;
  r.querySelector('img').src='data:image/png;base64,'+d.img;
  const s=r.querySelector('select'); s.value=d.kind; s.onchange=()=>{d.kind=s.value; r.className='row '+d.kind;};
  const t=r.querySelector('input'); t.value=d.text; const orig=d.text;
  t.oninput=()=>{d.text=t.value; r.classList.toggle('changed', t.value!==orig);};
  t.onfocus=()=>last=t;
  m.appendChild(r);
});
// mousedown + preventDefault keeps the cursor in the text field
document.querySelectorAll('.pal button').forEach(b=>b.onmousedown=e=>{
  e.preventDefault(); if(!last) return;
  const p=last.selectionStart, v=last.value;
  last.value=v.slice(0,p)+b.dataset.ins+v.slice(last.selectionEnd);
  last.selectionStart=last.selectionEnd=p+b.dataset.ins.length; last.dispatchEvent(new Event('input'));
});
window.addEventListener('beforeunload',e=>{ if(document.querySelector('.row.changed')) e.preventDefault(); });
document.getElementById('save').onclick=()=>{
  const lines=['file\tkind\ttext'].concat(DATA.map(d=>[d.file,d.kind,d.text.normalize('NFC').replace(/\t/g,' ')].join('\t')));
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/tab-separated-values'}));
  a.download=PAGE+'.gt.tsv'; a.click();
  document.querySelectorAll('.row.changed').forEach(r=>r.classList.remove('changed'));
};
</script></body></html>"""

if __name__ == "__main__":
    main()
