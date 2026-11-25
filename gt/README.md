Put checked pages here: `<page>.gt.tsv` files saved from the proofreading page.

They are the "ground truth": `gt_export.py` builds training data from them and
`check_text.py` uses them as the list of known words and characters.

Columns: `file` (line image name), `kind` (`text` / `gloss` / `skip`), `text`.
