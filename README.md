# Handwritten interlinear notes → text

These scripts turn **scanned fieldwork notebooks** into text that can be
checked line by line, and train a handwriting model on your own pages. The
pages are written in the usual interlinear layout:

```
bə̀lò   gó     gə́rám    èŋ  gə̀ŋ  gàkűró        ← line in the language, with tone marks
|homme| avec| femmes| de | lui| trois|          ← gloss line
```

Everything runs on your own computer. You need no account, subscription or
internet connection, and no AI service.

## Why train a model at all

General OCR tools and AI chatbots get the base letters of such notes roughly
right, but they **invent or drop tone marks**. On our test page a chatbot
made errors on 10 % of the characters and on 11 % of the tone marks: `ə→a`,
`ó→ő`, `ā→à`, and even a made-up capital `Ý`. Recent studies of handwriting in
under-resourced languages reach the same conclusion: you need a model trained
on *your* handwriting, using a few hundred to a few thousand transcribed lines
([Lakota archival OCR, 2026](https://arxiv.org/pdf/2610.25440)).

This repository gives you the whole loop:

```
 scan ──► 1 segment_lines.py ──► line images
                                     │
              5 recognize.py ◄───────┤ (once you have a model)
                    │                ▼
                    └──────► 2 proof_page.py ──► you correct in the browser ──► gt/<page>.gt.tsv
                                                                                    │
 4 train_kraken.sh ◄── 3 gt_export.py ◄─────────────────────────────────────────────┘
        │                                         check_text.py: flags errors, measures accuracy
        └──► models/*.mlmodel ──► back to 5
```

Each page you correct is added to `gt/` (the "ground truth"). It improves the
next model and gives the checker more known words. At first you type the
pages yourself on the proofreading page. Once a model exists, it writes the
draft and you only correct it.

## Installation

Python 3.9 or newer.

```
git clone https://github.com/bomolopuu/handwritten-interlinear-htr.git
cd handwritten-interlinear-htr
pip install -r requirements.txt          # numpy + Pillow: enough for steps 1-3
```

### Installing Kraken (for steps 4-5)

[Kraken](https://kraken.re) is a free, open-source handwriting recognition
engine. It runs on **Linux and macOS**. On **Windows**, use WSL: run
`wsl --install` once in PowerShell, open "Ubuntu", and type the commands below
there.

```
python3 -m venv ~/venvs/kraken
~/venvs/kraken/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/venvs/kraken/bin/pip install kraken
~/venvs/kraken/bin/kraken --version
```

The second line installs the CPU-only version of PyTorch (about 200 MB
instead of several GB). On a computer with an NVIDIA GPU, skip that line.

If `venv` complains that `ensurepip is not available` and you have no admin
rights, use `python3 -m venv --without-pip ~/venvs/kraken`, then
`curl -sSO https://bootstrap.pypa.io/get-pip.py && ~/venvs/kraken/bin/python get-pip.py`.

## Step by step

### 1. Cut the page into lines

```
python segment_lines.py scans/page002.jpg --out lines --overlay
```

This writes `lines/page002_01.png`, `_02.png`, … and `lines/page002.overlay.png`
with a box around every line. **Look at the overlay.** Every line of text and
every gloss line should have its own box, and the tone marks should be inside
it.

- If two lines got one box, try `--valley 0.3`.
- If one line got split in two, try `--valley 0.1`.
- If the scan is pale, set the ink threshold by hand, e.g. `--ink 200`.
- Photos taken at an angle should be straightened first; any photo app can
  do this.

### 2. Type or correct the text

```
python proof_page.py lines/page002.lines.tsv
```

Open `proof/page002.html` in a browser. Each line image is shown above a text
field.

- Type the line, or correct the draft if you passed one with `--text`.
- To add a tone, put the cursor right after the letter and click the tone
  button. Change the letter buttons for your language with
  `--letters "əɛɔŋ"`.
- Set each line's type: `text`, `gloss`, or `skip` for crossed-out lines or
  page numbers.
- Click **Save .gt.tsv** and move the downloaded file into `gt/`.

### 3. Build the training set

```
python gt_export.py
```

This copies every checked line into `train/` as an image + `.gt.txt` pair and
holds back every 10th line to measure accuracy. It also lists characters seen
fewer than 5 times; the model cannot learn those yet.

> **Status.** Steps 1–3 and `check_text.py` have been tested on real
> notebook pages. Steps 4–5 call Kraken's documented commands
> (`ketos train`, `kraken … ocr --no-segmentation`), but we have not run
> them end-to-end yet. If something breaks, please open an issue.
> Models are not included: you train your own, on your own handwriting.

### 4. Train a model

Train once you have roughly 300 or more checked lines; more is better, see
below.

```
bash train_kraken.sh                                  # first model
BASE=models/htr_best.mlmodel bash train_kraken.sh     # continue from the last one
```

On a laptop CPU this takes hours; with a GPU (`DEVICE=cuda:0`), minutes.
Kraken prints the validation accuracy after every round. The best model is
saved as `models/htr_best.mlmodel`. The script also runs under SLURM on a
computing cluster (`sbatch train_kraken.sh`).

### 5. Recognise new pages

```
python recognize.py scans/page003.jpg --model models/htr_best.mlmodel
```

This cuts the page into lines, has the model read every line, and opens the
proofreading page already filled in. Correct it and save it into `gt/` (back
to step 2): each corrected page makes the next model better.

### Checking any transcription

`check_text.py` works with or without a model. It reads `.gt.tsv` files,
chatbot-style text (`* Text: … / * Gloss: …`) or plain lines.

```
python check_text.py ocr/page003.ocr.tsv
python check_text.py ocr/page003.ocr.tsv --ref gt/page003.gt.tsv
```

**Without a reference**, it compares the text with all pages in `gt/` and
flags three things:

- characters never seen before, e.g. `ő` when the writer only ever puts a
  double acute on `u`;
- words never seen before, with the closest known words;
- lines where the number of words differs from the number of gloss cells.

**With a reference** (the corrected version of the same page), it prints the
error rate for all characters, for the base letters only and for the tone
marks only, plus the word-by-word differences.

## How much to transcribe

- Every tone mark and special letter has to appear many times. `gt_export.py`
  tells you which ones are still rare.
- A rough guide: a few hundred lines give a first, usable model when you
  fine-tune with `BASE=`. A few thousand are needed for a good model trained
  from scratch. Measure rather than guess: set one checked page aside and run
  `check_text.py --ref` on the model's output for it.
- One model per handwriting works best. If the notebooks come from several
  hands, start with the most frequent one and add the others later.

## Other tools you can use instead

| tool | good for | catch |
|---|---|---|
| [eScriptorium](https://escriptorium-tutorial.readthedocs.io/en/latest/train/) | the same Kraken engine with a shared web interface for a team: segmentation, transcription, training | someone has to host the server (Docker) |
| [Transkribus](https://www.transkribus.org) | hosted service, trains a model on your pages, no installation | credits cost money; scans are uploaded |
| Fine-tuned vision-language model (Qwen2.5-VL-3B / Qwen3-VL-4B + LoRA) | the strongest results in recent low-resource studies, on **line images** like the ones `segment_lines.py` makes ([Lakota](https://arxiv.org/pdf/2610.25440), [AtlasOCR](https://arxiv.org/pdf/2604.08070)) | needs a GPU and some programming; the `train/` folder made here can be reused as training data |
| Tesseract, Windows OCR, AI chatbots | the French/English gloss lines; finding things on a page | not reliable for tone marks without training |

## Files

| file | what it does | needs |
|---|---|---|
| `segment_lines.py` | page → line images, tone marks kept with their line | numpy, Pillow |
| `proof_page.py` | offline HTML page for typing/correcting lines | — |
| `check_text.py` | flags suspicious characters and words; error rates against a reference | — |
| `gt_export.py` | checked pages → Kraken training data | — |
| `train_kraken.sh` | trains or fine-tunes a Kraken model (laptop or SLURM) | Kraken |
| `recognize.py` | scan → lines → Kraken → pre-filled proofreading page | Kraken + a model |
| `gt/` | your checked pages (`*.gt.tsv`); not uploaded by default, see `.gitignore` | |

All text is handled as Unicode. Files are saved in NFC form. For training,
every tone mark is split off as a separate character (NFD), so the model
learns tones separately from letters.

## License

MIT, see `LICENSE`.
