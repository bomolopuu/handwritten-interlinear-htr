#!/bin/bash
# Train (or continue training) a Kraken handwriting model on train/.
#
#   python gt_export.py                                  # first: build train/ from gt/
#   bash train_kraken.sh                                 # new model, on CPU
#   DEVICE=cuda:0 bash train_kraken.sh                   # on an NVIDIA GPU
#   BASE=models/htr_best.mlmodel bash train_kraken.sh    # continue from an earlier model
#   sbatch train_kraken.sh                               # on a SLURM cluster (edit lines below)
#
# Settings can be given as environment variables:
#   KETOS   path to the ketos program   (default: ketos from your PATH / active venv)
#   DEVICE  cpu or cuda:0               (default: cpu)
#   BASE    an existing .mlmodel to fine-tune (default: none = train from scratch)
#   NAME    output name                 (default: htr -> models/htr_best.mlmodel)
#
#SBATCH --job-name=htr_kraken
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=htr_kraken_%j.log

set -euo pipefail
cd "$(dirname "$0")"
KETOS=${KETOS:-ketos}
DEVICE=${DEVICE:-cpu}
NAME=${NAME:-htr}
mkdir -p models

# ketos wants lists of image paths; gt_export.py wrote bare file names
sed 's#^#train/#' train/split_train.txt > train/_train_paths.txt
sed 's#^#train/#' train/split_val.txt   > train/_val_paths.txt

ARGS=(
  -f path                 # images with .gt.txt files next to them
  -u NFD                  # split tone marks off letters: each tone becomes its own symbol
  --augment               # random distortions of the images: important with little data
  -d "$DEVICE"
  -o "models/$NAME"       # kraken saves models/<NAME>_<n>.mlmodel and models/<NAME>_best.mlmodel
  -t train/_train_paths.txt
  -e train/_val_paths.txt
)
if [ -n "${BASE:-}" ]; then
  ARGS+=(-i "$BASE" --resize union)   # keep the old alphabet and add new characters
fi

"$KETOS" train "${ARGS[@]}"
echo "done. Best model: models/${NAME}_best.mlmodel (validation accuracy is printed above)"
