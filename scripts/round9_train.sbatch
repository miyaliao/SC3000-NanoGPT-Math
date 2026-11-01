#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=round9_train
#SBATCH --output=round9_train_%j.out
#SBATCH --error=round9_train_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
ARM="$1"
case "$ARM" in
 replay|random|structured|hardneg) ;;
 *) echo "Usage: sbatch scripts/round9_train.sbatch replay|random|structured|hardneg" >&2; exit 2 ;;
esac
python scripts/train_round9.py --arm "$ARM" --device cuda \
  --epochs 1 --batch-size 64 --lr 1e-5 --max-new-tokens 80
