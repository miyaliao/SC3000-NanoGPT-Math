#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=r8_train
#SBATCH --output=round8_train_%x_%j.out
#SBATCH --error=round8_train_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
arm="${1:?Usage: sbatch scripts/round8_train.sbatch control|hardneg}"
case "$arm" in
  control|hardneg) ;;
  *) echo "Unknown Round8 arm: $arm" >&2; exit 2;;
esac
python scripts/train_round8_matched.py \
  --arm "$arm" --device cuda --epochs 1 --batch-size 64 \
  --lr 1e-5 --max-new-tokens 80
