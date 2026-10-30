#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=round7_train
#SBATCH --output=round7_%x_%j.out
#SBATCH --error=round7_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
ARM="${1:?Usage: sbatch scripts/round7_matched.sbatch replay|targeted}"
case "$ARM" in
  replay|targeted) ;;
  *) echo "Invalid arm: $ARM" >&2; exit 2;;
esac
python scripts/train_round7_matched.py \
  --arm "$ARM" --device cuda --epochs 1 --batch-size 64 \
  --lr 1e-5 --max-new-tokens 80
