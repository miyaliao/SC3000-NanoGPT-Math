#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=r8_mine
#SBATCH --output=round8_mine_%j.out
#SBATCH --error=round8_mine_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/mine_round8_train_hard_negatives.py \
  --device cuda --per-category 120 --variants-per-error 2 \
  --max-new-tokens 80 --seed 42
