#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=round6_smoke
#SBATCH --output=round6_smoke_%j.out
#SBATCH --error=round6_smoke_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/evaluate_round6_formats.py
python scripts/train_round6_from_sft.py \
  --arm decomposition_before --epoch 1 --smoke \
  --device cuda --lr 1e-4 --batch-size 64 --max-new-tokens 160
