#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=r5_smoke
#SBATCH --output=r5_smoke_%j.out
#SBATCH --error=r5_smoke_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python scripts/train_round5_decomposition.py --device cuda \
  --arm decomposition --smoke \
  --epochs 1 --batch-size 64 --lr 1e-5 \
  --out-dir dpo/experiments/round5_smoke_decomposition
