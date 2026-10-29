#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=term256
#SBATCH --output=term256_%j.out
#SBATCH --error=term256_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/overfit256_termination_aware.py --device cuda \
  --epochs 128 --batch-size 64 --answer-weight 1.0 \
  --completion-weight 0.25 --end-weight 1.0
