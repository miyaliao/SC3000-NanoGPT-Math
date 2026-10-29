#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=number_pos
#SBATCH --output=number_pos_%j.out
#SBATCH --error=number_pos_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# Read-only: one frozen gamma=0.5 checkpoint; no gradient/update.
python scripts/diagnose_number_positions.py \
  --device cuda \
  --checkpoint dpo/experiments/overfit256_termination_gamma050/dpo.pt \
  --out-dir results/number_position_gamma050 \
  --batch-size 32
