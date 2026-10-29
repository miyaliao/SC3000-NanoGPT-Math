#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=term_025
#SBATCH --output=term_025_%j.out
#SBATCH --error=term_025_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# Same Round 1C sampled 256 problems, original pretrained initialization.
# Independent checkpoint + summary, no overwriting gamma=1.0.
python scripts/overfit256_termination_aware.py --device cuda \
  --seed 42 --epochs 128 --batch-size 64 \
  --answer-weight 1.0 --completion-weight 0.25 \
  --end-weight 0.25 \
  --out-dir dpo/experiments/overfit256_termination_gamma025 \
  --summary-path results/overfit256_termination_gamma025_summary.json
