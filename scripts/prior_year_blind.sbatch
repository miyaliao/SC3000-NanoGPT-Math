#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=prior_blind
#SBATCH --output=prior_blind_%j.out
#SBATCH --error=prior_blind_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python scripts/benchmark_prior_year_blind.py \
  --device cuda --decode both --max-new-tokens 200 --seed 42 \
  --out-dir dpo/experiments/prior_year_blind_2025
