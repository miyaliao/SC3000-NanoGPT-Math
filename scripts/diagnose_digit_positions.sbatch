#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=digitpos256
#SBATCH --output=digitpos_%j.out
#SBATCH --error=digitpos_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/diagnose_digit_positions.py --device cuda