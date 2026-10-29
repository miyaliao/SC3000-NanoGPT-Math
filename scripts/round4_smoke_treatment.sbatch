#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=r4_smoke
#SBATCH --output=r4_smoke_%j.out
#SBATCH --error=r4_smoke_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python scripts/train_round4_matched.py --device cuda --arm treatment --smoke --data-dir dpo/data/round4_matched_full --out-dir dpo/experiments/round4_smoke_treatment
