#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=r4_target
#SBATCH --output=r4_target_%j.out
#SBATCH --error=r4_target_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python scripts/train_round4_matched.py --device cuda --arm treatment --epochs 2 --batch-size 64 --lr 2e-5 --data-dir dpo/data/round4_matched_full --out-dir dpo/experiments/round4_treatment
