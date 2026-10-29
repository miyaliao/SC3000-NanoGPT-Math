#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=round3_100k
#SBATCH --output=round3_100k_%j.out
#SBATCH --error=round3_100k_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

python scripts/train_round3_course.py --device cuda --epochs 5 --batch-size 64 \
  --out-dir dpo/experiments/round3_100k
