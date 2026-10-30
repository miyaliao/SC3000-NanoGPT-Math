#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=teacher680_eval
#SBATCH --output=teacher680_%j.out
#SBATCH --error=teacher680_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/evaluate_teacher_style_680.py \
  --device cuda --max-new-tokens 80 --models round5_A,round7_replay
