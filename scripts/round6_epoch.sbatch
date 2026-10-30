#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=round6_epoch
#SBATCH --output=round6_%x_%j.out
#SBATCH --error=round6_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
arm="${1:?Usage: sbatch scripts/round6_epoch.sbatch ARM EPOCH}"
epoch="${2:?Need epoch 1..5}"
python scripts/train_round6_from_sft.py --arm "${arm}" --epoch "${epoch}" \
  --device cuda --lr 1e-4 --batch-size 64 --max-new-tokens 160
