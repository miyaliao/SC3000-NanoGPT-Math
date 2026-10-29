#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=dpo_smoke
#SBATCH --output=dpo_smoke_%j.out
#SBATCH --error=dpo_smoke_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/NanoGPT-Math}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/pipeline_smoke.py --device cuda --samples 48 --batch-size 16 \
  --out-dir "dpo/smoke_outputs/job_${SLURM_JOB_ID}"
