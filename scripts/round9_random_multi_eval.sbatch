#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=R9_random_E3_E6_eval
#SBATCH --output=round9_random_multi_eval_%j.out
#SBATCH --error=round9_random_multi_eval_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/eval_round9_random_multi.py --device cuda
