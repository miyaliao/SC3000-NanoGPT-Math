#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=nanogpt_e5_viz
#SBATCH --output=nanogpt_e5_viz_%j.out
#SBATCH --error=nanogpt_e5_viz_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/NanoGPT-Math}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python scripts/nanogpt_e5_final_figures.py --repo . --checkpoint dpo/dpo.pt --device cuda
