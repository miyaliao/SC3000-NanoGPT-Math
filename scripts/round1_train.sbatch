#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=sc3000_round1
#SBATCH --output=round1_%j.out
#SBATCH --error=round1_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# Evaluate on the SAME fixed 256 validation questions before and after training.
python scripts/evaluate_math.py \
    --test dpo/data/round1_full/round1_val.json \
    --checkpoint sft/gpt.pt \
    --out results/round1_val_baseline_256.jsonl \
    --limit 256 --device cuda

# Executes the real Notebook Step 2-7 cells; does not replace the instructor loss.
python scripts/train_round1.py --device cuda

python scripts/evaluate_math.py \
    --test dpo/data/round1_full/round1_val.json \
    --checkpoint dpo/dpo.pt \
    --out results/round1_val_dpo_256.jsonl \
    --limit 256 --device cuda
