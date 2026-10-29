#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=sc3000_round1c
#SBATCH --output=round1c_%j.out
#SBATCH --error=round1c_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# Confirm frozen source and exact labels before allocating training time.
python scripts/generate_round1c.py --verify-only

# Execute existing course Notebook Steps 2-7; saves only the isolated 1C checkpoint.
python scripts/train_round1c.py --device cuda

# Same N=256 integer validation prompts as 1A/1B, original '.'/newline stop tokens.
python scripts/evaluate_ta_integer.py \
    --test dpo/experiments/round1b_integer/common_integer_val_256.json \
    --checkpoint dpo/experiments/round1c_incorrect/dpo.pt \
    --out results/round1c_common_round1c.jsonl --device cuda
