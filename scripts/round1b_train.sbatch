#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=sc3000_round1b
#SBATCH --output=round1b_%j.out
#SBATCH --error=round1b_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONNOUSERSITE=1

# Integer-only fixed validation under the original '.' / newline stopping rule.
python scripts/evaluate_ta_integer.py \
  --test dpo/experiments/round1b_integer/common_integer_val_256.json \
  --checkpoint sft/gpt.pt \
  --out results/round1b_common_baseline.jsonl --device cuda

python scripts/evaluate_ta_integer.py \
  --test dpo/experiments/round1b_integer/common_integer_val_256.json \
  --checkpoint dpo/dpo.pt \
  --out results/round1b_common_round1a.jsonl --device cuda

# Execute the real instructor Notebook cells on the isolated integer data.
python scripts/train_round1b.py --device cuda

python scripts/evaluate_ta_integer.py \
  --test dpo/experiments/round1b_integer/common_integer_val_256.json \
  --checkpoint dpo/experiments/round1b_integer/dpo.pt \
  --out results/round1b_common_round1b.jsonl --device cuda
