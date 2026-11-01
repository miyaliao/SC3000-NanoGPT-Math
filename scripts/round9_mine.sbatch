#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=round9_mine
#SBATCH --output=round9_mine_%j.out
#SBATCH --error=round9_mine_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
MODE="${1:-}"
RESUME="${2:-}"
if [[ "$MODE" != "pilot" && "$MODE" != "full" ]]; then
  echo "Usage: sbatch scripts/round9_mine.sbatch pilot|full [--resume]" >&2
  exit 2
fi
if [[ -n "$RESUME" && "$RESUME" != "--resume" ]]; then
  echo "Second argument must be --resume" >&2
  exit 2
fi
if [[ "$MODE" == "pilot" && -n "$RESUME" ]]; then
  echo "Pilot does not support resume" >&2
  exit 2
fi
ARGS=(--mine "$MODE" --device cuda --max-new-tokens 80)
if [[ "$MODE" == "full" ]]; then
  # Up to 8000 unique TRAIN prompts per job, save progress every 200.
  ARGS+=(--max-questions-per-job 8000)
fi
if [[ "$RESUME" == "--resume" ]]; then
  ARGS+=(--resume)
fi
python scripts/mine_round9_hardneg.py "${ARGS[@]}"
