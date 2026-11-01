#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=round9_mine
#SBATCH --output=round9_mine_%j.out
#SBATCH --error=round9_mine_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
MODE="$1"
case "$MODE" in
 pilot|full) ;;
 *) echo "Usage: sbatch scripts/round9_mine.sbatch pilot|full" >&2; exit 2 ;;
esac
python scripts/mine_round9_hardneg.py --mine "$MODE" --device cuda --max-new-tokens 80
