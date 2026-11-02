#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=R10_elementary
#SBATCH --output=round10_train_%j.out
#SBATCH --error=round10_train_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONNOUSERSITE=1
ARM="${1:?Usage: sbatch scripts/round10_train.sbatch control|targeted [--smoke|--resume]}"
shift
case "$ARM" in
  control|targeted) ;;
  *) echo "Invalid arm $ARM" >&2; exit 2;;
esac
python scripts/train_round10_elementary.py --arm "$ARM" --device cuda "$@"
