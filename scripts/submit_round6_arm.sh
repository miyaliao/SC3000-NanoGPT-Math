#!/bin/bash
# Submit five dependency-linked jobs for ONE Round6 arm, total 5 epochs.
# IMPORTANT: inspect GPU smoke first; do not run this script until it passes.
set -euo pipefail
arm="${1:?Usage: bash scripts/submit_round6_arm.sh original|answer_only|decomposition_after|decomposition_before}"
case "$arm" in
  original|answer_only|decomposition_after|decomposition_before) ;;
  *) echo "Unknown Round6 arm: $arm" >&2; exit 2;;
esac
cd "$(dirname "$0")/.."
test -f dpo/data/round6_from_sft/manifest.json || {
 echo "Generate Round6 data first" >&2; exit 2;
}
previous=""
for epoch in 1 2 3 4 5; do
  if [[ -n "$previous" ]]; then
    job=$(sbatch --parsable --dependency="afterok:$previous" \
      --job-name="r6_${arm}_e${epoch}" scripts/round6_epoch.sbatch "$arm" "$epoch")
  else
    job=$(sbatch --parsable --job-name="r6_${arm}_e${epoch}" \
      scripts/round6_epoch.sbatch "$arm" "$epoch")
  fi
  previous="${job%%;*}"
  echo "$arm epoch $epoch -> job $previous"
done
echo "Each next epoch starts only after the previous epoch finishes successfully."
echo "Check: squeue -u \"$USER\""
