#!/bin/bash
# Submit five dependency-linked jobs for ONE Round6 arm, total 5 epochs.
# IMPORTANT: inspect GPU smoke first; do not run this script until it passes.
set -euo pipefail
arm="${1:?Usage: bash scripts/submit_round6_arm.sh ARM [START_EPOCH=1]}"
start="${2:-1}"
case "$arm" in
  original|answer_only|decomposition_after|decomposition_before) ;;
  *) echo "Unknown Round6 arm: $arm" >&2; exit 2;;
esac
cd "$(dirname "$0")/.."
case "$start" in 1|2|3|4|5) ;; *) echo "Start epoch must be 1..5" >&2; exit 2;; esac
if [[ "$start" -gt 1 ]]; then
  prior=$((start-1))
  previous_summary=$(printf "dpo/experiments/round6_from_sft/%s/epoch%02d/epoch_summary.json" "$arm" "$prior")
  test -f "$previous_summary" || {
    echo "Cannot resume: completed epoch summary missing: $previous_summary" >&2
    exit 2
  }
fi
test -f dpo/data/round6_from_sft/manifest.json || {
 echo "Generate Round6 data first" >&2; exit 2;
}
previous=""
for epoch in 1 2 3 4 5; do
  [[ "$epoch" -ge "$start" ]] || continue
  epoch_folder=$(printf "dpo/experiments/round6_from_sft/%s/epoch%02d" "$arm" "$epoch")
  if [[ -e "$epoch_folder" ]]; then
    echo "Refusing to resubmit existing epoch directory: $epoch_folder" >&2
    exit 2
  fi
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
