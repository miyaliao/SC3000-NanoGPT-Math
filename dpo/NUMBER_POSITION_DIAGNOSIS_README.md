# SC3000 Number Position Diagnostic — Gamma 0.50

This read-only diagnosis examines why 18/256 seen training questions but 0/256
held-out validation questions were answered correctly with gamma=0.50.

## Questions

Score the next-character distribution conditioned on the TRUE previous digits
(Teacher Forcing). Report:
- Ones/tens/hundreds/thousands: position from the RIGHT
- First/second/third digit: autoregressive position from the LEFT
- Minus sign as a distinct category, not a digit
- Answer digit count, sign and math problem type
- Two-digit matched-pair comparison of tens and ones on EXACTLY the same prompts
- Gold terminal period (probability and Top-1), and saved greedy outcomes

The result is conditional digit prediction quality, NOT free-generation accuracy.

## Run in TC1

    cd ~/NanoGPT-Math
    git status --short
    git pull --ff-only origin main
    source .venv/bin/activate
    python scripts/diagnose_number_positions.py --self-test-only
    sbatch scripts/diagnose_number_positions.sbatch

Slurm logs: number_pos_JOBID.out and number_pos_JOBID.err

Summary: results/number_position_gamma050/summary.json

Per-token and per-question JSONL are also saved under that directory for each
of train and val. Existing frozen predictions are read when present; the script
does not regenerate answers, train, update weights, or alter original datasets.

It refuses to overwrite an existing output directory.

## How to inspect results

The Slurm output displays overall digit Top-1, rates by place value, a matched
two-digit tens-versus-ones comparison and gold terminal period Top-1 for both
training and validation. More detailed breakdowns by answer length and task
category are in the summary JSON.

Interpret cautiously:
- The two-digit seen train group has about 183 questions; other groups are smaller.
- Teacher Forcing supplies gold earlier digits; it is easier than free generation.
- A difference by place is evidence for a position-specific error, not proof
  that raising the corresponding Loss weight will fix it.
- We use the teacher model's period/newline stopping convention for cached
  greedy predictions, not a claim of access to the TA's hidden grader.
