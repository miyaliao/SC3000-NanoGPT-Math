#!/usr/bin/env python3
"""Compare Round3 baseline, Round4 replay-only and Round4 targeted continuation.

Read-only. Same fixed in-domain 768, old OOD 256, original Round3 train probe.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dpo/experiments/round3_100k/val_summary.json"
REPLAY=ROOT/"dpo/experiments/round4_replay_only/summary.json"
TREAT=ROOT/"dpo/experiments/round4_treatment/summary.json"
WATCH=("add","subtract","multiply","divide","x_over_a","x_times_a","a_over_x")


def get(path):
    if not path.exists():
        raise SystemExit(f"Missing {path}; wait for both jobs to finish")
    return json.loads(path.read_text(encoding="utf-8"))


def line(label,stage,split,report):
    count,total=report["correct"],report["n"]
    return (f"{label:14} {stage:7} {split:11}: "
            f"{count:3d}/{total:3d} = {count/total:6.2%}")


def main():
    base=get(BASE)
    control=get(REPLAY)
    treatment=get(TREAT)
    for key in ("init_checkpoint_sha256","seed","lr","epochs","batch_size",
                "steps_per_epoch","total_updates","source_manifest_sha256"):
        if control[key]!=treatment[key]:
            raise SystemExit(f"Not matched on {key}: {control[key]} vs {treatment[key]}")
    print("Matched controls verified: same Round3 initialization, epochs, LR, steps, seed")
    print(f"\nRound3 baseline    val: {base['correct']}/{base['n']} = {base['accuracy']:.2%}")
    for name,model in (("Replay-only",control),("Treatment",treatment)):
        for e in model["epoch_history"]:
            for split in ("train_probe","val","old_ood"):
                print(line(name,f"Epoch{e['epoch']}",split,e["evaluations"][split]))
    print("\n=== Fixed in-domain validation: per-category comparison ===")
    b=base["by_category"]
    r=control["epoch_history"][-1]["evaluations"]["val"]["by_category"]
    t=treatment["epoch_history"][-1]["evaluations"]["val"]["by_category"]
    for cat in WATCH:
        def sc(d):return d.get(cat,{}).get("correct",0)
        print(f"{cat:14} baseline={sc(b):2}/64 "
              f"replay={sc(r):2}/64 treatment={sc(t):2}/64 "
              f"treatment-minus-replay={sc(t)-sc(r):+d} questions")
    bc=base["correct"]
    rc=control["epoch_history"][-1]["evaluations"]["val"]["correct"]
    tc=treatment["epoch_history"][-1]["evaluations"]["val"]["correct"]
    print(f"\nVAL net gain vs replay: {tc-rc:+d}/768 ({(tc-rc)/768:+.2%})")
    print(f"VAL net gain vs Round3 baseline: {tc-bc:+d}/768 ({(tc-bc)/768:+.2%})")
    print("Note: additional training vs data-targeting are separable only "
          "through the replay-only control. Do not interpret test-set results.")


if __name__=="__main__":
    main()
