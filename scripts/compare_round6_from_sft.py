#!/usr/bin/env python3
"""Round6 four-format epoch curves on ONE fixed primary scoring protocol.

PRIMARY: legacy 'The answer is <integer>' at output beginning, for ALL arms.
It is the historical Round3/4/5 convention, NOT a verified TA scoring rule.
Optional --diagnostics shows separately pre-answer step/answer measurements;
those are NOT eligible for the primary ranking or final model choice.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dpo/experiments/round6_from_sft"
ARMS=("original","answer_only","decomposition_after","decomposition_before")
SPLITS=("val","old_ood","extended_val","train_probe")
DIRECT={"multiply","x_over_a"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def path(arm,epoch,name):
    return BASE/arm/f"epoch{epoch:02d}"/name


def paired(left,right,mode,filter_direct=False):
    if len(left)!=len(right):raise ValueError("Different question counts")
    pairs=Counter()
    for i,(a,b) in enumerate(zip(left,right)):
        for field in ("prompt","answer","category"):
            if a[field]!=b[field]:
                raise ValueError(f"Mismatched paired {field} row={i}")
        if filter_direct and a["category"] not in DIRECT:continue
        aok=a[mode]=="correct"
        bok=b[mode]=="correct"
        case="both" if aok and bok else "left_only" if aok else "right_only" if bok else "neither"
        pairs[case]+=1
    n=sum(pairs.values())
    if not n:raise ValueError("Empty filtered comparison")
    return {
      "n":n,
      "left_correct":pairs["both"]+pairs["left_only"],
      "right_correct":pairs["both"]+pairs["right_only"],
      "right_minus_left":pairs["right_only"]-pairs["left_only"],
      "left_only":pairs["left_only"],"right_only":pairs["right_only"]
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics",action="store_true",
        help="Show secondary exploratory final-answer/step metrics separately; never rank on them")
    args=parser.parse_args()
    manifest=read(ROOT/"dpo/data/round6_from_sft/manifest.json")
    print("Round6 from SFT: 100k examples per arm; decomposition moved ahead of answer on",
          manifest["modified_before_after_rows"],"records")
    summaries={}
    for epoch in range(1,6):
        current={}
        for arm in ARMS:
            file=path(arm,epoch,"epoch_summary.json")
            if file.is_file():
                data=read(file)
                if data["smoke"] or data["epoch"]!=epoch or data["arm"]!=arm:
                    raise ValueError(f"Wrong full-run summary {file}")
                current[arm]=data
        if not current:
            print(f"Epoch {epoch}: not yet run")
            continue
        if len(current)!=len(ARMS):
            print(f"Epoch {epoch}: only {','.join(current)} complete; "
                  "waiting for all arms before paired analysis")
            continue
        for field in ("init_sft_sha256","source_manifest_sha256","loss",
                      "batch_size","training_rows","lr","seed","updates_this_epoch",
                      "total_schedule_updates","max_new_tokens","order_sha256"):
            values={current[arm][field] for arm in ARMS}
            if len(values)!=1:raise ValueError(f"Epoch{epoch} arms not matched: {field}")
        summaries[epoch]=current
        print(f"\n=== EPOCH {epoch}: PRIMARY UNIFORM LEGACY SCORING ===")
        print("All arms: first 'The answer is <integer>' MUST begin the completion.")
        print("This matches our earlier comparison convention; actual TA parser is UNKNOWN.")
        for arm in ARMS:
            data=current[arm]["evaluation"]
            print(f"{arm:22} "
                  f"val={data['val']['strict_correct']:3}/768 "
                  f"extended={data['extended_val']['strict_correct']:3}/384 "
                  f"old_ood={data['old_ood']['strict_correct']:3}/256 "
                  f"train_probe={data['train_probe']['strict_correct']:3}/256")
        for split in SPLITS:
            outputs={arm:read(path(arm,epoch,f"{split}_predictions.json")) for arm in ARMS}
            for tag,left,right in (
                ("H1 answer-only minus original","original","answer_only"),
                ("H2 reasoning-before minus after","decomposition_after","decomposition_before")
            ):
                result=paired(outputs[left],outputs[right],"strict_status")
                print(f"{tag:36} {split:12} PRIMARY: "
                      f"{result['left_correct']:3}->{result['right_correct']:3} "
                      f"delta={result['right_minus_left']:+d} "
                      f"(left-only={result['left_only']}, right-only={result['right_only']})")
                if split=="extended_val":
                    result=paired(outputs[left],outputs[right],"strict_status",filter_direct=True)
                    print(f"{tag:36} DIRECT 128   PRIMARY: "
                          f"{result['left_correct']:3}->{result['right_correct']:3} "
                          f"delta={result['right_minus_left']:+d} "
                          f"(left-only={result['left_only']}, right-only={result['right_only']})")
        if args.diagnostics:
            print("\n--- SECONDARY REASONING DIAGNOSTICS (NOT PRIMARY ACCURACY) ---")
            for arm in ARMS:
                data=current[arm]["evaluation"]["extended_val"]
                print(f"{arm:22} final_answer={data['final_correct']:3}/384 "
                      f"pre_answer_pattern={data['reason_before_answer']:3}/384 "
                      f"internally_valid_steps={data['valid_calculation']:3}/384")
            for split in SPLITS:
                outputs={arm:read(path(arm,epoch,f"{split}_predictions.json")) for arm in ARMS}
                result=paired(outputs["decomposition_after"],outputs["decomposition_before"],"final_status")
                print(f"H2 secondary final-answer  {split:12}: "
                      f"{result['left_correct']}->{result['right_correct']} "
                      f"(delta={result['right_minus_left']:+d}; NOT official grading)")
    print("\nPrimary legacy scoring applied UNCHANGED to ALL four groups.")
    print("The original teacher notebook does not disclose the TA's private answer parser.")
    print("Secondary final-answer-aware reasoning numbers are diagnostics ONLY.")
    print("Round3/4/5 are historical trajectories with different train schedules; "
          "do not attribute a difference from those models solely to output format.")
    print("All 768/256/384 validation splits are fixed. Sealed Test is not read.")


if __name__=="__main__":main()
