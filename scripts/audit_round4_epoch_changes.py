#!/usr/bin/env python3
"""Read-only paired error analysis for Round4 matched continued training.

Run after BOTH Round4 jobs finish:
    python scripts/audit_round4_epoch_changes.py
    python scripts/audit_round4_epoch_changes.py --save

It compares exactly the SAME prompts between replay-only and targeted
treatment at epochs 1/2, plus treatment's epoch-1-to-2 changes. It never
loads a model, trains, or accesses the sealed Round3 test split.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=ROOT/"dpo/experiments"
ARMS={"replay":"round4_replay_only","treatment":"round4_treatment"}
SPLITS={"val":768,"old_ood":256,"extended_val":384,"train_probe":256}
OUTPUT=ROOT/"results/round4_matched_epoch_audit.json"


def read_predictions(arm, epoch, split):
    file=(EXPERIMENTS/ARMS[arm]/f"eval_epoch{epoch}"/f"{split}_predictions.json")
    if not file.is_file():
        raise FileNotFoundError(f"Missing completed evaluation: {file}")
    rows=json.loads(file.read_text(encoding="utf-8"))
    if len(rows)!=SPLITS[split]:
        raise ValueError(f"Unexpected question count {file}: {len(rows)}")
    mapping={r["prompt"]:r for r in rows}
    if len(mapping)!=len(rows):
        raise ValueError(f"Duplicate prompts in {file}")
    for r in rows:
        if str(r.get("answer")) not in ("", "None") and r["status"]=="correct":
            if r.get("prediction") is None or int(r["prediction"])!=int(r["answer"]):
                raise ValueError(f"Incorrect 'correct' label: {file} {r['prompt']}")
    return mapping


def compare(a,b):
    if set(a)!=set(b):
        raise ValueError("The two evaluations did not use identical prompts")
    counts=Counter()
    by_category=defaultdict(Counter)
    by_digits=defaultdict(Counter)
    examples={"a_only":[],"b_only":[]}
    for prompt in sorted(a):
        x,y=a[prompt],b[prompt]
        if str(x["answer"])!=str(y["answer"]) or x["category"]!=y["category"]:
            raise ValueError(f"Different truth labels/category for {prompt}")
        xok=x["status"]=="correct"
        yok=y["status"]=="correct"
        key="both" if xok and yok else "a_only" if xok else "b_only" if yok else "neither"
        counts[key]+=1
        by_category[x["category"]][key]+=1
        by_digits[len(str(x["answer"]).lstrip("-"))][key]+=1
        if key in examples and len(examples[key])<12:
            examples[key].append({
                "prompt":prompt,"gold":x["answer"],
                "a_prediction":x.get("prediction"),"b_prediction":y.get("prediction"),
                "a_completion":x.get("completion"),"b_completion":y.get("completion"),
            })
    n=len(a)
    return {
        "n":n,
        "a_correct":counts["both"]+counts["a_only"],
        "b_correct":counts["both"]+counts["b_only"],
        "a_minus_b_correct":counts["a_only"]-counts["b_only"],
        "a_minus_b_percentage_points":100*(counts["a_only"]-counts["b_only"])/n,
        "paired_outcomes":dict(counts),
        "by_category":{cat:{"n":sum(ct.values()),
                            "a_correct":ct["both"]+ct["a_only"],
                            "b_correct":ct["both"]+ct["b_only"],
                            "a_only":ct["a_only"],"b_only":ct["b_only"]}
                       for cat,ct in sorted(by_category.items())},
        "by_digit_count":{str(d):{"n":sum(ct.values()),
                                   "a_correct":ct["both"]+ct["a_only"],
                                   "b_correct":ct["both"]+ct["b_only"],
                                   "a_only":ct["a_only"],"b_only":ct["b_only"]}
                          for d,ct in sorted(by_digits.items())},
        "examples":examples,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save",action="store_true",
                   help="Write a new, non-overwriting JSON report to results/")
    args=p.parse_args()
    cache={}
    for arm in ARMS:
        for epoch in (1,2):
            for split in SPLITS:
                cache[arm,epoch,split]=read_predictions(arm,epoch,split)

    result={"comparing":"a=treatment, b=replay or prior treatment epoch",
            "same_prompts_required":True,
            "treatment_minus_replay":{},
            "treatment_epoch2_minus_epoch1":{},
            "replay_epoch2_minus_epoch1":{}}
    print("=== SAME-EPOCH TREATMENT (A) vs REPLAY-ONLY (B) ===")
    for epoch in (1,2):
        for split in SPLITS:
            ct=compare(cache["treatment",epoch,split],
                       cache["replay",epoch,split])
            result["treatment_minus_replay"][f"epoch{epoch}_{split}"]=ct
            print(f"epoch={epoch} {split:12} "
                  f"treatment={ct['a_correct']}/{ct['n']} "
                  f"replay={ct['b_correct']}/{ct['n']} "
                  f"delta={ct['a_minus_b_correct']:+d} "
                  f"treatment-only={ct['paired_outcomes'].get('a_only',0)} "
                  f"replay-only={ct['paired_outcomes'].get('b_only',0)}")
    print("\n=== EPOCH 2 (A) vs EPOCH 1 (B), SAME ARM ===")
    for arm in ARMS:
        for split in SPLITS:
            ct=compare(cache[arm,2,split],cache[arm,1,split])
            result[f"{arm}_epoch2_minus_epoch1"][split]=ct
            print(f"{arm:10} {split:12} "
                  f"epoch2={ct['a_correct']}/{ct['n']} "
                  f"epoch1={ct['b_correct']}/{ct['n']} "
                  f"delta={ct['a_minus_b_correct']:+d} "
                  f"newly-correct={ct['paired_outcomes'].get('a_only',0)} "
                  f"regressed={ct['paired_outcomes'].get('b_only',0)}")
    print("\n=== EXTENDED-VAL CATEGORY DIFFERENCES (TREATMENT EPOCH 2 vs REPLAY EPOCH 2) ===")
    cat=result["treatment_minus_replay"]["epoch2_extended_val"]["by_category"]
    for name,row in cat.items():
        print(f"{name:15} {row['a_correct']:2}/{row['n']} treatment "
              f"vs {row['b_correct']:2}/{row['n']} replay; "
              f"delta={row['a_correct']-row['b_correct']:+d}")

    print("\n=== TREATMENT EXTENDED-VAL CHANGES (EPOCH 2 vs EPOCH 1) ===")
    cat=result["treatment_epoch2_minus_epoch1"]["extended_val"]["by_category"]
    for name,row in cat.items():
        print(f"{name:15} {row['a_correct']:2}/{row['n']} E2 "
              f"vs {row['b_correct']:2}/{row['n']} E1; "
              f"delta={row['a_correct']-row['b_correct']:+d}")
    print("\nNote: questions can share underlying math facts; paired per-question "
          "counts are descriptive, not independent-sample significance tests.")
    if args.save:
        if OUTPUT.exists():
            raise FileExistsError(f"Refusing to overwrite previous audit: {OUTPUT}")
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        with OUTPUT.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
            f.write("\n")
        print("Saved:",OUTPUT)


if __name__=="__main__":
    main()
