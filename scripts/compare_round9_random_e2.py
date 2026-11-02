#!/usr/bin/env python3
"""Read-only paired comparison: Round9 Random E1 vs restarted E2.

Reports exact same-question accuracy (including regressions/gains) for all
five fixed splits. Does NOT read the sealed Round3 final test.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from compare_round9 import paired, inspect

ROOT=Path(__file__).resolve().parents[1]
E1=ROOT/"dpo/experiments/round9_random"
E2=ROOT/"dpo/experiments/round9_random_e2_restart"
SPLITS={"train_probe":256,"val":768,"old_ood":256,
        "extended_val":384,"teacher680":680}


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only",action="store_true",
                   help="Read/compare without creating a summary file")
    a=p.parse_args()
    e1=read(E1/"summary.json")
    e2=read(E2/"summary.json")
    if e1.get("arm")!="random" or e1.get("smoke") or \
       e2.get("arm")!="random_e2_restart" or e2.get("smoke"):
        raise RuntimeError("Wrong parent/continuation checkpoint or smoke result")
    if (e2.get("parent_e1_evaluation")!=e1.get("evaluation") or
        e2.get("dataset_sha256")!=e1.get("dataset_sha256") or
        e2.get("round9_manifest_sha256")!=e1.get("round9_manifest_sha256") or
        e2.get("continuous_optimizer_state") is not False):
        raise RuntimeError("Continuation is not paired to the frozen Random E1")
    report={"source_model":"Round9 Random E1",
            "continuation_model":"Round9 Random restarted optimizer E2",
            "initialization":"Random E1 weights; optimizer and LR schedule restarted",
            "comparisons":{}}
    for name,n in SPLITS.items():
        left=read(E1/(name+"_predictions.json"))
        right=read(E2/(name+"_predictions.json"))
        if len(left)!=n or len(right)!=n:
            raise RuntimeError(f"Wrong fixed sample count {name}")
        results=paired(left,right)
        l=inspect(left); r=inspect(right)
        report["comparisons"][name]={
            "e1_correct":l["correct"],"e2_correct":r["correct"],
            "n":n,"paired":results,
            "e1_by_category":l["by_category"],
            "e2_by_category":r["by_category"]
        }
        print(f"{name:13} E1={l['correct']:3}/{n} "
              f"E2={r['correct']:3}/{n} "
              f"delta={results['right_minus_left']:+d} "
              f"E1-only={results['left_only']} E2-only={results['right_only']}",
              flush=True)
    # Most challenging extended subset: direct 401–999 product answers.
    l=read(E1/"extended_val_predictions.json")
    r=read(E2/"extended_val_predictions.json")
    left=[x for x in l if x["category"] in ("multiply","x_over_a")]
    right=[x for x in r if x["category"] in ("multiply","x_over_a")]
    if len(left)!=128 or len(right)!=128:
        raise RuntimeError("Direct multiplication subset should contain 128 questions")
    result=paired(left,right)
    report["comparisons"]["direct128"]=result
    print(f"direct128      E1={inspect(left)['correct']}/128 "
          f"E2={inspect(right)['correct']}/128 "
          f"delta={result['right_minus_left']:+d}",flush=True)
    dest=E2/"paired_vs_random_e1.json"
    if not a.check_only:
        if dest.exists():raise FileExistsError(f"Preserve existing paired audit: {dest}")
        with dest.open("x",encoding="utf-8") as f:
            json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
        print(f"SAVED {dest}",flush=True)


if __name__=="__main__":
    main()
