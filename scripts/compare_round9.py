#!/usr/bin/env python3
"""Round9 read-only fixed-question/pairwise comparison (R,U,S,H).

Primary: strict answer-first numerical accuracy with original period stop.
Teacher680 is all historical Seen facts; interpret as competency, NOT OOD.
Round3 final Test is deliberately absent.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=("replay","random","structured","hardneg")
SPLITS=("val","old_ood","extended_val","teacher680")
EXPECTED={"val":768,"old_ood":256,"extended_val":384,"teacher680":680}


def read(p):return json.loads(p.read_text(encoding="utf-8"))


def inspect(records):
    c=sum(x["status"]=="correct" for x in records)
    return {"correct":c,"n":len(records),"accuracy":c/len(records),
            "by_category":{k:{"correct":sum(x["status"]=="correct" for x in records
                                         if x["category"]==k),
                              "n":sum(x["category"]==k for x in records)}
                           for k in sorted(set(x["category"] for x in records))}}


def paired(left,right):
    if len(left)!=len(right):raise RuntimeError("Rows not aligned")
    for i,(a,b) in enumerate(zip(left,right)):
        if (a["prompt"],a["answer"],a["category"])!=(b["prompt"],b["answer"],b["category"]):
            raise RuntimeError(f"Paired eval questions differ at {i}")
    lo=sum(a["status"]=="correct" and b["status"]!="correct" for a,b in zip(left,right))
    ro=sum(b["status"]=="correct" and a["status"]!="correct" for a,b in zip(left,right))
    return {"left_only":lo,"right_only":ro,"right_minus_left":ro-lo,
            "right_minus_left_pp":100*(ro-lo)/len(left)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test-only",action="store_true")
    p.add_argument("--require-all",action="store_true")
    args=p.parse_args()
    if args.self_test_only:
        a=[{"prompt":f"{i}+1=?","answer":str(i+1),"category":"add",
            "status":"correct" if i%2 else "math_error"} for i in range(12)]
        b=[dict(x,status="correct" if i%3 else "math_error") for i,x in enumerate(a)]
        r=paired(a,b)
        assert r["right_minus_left"]==sum(x["status"]=="correct" for x in b)-\
                                     sum(x["status"]=="correct" for x in a)
        print("SELFTEST PASS paired alignment / wins")
        return
    outputs={}
    for arm in ARMS:
        base=ROOT/"dpo/experiments"/("round9_"+arm)
        if not (base/"summary.json").is_file():
            if args.require_all:raise FileNotFoundError(base/"summary.json")
            print(f"SKIP {arm}: no completed checkpoint")
            continue
        summary=read(base/"summary.json")
        if summary["arm"]!=arm or summary["smoke"]:
            raise RuntimeError("Refusing smoke or mislabelled result")
        outputs[arm]={}
        for split in SPLITS:
            path=base/f"{split}_predictions.json"
            data=read(path)
            if len(data)!=EXPECTED[split]:raise RuntimeError(f"{arm}/{split}: unexpected size")
            outputs[arm][split]=data
            results=inspect(data)
            print(f"FINAL {arm:12} {split:12} {results['correct']}/{results['n']}")
    if not outputs:raise RuntimeError("No Round9 results exist yet")
    report={"description":"Round9 matched fixed question comparison; 680 is Seen-heavy",
            "scores":{},"matched":{}}
    for arm,sets in outputs.items():
        report["scores"][arm]={k:inspect(v) for k,v in sets.items()}
    for left,right in (("replay","random"),("random","structured"),
                       ("structured","hardneg")):
        if left not in outputs or right not in outputs:continue
        key=f"{left}_vs_{right}"
        report["matched"][key]={}
        for split in SPLITS:
            result=paired(outputs[left][split],outputs[right][split])
            report["matched"][key][split]=result
            print(f"PAIRED {key:24} {split:12} "
                  f"delta={result['right_minus_left']:+d} "
                  f"({left}-only={result['left_only']} {right}-only={result['right_only']})")
        # Primary directly producing 401..999 products: multiply + x_over_a.
        a=[x for x in outputs[left]["extended_val"]
           if x["category"] in ("multiply","x_over_a")]
        b=[x for x in outputs[right]["extended_val"]
           if x["category"] in ("multiply","x_over_a")]
        if len(a)!=128 or len(b)!=128:raise RuntimeError("Direct product subset not 128")
        report["matched"][key]["direct128"]=paired(a,b)
    # Reporting only; never overwrite results or use independent hidden test.
    dest=ROOT/"dpo/experiments/round9_comparison_summary.json"
    if dest.exists():raise FileExistsError(f"Refusing overwrite {dest}")
    with dest.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
    print(f"SAVED {dest}")


if __name__=="__main__":main()
