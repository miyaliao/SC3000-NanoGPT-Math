#!/usr/bin/env python3
"""Round 3 full 768-case cross-tab audit; read-only and CPU-only.

Uses saved validation predictions and ground-truth metadata. No model training.
"""
from __future__ import annotations
import json
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRED=ROOT/"dpo/experiments/round3_100k/val_predictions.json"
GOLD=ROOT/"dpo/data/round3_structured_full/val.json"
OUT=ROOT/"results/round3_val_cross_audit.json"


def compute():
    predictions=json.loads(PRED.read_text(encoding="utf-8"))
    metadata=json.loads(GOLD.read_text(encoding="utf-8"))
    by_prompt={r["prompt"]:r for r in metadata}
    if len(predictions)!=768 or len(by_prompt)!=768 or {r["prompt"] for r in predictions}!=set(by_prompt):
        raise RuntimeError("Expected the fixed 768-question validation set")
    rows=[]
    for row in predictions:
        gold=by_prompt[row["prompt"]]
        if row["answer"]!=gold["answer"] or row["category"]!=gold["category"]:
            raise RuntimeError(f"Mismatch: {row['prompt']}")
        predicted=row.get("prediction")
        correct=(row["status"]=="correct")
        if correct!=(predicted is not None and int(predicted)==int(gold["answer"])):
            raise RuntimeError(f"Scorer inconsistency: {row['prompt']}")
        digits=len(str(gold["answer"]).lstrip("-"))
        tag=gold["curriculum_tag"]
        rows.append({**row,"digit_count":digits,"tag":tag})
    return rows


def tabulate(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(str(r[k]) for k in keys)].append(r)
    results=[]
    for key,items in sorted(groups.items()):
        n=len(items); c=sum(r["status"]=="correct" for r in items)
        results.append({"group":dict(zip(keys,key)),"n":n,"correct":c,"errors":n-c,
                        "accuracy":c/n})
    return results


def summarize(rows):
    tabs={
        "category_x_digit_count":tabulate(rows,["category","digit_count"]),
        "category_x_curriculum":tabulate(rows,["category","tag"]),
        "curriculum_x_digit_count":tabulate(rows,["tag","digit_count"]),
    }
    print("=== CATEGORY × ANSWER DIGITS ===")
    for r in tabs["category_x_digit_count"]:
        print(f"{r['group']['category']:15s} {r['group']['digit_count']} digits "
              f"{r['correct']:3d}/{r['n']:3d}={r['accuracy']:.1%}")
    print("\n=== LARGER RESULTS: BY CATEGORY ===")
    for r in tabs["category_x_curriculum"]:
        if r["group"]["tag"]=="larger_results":
            print(f"{r['group']['category']:15s} {r['correct']:3d}/{r['n']:3d}={r['accuracy']:.1%}")
    print("\n=== LARGER RESULTS: BY ANSWER DIGITS ===")
    for r in tabs["curriculum_x_digit_count"]:
        if r["group"]["tag"]=="larger_results":
            print(f"{r['group']['digit_count']} digits: {r['correct']}/{r['n']}={r['accuracy']:.1%}")
    print("\n=== HIGH-ERROR STRATA (N>=10, ascending accuracy) ===")
    for r in sorted((r for r in tabs["category_x_digit_count"] if r["n"]>=10),
                    key=lambda t:(t["accuracy"],-t["n"]))[:12]:
        print(f"{r['group']}: {r['correct']}/{r['n']}={r['accuracy']:.1%}")
    return {"n":len(rows),"correct":sum(r["status"]=="correct" for r in rows),
            "tables":tabs,
            "important_caveat":"curriculum_tag is a prioritized label, not an independent factorial category; do not infer causal effects from observational strata."}


def main():
    if OUT.exists():raise FileExistsError(f"Refusing to overwrite {OUT}")
    rows=compute()
    output=summarize(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2);f.write("\n")
    print("\nSaved:",OUT)


if __name__=="__main__":main()
