#!/usr/bin/env python3
"""Read-only Round4 extended multiplication-error audit; CPU, no training.

Compares treatment epoch 1/2 and replay epoch 2 on the exact frozen 384
extended-val records. Focus: multiplicand lengths, product ranges, error
magnitudes, and the two classes predicting a 401..999 answer directly.

Usage:
    python scripts/audit_round4_large_products.py
    python scripts/audit_round4_large_products.py --save
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation

VAL=ROOT/"dpo/data/round4_matched_full/extension_val.json"
PRED={
    "treatment_e1": ROOT/"dpo/experiments/round4_treatment/eval_epoch1/extended_val_predictions.json",
    "treatment_e2": ROOT/"dpo/experiments/round4_treatment/eval_epoch2/extended_val_predictions.json",
    "replay_e2": ROOT/"dpo/experiments/round4_replay_only/eval_epoch2/extended_val_predictions.json",
}
OUT=ROOT/"results/round4_large_product_audit.json"
DIRECT={"multiply","x_over_a"}


def load():
    records=json.loads(VAL.read_text(encoding="utf-8"))
    if len(records)!=384:
        raise ValueError("Expected exactly 384 frozen extended validation rows")
    predictions={}
    for name,path in PRED.items():
        p=json.loads(path.read_text(encoding="utf-8"))
        if len(p)!=len(records):
            raise ValueError(f"Mismatched prediction count: {path}")
        for i,(gold,pred) in enumerate(zip(records,p)):
            if (gold["prompt"]!=pred["prompt"] or
                str(gold["answer"])!=str(pred["answer"]) or
                gold["category"]!=pred["category"]):
                raise ValueError(f"Ground truth misaligned at row {i} in {path}")
            key=parse_relation(gold["prompt"],str(gold["answer"]))
            if tuple(gold["fact_key"])!=key:
                raise ValueError(f"Invalid canonical mathematical fact in val row {i}")
            if key[0]!="mul" or not 401<=int(key[3])<=999:
                raise ValueError(f"Unexpected extended math fact: {key}")
            correct=pred["status"]=="correct"
            if correct!=(pred.get("prediction") is not None and
                        int(pred["prediction"])==int(gold["answer"])):
                raise ValueError(f"Incorrect score status for {gold['prompt']} in {path}")
        predictions[name]=p
    return records,predictions


def summarize(items):
    n=len(items)
    if n==0:return {"n":0}
    correct=sum(r["correct"] for r in items)
    wrong=[r for r in items if not r["correct"]]
    wrong_numeric=[r for r in wrong if r["prediction"] is not None]
    return {
        "n":n,"correct":correct,"accuracy":correct/n,
        "incorrect":len(wrong),
        "numeric_error_count":len(wrong_numeric),
        "format_failure_count":len(wrong)-len(wrong_numeric),
        "wrong_with_correct_digit_length":sum(
            len(str(r["prediction"]).lstrip("-"))==len(str(r["answer"]).lstrip("-"))
            for r in wrong_numeric),
        "wrong_with_abs_error_at_most_10":sum(
            abs(int(r["prediction"])-int(r["answer"]))<=10
            for r in wrong_numeric),
        "wrong_with_abs_error_at_most_50":sum(
            abs(int(r["prediction"])-int(r["answer"]))<=50
            for r in wrong_numeric),
    }


def group(rows,key):
    sections=defaultdict(list)
    for r in rows:sections[str(key(r))].append(r)
    return {k:summarize(v) for k,v in sorted(sections.items())}


def compute():
    records,preds=load()
    results={}
    for name,rows in preds.items():
        dataset=[]
        for gold,p in zip(records,rows):
            key=tuple(gold["fact_key"])
            a,b=int(key[1]),int(key[2])
            product=int(key[3])
            bucket="401-599" if product<600 else "600-799" if product<800 else "800-999"
            pattern="1x2" if min(a,b)<10 and max(a,b)>=10 else ("2x2" if min(a,b)>=10 else "1x1")
            dataset.append({
                "prompt":gold["prompt"],"category":gold["category"],
                "answer":int(gold["answer"]),"prediction":p.get("prediction"),
                "completion":p.get("completion"),
                "correct":p["status"]=="correct",
                "product":product,"product_band":bucket,"factor_pattern":pattern,
                "direct_product_answer":gold["category"] in DIRECT,
            })
        direct=[r for r in dataset if r["direct_product_answer"]]
        inverse=[r for r in dataset if not r["direct_product_answer"]]
        results[name]={
            "overall":summarize(dataset),
            "direct_product":summarize(direct),
            "factor_retrieval":summarize(inverse),
            "by_category":group(dataset,lambda r:r["category"]),
            "direct_by_product_band":group(direct,lambda r:r["product_band"]),
            "direct_by_factor_pattern":group(direct,lambda r:r["factor_pattern"]),
            "factor_retrieval_by_product_band":group(inverse,lambda r:r["product_band"]),
            "direct_error_examples":[
                {"prompt":r["prompt"],"gold":r["answer"],
                 "pred":r["prediction"],"product":r["product"]}
                for r in direct if not r["correct"]
            ][:30],
        }
        print(f"=== {name} ===")
        for label,stat in (("All",results[name]["overall"]),
                           ("Direct product answer",results[name]["direct_product"]),
                           ("Small factor answer",results[name]["factor_retrieval"])):
            print(f"{label:24} {stat['correct']:3}/{stat['n']:3} = {stat['accuracy']:.2%}")
        print("Direct product by true product size:")
        for k,v in results[name]["direct_by_product_band"].items():
            print(f"  {k}: {v['correct']}/{v['n']} = {v['accuracy']:.1%}")
        print("Direct product by factor digit pattern:")
        for k,v in results[name]["direct_by_factor_pattern"].items():
            print(f"  {k}: {v['correct']}/{v['n']} = {v['accuracy']:.1%}")
        d=results[name]["direct_product"]
        print(f"Direct-product wrong numeric answers within 10: {d['wrong_with_abs_error_at_most_10']}"
              f"/{d['numeric_error_count']} | correct length: {d['wrong_with_correct_digit_length']}"
              f"/{d['numeric_error_count']}")
    return {"source":"frozen Round4 extended_val 384 and saved predictions",
            "note":"Product size is 401..999 for all math facts. Math fact repeats across"
                    " categories; per-row statistics are descriptive.",
            "results":results}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save",action="store_true")
    args=ap.parse_args()
    output=compute()
    if args.save:
        if OUT.exists():
            raise FileExistsError(f"Refusing to overwrite {OUT}; run without --save")
        OUT.parent.mkdir(parents=True,exist_ok=True)
        with OUT.open("x",encoding="utf-8") as f:
            json.dump(output,f,ensure_ascii=False,indent=2)
            f.write("\n")
        print("Saved:",OUT)


if __name__=="__main__":
    main()
