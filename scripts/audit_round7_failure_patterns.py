#!/usr/bin/env python3
"""CPU/read-only failure-type audit to guide future Round7 math curriculum.

The evaluation data here are for DIAGNOSIS ONLY. This script NEVER generates
training examples from validation/public-blind prompt strings or gold labels.
The next training generator must use independently fact-split train pools.

Usage:
    python scripts/audit_round7_failure_patterns.py
    python scripts/audit_round7_failure_patterns.py --save
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation

R5=ROOT/"dpo/experiments/round5_original"
PUBLIC=ROOT/"dpo/experiments/prior_year_blind_2025"
OUT=ROOT/"results/round7_failure_patterns.json"

SOURCES={
    "val768":R5/"val_predictions.json",
    "old_ood256":R5/"old_ood_predictions.json",
    "extended384":R5/"extended_val_predictions.json",
    "public_blind61":PUBLIC/"round5_A_greedy_blind61_predictions.json",
    "public_blind41":PUBLIC/"round5_A_greedy_blind41_predictions.json",
}
LENGTHS={"val768":768,"old_ood256":256,"extended384":384,"public_blind61":61,"public_blind41":41}
ARITH=re.compile(r"^(\d+)([+\-*/])(\d+)=\?$")
EQ=re.compile(r"^(\d+|x)([+\-*/])(\d+|x)=(\d+),x=\?$")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def family_and_kind(prompt,answer,record):
    if "category" in record:
        category=record["category"]
    else:
        m=ARITH.fullmatch(prompt)
        if m:
            category={"+":"add","-":"subtract","*":"multiply","/":"divide"}[m.group(2)]
        else:
            m=EQ.fullmatch(prompt)
            if not m:raise ValueError(f"Unknown question kind: {prompt}")
            left,op,right,_=m.groups()
            sub={"+":"plus","-":"minus","*":"times","/":"over"}[op]
            category=("x_" if left=="x" else "a_")+sub+("_a" if left=="x" else "_x")
    fact=parse_relation(prompt,str(answer))
    return category,fact

def characteristic(prompt,answer,category,fact):
    result=int(answer)
    op=ARITH.fullmatch(prompt) or EQ.fullmatch(prompt)
    a,b=int(fact[1]),int(fact[2])
    numeric_len=len(str(abs(result)))
    if fact[0]=="mul":
        product=int(fact[3])
        size=("<=100" if product<=100 else "101-400" if product<=400 else
              "401-999" if product<=999 else ">=1000")
        factor=("identity" if a<=1 else "single_by_multi" if a<10 and b>=10
                else "two_by_two" if a>=10 and b>=10 else "small_pair")
        stage=("direct_product" if category in ("multiply","x_over_a") else "inverse_factor")
        return {"math_family":"mul","numeric_digits":numeric_len,
                "product_band":size,"factor_pattern":factor,"target":stage}
    if fact[0]=="add":
        s=a%10+b%10
        mode=("carry" if s>=10 else "no_carry")
        size=("answer>=100" if result>=100 else "answer<100")
        return {"math_family":"add","numeric_digits":numeric_len,
                "carry_pattern":mode,"answer_band":size,
                "equation":"equation" if category not in ("add","subtract") else "arithmetic"}
    raise ValueError(f"Unexpected fact: {fact}")

def summarize(rows):
    c=Counter()
    for row in rows:
        c["n"]+=1
        c["correct"]+=row["correct"]
        c["math_error"]+=row["status"]=="math_error"
        c["format_failure"]+=row["status"]=="format_failure"
    return {"n":c["n"],"correct":c["correct"],
            "accuracy":c["correct"]/c["n"] if c["n"] else None,
            "math_error":c["math_error"],
            "format_failure":c["format_failure"]}

def evaluate_group(rows,expected,group):
    if len(rows)!=expected:
        raise ValueError(f"{group}: expected {expected}, got {len(rows)}")
    result=[]
    for i,r in enumerate(rows):
        prompt,gold=r["prompt"],str(r["answer"])
        category,fact=family_and_kind(prompt,gold,r)
        status=r.get("status",r.get("strict_status"))
        if status not in ("correct","math_error","format_failure"):
            raise ValueError(f"Unrecognized score in {group}[{i}]: {status}")
        pred=r.get("prediction",r.get("strict_prediction"))
        if status=="correct" and (pred is None or int(pred)!=int(gold)):
            raise ValueError(f"Correct label inconsistent in {group}[{i}]")
        feat=characteristic(prompt,gold,category,fact)
        result.append({"prompt":prompt,"category":category,"answer":gold,
                       "status":status,"prediction":pred,
                       "correct":status=="correct","features":feat})
    totals=summarize(result)
    summaries={"overall":totals,"by_category":{},"by_feature":{}}
    for cat in sorted({r["category"] for r in result}):
        summaries["by_category"][cat]=summarize([r for r in result if r["category"]==cat])
    for field in ("math_family","numeric_digits","product_band","factor_pattern",
                  "target","carry_pattern","answer_band","equation"):
        values=sorted({str(r["features"][field]) for r in result if field in r["features"]})
        if not values:continue
        summaries["by_feature"][field]={
            value:summarize([r for r in result if str(r["features"].get(field))==value])
            for value in values
        }
    print(f"=== {group}: {totals['correct']}/{totals['n']} "
          f"({totals['accuracy']:.1%}), math_error={totals['math_error']}, "
          f"format_failure={totals['format_failure']} ===")
    for category,data in summaries["by_category"].items():
        print(f"  {category:14} {data['correct']:3}/{data['n']:3} "
              f"({data['accuracy']:.1%})")
    for name in ("target","product_band","factor_pattern","carry_pattern"):
        if name not in summaries["by_feature"]:continue
        print(f"  - {name}:")
        for value,data in summaries["by_feature"][name].items():
            print(f"      {value:19} {data['correct']:3}/{data['n']:3} "
                  f"({data['accuracy']:.1%})")
    return summaries

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save",action="store_true")
    args=ap.parse_args()
    report={
        "scope":"Frozen Round5-A validation predictions and public benchmark failures",
        "purpose":"Descriptive errors only. Never source Round7 training examples from these rows.",
        "strict_score":"The answer is <integer> at the completion start",
        "sources":{},
        "no_sealed_test_access":True,
    }
    for name,path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved predictions: {path}")
        rows=read(path)
        report["sources"][name]=evaluate_group(rows,LENGTHS[name],name)
    print("\nDo NOT train directly on these validation/public-blind examples or "
          "tune toward the public Blind61/41 headline results.")
    print("For a valid hard-example intervention, mine only TRAIN-split facts, "
          "protect existing fixed Val/OOD/Extended, and compare a matched replay control.")
    if args.save:
        if OUT.exists():raise FileExistsError(f"Refusing overwrite: {OUT}")
        OUT.parent.mkdir(parents=True,exist_ok=True)
        with OUT.open("x",encoding="utf-8") as f:
            json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
        print("Saved:",OUT)

if __name__=="__main__":
    main()
