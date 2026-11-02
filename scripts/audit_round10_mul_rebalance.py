#!/usr/bin/env python3
"""Round10 CPU-only audit: six multiplicative task families in frozen Round9 Random.

3-digit is defined by underlying CANONICAL PRODUCT 100..999, not by
answer length (e.g. 722/19=38 still counts as 3-digit-product fact).
Never edits the source dataset, model weights, or frozen benchmarks.

Run: python scripts/audit_round10_mul_rebalance.py
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation
from mine_round8_train_hard_negatives import kind

SRC=ROOT/"dpo/data/round9_balanced_full/random_pairs.json"
MULT=("multiply","divide","x_times_a","a_times_x","x_over_a","a_over_x")
POS_MARKER=" The answer is "
BINS=("1digit","2digit","3digit","4plus")

def value_bin(product):
    v=abs(int(product))
    return "1digit" if v<10 else ("2digit" if v<100 else
           ("3digit" if v<1000 else "4plus"))

def extract(row):
    text=row["positive"]
    if POS_MARKER not in text:
        raise RuntimeError("Expected standard course Chosen format")
    prompt,tail=text.split(POS_MARKER,1)
    answer=tail.split(" because ",1)[0].rstrip(".")
    cat=kind(prompt)
    fact=parse_relation(prompt,answer)
    if cat in MULT and fact[0]!="mul":
        raise RuntimeError("Non-multiplicative canonical fact")
    return prompt,cat,fact

def report_group(label,rows):
    total=len(rows)
    counts=Counter(value_bin(key[3]) for _,key in rows)
    uniq_prompts=len({p for p,_ in rows})
    uniq_facts=len({k for _,k in rows})
    print(f"  {label:<14} rows={total:6} unique_prompt={uniq_prompts:5} "
          f"unique_fact={uniq_facts:5} repeats={total-uniq_prompts:6}")
    for b in BINS:
        n=counts[b]
        print(f"    {b:<8}: {n:6} / {total:6} = {n/total*100 if total else 0:6.2f}%")
    return counts

def main():
    if not SRC.is_file():raise FileNotFoundError(SRC)
    with SRC.open(encoding="utf-8") as file:
        data=json.load(file)
    if len(data)!=100000:
        raise RuntimeError(f"Expected exactly 100000 Random rows, got {len(data)}")
    groups=defaultdict(list)
    mul_all=[]
    for row in data:
        prompt,category,key=extract(row)
        if category in MULT:
            groups[category].append((prompt,key))
            mul_all.append((prompt,key))
    print(f"SOURCE {SRC}")
    print("All six multiplication/division families, including algebra:")
    for category in MULT:
        report_group(category,groups[category])
    print("\nOVERALL MULTIPLICATIVE:")
    totals=report_group("ALL MUL/DIV",mul_all)
    n=len(mul_all)
    assert n and sum(totals.values())==n
    n3=totals["3digit"]
    target3=n3//2
    n2=totals["2digit"]
    proposed2=n2+(n3-target3)
    print("\nREQUESTED HALVING (same multiplicative-row count):")
    print(f"  Original product 100-999: {n3}/{n} = {n3/n:.2%}")
    print(f"  Proposed product 100-999: {target3}/{n} = {target3/n:.2%}")
    print(f"  Proposed product 10-99  : {proposed2}/{n} = {proposed2/n:.2%}")
    print(f"  Replace {n3-target3} three-digit-product rows with two-digit-product rows")
    print("  This is an EXPOSURE quota, not yet a feasible unique-question set.")
    unique2={p for p,k in mul_all if value_bin(k[3])=="2digit"}
    unique2facts={k for p,k in mul_all if value_bin(k[3])=="2digit"}
    print("\nDEDUP CAPACITY IN ORIGINAL RANDOM:")
    print(f"  Existing distinct 2-digit-product prompts={len(unique2)} "
          f"distinct canonical facts={len(unique2facts)}")
    print(f"  Required 2-digit rows={proposed2}; "
          f"shortfall for strict unique prompts={max(0,proposed2-len(unique2))}")
    print("  A broader TRAIN-only candidate pool may increase uniqueness, but")
    print("  cannot make tens of thousands of distinct small products appear.")
    print("  Mathematical upper bound: 241 distinct unordered facts for")
    print("  positive factors 1..99 and product <=99; only 142 with both factors >=2.")
    print("\nNOTE: unique_prompt permits the same canonical math fact in")
    print("different equation templates. unique_fact removes these too.")
    print("No source file or checkpoint was modified.")

if __name__=="__main__":
    main()
