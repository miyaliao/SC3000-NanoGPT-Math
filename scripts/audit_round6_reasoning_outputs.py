#!/usr/bin/env python3
"""Round6 read-only audit of reasoning-first completion quality.

Distinct measures:
(1) generated a complete pre-answer arithmetic expression,
(2) each displayed arithmetic step is internally valid,
(3) the displayed factors actually match the question,
(4) the final predicted number matches the gold answer.

This does NOT evaluate the sealed final test or load GPU/model weights.

Usage:
    python scripts/audit_round6_reasoning_outputs.py --epoch 1
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
from evaluate_round6_formats import REASON,parse_completion

PRED_DIR=ROOT/"dpo/experiments/round6_from_sft/decomposition_before"
MUL=re.compile(r"(\d+)\*(\d+)=\?")
XO=re.compile(r"x/(\d+)=(\d+),x=\?")
DIRECT={"multiply","x_over_a"}


def expected_factors(prompt):
    m=MUL.fullmatch(prompt)
    if m:return tuple(sorted((int(m.group(1)),int(m.group(2)))))
    m=XO.fullmatch(prompt)
    if m:return tuple(sorted((int(m.group(2)),int(m.group(1)))))
    return None


def metric(rows):
    totals=Counter()
    samples={"reason_but_wrong_answer":[],"internally_wrong_steps":[],
             "reason_not_for_question":[],"unparseable_direct":[]}
    for i,r in enumerate(rows):
        prompt=r["prompt"]
        gold=int(r["answer"])
        completion=r["completion"]
        parsed=parse_completion(completion,gold)
        if (parsed["output_mode"]!=r["output_mode"] or
            parsed["final_status"]!=r["final_status"] or
            parsed["calculation_valid"]!=r["calculation_valid"]):
            raise ValueError(f"Recomputed parser differs from saved evaluation at row {i}")
        totals["n"]+=1
        direct=r["category"] in DIRECT
        if direct:totals["direct_n"]+=1
        if parsed["strict_status"]=="correct":totals["strict_correct"]+=1
        if parsed["final_status"]=="correct":totals["final_correct"]+=1
        if parsed["output_mode"]!="reason_before_answer":
            if direct and len(samples["unparseable_direct"])<8:
                samples["unparseable_direct"].append({"prompt":prompt,"completion":completion})
            continue
        totals["reason_before_answer"]+=1
        if direct:totals["direct_reason_before_answer"]+=1
        else:totals["non_direct_reason_before_answer"]+=1
        m=REASON.fullmatch(completion)
        if not m:raise AssertionError("Parser/reason expression mismatch")
        a,tens,p1,b,ones,p2,q1,q2,total,n=(int(x) for x in m.groups())
        internally_valid=(
          a==b and tens>=10 and tens%10==0 and ones>0
          and a*tens==p1 and a*ones==p2 and p1==q1 and p2==q2
          and p1+p2==total==n
        )
        if internally_valid:totals["internally_valid_steps"]+=1
        elif len(samples["internally_wrong_steps"])<8:
            samples["internally_wrong_steps"].append({"prompt":prompt,"gold":gold,"completion":completion})
        factors=expected_factors(prompt)
        matches_prompt=(factors is not None and
                        tuple(sorted((a,tens+ones)))==factors)
        if matches_prompt:totals["factor_alignment"]+=1
        elif len(samples["reason_not_for_question"])<8:
            samples["reason_not_for_question"].append({
                "prompt":prompt,"gold":gold,"completion":completion
            })
        if internally_valid and matches_prompt:
            totals["fully_valid_reason_for_question"]+=1
        if n==gold:
            totals["reason_final_number_correct"]+=1
        elif len(samples["reason_but_wrong_answer"])<8:
            samples["reason_but_wrong_answer"].append({
                "prompt":prompt,"gold":gold,"prediction":n,
                "steps_internally_valid":internally_valid,
                "factors_match_prompt":matches_prompt,
                "completion":completion,
            })
    return totals,samples


def run(epoch):
    file=PRED_DIR/f"epoch{epoch:02d}"/"extended_val_predictions.json"
    if not file.is_file():
        raise FileNotFoundError(file)
    records=json.loads(file.read_text(encoding="utf-8"))
    if len(records)!=384:
        raise ValueError("Expected 384 frozen extended validation rows")
    categories=defaultdict(list)
    for record in records:categories[record["category"]].append(record)
    overall,examples=metric(records)
    print(f"=== ROUND6 DECOMPOSITION-BEFORE EPOCH {epoch} ===")
    print(f"Overall: N={overall['n']} strict={overall['strict_correct']} "
          f"final={overall['final_correct']}")
    print(f"Complete pre-answer expression: {overall['reason_before_answer']}/384")
    print(f"Of these: internally correct steps={overall['internally_valid_steps']}, "
          f"correct factor relation to prompt={overall['factor_alignment']}, "
          f"both={overall['fully_valid_reason_for_question']}, "
          f"final gold number correct={overall['reason_final_number_correct']}")
    print(f"Reasoning output in DIRECT 128={overall['direct_reason_before_answer']}; "
          f"elsewhere={overall['non_direct_reason_before_answer']}")
    print("\nPer category: N, reasoning pattern, valid steps, aligned prompt, "
          "fully valid question-proof, final correct")
    for category,rows in sorted(categories.items()):
        counts,_=metric(rows)
        print(f"  {category:15} N={len(rows):3} "
              f"reason={counts['reason_before_answer']:3} "
              f"valid_steps={counts['internally_valid_steps']:3} "
              f"aligned={counts['factor_alignment']:3} "
              f"fully_valid={counts['fully_valid_reason_for_question']:3} "
              f"correct={counts['final_correct']:3}")
    for label,rows in examples.items():
        if rows:
            print(f"\n{label} (up to 8 examples):")
            for row in rows[:4]:print("  ",json.dumps(row,ensure_ascii=False))
    print("\nRead-only audit; no evaluation or training jobs submitted; "
          "reported measures differ from first-answer accuracy.")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epoch",type=int,default=1,choices=range(1,6))
    args=ap.parse_args()
    run(args.epoch)
