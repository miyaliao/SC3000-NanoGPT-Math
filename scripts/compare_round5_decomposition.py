#!/usr/bin/env python3
"""Round5 paired original-vs-decomposition comparison on identical fixed questions.

Original=A; decomposition=B. Refuses unmatched source manifests, schedules,
checkpoints, or question orders. No model execution required.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"dpo/experiments/round5_original"
B=ROOT/"dpo/experiments/round5_decomposition"
SPLITS=("val","old_ood","extended_val","train_probe")
EXT_TARGET=("multiply","x_over_a")


def read(path):
    if not path.is_file():raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compare_rows(left,right):
    if len(left)!=len(right):raise RuntimeError("Different row counts")
    pair=Counter()
    by_category=defaultdict(Counter)
    examples=[]
    for i,(a,b) in enumerate(zip(left,right)):
        for key in ("prompt","answer","category"):
            if a[key]!=b[key]:
                raise RuntimeError(f"Different fixed evaluation data at row {i}: {key}")
        wa=a["status"]=="correct"
        wb=b["status"]=="correct"
        k="both" if wa and wb else "A_only" if wa else "B_only" if wb else "neither"
        pair[k]+=1
        by_category[a["category"]][k]+=1
        if len(examples)<12 and k in ("A_only","B_only"):
            examples.append({"index":i,"prompt":a["prompt"],"answer":a["answer"],
                             "A_pred":a.get("prediction"),"B_pred":b.get("prediction"),
                             "outcome":k})
    n=len(left)
    data={"n":n,"A_correct":pair["both"]+pair["A_only"],
          "B_correct":pair["both"]+pair["B_only"],
          "B_minus_A_count":pair["B_only"]-pair["A_only"],
          "B_minus_A_percentage_points":100*(pair["B_only"]-pair["A_only"])/n,
          "paired":dict(pair),
          "unique_prompts":len({r["prompt"] for r in left}),
          "by_category":{
              kind:{"n":sum(x.values()),"A_correct":x["both"]+x["A_only"],
                    "B_correct":x["both"]+x["B_only"],
                    "B_minus_A":x["B_only"]-x["A_only"]}
              for kind,x in sorted(by_category.items())},
          "changed_examples":examples}
    return data


def main():
    sa,sb=read(A/"summary.json"),read(B/"summary.json")
    for key in ("init_sha256","source_round5_manifest_sha256","seed",
                "lr","epochs","batch_size","updates"):
        if sa[key]!=sb[key]:
            raise RuntimeError(f"Matched settings differ on {key}")
    m=read(ROOT/"dpo/data/round5_decomposition/manifest.json")
    if sa["source_training_sha256"]!=m["file_sha256"]["original"] or sb["source_training_sha256"]!=m["file_sha256"]["decomposition"]:
        raise RuntimeError("Training dataset hash does not match Round5 paired data manifest")
    if sa["smoke"] or sb["smoke"]:
        raise RuntimeError("Cannot compare smoke instead of full experiment")
    print("PASS: matched Round4 E1 initialization, seed, Course Loss, steps, LR and data manifest")
    print(f"Changed Positive explanation rows: {m['modified_rows']}/{m['source_round4_training_rows']} "
          f"({m['modified_fraction']:.2%})")
    print("Modified by category:",m["modified_by_category"])
    summaries={}
    for split in SPLITS:
        pa,pb=read(A/f"{split}_predictions.json"),read(B/f"{split}_predictions.json")
        stats=compare_rows(pa,pb)
        if stats["A_correct"]!=sa["evaluation"][split]["correct"] or stats["B_correct"]!=sb["evaluation"][split]["correct"]:
            raise RuntimeError(f"Summary/prediction conflict: {split}")
        summaries[split]=stats
        print(f"{split:14} A={stats['A_correct']:3}/{stats['n']:3} "
              f"B={stats['B_correct']:3}/{stats['n']:3} "
              f"B-A={stats['B_minus_A_count']:+d} "
              f"A_only={stats['paired'].get('A_only',0)}, "
              f"B_only={stats['paired'].get('B_only',0)}")
    print("\n=== FIXED EXTENDED VAL: SIX ARITHMETIC CATEGORIES ===")
    c=summaries["extended_val"]["by_category"]
    for category,data in c.items():
        print(f"{category:15} A={data['A_correct']:2}/{data['n']:2} "
              f"B={data['B_correct']:2}/{data['n']:2} delta={data['B_minus_A']:+d}")
    target=[cat for cat in EXT_TARGET if cat in c]
    if len(target)!=2:raise RuntimeError("Missing multiply/x_over_a target categories")
    a_total=sum(c[k]["A_correct"] for k in target)
    b_total=sum(c[k]["B_correct"] for k in target)
    n=sum(c[k]["n"] for k in target)
    print(f"\nDIRECT LARGE-PRODUCT TARGET: A={a_total}/{n} ({a_total/n:.2%}) "
          f"B={b_total}/{n} ({b_total/n:.2%}) "
          f"delta={b_total-a_total:+d} / {n}")
    print("This isolates explanation-text treatment at equal questions/steps;")
    print("different explanation lengths still change full-sequence Loss weighting.")
    print("No final Test-set data accessed.")


if __name__=="__main__":main()
