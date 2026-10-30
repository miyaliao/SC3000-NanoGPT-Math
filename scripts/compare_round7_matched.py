#!/usr/bin/env python3
"""Compare matched Round7 one-epoch replay-only vs targeted curriculum (CPU).

Read saved predictions only; same answer-first primary scoring for both arms.
No GPU or independent sealed test involved.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_round1 import parse_relation

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dpo/experiments"
ARMS=("replay","targeted")
SPLITS=(("val",768),("old_ood",256),("extended_val",384),("train_probe",256))
DIRECT={"multiply","x_over_a"}

def read(file):
    return json.loads(file.read_text(encoding="utf-8"))

def paired(a,b,direct=False):
    if len(a)!=len(b):raise RuntimeError("Paired question set lengths differ")
    c=Counter()
    for i,(x,y) in enumerate(zip(a,b)):
        for key in ("prompt","answer","category"):
            if x[key]!=y[key]:
                raise RuntimeError(f"Question mismatch {i}: {key}")
        if direct and x["category"] not in DIRECT:continue
        if x["status"] not in ("correct","math_error","format_failure"):
            raise RuntimeError(f"Unexpected control score: {x['status']}")
        if y["status"] not in ("correct","math_error","format_failure"):
            raise RuntimeError(f"Unexpected target score: {y['status']}")
        ac=x["status"]=="correct"
        bc=y["status"]=="correct"
        c["n"]+=1
        c["replay_correct"]+=ac
        c["targeted_correct"]+=bc
        c["replay_only"]+=ac and not bc
        c["targeted_only"]+=bc and not ac
        c["both_correct"]+=ac and bc
        c["both_wrong"]+=not(ac or bc)
    if direct and c["n"]!=128:raise RuntimeError("Expected direct 128 subset")
    return dict(c)

def report_product_bands(control, treatment, split):
    """Read-only paired accuracy by canonical product magnitude."""
    if split not in ("val","old_ood","extended_val"):return
    bands={
        "<=100":[], "101-400":[], "401-999":[],
        "1000-4999":[], ">=5000":[]
    }
    for i,(a,b) in enumerate(zip(control,treatment)):
        if a["prompt"]!=b["prompt"] or a["answer"]!=b["answer"]:
            raise RuntimeError(f"Unaligned split at index {i}")
        fact=parse_relation(a["prompt"],str(a["answer"]))
        if fact[0]!="mul":continue
        product=int(fact[3])
        band=("<=100" if product<=100 else "101-400" if product<=400 else
              "401-999" if product<=999 else "1000-4999" if product<=4999
              else ">=5000")
        bands[band].append((a,b))
    print("  by canonical multiplication product range (same questions):")
    for name,rows in bands.items():
        if not rows:continue
        pair=paired([p[0] for p in rows],[p[1] for p in rows])
        replay=pair["replay_correct"]
        target=pair["targeted_correct"]
        print(f"    {name:10} N={pair['n']:3} "
              f"replay={replay:3}/{pair['n']} "
              f"target={target:3}/{pair['n']} "
              f"TARGET-REPLAY={target-replay:+d} "
              f"(replay-only={pair['replay_only']}, targeted-only={pair['targeted_only']})")


def run():
    files={a:BASE/f"round7_{a}"/"summary.json" for a in ARMS}
    for a,path in files.items():
        if not path.is_file():
            print(f"Waiting for {a} completion: {path}")
            return
    summaries={a:read(files[a]) for a in ARMS}
    protocol=("init_sha256","source_round7_manifest_sha256","seed",
              "lr","epochs","batch_size","updates","loss","smoke")
    for name in protocol:
        vals=[summaries[a][name] for a in ARMS]
        if vals[0]!=vals[1]:raise RuntimeError(f"Unmatched Round7 {name}: {vals}")
    if summaries["replay"]["smoke"] or summaries["targeted"]["smoke"]:
        raise RuntimeError("Do not compare smoke pilot against full train")
    base={
       "val":(560,768),
       "old_ood":(106,256),
       "extended_val":(168,384),
       "train_probe":(208,256),
    }
    print("MATCHED Round7 Course Loss, 1 epoch, same Round5-A initialization")
    print(f"init_sha256={summaries['replay']['init_sha256']}")
    for split,n in SPLITS:
        data={}
        for a in ARMS:
            path=BASE/f"round7_{a}"/f"{split}_predictions.json"
            data[a]=read(path)
            if len(data[a])!=n:raise RuntimeError(f"Unexpected {split} size")
            recomputed=sum(r["status"]=="correct" for r in data[a])
            if recomputed!=summaries[a]["evaluation"][split]["correct"]:
                raise RuntimeError(f"Summary/predictions mismatch: {split} {a}")
        comparison=paired(data["replay"],data["targeted"])
        if comparison["n"]!=n:raise RuntimeError("Unknown paired count")
        a,b=comparison["replay_correct"],comparison["targeted_correct"]
        print(f"{split:13} base={base[split][0]}/{n} "
              f"replay={a}/{n} targeted={b}/{n} "
              f"TARGET-REPLAY={b-a:+d} "
              f"(replay-only={comparison['replay_only']}; "
              f"targeted-only={comparison['targeted_only']})")
        if split=="extended_val":
            d=paired(data["replay"],data["targeted"],True)
            print(f"  {'DIRECT 128':11} replay={d['replay_correct']}/128 "
                  f"targeted={d['targeted_correct']}/128 "
                  f"TARGET-REPLAY={d['targeted_correct']-d['replay_correct']:+d}")
        bycat=Counter()
        for row in data["replay"]:
            bycat[row["category"]]+=row["status"]=="correct"
        bycat_new=Counter()
        for row in data["targeted"]:
            bycat_new[row["category"]]+=row["status"]=="correct"
        print("  by category TARGET - REPLAY:",
              " ".join(f"{c}={bycat_new[c]-bycat[c]:+d}"
                       for c in sorted(bycat.keys()|bycat_new.keys())))
        report_product_bands(data["replay"],data["targeted"],split)
    print("No public Blind61/41 or sealed independent test read.")
    print("The matched targeted-control gap, not targeted-vs-Round5 alone, "
          "is the main estimate of whether curriculum helped.")

if __name__=="__main__":
    run()
