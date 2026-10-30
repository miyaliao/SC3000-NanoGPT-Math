#!/usr/bin/env python3
"""Round6 A-vs-D only, after the decision to pause B/C following Epoch2.

This is read-only: one SAME legacy answer-first score for A and D.
It reads existing epoch summaries and prediction JSON files; no model/GPU
needed. Secondary --diagnostics are separate and NEVER rank checkpoints.

A=original because, D=decomposition before answer on 16,271 train rows.
A-vs-D is a bundled output-format intervention, NOT a pure comparison of
decomposition position. Past B/C Epoch1-2 artifacts remain untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dpo/experiments/round6_from_sft"
A="original"
D="decomposition_before"
SPLITS={"val":768,"old_ood":256,"extended_val":384,"train_probe":256}
DIRECT={"multiply","x_over_a"}
MATCH_FIELDS=(
    "init_sft_sha256","source_manifest_sha256","loss","batch_size",
    "training_rows","lr","seed","updates_this_epoch",
    "total_schedule_updates","max_new_tokens","order_sha256"
)


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def file(arm,epoch,name):
    return BASE/arm/f"epoch{epoch:02d}"/name


def paired(a,b,criterion,focus=False):
    if len(a)!=len(b):
        raise ValueError("Comparands have different evaluation sizes")
    outcomes=Counter()
    category=Counter()
    for i,(x,y) in enumerate(zip(a,b)):
        for field in ("prompt","answer","category"):
            if x[field]!=y[field]:
                raise ValueError(f"Different fixed question at index {i}: {field}")
        if focus and x["category"] not in DIRECT:
            continue
        valid={"correct","math_error","format_failure"}
        if x[criterion] not in valid or y[criterion] not in valid:
            raise ValueError(f"Unknown score status at row {i}")
        ax=x[criterion]=="correct"
        dy=y[criterion]=="correct"
        case=("both" if ax and dy else
              "A_only" if ax else
              "D_only" if dy else
              "neither")
        outcomes[case]+=1
        category[x["category"]]+=1
    n=sum(outcomes.values())
    if focus and (n!=128 or category!={"multiply":64,"x_over_a":64}):
        raise ValueError(f"Primary 128 direct-product subset changed: {category}")
    if not n:
        raise ValueError("Empty paired comparison")
    return {
        "n":n,"a_correct":outcomes["both"]+outcomes["A_only"],
        "d_correct":outcomes["both"]+outcomes["D_only"],
        "d_minus_a":outcomes["D_only"]-outcomes["A_only"],
        "a_only":outcomes["A_only"],"d_only":outcomes["D_only"],
        "both":outcomes["both"],"neither":outcomes["neither"],
    }


def brief(result):
    return (f"A={result['a_correct']}/{result['n']} "
            f"D={result['d_correct']}/{result['n']} "
            f"D-A={result['d_minus_a']:+d} "
            f"(A-only={result['a_only']}; D-only={result['d_only']})")


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics",action="store_true",
                   help="Print D secondary final-answer and step metrics separately")
    args=p.parse_args()
    m=read(ROOT/"dpo/data/round6_from_sft/manifest.json")
    if (m["version"]!="round6_sft_four_formats_v1" or
        m["train_rows_per_arm"]!=100000 or
        m["modified_before_after_rows"]!=16271):
        raise ValueError("Unexpected Round6 training-data manifest")
    print("ROUND6 CONTINUATION: A Original vs D Decomposition-before")
    print("Primary = original uniform answer-first evaluator (NOT confirmed TA parser)")
    print("D differs on 16,271/100,000 rows. B/C stopped after Epoch2, "
          "and are not required for this report.")
    previous={}
    complete=0
    for epoch in range(1,6):
        names={arm:file(arm,epoch,"epoch_summary.json") for arm in (A,D)}
        present={arm:path.is_file() for arm,path in names.items()}
        if not all(present.values()):
            state="; ".join(
                arm+"="+("complete" if ok else "missing")
                for arm,ok in present.items()
            )
            print(f"Epoch {epoch}: paired report unavailable ({state})")
            continue
        sa,sd=read(names[A]),read(names[D])
        for arm,s in ((A,sa),(D,sd)):
            if s.get("smoke") or s.get("epoch")!=epoch or s.get("arm")!=arm:
                raise ValueError(f"Inconsistent non-smoke epoch summary: {names[arm]}")
        for field in MATCH_FIELDS:
            if sa[field]!=sd[field]:
                raise ValueError(f"Matched protocol mismatch epoch={epoch}, {field}")
        datasets={}
        print(f"\n=== EPOCH {epoch}: SAME PRIMARY SCORER ON A AND D ===")
        for split,n in SPLITS.items():
            a=read(file(A,epoch,f"{split}_predictions.json"))
            d=read(file(D,epoch,f"{split}_predictions.json"))
            if len(a)!=n or len(d)!=n:
                raise ValueError(f"Wrong frozen eval size at {split}, E{epoch}")
            observed=paired(a,d,"strict_status")
            if (observed["a_correct"]!=sa["evaluation"][split]["strict_correct"] or
                observed["d_correct"]!=sd["evaluation"][split]["strict_correct"]):
                raise ValueError(f"Prediction-score conflict: E{epoch} {split}")
            print(f"{split:13} {brief(observed)}")
            datasets[split]=(a,d)
        primary=paired(*datasets["extended_val"],"strict_status",focus=True)
        print(f"{'DIRECT 128':13} {brief(primary)}")
        for arm,s in ((A,sa),(D,sd)):
            key=(arm,"val")
            current=s["evaluation"]["val"]["strict_correct"]
            if key in previous:
                print(f"  {arm} Val: E{epoch-1}={previous[key]} "
                      f"→ E{epoch}={current} ({current-previous[key]:+d})")
            previous[key]=current
        if args.diagnostics:
            print("--- D REASONING DIAGNOSTICS: NOT PRIMARY MODEL ACCURACY ---")
            dsum=sd["evaluation"]["extended_val"]
            print(f"D extended: final-aware={dsum['final_correct']}/384, "
                  f"pre-answer syntax={dsum['reason_before_answer']}/384, "
                  f"internally valid arithmetic={dsum['valid_calculation']}/384")
            da,dd=datasets["extended_val"]
            d_direct=[r for r in dd if r["category"] in DIRECT]
            print("D direct 128 final-aware correct=",
                  sum(r["final_status"]=="correct" for r in d_direct),
                  "/128; compare ONLY as diagnostics, not against A primary score.")
        complete+=1
    print(f"\n{complete} matched A/D epoch(s) available.")
    print("Primary rule identical across arms. TA extraction code is still unknown.")
    print("No GPU, model weights, B/C Epoch3+, or sealed final Test read.")


if __name__=="__main__":
    run()
