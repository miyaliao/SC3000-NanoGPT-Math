#!/usr/bin/env python3
"""Round8 same-positives refusal vs self-generated actual wrong-answer negatives.

Both arms train on EXACT SAME positive strings, prompts, order and 100k rows.
Read-only strict primary accuracy; no hidden test or GPU.
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
ARMS=("control","hardneg")
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
        c["control_correct"]+=ac
        c["hardneg_correct"]+=bc
        c["control_only"]+=ac and not bc
        c["hardneg_only"]+=bc and not ac
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
        control=pair["control_correct"]
        target=pair["hardneg_correct"]
        print(f"    {name:10} N={pair['n']:3} "
              f"control={control:3}/{pair['n']} "
              f"target={target:3}/{pair['n']} "
              f"HARDNEG-CONTROL={target-control:+d} "
              f"(control-only={pair['control_only']}, hardneg-only={pair['hardneg_only']})")


def run():
    files={a:BASE/f"round8_{a}"/"summary.json" for a in ARMS}
    for a,path in files.items():
        if not path.is_file():
            print(f"Waiting for {a} completion: {path}")
            return
    summaries={a:read(files[a]) for a in ARMS}
    protocol=("init_sha256","source_round8_manifest_sha256","seed",
              "lr","epochs","batch_size","updates","loss","smoke")
    for name in protocol:
        vals=[summaries[a][name] for a in ARMS]
        if vals[0]!=vals[1]:raise RuntimeError(f"Unmatched Round8 {name}: {vals}")
    if summaries["control"]["smoke"] or summaries["hardneg"]["smoke"]:
        raise RuntimeError("Do not compare smoke pilot against full train")
    base={
       "val":(567,768),
       "old_ood":(109,256),
       "extended_val":(174,384),
       "train_probe":(211,256),
    }
    print("ROUND8 MATCHED same-positive preference: refusal CONTROL vs model wrong-answer HARDNEG")
    print(f"init_sha256={summaries['control']['init_sha256']}")
    print("Round7 Replay fixed baseline: Val567/768, OOD109/256, Ext174/384, Direct28/128")
    print("Only Rejected differs: every Chosen string and its training order identical")
    for split,n in SPLITS:
        data={}
        for a in ARMS:
            path=BASE/f"round8_{a}"/f"{split}_predictions.json"
            data[a]=read(path)
            if len(data[a])!=n:raise RuntimeError(f"Unexpected {split} size")
            recomputed=sum(r["status"]=="correct" for r in data[a])
            if recomputed!=summaries[a]["evaluation"][split]["correct"]:
                raise RuntimeError(f"Summary/predictions mismatch: {split} {a}")
        comparison=paired(data["control"],data["hardneg"])
        if comparison["n"]!=n:raise RuntimeError("Unknown paired count")
        a,b=comparison["control_correct"],comparison["hardneg_correct"]
        print(f"{split:13} base={base[split][0]}/{n} "
              f"control={a}/{n} hardneg={b}/{n} "
              f"HARDNEG-CONTROL={b-a:+d} "
              f"(control-only={comparison['control_only']}; "
              f"hardneg-only={comparison['hardneg_only']})")
        if split=="extended_val":
            d=paired(data["control"],data["hardneg"],True)
            print(f"  {'DIRECT 128':11} control={d['control_correct']}/128 "
                  f"hardneg={d['hardneg_correct']}/128 "
                  f"HARDNEG-CONTROL={d['hardneg_correct']-d['control_correct']:+d}")
        bycat=Counter()
        for row in data["control"]:
            bycat[row["category"]]+=row["status"]=="correct"
        bycat_new=Counter()
        for row in data["hardneg"]:
            bycat_new[row["category"]]+=row["status"]=="correct"
        print("  by category HARDNEG - CONTROL:",
              " ".join(f"{c}={bycat_new[c]-bycat[c]:+d}"
                       for c in sorted(bycat.keys()|bycat_new.keys())))
        report_product_bands(data["control"],data["hardneg"],split)
    print("No public Blind61/41 or sealed independent test read.")
    print("The matched HARDNEG-CONTROL gap, not either arm vs old Replay alone, "
          "tests the added value of the model's own wrong Rejected completions.")

if __name__=="__main__":
    run()
