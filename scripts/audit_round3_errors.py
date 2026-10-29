#!/usr/bin/env python3
"""Round 3 read-only audit: category/answer errors and old frozen-256 OOD test.

CPU-only error breakdown runs without torch or GPU.
OOD generation needs CUDA and a Slurm allocation; never trains or changes weights.
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
RUN=ROOT/"dpo/experiments/round3_100k"
ROUND3=ROOT/"dpo/data/round3_structured_full"
OLD_VAL=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(rows, key):
    buckets=defaultdict(list)
    for r in rows:buckets[str(key(r))].append(r)
    result={}
    for label, group in sorted(buckets.items()):
        n=len(group);good=sum(r["status"]=="correct" for r in group)
        result[label]={"n":n,"correct":good,"accuracy":good/n}
    return result


def audit_predictions(rows, metadata):
    by_prompt={r["prompt"]:r for r in metadata}
    if len(rows)!=len(by_prompt) or len(set(r["prompt"] for r in rows))!=len(rows):
        raise ValueError("Prediction/validation count or duplicate mismatch")
    if set(r["prompt"] for r in rows)!=set(by_prompt):
        raise ValueError("Prediction prompts not equal to validation prompts")
    for row in rows:
        meta=by_prompt[row["prompt"]]
        if row["category"]!=meta["category"] or str(row["answer"])!=str(meta["answer"]):
            raise ValueError("Wrong prediction/ground truth alignment")
    errors=[r for r in rows if r["status"]!="correct"]
    print(f"Val: correct={len(rows)-len(errors)}/{len(rows)}; errors={len(errors)}")
    groups={
        "by_category": aggregate(rows, lambda r:r["category"]),
        "by_answer_digits":aggregate(rows,lambda r:len(str(r["answer"]).lstrip("-"))),
        "by_curriculum_tag": aggregate(rows,lambda r:by_prompt[r["prompt"]]["curriculum_tag"]),
        "by_stop_reason":aggregate(rows,lambda r:r["stop_reason"]),
    }
    for name,data in groups.items():
        print(f"\n=== {name} ===")
        for label,stats in data.items():
            print(f"{label:30} {stats['correct']:3}/{stats['n']:3} = {stats['accuracy']:.2%}")
    print("\n=== First 20 wrong answers ===")
    for r in errors[:20]:
        print(f"{r['prompt']} | gold={r['answer']} | pred={r.get('prediction')} "
              f"| completion={r['completion']!r}")
    return {"n":len(rows),"errors":len(errors),"breakdowns":groups,
            "first_40_errors":errors[:40]}


def run_ood(checkpoint, device, out):
    import torch
    import pickle
    from diagnose_generation_gap import load_model
    from evaluate_ta_integer import greedy_original_stop
    from train_round3_course import evaluate_text, write_once
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not OLD_VAL.is_file():raise FileNotFoundError(OLD_VAL)
    if out.exists():raise FileExistsError(out)
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Use Slurm CUDA allocation")
    with (ROOT/"sft/meta.pkl").open("rb") as f: vocab=pickle.load(f)
    stoi,itos=vocab["stoi"],vocab["itos"]
    model=load_model(checkpoint,device)
    old=load_json(OLD_VAL)
    counts=Counter();by_kind=defaultdict(Counter);rows=[]
    for i,r in enumerate(old):
        prompt,gold=r["prompt"],str(r["answer"])
        if "." in prompt+gold or not re.fullmatch(r"[+-]?\d+",gold):
            raise ValueError(f"Not integer-only: {prompt}")
        completion,stop=greedy_original_stop(model,prompt,stoi,itos,80,device)
        status,pred,proof=evaluate_text(completion,gold)
        counts[status]+=1;by_kind[r["category"]][status]+=1
        rows.append({"prompt":prompt,"answer":gold,"category":r["category"],
                     "completion":completion,"prediction":pred,"status":status,
                     "explanation_valid":proof,"stop_reason":stop})
        if i<8:print(f"OOD: {prompt} -> {completion!r} [{status}]",flush=True)
    n=len(rows)
    summary={"n":n,"correct":counts["correct"],"accuracy":counts["correct"]/n,
             "counts":dict(counts),"by_category":{k:dict(v) for k,v in sorted(by_kind.items())},
             "checkpoint":str(checkpoint),"evaluator":"first-number because-compatible greedy"}
    out.mkdir(parents=True,exist_ok=False)
    write_once(out/"summary.json",summary)
    write_once(out/"predictions.json",rows)
    print(f"=== OLD FROZEN VAL OOD === correct={summary['correct']}/{n} "
          f"({summary['accuracy']:.2%})",flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ood",action="store_true")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--checkpoint",type=Path,default=RUN/"dpo.pt")
    p.add_argument("--out-dir",type=Path,default=ROOT/"results/round3_old_val_ood")
    args=p.parse_args()
    if args.ood:
        checkpoint=args.checkpoint if args.checkpoint.is_absolute() else ROOT/args.checkpoint
        out=args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir
        run_ood(checkpoint,args.device,out)
    else:
        rows=load_json(RUN/"val_predictions.json")
        metadata=load_json(ROUND3/"val.json")
        report=audit_predictions(rows,metadata)
        dest=ROOT/"results/round3_val_error_audit.json"
        if dest.exists():raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(f"\nSaved {dest}")


if __name__=="__main__":main()
