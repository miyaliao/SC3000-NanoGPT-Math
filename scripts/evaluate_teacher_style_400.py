#!/usr/bin/env python3
"""Evaluate frozen teacher-style 400, grouped by operation/subskill/Seen-Unseen.

READ ONLY: training never uses benchmark rows. This is a DEVELOPMENT benchmark,
not the sealed official independent Test or a known TA scorer.

Run with a GPU Slurm allocation:
 python scripts/evaluate_teacher_style_400.py --device cuda
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
BENCH=ROOT/"dpo/benchmarks/teacher_style_400_v1/questions.json"
MANIFEST=ROOT/"dpo/benchmarks/teacher_style_400_v1/manifest.json"
DEFAULT_OUT=ROOT/"dpo/experiments/teacher_style_400_v1_eval"
MODELS={
    "round5_A":ROOT/"dpo/experiments/round5_original/dpo.pt",
    "round7_replay":ROOT/"dpo/experiments/round7_replay/dpo.pt",
}
GROUPS=("add","multiply","divide","a_minus_x","x_times_a")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(rows):
    total=len(rows)
    c=sum(r["strict_status"]=="correct" for r in rows)
    return {"n":total,"strict_correct":c,"strict_accuracy":c/total if total else None,
            "format_failures":sum(r["strict_status"]=="format_failure" for r in rows)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--max-new-tokens",type=int,default=80)
    ap.add_argument("--model-names",default="round5_A,round7_replay",
                    help="Comma-separated existing checkpoint names")
    ap.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    selected=[s.strip() for s in args.model_names.split(",")]
    if not selected or len(set(selected))!=len(selected) or any(s not in MODELS for s in selected):
        ap.error("Unknown or duplicate model in --model-names")
    if args.max_new_tokens<1:
        ap.error("Require max_new_tokens>0")
    if not BENCH.is_file() or not MANIFEST.is_file():
        ap.error("Freeze 400 before evaluating: build_teacher_style_400.py --freeze")
    manifest=read(MANIFEST)
    rows=read(BENCH)
    if (manifest.get("version")!="teacher_style_400_v1" or len(rows)!=400 or
        sha(BENCH)!=manifest["questions_sha256"]):
        raise RuntimeError("Frozen benchmark data SHA mismatch or wrong size")
    from generate_round1 import parse_relation,stable_split
    if len({tuple(r["fact_key"]) for r in rows})!=400:
        raise RuntimeError("Duplicated mathematical facts")
    for row in rows:
        k=parse_relation(row["prompt"],str(row["answer"]))
        if k!=tuple(row["fact_key"]) or stable_split(k,42)!="train":
            raise RuntimeError("Benchmark fact/partition corrupted")
    import torch
    from model import GPT,GPTConfig
    from evaluate_ta_integer import greedy_original_stop
    from evaluate_round6_formats import parse_completion
    if args.device=="cuda" and not torch.cuda.is_available():
        ap.error("No GPU allocation. Use TC1 Slurm sbatch, not login node.")
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        meta=pickle.load(f)
    stoi,itos=meta["stoi"],meta["itos"]
    if any(set(r["prompt"])-set(stoi) for r in rows):
        raise RuntimeError("Benchmark has prompt chars missing from SFT tokenizer")
    out=(args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir).resolve()
    base=(ROOT/"dpo/experiments").resolve()
    if not out.is_relative_to(base) or out==base or out.exists():
        ap.error(f"Output must be a NEW folder under experiments: {out}")
    report={"benchmark":"teacher_style_400_v1",
            "questions_sha256":sha(BENCH),"seed":manifest["seed"],
            "decode":"greedy, model.py period/newline stop",
            "max_new_tokens":args.max_new_tokens,
            "scorer":"historical anchored The answer is <integer>; not verified TA parser",
            "test_policy":"DEVELOPMENT benchmark, sealed independent Test never accessed",
            "models":{}}
    for name in selected:
        ckpt_path=MODELS[name]
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Model {name}: {ckpt_path}")
    out.mkdir(parents=True,exist_ok=False)
    for name in selected:
        ckpt_path=MODELS[name]
        ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=True)
        model=GPT(GPTConfig(**ckpt["model_args"]))
        weights=ckpt.get("model_state_dict",ckpt.get("model"))
        if weights is None:raise RuntimeError("Checkpoint missing weights")
        model.load_state_dict({
            k.removeprefix("_orig_mod."):v for k,v in weights.items()
        },strict=True)
        model.to(args.device).eval()
        predictions=[]
        for i,r in enumerate(rows):
            completion,stop=greedy_original_stop(
                model,r["prompt"],stoi,itos,args.max_new_tokens,args.device)
            evaluation=parse_completion(completion,r["answer"])
            predictions.append({
                **r,"completion":completion,"stop_reason":stop,**evaluation})
            if (i+1)%80==0:
                current=predictions[-80:]
                print(f"{name} group={r['category']} current80 "
                      f"{summarize(current)['strict_correct']}/80",flush=True)
        split={}
        split["overall"]=summarize(predictions)
        split["by_category"]={}
        split["by_status"]={}
        split["by_subskill"]={}
        for cat in GROUPS:
            subset=[r for r in predictions if r["category"]==cat]
            split["by_category"][cat]=summarize(subset)
            split["by_subskill"][cat]={
                sub:summarize([r for r in subset if r["subskill"]==sub])
                for sub in sorted({r["subskill"] for r in subset})
            }
        for status in ("seen","unseen"):
            subset=[r for r in predictions if r["seen_status"]==status]
            split["by_status"][status]=summarize(subset)
        report["models"][name]={
            "checkpoint_sha256":sha(ckpt_path),
            "scores":split
        }
        with (out/f"{name}_predictions.json").open("x",encoding="utf-8") as f:
            json.dump(predictions,f,indent=2,ensure_ascii=False);f.write("\n")
        print(f"FINAL {name}: overall={split['overall']['strict_correct']}/400 "
              f"seen={split['by_status']['seen']['strict_correct']}/"
              f"{split['by_status']['seen']['n']} "
              f"unseen={split['by_status']['unseen']['strict_correct']}/"
              f"{split['by_status']['unseen']['n']}",flush=True)
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
    with (out/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write("\n")
    print("Saved:",out/"summary.json",flush=True)
    print("Do NOT use this frozen development benchmark's gold to construct training.")


if __name__=="__main__":main()
