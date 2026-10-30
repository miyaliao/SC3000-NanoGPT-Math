#!/usr/bin/env python3
"""Uniform score on the two immutable teacher-style benchmark files (400+280).

DEVELOPMENT benchmark only. Both candidate checkpoints receive EXACT same
680 prompts, greedy decoding, model.py original stop and strict answer-first
grading. Sealed independent Test and prior-year public tests are not read.

Slurm: sbatch scripts/teacher_style_680_eval.sbatch
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
CORE=ROOT/"dpo/benchmarks/teacher_style_400_v1"
EXT=ROOT/"dpo/benchmarks/teacher_style_extension_280_v1"
OUT=ROOT/"dpo/experiments/teacher_style_680_v1_eval"
MODELS={
    "round5_A":ROOT/"dpo/experiments/round5_original/dpo.pt",
    "round7_replay":ROOT/"dpo/experiments/round7_replay/dpo.pt",
    "round6_A":ROOT/"dpo/experiments/round6_from_sft/original/dpo.pt",
    "round6_D":ROOT/"dpo/experiments/round6_from_sft/decomposition_before/dpo.pt",
}
CORE_GROUPS=("add","multiply","divide","a_minus_x","x_times_a")
EXT_GROUPS=("subtract","x_plus_a","a_plus_x","x_minus_a","a_times_x","x_over_a","a_over_x")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frozen(path,version,n,unique_key):
    p=path/"questions.json"
    manifest=path/"manifest.json"
    if not p.exists() or not manifest.exists():
        raise FileNotFoundError(f"First freeze the two benchmark parts: {path}")
    meta=read(manifest);records=read(p)
    if meta.get("version")!=version or len(records)!=n:
        raise RuntimeError(f"Incorrect frozen benchmark version or size {path}")
    if sha(p)!=meta["questions_sha256"]:
        raise RuntimeError(f"Frozen benchmark checksum mismatch {path}")
    if len({r["prompt"] for r in records})!=n:
        raise RuntimeError(f"Benchmark contains repeated prompt {path}")
    if len({tuple(r["fact_key"]) for r in records})!=meta[unique_key]:
        raise RuntimeError(f"Benchmark unique fact count changed: {path}")
    return records,meta

def validate():
    from generate_round1 import parse_relation,stable_split
    from pipeline_smoke import verify_math
    core,cm=frozen(CORE,"teacher_style_400_v1",400,"unique_fact_keys_total")
    ext,em=frozen(EXT,"teacher_style_extension_280_v1",280,"unique_math_facts_total")
    if em["core_reference"]["core_questions_sha256"]!=sha(CORE/"questions.json"):
        raise RuntimeError("Extension was frozen against a different Core400")
    if em["core_reference"]["core_manifest_sha256"]!=sha(CORE/"manifest.json"):
        raise RuntimeError("Core400 manifest was modified after extension selection")
    if len({r["prompt"] for r in core+ext})!=680:
        raise RuntimeError("Core+Extension duplicate prompt")
    core_facts={tuple(r["fact_key"]) for r in core}
    ext_facts={tuple(r["fact_key"]) for r in ext}
    if core_facts & ext_facts:
        raise RuntimeError("Core and Extension share canonical arithmetic facts")
    if [r["category"] for r in core] != sorted(
        [r["category"] for r in core],key=lambda k:CORE_GROUPS.index(k)
    ):
        raise RuntimeError("Core group order has changed")
    groups=Counter(r["category"] for r in core+ext)
    for cat in CORE_GROUPS:
        if groups[cat]!=80:raise RuntimeError(f"Wrong core group size {cat}")
    for cat in EXT_GROUPS:
        if groups[cat]!=40:raise RuntimeError(f"Wrong extension group size {cat}")
    for label,rr in (("core400",core),("extension280",ext)):
        for r in rr:
            f=parse_relation(r["prompt"],str(r["answer"]))
            if (f!=tuple(r["fact_key"]) or stable_split(f,42)!="train" or
                not verify_math(r["prompt"],r["answer"])):
                raise RuntimeError(f"Invalid frozen benchmark item {label}: {r['prompt']}")
    print("PASS frozen Core400 + Extension280: 680 different prompts, "
          f"{len(core_facts)+len(ext_facts)} unique disjoint math facts; "
          "two independent manifests intact.")
    return core,ext,cm,em

def stats(rows):
    n=len(rows)
    correct=sum(r["strict_status"]=="correct" for r in rows)
    return {"n":n,"correct":correct,
            "accuracy":correct/n if n else None,
            "math_error":sum(r["strict_status"]=="math_error" for r in rows),
            "format_failure":sum(r["strict_status"]=="format_failure" for r in rows)}

def analyze(records):
    result={"overall":stats(records),"by_section":{},"by_category":{},
            "by_category_subskill":{},"by_seen_status":{},"by_section_seen_status":{}}
    for section in ("core400","extension280"):
        subset=[x for x in records if x["section"]==section]
        result["by_section"][section]=stats(subset)
        result["by_section_seen_status"][section]={
            label:stats([x for x in subset if x["seen_status"]==label])
            for label in ("seen","unseen")
        }
    for cat in CORE_GROUPS+EXT_GROUPS:
        subset=[x for x in records if x["category"]==cat]
        result["by_category"][cat]=stats(subset)
        result["by_category_subskill"][cat]={
            sub:stats([x for x in subset if x["subskill"]==sub])
            for sub in sorted({x["subskill"] for x in subset})
        }
    for label in ("seen","unseen"):
        result["by_seen_status"][label]=stats(
            [x for x in records if x["seen_status"]==label])
    return result

def diff(a,b):
    if len(a)!=len(b):raise RuntimeError("Models have different prompt counts")
    c=Counter()
    for x,y in zip(a,b):
        if x["prompt"]!=y["prompt"] or x["answer"]!=y["answer"]:
            raise RuntimeError("Question list differs between checkpoints")
        left=x["strict_status"]=="correct"
        right=y["strict_status"]=="correct"
        c["n"]+=1;c["a_correct"]+=left;c["b_correct"]+=right
        c["a_only"]+=left and not right;c["b_only"]+=right and not left
    return dict(c)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--models",default="round5_A,round7_replay",
                   help="Comma-separated known checkpoint aliases")
    p.add_argument("--out-dir",type=Path,default=OUT)
    p.add_argument("--check-only",action="store_true",
                   help="Validate both frozen benchmark parts without GPU inference")
    args=p.parse_args()
    core,ext,cm,em=validate()
    if args.check_only:
        print("PASS 680 read-only benchmark freeze validation; no GPU run")
        return
    selected=[n.strip() for n in args.models.split(",")]
    if (not selected or len(set(selected))!=len(selected) or
        any(n not in MODELS for n in selected)):
        p.error("Unknown or duplicate model name")
    if args.max_new_tokens<1:p.error("Max new tokens must be positive")
    output=(args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir).resolve()
    experiments=(ROOT/"dpo/experiments").resolve()
    if not output.is_relative_to(experiments) or output==experiments or output.exists():
        p.error(f"Refusing existing/outside output directory {output}")
    import torch
    from model import GPT,GPTConfig
    from evaluate_ta_integer import greedy_original_stop
    from train_round3_course import evaluate_text
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("No CUDA GPU allocation; run through Slurm")
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        meta=pickle.load(f)
    stoi,itos=meta["stoi"],meta["itos"]
    all_rows=[dict(x,section="core400") for x in core]+[
        dict(x,section="extension280") for x in ext]
    if any(set(row["prompt"])-set(stoi) for row in all_rows):
        raise RuntimeError("Unknown tokenizer characters in benchmark")
    for name in selected:
        if not MODELS[name].is_file():raise FileNotFoundError(MODELS[name])
    output.mkdir(parents=True,exist_ok=False)
    report={
        "benchmark":"teacher_style_680_v1","core_questions_sha256":sha(CORE/"questions.json"),
        "extension_questions_sha256":sha(EXT/"questions.json"),
        "max_new_tokens":args.max_new_tokens,
        "decode":"greedy NanoGPT, stop on first period or newline",
        "primary_scorer":"strict The answer is <integer> at beginning of completion, NOT verified TA",
        "benchmark_role":"fixed teacher-style development competency suite; seen facts are NOT unseen generalization",
        "models":{},"paired":{}
    }
    predictions_by_model={}
    for name in selected:
        file=MODELS[name]
        ckpt=torch.load(file,map_location="cpu",weights_only=True)
        model=GPT(GPTConfig(**ckpt["model_args"]))
        weights=ckpt.get("model_state_dict",ckpt.get("model"))
        if weights is None:raise RuntimeError(f"Missing model weights {file}")
        model.load_state_dict({k.removeprefix("_orig_mod."):v
                               for k,v in weights.items()},strict=True)
        model.to(args.device).eval()
        outputs=[]
        for i,row in enumerate(all_rows):
            completion,stop=greedy_original_stop(
                model,row["prompt"],stoi,itos,args.max_new_tokens,args.device)
            status,prediction,proof=evaluate_text(completion,row["answer"])
            outputs.append({
                **row,"completion":completion,"stop_reason":stop,
                "strict_status":status,"strict_prediction":prediction,
                "explanation_valid":proof
            })
            if (i+1)%40==0:
                previous=outputs[-40:]
                print(f"{name:14} section={row['section']:12} "
                      f"category={row['category']:12} "
                      f"correct={stats(previous)['correct']:2}/40",flush=True)
        scores=analyze(outputs)
        predictions_by_model[name]=outputs
        report["models"][name]={"checkpoint_sha256":sha(file),
                                "scores":scores}
        with (output/f"{name}_predictions.json").open("x",encoding="utf-8") as f:
            json.dump(outputs,f,indent=2,ensure_ascii=False);f.write("\n")
        print(f"FINAL {name:14} total={scores['overall']['correct']}/680 "
              f"core={scores['by_section']['core400']['correct']}/400 "
              f"extension={scores['by_section']['extension280']['correct']}/280 "
              f"seen={scores['by_seen_status']['seen']['correct']}/"
              f"{scores['by_seen_status']['seen']['n']}, "
              f"unseen={scores['by_seen_status']['unseen']['correct']}/"
              f"{scores['by_seen_status']['unseen']['n']}",flush=True)
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
    if len(selected)==2:
        a,b=selected
        for section in ("all","core400","extension280"):
            left=[r for r in predictions_by_model[a]
                  if section=="all" or r["section"]==section]
            right=[r for r in predictions_by_model[b]
                   if section=="all" or r["section"]==section]
            d=diff(left,right)
            report["paired"][section]={"left":a,"right":b,**d}
            print(f"PAIRED {section}: {a}={d['a_correct']} {b}={d['b_correct']}, "
                  f"{b}-{a}={d['b_correct']-d['a_correct']:+d} "
                  f"({a}-only={d['a_only']}; {b}-only={d['b_only']})")
    with (output/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write("\n")
    print("SAVED",output/"summary.json")

if __name__=="__main__": main()
