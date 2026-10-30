#!/usr/bin/env python3
"""Reproduce earlier 2025 SC3000 GitHub blind tests on our three trained NanoGPT models.

Reference: https://github.com/JamesRobertJohns/3000
Fixed 61-question test (notebook cell 44, pre-extra-finetune checkpoint) and
41-question test (cell 61, after extra fine-tuning). The public notebook has
a single STORED stochastic sample per question, NOT an independent hidden test.

Our checkpoints are LOCAL to TC1. This script CANNOT produce new model results
without their files. It only reads frozen public questions & OUR training
pairs; it NEVER reads the sealed Round3 test.
Usage (run on TC1 with one GPU allocation):
  python scripts/benchmark_prior_year_blind.py --self-test-only
  python scripts/benchmark_prior_year_blind.py --device cuda --decode both
  python scripts/benchmark_prior_year_blind.py --device cuda --decode greedy
  python scripts/benchmark_prior_year_blind.py --device cuda --decode sampled

Generated outputs are isolated, never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))

REFERENCE=ROOT/"dpo/reference_2025_blind_sets.json"
TRAIN=ROOT/"dpo/data/round5_decomposition/original_pairs.json"
ROUND5_MANIFEST=ROOT/"dpo/data/round5_decomposition/manifest.json"
DEFAULT_OUT=ROOT/"dpo/experiments/prior_year_blind_2025"
CHECKPOINTS={
    "round5_A":ROOT/"dpo/experiments/round5_original/dpo.pt",
    "round6_A":ROOT/"dpo/experiments/round6_from_sft/original/dpo.pt",
    "round6_D":ROOT/"dpo/experiments/round6_from_sft/decomposition_before/dpo.pt",
}
GROUPS=("blind61","blind41")
CORRECT_RE=re.compile(r"The answer is\s+(-?\d+)",re.I)
TRAIN_POS_RE=re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>\d+)(?:\s|[.])")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_gold(prompt):
    """Use exact Fraction arithmetic, independently from source training generators."""
    from pipeline_smoke import verify_math
    if not isinstance(prompt,str) or not prompt.endswith("=?"):
        raise ValueError(f"Unrecognized published question: {prompt}")
    m=re.fullmatch(r"(\d+)([+\-*/])(\d+)=\?",prompt)
    if m:
        a,op,b=Fraction(m[1]),m[2],Fraction(m[3])
        value=(a+b if op=="+" else a-b if op=="-" else
               a*b if op=="*" else a/b if b else None)
        if value is None:raise ValueError("Division by zero")
        if value.denominator!=1:raise ValueError("Unexpected fractional gold")
        return str(value.numerator)
    m=re.fullmatch(r"(x|\d+)([+\-*/])(x|\d+)=(\d+),x=\?",prompt)
    if m is None:raise ValueError(f"Unsupported published question: {prompt}")
    l,op,r,c=m.groups()
    c=Fraction(c)
    if l=="x" and r!="x":
        n=Fraction(r)
        x=(c-n if op=="+" else c+n if op=="-" else
           c/n if op=="*" and n else c*n if op=="/" else None)
    elif r=="x" and l!="x":
        n=Fraction(l)
        x=(c-n if op=="+" else n-c if op=="-" else
           c/n if op=="*" and n else n/c if op=="/" and c else None)
    else:
        raise ValueError(f"Equation must have exactly one x: {prompt}")
    if x is None or x.denominator!=1 or not verify_math(prompt,str(x)):
        raise ValueError(f"Invalid mathematical answer for {prompt}: {x}")
    return str(x.numerator)


def validate_reference(data):
    if data["meta"]["source_path"]!="src/dpo/dpo.ipynb":
        raise ValueError("Wrong reference source")
    all_questions=[]
    expected={"blind61":(61,52),"blind41":(41,39)}
    for group in GROUPS:
        rows=data["groups"][group]["records"]
        n,correct=expected[group]
        if len(rows)!=n:raise ValueError(f"Bad {group} length")
        if len({r["prompt"] for r in rows})!=n:
            raise ValueError(f"Duplicate test question inside {group}")
        observed=0
        for r in rows:
            gold=independent_gold(r["prompt"])
            if r["answer"]!=gold:
                raise ValueError(f"Published gold mismatch: {r['prompt']}")
            if r["prompt"] not in r["reference_generated"]:
                raise ValueError(f"Printed reference output not for {r['prompt']}")
            match=CORRECT_RE.search(r["reference_generated"])
            if not match or match.group(1)!=r["reference_prediction"]:
                raise ValueError(f"Incorrect reference extraction: {r['prompt']}")
            was_correct=match.group(1)==gold
            if was_correct!=r["reference_correct"]:
                raise ValueError(f"Reference output score mismatch: {r['prompt']}")
            observed+=was_correct
            all_questions.append(r["prompt"])
        if observed!=correct:
            raise ValueError(f"Reference score expected {correct}, got {observed}")
    if len(set(all_questions))!=101:
        raise ValueError("Expected 101 distinct prompts across two published blind sets")
    print("PASS: reference source outputs independently verified "
          "blind61=52/61 and blind41=39/41; 101 unique prompts.")


def exact_fact(prompt,answer):
    from generate_round1 import parse_relation
    return parse_relation(prompt,str(answer))


def read_training_facts():
    m=read(ROUND5_MANIFEST)
    if sha(TRAIN)!=m["file_sha256"]["original"]:
        raise ValueError("Round5 original training dataset SHA mismatch")
    pairs=read(TRAIN)
    if len(pairs)!=100000:raise ValueError("Expected Round4 treatment source 100k")
    facts=set()
    prompts=set()
    prompt_counts=Counter()
    for i,pair in enumerate(pairs):
        pos=pair["positive"]
        match=TRAIN_POS_RE.match(pos)
        if match is None:raise ValueError(f"Training format not recognized at {i}")
        prompt,answer=match.group("prompt"),match.group("answer")
        facts.add(exact_fact(prompt,answer))
        prompts.add(prompt)
        prompt_counts[prompt]+=1
    print(f"PASS: original Round5 training {len(pairs)} records, "
          f"{len(prompts)} unique prompts, {len(facts)} unique math facts.")
    return facts,prompts,prompt_counts


def attach_overlap(reference,facts,prompts,prompt_counts):
    result={}
    for name in GROUPS:
        annotated=[]
        for r in reference["groups"][name]["records"]:
            prompt,gold=r["prompt"],r["answer"]
            key=exact_fact(prompt,gold)
            annotated.append({
                **r,
                "fact_key":list(key),
                "in_training_prompt":prompt in prompts,
                "in_training_fact":key in facts,
                "training_prompt_exposures":prompt_counts[prompt],
            })
        result[name]=annotated
        seen_prompt=sum(r["in_training_prompt"] for r in annotated)
        seen_fact=sum(r["in_training_fact"] for r in annotated)
        print(f"TRAINING OVERLAP {name}: identical prompt={seen_prompt}/{len(annotated)}, "
              f"same mathematical fact incl inverse variants={seen_fact}/{len(annotated)}, "
              f"unseen math facts={len(annotated)-seen_fact}/{len(annotated)}")
    return result


def load_model(path,device):
    import torch
    from model import GPT,GPTConfig
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint unavailable on TC1: {path}")
    ckpt=torch.load(path,map_location="cpu",weights_only=True)
    weights=ckpt.get("model_state_dict",ckpt.get("model"))
    if weights is None:raise ValueError(f"Checkpoint contains no model weights: {path}")
    weights={name.removeprefix("_orig_mod."):value for name,value in weights.items()}
    model=GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(weights,strict=True)
    model.to(device).eval()
    return model


def generate(model,prompt,stoi,itos,device,mode,limit,seed):
    import torch
    from evaluate_ta_integer import greedy_original_stop
    if mode=="greedy":
        return greedy_original_stop(model,prompt,stoi,itos,limit,device)
    # The source notebook uses model.generate with temperature=.8/top_k=200.
    # Seed per question makes stochastic model comparisons repeatable; their
    # reference publication did NOT expose its own generation RNG stream.
    torch.manual_seed(seed)
    if device=="cuda":torch.cuda.manual_seed_all(seed)
    ids=torch.tensor([[stoi[c] for c in prompt]],dtype=torch.long,device=device)
    with torch.inference_mode():
        output=model.generate(ids, max_new_tokens=limit,temperature=0.8,top_k=200)
    result=output[0][0].tolist()
    return "".join(itos[i] for i in result[len(prompt):]),"model.generate_sampling"


def report_slice(rows):
    c=Counter()
    for r in rows:
        c["n"]+=1
        c["correct"]+=r["strict_status"]=="correct"
        c["reference_correct"]+=r["reference_correct"]
        c["unseen_fact"]+=not r["in_training_fact"]
        if r["in_training_fact"]:
            c["seen_fact_n"]+=1
            c["seen_fact_correct"]+=r["strict_status"]=="correct"
        else:
            c["unseen_fact_n"]+=1
            c["unseen_fact_correct"]+=r["strict_status"]=="correct"
        c["format_failure"]+=r["strict_status"]=="format_failure"
        c["final_aware_correct"]+=r["final_status"]=="correct"
        c["reason_before_answer"]+=r["output_mode"]=="reason_before_answer"
        c["valid_calculation"]+=r["calculation_valid"]
    return dict(c)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--decode",choices=("greedy","sampled","both"),default="both")
    ap.add_argument("--max-new-tokens",type=int,default=200)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--self-test-only",action="store_true")
    args=ap.parse_args()
    reference=read(REFERENCE)
    validate_reference(reference)
    if args.self_test_only:
        return
    if args.max_new_tokens<1:ap.error("max_new_tokens must be positive")
    out=(args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir).resolve()
    # Outputs must be new and kept under experiments; never overwrite prior results.
    if out==ROOT or not out.is_relative_to((ROOT/"dpo/experiments").resolve()):
        ap.error("Output must be a NEW folder under dpo/experiments/")
    if out.exists():ap.error(f"Refusing to overwrite existing evaluation folder: {out}")
    facts,prompts,prompt_counts=read_training_facts()
    records=attach_overlap(reference,facts,prompts,prompt_counts)
    import torch
    from evaluate_round6_formats import parse_completion
    if args.device=="cuda" and not torch.cuda.is_available():
        ap.error("No CUDA device allocated; submit a Slurm GPU job")
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        meta=pickle.load(f)
    stoi,itos=meta["stoi"],meta["itos"]
    for group,rs in records.items():
        for r in rs:
            unknown=set(r["prompt"])-set(stoi)
            if unknown:raise ValueError(f"Model tokenizer lacks {unknown}: {r['prompt']}")
    out.mkdir(parents=True,exist_ok=False)
    (out/"source_reference_and_leakage.json").write_text(
        json.dumps({"reference":reference["meta"],"groups":records,
                    "shared_training_source":str(TRAIN),
                    "training_sha256":sha(TRAIN)},ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )
    modes=("greedy","sampled") if args.decode=="both" else (args.decode,)
    overall={}
    for model_name,checkpoint in CHECKPOINTS.items():
        model=load_model(checkpoint,args.device)
        model_info={"checkpoint":str(checkpoint),"sha256":sha(checkpoint)}
        for mode in modes:
            grouped={}
            for group,rs in records.items():
                outputs=[]
                for i,r in enumerate(rs):
                    question=r["prompt"]
                    answer=r["answer"]
                    completion,stop=generate(model,question,stoi,itos,args.device,mode,
                                             args.max_new_tokens,args.seed+i)
                    parsed=parse_completion(completion,answer)
                    outputs.append({
                        "prompt":question,"answer":answer,
                        "reference_correct":r["reference_correct"],
                        "reference_prediction":r["reference_prediction"],
                        "in_training_prompt":r["in_training_prompt"],
                        "in_training_fact":r["in_training_fact"],
                        "training_prompt_exposures":r["training_prompt_exposures"],
                        "completion":completion,"stop_reason":stop,
                        **parsed
                    })
                report=report_slice(outputs)
                grouped[group]=report
                (out/f"{model_name}_{mode}_{group}_predictions.json").write_text(
                    json.dumps(outputs,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                print(f"{model_name:12} {mode:8} {group:7}: "
                      f"PRIMARY={report['correct']}/{report['n']} "
                      f"({100*report['correct']/report['n']:.2f}%), "
                      f"published_reference={report['reference_correct']}/{report['n']}, "
                      f"unseen_fact={report.get('unseen_fact_correct',0)}/"
                      f"{report.get('unseen_fact_n',0)}, "
                      f"exact_math_fact_seen={report.get('seen_fact_n',0)}/{report['n']}",
                      flush=True)
            overall[f"{model_name}_{mode}"]={"model":model_info,"groups":grouped}
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
    report={
        "benchmark":"2025_prior_year_published_blind61_and_blind41",
        "reference_source":reference["meta"],
        "evaluation":{
            "device":args.device,"mode":args.decode,
            "max_new_tokens":args.max_new_tokens,"seed":args.seed,
            "sampling_temperature":0.8,"sampling_top_k":200,
            "primary":"historical strict answer-first first numeric answer; same for ALL models",
            "secondary":"final-answer aware diagnostic for reasoning-first; not TA grading",
        },
        "training_source_sha256":sha(TRAIN),
        "models":overall,
    }
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SAVED:",out/"summary.json")
    print("Note: 61 and 41 are different published reference model stages; "
          "not official TA tests, and user train-fact overlap affects fairness.")
    print("The same legacy primary scoring is used for Round5-A, Round6-A, Round6-D.")


if __name__=="__main__":
    main()
