#!/usr/bin/env python3
"""Round8: mine OUR model's real wrong answers on TRAIN-only facts and nearby TRAIN-only number variants.

Read-only inference, no training. GPU uses frozen Round7 Replay checkpoint.
Never use Val/OOD/Extended or public prior-year benchmark examples as mining
questions. Public sources are used ONLY to exclude their canonical math facts.
The independent sealed Round3 final Test is NEVER read.

Small pilot:
    python scripts/mine_round8_train_hard_negatives.py --self-test-only
    sbatch scripts/round8_mining.sbatch
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
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
SEED=42
BASE=ROOT/"dpo/experiments/round7_replay/dpo.pt"
DATA=ROOT/"dpo/data/round5_decomposition/original_pairs.json"
MANIFEST=ROOT/"dpo/data/round5_decomposition/manifest.json"
R3=ROOT/"dpo/data/round3_structured_full"
R4=ROOT/"dpo/data/round4_matched_full"
OOD=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"
PUBLIC=ROOT/"dpo/reference_2025_blind_sets.json"
OUT=ROOT/"dpo/experiments/round8_hard_negative_mining_seed42"

POS_RE=re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>\d+) because .+\.$")
HARD_RE=re.compile(
    r"^\s*The answer is\s+(\d+)\s+because\s+[^.\n]+\.\s*$",
    flags=re.IGNORECASE
)
CATEGORIES=(
    "add","subtract","multiply","divide","x_plus_a","a_plus_x",
    "x_minus_a","a_minus_x","x_times_a","a_times_x","x_over_a","a_over_x"
)
OPS={"+":"plus","-":"minus","*":"times","/":"over"}
ARITH=re.compile(r"^(\d+)([+\-*/])(\d+)=\?$")
EQ=re.compile(r"^(\d+|x)([+\-*/])(\d+|x)=(\d+),x=\?$")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kind(prompt):
    m=ARITH.fullmatch(prompt)
    if m:return {"+":"add","-":"subtract","*":"multiply","/":"divide"}[m[2]]
    m=EQ.fullmatch(prompt)
    if m:
        l,op,r,_=m.groups()
        if (l=="x") == (r=="x"):raise ValueError(prompt)
        return ("x_"+OPS[op]+"_a" if l=="x" else
                "a_"+OPS[op]+"_x")
    raise ValueError(f"Unsupported train-only prompt: {prompt}")


def load_train_only():
    from generate_round1 import parse_relation,RESERVED
    m=read(MANIFEST)
    if sha(DATA)!=m["file_sha256"]["original"]:
        raise RuntimeError("Round5 original 100k source changed")
    r3m=read(R3/"manifest.json")
    r4m=read(R4/"manifest.json")
    if sha(R3/"val.json")!=r3m["sha256"]["val"]:
        raise RuntimeError("Frozen val checksum failed")
    if sha(OOD)!=r4m["source_sha256"]["old_val_sha256"]:
        raise RuntimeError("Frozen OOD checksum failed")
    if sha(R4/"extension_val.json")!=r4m["file_sha256"]["extension_val"]:
        raise RuntimeError("Frozen extension checksum failed")
    protected=set()
    for path in (R3/"val.json",OOD,R4/"extension_val.json"):
        for r in read(path):
            protected.add(parse_relation(r["prompt"],str(r["answer"])))
    for p,a in RESERVED.items():
        if "." not in p+a:
            protected.add(parse_relation(p,a))
    public=read(PUBLIC)
    public_facts=set()
    for group in ("blind61","blind41"):
        for row in public["groups"][group]["records"]:
            public_facts.add(parse_relation(row["prompt"],str(row["answer"])))
    protected |= public_facts
    lines=read(DATA)
    if len(lines)!=100000:
        raise RuntimeError("Expected exactly 100,000 original training preference rows")
    facts=set()
    by_category=defaultdict(dict)
    for i,p in enumerate(lines):
        m=POS_RE.fullmatch(p["positive"])
        if not m:raise ValueError(f"Unexpected Round5-A Positive at row {i}")
        prompt,answer=m["prompt"],m["answer"]
        from generate_round1 import parse_relation
        fact=parse_relation(prompt,answer)
        facts.add(fact)
        if fact in protected:continue
        cat=kind(prompt)
        if p["negative"]!=f"{prompt} Sorry, I don't know.":
            raise ValueError(f"Source negative changed at row {i}")
        by_category[cat].setdefault(prompt,{
            "prompt":prompt,"answer":answer,"category":cat,
            "fact_key":list(fact),"positive":p["positive"],
            "source_train_row":i
        })
    return by_category,facts,protected,{
        "round5_pairs_sha256":sha(DATA),"round5_manifest_sha256":sha(MANIFEST),
        "checkpoint_sha256":sha(BASE),
        "round3_val_sha256":sha(R3/"val.json"),
        "old_ood_sha256":sha(OOD),
        "extended_val_sha256":sha(R4/"extension_val.json"),
        "public_exclusion_file_sha256":sha(PUBLIC),
        "public_protected_fact_count":len(public_facts),
        "total_original_train_facts":len(facts),
    }


def nearby_facts(seed,train_facts,excluded,rng,max_candidates=64):
    """Change arithmetic operands by <=3; preserve the original question form and bounded difficulty.

    Changing a digit is NOT inserting the original wrong validation question.
    Every new math fact must hash to the original train partition and be absent
    from previous train facts and ALL held-out/public protected math facts.
    """
    from generate_round1 import stable_split,parse_relation
    from generate_round3_structured import sum_relations,product_relations
    old=tuple(seed["fact_key"])
    fam,a,b,_=old
    if fam not in ("mul","add"):return []
    a,b=int(a),int(b)
    category=seed["category"]
    candidates=[]
    for da in (-3,-2,-1,0,1,2,3):
        for db in (-3,-2,-1,0,1,2,3):
            if da==db==0:continue
            x,y=a+da,b+db
            if fam=="mul":
                if x<2 or y<2 or x>99 or y>99 or not 101<=x*y<=999:
                    continue
                rows=product_relations(x,y)
            else:
                if x<0 or y<0 or x>99 or y>99 or x+y>199:
                    continue
                rows=sum_relations(x,y)
            for row in rows:
                if row["category"]!=category:continue
                fact=parse_relation(row["prompt"],str(row["answer"]))
                if (fact in excluded or fact in train_facts or
                    stable_split(fact,SEED)!="train"):
                    continue
                candidates.append({
                    "prompt":row["prompt"],"answer":row["answer"],
                    "category":category,"fact_key":list(fact),
                    "positive":row["positive"],
                    "source_error_prompt":seed["prompt"],
                    "source_error_category":category,
                    "source_error_model_prediction":seed["prediction"]
                })
    rng.shuffle(candidates)
    unique=[]
    seen=set()
    for row in candidates:
        fact=tuple(row["fact_key"])
        if fact in seen:continue
        unique.append(row)
        seen.add(fact)
        if len(unique)>=max_candidates:break
    return unique


def parse_model_error(completion,answer):
    """Use the model's ACTUAL generated entire answer, not a fabricated corruption."""
    m=HARD_RE.fullmatch(completion)
    if not m:return None
    n=int(m[1])
    if n==int(answer):return None
    return n


def mine(args):
    import torch
    from model import GPT,GPTConfig
    from evaluate_ta_integer import greedy_original_stop
    from generate_round1 import parse_relation
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU not allocated; run via sbatch")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing mining results: {OUT}")
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    if args.per_category<1 or args.variants_per_error<1:
        raise ValueError("Positive sampling quota needed")
    groups,train_facts,protected,checksums=load_train_only()
    rng=random.Random(args.seed)
    source=[]
    for category in CATEGORIES:
        pool=list(groups[category].values())
        rng.shuffle(pool)
        source.extend(pool[:args.per_category])
        print(f"train-only eligible unique prompts: {category:14} {len(pool):5}; "
              f"mining={min(len(pool),args.per_category)}",flush=True)
    rng.shuffle(source)
    print(f"Total train-only mining prompts={len(source)}; "
          f"protected canonical facts={len(protected)}",flush=True)
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        meta=pickle.load(f)
    stoi,itos=meta["stoi"],meta["itos"]
    ckpt=torch.load(BASE,map_location="cpu",weights_only=True)
    model=GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt.get("model_state_dict",ckpt.get("model")),strict=True)
    model.to(args.device).eval()
    def answer(row):
        if set(row["prompt"])-set(stoi):
            raise ValueError(f"Tokenizer unknown input: {row['prompt']}")
        completion,stop=greedy_original_stop(
            model,row["prompt"],stoi,itos,args.max_new_tokens,args.device)
        prediction=parse_model_error(completion,row["answer"])
        return completion,stop,prediction

    stats=Counter()
    original_wrong=[]
    original_pilot=[]
    for i,row in enumerate(source):
        completion,stop,prediction=answer(row)
        stats["mined_original_questions"]+=1
        if prediction is None:
            if "The answer is" in completion:
                stats["original_correct_or_invalid"]+=1
            else:stats["original_no_answer"]+=1
        else:
            stats["original_strict_form_wrong"]+=1
            record={**row,"completion":completion,"stop_reason":stop,
                    "prediction":prediction,
                    "negative":row["prompt"]+
                               (completion if completion[:1].isspace() else " "+completion)}
            if (record["negative"]==record["positive"] or
                not parse_relation(record["prompt"],record["answer"])==
                   tuple(record["fact_key"])):
                raise RuntimeError("Invalid mined paired preference")
            if max(len(record["negative"]),len(record["positive"]))+4>256:
                stats["original_over_context"]+=1
            else:
                original_wrong.append(record)
        if i<8:
            original_pilot.append({"prompt":row["prompt"],"gold":row["answer"],
                                   "generated":completion,"wrong":prediction is not None})
        if (i+1)%200==0:
            print(f"MINED {i+1}/{len(source)}, strict numeric wrong and well-formed="
                  f"{len(original_wrong)}",flush=True)

    # Numerical transformations are driven only by patterns of TRAIN questions
    # that the model actually answered wrongly. No validation/public prompt is
    # used as a seed. New relations are constrained to products <=999.
    variants=[]
    seen_variant_facts=set()
    for seed in original_wrong:
        nearby=nearby_facts(seed,train_facts,protected|seen_variant_facts,rng)
        accepted=0
        for row in nearby:
            fact=tuple(row["fact_key"])
            if fact in seen_variant_facts:continue
            seen_variant_facts.add(fact)
            variants.append(row)
            accepted+=1
            if accepted>=args.variants_per_error:
                break
    rng.shuffle(variants)
    # Keep only actually INCORRECT model responses as hard-negative pairs.
    variation_wrong=[]
    for i,row in enumerate(variants):
        completion,stop,prediction=answer(row)
        stats["mined_variant_questions"]+=1
        if prediction is not None:
            record={**row,"completion":completion,"stop_reason":stop,
                    "prediction":prediction,
                    "negative":row["prompt"]+
                               (completion if completion[:1].isspace() else " "+completion)}
            if max(len(record["positive"]),len(record["negative"]))+4>256:
                stats["variant_over_context"]+=1
            else:
                variation_wrong.append(record)
                stats["variant_strict_form_wrong"]+=1
        if (i+1)%200==0:
            print(f"VARIANTS {i+1}/{len(variants)} tested; "
                  f"real well-formed wrong={len(variation_wrong)}",flush=True)

    # Defensive cross-file, exact gold and anti-leakage audits.
    for group in (original_wrong,variation_wrong):
        for row in group:
            k=tuple(row["fact_key"])
            if k in protected or row["positive"]==row["negative"]:
                raise RuntimeError("Protected fact in mined training, or invalid preference")
            if parse_relation(row["prompt"],row["answer"])!=k:
                raise RuntimeError("Incorrect gold labels")
            if parse_model_error(row["completion"],row["answer"])!=row["prediction"]:
                raise RuntimeError("Negative was not a verbatim model numerical error")
    output={
        "version":"round8_train_only_mined_errors_v1",
        "source_model":"round7_replay",
        "seed":args.seed,"per_category":args.per_category,
        "variants_per_error":args.variants_per_error,
        "max_new_tokens":args.max_new_tokens,
        "source_and_protection_sha256":checksums,
        "original_source_questions":len(source),
        "original_errors":len(original_wrong),
        "probed_nearby_variations":len(variants),
        "variation_errors":len(variation_wrong),
        "mined_source_error_category":dict(Counter(r["category"] for r in original_wrong)),
        "mined_variant_error_category":dict(Counter(r["category"] for r in variation_wrong)),
        "counters":dict(stats),
        "controls":"All evaluation and published external facts excluded. Sealed independent Test unopened. Numerically wrong answers generated by frozen base model; no synthetic invented Rejected."
    }
    OUT.mkdir(parents=True,exist_ok=False)
    for name,value in (("manifest.json",output),
                       ("original_hard_negatives.json",original_wrong),
                       ("variant_hard_negatives.json",variation_wrong),
                       ("sample_generation_preview.json",original_pilot)):
        path=OUT/name
        with path.open("x",encoding="utf-8") as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n")
    print("ROUND8 TRAIN-ONLY ERROR MINING COMPLETE")
    print(f"Original model-generated incorrect responses: {len(original_wrong)}")
    print(f"New small-number variations PROBED: {len(variants)}")
    print(f"Variants with real incorrect model responses: {len(variation_wrong)}")
    print("Outputs:",OUT,flush=True)
    print("NO training performed. No Val/OOD/Extended/public blind gold used as target.")
    print("NEXT: inspect representative errors and build balanced control/treatment.")


def selftest():
    from generate_round1 import parse_relation,stable_split
    for prompt,ans,cat in (
        ("47*13=?","611","multiply"),
        ("x/13=47,x=?","611","x_over_a"),
        ("84/7=?","12","divide"),
        ("17+19=?","36","add"),
    ):
        if kind(prompt)!=cat:raise AssertionError((prompt,cat,kind(prompt)))
        if not parse_relation(prompt,ans):
            raise AssertionError(prompt)
    assert parse_model_error(" The answer is 601 because 47*13 equals 601.","611")==601
    assert parse_model_error(" The answer is 611 because 47*13 equals 611.","611") is None
    assert parse_model_error(" 47*10=470 so The answer is 601.","611") is None
    example={
        "prompt":"47*13=?","answer":"611","category":"multiply",
        "fact_key":list(parse_relation("47*13=?","611")),
        "prediction":601
    }
    variants=nearby_facts(example,set(),set(),random.Random(42))
    assert variants
    assert all(int(r["fact_key"][3])<=999 for r in variants)
    assert all(stable_split(tuple(r["fact_key"]),42)=="train" for r in variants)
    assert all(r["positive"].startswith(r["prompt"]+" The answer is ") for r in variants)
    assert all(r["negative"]=="Sorry, I don't know." if "negative" in r else True for r in variants)
    print(f"PASS: exact fact, correct category, strict self-generated error, "
          f"TRAIN-split bounded math variants ({len(variants)} candidates), no GPU")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--per-category",type=int,default=120,
                    help="Max unique original TRAIN prompts per each of 12 categories")
    ap.add_argument("--variants-per-error",type=int,default=2)
    ap.add_argument("--max-new-tokens",type=int,default=80)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--self-test-only",action="store_true")
    args=ap.parse_args()
    if args.self_test_only:
        selftest()
        return
    mine(args)


if __name__=="__main__":
    main()
