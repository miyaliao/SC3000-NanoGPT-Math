#!/usr/bin/env python3
"""Round9: genuine wrong-answer mining and matched HardNeg dataset assembly.

--mine pilot  : small stratified read-only GPU pilot (up to 50 unique prompts/category)
--mine full   : score every eligible unique TRAIN prompt once, not every repeated row
--assemble    : CPU-only, change ONLY Rejected for <=5000 structured rows
No evaluation question is used for mining. Frozen Round7 Replay never modified.
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
from generate_round1 import parse_relation,stable_split
from generate_round3_structured import REFUSAL,KINDS
from build_teacher_style_400 import source_and_protected
from evaluate_teacher_style_680 import validate as validate_680
from mine_round8_train_hard_negatives import kind,parse_model_error
from train_round3_course import evaluate_text

D=ROOT/"dpo/data/round9_balanced_full"
BASE=ROOT/"dpo/experiments/round7_replay/dpo.pt"
OUT=ROOT/"dpo/experiments/round9_mining"
POS_RE=re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>-?\d+) because .+\.$")
MAX_REPLACEMENTS=5000
MAX_PER_UNIQUE_PROMPT=5


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blob(o):return (json.dumps(o,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")


def inputs():
    m=read(D/"manifest.json")
    if m["version"]!="round9_structured_v1" or m["train_rows"]!=100000:
        raise RuntimeError("Round9 source manifest invalid")
    for key in ("structured","structured_metadata"):
        p=D/("structured_metadata.json" if key=="structured_metadata" else
             "structured_pairs.json")
        if sha(p)!=m["files_sha256"][key]:raise RuntimeError("Round9 file SHA changed")
    if sha(BASE)!=m["base_checkpoint_sha256"]:
        raise RuntimeError("Round7 Replay checkpoint changed")
    core,ext,_,_=validate_680()
    if (sha(ROOT/"dpo/benchmarks/teacher_style_400_v1/questions.json")!=m["core400_sha256"] or
        sha(ROOT/"dpo/benchmarks/teacher_style_extension_280_v1/questions.json")!=m["extension280_sha256"]):
        raise RuntimeError("Frozen teacher benchmark changed")
    _,protected,_=source_and_protected()
    protected.update(tuple(x["fact_key"]) for x in core+ext)
    rows=read(D/"structured_pairs.json")
    meta=read(D/"structured_metadata.json")
    if len(rows)!=100000 or len(meta)!=100000:raise RuntimeError("Wrong row count")
    for row,info in zip(rows,meta):
        pm=POS_RE.fullmatch(row["positive"])
        if pm is None or pm["prompt"]!=info["prompt"] or pm["answer"]!=info["answer"]:
            raise RuntimeError("Misaligned structured row metadata")
        key=parse_relation(pm["prompt"],pm["answer"])
        if tuple(info["fact_key"])!=key or stable_split(key,42)!="train":
            raise RuntimeError("Bad math fact / wrong split")
        if row["negative"]!=pm["prompt"]+" "+REFUSAL:
            raise RuntimeError("Source negative changed")
    return rows,meta,m,protected


def mine(mode,device,max_new_tokens,pilot_per_category):
    import torch
    from model import GPT,GPTConfig
    from evaluate_ta_integer import greedy_original_stop
    rows,meta,m,protected=inputs()
    dest=OUT/mode
    if dest.exists():raise FileExistsError(f"Refuse overwrite {dest}")
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run GPU inference through Slurm allocation")
    index={}
    for row,info in zip(rows,meta):
        p=info["prompt"]
        if tuple(info["fact_key"]) in protected:continue
        if p not in index:
            index[p]={"prompt":p,"answer":info["answer"],
                      "category":kind(p),"source":info["source"],
                      "fact_key":info["fact_key"]}
        elif index[p]["answer"]!=info["answer"]:
            raise RuntimeError("Identical prompt with conflicting gold")
    pools=defaultdict(list)
    for rec in index.values():
        if rec["category"] not in KINDS:
            raise RuntimeError(f"Unrecognized category {rec['prompt']}")
        pools[rec["category"]].append(rec)
    rng=random.Random(4200919)
    selected=[]
    for cat in KINDS:
        cur=sorted(pools[cat],key=lambda r:r["prompt"])
        # Favor newly synthesized prompts in pilot, then include old replay.
        new=[r for r in cur if r["source"]=="new"]
        old=[r for r in cur if r["source"]!="new"]
        rng.shuffle(new);rng.shuffle(old)
        take=(new+old)[:pilot_per_category] if mode=="pilot" else new+old
        selected.extend(take)
        print(f"MINE {cat:12} eligible={len(cur):5} selected={len(take):5}",flush=True)
    rng.shuffle(selected)
    print(f"Unique eligible train prompts: {len(index)}; running {len(selected)}",flush=True)
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        meta_vocab=pickle.load(f)
    stoi,itos=meta_vocab["stoi"],meta_vocab["itos"]
    checkpoint=torch.load(BASE,map_location="cpu",weights_only=True)
    model=GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict({k.removeprefix("_orig_mod."):v for k,v in
                           checkpoint.get("model_state_dict",checkpoint.get("model")).items()},
                          strict=True)
    model.to(device).eval()
    errors=[]
    counts=Counter()
    percat=defaultdict(Counter)
    previews=[]
    for i,r in enumerate(selected,1):
        if set(r["prompt"])-set(stoi):raise ValueError("Tokenizer OOV")
        completion,stop=greedy_original_stop(
            model,r["prompt"],stoi,itos,max_new_tokens,device)
        status,predicted,proof=evaluate_text(completion,r["answer"])
        wrong=parse_model_error(completion,r["answer"])
        eligible=(stop=="period" and wrong is not None and status=="math_error")
        if eligible:
            negative=r["prompt"]+(completion if completion.startswith(" ") else " "+completion)
            if max(len(negative),len(r["prompt"]+" The answer is "+r["answer"]+".")+4>256:
                eligible=False
                counts["too_long"]+=1
            elif set(negative)-set(stoi):
                eligible=False
                counts["oov"]+=1
            else:
                errors.append({**r,"completion":completion,"negative":negative,
                               "predicted":wrong,"stop_reason":stop})
        label="numeric_wrong" if eligible else ("numeric_correct" if status=="correct" else status)
        counts[label]+=1;percat[r["category"]][label]+=1
        if len(previews)<24 and (eligible or len(previews)<6):
            previews.append({"prompt":r["prompt"],"gold":r["answer"],
                             "completion":completion,"eligible":eligible,"status":status})
        if i%200==0:print(f"PROGRESS {i}/{len(selected)} usable_wrong={len(errors)}",flush=True)
    outm={
        "version":"round9_mined_v1","mode":mode,
        "source_structured_sha256":m["files_sha256"]["structured"],
        "source_metadata_sha256":m["files_sha256"]["structured_metadata"],
        "base_checkpoint_sha256":sha(BASE),
        "max_new_tokens":max_new_tokens,"decode":"greedy original-stop",
        "eligible_unique_training_prompts":len(index),
        "evaluated_unique_prompts":len(selected),"usable_wrong_unique":len(errors),
        "counts":dict(counts),
        "by_category":{cat:dict(percat[cat]) for cat in KINDS},
        "previews":previews,"errors_sha256":hashlib.sha256(blob(errors)).hexdigest()
    }
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"errors.json").write_bytes(blob(errors))
    (dest/"manifest.json").write_bytes(blob(outm))
    print(f"MINE PASS mode={mode}, usable={len(errors)}/{len(selected)} => {dest}")


def assemble():
    rows,meta,m,protected=inputs()
    minedir=OUT/"full"
    info=read(minedir/"manifest.json")
    errors=read(minedir/"errors.json")
    if info["mode"]!="full" or info["source_structured_sha256"]!=m["files_sha256"]["structured"] or \
       info["base_checkpoint_sha256"]!=sha(BASE) or \
       info["errors_sha256"]!=sha(minedir/"errors.json"):
        raise RuntimeError("Stale/altered full mining result")
    available={}
    for r in errors:
        key=tuple(r["fact_key"])
        if key in protected or stable_split(key,42)!="train":raise RuntimeError("Protected mined fact")
        if parse_relation(r["prompt"],r["answer"])!=key or \
           parse_model_error(r["completion"],r["answer"]) is None:
            raise RuntimeError("Invalid real model error")
        if r["negative"]!=r["prompt"]+(r["completion"] if r["completion"].startswith(" ") else " "+r["completion"]):
            raise RuntimeError("Mined negative not verbatim")
        available[r["prompt"]]=r
    if len(available)<20:
        raise RuntimeError("Fewer than 20 usable unique model errors; do not fake negatives")
    candidates=defaultdict(list)
    for i,r in enumerate(meta):
        if r["prompt"] in available and tuple(r["fact_key"]) not in protected:
            candidates[kind(r["prompt"])].append(i)
    rng=random.Random(4200927)
    chosen=[]
    uses=Counter()
    # Fairness: do not allow one operation to monopolize all 5000 substitutions.
    max_per_category=500
    for category in KINDS:
        ids=candidates[category][:]
        rng.shuffle(ids)
        taken=0
        for i in ids:
            if len(chosen)>=MAX_REPLACEMENTS or taken>=max_per_category:break
            prompt=meta[i]["prompt"]
            if uses[prompt]>=MAX_PER_UNIQUE_PROMPT:continue
            chosen.append(i);uses[prompt]+=1;taken+=1
    if not chosen:raise RuntimeError("No eligible hard negatives to apply")
    hard=[dict(r) for r in rows]
    for i in chosen:
        r=available[meta[i]["prompt"]]
        hard[i]["negative"]=r["negative"]
        if hard[i]["positive"]!=rows[i]["positive"] or \
           hard[i]["negative"]==rows[i]["negative"]:
            raise RuntimeError("Hard-negative replacement failed")
        if max(map(len,(hard[i]["positive"],hard[i]["negative"])))+4>256:
            raise RuntimeError("Hard-negative exceeds block_size")
    if any(a["positive"]!=b["positive"] for a,b in zip(rows,hard)):
        raise RuntimeError("Positive samples or order changed")
    if sum(a["negative"]!=b["negative"] for a,b in zip(rows,hard))!=len(chosen):
        raise RuntimeError("Negative replacements do not match manifest")
    dest=D/"hardneg_pairs.json"
    manifest=D/"hardneg_manifest.json"
    if dest.exists() or manifest.exists():raise FileExistsError("HardNeg output already exists")
    payload=blob(hard)
    md={
        "version":"round9_hardneg_v1",
        "base_checkpoint_sha256":sha(BASE),
        "structured_sha256":m["files_sha256"]["structured"],
        "full_mining_manifest_sha256":sha(minedir/"manifest.json"),
        "full_mining_errors_sha256":sha(minedir/"errors.json"),
        "hardneg_sha256":hashlib.sha256(payload).hexdigest(),
        "replaced_rows":len(chosen),
        "replaced_unique_prompts":len(uses),
        "per_category":dict(Counter(kind(meta[i]["prompt"]) for i in chosen)),
        "max_per_prompt":MAX_PER_UNIQUE_PROMPT
    }
    with dest.open("xb") as f:f.write(payload)
    with manifest.open("xb") as f:f.write(blob(md))
    print(f"ASSEMBLE PASS: {len(chosen)} ONLY-negative replacements, "
          f"{len(uses)} unique prompts; {dest}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mine",choices=("pilot","full"))
    p.add_argument("--assemble",action="store_true")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--pilot-per-category",type=int,default=50)
    args=p.parse_args()
    if bool(args.mine)==bool(args.assemble):
        p.error("Choose exactly one of --mine pilot|full or --assemble")
    if args.pilot_per_category<1 or args.max_new_tokens<1:
        p.error("Positive limits required")
    if args.assemble:assemble()
    else:mine(args.mine,args.device,args.max_new_tokens,args.pilot_per_category)


if __name__=="__main__":main()
