#!/usr/bin/env python3
"""Round7: mathematically isolated replay vs targeted multiplication curriculum.

Only OLD TRAIN rows may be replayed. New 11..99 times 11..99 facts with product
1000..4999 use the original stable 42 TRAIN partition, excluding the existing
Val768, Old OOD256, Extended384 and all previously trained facts.

NEVER open the independent sealed Round3 test or the prior-year public blind
questions. Only predeclared skill-level strata from earlier analysis guide
the synthetic math generation.

Usage:
  python scripts/build_round7_targeted_data.py --profile mini --verify-only
  python scripts/build_round7_targeted_data.py --profile full --verify-only
  python scripts/build_round7_targeted_data.py --profile full
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation,stable_split,RESERVED
from generate_round3_structured import product_relations,REFUSAL

SEED=42
DATA=ROOT/"dpo/data/round5_decomposition"
OLD=DATA/"original_pairs.json"
R3=ROOT/"dpo/data/round3_structured_full"
R4=ROOT/"dpo/data/round4_matched_full"
OOD=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"

# Errors in Round5-A were concentrated in direct large-product arithmetic.
# Keep smaller inverse-form training to avoid losing them.
QUOTA={"multiply":3750,"x_over_a":3750,"x_times_a":2250,
       "a_times_x":2250,"divide":1500,"a_over_x":1500}
assert sum(QUOTA.values())==15000

POS_RE=re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>\d+) because .+\.$")
FAMILY_PATTERNS={
    "multiply":re.compile(r"\d+\*\d+=\?"),
    "x_over_a":re.compile(r"x/\d+=\d+,x=\?"),
    "x_times_a":re.compile(r"x\*\d+=\d+,x=\?"),
    "a_times_x":re.compile(r"\d+\*x=\d+,x=\?"),
    "divide":re.compile(r"\d+/\d+=\?"),
    "a_over_x":re.compile(r"\d+/x=\d+,x=\?"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_and_guards():
    m=load(DATA/"manifest.json")
    if m.get("source_round4_training_rows")!=100000:
        raise ValueError("Round5 source is not 100k")
    if sha(OLD)!=m["file_sha256"]["original"]:
        raise ValueError("Round5 source train sha differs from manifest")
    rm=load(R3/"manifest.json")
    fm=load(R4/"manifest.json")
    if sha(R3/"val.json")!=rm["sha256"]["val"]:
        raise ValueError("Round3 frozen validation modified")
    if sha(OOD)!=fm["source_sha256"]["old_val_sha256"]:
        raise ValueError("Old OOD validation modified")
    if sha(R4/"extension_val.json")!=fm["file_sha256"]["extension_val"]:
        raise ValueError("Extended validation modified")
    val=load(R3/"val.json")
    ood=load(OOD)
    ext=load(R4/"extension_val.json")
    if (len(val),len(ood),len(ext))!=(768,256,384):
        raise ValueError("Unexpected number of frozen validation records")
    protected=set()
    for records in (val,ood,ext):
        for r in records:
            protected.add(parse_relation(r["prompt"],str(r["answer"])))
    protected.update(parse_relation(p,a) for p,a in RESERVED.items()
                     if "." not in p+a)
    old=load(OLD)
    if len(old)!=100000:
        raise ValueError("Expected 100k Round5 original training rows")
    parsed=[]
    prior_facts=set()
    by_cat={}
    for cat in QUOTA:by_cat[cat]=[]
    for i,p in enumerate(old):
        match=POS_RE.fullmatch(p["positive"])
        if match is None:raise ValueError(f"Unexpected original training positive row {i}")
        prompt,answer=match["prompt"],match["answer"]
        if p["negative"]!=f"{prompt} {REFUSAL}":
            raise ValueError(f"Negative preference template changed at {i}")
        fact=parse_relation(prompt,answer)
        if fact in protected:
            raise ValueError(f"Original training data overlaps a protected validation fact: {prompt}")
        prior_facts.add(fact)
        cat=next((k for k,pattern in FAMILY_PATTERNS.items()
                  if pattern.fullmatch(prompt)),None)
        product=int(fact[3]) if fact[0]=="mul" else None
        if cat in QUOTA and product is not None and product>=101:
            # Replay difficult mathematical facts already seen at training,
            # including direct products 101..999 and reverse equations.
            by_cat[cat].append(i)
        parsed.append((prompt,answer,fact,cat))
    for category,candidates in by_cat.items():
        if not candidates:
            raise RuntimeError(f"No prior training examples for {category}")
    source_hashes={
        "old_100k_sha256":sha(OLD),
        "round5_manifest_sha256":sha(DATA/"manifest.json"),
        "round3_val_sha256":sha(R3/"val.json"),
        "old_ood_sha256":sha(OOD),
        "extended_val_sha256":sha(R4/"extension_val.json"),
    }
    # We intentionally do NOT open R3/test.json.
    # New fact stable_split == train proves it cannot be in the old test split.
    return old,prior_facts,protected,by_cat,source_hashes


def novel_pool(prior_facts,protected):
    pools={kind:{} for kind in QUOTA}
    facts=set()
    for a in range(11,100):
        for b in range(a,100):
            product=a*b
            if not 1000<=product<=4999:
                continue
            for left,right in (((a,b),(b,a)) if a!=b else ((a,b),)):
                for r in product_relations(left,right):
                    cat=r["category"]
                    if cat not in pools:continue
                    key=parse_relation(r["prompt"],r["answer"])
                    if (key in prior_facts or key in protected or
                        stable_split(key,SEED)!="train"):
                        continue
                    if key[0]!="mul" or int(key[3])!=product:
                        raise ValueError("Bad synthetic multiplication fact")
                    pools[cat].setdefault(r["prompt"],r)
                    facts.add(key)
    return {k:list(v.values()) for k,v in pools.items()},facts


def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(content,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
    with path.open("xb") as f:f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def assemble(profile):
    rng=random.Random(42007)
    old,prior_facts,protected,hard_by_cat,source_hashes=source_and_guards()
    pool,possible_facts=novel_pool(prior_facts,protected)
    if profile=="full":
        shared=70000
        old_quota=QUOTA
        new_quota=QUOTA
    else:
        shared=448
        old_quota={cat:16 for cat in QUOTA}
        new_quota={cat:16 for cat in QUOTA}
    hard_n=sum(old_quota.values())
    new_n=sum(new_quota.values())
    if shared+hard_n+new_n!=(100000 if profile=="full" else 640):
        raise AssertionError("Round7 arm length wrong")
    permutation=list(range(len(old)))
    rng.shuffle(permutation)
    common=permutation[:shared]
    remaining=permutation[shared:]
    # SAME shared rows; control fills with leftover ordinary old TRAIN rows.
    control_extra=rng.sample(remaining,hard_n+new_n)
    control=[old[i] for i in common+control_extra]
    target=[old[i] for i in common]
    hard_counts={}
    for cat,count in old_quota.items():
        selected=rng.choices(hard_by_cat[cat],k=count)
        target.extend(old[i] for i in selected)
        hard_counts[cat]={
            "rows":count,"distinct_source_rows":len(set(selected)),
            "available_train_source_rows":len(hard_by_cat[cat])
        }
    new_counts={}
    novel_rows=[]
    for cat,count in new_quota.items():
        choices=pool[cat]
        if len(choices)<count:
            raise RuntimeError(f"New TRAIN-only pool {cat}: {len(choices)} < {count}")
        chosen=rng.sample(choices,count)
        novel_rows.extend(chosen)
        target.extend({"positive":r["positive"],"negative":r["negative"]}
                      for r in chosen)
        new_counts[cat]={
            "rows":count,"available_pool_prompts":len(choices),
            "unique_prompts":len({r["prompt"] for r in chosen}),
        }
    new_facts={tuple(r["fact_key"]) for r in novel_rows}
    if new_facts & protected or new_facts & prior_facts:
        raise RuntimeError("Novel candidate collides with previous training or validation")
    if any(stable_split(k,SEED)!="train" for k in new_facts):
        raise RuntimeError("New math facts must be in the original train partition")
    rng.shuffle(control)
    rng.shuffle(target)
    if len(control)!=len(target):
        raise RuntimeError("Unequal experimental training rows")
    vocab_path=ROOT/"sft/meta.pkl"
    with vocab_path.open("rb") as f:vocab=set(pickle.load(f)["stoi"])
    max_len=0
    for arm,rows in (("replay",control),("targeted",target)):
        for i,p in enumerate(rows):
            pos,neg=p["positive"],p["negative"]
            invalid=set(pos+neg)-vocab
            if invalid:
                raise ValueError(f"{arm} tokenizer OOV row {i}: {sorted(invalid)}")
            length=max(len(pos)+4,len(neg)+4)
            max_len=max(max_len,length)
            if length>256:
                raise ValueError(f"{arm} exceeds model context 256 row {i}: {length}")
            match=POS_RE.fullmatch(pos)
            if match is None:
                raise ValueError(f"{arm} unusual positive row {i}")
            p_prompt,p_answer=match["prompt"],match["answer"]
            if neg!=f"{p_prompt} {REFUSAL}":
                raise ValueError("Course negative preference was changed")
            key=parse_relation(p_prompt,p_answer)
            if key in protected or (arm=="targeted" and
                                    key not in prior_facts and key not in new_facts):
                raise ValueError(f"{arm} protected/unknown math fact: {p_prompt}")
    info={
        "version":"round7_matched_mult_v1",
        "profile":profile,"seed":42007,
        "source_training_sha256":source_hashes,
        "base_checkpoint":"dpo/experiments/round5_original/dpo.pt",
        "strategy":"train-only multiplication failure-type reinforcement",
        "same_shared_old_records":shared,
        "control_other_old_records":hard_n+new_n,
        "target_old_hard_records":hard_n,
        "target_new_math_records":new_n,
        "old_hard_by_category":hard_counts,
        "new_by_category":new_counts,
        "new_product_band":"1000..4999 (two-digit by two-digit)",
        "new_unique_math_facts":len(new_facts),
        "new_unique_prompts":len({r["prompt"] for r in novel_rows}),
        "possible_new_unique_math_facts":len(possible_facts),
        "previous_training_math_facts":len(prior_facts),
        "validation_protected_math_facts":len(protected),
        "every_new_fact_stable_split_train":True,
        "no_sealed_test_read":True,
        "no_public_blind_read":True,
        "max_sequence_length_with_padding":max_len,
        "rows_per_arm":len(control),
        "batch_size":64,
        "optimizer_updates_per_arm":len(control)//64,
        "discarded_incomplete_batch":len(control)%64,
        "init_model_same_for_both_arms":True,
        "training_plan":"from Round5-A; 1 epoch; Course Loss; same batch64 LR1e-5 seed; evaluate fixed primary",
    }
    if profile=="full" and len(new_facts)<1000:
        raise RuntimeError(f"Insufficient novel mathematical facts: {len(new_facts)}")
    print("=== ROUND7 MATCHED TRAINING DATA ===")
    print(f"control={len(control)} targeted={len(target)} shared_old={shared}")
    print(f"target old hard={hard_n}, target NEW TRAIN-split rows={new_n}")
    print("old hard:",json.dumps(hard_counts,sort_keys=True))
    print("new:",json.dumps(new_counts,sort_keys=True))
    print("new unique math facts:",len(new_facts),"old math facts:",len(prior_facts))
    print("protected facts:",len(protected),"max tokens incl padding:",max_len)
    print("No sealed final test or public blind tests opened.")
    return control,target,info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=("mini","full"),default="mini")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--out-dir",type=Path)
    args=p.parse_args()
    out=(args.out_dir if args.out_dir is not None else
         ROOT/f"dpo/data/round7_targeted_{args.profile}").resolve()
    if out.exists() and not args.verify_only:
        p.error(f"Cannot overwrite Round7 folder: {out}")
    control,target,info=assemble(args.profile)
    if args.verify_only:
        print("PASS: Round7 balanced TRAIN-only data audited, nothing written")
        return
    out.mkdir(parents=True,exist_ok=False)
    hashes={
        "replay":write_once(out/"replay_pairs.json",control),
        "targeted":write_once(out/"targeted_pairs.json",target),
    }
    info["file_sha256"]=hashes
    write_once(out/"manifest.json",info)
    print("PASS: saved",out)


if __name__=="__main__":
    main()
