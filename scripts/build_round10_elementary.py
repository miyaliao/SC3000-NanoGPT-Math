#!/usr/bin/env python3
"""Round10: 20k paired Control vs elementary multiplication curriculum.

Targeted: 8k additive replay + 670 product<10 + 6720 product10..99
          +4610 product100..999; 35% of 2digit-product exposure carries.
Control: 20k original Random replay. Both start from frozen E5 checkpoint.
Never overwrite old data; no new training facts in validation/test partitions.
No sealed final test read. Exposures can repeat, but are strictly capped.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left
from collections import Counter,defaultdict
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
from generate_round3_structured import product_relations, KINDS
from mine_round8_train_hard_negatives import kind
from build_teacher_style_400 import source_and_protected
from evaluate_teacher_style_680 import validate as validate_680

SOURCE=ROOT/"dpo/data/round9_balanced_full/random_pairs.json"
SOURCE_MANIFEST=ROOT/"dpo/data/round9_balanced_full/manifest.json"
E5=ROOT/"dpo/experiments/round9_random_multi_e3_e6/epoch_e05.pt"
OUT=ROOT/"dpo/data/round10_elementary_mul_20k"
SEED=4201002
MULT=("multiply","divide","x_times_a","a_times_x","x_over_a","a_over_x")
ADD=tuple(x for x in KINDS if x not in MULT)
# Intentionally higher DIRECT multiplication exposure.
QUOTA={
    "1digit":dict(zip(MULT,(190,110,95,95,95,85))),
    "2digit":dict(zip(MULT,(2000,1000,900,900,1000,920))),
    "3digit":dict(zip(MULT,(930,722,691,707,868,692)))
}
assert {k:sum(v.values()) for k,v in QUOTA.items()}=={
    "1digit":670,"2digit":6720,"3digit":4610}
assert 8000+sum(sum(v.values()) for v in QUOTA.values())==20000

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def asbytes(obj):return (json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
def group(fact):
    v=int(fact[3])
    if v<0:raise RuntimeError("Negative products not in Round10")
    return "1digit" if v<10 else ("2digit" if v<100 else "3digit" if v<1000 else "4plus")
def has_carry(fact):
    a,b=map(int,fact[1:3])
    small,big=min(a,b),max(a,b)
    return group(fact)=="2digit" and 2<=small<=9 and 10<=big<=99 and (small*(big%10))>=10

def as_record(row):
    pos=row["positive"]; mark=" The answer is "
    if mark not in pos:raise RuntimeError("Malformed Chosen prefix")
    prompt,tail=pos.split(mark,1)
    ans=tail.split(" because ",1)[0].rstrip(".")
    cat=kind(prompt)
    fact=parse_relation(prompt,ans)
    return {"positive":pos,"negative":row["negative"],"prompt":prompt,
            "category":cat,"fact":fact,"bin":group(fact) if cat in MULT else "additive",
            "carry":has_carry(fact) if cat in MULT else False}

def source():
    old_manifest=read(SOURCE_MANIFEST)
    if sha(SOURCE)!=old_manifest["files_sha256"]["random"]:
        raise RuntimeError("Frozen Round9 Random SHA mismatch")
    old=read(SOURCE)
    if len(old)!=100000:raise RuntimeError("Expected 100k Random rows")
    e5_history=read(E5.parent/"train_manifest.json")
    if not any(x["checkpoint"]==E5.name and x["checkpoint_sha256"]==sha(E5)
               for x in e5_history["history"]):
        raise RuntimeError("E5 checkpoint not in frozen Random E3-E6 history")
    old_records=[as_record(x) for x in old]
    return old,old_records

def eligible_generated(protected):
    # Original Round9 numeric domain; fact-based train partition & protected exclusions.
    bank={}
    for a in range(1,100):
        for b in range(a,100):
            if a*b>999:continue
            for left,right in ((a,b),(b,a)) if a!=b else ((a,b),):
                for item in product_relations(left,right):
                    rec=as_record(item)
                    if tuple(rec["fact"]) in protected or stable_split(rec["fact"],42)!="train":
                        continue
                    bank.setdefault(rec["prompt"],rec)
    return bank

def roll(pool, rng, prompt_uses, fact_uses, pcap, fcap):
    # Static weighted distribution; hard exposure caps checked after each proposal.
    weights=[(3.0 if rec["carry"] else 1.0) *
             (0.15 if 1 in tuple(map(int,rec["fact"][1:3])) else 1.0)
             for rec in pool]
    cumul=[];run=0.
    for w in weights:run+=w;cumul.append(run)
    for _ in range(240):
        ix=bisect_left(cumul,rng.random()*run)
        item=pool[min(ix,len(pool)-1)]
        fact=tuple(item["fact"])
        if prompt_uses[item["prompt"]]<pcap and fact_uses[fact]<fcap:
            prompt_uses[item["prompt"]]+=1;fact_uses[fact]+=1
            return item
    available=[x for x in pool if prompt_uses[x["prompt"]]<pcap and
               fact_uses[tuple(x["fact"])]<fcap]
    if not available:raise RuntimeError("Strict caps exhaust quota: adjust SIZE, not silent duplicate")
    item=rng.choice(available)
    prompt_uses[item["prompt"]]+=1;fact_uses[tuple(item["fact"])]+=1
    return item

def select_targeted(old_records,protected,teacher):
    rng=random.Random(SEED)
    additive={}
    for rec in old_records:
        if rec["category"] in ADD:additive.setdefault(rec["prompt"],rec)
    if len(additive)<8000:
        raise RuntimeError(f"Only {len(additive)} distinct additive replay prompts; can't select unique 8000")
    selected_add=rng.sample(list(additive.values()),8000)
    # Reuse historical TRAIN examples and generate additional TRAIN-only surface prompts.
    pools=eligible_generated(protected)
    for rec in old_records:
        if rec["category"] in MULT and tuple(rec["fact"]) not in protected and \
           stable_split(rec["fact"],42)=="train":
            pools.setdefault(rec["prompt"],rec)
    grouped=defaultdict(list)
    for rec in pools.values():
        if rec["category"] in MULT and rec["bin"] in QUOTA:
            key=(rec["bin"],rec["category"], "carry" if rec["carry"] else "nocarry")
            grouped[key].append(rec)
    counts=Counter()
    fact_counts=Counter()
    targeted=list(selected_add)
    details=Counter()
    for bucket in ("1digit","2digit","3digit"):
        quotas=QUOTA[bucket]
        # Round-robin task families prevents one category from taking every shared fact.
        remaining=dict(quotas)
        carry_target={cat:int(q*0.35) for cat,q in quotas.items()} if bucket=="2digit" else {}
        carry_used=Counter()
        pcap={"1digit":15,"2digit":10,"3digit":3}[bucket]
        fcap={"1digit":100,"2digit":70,"3digit":20}[bucket]
        for cat in MULT:
            if not grouped[(bucket,cat,"carry")] and not grouped[(bucket,cat,"nocarry")]:
                raise RuntimeError(f"No eligible items for {bucket}/{cat}")
        while any(remaining.values()):
            for cat in MULT:
                if remaining[cat]<=0:continue
                if bucket=="2digit" and carry_used[cat]<carry_target[cat]:
                    key=(bucket,cat,"carry")
                else:key=(bucket,cat,"nocarry")
                pool=grouped[key]
                try:
                    rec=roll(pool,rng,counts,fact_counts,pcap,fcap)
                except RuntimeError:
                    # Prefer a legal alternative only when the preferred stratum is exhausted.
                    other="nocarry" if key[-1]=="carry" else "carry"
                    rec=roll(grouped[(bucket,cat,other)],rng,counts,fact_counts,pcap,fcap)
                targeted.append(rec)
                remaining[cat]-=1
                if rec["carry"]:carry_used[cat]+=1
                details[(bucket,cat)]+=1
        if bucket=="2digit":
            n=sum(details[("2digit",cat)] for cat in MULT)
            c=sum(1 for rec in targeted[8000:] if rec["bin"]=="2digit" and rec["carry"])
            if c<0.30*n:
                raise RuntimeError(f"Carry exposure unexpectedly low: {c}/{n}")
            print(f"2DIGIT carry exposure={c}/{n} ({c/n:.2%}); target >=30%",flush=True)
    assert len(targeted)==20000
    # Additive 8000 are all unique, multiplicative identities can recur at capped exposure.
    rng.shuffle(targeted)
    return targeted,details,counts,fact_counts

def build_basic_probe(seen,protected):
    """Val-split multiplication facts unseen in ALL historical E5 training;
    not the sealed final test. Also preserve the already RESERVED teacher probes."""
    bank={}
    for a in range(1,100):
        for b in range(a,100):
            if not 10<=a*b<=99:continue
            for left,right in ((a,b),(b,a)) if a!=b else ((a,b),):
                for item in product_relations(left,right):
                    rec=as_record(item)
                    fact=tuple(rec["fact"])
                    if stable_split(fact,42)!="val" or fact in seen or fact in protected:
                        continue
                    bank.setdefault(rec["prompt"],rec)
    rng=random.Random(SEED+50)
    available=list(bank.values())
    rng.shuffle(available)
    seen_facts=Counter()
    result=[]
    for rec in available:
        fact=tuple(rec["fact"])
        if seen_facts[fact]>=2:continue
        result.append({"prompt":rec["prompt"],"answer":rec["fact"][3] if rec["category"] in ("multiply","x_over_a") else rec["positive"].split(" The answer is ")[1].split(" because ")[0],
                       "category":rec["category"],"fact_key":list(fact)})
        seen_facts[fact]+=1
        if len(result)>=60:break
    if len(result)<10:
        print(f"WARNING: only {len(result)} independent elementary held-out prompts; do not claim a 60-item benchmark",flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--verify-only",action="store_true")
    ap.add_argument("--build",action="store_true")
    args=ap.parse_args()
    if args.verify_only==args.build:ap.error("Specify one of --verify-only and --build")
    if args.build and OUT.exists():raise FileExistsError(OUT)
    old,old_records=source()
    core,ext,_,_=validate_680()
    seen,protected,_=source_and_protected()
    protected.update(tuple(x["fact_key"]) for x in core+ext)
    rng=random.Random(SEED+1)
    control=[old[i] for i in rng.sample(range(len(old)),20000)]
    chosen,detail,uses,fact_uses=select_targeted(old_records,protected,core+ext)
    targeted=[{"positive":x["positive"],"negative":x["negative"]} for x in chosen]
    if len(targeted)!=len(control):raise RuntimeError("Unmatched 20k")
    # Explicit retained 5 instructor/public probes; never train on their facts in new rows.
    probes=[{"prompt":p,"answer":a,"category":kind(p)} for p,a in
            RESERVED.items() if "." not in p+a and p in (
            "17+19=?","3*17=?","72/4=?","72-x=34,x=?","x*11=44,x=?")]
    benchmark=build_basic_probe(seen,protected)
    manifest={
        "version":"round10_elementary_20k_v1","seed":SEED,
        "source_random_sha256":sha(SOURCE),"e5_checkpoint_sha256":sha(E5),
        "rows":20000,"controls":"random 20k rows from frozen Round9 Random; sampled without replacement",
        "targeted":"8000 unique additive replay + 670 product<10 + 6720 product10..99 + 4610 product100..999",
        "targeted_counts":dict(Counter(x["bin"] for x in chosen)),
        "targeted_by_bin_and_category":{f"{b}/{cat}":v for (b,cat),v in detail.items()},
        "multiplicative_distinct_prompts":len(uses),
        "multiplicative_distinct_facts":len(fact_uses),
        "multiplicative_max_prompt_exposure":max(uses.values()),
        "multiplicative_max_fact_exposure":max(fact_uses.values()),
        "carry_2digit_exposures":sum(x["carry"] for x in chosen if x["bin"]=="2digit"),
        "basic_probe_n":len(benchmark),
        "basic_probe_distinct_facts":len({tuple(x["fact_key"]) for x in benchmark}),
        "reserved_probes_n":len(probes),
        "files_sha256":{}
    }
    files={"control_pairs.json":control,"targeted_pairs.json":targeted,
           "basic_probe.json":benchmark,"reserved_probes.json":probes}
    for name,data in files.items():manifest["files_sha256"][name]=hashlib.sha256(asbytes(data)).hexdigest()
    print(json.dumps({k:v for k,v in manifest.items() if k!="files_sha256"},indent=2),flush=True)
    if args.verify_only:
        print("ROUND10 AUDIT PASS; nothing written",flush=True)
        return
    OUT.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        (OUT/name).write_bytes(asbytes(data))
    (OUT/"manifest.json").write_bytes(asbytes(manifest))
    for name,hashval in manifest["files_sha256"].items():
        if sha(OUT/name)!=hashval:raise RuntimeError(f"Output SHA mismatch {name}")
    print("ROUND10 BUILD PASS: matched 20k Control/Targeted + frozen development probes",flush=True)

if __name__=="__main__":main()
