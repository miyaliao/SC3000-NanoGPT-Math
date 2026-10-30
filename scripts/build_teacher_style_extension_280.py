#!/usr/bin/env python3
"""Freeze 280 more easy SC3000 problems in 7 missing arithmetic/equation forms.

This is an EXTENSION, not a replacement for frozen Teacher-style Core400.
Exactly seven categories x four subskills x ten prompts = 280 distinct prompts.
All new canonical math facts are DISJOINT FROM the frozen Core400 and all
historical Val/OOD/Extended/public tests. Any historical TRAIN overlap is
labelled Seen; unseen is reported truthfully, never fabricated.

CPU:
 python scripts/build_teacher_style_extension_280.py --self-test-only
 python scripts/build_teacher_style_extension_280.py --audit-only
 python scripts/build_teacher_style_extension_280.py --freeze
 python scripts/build_teacher_style_extension_280.py --check-frozen
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation, stable_split
from pipeline_smoke import verify_math
from build_teacher_style_400 import (
    source_and_protected, sha, read, count_carries
)

CORE=ROOT/"dpo/benchmarks/teacher_style_400_v1"
OUT=ROOT/"dpo/benchmarks/teacher_style_extension_280_v1"
SEED=20261031
PARTITION_SEED=42
GROUPS=(
    "subtract", "x_plus_a", "a_plus_x", "x_minus_a",
    "a_times_x", "x_over_a", "a_over_x",
)
PER_STRATUM=10
PER_CATEGORY=40
EXPECTED=280
STRATA={
    "subtract":[f"result_{n}digit_borrow{c}" for n in (1,2) for c in (0,1)],
    "x_plus_a":[f"answer_{n}digit_carry{c}" for n in (1,2) for c in (0,1)],
    "a_plus_x":[f"answer_{n}digit_carry{c}" for n in (1,2) for c in (0,1)],
    "x_minus_a":[f"subtrahend_{n}digit_borrow{c}" for n in (1,2) for c in (0,1)],
    "a_times_x":[
        "unknown_1digit_known_2digit",
        "unknown_2digit_known_1digit",
        "both_2digit_product_100_400",
        "both_2digit_product_401_999",
    ],
    "x_over_a":[
        "divisor_1digit_quotient_2digit",
        "divisor_2digit_quotient_1digit",
        "both_2digit_product_100_400",
        "both_2digit_product_401_999",
    ],
    "a_over_x":[
        "answer_1digit_quotient_2digit",
        "answer_2digit_quotient_1digit",
        "both_2digit_product_100_400",
        "both_2digit_product_401_999",
    ],
}
assert len(GROUPS)==7 and all(len(STRATA[g])==4 for g in GROUPS)
assert 7*4*10==280

def core_facts():
    questions=CORE/"questions.json"
    manifest=CORE/"manifest.json"
    if not questions.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            "Core400 must be FROZEN and committed before extension generation."
        )
    m=read(manifest);rows=read(questions)
    if (m.get("version")!="teacher_style_400_v1" or len(rows)!=400 or
        sha(questions)!=m["questions_sha256"] or
        len({r["prompt"] for r in rows})!=400):
        raise RuntimeError("Frozen Core400 content/SHA mismatch")
    facts=set()
    for row in rows:
        fact=parse_relation(row["prompt"],str(row["answer"]))
        if fact!=tuple(row["fact_key"]):
            raise RuntimeError("Core400 fact key corrupted")
        facts.add(fact)
    if len(facts)!=m["unique_fact_keys_total"]:
        raise RuntimeError("Core400 canonical fact count mismatch")
    return facts,{"core_questions_sha256":sha(questions),
                  "core_manifest_sha256":sha(manifest),
                  "core_n_prompts":400,"core_n_canonical_facts":len(facts)}

def classify(group,prompt,answer):
    n=int(answer)
    if group=="subtract":
        m=re.fullmatch(r"(\d+)-(\d+)=\?",prompt)
        if not m:raise ValueError(prompt)
        a,b=map(int,m.groups())
        return f"result_{len(str(n))}digit_borrow{int(a%10<b%10)}"
    if group in ("x_plus_a","a_plus_x"):
        m=(re.fullmatch(r"x\+(\d+)=(\d+),x=\?",prompt)
           if group=="x_plus_a" else
           re.fullmatch(r"(\d+)\+x=(\d+),x=\?",prompt))
        if not m:raise ValueError(prompt)
        a,result=map(int,m.groups())
        return f"answer_{len(str(n))}digit_carry{int(n%10+a%10>=10)}"
    if group=="x_minus_a":
        m=re.fullmatch(r"x-(\d+)=(\d+),x=\?",prompt)
        if not m:raise ValueError(prompt)
        a,_=map(int,m.groups())
        return f"subtrahend_{len(str(a))}digit_borrow{int(n%10<a%10)}"
    if group=="a_times_x":
        m=re.fullmatch(r"(\d+)\*x=(\d+),x=\?",prompt)
        if not m:raise ValueError(prompt)
        multiplier,_=map(int,m.groups())
        product=multiplier*n
        if n<=9 and multiplier>=10:
            return "unknown_1digit_known_2digit"
        if n>=10 and multiplier<=9:
            return "unknown_2digit_known_1digit"
        if n>=10 and multiplier>=10:
            return ("both_2digit_product_100_400" if product<=400
                    else "both_2digit_product_401_999")
        return None
    if group=="x_over_a":
        m=re.fullmatch(r"x/(\d+)=(\d+),x=\?",prompt)
        if not m:raise ValueError(prompt)
        divisor,quotient=map(int,m.groups())
        product=divisor*quotient
        if divisor<=9 and quotient>=10:
            return "divisor_1digit_quotient_2digit"
        if divisor>=10 and quotient<=9:
            return "divisor_2digit_quotient_1digit"
        if divisor>=10 and quotient>=10:
            return ("both_2digit_product_100_400" if product<=400
                    else "both_2digit_product_401_999")
        return None
    if group=="a_over_x":
        m=re.fullmatch(r"(\d+)/x=(\d+),x=\?",prompt)
        if not m:raise ValueError(prompt)
        _,quotient=map(int,m.groups())
        product=n*quotient
        if n<=9 and quotient>=10:
            return "answer_1digit_quotient_2digit"
        if n>=10 and quotient<=9:
            return "answer_2digit_quotient_1digit"
        if n>=10 and quotient>=10:
            return ("both_2digit_product_100_400" if product<=400
                    else "both_2digit_product_401_999")
        return None
    raise ValueError(group)

def examples():
    """Elementary arithmetic; no 4-digit computed products, no fractions."""
    for a in range(10,100):
        for b in range(1,a):
            answer=a-b
            if 1<=answer<=99:
                yield "subtract",f"{a}-{b}=?",str(answer)
    for known in range(1,100):
        for x in range(1,100):
            result=known+x
            if result<=199:
                yield "x_plus_a",f"x+{known}={result},x=?",str(x)
                yield "a_plus_x",f"{known}+x={result},x=?",str(x)
    # x-a=b: keep x two-digit; include single/two-digit subtrahends;
    # the 'borrow' label is the units column of x-a=b.
    for x in range(10,100):
        for a in range(1,x):
            b=x-a
            if 1<=b<=99:
                yield "x_minus_a",f"x-{a}={b},x=?",str(x)
    # Both a*x=b and x/a=b are elementary multiplication relations;
    # limit the result to <=999 for all multiplication-derived cases.
    for multiplier in range(2,41):
        for x in range(1,80):
            result=multiplier*x
            if result<=999:
                yield "a_times_x",f"{multiplier}*x={result},x=?",str(x)
    for divisor in range(2,41):
        for quotient in range(1,80):
            result=divisor*quotient
            if result<=999:
                yield "x_over_a",f"x/{divisor}={quotient},x=?",str(result)
    for quotient in range(2,41):
        for x in range(1,80):
            numerator=x*quotient
            if numerator<=999:
                yield "a_over_x",f"{numerator}/x={quotient},x=?",str(x)

def make_buckets(seen,protected):
    buckets=defaultdict(dict)
    rejected=Counter()
    for category,prompt,answer in examples():
        fact=parse_relation(prompt,answer)
        if fact in protected:
            rejected["protected"]+=1;continue
        if stable_split(fact,PARTITION_SEED)!="train":
            rejected["original_split_not_train"]+=1;continue
        stratum=classify(category,prompt,answer)
        if stratum is None:
            # Both multiplication factors are single-digit. This tiny fact
            # universe is heavily represented by frozen Core400 and is not
            # used to pad new independent extension categories.
            continue
        if stratum not in STRATA[category]:
            raise RuntimeError(f"Unrecognized stratum {category}/{stratum}")
        if not verify_math(prompt,answer):
            raise RuntimeError(f"Incorrect arithmetic ground truth: {prompt} => {answer}")
        if category in ("a_times_x","x_over_a","a_over_x") and int(fact[3])>=1000:
            raise RuntimeError(f"Unintended four-digit product: {prompt}")
        row={
            "prompt":prompt,"answer":answer,"category":category,
            "subskill":stratum,"fact_key":list(fact),
            "seen_status":"seen" if fact in seen else "unseen",
        }
        # Commuted/inverse mathematical facts are deduplicated in a stratum;
        # one representative is selected, while exact prompts stay unique.
        buckets[(category,stratum)].setdefault(fact,row)
    return buckets,rejected

def allocate(group,buckets,used_global,rng):
    """Maximum fact-unique matching, then disclose any cross-stratum reuses.

    Reusing math facts ACROSS different types of questions is permitted only
    if the finite elementary candidate pool makes complete independence
    infeasible. Within each subskill all ten facts and prompts must differ.
    """
    slot_edges={}
    fact_degree=Counter()
    for stratum in STRATA[group]:
        facts=buckets[(group,stratum)]
        if len(facts)<PER_STRATUM:
            raise RuntimeError(f"Not enough candidates for {group}/{stratum}: "
                               f"only {len(facts)} canonical facts, need {PER_STRATUM}")
        fact_degree.update(facts)
    slots=[(s,i) for s in
           sorted(STRATA[group],key=lambda s:(len(buckets[(group,s)]),s))
           for i in range(PER_STRATUM)]
    for slot in slots:
        sub,i=slot
        keys=list(buckets[(group,sub)])
        rng.shuffle(keys)
        preferred="seen" if i<5 else "unseen"
        keys.sort(key=lambda k:(
            0 if buckets[(group,sub)][k]["seen_status"]==preferred else 1,
            fact_degree[k],
            0 if k not in used_global else 1))
        slot_edges[slot]=keys
    owners={}
    assignment={}
    def aug(slot,visited):
        for fact in slot_edges[slot]:
            if fact in visited:continue
            visited.add(fact)
            prev=owners.get(fact)
            if prev is None or aug(prev,visited):
                owners[fact]=slot
                assignment[slot]=fact
                return True
        return False
    missing=[]
    for slot in slots:
        if not aug(slot,set()):
            missing.append(slot)
    # Unlike a global 'unique 280 facts' quota, only prompt uniqueness and
    # within-stratum mathematical-fact uniqueness are mandatory.
    for slot in missing:
        sub=slot[0]
        within={assignment[(sub,i)] for i in range(PER_STRATUM)
                if (sub,i) in assignment}
        candidates=[fact for fact in slot_edges[slot] if fact not in within]
        if not candidates:
            raise RuntimeError(
                f"Can't fill {group}/{sub} despite cross-stratum reuse; "
                "do not weaken protected-fact exclusions"
            )
        count=Counter(assignment.values())
        candidates.sort(key=lambda k:(count[k],0 if k not in used_global else 1,
                                      fact_degree[k]))
        assignment[slot]=candidates[0]
    rows=[]
    for sub in STRATA[group]:
        chosen=[buckets[(group,sub)][assignment[(sub,i)]]
                for i in range(PER_STRATUM)]
        if len({tuple(r["fact_key"]) for r in chosen})!=PER_STRATUM:
            raise RuntimeError(f"Repeated mathematical fact WITHIN {group}/{sub}")
        rows.extend(chosen)
    if len(rows)!=PER_CATEGORY or len({r["prompt"] for r in rows})!=PER_CATEGORY:
        raise RuntimeError(f"Incorrect 40 unique prompts for {group}")
    return rows

def build():
    rng=random.Random(SEED)
    core,core_meta=core_facts()
    seen,protected,source_sha=source_and_protected()
    protected.update(core)
    buckets,rejections=make_buckets(seen,protected)
    out=[]
    used=set()
    for group in GROUPS:
        print(f"=== {group} candidate canonical facts ===",flush=True)
        for sub in STRATA[group]:
            counts=Counter(r["seen_status"] for r in buckets[(group,sub)].values())
            print(f"  {sub:41} unique={len(buckets[(group,sub)]):4} "
                  f"seen={counts['seen']:4} unseen={counts['unseen']:4}",flush=True)
        rows=allocate(group,buckets,used,rng)
        out.extend(rows)
        used.update(tuple(r["fact_key"]) for r in rows)
    order={g:i for i,g in enumerate(GROUPS)}
    suborder={(g,s):i for g in GROUPS for i,s in enumerate(STRATA[g])}
    out.sort(key=lambda r:(order[r["category"]],
                           suborder[(r["category"],r["subskill"])],r["prompt"]))
    if len(out)!=EXPECTED or len({r["prompt"] for r in out})!=EXPECTED:
        raise RuntimeError("Expected 280 unique prompts")
    if not core.isdisjoint({tuple(r["fact_key"]) for r in out}):
        raise RuntimeError("New extension canonical fact overlaps frozen Core400")
    counts=Counter((r["category"],r["subskill"]) for r in out)
    for g in GROUPS:
        if sum(r["category"]==g for r in out)!=PER_CATEGORY:
            raise RuntimeError(f"Not 40 {g} questions")
        for sub in STRATA[g]:
            if counts[(g,sub)]!=PER_STRATUM:
                raise RuntimeError(f"Not 10 questions in {g}/{sub}")
            facts={tuple(r["fact_key"]) for r in out
                   if r["category"]==g and r["subskill"]==sub}
            if len(facts)!=PER_STRATUM:
                raise RuntimeError(f"Fact collision within {g}/{sub}")
    by_status={}
    by_group_facts={}
    for group in GROUPS:
        rr=[r for r in out if r["category"]==group]
        by_status[group]=dict(Counter(r["seen_status"] for r in rr))
        by_group_facts[group]=len({tuple(r["fact_key"]) for r in rr})
        print(f"{group:13} N=40 fact_keys={by_group_facts[group]} "
              f"seen={by_status[group].get('seen',0)} "
              f"unseen={by_status[group].get('unseen',0)}",flush=True)
        for sub in STRATA[group]:
            rr2=[r for r in rr if r["subskill"]==sub]
            seen_n=sum(r["seen_status"]=="seen" for r in rr2)
            print(f"  {sub:41} {len(rr2):2} seen={seen_n:2} "
                  f"unseen={len(rr2)-seen_n:2}",flush=True)
    all_facts={tuple(r["fact_key"]) for r in out}
    if any(tuple(r["fact_key"]) in protected for r in out):
        raise RuntimeError("Protected validation/public/teacher/Core fact entered extension")
    if any(stable_split(tuple(r["fact_key"]),42)!="train" for r in out):
        raise RuntimeError("Non-train original partition math fact entered extension")
    manifest={
        "version":"teacher_style_extension_280_v1",
        "seed":SEED,
        "design":"7 remaining original SC3000 math categories, each 4x10; inspired by ICLR 2024 digit/operation difficulty",
        "paper_repo":"https://github.com/lee-ny/teaching_arithmetic",
        "not_an_official_ICLR_dataset":True,
        "not_a_TA_hidden_test":True,
        "core_reference":core_meta,
        "source_sha256":source_sha,
        "groups":list(GROUPS),
        "subskills":{g:list(STRATA[g]) for g in GROUPS},
        "quota_per_group":PER_CATEGORY,"quota_per_subskill":PER_STRATUM,
        "selected_n":len(out),
        "all_unique_prompts":True,
        "unique_math_facts_total":len(all_facts),
        "within_subskill_unique_math_facts":True,
        "within_group_unique_math_facts":by_group_facts,
        "by_seen_status":by_status,
        "total_seen":sum(r["seen_status"]=="seen" for r in out),
        "total_unseen":sum(r["seen_status"]=="unseen" for r in out),
        "historical_seen_is_not_generalization":True,
        "canonical_fact_disjoint_with_core400":True,
        "canonical_fact_disjoint_with_old_val_ood_extended_public_and_teacher":True,
        "original_stable_partition_train_only":True,
        "future_round8_training_on_these_facts_forbidden":True,
        "no_sealed_test_file_opened":True,
        "selection":"maximum matching per group with disclosed cross-stratum reuse if needed",
        "rejected_candidate_counts":dict(rejections),
        "evaluation":"greedy model.py period/newline stopping, fixed strict answer-first parser",
    }
    print(f"PASS extension280: 280 distinct prompts, {len(all_facts)} "
          f"unique canonical math facts; zero overlap with Core400; "
          "independent original final Test unopened")
    return out,manifest

def serialize(obj):
    return (json.dumps(obj,indent=2,ensure_ascii=False)+"\n").encode("utf-8")

def freeze():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing extension: {OUT}")
    rows,manifest=build()
    OUT.mkdir(parents=True,exist_ok=False)
    blob=serialize(rows)
    with (OUT/"questions.json").open("xb") as f:f.write(blob)
    manifest["questions_sha256"]=hashlib.sha256(blob).hexdigest()
    with (OUT/"manifest.json").open("xb") as f:f.write(serialize(manifest))
    print("FROZEN Extension280",OUT)
    print("questions_sha256:",manifest["questions_sha256"])
    print("Next commit exactly questions.json + manifest.json")

def check_frozen():
    questions=OUT/"questions.json"
    manifest=OUT/"manifest.json"
    if not questions.is_file() or not manifest.is_file():
        raise FileNotFoundError("Extension not yet frozen; pass --audit-only then --freeze.")
    m=read(manifest);rows=read(questions)
    if m.get("version")!="teacher_style_extension_280_v1" or len(rows)!=280:
        raise RuntimeError("Wrong extension version or size")
    if sha(questions)!=m["questions_sha256"]:
        raise RuntimeError("Frozen extension questions SHA mismatch")
    if len({r["prompt"] for r in rows})!=280:
        raise RuntimeError("Duplicate extension prompt")
    if len({tuple(r["fact_key"]) for r in rows})!=m["unique_math_facts_total"]:
        raise RuntimeError("Frozen extension canonical fact count changed")
    core,meta=core_facts()
    if meta!=m["core_reference"]:
        raise RuntimeError("Core400 files changed after extension freeze")
    if {tuple(r["fact_key"]) for r in rows} & core:
        raise RuntimeError("Extension shares a mathematical fact with Core400")
    for r in rows:
        if parse_relation(r["prompt"],r["answer"])!=tuple(r["fact_key"]):
            raise RuntimeError("Extension question/answer/fact mismatch")
        if not verify_math(r["prompt"],r["answer"]):
            raise RuntimeError("Extension incorrect gold")
    print(f"PASS frozen Extension280 and Core400 disjoint; sha={m['questions_sha256']}")

def selftest():
    expected={
        ("subtract","52-19=?","33"):"result_2digit_borrow1",
        ("x_plus_a","x+17=36,x=?","19"):"answer_2digit_carry1",
        ("a_plus_x","17+x=36,x=?","19"):"answer_2digit_carry1",
        ("x_minus_a","x-17=20,x=?","37"):"subtrahend_2digit_borrow0",
        ("a_times_x","11*x=44,x=?","4"):"unknown_1digit_known_2digit",
        ("x_over_a","x/4=18,x=?","72"):"divisor_1digit_quotient_2digit",
        ("a_over_x","72/x=4,x=?","18"):"answer_2digit_quotient_1digit"
    }
    for (group,prompt,answer),stratum in expected.items():
        assert classify(group,prompt,answer)==stratum
        assert verify_math(prompt,answer)
    assert len(GROUPS)==7 and len(STRATA)==7 and all(len(s)==4 for s in STRATA.values())
    assert 7*4*10==280
    print("PASS Extension280: 7 original course categories × 4 strata × 10, correct gold and labels")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test-only",action="store_true")
    actions.add_argument("--audit-only",action="store_true")
    actions.add_argument("--freeze",action="store_true")
    actions.add_argument("--check-frozen",action="store_true")
    a=p.parse_args()
    if a.self_test_only:selftest()
    elif a.audit_only:build();print("AUDIT PASS; nothing written")
    elif a.freeze:freeze()
    else:check_frozen()

if __name__=="__main__":main()
