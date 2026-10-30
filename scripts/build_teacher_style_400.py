#!/usr/bin/env python3
"""Freeze a NEW 400-question SC3000 teacher-style, ICLR-2024-inspired benchmark.

NOT a verbatim ICLR dataset. The paper's balanced digit/carry idea is adapted
to five SC3000 example categories, with exactly four subskills x 20 questions.
Separately label Seen versus Unseen canonical mathematical facts relative to
the frozen Round3/Round4/Round5 training source. Existing Val/OOD/Extended,
prior-year public blind questions and professor's examples are excluded.

ALL selected facts must belong to the ORIGINAL stable_split(key,42) TRAIN
partition, even for UNSEEN questions. This is necessary to keep the sealed
Round3 independent TEST partition untouched without reading the Test file.

Commands (CPU):
  python scripts/build_teacher_style_400.py --self-test-only
  python scripts/build_teacher_style_400.py --audit-only
  python scripts/build_teacher_style_400.py --freeze

Once written, NEVER regenerate, rebalance or modify its 400 rows for tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation,stable_split,RESERVED
from pipeline_smoke import verify_math

R3=ROOT/"dpo/data/round3_structured_full"
R4=ROOT/"dpo/data/round4_matched_full"
R5=ROOT/"dpo/data/round5_decomposition"
OOD=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"
PUBLIC=ROOT/"dpo/reference_2025_blind_sets.json"
OUT=ROOT/"dpo/benchmarks/teacher_style_400_v1"

SEED=20261030
GROUPS=("add","multiply","divide","a_minus_x","x_times_a")
SUBSKILL=20
PER_GROUP=4*SUBSKILL
EXPECTED_TOTAL=5*PER_GROUP

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def count_carries(a,b):
    """Digit sum identity from the paper's carry-balancing data code."""
    ds=lambda n:sum(int(ch) for ch in str(n))
    return (ds(a)+ds(b)-ds(a+b))//9

def attr(group,prompt,answer):
    m=re.fullmatch(r"(\d+)\+(\d+)=\?",prompt) if group=="add" else None
    if m:
        a,b=map(int,m.groups())
        u=(a%10+b%10)>=10
        t=(a//10+b//10+int(u))>=10
        assert count_carries(a,b)==int(u)+int(t)
        return f"ones_carry{int(u)}_tens_carry{int(t)}"
    m=re.fullmatch(r"(\d+)\*(\d+)=\?",prompt) if group=="multiply" else None
    if m:
        a,b=map(int,m.groups())
        return f"result_{len(str(int(answer)))}digit_ones_carry{int(a*(b%10)>=10)}"
    m=re.fullmatch(r"(\d+)/(\d+)=\?",prompt) if group=="divide" else None
    if m:
        a,b=map(int,m.groups())
        return f"quotient_{len(str(int(answer)))}digit_divisor_{'even' if b%2==0 else 'odd'}"
    m=re.fullmatch(r"(\d+)-x=(\d+),x=\?",prompt) if group=="a_minus_x" else None
    if m:
        a,b=map(int,m.groups())
        return f"answer_{len(str(int(answer)))}digit_borrow{int(a%10<b%10)}"
    m=re.fullmatch(r"x\*(\d+)=(\d+),x=\?",prompt) if group=="x_times_a" else None
    if m:
        a,b=map(int,m.groups())
        return f"answer_{len(str(int(answer)))}digit_multiplier_{len(str(a))}digit"
    raise ValueError(f"Unrecognized {group} expression {prompt}")

STRATA={
    "add":tuple(f"ones_carry{u}_tens_carry{t}" for u in (0,1) for t in (0,1)),
    "multiply":tuple(f"result_{n}digit_ones_carry{c}" for n in (2,3) for c in (0,1)),
    "divide":tuple(f"quotient_{n}digit_divisor_{p}" for n in (1,2) for p in ("even","odd")),
    "a_minus_x":tuple(f"answer_{n}digit_borrow{b}" for n in (1,2) for b in (0,1)),
    "x_times_a":tuple(f"answer_{n}digit_multiplier_{m}digit" for n in (1,2) for m in (1,2)),
}

def examples():
    """Enumerate the intended elementary difficulty, no 4-digit product."""
    for a in range(10,100):
        for b in range(10,100):
            yield "add",f"{a}+{b}=?",str(a+b)
    for a in range(2,10):
        for b in range(10,100):
            yield "multiply",f"{a}*{b}=?",str(a*b)
    for divisor in range(2,10):
        for quotient in range(1,100):
            dividend=divisor*quotient
            if 1<=dividend<=99:
                yield "divide",f"{dividend}/{divisor}=?",str(quotient)
    for known in range(20,100):
        for result in range(10,known):
            x=known-result
            if 1<=x<=99:
                yield "a_minus_x",f"{known}-x={result},x=?",str(x)
    for multiplier in range(2,16):
        for x in range(1,80):
            product=multiplier*x
            if product<=999:
                yield "x_times_a",f"x*{multiplier}={product},x=?",str(x)

def source_and_protected():
    r3m=read(R3/"manifest.json")
    r4m=read(R4/"manifest.json")
    r5m=read(R5/"manifest.json")
    sources={
        "round3_train_meta":R3/"train_metadata.json",
        "round3_val":R3/"val.json",
        "round4_treatment":R4/"treatment_pairs.json",
        "round4_extended_val":R4/"extension_val.json",
        "round5_train":R5/"original_pairs.json",
        "old_ood":OOD,
        "published_public_blind":PUBLIC,
    }
    if sha(sources["round3_train_meta"])!=r3m["sha256"]["train_metadata"]:
        raise RuntimeError("Round3 original training metadata SHA differs")
    if sha(sources["round3_val"])!=r3m["sha256"]["val"]:
        raise RuntimeError("Round3 frozen val SHA differs")
    if sha(sources["round4_treatment"])!=r4m["file_sha256"]["treatment"]:
        raise RuntimeError("Round4 treatment training SHA differs")
    if sha(sources["round4_extended_val"])!=r4m["file_sha256"]["extension_val"]:
        raise RuntimeError("Round4 extended val SHA differs")
    if sha(sources["round5_train"])!=r5m["file_sha256"]["original"]:
        raise RuntimeError("Round5 original training SHA differs")
    if sha(sources["old_ood"])!=r4m["source_sha256"]["old_val_sha256"]:
        raise RuntimeError("Old OOD validation SHA differs")
    seen=set()
    r3_train=read(sources["round3_train_meta"])
    if len(r3_train)!=100000:raise RuntimeError("Missing old 100k train source")
    for r in r3_train:
        seen.add(tuple(r["fact_key"]))
    regex=re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>\d+) because ")
    for name in ("round4_treatment","round5_train"):
        pairs=read(sources[name])
        if len(pairs)!=100000:raise RuntimeError(f"Unexpected 100k rows: {name}")
        for pair in pairs:
            m=regex.match(pair["positive"])
            if not m:raise ValueError(f"Old TRAIN positive not recognized: {name}")
            seen.add(parse_relation(m["prompt"],m["answer"]))
    protected=set()
    for name in ("round3_val","old_ood","round4_extended_val"):
        for row in read(sources[name]):
            protected.add(parse_relation(row["prompt"],str(row["answer"])))
    for prompt,answer in RESERVED.items():
        if "." not in prompt+answer:
            protected.add(parse_relation(prompt,answer))
    public=read(PUBLIC)
    for group in ("blind61","blind41"):
        for row in public["groups"][group]["records"]:
            protected.add(parse_relation(row["prompt"],str(row["answer"])))
    if not seen.isdisjoint(protected):
        # Some protected public questions are known to overlap historical TRAIN.
        # This is acceptable: those facts are EXCLUDED from new benchmark.
        print("NOTE: previously used public/instructor facts overlap old TRAIN; "
              "they are excluded from the new benchmark.")
    return seen,protected,{k:sha(v) for k,v in sources.items()}

def candidates(seen,protected):
    buckets=defaultdict(list)
    encounter=set()
    rejection=Counter()
    for group,prompt,answer in examples():
        fact=parse_relation(prompt,answer)
        if fact in protected:
            rejection["protected"]+=1;continue
        if stable_split(fact,42)!="train":
            rejection["not_original_train_partition"]+=1;continue
        if (group,prompt) in encounter:continue
        encounter.add((group,prompt))
        sub=attr(group,prompt,answer)
        if sub not in STRATA[group]:
            raise ValueError(f"Unexpected subskill {group} {sub}")
        if not verify_math(prompt,answer):
            raise ValueError(f"Incorrect generated answer: {prompt} -> {answer}")
        if "?" not in prompt or any(ord(ch)>127 for ch in prompt):
            raise ValueError(prompt)
        buckets[(group,sub)].append({
            "prompt":prompt,"answer":answer,
            "category":group,"subskill":sub,
            "fact_key":list(fact),
            "seen_status":"seen" if fact in seen else "unseen",
        })
    return buckets,rejection

def select_group(group,buckets,used_global,rng):
    """Jointly allocate 20 distinct facts to each of a group's four subskills.

    Greedy, one-subskill-at-a-time selection can consume the same math fact in
    two neighboring strata. For example 8*7=56 could appear as 56/7=8 or
    56/8=7, across odd/even divisor groups. A bipartite augmenting-path
    matching reassigns previously chosen facts so a feasible 4x20 solution is
    found instead of failing due to the order of sampling.

    Historical Seen/Unseen is a preference, NOT a hard quota. Never invent
    missing unseen facts just to force 10/10. Preserve 400 distinct prompts
    and distinct canonical facts WITHIN each operation group.
    """
    subs=STRATA[group]
    # Within each subskill multiple surface variants can represent one fact.
    # Keep ONE representative per (subskill,fact_key) deterministically.
    per_sub={}
    for sub in subs:
        rows=list(buckets[(group,sub)])
        rng.shuffle(rows)
        unique={}
        for row in rows:
            unique.setdefault(tuple(row["fact_key"]),row)
        per_sub[sub]=unique
        if len(unique)<SUBSKILL:
            counts=Counter(x["seen_status"] for x in unique.values())
            raise RuntimeError(
                f"{group}/{sub}: only {len(unique)} unique mathematical facts "
                f"available, fewer than {SUBSKILL}; Seen/Unseen={dict(counts)}."
            )

    fact_degree=Counter()
    for unique in per_sub.values():
        fact_degree.update(unique.keys())

    # Give scarce strata first access and minimize overlap where possible.
    # The augmenter may subsequently move any selected fact to free a
    # constrained slot; there is NO irreversible greedy reservation.
    sub_order=sorted(subs,key=lambda sub:(len(per_sub[sub]),sub))
    slot_order=[(sub,i) for sub in sub_order for i in range(SUBSKILL)]
    edges={}
    for sub,i in slot_order:
        keys=list(per_sub[sub])
        rng.shuffle(keys)
        # Prefer Seen in first 10 slots, Unseen in last 10. Scarcity of the
        # other label is allowed; the output reports actual counts.
        preferred="seen" if i<SUBSKILL//2 else "unseen"
        keys.sort(key=lambda k:(
            0 if per_sub[sub][k]["seen_status"]==preferred else 1,
            fact_degree[k],
            0 if k not in used_global else 1,
        ))
        edges[(sub,i)]=keys

    owner={}      # canonical math fact -> slot holding it
    assignment={} # slot -> canonical math fact

    def augment(slot,visited):
        for key in edges[slot]:
            if key in visited:
                continue
            visited.add(key)
            previous_owner=owner.get(key)
            if previous_owner is None or augment(previous_owner,visited):
                owner[key]=slot
                assignment[slot]=key
                return True
        return False

    for slot in slot_order:
        if not augment(slot,set()):
            sub=slot[0]
            availability={s:len(per_sub[s]) for s in subs}
            counts={s:dict(Counter(r["seen_status"] for r in per_sub[s].values()))
                    for s in subs}
            raise RuntimeError(
                f"{group}: mathematically impossible or insufficient "
                f"fact-disjoint 4x{SUBSKILL} stratification at {sub}. "
                f"Unique fact candidates={availability}, "
                f"Seen/Unseen candidates={counts}. "
                "Do NOT weaken anti-leakage constraints or add duplicates."
            )

    result=[]
    for sub in subs:
        rows=[per_sub[sub][assignment[(sub,i)]] for i in range(SUBSKILL)]
        if len({tuple(x["fact_key"]) for x in rows})!=SUBSKILL:
            raise RuntimeError(f"Matching produced duplicate facts in {group}/{sub}")
        result.extend(rows)
    expected=len(subs)*SUBSKILL
    if len(result)!=expected or len({tuple(x["fact_key"]) for x in result})!=expected:
        raise RuntimeError(f"Joint matching failed {group} canonical fact uniqueness")
    return result


def test_matching():
    """Exercise the REAL matcher on Hall-tight synthetic overlapping strata."""
    # A has 29 math facts, B has 30, overlap 19 and union precisely 40.
    # A naive sequential sampling can waste the overlap and fail, although
    # a valid 20+20 matching exists. This case tests displacement/reassignment.
    label="__two_subskill_match_test__"
    if label in STRATA:
        raise AssertionError("Synthetic test group unexpectedly exists")
    STRATA[label]=("A","B")
    def make(i):
        return {"fact_key":["mul",str(i),"2",str(i*2)],
                "seen_status":"seen" if i%2 else "unseen",
                "prompt":str(i),"subskill":"A"}
    buckets={
        (label,"A"):[make(i) for i in range(29)],
        (label,"B"):[dict(make(i),subskill="B")
                     for i in list(range(19))+list(range(29,40))]
    }
    try:
        chosen=select_group(label,buckets,set(),random.Random(42))
    finally:
        del STRATA[label]
    assert len(chosen)==40
    assert Counter(x["subskill"] for x in chosen)=={"A":20,"B":20}
    assert len({tuple(x["fact_key"]) for x in chosen})==40
    print("PASS real joint matcher: 29 vs 30 candidates, overlap 19, "
          "40 distinct facts allocated")

def build():
    rng=random.Random(SEED)
    seen,protected,source_hashes=source_and_protected()
    buckets,rejection=candidates(seen,protected)
    chosen=[]
    used=set()
    # Constrained group matching, not greedy stratum-by-stratum selection.
    # This preserves 20 facts per stratum and 80 unique facts per operation.
    for group in GROUPS:
        print(f"Candidate availability for {group} (unique facts per subskill):",flush=True)
        for sub in STRATA[group]:
            unique={tuple(r["fact_key"]):r for r in buckets[(group,sub)]}
            counts=Counter(r["seen_status"] for r in unique.values())
            print(f"  {sub:38} unique={len(unique):4} "
                  f"seen={counts['seen']:4} unseen={counts['unseen']:4}",flush=True)
        rows=select_group(group,buckets,used,rng)
        chosen.extend(rows)
        used.update(tuple(r["fact_key"]) for r in rows)
    order={group:i for i,group in enumerate(GROUPS)}
    suborder={(g,s):i for g in GROUPS for i,s in enumerate(STRATA[g])}
    chosen.sort(key=lambda r:(order[r["category"]],
                              suborder[(r["category"],r["subskill"])],
                              r["prompt"]))
    if len(chosen)!=EXPECTED_TOTAL:
        raise RuntimeError(f"Wrong total: {len(chosen)}")
    if len({r["prompt"] for r in chosen})!=EXPECTED_TOTAL:
        raise RuntimeError("Benchmark contains duplicate exact prompts")
    if any(len({tuple(r["fact_key"]) for r in chosen if r["category"]==g})!=PER_GROUP
           for g in GROUPS):
        raise RuntimeError("Within-group benchmark has duplicate mathematical facts")
    by_group=Counter(r["category"] for r in chosen)
    by_stratum=Counter((r["category"],r["subskill"]) for r in chosen)
    if any(by_group[g]!=PER_GROUP for g in GROUPS):
        raise RuntimeError(f"Wrong group quotas: {by_group}")
    if any(by_stratum[(g,s)]!=SUBSKILL for g in GROUPS for s in STRATA[g]):
        raise RuntimeError(f"Wrong subskill quotas: {by_stratum}")
    by_status=defaultdict(Counter)
    for r in chosen:
        fact=tuple(r["fact_key"])
        if fact in protected or stable_split(fact,42)!="train":
            raise RuntimeError(f"Protected or non-TRAIN fact: {r['prompt']}")
        if (fact in seen)!=(r["seen_status"]=="seen"):
            raise RuntimeError("Incorrect seen/unseen label")
        if not verify_math(r["prompt"],r["answer"]):
            raise RuntimeError(f"Incorrect arithmetic gold: {r}")
        if r["category"]=="multiply" and int(r["answer"])>=1000:
            raise RuntimeError("Four digit multiplication is not teacher style")
        by_status[r["category"]][r["seen_status"]]+=1
    manifest={
        "version":"teacher_style_400_v1",
        "seed":SEED,
        "paper":"Nayoung Lee et al., Teaching Arithmetic to Small Transformers, ICLR 2024",
        "paper_repo":"https://github.com/lee-ny/teaching_arithmetic",
        "note":"New SC3000 benchmark inspired by digit/carry balance; NOT paper's raw dataset",
        "teacher_example_questions_kept_separate":True,
        "groups":list(GROUPS),
        "quota_per_group":PER_GROUP,
        "subskill_quota":SUBSKILL,
        "subskills":{g:list(STRATA[g]) for g in GROUPS},
        "selected_n":len(chosen),
        "by_seen_status":{g:dict(by_status[g]) for g in GROUPS},
        "total_seen":sum(r["seen_status"]=="seen" for r in chosen),
        "total_unseen":sum(r["seen_status"]=="unseen" for r in chosen),
        "original_stable_partition":"train for ALL benchmark facts",
        "fact_disjoint_with_existing_val_ood_extended_public_and_teacher_examples":True,
        "all_unique_prompts":True,
        "unique_fact_keys_total":len({tuple(r["fact_key"]) for r in chosen}),
        "math_fact_reuses_between_operation_groups":
            EXPECTED_TOTAL-len({tuple(r["fact_key"]) for r in chosen}),
        "within_each_operation_unique_fact_keys":True,
        "selection_strategy":"joint augmenting-path canonical-fact matching per operation",
        "no_four_digit_multiplication_results":True,
        "no_sealed_round3_test_file_read":True,
        "source_sha256":source_hashes,
        "new_candidate_rejections":dict(rejection),
        "no_future_training_on_any_of_these_400_facts":True,
        "evaluation":"same original greedy model.py stop and strict starts-with-The answer is scoring; TA extractor unknown",
    }
    print("=== NEW TEACHER-STYLE 400 QUESTION BENCHMARK ===")
    for g in GROUPS:
        print(f"{g:12} {by_group[g]}/80 seen={by_status[g]['seen']} "
              f"unseen={by_status[g]['unseen']}")
        for sub in STRATA[g]:
            rs=[x for x in chosen if x["category"]==g and x["subskill"]==sub]
            ss=Counter(x["seen_status"] for x in rs)
            print(f"  {sub:38} {len(rs):2} seen={ss['seen']:2} unseen={ss['unseen']:2}")
    print("400 unique prompt strings;",
          len({tuple(r["fact_key"]) for r in chosen}),
          "unique canonical math facts (cross-task reuse disclosed); no old validation/public facts; "
          "sealed old Test NEVER opened.")
    return chosen,manifest

def write(path,obj):
    return (json.dumps(obj,ensure_ascii=False,indent=2)+"\n").encode("utf-8")

def freeze():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen benchmark folder {OUT}")
    rows,manifest=build()
    OUT.mkdir(parents=True,exist_ok=False)
    blob=write(OUT/"questions.json",rows)
    with (OUT/"questions.json").open("xb") as f:f.write(blob)
    manifest["questions_sha256"]=hashlib.sha256(blob).hexdigest()
    with (OUT/"manifest.json").open("xb") as f:f.write(write(OUT/"manifest.json",manifest))
    print("FROZEN benchmark:",OUT)
    print("questions_sha256:",manifest["questions_sha256"])
    print("Commit questions.json + manifest.json before Round8 mining/training.")

def validate_frozen():
    questions=OUT/"questions.json";manifest=OUT/"manifest.json"
    if not questions.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            "Benchmark not yet frozen. Run --audit-only successfully, then "
            "--freeze, then --check-frozen."
        )
    m=read(manifest); rows=read(questions)
    if m["version"]!="teacher_style_400_v1" or len(rows)!=400:
        raise RuntimeError("Wrong frozen benchmark version/size")
    if sha(questions)!=m["questions_sha256"]:
        raise RuntimeError("Frozen benchmark SHA MISMATCH: do not silently regenerate")
    if len({r["prompt"] for r in rows})!=400:
        raise RuntimeError("Duplicate exact prompts")
    if len({tuple(r["fact_key"]) for r in rows})!=m["unique_fact_keys_total"]:
        raise RuntimeError("Frozen canonical fact count changed")
    for r in rows:
        if tuple(r["fact_key"])!=parse_relation(r["prompt"],r["answer"]):
            raise RuntimeError("Benchmark fact key corrupted")
        if not verify_math(r["prompt"],r["answer"]):
            raise RuntimeError("Benchmark label incorrect")
    print("PASS frozen 400 and SHA:",m["questions_sha256"])

def selftest():
    assert count_carries(17,19)==1
    assert count_carries(55,66)==2
    assert attr("add","17+19=?","36")=="ones_carry1_tens_carry0"
    assert attr("multiply","3*17=?","51")=="result_2digit_ones_carry1"
    assert attr("divide","72/4=?","18")=="quotient_2digit_divisor_even"
    assert attr("a_minus_x","72-x=34,x=?","38")=="answer_2digit_borrow1"
    assert attr("x_times_a","x*11=44,x=?","4")=="answer_1digit_multiplier_2digit"
    assert sum(1 for _ in GROUPS)==5
    assert all(len(v)==4 for v in STRATA.values())
    assert 5*4*20==400
    test_matching()
    print("PASS 5 categories, 20 strata x20, carry/borrow labels, original teacher samples")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test-only",action="store_true")
    action.add_argument("--audit-only",action="store_true")
    action.add_argument("--freeze",action="store_true")
    action.add_argument("--check-frozen",action="store_true")
    args=p.parse_args()
    if args.self_test_only:selftest()
    elif args.audit_only:build();print("AUDIT PASS; no files written")
    elif args.freeze:freeze()
    else:validate_frozen()

if __name__=="__main__": main()
