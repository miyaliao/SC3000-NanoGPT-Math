#!/usr/bin/env python3
"""Round 4: matched continuation-vs-replay data from frozen Round 3 train facts.

Experiment design:
    Replay-only: 70k SAME old pairs + 30k other old pairs.
    Treatment:   70k SAME old pairs + 15k harder old train pairs +
                 15k NEW hard multiplication/inverse facts (products 401..999).
Both start from the exact same Round 3 checkpoint, with same training schedule.

Do not inspect or train on the sealed Round3 test set (its fact keys are used
only to guard against leakage). New examples require stable_split(key,42)=train
and must not share facts with Round3 val/test or old Round1C common val.

CPU only:
    python scripts/build_round4_data.py --profile mini --verify-only
    python scripts/build_round4_data.py --profile full --verify-only
    python scripts/build_round4_data.py --profile full
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation, stable_split, RESERVED
from generate_round3_structured import product_relations

SOURCE=ROOT/"dpo/data/round3_structured_full"
OLD_VAL=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"
SEED=42
MULTIPLICATIVE=("multiply","divide","x_times_a","a_times_x","x_over_a","a_over_x")
FOCUS_QUOTA={"multiply":4000,"x_over_a":4000,"x_times_a":2000,
             "a_times_x":2000,"divide":1500,"a_over_x":1500}
EXTEND_QUOTA={c:2500 for c in MULTIPLICATIVE}
assert sum(FOCUS_QUOTA.values())==sum(EXTEND_QUOTA.values())==15000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_existing(out):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {out}")


def validate_sources():
    manifest=read(SOURCE/"manifest.json")
    pairs=read(SOURCE/"pos_neg_pairs.json")
    meta=read(SOURCE/"train_metadata.json")
    if len(pairs)!=100000 or len(meta)!=100000:
        raise RuntimeError("Unexpected Round3 source length")
    required={"train":"pos_neg_pairs.json","train_metadata":"train_metadata.json",
              "val":"val.json","test":"test.json","train_probe":"train_probe_256.json"}
    for k,file in required.items():
        if sha(SOURCE/file)!=manifest["sha256"][k]:
            raise RuntimeError(f"Round3 source hash mismatch: {file}")
    keys_val={parse_relation(r["prompt"],str(r["answer"])) for r in read(SOURCE/"val.json")}
    keys_test={parse_relation(r["prompt"],str(r["answer"])) for r in read(SOURCE/"test.json")}
    if keys_val & keys_test:
        raise RuntimeError("Round3 val/test leakage")
    protected=keys_val | keys_test
    if not OLD_VAL.is_file():
        raise RuntimeError("Old 256 OOD validation missing: cannot certify isolation")
    old=read(OLD_VAL)
    if len(old)!=256:raise RuntimeError("Old frozen validation is not 256")
    old_keys={parse_relation(r["prompt"],str(r["answer"])) for r in old}
    protected |= old_keys
    protected |= {parse_relation(q,a) for q,a in RESERVED.items() if "." not in q+a}
    for pair,m in zip(pairs,meta):
        if not pair["positive"].startswith(m["prompt"]+" The answer is "+str(m["answer"])+" because "):
            raise RuntimeError("Round3 pair/metadata misaligned")
        if parse_relation(m["prompt"],str(m["answer"]))!=tuple(m["fact_key"]):
            raise RuntimeError("Incorrect Round3 fact")
        if tuple(m["fact_key"]) in protected:
            raise RuntimeError("Round3 train already overlaps a protected fact")
    return pairs, meta, protected, {
        "round3_manifest_sha256":sha(SOURCE/"manifest.json"),
        "round3_train_sha256":sha(SOURCE/"pos_neg_pairs.json"),
        "round3_metadata_sha256":sha(SOURCE/"train_metadata.json"),
        "round3_val_sha256":sha(SOURCE/"val.json"),
        "round3_test_sha256":sha(SOURCE/"test.json"),
        "old_val_sha256":sha(OLD_VAL),
    }


def get_extension_pool(protected, split="train"):
    by_kind={k:{} for k in MULTIPLICATIVE}
    for a in range(2,101):
        for b in range(a,101):
            if not (401 <= a*b <= 999):continue
            for left,right in ((a,b),(b,a)) if a!=b else ((a,b),):
                for row in product_relations(left,right):
                    key=row["fact_key"]
                    if key in protected or stable_split(key,SEED)!=split:
                        continue
                    by_kind[row["category"]].setdefault(row["prompt"],row)
    return by_kind


def build(profile):
    old,meta,protected,source_hashes=validate_sources()
    rng=random.Random(42004)
    if profile=="full":
        replay_count=70000
        focus_quota=FOCUS_QUOTA
        extend_quota=EXTEND_QUOTA
    else:
        replay_count=448
        focus_quota={k:16 for k in MULTIPLICATIVE}
        extend_quota={k:16 for k in MULTIPLICATIVE}
    focus_n=sum(focus_quota.values())
    extend_n=sum(extend_quota.values())
    assert replay_count+focus_n+extend_n==(100000 if profile=="full" else 640)

    # SAME shared 70% in both arms (sample old source without replacement).
    order=list(range(len(old)))
    rng.shuffle(order)
    shared_idx=order[:replay_count]
    remaining_idx=order[replay_count:]
    # Replay-only control contains old pairs, with the same total sample count.
    control_idx=shared_idx + rng.sample(remaining_idx, focus_n+extend_n)
    treatment_pairs=[old[i] for i in shared_idx]
    focus_counts=Counter()
    for category,count in focus_quota.items():
        candidates=[]
        for i,m in enumerate(meta):
            if m["category"]!=category:continue
            f=tuple(m["fact_key"])
            if f[0]!="mul" or int(f[1])<=1 or int(f[2])<=1:
                continue
            ans=int(m["answer"])
            # Target numerically hard examples, including inverse operations.
            if category in ("multiply","x_over_a") and ans<100:
                continue
            if category in ("divide","x_times_a","a_times_x","a_over_x") and ans<10:
                continue
            candidates.append(i)
        if not candidates:
            raise RuntimeError(f"No harder in-domain samples for {category}")
        selected=rng.choices(candidates,k=count)
        treatment_pairs.extend(old[i] for i in selected)
        focus_counts[category]=count
    extension=get_extension_pool(protected, split="train")
    extension_val_pool=get_extension_pool(protected, split="val")
    extension_val=[]
    for category in MULTIPLICATIVE:
        candidates=list(extension_val_pool[category].values())
        if len(candidates)<64:
            raise RuntimeError(f"Not enough independent extended validation examples: {category}")
        extension_val.extend(rng.sample(candidates,64))
    rng.shuffle(extension_val)
    extension_rows=[]
    extension_stats={}
    for category,count in extend_quota.items():
        pool=list(extension[category].values())
        if len(pool)<count:
            # May repeat only when truly needed. Manifest reports uniqueness.
            chosen=pool+rng.choices(pool,k=count-len(pool))
        else:
            chosen=rng.sample(pool,count)
        if len(chosen)!=count:
            raise AssertionError("Extended example count mismatch")
        extension_rows.extend(chosen)
        extension_stats[category]={"total_rows":len(chosen),
                                   "unique_prompts":len({x["prompt"] for x in chosen}),
                                   "unique_math_facts":len({x["fact_key"] for x in chosen})}
    treatment_pairs.extend({"positive":r["positive"],"negative":r["negative"]}
                           for r in extension_rows)
    control_pairs=[old[i] for i in control_idx]
    rng.shuffle(control_pairs)
    rng.shuffle(treatment_pairs)
    if len(control_pairs)!=len(treatment_pairs):
        raise AssertionError("Control and treatment length mismatch")
    # Preserve the instructor's get_batches() behavior: any final <64 rows are
    # skipped after shuffling. Both arms have identical count and skipped rows.
    # Audit ALL newly created samples; neither arm can introduce fact leakage.
    vocab_path=ROOT/"sft/meta.pkl"
    with vocab_path.open("rb") as f:vocab=set(pickle.load(f)["stoi"])
    existing_train_facts={tuple(m["fact_key"]) for m in meta}
    extra_facts=set()
    max_len=0
    for r in extension_rows:
        k=parse_relation(r["prompt"],r["answer"])
        if k!=r["fact_key"] or k in protected or k in existing_train_facts:
            raise RuntimeError("New extension fact leaks into old train or val/test")
        if stable_split(k,SEED)!="train":
            raise RuntimeError("Wrong split for extension")
        if not (401<=int(k[3])<=999):
            raise RuntimeError(f"Unexpected product difficulty: {k}")
        if set(r["positive"]+r["negative"])-vocab:
            raise RuntimeError("Tokenizer OOV")
        if max(len(r["positive"]),len(r["negative"]))+4>256:
            raise RuntimeError("Context exceeded")
        extra_facts.add(k)
    for arm in (control_pairs,treatment_pairs):
        for p in arm:
            if set(p["positive"]+p["negative"])-vocab:
                raise RuntimeError("OOV in training pair")
            max_len=max(max_len,len(p["positive"])+4,len(p["negative"])+4)
            if max_len>256:raise RuntimeError("Context exceeded")
    info={
      "profile":profile,"seed":SEED,"source_sha256":source_hashes,
      "old_validation_protected_count":len(protected),
      "replay_shared_count":replay_count,"replay_only_extra_count":focus_n+extend_n,
      "treatment_in_domain_focus_count":focus_n,
      "treatment_extended_count":extend_n,
      "treatment_focus_categories":dict(focus_counts),
      "new_extension_unique_math_facts":len(extra_facts),
      "new_extension_unique_prompts":len({r["prompt"] for r in extension_rows}),
      "new_extension_by_category":extension_stats,
      "max_sequence_with_padding":max_len,
      "training_arm_size":len(control_pairs),
      "full_batches_per_epoch_batch64":len(control_pairs)//64,
      "discarded_incomplete_batch_per_epoch":len(control_pairs)%64,
      "training_rows_not_unique_prompts":True,
      "training_plan":"Round3 checkpoint; original Course Loss; 2 epochs, batch 64, lr=2e-5; replay-only control",
      "evaluation":"Existing Round3 768 Val + previous Round1C 256 OOD; never Round3 final Test",
      "note":"All 70% shared old examples are identical between arms. Treatment additionally reweights hard old train facts and adds product 401..999 facts; control replays older training examples only.",
    }
    info["extension_val_count"]=len(extension_val)
    info["extension_val_unique_math_facts"]=len({r["fact_key"] for r in extension_val})
    if {r["fact_key"] for r in extension_rows} & {r["fact_key"] for r in extension_val}:
        raise RuntimeError("Extended validation intersects training facts")
    for row in extension_val:
        if stable_split(row["fact_key"],SEED)!="val" or row["fact_key"] in protected:
            raise RuntimeError("Invalid extended validation assignment")
        if set(row["positive"]+row["negative"])-vocab:
            raise RuntimeError("Invalid extended validation vocabulary")
    return control_pairs,treatment_pairs,extension_val,info


def write_new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(data,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
    with path.open("xb") as f:f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=("mini","full"),default="mini")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--out-dir",type=Path)
    a=p.parse_args()
    out=(a.out_dir if a.out_dir else ROOT/f"dpo/data/round4_matched_{a.profile}").resolve()
    if out.exists() and not a.verify_only:refuse_existing(out)
    control,treatment,extension_val,info=build(a.profile)
    print("=== ROUND 4 MATCHED DATA CHECK ===")
    print(f"Replay-only={len(control)}, Treatment={len(treatment)}; "
          f"shared_old={info['replay_shared_count']}, "
          f"old_focus={info['treatment_in_domain_focus_count']}, "
          f"new_extension={info['treatment_extended_count']}")
    print(f"New unique math facts={info['new_extension_unique_math_facts']}; "
          f"new unique prompts={info['new_extension_unique_prompts']}; "
          f"max_len={info['max_sequence_with_padding']}")
    if a.verify_only:
        print("PASS: mathematically disjoint, no files written")
        return
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    hashes["replay_only"]=write_new(out/"replay_only_pairs.json",control)
    hashes["treatment"]=write_new(out/"treatment_pairs.json",treatment)
    hashes["extension_val"]=write_new(out/"extension_val.json",[
        {"prompt":r["prompt"],"answer":r["answer"],"category":r["category"],
         "fact_key":list(r["fact_key"])} for r in extension_val])
    info["file_sha256"]=hashes
    write_new(out/"manifest.json",info)
    print("PASS: no protected validation/test facts entered the new training data")
    print(f"Saved: {out}")


if __name__=="__main__":main()
