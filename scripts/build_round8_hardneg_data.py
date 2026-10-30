#!/usr/bin/env python3
"""Round8 paired data: SAME positive examples, DIFFERENT negative preference text.

CONTROL: source replay plus mined difficult train questions & number variants,
         but standard "Sorry, I don't know." rejected response.
HARDNEG: EXACT same prompts, positive completions, order and repeat counts;
         only the mined rows' Rejected is the frozen model's REAL wrong output.

This isolates model-error negative quality from skill targeting & augmentation.
No sealed Test is read and no validation/public math fact enters training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation,stable_split
from mine_round8_train_hard_negatives import (
    OUT as MINED, DATA, MANIFEST, BASE, load_train_only, parse_model_error
)
OUT=ROOT/"dpo/data/round8_hardneg_full"
SEED=42008


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path,content):
    blob=(json.dumps(content,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
    with path.open("xb") as f:f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def capped_replay(rows,limit,max_repeat,rng):
    if not rows:return []
    if len(rows)!=len({r["prompt"] for r in rows}):
        raise RuntimeError("Mined prompt duplicates: each error must be independently unique")
    by_prompt=rows[:]
    rng.shuffle(by_prompt)
    selected=[]
    for _ in range(max_repeat):
        shuffled=by_prompt[:]
        rng.shuffle(shuffled)
        for r in shuffled:
            if len(selected)>=limit:return selected
            selected.append(r)
    return selected


def build():
    rng=random.Random(SEED)
    mining=read(MINED/"manifest.json")
    if mining.get("version")!="round8_train_only_mined_errors_v1":
        raise RuntimeError("Unexpected mining run version")
    if mining["source_and_protection_sha256"]["checkpoint_sha256"]!=sha(BASE):
        raise RuntimeError("Frozen Round7 Replay checkpoint differs from mined model")
    if mining["source_and_protection_sha256"]["round5_pairs_sha256"]!=sha(DATA):
        raise RuntimeError("Frozen old train rows changed from mining")
    original=read(MINED/"original_hard_negatives.json")
    variants=read(MINED/"variant_hard_negatives.json")
    if len(original)!=mining["original_errors"] or len(variants)!=mining["variation_errors"]:
        raise RuntimeError("Mined counts mismatch manifest")
    # Reload exact old TRAIN-only facts and all public/Val/OOD/Extended guards;
    # sealed independent Test remains unopened.
    groups,old_facts,protected,checks=load_train_only()
    if checks!=mining["source_and_protection_sha256"]:
        raise RuntimeError("Mining source protection hashes no longer match")
    if len(original)<20 or len(variants)<10:
        raise RuntimeError(
            f"Insufficient mined errors for a valid paired experiment: "
            f"{len(original)} original / {len(variants)} variant. "
            "Mine more TRAIN-only prompts, do not create fabricated negatives."
        )
    candidates=[("original",r) for r in original]+[("variant",r) for r in variants]
    for stage,r in candidates:
        prompt,answer=r["prompt"],str(r["answer"])
        f=parse_relation(prompt,answer)
        if tuple(r["fact_key"])!=f or f in protected:
            raise RuntimeError(f"Forbidden or incorrect mined fact: {prompt}")
        if stage=="variant":
            if stable_split(f,42)!="train" or f in old_facts:
                raise RuntimeError(f"Variant is not a fresh TRAIN-partition math fact: {prompt}")
            if f[0]=="mul" and int(f[3])>999:
                raise RuntimeError(f"Variant multiplies to >=1000: {prompt}")
        else:
            if f not in old_facts or prompt not in groups[r["category"]]:
                raise RuntimeError(f"Original error was not from eligible old TRAIN: {prompt}")
        if parse_model_error(r["completion"],answer)!=r["prediction"]:
            raise RuntimeError(f"Rejected not a real model wrong answer: {prompt}")
        expected_negative=prompt+(r["completion"] if r["completion"][:1].isspace()
                                  else " "+r["completion"])
        if r["negative"]!=expected_negative:
            raise RuntimeError(f"Rejected not verbatim model completion: {prompt}")
        if r["negative"]==r["positive"]:
            raise RuntimeError("Positive and negative have same response")
        if not r["positive"].startswith(prompt+" The answer is "+answer+" because "):
            raise RuntimeError("The positive answer template was changed")
    max_repeat=5
    selected_old=capped_replay(original,5000,max_repeat,rng)
    selected_variant=capped_replay(variants,5000,max_repeat,rng)
    n_hard=len(selected_old)+len(selected_variant)
    if n_hard>10000:raise AssertionError(n_hard)
    shared_n=100000-n_hard
    old=read(DATA)
    permutation=list(range(len(old)))
    rng.shuffle(permutation)
    baseline=[old[i] for i in permutation[:shared_n]]
    shared_rows=[{"positive":r["positive"],"negative":r["negative"]} for r in baseline]
    control=shared_rows[:]
    hardneg=shared_rows[:]
    for row in selected_old+selected_variant:
        control.append({"positive":row["positive"],
                        "negative":row["prompt"]+" Sorry, I don't know."})
        hardneg.append({"positive":row["positive"],"negative":row["negative"]})
    # One exact row order for both arms; only Rejected differs for mined rows.
    order=list(range(100000))
    rng.shuffle(order)
    control=[control[i] for i in order]
    hardneg=[hardneg[i] for i in order]
    if any(c["positive"]!=h["positive"] for c,h in zip(control,hardneg)):
        raise RuntimeError("Control and hardneg positives are not identical")
    diff=sum(c["negative"]!=h["negative"] for c,h in zip(control,hardneg))
    if diff!=n_hard:
        raise RuntimeError(f"Expected exactly {n_hard} replaced Rejected rows; got {diff}")
    with (ROOT/"sft/meta.pkl").open("rb") as f:vocab=set(pickle.load(f)["stoi"])
    max_length=0
    for i,(c,h) in enumerate(zip(control,hardneg)):
        for record in (c,h):
            if set(record["positive"]+record["negative"])-vocab:
                raise RuntimeError(f"Unsupported training character row {i}")
            max_length=max(max_length,len(record["positive"])+4,
                           len(record["negative"])+4)
            if max_length>256:
                raise RuntimeError(f"Context >256 at row {i}, max={max_length}")
    for i,(c,h) in enumerate(zip(control,hardneg)):
        if c["negative"]!=h["negative"] and
           not c["negative"].endswith(" Sorry, I don't know."):
            raise RuntimeError(f"Control mined row lacks original refusal negative at {i}")
    manifest={
        "version":"round8_matched_hardneg_v1",
        "seed":SEED,
        "base":"dpo/experiments/round7_replay/dpo.pt",
        "base_sha256":sha(BASE),
        "source_pair_sha256":sha(DATA),
        "mine_manifest_sha256":sha(MINED/"manifest.json"),
        "source_and_protection_sha256":checks,
        "training_rows_each":len(control),
        "shared_old_replay_rows":shared_n,
        "mined_original_candidate_count":len(original),
        "mined_variant_candidate_count":len(variants),
        "mined_original_training_rows":len(selected_old),
        "mined_variant_training_rows":len(selected_variant),
        "hard_negative_replaced_rows":diff,
        "hard_negative_fraction":diff/len(control),
        "max_repeat_per_mined_unique_question":max_repeat,
        "exact_same_positive_response_order_between_arms":True,
        "no_heldout_fact_leakage":True,
        "all_variant_multiplication_products_le_999":True,
        "refusals_control_vs_real_model_wrong_hardneg":True,
        "max_model_input_seq_plus_pad":max_length,
        "training_steps_per_epoch":len(control)//64,
        "no_sealed_test_read":True,
        "no_gpu_training_performed":True,
    }
    print("ROUND8 SAME-POSITIVE MATCHED HARD-NEGATIVE DATA")
    print("Train rows per arm: 100000")
    print("Identical plain old replay rows:",shared_n)
    print("Actual mined original TRAIN error repetitions:",len(selected_old))
    print("Actual mined number variants error repetitions:",len(selected_variant))
    print("Negative text changed at exactly:",diff,"rows",
          f"({diff/1000:.2f}% of 100k)")
    print("Max repeat per wrong unique prompt:",max_repeat)
    print("Max total token characters + 4 pad:",max_length)
    print("Full pairs and manifest SHA checked; no validation/public/hidden test seeded into training.")
    return control,hardneg,manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-only",action="store_true")
    args=ap.parse_args()
    control,hard,manifest=build()
    if args.verify_only:
        print("PASS: no files written")
        return
    if OUT.exists():ap.error(f"Refusing to overwrite existing Round8 data folder: {OUT}")
    OUT.mkdir(parents=True,exist_ok=False)
    manifest["file_sha256"]={
        "control":write_once(OUT/"control_pairs.json",control),
        "hardneg":write_once(OUT/"hardneg_pairs.json",hard)
    }
    write_once(OUT/"manifest.json",manifest)
    print("Saved:",OUT)


if __name__=="__main__":
    main()
