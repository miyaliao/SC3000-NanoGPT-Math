#!/usr/bin/env python3
"""Build four matched 100k Round6 preference datasets from Round5 audited data.

All arms initialize FROM sft/gpt.pt during training; this generator does NOT train.
A: original 'because' (Round5-A data)
B: answer only ('The answer is N.')
C: original-answer + post-answer decomposition (Round5-B data)
D: on the same 16271 rows as C, put decomposition BEFORE 'The answer is N.'.

All prompts, gold numbers, negatives, row order and underlying math facts match.
No sealed test labels are used. Source files and outputs are SHA256-checked.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from generate_round1 import parse_relation

SRC=ROOT/"dpo/data/round5_decomposition"
OUT=ROOT/"dpo/data/round6_from_sft"
ARMS=("original","answer_only","decomposition_after","decomposition_before")
POS=re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>\d+) because (?P<tail>.+)\.$")
PROOF=re.compile(
    r"(?P<a>\d+)\*(?P<tens>\d+)=(?P<p1>\d+) and "
    r"(?P<aa>\d+)\*(?P<ones>\d+)=(?P<p2>\d+) and "
    r"(?P<q1>\d+)\+(?P<q2>\d+)=(?P<total>\d+)"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pair(row):
    m=POS.fullmatch(row["positive"])
    if not m:raise ValueError(f"Original training Positive not recognized: {row['positive'][:120]}")
    return m.group("prompt"),int(m.group("answer")),m.group("tail")


def proof_valid(explanation,answer):
    m=PROOF.fullmatch(explanation)
    if m is None:return False
    x={k:int(v) for k,v in m.groupdict().items()}
    return (x["a"]==x["aa"] and x["tens"]>=10 and x["tens"]%10==0
            and x["ones"]>=1 and x["p1"]==x["a"]*x["tens"]
            and x["p2"]==x["a"]*x["ones"] and
            x["q1"]==x["p1"] and x["q2"]==x["p2"]
            and x["total"]==x["p1"]+x["p2"]==answer)


def build():
    old_manifest=read(SRC/"manifest.json")
    sources={
        "original":SRC/"original_pairs.json",
        "decomposition":SRC/"decomposition_pairs.json",
    }
    if any(sha(path)!=old_manifest["file_sha256"][key] for key,path in sources.items()):
        raise ValueError("Existing Round5 dataset SHA256 does not match manifest")
    aa=read(sources["original"])
    cc=read(sources["decomposition"])
    if len(aa)!=100000 or len(cc)!=100000:raise ValueError("Expected 100k Round5 source rows")
    with (ROOT/"sft/meta.pkl").open("rb") as f:stoi=pickle.load(f)["stoi"]
    char_vocab=set(stoi)
    generated={arm:[] for arm in ARMS}
    counts=Counter()
    changed_category=Counter()
    examples={}
    max_chars={arm:0 for arm in ARMS}
    for i,(a,c) in enumerate(zip(aa,cc)):
        prompt,answer,tail=parse_pair(a)
        prompt2,answer2,tail2=parse_pair(c)
        if (prompt,answer)!=(prompt2,answer2) or a["negative"]!=c["negative"]:
            raise ValueError(f"Round5 source prompt/answer/Negative mismatch at {i}")
        # Independent mathematics check, also for the non-decomposed examples.
        parse_relation(prompt,str(answer))
        if not a["negative"].startswith(prompt+" "):
            raise ValueError(f"Negative question mismatch at {i}")
        modified=a["positive"]!=c["positive"]
        if modified:
            if not proof_valid(tail2,answer):
                raise ValueError(f"Mathematically invalid decomposition row {i}: {tail2}")
            if not re.fullmatch(r"\d+\*\d+=\?",prompt) and not re.fullmatch(r"x/\d+=\d+,x=\?",prompt):
                raise ValueError(f"Unexpected modified category: {prompt}")
            category="multiply" if re.fullmatch(r"\d+\*\d+=\?",prompt) else "x_over_a"
            changed_category[category]+=1
            counts["modified"]+=1
            b_before=f"{prompt} {tail2} so The answer is {answer}."
            if b_before.count(".")!=1 or "\n" in b_before:
                raise ValueError(f"Early generation stop in pre-answer row {i}")
            if " The answer is " in tail2 or "because" in tail2:
                raise ValueError(f"Unexpected reasoning expression {i}")
        else:
            b_before=c["positive"]
            counts["unchanged"]+=1
        texts={
            "original":a["positive"],
            "answer_only":f"{prompt} The answer is {answer}.",
            "decomposition_after":c["positive"],
            "decomposition_before":b_before,
        }
        for arm,positive in texts.items():
            if set(positive+a["negative"])-char_vocab:
                raise ValueError(f"Tokenizer OOV in {arm} row {i}")
            if len(positive)+4>256 or len(a["negative"])+4>256:
                raise ValueError(f"Model block 256 exceeded in {arm} row {i}")
            max_chars[arm]=max(max_chars[arm],len(positive)+4,len(a["negative"])+4)
            generated[arm].append({"positive":positive,"negative":a["negative"]})
        if modified and len(examples)<8:
            examples[str(i)]={"original":a["positive"],
                "answer_only":texts["answer_only"],
                "after":c["positive"],"before":texts["decomposition_before"]}
    if counts["modified"]!=old_manifest["modified_rows"]:
        raise ValueError(f"Round5 modified count changed: {dict(counts)}")
    if dict(changed_category)!=old_manifest["modified_by_category"]:
        raise ValueError("Round5 decomposed category distribution mismatch")
    for arm in ARMS:
        if len(generated[arm])!=100000:raise AssertionError(arm)
    manifest={
        "version":"round6_sft_four_formats_v1",
        "train_rows_per_arm":100000,
        "source_sha256":{name:sha(path) for name,path in sources.items()},
        "source_manifest_sha256":sha(SRC/"manifest.json"),
        "same_prompt_answer_negative_order":True,
        "initial_checkpoint":"sft/gpt.pt",
        "modified_before_after_rows":counts["modified"],
        "unchanged_before_after_rows":counts["unchanged"],
        "modified_by_category":dict(changed_category),
        "answer_only_modified_rows":100000,
        "max_sequence_characters_including_four_padding":max_chars,
        "examples":examples,
        "test_split_never_read":True,
    }
    return generated,manifest


def save_new(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
    with path.open("xb") as handle:handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    if OUT.exists() and not args.verify_only:
        p.error(f"Output exists and will not be overwritten: {OUT}")
    datasets,m=build()
    print("=== ROUND6 FOUR-FORMAT DATASET AUDIT ===")
    print("Rows per arm:",m["train_rows_per_arm"])
    print("Answer-only rows:",m["answer_only_modified_rows"])
    print("Reasoning moved before answer:",m["modified_before_after_rows"])
    print("By category:",m["modified_by_category"])
    print("Max lengths (incl. padding):",m["max_sequence_characters_including_four_padding"])
    print("Example:",next(iter(m["examples"].values())))
    if args.verify_only:
        print("PASS (read-only): all four formats mathematically/source/tokenizer verified")
        return
    OUT.mkdir(parents=True,exist_ok=False)
    m["file_sha256"]={}
    for arm,records in datasets.items():
        m["file_sha256"][arm]=save_new(OUT/f"{arm}_pairs.json",records)
    save_new(OUT/"manifest.json",m)
    print("PASS saved all four paired training files + SHA256 manifest:",OUT)


if __name__=="__main__":
    main()
