#!/usr/bin/env python3
"""Round5: paired original-vs-decomposition explanations on SAME Round4 Treatment rows.

Both output files have exactly 100,000 rows, identical order/prompts/answers/
negatives. Arm A copies original Positive. Arm B modifies ONLY the explanatory
tail of mathematically eligible product-answer questions.

No model weights or original dataset are changed.
Usage:
  python scripts/build_round5_decomposition.py --verify-only
  python scripts/build_round5_decomposition.py
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
from generate_round3_structured import REFUSAL

SOURCE=ROOT/"dpo/data/round4_matched_full"
SOURCE_FILE=SOURCE/"treatment_pairs.json"
DEST=ROOT/"dpo/data/round5_decomposition"
NUM=r"[0-9]+"
DIRECT_RE=re.compile(rf"^({NUM})\*({NUM})=\?$")
UNKNOWN_RE=re.compile(rf"^x/({NUM})=({NUM}),x=\?$")
POS_RE=re.compile(
    rf"^(?P<prompt>.+?=\?) The answer is (?P<answer>{NUM}) "
    rf"because (?P<explanation>.+)\.$"
)
KINDS=("multiply","x_over_a")


def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    manifest=json.loads((SOURCE/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile")!="full" or manifest.get("training_arm_size")!=100000:
        raise RuntimeError("Round4 source is not the frozen full 100k experiment")
    if digest(SOURCE_FILE)!=manifest["file_sha256"]["treatment"]:
        raise RuntimeError("Round4 treatment SHA256 mismatch")
    pairs=json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    if len(pairs)!=100000:
        raise RuntimeError("Expected exactly 100,000 Round4 treatment rows")
    return manifest,pairs


def exact_factors(prompt:str,answer:int):
    m=DIRECT_RE.fullmatch(prompt)
    if m:
        a,b=int(m.group(1)),int(m.group(2))
        kind="multiply"
    else:
        m=UNKNOWN_RE.fullmatch(prompt)
        if not m:return None
        denominator,quotient=int(m.group(1)),int(m.group(2))
        a,b=quotient,denominator
        kind="x_over_a"
    if a*b!=answer or answer<100 or a<2 or b<2:
        return None
    return a,b,kind


def split_factors(a:int,b:int):
    """Choose an arithmetic split with two strictly positive partial products.

    Prefer split of the second operand to follow question order (23*17).
    If necessary, split the first operand. Never use +0 disguised as teaching.
    """
    for left,right in ((a,b),(b,a)):
        tens=right//10*10
        ones=right%10
        if tens>=10 and ones>=1:
            return left,tens,ones
    return None


def decomposition(pair:dict):
    pos,neg=pair["positive"],pair["negative"]
    m=POS_RE.fullmatch(pos)
    if not m:
        raise ValueError(f"Unexpected original Positive format: {pos[:90]}")
    prompt,ans=m.group("prompt"),int(m.group("answer"))
    if neg!=f"{prompt} {REFUSAL}":
        raise ValueError(f"Negative is not the unchanged course refusal: {prompt}")
    # Every row is independently verified against its printed mathematical task.
    parse_relation(prompt,str(ans))
    factors=exact_factors(prompt,ans)
    if factors is None:
        return pos,None
    a,b,kind=factors
    split=split_factors(a,b)
    if split is None:
        return pos,None
    x,tens,ones=split
    first=x*tens
    second=x*ones
    if first+second!=ans or not(first>0 and second>0):
        raise AssertionError(f"Arithmetic decomposition wrong for {prompt}")
    # Canonical phrase; all intermediate expressions have been checked.
    explanation=(f"{x}*{tens}={first} and "
                 f"{x}*{ones}={second} and "
                 f"{first}+{second}={ans}")
    new_pos=f"{prompt} The answer is {ans} because {explanation}."
    if new_pos==pos or new_pos.count(".")!=1:
        raise AssertionError(f"Invalid decomposition text: {new_pos}")
    return new_pos, {
        "prompt":prompt,"answer":ans,"category":kind,
        "factors":[a,b],"split":[x,tens,ones],
        "partial_products":[first,second],
    }


def assemble():
    source_manifest,pairs=load_source()
    a_pairs=[]
    b_pairs=[]
    examples=[]
    counts=Counter()
    eligible_by_answer=Counter()
    reasons=Counter()
    vocab_path=ROOT/"sft/meta.pkl"
    with vocab_path.open("rb") as f:
        vocab=set(pickle.load(f)["stoi"])
    max_length=0
    for i,pair in enumerate(pairs):
        old=pair["positive"];neg=pair["negative"]
        new,detail=decomposition(pair)
        if detail is not None:
            counts[detail["category"]]+=1
            band = ("100-400" if detail["answer"]<=400 else
                    "401-999" if detail["answer"]<=999 else "1000+")
            eligible_by_answer[band]+=1
            if len(examples)<12:examples.append({"row":i,"original":old,"decomposition":new,**detail})
        else:
            if "=? The answer is " not in old:
                raise AssertionError(f"Unexpected math template: {old[:90]}")
            reasons["unchanged"]+=1
        if set(old+new+neg)-vocab:
            raise RuntimeError(f"Tokenizer OOV in row {i}")
        max_length=max(max_length,len(old)+4,len(new)+4,len(neg)+4)
        if max_length>256:
            raise RuntimeError(f"Sequence exceeds model block 256 at row {i}: {len(new)}")
        a_pairs.append({"positive":old,"negative":neg})
        b_pairs.append({"positive":new,"negative":neg})
    changed=sum(counts.values())
    if changed==0:
        raise RuntimeError("No eligible decomposition samples found; check source data")
    if changed<5000:
        print(f"WARNING: only {changed} decomposition-modified rows; "
              "interpret a null result as possibly limited treatment intensity",flush=True)
    for pa,pb in zip(a_pairs,b_pairs):
        if pa["negative"]!=pb["negative"]:
            raise AssertionError("Control negative differed")
        ap=POS_RE.fullmatch(pa["positive"])
        bp=POS_RE.fullmatch(pb["positive"])
        if ap.group("prompt")!=bp.group("prompt") or ap.group("answer")!=bp.group("answer"):
            raise AssertionError("Control prompt/answer differed")
    report={
        "version":"round5-decomposition-v1",
        "source_round4_treatment_sha256":digest(SOURCE_FILE),
        "source_round4_manifest_sha256":digest(SOURCE/"manifest.json"),
        "source_round4_training_rows":len(pairs),
        "arm_A":"original one-step because",
        "arm_B":"exact distributive decomposition after correct numerical answer",
        "same_rows_order_answers_negatives":True,
        "modified_rows":changed,
        "unmodified_rows":len(pairs)-changed,
        "modified_fraction":changed/len(pairs),
        "modified_by_category":dict(counts),
        "modified_by_answer_band":dict(eligible_by_answer),
        "max_required_length_with_padding":max_length,
        "max_model_context":256,
        "selected_examples":examples,
        "training_plan":"both from Round4 Treatment Epoch1, 1 epoch, same seed/batch/LR, original course Loss",
        "no_training_or_test_evaluation":True,
    }
    return a_pairs,b_pairs,report


def write_new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(data,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
    with path.open("xb") as f:f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--out-dir",type=Path,default=DEST)
    args=p.parse_args()
    out=args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir
    if out.exists() and not args.verify_only:
        p.error(f"Refusing to overwrite existing Round5 folder: {out}")
    a,b,manifest=assemble()
    print("=== ROUND 5 DATA AUDIT ===")
    print(f"Training rows per arm: {len(a)}")
    print(f"Decomposition-modified rows: {manifest['modified_rows']} "
          f"({manifest['modified_fraction']:.2%})")
    print(f"By category: {manifest['modified_by_category']}")
    print(f"By answer band: {manifest['modified_by_answer_band']}")
    print(f"Unmodified rows: {manifest['unmodified_rows']}")
    print(f"Max sequence length including 4 padding characters: {manifest['max_required_length_with_padding']}")
    print("Example A:",manifest["selected_examples"][0]["original"])
    print("Example B:",manifest["selected_examples"][0]["decomposition"])
    if args.verify_only:
        print("PASS: verified without creating files")
        return
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    hashes["original"]=write_new(out/"original_pairs.json",a)
    hashes["decomposition"]=write_new(out/"decomposition_pairs.json",b)
    manifest["file_sha256"]=hashes
    write_new(out/"manifest.json",manifest)
    print("PASS: paired datasets and manifest saved to",out)


if __name__=="__main__":
    main()
