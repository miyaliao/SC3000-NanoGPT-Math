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


def mine(mode,device,max_new_tokens,pilot_per_category,resume=False,max_questions_per_job=0):
    import torch
    from model import GPT,GPTConfig
    from evaluate_ta_integer import greedy_original_stop
    rows,meta,m,protected=inputs()
    dest=OUT/mode
    if resume:
        if mode != "full" or not (dest/"progress.json").is_file():
            raise RuntimeError("--resume requires incomplete full/progress.json")
        if (dest/"manifest.json").exists():
            raise RuntimeError("Mining is already complete; refusing resume")
    elif dest.exists():
        raise FileExistsError(f"Refuse overwrite existing mining output {dest}")
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
    selection_sha256=hashlib.sha256(blob([r["prompt"] for r in selected])).hexdigest()
    source_sha256=m["files_sha256"]["structured"]
    metadata_sha256=m["files_sha256"]["structured_metadata"]
    base_sha256=sha(BASE)
    dest.mkdir(parents=True,exist_ok=False) if not resume else None
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        meta_vocab=pickle.load(f)
    stoi,itos=meta_vocab["stoi"],meta_vocab["itos"]
    checkpoint=torch.load(BASE,map_location="cpu",weights_only=True)
    model=GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict({k.removeprefix("_orig_mod."):v for k,v in
                           checkpoint.get("model_state_dict",checkpoint.get("model")).items()},
                          strict=True)
    model.to(device).eval()
    progress_path=dest/"progress.json"
    if resume:
        progress=read(progress_path)
        expected={"mode":mode,"source_structured_sha256":source_sha256,
                  "source_metadata_sha256":metadata_sha256,
                  "base_checkpoint_sha256":base_sha256,
                  "selected_prompt_sha256":selection_sha256,
                  "max_new_tokens":max_new_tokens,
                  "total_selected":len(selected)}
        if any(progress.get(k)!=v for k,v in expected.items()):
            raise RuntimeError("Resume checkpoint does not match frozen selection/model/data")
        processed=progress["processed"]
        if not 0 <= processed <= len(selected):raise RuntimeError("Corrupt progress offset")
        errors=progress["errors"]
        counts=Counter(progress["counts"])
        percat=defaultdict(Counter,{k:Counter(v) for k,v in progress["by_category"].items()})
        previews=progress["previews"]
        print(f"RESUME {processed}/{len(selected)} usable_wrong={len(errors)}",flush=True)
    else:
        processed=0
        errors=[]
        counts=Counter()
        percat=defaultdict(Counter)
        previews=[]
    def checkpoint(step):
        state={"version":"round9_mining_progress_v1","mode":mode,
               "source_structured_sha256":source_sha256,
               "source_metadata_sha256":metadata_sha256,
               "base_checkpoint_sha256":base_sha256,
               "selected_prompt_sha256":selection_sha256,
               "max_new_tokens":max_new_tokens,"total_selected":len(selected),
               "processed":step,"errors":errors,"counts":dict(counts),
               "by_category":{k:dict(v) for k,v in percat.items()},
               "previews":previews}
        temp=progress_path.with_suffix(".json.tmp")
        temp.write_bytes(blob(state))
        temp.replace(progress_path)
    if not resume:checkpoint(0)
    for i,r in enumerate(selected[processed:],processed+1):
        if set(r["prompt"])-set(stoi):raise ValueError("Tokenizer OOV")
        completion,stop=greedy_original_stop(
            model,r["prompt"],stoi,itos,max_new_tokens,device)
        status,predicted,proof=evaluate_text(completion,r["answer"])
        wrong=parse_model_error(completion,r["answer"])
        eligible=(stop=="period" and wrong is not None and status=="math_error")
        if eligible:
            negative=r["prompt"]+(completion if completion.startswith(" ") else " "+completion)
            if max(len(negative), len(r["prompt"]+" The answer is "+r["answer"]+".")) + 4 > 256:
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
        if i%200==0:
            checkpoint(i)
            print(f"PROGRESS {i}/{len(selected)} usable_wrong={len(errors)}",flush=True)
        if max_questions_per_job and i-processed>=max_questions_per_job:
            checkpoint(i)
            if i<len(selected):
                print(f"PARTIAL {i}/{len(selected)} usable_wrong={len(errors)}; "
                      "submit 'sbatch scripts/round9_mine.sbatch full --resume' to continue",flush=True)
                return
    checkpoint(len(selected))
    outm={
        "version":"round9_mined_v1","mode":mode,
        "source_structured_sha256":source_sha256,
        "source_metadata_sha256":metadata_sha256,
        "base_checkpoint_sha256":base_sha256,
        "max_new_tokens":max_new_tokens,"decode":"greedy original-stop",
        "eligible_unique_training_prompts":len(index),
        "evaluated_unique_prompts":len(selected),"usable_wrong_unique":len(errors),
        "counts":dict(counts),
        "by_category":{cat:dict(percat[cat]) for cat in KINDS},
        "previews":previews,"errors_sha256":hashlib.sha256(blob(errors)).hexdigest()
    }
    with (dest/"errors.json").open("xb") as f:f.write(blob(errors))
    with (dest/"manifest.json").open("xb") as f:f.write(blob(outm))
    print(f"MINE PASS mode={mode}, usable={len(errors)}/{len(selected)} => {dest}")



def finalize_partial(preview_only=False):
    """CPU-only: validate a durable partial FULL progress snapshot before early finalization."""
    rows, metadata, manifest, protected = inputs()
    folder = OUT / "full"
    progress_path = folder / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError("No resumable full/progress.json; nothing to finalize")
    if (folder / "manifest.json").exists() or (folder / "errors.json").exists():
        raise RuntimeError("Existing finalized outputs; refusing to overwrite")

    state = read(progress_path)
    if state.get("version") != "round9_mining_progress_v1":
        raise RuntimeError("Unknown progress format")
    checks = {
        "mode":"full",
        "source_structured_sha256":manifest["files_sha256"]["structured"],
        "source_metadata_sha256":manifest["files_sha256"]["structured_metadata"],
        "base_checkpoint_sha256":sha(BASE),
        "max_new_tokens":80,
    }
    for name, expected in checks.items():
        if state.get(name) != expected:
            raise RuntimeError(f"Frozen mining progress mismatch: {name}")

    # Regenerate original deterministic selection on CPU. This guarantees that
    # the 24k processed questions correspond to the previously fixed TRAIN-only
    # ordering, and not to hand-selected validation examples.
    unique = {}
    for item in metadata:
        key = tuple(item["fact_key"])
        if key in protected:
            continue
        prompt = item["prompt"]
        if prompt not in unique:
            unique[prompt] = {
                "prompt":prompt, "answer":item["answer"],
                "category":kind(prompt), "source":item["source"],
                "fact_key":item["fact_key"]
            }
        elif unique[prompt]["answer"] != item["answer"]:
            raise RuntimeError("Conflicting answers for one prompt")

    groups = defaultdict(list)
    for item in unique.values():
        groups[item["category"]].append(item)
    randomizer = random.Random(4200919)
    selected = []
    for category in KINDS:
        ordered = sorted(groups[category], key=lambda x:x["prompt"])
        new = [x for x in ordered if x["source"] == "new"]
        old = [x for x in ordered if x["source"] != "new"]
        randomizer.shuffle(new)
        randomizer.shuffle(old)
        selected.extend(new + old)
    randomizer.shuffle(selected)
    expected_selection_sha = hashlib.sha256(
        blob([r["prompt"] for r in selected])
    ).hexdigest()
    processed = state.get("processed")
    if (state.get("selected_prompt_sha256") != expected_selection_sha or
        state.get("total_selected") != len(selected) or
        not isinstance(processed, int) or
        not 0 < processed <= len(selected)):
        raise RuntimeError("Selection hash or processed offset mismatch")

    all_errors = state["errors"]
    if len(all_errors) != state["counts"].get("numeric_wrong", 0):
        raise RuntimeError("Saved numerical error count mismatch")
    examined = {r["prompt"]:r for r in selected[:processed]}
    accepted = {}
    for error in all_errors:
        prompt = error["prompt"]
        if prompt not in examined or prompt in accepted:
            raise RuntimeError("Mined error was not a distinct examined TRAIN question")
        expected = examined[prompt]
        fact = tuple(error["fact_key"])
        if (error["answer"] != expected["answer"] or
            fact != tuple(expected["fact_key"]) or
            fact in protected or stable_split(fact,42) != "train" or
            parse_relation(prompt,error["answer"]) != fact):
            raise RuntimeError("Mined answer/fact protection mismatch")
        if (error.get("stop_reason") != "period" or
            parse_model_error(error["completion"],error["answer"]) is None):
            raise RuntimeError("Saved response was not a complete math-wrong model answer")
        negative = prompt + (
            error["completion"] if error["completion"].startswith(" ")
            else " " + error["completion"])
        if error["negative"] != negative or len(negative)+4 > 256:
            raise RuntimeError("Mined negative text/context mismatch")
        accepted[prompt] = error

    candidate_rows = defaultdict(list)
    for idx, item in enumerate(metadata):
        if item["prompt"] in accepted and tuple(item["fact_key"]) not in protected:
            candidate_rows[kind(item["prompt"])].append(idx)
    max_rows = 0
    possible_by_category = {}
    for category in KINDS:
        frequencies = Counter(metadata[i]["prompt"] for i in candidate_rows[category])
        count = min(500, sum(min(n, MAX_PER_UNIQUE_PROMPT) for n in frequencies.values()))
        possible_by_category[category] = count
        max_rows += count
    replacement_capacity = min(MAX_REPLACEMENTS,max_rows)
    print(f"PARTIAL CHECK PASS processed={processed}/{len(selected)} "
          f"unique_valid_wrong={len(all_errors)}",flush=True)
    print(f"H training replacement capacity={replacement_capacity}/{MAX_REPLACEMENTS} "
          f"by_category={possible_by_category}",flush=True)
    if preview_only:
        print("PREVIEW ONLY; no files written")
        return
    if len(all_errors) < 20 or replacement_capacity < 1:
        raise RuntimeError("Insufficient usable errors to finalize partial mining")
    counts = state["counts"]
    if sum(counts.get(status,0) for status in
           ("numeric_wrong","numeric_correct","math_error","format_failure","refusal")) != processed:
        raise RuntimeError("Progress response categories do not sum to processed questions")
    audit = {
        "version":"round9_mined_v1","mode":"full",
        "partial_finalized":processed < len(selected),
        "full_selection_size":len(selected),
        "source_structured_sha256":checks["source_structured_sha256"],
        "source_metadata_sha256":checks["source_metadata_sha256"],
        "base_checkpoint_sha256":checks["base_checkpoint_sha256"],
        "selected_prompt_sha256":expected_selection_sha,
        "max_new_tokens":checks["max_new_tokens"],
        "decode":"greedy original-stop",
        "eligible_unique_training_prompts":len(unique),
        "evaluated_unique_prompts":processed,
        "usable_wrong_unique":len(all_errors),
        "counts":counts,
        "by_category":state["by_category"],
        "previews":state["previews"],
        "hardneg_replacement_capacity":replacement_capacity,
        "replacement_capacity_by_category":possible_by_category,
        "errors_sha256":hashlib.sha256(blob(all_errors)).hexdigest()
    }
    with (folder/"errors.json").open("xb") as file:
        file.write(blob(all_errors))
    with (folder/"manifest.json").open("xb") as file:
        file.write(blob(audit))
    print(f"PARTIAL FINALIZED {processed}/{len(selected)} "
          f"usable_wrong={len(all_errors)}; ready for --assemble",flush=True)



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
    per_category=Counter()
    # Round-robin allocation prevents early task types from consuming all 5000
    # presentations before e.g. x_over_a / a_over_x are even considered.
    max_per_category=500
    queues={}
    for category in KINDS:
        ids=candidates[category][:]
        rng.shuffle(ids)
        queues[category]=ids
    while len(chosen)<MAX_REPLACEMENTS:
        made_progress=False
        for category in KINDS:
            if len(chosen)>=MAX_REPLACEMENTS:break
            if per_category[category]>=max_per_category:continue
            ids=queues[category]
            while ids:
                i=ids.pop()
                prompt=meta[i]["prompt"]
                if uses[prompt]>=MAX_PER_UNIQUE_PROMPT:continue
                chosen.append(i)
                uses[prompt]+=1
                per_category[category]+=1
                made_progress=True
                break
        if not made_progress:break
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
        "mining_partial_finalized":bool(info.get("partial_finalized",False)),
        "mining_evaluated_unique_prompts":info["evaluated_unique_prompts"],
        "mining_usable_wrong_unique":info["usable_wrong_unique"],
        "hardneg_replacement_capacity":info.get("hardneg_replacement_capacity"),
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
    p.add_argument("--preview-partial",action="store_true",help="CPU-only review of saved full-mining progress and H replacement capacity")
    p.add_argument("--finalize-partial",action="store_true",help="CPU-only finalize vetted saved full-mining progress without more GPU inference")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--pilot-per-category",type=int,default=50)
    p.add_argument("--resume",action="store_true",
                   help="Continue interrupted full mining using audited progress.json")
    p.add_argument("--max-questions-per-job",type=int,default=0,
                   help="Optional chunk cap; 0 means finish all selected questions")
    args=p.parse_args()
    if sum((bool(args.mine),args.assemble,args.preview_partial,args.finalize_partial)) != 1:
        p.error("Choose one of --mine pilot|full, --assemble, --preview-partial, --finalize-partial")
    if args.pilot_per_category<1 or args.max_new_tokens<1 or args.max_questions_per_job<0:
        p.error("Positive limits required")
    if args.preview_partial:finalize_partial(preview_only=True)
    elif args.finalize_partial:finalize_partial(preview_only=False)
    elif args.assemble:assemble()
    else:mine(args.mine,args.device,args.max_new_tokens,args.pilot_per_category,
              resume=args.resume,max_questions_per_job=args.max_questions_per_job)


if __name__=="__main__":main()
