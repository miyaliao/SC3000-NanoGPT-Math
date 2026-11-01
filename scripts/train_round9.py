#!/usr/bin/env python3
"""Round9: 4-arm matched DPO continuation from frozen Round7 Replay.

Arms replay / random / structured / hardneg. Original teacher Course Loss,
identical seed, schedule, one epoch and strict, frozen evaluations.
No final Round3 Test access or past checkpoint overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from pipeline_smoke import execute_step,notebook_steps
from train_round3_course import score,write_once
from evaluate_teacher_style_680 import validate as validate_680
from generate_round1 import parse_relation,stable_split

DATA=ROOT/"dpo/data/round9_balanced_full"
EXPERIMENTS=ROOT/"dpo/experiments"
BASE=EXPERIMENTS/"round7_replay/dpo.pt"
R3=ROOT/"dpo/data/round3_structured_full"
R4=ROOT/"dpo/data/round4_matched_full"
OOD=EXPERIMENTS/"round1c_incorrect/common_integer_val_256.json"
PAIRS={a:DATA/(a+"_pairs.json") for a in ("replay","random","structured","hardneg")}


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(arm):
    manifest=read(DATA/"manifest.json")
    if manifest["version"]!="round9_structured_v1" or manifest["train_rows"]!=100000:
        raise RuntimeError("Wrong Round9 dataset manifest")
    if sha(BASE)!=manifest["base_checkpoint_sha256"]:
        raise RuntimeError("Frozen Round7 Replay initialization mismatch")
    if sha(ROOT/"dpo/data/round5_decomposition/original_pairs.json")!=manifest["old_100k_sha256"]:
        raise RuntimeError("Original Round5 training source modified")
    frozen_files={
        "round3_val":R3/"val.json",
        "old_ood":OOD,
        "round4_extended_val":R4/"extension_val.json"
    }
    for key,path in frozen_files.items():
        if sha(path)!=manifest["source_hashes"][key]:
            raise RuntimeError(f"Frozen {key} changed")
    core,ext,_,_=validate_680()
    if sha(ROOT/"dpo/benchmarks/teacher_style_400_v1/questions.json")!=manifest["core400_sha256"] or \
       sha(ROOT/"dpo/benchmarks/teacher_style_extension_280_v1/questions.json")!=manifest["extension280_sha256"]:
        raise RuntimeError("Frozen teacher benchmark changed")
    expected=manifest["files_sha256"].get(arm)
    if arm=="hardneg":
        hm=read(DATA/"hardneg_manifest.json")
        if hm["version"]!="round9_hardneg_v1" or \
           hm["base_checkpoint_sha256"]!=manifest["base_checkpoint_sha256"] or \
           hm["structured_sha256"]!=manifest["files_sha256"]["structured"]:
            raise RuntimeError("HardNeg is not matched to Round9 Structured")
        expected=hm["hardneg_sha256"]
    if expected is None or not PAIRS[arm].is_file() or sha(PAIRS[arm])!=expected:
        raise RuntimeError(f"Unverified {arm} training dataset")
    pairs=read(PAIRS[arm])
    if len(pairs)!=100000:raise RuntimeError("100000 rows required")
    if arm=="hardneg":
        source=read(PAIRS["structured"])
        differences=0
        for a,b in zip(source,pairs):
            if a["positive"]!=b["positive"]:
                raise RuntimeError("HardNeg modified a Chosen or row order")
            differences+=(a["negative"]!=b["negative"])
        if differences!=hm["replaced_rows"]:
            raise RuntimeError("HardNeg replacement count disagrees with manifest")
    # Other pairs are already audited on build, but re-check here before GPU.
    from train_round3_course import FULL_EXPLANATION
    for row in pairs:
        if not row["positive"].endswith(".") or not row["negative"].endswith("."):
            raise RuntimeError("Missing terminal period")
        # Do NOT accept a newly generated row whose numeric answer lacks math verification.
        pos=row["positive"]
        stem=pos.split(" The answer is ",1)
        if len(stem)!=2:raise RuntimeError("Positive prefix malformed")
        gold=stem[1].split(" because ",1)[0]
        if not gold.lstrip("-").isdigit():raise RuntimeError("Noninteger Chosen")
        key=parse_relation(stem[0],gold)
        if stable_split(key,42)!="train":
            raise RuntimeError("Protected non-train fact")
        if max(len(row["positive"]),len(row["negative"]))+4>256:
            raise RuntimeError("Over blocksize")
    return pairs,manifest,core+ext


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=tuple(PAIRS),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out-dir",type=Path)
    args=p.parse_args()
    if args.epochs!=1 or args.batch_size<1 or args.lr<=0 or args.max_new_tokens<1:
        p.error("Round9 study uses exactly 1 epoch; positive batch/LR/limit required")
    pairs,manifest,teacher=verify(args.arm)
    if args.smoke:
        pairs=pairs[:128]
        args.batch_size=16
    nsteps=len(pairs)//args.batch_size
    if nsteps<=0:raise RuntimeError("No batches")
    print(f"VERIFIED arm={args.arm} rows={len(pairs)} steps={nsteps} "
          f"base_sha={manifest['base_checkpoint_sha256'][:16]}",flush=True)
    if args.check_only:
        print("CHECK PASS; no GPU model changes")
        return
    out=(args.out_dir or EXPERIMENTS/
         ("round9_smoke_"+args.arm if args.smoke else "round9_"+args.arm)).resolve()
    if out.exists() or not out.is_relative_to(EXPERIMENTS.resolve()) or out==EXPERIMENTS.resolve():
        p.error(f"Refusing existing or unsafe output: {out}")
    import torch
    import torch.nn.functional as F
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("Use a Slurm CUDA GPU allocation")
    os.chdir(ROOT)
    random.seed(42);torch.manual_seed(42)
    if args.device=="cuda":torch.cuda.manual_seed_all(42)
    env={"__name__":"__round9__"}
    steps_nb=notebook_steps()
    execute_step(2,steps_nb,env)
    env["device"]=args.device
    env["epochs"]=1;env["batch_size"]=args.batch_size
    env["base_lr"]=args.lr;env["loss_mode"]="course"
    execute_step(3,steps_nb,env)
    execute_step(4,steps_nb,env)
    original=torch.load(BASE,map_location="cpu",weights_only=True)
    if original["model_args"]!=env["ckpt"]["model_args"]:
        raise RuntimeError("Model architecture changed")
    env["gpt"].load_state_dict(original["model_state_dict"],strict=True)
    vocab=set(env["stoi"])
    for i,row in enumerate(pairs):
        if set(row["positive"]+row["negative"])-vocab:
            raise RuntimeError(f"Tokenizer OOV row {i}")
    out.mkdir(parents=True,exist_ok=False)
    write_once(out/"pos_neg_pairs.json",pairs)
    env["dpo_dir"]=out
    execute_step(5,steps_nb,env)
    if len(env["lines"])!=len(pairs) or \
       env["data_path"].resolve()!=(out/"pos_neg_pairs.json").resolve():
        raise RuntimeError("Wrong training file loaded")
    execute_step(6,steps_nb,env)
    model=env["gpt"];optimizer=env["optimizer"];scheduler=env["scheduler"]
    model.train()
    total=0.
    for step,(negative,positive) in enumerate(env["get_batches"](env["lines"],args.batch_size),1):
        optimizer.zero_grad(set_to_none=True)
        neg_lp=env["compute_logprob"](negative)
        pos_lp=env["compute_logprob"](positive)
        loss=(-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()
              -0.1*pos_lp.mean())
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite preference loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        optimizer.step();scheduler.step()
        total+=float(loss.item())
        if step==1 or step%250==0 or step==nsteps:
            print(f"TRAIN {args.arm} {step}/{nsteps} loss={loss.item():.5f}",flush=True)
    if step!=nsteps:raise RuntimeError("Unexpected training updates")
    torch.save({"model_state_dict":model.state_dict(),
                "model_args":original["model_args"]},out/"dpo.pt")
    tasks={
        "train_probe":read(R3/"train_probe_256.json"),
        "val":read(R3/"val.json"),
        "old_ood":read(OOD),
        "extended_val":read(R4/"extension_val.json"),
        "teacher680":teacher
    }
    reports={}
    for name,records in tasks.items():
        take=records[:12] if args.smoke else records
        metrics,preds=score(model,take,env["stoi"],env["itos"],
                            args.device,args.max_new_tokens)
        write_once(out/f"{name}_summary.json",metrics)
        write_once(out/f"{name}_predictions.json",preds)
        reports[name]=metrics
        print(f"EVAL {name}: {metrics['correct']}/{metrics['n']}",flush=True)
    if sha(BASE)!=manifest["base_checkpoint_sha256"]:
        raise RuntimeError("Frozen source modified")
    write_once(out/"summary.json",{
        "arm":args.arm,"round9_manifest_sha256":sha(DATA/"manifest.json"),
        "dataset_sha256":sha(PAIRS[args.arm]),
        "init_checkpoint_sha256":sha(BASE),
        "loss":"original teacher full-sequence mean preference Course Loss",
        "seed":42,"batch_size":args.batch_size,"lr":args.lr,
        "epochs":1,"steps":nsteps,"mean_train_loss":total/nsteps,
        "max_new_tokens":args.max_new_tokens,"smoke":args.smoke,
        "evaluation":reports
    })
    print(f"ROUND9 PASS arm={args.arm} checkpoint={out/'dpo.pt'}",flush=True)


if __name__=="__main__":main()
