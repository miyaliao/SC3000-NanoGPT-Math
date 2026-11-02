#!/usr/bin/env python3
"""Round9 Random: one more epoch with a FRESH optimizer/scheduler.

Initial weights: completed Round9 Random E1 (not Round7 Replay directly).
Exact original 100k Random dataset, instructor Course Loss, LR=1e-5,
batch64, seed42 and frozen evaluation.  Optimizer + scheduler states were
NOT saved in Round9 E1, so this is a RESTARTED continuation, not a
strict uninterrupted second epoch. Does not overwrite any prior artifacts.
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
PARENT=EXPERIMENTS/"round9_random/dpo.pt"
PARENT_SUMMARY=EXPERIMENTS/"round9_random/summary.json"
R3=ROOT/"dpo/data/round3_structured_full"
R4=ROOT/"dpo/data/round4_matched_full"
OOD=EXPERIMENTS/"round1c_incorrect/common_integer_val_256.json"
PAIRS={"random":DATA/"random_pairs.json"}


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
    if expected is None or not PAIRS[arm].is_file() or sha(PAIRS[arm])!=expected:
        raise RuntimeError(f"Unverified {arm} training dataset")
    pairs=read(PAIRS[arm])
    if len(pairs)!=100000:raise RuntimeError("100000 rows required")
    # E2 MUST train on the exact same Random data as E1; no new dataset generation.
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
    if not PARENT.is_file() or not PARENT_SUMMARY.is_file():
        raise FileNotFoundError("Round9 Random E1 checkpoint/summary is missing")
    parent=read(PARENT_SUMMARY)
    if (parent.get("arm")!="random" or parent.get("smoke") or
        parent.get("round9_manifest_sha256")!=sha(DATA/"manifest.json") or
        parent.get("dataset_sha256")!=sha(PAIRS["random"]) or
        parent.get("init_checkpoint_sha256")!=manifest["base_checkpoint_sha256"] or
        parent.get("seed")!=42 or parent.get("steps")!=1562 or
        parent.get("batch_size")!=64 or parent.get("lr")!=1e-5 or
        parent.get("epochs")!=1):
        raise RuntimeError("Parent Round9 Random E1 protocol is not the expected frozen one")
    return pairs,manifest,core+ext,parent,sha(PARENT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("random",),default="random")
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
    pairs,manifest,teacher,parent,parent_sha=verify(args.arm)
    if args.smoke:
        pairs=pairs[:128]
        args.batch_size=16
    nsteps=len(pairs)//args.batch_size
    if nsteps<=0:raise RuntimeError("No batches")
    print(f"VERIFIED arm={args.arm} rows={len(pairs)} steps={nsteps} "
          f"parent_random_e1_sha={parent_sha[:16]} "
          f"restart_optimizer=True",flush=True)
    if args.check_only:
        print("CHECK PASS; no GPU model changes")
        return
    out=(args.out_dir or EXPERIMENTS/
         ("round9_smoke_random_e2_restart" if args.smoke else "round9_random_e2_restart")).resolve()
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
    # Round9 E1 saved weights only. Reinitialize AdamW/scheduler once for this one-epoch continuation.
    original=torch.load(PARENT,map_location="cpu",weights_only=True)
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
    if sha(BASE)!=manifest["base_checkpoint_sha256"] or sha(PARENT)!=parent_sha:
        raise RuntimeError("Frozen baseline/parent checkpoint modified")
    write_once(out/"summary.json",{
        "arm":"random_e2_restart","round9_manifest_sha256":sha(DATA/"manifest.json"),
        "parent_e1_checkpoint":str(PARENT),
        "parent_e1_checkpoint_sha256":parent_sha,
        "parent_e1_summary_sha256":sha(PARENT_SUMMARY),
        "parent_e1_evaluation":parent["evaluation"],
        "initialization":"Round9 Random E1 weights, newly initialized AdamW and cosine scheduler",
        "continuous_optimizer_state":False,
        "dataset_sha256":sha(PAIRS[args.arm]),
        "init_checkpoint_sha256":parent_sha,
        "original_round7_init_sha256":sha(BASE),
        "loss":"original teacher full-sequence mean preference Course Loss",
        "seed":42,"batch_size":args.batch_size,"lr":args.lr,
        "epochs":1,"steps":nsteps,"mean_train_loss":total/nsteps,
        "max_new_tokens":args.max_new_tokens,"smoke":args.smoke,
        "evaluation":reports
    })
    print(f"ROUND9 CONTINUATION PASS arm=random_e2_restart "
          f"checkpoint={out/'dpo.pt'}",flush=True)


if __name__=="__main__":main()
