#!/usr/bin/env python3
"""Train Random E3-E6 from frozen Random E2, saving each epoch; NO evaluation.

The FIRST optimizer restart is at E2 -> E3 because old E2 saved weights only.
Inside this run AdamW + CosineAnnealingLR are CONTINUOUS across all epochs.
Checkpoint/state files after each epoch allow exact epoch-boundary resume.
No sealed final test or validation predictions are accessed during training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from train_round9_random_e2 import (
    verify as verify_e1, sha, DATA, EXPERIMENTS, BASE, PARENT,
    PARENT_SUMMARY
)
from pipeline_smoke import execute_step, notebook_steps
from train_round3_course import write_once

PARENT_E2=EXPERIMENTS/"round9_random_e2_restart/dpo.pt"
PARENT_E2_SUMMARY=EXPERIMENTS/"round9_random_e2_restart/summary.json"
RUN=EXPERIMENTS/"round9_random_multi_e3_e6"
SEED=42
START_EPOCH=3

def read(p):return json.loads(p.read_text(encoding="utf-8"))

def verify_parent():
    pairs, data_manifest, _, parent_e1, e1_sha=verify_e1("random")
    if not PARENT_E2.is_file() or not PARENT_E2_SUMMARY.is_file():
        raise FileNotFoundError("Completed Random E2 checkpoint and summary required")
    e2=read(PARENT_E2_SUMMARY)
    expected={
        "arm":"random_e2_restart","batch_size":64,"lr":1e-5,
        "steps":1562,"epochs":1,"seed":42,"smoke":False,
        "max_new_tokens":80,"continuous_optimizer_state":False,
        "parent_e1_checkpoint_sha256":e1_sha,
        "dataset_sha256":sha(DATA/"random_pairs.json"),
        "round9_manifest_sha256":sha(DATA/"manifest.json"),
        "original_round7_init_sha256":sha(BASE)
    }
    for k,v in expected.items():
        if e2.get(k)!=v:
            raise RuntimeError(f"Frozen Random E2 summary differs at {k}: {e2.get(k)!r}")
    if e2.get("parent_e1_evaluation") != parent_e1.get("evaluation"):
        raise RuntimeError("Random E2 lineage does not match its E1 parent")
    return pairs,data_manifest,e2,sha(PARENT_E2)

def atomic_torch_save(data,path):
    import torch
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint/state: {path}")
    tmp=path.with_suffix(path.suffix+".partial")
    if tmp.exists():
        raise RuntimeError(f"Incomplete temporary checkpoint: {tmp}")
    torch.save(data,tmp)
    tmp.rename(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--epochs",type=int,default=4,help="Extra epochs from E2; default E3..E6")
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true",help="Resume at last fully saved epoch")
    args=p.parse_args()
    if not (1<=args.epochs<=8) or args.batch_size!=64 or args.lr!=1e-5:
        p.error("Fixed protocol: 1-8 extra epochs; batch64 and LR1e-5")
    if args.smoke and args.resume:
        p.error("Do not resume a smoke run")
    pairs, manifest, e2, e2sha=verify_parent()
    if args.smoke:
        pairs=pairs[:128]
        args.batch_size=16
        args.epochs=2
    nsteps=len(pairs)//args.batch_size
    out=(EXPERIMENTS/"round9_random_multi_smoke" if args.smoke else RUN).resolve()
    if not out.is_relative_to(EXPERIMENTS.resolve()) or out==EXPERIMENTS.resolve():
        p.error("Unsafe output destination")
    print(f"VERIFIED Random E2 parent_sha={e2sha[:16]} rows={len(pairs)} "
          f"steps_per_epoch={nsteps} extra_epochs={args.epochs} "
          f"effective_epochs={START_EPOCH}..{START_EPOCH+args.epochs-1}",
          flush=True)
    if args.check_only:
        print(f"CHECK PASS; no GPU model changes; output={out}",flush=True)
        return
    import torch
    import torch.nn.functional as F
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("Use a Slurm GPU allocation")
    os.chdir(ROOT)
    random.seed(SEED);torch.manual_seed(SEED)
    if args.device=="cuda":torch.cuda.manual_seed_all(SEED)
    env={"__name__":"__random_multi__"}
    cells=notebook_steps()
    execute_step(2,cells,env)
    env.update(device=args.device,epochs=args.epochs,batch_size=args.batch_size,
               base_lr=args.lr,loss_mode="course")
    execute_step(3,cells,env)
    execute_step(4,cells,env)
    parent=torch.load(PARENT_E2,map_location="cpu",weights_only=True)
    if parent["model_args"]!=env["ckpt"]["model_args"]:
        raise RuntimeError("Model architecture changed from Random E2")
    model=env["gpt"]
    model.load_state_dict(parent["model_state_dict"],strict=True)
    vocab=set(env["stoi"])
    if any(set(r["positive"]+r["negative"])-vocab for r in pairs):
        raise RuntimeError("Training tokenizer OOV")
    config={
        "version":"round9_random_multi_v1","parent_e2_sha256":e2sha,
        "parent_e2_summary_sha256":sha(PARENT_E2_SUMMARY),
        "parent_e1_sha256":sha(PARENT),
        "dataset_sha256":sha(DATA/"random_pairs.json"),
        "round9_manifest_sha256":sha(DATA/"manifest.json"),
        "start_epoch":START_EPOCH,"extra_epochs":args.epochs,
        "rows":len(pairs),"steps_per_epoch":nsteps,"batch_size":args.batch_size,
        "lr":args.lr,"seed":SEED,"smoke":args.smoke,
        "objective":"teacher full-sequence mean preference Course Loss",
        "scheduler":"one CosineAnnealingLR over all extra epochs, eta_min=0.1*lr",
        "optimizer_restart_from_e2":True,
        "continuous_optimizer_within_e3_e6":True
    }
    progress_path=out/"train_progress.json"
    if args.resume:
        if not out.is_dir() or not progress_path.is_file():
            raise RuntimeError("No resumable output/progress found")
        progress=read(progress_path)
        if progress["config"]!=config:
            raise RuntimeError("Resume does not match frozen training protocol")
        finished=progress["completed"]
        if finished<1 or finished>args.epochs:
            raise RuntimeError("Bad completed-epoch index")
    else:
        if out.exists():
            p.error(f"Refusing existing experiment output: {out}")
        out.mkdir(parents=True,exist_ok=False)
        finished=0
        write_once(out/"config.json",config)
    # A verified symlink avoids a second large JSON serialization and prevents
    # accidentally truncated copies from being loaded by Notebook Step 5.
    source_pairs=(DATA/"random_pairs.json").resolve()
    expected_pairs_sha=config["dataset_sha256"]
    if not args.resume:
        link=out/"pos_neg_pairs.json"
        link.symlink_to(source_pairs)
    if not (out/"pos_neg_pairs.json").is_file() or sha(out/"pos_neg_pairs.json")!=expected_pairs_sha:
        raise RuntimeError("Notebook training data link differs from verified Random pairs")
    if args.resume and sha(out/"pos_neg_pairs.json")!=progress["train_copy_sha256"]:
        raise RuntimeError("Saved Random pairs changed during resume")
    print(f"TRAIN DATA VERIFIED sha={expected_pairs_sha[:16]} "
          f"path={(out/'pos_neg_pairs.json').resolve()}",flush=True)
    env["dpo_dir"]=out
    execute_step(5,cells,env)
    if len(env["lines"])!=len(pairs) or env["data_path"].resolve()!=(out/"pos_neg_pairs.json").resolve():
        raise RuntimeError("Course Step5 loaded wrong training data")
    execute_step(6,cells,env)
    if env["steps_per_epoch"]!=nsteps or env["total_steps"]!=args.epochs*nsteps:
        raise RuntimeError("Scheduler not configured across all extra epochs")
    optimizer,scheduler=env["optimizer"],env["scheduler"]
    if args.resume:
        epoch=START_EPOCH+finished-1
        wpath=out/f"epoch_e{epoch:02d}.pt"
        spath=out/f"state_e{epoch:02d}.pt"
        if sha(wpath)!=progress["last_model_sha256"] or sha(spath)!=progress["last_state_sha256"]:
            raise RuntimeError("Saved weights/state hash mismatch")
        model.load_state_dict(torch.load(wpath,map_location="cpu",weights_only=True)["model_state_dict"],
                              strict=True)
        state=torch.load(spath,map_location="cpu",weights_only=True)
        if state["epoch"]!=epoch or state["parent_e2_sha256"]!=e2sha:
            raise RuntimeError("Resume epoch/lineage mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        # Restore the exact shuffled order from the end of the previous epoch,
        # so Python's in-place random.shuffle continues identically.
        env["lines"]=state["lines"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if args.device=="cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"RESUME after E{epoch}; next=E{epoch+1}",flush=True)
    for local_epoch in range(finished,args.epochs):
        global_epoch=START_EPOCH+local_epoch
        model.train()
        total=0.0
        completed=0
        for step,(negative,positive) in enumerate(
                env["get_batches"](env["lines"],args.batch_size),1):
            optimizer.zero_grad(set_to_none=True)
            neg_lp=env["compute_logprob"](negative)
            pos_lp=env["compute_logprob"](positive)
            loss=(-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()
                  -0.1*pos_lp.mean())
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Nonfinite loss E{global_epoch} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step();scheduler.step()
            completed+=1
            total+=float(loss.item())
            if step==1 or step%250==0 or step==nsteps:
                print(f"TRAIN E{global_epoch} step={step}/{nsteps} "
                      f"loss={loss.item():.5f} lr={scheduler.get_last_lr()[0]:.8g}",
                      flush=True)
        if completed!=nsteps:
            raise RuntimeError(f"E{global_epoch}: expected {nsteps} updates, got {completed}")
        if sha(PARENT_E2)!=e2sha:
            raise RuntimeError("Frozen E2 parent unexpectedly modified")
        model_path=out/f"epoch_e{global_epoch:02d}.pt"
        state_path=out/f"state_e{global_epoch:02d}.pt"
        atomic_torch_save({"model_state_dict":model.state_dict(),
                           "model_args":parent["model_args"]},model_path)
        atomic_torch_save({
            "epoch":global_epoch,"parent_e2_sha256":e2sha,
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),
            "python_rng":random.getstate(),
            "torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all() if args.device=="cuda" else [],
            "lines":env["lines"]
        },state_path)
        progress={
            "config":config,"completed":local_epoch+1,
            "last_model_sha256":sha(model_path),
            "last_state_sha256":sha(state_path),
            "train_copy_sha256":sha(out/"pos_neg_pairs.json"),
            "history":[] if local_epoch==0 else read(progress_path)["history"]
        }
        progress["history"].append({
            "epoch":global_epoch,"steps":nsteps,"mean_train_loss":total/completed,
            "checkpoint":model_path.name,"checkpoint_sha256":progress["last_model_sha256"],
            "state":state_path.name,"state_sha256":progress["last_state_sha256"]
        })
        tmp=progress_path.with_suffix(".json.tmp")
        with tmp.open("x",encoding="utf-8") as f:
            json.dump(progress,f,indent=2,ensure_ascii=False);f.write("\n")
        tmp.replace(progress_path)
        print(f"EPOCH SAVED E{global_epoch} mean_loss={total/completed:.6f} "
              f"weights={model_path.name} state={state_path.name}",flush=True)
    write_once(out/"train_manifest.json",progress)
    print(f"MULTI TRAIN PASS E{START_EPOCH}..E{START_EPOCH+args.epochs-1} "
          f"no validation inference performed; {out}",flush=True)

if __name__=="__main__":main()
