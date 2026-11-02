#!/usr/bin/env python3
"""Round10: E5->T1..T4, independent Control/Targeted, per-epoch snapshots.

Same 20k step budget, batch64, lr3e-6, teacher Course Loss, seed42.
Fresh AdamW at E5->T1 because E5 stores weights only; continuous AdamW+cosine
inside each four-epoch run. No evaluation during training. Exact epoch boundary
resume saves optimizer, scheduler, shuffled order, and RNG states.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from pipeline_smoke import execute_step,notebook_steps
from train_round3_course import write_once
from build_round10_elementary import OUT as DATA,E5,sha,read

EXPERIMENTS=ROOT/"dpo/experiments"
SEED=42
EPOCHS=4
LR=3e-6
BATCH=64

def preflight(arm):
    m=read(DATA/"manifest.json")
    if m["version"]!="round10_elementary_20k_v1" or m["rows"]!=20000:
        raise RuntimeError("Round10 dataset manifest invalid")
    if m["e5_checkpoint_sha256"]!=sha(E5):
        raise RuntimeError("Frozen E5 parent changed")
    source=DATA/f"{arm}_pairs.json"
    if sha(source)!=m["files_sha256"][source.name]:
        raise RuntimeError("Round10 dataset was modified")
    rows=read(source)
    if len(rows)!=20000 or any(not x["positive"].endswith(".") or not x["negative"].endswith(".") for x in rows):
        raise RuntimeError("Unexpected Round10 pair count/format")
    return rows,m,sha(source),sha(E5)

def atomic_save(data,p):
    import torch
    if p.exists():raise FileExistsError(p)
    tmp=p.with_suffix(p.suffix+".partial")
    if tmp.exists():raise RuntimeError(f"Stale checkpoint temporary file {tmp}")
    torch.save(data,tmp)
    tmp.rename(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("control","targeted"),required=True)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.smoke and args.resume:p.error("Cannot resume a smoke")
    allrows,m,source_sha,parent_sha=preflight(args.arm)
    rows=allrows[:128] if args.smoke else allrows
    batch=16 if args.smoke else BATCH
    epochs=2 if args.smoke else EPOCHS
    steps=len(rows)//batch
    out=EXPERIMENTS/(f"round10_smoke_{args.arm}" if args.smoke else f"round10_{args.arm}_e5_t1_t4")
    config={
        "version":"round10_train_v1","arm":args.arm,
        "dataset_manifest_sha256":sha(DATA/"manifest.json"),
        "source_pairs_sha256":source_sha,"parent_e5_sha256":parent_sha,
        "batch_size":batch,"lr":LR,"seed":SEED,"epochs":epochs,
        "steps_per_epoch":steps,"rows":len(rows),"smoke":args.smoke,
        "objective":"original full-sequence mean instructor preference Course Loss",
        "optimizer":"AdamW, fresh at E5->T1, continuous T1..T4",
        "scheduler":"CosineAnnealingLR T_max=(epochs*steps), eta_min=0.1*lr",
        "inference_during_training":False
    }
    print(f"ROUND10 VERIFIED arm={args.arm} E5_sha={parent_sha[:16]} "
          f"rows={len(rows)} steps_per_epoch={steps} epochs={epochs}",flush=True)
    if args.check_only:
        print("ROUND10 CHECK PASS; no model changes",flush=True);return
    import torch
    import torch.nn.functional as F
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("Requires Slurm GPU allocation")
    os.chdir(ROOT)
    random.seed(SEED);torch.manual_seed(SEED)
    if args.device=="cuda":torch.cuda.manual_seed_all(SEED)
    env={"__name__":"__round10__"}
    cells=notebook_steps()
    execute_step(2,cells,env)
    env.update(device=args.device,epochs=epochs,batch_size=batch,base_lr=LR,loss_mode="course")
    execute_step(3,cells,env)
    execute_step(4,cells,env)
    parent=torch.load(E5,map_location="cpu",weights_only=True)
    if parent["model_args"]!=env["ckpt"]["model_args"]:
        raise RuntimeError("E5 architecture differs from original SFT")
    model=env["gpt"]
    model.load_state_dict(parent["model_state_dict"],strict=True)
    if any(set(x["positive"]+x["negative"])-set(env["stoi"]) for x in rows):
        raise RuntimeError("Unrecognized tokenizer character in Round10 data")
    out=out.resolve()
    progress_path=out/"train_progress.json"
    if args.resume:
        progress=read(progress_path)
        if progress["config"]!=config:raise RuntimeError("Resume protocol differs")
        completed=progress["completed"]
        if not 1<=completed<=epochs:raise RuntimeError("Invalid resume epoch")
    else:
        if out.exists():p.error(f"Refusing existing output directory: {out}")
        out.mkdir(parents=True,exist_ok=False)
        write_once(out/"config.json",config)
        completed=0
        if args.smoke:
            write_once(out/"pos_neg_pairs.json",rows)
        else:
            (out/"pos_neg_pairs.json").symlink_to((DATA/f"{args.arm}_pairs.json").resolve())
    if args.smoke:
        if len(read(out/"pos_neg_pairs.json"))!=len(rows):
            raise RuntimeError("Smoke source count mismatched")
    elif sha(out/"pos_neg_pairs.json")!=source_sha:
        raise RuntimeError("Verified Round10 source changed during training")
    env["dpo_dir"]=out
    execute_step(5,cells,env)
    if (len(env["lines"])!=len(rows) or
        env["data_path"].resolve()!=(out/"pos_neg_pairs.json").resolve()):
        raise RuntimeError("Notebook Step 5 loaded wrong data")
    execute_step(6,cells,env)
    if env["steps_per_epoch"]!=steps or env["total_steps"]!=steps*epochs:
        raise RuntimeError("Incorrect continuous cosine scheduler")
    optimizer,scheduler=env["optimizer"],env["scheduler"]
    if args.resume:
        last=completed
        modelpath=out/f"epoch_t{last:02d}.pt"
        statepath=out/f"state_t{last:02d}.pt"
        if sha(modelpath)!=progress["last_model_sha256"] or sha(statepath)!=progress["last_state_sha256"]:
            raise RuntimeError("Epoch-boundary resume hash mismatch")
        weights=torch.load(modelpath,map_location="cpu",weights_only=True)
        state=torch.load(statepath,map_location="cpu",weights_only=True)
        if state["t"]!=last or state["parent_e5_sha256"]!=parent_sha:
            raise RuntimeError("Resume provenance mismatch")
        model.load_state_dict(weights["model_state_dict"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        env["lines"]=state["lines"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if args.device=="cuda":torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"RESUME after T{last} / T{epochs}",flush=True)
    for t in range(completed+1,epochs+1):
        model.train();total=0.;done=0
        for step,(neg,pos) in enumerate(env["get_batches"](env["lines"],batch),1):
            optimizer.zero_grad(set_to_none=True)
            neg_lp=env["compute_logprob"](neg)
            pos_lp=env["compute_logprob"](pos)
            loss=(-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()
                  -0.1*pos_lp.mean())
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Nonfinite loss {args.arm} T{t} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            optimizer.step();scheduler.step()
            total+=float(loss.item());done+=1
            if step==1 or step%100==0 or step==steps:
                print(f"TRAIN arm={args.arm} T{t} {step}/{steps} "
                      f"loss={loss.item():.5f} lr={scheduler.get_last_lr()[0]:.9f}",flush=True)
        if done!=steps:raise RuntimeError("Unexpected batch count")
        if sha(E5)!=parent_sha:raise RuntimeError("Immutable E5 altered")
        weightpath=out/f"epoch_t{t:02d}.pt"
        statepath=out/f"state_t{t:02d}.pt"
        atomic_save({"model_args":parent["model_args"],
                     "model_state_dict":model.state_dict()},weightpath)
        atomic_save({"t":t,"parent_e5_sha256":parent_sha,
                     "optimizer":optimizer.state_dict(),
                     "scheduler":scheduler.state_dict(),
                     "python_rng":random.getstate(),
                     "torch_rng":torch.get_rng_state(),
                     "cuda_rng":torch.cuda.get_rng_state_all() if args.device=="cuda" else [],
                     "lines":env["lines"]},statepath)
        history=[] if t==1 else read(progress_path)["history"]
        h={"t":t,"steps":steps,"mean_loss":total/done,
           "weights":weightpath.name,"weights_sha256":sha(weightpath),
           "state":statepath.name,"state_sha256":sha(statepath)}
        history.append(h)
        report={"config":config,"completed":t,"history":history,
                "last_model_sha256":h["weights_sha256"],
                "last_state_sha256":h["state_sha256"]}
        temp=progress_path.with_suffix(".json.tmp")
        with temp.open("x",encoding="utf-8") as f:
            json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
        temp.replace(progress_path)
        print(f"EPOCH SAVED arm={args.arm} T{t} mean_loss={total/done:.6f} "
              f"checkpoint={weightpath.name}",flush=True)
    write_once(out/"train_manifest.json",report)
    print(f"ROUND10 TRAIN PASS arm={args.arm} T1..T{epochs}; no evaluation run",flush=True)

if __name__=="__main__":main()
