#!/usr/bin/env python3
"""Round6 SFT-start FOUR-ARM training, exactly ONE EPOCH per reproducible job.

Each arm uses the same sft/gpt.pt initialization, same underlying 100k math
questions/negative refusals, Course Loss, optimizer and 5-epoch cosine schedule.
One epoch per Slurm job makes 20 GPU stages resumable without losing AdamW
or dropout RNG states. Run stages with afterok dependencies per arm.

The old sealed Round3 final Test is NEVER opened/evaluated here.
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
from pipeline_smoke import notebook_steps, execute_step
from train_round3_course import hashfile,write_once
from evaluate_round6_formats import score_formats

DATA=ROOT/"dpo/data/round6_from_sft"
RUNROOT=ROOT/"dpo/experiments/round6_from_sft"
SFT=ROOT/"sft/gpt.pt"
R3=ROOT/"dpo/data/round3_structured_full"
R4=ROOT/"dpo/data/round4_matched_full"
OOD=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"
ARMS=("original","answer_only","decomposition_after","decomposition_before")
SPLITS={
    "val":(R3/"val.json",768),
    "old_ood":(OOD,256),
    "extended_val":(R4/"extension_val.json",384),
    "train_probe":(R3/"train_probe_256.json",256),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_data(arm):
    m=read(DATA/"manifest.json")
    if m["version"]!="round6_sft_four_formats_v1":
        raise ValueError("Wrong Round6 version")
    if m["train_rows_per_arm"]!=100000 or not m["same_prompt_answer_negative_order"]:
        raise ValueError("Round6 100k mathematical-question invariants failed")
    for source,path in (("original",ROOT/"dpo/data/round5_decomposition/original_pairs.json"),
                        ("decomposition",ROOT/"dpo/data/round5_decomposition/decomposition_pairs.json")):
        if hashfile(path)!=m["source_sha256"][source]:
            raise ValueError("Round5 upstream training dataset modified")
    pair_path=DATA/f"{arm}_pairs.json"
    if hashfile(pair_path)!=m["file_sha256"][arm]:
        raise ValueError("Round6 training pair SHA256 mismatch")
    if not SFT.exists():raise FileNotFoundError(SFT)
    source_manifest=read(R3/"manifest.json")
    round4_manifest=read(R4/"manifest.json")
    for name,(eval_path,expected_n) in SPLITS.items():
        if name=="old_ood":
            expected=round4_manifest["source_sha256"]["old_val_sha256"]
        elif name=="extended_val":
            expected=round4_manifest["file_sha256"]["extension_val"]
        else:
            expected=source_manifest["sha256"]["val" if name=="val" else "train_probe"]
        if hashfile(eval_path)!=expected:
            raise ValueError(f"Frozen evaluation file changed: {eval_path}")
    # NEVER return the path of the last validation split here.
    if pair_path!=DATA/f"{arm}_pairs.json":
        raise AssertionError("Training source path changed during validation checks")
    return m,pair_path


def stable_order(epoch,n):
    indices=list(range(n))
    random.Random(42006000+epoch).shuffle(indices)
    return indices


def atomic_save(obj,path):
    import torch
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+".incomplete")
    if tmp.exists():raise FileExistsError(tmp)
    try:
        torch.save(obj,tmp)
        os.replace(tmp,path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def is_safe_pretraining_recovery(out, pairs, epoch, smoke):
    """Recover a failed epoch startup without overwriting any trained state.

    The prior RNG-restore exception happened AFTER the pairs were written,
    but BEFORE the first optimizer update. Allow reuse only when the directory
    contains exactly the original pairs file and no checkpoint/evaluations.
    Any other existing content still requires manual inspection.
    """
    if not out.is_dir() or epoch<2 or smoke:
        return False
    contents={entry.name for entry in out.iterdir()}
    expected=out/"pos_neg_pairs.json"
    if contents!={"pos_neg_pairs.json"} or expected.is_symlink():
        return False
    if read(expected)!=pairs:
        raise ValueError(f"Cannot safely resume: existing pairs differ in {out}")
    return True


def load_eval_rows(smoke):
    splits={}
    for name,(path,n) in SPLITS.items():
        rows=read(path)
        if len(rows)!=n:raise ValueError(f"Wrong {name} set length: {len(rows)}")
        splits[name]=rows[:12] if smoke else rows
    return splits


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",required=True,choices=ARMS)
    p.add_argument("--epoch",type=int,required=True,choices=range(1,6))
    p.add_argument("--device",default="cuda",choices=("cuda","cpu"))
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--max-new-tokens",type=int,default=160)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--eval-only",action="store_true",
                   help="Recover evaluations using a completed training_state.pt")
    args=p.parse_args()
    if args.smoke and args.epoch!=1:p.error("Smoke only supports epoch 1")
    if args.lr<=0 or args.batch_size<=0 or args.max_new_tokens<=0:p.error("Invalid hyperparameters")
    if args.eval_only and args.check_only:p.error("Choose check-only or eval-only, not both")
    manifest,path=verify_data(args.arm)
    if path != DATA/f"{args.arm}_pairs.json":
        raise AssertionError(f"Round6 expected training pairs, got {path}")
    pairs=read(path)
    if len(pairs)!=manifest["train_rows_per_arm"]:
        raise ValueError(f"Wrong full training rows: got {len(pairs)} from {path}")
    if args.smoke:pairs=pairs[:128]
    print(f"PASS Round6 arm={args.arm} epoch={args.epoch}/5 "
          f"rows={len(pairs)} source={hashfile(path)[:16]}",flush=True)
    if args.check_only:return
    import torch
    import torch.nn.functional as F
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("A Slurm GPU allocation is required")
    os.chdir(ROOT)
    torch.manual_seed(42)
    random.seed(42)
    if args.device=="cuda":torch.cuda.manual_seed_all(42)
    env={"__name__":"__round6_sft_formats__"}
    steps=notebook_steps()
    execute_step(2,steps,env)
    env.update({
        "device":args.device,"epochs":5,"batch_size":16 if args.smoke else args.batch_size,
        "base_lr":args.lr,"loss_mode":"course",
    })
    execute_step(3,steps,env)
    execute_step(4,steps,env)  # Reads SFT gpt.pt, never Round4/5 DPO weights!
    expected_sft_sha=hashfile(SFT)
    directory=(RUNROOT/("_smoke_"+args.arm if args.smoke else args.arm))
    out=directory/f"epoch{args.epoch:02d}"
    previous=directory/f"epoch{args.epoch-1:02d}"/"training_state.pt"
    if args.epoch>1 and not args.smoke and not previous.is_file():
        raise FileNotFoundError(f"Previous epoch training state missing: {previous}")
    # A failed Epoch 2+ RNG restore can leave ONLY the unmodified
    # pos_neg_pairs.json, with no saved model or optimizer state. This exact
    # situation is safely restartable from the previous completed epoch.
    recovering=is_safe_pretraining_recovery(out,pairs,args.epoch,args.smoke) if (
        out.exists() and not args.eval_only) else False
    if out.exists() and not args.eval_only and not recovering:
        p.error(f"Refusing overwrite of prior Round6 run: {out}. "
                "Only a matching, pairs-only pretraining directory is recoverable.")
    if args.eval_only and not (out/"training_state.pt").is_file():
        p.error("Cannot eval-only without a complete training_state.pt")
    if not args.eval_only:
        vocab=set(env["stoi"])
        for i,r in enumerate(pairs):
            if set(r["positive"]+r["negative"])-vocab:
                raise ValueError(f"Unknown tokenizer characters at row {i}")
            if max(len(r["positive"]),len(r["negative"]))+4>env["gpt"].config.block_size:
                raise ValueError(f"Context overflow at row {i}")
        if recovering:
            print(f"SAFE PRE-TRAINING RECOVERY: reusing matching pairs-only {out}. "
                  "No optimizer updates or checkpoint existed in this directory.",flush=True)
        else:
            out.mkdir(parents=True,exist_ok=False)
            # Step5 expects this exact filename in the per-epoch directory.
            write_once(out/"pos_neg_pairs.json",pairs)
    env["dpo_dir"]=out
    execute_step(5,steps,env)
    if len(env["lines"])!=len(pairs):
        raise ValueError("Notebook Step5 read the wrong dataset")
    execute_step(6,steps,env)
    model=env["gpt"]
    optimizer=env["optimizer"]
    scheduler=env["scheduler"]
    num_batches=len(pairs)//env["batch_size"]
    if num_batches<=0:raise ValueError("No batches")
    init_sha=expected_sft_sha
    if args.eval_only:
        state=torch.load(out/"training_state.pt",map_location=args.device,weights_only=True)
        model.load_state_dict(state["model_state_dict"],strict=True)
        if state["epoch"]!=args.epoch:raise ValueError("Wrong epoch in recovery checkpoint")
        order_hash=state["order_hash"]
        training_loss=state["mean_loss"]
    elif args.epoch>1:
        prior=torch.load(previous,map_location=args.device,weights_only=True)
        if (prior["epoch"]!=args.epoch-1 or
            prior["init_sft_sha256"]!=init_sha or
            prior["dataset_sha256"]!=hashfile(path) or
            prior["lr"]!=args.lr or prior["batch_size"]!=env["batch_size"]):
            raise ValueError("Prior checkpoint optimizer/schedule/protocol mismatch")
        model.load_state_dict(prior["model_state_dict"],strict=True)
        optimizer.load_state_dict(prior["optimizer_state_dict"])
        scheduler.load_state_dict(prior["scheduler_state_dict"])
        # CUDA RNG state is a CPU uint8 ByteTensor. Loading the checkpoint
        # with map_location='cuda' moves these buffers onto GPU as well; move
        # RNG buffers explicitly back to CPU before restoring generators.
        cpu_rng=prior["rng_cpu"].cpu()
        cuda_rng=[state.cpu() for state in prior["rng_cuda"]]
        if cpu_rng.dtype!=torch.uint8 or any(x.dtype!=torch.uint8 for x in cuda_rng):
            raise TypeError("Saved RNG states must be CPU torch.uint8 ByteTensors")
        if args.device=="cuda" and len(cuda_rng)!=torch.cuda.device_count():
            raise ValueError(f"GPU count changed: checkpoint has {len(cuda_rng)} RNG "
                             f"states, current allocation has {torch.cuda.device_count()}")
        torch.set_rng_state(cpu_rng)
        if args.device=="cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
    if not args.eval_only:
        model.train()
        indices=stable_order(args.epoch,len(pairs))
        order_hash=hashlib.sha256(json.dumps(indices,separators=(",",":")).encode("utf-8")).hexdigest()
        runloss=0.0
        batch=0
        width=env["max_length"]
        for start in range(0,num_batches*env["batch_size"],env["batch_size"]):
            group=[env["lines"][j] for j in indices[start:start+env["batch_size"]]]
            pos=torch.tensor([env["pad_or_truncate"](
                env["encode"](row["positive"]+"\n\n\n\n"),width) for row in group],
                dtype=torch.long,device=args.device)
            neg=torch.tensor([env["pad_or_truncate"](
                env["encode"](row["negative"]+"\n\n\n\n"),width) for row in group],
                dtype=torch.long,device=args.device)
            optimizer.zero_grad(set_to_none=True)
            pos_lp=env["compute_logprob"](pos)
            neg_lp=env["compute_logprob"](neg)
            loss=-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()-0.1*pos_lp.mean()
            if not torch.isfinite(loss).item():raise RuntimeError("Nonfinite course Loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()
            batch+=1
            runloss+=float(loss.item())
            if batch==1 or batch%250==0 or batch==num_batches:
                print(f"arm={args.arm} epoch={args.epoch} batch={batch}/{num_batches} "
                      f"loss={loss.item():.5f}",flush=True)
        training_loss=runloss/num_batches
        if batch!=num_batches:raise RuntimeError("Unexpected updates")
        atomic_save({
            "epoch":args.epoch,"arm":args.arm,"model_state_dict":model.state_dict(),
            "model_args":env["ckpt"]["model_args"],
            "optimizer_state_dict":optimizer.state_dict(),
            "scheduler_state_dict":scheduler.state_dict(),
            "rng_cpu":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state_all() if args.device=="cuda" else [],
            "init_sft_sha256":init_sha,
            "dataset_sha256":hashfile(path),
            "protocol":"round6_sft_four_formats_v1",
            "lr":args.lr,"batch_size":env["batch_size"],
            "mean_loss":training_loss,"order_hash":order_hash,
            "updates_this_epoch":num_batches,
        },out/"training_state.pt")
        print(f"Saved resumable epoch state (optimizer + LR scheduler + RNG): {out}",flush=True)
    model.eval()
    reports={}
    for split,rows in load_eval_rows(args.smoke).items():
        pred_path=out/f"{split}_predictions.json"
        stat_path=out/f"{split}_summary.json"
        if pred_path.exists() and stat_path.exists():
            report=read(stat_path)
            if report["n"]!=len(rows):raise ValueError("Prior evaluation count mismatch")
            print(f"Existing {split} evaluation reused",flush=True)
        else:
            if pred_path.exists() or stat_path.exists():
                raise ValueError(f"Partial evaluation exists for {split}, inspect before retry")
            print(f"Evaluating {split} N={len(rows)}",flush=True)
            report,preds=score_formats(model,rows,env["stoi"],env["itos"],
                                      args.device,args.max_new_tokens)
            write_once(pred_path,preds)
            write_once(stat_path,report)
        reports[split]=report
        print(f"{split:12} strict={report['strict_correct']}/{report['n']} "
              f"final={report['final_correct']}/{report['n']} "
              f"pre-answer={report['reason_before_answer']} "
              f"valid-steps={report['valid_calculation']}",flush=True)
    summary=out/"epoch_summary.json"
    if not summary.exists():
        write_once(summary,{
            "arm":args.arm,"epoch":args.epoch,
            "initial_checkpoint":"sft/gpt.pt","init_sft_sha256":init_sha,
            "dataset_sha256":hashfile(path),
            "source_manifest_sha256":hashfile(DATA/"manifest.json"),
            "loss":"original instructor Course Loss, full-sequence-mean",
            "seed":42,"order_sha256":order_hash,
            "batch_size":env["batch_size"],"training_rows":len(pairs),
            "updates_this_epoch":num_batches,
            "total_schedule_updates":5*num_batches,
            "lr":args.lr,"mean_loss":training_loss,
            "max_new_tokens":args.max_new_tokens,
            "eval_protocol":"round6 strict and final-answer paired scoring",
            "evaluation":reports,"smoke":args.smoke,
        })
    if args.epoch==5 and not args.smoke:
        final=directory/"dpo.pt"
        if not final.exists():
            atomic_save({"model_state_dict":model.state_dict(),
                         "model_args":env["ckpt"]["model_args"]},final)
        print(f"Final arm model: {final}",flush=True)
    if hashfile(SFT)!=init_sha:
        raise RuntimeError("Initial SFT checkpoint modified during training")
    print(f"PASS Round6 {args.arm} epoch{args.epoch:02d} complete",flush=True)


if __name__=="__main__":
    run()
