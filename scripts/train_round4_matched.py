#!/usr/bin/env python3
"""Round4 matched continued fine-tuning from the SAME frozen Round3 checkpoint.

Two arms: 'treatment' (targeted arithmetic) and 'replay_only' (extra old data).
Same model, original course loss, LR, steps, seed, evaluation, and saved probe.
Outputs separated. Run --check-only on login CPU; train only inside Slurm.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from pipeline_smoke import notebook_steps,execute_step
from train_round3_course import score,hashfile,write_once

SOURCE=ROOT/"dpo/data/round4_matched_full"
BASE=ROOT/"dpo/experiments/round3_100k/dpo.pt"
EVAL_ROOT=ROOT/"dpo/data/round3_structured_full"
OOD=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"
RUNROOT=ROOT/"dpo/experiments"


def check_data(data_dir,arm):
    m=json.loads((data_dir/"manifest.json").read_text(encoding="utf-8"))
    if m.get("seed")!=42 or m.get("profile") not in ("full","mini"):
        raise RuntimeError("Unexpected Round4 data manifest")
    for k,expected in m["source_sha256"].items():
        sources={
          "round3_manifest_sha256":EVAL_ROOT/"manifest.json",
          "round3_train_sha256":EVAL_ROOT/"pos_neg_pairs.json",
          "round3_metadata_sha256":EVAL_ROOT/"train_metadata.json",
          "round3_val_sha256":EVAL_ROOT/"val.json",
          "round3_test_sha256":EVAL_ROOT/"test.json",
          "old_val_sha256":OOD,
        }
        if hashfile(sources[k])!=expected:raise RuntimeError(f"Source was modified: {k}")
    file=data_dir/f"{arm}_pairs.json"
    if hashfile(file)!=m["file_sha256"][arm]:
        raise RuntimeError(f"Training arm SHA256 mismatch: {file}")
    if not BASE.is_file():
        raise RuntimeError("Missing Round3 initial checkpoint")
    n=m["training_arm_size"]
    if n!=(100000 if m["profile"]=="full" else 640):
        raise RuntimeError("Unexpected training row count")
    return m,file


def eval_cached(model,records,env,device,max_new_tokens,folder,split):
    print(f"=== {split}: {len(records)} fixed questions ===",flush=True)
    result,pred=score(model,records,env["stoi"],env["itos"],device,max_new_tokens)
    write_once(folder/f"{split}_summary.json",result)
    write_once(folder/f"{split}_predictions.json",pred)
    print(f"{split} correct={result['correct']}/{result['n']} "
          f"accuracy={result['accuracy']:.2%} "
          f"format_failure={result['format_failure']}",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("treatment","replay_only"),required=True)
    p.add_argument("--data-dir",type=Path,default=SOURCE)
    p.add_argument("--out-dir",type=Path)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0:
        p.error("epochs, batch size and LR must be positive")
    if args.max_new_tokens<2:p.error("max_new_tokens too small")
    data_dir=args.data_dir if args.data_dir.is_absolute() else ROOT/args.data_dir
    manifest,file=check_data(data_dir,args.arm)
    pairs=json.loads(file.read_text(encoding="utf-8"))
    if len(pairs)!=manifest["training_arm_size"]:
        raise RuntimeError("Training dataset length mismatch")
    if args.smoke:
        pairs=pairs[:128];args.epochs=1;args.batch_size=16
    if len(pairs)<args.batch_size:
        raise RuntimeError("Insufficient training rows")
    steps_per_epoch=len(pairs)//args.batch_size
    print(f"PASS Round4 data integrity, arm={args.arm}, "
          f"rows={len(pairs)}, steps/epoch={steps_per_epoch}, "
          f"skipped trailing rows={len(pairs)%args.batch_size}",flush=True)
    if args.check_only:return

    dest=(args.out_dir or RUNROOT/f"round4_{args.arm}").resolve()
    root=RUNROOT.resolve()
    if dest==root or not dest.is_relative_to(root) or dest.exists():
        p.error("Use a NEW output directory under dpo/experiments/ (no overwrite)")
    import torch
    import torch.nn.functional as F
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; submit a GPU Slurm job")
    # Use exact same seed for both arms and original notebook routines.
    os.chdir(ROOT)
    random.seed(42);torch.manual_seed(42)
    if args.device=="cuda":torch.cuda.manual_seed_all(42)
    env={"__name__":"__round4_continuation__"}
    steps=notebook_steps()
    execute_step(2,steps,env)
    env["device"]=args.device
    env["epochs"]=args.epochs
    env["batch_size"]=args.batch_size
    env["base_lr"]=args.lr  # consumed by original Step 6
    env["loss_mode"]="course"
    execute_step(3,steps,env)
    execute_step(4,steps,env) # Initializes architecture as in course Notebook
    # Important: replace SFT weights with Round3 weights for BOTH arms, strictly.
    base_sha=hashfile(BASE)
    ckpt=torch.load(BASE,map_location="cpu",weights_only=True)
    if ckpt["model_args"]!=env["ckpt"]["model_args"]:
        raise RuntimeError("Checkpoint architecture differs from original SFT")
    env["gpt"].load_state_dict(ckpt["model_state_dict"],strict=True)
    env["gpt"].train()
    vocab=set(env["stoi"])
    for row in pairs:
        if set(row["positive"]+row["negative"])-vocab:
            raise RuntimeError("Tokenizer out-of-vocab in Round4 training")
        if max(len(row["positive"]),len(row["negative"]))+4>env["gpt"].config.block_size:
            raise RuntimeError("Input exceeds block_size")
    dest.mkdir(parents=True,exist_ok=False)
    # Notebook Step5 expects this canonical filename in the per-run directory.
    write_once(dest/"pos_neg_pairs.json",pairs)
    env["dpo_dir"]=dest
    execute_step(5,steps,env)
    if len(env["lines"])!=len(pairs) or env["data_path"].resolve()!=(dest/"pos_neg_pairs.json").resolve():
        raise RuntimeError("Notebook did not read the correct Round4 dataset")
    execute_step(6,steps,env)
    if abs(env["optimizer"].param_groups[0]["lr"]-args.lr)>1e-10:
        raise RuntimeError("Learning-rate override failed")
    model=env["gpt"]
    records={
        "train_probe":json.loads((EVAL_ROOT/"train_probe_256.json").read_text(encoding="utf-8")),
        "val":json.loads((EVAL_ROOT/"val.json").read_text(encoding="utf-8")),
        "old_ood":json.loads(OOD.read_text(encoding="utf-8")),
    }
    # Do not read or score the sealed Round3 final test JSON!
    epoch_history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        total=0.0
        for batch,(negative,positive) in enumerate(env["get_batches"](env["lines"],args.batch_size),1):
            env["optimizer"].zero_grad(set_to_none=True)
            pos_lp=env["compute_logprob"](positive)
            neg_lp=env["compute_logprob"](negative)
            loss=(-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()
                  -0.1*pos_lp.mean())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite Loss at epoch={epoch} batch={batch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            env["optimizer"].step()
            env["scheduler"].step()
            total+=float(loss.item())
            if batch==1 or batch%250==0 or batch==steps_per_epoch:
                print(f"arm={args.arm} epoch={epoch}/{args.epochs} "
                      f"batch={batch}/{steps_per_epoch} loss={loss.item():.5f}",flush=True)
        if batch!=steps_per_epoch:
            raise RuntimeError("Unexpected number of batches")
        mean_loss=total/steps_per_epoch
        print(f"=== Epoch {epoch}/{args.epochs}: mean course Loss={mean_loss:.6f} ===",flush=True)
        snapshot=dest/f"dpo_epoch{epoch}.pt"
        torch.save({"model_state_dict":model.state_dict(),
                    "model_args":ckpt["model_args"]},snapshot)
        metrics={"epoch":epoch,"mean_loss":mean_loss,
                 "checkpoint":str(snapshot),"evaluations":{}}
        for split,all_rows in records.items():
            this=all_rows[:16] if args.smoke else all_rows
            epoch_dir=dest/f"eval_epoch{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            metrics["evaluations"][split]=eval_cached(model,this,env,args.device,
                                                       args.max_new_tokens,epoch_dir,split)
        epoch_history.append(metrics)
        write_once(dest/f"epoch{epoch}_results.json",metrics)
    if hashfile(BASE)!=base_sha:
        raise RuntimeError("Round3 checkpoint changed unexpectedly")
    shutil.copy2(dest/f"dpo_epoch{args.epochs}.pt",dest/"dpo.pt")
    summary={
        "arm":args.arm,"checkpoint":str(dest/"dpo.pt"),
        "init_checkpoint":str(BASE),"init_checkpoint_sha256":base_sha,
        "source_manifest_sha256":hashfile(data_dir/"manifest.json"),
        "training_data_sha256":manifest["file_sha256"][args.arm],
        "loss":"exact instructor whole-sequence mean pref + 0.1 chosen logprob",
        "seed":42,"lr":args.lr,"epochs":args.epochs,
        "batch_size":args.batch_size,"steps_per_epoch":steps_per_epoch,
        "total_updates":args.epochs*steps_per_epoch,
        "smoke":args.smoke,
        "epoch_history":epoch_history,
    }
    write_once(dest/"summary.json",summary)
    print(f"=== ROUND4 {args.arm.upper()} COMPLETE ===",flush=True)
    for split,v in epoch_history[-1]["evaluations"].items():
        print(f"{split}: {v['correct']}/{v['n']} ({v['accuracy']:.2%})",flush=True)
    print("Summary:",dest/"summary.json",flush=True)


if __name__=="__main__":main()
