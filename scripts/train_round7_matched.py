#!/usr/bin/env python3
"""Round7 matched 1-epoch replay vs targeted multiplication fine-tuning.

Both arms start the EXACT same frozen Round5-A checkpoint. Training rows
differ (standard replay vs train-only targeted arithmetic), not Course Loss,
batch count, initialization, LR or fixed evaluation. Sealed final Test and
public prior-year Blind sets are never read. Never overwrite the base.
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
from train_round3_course import score, write_once, hashfile

DATA=ROOT/"dpo/data/round7_targeted_full"
RUNROOT=ROOT/"dpo/experiments"
SOURCE=ROOT/"dpo/data/round4_matched_full"
BASE=ROOT/"dpo/experiments/round5_original/dpo.pt"
ROUND3=ROOT/"dpo/data/round3_structured_full"
OLD_OOD=ROOT/"dpo/experiments/round1c_incorrect/common_integer_val_256.json"
ARMS={"replay":"replay_pairs.json","targeted":"targeted_pairs.json"}


def verify(arm):
    m=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    if (m.get("version")!="round7_matched_mult_v1" or
        m.get("profile")!="full" or m.get("rows_per_arm")!=100000 or
        m.get("same_shared_old_records")!=70000 or
        m.get("target_old_hard_records")!=15000 or
        m.get("target_new_math_records")!=15000):
        raise RuntimeError("Round7 matched full dataset manifest unexpected")
    old_data=ROOT/"dpo/data/round5_decomposition/original_pairs.json"
    if hashfile(old_data)!=m["source_training_sha256"]["old_100k_sha256"]:
        raise RuntimeError("Round7 data source SHA mismatch")
    old_manifest=ROOT/"dpo/data/round5_decomposition/manifest.json"
    if hashfile(old_manifest)!=m["source_training_sha256"]["round5_manifest_sha256"]:
        raise RuntimeError("Round7 source manifest SHA mismatch")
    src=DATA/ARMS[arm]
    if hashfile(src)!=m["file_sha256"][arm]:
        raise RuntimeError("Round7 training arm SHA256 mismatch")
    if not BASE.is_file():
        raise FileNotFoundError(f"Missing frozen Round5-A checkpoint: {BASE}")
    source_m=json.loads((SOURCE/"manifest.json").read_text(encoding="utf-8"))
    old_m=json.loads((ROUND3/"manifest.json").read_text(encoding="utf-8"))
    for name,path,expected in (
      ("Round4 extended val",SOURCE/"extension_val.json",
       source_m["file_sha256"]["extension_val"]),
      ("Round3 val",ROUND3/"val.json",old_m["sha256"]["val"]),
      ("Round3 old OOD",OLD_OOD,
       source_m["source_sha256"]["old_val_sha256"]),
    ):
        if hashfile(path)!=expected or hashfile(path)!=m["source_training_sha256"][{
            "Round4 extended val":"extended_val_sha256",
            "Round3 val":"round3_val_sha256",
            "Round3 old OOD":"old_ood_sha256"
        }[name]]:
            raise RuntimeError(f"Frozen evaluation file changed: {name}")
    return m,src


def summarize_split(model, env, device, rows, split, dest, limit):
    print(f"Evaluating {split}: {len(rows)} fixed records",flush=True)
    report,predictions=score(model,rows,env["stoi"],env["itos"],device,limit)
    write_once(dest/f"{split}_summary.json",report)
    write_once(dest/f"{split}_predictions.json",predictions)
    print(f"{split}: {report['correct']}/{report['n']} = {report['accuracy']:.2%}, "
          f"format_failures={report['format_failure']}",flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=tuple(ARMS),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--out-dir",type=Path)
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.epochs!=1:
        p.error("Round7 matched pilot locked to 1 epoch")
    if args.batch_size<1 or args.lr<=0 or args.max_new_tokens<1:
        p.error("Invalid learning rate or batch size")
    manifest,source=verify(args.arm)
    pairs=json.loads(source.read_text(encoding="utf-8"))
    if len(pairs)!=100000:
        raise RuntimeError("Expected full 100k paired dataset")
    if args.smoke:
        pairs=pairs[:128]
        args.batch_size=16
    steps=len(pairs)//args.batch_size
    if not steps:
        raise RuntimeError("No complete training batches")
    print(f"PASS Round7 arm={args.arm}, rows={len(pairs)}, epochs=1, "
          f"batch={args.batch_size}, steps={steps}, "
          f"last_incomplete={len(pairs)%args.batch_size}",flush=True)
    print(f"Round7: targeted hard old={manifest['target_old_hard_records']}, "
          f"new synthetic TRAIN-only={manifest['target_new_math_records']}",flush=True)
    if args.check_only:
        return
    out=(args.out_dir or RUNROOT/f"round7_{args.arm}").resolve()
    base_root=RUNROOT.resolve()
    if out.exists() or out==base_root or not out.is_relative_to(base_root):
        p.error(f"Refusing output path (exists/not under experiments): {out}")

    import torch
    import torch.nn.functional as F
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("GPU unavailable: run in a Slurm GPU allocation")
    os.chdir(ROOT)
    random.seed(42)
    torch.manual_seed(42)
    if args.device=="cuda":torch.cuda.manual_seed_all(42)
    env={"__name__":"__round7_matched__"}
    steps_nb=notebook_steps()
    execute_step(2,steps_nb,env)
    env["device"]=args.device
    env["epochs"]=1
    env["batch_size"]=args.batch_size
    env["base_lr"]=args.lr
    env["loss_mode"]="course"
    execute_step(3,steps_nb,env)
    execute_step(4,steps_nb,env)
    init_sha=hashfile(BASE)
    checkpoint=torch.load(BASE,map_location="cpu",weights_only=True)
    if checkpoint["model_args"]!=env["ckpt"]["model_args"]:
        raise RuntimeError("Round4 checkpoint architecture changed")
    env["gpt"].load_state_dict(checkpoint["model_state_dict"],strict=True)
    vocab=set(env["stoi"])
    for i,pair in enumerate(pairs):
        if set(pair["positive"]+pair["negative"])-vocab:
            raise RuntimeError(f"Tokenizer OOV at row {i}")
        if max(len(pair["positive"]),len(pair["negative"]))+4>env["gpt"].config.block_size:
            raise RuntimeError(f"Sequence longer than model block at row {i}")
    out.mkdir(parents=True,exist_ok=False)
    write_once(out/"pos_neg_pairs.json",pairs)
    env["dpo_dir"]=out
    execute_step(5,steps_nb,env)
    if (len(env["lines"])!=len(pairs) or
        env["data_path"].resolve()!=(out/"pos_neg_pairs.json").resolve()):
        raise RuntimeError("Wrong training pair dataset loaded")
    execute_step(6,steps_nb,env)
    model=env["gpt"]
    optimizer=env["optimizer"]
    scheduler=env["scheduler"]
    model.train()
    running=0.0
    for batch,(negative,positive) in enumerate(env["get_batches"](env["lines"],args.batch_size),1):
        optimizer.zero_grad(set_to_none=True)
        neg_lp=env["compute_logprob"](negative)
        pos_lp=env["compute_logprob"](positive)
        loss=(-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()
              -0.1*pos_lp.mean())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite Loss in batch {batch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running+=float(loss.item())
        if batch==1 or batch%250==0 or batch==steps:
            print(f"arm={args.arm} batch={batch}/{steps} loss={loss.item():.5f}",flush=True)
    if batch!=steps:
        raise RuntimeError("Round7 optimizer update count mismatch")
    mean_loss=running/steps
    print(f"Original Course Loss mean={mean_loss:.6f}",flush=True)
    ckpt_path=out/"dpo.pt"
    torch.save({"model_state_dict":model.state_dict(),
                "model_args":checkpoint["model_args"]},ckpt_path)
    # All eval records are frozen and identical for the two arms. DO NOT evaluate
    # the independent Round3 final Test; its labels are reserved for final work.
    records={
      "train_probe":json.loads((ROUND3/"train_probe_256.json").read_text(encoding="utf-8")),
      "val":json.loads((ROUND3/"val.json").read_text(encoding="utf-8")),
      "old_ood":json.loads(OLD_OOD.read_text(encoding="utf-8")),
      "extended_val":json.loads((SOURCE/"extension_val.json").read_text(encoding="utf-8")),
    }
    report={}
    for name,rows in records.items():
        current=rows[:16] if args.smoke else rows
        report[name]=summarize_split(model,env,args.device,current,name,out,args.max_new_tokens)
    if hashfile(BASE)!=init_sha:
        raise RuntimeError("Base checkpoint modified unexpectedly")
    write_once(out/"summary.json",{
      "arm":args.arm,"init_checkpoint":str(BASE),"init_sha256":init_sha,
      "source_round7_manifest_sha256":hashfile(DATA/"manifest.json"),
      "source_training_sha256":manifest["file_sha256"][args.arm],
      "loss":"original Course Loss, full-sequence mean",
      "seed":42,"lr":args.lr,"epochs":1,
      "batch_size":args.batch_size,"updates":steps,
      "mean_loss":mean_loss,"smoke":args.smoke,
      "target_new_synthetic_rows_full":manifest["target_new_math_records"],
      "shared_old_train_rows":manifest["same_shared_old_records"],
      "evaluation":report,"checkpoint":str(ckpt_path)
    })
    print("=== ROUND7 COMPLETED ===",args.arm,flush=True)
    print("Summary:",out/"summary.json",flush=True)


if __name__=="__main__":
    main()
