#!/usr/bin/env python3
"""After training: evaluate ALL frozen Random E3..E6 epoch checkpoints together.

Read-only fixed Val768, Old OOD256, Extended384, Teacher680, TrainProbe256;
never reads sealed final Test768. Saves predictions immediately per checkpoint
so interrupted 6h QoS inference can be resumed without recomputing earlier
epochs or overwriting any result. Includes paired E2-vs-each-epoch reporting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from train_round9_random_multi import RUN, verify_parent, read, sha
from train_round9_random_e2 import R3, R4, OOD, EXPERIMENTS
from train_round3_course import score, write_once
from compare_round9 import paired,inspect
from evaluate_teacher_style_680 import validate as validate_teacher

# Evaluator intentionally repeats the fixed paths rather than touching Test.
E2=EXPERIMENTS/"round9_random_e2_restart"
DATASETS={
    "train_probe":(R3/"train_probe_256.json",256),
    "val":(R3/"val.json",768),
    "old_ood":(OOD,256),
    "extended_val":(R4/"extension_val.json",384)
}
EXPECTED_SPLITS={"train_probe":256,"val":768,"old_ood":256,
                 "extended_val":384,"teacher680":680}
METRICS=("train_probe","val","old_ood","extended_val","teacher680")

def filehash(p):
    if not p.is_file():raise FileNotFoundError(p)
    return sha(p)

def verify():
    _,manifest,e2,e2sha=verify_parent()
    path=RUN/"train_manifest.json"
    if not path.is_file():
        raise RuntimeError("Training is not complete: train_manifest.json missing")
    completed=read(path)
    config=completed["config"]
    if config.get("version")!="round9_random_multi_v1" or config.get("smoke"):
        raise RuntimeError("Refusing smoke/unknown training run")
    if (config["parent_e2_sha256"]!=e2sha or
        config["parent_e2_summary_sha256"]!=sha(E2/"summary.json") or
        config["dataset_sha256"]!=sha(ROOT/"dpo/data/round9_balanced_full/random_pairs.json") or
        config["round9_manifest_sha256"]!=sha(ROOT/"dpo/data/round9_balanced_full/manifest.json") or
        config["steps_per_epoch"]!=1562 or config["batch_size"]!=64):
        raise RuntimeError("Training manifest lineage mismatch")
    history=completed["history"]
    if len(history)!=config["extra_epochs"] or completed["completed"]!=len(history):
        raise RuntimeError("Training has not reached the intended final epoch")
    checkpoints=[]
    for i,x in enumerate(history):
        ep=config["start_epoch"]+i
        p=RUN/x["checkpoint"]
        if ep!=x["epoch"] or sha(p)!=x["checkpoint_sha256"]:
            raise RuntimeError(f"Tampered or missing E{ep} checkpoint")
        checkpoints.append((ep,p,x["checkpoint_sha256"]))
    if len(checkpoints)!=config["extra_epochs"]:
        raise RuntimeError("Mismatched checkpoint count")
    tcore,text,_,_=validate_teacher()
    tasks={k:read(p) for k,(p,n) in DATASETS.items()}
    tasks["teacher680"]=tcore+text
    for k,n in EXPECTED_SPLITS.items():
        if len(tasks[k])!=n:raise RuntimeError(f"{k} frozen eval size changed")
    if (sha(ROOT/"dpo/benchmarks/teacher_style_400_v1/questions.json")!=
        manifest["core400_sha256"] or
        sha(ROOT/"dpo/benchmarks/teacher_style_extension_280_v1/questions.json")!=
        manifest["extension280_sha256"]):
        raise RuntimeError("Frozen teacher SHA modified")
    for k,filekey in (("val","round3_val"),("old_ood","old_ood"),
                      ("extended_val","round4_extended_val")):
        if sha(DATASETS[k][0])!=manifest["source_hashes"][filekey]:
            raise RuntimeError(f"Frozen eval source {k} modified")
    for k in METRICS:
        p=E2/f"{k}_predictions.json"
        if not p.is_file() or len(read(p))!=EXPECTED_SPLITS[k]:
            raise RuntimeError(f"Missing E2 fixed baseline predictions: {k}")
    return config,checkpoints,tasks,e2sha

def compare_all(checkpoints,output):
    report={"version":"random_multi_eval_v1",
            "description":"Frozen matched strict greedy evaluation after all training",
            "baseline":"Round9 Random E2 restarted continuation",
            "baseline_e2_scores":{},
            "epochs":{}}
    for name in METRICS:
        base=read(E2/f"{name}_predictions.json")
        report["baseline_e2_scores"][name]=inspect(base)
    for ep,p,ck_sha in checkpoints:
        label=f"e{ep:02d}"
        epochdir=output/label
        now={}
        for name in METRICS:
            pred=read(epochdir/f"{name}_predictions.json")
            old=read(E2/f"{name}_predictions.json")
            if len(pred)!=EXPECTED_SPLITS[name]:
                raise RuntimeError(f"{label}: {name} eval size mismatch")
            res=inspect(pred)
            difference=paired(old,pred)
            now[name]={"scores":res,"paired_vs_e2":difference}
            print(f"COMPARE {label} {name:13} E2={report['baseline_e2_scores'][name]['correct']}"
                  f" NOW={res['correct']} delta={difference['right_minus_left']:+d} "
                  f"E2_only={difference['left_only']} epoch_only={difference['right_only']}",
                  flush=True)
        extended_old=read(E2/"extended_val_predictions.json")
        extended_now=read(epochdir/"extended_val_predictions.json")
        a=[x for x in extended_old if x["category"] in ("multiply","x_over_a")]
        b=[x for x in extended_now if x["category"] in ("multiply","x_over_a")]
        if len(a)!=128 or len(b)!=128:
            raise RuntimeError("Direct128 frozen subset not aligned")
        now["direct128"]={"e2_correct":inspect(a)["correct"],
                          "epoch_correct":inspect(b)["correct"],
                          "paired_vs_e2":paired(a,b)}
        print(f"COMPARE {label} direct128 E2={now['direct128']['e2_correct']} "
              f"NOW={now['direct128']['epoch_correct']}",flush=True)
        report["epochs"][label]=now
    dest=output/"all_epochs_comparison.json"
    if dest.exists():
        raise FileExistsError("Refusing to overwrite completed E3-E6 comparison")
    write_once(dest,report)
    print(f"COMPARISON PASS {dest}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--check-only",action="store_true")
    args=p.parse_args()
    config,checkpoints,tasks,e2sha=verify()
    if args.max_new_tokens!=80:
        p.error("Fixed evaluation: max_new_tokens=80")
    print(f"VERIFIED {len(checkpoints)} frozen epoch checkpoints "
          f"E{checkpoints[0][0]}..E{checkpoints[-1][0]}, "
          f"all fixed evaluation prompts; E2_sha={e2sha[:16]}",
          flush=True)
    if args.check_only:
        print("EVAL CHECK PASS; no GPU inference performed",flush=True)
        return
    import torch
    from model import GPT,GPTConfig
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("Requires Slurm GPU allocation")
    with (ROOT/"sft/meta.pkl").open("rb") as f:
        tokenizer=pickle.load(f)
    stoi,itos=tokenizer["stoi"],tokenizer["itos"]
    output=RUN/"eval"
    output.mkdir(exist_ok=True)
    baseline_model_args=torch.load(E2/"dpo.pt",map_location="cpu",
                                   weights_only=True)["model_args"]
    for epoch,path,checkpoint_sha in checkpoints:
        label=f"e{epoch:02d}"
        epout=output/label
        if (epout/"summary.json").exists():
            done=read(epout/"summary.json")
            if (done.get("checkpoint_sha256")!=checkpoint_sha or
                done.get("max_new_tokens")!=80 or
                set(done.get("evaluation",{}))!=set(METRICS)):
                raise RuntimeError(f"Stale evaluation record for {label}")
            print(f"SKIP completed {label} evaluation",flush=True)
            continue
        epout.mkdir(exist_ok=True)
        ck=torch.load(path,map_location="cpu",weights_only=True)
        if ck["model_args"]!=baseline_model_args:
            raise RuntimeError(f"{label}: architecture does not match E2")
        model=GPT(GPTConfig(**ck["model_args"]))
        model.load_state_dict(ck["model_state_dict"],strict=True)
        model.to(args.device).eval()
        reports={}
        for name,records in tasks.items():
            summarypath=epout/f"{name}_summary.json"
            predpath=epout/f"{name}_predictions.json"
            if summarypath.exists() and predpath.exists():
                metrics=read(summarypath)
                preds=read(predpath)
                if metrics["n"]!=EXPECTED_SPLITS[name] or len(preds)!=EXPECTED_SPLITS[name]:
                    raise RuntimeError(f"{label}/{name} saved size mismatch")
                if sum(x["status"]=="correct" for x in preds)!=metrics["correct"]:
                    raise RuntimeError(f"{label}/{name} saved score mismatch")
                print(f"SKIP completed {label}/{name}",flush=True)
            elif summarypath.exists() or predpath.exists():
                raise RuntimeError(f"Incomplete {label}/{name} eval files; inspect instead of overwriting")
            else:
                metrics,preds=score(model,records,stoi,itos,args.device,80)
                write_once(predpath,preds)
                write_once(summarypath,metrics)
                print(f"EVAL {label:5} {name:13} "
                      f"{metrics['correct']}/{metrics['n']}",flush=True)
            reports[name]=metrics
        if sha(path)!=checkpoint_sha:
            raise RuntimeError(f"{label} checkpoint changed during inference")
        write_once(epout/"summary.json",{
            "epoch":epoch,"checkpoint_sha256":checkpoint_sha,
            "parent_e2_sha256":e2sha,"max_new_tokens":80,
            "evaluation":reports
        })
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
        print(f"EVAL COMPLETE {label}",flush=True)
    if not (output/"all_epochs_comparison.json").exists():
        compare_all(checkpoints,output)
    else:
        print("Existing all_epochs_comparison.json preserved",flush=True)
    print("MULTI EVAL PASS; no sealed final Test touched",flush=True)

if __name__=="__main__":main()
