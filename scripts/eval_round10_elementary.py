#!/usr/bin/env python3
"""Round10 unified delayed evaluation: E5, Control T1..T4, Targeted T1..T4.

Only ORIGINAL frozen Val768, TrainProbe256, Teacher680 and new transparent
elementary development probes. No OOD256, no Extended384, no sealed final Test.
Per-checkpoint/per-split outputs support safe reruns if a 6h GPU job stops.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_round10_elementary import OUT as DATA,E5,sha,read
from train_round3_course import score,write_once
from compare_round9 import paired,inspect
from evaluate_teacher_style_680 import validate as validate680
from generate_round1 import parse_relation

MUL_CATS=("multiply","divide","x_times_a","a_times_x","x_over_a","a_over_x")

def basic_slices(val_predictions):
    """Read-only subgroups OF Val768, not an independent extra benchmark."""
    groups={"product_10_99":[], "direct_multiply_10_99":[],
            "carry_10_99":[]}
    for item in val_predictions:
        if item["category"] not in MUL_CATS:continue
        fact=parse_relation(item["prompt"],str(item["answer"]))
        if fact[0]!="mul":raise RuntimeError("Val product category mismatched fact")
        if not 10<=int(fact[3])<=99:continue
        groups["product_10_99"].append(item)
        if item["category"]=="multiply":
            groups["direct_multiply_10_99"].append(item)
        a,b=map(int,fact[1:3])
        if 2<=a<=9 and b>=10 and a*(b%10)>=10:
            groups["carry_10_99"].append(item)
    return groups

EXP=ROOT/"dpo/experiments"
OUTPUT=EXP/"round10_joint_eval"
R3=ROOT/"dpo/data/round3_structured_full"
TASKS={
    "val":(R3/"val.json",768),
    "train_probe":(R3/"train_probe_256.json",256)
}
KNOWN_E5={"val":614,"teacher680":521,"train_probe":226}

def verify():
    m=read(DATA/"manifest.json")
    if m.get("version")!="round10_elementary_20k_v1" or sha(E5)!=m["e5_checkpoint_sha256"]:
        raise RuntimeError("Round10 dataset or parent E5 changed")
    for name,expected in m["files_sha256"].items():
        if sha(DATA/name)!=expected:raise RuntimeError(f"Dataset/benchmark SHA mismatch: {name}")
    core,ext,_,_=validate680()
    bench=read(DATA/"basic_probe.json")
    probes=read(DATA/"reserved_probes.json")
    if len(bench)!=m["basic_probe_n"] or len(probes)!=m["reserved_probes_n"]:
        raise RuntimeError("Round10 new probe count changed")
    from generate_round1 import parse_relation,stable_split
    facts=set()
    for record in bench:
        fact=parse_relation(record["prompt"],str(record["answer"]))
        if stable_split(fact,42)!="val" or fact!=tuple(record["fact_key"]):
            raise RuntimeError("New basic probe invalid/partition changed")
        facts.add(fact)
    if len(facts)!=m["basic_probe_distinct_facts"]:
        raise RuntimeError("New probe unique-fact count mismatch")
    models={"e5":(E5,sha(E5))}
    for arm in ("control","targeted"):
        folder=EXP/f"round10_{arm}_e5_t1_t4"
        manifest_path=folder/"train_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"First finish {arm} training: {manifest_path}")
        t=read(manifest_path)
        config=t["config"]
        if (config.get("version")!="round10_train_v1" or
            config.get("arm")!=arm or config.get("epochs")!=4 or
            config.get("smoke") or config.get("parent_e5_sha256")!=sha(E5) or
            config.get("dataset_manifest_sha256")!=sha(DATA/"manifest.json") or
            config.get("source_pairs_sha256")!=sha(DATA/f"{arm}_pairs.json") or
            config.get("batch_size")!=64 or config.get("lr")!=3e-6 or
            config.get("steps_per_epoch")!=312):
            raise RuntimeError(f"{arm} training protocol mismatch")
        if t["completed"]!=4 or len(t["history"])!=4:
            raise RuntimeError(f"Unfinished {arm} training")
        for i,row in enumerate(t["history"],1):
            f=folder/row["weights"]
            if row["t"]!=i or sha(f)!=row["weights_sha256"]:
                raise RuntimeError(f"{arm} T{i} checkpoint corrupt")
            models[f"{arm}_t{i}"]=(f,row["weights_sha256"])
    tasks={key:read(path) for key,(path,n) in TASKS.items()}
    tasks["teacher680"]=core+ext
    if bench: tasks["basic_probe"]=bench
    tasks["reserved_probes"]=probes
    for name,rows in tasks.items():
        if name in TASKS and len(rows)!=TASKS[name][1]:
            raise RuntimeError(f"Frozen eval {name} size mismatch")
        if not rows:raise RuntimeError(f"Empty evaluation split {name}")
    return models,tasks

def recover(pred):
    from collections import Counter,defaultdict
    counts=Counter(r["status"] for r in pred)
    stops=Counter(r["stop_reason"] for r in pred)
    cats=defaultdict(Counter)
    for r in pred:cats[r["category"]][r["status"]]+=1
    n=len(pred)
    return {"n":n,"correct":counts["correct"],"accuracy":counts["correct"]/n,
            "math_error":counts["math_error"],"format_failure":counts["format_failure"],
            "valid_explanations":sum(r["explanation_valid"] for r in pred),
            "stops":dict(stops),
            "by_category":{k:dict(v) for k,v in sorted(cats.items())}}

def compare(models,tasks):
    report={"version":"round10_paired_v1",
            "source":"Frozen E5; two 20k curricula, 4 epochs each",
            "benchmark_warning":"Basic probe may be small and VAL-split only; Teacher680 is historical Seen facts",
            "datasets":list(tasks),"models":{},"paired":{}}
    results={}
    for name in models:
        results[name]={}
        report["models"][name]={}
        for split in tasks:
            rows=read(OUTPUT/name/f"{split}_predictions.json")
            if len(rows)!=len(tasks[split]):raise RuntimeError("Incorrect inference sample count")
            stat=inspect(rows)
            results[name][split]=rows
            report["models"][name][split]=stat
            print(f"RESULT {name:15} {split:15} {stat['correct']}/{stat['n']}",flush=True)
            if name=="e5" and split in KNOWN_E5 and stat["correct"]!=KNOWN_E5[split]:
                raise RuntimeError(f"E5 evaluation regression {split}: {stat['correct']}")
    # Extra basic skills are strictly diagnostic slices of the SAME frozen
    # Val768 predictions. They share questions with Val and are NOT
    # independent validation samples or an additional testing source.
    subgroup_names=("product_10_99","direct_multiply_10_99","carry_10_99")
    report["val_basic_diagnostics"]={
        "description":"Subgroups of frozen Val768 only; counts overlap Val768, not independent",
        "models":{},"paired":{}
    }
    subgroups={}
    for name in models:
        subgroups[name]=basic_slices(results[name]["val"])
        report["val_basic_diagnostics"]["models"][name]={}
        for group in subgroup_names:
            records=subgroups[name][group]
            report["val_basic_diagnostics"]["models"][name][group]=(
                inspect(records) if records else {"correct":0,"n":0,"accuracy":None})
            n=len(records)
            print(f"VAL BASIC {name:15} {group:24} "
                  f"{sum(x['status']=='correct' for x in records)}/{n}",
                  flush=True)
    for t in range(1,5):
        key=f"t{t}"
        report["val_basic_diagnostics"]["paired"][key]={}
        for group in subgroup_names:
            old=subgroups["e5"][group]
            control=subgroups[f"control_t{t}"][group]
            targeted=subgroups[f"targeted_t{t}"][group]
            if not old:
                report["val_basic_diagnostics"]["paired"][key][group]=None
                continue
            report["val_basic_diagnostics"]["paired"][key][group]={
                "e5_vs_control":paired(old,control),
                "e5_vs_targeted":paired(old,targeted),
                "control_vs_targeted":paired(control,targeted)
            }
    for t in range(1,5):
        c=f"control_t{t}";h=f"targeted_t{t}"
        report["paired"][f"t{t}"]={}
        for split in tasks:
            base=results["e5"][split]
            cc=results[c][split]
            tt=results[h][split]
            report["paired"][f"t{t}"][split]={
                "e5_vs_control":paired(base,cc),
                "e5_vs_targeted":paired(base,tt),
                "control_vs_targeted":paired(cc,tt)
            }
            pair=report["paired"][f"t{t}"][split]["control_vs_targeted"]
            print(f"PAIRED T{t} {split:15} Target-Control "
                  f"delta={pair['right_minus_left']:+d} "
                  f"control_only={pair['left_only']} target_only={pair['right_only']}",
                  flush=True)
        # Direct multiply is a key targeted metric within Val768, not a new independent test.
        for cat in ("multiply","divide","x_over_a"):
            a=[x for x in results[c]["val"] if x["category"]==cat]
            b=[x for x in results[h]["val"] if x["category"]==cat]
            report["paired"][f"t{t}"][f"val_{cat}"]=paired(a,b)
    output=OUTPUT/"comparison.json"
    if output.exists():raise FileExistsError(output)
    write_once(output,report)
    print(f"ROUND10 COMPARISON PASS {output}",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--check-only",action="store_true")
    args=ap.parse_args()
    models,tasks=verify()
    print(f"ROUND10 EVAL VERIFIED models={len(models)} "
          f"splits={list(tasks)} basic_probe_n={len(tasks.get('basic_probe',[]))}",
          flush=True)
    if args.check_only:
        print("EVAL CHECK PASS; no GPU inference",flush=True);return
    import torch
    from model import GPT,GPTConfig
    if args.device=="cuda" and not torch.cuda.is_available():
        ap.error("Run evaluation with Slurm GPU")
    with (ROOT/"sft/meta.pkl").open("rb") as f:meta=pickle.load(f)
    stoi,itos=meta["stoi"],meta["itos"]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    parent_args=torch.load(E5,map_location="cpu",weights_only=True)["model_args"]
    for name,(ckptpath,hashval) in models.items():
        out=OUTPUT/name
        summarypath=out/"summary.json"
        if summarypath.exists():
            summary=read(summarypath)
            if (summary["checkpoint_sha256"]!=hashval or
                set(summary["evaluation"])!=set(tasks)):
                raise RuntimeError(f"Stale previous result: {name}")
            print(f"SKIP completed {name}",flush=True);continue
        out.mkdir(parents=True,exist_ok=True)
        checkpoint=torch.load(ckptpath,map_location="cpu",weights_only=True)
        if checkpoint["model_args"]!=parent_args:
            raise RuntimeError(f"{name} architecture mismatch")
        model=GPT(GPTConfig(**checkpoint["model_args"]))
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        model.to(args.device).eval()
        scores={}
        for split,records in tasks.items():
            preds_path=out/f"{split}_predictions.json"
            metrics_path=out/f"{split}_summary.json"
            if preds_path.exists() and metrics_path.exists():
                pred=read(preds_path);stat=read(metrics_path)
                if len(pred)!=len(records) or stat["correct"]!=sum(x["status"]=="correct" for x in pred):
                    raise RuntimeError(f"Bad cached {name}/{split}")
            elif preds_path.exists() and not metrics_path.exists():
                pred=read(preds_path)
                if len(pred)!=len(records):
                    raise RuntimeError(f"Partial prior predictions {name}/{split}")
                stat=recover(pred)
                write_once(metrics_path,stat)
            elif metrics_path.exists():
                raise RuntimeError(f"Missing saved predictions {name}/{split}")
            else:
                stat,pred=score(model,records,stoi,itos,args.device,80)
                write_once(preds_path,pred)
                write_once(metrics_path,stat)
            scores[split]=stat
            print(f"EVAL {name:15} {split:15} {stat['correct']}/{stat['n']}",flush=True)
        if sha(ckptpath)!=hashval:
            raise RuntimeError(f"Checkpoint changed during {name} inference")
        write_once(summarypath,{"name":name,"checkpoint_sha256":hashval,
                                "max_new_tokens":80,"evaluation":scores})
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
    if not (OUTPUT/"comparison.json").exists():
        compare(models,tasks)
    else:print("Existing comparison preserved",flush=True)
    print("ROUND10 EVAL PASS; no sealed Test used",flush=True)

if __name__=="__main__":main()
