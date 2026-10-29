#!/usr/bin/env python3
"""Round3 structured 100k: isolated original-course-loss train and evaluation.

Uses actual notebook Steps 2-6 for model, tokenizer, data-loader and scheduler.
No modification of notebook, original sft checkpoint or previous DPO models.
The generated positive answer is followed by 'because'; evaluate first numeric
answer separately from the explanation, with the original period stop.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_smoke import execute_step, notebook_steps
from evaluate_ta_integer import greedy_original_stop

DATA = ROOT / "dpo/data/round3_structured_full"
RUNROOT = ROOT / "dpo/experiments"
ANSWER = re.compile(r"^\s*The answer is\s+([+-]?\d+)(?=\s|[.]|$)", re.I)
FULL_EXPLANATION = re.compile(
    r"^\s*The answer is\s+([+-]?\d+)\s+because\s+"
    r"([+-]?\d+)([+\-*/])([+-]?\d+)\s+equals\s+([+-]?\d+)\.\s*$",
    re.I,
)


def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_text(text: str, answer: str):
    """Only the first explicitly marked integer answer counts as math accuracy."""
    m = ANSWER.match(text)
    if not m:
        return "format_failure", None, False
    predicted = int(m.group(1))
    correct = predicted == int(answer)
    proof = False
    ex = FULL_EXPLANATION.fullmatch(text)
    if ex:
        try:
            a, op, b, rhs = int(ex.group(2)), ex.group(3), int(ex.group(4)), int(ex.group(5))
            if op == "+": value = a+b
            elif op == "-": value = a-b
            elif op == "*": value = a*b
            elif b: value = a/b
            else: value = float("nan")
            proof = value == rhs == predicted
        except (ValueError, ZeroDivisionError):
            proof = False
    return ("correct" if correct else "math_error"), predicted, proof


def score(model, records, stoi, itos, device, max_new_tokens):
    model.eval()
    counts, stops = Counter(), Counter()
    categories = defaultdict(Counter)
    rows=[]
    for i, item in enumerate(records):
        prompt, gold = item["prompt"], str(item["answer"])
        completion, reason = greedy_original_stop(model, prompt, stoi, itos, max_new_tokens, device)
        status, predicted, proof = evaluate_text(completion, gold)
        row={"prompt":prompt,"answer":gold,"category":item["category"],
             "completion":completion,"prediction":predicted,"status":status,
             "explanation_valid":proof,"stop_reason":reason}
        rows.append(row)
        counts[status]+=1;stops[reason]+=1;categories[item["category"]][status]+=1
        if i<5: print(f"  {prompt} => {completion!r} [{status}]",flush=True)
    n=len(records)
    return {"n":n,"correct":counts["correct"],"accuracy":counts["correct"]/n,
            "math_error":counts["math_error"],"format_failure":counts["format_failure"],
            "valid_explanations":sum(r["explanation_valid"] for r in rows),
            "stops":dict(stops),"by_category":{k:dict(v) for k,v in sorted(categories.items())}}, rows


def write_once(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
        f.write("\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--out-dir",type=Path)
    p.add_argument("--max-new-tokens",type=int,default=80)
    p.add_argument("--check-only",action="store_true")
    a=p.parse_args()
    if a.batch_size<1 or a.epochs<1 or a.max_new_tokens<1:
        p.error("All positive hyperparameters required")
    d=DATA
    manifest=json.loads((d/"manifest.json").read_text())
    for key,filename in (("train","pos_neg_pairs.json"),("train_metadata","train_metadata.json"),
                         ("val","val.json"),("test","test.json"),("train_probe","train_probe_256.json")):
        expected=manifest["sha256"].get(key)
        if not expected or hashfile(d/filename)!=expected:
            p.error(f"Data hash mismatch: {filename}")
    all_pairs=json.loads((d/"pos_neg_pairs.json").read_text())
    assert len(all_pairs)==100000
    metadata=json.loads((d/"train_metadata.json").read_text())
    val=json.loads((d/"val.json").read_text())
    probe=json.loads((d/"train_probe_256.json").read_text())
    assert len(metadata)==100000 and len(val)==768 and len(probe)==256
    if a.smoke: pairs=all_pairs[:64];a.epochs=1;a.batch_size=16
    else: pairs=all_pairs
    if len(pairs) < a.batch_size:
        p.error("Training rows must be at least one full batch")
    remainder = len(pairs) % a.batch_size
    if remainder:
        print(f"NOTE: {remainder} rows in the final incomplete batch will be skipped "
              f"per epoch by the instructor's get_batches(); "
              f"{len(pairs)//a.batch_size} full batches/epoch.", flush=True)
    print(f"PASS dataset SHA256 and lengths: total={len(all_pairs)} "
          f"selected={len(pairs)}, val={len(val)}",flush=True)
    if a.check_only:return
    dest=(a.out_dir or RUNROOT/("round3_smoke" if a.smoke else "round3_course_100k")).resolve()
    if dest==RUNROOT.resolve() or not dest.is_relative_to(RUNROOT.resolve()) or dest.exists():
        p.error("Choose a NEW, nonexisting directory inside dpo/experiments/")
    import torch
    import torch.nn.functional as F
    if a.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable: run inside a Slurm GPU allocation")
    # Read-only notebook Step 5 expects its dataset in the designated dpo_dir.
    os.chdir(ROOT)
    env={"__name__":"__round3_course_train__"}
    steps=notebook_steps()
    random.seed(42);torch.manual_seed(42)
    if a.device=="cuda":torch.cuda.manual_seed_all(42)
    execute_step(2,steps,env)
    env["device"]=a.device;env["epochs"]=a.epochs
    env["batch_size"]=a.batch_size;env["loss_mode"]="course"
    execute_step(3,steps,env)
    execute_step(4,steps,env)
    # Make sure every preference record fits and all chars are in tokenizer.
    vocab=set(env["stoi"])
    for row in pairs:
        if set(row["positive"]+row["negative"])-vocab:
            raise RuntimeError("Invalid training character")
        if max(len(row["positive"]),len(row["negative"]))+4>env["gpt"].config.block_size:
            raise RuntimeError("Training data exceeds model context")
    dest.mkdir(parents=True,exist_ok=False)
    write_once(dest/"pos_neg_pairs.json",pairs)
    env["dpo_dir"]=dest
    execute_step(5,steps,env)
    if len(env["lines"])!=len(pairs) or env["data_path"].resolve()!=(dest/"pos_neg_pairs.json").resolve():
        raise RuntimeError("Notebook did not load selected pair data")
    execute_step(6,steps,env)
    gpt=env["gpt"]
    optimizer=env["optimizer"];scheduler=env["scheduler"]
    losses=[]
    gpt.train()
    for epoch in range(a.epochs):
        total=0.0
        for neg,pos in env["get_batches"](env["lines"],a.batch_size):
            optimizer.zero_grad(set_to_none=True)
            neg_lp=env["compute_logprob"](neg)
            pos_lp=env["compute_logprob"](pos)
            pref=-F.logsigmoid((pos_lp-neg_lp)/env["beta"]).mean()
            loss=pref-0.1*pos_lp.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss epoch={epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt.parameters(),1.0)
            optimizer.step();scheduler.step()
            total+=float(loss.item())
        losses.append(total/env["steps_per_epoch"])
        print(f"Epoch {epoch+1}/{a.epochs} original course loss={losses[-1]:.6f}",flush=True)
        write_once(dest/f"epoch_{epoch+1:02d}_summary.json",
                   {"epoch":epoch+1,"mean_loss":losses[-1],"steps_per_epoch":env["steps_per_epoch"]})
    torch.save({"model_state_dict":gpt.state_dict(),
                "model_args":env["ckpt"]["model_args"]},dest/"dpo.pt")
    for split,records in (("train_probe",probe[:32] if a.smoke else probe),
                          ("val",val[:32] if a.smoke else val)):
        print(f"=== Evaluation: {split} ===",flush=True)
        report,rows=score(gpt,records,env["stoi"],env["itos"],a.device,a.max_new_tokens)
        print(f"{split} correct={report['correct']}/{report['n']} "
              f"({report['accuracy']:.2%}) format_failure={report['format_failure']} "
              f"valid_explanations={report['valid_explanations']}",flush=True)
        write_once(dest/f"{split}_predictions.json",rows)
        write_once(dest/f"{split}_summary.json",report)
    write_once(dest/"run_manifest.json",
               {"seed":42,"data_sha256":manifest["sha256"]["train"],
                "init_checkpoint":"sft/gpt.pt","loss":"original course whole-sequence mean",
                "epochs":a.epochs,"batch_size":a.batch_size,
                "steps":a.epochs*env["steps_per_epoch"],
                "smoke":a.smoke,"checkpoint":str(dest/"dpo.pt")})
    print("ROUND3 COMPLETE, isolated output:",dest,flush=True)


if __name__=="__main__":
    main()
