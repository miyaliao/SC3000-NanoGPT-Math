#!/usr/bin/env python3
"""SC3000 DPO controlled overfitting diagnostic (no modifications to course notebook).

Train on 256 fixed Round-1C preference pairs, using the instructor's EXACT
whole-sequence-mean loss and the notebook's original Steps 2-6 helpers.

Run on a Slurm-allocated GPU:
    python scripts/overfit256_course.py --device cuda

This is NOT a replacement for the assignment notebook or a formal Round-2 run.
Original checkpoints and datasets are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from pipeline_smoke import execute_step, notebook_steps  # noqa: E402
from evaluate_ta_integer import evaluate_answer, greedy_original_stop  # noqa: E402

SOURCE_DIR = ROOT / "dpo/experiments/round1c_incorrect"
DEST_DIR = ROOT / "dpo/experiments/overfit256_course"
EXPECTED_SHA = "547d8d46f6feccfff16f6871ef54ad57424db4f5803d97efd911901128494677"
CATEGORIES = (
    "add", "subtract", "multiply", "divide", "x_plus_a", "a_plus_x",
    "x_minus_a", "a_minus_x", "x_times_a", "a_times_x", "x_over_a", "a_over_x",
)
CATEGORY_QUOTA = {name: (32 if i < 4 else 16) for i, name in enumerate(CATEGORIES)}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_rows(seed: int = 42):
    pairs_path = SOURCE_DIR / "pos_neg_pairs.json"
    meta_path = SOURCE_DIR / "round1c_train_metadata.json"
    validation_path = SOURCE_DIR / "common_integer_val_256.json"
    manifest_path = SOURCE_DIR / "round1c_manifest.json"
    for p in (pairs_path, meta_path, validation_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError("Generate Round 1C first: " + str(p))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(pairs_path) != EXPECTED_SHA or sha256(pairs_path) != manifest["file_sha256"]["train"]:
        raise RuntimeError("Round 1C training pairs SHA-256 mismatch")
    if sha256(meta_path) != manifest["file_sha256"]["metadata"]:
        raise RuntimeError("Round 1C metadata SHA-256 mismatch")
    if sha256(validation_path) != manifest["file_sha256"]["val"]:
        raise RuntimeError("Frozen validation SHA-256 mismatch")
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if len(pairs) != len(meta) or len(pairs) != 20480 or len(validation) != 256:
        raise RuntimeError("Unexpected Round 1C dataset sizes")
    by_kind: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_kind[m["category"]].append(i)
        prompt = m["prompt"]
        if pairs[i]["positive"] != f"{prompt} The answer is {m['correct_answer']}.":
            raise RuntimeError("Round 1C positive data / metadata mismatch")
        if pairs[i]["negative"] != f"{prompt} The answer is {m['wrong_answer']}.":
            raise RuntimeError("Round 1C negative data / metadata mismatch")
    rng = random.Random(seed)
    selected_indices = []
    for category in CATEGORIES:
        if len(by_kind[category]) < CATEGORY_QUOTA[category]:
            raise RuntimeError("Not enough training examples in " + category)
        selected_indices.extend(rng.sample(by_kind[category], CATEGORY_QUOTA[category]))
    rng.shuffle(selected_indices)
    selection = [pairs[i] for i in selected_indices]
    metadata = [meta[i] for i in selected_indices]
    assert len(selection) == 256
    train_prompts = {r["prompt"] for r in metadata}
    val_prompts = {r["prompt"] for r in validation}
    if len(train_prompts) != 256 or train_prompts & val_prompts:
        raise RuntimeError("Training/validation prompt overlap or duplicate")
    train_facts = {tuple(r["fact_key"]) for r in metadata}
    for item in validation:
        # Frozen Round 1C already establishes fact-group isolation; require a
        # re-check using the original exact-fraction parser to be safe.
        from generate_round1 import parse_relation
        if parse_relation(item["prompt"], str(item["answer"])) in train_facts:
            raise RuntimeError("Math-fact leakage into validation")
    assert len(selection) == sum(CATEGORY_QUOTA.values())
    return selection, metadata, validation, selected_indices


def run_evaluation(model, records, stoi, itos, device, max_new_tokens: int):
    model.eval()
    rows = []
    counts = Counter()
    by_category = defaultdict(Counter)
    for item in records:
        prompt, answer = item["prompt"], str(item.get("answer", item.get("correct_answer")))
        if set(prompt) - set(stoi):
            raise RuntimeError("Prompt contains unrecognized characters")
        text, stop = greedy_original_stop(model, prompt, stoi, itos, max_new_tokens, device)
        status, prediction = evaluate_answer(text, answer)
        row = {"prompt":prompt, "answer": answer, "category": item["category"],
               "completion": text, "status": status, "prediction": prediction, "stop_reason":stop}
        rows.append(row)
        counts[status] += 1
        by_category[item["category"]][status] += 1
    return {
        "n": len(rows), "correct": counts["correct"],
        "accuracy": counts["correct"] / len(rows),
        "refusal": counts["refusal"], "math_error": counts["math_error"],
        "format_failure": counts["format_failure"],
        "unique_predictions": len({str(r["prediction"]) for r in rows}),
        "by_category": {k: dict(v) for k,v in sorted(by_category.items())},
    }, rows


def score_preferences(env, pairs, batch_size: int):
    import torch
    model = env["gpt"]
    model.eval()
    pos_logprobs, neg_logprobs = [], []
    with torch.inference_mode():
        for j in range(0, len(pairs), batch_size):
            portion = pairs[j:j+batch_size]
            pos = [env["pad_or_truncate"](env["encode"](r["positive"] + "\n\n\n\n"), env["max_length"]) for r in portion]
            neg = [env["pad_or_truncate"](env["encode"](r["negative"] + "\n\n\n\n"), env["max_length"]) for r in portion]
            pp = env["compute_logprob"](torch.tensor(pos, device=env["device"])).detach().cpu().tolist()
            nn = env["compute_logprob"](torch.tensor(neg, device=env["device"])).detach().cpu().tolist()
            pos_logprobs.extend(pp)
            neg_logprobs.extend(nn)
    margin = [p-n for p,n in zip(pos_logprobs, neg_logprobs)]
    return {"n": len(margin), "pairwise_wins": sum(v>0 for v in margin),
            "win_rate": sum(v>0 for v in margin)/len(margin),
            "mean_full_seq_margin": sum(margin)/len(margin),
            "mean_pos_full_logprob":sum(pos_logprobs)/len(margin),
            "mean_neg_full_logprob":sum(neg_logprobs)/len(margin)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=128,
                        help="128 epochs x four batches = 512 optimizer steps by default")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or 256 % args.batch_size:
        parser.error("epochs positive; batch size must evenly divide 256")
    if args.max_new_tokens < 2:
        parser.error("max-new-tokens must be >= 2")
    data, metadata, validation, indices = select_rows(args.seed)
    print("PASS: selected 256 fixed seen questions; no validation leakage; Round 1C hashes verified",flush=True)
    if args.check_data_only:
        print("DATA CHECK PASSED. No checkpoint loaded or GPU required.",flush=True)
        return
    if DEST_DIR.exists():
        parser.error(f"Refusing to reuse existing experiment directory {DEST_DIR}. Backup and explicitly remove it if you truly wish to repeat.")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable, use Slurm to request a GPU")
    DEST_DIR.mkdir(parents=True, exist_ok=False)
    save_json(DEST_DIR / "pos_neg_pairs.json", data)
    save_json(DEST_DIR / "train_metadata.json", metadata)
    save_json(DEST_DIR / "selection.json", {"seed": args.seed, "indices": indices,
                                          "source_sha256":EXPECTED_SHA,
                                          "category_quota": CATEGORY_QUOTA})
    os.chdir(ROOT)
    env = {"__name__": "__overfit256__"}
    steps = notebook_steps()
    execute_step(2, steps, env)
    env["dpo_dir"] = DEST_DIR
    env["device"] = args.device
    env["epochs"] = args.epochs
    env["batch_size"] = args.batch_size
    execute_step(3, steps, env)
    execute_step(4, steps, env)
    execute_step(5, steps, env)
    if env["data_path"].resolve() != (DEST_DIR / "pos_neg_pairs.json").resolve() or len(env["lines"]) != 256:
        raise AssertionError("Notebook helpers loaded an unexpected dataset")
    execute_step(6, steps, env)
    model = env["gpt"]
    scorer_before = score_preferences(env, data, args.batch_size)
    print("Initial pairwise preference:",scorer_before,flush=True)
    import torch.nn.functional as F
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    optimizer = env["optimizer"]
    scheduler = env["scheduler"]
    beta = env["beta"]
    model.train()
    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_total = 0.0
        for neg_tensor,pos_tensor in env["get_batches"](env["lines"],args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            neg_logprob = env["compute_logprob"](neg_tensor)
            pos_logprob = env["compute_logprob"](pos_tensor)
            pref = -F.logsigmoid((pos_logprob-neg_logprob)/beta).mean()
            chosen = -0.1*pos_logprob.mean()
            loss = pref+chosen
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_total += float(loss.item())
        epoch_losses.append(epoch_total / env["steps_per_epoch"])
        if epoch == 0 or (epoch+1)%16==0 or epoch == args.epochs-1:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: mean course loss={epoch_losses[-1]:.6f}",flush=True)
    ckpt = DEST_DIR / "dpo.pt"
    torch.save({"model_state_dict":model.state_dict(), "model_args":env["ckpt"]["model_args"]},ckpt)
    model.eval()
    train_summary,train_preds = run_evaluation(model,metadata,env["stoi"],env["itos"],args.device,args.max_new_tokens)
    val_summary,val_preds = run_evaluation(model,validation,env["stoi"],env["itos"],args.device,args.max_new_tokens)
    scorer_after = score_preferences(env,data,args.batch_size)
    save_json(DEST_DIR / "train_predictions.json",train_preds)
    save_json(DEST_DIR / "val_predictions.json",val_preds)
    summary = {"stage":"overfit256_course_loss","init_checkpoint":"sft/gpt.pt",
               "source_round1c_sha256":EXPECTED_SHA,"sample_seed":args.seed,
               "training_size":256,"epochs":args.epochs,"batch_size":args.batch_size,
               "total_steps":args.epochs*env["steps_per_epoch"],
               "loss":"EXACT instructor preference and chosen bonus on full-sequence mean logprob",
               "initial_preference":scorer_before,"final_preference":scorer_after,
               "epoch_losses":epoch_losses,"seen_train":train_summary,"unseen_val":val_summary,
               "checkpoint":str(ckpt),"note":"Small-set memorization diagnostic; not formal validation and not final DPO model."}
    save_json(ROOT/"results/overfit256_course_summary.json",summary)
    print("=== OVERFIT256 COURSE LOSS COMPLETE ===",flush=True)
    print(f"TRAIN accuracy: {train_summary['correct']}/256 = {train_summary['accuracy']:.2%}",flush=True)
    print(f"VAL accuracy: {val_summary['correct']}/256 = {val_summary['accuracy']:.2%}",flush=True)
    print(f"Preference pair wins: {scorer_after['pairwise_wins']}/256 = {scorer_after['win_rate']:.2%}",flush=True)
    print("Initial pair wins:",scorer_before["pairwise_wins"],flush=True)
    print("Summary: results/overfit256_course_summary.json",flush=True)
    print("Checkpoint:",ckpt,flush=True)


if __name__=="__main__":
    main()
