#!/usr/bin/env python3
"""SC3000 Overfit-256: only add gold-answer period NLL to answer-focused loss.

This is an isolated *diagnostic ablation*, not the instructor's original loss.
Everything except the additional chosen-answer end-token NLL remains matched to
scripts/overfit256_answer_focused.py: exact selected 256 pairs, initial SFT
checkpoint, 128 epochs, 64 batch, optimizer and scheduler, and frozen val.

Run from the repository root:
  python scripts/overfit256_termination_aware.py --check-data-only
  sbatch scripts/overfit256_termination_aware.sbatch

Never writes to Round 1A/B/C, course-loss, or answer-focused experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from overfit256_course import (  # noqa: E402
    CATEGORY_QUOTA, EXPECTED_SHA, run_evaluation, save_json,
    score_preferences, select_rows,
)
from pipeline_smoke import execute_step, notebook_steps  # noqa: E402

DEFAULT_DEST = ROOT / "dpo/experiments/overfit256_termination_aware"
SUMMARY_PATH = ROOT / "results/overfit256_termination_aware_summary.json"


def final_period_mask(input_ids, stoi):
    """Select the one period AFTER each integer answer, in shifted targets.

    input_ids is [B,L] with ID 0 (newline) serving as padding.  All examples
    are '<question>=? The answer is <integer>.', with exactly one dot, hence
    the final non-padding target must be '.'.  The returned [B,L-1] mask
    indexes *next-token targets*, never the character preceding the period.
    """
    import torch

    if input_ids.ndim != 2 or input_ids.size(1) < 3:
        raise ValueError("Expected token IDs [B,L], L>=3")
    targets = input_ids[:, 1:]
    valid = targets != 0
    if not bool(valid.any(dim=1).all().item()):
        raise ValueError("Empty sample in batch")
    dot = stoi["."]
    if not bool(((targets == dot) & valid).sum(dim=1).eq(1).all().item()):
        raise ValueError("Termination-Aware experiment requires one final period, no decimals")
    positions = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
    last = torch.where(valid, positions, -1).max(dim=1).values
    if not bool(targets.gather(1, last.unsqueeze(1)).eq(dot).all().item()):
        raise ValueError("Last non-padding character is not the end period")
    return positions == last[:, None]


def termination_aware_scores(env, input_ids):
    """Same answer-focused scores PLUS one teacher-forced end-period score.

    Returns response_sum, response_mean, number_mean, stop_logprob, stop_top1.
    Compute them from a SINGLE forward pass for efficiency and alignment.
    """
    import torch
    import torch.nn.functional as F

    logits, _ = env["gpt"](input_ids[:, :-1], full_seq=True)
    targets = input_ids[:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    response_mask, number_mask = env["answer_focused_masks"](input_ids)
    stop_mask = final_period_mask(input_ids, env["stoi"])
    if not bool((response_mask & stop_mask).eq(stop_mask).all().item()):
        raise AssertionError("End period must fall within the answer-response mask")
    if bool((number_mask & stop_mask).any().item()):
        raise AssertionError("End period must NOT be counted as a number digit")

    resp_sum = (token_logprobs * response_mask).sum(dim=1)
    resp_mean = resp_sum / response_mask.sum(dim=1).clamp_min(1)
    number_sum = (token_logprobs * number_mask).sum(dim=1)
    number_mean = number_sum / number_mask.sum(dim=1).clamp_min(1)
    # The only term added to the previous Answer-Focused loss:
    stop_logprob = (token_logprobs * stop_mask).sum(dim=1)
    stop_top1 = ((logits.argmax(dim=-1) == env["stoi"]["."]) & stop_mask).any(dim=1)
    return resp_sum, resp_mean, number_mean, stop_logprob, stop_top1


def self_test_masks(env):
    """Check shifted target alignment, including 1- and 4-digit negatives."""
    import torch

    examples = [
        ("79-7=?", "72"),
        ("x+55=95,x=?", "40"),
        ("5-9=?", "-4"),
        ("x/17=80,x=?", "1360"),
        ("9*1=?", "9"),
    ]
    encoded = []
    for p, answer in examples:
        sample = f"{p} The answer is {answer}."
        if len(sample) + 4 > env["max_length"]:
            raise RuntimeError("Example exceeds training max_length")
        encoded.append(env["pad_or_truncate"](env["encode"](sample+"\n\n\n\n"),env["max_length"]))
    ids = torch.tensor(encoded, dtype=torch.long, device=env["device"])
    stop_mask = final_period_mask(ids, env["stoi"])
    resp_mask, num_mask = env["answer_focused_masks"](ids)
    for i, (p, ans) in enumerate(examples):
        labels = ids[i, 1:].tolist()
        def selected(mask):
            return env["decode"]([int(t) for t, m in zip(labels, mask[i].tolist()) if m])
        assert selected(resp_mask) == f" The answer is {ans}."
        assert selected(num_mask) == ans
        assert selected(stop_mask) == "."
        # Important: it should select the final period, not the preceding digit.
        correct_pos = len(f"{p} The answer is {ans}")
        observed_positions = torch.nonzero(stop_mask[i], as_tuple=False).flatten().tolist()
        assert observed_positions == [correct_pos - 1], (p, observed_positions, correct_pos)
    print("PASS: shifted stop masks select ONLY the period after the correct integer", flush=True)


def score_preference_and_end(env, pairs, batch_size):
    """Read-only preference/stop statistics on 256 seen training pairs."""
    import torch

    env["gpt"].eval()
    margins, chosen_stop_lp, chosen_end_top1, chosen_number_lp = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start+batch_size]
            to_t = lambda key: torch.tensor([
                env["pad_or_truncate"](env["encode"](r[key]+"\n\n\n\n"),env["max_length"])
                for r in chunk
            ], dtype=torch.long,device=env["device"])
            pos = to_t("positive")
            neg = to_t("negative")
            p_sum, _, p_num, p_stop, p_top1 = termination_aware_scores(env,pos)
            n_sum, _, _, _, _ = termination_aware_scores(env,neg)
            margins.extend((p_sum - n_sum).cpu().tolist())
            chosen_stop_lp.extend(p_stop.cpu().tolist())
            chosen_end_top1.extend(p_top1.cpu().tolist())
            chosen_number_lp.extend(p_num.cpu().tolist())
    n = len(margins)
    return {
        "n": n,
        "pairwise_wins": sum(x>0 for x in margins),
        "pairwise_win_rate": sum(x>0 for x in margins)/n,
        "mean_completion_sum_margin": sum(margins)/n,
        "mean_gold_stop_logprob": sum(chosen_stop_lp)/n,
        "mean_gold_stop_probability": sum(math.exp(v) for v in chosen_stop_lp)/n,
        "gold_stop_top1_rate": sum(chosen_end_top1)/n,
        "mean_gold_number_logprob": sum(chosen_number_lp)/n,
    }


def generation_diagnostics(rows):
    """Never score a mere correct-number prefix as an actual correct answer."""
    n = len(rows)
    counts = Counter(r["stop_reason"] for r in rows)
    prefix = " The answer is "
    correct_prefix = sum(
        r["completion"].startswith(prefix + str(r["answer"])) for r in rows
    )
    exact_end = sum(
        r["completion"] == prefix + str(r["answer"]) + "." for r in rows
    )
    return {
        "n": n,
        "period_stops": counts["period"],
        "period_stop_rate": counts["period"] / n,
        "newline_stops": counts["newline"],
        "max_token_stops": counts["max_new_tokens"],
        "gold_answer_prefix_count": correct_prefix,
        "gold_answer_prefix_rate": correct_prefix / n,
        "exact_answer_and_period_count": exact_end,
        "exact_answer_and_period_rate": exact_end / n,
        "note": "Prefix count is diagnostic, NOT mathematical accuracy.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--completion-weight", type=float, default=0.25)
    p.add_argument("--end-weight", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DEST)
    p.add_argument("--summary-path", type=Path, default=None,
                   help="Unique summary JSON inside results/ for this experiment")
    p.add_argument("--check-data-only", action="store_true")
    args = p.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or 256 % args.batch_size
            or args.max_new_tokens < 1 or args.end_weight <= 0
            or args.answer_weight < 0 or args.completion_weight < 0):
        p.error("Invalid training settings/weights")
    if args.seed != 42:
        p.error("Fixed 256-question comparison requires seed=42")

    pairs, metadata, validation, indices = select_rows(args.seed)
    if len(pairs) != 256 or len(validation) != 256:
        raise AssertionError("Unexpected dataset sizes")
    print("PASS: exactly the same 256 train/val pairs as previous Overfit-256 experiments",flush=True)
    if args.check_data_only:
        print("DATA CHECK PASSED: no GPU needed, source data hash verified",flush=True)
        return

    dest = (args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir).resolve()
    experiment_root = (ROOT/"dpo/experiments").resolve()
    if dest == experiment_root or not dest.is_relative_to(experiment_root) or dest.exists():
        p.error("Specify a NEW output directory below dpo/experiments/ (never overwrite)")
    if args.summary_path is None:
        summary_path = SUMMARY_PATH if dest == DEFAULT_DEST.resolve() else ROOT / "results" / f"{dest.name}_summary.json"
    else:
        summary_path = args.summary_path if args.summary_path.is_absolute() else ROOT / args.summary_path
    summary_path = summary_path.resolve()
    results_root = (ROOT / "results").resolve()
    if summary_path == results_root or not summary_path.is_relative_to(results_root):
        p.error("Summary path must be a file below results/")
    if summary_path.exists():
        p.error(f"Summary exists; refusing to overwrite {summary_path}")

    import torch
    import torch.nn.functional as F
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; use sbatch for GPU")
    os.chdir(ROOT)
    env = {"__name__": "__overfit256_termination_aware__"}
    steps = notebook_steps()
    execute_step(2,steps,env)
    env["device"] = args.device
    env["epochs"] = args.epochs
    env["batch_size"] = args.batch_size
    env["loss_mode"] = "answer_focused"
    env["answer_weight"] = args.answer_weight
    env["completion_weight"] = args.completion_weight
    execute_step(3,steps,env)
    if not all(k in env for k in ("answer_focused_masks","compute_answer_focused_logprob")):
        raise RuntimeError("Notebook Step 3 lacks answer-focused helper functions")
    execute_step(4,steps,env)
    env["max_length"] = max(env["max_length"], max(len(x["positive"])+4 for x in pairs))
    self_test_masks(env)
    dest.mkdir(parents=True,exist_ok=False)
    save_json(dest/"pos_neg_pairs.json",pairs)
    save_json(dest/"train_metadata.json",metadata)
    save_json(dest/"selection.json",{"seed":args.seed,"indices":indices,"source_sha256":EXPECTED_SHA,
                                     "category_quota":CATEGORY_QUOTA})
    env["dpo_dir"] = dest
    execute_step(5,steps,env)
    if env["data_path"].resolve() != (dest/"pos_neg_pairs.json").resolve() or len(env["lines"]) != 256:
        raise RuntimeError("Notebook loaded incorrect dataset")
    execute_step(6,steps,env)

    model = env["gpt"]
    before = score_preference_and_end(env,pairs,args.batch_size)
    print("Initial preference wins:",before["pairwise_wins"],flush=True)
    print("Initial gold stop P(.):",f"{before['mean_gold_stop_probability']:.4f}",flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model.train()
    losses, pref_losses, number_losses, comp_losses, stop_losses = [], [], [], [], []
    for epoch in range(args.epochs):
        total = [0.0]*5
        for negative, positive in env["get_batches"](env["lines"],args.batch_size):
            env["optimizer"].zero_grad(set_to_none=True)
            ps, pcm, pn, pend, _ = termination_aware_scores(env,positive)
            ns, _, _, _, _ = termination_aware_scores(env,negative)
            preference = -F.logsigmoid((ps-ns)/env["beta"]).mean()
            number_nll = -pn.mean()
            completion_nll = -pcm.mean()
            stop_nll = -pend.mean()
            loss = (preference + args.answer_weight*number_nll
                    + args.completion_weight*completion_nll
                    + args.end_weight*stop_nll)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite loss at epoch " + str(epoch+1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            env["optimizer"].step()
            env["scheduler"].step()
            for i, value in enumerate((loss,preference,number_nll,completion_nll,stop_nll)):
                total[i] += float(value.item())
        for series, value in zip((losses,pref_losses,number_losses,comp_losses,stop_losses),total):
            series.append(value/env["steps_per_epoch"])
        if epoch == 0 or (epoch+1)%16 == 0 or epoch == args.epochs-1:
            print(f"Epoch {epoch+1:3d}/{args.epochs} total={losses[-1]:.5f} "
                  f"pref={pref_losses[-1]:.5f} number={number_losses[-1]:.5f} "
                  f"template={comp_losses[-1]:.5f} stop={stop_losses[-1]:.5f}",flush=True)

    checkpoint = dest/"dpo.pt"
    torch.save({"model_state_dict":model.state_dict(),"model_args":env["ckpt"]["model_args"]},checkpoint)
    model.eval()
    train_score, train_rows = run_evaluation(model,metadata,env["stoi"],env["itos"],args.device,args.max_new_tokens)
    val_score, val_rows = run_evaluation(model,validation,env["stoi"],env["itos"],args.device,args.max_new_tokens)
    after = score_preference_and_end(env,pairs,args.batch_size)
    train_gen = generation_diagnostics(train_rows)
    val_gen = generation_diagnostics(val_rows)
    save_json(dest/"train_predictions.json",train_rows)
    save_json(dest/"val_predictions.json",val_rows)
    result = {
        "stage":"overfit256_termination_aware",
        "checkpoint":str(checkpoint),
        "checkpoint_bytes":checkpoint.stat().st_size,
        "init_checkpoint":"sft/gpt.pt",
        "source_round1c_sha256":EXPECTED_SHA,
        "seed":args.seed,"training_size":256,"validation_size":256,
        "epochs":args.epochs,"batch_size":args.batch_size,"steps":args.epochs*env["steps_per_epoch"],
        "beta":env["beta"],"answer_weight":args.answer_weight,
        "completion_weight":args.completion_weight,"end_weight":args.end_weight,
        "loss_definition":"answer-focused preference + numeric NLL + completion NLL + gold end-period NLL, no reference model",
        "initial_training_preference_and_stop":before,
        "final_training_preference_and_stop":after,
        "epoch_total_losses":losses,"epoch_preference_losses":pref_losses,
        "epoch_number_losses":number_losses,"epoch_completion_losses":comp_losses,
        "epoch_stop_losses":stop_losses,
        "seen_train":train_score,"unseen_val":val_score,
        "seen_train_generation":train_gen,"unseen_val_generation":val_gen,
        "note":"Additional termination-loss ablation; teacher-provided target, not TA evaluation. Existing checkpoints unchanged.",
    }
    save_json(summary_path,result)
    print("=== OVERFIT256 TERMINATION-AWARE LOSS COMPLETE ===",flush=True)
    print(f"TRAIN accuracy: {train_score['correct']}/256 = {train_score['accuracy']:.2%}",flush=True)
    print(f"VAL accuracy: {val_score['correct']}/256 = {val_score['accuracy']:.2%}",flush=True)
    print(f"Natural period stops: {train_gen['period_stops']}/256 = {train_gen['period_stop_rate']:.2%}",flush=True)
    print(f"Gold answer prefix: {train_gen['gold_answer_prefix_count']}/256 = {train_gen['gold_answer_prefix_rate']:.2%}",flush=True)
    print(f"Gold end-period top-1 (teacher forced): {after['gold_stop_top1_rate']:.2%}",flush=True)
    print(f"Mean gold end-period P(.): {after['mean_gold_stop_probability']:.4f}",flush=True)
    print(f"Preference wins: {after['pairwise_wins']}/256 = {after['pairwise_win_rate']:.2%}",flush=True)
    print("Saved:",summary_path,flush=True)
    print("Checkpoint:",checkpoint,flush=True)


if __name__=="__main__":
    main()
