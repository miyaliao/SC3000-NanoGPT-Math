#!/usr/bin/env python3
"""Read-only generation-vs-teacher-forcing diagnosis for SC3000 NanoGPT-Math.

Compare the SAME 256 seen Round-1C questions across the course-loss and
answer-focused 256-question overfit checkpoints. Does not change any weights.

Checks:
  (1) natural greedy generation, original '.' / newline stopping
  (2) greedy generation AFTER supplying only ' The answer is ' (not digits)
  (3) teacher-forced true answer characters: argmax vs gold, rank, mean NLL

Run on an allocated GPU: python scripts/diagnose_generation_gap.py --device cuda
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from model import GPT, GPTConfig  # noqa: E402
from evaluate_ta_integer import evaluate_answer, greedy_original_stop  # noqa: E402
from overfit256_course import select_rows  # noqa: E402

PFX = " The answer is "
DEFAULTS = [
    "dpo/experiments/overfit256_course/dpo.pt",
    "dpo/experiments/overfit256_answer_focused/dpo.pt",
]


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    sd = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if sd is None:
        raise ValueError(f"Missing weights in {path}")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def gold_character_scores(model, row, stoi, device):
    """Compare greedy TOP-1 token with each gold sign/digit, using gold prefix.

    This isolates a teacher-forced next-character task; it is not free generation.
    """
    prompt, answer = row["prompt"], str(row["correct_answer"])
    if not answer.lstrip("-").isdigit():
        raise ValueError("Integer-only answer required")
    history = prompt + PFX
    sequence = history + answer  # score digits/sign only, not final '.'
    ids = torch.tensor([[stoi[c] for c in sequence]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(ids[:, :-1], full_seq=True)
    # logits at position j-1 predict character j in full sequence.
    positions = list(range(len(history), len(sequence)))
    rows = []
    for position in positions:
        logit = logits[0, position - 1]
        true_id = int(ids[0, position].item())
        predicted_id = int(logit.argmax().item())
        logprob = float(F.log_softmax(logit, dim=-1)[true_id].item())
        rank = 1 + int((logit > logit[true_id]).sum().item())
        rows.append({"character": sequence[position], "top1": true_id == predicted_id,
                     "rank": rank, "logprob": logprob})
    return {"all_digits_top1": all(r["top1"] for r in rows),
            "first_digit_top1": rows[0]["top1"],
            "token_top1": sum(r["top1"] for r in rows),
            "token_total": len(rows),
            "mean_logprob": sum(r["logprob"] for r in rows)/len(rows),
            "first_digit_rank": rows[0]["rank"]}


def probe_model(model, records, stoi, itos, device, max_new_tokens):
    normal_counts, forced_counts = Counter(), Counter()
    kinds = defaultdict(lambda: {"normal": Counter(), "forced": Counter()})
    rows = []
    scores = []
    for r in records:
        prompt, expected, kind = r["prompt"], str(r["correct_answer"]), r["category"]
        if any(c not in stoi for c in prompt + PFX + expected + "."):
            raise ValueError(f"Out-of-vocab math item: {prompt}")
        with torch.inference_mode():
            natural, normal_stop = greedy_original_stop(
                model, prompt, stoi, itos, max_new_tokens, device)
            forced, forced_stop = greedy_original_stop(
                model, prompt + PFX, stoi, itos, max_new_tokens, device)
        natural_status, _ = evaluate_answer(natural, expected)
        forced_status, _ = evaluate_answer(forced, expected)
        normal_counts[natural_status] += 1
        forced_counts[forced_status] += 1
        kinds[kind]["normal"][natural_status] += 1
        kinds[kind]["forced"][forced_status] += 1
        score = gold_character_scores(model, r, stoi, device)
        scores.append(score)
        rows.append({"prompt": prompt, "answer": expected, "category": kind,
                     "natural": natural, "natural_status": natural_status,
                     "natural_stop": normal_stop,
                     "forced_prefix": forced, "forced_status": forced_status,
                     "forced_stop": forced_stop, **score})
    n = len(rows)
    first_top1 = sum(s["first_digit_top1"] for s in scores)
    all_top1 = sum(s["all_digits_top1"] for s in scores)
    tokens_correct = sum(s["token_top1"] for s in scores)
    tokens_total = sum(s["token_total"] for s in scores)
    summary = {"n": n,
        "natural": {"accuracy":normal_counts['correct']/n,"counts":dict(normal_counts)},
        "fixed_answer_prefix": {"accuracy":forced_counts['correct']/n,"counts":dict(forced_counts)},
        "teacher_forced": {
           "first_sign_or_digit_top1_rate":first_top1/n,
           "all_answer_characters_top1_rate":all_top1/n,
           "gold_character_top1_rate":tokens_correct/tokens_total,
           "gold_mean_character_logprob":statistics.mean(s['mean_logprob'] for s in scores),
           "gold_first_character_median_rank":statistics.median(s['first_digit_rank'] for s in scores)},
        "by_category": {kind:{typ:dict(counts) for typ,counts in types.items()}
                         for kind,types in sorted(kinds.items())}}
    return summary, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--checkpoints', nargs='+', default=DEFAULTS)
    p.add_argument('--out', default='results/generation_gap_256.json')
    args = p.parse_args()
    if args.max_new_tokens < 1: p.error('Token budget must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available(): p.error('No CUDA device')
    out = Path(args.out)
    if not out.is_absolute(): out = ROOT / out
    if not out.resolve().is_relative_to((ROOT / 'results').resolve()):
        p.error('Only output within results/ is supported')
    if out.exists(): p.error(f'Refusing to overwrite prior diagnosis: {out}')

    _, metadata, _, _ = select_rows(args.seed)
    with (ROOT / 'sft/meta.pkl').open('rb') as handle:
        vocab = pickle.load(handle)  # trusted course file
    stoi, itos = vocab['stoi'], vocab['itos']
    result = {"sample_seed": args.seed, "n": len(metadata),
              "experiment":"read-only, original-stop deterministic greedy vs forced-prefix vs gold character teacher forcing",
              "models":{}}
    for path in args.checkpoints:
        ckpt = Path(path)
        if not ckpt.is_absolute(): ckpt = ROOT / ckpt
        if not ckpt.is_file(): raise FileNotFoundError(f'Missing checkpoint: {ckpt}')
        model = load_model(ckpt, args.device)
        summary, rows = probe_model(model, metadata, stoi, itos, args.device, args.max_new_tokens)
        result['models'][str(path)] = {"summary":summary, "examples":rows}
        print(f"=== {path} ===", flush=True)
        print(f"Natural: {summary['natural']['counts'].get('correct',0)}/{len(rows)} = {summary['natural']['accuracy']:.2%}", flush=True)
        print(f"Forced prefix: {summary['fixed_answer_prefix']['counts'].get('correct',0)}/{len(rows)} = {summary['fixed_answer_prefix']['accuracy']:.2%}", flush=True)
        print(f"Gold first character top-1: {summary['teacher_forced']['first_sign_or_digit_top1_rate']:.2%}", flush=True)
        print(f"Gold ALL characters top-1: {summary['teacher_forced']['all_answer_characters_top1_rate']:.2%}", flush=True)
        print(f"Gold digit token top-1: {summary['teacher_forced']['gold_character_top1_rate']:.2%}", flush=True)
        print(f"Gold character logprob (mean): {summary['teacher_forced']['gold_mean_character_logprob']:.4f}", flush=True)
        print('Example first five:', flush=True)
        for r in rows[:5]:
            print(f"  {r['prompt']} expected={r['answer']}, natural={r['natural']!r}, fixed-prefix={r['forced_prefix']!r}", flush=True)
        del model
        if args.device == 'cuda': torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print('=== GENERATION GAP DIAGNOSIS COMPLETE ===',flush=True)
    print('Output:',out,flush=True)

if __name__ == '__main__':
    main()
