#!/usr/bin/env python3
"""Evaluate a frozen NanoGPT-Math checkpoint on fixed integer/decimal questions.

Examples:
  python scripts/evaluate_math.py --test data/test_math.jsonl
  python scripts/evaluate_math.py --test data/test_math.jsonl --limit 20 --device cpu
  python scripts/evaluate_math.py --test data/test_math.jsonl --checkpoint dpo/dpo.pt

Expected JSONL/JSON record:
  {"prompt": "7/2=?", "answer": "3.5", "category": "division"}

The test set must be prepared and frozen independently. This program never
creates questions or modifies model parameters. Decimal answers use exact
Fraction arithmetic rather than floating-point equality. The decoder peeks
past a numeric '.' instead of terminating before a fractional digit.
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig  # noqa: E402
from math_eval_utils import greedy_completion, grade_completion, parse_expected  # noqa: E402


def path_from_root(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_test(path):
    if not path.is_file():
        raise FileNotFoundError("Missing test file: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Test file must contain a nonempty JSON array or JSONL rows")

    prompts = set()
    for idx, item in enumerate(records):
        if not isinstance(item, dict) or not {"prompt", "answer"} <= item.keys():
            raise ValueError(f"Row {idx} needs prompt and answer")
        if not isinstance(item["prompt"], str) or not item["prompt"]:
            raise ValueError(f"Row {idx}: prompt must be a nonempty string")
        if item["prompt"] in prompts:
            raise ValueError(f"Duplicate prompt in test file: {item['prompt']!r}")
        prompts.add(item["prompt"])
        try:
            parse_expected(item["answer"])
        except ValueError as exc:
            raise ValueError(f"Row {idx}: {exc}") from exc
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="Frozen JSONL/JSON test set")
    parser.add_argument("--checkpoint", default="sft/gpt.pt")
    parser.add_argument("--meta", default="sft/meta.pkl")
    parser.add_argument("--out", default="results/baseline_predictions.jsonl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, help="Optional first N rows for smoke")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu")

    records = load_test(path_from_root(args.test))
    if args.limit is not None:
        records = records[:args.limit]
    with path_from_root(args.meta).open("rb") as handle:
        meta = pickle.load(handle)  # Use only the trusted course-supplied meta.pkl.
    stoi, itos = meta["stoi"], meta["itos"]
    ckpt = torch.load(path_from_root(args.checkpoint), map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    original = ckpt.get("model", ckpt.get("model_state_dict"))
    if original is None:
        raise ValueError("Checkpoint missing model weights")
    clean_state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in original.items()
    }
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device).eval()

    by_category = defaultdict(lambda: defaultdict(int))
    rows = []
    for idx, item in enumerate(records, start=1):
        prompt = item["prompt"]
        missing = sorted(set(prompt) - set(stoi))
        if missing:
            raise ValueError(
                f"Prompt {idx} contains characters absent from tokenizer: {missing}"
            )
        if len(prompt) > model.config.block_size:
            raise ValueError(f"Prompt {idx} exceeds block_size")
        completion, stop = greedy_completion(
            model, [stoi[ch] for ch in prompt], itos, args.max_new_tokens, device
        )
        status, prediction = grade_completion(completion, item["answer"])
        category = item.get("category") or "uncategorized"
        row = {
            "id": item.get("id", idx),
            "category": category,
            "prompt": prompt,
            "answer": str(item["answer"]),
            "completion": completion,
            "prediction": prediction,
            "status": status,
            "stop_reason": stop,
        }
        rows.append(row)
        by_category["overall"][status] += 1
        by_category[category][status] += 1
        if idx <= 5:
            print(f"#{idx} {prompt!r} -> {completion!r} [{status}]")

    output_path = path_from_root(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nModel: {args.checkpoint} | Device: {device} | Tested: {len(rows)}")
    print(f"Test source: {path_from_root(args.test)}")
    print(f"Predictions: {output_path}")
    for category, counts in sorted(by_category.items()):
        total = sum(counts.values())
        print(
            f"{category:20s} N={total:5d} "
            f"accuracy={counts['correct']/total:.2%} "
            f"refusal={counts['refusal']/total:.2%} "
            f"math_error={counts['math_error']/total:.2%} "
            f"format_failure={counts['format_failure']/total:.2%}"
        )


if __name__ == "__main__":
    main()
