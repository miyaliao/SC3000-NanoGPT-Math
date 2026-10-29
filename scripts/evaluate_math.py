#!/usr/bin/env python3
"""Evaluate a frozen NanoGPT-Math checkpoint on a fixed, integer-answer test set.

Examples (run from any working directory):
  python scripts/evaluate_math.py --test data/test_math.jsonl
  python scripts/evaluate_math.py --test data/test_math.jsonl --limit 20 --device cpu

Expected JSONL row: {"prompt": "79-7=?", "answer": "72", "category": "subtraction"}
A JSON array containing the same dictionaries also works.

The test file must be prepared/frozen independently. This program never generates
questions and does not train or change model parameters.
"""

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig  # noqa: E402


ANSWER_PATTERN = re.compile(
    r"^\s*(?:The answer is\s+)?([+-]?\d+)(?=\s*(?:[.!]|because\b|$))",
    flags=re.IGNORECASE,
)
REFUSAL_PATTERN = re.compile(
    r"\b(?:sor+y|unknown)\b|\b(?:do not know|don't know|cannot answer|can't answer)\b",
    flags=re.IGNORECASE,
)


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
        if not re.fullmatch(r"[+-]?\d+", str(item["answer"]).strip()):
            raise ValueError(
                f"Row {idx}: expected integer answer, got {item['answer']!r}"
            )
    return records


def generate(model, prompt_ids, itos, max_new_tokens, device):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids = []
    stop = "max_new_tokens"
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.block_size :]
            logits, _ = model(context)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == 0:
                stop = "newline"
                break
            new_ids.append(next_id)
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id == 7:
                stop = "period"
                break
    return "".join(itos[i] for i in new_ids), stop


def assess(completion, expected):
    """Grade a generated completion, not tokens already supplied in its prompt."""
    match = ANSWER_PATTERN.match(completion)
    if match:
        predicted = int(match.group(1))
        return ("correct" if predicted == expected else "math_error"), predicted
    if REFUSAL_PATTERN.search(completion):
        return "refusal", None
    return "format_failure", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="Fixed JSONL/JSON test set")
    parser.add_argument("--checkpoint", default="sft/gpt.pt")
    parser.add_argument("--meta", default="sft/meta.pkl")
    parser.add_argument("--out", default="results/baseline_predictions.jsonl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, help="Optional smoke subset")
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

    test_path = path_from_root(args.test)
    records = load_test(test_path)
    if args.limit is not None:
        records = records[:args.limit]

    with path_from_root(args.meta).open("rb") as handle:
        meta = pickle.load(handle)  # Trusted course-supplied meta.pkl only.
    stoi, itos = meta["stoi"], meta["itos"]
    ckpt = torch.load(
        path_from_root(args.checkpoint), map_location="cpu", weights_only=True
    )
    model = GPT(GPTConfig(**ckpt["model_args"]))
    original = ckpt.get("model", ckpt.get("model_state_dict"))
    if original is None:
        raise ValueError("Checkpoint missing model weights")
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in original.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    by_category = defaultdict(lambda: defaultdict(int))
    all_rows = []
    for idx, item in enumerate(records, start=1):
        prompt = item["prompt"]
        missing = sorted(set(prompt) - set(stoi))
        if missing:
            raise ValueError(
                f"Prompt {idx} contains characters absent from tokenizer: {missing}. "
                "Adapt the external dataset before using it."
            )
        if len(prompt) > model.config.block_size:
            raise ValueError(f"Prompt {idx} exceeds block_size")
        completion, stop = generate(
            model, [stoi[char] for char in prompt], itos, args.max_new_tokens, device
        )
        expected = int(str(item["answer"]).strip())
        status, predicted = assess(completion, expected)
        category = item.get("category") or "uncategorized"
        row = {
            "id": item.get("id", idx),
            "category": category,
            "prompt": prompt,
            "answer": expected,
            "completion": completion,
            "prediction": predicted,
            "status": status,
            "stop_reason": stop,
        }
        all_rows.append(row)
        by_category["overall"][status] += 1
        by_category[category][status] += 1
        if idx <= 5:
            print(f"#{idx} {prompt!r} -> {completion!r} [{status}]")

    output_path = path_from_root(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nModel: {args.checkpoint} | Device: {device} | Tested: {len(all_rows)}")
    print(f"Test source: {test_path}")
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
