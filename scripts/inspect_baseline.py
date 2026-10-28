#!/usr/bin/env python3
"""Read-only inspection of the instructor-provided NanoGPT checkpoint.

Usage (from repository root):
    python scripts/inspect_baseline.py
    python scripts/inspect_baseline.py --smoke --device cuda
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def greedy_completion(model, stoi, itos, prompt, max_new_tokens, device):
    token_ids = [stoi[ch] for ch in prompt]
    input_len = len(token_ids)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    stop_reason = "length_limit"

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.block_size :]
            logits, _ = model(context)
            next_id = int(logits[0, -1].argmax().item())
            if next_id == 0:  # newline/end marker: original generate() skips it
                stop_reason = "token_0"
                break
            next_tensor = torch.tensor([[next_id]], device=device)
            ids = torch.cat((ids, next_tensor), dim=1)
            if next_id == 7:  # period/end marker: original generate() retains it
                stop_reason = "token_7"
                break

    generated_ids = ids[0].tolist()[input_len:]
    completion = "".join(itos[token] for token in generated_ids)
    return completion, stop_reason


def main():
    parser = argparse.ArgumentParser(description="Inspect NanoGPT-Math baseline safely")
    parser.add_argument("--checkpoint", default="sft/gpt.pt")
    parser.add_argument("--meta", default="sft/meta.pkl")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run one deterministic inference example (not an accuracy test)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    checkpoint_path = resolve(args.checkpoint)
    meta_path = resolve(args.meta)
    for path in (checkpoint_path, meta_path):
        if not path.is_file():
            parser.error("Required local file not found: " + str(path))

    # Only load checkpoint tensors and safe Python primitives.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        parser.error("Expected a checkpoint dictionary")
    if "model_args" not in checkpoint:
        parser.error("Checkpoint has no 'model_args'")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state_dict is None:
        parser.error("Checkpoint has neither 'model' nor 'model_state_dict'")

    with meta_path.open("rb") as f:
        meta = pickle.load(f)  # Only load meta.pkl from the trusted course repository.
    stoi, itos = meta["stoi"], meta["itos"]

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    clean_state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    non_pos_params = model.get_num_params(non_embedding=True)

    print("=== Checkpoint ===")
    print("File:", checkpoint_path)
    print("Size (MiB):", round(checkpoint_path.stat().st_size / (1024**2), 2))
    print("Keys:", sorted(checkpoint.keys()))
    print("State dict: loaded with strict=True")
    print("Model parameters (total):", f"{total_params:,}")
    print("Model parameters (excluding position embeddings):", f"{non_pos_params:,}")
    print("=== GPTConfig ===")
    for key in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size", "dropout", "bias"):
        print(key + ":", getattr(config, key))
    print("=== Character tokenizer ===")
    print("meta vocab_size:", meta.get("vocab_size"))
    print("stoi entries:", len(stoi), "| itos entries:", len(itos))
    print("Model vocab matches tokenizer:", config.vocab_size == len(itos))
    print("ID 0 character:", repr(itos.get(0)))
    print("ID 7 character:", repr(itos.get(7)))
    print("Characters:", repr("".join(itos[i] for i in sorted(itos))))

    prompts = ["79-7=?", "74+8=?", "x+55=95,x=?", "1*x=6,x=?"]
    print("=== Prompt encoding check ===")
    for prompt in prompts:
        missing = sorted(set(prompt) - set(stoi))
        print(repr(prompt), "=>", "OK" if not missing else "MISSING " + repr(missing))

    if args.smoke:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable. Run without --smoke on login node, or use a GPU job.")
        prompt = prompts[0]
        if set(prompt) - set(stoi):
            parser.error("Smoke prompt contains out-of-vocabulary characters")
        model.to(args.device)
        completion, stop_reason = greedy_completion(
            model, stoi, itos, prompt, args.max_new_tokens, args.device
        )
        print("=== Smoke generation (greedy, 1 question) ===")
        print("Device:", args.device)
        print("Prompt:", repr(prompt))
        print("Completion:", repr(completion))
        print("Stop reason:", stop_reason)
        print("NOTE: This is only a format check, NOT a baseline accuracy score.")


if __name__ == "__main__":
    main()
