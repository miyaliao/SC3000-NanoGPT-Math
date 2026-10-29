#!/usr/bin/env python3
"""Execute the existing dpo.ipynb training cells (Steps 2-7) for Round 1.

Do not introduce a second training implementation: this runner executes the
Notebook cells directly, just like the successful pipeline smoke test.

Run only on allocated TC1 GPU:
    python scripts/train_round1.py --device cuda

The script will NOT overwrite an existing dpo/dpo.pt unless --overwrite is
specified. It also requires the installed training data to match the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_smoke import execute_step, notebook_steps  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--overwrite", action="store_true", help="Explicitly overwrite prior dpo/dpo.pt")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive")
    model_file = ROOT / "dpo" / "dpo.pt"
    if model_file.exists() and not args.overwrite:
        p.error(f"Refusing to replace existing {model_file}. Remove/backup it or use --overwrite.")
    train_file = ROOT / "dpo" / "pos_neg_pairs.json"
    manifest_file = ROOT / "dpo" / "data" / "round1_full" / "round1_manifest.json"
    if not train_file.is_file() or not manifest_file.is_file():
        p.error("Run python scripts/generate_round1.py --profile full --install-train first")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    current_sha = hashlib.sha256(train_file.read_bytes()).hexdigest()
    if current_sha != manifest["file_sha256"]["train"]:
        p.error("Training data differs from generated manifest; refusing to train")
    if manifest["data"]["train"]["total"] != 20480:
        p.error("Unexpected train size")

    os.chdir(ROOT)
    env = {"__name__": "__round1__"}
    steps = notebook_steps()
    print("=== Step 2: load NanoGPT setup ===", flush=True)
    execute_step(2, steps, env)
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("No CUDA GPU. Submit through Slurm, not TC1 login node.")
    env["device"] = args.device
    env["epochs"] = args.epochs
    env["batch_size"] = args.batch_size
    print("=== Steps 3-4: use original Notebook helpers and pretrained model ===", flush=True)
    execute_step(3, steps, env)
    execute_step(4, steps, env)
    print("=== Steps 5-7: load the full data and run course preference loss ===", flush=True)
    execute_step(5, steps, env)
    if len(env["lines"]) != 20480:
        raise RuntimeError("Incorrect number of training pairs loaded")
    execute_step(6, steps, env)
    execute_step(7, steps, env)
    if not model_file.is_file():
        raise RuntimeError("Final checkpoint missing")
    import math
    if not all(math.isfinite(v) for v in env["training_loss"]):
        raise RuntimeError("Nonfinite epoch loss")
    summary = {"stage": "round1-refusal-v1", "epochs": args.epochs,
               "batch_size": env["batch_size"], "steps_per_epoch": env["steps_per_epoch"],
               "total_steps": args.epochs * env["steps_per_epoch"],
               "train_size": len(env["lines"]), "epoch_mean_losses": env["training_loss"],
               "training_data_sha256": current_sha,
               "checkpoint": str(model_file), "checkpoint_bytes": model_file.stat().st_size,
               "note": "Runs exact course Notebook training cells 2-7; not a baseline/accuracy score"}
    dest = ROOT / "results" / "round1_training_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== ROUND 1 TRAINING COMPLETED ===")
    print("Summary:", dest)
    print("Checkpoint:", model_file)


if __name__ == "__main__":
    main()
