#!/usr/bin/env python3
"""Compare three isolated Overfit-256 termination-loss gamma values.

Pure JSON reading: runs on login node and does not load a model or need GPU.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("0.25", "results/overfit256_termination_gamma025_summary.json"),
    ("0.50", "results/overfit256_termination_gamma050_summary.json"),
    ("1.00", "results/overfit256_termination_aware_summary.json"),
]


def main():
    records = []
    for gamma, filename in CONFIGS:
        path = ROOT / filename
        if not path.exists():
            raise SystemExit(f"Missing summary for gamma={gamma}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert abs(float(data["end_weight"]) - float(gamma)) < 1e-9, path
        assert data["training_size"] == data["validation_size"] == 256, path
        assert data["epochs"] == 128 and data["batch_size"] == 64 and data["steps"] == 512, path
        assert data["init_checkpoint"] == "sft/gpt.pt", path
        records.append((gamma, data))

    for key in ("source_round1c_sha256", "beta", "answer_weight", "completion_weight", "seed"):
        first = records[0][1][key]
        assert all(d[key] == first for _, d in records), f"Experiment mismatch on {key}"

    head = ("gamma", "train", "val", "period-stop", "gold-prefix", "end P(.)",
            "end top-1", "pair wins")
    print(" | ".join(f"{x:>12}" for x in head))
    print("-" * 122)
    for gamma, d in records:
        t, v = d["seen_train"], d["unseen_val"]
        gen = d["seen_train_generation"]
        m = d["final_training_preference_and_stop"]
        vals = (
            gamma,
            f"{t['correct']}/256 ({t['accuracy']:.2%})",
            f"{v['correct']}/256 ({v['accuracy']:.2%})",
            f"{gen['period_stops']}/256",
            f"{gen['gold_answer_prefix_count']}/256",
            f"{m['mean_gold_stop_probability']:.4f}",
            f"{m['gold_stop_top1_rate']:.2%}",
            f"{m['pairwise_wins']}/256",
        )
        print(" | ".join(f"{str(x):>12}" for x in vals))
    print("\nSame 256 training pairs and held-out validation questions; all from sft/gpt.pt.")
    print("Period-stop rate alone is not accuracy; inspect early vs correct stopping.")


if __name__ == "__main__":
    main()
