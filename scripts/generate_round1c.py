#!/usr/bin/env python3
"""Round 1C: keep Round 1B integer prompts/positives, replace refusals with wrong answers.

All negatives have the same 'The answer is ... .' format, sign and string
length as the correct response. All mathematics is validated independently.
The original Round 1A and Round 1B training data/checkpoints are untouched.

From the repository root:
  python scripts/generate_round1c.py
  python scripts/generate_round1c.py --verify-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import generate_round1 as base  # noqa: E402

SEED = 31042
SOURCE = ROOT / "dpo/experiments/round1b_integer"
DEST = ROOT / "dpo/experiments/round1c_incorrect"
EXPECTED_SOURCE_SHA = "0c641927173c7d0e9d221050ded48d273eac9b25da2752b1c6f0569015512164"
PREFIX = " The answer is "


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p: Path, value) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wrong_number(original: str, rng: random.Random) -> tuple[str, str]:
    """Generate a wrong integer with exactly the same sign and character length."""
    assert original and original.lstrip("-").isdigit(), original
    negative = original.startswith("-")
    digits = original[1:] if negative else original
    assert digits == "0" or digits[0] != "0", original
    assert not (negative and digits == "0")

    strategy = rng.choices(["digit_change", "off_by_one", "place_error"], [6, 3, 1])[0]
    n = int(digits)
    wrong: int | None = None
    if strategy != "digit_change":
        offset = 1 if strategy == "off_by_one" else 10 ** rng.randrange(1, len(digits)) if len(digits) > 1 else 1
        for candidate in rng.sample([n - offset, n + offset], 2):
            if candidate >= (1 if negative else 0) and len(str(candidate)) == len(digits) and candidate != n:
                wrong = candidate
                break
    if wrong is None:
        strategy = "digit_change"  # stable fallback if near-miss changes number of digits/sign
        idx = rng.randrange(len(digits))
        options = [str(d) for d in range(10)
                   if str(d) != digits[idx] and (idx != 0 or len(digits) == 1 or d != 0)
                   and (not negative or len(digits) != 1 or d != 0)]
        if not options:
            raise RuntimeError(f"No legal digit substitution for {original}")
        replacement = rng.choice(options)
        candidate = digits[:idx] + replacement + digits[idx + 1:]
        wrong = int(candidate)
    result = ("-" if negative else "") + str(wrong)
    assert result != original and len(result) == len(original)
    assert result.startswith("-") == negative
    return result, strategy


def source_files():
    manifest_file = SOURCE / "round1b_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    paths = {
        "train": SOURCE / "pos_neg_pairs.json",
        "metadata": SOURCE / "round1b_train_metadata.json",
        "validation": SOURCE / "common_integer_val_256.json",
    }
    for key, f in paths.items():
        assert f.exists(), f"Missing Round 1B source: {f}"
        mkey = "val" if key == "validation" else key
        if sha256(f) != manifest["file_sha256"][mkey]:
            raise RuntimeError(f"Round 1B {key} changed since generation")
    if sha256(paths["train"]) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Round 1B training hash differs from previously audited source")
    return paths, manifest


def verify_records(pairs, meta, validation):
    if len(pairs) != len(meta) or len(pairs) != 20480:
        raise AssertionError("Wrong Round 1C size")
    if len(validation) != 256:
        raise AssertionError("Expected frozen common integer validation N=256")
    prompts, facts, kinds = set(), set(), Counter()
    max_len = 0
    with (ROOT / "sft/meta.pkl").open("rb") as f:
        stoi = pickle.load(f)["stoi"]
    for pair, r in zip(pairs, meta):
        p, answer, wrong = r["prompt"], r["correct_answer"], r["wrong_answer"]
        assert p not in prompts
        prompts.add(p)
        fact = base.parse_relation(p, answer)  # validates exact printed question and correct value
        facts.add(fact)
        assert p.endswith("=?"), p
        assert pair["positive"] == f"{p}{PREFIX}{answer}."
        assert pair["negative"] == f"{p}{PREFIX}{wrong}."
        assert len(pair["positive"]) == len(pair["negative"])  # avoids whole-sequence-length shortcut
        assert answer != wrong and answer.lstrip("-").isdigit() and wrong.lstrip("-").isdigit()
        assert answer.startswith("-") == wrong.startswith("-")
        assert len(answer) == len(wrong)
        try:
            base.parse_relation(p, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Incorrect answer is unexpectedly mathematically valid: {p} -> {wrong}")
        for v in pair.values():
            assert not (set(v) - set(stoi)), f"Tokenizer OOV in {v!r}"
            assert len(v) + 4 <= 256
            max_len = max(max_len, len(v))
        kinds[r["category"]] += 1
    for r in validation:
        assert r["prompt"] not in prompts
        key = base.parse_relation(r["prompt"], str(r["answer"]))
        assert key not in facts, "Validation math-fact overlaps training"
        assert "." not in r["prompt"] + str(r["answer"])
    return {"max_pair_chars": max_len, "types": dict(sorted(kinds.items())),
            "unique_prompts": len(prompts), "unique_facts": len(facts)}


def create():
    paths, source_manifest = source_files()
    source_pairs = json.loads(paths["train"].read_text(encoding="utf-8"))
    source_meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    frozen_val = json.loads(paths["validation"].read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    pairs, meta = [], []
    for old, r in zip(source_pairs, source_meta):
        prompt, answer = r["prompt"], str(r["answer"])
        assert old["positive"] == f"{prompt}{PREFIX}{answer}."
        wrong, strategy = wrong_number(answer, rng)
        pairs.append({"positive": old["positive"], "negative": f"{prompt}{PREFIX}{wrong}."})
        meta.append({"prompt": prompt, "correct_answer": answer, "wrong_answer": wrong,
                     "category": r["category"], "strategy": strategy,
                     "fact_key": r["fact_key"]})
    summary = verify_records(pairs, meta, frozen_val)
    train_path = DEST / "pos_neg_pairs.json"
    meta_path = DEST / "round1c_train_metadata.json"
    val_path = DEST / "common_integer_val_256.json"
    write_json(train_path, pairs)
    write_json(meta_path, meta)
    write_json(val_path, frozen_val)
    manifest = {
        "stage": "round1c-incorrect-negative", "seed": SEED,
        "training_size": len(pairs), "validation_size": len(frozen_val),
        "init_checkpoint": "sft/gpt.pt",
        "source_round1b_train_sha256": sha256(paths["train"]),
        "source_round1b_validation_sha256": sha256(paths["validation"]),
        "source_round1b_manifest_sha256": sha256(SOURCE / "round1b_manifest.json"),
        "templates": {"positive": "{prompt} The answer is {correct}.",
                      "negative": "{prompt} The answer is {wrong}."},
        "rules": "Wrong integer; same sign and string length; exact math verifier; fixed seed; source prompts and positives unchanged.",
        "summary": summary,
        "strategies": dict(Counter(r["strategy"] for r in meta)),
        "file_sha256": {"train": sha256(train_path), "metadata": sha256(meta_path), "val": sha256(val_path)},
        "note": "Round 1C differs from Round 1B only in the negative completion. A negative that is a plausible wrong number is synthetic, not model-mined.",
    }
    write_json(DEST / "round1c_manifest.json", manifest)
    print("=== ROUND 1C DATASET PASSED ===")
    print("Training pairs:", len(pairs), "| Frozen common validation:", len(frozen_val))
    print("Same positive/prompts as Round 1B:", len(pairs))
    print("Wrong-answer strategies:", manifest["strategies"])
    print("Max pair length:", summary["max_pair_chars"])
    print("Train SHA256:", manifest["file_sha256"]["train"])
    print("Output:", DEST)


def verify_only():
    paths, _ = source_files()
    manifest = json.loads((DEST / "round1c_manifest.json").read_text(encoding="utf-8"))
    assert sha256(paths["train"]) == manifest["source_round1b_train_sha256"]
    assert sha256(paths["validation"]) == manifest["source_round1b_validation_sha256"]
    outputs = {"train": DEST / "pos_neg_pairs.json",
               "metadata": DEST / "round1c_train_metadata.json",
               "val": DEST / "common_integer_val_256.json"}
    for kind, path in outputs.items():
        assert sha256(path) == manifest["file_sha256"][kind], f"Hash mismatch: {kind}"
    meta = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    pairs = json.loads(outputs["train"].read_text(encoding="utf-8"))
    val = json.loads(outputs["val"].read_text(encoding="utf-8"))
    summary = verify_records(pairs, meta, val)
    assert summary == manifest["summary"]
    original = json.loads(paths["train"].read_text(encoding="utf-8"))
    assert all(a["positive"] == b["positive"] for a,b in zip(pairs,original))
    print("PASS: hashes, math correctness, integer-only, equal lengths, identical positives, frozen validation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only:
        verify_only()
    else:
        create()


if __name__ == "__main__":
    main()
