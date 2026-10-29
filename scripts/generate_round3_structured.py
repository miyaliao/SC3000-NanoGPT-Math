#!/usr/bin/env python3
"""Round 3: structured, integer-only, one-step-explanation SC3000 preference data.

Core design: dense 0..100 arithmetic facts and related equation templates,
exact arithmetic verification, stable *fact-group* isolation, target 100,000
training rows (deliberate repeat exposure; report actual unique prompts).
Refusal negatives and the instructor's original course loss are kept fixed.

No checkpoints or notebook files are changed. Never automatically install
the dataset into dpo/pos_neg_pairs.json.

Usage:
  python scripts/generate_round3_structured.py --profile mini
  python scripts/generate_round3_structured.py --profile full
  python scripts/generate_round3_structured.py --profile full --verify-only
  python scripts/generate_round3_structured.py --profile full --write-control
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_round1 import RESERVED, parse_relation, stable_split  # noqa: E402

SEED = 42
REFUSAL = "Sorry, I don't know."
KINDS = (
    "add", "subtract", "multiply", "divide",
    "x_plus_a", "a_plus_x", "x_minus_a", "a_minus_x",
    "x_times_a", "a_times_x", "x_over_a", "a_over_x",
)
TRAIN_QUOTA = {
    "add": 14000, "subtract": 10000, "multiply": 11000, "divide": 9000,
    "x_plus_a": 7000, "a_plus_x": 7000, "x_minus_a": 7000, "a_minus_x": 7000,
    "x_times_a": 7000, "a_times_x": 7000, "x_over_a": 7000, "a_over_x": 7000,
}
assert sum(TRAIN_QUOTA.values()) == 100000
DEFAULT_OUT = ROOT / "dpo/data/round3_structured_full"


def make_record(prompt: str, answer: int, explanation: tuple[int, str, int],
                category: str) -> dict:
    """Construct a correctly explained mathematical statement."""
    a, op, b = explanation
    expected = {"+": a + b, "-": a - b, "*": a * b}
    if op == "/":
        if b == 0 or a % b:
            raise ValueError(f"Non-exact explanation division: {a}/{b}")
        value = a // b
    else:
        value = expected[op]
    if value != answer:
        raise ValueError(f"Incorrect reasoning: {prompt}, explanation={explanation}, answer={answer}")
    fact_key = parse_relation(prompt, str(answer))
    explanation_text = f"{a}{op}{b} equals {answer}"
    pos = f"{prompt} The answer is {answer} because {explanation_text}."
    control = f"{prompt} The answer is {answer}."
    neg = f"{prompt} {REFUSAL}"
    if pos.count(".") != 1 or control.count(".") != 1 or neg.count(".") != 1:
        raise ValueError("Unexpected period in integer-only examples")
    if "?" not in prompt or not prompt.endswith("=?"):
        raise ValueError("Bad problem form")
    return {
        "prompt": prompt, "answer": str(answer), "category": category,
        "fact_key": fact_key,
        "positive": pos, "control_positive": control, "negative": neg,
    }


def sum_relations(a: int, b: int):
    """One a+b=c fact, six question families in both orientations."""
    c = a + b
    yield make_record(f"{a}+{b}=?", c, (a, "+", b), "add")
    yield make_record(f"{c}-{a}=?", b, (c, "-", a), "subtract")
    yield make_record(f"x+{a}={c},x=?", b, (c, "-", a), "x_plus_a")
    yield make_record(f"{a}+x={c},x=?", b, (c, "-", a), "a_plus_x")
    yield make_record(f"x-{a}={b},x=?", c, (b, "+", a), "x_minus_a")
    yield make_record(f"{c}-x={a},x=?", b, (c, "-", a), "a_minus_x")


def product_relations(a: int, b: int):
    """One a*b=c fact. Never create equations with nonunique solutions."""
    c = a * b
    yield make_record(f"{a}*{b}=?", c, (a, "*", b), "multiply")
    if a > 0:
        yield make_record(f"{c}/{a}=?", b, (c, "/", a), "divide")
        yield make_record(f"x*{a}={c},x=?", b, (c, "/", a), "x_times_a")
        yield make_record(f"{a}*x={c},x=?", b, (c, "/", a), "a_times_x")
        yield make_record(f"x/{a}={b},x=?", c, (b, "*", a), "x_over_a")
        # x is the denominator, must be nonzero.
        if b > 0:
            yield make_record(f"{c}/x={a},x=?", b, (c, "/", a), "a_over_x")


def curriculum_tag(record: dict) -> str:
    q = record["prompt"]
    ans = int(record["answer"])
    key = record["fact_key"]
    a, b = int(key[1]), int(key[2])
    if a in (0, 1) or b in (0, 1):
        return "identity_0_1"
    if a <= 10 and b <= 10:
        return "basic_facts"
    if a % 10 == 0 or b % 10 == 0:
        return "tens_and_round_numbers"
    if key[0] == "add" and (a % 10 + b % 10 >= 10):
        return "carry_borrow"
    if key[0] == "mul" and a <= 12 and b <= 12:
        return "small_times_table"
    if ans >= 100:
        return "larger_results"
    return "standard"


TAG_WEIGHT = {
    "identity_0_1": 4.0,
    "basic_facts": 3.0,
    "tens_and_round_numbers": 2.5,
    "carry_borrow": 2.0,
    "small_times_table": 3.0,
    "larger_results": 1.0,
    "standard": 1.0,
}


def existing_protected_keys():
    """Also protect the previous common val-256 when available on TC1."""
    keys = {parse_relation(p, ans) for p, ans in RESERVED.items()
            if "." not in p + ans}  # integer portion of course demos
    old = ROOT / "dpo/experiments/round1c_incorrect/common_integer_val_256.json"
    if old.is_file():
        for r in json.loads(old.read_text(encoding="utf-8")):
            keys.add(parse_relation(r["prompt"], str(r["answer"])))
    return keys


def build_pools(seed: int):
    held_out = existing_protected_keys()
    pools = {split: {kind: {} for kind in KINDS}
             for split in ("train", "val", "test")}
    candidates = Counter()
    for op in ("add", "mul"):
        if op == "add":
            # 0..100 densely, all unordered sum facts, then both directions.
            fact_pairs = ((a, b) for a in range(101) for b in range(a, 101))
            builder = sum_relations
        else:
            # Strictly limited products; zero identities include only
            # valid, uniquely solved equations.
            fact_pairs = ((a, b) for a in range(101) for b in range(max(1, a), 101)
                          if a * b <= 400 and (a > 0 or b > 0))
            builder = product_relations
        for a, b in fact_pairs:
            for left, right in ((a, b), (b, a)) if a != b else ((a, b),):
                for row in builder(left, right):
                    candidates["generated"] += 1
                    key = row["fact_key"]
                    if key in held_out:
                        continue
                    split = stable_split(key, seed)
                    # Deduplicate within each split and category, while allowing
                    # repeated exposure in the final 100k training rows.
                    pools[split][row["category"]].setdefault(row["prompt"], row)
    for split, categories in pools.items():
        for category, items in categories.items():
            if not items:
                raise RuntimeError(f"No examples in pool {split}/{category}")
    return pools, candidates, len(held_out)


def sample_rows(pools: dict, profile: str, seed: int):
    rng = random.Random(seed + 3000)
    if profile == "mini":
        train_quota = {k: 12 for k in KINDS}
        heldout_quota = {k: 4 for k in KINDS}
    else:
        train_quota = TRAIN_QUOTA
        heldout_quota = {k: 64 for k in KINDS}

    result = {}
    for split in ("train", "val", "test"):
        out = []
        for kind in KINDS:
            pool = list(pools[split][kind].values())
            rng.shuffle(pool)
            count = train_quota[kind] if split == "train" else heldout_quota[kind]
            if split == "train":
                # Unique items first, then weighted repeat exposure. Do not
                # misreport 100k rows as 100k unique math problems.
                if len(pool) >= count:
                    chosen = rng.sample(pool, count)
                else:
                    weights = [TAG_WEIGHT[curriculum_tag(row)] for row in pool]
                    chosen = pool + rng.choices(pool, weights=weights, k=count - len(pool))
                assert len(chosen) == count
            else:
                if count > len(pool):
                    raise ValueError(f"Only {len(pool)} unique {split} records for {kind}; requested {count}")
                chosen = rng.sample(pool, count)
            out.extend(chosen)
        rng.shuffle(out)
        result[split] = out
    return result


def verify_records(records: dict, seed: int, vocab: set[str] | None):
    # Called before writing ANY outputs.
    manifest_splits = {}
    seen_fact_keys = {split: set() for split in records}
    seen_prompts = {split: set() for split in records}
    for split, rows in records.items():
        max_len = 0
        cats, tags = Counter(), Counter()
        for row in rows:
            p, ans, pos, neg = row["prompt"], row["answer"], row["positive"], row["negative"]
            key = parse_relation(p, ans)
            if key != row["fact_key"] or stable_split(key, seed) != split:
                raise RuntimeError(f"Fact key/split mismatch: {p}")
            # Independently re-check the explained arithmetic expression
            # encoded in the Positive string, rather than trusting a metadata field.
            stem = f"{p} The answer is {ans} because "
            if not pos.startswith(stem) or not pos.endswith(f" equals {ans}."):
                raise RuntimeError("Bad Positive format: " + pos)
            expression = pos[len(stem):-(len(f" equals {ans}."))]
            import re
            m = re.fullmatch(r"(-?\d+)([+\-*/])(-?\d+)", expression)
            if not m:
                raise RuntimeError("Bad mathematical explanation: " + expression)
            a, op, b = int(m[1]), m[2], int(m[3])
            if op == "+":
                actual = a + b
            elif op == "-":
                actual = a - b
            elif op == "*":
                actual = a * b
            else:
                if b == 0 or a % b:
                    raise RuntimeError("Invalid division explanation")
                actual = a // b
            if actual != int(ans):
                raise RuntimeError(f"Incorrect explanation: {pos}")
            if neg != f"{p} {REFUSAL}" or row["control_positive"] != f"{p} The answer is {ans}.":
                raise RuntimeError("Pair template inconsistency")
            if vocab is not None:
                unknown = set(pos + neg + row["control_positive"]) - vocab
                if unknown:
                    raise RuntimeError(f"Tokenizer OOV: {sorted(unknown)} from {p}")
            longest = max(len(pos), len(neg), len(row["control_positive"])) + 4
            if longest > 256:
                raise RuntimeError(f"Context >256: {p}")
            max_len = max(max_len, longest)
            seen_prompts[split].add(p)
            seen_fact_keys[split].add(key)
            cats[row["category"]] += 1
            tags[curriculum_tag(row)] += 1
        manifest_splits[split] = {
            "rows": len(rows), "unique_prompts": len(seen_prompts[split]),
            "unique_math_facts": len(seen_fact_keys[split]),
            "by_category": dict(cats), "by_curriculum_tag": dict(tags),
            "max_required_sequence_length": max_len,
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if seen_fact_keys[a] & seen_fact_keys[b] or seen_prompts[a] & seen_prompts[b]:
            raise RuntimeError(f"Train/validation/test math-fact leakage: {a} and {b}")
    return manifest_splits


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_new(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(value)
    with path.open("xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=("mini", "full"), default="mini")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--write-control", action="store_true",
                    help="Also write matching answer-only preferences for later ablation")
    ap.add_argument("--verify-only", action="store_true",
                    help="Build and validate in memory without writing outputs")
    args = ap.parse_args()
    if args.seed != SEED:
        ap.error("For controlled comparison keep seed=42")
    out = (args.out_dir if args.out_dir else
           ROOT / f"dpo/data/round3_structured_{args.profile}").resolve()
    if out.exists() and not args.verify_only:
        ap.error(f"Refusing to overwrite existing output: {out}")

    vocab_path = ROOT / "sft/meta.pkl"
    vocab = None
    if vocab_path.exists():
        with vocab_path.open("rb") as f:
            vocab = set(pickle.load(f)["stoi"])
    pools, attempted, protected_count = build_pools(args.seed)
    rows = sample_rows(pools, args.profile, args.seed)
    info = verify_records(rows, args.seed, vocab)
    print("=== ROUND 3 STRUCTURED MATH DATA VERIFIED ===")
    for split in ("train", "val", "test"):
        v = info[split]
        print(f"{split:5} rows={v['rows']:6} unique_prompts={v['unique_prompts']:6} "
              f"unique_facts={v['unique_math_facts']:5} "
              f"max_sequence={v['max_required_sequence_length']}")
    print(f"protected_existing_math_facts={protected_count}")
    if args.verify_only:
        print("PASS: verify-only; no data files written")
        return
    hashes = {}
    hashes["train"] = write_new(
        out / "pos_neg_pairs.json",
        [{"positive": r["positive"], "negative": r["negative"]} for r in rows["train"]])
    if args.write_control:
        hashes["answer_only_control"] = write_new(
            out / "pos_neg_pairs_answer_only.json",
            [{"positive": r["control_positive"], "negative": r["negative"]}
             for r in rows["train"]])
    hashes["train_metadata"] = write_new(out / "train_metadata.json",
        [{"prompt": r["prompt"], "answer": r["answer"], "category": r["category"],
          "fact_key": list(r["fact_key"]), "curriculum_tag": curriculum_tag(r)}
         for r in rows["train"]])
    for split in ("val", "test"):
        hashes[split] = write_new(out / f"{split}.json",
            [{"prompt": r["prompt"], "answer": r["answer"],
              "category": r["category"], "fact_key": list(r["fact_key"]),
              "curriculum_tag": curriculum_tag(r)} for r in rows[split]])
    rng = random.Random(SEED + 100)
    sample = rng.sample(rows["train"], min(256, len(rows["train"])))
    hashes["train_probe"] = write_new(out / "train_probe_256.json",
        [{"prompt": r["prompt"], "answer": r["answer"], "category": r["category"]}
         for r in sample])
    manifest = {
        "version": "round3-structured-because-v1",
        "status": "GENERATED_NOT_TRAINED",
        "seed": args.seed, "profile": args.profile,
        "sampling": ("All distinct train records sampled first where possible; "
                     "weighted repeat exposure for 0/1, 1-digit facts, round numbers "
                     "and carrying; exactly 100k train rows in full mode."),
        "split": ("Stable sha256 split by canonical addition/multiplication math fact; "
                  "80% train / 10% val / 10% test; same fact never crosses splits."),
        "note": ("Older common_integer_val_256 facts are excluded when that source "
                 "file is available locally during generation. They remain OOD "
                 "and are NOT a substitute for the new fixed validation set."),
        "positive": "{prompt} The answer is {answer} because {verified expression} equals {answer}.",
        "negative": f"{{prompt}} {REFUSAL}",
        "loss_plan": "Instructor's original course loss; start from sft/gpt.pt",
        "held_out_keys_protected": protected_count,
        "candidate_count_before_protection": attempted["generated"],
        "per_split": info,
        "sha256": hashes,
        "write_control": args.write_control,
    }
    write_new(out / "manifest.json", manifest)
    print(f"PASS: exact-answer and explanation checks, vocabulary, context, and fact-group splits")
    print(f"Dataset directory: {out}")
    print("Files never overwrite dpo/pos_neg_pairs.json or course checkpoints.")
    print("Next: run small GPU smoke and original course-loss training separately.")


if __name__ == "__main__":
    main()
