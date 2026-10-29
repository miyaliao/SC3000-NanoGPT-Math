#!/usr/bin/env python3
"""Reproducible, exact-math SC3000 NanoGPT-Math Round 1 preference dataset.

Uses Python standard library only. Generates 12 math templates, exact finite
quarters/halves, strong math-fact-based split isolation and a frozen test set.

From repository root:
    python scripts/generate_round1.py --profile mini
    python scripts/generate_round1.py --profile full
    python scripts/generate_round1.py --profile full --install-train

Without --install-train, NEVER replaces dpo/pos_neg_pairs.json.
The existing instructor demo is backed up on first installation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import shutil

REPO = Path(__file__).resolve().parents[1]
KINDS = ("add", "subtract", "multiply", "divide",
         "x_plus_a", "a_plus_x", "x_minus_a", "a_minus_x",
         "x_times_a", "a_times_x", "x_over_a", "a_over_x")
OPERATORS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/",
             "x_plus_a": "+", "a_plus_x": "+", "x_minus_a": "-", "a_minus_x": "-",
             "x_times_a": "*", "a_times_x": "*", "x_over_a": "/", "a_over_x": "/"}
NUM = r"[+-]?\d+(?:\.\d+)?"
ARITH = re.compile(rf"^({NUM})([+*/-])({NUM})=\?$")
EQUATION = re.compile(rf"^({NUM}|x)([+*/-])({NUM}|x)=({NUM}),x=\?$")
REFUSAL = "Sorry, I don't know."
DEFAULT_CHARS = set("\n '*+,-./=?’0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
SEED = 42
# Reserve course demos, Notebook Step 8, and previous independent smoke probes.
RESERVED = {
    "79-7=?": "72", "74+8=?": "82", "x+55=95,x=?": "40", "1*x=6,x=?": "6",
    "17+19=?": "36", "3*17=?": "51", "72/4=?": "18", "72-x=34,x=?": "38",
    "x*11=44,x=?": "4", "7/2=?": "3.5", "5-9=?": "-4",
}


def canonical_number(v: Fraction) -> str:
    """Signed integer / exact terminating decimal with up to 2 decimal places."""
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    n = v * 100
    if n.denominator != 1:
        raise ValueError(f"Not expressible in <=2 decimal places: {v}")
    n = int(n)
    sign = "-" if n < 0 else ""
    whole, rest = divmod(abs(n), 100)
    return (f"{sign}{whole}.{rest:02d}").rstrip("0").rstrip(".")


def signed_int(rng: random.Random, lo: int, hi: int, negative_chance: float = 0) -> Fraction:
    n = rng.randint(lo, hi)
    return Fraction(-n if rng.random() < negative_chance else n)


def fractional(rng: random.Random, lo: int = 1, hi: int = 120, negative_chance: float = 0) -> Fraction:
    # Only .25, .5, .75; no float artifacts, no repeating decimals.
    n = rng.randint(lo * 4, hi * 4)
    while n % 4 == 0:
        n = rng.randint(lo * 4, hi * 4)
    return Fraction(-n if rng.random() < negative_chance else n, 4)


def problem(kind: str, decimal: bool, rng: random.Random) -> tuple[str, str]:
    """Construct true relations backwards; independently verified later."""
    a = signed_int(rng, 1, 100)
    b = signed_int(rng, 1, 100)
    x = fractional(rng, negative_chance=0.2) if decimal else signed_int(rng, 1, 100, .2)
    small = rng.randint(1, 30)
    if kind == "add":
        l, r = (x, b) if decimal else (a, b)
        q, ans = f"{canonical_number(l)}+{canonical_number(r)}=?", l + r
    elif kind == "subtract":
        l, r = (x, b) if decimal else (a, b)
        q, ans = f"{canonical_number(l)}-{canonical_number(r)}=?", l - r
    elif kind == "multiply":
        l, r = (x, Fraction(small)) if decimal else (a, Fraction(small))
        q, ans = f"{canonical_number(l)}*{canonical_number(r)}=?", l * r
    elif kind == "divide":
        if decimal:
            quotient = fractional(rng, lo=1, hi=100)
            divisor = Fraction(4 * rng.randint(1, 10))
            numerator = quotient * divisor
        else:
            quotient = signed_int(rng, 1, 100, .15)
            divisor = Fraction(small)
            numerator = quotient * divisor
        q, ans = f"{canonical_number(numerator)}/{canonical_number(divisor)}=?", quotient
    elif kind == "x_plus_a":
        q, ans = f"x+{canonical_number(a)}={canonical_number(x+a)},x=?", x
    elif kind == "a_plus_x":
        q, ans = f"{canonical_number(a)}+x={canonical_number(a+x)},x=?", x
    elif kind == "x_minus_a":
        q, ans = f"x-{canonical_number(a)}={canonical_number(x-a)},x=?", x
    elif kind == "a_minus_x":
        q, ans = f"{canonical_number(a)}-x={canonical_number(a-x)},x=?", x
    elif kind == "x_times_a":
        q, ans = f"x*{small}={canonical_number(x*small)},x=?", x
    elif kind == "a_times_x":
        q, ans = f"{small}*x={canonical_number(x*small)},x=?", x
    elif kind == "x_over_a":
        divisor = Fraction(rng.choice((2, 4, 5, 10))) if decimal else Fraction(small)
        quotient = fractional(rng, lo=1, hi=100, negative_chance=0.2) if decimal else signed_int(rng, 1, 100, .2)
        x = quotient * divisor
        q, ans = f"x/{canonical_number(divisor)}={canonical_number(quotient)},x=?", x
    elif kind == "a_over_x":
        quotient = signed_int(rng, 1, 30, .2)
        x = fractional(rng, lo=1, hi=100, negative_chance=0.2) if decimal else signed_int(rng, 1, 100, .2)
        numerator = x * quotient
        q, ans = f"{canonical_number(numerator)}/x={canonical_number(quotient)},x=?", x
    else:
        raise ValueError(kind)
    answer = canonical_number(ans)
    if decimal and "." not in (q + answer):
        raise AssertionError(f"Expected decimal case: {q} -> {answer}")
    if not decimal and "." in (q + answer):
        raise AssertionError(f"Integer-only case unexpectedly decimal: {q} -> {answer}")
    return q, answer


def calculate(left: Fraction, op: str, right: Fraction) -> Fraction:
    if op == "+": return left + right
    if op == "-": return left - right
    if op == "*": return left * right
    if op == "/" and right != 0: return left / right
    raise ValueError("Division by zero")


def parse_relation(prompt: str, answer: str):
    """Parse and verify a displayed problem using substitution, not generator logic."""
    answer_f = Fraction(answer)
    match = EQUATION.fullmatch(prompt)
    if match:
        l, op, r, rhs = match.groups()
        if (l == "x") == (r == "x"):
            raise ValueError(f"Must contain exactly one x: {prompt}")
        lv = answer_f if l == "x" else Fraction(l)
        rv = answer_f if r == "x" else Fraction(r)
        cv = Fraction(rhs)
        if calculate(lv, op, rv) != cv:
            raise ValueError(f"Wrong answer: {prompt} -> {answer}")
    else:
        match = ARITH.fullmatch(prompt)
        if not match:
            raise ValueError(f"Invalid math prompt: {prompt}")
        l, op, r = match.groups()
        lv, rv, cv = Fraction(l), Fraction(r), answer_f
        if calculate(lv, op, rv) != cv:
            raise ValueError(f"Wrong answer: {prompt} -> {answer}")

    # Group equivalent arithmetic identities across question templates and rearrangements.
    if op == "+":
        family, left, right, result = "add", lv, rv, cv
    elif op == "-":
        family, left, right, result = "add", rv, cv, lv
    elif op == "*":
        family, left, right, result = "mul", lv, rv, cv
    else:
        family, left, right, result = "mul", rv, cv, lv
    if family == "mul" and left == 0 and right != 0:
        # Zero relations are degenerate for inversion-based group matching.
        pass
    x, y = sorted((left, right))
    return (family, str(x), str(y), str(result))


def stable_split(group_key, seed: int) -> str:
    encoded = json.dumps([seed, *group_key], separators=(",", ":"), ensure_ascii=True).encode()
    point = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % 10000
    return "train" if point < 8000 else "val" if point < 9000 else "test"


def quota(profile: str):
    if profile == "full":
        return {
            "train": {kind: (2560 if i < 4 else 1280) for i, kind in enumerate(KINDS)},
            "val": {kind: (256 if i < 4 else 128) for i, kind in enumerate(KINDS)},
            "test": {kind: (256 if i < 4 else 128) for i, kind in enumerate(KINDS)},
        }
    if profile == "mini":
        return {
            "train": {kind: (16 if i < 4 else 8) for i, kind in enumerate(KINDS)},
            "val": {kind: (4 if i < 4 else 2) for i, kind in enumerate(KINDS)},
            "test": {kind: (4 if i < 4 else 2) for i, kind in enumerate(KINDS)},
        }
    raise ValueError(profile)


def accepted_records(profile: str, seed: int):
    rng = random.Random(seed)
    targets = quota(profile)
    by_split = {split: [] for split in targets}
    seen_questions = set()
    reserved_keys = {parse_relation(p, answer) for p, answer in RESERVED.items()}
    seen_keys_by_split = {split: set() for split in targets}
    attempted = Counter()
    for split in ("train", "val", "test"):
        for kind in KINDS:
            target = targets[split][kind]
            decimal_target = round(target * .15)
            for decimal, count in ((False, target - decimal_target), (True, decimal_target)):
                for _ in range(count):
                    for tries in range(6000):
                        attempted[split] += 1
                        q, ans = problem(kind, decimal, rng)
                        if q in RESERVED or q in seen_questions:
                            continue
                        key = parse_relation(q, ans)
                        if key in reserved_keys or stable_split(key, seed) != split:
                            continue
                        seen_questions.add(q)
                        seen_keys_by_split[split].add(key)
                        by_split[split].append({
                            "prompt": q, "answer": ans, "category": kind,
                            "number_type": "decimal" if decimal else "integer",
                            "fact_key": key,
                        })
                        break
                    else:
                        raise RuntimeError(f"Could not fill {split}:{kind}:decimal={decimal}; expand candidate ranges")
        rng.shuffle(by_split[split])
    groups = list(seen_keys_by_split.values())
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    return by_split, attempted


def validate_all(by_split, stoi: set[str] | None = None):
    alphabet = stoi if stoi is not None else DEFAULT_CHARS
    prompts = set()
    keys_by_split = {}
    summary = {}
    for split, records in by_split.items():
        keys_by_split[split] = set()
        cats = Counter()
        types = Counter()
        max_pair_length = 0
        for item in records:
            p, a = item["prompt"], item["answer"]
            key = parse_relation(p, a)
            assert tuple(item["fact_key"]) == key
            assert key not in {parse_relation(r, v) for r, v in RESERVED.items()}
            assert p not in prompts
            assert len(p) < 256
            prompts.add(p)
            keys_by_split[split].add(key)
            assert canonical_number(Fraction(a)) == a
            pos = f"{p} The answer is {a}."
            neg = f"{p} {REFUSAL}"
            assert pos != neg
            assert set(pos + neg).issubset(alphabet), (p, sorted(set(pos + neg) - alphabet))
            assert max(len(pos), len(neg)) + 4 <= 256
            max_pair_length = max(max_pair_length, len(pos), len(neg))
            cats[item["category"]] += 1
            types[item["number_type"]] += 1
        summary[split] = {"total": len(records), "categories": dict(cats),
                          "number_types": dict(types), "max_pair_length": max_pair_length,
                          "unique_facts": len(keys_by_split[split])}
    assert not(keys_by_split["train"] & keys_by_split["val"])
    assert not(keys_by_split["train"] & keys_by_split["test"])
    assert not(keys_by_split["val"] & keys_by_split["test"])
    return summary


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(profile: str, seed: int, out_dir: Path, stoi: set[str] | None):
    by_split, attempts = accepted_records(profile, seed)
    summary = validate_all(by_split, stoi=stoi)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, records in by_split.items():
        if split == "train":
            path = out_dir / "pos_neg_pairs.json"
            data = [{"positive": f"{r['prompt']} The answer is {r['answer']}.",
                     "negative": f"{r['prompt']} {REFUSAL}"} for r in records]
        else:
            path = out_dir / f"round1_{split}.json"
            data = [{"prompt": r["prompt"], "answer": r["answer"],
                     "category": r["category"], "number_type": r["number_type"]} for r in records]
        write_json(path, data)
        paths[split] = path
    write_json(out_dir / "round1_train_metadata.json",
               [{"prompt": r["prompt"], "answer": r["answer"],
                 "category": r["category"], "number_type": r["number_type"],
                 "fact_key": list(r["fact_key"])} for r in by_split["train"]])
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()}
    manifest = {"version": "round1-refusal-v1", "seed": seed, "profile": profile,
                "positive_template": "{prompt} The answer is {answer}.",
                "negative_template": "{prompt} Sorry, I don't know.",
                "sampling": "Exact Fraction; 12 kinds; ~15% decimal; fact-group hash split; no leakage",
                "hash_rule": "sha256(json.dumps([seed, *fact_key], separators=(',', ':'))) first8bytes modulo 10000; train 0-7999 val 8000-8999 test 9000-9999",
                "data": summary, "attempts": dict(attempts), "file_sha256": hashes,
                "reserved_prompts": sorted(RESERVED),
                "test_policy": "Use validation for tuning; do not train on validation/test, including model-mined errors."}
    write_json(out_dir / "round1_manifest.json", manifest)
    return manifest, paths


def install(train_path: Path, repo: Path):
    target = repo / "dpo" / "pos_neg_pairs.json"
    backup = repo / "dpo" / "data" / "course_examples.json"
    existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
    if existing is not None and len(existing) == 4 and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, backup)
    elif existing is not None and len(existing) not in (4, 20480, 128):
        raise RuntimeError("Refusing to overwrite a non-demo, non-Round-1 training dataset")
    shutil.copyfile(train_path, target)
    print(f"Installed Round 1 training JSON into {target} (previous 4-demo backed up if applicable)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("mini", "full"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, help="Defaults to dpo/data/round1_{profile}")
    parser.add_argument("--install-train", action="store_true",
                        help="Install the training data into dpo/pos_neg_pairs.json after validation")
    args = parser.parse_args()
    out = args.out_dir if args.out_dir else REPO / "dpo" / "data" / f"round1_{args.profile}"
    if args.install_train and args.profile != "full":
        parser.error("Only the full dataset may be installed into the formal training path")
    vocab = None
    meta = REPO / "sft" / "meta.pkl"
    if meta.is_file():
        with meta.open("rb") as f:
            vocab = set(pickle.load(f)["stoi"])
    manifest, paths = generate(args.profile, args.seed, out, vocab)
    if args.install_train:
        install(paths["train"], REPO)
    for split, item in manifest["data"].items():
        print(f"{split}: {item['total']:,} unique questions | {item['unique_facts']:,} unique math facts | "
              f"{item['number_types']} | longest pair {item['max_pair_length']} chars")
    print("PASS: exact mathematical verification, no duplicate questions, no fact-level split overlap")
    print("PASS: character vocabulary, 256-character context, correct positive/negative templates")
    print(f"Manifest: {out / 'round1_manifest.json'}")


if __name__ == "__main__":
    main()
