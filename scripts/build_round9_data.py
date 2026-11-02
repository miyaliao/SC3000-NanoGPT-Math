#!/usr/bin/env python3
"""Round9 data: matched Replay / Random / Structured from ONE candidate universe.

All rows retain the instructor Course-Loss answer-first format and refusal
negative.  Only TRAIN-partition canonical facts enter newly synthesized rows.
Known validation / public / teacher-style 680 facts are excluded from new rows.
The historical 70k replay may contain 680 Seen facts (report, do not hide).

CPU:
 python scripts/build_round9_data.py --self-test-only
 python scripts/build_round9_data.py --verify-only
 python scripts/build_round9_data.py --build
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_round1 import parse_relation, stable_split
from generate_round3_structured import (
    KINDS, REFUSAL, sum_relations, product_relations
)
from build_teacher_style_400 import source_and_protected, count_carries
from evaluate_teacher_style_680 import validate as validate_680

SEED = 42
OUT = ROOT / "dpo/data/round9_balanced_full"
OLD = ROOT / "dpo/data/round5_decomposition/original_pairs.json"
OLD_MANIFEST = ROOT / "dpo/data/round5_decomposition/manifest.json"
BASE = ROOT / "dpo/experiments/round7_replay/dpo.pt"
REPLAY_N, NEW_PER_CAT, ROWS = 70000, 2500, 100000
BINS = ("b0", "b1", "b2", "b3")
ADDITIVE = set(KINDS[:2]) | {"x_plus_a","a_plus_x","x_minus_a","a_minus_x"}
POS = re.compile(r"^(?P<prompt>.+?=\?) The answer is (?P<answer>-?\d+) because .+\.$")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bytes_json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def check_one(row, stoi, protected, new):
    m = POS.fullmatch(row["positive"])
    if not m:
        raise ValueError("Positive must be full answer-first because format")
    p, ans = m["prompt"], m["answer"]
    key = parse_relation(p, ans)
    if row["negative"] != p + " " + REFUSAL:
        raise ValueError("Non-refusal negative or mismatched prompt")
    if max(len(row["positive"]), len(row["negative"])) + 4 > 256:
        raise ValueError("Exceeds block_size=256")
    if set(row["positive"] + row["negative"]) - stoi:
        raise ValueError("Tokenizer out-of-vocabulary")
    if new and (key in protected or stable_split(key, SEED) != "train"):
        raise ValueError("New example contains protected/non-train math fact")
    return p, ans, key


def bin_for(category, key):
    """Four feasible mathematical strata per category; no fake examples."""
    if category in ADDITIVE:
        if key[0] != "add":
            raise ValueError("Wrong additive family")
        a, b = int(key[1]), int(key[2])
        # Balanced ranges + any carry; subtraction borrow is the inverse carry.
        return "b" + str(2 * int(max(a, b) > 30) + int(count_carries(a, b) > 0))
    if key[0] != "mul":
        raise ValueError("Wrong multiplicative family")
    a, b, product = int(key[1]), int(key[2]), int(key[3])
    if product <= 99: return "b0"
    if product <= 400 and min(a, b) <= 9: return "b1"
    if product <= 400: return "b2"
    if product <= 999: return "b3"
    raise ValueError("Product exceeds Round9 training domain (999)")


def pool_records(protected):
    pools = {c: {} for c in KINDS}
    def admit(record):
        key = tuple(record["fact_key"])
        if key in protected or stable_split(key, SEED) != "train":
            return
        cat = record["category"]
        bucket = bin_for(cat, key)
        pools[cat].setdefault(record["prompt"], {
            "positive": record["positive"], "negative": record["negative"],
            "prompt": record["prompt"], "answer": record["answer"],
            "fact_key": list(key), "category": cat, "bin": bucket
        })
    for a in range(0, 101):
        for b in range(a, 101):
            for x, y in (((a,b),(b,a)) if a != b else ((a,b),)):
                for rec in sum_relations(x, y):
                    admit(rec)
    for a in range(1, 100):
        for b in range(a, 100):
            if a * b > 999: continue
            for x, y in (((a,b),(b,a)) if a != b else ((a,b),)):
                for rec in product_relations(x, y):
                    admit(rec)
    by_cat = {}
    for cat in KINDS:
        records = list(pools[cat].values())
        counts = Counter(r["bin"] for r in records)
        if not records or any(counts[b] == 0 for b in BINS):
            raise RuntimeError(f"Insufficient eligible TRAIN-only candidates: {cat} {dict(counts)}")
        by_cat[cat] = records
        print(f"POOL {cat:12s}: n={len(records):5d}, "
              + " ".join(f"{b}={counts[b]}" for b in BINS), flush=True)
    return by_cat


def make():
    # Both benchmark parts MUST be frozen and hash-valid before generating training.
    core, extension, _, _ = validate_680()
    seen, protected, source_hashes = source_and_protected()
    teacher_keys = {tuple(r["fact_key"]) for r in core + extension}
    protected |= teacher_keys
    if len(core) + len(extension) != 680: raise RuntimeError("Frozen benchmark incomplete")
    if not OLD.is_file() or not BASE.is_file(): raise FileNotFoundError("Old data / Round7 replay missing")
    source_m = read(OLD_MANIFEST)
    if digest(OLD) != source_m["file_sha256"]["original"]:
        raise RuntimeError("Old Round5 TRAIN bytes changed from manifest")
    with (ROOT / "sft/meta.pkl").open("rb") as f:
        vocab = set(pickle.load(f)["stoi"])
    old = read(OLD)
    if len(old) != ROWS: raise RuntimeError("Old replay source must have 100000 rows")
    for pair in old:
        check_one(pair, vocab, protected, new=False)
    pools = pool_records(protected)
    rng = random.Random(42009)
    old_ids = list(range(ROWS))
    rng.shuffle(old_ids)
    common = [old[i] for i in old_ids[:REPLAY_N]]
    replay_extra = [old[i] for i in old_ids[REPLAY_N:]]
    random_extra, structured_extra = [], []
    for cat in KINDS:
        candidates = pools[cat]
        random_extra.extend(rng.choices(candidates, k=NEW_PER_CAT))
        by_bin = {b:[x for x in candidates if x["bin"] == b] for b in BINS}
        for bucket in BINS:
            structured_extra.extend(rng.choices(by_bin[bucket], k=NEW_PER_CAT // 4))
    if len(random_extra) != 30000 or len(structured_extra) != 30000:
        raise RuntimeError("30k newly sampled rows missing")
    full = {
        "replay": common + replay_extra,
        "random": common + random_extra,
        "structured": common + structured_extra
    }
    # Use the SAME positional shuffle for all three arms and metadata.
    order = list(range(ROWS))
    random.Random(420091).shuffle(order)
    pairs = {
        arm: [{"positive": src[i]["positive"], "negative": src[i]["negative"]}
              for i in order] for arm, src in full.items()
    }
    # Metadata mirrors structured arm row-for-row (helps exact error mining).
    structured_meta = []
    summary = {}
    for arm, src in full.items():
        rows = [src[i] for i in order]
        new_rows = [src[i] for i in order if i >= REPLAY_N]
        for r in rows:
            check_one(r, vocab, protected, new=False)
        for r in new_rows:
            if arm != "replay":
                check_one(r, vocab, protected, new=True)
        keys = {parse_relation(POS.fullmatch(r["positive"])["prompt"],
                               POS.fullmatch(r["positive"])["answer"]) for r in new_rows}
        summary[arm] = {
            "total_rows": len(rows), "extra_rows": len(new_rows),
            "extra_unique_facts": len(keys), "extra_new_to_historical_facts": len(keys - seen),
            "extra_unique_prompts": len({POS.fullmatch(r["positive"])["prompt"] for r in new_rows}),
            "extra_by_category": (dict(Counter(r["category"] for r in new_rows))
                                  if arm != "replay" else {}),
            "extra_by_category_bin": (
                {k:dict(Counter(r["bin"] for r in new_rows if r["category"]==k))
                 for k in KINDS} if arm != "replay" else {})
        }
        if arm == "structured":
            for idx, r in zip(order, rows):
                m = POS.fullmatch(r["positive"])
                key = parse_relation(m["prompt"], m["answer"])
                # Old replay rows can include other values / categories.
                structured_meta.append({
                    "prompt":m["prompt"], "answer":m["answer"],
                    "source":"old" if idx < REPLAY_N else "new",
                    "category": (r["category"] if idx >= REPLAY_N else None),
                    "bin": (r["bin"] if idx >= REPLAY_N else None),
                    "fact_key":list(key)
                })
    if pairs["structured"] == pairs["random"]:
        raise RuntimeError("Random and Structured samples unexpectedly identical")
    if len(structured_meta) != ROWS: raise RuntimeError("Misaligned metadata")
    manifest = {
        "version":"round9_structured_v1", "seed":SEED,
        "base_checkpoint_sha256":digest(BASE),
        "old_100k_sha256":digest(OLD),
        "old_manifest_sha256":digest(OLD_MANIFEST),
        "source_hashes":source_hashes,
        "core400_sha256":digest(ROOT/"dpo/benchmarks/teacher_style_400_v1/questions.json"),
        "extension280_sha256":digest(ROOT/"dpo/benchmarks/teacher_style_extension_280_v1/questions.json"),
        "protected_teacher_fact_count":len(teacher_keys),
        "train_rows": ROWS,"shared_old":REPLAY_N,
        "new_per_category":NEW_PER_CAT,"bin_quota_per_category":NEW_PER_CAT//4,
        "strata_additive":"max underlying addend <=30/>30 x zero/positive carry",
        "strata_multiplicative":"product<=99 / 100-400 one-digit factor / 100-400 two-digit factors / 401-999",
        "shared_permutation_seed":420091,
        "statistics":summary,
        "files_sha256":{}
    }
    for arm in pairs: manifest["files_sha256"][arm] = hashlib.sha256(bytes_json(pairs[arm])).hexdigest()
    manifest["files_sha256"]["structured_metadata"] = hashlib.sha256(bytes_json(structured_meta)).hexdigest()
    return pairs, structured_meta, manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test-only",action="store_true")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--build",action="store_true")
    args=p.parse_args()
    if args.self_test_only:
        for a,b in ((7,8),(29,17),(40,51)):
            for record in sum_relations(a,b):
                assert bin_for(record["category"],tuple(record["fact_key"])) in BINS
        for a,b in ((8,7),(8,24),(15,18),(41,19)):
            for record in product_relations(a,b):
                assert bin_for(record["category"],tuple(record["fact_key"])) in BINS
        print("SELFTEST PASS: 12 categories, canonical fact bins")
        return
    if args.build == args.verify_only:
        p.error("Choose exactly one of --verify-only or --build")
    if OUT.exists() and args.build:
        raise FileExistsError(f"Refusing to overwrite dataset: {OUT}")
    pairs, meta, manifest = make()
    for a, s in manifest["statistics"].items(): print(a,s,flush=True)
    if args.verify_only:
        print("AUDIT PASS; no files written")
        return
    OUT.mkdir(parents=True,exist_ok=False)
    for arm in ("replay","random","structured"):
        (OUT/f"{arm}_pairs.json").write_bytes(bytes_json(pairs[arm]))
    (OUT/"structured_metadata.json").write_bytes(bytes_json(meta))
    (OUT/"manifest.json").write_bytes(bytes_json(manifest))
    for item, hash_value in manifest["files_sha256"].items():
        path=OUT/("structured_metadata.json" if item=="structured_metadata" else f"{item}_pairs.json")
        if digest(path) != hash_value: raise RuntimeError(f"Output SHA mismatch: {path}")
    print(f"BUILD PASS: {OUT}; SHA256 and protected facts verified")


if __name__ == "__main__":
    main()
