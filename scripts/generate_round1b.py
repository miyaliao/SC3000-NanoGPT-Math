#!/usr/bin/env python3
"""Create a controlled integer-only SC3000 Round 1B DPO dataset.

Reuses Round 1A's deterministic generator and exact mathematical validator.
Keeps ALL 17,408 Round 1A integer training examples and substitutes new
integer examples for the 3,072 decimal training examples, holding all other
experiment dimensions fixed. Does NOT replace dpo/pos_neg_pairs.json or dpo/dpo.pt.

A fixed, category-stratified set of 256 INTEGER validation questions is drawn
from the already-frozen Round 1A validation set (never from training).

Examples:
  python scripts/generate_round1b.py
  python scripts/generate_round1b.py --verify-only
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
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import generate_round1 as base  # noqa: E402

SEED = 42
EXTRA_SEED = 14042
VALIDATION_SEED = 24042
OUT = ROOT / 'dpo' / 'experiments' / 'round1b_integer'
OLD_VALIDATION = ROOT / 'dpo' / 'data' / 'round1_full' / 'round1_val.json'
OLD_MANIFEST = ROOT / 'dpo' / 'data' / 'round1_full' / 'round1_manifest.json'
KIND_TARGETS = {k: (2560 if i < 4 else 1280) for i, k in enumerate(base.KINDS)}
VAL_COUNTS = {k: (32 if i < 4 else 16) for i, k in enumerate(base.KINDS)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def record_is_integer(row):
    return '.' not in (row['prompt'] + str(row['answer'])) and Fraction(row['answer']).denominator == 1


def make_dataset():
    original_manifest = json.loads(OLD_MANIFEST.read_text(encoding='utf-8'))
    assert original_manifest['seed'] == SEED and original_manifest['profile'] == 'full'
    val_rows = json.loads(OLD_VALIDATION.read_text(encoding='utf-8'))
    if sha256(OLD_VALIDATION) != original_manifest['file_sha256']['val']:
        raise RuntimeError('Frozen Round 1A validation JSON changed; aborting')

    source, attempts = base.accepted_records('full', SEED)
    original_integer = [r for r in source['train'] if r['number_type'] == 'integer']
    original_all_questions = {r['prompt'] for split in source for r in source[split]}
    old_train_fact = {tuple(r['fact_key']) for r in source['train']}
    other_fact = {tuple(r['fact_key']) for split in ('val', 'test') for r in source[split]}
    reserved_fact = {base.parse_relation(q, answer) for q, answer in base.RESERVED.items()}
    assert not (old_train_fact & other_fact)
    assert len(original_integer) == 17408

    selected = list(original_integer)
    rng = random.Random(EXTRA_SEED)
    counts = Counter(r['category'] for r in selected)
    seen_questions = set(original_all_questions)
    for kind in base.KINDS:
        missing = KIND_TARGETS[kind] - counts[kind]
        assert missing > 0
        for _ in range(missing):
            for attempt in range(20000):
                if kind == 'multiply':
                    # Round 1A already occupies most (1..100)*(1..30) train
                    # candidates. Add the same single-step multiplication
                    # template with a modestly extended 31..50 multiplier.
                    left = rng.randint(1, 100)
                    if rng.random() < 0.05:
                        left = -left
                    right = rng.randint(31, 50)
                    question, answer = f'{left}*{right}=?', str(left * right)
                else:
                    question, answer = base.problem(kind, decimal=False, rng=rng)
                if question in seen_questions or question in base.RESERVED:
                    continue
                fact_key = base.parse_relation(question, answer)
                if fact_key in reserved_fact or fact_key in other_fact:
                    continue
                if base.stable_split(fact_key, SEED) != 'train':
                    continue
                selected.append({'prompt': question, 'answer': answer,
                                 'category': kind, 'number_type': 'integer',
                                 'fact_key': fact_key})
                seen_questions.add(question)
                counts[kind] += 1
                break
            else:
                raise RuntimeError(f'Unable to generate unique {kind} integer problem')
    random.Random(34042).shuffle(selected)
    assert len(selected) == 20480
    assert all(counts[k] == KIND_TARGETS[k] for k in base.KINDS)
    assert len({r['prompt'] for r in selected}) == len(selected)
    train_facts = {tuple(r['fact_key']) for r in selected}
    assert not (train_facts & other_fact)
    assert all(base.parse_relation(r['prompt'], r['answer']) == tuple(r['fact_key']) for r in selected)
    assert all(record_is_integer(r) for r in selected)

    valid_old = [r for r in val_rows if record_is_integer(r)]
    assert len(valid_old) == 1744, f'Unexpected old validation integer size: {len(valid_old)}'
    by_kind = {kind: [] for kind in base.KINDS}
    for row in valid_old:
        by_kind[row['category']].append(row)
    val_rng = random.Random(VALIDATION_SEED)
    selected_val = []
    for kind in base.KINDS:
        category_rows = by_kind[kind]
        assert len(category_rows) >= VAL_COUNTS[kind]
        selected_val.extend(val_rng.sample(category_rows, VAL_COUNTS[kind]))
    val_rng.shuffle(selected_val)
    assert len(selected_val) == 256 and len({r['prompt'] for r in selected_val}) == 256
    assert all(base.parse_relation(r['prompt'], str(r['answer'])) not in train_facts for r in selected_val)
    assert all(record_is_integer(r) for r in selected_val)

    meta_path = ROOT / 'sft' / 'meta.pkl'
    stoi = None
    if meta_path.is_file():
        with meta_path.open('rb') as f:
            stoi = set(pickle.load(f)['stoi'])
    base.validate_all({'train': selected, 'val': [], 'test': []}, stoi=stoi)
    for row in selected_val:
        base.parse_relation(row['prompt'], str(row['answer']))

    pairs = [{'positive': f"{r['prompt']} The answer is {r['answer']}.",
              'negative': f"{r['prompt']} {base.REFUSAL}"} for r in selected]
    train_path = OUT / 'pos_neg_pairs.json'
    val_path = OUT / 'common_integer_val_256.json'
    meta_out = OUT / 'round1b_train_metadata.json'
    save_json(train_path, pairs)
    save_json(val_path, selected_val)
    save_json(meta_out, [{k: list(v) if k == 'fact_key' else v for k,v in r.items()}
                         for r in selected])
    manifest = {
        'stage': 'round1b-integer-only', 'seed': SEED, 'extra_seed': EXTRA_SEED,
        'validation_seed': VALIDATION_SEED,
        'training_size': len(pairs), 'validation_size': len(selected_val),
        'original_train_integer_retained': len(original_integer),
        'new_integer_replacements': len(selected)-len(original_integer),
        'train_by_category': {k:counts[k] for k in base.KINDS},
        'validation_by_category': {k:VAL_COUNTS[k] for k in base.KINDS},
        'positive_format':'{prompt} The answer is {answer}.',
        'negative_format':"{prompt} Sorry, I don't know.",
        'original_validation_sha256': sha256(OLD_VALIDATION),
        'original_round1a_train_sha256': original_manifest['file_sha256']['train'],
        'file_sha256': {'train':sha256(train_path),'val':sha256(val_path),'metadata':sha256(meta_out)},
        'notes': 'Integer-only ablation; same 12 categories, 5 epochs, 64 batch, 1,600 steps. Original Round 1A files unchanged. Validation is a frozen integer-only subset of Round 1A validation.'
    }
    save_json(OUT / 'round1b_manifest.json', manifest)
    print('=== ROUND 1B DATASET PASSED ===')
    print('Train:',len(pairs),'| Retained original integer:',len(original_integer),
          '| New integer:',len(selected)-len(original_integer))
    print('Validation (frozen common):',len(selected_val))
    print('Train SHA256:',manifest['file_sha256']['train'])
    print('Output:',OUT)
    return manifest


def verify_only():
    manifest = json.loads((OUT / 'round1b_manifest.json').read_text(encoding='utf-8'))
    for key, path in [('train', OUT / 'pos_neg_pairs.json'),
                      ('val', OUT / 'common_integer_val_256.json'),
                      ('metadata', OUT / 'round1b_train_metadata.json')]:
        assert sha256(path)==manifest['file_sha256'][key], f'SHA mismatch: {key}'
    rows = json.loads((OUT / 'round1b_train_metadata.json').read_text())
    val = json.loads((OUT / 'common_integer_val_256.json').read_text())
    base.validate_all({'train': rows, 'val': [], 'test': []})
    assert all(record_is_integer(r) for r in rows+val)
    train_keys = {base.parse_relation(r['prompt'],r['answer']) for r in rows}
    assert all(base.parse_relation(r['prompt'],r['answer']) not in train_keys for r in val)
    print('PASS: integer-only, exact math, frozen val separation and all hashes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        verify_only()
    else:
        make_dataset()


if __name__=='__main__':
    main()
