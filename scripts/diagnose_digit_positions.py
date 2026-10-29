#!/usr/bin/env python3
"""Read-only, teacher-forced digit-position diagnosis for SC3000 Overfit-256.

Compare fixed 256 SEEN Round-1C training prompts (never train on validation).
For each integer gold answer, measure next-character top-1 and probability
at each place (units/tens/hundreds/thousands) and at the final period.
Teacher forcing supplies the GOLD previous characters; these are not free-
generation accuracy estimates. Sign is scored separately, not as a digit.

Usage:
  python scripts/diagnose_digit_positions.py --self-test-only
  python scripts/diagnose_digit_positions.py --check-data-only
  python scripts/diagnose_digit_positions.py --device cuda
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import log
from pathlib import Path
import pickle
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from diagnose_generation_gap import load_model  # noqa: E402
from overfit256_course import select_rows, EXPECTED_SHA  # noqa: E402

PREFIX = ' The answer is '
CHECKPOINTS = (
    'dpo/experiments/overfit256_termination_gamma025/dpo.pt',
    'dpo/experiments/overfit256_termination_gamma050/dpo.pt',
    'dpo/experiments/overfit256_termination_aware/dpo.pt',
)


def position_records(answer: str):
    """Classify expected numeral characters by decimal place; minus is separate."""
    if not answer or not answer.lstrip('-').isdigit() or answer == '-0':
        raise ValueError(f'Expected canonical integer answer: {answer!r}')
    digits = answer.lstrip('-')
    if len(digits) > 1 and digits[0] == '0':
        raise ValueError(f'Leading zeros are not canonical: {answer!r}')
    sign_len = len(answer) - len(digits)
    out = []
    for idx, char in enumerate(answer):
        if idx < sign_len:
            out.append({'char': char, 'kind': 'sign', 'place': None, 'digit_index': None})
        else:
            d_idx = idx - sign_len
            out.append({'char': char, 'kind': 'digit',
                        'place': len(digits) - d_idx - 1, 'digit_index': d_idx})
    return out


def self_test():
    assert [(x['kind'], x['place']) for x in position_records('63')] == [('digit', 1), ('digit', 0)]
    assert [(x['char'], x['place']) for x in position_records('-340')] == [('-', None), ('3', 2), ('4', 1), ('0', 0)]
    assert [x['place'] for x in position_records('9')] == [0]
    for bad in ('', '3.5', '-0', '007', '12x'):
        try:
            position_records(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Unexpected accepted answer {bad!r}')
    print('PASS: sign separated from digits, place values indexed right-to-left')


def new_stats():
    return {'correct': 0, 'total': 0, 'prob_sum': 0., 'nll_sum': 0.,
            'premature_period_prob_sum': 0.}


def add(stats, top1: bool, probability: float, period_probability: float):
    stats['total'] += 1
    stats['correct'] += int(top1)
    stats['prob_sum'] += probability
    stats['nll_sum'] -= log(max(probability, 1e-300))
    stats['premature_period_prob_sum'] += period_probability


def finish(stats):
    n = stats['total']
    return {'n': n, 'correct': stats['correct'],
            'top1_rate': stats['correct'] / n if n else None,
            'mean_gold_probability': stats['prob_sum'] / n if n else None,
            'mean_gold_nll': stats['nll_sum'] / n if n else None,
            'mean_early_period_probability': stats['premature_period_prob_sum'] / n if n else None}


def audit(model, rows, stoi, device):
    by_place, by_length, by_category = defaultdict(new_stats), defaultdict(new_stats), defaultdict(new_stats)
    by_length_place = defaultdict(lambda: defaultdict(new_stats))
    sign = new_stats()
    total_numerals = defaultdict(int)
    last_period_probs = []
    period_top1 = 0
    individual = []
    dot_id = stoi['.']
    with torch.inference_mode():
        for row in rows:
            prompt, gold = row['prompt'], str(row['correct_answer'])
            classified = position_records(gold)
            sequence = prompt + PREFIX + gold + '.'
            unknown = set(sequence) - set(stoi)
            if unknown:
                raise ValueError(f'Unknown tokenizer chars: {unknown}')
            seq_ids = [stoi[ch] for ch in sequence]
            if len(seq_ids) > model.config.block_size:
                raise ValueError('Example exceeds context length')
            tokens = torch.tensor([seq_ids[:-1]], dtype=torch.long, device=device)
            logits, _ = model(tokens, full_seq=True)
            start = len(prompt + PREFIX)
            digit_flags = []
            chars = []
            for j, item in enumerate(classified):
                absolute = start + j
                logits_here = logits[0, absolute - 1]
                probabilities = F.softmax(logits_here, dim=-1)
                true_id = stoi[item['char']]
                probability = float(probabilities[true_id].item())
                period_probability = float(probabilities[dot_id].item())
                top1 = int(logits_here.argmax().item()) == true_id
                if item['kind'] == 'digit':
                    add(by_place[item['place']], top1, probability, period_probability)
                    length = len(gold.lstrip('-'))
                    add(by_length[length], top1, probability, period_probability)
                    add(by_length_place[length][item['place']], top1, probability, period_probability)
                    add(by_category[row['category']], top1, probability, period_probability)
                    digit_flags.append(top1)
                else:
                    add(sign, top1, probability, period_probability)
                chars.append({'gold': item['char'], 'kind': item['kind'], 'place': item['place'],
                              'top1': top1, 'gold_probability': probability,
                              'period_probability_before_char': period_probability})
            period_logits = logits[0, len(sequence)-2]
            period_probs = F.softmax(period_logits, dim=-1)
            last_period_probs.append(float(period_probs[dot_id].item()))
            period_top1 += int(period_logits.argmax().item()) == dot_id
            total_numerals['all_digits_top1'] += int(all(digit_flags))
            total_numerals['first_digit_top1'] += int(digit_flags[0])
            total_numerals['n'] += 1
            individual.append({'prompt': prompt, 'answer': gold, 'category': row['category'],
                               'answer_digits': len(gold.lstrip('-')),
                               'all_digits_top1': all(digit_flags),
                               'first_digit_top1': digit_flags[0],
                               'gold_end_probability': last_period_probs[-1], 'characters': chars})
    n = total_numerals['n']
    result = {
        'n': n, 'checkpoint_type': 'teacher_forced_gold_prefix',
        'all_digits_top1': {'count': total_numerals['all_digits_top1'], 'rate': total_numerals['all_digits_top1']/n},
        'first_digit_top1': {'count': total_numerals['first_digit_top1'], 'rate': total_numerals['first_digit_top1']/n},
        'end_period_top1': {'count': period_top1, 'rate': period_top1/n,
                            'mean_probability': sum(last_period_probs)/n},
        'digit_places': {str(k): finish(v) for k, v in sorted(by_place.items())},
        'by_answer_length': {str(k): finish(v) for k, v in sorted(by_length.items())},
        'by_category': {str(k): finish(v) for k, v in sorted(by_category.items())},
        'by_length_and_place': {str(length): {str(place): finish(v) for place, v in sorted(sub.items())}
                                for length, sub in sorted(by_length_place.items())},
        'sign': finish(sign),
    }
    return result, individual


def show(name, result):
    print(f'=== {name} ===')
    print(f"Gold all digits top-1: {result['all_digits_top1']['count']}/{result['n']} ({result['all_digits_top1']['rate']:.2%})")
    print(f"Gold first digit top-1: {result['first_digit_top1']['count']}/{result['n']} ({result['first_digit_top1']['rate']:.2%})")
    end = result['end_period_top1']
    print(f"Gold end-period top-1: {end['count']}/{result['n']} ({end['rate']:.2%}); mean P(.)={end['mean_probability']:.4f}")
    for place, s in result['digit_places'].items():
        label = ['units', 'tens', 'hundreds', 'thousands'][int(place)] if int(place) < 4 else '10^'+place
        print(f"  {label:>10}: {s['correct']}/{s['n']} ({s['top1_rate']:.2%})  mean P(gold)={s['mean_gold_probability']:.4f}")
    for length, cells in result['by_length_and_place'].items():
        if int(length) in (2, 3):
            vals = '  '.join(f"10^{place}: {c['correct']}/{c['n']} ({c['top1_rate']:.1%})" for place, c in sorted(cells.items()))
            print(f'  within {length}-digit answers: {vals}')
    sign = result['sign']
    if sign['n']:
        print(f"  {'minus sign':>10}: {sign['correct']}/{sign['n']} ({sign['top1_rate']:.2%})")
    print('NOTE: gold-prefix teacher forcing; these are not free-generation accuracies.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoints', nargs='+', default=list(CHECKPOINTS))
    p.add_argument('--out', default='results/digit_position_gamma_256.json')
    p.add_argument('--self-test-only', action='store_true')
    p.add_argument('--check-data-only', action='store_true')
    args = p.parse_args()
    self_test()
    if args.self_test_only:
        return
    if args.seed != 42:
        p.error('Use identical preselected 256 training questions (seed 42)')
    _, rows, _, _ = select_rows(args.seed)
    assert len(rows) == 256
    if args.check_data_only:
        print('PASS: hash-verified exact same 256 Round 1C seen training questions')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA not available: use sbatch on TC1')
    with (ROOT/'sft/meta.pkl').open('rb') as f:
        meta = pickle.load(f)  # only course-trusted local meta.pkl
    stoi = meta['stoi']
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT/out
    if not out.resolve().is_relative_to((ROOT/'results').resolve()):
        p.error('Output must reside under results/')
    if out.exists():
        p.error(f'Output exists; refusing to overwrite: {out}')
    reports = {'seed': args.seed, 'source_round1c_sha256': EXPECTED_SHA,
               'n': 256, 'definition': 'Teacher forcing with gold preceding characters, not natural greedy generation',
               'models': {}}
    for name in args.checkpoints:
        checkpoint = Path(name)
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT/checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f'Missing model: {checkpoint}')
        model = load_model(checkpoint, args.device)
        report, detailed = audit(model, rows, stoi, args.device)
        reports['models'][name] = {'summary': report, 'per_question': detailed}
        show(name, report)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('=== DIGIT POSITION DIAGNOSIS COMPLETE ===')
    print('Saved:', out)


if __name__ == '__main__':
    main()