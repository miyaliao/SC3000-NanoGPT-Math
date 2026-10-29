#!/usr/bin/env python3
"""Integer-only evaluator with original NanoGPT '.'/newline stopping semantics.

This reproduces the instructor model.py STOP TOKENS with greedy argmax for
reproducibility. It is a TA-like diagnostic, NOT a claim about undisclosed
TA sampling/parsing code. Do not use this evaluator for decimal questions.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import re
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(SCRIPT_DIR))
from model import GPT, GPTConfig  # noqa: E402
from math_eval_utils import REFUSAL_PATTERN  # noqa: E402

ANSWER=re.compile(r"\s*(?:The answer is\s+)?([+-]?\d+)\.?\s*\Z", re.I)


def evaluate_answer(completion, answer):
    m=ANSWER.fullmatch(completion)
    if m:
        p=int(m.group(1))
        return ('correct' if p==int(answer) else 'math_error'), p
    if REFUSAL_PATTERN.search(completion):
        return 'refusal',None
    return 'format_failure',None


def greedy_original_stop(gpt,prompt,stoi,itos,limit,device):
    tokens=[stoi[c] for c in prompt]
    original=len(tokens)
    if len(tokens)>gpt.config.block_size:
        raise ValueError('Prompt over model context length')
    result=[]
    stop='max_new_tokens'
    with torch.inference_mode():
        for _ in range(limit):
            ids=torch.tensor([tokens[-gpt.config.block_size:]],dtype=torch.long,device=device)
            logits,_=gpt(ids)
            next_id=int(logits[0,-1].argmax().item())
            if next_id==0:  # original model.py: newline not appended
                stop='newline'
                break
            tokens.append(next_id)
            result.append(next_id)
            if next_id==7:  # original model.py: '.' appended and then stop
                stop='period'
                break
    return ''.join(itos[t] for t in result),stop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--test',default='dpo/experiments/round1b_integer/common_integer_val_256.json',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--max-new-tokens',type=int,default=80)
    args=p.parse_args()
    if args.max_new_tokens<1:p.error('--max-new-tokens must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('GPU unavailable')
    path=lambda x: x if x.is_absolute() else ROOT/x
    rows=json.loads(path(args.test).read_text(encoding='utf-8'))
    with (ROOT/'sft/meta.pkl').open('rb') as f:
        vocab=pickle.load(f)
    stoi,itos=vocab['stoi'],vocab['itos']
    ckpt=torch.load(path(args.checkpoint),map_location='cpu',weights_only=True)
    gpt=GPT(GPTConfig(**ckpt['model_args']))
    original=ckpt.get('model',ckpt.get('model_state_dict'))
    if original is None:raise ValueError('Checkpoint missing model weights')
    state={k.removeprefix('_orig_mod.'):v for k,v in original.items()}
    gpt.load_state_dict(state,strict=True)
    gpt.to(args.device).eval()
    preds=[]
    overall=Counter()
    by_category=defaultdict(Counter)
    for i,r in enumerate(rows):
        if '.' in (r['prompt']+str(r['answer'])):raise ValueError('Decimal detected in integer-only val')
        if not re.fullmatch(r'[+-]?\d+',str(r['answer'])):raise ValueError('Non-integer expected answer')
        if set(r['prompt'])-set(stoi):raise ValueError('Unknown input chars')
        completion,stop=greedy_original_stop(gpt,r['prompt'],stoi,itos,args.max_new_tokens,args.device)
        status,prediction=evaluate_answer(completion,r['answer'])
        overall[status]+=1;by_category[r['category']][status]+=1
        preds.append({'prompt':r['prompt'],'answer':str(r['answer']),
                      'category':r['category'],'completion':completion,'status':status,
                      'prediction':prediction,'stop_reason':stop})
        if i<8:print(f"{r['prompt']} expected={r['answer']} -> {completion!r} [{status}]",flush=True)
    out=path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in preds:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f"MODEL {args.checkpoint} | N={len(rows)} | output={out}")
    for kind,counts in [('overall',overall),*sorted(by_category.items())]:
        n=sum(counts.values())
        print(f"{kind:15} N={n:3d} correct={counts['correct']:3d} ({counts['correct']/n:.1%}) "
              f"math_error={counts['math_error']:3d} refusal={counts['refusal']:3d} "
              f"format_failure={counts['format_failure']:3d}")


if __name__=='__main__':
    main()
