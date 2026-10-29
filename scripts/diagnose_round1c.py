#!/usr/bin/env python3
"""Read-only Round 1C diagnosis: seen-train vs held-out accuracy + preference margins.

Run in NanoGPT-Math repo via a Slurm GPU allocation:
    python scripts/diagnose_round1c.py --device cuda

Never trains, changes data, writes a checkpoint, or touches held-out test data.
Uses same greedy argmax and first-period/newline stop as evaluate_ta_integer.py.
Scoring reproduces course compute_logprob()'s nonzero target mask, and also
reports suffix-only summed/mean token log-probabilities for the SAME pairs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402
from evaluate_ta_integer import evaluate_answer, greedy_original_stop  # noqa: E402

TRAIN_ROOT = ROOT / 'dpo/experiments/round1c_incorrect'
DEFAULT_CHECKPOINT = TRAIN_ROOT / 'dpo.pt'
OUT = ROOT / 'results/diagnosis_round1c'
CATEGORIES = ('add','subtract','multiply','divide',
              'x_plus_a','a_plus_x','x_minus_a','a_minus_x',
              'x_times_a','a_times_x','x_over_a','a_over_x')
QUOTAS = {k: (32 if i < 4 else 16) for i,k in enumerate(CATEGORIES)}
PFX = ' The answer is '


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    man = read_json(TRAIN_ROOT / 'round1c_manifest.json')
    paths = {'train': TRAIN_ROOT / 'pos_neg_pairs.json',
             'metadata': TRAIN_ROOT / 'round1c_train_metadata.json',
             'val': TRAIN_ROOT / 'common_integer_val_256.json'}
    for key,path in paths.items():
        if sha256(path) != man['file_sha256'][key]:
            raise RuntimeError(f'Round 1C {key} file SHA-256 differs from frozen manifest')
    pairs,meta,val = (read_json(paths[k]) for k in ('train','metadata','val'))
    if not (len(pairs)==len(meta)==20480 and len(val)==256):
        raise ValueError('Unexpected Round 1C data count')
    for p,m in zip(pairs,meta):
        question = m['prompt']
        if p['positive'] != question + PFX + m['correct_answer'] + '.' or \
           p['negative'] != question + PFX + m['wrong_answer'] + '.':
            raise ValueError('Round 1C pair/metadata mismatch')
        if len(p['positive']) != len(p['negative']):
            raise ValueError('Pair has mismatched lengths')
    train_q = {m['prompt'] for m in meta}
    if len(train_q)!=len(meta) or any(v['prompt'] in train_q for v in val):
        raise ValueError('Prompt overlap within train or between train and validation')
    return pairs,meta,val,man


def sample_stratified(meta, seed):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for i,m in enumerate(meta):
        by_cat[m['category']].append(i)
    if set(by_cat) != set(CATEGORIES):
        raise RuntimeError('Unexpected categories in training metadata')
    picks=[]
    for cat in CATEGORIES:
        if len(by_cat[cat]) < QUOTAS[cat]:
            raise RuntimeError('Insufficient training items for '+cat)
        picks.extend(rng.sample(by_cat[cat], QUOTAS[cat]))
    rng.shuffle(picks)
    if len(picks)!=256 or len(set(picks))!=256:
        raise AssertionError('Invalid sampling')
    return picks


def load_model(checkpoint, device):
    with (ROOT/'sft/meta.pkl').open('rb') as f:
        vocab = pickle.load(f)  # trusted teacher-supplied tokenizer only
    stoi,itos = vocab['stoi'],vocab['itos']
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    gpt = GPT(GPTConfig(**ckpt['model_args']))
    state = ckpt.get('model', ckpt.get('model_state_dict'))
    if state is None: raise ValueError('Missing checkpoint model weights')
    gpt.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},strict=True)
    gpt.to(device).eval()
    return gpt,stoi,itos


def grade_set(model, stoi,itos,rows,device,max_new_tokens):
    output=[]
    for i,r in enumerate(rows,1):
        prompt=r['prompt']; answer=str(r['answer'])
        if set(prompt)-set(stoi): raise ValueError(f'Unknown characters: {prompt}')
        if '.' in prompt+answer: raise ValueError('Not integer-only data')
        generated,stop=greedy_original_stop(model,prompt,stoi,itos,max_new_tokens,device)
        status,pred=evaluate_answer(generated,answer)
        output.append({'prompt':prompt,'answer':answer,'category':r['category'],
                       'completion':generated,'prediction':pred,'status':status,
                       'stop_reason':stop})
        if i <= 5:
            print(f"{i:03d} {prompt} expected={answer} -> {generated!r} [{status}]", flush=True)
    counts=Counter(row['status'] for row in output)
    by_category={}
    for category in CATEGORIES:
        subset=[r for r in output if r['category']==category]
        c=Counter(row['status'] for row in subset)
        by_category[category]={'n':len(subset),'correct':c['correct'],
                              'accuracy':c['correct']/len(subset) if subset else 0.0,
                              'refusal':c['refusal'],'math_error':c['math_error'],
                              'format_failure':c['format_failure']}
    summary={'n':len(output),'correct':counts['correct'],
             'accuracy':counts['correct']/len(output),'refusal':counts['refusal'],
             'math_error':counts['math_error'],'format_failure':counts['format_failure'],
             'unique_predictions':len({r['prediction'] for r in output if r['prediction'] is not None}),
             'most_common_predictions':Counter(str(r['prediction']) for r in output
                                              if r['prediction'] is not None).most_common(8),
             'stops':dict(Counter(r['stop_reason'] for r in output)),
             'by_category':by_category}
    return output,summary


def score_batch(model,stoi, batch,device):
    """Teacher-forced pair scores, exact same targets/mask as Notebook Step 3.

    Returns full-sequence mean logP, completion-only sum/mean logP, and margins.
    Both sequences have the same length and same prefix, so without dropout
    full_mean_margin * (L-1) ~= completion_sum_margin (up to floating point).
    """
    texts=[]; starts=[]; pair_lengths=[]
    for pair,meta in batch:
        p=meta['prompt']
        if not pair['positive'].startswith(p) or not pair['negative'].startswith(p):
            raise AssertionError('Pair prompt mismatch')
        pos,neg=pair['positive'],pair['negative']
        if len(pos)!=len(neg): raise AssertionError('Different pair lengths')
        for s in (pos,neg):
            if set(s)-set(stoi): raise ValueError('Out-of-vocabulary text')
            if len(s)>model.config.block_size: raise ValueError('Sequence too long')
            texts.append([stoi[c] for c in s])
            starts.append(len(p))
        pair_lengths.append(len(pos))
    max_len=max(map(len,texts))
    tokens=torch.zeros((len(texts),max_len),dtype=torch.long,device=device)
    for i,s in enumerate(texts):
        tokens[i,:len(s)] = torch.tensor(s,dtype=torch.long,device=device)
    logits,_=model(tokens[:,:-1],full_seq=True)
    targets=tokens[:,1:]
    logp=F.log_softmax(logits.float(),dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    idx=torch.arange(1,max_len,device=device).unsqueeze(0)
    lengths=torch.tensor([len(x) for x in texts],device=device).unsqueeze(1)
    start=torch.tensor(starts,device=device).unsqueeze(1)
    valid=(idx<lengths)&(targets!=0)  # teacher's ignore_index=0
    prefix=valid&(idx<start)
    completion=valid&(idx>=start)
    assert torch.all(valid.sum(dim=1)>0) and torch.all(completion.sum(dim=1)>0)
    whole_sum=(logp*valid).sum(dim=1)
    whole_mean=whole_sum/valid.sum(dim=1)
    suffix_sum=(logp*completion).sum(dim=1)
    suffix_mean=suffix_sum/completion.sum(dim=1)
    prefix_sum=(logp*prefix).sum(dim=1)
    # Scoring differences should not be polluted by prompt-token likelihoods.
    if (prefix_sum[0::2]-prefix_sum[1::2]).abs().max().item() > 1e-3:
        raise AssertionError('Identical prefixes were assigned different scores')
    result=[]
    for j in range(len(batch)):
        a,b=2*j,2*j+1
        score={'full_mean_positive':float(whole_mean[a]),
               'full_mean_negative':float(whole_mean[b]),
               'completion_sum_positive':float(suffix_sum[a]),
               'completion_sum_negative':float(suffix_sum[b]),
               'completion_mean_positive':float(suffix_mean[a]),
               'completion_mean_negative':float(suffix_mean[b]),
               'full_mean_margin':float(whole_mean[a]-whole_mean[b]),
               'completion_sum_margin':float(suffix_sum[a]-suffix_sum[b]),
               'completion_mean_margin':float(suffix_mean[a]-suffix_mean[b]),
               'total_supervised_tokens':int(valid[a].sum()),
               'completion_tokens':int(completion[a].sum())}
        # Equal-length pairs: preference ranking must be the same under sum or mean.
        # True for exact arithmetic; float32 small differences near 0 are tolerated.
        if abs(score['full_mean_margin'] * score['total_supervised_tokens'] -
               score['completion_sum_margin']) > 0.006:
            raise AssertionError('Unexpected full/suffix margin mismatch')
        result.append(score)
    return result


def score_preferences(model,stoi,pairs,meta,selection,device,batch_size):
    results=[]
    with torch.inference_mode():
        for i in range(0,len(selection),batch_size):
            indices=selection[i:i+batch_size]
            items=[(pairs[j],meta[j]) for j in indices]
            results.extend(score_batch(model,stoi,items,device))
    rows=[]
    for idx,score in zip(selection,results):
        rows.append({'prompt':meta[idx]['prompt'], 'category':meta[idx]['category'],
                     'correct_answer':meta[idx]['correct_answer'],
                     'wrong_answer':meta[idx]['wrong_answer'],**score})
    if len(rows)!=len(selection): raise AssertionError('Incomplete probability scoring')
    margins=[r['full_mean_margin'] for r in rows]
    msum=[r['completion_sum_margin'] for r in rows]
    def stats(vals):
        return {'mean':statistics.fmean(vals),'median':statistics.median(vals),
                'positive':sum(x>1e-6 for x in vals),
                'negative':sum(x< -1e-6 for x in vals),
                'ties':sum(abs(x)<=1e-6 for x in vals),
                'min':min(vals),'max':max(vals)}
    summary={'n':len(rows),'full_mean_margin':stats(margins),
             'completion_sum_margin':stats(msum),
             'pairwise_preference_accuracy':sum(x>1e-6 for x in margins)/len(margins),
             'mean_chosen_full_logprob':statistics.fmean(r['full_mean_positive'] for r in rows),
             'mean_rejected_full_logprob':statistics.fmean(r['full_mean_negative'] for r in rows),
             'mean_chosen_completion_logprob':statistics.fmean(r['completion_sum_positive'] for r in rows),
             'mean_rejected_completion_logprob':statistics.fmean(r['completion_sum_negative'] for r in rows)}
    return rows,summary


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-new-tokens',type=int,default=80)
    parser.add_argument('--out-dir',type=Path,default=OUT)
    args=parser.parse_args()
    if args.batch_size<1 or args.max_new_tokens<1:parser.error('Positive batch size and token budget required')
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable; use Slurm GPU allocation')
    path=lambda p:p if p.is_absolute() else ROOT/p
    out=path(args.out_dir).resolve()
    for protected in (TRAIN_ROOT, ROOT/'dpo', ROOT/'sft'):
        if out==protected:parser.error('Output directory is protected')
    if not out.is_relative_to(ROOT/'results'):
        parser.error('--out-dir must be inside repository results/')
    if out.exists() and any(out.iterdir()):
        parser.error(f'Refusing to overwrite nonempty diagnostic folder: {out}')
    pairs,meta,val,man=load_inputs()
    selection=sample_stratified(meta,args.seed)
    selected=[{'prompt':meta[i]['prompt'],'answer':meta[i]['correct_answer'],
               'category':meta[i]['category']} for i in selection]
    if set(m['prompt'] for m in selected)&{r['prompt'] for r in val}:
        raise AssertionError('Training/validation overlap')
    gpt,stoi,itos=load_model(path(args.checkpoint),args.device)
    print('=== Round 1C DIAGNOSIS (read only) ===',flush=True)
    print('Checkpoint:',path(args.checkpoint),flush=True)
    print('Train selection: 256 stratified SEEN prompts | Validation: 256 UNSEEN prompts',flush=True)
    print('=== Generating on SEEN training questions ===',flush=True)
    train_out,train_summary=grade_set(gpt,stoi,itos,selected,args.device,args.max_new_tokens)
    print('TRAIN accuracy:',train_summary['correct'],'/256 =',f"{train_summary['accuracy']:.2%}",flush=True)
    print('=== Generating on UNSEEN validation questions ===',flush=True)
    val_out,val_summary=grade_set(gpt,stoi,itos,val,args.device,args.max_new_tokens)
    print('VAL accuracy:',val_summary['correct'],'/256 =',f"{val_summary['accuracy']:.2%}",flush=True)
    print('=== Teacher-forced positive vs incorrect-negative likelihoods ===',flush=True)
    margin_rows,margin_summary=score_preferences(gpt,stoi,pairs,meta,selection,args.device,args.batch_size)
    print('Preference pair wins:',margin_summary['full_mean_margin']['positive'],'/256',flush=True)
    print('Mean full-seq margin:',round(margin_summary['full_mean_margin']['mean'],6),flush=True)
    print('Mean completion-sum margin:',round(margin_summary['completion_sum_margin']['mean'],6),flush=True)
    by_prompt={r['prompt']:r for r in margin_rows}
    for row in train_out:
        row['full_mean_margin']=by_prompt[row['prompt']]['full_mean_margin']
        row['completion_sum_margin']=by_prompt[row['prompt']]['completion_sum_margin']
    summary={'checkpoint':str(path(args.checkpoint)),
             'training_data_sha256':man['file_sha256']['train'],
             'sample_seed':args.seed,'train_n':len(selected),'val_n':len(val),
             'note':'No training, held-out test not accessed. Greedy original-stop; eval mode for preference scoring.',
             'seen_train':train_summary,'unseen_validation':val_summary,
             'train_pairwise_scoring':margin_summary}
    write_json(out/'diagnosis_summary.json',summary)
    write_jsonl(out/'train_predictions_256.jsonl',train_out)
    write_jsonl(out/'validation_predictions_256.jsonl',val_out)
    write_jsonl(out/'train_pair_margins_256.jsonl',margin_rows)
    write_json(out/'train_sample_256.json',selected)
    print('=== DIAGNOSIS COMPLETE ===',flush=True)
    print('Summary:',out/'diagnosis_summary.json',flush=True)


if __name__=='__main__':
    main()
