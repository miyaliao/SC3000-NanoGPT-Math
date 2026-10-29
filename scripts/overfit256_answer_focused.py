#!/usr/bin/env python3
"""Isolated 256-example DPO loss ablation: completion preference + answer NLL.

Uses the EXACT same 256 Round 1C pairs selected by overfit256_course.py,
initial sft/gpt.pt, batch size, LR schedule, steps, and held-out integer
validation. Only the loss is changed. All model weights remain untouched
except within the new dpo/experiments/overfit256_answer_focused/ directory.

  python scripts/overfit256_answer_focused.py --check-data-only
  python scripts/overfit256_answer_focused.py --device cuda
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from overfit256_course import (select_rows, run_evaluation, score_preferences, save_json,
                               EXPECTED_SHA, CATEGORY_QUOTA)
from pipeline_smoke import execute_step, notebook_steps

DEFAULT_DEST = ROOT / 'dpo/experiments/overfit256_answer_focused'


def self_test_token_masks(env):
    import torch
    samples = [
        ('79-7=? The answer is 72.', ' The answer is 72.', '72'),
        ('x+55=95,x=? The answer is 40.', ' The answer is 40.', '40'),
        ('5-9=? The answer is -4.', ' The answer is -4.', '-4'),
        ('9*1=? The answer is 9.', ' The answer is 9.', '9'),
    ]
    encoded = []
    for sample, _, _ in samples:
        assert len(sample) + 4 <= env['max_length']
        encoded.append(env['pad_or_truncate'](env['encode'](sample+'\n\n\n\n'),env['max_length']))
    ids = torch.tensor(encoded, dtype=torch.long,device=env['device'])
    comp, num = env['answer_focused_masks'](ids)
    for i,(sample,want_comp,want_num) in enumerate(samples):
        decoded_response = env['decode']([int(t) for t,m in zip(ids[i,1:].tolist(),comp[i].tolist()) if m])
        decoded_number = env['decode']([int(t) for t,m in zip(ids[i,1:].tolist(),num[i].tolist()) if m])
        assert decoded_response==want_comp, (sample,decoded_response,want_comp)
        assert decoded_number==want_num, (sample,decoded_number,want_num)
    print('PASS: causal shifted-token masks isolate completions and numeric answer tokens',flush=True)


def score_answer_focused(env,pairs,batch_size):
    import torch
    gpt=env['gpt']
    gpt.eval()
    margins=[]; chosen_numeric=[]; chosen_completion=[]
    with torch.inference_mode():
        for start in range(0,len(pairs),batch_size):
            batch=pairs[start:start+batch_size]
            encode=env['encode']; pad=env['pad_or_truncate']; L=env['max_length']
            pos=torch.tensor([pad(encode(x['positive']+'\n\n\n\n'),L) for x in batch],
                             device=env['device'],dtype=torch.long)
            neg=torch.tensor([pad(encode(x['negative']+'\n\n\n\n'),L) for x in batch],
                             device=env['device'],dtype=torch.long)
            ps,pm,pnum=env['compute_answer_focused_logprob'](pos)
            ns,_,_=env['compute_answer_focused_logprob'](neg)
            margins.extend((ps-ns).detach().cpu().tolist())
            chosen_numeric.extend(pnum.detach().cpu().tolist())
            chosen_completion.extend(pm.detach().cpu().tolist())
    return {'n':len(margins),'pairwise_wins':sum(x>0 for x in margins),
            'win_rate':sum(x>0 for x in margins)/len(margins),
            'mean_completion_sum_margin':sum(margins)/len(margins),
            'mean_chosen_numeric_logprob':sum(chosen_numeric)/len(chosen_numeric),
            'mean_chosen_completion_logprob':sum(chosen_completion)/len(chosen_completion)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--max-new-tokens',type=int,default=48)
    parser.add_argument('--answer-weight',type=float,default=1.0)
    parser.add_argument('--completion-weight',type=float,default=0.25)
    parser.add_argument('--out-dir',type=Path,default=DEFAULT_DEST)
    parser.add_argument('--check-data-only',action='store_true')
    args=parser.parse_args()
    if (args.epochs<1 or args.batch_size<1 or 256 % args.batch_size
        or args.answer_weight<0 or args.completion_weight<0
        or args.answer_weight+args.completion_weight==0 or args.max_new_tokens<1):
        parser.error('Invalid loss/training parameters')
    pairs,metadata,validation,indices=select_rows(args.seed)
    print('PASS: exact same 256 seen pairs and 256 held-out val as course-loss overfit',flush=True)
    if args.check_data_only:
        assert len(pairs)==256 and len(validation)==256
        print('DATA CHECK PASSED: no GPU and no checkpoint required',flush=True)
        return
    dest=(args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir).resolve()
    exp=(ROOT/'dpo/experiments').resolve()
    if not dest.is_relative_to(exp) or dest==exp or dest.exists():
        parser.error('Choose a NEW output folder below dpo/experiments/; no overwrites allowed')
    import torch
    import torch.nn.functional as F
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; submit the Slurm job')
    os.chdir(ROOT)
    env={'__name__':'__overfit256_answer_focused__'}
    steps=notebook_steps()
    execute_step(2,steps,env)
    env['device']=args.device
    env['epochs']=args.epochs
    env['batch_size']=args.batch_size
    env['loss_mode']='answer_focused'
    env['answer_weight']=args.answer_weight
    env['completion_weight']=args.completion_weight
    # Load the source notebook functions/model/optimizer. Nothing else is reimplemented.
    execute_step(3,steps,env)
    if 'answer_focused_masks' not in env or 'compute_answer_focused_logprob' not in env:
        raise RuntimeError('The GitHub Notebook is missing the answer_focused loss helpers; git pull')
    execute_step(4,steps,env)
    # Only create output after all prerequisites and tokenized masks have passed.
    env['max_length']=max(env['max_length'],max(len(p['positive'])+4 for p in pairs))
    self_test_token_masks(env)
    dest.mkdir(parents=True,exist_ok=False)
    save_json(dest/'pos_neg_pairs.json',pairs)
    save_json(dest/'train_metadata.json',metadata)
    save_json(dest/'selection.json',{'seed':args.seed,'indices':indices,
              'source_sha256':EXPECTED_SHA,'category_quota':CATEGORY_QUOTA})
    env['dpo_dir']=dest
    execute_step(5,steps,env)
    if env['data_path'].resolve()!=(dest/'pos_neg_pairs.json').resolve() or len(env['lines'])!=256:
        raise AssertionError('Notebook loaded wrong dataset')
    execute_step(6,steps,env)
    model=env['gpt'];model.eval()
    initial_new=score_answer_focused(env,pairs,args.batch_size)
    initial_course=score_preferences(env,pairs,args.batch_size)
    print('Initial pairwise wins, answer-focused:',initial_new['pairwise_wins'],flush=True)
    print('Initial pairwise wins, course:',initial_course['pairwise_wins'],flush=True)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda':torch.cuda.manual_seed_all(args.seed)
    model.train()
    losses=[];pref_losses=[];chosen_losses=[]
    for epoch in range(args.epochs):
        running=0.;running_pref=0.;running_chosen=0.
        for neg,pos in env['get_batches'](env['lines'],args.batch_size):
            env['optimizer'].zero_grad(set_to_none=True)
            ps,pm,pnum=env['compute_answer_focused_logprob'](pos)
            ns,_,_=env['compute_answer_focused_logprob'](neg)
            pref=-F.logsigmoid((ps-ns)/env['beta']).mean()
            chosen=(-args.answer_weight*pnum.mean()
                    -args.completion_weight*pm.mean())
            loss=pref+chosen
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('Nonfinite loss at epoch '+str(epoch+1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            env['optimizer'].step();env['scheduler'].step()
            running+=float(loss.item());running_pref+=float(pref.item());running_chosen+=float(chosen.item())
        losses.append(running/env['steps_per_epoch'])
        pref_losses.append(running_pref/env['steps_per_epoch'])
        chosen_losses.append(running_chosen/env['steps_per_epoch'])
        if epoch==0 or (epoch+1)%16==0 or epoch==args.epochs-1:
            print(f"Epoch {epoch+1:3d}/{args.epochs} total={losses[-1]:.5f} pref={pref_losses[-1]:.5f} chosen={chosen_losses[-1]:.5f}",flush=True)
    ckpt=dest/'dpo.pt'
    torch.save({'model_state_dict':model.state_dict(),
                'model_args':env['ckpt']['model_args']},ckpt)
    model.eval()
    train_summary,train_preds=run_evaluation(model,metadata,env['stoi'],env['itos'],args.device,args.max_new_tokens)
    val_summary,val_preds=run_evaluation(model,validation,env['stoi'],env['itos'],args.device,args.max_new_tokens)
    final_new=score_answer_focused(env,pairs,args.batch_size)
    final_course=score_preferences(env,pairs,args.batch_size)
    save_json(dest/'train_predictions.json',train_preds)
    save_json(dest/'val_predictions.json',val_preds)
    summary={'stage':'overfit256_answer_focused','init_checkpoint':'sft/gpt.pt',
      'source_round1c_sha256':EXPECTED_SHA, 'sample_seed':args.seed,
      'training_size':256,'validation_size':256,'epochs':args.epochs,
      'batch_size':args.batch_size,'steps':args.epochs*env['steps_per_epoch'],
      'loss_mode':'answer_focused','beta':env['beta'],
      'answer_weight':args.answer_weight,'completion_weight':args.completion_weight,
      'loss_definition':'completion-logprob-sum preference + chosen numeric mean NLL + weighted completion mean NLL; no reference model',
      'initial_preference_answer_focused':initial_new,'final_preference_answer_focused':final_new,
      'initial_preference_course':initial_course,'final_preference_course':final_course,
      'epoch_total_losses':losses,'epoch_preference_losses':pref_losses,
      'epoch_chosen_losses':chosen_losses,
      'seen_train':train_summary,'unseen_val':val_summary,
      'checkpoint':str(ckpt),'note':'Explicit additional loss ablation; not the original course loss. Same seen 256 as overfit256_course.'}
    save_json(ROOT/'results/overfit256_answer_focused_summary.json',summary)
    print('=== OVERFIT256 ANSWER-FOCUSED LOSS COMPLETE ===',flush=True)
    print(f"TRAIN accuracy: {train_summary['correct']}/256 = {train_summary['accuracy']:.2%}",flush=True)
    print(f"VAL accuracy: {val_summary['correct']}/256 = {val_summary['accuracy']:.2%}",flush=True)
    print(f"Preference pair wins: {final_new['pairwise_wins']}/256 = {final_new['win_rate']:.2%}",flush=True)
    print('Initial pair wins:',initial_new['pairwise_wins'],flush=True)
    print('Saved:',ROOT/'results/overfit256_answer_focused_summary.json',flush=True)
    print('Checkpoint:',ckpt,flush=True)

if __name__=='__main__':
    main()
