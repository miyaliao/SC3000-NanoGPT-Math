#!/usr/bin/env python3
"""Run instructor Notebook Step 2-7 on isolated Round 1B integer-only data.

Safe: dpo/dpo.pt and dpo/pos_neg_pairs.json remain unchanged.
Round 1B checkpoint is dpo/experiments/round1b_integer/dpo.pt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR))
from pipeline_smoke import execute_step,notebook_steps  # noqa: E402


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:p.error('Invalid training hyperparameters')
    root=ROOT/'dpo/experiments/round1b_integer'
    manifest_file=root/'round1b_manifest.json'
    train_file=root/'pos_neg_pairs.json'
    ckpt_file=root/'dpo.pt'
    if ckpt_file.exists() and not args.overwrite:p.error('Round 1B checkpoint exists; use --overwrite explicitly')
    if not manifest_file.is_file() or not train_file.is_file():p.error('Run scripts/generate_round1b.py first')
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    sha=hashlib.sha256(train_file.read_bytes()).hexdigest()
    if sha!=manifest['file_sha256']['train'] or manifest['training_size']!=20480:
        p.error('Training dataset differs from manifest')
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; use Slurm allocation')
    os.chdir(ROOT)
    steps=notebook_steps();env={'__name__':'__round1b__'}
    print('=== Executing actual Notebook Step 2 ===',flush=True)
    execute_step(2,steps,env)
    env['dpo_dir']=root
    env['device']=args.device
    env['epochs']=args.epochs
    env['batch_size']=args.batch_size
    print('=== Executing actual Notebook Steps 3-7 ===',flush=True)
    for step in range(3,8):
        execute_step(step,steps,env)
        if step==5:
            if env['data_path'].resolve()!=train_file.resolve() or len(env['lines'])!=20480:
                raise AssertionError('Notebook Step5 loaded incorrect data')
        if step==6 and env['steps_per_epoch'] != 20480 // args.batch_size:
            raise AssertionError('Unexpected step count')
    if not ckpt_file.is_file():raise RuntimeError('Round1B checkpoint not saved')
    import math
    if not all(math.isfinite(x) for x in env['training_loss']):raise RuntimeError('Nonfinite training loss')
    summary={'stage':'round1b-integer-refusal','init_checkpoint':'sft/gpt.pt',
             'checkpoint':str(ckpt_file),'data_sha256':sha,'epoch_losses':env['training_loss'],
             'epochs':args.epochs,'batch_size':args.batch_size,
             'steps_per_epoch':env['steps_per_epoch'],'total_steps':env['steps_per_epoch']*args.epochs,
             'note':'Executed actual Notebook Steps 2-7 with instructor loss; no extra SFT.'}
    dest=ROOT/'results/round1b_training_summary.json';dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('=== ROUND 1B TRAINING COMPLETED ===',flush=True)
    print('Summary:',dest,'Checkpoint:',ckpt_file,flush=True)


if __name__=='__main__':
    main()
