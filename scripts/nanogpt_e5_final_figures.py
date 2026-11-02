#!/usr/bin/env python3
"""NanoGPT-Math: two publication-ready GitHub figures from ONE submitted model.

Source of truth: dpo/dpo.pt (the selected Round9 Random E5 export).
This script verifies its weights against the archived Random E5 epoch when available,
evaluates fixed *development* sets only, saves per-prompt predictions and category
statistics, and draws both figures from the SAME cached evaluation.

    python scripts/nanogpt_e5_final_figures.py --repo . --checkpoint dpo/dpo.pt --device cuda

For repeat plotting without running inference:
    python scripts/nanogpt_e5_final_figures.py --repo . --checkpoint dpo/dpo.pt --plot-only

The independent sealed test is NEVER accessed. Uses the notebook's Step 8
strict answer-first integer grader, greedy decoding, 80-token maximum,
period / newline stopping. Requires torch + matplotlib + numpy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

STAGE = "Round9 Random E5 (submitted dpo.pt)"
ARCHIVED = Path('dpo/experiments/round9_random_multi_e3_e6/epoch_e05.pt')
VAL = Path('dpo/data/round3_structured_full/val.json')
TEACHER_CORE = Path('dpo/benchmarks/teacher_style_400_v1/questions.json')
TEACHER_EXT = Path('dpo/benchmarks/teacher_style_extension_280_v1/questions.json')

KIND_ORDER = [
    'add', 'subtract', 'multiply', 'divide',
    'a_minus_x','x_minus_a','x_plus_a','a_plus_x',
    'x_times_a','a_times_x','x_over_a','a_over_x',
]
KIND_LABELS = {
    'add': 'Addition', 'subtract': 'Subtraction',
    'multiply': 'Multiplication', 'divide': 'Division',
    'a_minus_x': 'a - x = b', 'x_minus_a': 'x - a = b',
    'x_plus_a': 'x + a = b', 'a_plus_x': 'a + x = b',
    'x_times_a': 'x × a = b', 'a_times_x': 'a × x = b',
    'x_over_a': 'x / a = b', 'a_over_x': 'a / x = b',
}
N_EXPECT = {k: (80 if k in {'add','multiply','divide','a_minus_x','x_times_a'} else 40) for k in KIND_ORDER}
RE = re.compile(r'^\s*The answer is\s+([+-]?\d+)(?=\s|[.]|$)', re.I)

BG = '#F8FAFC'
INK = '#14243B'
SUB = '#64748B'
BLUE = '#285DD5'
TEAL = '#0F9183'
AMBER = '#EDA74C'
GRID = '#E6EAF1'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoint_weights(payload):
    state = payload.get('model_state_dict')
    if state is None:
        state = payload.get('model')
    if state is None:
        raise ValueError('Checkpoint has neither model_state_dict nor model.')
    return {k.removeprefix('_orig_mod.'): v for k, v in state.items()}


def checked_data(root):
    val_path = root / VAL
    val_manifest = read_json(root / 'dpo/data/round3_structured_full/manifest.json')
    if digest(val_path) != val_manifest['sha256']['val']:
        raise ValueError('Val768 SHA256 differs from frozen manifest.')
    val = read_json(val_path)
    if len(val) != 768:
        raise ValueError('Expected exactly 768 Val records.')

    allteacher = []
    for relative, expected in ((TEACHER_CORE, 400),(TEACHER_EXT, 280)):
        path = root / relative
        manifest = read_json(path.parent / 'manifest.json')
        if digest(path) != manifest['questions_sha256']:
            raise ValueError(f'Teacher benchmark checksum mismatch: {path}')
        records = read_json(path)
        if len(records) != expected:
            raise ValueError(f'Incorrect teacher split length: {path}')
        allteacher.extend(records)
    if len({r['prompt'] for r in allteacher}) != 680:
        raise ValueError('Teacher680 must have 680 distinct prompts.')
    if Counter(r['category'] for r in allteacher) != Counter(N_EXPECT):
        raise ValueError('Teacher680 category distribution changed.')
    for split, rows in [('Val768',val),('Teacher680',allteacher)]:
        for rec in rows:
            if not all(k in rec for k in ('prompt','answer','category')):
                raise ValueError(f'{split} missing required record keys')
    return {'Val768': val, 'Teacher680': allteacher}, {'Val768':digest(val_path),
               'Teacher400':digest(root/TEACHER_CORE),'Teacher280':digest(root/TEACHER_EXT)}


def validate_selected_model(root, checkpoint, torch, *, allow_unverified=False):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint_weights(payload)
    archived = root / ARCHIVED
    verify = {'archived_path':str(ARCHIVED), 'archived_exists':archived.exists(),
              'model_weights_equal_archived_e5':None}
    if archived.is_file():
        other = torch.load(archived, map_location='cpu', weights_only=True)
        base = checkpoint_weights(other)
        is_identical = set(state)==set(base) and all(torch.equal(state[k],base[k]) for k in state)
        verify.update({'archived_sha256':digest(archived),
                       'model_weights_equal_archived_e5':is_identical})
        if not is_identical:
            raise RuntimeError(f'{checkpoint} weights do NOT equal archived Random E5. '
                               'Do not combine their evaluation results.')
        print('PASS: final submission weights match archived Random E5 bitwise.')
    elif not allow_unverified:
        raise FileNotFoundError(f'Cannot verify Random E5 identity; archived checkpoint missing: {archived}. '
                                'Use --allow-unverified only if you have independently established provenance.')
    else:
        print('WARNING: archived E5 not found; model identity is not independently verified.')
    return payload, state, verify


def run_eval(root, checkpoint, device, allow_unverified=False):
    import torch
    from model import GPT, GPTConfig
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; run inside a Slurm GPU allocation.')
    datasets, data_hashes = checked_data(root)
    payload, state, verification = validate_selected_model(root,checkpoint,torch,
                                                            allow_unverified=allow_unverified)
    conf = GPTConfig(**payload['model_args'])
    model = GPT(conf)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    meta_file = root/'sft/meta.pkl'
    with meta_file.open('rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    assert len(stoi)==len(itos)==74 and stoi['\n']==0 and stoi['.']==7
    report={'model':STAGE,'checkpoint':str(checkpoint.relative_to(root)),
            'checkpoint_sha256':digest(checkpoint), 'model_verification':verification,
            'model_args':payload['model_args'], 'data_hashes':data_hashes,
            'protocol':{'greedy':True,'max_new_tokens':80,'stop':['period','newline'],
                        'grading':'strict answer-first integer exact match'},
            'sets':{},'created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'development evaluation ONLY; no independent sealed test'}
    predictions={}
    for split,records in datasets.items():
        scored=[]
        for i,rec in enumerate(records,1):
            tokens=[stoi[c] for c in rec['prompt']]
            if len(tokens)>model.config.block_size:
                raise ValueError(f'{split} prompt length exceeds block_size.')
            gen=[]
            reason='max_new_tokens'
            with torch.inference_mode():
                for _ in range(80):
                    x = torch.tensor([tokens[-model.config.block_size:]],dtype=torch.long,device=device)
                    logits, _ = model(x)
                    t=int(logits[0,-1].argmax().item())
                    if t==0:
                        reason='newline'
                        break
                    tokens.append(t)
                    gen.append(t)
                    if t==7:
                        reason='period'
                        break
            completion=''.join(itos[t] for t in gen)
            match=RE.match(completion)
            status=('format_failure' if not match else
                    'correct' if int(match.group(1))==int(rec['answer']) else 'math_error')
            scored.append({'prompt':rec['prompt'],'answer':str(rec['answer']),
                           'category':rec['category'],'completion':completion,'status':status,
                           'stop_reason':reason})
            if i % 100 == 0 or i == len(records):
                print(f'{split}: {i}/{len(records)} complete',flush=True)
        counts=Counter(r['status'] for r in scored)
        by_category={}
        for category in sorted({r['category'] for r in scored}):
            subset=[r for r in scored if r['category']==category]
            c=Counter(r['status'] for r in subset)
            by_category[category]={'n':len(subset),'correct':c['correct'],
                 'math_error':c['math_error'],'format_failure':c['format_failure']}
        metrics={'n':len(scored),'correct':counts['correct'],
                 'accuracy':counts['correct']/len(scored),
                 'math_error':counts['math_error'],
                 'format_failure':counts['format_failure'],
                 'by_category':by_category}
        assert counts['correct']+counts['math_error']+counts['format_failure']==len(scored)
        report['sets'][split]=metrics
        predictions[split]=scored
        print(f'{split}: {metrics["correct"]}/{metrics["n"]} = {metrics["accuracy"]:.2%}')
    return report,predictions


def import_notebook_eval(root, checkpoint, *, allow_unverified=False):
    """Reuse the exact same evaluation saved by the notebook: NO GPU run needed.

    The final_reproduction_manifest records scores for the *submitted* checkpoint,
    not necessarily the newly replay-trained one. We confirm that checkpoint's
    provenance and validate every saved prediction against recorded metrics.
    """
    import torch
    base=root/'dpo/experiments/final_notebook_e5_reproduction'
    manifest_path=base/'final_reproduction_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Final notebook manifest not found; run a full notebook evaluation, '
                                'or omit --from-notebook to infer the two development splits.')
    manifest=read_json(manifest_path)
    source=root/ARCHIVED if manifest.get('export_checkpoint_origin')=='selected_archived_E5' else None
    if source is None:
        stage=manifest['stage_history']['round9_random_e3_e5_from_e2']['checkpoint']
        source=root/stage
    if not source.exists() or digest(source)!=manifest['export_source_sha256']:
        raise ValueError('Final manifest export source not present or SHA256 mismatch.')
    payload, current, verification=validate_selected_model(root,checkpoint,torch,
                                                             allow_unverified=allow_unverified)
    source_payload=torch.load(source,map_location='cpu',weights_only=True)
    original=checkpoint_weights(source_payload)
    if set(current)!=set(original) or any(not torch.equal(current[k],original[k]) for k in current):
        raise ValueError('Submitted dpo.pt weights are NOT equal to notebook-exported source.')
    set_dir=base/('submitted_e5_development' if manifest['export_checkpoint_origin']=='selected_archived_E5'
                  else 'development_e5')
    datasets,data_hashes=checked_data(root)
    report={'model':STAGE,'checkpoint':str(checkpoint.relative_to(root)),
            'checkpoint_sha256':digest(checkpoint),'model_verification':verification,
            'model_args':payload['model_args'],'data_hashes':data_hashes,
            'protocol':{'greedy':True,'max_new_tokens':80,'stop':['period','newline'],
                        'grading':'strict answer-first integer exact match'},
            'sets':{},'created_utc':manifest.get('sources',{}).get('utc','archived notebook evaluation'),
            'status':'development evaluation ONLY; no independent sealed test',
            'source':'verified final notebook evaluation predictions'}
    for split,records in datasets.items():
        key='val' if split=='Val768' else 'teacher680'
        pred_path=set_dir/f'{key}_predictions.json'
        metric_path=set_dir/f'{key}_metrics.json'
        if not pred_path.exists() or not metric_path.exists():
            raise FileNotFoundError(f'Missing {split} notebook saved evaluation under {set_dir}')
        preds=read_json(pred_path)
        metrics=read_json(metric_path)
        if len(records)!=len(preds):
            raise ValueError(f'{split} saved predictions length changed')
        for row,ref in zip(preds,records):
            if (row['prompt'],str(row['answer']),row['category'])!=(ref['prompt'],str(ref['answer']),ref['category']):
                raise ValueError(f'{split} prediction order/content is not the frozen dataset')
            if row['status'] not in {'correct','math_error','format_failure'}:
                raise ValueError(f'Unexpected score category in {split}')
        counts=Counter(row['status'] for row in preds)
        bycat={}
        for kind in sorted({row['category'] for row in preds}):
            subset=[row for row in preds if row['category']==kind]
            vals=Counter(row['status'] for row in subset)
            bycat[kind]={'n':len(subset),'correct':vals['correct'],
                         'math_error':vals['math_error'],'format_failure':vals['format_failure']}
            old=metrics['by_category'][kind]
            if (old['n'],old['correct'])!=(len(subset),vals['correct']):
                raise ValueError(f'{split} category counts do not match notebook metrics')
        exact={'n':len(preds),'correct':counts['correct'],
               'accuracy':counts['correct']/len(preds),
               'math_error':counts['math_error'],
               'format_failure':counts['format_failure'],
               'by_category':bycat}
        if (exact['n'],exact['correct'],exact['format_failure']) != (metrics['n'],metrics['correct'],metrics['format_failure']):
            raise ValueError(f'{split} predictions do not match their saved notebook metrics')
        anchor=manifest['measured_dev_scores_of_submitted_checkpoint'][key]
        if (anchor['n'],anchor['correct'])!=(exact['n'],exact['correct']):
            raise ValueError(f'{split} Notebook final manifest disagrees with evaluation')
        report['sets'][split]=exact
        print(f'REUSED verified notebook evaluation: {split} {exact["correct"]}/{exact["n"]}')
    return report


def set_visual_theme():
    import matplotlib as mpl
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'figure.facecolor':BG,'savefig.facecolor':BG,
                         'axes.facecolor':BG,'axes.edgecolor':GRID,
                         'text.color':INK,'xtick.color':SUB,'ytick.color':SUB})


def save_image(fig, out):
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=200,bbox_inches='tight',pad_inches=.18,facecolor=BG)
    fig.savefig(out.with_suffix('.svg'),bbox_inches='tight',pad_inches=.18,facecolor=BG)
    print('CREATED',out)


def figure1(data, out):
    """Same Random E5 checkpoint: fact-disjoint dev and seen-heavy retention."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    set_visual_theme()
    fig=plt.figure(figsize=(12.4,6.1))
    fig.text(.055,.935,'01 / FINAL MODEL PERFORMANCE',color=TEAL,weight='bold',size=11)
    fig.text(.055,.862,'One selected checkpoint. Two complementary evaluations.',weight='bold',size=21)
    fig.text(.055,.805,'Round9 Random E5  |  8.84M parameters  |  Greedy decoding  |  Exact-answer accuracy',color=SUB,size=11)
    metrics=[('Val768','Fact-disjoint development','#285DD5',.055),
             ('Teacher680','Seen-heavy retention','#0F9183',.53)]
    for key,subtitle,color,left in metrics:
        value=data['sets'][key]
        ax=fig.add_axes([left,.28,.415,.39]);ax.axis('off')
        ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=0.01,rounding_size=.06',
                                    facecolor='white',edgecolor=GRID,linewidth=1.5,transform=ax.transAxes))
        ax.text(.06,.82,key,fontweight='bold',fontsize=14,color=INK,transform=ax.transAxes)
        ax.text(.06,.68,subtitle,fontsize=10,color=SUB,transform=ax.transAxes)
        ax.text(.06,.35,f'{value["accuracy"]*100:.2f}%',fontsize=32,fontweight='bold',color=color,transform=ax.transAxes)
        ax.text(.06,.16,f'{value["correct"]} / {value["n"]} correct',fontsize=11,color=SUB,transform=ax.transAxes)
        ax.add_patch(FancyBboxPatch((.06,.08),.87,.025,boxstyle='round,pad=0,rounding_size=.01',
                                    facecolor=GRID,edgecolor='none',transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((.06,.08),.87*value['accuracy'],.025,boxstyle='round,pad=0,rounding_size=.01',
                                    facecolor=color,edgecolor='none',transform=ax.transAxes))
    fig.text(.055,.16,'Interpretation',fontsize=11,fontweight='bold',color=INK)
    fig.text(.055,.105,'These benchmarks have different roles and difficulties; their accuracy values are not directly comparable.',fontsize=10.4,color=SUB)
    fig.text(.055,.057,'Development-selected model; independent sealed final test was not accessed. Figure 4 uses the same checkpoint.',fontsize=10.2,color=SUB)
    save_image(fig,out);plt.close(fig)


def figure4(data, out):
    """Single E5 checkpoint: by-skill correct vs math-error vs format-failure."""
    import numpy as np
    import matplotlib.pyplot as plt
    set_visual_theme()
    rows=data['sets']['Teacher680']['by_category']
    if set(rows)!=set(KIND_ORDER):
        raise ValueError(f'Expected exactly 12 skill categories: {set(rows)^set(KIND_ORDER)}')
    fig=plt.figure(figsize=(12.6,9.7))
    fig.text(.055,.953,'04 / SKILL-WISE ERROR PROFILE',color=TEAL,weight='bold',size=11)
    fig.text(.055,.911,'Which math skills remain difficult for Random E5?',fontsize=21,weight='bold')
    fig.text(.055,.873,'One submitted checkpoint  |  Teacher680 development suite  |  12 math templates',color=SUB,fontsize=11)
    ax=fig.add_axes([.20,.18,.73,.59])
    ys=np.arange(len(KIND_ORDER))[::-1]
    correct=np.array([100*rows[k]['correct']/rows[k]['n'] for k in KIND_ORDER])
    matherr=np.array([100*rows[k]['math_error']/rows[k]['n'] for k in KIND_ORDER])
    fmt=np.array([100*rows[k]['format_failure']/rows[k]['n'] for k in KIND_ORDER])
    ax.barh(ys,correct,color=TEAL,height=.65,label='Correct')
    ax.barh(ys,matherr,left=correct,color=AMBER,height=.65,label='Wrong answer')
    ax.barh(ys,fmt,left=correct+matherr,color='#C7D1DF',height=.65,label='Format failure')
    ax.set_xlim(0,106)
    ax.set_yticks(ys,labels=[KIND_LABELS[k] for k in KIND_ORDER])
    ax.set_xticks([0,25,50,75,100],labels=['0%','25%','50%','75%','100%'])
    ax.tick_params(axis='y',length=0,pad=14,labelsize=11)
    ax.grid(axis='x',color=GRID,zorder=0,linewidth=.9)
    ax.set_axisbelow(True)
    for edge in ax.spines.values():edge.set_visible(False)
    for y,k,p in zip(ys,KIND_ORDER,correct):
        n=rows[k]['n']
        ax.text(103,y,f'{rows[k]["correct"]}/{n}',ha='left',va='center',fontsize=9.5,color=INK)
        if p>=15:
            ax.text(max(p/2,4),y,f'{p:.0f}%',ha='center',va='center',color='white',fontweight='bold',fontsize=10)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=3,frameon=False,fontsize=11,
              handlelength=1.8,columnspacing=2.4)
    fig.text(.055,.095,'Five core categories have 80 questions each; seven extension categories have 40 each.',fontsize=10,color=SUB)
    fig.text(.055,.058,'Teacher680 contains previously encountered facts: this measures competence / retention, not independent generalization.',fontsize=9.7,color=SUB)
    save_image(fig,out);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('.'))
    parser.add_argument('--checkpoint',type=Path,default=Path('dpo/dpo.pt'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--plot-only',action='store_true',help='Read cached evaluation but still verify submitted checkpoint SHA.')
    parser.add_argument('--from-notebook',action='store_true',help='Reuse the final notebook saved model evaluation; no GPU required.')
    parser.add_argument('--allow-unverified',action='store_true',help='Archived E5 missing; proceed with explicit warning.')
    parser.add_argument('--preflight',action='store_true',help='Check local files and model identity on CPU, without running inference.')
    args=parser.parse_args()
    root=args.repo.expanduser().resolve()
    cp=args.checkpoint if args.checkpoint.is_absolute() else root/args.checkpoint
    if not cp.is_file():
        raise FileNotFoundError(f'Submitted-model file not found: {cp}. Export dpo/dpo.pt first.')
    out=root/'results/portfolio_e5'
    report_file=out/'e5_development_metrics.json'
    if sum((args.plot_only,args.from_notebook,args.preflight))>1:
        parser.error('Choose only one of --plot-only, --from-notebook, or --preflight.')
    if args.preflight:
        import torch
        datasets,data_hashes=checked_data(root)
        _,state,verification=validate_selected_model(root,cp,torch,allow_unverified=args.allow_unverified)
        print('PREFLIGHT PASS: checkpoint=',cp)
        print('PREFLIGHT PASS: tensors=',len(state))
        print('PREFLIGHT PASS: dataset counts=',{k:len(v) for k,v in datasets.items()})
        print('PREFLIGHT PASS: benchmark SHA256=',data_hashes)
        print('PREFLIGHT PASS: archived E5 identity=',verification['model_weights_equal_archived_e5'])
        return
    if args.plot_only:
        data=read_json(report_file)
        if data['checkpoint_sha256']!=digest(cp):
            raise ValueError('Cached results belong to DIFFERENT weights. Do not reuse.')
        current_data_hashes=checked_data(root)[1]
        if current_data_hashes!=data['data_hashes']:
            raise ValueError('Cached results belong to different benchmark data.')
    else:
        if args.from_notebook:
            data=import_notebook_eval(root,cp,allow_unverified=args.allow_unverified)
            predictions={}
        else:
            sys.path.insert(0,str(root))
            data,predictions=run_eval(root,cp,args.device,allow_unverified=args.allow_unverified)
        out.mkdir(parents=True,exist_ok=True)
        if report_file.exists():
            old=read_json(report_file)
            # Preserve a verified existing evaluation; never quietly overwrite its evidence.
            keys=['checkpoint_sha256','data_hashes','sets','protocol']
            if any(old[k]!=data[k] for k in keys):
                raise RuntimeError(f'Existing E5 metrics differ: {report_file}. Audit before overwriting.')
            print('VERIFIED existing evaluation metrics')
        else:
            report_file.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for split,rows in predictions.items():
            dest=out/f'{split.lower()}_predictions.json'
            if dest.exists() and read_json(dest)!=rows:
                raise RuntimeError(f'Existing predictions differ: {dest}')
            if not dest.exists():
                dest.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    figure1(data,root/'assets/figure1_final_e5.png')
    figure4(data,root/'assets/figure4_e5_skill_errors.png')
    print('FINAL PLOT SOURCE:', data['model'], data['checkpoint'], data['checkpoint_sha256'])
    print('No sealed test was accessed.')

if __name__=='__main__':
    main()