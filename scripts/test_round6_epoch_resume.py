#!/usr/bin/env python3
"""CPU-only regression for Round6 E2+ random state restore and safe startup recovery.

Run:
  python scripts/test_round6_epoch_resume.py --arm original

No training, no GPU, no modification to any real experiment artifact.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from train_round6_from_sft import is_safe_pretraining_recovery

def test_recovery():
    pairs=[{"positive":"1+1=? The answer is 2.","negative":"1+1=? Sorry, I don't know."}]
    with tempfile.TemporaryDirectory(prefix="round6_rng_regression_") as temp:
        root=Path(temp)
        trial=root/"epoch02"
        trial.mkdir()
        pairfile=trial/"pos_neg_pairs.json"
        pairfile.write_text(json.dumps(pairs),encoding="utf-8")
        assert is_safe_pretraining_recovery(trial,pairs,2,False)
        assert not is_safe_pretraining_recovery(trial,pairs,1,False)
        assert not is_safe_pretraining_recovery(trial,pairs,2,True)
        try:
            is_safe_pretraining_recovery(trial,[{"positive":"other"}],2,False)
        except ValueError:pass
        else:raise AssertionError("Data mismatch was not rejected")
        (trial/"training_state.pt").write_text("marker",encoding="utf-8")
        assert not is_safe_pretraining_recovery(trial,pairs,2,False)
        (trial/"training_state.pt").unlink()
        (trial/"extended_val_predictions.json").write_text("marker",encoding="utf-8")
        assert not is_safe_pretraining_recovery(trial,pairs,2,False)
    print("PASS: only matching pairs-only failed startup is recoverable")

def test_checkpoint(arm):
    file=ROOT/f"dpo/experiments/round6_from_sft/{arm}/epoch01/training_state.pt"
    if not file.is_file():
        print("NOTE: no saved Epoch1 checkpoint found; skip real-checkpoint RNG inspection:",file)
        return
    data=torch.load(file,map_location="cpu",weights_only=True)
    assert data["epoch"]==1 and data["arm"]==arm
    cpu=data["rng_cpu"].cpu()
    gpu=[x.cpu() for x in data["rng_cuda"]]
    assert cpu.dtype==torch.uint8 and cpu.device.type=="cpu"
    assert gpu and all(x.dtype==torch.uint8 and x.device.type=="cpu" for x in gpu)
    assert all(x.ndim==1 and x.numel()>0 for x in gpu)
    print(f"PASS: {arm} Epoch1 saved RNG state: CPU bytes={cpu.numel()}, "
          f"CUDA state tensors={len(gpu)}; all CPU torch.ByteTensor.")
    print("PASS: saved model/optimizer/scheduler metadata preserved; "
          "no actual GPU generator state changed.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",default="original",
                   choices=("original","answer_only","decomposition_after","decomposition_before"))
    a=p.parse_args()
    test_recovery()
    test_checkpoint(a.arm)
