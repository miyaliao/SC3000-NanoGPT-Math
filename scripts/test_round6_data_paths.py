#!/usr/bin/env python3
"""CPU-only guard that Round6 check-only always reads all FOUR 100k train files.

Regression: verify_data() previously overwrote the path variable while checking
fixed validation files and accidentally returned train_probe_256.json.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_round6_from_sft import verify_data,DATA,ARMS,read

def main():
    for arm in ARMS:
        manifest,pair_path=verify_data(arm)
        expected=DATA/f"{arm}_pairs.json"
        if pair_path!=expected:
            raise AssertionError(f"{arm}: expected {expected}, got {pair_path}")
        records=read(pair_path)
        n=manifest["train_rows_per_arm"]
        if len(records)!=n or n!=100000:
            raise AssertionError(f"{arm}: expected 100000 train rows, got {len(records)}")
        if not all(isinstance(r,dict) and isinstance(r.get("positive"),str)
                   and isinstance(r.get("negative"),str)
                   for r in (records[0],records[-1])):
            raise AssertionError(f"{arm}: corrupt first/last training row")
        print(f"PASS {arm:21} training source={pair_path.name}, rows={len(records)}")
    print("PASS: no evaluation split path was returned as training data")

if __name__=="__main__":
    main()
