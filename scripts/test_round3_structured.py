#!/usr/bin/env python3
"""CPU-only invariants for the Round3 generator.

Run: python scripts/test_round3_structured.py
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_round3_structured as g


def main():
    cases = [
        ("79-7=?", 72, (79, "-", 7), "subtract"),
        ("x+55=95,x=?", 40, (95, "-", 55), "x_plus_a"),
        ("72-x=34,x=?", 38, (72, "-", 34), "a_minus_x"),
        ("72/x=9,x=?", 8, (72, "/", 9), "a_over_x"),
        ("x-9=5,x=?", 14, (5, "+", 9), "x_minus_a"),
        ("x/4=7,x=?", 28, (7, "*", 4), "x_over_a"),
        ("0*12=?", 0, (0, "*", 12), "multiply"),
    ]
    for q, ans, expression, kind in cases:
        row = g.make_record(q, ans, expression, kind)
        assert f"The answer is {ans} because " in row["positive"]
        assert row["negative"] == f"{q} {g.REFUSAL}"

    for bad in [
        ("72/x=9,x=?", 9, (72, "/", 9), "a_over_x"),
        ("72/x=9,x=?", 8, (72, "/", 0), "a_over_x"),
    ]:
        try:
            g.make_record(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid explanation was not rejected")

    zero = list(g.product_relations(0, 12)) + list(g.product_relations(12, 0))
    assert not any(r["prompt"] in ("0*x=0,x=?", "x*0=0,x=?", "0/x=0,x=?")
                   for r in zero)
    assert any(r["prompt"] == "x*12=0,x=?" and r["answer"] == "0"
               for r in zero)

    pools, _, _ = g.build_pools(g.SEED)
    a = g.sample_rows(pools, "mini", g.SEED)
    b = g.sample_rows(pools, "mini", g.SEED)
    assert a == b, "Sample selection is nondeterministic"
    info = g.verify_records(a, g.SEED, vocab=None)
    assert info["train"]["rows"] == 144
    assert info["val"]["rows"] == 48
    assert info["test"]["rows"] == 48
    assert all(x["max_required_sequence_length"] <= 256 for x in info.values())
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / "sample.json"
        sha = g.write_new(p, [{"positive": r["positive"], "negative": r["negative"]}
                              for r in a["train"]])
        assert len(json.loads(p.read_text())) == 144 and len(sha) == 64
        try:
            g.write_new(p, [])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Unexpected dataset overwrite")
    print("PASS: mathematics, deterministic sampling, fact-disjoint splits,")
    print("      context length, JSON hash and overwrite safety")
    print({s: x["rows"] for s, x in info.items()})


if __name__ == "__main__":
    main()
