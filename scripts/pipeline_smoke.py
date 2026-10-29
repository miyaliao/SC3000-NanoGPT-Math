#!/usr/bin/env python3
"""End-to-end SC3000 DPO pipeline smoke test against the ACTUAL notebook cells.

No permanent training data or final dpo.pt is touched. Runs Notebook Steps 2-8
(except Step 1 pip installs), with a tiny disposable preference dataset.

From repository root, inside the prepared .venv:
    python scripts/pipeline_smoke.py --self-test-only
    python scripts/pipeline_smoke.py --device cpu
    python scripts/pipeline_smoke.py --device cuda --samples 96 --batch-size 16
    python scripts/pipeline_smoke.py --device cuda --out-dir dpo/smoke_outputs/run1

For TC1 GPU use an interactive allocation or sbatch. Do not run CUDA training
on a login node. The sample generator here is ONLY for pipeline verification;
Round 1 will use a separate, fully audited dataset generator.
"""

import argparse
import contextlib
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "dpo" / "dpo.ipynb"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_eval_utils import greedy_completion, grade_completion, parse_expected, self_test as test_eval  # noqa: E402

RESERVED_PROBES = {"79-7=?", "x+55=95,x=?", "7/2=?", "5-9=?"}
KINDS = (
    "add", "subtract", "multiply", "divide",
    "x_plus_a", "a_plus_x", "x_minus_a", "a_minus_x",
    "x_times_a", "a_times_x", "x_over_a", "a_over_x",
)
NUM = r"[+-]?\d+(?:\.\d+)?"
ARITH_RE = re.compile(rf"^({NUM})([+*/-])({NUM})=\?$")
EQUATION_RE = re.compile(rf"^({NUM}|x)([+*/-])({NUM}|x)=({NUM}),x=\?$")


def fmt_number(value):
    """Render fractions whose decimal expansion ends within two digits."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    scaled = value * 100
    if scaled.denominator != 1:
        raise ValueError(f"Non-terminating or >2-place decimal: {value}")
    v = int(scaled)
    sign = "-" if v < 0 else ""
    whole, decimal = divmod(abs(v), 100)
    return (f"{sign}{whole}.{decimal:02d}").rstrip("0").rstrip(".")


def apply_op(left, op, right):
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/" and right != 0:
        return left / right
    raise ValueError(f"Invalid math operation or division by zero: {op}")


def verify_math(prompt, answer):
    """Independent verification by parsing the printed question and substitution."""
    target = parse_expected(answer)
    equation = EQUATION_RE.fullmatch(prompt)
    if equation:
        lhs, op, rhs_term, rhs = equation.groups()
        if (lhs == "x") == (rhs_term == "x"):
            raise ValueError(f"Equation does not have exactly one x: {prompt}")
        left = target if lhs == "x" else parse_expected(lhs)
        right = target if rhs_term == "x" else parse_expected(rhs_term)
        return apply_op(left, op, right) == parse_expected(rhs)
    arithmetic = ARITH_RE.fullmatch(prompt)
    if arithmetic:
        left, op, right = arithmetic.groups()
        return apply_op(parse_expected(left), op, parse_expected(right)) == target
    raise ValueError(f"Unrecognized math prompt: {prompt}")


def new_problem(kind, rng, decimal=False):
    a = rng.randint(1, 35)
    b = rng.randint(1, 35)
    x = Fraction(rng.randint(1, 35), 1)
    if decimal:
        x = Fraction(rng.randrange(1, 70, 2), 2)
    if kind == "add":
        left = x if decimal else Fraction(a)
        prompt, result = f"{fmt_number(left)}+{b}=?", left + b
    elif kind == "subtract":
        left = x if decimal else Fraction(a)
        prompt, result = f"{fmt_number(left)}-{b}=?", left - b
    elif kind == "multiply":
        right = rng.randrange(1, 16, 2) if decimal else rng.randint(1, 15)
        left = x if decimal else Fraction(a)
        prompt, result = f"{fmt_number(left)}*{right}=?", left * right
    elif kind == "divide":
        if decimal:
            dividend, divisor = x, rng.choice((1, 2, 5))
            result = dividend / divisor
        else:
            dividend, divisor = a * b, b
            result = Fraction(a)
        prompt = f"{fmt_number(dividend)}/{divisor}=?"
    elif kind == "x_plus_a":
        prompt, result = f"x+{a}={fmt_number(x+a)},x=?", x
    elif kind == "a_plus_x":
        prompt, result = f"{a}+x={fmt_number(a+x)},x=?", x
    elif kind == "x_minus_a":
        prompt, result = f"x-{a}={fmt_number(x-a)},x=?", x
    elif kind == "a_minus_x":
        prompt, result = f"{a}-x={fmt_number(a-x)},x=?", x
    elif kind == "x_times_a":
        prompt, result = f"x*{a}={fmt_number(x*a)},x=?", x
    elif kind == "a_times_x":
        prompt, result = f"{a}*x={fmt_number(a*x)},x=?", x
    elif kind == "x_over_a":
        divisor = rng.choice((1, 2, 5)) if decimal else b
        if not decimal:
            x = Fraction(rng.randint(1, 20) * divisor)
        prompt, result = f"x/{divisor}={fmt_number(x/divisor)},x=?", x
    elif kind == "a_over_x":
        quotient = rng.randint(1, 20)
        prompt, result = f"{fmt_number(x*quotient)}/x={quotient},x=?", x
    else:
        raise ValueError(f"Unknown problem kind {kind}")
    result = fmt_number(result)
    if not verify_math(prompt, result):
        raise AssertionError(f"Invalid positive label: {prompt} -> {result}")
    return prompt, result


def make_pairs(n, seed):
    rng = random.Random(seed)
    records, seen = [], set()
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        for attempt in range(2000):
            prompt, answer = new_problem(kind, rng, decimal=(i % 7 == 0))
            if prompt not in seen and prompt not in RESERVED_PROBES:
                seen.add(prompt)
                break
        else:
            raise RuntimeError(f"Cannot get a unique prompt for kind {kind}")
        records.append({
            "positive": f"{prompt} The answer is {answer}.",
            "negative": f"{prompt} Sorry, I don't know.",
            "prompt": prompt, "answer": answer, "category": kind,
        })
    return records


def notebook_steps():
    with NOTEBOOK.open(encoding="utf-8") as handle:
        nb = json.load(handle)
    by_step = {}
    current = None
    for cell in nb["cells"]:
        source = cell.get("source", [])
        code = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] == "markdown":
            found = re.search(r"###\s+Step\s+(\d+)\s*:", code)
            if found:
                current = int(found.group(1))
        elif cell["cell_type"] == "code" and current is not None:
            by_step.setdefault(current, []).append(code)
    for step in range(2, 9):
        if step not in by_step:
            raise RuntimeError(f"Notebook is missing Step {step} code")
    return by_step


def execute_step(step, chunks, globals_dict):
    for i, source in enumerate(chunks[step]):
        if source.lstrip().startswith(("!", "%")):
            raise RuntimeError(f"Cannot execute notebook shell/magic at Step {step}")
        exec(compile(source, f"{NOTEBOOK}:step{step}:cell{i}", "exec"), globals_dict, globals_dict)


def compare_predictions(model, stoi, itos, prompts, token_budget, device):
    model.eval()
    rows = []
    for prompt, answer in prompts:
        missing = sorted(set(prompt) - set(stoi))
        if missing:
            raise ValueError(f"Evaluation prompt has unknown characters: {missing}")
        completion, stop = greedy_completion(
            model, [stoi[ch] for ch in prompt], itos, token_budget, device
        )
        status, prediction = grade_completion(completion, answer)
        rows.append({"prompt": prompt, "expected": answer, "completion": completion,
                     "prediction": prediction, "status": status, "stop_reason": stop})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--out-dir", help="Optional NEW directory for persistent smoke artifacts")
    args = parser.parse_args()

    test_eval()
    for record in make_pairs(240, 42):
        assert verify_math(record["prompt"], record["answer"])
        assert record["positive"].startswith(record["prompt"])
        assert record["negative"].startswith(record["prompt"])
    print("PASS: all 12 synthetic problem templates, exact answers, unique prompts")
    if args.self_test_only:
        print("SELF-TEST PASSED (model not loaded; no gradient steps run)")
        return

    if args.samples < 12 or args.batch_size < 1 or args.samples % args.batch_size:
        parser.error("Use samples >= 12, batch_size >= 1, and samples divisible by batch_size")
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be >= 2")
    if not (ROOT / "sft" / "gpt.pt").is_file():
        parser.error("Missing trusted course checkpoint: sft/gpt.pt")
    if not (ROOT / "sft" / "meta.pkl").is_file():
        parser.error("Missing course tokenizer: sft/meta.pkl")

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable: use an allocated GPU Slurm job or --device cpu")
    steps = notebook_steps()
    records = make_pairs(args.samples, args.seed)

    # Four independent probes; never use their prompts for smoke training.
    probes = [("79-7=?", "72"), ("x+55=95,x=?", "40"),
              ("7/2=?", "3.5"), ("5-9=?", "-4")]
    train_prompts = {row["prompt"] for row in records}
    if train_prompts.intersection(prompt for prompt, _ in probes):
        raise AssertionError("Smoke probes unexpectedly overlap training data")

    os.chdir(ROOT)  # Notebook Step 2 deliberately checks cwd.
    with contextlib.ExitStack() as stack:
        if args.out_dir:
            out = Path(args.out_dir).expanduser().resolve()
            smoke_root = (ROOT / "dpo" / "smoke_outputs").resolve()
            if out == smoke_root or not out.is_relative_to(smoke_root):
                parser.error("--out-dir must be a new subdirectory of dpo/smoke_outputs/")
            out.mkdir(parents=True, exist_ok=False)
        else:
            out = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="sc3000_dpo_smoke_")))
        (out / "pos_neg_pairs.json").write_text(
            json.dumps([{"positive": r["positive"], "negative": r["negative"]}
                        for r in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        env = {"__name__": "__dpo_smoke__"}
        print("=== Executing the actual dpo.ipynb Step 2 ===")
        execute_step(2, steps, env)
        env["device"] = device
        env["batch_size"] = args.batch_size
        env["epochs"] = 1
        env["dpo_dir"] = out  # prevents final data/checkpoint overwrite
        print(f"Device={device} | disposable pairs={len(records)} | batch={args.batch_size}")

        print("=== Executing Notebook Steps 3-4: helpers and baseline weights ===")
        execute_step(3, steps, env)
        execute_step(4, steps, env)
        assert int(env["gpt"].config.block_size) == 256, "Unexpected model context length"
        assert env["gpt"].config.vocab_size == len(env["stoi"])
        for row in records:
            for field in ("positive", "negative"):
                unknown = set(row[field]) - set(env["stoi"])
                if unknown:
                    raise ValueError(f"OOV characters in {field}: {unknown}")
                if len(row[field]) + 4 > env["gpt"].config.block_size:
                    raise ValueError("Sample exceeds model block_size")
        print("PASS: tokenizer, model config, full text sequence lengths")

        itos, stoi = env["itos"], env["stoi"]
        before = compare_predictions(env["gpt"], stoi, itos, probes, args.max_new_tokens, device)
        tracked_parameter = env["gpt"].transformer.h[0].attn.c_attn.weight
        before_parameter = tracked_parameter.detach().cpu().clone()

        print("=== Executing Notebook Steps 5-7: data, optimizer and actual training ===")
        execute_step(5, steps, env)
        assert env["data_path"].resolve() == (out / "pos_neg_pairs.json").resolve()
        assert len(env["lines"]) == len(records)
        execute_step(6, steps, env)
        assert env["steps_per_epoch"] == args.samples // args.batch_size
        assert env["epochs"] == 1 and env["dpo_dir"] == out
        execute_step(7, steps, env)
        assert len(env["training_loss"]) == 1
        assert all(__import__("math").isfinite(v) for v in env["training_loss"])
        delta = (tracked_parameter.detach().cpu() - before_parameter).abs().max().item()
        if delta <= 0:
            raise AssertionError("No parameter update detected")
        print(f"PASS: gradient updates, finite loss, optimizer/scheduler | max parameter delta={delta:.8g}")

        print("=== Executing Notebook Step 8: checkpoint loading ===")
        checkpoint_path = out / "dpo.pt"
        assert checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
        expected_after = tracked_parameter.detach().cpu().clone()
        execute_step(8, steps, env)  # Current Notebook only reloads; evaluation is TODO.
        restored = env["gpt"].transformer.h[0].attn.c_attn.weight.detach().cpu()
        if not torch.equal(expected_after, restored):
            raise AssertionError("Checkpoint reload does not reproduce trained weights")
        print("PASS: dpo.pt saved and reloaded exactly using Notebook Step 8")
        after = compare_predictions(env["gpt"], stoi, itos, probes, args.max_new_tokens, device)

        for b, a in zip(before, after):
            print(f"PROBE {b['prompt']!r} (answer {b['expected']}):")
            print(f"  baseline: {b['completion']!r} [{b['status']}, {b['stop_reason']}]")
            print(f"  trained:  {a['completion']!r} [{a['status']}, {a['stop_reason']}]")

        report = {"notebook": str(NOTEBOOK), "device": device, "samples": len(records),
                  "batch_size": args.batch_size, "train_steps": env["steps_per_epoch"],
                  "training_loss": env["training_loss"], "parameter_max_change": delta,
                  "categories": dict(Counter(r["category"] for r in records)),
                  "baseline_probes": before, "post_smoke_probes": after,
                  "checkpoint_bytes": checkpoint_path.stat().st_size,
                  "note": "Pipeline only: not a formal accuracy baseline or Round 1."}
        (out / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("=== PIPELINE SMOKE TEST PASSED ===")
        print(f"Artifacts: {out}")
        if not args.out_dir:
            print("Temporary artifacts will be removed; use --out-dir to keep them.")


if __name__ == "__main__":
    main()
