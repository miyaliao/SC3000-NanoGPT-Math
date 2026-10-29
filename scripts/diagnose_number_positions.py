#!/usr/bin/env python3
"""Read-only number-position diagnosis of NanoGPT-Math.

Teacher forcing: score the NEXT-character distribution at gold answer positions.
Report digit places separately from minus sign, answer lengths, and arithmetic
categories. Optionally cross-reference existing (never regenerated) greedy
predictions. This is not training and does not use validation labels to update
the model.

Run:
    python scripts/diagnose_number_positions.py --self-test-only
    python scripts/diagnose_number_positions.py --device cuda
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from diagnose_generation_gap import load_model   # noqa: E402
from overfit256_course import select_rows         # noqa: E402

PREFIX = " The answer is "
DEFAULT_CHECKPOINT = "dpo/experiments/overfit256_termination_gamma050/dpo.pt"
DEFAULT_OUTPUT = "results/number_position_gamma050"
PLACES = ("ones", "tens", "hundreds", "thousands", "ten_thousands")


def location_of_answer_char(answer: str, index: int) -> dict:
    """Position labels for one character; minus sign isn't a numeric place."""
    if not answer or (not answer.lstrip("-").isdigit()):
        raise ValueError(f"Only canonical integer answers allowed: {answer!r}")
    if answer.startswith("-") and answer[1:] == "0":
        raise ValueError("Negative zero is not canonical")
    negative = answer.startswith("-")
    digits = answer[1:] if negative else answer
    if negative and index == 0:
        return {"kind": "sign", "place_from_right": None,
                "place": "minus_sign", "digit_from_left": None}
    j = index - int(negative)
    if j < 0 or j >= len(digits):
        raise IndexError(index)
    exponent = len(digits) - 1 - j
    return {
        "kind": "digit",
        "place_from_right": exponent,
        "place": PLACES[exponent] if exponent < len(PLACES) else f"10^{exponent}",
        "digit_from_left": j + 1,
    }


def self_test():
    def places(answer):
        return [(answer[i], location_of_answer_char(answer, i)["place"])
                for i in range(len(answer))]
    assert places("63") == [("6", "tens"), ("3", "ones")]
    assert places("-34") == [("-", "minus_sign"), ("3", "tens"), ("4", "ones")]
    assert places("1360") == [("1", "thousands"), ("3", "hundreds"),
                              ("6", "tens"), ("0", "ones")]
    assert places("0") == [("0", "ones")]
    assert location_of_answer_char("-34", 1)["digit_from_left"] == 1
    try:
        location_of_answer_char("2.5", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Decimal answer not rejected")
    print("PASS: integer place mapping, left-to-right mapping, negative sign isolation")


def summarize_token_rows(items: list[dict]) -> dict:
    if not items:
        return {"n": 0}
    return {
        "n": len(items),
        "top1_count": sum(x["top1"] for x in items),
        "top1_rate": sum(x["top1"] for x in items) / len(items),
        "mean_gold_probability": statistics.mean(x["gold_probability"] for x in items),
        "mean_gold_logprob": statistics.mean(x["gold_logprob"] for x in items),
        "mean_gold_rank": statistics.mean(x["gold_rank"] for x in items),
        "mean_early_period_probability": statistics.mean(
            x["early_period_probability"] for x in items),
    }


def group_tokens(items: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for item in items:
        groups[str(item[field])].append(item)
    return {key: summarize_token_rows(group) for key, group in sorted(groups.items())}


def load_existing_predictions(model_path: Path, split: str, records: list[dict]):
    """Reference the model's saved greedy outputs; never call generate here."""
    file = model_path.parent / f"{split}_predictions.json"
    if not file.is_file():
        return None
    rows = json.loads(file.read_text(encoding="utf-8"))
    by_prompt = {r["prompt"]: r for r in rows}
    if len(rows) != len(records) or len(by_prompt) != len(records):
        raise RuntimeError(f"Existing {split} predictions differ from selected dataset: {file}")
    expected_prompts = {r["prompt"] for r in records}
    if set(by_prompt) != expected_prompts:
        raise RuntimeError(f"Existing {split} predictions don't match selected prompts")
    for r in records:
        pred = by_prompt[r["prompt"]]
        expected = str(r.get("correct_answer", r.get("answer")))
        if str(pred["answer"]) != expected or pred["category"] != r["category"]:
            raise RuntimeError(f"Existing prediction data mismatch: {r['prompt']}")
    return by_prompt


def summarize_questions(question_rows: list[dict]) -> dict:
    n = len(question_rows)
    if not n:
        return {"n": 0}
    out = {
        "n": n,
        "all_digits_top1_count": sum(q["all_digits_top1"] for q in question_rows),
        "all_digits_top1_rate": sum(q["all_digits_top1"] for q in question_rows) / n,
        "first_digit_top1_count": sum(q["first_digit_top1"] for q in question_rows),
        "first_digit_top1_rate": sum(q["first_digit_top1"] for q in question_rows) / n,
        "gold_period_top1_count": sum(q["gold_period_top1"] for q in question_rows),
        "gold_period_top1_rate": sum(q["gold_period_top1"] for q in question_rows) / n,
        "gold_period_mean_probability": statistics.mean(
            q["gold_period_probability"] for q in question_rows),
    }
    known = [q for q in question_rows if q.get("greedy_status") is not None]
    if known:
        out["greedy_accuracy"] = sum(q["greedy_status"] == "correct" for q in known) / len(known)
        out["greedy_correct"] = sum(q["greedy_status"] == "correct" for q in known)
    return out


def group_questions(question_rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for q in question_rows:
        groups[str(q[field])].append(q)
    return {k: summarize_questions(v) for k, v in sorted(groups.items())}


def score_split(model, records: list[dict], stoi: dict, device: str,
                batch_size: int, predictions):
    import torch
    import torch.nn.functional as F

    tokens = []
    questions = []
    prefix = PREFIX
    dot_id = stoi["."]
    for begin in range(0, len(records), batch_size):
        batch = records[begin:begin + batch_size]
        sequences, spans = [], []
        for r in batch:
            prompt = r["prompt"]
            answer = str(r.get("correct_answer", r.get("answer")))
            if not answer or not answer.lstrip("-").isdigit() or (answer.startswith("0") and len(answer)>1):
                raise ValueError(f"Not a canonical integer answer: {answer!r}")
            if any(ch not in stoi for ch in prompt + prefix + answer + "."):
                raise ValueError(f"Tokenizer OOV in {prompt}")
            seq = prompt + prefix + answer + "."
            if len(seq) > model.config.block_size:
                raise ValueError(f"Exceeds model context: {prompt}")
            sequences.append(seq)
            spans.append((len(prompt) + len(prefix), answer))
        width = max(map(len, sequences))
        ids = torch.tensor([[stoi[ch] for ch in s] + [0] * (width - len(s))
                            for s in sequences], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits, _ = model(ids[:, :-1], full_seq=True)
            logprobs = F.log_softmax(logits, dim=-1)

        for b, r in enumerate(batch):
            start, answer = spans[b]
            q_tokens = []
            for j, ch in enumerate(answer):
                char_pos = start + j  # absolute gold character index in full sequence
                gold_id = stoi[ch]
                l = logits[b, char_pos - 1]  # position i-1 predicts character i
                lp = float(logprobs[b, char_pos - 1, gold_id].item())
                early_period_probability = math.exp(
                    float(logprobs[b, char_pos - 1, dot_id].item()))
                rank = 1 + int((l > l[gold_id]).sum().item())
                pos = location_of_answer_char(answer, j)
                info = {
                    "prompt": r["prompt"], "answer": answer, "category": r["category"],
                    "answer_digit_count": len(answer.lstrip("-")),
                    "gold_character": ch,
                    "kind": pos["kind"], "place": pos["place"],
                    "place_from_right": pos["place_from_right"],
                    "digit_from_left": pos["digit_from_left"],
                    "gold_probability": math.exp(lp),
                    "gold_logprob": lp,
                    "early_period_probability": early_period_probability,
                    "gold_rank": rank,
                    "top1": rank == 1,
                    "greedy_top_character": None,  # optional token text not needed for summary
                }
                tokens.append(info)
                q_tokens.append(info)

            period_index = start + len(answer)
            end_logits = logits[b, period_index - 1]
            end_lp = float(logprobs[b, period_index - 1, dot_id].item())
            digit_tokens = [t for t in q_tokens if t["kind"] == "digit"]
            if not digit_tokens:
                raise AssertionError("Missing digits")
            old = None if predictions is None else predictions[r["prompt"]]
            questions.append({
                "prompt": r["prompt"],
                "answer": answer,
                "category": r["category"],
                "answer_digit_count": len(answer.lstrip("-")),
                "negative_answer": answer.startswith("-"),
                "first_digit_top1": digit_tokens[0]["top1"],
                "all_digits_top1": all(t["top1"] for t in digit_tokens),
                "gold_period_top1": int(end_logits.argmax().item()) == dot_id,
                "gold_period_probability": math.exp(end_lp),
                "greedy_status": None if old is None else old["status"],
                "greedy_completion": None if old is None else old["completion"],
            })
    if len(questions) != len(records):
        raise AssertionError("Incorrect number of questions scored")
    return tokens, questions


def assemble_split(tokens: list[dict], questions: list[dict]) -> dict:
    digit_tokens = [x for x in tokens if x["kind"] == "digit"]
    sign_tokens = [x for x in tokens if x["kind"] == "sign"]
    two_digit_questions = [q for q in questions if q["answer_digit_count"] == 2]
    two_digit_tokens = [x for x in digit_tokens if x["answer_digit_count"] == 2]

    # Paired contingency: how often does one digit succeed when the other fails?
    twos = defaultdict(dict)
    for x in two_digit_tokens:
        twos[x["prompt"]][x["place"]] = x["top1"]
    paired = Counter()
    for q in two_digit_questions:
        pair = twos[q["prompt"]]
        if set(pair) != {"ones", "tens"}:
            raise AssertionError(f"Two-digit mapping failed for {q['prompt']}")
        paired[("tens_top1" if pair["tens"] else "tens_wrong",
                "ones_top1" if pair["ones"] else "ones_wrong")] += 1

    return {
        "questions": summarize_questions(questions),
        "all_digit_positions": summarize_token_rows(digit_tokens),
        "minus_sign_positions": summarize_token_rows(sign_tokens),
        "by_place_from_right": group_tokens(digit_tokens, "place"),
        "by_position_from_left": group_tokens(digit_tokens, "digit_from_left"),
        "by_answer_digit_count": group_questions(questions, "answer_digit_count"),
        "by_category": group_questions(questions, "category"),
        "by_negative_answer": group_questions(questions, "negative_answer"),
        "two_digit_only": {
            "n": len(two_digit_questions),
            "tens": summarize_token_rows([x for x in two_digit_tokens if x["place"] == "tens"]),
            "ones": summarize_token_rows([x for x in two_digit_tokens if x["place"] == "ones"]),
            "paired_top1": {
                f"{a}__{b}": n for (a, b), n in sorted(paired.items())
            },
        },
    }


def write_once(path: Path, contents: str):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite previous diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    self_test()
    if args.self_test_only:
        return
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        parser.error(f"Checkpoint not found: {path}")
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if out == (ROOT / "results").resolve() or not out.is_relative_to((ROOT / "results").resolve()):
        parser.error("--out-dir must be under results/")
    if out.exists():
        parser.error(f"Refusing to overwrite prior result folder: {out}")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; run using Slurm GPU allocation or --device cpu")
    pairs, train, val, indices = select_rows(seed=42)
    assert len(pairs) == len(train) == len(val) == 256
    train_prompts = {x["prompt"] for x in train}
    assert not train_prompts.intersection(x["prompt"] for x in val)
    with (ROOT / "sft" / "meta.pkl").open("rb") as f:
        vocab = pickle.load(f)  # course trusted file
    stoi = vocab["stoi"]
    model = load_model(path, args.device)

    result = {
        "checkpoint": str(path), "experiment": "read-only_teacher_forced_number_position",
        "sample_seed": 42, "train_size": len(train), "validation_size": len(val),
        "method": "gold-prefix next-character Top-1, probability and rank; '-' separated",
        "notes": ("Teacher-forced digit Top-1 uses gold previous digits, so it is "
                  "NOT free generation accuracy. Validation only used for analysis."),
        "splits": {},
    }
    outputs = {}
    for split, records in (("train", train), ("val", val)):
        predictions = load_existing_predictions(path, split, records)
        token_rows, q_rows = score_split(
            model, records, stoi, args.device, args.batch_size, predictions
        )
        result["splits"][split] = assemble_split(token_rows, q_rows)
        outputs[split] = (token_rows, q_rows)
        summary = result["splits"][split]
        print(f"=== {split.upper()} | {len(records)} questions ===")
        print("Digits Top-1:", summary["all_digit_positions"]["top1_count"],
              "/", summary["all_digit_positions"]["n"],
              f"({summary['all_digit_positions']['top1_rate']:.2%})")
        print("All digits Top-1:", summary["questions"]["all_digits_top1_count"],
              "/", len(records), f"({summary['questions']['all_digits_top1_rate']:.2%})")
        for place in ("ones", "tens", "hundreds", "thousands"):
            value = summary["by_place_from_right"].get(place)
            if value:
                print(f"{place:10} {value['top1_count']:3}/{value['n']:3} "
                      f"= {value['top1_rate']:.2%}, mean P(gold)={value['mean_gold_probability']:.3f}, "
                      f"mean early P(.)={value['mean_early_period_probability']:.3f}")
        t = summary["two_digit_only"]
        print(f"Two-digit answers: N={t['n']} | tens={t['tens']['top1_rate']:.2%}"
              f" | ones={t['ones']['top1_rate']:.2%}")
        print("Gold period Top-1:", f"{summary['questions']['gold_period_top1_rate']:.2%}")
        if "greedy_accuracy" in summary["questions"]:
            print("Saved greedy accuracy:", f"{summary['questions']['greedy_accuracy']:.2%}")

    write_once(out / "summary.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for split, (token_rows, q_rows) in outputs.items():
        write_once(out / f"{split}_digit_positions.jsonl",
                   "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in token_rows))
        write_once(out / f"{split}_questions.jsonl",
                   "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in q_rows))
    print("=== NUMBER POSITION DIAGNOSIS COMPLETE ===")
    print("Output:", out / "summary.json")


if __name__ == "__main__":
    main()
