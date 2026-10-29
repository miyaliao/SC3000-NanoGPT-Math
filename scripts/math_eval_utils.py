"""Shared exact-numeric grading and decimal-safe greedy decoding for NanoGPT-Math.

The course model's generate() stops at the first '.', which truncates decimals.
These helpers leave the supplied model.py unchanged.
"""

from fractions import Fraction
import re

NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?\Z")
ANSWER_PATTERN = re.compile(
    r"^\s*(?:The answer is\s+)?([+-]?\d+(?:\.\d+)?)(?=\s*(?:\.(?!\d)|because\b|$))",
    flags=re.IGNORECASE,
)
REFUSAL_PATTERN = re.compile(
    r"\b(?:sor+y|unknown)\b|\b(?:do not know|don't know|cannot answer|can't answer)\b",
    flags=re.IGNORECASE,
)


def parse_expected(value):
    """Parse an integer or finite decimal exactly, without binary floats."""
    s = str(value).strip()
    if not NUMBER_PATTERN.fullmatch(s):
        raise ValueError(f"Expected an integer or finite decimal, got {value!r}")
    return Fraction(s)


def grade_completion(completion, expected):
    """Return (correct/math_error/refusal/format_failure, parsed_answer_or_None)."""
    match = ANSWER_PATTERN.match(completion)
    if match:
        answer = match.group(1)
        result = "correct" if Fraction(answer) == parse_expected(expected) else "math_error"
        return result, answer
    if REFUSAL_PATTERN.search(completion):
        return "refusal", None
    return "format_failure", None


def decode_from_next(predict_next, prompt_ids, itos, max_new_tokens):
    """Greedy token streaming. Peek after a numeric '.' to distinguish 1.5 from 1.

    predict_next(input_ids) -> int, using the already-emitted token IDs.
    A non-digit lookahead is discarded, just like the original period stop.
    """
    dot_id = next((i for i, ch in itos.items() if ch == "."), None)
    newline_id = next((i for i, ch in itos.items() if ch == "\n"), None)
    digit_ids = {i for i, ch in itos.items() if ch in "0123456789"}
    if dot_id is None or newline_id is None or max_new_tokens <= 0:
        raise ValueError("Tokenizer must contain '.' and newline, and token budget must be positive")

    context = list(prompt_ids)
    generated = []
    stop_reason = "max_new_tokens"
    while len(generated) < max_new_tokens:
        token = int(predict_next(context))
        if token == newline_id:
            stop_reason = "newline"
            break
        prev_is_digit = bool(generated) and generated[-1] in digit_ids
        context.append(token)
        generated.append(token)
        if token == dot_id:
            if prev_is_digit and len(generated) < max_new_tokens:
                next_token = int(predict_next(context))
                if next_token in digit_ids:
                    context.append(next_token)
                    generated.append(next_token)
                    continue
            stop_reason = "period"
            break
    return "".join(itos[token] for token in generated), stop_reason


def greedy_completion(model, prompt_ids, itos, max_new_tokens, device):
    import torch

    block_size = model.config.block_size

    def next_token(context):
        ids = torch.tensor([context[-block_size:]], dtype=torch.long, device=device)
        logits, _ = model(ids)
        return int(logits[0, -1].argmax().item())

    with torch.inference_mode():
        return decode_from_next(next_token, prompt_ids, itos, max_new_tokens)


def self_test():
    assert parse_expected("1.50") == parse_expected("1.5")
    assert parse_expected("-3") == Fraction(-3)
    for bad in ("1/3", "nan", "1e4", "1.2.3"):
        try:
            parse_expected(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected rejection of {bad!r}")
    assert grade_completion(" The answer is 1.50.", "1.5")[0] == "correct"
    assert grade_completion(" The answer is 1.", "1.5")[0] == "math_error"
    assert grade_completion(" The answer is 1.5.0", "1.5")[0] == "format_failure"
    assert grade_completion(" Sorry, I don't know.", "2")[0] == "refusal"
    assert grade_completion(" Something else.", "2")[0] == "format_failure"
    assert grade_completion(" The answer is -4.", "-4")[0] == "correct"

    chars = "\n .-0123456789The answer isSory,Idn'tkow"
    unique = list(dict.fromkeys(chars))
    itos = dict(enumerate(unique))
    stoi = {v: k for k, v in itos.items()}
    for answer, expected, outcome in (
        (" The answer is 1.5.", "1.5", "correct"),
        (" The answer is -4.", "-4", "correct"),
        (" Sorry, I don't know.", "2", "refusal"),
    ):
        scripted = [stoi[x] for x in answer]
        prefix = [stoi["T"]]

        def scripted_predict(tokens):
            i = len(tokens) - len(prefix)
            return scripted[i] if i < len(scripted) else stoi["\n"]

        completion, stop = decode_from_next(scripted_predict, prefix, itos, 50)
        assert (completion, stop) == (answer, "period"), (completion, stop)
        assert grade_completion(completion, expected)[0] == outcome
    print("PASS: decimal-aware stopping, exact decimal grading, refusal grading")


if __name__ == "__main__":
    self_test()
