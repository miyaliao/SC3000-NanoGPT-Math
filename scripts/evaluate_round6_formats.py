#!/usr/bin/env python3
"""Uniform Round6 evaluator: original fixed answer-first + final-answer-after-steps.

Uses the EXACT original period/newline stop behavior with greedy argmax.
Reports strict first-answer accuracy (comparable to Round3/4/5) AND a separate
format-flexible final-answer accuracy. Never claims TA grader compatibility.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import re

from evaluate_ta_integer import greedy_original_stop
from train_round3_course import evaluate_text

START=re.compile(r"^\s*The answer is\s+([+-]?\d+)(?=\s|[.]|$)",re.I)
REASON=re.compile(
    r"^\s*(\d+)\*(\d+)=(\d+) and "
    r"(\d+)\*(\d+)=(\d+) and "
    r"(\d+)\+(\d+)=(\d+) so The answer is\s+(\d+)\.\s*$",re.I
)

def parse_completion(text,gold):
    strict_status,strict_prediction,_=evaluate_text(text,str(gold))
    start=START.match(text)
    reason=REASON.fullmatch(text)
    mode="unparsed"
    parsed=None
    calculation_valid=False
    if start:
        mode="answer_first"
        parsed=int(start.group(1))
    elif reason:
        mode="reason_before_answer"
        x,tens,p1,xx,ones,p2,q1,q2,total,n=(int(v) for v in reason.groups())
        parsed=n
        calculation_valid=(x==xx and tens>=10 and tens%10==0 and ones>0
                           and x*tens==p1 and x*ones==p2 and
                           p1==q1 and p2==q2 and p1+p2==total==n)
    numeric_status=("correct" if parsed==int(gold) else
                    "math_error" if parsed is not None else "format_failure")
    return {
        "strict_status":strict_status,"strict_prediction":strict_prediction,
        "final_status":numeric_status,"final_prediction":parsed,
        "output_mode":mode,"calculation_valid":calculation_valid
    }


def score_formats(model,rows,stoi,itos,device,max_new_tokens):
    model.eval()
    counts=Counter()
    by_category=defaultdict(Counter)
    stops=Counter()
    predictions=[]
    for i,r in enumerate(rows):
        completion,stop=greedy_original_stop(model,r["prompt"],stoi,itos,
                                             max_new_tokens,device)
        parsed=parse_completion(completion,r["answer"])
        category=r["category"]
        predictions.append({
            "prompt":r["prompt"],"answer":str(r["answer"]),
            "category":category,"completion":completion,
            "stop_reason":stop,**parsed})
        counts["n"]+=1
        counts["strict_correct"]+=(parsed["strict_status"]=="correct")
        counts["final_correct"]+=(parsed["final_status"]=="correct")
        counts["strict_format_failure"]+=(parsed["strict_status"]=="format_failure")
        counts["final_format_failure"]+=(parsed["final_status"]=="format_failure")
        counts["reason_before_answer"]+=(parsed["output_mode"]=="reason_before_answer")
        counts["valid_calculation"]+=parsed["calculation_valid"]
        by_category[category]["n"]+=1
        by_category[category]["strict_correct"]+=(parsed["strict_status"]=="correct")
        by_category[category]["final_correct"]+=(parsed["final_status"]=="correct")
        by_category[category]["reason_before_answer"]+=(parsed["output_mode"]=="reason_before_answer")
        by_category[category]["valid_calculation"]+=parsed["calculation_valid"]
        stops[stop]+=1
        if i<3:
            print(f"  {r['prompt']} -> {completion!r}, strict={parsed['strict_status']}, "
                  f"final={parsed['final_status']}",flush=True)
    n=counts["n"]
    report={
        **dict(counts),
        "strict_accuracy":counts["strict_correct"]/n,
        "final_accuracy":counts["final_correct"]/n,
        "by_category":dict(by_category),"stops":dict(stops),
        "max_new_tokens":max_new_tokens,
        "meaning":"Strict begins The answer is; final also recognizes exact "
                  "distributive reasoning before one terminal answer. "
                  "Neither asserts TA acceptance."
    }
    return report,predictions


def self_test():
    data=[
        (" The answer is 391 because 23*17 equals 391.","391","answer_first",True,True,False),
        (" The answer is 391.","391","answer_first",True,True,False),
        (" 23*10=230 and 23*7=161 and 230+161=391 so The answer is 391.",
         "391","reason_before_answer",False,True,True),
        (" 23*10=231 and 23*7=161 and 231+161=392 so The answer is 391.",
         "391","reason_before_answer",False,True,False),
        (" 23*10=230 and 23*7=161 and 230+161=391.","391","unparsed",False,False,False),
        (" Sorry, I don't know.","391","unparsed",False,False,False),
        (" The answer is 390.","391","answer_first",False,False,False),
    ]
    for text,ans,mode,strict,final,valid in data:
        x=parse_completion(text,ans)
        assert x["output_mode"]==mode,(x,text)
        assert (x["strict_status"]=="correct")==strict,(x,text)
        assert (x["final_status"]=="correct")==final,(x,text)
        assert x["calculation_valid"]==valid,(x,text)
    print("PASS Round6 format parser: strict/first-answer, reasoning-first/final, "
          "arithmetic verification and incomplete-output rejection")


if __name__=="__main__":
    self_test()
