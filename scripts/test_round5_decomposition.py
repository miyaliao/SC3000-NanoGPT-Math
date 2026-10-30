#!/usr/bin/env python3
"""CPU-only self-test of Round5 decomposition math and template invariants."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_round5_decomposition as m

def case(prompt,answer,original_explanation):
    return {"positive":f"{prompt} The answer is {answer} because {original_explanation}.",
            "negative":f"{prompt} {m.REFUSAL}"}

def main():
    tests=[
        (case("23*17=?","391","23*17 equals 391"),True,["23*10=230","23*7=161","230+161=391"]),
        (case("x/17=23,x=?","391","23*17 equals 391"),True,["23*10=230","23*7=161","230+161=391"]),
        (case("7*53=?","371","7*53 equals 371"),True,["7*50=350","7*3=21","350+21=371"]),
        (case("25*20=?","500","25*20 equals 500"),True,["20*20=400","20*5=100","400+100=500"]),
        (case("10*20=?","200","10*20 equals 200"),False,[]),
        (case("3*7=?","21","3*7 equals 21"),False,[]),
        (case("7*1=?","7","7*1 equals 7"),False,[]),
        (case("17+23=?","40","17+23 equals 40"),False,[]),
        (case("x*17=391,x=?","23","391/17 equals 23"),False,[]),
    ]
    for pair,expected,parts in tests:
        text,info=m.decomposition(pair)
        assert (info is not None)==expected, pair["positive"]
        if expected:
            for piece in parts:
                assert piece in text,(piece,text)
            assert text.startswith(pair["positive"].split(" because ")[0]+" because ")
            assert text.count(".")==1
        else:
            assert text==pair["positive"]
    incorrect=case("23*17=?","392","23*17 equals 392")
    try:m.decomposition(incorrect)
    except ValueError:pass
    else:raise AssertionError("Wrong gold answer accepted")
    print("PASS: direct and inverse product decomposition; no trivial +0 split;")
    print("      unchanged other math categories and identities; wrong gold rejected")


if __name__=="__main__":main()
