"""Generate a small synthetic SFT dataset (module 09).

Real datasets: OpenAssistant, UltraChat (HuggingFace). This script produces a
tiny template-based set so the whole SFT path runs anywhere, offline.
Output: data/sft.jsonl with {"prompt": ..., "response": ...}
"""
import json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

PROMPTS = [
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Name three planets.", "Mercury, Venus, and Earth are three planets."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What color is the sky on a clear day?", "On a clear day, the sky appears blue."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
    ("Finish: the sun rises in the", "The sun rises in the east."),
    ("What does 'H2O' mean?", "H2O is the chemical formula for water."),
    ("Name a prime number greater than 10.", "11 is a prime number greater than 10."),
]
# domain-flavored rephrasings to get volume + template robustness
REPHRASES = [
    "{q}", "Question: {q}", "Answer the following: {q}", "Please tell me: {q}",
    "Can you answer: {q}", "I need to know: {q}", "{q} Answer briefly.",
    "Quick question: {q}", "Help me out: {q}", "Just answer this: {q}",
]


def main():
    out = []
    for q, a in PROMPTS:
        for r in REPHRASES:
            out.append({"prompt": r.format(q=q), "response": a})
    os.makedirs("data", exist_ok=True)
    with open("data/sft.jsonl", "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(out)} samples to data/sft.jsonl")


if __name__ == "__main__":
    main()
