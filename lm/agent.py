"""Tool-use agent loop (module 24): ReAct at course scale.

The o-style loop, skeleton: the model alternates THINK/tool-call turns with
observations until it emits an answer. The calculator tool is verifiable —
the same kind of RLVR reward (module 10) that trains frontier agents.

Turn grammar (taught by the SFT/RL phase in real systems):
    CALC: <expression>          -> tool call
    RESULT: <value>             -> observation (appended by the loop)
    ANSWER: <text>              -> final answer, loop ends
"""
import re

import torch

from .tokenizer import get_tokenizer

TOOL_CALL = re.compile(r"CALC:\s*([^\n]+)")
FINAL = re.compile(r"ANSWER:\s*(.*)", re.DOTALL)


class Calculator:
    """The toy tool: a safe arithmetic evaluator (the RLVR reward source)."""

    def run(self, expr: str) -> str:
        expr = expr.strip().replace(" ", "")
        if not re.fullmatch(r"[0-9+\-*/().]+", expr):
            return "ERROR: invalid expression"
        try:
            # eval is unsafe in general; guarded by the charset whitelist above
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return "ERROR: evaluation failed"


@torch.no_grad()
def agent_loop(model, task: str, tool: Calculator, max_turns: int = 5,
               max_new: int = 64, device: str = "cuda",
               prompt_fn=None) -> tuple[str, list[str]]:
    """Run the loop: generate -> parse CALC -> execute -> append RESULT ->
    repeat until ANSWER or max_turns. Returns (final_answer, turn_log)."""
    from .generate import generate
    enc = get_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = prompt_fn(task) if prompt_fn else (
        f"Solve this step by step. You may call the calculator by writing "
        f"`CALC: <expression>` on its own line; you will receive "
        f"`RESULT: <value>`. End with `ANSWER: <answer>`.\n\nTask: {task}\n")
    history = prompt
    turn_log = []
    answer = ""
    max_ctx = model.cfg.max_seq_len
    for turn in range(max_turns):
        # sliding window: the model can only see the last max_seq_len tokens
        ctx_ids = enc.encode_ordinary(history)[-max_ctx:]
        out = generate(model, ctx_ids, max_new=max_new,
                       temperature=0.7, top_p=0.9, device=device)
        text = enc.decode(out)
        history += text
        m = TOOL_CALL.search(text)
        f = FINAL.search(text)
        if m and (f is None or m.start() < f.start()):
            result = tool.run(m.group(1))
            history += f"\nRESULT: {result}\n"
            turn_log.append(f"turn {turn}: CALC {m.group(1)} -> {result}")
            continue
        if f:
            answer = f.group(1).strip()
            turn_log.append(f"turn {turn}: ANSWER {answer[:60]}")
            break
        turn_log.append(f"turn {turn}: no parseable action")
        break
    return answer, turn_log
