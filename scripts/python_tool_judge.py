"""Tool-loop judge: the judge model has a `python` tool (native vLLM tool-calling,
requires --enable-auto-tool-choice --tool-call-parser qwen3_xml). It writes code,
sees real execution output, iterates if needed, then commits to a reference answer.

Run as a script to compare tool-judge references against an existing bank's
3-program-majority references.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import urllib.request

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute a standalone Python 3 script in a sandbox (15s limit) and return its stdout/stderr. State does NOT persist between calls; each call must be a complete script that print()s what you need.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Complete Python 3 source to execute."}},
            "required": ["code"],
        },
    },
}

SYSTEM = (
    "You are a math verification judge with a python tool. Your job: compute the single correct "
    "integer answer to the question by DIRECT COMPUTATION or BRUTE FORCE with the tool - never by "
    "head math alone. Write a complete script that print()s the answer. If the output is empty, "
    "errors, times out, or looks suspicious, revise the code and run it again. Cross-check with a "
    "second, independently-written approach when feasible. When confident, reply with exactly one "
    "final line:\nFINAL_REFERENCE: <integer>\n"
    "If the question is ambiguous or has no single integer answer, reply instead with:\n"
    "FINAL_REFERENCE: UNVERIFIABLE"
)

FINAL_RE = re.compile(r"FINAL_REFERENCE:\s*(-?\d+|UNVERIFIABLE)")


def run_code(code, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        out = subprocess.run(["python", path], capture_output=True, text=True, timeout=timeout)
        stdout = out.stdout.strip()[:2000]
        stderr = out.stderr.strip()[-1000:]
        if not stdout and not stderr:
            return "(no output - did you forget print()?)"
        return "stdout:\n%s%s" % (stdout or "(empty)", ("\nstderr:\n" + stderr) if stderr else "")
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after %ds - use a faster approach" % timeout
    except Exception as e:
        return "ERROR: %s" % e
    finally:
        os.unlink(path)


def chat_raw(messages, temperature=0.6, max_tokens=2000):
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": [PYTHON_TOOL],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    import time as _time
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())["choices"][0]
        except Exception as e:
            last = e
            _time.sleep(min(60, 5 * (2 ** attempt)))
    raise last


def judge_reference(question, max_turns=6, temperature=0.6):
    """Returns (reference_int_or_None, n_tool_calls, trace). None => UNVERIFIABLE/failed."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Question:\n%s" % question},
    ]
    trace = []
    n_calls = 0
    for _ in range(max_turns):
        choice = chat_raw(messages, temperature=temperature)
        msg = choice["message"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc in tool_calls:
                n_calls += 1
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except Exception:
                    code = ""
                result = run_code(code)
                trace.append({"code": code, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            continue
        messages.append({"role": "assistant", "content": content})
        m = FINAL_RE.search(content)
        if m:
            val = m.group(1)
            return (None if val == "UNVERIFIABLE" else int(val)), n_calls, trace
        # No tool call and no final line: nudge once.
        messages.append({"role": "user", "content": "Reply with FINAL_REFERENCE: <integer> (or UNVERIFIABLE), using the python tool first if you have not verified by code."})
    return None, n_calls, trace


SOLUTION_SYSTEM = (
    "You are a verification judge with a python tool. You are given a question and a candidate "
    "solution. Decide whether the candidate's final answer is CORRECT by INDEPENDENT computation "
    "with the tool - write a complete script that print()s what you need; never trust the "
    "candidate's reasoning or arithmetic. If the question is ambiguous, judge the answer within "
    "the candidate's stated, reasonable interpretation. If code output is empty, errors, or looks "
    "suspicious, revise and rerun. When confident, reply with exactly one final line:\n"
    "VERDICT: CORRECT\nor\nVERDICT: INCORRECT"
)

VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)")


def judge_solution(question, solution, max_turns=6, temperature=0.6):
    """Independent tool-loop judge of one artifact. Returns (verdict_bool_or_None, n_tool_calls, trace)."""
    messages = [
        {"role": "system", "content": SOLUTION_SYSTEM},
        {"role": "user", "content": "Question:\n%s\n\nCandidate solution:\n%s" % (question, solution[-6000:])},
    ]
    trace = []
    n_calls = 0
    for _ in range(max_turns):
        choice = chat_raw(messages, temperature=temperature)
        msg = choice["message"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc in tool_calls:
                n_calls += 1
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except Exception:
                    code = ""
                result = run_code(code)
                trace.append({"code": code, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            continue
        messages.append({"role": "assistant", "content": content})
        m = VERDICT_RE.search(content)
        if m:
            trace.append({"final_response": content})
            return m.group(1) == "CORRECT", n_calls, trace
        messages.append({"role": "user", "content": "Reply with VERDICT: CORRECT or VERDICT: INCORRECT, after verifying with the python tool."})
    return None, n_calls, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=str, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.bank) if l.strip()]
    # mix: normal refs, discarded (ref None), stride-sampled
    sample = [r for r in rows if r.get("reference_answer") is None][:3]
    rest = [r for r in rows if r.get("reference_answer") is not None]
    sample += rest[:: max(1, len(rest) // max(1, args.n - len(sample)))][: args.n - len(sample)]

    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        ref, n_calls, trace = judge_reference(r["question"])
        return r, ref, n_calls, trace

    agree = disagree = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, ref, n_calls, trace in ex.map(one, sample):
            old = r.get("reference_answer")
            status = "AGREE" if ref == old else ("RESOLVED-DISCARD" if old is None and ref is not None else "DISAGREE")
            if status == "AGREE":
                agree += 1
            elif status == "DISAGREE":
                disagree += 1
            print("[%s] tool_calls=%d old=%s new=%s | %s" % (status, n_calls, old, ref, r["question"][:100]), flush=True)
            if status == "DISAGREE":
                for t in trace[-2:]:
                    print("    last code result: %s" % t["result"][:200].replace("\n", " | "), flush=True)
    print("agree=%d disagree=%d of %d" % (agree, disagree, len(sample)), flush=True)


if __name__ == "__main__":
    main()
