"""Scaled agentic question generation with code-judge measurement.

Runs C parallel proposer chains. Each chain round: propose 8 questions
conditioned on accumulated [pass | program-agreement] feedback, measure with
K solver attempts, grade against a 3-program code-judge consensus, then feed
the measured results back into the next round's prompt.

Every measured question is appended to a JSONL bank compatible with
train_grpo_trl's pairwise_jsonl loader (question, reference_answer,
trainable_for_solver).
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
K = 8

GOLDEN = [
    "How many integers n with 1 <= n <= 150 satisfy the condition that the number of distinct prime factors of n (omega(n)) is exactly 2, the number of positive divisors of n is even, and the sum of the decimal digits of n is a multiple of 5?",
    "Find the number of integers n with 1 <= n <= 300 such that n is a Harshad number (divisible by the sum of its decimal digits), the sum of the positive divisors of n is divisible by 7, and n is congruent to 2 modulo 4.",
    "Define a sequence by a_0 = 0 and a_{n+1} = a_n + floor(a_n / 2) + n. How many integers n with 1 <= n <= 40 satisfy that a_n is divisible by 9 and n is a prime number?",
    "How many integers n with 1 <= n <= 300 satisfy that the product of the decimal digits of n is a multiple of 9, n is odd, and n is not a multiple of 10?",
    "Let f(n) be the sum of the proper divisors of n. Find the smallest integer n greater than 50 such that f(n) = n - 1.",
]

# Measured feedback from validation rounds R6-R8 (codejudge + final2 tests).
SEED_HISTORY = [
    ("Let S = {1, 2, ..., 12}. How many subsets of S have an even element sum and contain no multiple of 3?", 8, "3/3", "all attempts same answer"),
    ("Sequence a1=2, a2=3, a_n=2a_{n-1}+a_{n-2}; find a_15 mod 100.", 8, "3/3", "all attempts same answer"),
    ("How many lattice points (x, y) with integer coordinates satisfy x^2 + y^2 <= 2025?", 1, "3/3", "7 of 8 attempts failed to finish"),
    ("A 4x4 grid: how many ways to place 6 indistinguishable tokens so no two share a row or column?", 8, "3/3", "all attempts same answer (trick question, answer 0)"),
    ("Find the number of integers n, 1<=n<=1000, with floor(n/2)+floor(n/3)+floor(n/5) = n-3.", 6, "3/3", "2 of 8 attempts failed to finish (answer 0)"),
    ("Color hexagon vertices with 2 colors, no two adjacent same, rotations identical: how many ways?", 8, "2/3", "all attempts same answer"),
    ("How many positive integer solutions (x,y), x<=100, satisfy x(x+1)(x+2)=y(y+1)(y+2)?", 8, "3/3", "all attempts same answer"),
    ("N = number of domino tilings of 3x5 rectangle; find N*3 mod 1000.", 7, "3/3", "trick question, answer 0"),
    ("How many subsets of the set {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} have a sum of elements exactly equal to 20?", 2, "3/3", "4 of 8 attempts failed to finish"),
    ("How many integer lattice points (x, y) satisfy the inequality x^2 + y^2 <= 50?", 8, "3/3", "all attempts same answer"),
    ("How many integers n with 1 <= n <= 100 satisfy the condition that floor(sqrt(n)) is a multiple of 3?", 7, "3/3", "all attempts answered"),
    ("How many positive integers n with 1 <= n <= 100 have exactly 3 prime factors when counted with multiplicity?", 8, "3/3", "all attempts same answer"),
    ("How many pairs of integers (x, y) with 1 <= x <= y satisfy the equation x^2 + y^2 = 50?", 8, "3/3", "all attempts same answer"),
    ("How many ways are there to arrange the letters of the word \"MISSISSIPPI\" such that no two 'S' characters are adjacent?", 8, "3/3", "all attempts same answer"),
    ("How many triples of distinct integers (a, b, c) chosen from the set {1, 2, ..., 10} have a sum that is divisible by 3?", 8, "3/3", "all attempts same answer"),
    ("A sequence is defined by a_1 = 1 and a_{n+1} = a_n + d(n), where d(n) is the number of positive divisors of n. Find a_20.", 8, "2/3", "all attempts same answer"),
    ("How many integer pairs (x, y) satisfy the inequality x^2 + y^2 <= 150 with the constraints x > 0 and y > 0?", 7, "3/3", "all attempts answered"),
    ("Consider the set {1, 2, ..., 12}. How many subsets of this set have a sum of elements that is divisible by 5?", 6, "3/3", "1 of 8 attempts failed to finish"),
    ("How many integers n between 1 and 500 have exactly 6 positive divisors?", 2, "3/3", "all attempts answered"),
    ("How many permutations of the set {1, 2, 3, 4, 5, 6, 7} have exactly two fixed points?", 8, "3/3", "all attempts same answer"),
    ("Let a sequence be defined by a_1 = 1 and a_{n+1} = a_n + (d(n))^2, where d(n) is the number of positive divisors of n. Find the value of a_{15}.", 8, "3/3", "all attempts same answer"),
    ("How many integer solutions (x, y) exist for the equation x^2 + y^2 = 325?", 7, "3/3", "all attempts answered"),
    ("Find the number of integers n with 1 <= n <= 80 such that floor(n/2) + floor(n/3) = n - 4.", 8, "3/3", "all attempts same answer"),
    ("How many distinct permutations of the letters in \"MISSISSIPPI\" are there such that all four 'I's are adjacent?", 7, "3/3", "all attempts same answer"),
]

CHAIN_FOCUS = [
    "Lean toward number theory: divisor counts, digit functions, modular constraints, factorials.",
    "Lean toward combinatorics: restricted permutations, subset counts with multiple constraints, arrangements.",
    "Lean toward geometric counting: lattice points, grid paths, polygon/region counting with constraints.",
    "Lean toward sequences and floor/mixed conditions: recurrences with arithmetic functions, floor equations, joint divisibility conditions.",
]


import random as _rnd

# Difficulty lean for the few-shot window: the demonstrations should be roughly the
# target difficulty so the proposer imitates ON-TARGET questions, not the easy majority.
# (Root cause of the v3 easy-drift: raw last-N window was ~60% 8/8 and got imitated.)
WINDOW_WEIGHT = {"mid": 2.2, "near": 1.3, "edge": 0.6}  # mid=2-4/8, near=5-6/8, edge=0/1 or 7/8


def _wt(passes):
    if 2 <= passes <= 4: return WINDOW_WEIGHT["mid"]
    if 5 <= passes <= 6: return WINDOW_WEIGHT["near"]
    return WINDOW_WEIGHT["edge"]


def select_window(history, n=60, recent_keep=12, seed=0):
    """Pick up to n feedback entries leaning toward on-target difficulty, but always
    keep the most recent `recent_keep` for continuity. Returned in chronological order."""
    if len(history) <= n:
        return history
    recent = history[-recent_keep:]
    pool = history[:-recent_keep]
    rng = _rnd.Random(seed + len(history))
    # weighted sample without replacement from the older pool
    idx = list(range(len(pool)))
    chosen = []
    weights = [_wt(pool[i][1]) for i in idx]
    k = n - len(recent)
    for _ in range(min(k, len(idx))):
        tot = sum(weights)
        r = rng.random() * tot
        acc = 0.0
        for j, w in enumerate(weights):
            acc += w
            if acc >= r:
                chosen.append(idx[j]); idx.pop(j); weights.pop(j); break
    keep = sorted(set(chosen)) 
    out = [pool[i] for i in keep] + list(recent)
    return out


def fmt_fb(entries):
    return "\n".join(
        "[pass %d/8 | verifier_programs_agree %s | %s] %s" % (p, prog, note, q)
        for q, p, prog, note in entries
    )


LABEL_PROMPT = (
    "Name the core mathematical structure of this question in 2-5 lowercase words "
    "(e.g. 'lattice point counting in disk', 'subset sum divisibility', 'floor function equation', "
    "'permutation fixed point count'). Output ONLY the label, nothing else.\n\nQuestion: %s"
)


def label_structure(q):
    out = chat(LABEL_PROMPT % q, 0.0, 1.0, 24, False).strip().lower()
    out = re.sub(r"[^a-z0-9 ]", "", out)
    return " ".join(out.split()[:6]) or "unlabeled"


def structure_report(labels):
    from collections import Counter
    c = Counter(labels)
    lines = [
        "STRUCTURE FREQUENCY REPORT - what you (all chains combined) have already produced, by structure.",
        "Why this matters: the solver trains on these questions; near-identical questions add almost no new",
        "training signal, so a batch's value comes from calibrated questions with NEW or RARE structures.",
        "Nothing is forbidden - use this information well.",
    ]
    for name, n in c.most_common(25):
        lines.append("- %s: %d produced so far" % (name, n))
    return "\n".join(lines)


def build_prompt(history_entries, focus, all_labels):
    return (
        "You are a math question designer inside a self-improvement loop. "
        "A strong solver model attempts each question 8 times. Each question is also verified by 3 "
        "independently written Python programs - if the programs do not agree, the question is DISCARDED "
        "as unverifiable and wastes the whole batch slot.\n\n"
        "Your target, in priority order:\n"
        "A. VERIFIABLE: a short brute-force Python program must be able to compute the answer. All 3 "
        "programs must agree. Single integer answer, unambiguous.\n"
        "B. CALIBRATED: solver passes about 3/8 (AIM LOWER - most recent questions were too easy). Recent batches averaged about 7/8 - STILL TOO EASY. "
        "Questions at 8/8 are wasted (too easy), 0/8 wasted (too hard). At least 5 of your 8 questions "
        "must be at least as hard as the past questions below that scored 4/8 or lower. Questions at 7/8 or 8/8 are WASTED - over half of recent questions were 8/8.\n"
        "C. DIVERSE: the structure frequency report below shows everything already produced. Treat it as "
        "feedback the same way you treat pass rates: heavily-produced structures are saturated, and "
        "parameter tweaks of saturated structures add little; rare or absent structures are where new "
        "value is. Do not reuse the answer-0-via-impossible-setup trick more than once per batch.\n"
        + focus + "\n\n"
        + structure_report(all_labels) + "\n\n"
        "GOLDEN TARGET EXAMPLES - these are the calibration anchor. Emulate their DIFFICULTY and\n"
        "style; aim for questions a strong solver gets right only ~3/8 of the time like these:\n"
        + "\n".join("- " + g for g in GOLDEN) + "\n\n"
        "DIFFICULTY GUIDE: routine textbook questions score 8/8 (wasted). Massive manual enumeration "
        "(6000+ items) scores 0-1/8 (mostly wasted). The 4/8 sweet spot is moderate enumeration or "
        "tricky multi-step case analysis - solvable by hand in ~4000 words but error-prone. Combining "
        "TWO constraints (e.g. a divisibility condition AND a digit condition) reliably increases "
        "difficulty while staying program-checkable.\n\n"
        "PAST QUESTIONS from ALL parallel chains with measured feedback (newest last):\n"
        + fmt_fb(history_entries) + "\n\n"
        "Write 8 NEW questions. Do NOT include solutions, answers, or hints.\n"
        "For each question output exactly two lines (replace the angle-bracket text):\n"
        "QUESTION: <question text>\n"
        "CONFIRM: program_checkable=<yes/no>; single_integer_answer=<yes/no>\n"
    )


def chat(p, temperature, top_p, max_tokens, thinking):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
        except Exception as e:
            if attempt == 3:
                return "ERROR: %s" % e
            time.sleep(10)


CODE_PROMPT = (
    "Write a standalone Python 3 program that computes the answer to this math question by direct "
    "computation or brute force. Self-contained, under 10 seconds, print EXACTLY one integer.\n\n"
    "Question:\n%s\n\nOutput only the Python code in a single ```python code block."
)


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m[-1] if m else None


def run_code(code):
    if code is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        out = subprocess.run(["python", path], capture_output=True, text=True, timeout=15)
        s = out.stdout.strip()
        m = re.fullmatch(r"-?\d+", s)
        return int(m.group(0)) if m else None
    except Exception:
        return None
    finally:
        os.unlink(path)


def code_judge(question):
    """One-shot 3-program majority (legacy judge)."""
    def one(_):
        return run_code(extract_code(chat(CODE_PROMPT % question, 0.6, 0.95, 4000, False)))

    with ThreadPoolExecutor(max_workers=3) as ex:
        outs = list(ex.map(one, range(3)))
    vals = [v for v in outs if v is not None]
    ref = max(set(vals), key=vals.count) if vals else None
    if ref is None or vals.count(ref) < 2:
        return None, outs
    return ref, outs


def tool_judge(question):
    """Tool-loop judge: 2 independent runs with a python tool; tiebreak run on disagreement.
    Requires vLLM started with --enable-auto-tool-choice --tool-call-parser qwen3_xml."""
    import python_tool_judge as PTJ

    def one(temp):
        try:
            ref, _, _ = PTJ.judge_reference(question, temperature=temp)
            return ref
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as ex:
        outs = list(ex.map(one, (0.2, 0.5)))
    if outs[0] is not None and outs[0] == outs[1]:
        return outs[0], outs
    outs.append(one(0.7))
    vals = [v for v in outs if v is not None]
    ref = max(set(vals), key=vals.count) if vals else None
    if ref is None or vals.count(ref) < 2:
        return None, outs
    return ref, outs


JUDGE_MODE = "tool"


def judge(question):
    return tool_judge(question) if JUDGE_MODE == "tool" else code_judge(question)


SOLVER_TMPL = (
    "Question:\n%s\n\nSolve the problem step by step, showing your reasoning concisely.\n"
    "End your response with one final line that is exactly:\nFINAL_ANSWER: <integer>"
)
FA = re.compile(r"FINAL_ANSWER:\s*(-?\d+)")


def parse_int(text):
    m = FA.findall(text)
    return m[-1] if m else None


def parse_questions(out):
    qs = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("QUESTION:"):
            q = s[len("QUESTION:"):].strip()
            if q and not q.startswith("<"):
                qs.append(q)
    return qs


_write_lock = threading.Lock()
_seen_questions: set[str] = set()
_shared_lock = threading.Lock()
_all_labels: list[str] = []  # structure label of every question produced, all chains + prior bank


def register_question(q):
    label = label_structure(q)
    with _shared_lock:
        n_same = sum(1 for x in _all_labels if x == label)
        _all_labels.append(label)
        return label, n_same


def write_records(out_path, records):
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            for r in records:
                key = r["question"].lower()
                if key in _seen_questions:
                    r["trainable_for_solver"] = False
                    r["duplicate"] = True
                else:
                    _seen_questions.add(key)
                f.write(json.dumps(r) + "\n")


def run_chain(chain_idx, rounds, out_path, shared_history):
    tag = "C%d" % chain_idx
    history = shared_history  # SHARED list across chains (append under lock)
    focus = CHAIN_FOCUS[chain_idx % len(CHAIN_FOCUS)]
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        with _shared_lock:
            hist_snapshot = list(history)
            labels_snapshot = list(_all_labels)
        prompt = build_prompt(select_window(hist_snapshot, n=60), focus, labels_snapshot)
        out = chat(prompt, 0.9, 0.95, 16000, True)
        dump_dir = os.path.join(os.path.dirname(out_path) or ".", "prompts")
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "%s_R%d.txt" % (tag, rnd)), "w", encoding="utf-8") as f:
            f.write("===== PROMPT =====\n" + prompt + "\n\n===== RAW OUTPUT =====\n" + out)
        qs = parse_questions(out)
        print("%s R%d proposer %.0fs parsed=%d" % (tag, rnd, time.time() - t0, len(qs)), flush=True)
        if not qs:
            print("%s R%d RAW OUTPUT HEAD: %r" % (tag, rnd, out[:300]), flush=True)
            continue
        jobs = [(qi, s) for qi in range(len(qs)) for s in range(K)]
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=16) as ex:
            sols = list(ex.map(lambda j: chat(SOLVER_TMPL % qs[j[0]], 1.1, 0.98, 8192, False), jobs))
        t2 = time.time()
        refs = {qi: judge(q) for qi, q in enumerate(qs)}
        print("%s R%d solver %.0fs judge %.0fs" % (tag, rnd, t2 - t1, time.time() - t2), flush=True)
        records = []
        for qi, q in enumerate(qs):
            ref, prog_outs = refs[qi]
            answers = [parse_int(sols[i]) for i, j in enumerate(jobs) if j[0] == qi]
            n_none = sum(1 for a in answers if a is None)
            if ref is None:
                passes = 0
                verif = "PROGRAMS DISAGREE - DISCARDED"
            else:
                passes = sum(1 for a in answers if a is not None and int(a) == ref)
                verif = "verifier_programs_agree %d/3" % sum(1 for v in prog_outs if v == ref)
            note = "%d of 8 attempts failed to finish" % n_none if n_none else "all attempts answered"
            label, n_same = register_question(q)
            note += "; structure: %s (%s)" % (label, "novel" if n_same == 0 else "%d produced before" % n_same)
            with _shared_lock:
                history.append((q, passes, verif.replace("verifier_programs_agree ", ""), note))
            trainable = ref is not None and 1 <= passes <= (K - 1)
            print(
                "%s R%d Q%d ref=%s progs=%s pass=%d/%d trainable=%s | %s"
                % (tag, rnd, qi + 1, ref, prog_outs, passes, K, trainable, q[:90]),
                flush=True,
            )
            records.append(
                {
                    "question": q,
                    "reference_answer": ref,
                    "pass_count": passes,
                    "k": K,
                    "program_outputs": prog_outs,
                    "solver_answers": answers,
                    "chain": chain_idx,
                    "round": rnd,
                    "structure_label": label,
                    "structure_seen_before": n_same,
                    "trainable_for_solver": trainable,
                }
            )
        write_records(out_path, records)
        novel = sum(1 for r in records if r["structure_seen_before"] == 0)
        print("%s R%d DIVERSITY: %d/%d novel structures | labels: %s" % (tag, rnd, novel, len(records), [r["structure_label"] for r in records]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", type=str, default="outputs/agentic_bank/bank.jsonl")
    ap.add_argument("--history_bank", type=str, default="",
                    help="Prior cycle bank JSONL; seeds feedback history and overuse stats.")
    ap.add_argument("--judge", type=str, default="tool", choices=["tool", "programs"],
                    help="tool = judge model iterates with a python tool (needs vLLM tool-calling); programs = legacy 3-program one-shot majority.")
    args = ap.parse_args()
    global JUDGE_MODE
    JUDGE_MODE = args.judge

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Resume support: pre-load existing questions so duplicates stay filtered.
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _seen_questions.add(json.loads(line)["question"].lower())
        print("resuming: %d existing questions in bank" % len(_seen_questions), flush=True)

    shared_history = list(SEED_HISTORY)
    if args.history_bank and os.path.exists(args.history_bank):
        prior = [json.loads(l) for l in open(args.history_bank, encoding="utf-8") if l.strip()]
        with ThreadPoolExecutor(max_workers=24) as ex:
            prior_labels = list(ex.map(lambda r: r.get("structure_label") or label_structure(r["question"]), prior))
        with _shared_lock:
            _all_labels.extend(prior_labels)
        # newest cycle's measurements replace the hardcoded seed entirely
        shared_history = [
            (r["question"], r.get("pass_count", 0),
             "3/3" if r.get("reference_answer") is not None else "PROGRAMS DISAGREE - DISCARDED",
             "structure: " + lbl)
            for r, lbl in zip(prior[-120:], prior_labels[-120:])
        ]
        print("seeded history from %s: %d prior questions, %d distinct structures"
              % (args.history_bank, len(prior), len(set(prior_labels))), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.chains) as ex:
        list(ex.map(lambda c: run_chain(c, args.rounds, args.out, shared_history), range(args.chains)))

    total = 0
    trainable = 0
    with open(args.out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            trainable += 1 if row.get("trainable_for_solver") else 0
    print(
        "==== DONE %.0fs | bank=%d questions | trainable=%d ====" % (time.time() - t0, total, trainable),
        flush=True,
    )


if __name__ == "__main__":
    main()
