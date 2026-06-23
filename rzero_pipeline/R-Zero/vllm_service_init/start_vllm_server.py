#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import multiprocessing as _mp
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit  # 1. Import the thread-safe 'stopit' library
import random
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "question_evaluate"))
from verify import extract_code, run_program, program_consensus, build_code_prompts, VERIFY_TEMPERATURE, self_critique_keep, SELF_FILTER

# Fraction of each request's questions that get a program-consensus verifiability
# check (0 disables; the challenger reward then matches upstream exactly).
VERIFY_SUBSAMPLE = float(os.getenv("VERIFY_SUBSAMPLE", "0"))
VERIFY_K_PROGRAMS = int(os.getenv("VERIFY_K_PROGRAMS", "3"))

# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
# Heavy init is spawn-unsafe at module level: with VLLM_WORKER_MULTIPROC_METHOD=spawn
# the engine-core child re-imports this module. Defer to _init_model() called only
# under the __main__ guard so the child does not re-create the vLLM engine.
tokenizer = None
model = None
sample_params = None


def _init_model():
    global tokenizer, model, sample_params
    print('[init] Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=float(os.getenv("SERVICE_GPU_MEM", str(args.gpu_mem_util))),
        # Qwen3.5 is hybrid mamba-attention: needs eager + bounded context (Milestone 1).
        enforce_eager=(os.getenv("WORKER_ENFORCE_EAGER","0")=="1"),
        dtype="bfloat16",
        max_model_len=int(os.getenv("SERVICE_MAX_MODEL_LEN", "8192")),
    )
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        stop_token_ids=[tokenizer.eos_token_id],
        n=10,
    )

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
# idle_thread.start() moved into __main__ guard (spawn-safe; touches CUDA)

# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # (vLLM generation logic remains unchanged)
    if valid_chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True,
                                              enable_thinking=False)
                for chat in valid_chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    def process_single(question, golden_answer, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        # print(f"[process_single] Processing question: '{question[:70]}...'")

        answer_counts = {}
        for res in results:
            if not res: continue # Skip empty results
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop over exist_ans
                
                # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
                try:
                    is_match = False
                    # First direction: res vs exist_ans
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    # Second direction (only if first failed): exist_ans vs res
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                             # Log timeout for the second direction as well
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop

                except Exception as e:
                    # Catch any other potential errors from the grader function itself.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue # Continue to the next comparison in the inner loop
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,
            'results':  results
        }

    results_all = []
    response_idx = 0
    for q, a in zip(questions, answers):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, a, response)
                results_all.append(item)
            else:
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')

    # ---------- Optional per-batch verifiability judge (subsample) ----------
    # Every challenger batch gets verifiability feedback at the same cadence as
    # the difficulty (majority-score) feedback above.
    if VERIFY_SUBSAMPLE > 0:
        candidates = [i for i, it in enumerate(results_all) if it.get('question') and it.get('score', -1) >= 0]
        k = max(1, int(len(candidates) * VERIFY_SUBSAMPLE)) if candidates else 0
        sampled = sorted(random.sample(candidates, k)) if k else []
        if sampled:
            code_chats = []
            for i in sampled:
                for p in build_code_prompts(results_all[i]['question'], VERIFY_K_PROGRAMS):
                    code_chats.append([{'role': 'user', 'content': p}])
            if tokenizer.chat_template:
                code_prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in code_chats]
            else:
                code_prompts = ['user: ' + c[0]['content'] for c in code_chats]
            code_params = vllm.SamplingParams(
                max_tokens=2048, temperature=VERIFY_TEMPERATURE, top_p=0.95, n=1,
                stop_token_ids=[tokenizer.eos_token_id],
            )
            code_responses = model.generate(code_prompts, sampling_params=code_params, use_tqdm=True)
            from concurrent.futures import ThreadPoolExecutor as _TPE
            _codes = [extract_code(r.outputs[0].text) for r in code_responses]
            with _TPE(max_workers=16) as _ex:
                _outs = list(_ex.map(run_program, _codes))
            # GRADED verifiability by program-agreement strength: 2*votes/K - 1 in [-1, 1].
            # 0 programs agree -> -1 (degenerate/uncheckable); all K agree -> +1 (robustly verifiable).
            def _graded(votes):
                return (2.0 * float(votes) / max(1, VERIFY_K_PROGRAMS)) - 1.0
            consensus = []
            for j, i in enumerate(sampled):
                outs = _outs[j * VERIFY_K_PROGRAMS:(j + 1) * VERIFY_K_PROGRAMS]
                ca, votes = program_consensus(outs)
                results_all[i]['program_outputs'] = outs
                results_all[i]['votes'] = int(votes)
                consensus.append((i, ca, int(votes)))
            if SELF_FILTER:
                # Graded among well-posed, checkable questions; full penalty (-1) for
                # uncheckable (no consensus) OR self-critique-rejected (ill-posed) ones.
                checkable = [(i, ca) for i, ca, _v in consensus if ca is not None]
                keep = self_critique_keep(
                    model, tokenizer, [(results_all[i]['question'], ca) for i, ca in checkable]
                ) if checkable else []
                keepmap = {checkable[t][0]: keep[t] for t in range(len(checkable))}
                for i, ca, votes in consensus:
                    if ca is None or not keepmap.get(i):
                        results_all[i]['verified'] = -1.0
                    else:
                        results_all[i]['verified'] = _graded(votes)
            else:
                for i, ca, votes in consensus:
                    results_all[i]['verified'] = _graded(votes)
        n_pos = sum(1 for it in results_all if it.get('verified', 0) > 0)
        n_neg = sum(1 for it in results_all if it.get('verified', 0) < 0)
        _vv = [it['verified'] for it in results_all if it.get('verified') is not None]
        _mv = (sum(_vv) / len(_vv)) if _vv else 0.0
        print(f'[server] verifiability(graded): {len(sampled)} judged, {n_pos} pos, {n_neg} neg, mean={_mv:.2f}.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__' and _mp.parent_process() is None:
    _init_model()
    idle_thread.start()
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
