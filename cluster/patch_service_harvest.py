import ast

p = "/work/pi_general_dartmouth_edu/dylan/R-Zero/vllm_service_init/start_vllm_server.py"
src = open(p).read()

# 1) process_single: carry full solution texts
a1 = """        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,
            'results':  results
        }"""
r1 = """        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,
            'results':  results,
            'sol_texts': [out.text for out in response.outputs],
        }"""
assert a1 in src, "anchor1"
src = src.replace(a1, r1, 1)

# 2) consensus loop: record the program-consensus answer on the item
a2 = """                results_all[i]['program_outputs'] = outs
                results_all[i]['votes'] = int(votes)"""
r2 = """                results_all[i]['program_outputs'] = outs
                results_all[i]['votes'] = int(votes)
                results_all[i]['verified_answer'] = ca"""
assert a2 in src, "anchor2"
src = src.replace(a2, r2, 1)

# 3) before writing results back to the caller: dump the harvest, strip bulky texts
a3 = """    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)"""
r3 = """    # SOLVER HARVEST: Stage-A reward calls already pay for n solver rollouts per
    # question — dump them (with labels) so solver replay-RL trains at zero extra
    # rollout cost. Inert unless HARVEST_DIR is set. Texts stripped from the caller
    # payload either way.
    _hdir = os.getenv("HARVEST_DIR")
    if _hdir:
        os.makedirs(_hdir, exist_ok=True)
        with open(os.path.join(_hdir, "solver_harvest.jsonl"), "a") as _hf:
            for _it in results_all:
                _hf.write(json.dumps({
                    "question": _it.get("question"), "majority_answer": _it.get("answer"),
                    "score": _it.get("score"), "score_cv": _it.get("score_cv"),
                    "verified": _it.get("verified"), "votes": _it.get("votes"),
                    "verified_answer": _it.get("verified_answer"),
                    "sol_answers": _it.get("results"), "sol_texts": _it.get("sol_texts"),
                }) + "\\n")
    for _it in results_all:
        _it.pop("sol_texts", None)

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)"""
assert a3 in src, "anchor3"
src = src.replace(a3, r3, 1)

ast.parse(src)
open(p, "w").write(src)
print("SERVICE_HARVEST_PATCH_OK")
