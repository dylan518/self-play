import ast

# --- evaluate.py: save full solution texts, not just extracted answers ---
p = "/work/pi_general_dartmouth_edu/dylan/R-Zero/question_evaluate/evaluate.py"
src = open(p).read()

a1 = """        try:
            # Extract the boxed content from all generated samples
            results = [extract_boxed_content(output.text) for output in response.outputs]"""
r1 = """        try:
            # Extract the boxed content from all generated samples
            solution_texts = [output.text for output in response.outputs]
            results = [extract_boxed_content(output.text) for output in response.outputs]"""
assert a1 in src, "anchor1"
src = src.replace(a1, r1, 1)

a2 = """            results_all.append({
                "question": question,
                "answer": majority_answer,
                "score": score,
                'results': results
            })"""
r2 = """            results_all.append({
                "question": question,
                "answer": majority_answer,
                "score": score,
                'results': results,
                # Full solver reasoning texts: Stage B already paid for these rollouts —
                # they are the free training data for solver RL/SFT. Never discard.
                'solution_texts': solution_texts,
            })"""
assert a2 in src, "anchor2"
src = src.replace(a2, r2, 1)
ast.parse(src)
open(p, "w").write(src)

# --- upload.py: archive results files instead of deleting them ---
p2 = "/work/pi_general_dartmouth_edu/dylan/R-Zero/question_evaluate/upload.py"
src2 = open(p2).read()
a3 = """        os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')"""
r3 = """        # Archive (never delete): these files carry the full solver rollout texts.
        os.makedirs(f'{STORAGE_PATH}/generated_question/raw_results', exist_ok=True)
        os.replace(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json',
                   f'{STORAGE_PATH}/generated_question/raw_results/{args.experiment_name}_{i}_results.json')"""
assert a3 in src2, "anchor3"
src2 = src2.replace(a3, r3, 1)
ast.parse(src2)
open(p2, "w").write(src2)
print("BOTH_PATCHES_OK")
