import ast

p = "/work/pi_general_dartmouth_edu/dylan/R-Zero/question_evaluate/evaluate.py"
src = open(p).read()

a1 = """        # Clean up the input file immediately after loading to save space
        os.remove(INPUT_FILE)"""
r1 = """        # Keep the input file: it is the ONLY copy of the generated questions until
        # results are written (deleting it made mid-run inspection impossible)."""
assert a1 in src, "anchor1"
src = src.replace(a1, r1, 1)

a2 = """    responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    print(f"[{args.suffix}] Generation complete.")

    # 4. Process and Grade Responses
    results_all = []
    print(f"[{args.suffix}] Grading responses...")
    for response, golden_answer, question in zip(responses, answers, questions):"""
r2 = """    # Chunked generate+grade: partial results land in OUTPUT_FILE.partial every
    # EVAL_CHUNK_QUESTIONS questions so long evals are inspectable mid-flight.
    CHUNK_Q = int(os.getenv("EVAL_CHUNK_QUESTIONS", "50"))
    results_all = []
    for _c0 in range(0, len(prompts), CHUNK_Q):
      responses = model.generate(prompts[_c0:_c0+CHUNK_Q], sampling_params=sample_params, use_tqdm=True)
      print(f"[{args.suffix}] chunk {_c0}..{_c0+len(responses)} generated; grading...", flush=True)
      for response, golden_answer, question in zip(responses, answers[_c0:_c0+CHUNK_Q], questions[_c0:_c0+CHUNK_Q]):"""
assert a2 in src, "anchor2"
src = src.replace(a2, r2, 1)

a3 = """        except Exception as e:
            print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
            continue

    # 5. Save Final Results"""
r3 = """        except Exception as e:
            print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
            continue
      with open(OUTPUT_FILE + ".partial", "w") as f:
          json.dump(results_all, f)
      print(f"[{args.suffix}] partial: {len(results_all)} scored through question {min(_c0+CHUNK_Q, len(prompts))}/{len(prompts)}", flush=True)

    # 5. Save Final Results"""
assert a3 in src, "anchor3"
src = src.replace(a3, r3, 1)

ast.parse(src)
open(p, "w").write(src)
print("PATCH_OK + SYNTAX_OK")
