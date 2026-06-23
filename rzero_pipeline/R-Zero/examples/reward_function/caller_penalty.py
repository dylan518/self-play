# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex as re
from typing import Dict, List
import json
from mathruler.grader import extract_boxed_content, grade_answer
import os
import sys as _sys
try:
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _sys.path.insert(0, os.path.join(os.getcwd(), 'examples', 'reward_function'))
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
STORAGE_PATH = os.getenv("STORAGE_PATH","/apdcephfs_sh2/share_300000800/user/chengchuang")
N_SERVICES = int(os.getenv("N_SERVICES", "4"))
# Base port for the vLLM solver services; service i listens on SERVICE_PORT_BASE+i.
# Parameterized so two arms can run concurrently on one box (verified=5000, majority=5100).
SERVICE_PORT_BASE = int(os.getenv("SERVICE_PORT_BASE", "5000"))
# Weight of the per-batch program-verifiability term in the challenger reward.
# 0 (default) = upstream baseline reward. The judged subset is controlled by
# VERIFY_SUBSAMPLE on the solver services; unjudged questions receive the batch
# mean so the term shifts only judged questions' advantages.
# Graded in [-1, 1] (2*votes/K - 1): worth a little more than the diversity/difficulty
# terms so verifiable, well-posed questions dominate the challenger's gradient.
VERIFY_WEIGHT = float(os.getenv("VERIFY_WEIGHT", "0.75"))
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "")
def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    _d = f"{STORAGE_PATH}/temp_results"
    os.makedirs(_d, exist_ok=True)
    return f"{_d}/{prefix}_{timestamp}_{rand_part}{suffix}"
def split_list(lst, n=N_SERVICES):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

def fetch(index,i):
    response = requests.get(f"http://0.0.0.0:{SERVICE_PORT_BASE+index}/hello?name={i}")
    print(response)
    return True

def generate_results(data):
    datas = split_list(data,N_SERVICES)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(N_SERVICES)]
    for i in range(N_SERVICES):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=N_SERVICES) as executor:
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(N_SERVICES)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(N_SERVICES):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(N_SERVICES):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results

def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, file_path: str = "") -> List[Dict[str, float]]:
    CLIP_EASY = float(os.environ.get("CLIP_EASY", "0"))  # clip reward to 0 when solver_score > CLIP_EASY (too easy)
    results = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)
    for i in range(len(predicts)):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    final_results = generate_results(results)
    # Per-sample Vendi-score (Nyström) diversity REWARD: each question's marginal
    # contribution to the diversity of all past samples (relative delta it adds to the
    # bank). Near-duplicate / degenerate questions add ~0; novel ones add ~full reward.
    from diversity import vendi_diversity_rewards
    diversity_reward, div_stats = vendi_diversity_rewards([result['question'] for result in final_results])
    assert len(diversity_reward) == len(final_results)

    # Per-batch GRADED verifiability term. verified in [-1, 1] = 2*votes/K - 1 (program
    # agreement strength): 0 programs agree -> -1 (degenerate/uncheckable), all K agree -> +1.
    judged = [r.get("verified") for r in final_results if r.get("verified") is not None]
    mean_verified = sum(judged) / len(judged) if judged else 0.0
    scores = []
    for i in range(len(final_results)):
        uncertainty = (min(final_results[i]["score"],1-final_results[i]["score"]) if final_results[i]['question'] else -1)
        verified = final_results[i].get("verified")
        # GRPO subtracts the per-GROUP baseline, so within-group variance drives the
        # gradient. Unjudged questions (verified None) get the batch mean = neutral. Judge
        # ALL questions (VERIFY_SUBSAMPLE=1.0) so groups have verifiability contrast.
        v = verified if verified is not None else mean_verified
        verif_term = VERIFY_WEIGHT * v
        final_score = uncertainty + diversity_reward[i] + verif_term
        if CLIP_EASY > 0 and final_results[i]['question'] and final_results[i]["score"] > CLIP_EASY:
            final_score = 0.0  # too easy: zero out the whole reward
        scores.append({"overall": final_score,"format": 1 if final_results[i]['question'] else 0,"accuracy": diversity_reward[i]})

    if ARTIFACTS_DIR:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, "challenger_batches.md")
        new_file = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            if new_file:
                f.write("# Challenger batches (every GRPO reward call)\n\n")
            f.write(f"## batch @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | n={len(final_results)} | mean_verified={mean_verified:.2f} (judged {len(judged)}) | vendi {div_stats['vs_before']:.1f}->{div_stats['vs_after']:.1f} (rel_delta {div_stats['rel_delta']:+.3f}, mean_novelty {div_stats['mean_novelty']:.2f}) | gate {div_stats['gate']:.2f} (bank_matched {div_stats['bank_matched']:.1f} / golden {div_stats['golden']:.1f})\n\n")
            f.write("| solver_score | verified(graded) | votes | diversity | reward | question |\n|---|---|---|---|---|---|\n")
            for i, r in enumerate(final_results):
                q = (r.get("question") or "").replace("|", "\\|").replace("\n", " ")[:160]
                v = r.get("verified")
                f.write(f"| {r.get('score', -1):.2f} | {'-' if v is None else f'{v:+.2f}'} | {r.get('votes', '-')} | {diversity_reward[i]:.2f} | {scores[i]['overall']:.3f} | {q} |\n")
            f.write("\n")
    return scores








