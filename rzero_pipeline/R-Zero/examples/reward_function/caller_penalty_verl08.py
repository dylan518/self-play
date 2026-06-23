# verl 0.8 mainline batch-reward adapter for R-Zero challenger reward.
# The batch reward manager (@register("batch")) calls:
#   compute_score(data_sources=, solution_strs=, ground_truths=, extra_infos=)
# once per batch (lists), merging reward_kwargs from
# reward.custom_reward_function.reward_kwargs. R-Zero core returns dicts keyed
# "overall"; verl 0.8 reads score["score"]. This shim maps solution_strs->predicts
# and overall->score, leaving all core logic (services solve + program verify +
# diversity penalty + verifiability term) untouched.
import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caller_penalty import compute_score as _core


def compute_score(data_sources=None, solution_strs=None, ground_truths=None,
                  extra_infos=None, format_weight=0.1, file_path="", **kwargs):
    predicts = list(solution_strs or [])
    if ground_truths is None:
        gts = ["" for _ in predicts]
    else:
        gts = list(ground_truths)
    out = _core(predicts, gts, format_weight, file_path)
    for d in out:
        if isinstance(d, dict) and "overall" in d and "score" not in d:
            d["score"] = d["overall"]
    return out
