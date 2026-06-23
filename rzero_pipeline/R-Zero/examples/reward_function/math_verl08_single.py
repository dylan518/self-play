# Per-sample solver reward for verl 0.8's experimental reward loop (naive manager).
# The reward loop calls run_single -> compute_score(data_source, solution_str,
# ground_truth, extra_info) once per rollout. Math grading is per-sample, so we
# reuse R-Zero's math.py grading on a 1-element batch and return {"score": ...}.
import os
import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("rzero_math_reward", os.path.join(_here, "math.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_core = _mod.compute_score


def compute_score(data_source=None, solution_str="", ground_truth="",
                  extra_info=None, format_weight=0.1, **kwargs):
    out = _core([solution_str], [str(ground_truth)], format_weight)
    d = out[0] if out else {}
    return {
        "score": float(d.get("overall", 0.0)),
        "format": float(d.get("format", 0.0)),
        "accuracy": float(d.get("accuracy", 0.0)),
    }
