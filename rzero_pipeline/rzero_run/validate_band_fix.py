import json
rows=[json.loads(l) for l in open("/data/selfplay/recon_it2_solver.jsonl")]
n=len(rows)
# current band (self-consistency) admitted ALL these 710 (they passed it). solve_rate = correctness vs program label.
def band(lo,hi): return sum(1 for r in rows if lo < r["solve_rate"] < hi)
zero=sum(1 for r in rows if r["solve_rate"]==0.0)
full=sum(1 for r in rows if r["solve_rate"]==1.0)
learn_02_08=band(0.2,0.8)
learn_0_1=sum(1 for r in rows if 0.0 < r["solve_rate"] < 1.0)
print(f"iter2 training set (passed the OLD self-consistency band): n={n}")
print(f"  solve_rate==0 (solver always WRONG, zero gradient): {zero} ({zero/n:.1%})")
print(f"  solve_rate==1 (solver always RIGHT, zero gradient):  {full} ({full/n:.1%})")
print(f"  ZERO-ADVANTAGE total: {zero+full} ({(zero+full)/n:.1%}) <- the dead weight the OLD band let in")
print(f"  learnable 0<sr<1 (any gradient):     {learn_0_1} ({learn_0_1/n:.1%})")
print(f"  CORRECTNESS-VARIANCE band (0.2,0.8): {learn_02_08} ({learn_02_08/n:.1%}) <- what the FIXED band keeps")
print(f"\nEffect: fixed band drops {n-learn_02_08} dead/near-dead questions, trains on {learn_02_08} high-signal ones.")
