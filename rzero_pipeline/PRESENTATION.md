# Self-Play for Math Reasoning: the Full Arc
### Qwen3 R-Zero → verified labels → RL collapse → selected-SFT
*Presentation source material, with real examples pulled from run data. (June–July 2026)*

---

## 0. Timeline at a glance

| phase | what | outcome |
|---|---|---|
| **P1** original self-play (this repo, `grpo_math`) | generator/solver/judge, pairwise Elo, GSM8K | judge-reliability metrics built; question-DPO blocked (holdout 0.71) |
| **P2** R-Zero replication — **Qwen3-4B** | paper's model + majority-vote pseudo-labels | eval ≈ 0.30 (matches external 0.33); base too weak — migrated |
| **P3** R-Zero on **Qwen3.5-4B**, two arms | **majority** (vanilla R-Zero) vs **verified** (program labels) | verified clearly wins; majority labels wrong 18% |
| **P4** the failures | U-shape, favorable-draw, band footgun, packing confound | iteration is *unreliable*: redo landed **below base** |
| **P5** the collapse investigation | surface-vs-skill diversity, Goodhart mechanisms | every RL run skill-collapsed; metrics hid it |
| **P6 (now)** **selected-SFT questioner** | curate → SFT from base → measure → repeat | v3: best diversity of campaign + verifier fixed |

⚠ Footgun that shaped P2→P3: **cross-base comparisons are invalid** (Qwen3 ≈0.30 vs Qwen3.5 ≈0.573 on the same eval — always check `config.json model_type` before comparing absolutes).

---

## 1. The pipeline (R-Zero, our extension)

```
A: questioner GRPO (6 steps; reward = difficulty-band + diversity + verifiability)
B: trained questioner → ~1–3k fresh Qs
   → band filter: solver self-consistency ∈ [0.3, 0.8]  (n=9 samples)
   → LABEL:  majority arm  = solver's majority answer   (vanilla R-Zero)
             verified arm  = K-program consensus        (our change — the ONLY diff)
C: solver GRPO on the curriculum (20 steps, n=5)
eval: OlympiadBench @8192 greedy — always report format_rate / acc|fmt / pass
```

---

## 2. Majority vs Verified: why program labels

**The solver's majority vote is wrong 18% of the time** (nopack: 168/946 relabeled by programs; sftsel: 21%). Real examples — solver-majority confidently wrong, 3/3 programs agree on the truth:

> *"…how many n ≤ 1000 where n³−n contains a digit 7?"* — majority said **500**, programs computed **491** `['491','491','491']`
> *"…n ≤ 1000 containing digit 7, sum f(n)…"* — majority said **667**, programs **739**
> *"…n ≤ 2023 with digit 7 and digit …"* — majority said **382**, programs **358**

**Head-to-head (OlympiadBench n=200, same protocol):**

| arm | eval | vs base |
|---|---|---|
| base Qwen3.5-4B | ~0.33 | — |
| **majority (vanilla R-Zero)** | 0.345 | **+1.5pp** (noise) |
| **verified, step-20** | **0.415** | **+8.5pp** (~2.4σ) |

Program-consensus labeling is the single change that made self-play train on *true* signal. (MATH-500: +3.8pp.) Verified-label accuracy, audited by independent re-solving: **98%** on the digit curriculum, 92% on diverse; failure mode = all K programs sharing one misreading of an ambiguous question.

---

## 3. R-Zero failures (why iteration didn't compound)

**3.1 The U-shape.** iter1 verified = **0.630**, iter2 = **0.545** — *below base 0.573*. Step-1 repro on iter1's exact inputs matched (0.554 ≈ 0.565) ⇒ the drop is in the **inputs** (questioner's curriculum), not code.

**3.2 The favorable-draw discovery (the deepest failure).** Re-running iter1 from base with byte-identical config → **0.474, below base**. The questioner is non-deterministic (diversity-bank cold start, wall-clock reward batching, stochastic n=10 scoring) and the original 0.630 was a **lucky curriculum draw**. Same config, three outcomes: 0.630 / 0.474 / (later) 0.6415.

**3.3 The band footgun.** The difficulty band filters on **self-consistency, not correctness** — a confidently-*wrong* question reads "easy" and passes/fails wrongly; iter2 had ~45% zero-gradient rows. The correctness-band fix (CVBAND): recovered 0.474→0.545 (**safety**, eliminates below-base collapse) but iter2 then found **0 learnable questions** (kept=0) → the loop **stalls instead of regressing**. Safe, not beneficial.

**3.4 Infrastructure confound.** `use_remove_padding=true` (sequence packing) silently corrupts Qwen3.5's hybrid-attention gradients: **−0.12 eval**, reproduced 2×2 across attention backends (no-pack 0.621/0.633 vs pack 0.508/0.495). Never in the real loop, but it corrupted a week of repro runs. *Lesson: isolate infra changes from experiment changes.*

**Final RL scoreboard (full Oly-675 @8192):** base 0.573 · **nopack 0.6415** · vfix 0.6296 · divfix 0.6326 — a 3-way statistical tie regardless of the diversity knob. Why? →

---

## 4. Diversity collapse — with the actual questions

**The questioner collapses to a digit-manipulation monoculture under RL.** Real samples from the nopack run:

**Step 1 (base questioner, before RL) — genuinely varied:**
> • *"Let S be the set of integer solutions (x,y) to (x²−y²)(x+y)³=0 … define a sequence aₙ …"*
> • *"…triples (x,y,z) ≤ 2023, pairwise distinct, x²+y²+z² divisible by 20…"*
> • *"…n ≤ 1000 where trailing zeros in the **binary representation of n!** equal …"*
> • *"…sequence a₁=n, a_{k+1}=(a_k²+1) mod 2025, with a₁₀₀ = a₀ …"*

**Step 6 (after 6 GRPO steps) — one template, re-rolled:**
> • *"…n ≤ 1000 whose decimal representation contains **a digit 7 and a digit 9** …"*
> • *"…n ≤ 2023 containing **a digit 5 and a digit 7** …"*
> • *"…n ≤ 1000 where the **sum of digits equals the product of digits** …"*
> • *"…n ≤ 1000 where the **digit sum divides n** …"*

**And the metrics lied about it.** Embedding-Vendi and dedup stayed flat while the LLM-judged *skill* distribution collapsed:

```
divfix step:                  1     2     3     4     5     6
Vendi (unique, embedding):  21.4  21.1  21.8  22.1  22.4  21.1   "diverse" ✗
exact-unique questions:     100%  100%  100%  100%  100%  100%   "no dupes" ✗
digit-manipulation share:    25%   30%   35%   40%   50%   55%   ← doubled
effective skills (exp-H):    6.2   5.0   6.5   5.2   4.4   3.6   ← halved
```
*Varied wording, one skill.* Diversity must be measured as **exp-entropy over LLM-judged skill tags**, not embeddings.

**Why RL collapses (mechanisms, all measured):**
- The **graded verify reward** (`2·votes/K − 1`) pays 3× more for a degenerate always-verifiable question than a borderline one — and the 3/3-unanimous set is 2.5× more templated. *The diversity lives exactly where the reward punishes.*
- **Selection on a noisy proxy** (3 programs @ temp 0.6) + iteration = optimizer's-curse ratchet into the degenerate corner.
- The usable question is the **conjunction difficult ∧ verifiable** — anti-correlated axes; per-step usable yield was only 4–10% while the reward drove verifiability 16%→94% *by collapsing*.

---

## 5. The pivot: selected-SFT (P6, current)

**Principle: RL only where reward = true objective (solver correctness). For question *quality* — a proxy — control the distribution by construction: curate a golden set, SFT the questioner from base, measure everything, rebalance, repeat.** No reward to game, no ratchet (always retrain *from base*), ~30 min/iteration.

| | base | **v1** (naive verified target) | **v2** (skill-quota) | **v3** (skill × difficulty strata) |
|---|---|---|---|---|
| digit share | 25% | **50%** ⚠ collapse | 20% | 23% |
| eff-skills | 6.2 | 4.9 ⚠ | 6.6 | **7.5** ✅ |
| trivial/sweet/hard | 0/25/75 | — | **48**/28/23 ⚠ easy-skew | 37/33/30 ✅ |
| verifiable (strong) | 92% | 92% | 92% | 94% |
| joint (usable) | ~18% | — | 20% | **23%** |

- **v1 re-created the diversity collapse *without RL***: the verify-filter's selection bias (base 25% → verified-subset 39% digit) + SFT mode-amplification (39% → 50%). Both mechanisms measured separately.
- **v2** fixed skill; exposed the second bias axis (verifiability selects for *easiness* → 48% trivial).
- **v3** stratified both axes + self-distilled its own verified sweet-spots. General law: **stratify the SFT target on every axis the downstream filter selects on.**
- Remaining ceiling: P(well-posed | hard) ≈ 70%, flat across targets — a 4B generation-capability limit. Real failed examples: *"exactly 4 distinct a mod 7 with x²≡a"* (max is 2), *"2n ≡ 1 mod 12"* (no solutions), *"50k!"* (precedence ambiguity). Next lever: scratchpad self-check at generation.

**Pipeline-real result (Stage B, 3,000 generations from the SFT questioner):**
```
in-band (real n=9 band):     301/3,000 = 10%    (easy-skew tax — v4's target)
K=10 MIN_AGREE=6 consensus:  249/301  = 83%     ← verifier bottleneck FIXED (was 16–48% @K=3)
TRAINABLE: 249 = 8.3% at full diversity          vs divfix RL: 21% but 77%-digit monoculture
```

**K=10 verification in action (persisted for the first time)** — *"How many N, 0≤N≤99, have #(multiples of 5 below N) = #(multiples of 7 below N)?"* → 10 programs: `[9,9,9,9,9,`**`10`**`,9,9,9,9]` → votes 9/10, label **9** (the dissenter mishandled the N=0 boundary). Consensus catches exactly this.

**Cost:** learnable questions per GPU·hour at held diversity — RL ~10–40 (rollouts then discarded) vs SFT ~1,500–2,500. Questioner training: 2h×4GPU+reward-service (RL) vs 10min×1GPU (SFT).

---

## 6. Transferable lessons

1. **Label with programs, not majority votes** — majority is wrong ~18–21%; program consensus is the single biggest win of the project (+7pp over majority).
2. **Measure skill diversity, not embedding diversity** — Vendi hid a full collapse.
3. **Gate, don't grade, noisy proxies** — graded rewards on noisy quality estimates are collapse gradients (Lagrangian form kept on the shelf: `E[v]=v*, E[b]=b*` marginals + `μ·v·b` co-occurrence).
4. **The conjunction (difficult ∧ verifiable) is the scarce resource** — anti-correlated; optimize jointly, never sequentially.
5. **Stratify the SFT target on every filter axis** — un-stratified axes leak selection bias into the clone, then amplify (epochs = fidelity knob, not quality knob).
6. **Always retrain from base** — kills compounding drift with zero machinery.
7. **De-noise the verifier with K** (K=10 majority): 83% consensus, enables IPW via votes/K.
8. **Self-play iteration is a variance problem before it is a learning problem** — 0.630 vs 0.474 from identical configs; guardrails (CVBAND) make it safe, not beneficial. Reliability must come from the curriculum-construction side.
9. **Persist everything** (verifier programs, per-step band/votes) — the investigation was rate-limited by discarded intermediates.

---

## 7. Where we are & what's next

**Now:** SFT-questioner quality campaign (solver GRPO deliberately deferred until question gates pass).
Targets: **trainable ≥40%** (at 8.3→23% depending on standard), **label-acc ≥90%** (✓ by K=10 construction), **well-formed ≥96% on-curriculum**.

**v4 (next):** ~200 mined verified sweet-spots (3× hard stratum) + **scratchpad self-check** generation format (the only lever on the 70% coherence ceiling) → probe → Stage-B pipeline-real. Then: rejection-SFT vs GRPO solver A/B, and the eval-vs-0.6415 test with yield matched and diversity as the isolated variable.


---

# Appendix: Full Example Questions (verbatim from run data)


## A. Diversity collapse — nopack questioner

### Step 1 (base, pre-RL):


**Step 1 (base, pre-RL) — Q1.** Let $S$ be the set of all positive integers $n$ such that the decimal representation of $n$ contains no digit strictly greater than $4$, and the sum of the digits of $n$ is a perfect square. Furthermore, for any $n \in S$, let $f(n)$ be the number of trailing zeros in the decimal representation of $n$. Determine the number of elements in $S$ such that $f(n) \ge 2$ and $n < 100,000$.


**Step 1 (base, pre-RL) — Q2.** Let $S$ be the set of all triplets of positive integers $(x, y, z)$ such that: 1. $x, y, z \le 50$, 2. $x \neq y$, $y \neq z$, and $z \neq x$, 3. $x + y \mid x + z + 7$ and $x + z \mid x + y + 7$. Find the number of distinct values of $x + y + z$ that can be formed by triplets in $S$.


**Step 1 (base, pre-RL) — Q3.** Let $S$ be the set of all integers $x$ such that $1 \le x \le 1000$ and $x$ can be written as the sum of at least two distinct powers of 2 (i.e., $x = \sum_{k \in K} 2^k$ where $K \subseteq \mathbb{Z}_{\ge 0}$, $|K| \ge 2$). For each $x \in S$, define $f(x)$ to be the number of ways to write $x$ as a product of two integers greater than 1. Calculate the sum $\sum_{x \in S} f(x)$.


### Step 6 (after 6 GRPO steps — the digit template):


**Step 6 (after RL) — Q1.** Let $S$ be the set of all integers $n$ such that $1 \le n \le 1000$ and the binary representation of $n$ contains at least two '1's, but does not contain the substring "11" (i.e., no two consecutive 1s). For each $n \in S$, let $d(n)$ be the number of divisors of $n$. Compute the sum $\sum_{n \in S} d(n)$.


**Step 6 (after RL) — Q2.** Let $S$ be the set of all integers $n$ such that $1 \le n \le 2024$ and the decimal representation of $n$ contains at least one digit '7'. For each $n \in S$, let $d(n)$ denote the number of '7's in the decimal representation of $n$. Compute the sum $\sum_{n \in S} d(n)$.


**Step 6 (after RL) — Q3.** Let $S$ be the set of all integers $n$ such that $1 \le n \le 2024$ and the sum of the digits of $n$ in base 10 is divisible by 3. Let $f(n)$ be the number of divisors of $n$. Compute the value of $\sum_{n \in S} f(n) \pmod{1000}$.


## B. Majority-vote label wrong (solver majority vs 3/3 program consensus)


**B1.** Let $S$ be the set of all integers $n$ such that $1 \le n \le 2023$ and the decimal representation of $n$ contains at least one digit equal to $7$ and at least one digit equal to $3$. Find the number of such integers $n$.

> solver-majority: **107** — programs (3/3): **108**


**B2.** Let $S$ be the set of all integers $n$ such that $1 \le n \le 1000$ and the decimal representation of $n^3 - n$ contains at least one digit equal to $7$. How many such integers $n$ are in $S$?

> solver-majority: **500** — programs (3/3): **491**


**B3.** Let $S$ be the set of all integers $n$ such that $1 \le n \le 1000$ and the decimal representation of $n$ contains at least one digit equal to 7. For each $n \in S$, let $f(n)$ be the number of distinct digits present in the decimal representation of $n$. Compute the sum $\sum_{n \in S} f(n)$.

> solver-majority: **667** — programs (3/3): **739**


## C. Ill-posed 'hard' questions (v3 sweet-spot failures — the coherence ceiling)


**C1.** How many integers $x$ with $1 \le x \le 1000$ are such that the equation $x^2 \equiv a \pmod 7$ has exactly 4 distinct solutions for $a \pmod 7$?

> why it fails: Incoherent: no a mod 7 gives 4 solutions to x^2=a (max 2); 'exactly 4 distinct a values' (QRs {0,1,2,4}) is a property of the modulus, independent of x. Readings give 1000, 0, or 858 - no single integer.


**C2.** A sequence $a_n$ is defined by $a_1 = 2$ and $a_{n} = a_{n-1} + 5$ for $n > 1$. Another sequence $b_n$ is defined by $b_1 = 3$ and $b_{n} = b_{n-1} + 7$ for $n > 1$. Let $S$ be the number of positive integers $n \leq 100$ such that $a_n \equiv b_n \pmod{12}$. Find the smallest prime factor of $S$.

> why it fails: S=0: 2n=1 mod 12 impossible, so no n satisfies congruence; smallest prime factor of 0 undefined


**C3.** Find the number of positive integers k less than 1000 such that k is not divisible by any integer greater than sqrt(k).

> why it fails: Literal reading trivial: k divides itself and k>sqrt(k) for k>1, so only k=1 qualifies (answer 1). Intended reading (no PROPER divisor > sqrt(k)) gives 1, primes, prime squares = 180. Brute force confirms both.


## D. GOOD v3 questions (in-band AND verified — the target product)


**D1.** How many triangles can be formed by the vertices $V = \{1, 2, \dots, 12\}$ such that for any two vertices $v_1, v_2$ in the triangle, $|v_1 - v_2|$ is divisible by at least one of the integers in $\{2, 3, 5, 7\}$?

> verified answer: **112**


**D2.** How many positive integers $n \le 1000$ are such that $n$ divides $2^n - 1$?

> verified answer: **1**


**D3.** Let $S$ be the set of ordered pairs $(a, b)$ of integers such that $1 \le a, b \le 50$ and $a + b \le 100$. Find the number of pairs in $S$ such that $a^2 + b^2$ is a perfect square.

> verified answer: **52**
