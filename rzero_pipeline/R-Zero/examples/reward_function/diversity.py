"""Embedding diversity signal for the challenger reward.

Keeps a growing bank of embeddings of every question generated this run
(persisted to disk so it spans batches and iterations).

Primary entry point: `vendi_diversity_rewards` — a per-sample diversity REWARD
equal to each question's MARGINAL contribution to the Vendi score (the effective
number of distinct questions) of all past samples, i.e. the relative delta it
adds to the bank. A near-duplicate / degenerate question lies in the span of
what's already been asked, so it adds ~0 (low reward); a genuinely novel
question adds a fresh direction (~1, full reward). The Vendi score and its
batch-level relative delta are computed via a Nyström approximation so the cost
stays O(n*m^2) as the bank grows to many thousands of samples.

`diversity_penalties` (the older nearest-neighbor penalty) is kept for
backward-compatibility but is no longer used by the reward.

CPU-only (small model) to avoid contending with training GPUs. No
sentence-transformers dependency: mean-pooled `transformers` AutoModel.
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

EMB_MODEL = os.getenv("DIVERSITY_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEM_PATH = os.getenv("DIVERSITY_MEM", os.path.join(os.getenv("STORAGE_PATH", "."), "diversity_mem.npy"))
DIVERSITY_WEIGHT = float(os.getenv("DIVERSITY_WEIGHT", "0.4"))  # weight of the per-sample diversity reward (Vendi marginal)
SIM_FLOOR = float(os.getenv("DIVERSITY_SIM_FLOOR", "0.5"))  # (legacy NN penalty) below this similarity = treated as novel
DIVERSITY_POWER = float(os.getenv("DIVERSITY_POWER", "2.0"))  # (legacy NN penalty) superlinear shaping
# --- Vendi (Nyström) diversity-reward knobs ---
VENDI_LANDMARKS = int(os.getenv("VENDI_LANDMARKS", "512"))   # # of Nyström landmarks (eig cost O(m^3))
VENDI_RIDGE = float(os.getenv("VENDI_RIDGE", "1e-3"))        # kernel ridge for numerical stability
VENDI_VS_CAP = int(os.getenv("VENDI_VS_CAP", "4000"))        # cap rows used for the logged VS metric
# Golden cap: stop rewarding diversity once the bank's Vendi (at the golden sample size)
# reaches the golden reference level (MATH-500 = ~85.6 over 500). 0 disables the cap.
VENDI_GOLDEN = float(os.getenv("VENDI_GOLDEN", "85.6091"))
VENDI_GOLDEN_N = int(os.getenv("VENDI_GOLDEN_N", "500"))     # matched sample size for the gate comparison

_MODEL = None
_TOK = None


def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(EMB_MODEL)
        _MODEL = AutoModel.from_pretrained(EMB_MODEL).eval()  # CPU on purpose
    return _MODEL, _TOK


def embed(texts, bs=128):
    model, tok = _load()
    chunks = []
    for i in range(0, len(texts), bs):
        b = [t if t else " " for t in texts[i:i + bs]]
        enc = tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        chunks.append(emb.numpy().astype(np.float32))
    return np.concatenate(chunks, 0)


def diversity_penalties(questions, update_memory=True):
    """Return (penalties, nn_similarities) per question; append to persistent bank."""
    cur = embed(questions)
    n, d = cur.shape
    hist = np.load(MEM_PATH) if os.path.exists(MEM_PATH) else np.zeros((0, d), dtype=np.float32)
    bank = np.concatenate([hist, cur], 0) if len(hist) else cur
    sims = cur @ bank.T  # cosine (rows are unit-normalized)
    for i in range(n):  # exclude self-match (current rows live at offset len(hist))
        sims[i, len(hist) + i] = -1.0
    nn = sims.max(1) if bank.shape[0] > 1 else np.zeros(n, dtype=np.float32)
    remapped = np.clip((nn - SIM_FLOOR) / max(1e-6, 1.0 - SIM_FLOOR), 0.0, 1.0)
    pen = DIVERSITY_WEIGHT * (remapped ** DIVERSITY_POWER)
    if update_memory:
        os.makedirs(os.path.dirname(MEM_PATH) or ".", exist_ok=True)
        np.save(MEM_PATH, bank)
    return pen.astype(float).tolist(), nn.astype(float).tolist()


# ---------------------------------------------------------------------------
# Vendi-score (Nyström) diversity REWARD.
# ---------------------------------------------------------------------------
def _vendi_from_eigs(eigs):
    """Vendi score = exp(Shannon entropy of the kernel's normalized eigenvalues)."""
    e = np.clip(np.asarray(eigs, dtype=np.float64).real, 0.0, None)
    s = e.sum()
    if s <= 0:
        return 1.0
    p = e / s
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def _even_idx(n, m):
    """m evenly-spaced unique indices into range(n) (deterministic; no RNG)."""
    if n <= m:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, m).astype(int))


def _vendi_nystrom(X):
    """Nyström-approximated Vendi score of the cosine kernel over rows of X (unit-norm).

    Uses landmarks L; nonzero eigenvalues of the full kernel K ~ eig of
    M = W^{-1/2} (C^T C) W^{-1/2}, where C = K[:,L], W = K[L,L]. Cost O(N*m^2 + m^3).
    """
    n = X.shape[0]
    if n < 2:
        return 1.0
    if n > VENDI_VS_CAP:
        X = X[_even_idx(n, VENDI_VS_CAP)]
    L = X[_even_idx(X.shape[0], VENDI_LANDMARKS)].astype(np.float64)
    W = L @ L.T + VENDI_RIDGE * np.eye(L.shape[0])
    wv, wU = np.linalg.eigh(W)
    wv = np.clip(wv, VENDI_RIDGE, None)
    W_isqrt = (wU * (1.0 / np.sqrt(wv))) @ wU.T
    C = X.astype(np.float64) @ L.T                 # (N, m)
    M = W_isqrt @ (C.T @ C) @ W_isqrt              # (m, m)
    eigs = np.linalg.eigvalsh((M + M.T) / 2)
    return _vendi_from_eigs(eigs)


def vendi_diversity_rewards(questions, update_memory=True):
    """Per-sample marginal Vendi diversity reward (Nyström residual vs all past samples).

    reward_i = DIVERSITY_WEIGHT * novelty_i, novelty_i = 1 - k_i^T (W+eI)^{-1} k_i in [0,1]
    (the residual of question i against the span of the accumulated bank's landmarks =
    the fresh diversity direction it adds). Also returns the batch-level relative delta
    of the Vendi score for monitoring. Appends the batch to the persistent bank.
    """
    cur = embed(questions)
    n, d = cur.shape
    hist = np.load(MEM_PATH) if os.path.exists(MEM_PATH) else np.zeros((0, d), dtype=np.float32)
    if len(hist) >= 2:
        # Marginal novelty = Nyström residual of each question against the span of all
        # past samples (landmarks drawn from the accumulated bank).
        L = hist[_even_idx(len(hist), VENDI_LANDMARKS)].astype(np.float64)
        W = L @ L.T + VENDI_RIDGE * np.eye(L.shape[0])
        W_inv = np.linalg.pinv(W)
        Kbl = cur.astype(np.float64) @ L.T                      # (n, m) sims to landmarks
        quad = np.einsum('nm,mk,nk->n', Kbl, W_inv, Kbl)       # k^T W^-1 k  ~ in [0,1]
        novelty = np.clip(1.0 - quad, 0.0, 1.0)               # fresh direction added
    else:
        # Cold start (empty bank): leave-self nearest-neighbor residual within the batch,
        # so the first batch still scores duplicates ~0 and novel questions ~1.
        S = (cur @ cur.T).astype(np.float64)
        np.fill_diagonal(S, -1.0)
        nn = S.max(1) if n > 1 else np.zeros(n)
        novelty = np.clip(1.0 - np.clip(nn, 0.0, 1.0), 0.0, 1.0)
    # Relative-delta of the Vendi score over the batch (logged).
    allX = np.concatenate([hist, cur], 0) if len(hist) else cur
    vs_before = _vendi_nystrom(hist) if len(hist) >= 2 else 1.0
    vs_after = _vendi_nystrom(allX)
    rel_delta = (vs_after - vs_before) / max(vs_before, 1.0)
    # Golden cap: ramp the reward to 0 as the bank's diversity (a golden-sized subsample,
    # so it's matched to the reference's N) approaches the golden Vendi level. Prevents the
    # questioner from reward-hacking novelty beyond a real-dataset diversity target.
    gate = 1.0
    bank_matched = vs_after
    if VENDI_GOLDEN > 0 and len(allX) >= VENDI_GOLDEN_N:
        bank_matched = _vendi_nystrom(allX[_even_idx(len(allX), VENDI_GOLDEN_N)])
        gate = float(np.clip((VENDI_GOLDEN - bank_matched) / VENDI_GOLDEN, 0.0, 1.0))
    rewards = (DIVERSITY_WEIGHT * gate * novelty).astype(float)
    if update_memory:
        os.makedirs(os.path.dirname(MEM_PATH) or ".", exist_ok=True)
        np.save(MEM_PATH, allX)
    stats = {"vs_before": vs_before, "vs_after": vs_after, "rel_delta": rel_delta,
             "mean_novelty": float(novelty.mean()), "n_hist": int(len(hist)),
             "gate": gate, "golden": VENDI_GOLDEN, "bank_matched": float(bank_matched)}
    return rewards.tolist(), stats
