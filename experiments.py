"""ALiBi experiments: slope sanity, locality of attention mass, helpers."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from alibi import alibi_bias, check_slope_sanity, get_alibi_slopes, sinusoidal_pe
from attention import TinyAttention, clone_attention, softmax


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def locality_attention_mass(
    d_model: int = 32,
    seq_len: int = 32,
    n_heads: int = 4,
    seed: int = 42,
    near_radius: int = 2,
) -> Dict[str, Any]:
    """With random Q/K, measure fraction of attention mass on nearby keys.

    ALiBi should concentrate mass nearer the query vs no-PE (flatter).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, seq_len, d_model))
    base = TinyAttention(d_model, n_heads=n_heads, pe="none", rng=rng, scale=0.15)

    out: Dict[str, Any] = {
        "seq_len": seq_len,
        "n_heads": n_heads,
        "near_radius": near_radius,
    }
    rows = []
    for pe in ("none", "absolute", "alibi"):
        attn = clone_attention(base, pe)  # type: ignore[arg-type]
        _, weights, _ = attn.forward(x, return_weights=True)  # (1,H,T,T)
        w = weights[0].mean(axis=0)  # (T,T)
        near_mass = []
        for i in range(seq_len):
            lo = max(0, i - near_radius)
            hi = min(seq_len, i + near_radius + 1)
            near_mass.append(float(w[i, lo:hi].sum()))
        mean_near = float(np.mean(near_mass))
        ent = []
        for i in range(seq_len):
            p = w[i] + 1e-12
            ent.append(float(-np.sum(p * np.log(p))))
        rows.append(
            {
                "pe": pe,
                "mean_near_mass": round(mean_near, 6),
                "mean_entropy": round(float(np.mean(ent)), 6),
                "mid_row": [round(float(v), 6) for v in w[seq_len // 2].tolist()],
            }
        )
        out[pe] = {
            "mean_near_mass": round(mean_near, 6),
            "mean_entropy": round(float(np.mean(ent)), 6),
        }

    out["per_pe"] = rows
    out["alibi_beats_none_near"] = bool(
        out["alibi"]["mean_near_mass"] > out["none"]["mean_near_mass"] + 0.05
    )
    return out


def bias_matrix_demo(seq_len: int = 16, n_heads: int = 4) -> Dict[str, Any]:
    """Return slopes + a sample bias matrix for steepest / shallowest head."""
    slopes = get_alibi_slopes(n_heads)
    bias = alibi_bias(seq_len, slopes, causal=False)  # (H,T,T)
    return {
        "slopes": [round(float(s), 8) for s in slopes.tolist()],
        "seq_len": seq_len,
        "n_heads": n_heads,
        "bias_head0": [[round(float(v), 4) for v in row] for row in bias[0].tolist()],
        "bias_head_last": [[round(float(v), 4) for v in row] for row in bias[-1].tolist()],
        "slope_check": check_slope_sanity(n_heads),
    }


def make_near_neighbor_batch(
    n: int,
    seq_len: int,
    d_model: int,
    offset: int,
    rng: np.random.Generator,
    *,
    n_distractors: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query must select key at relative -offset; distractors are farther away.

    Farther-only distractors make ALiBi's locality prior *help* (nearer = correct)
    while still defeating pure content matching. Absolute PE can memorize short
    slots but degrades when sequence length grows past training.
    """
    if offset < 1 or offset >= seq_len:
        raise ValueError("offset must be in 1..seq_len-1")
    X = rng.normal(size=(n, seq_len, d_model)) * 0.25
    labels = np.zeros(n, dtype=np.int64)
    q_idxs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        q = int(rng.integers(offset, seq_len))
        k_star = q - offset
        q_idxs[i] = q
        labels[i] = k_star
        src = rng.normal(size=d_model)
        src /= np.linalg.norm(src) + 1e-12
        X[i, q] = src * 1.6
        X[i, k_star] = src * 1.6 + rng.normal(size=d_model) * 0.02
        forbidden = {q, k_star}
        # candidates farther from q than the target (distance > offset)
        candidates = [
            k
            for k in range(seq_len)
            if k not in forbidden and abs(k - q) > offset
        ]
        rng.shuffle(candidates)
        for k in candidates[:n_distractors]:
            X[i, k] = src * 1.6 + rng.normal(size=d_model) * 0.02
            forbidden.add(k)
    return X, labels, q_idxs


def _query_logits(
    attn: TinyAttention,
    X: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
    *,
    mask_self: bool = True,
) -> np.ndarray:
    """(B, T) mean-over-heads logits from each example's query position.

    By default mask the query index itself: ALiBi's distance-0 peak would
    otherwise always win a retrieval task before any learning.
    """
    _, _, _, logits = attn.qkv_logits(X)  # (B,H,T,T)
    logits = logits.mean(axis=1) * logit_scale
    rows = logits[np.arange(len(q_idxs)), q_idxs, :].copy()
    if mask_self:
        rows[np.arange(len(q_idxs)), q_idxs.astype(int)] = -1e9
    return rows


def distance_discrimination_accuracy(
    attn: TinyAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
) -> float:
    rows = _query_logits(attn, X, q_idxs, logit_scale)
    return float(np.mean(np.argmax(rows, axis=-1) == labels))


def attention_mass_on_target(
    attn: TinyAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
    logit_scale: float = 1.0,
) -> float:
    rows = _query_logits(attn, X, q_idxs, logit_scale)
    probs = softmax(rows, axis=-1)
    return float(np.mean(probs[np.arange(len(labels)), labels]))


def mean_logit_gap(
    attn: TinyAttention,
    X: np.ndarray,
    labels: np.ndarray,
    q_idxs: np.ndarray,
) -> float:
    rows = _query_logits(attn, X, q_idxs, 1.0)
    gaps = []
    for i, y in enumerate(labels):
        target = rows[i, y]
        mask = np.ones(rows.shape[1], dtype=bool)
        mask[y] = False
        best_other = float(np.max(rows[i, mask]))
        gaps.append(float(target - best_other))
    return float(np.mean(gaps))


def train_relative_task(*args, **kwargs):
    from train_task import train_relative_task as _train

    return _train(*args, **kwargs)


def length_extrapolation(*args, **kwargs):
    from train_task import length_extrapolation as _ex

    return _ex(*args, **kwargs)
