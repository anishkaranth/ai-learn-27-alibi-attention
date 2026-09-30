"""Train loop for relative-offset retrieval + length-extrapolation eval."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from alibi import alibi_bias, get_alibi_slopes, sinusoidal_pe
from attention import TinyAttention, clone_attention, softmax
from experiments import (
    _query_logits,
    attention_mass_on_target,
    distance_discrimination_accuracy,
    make_near_neighbor_batch,
    mean_logit_gap,
)


def train_relative_task(
    pe: str,
    d_model: int = 32,
    seq_len: int = 16,
    offset: int = 3,
    n_heads: int = 4,
    n_train: int = 512,
    n_test: int = 192,
    steps: int = 140,
    lr: float = 0.08,
    seed: int = 42,
    n_distractors: int = 3,
    logit_scale: float = 8.0,
) -> Dict[str, Any]:
    """Train Q/K so attention at query selects the key at relative -offset."""
    pe_seed = {"none": 0, "absolute": 1, "alibi": 2}[pe]
    rng = np.random.default_rng(seed + pe_seed)
    rng_init = np.random.default_rng(seed)
    base = TinyAttention(
        d_model, n_heads=n_heads, pe="none", rng=rng_init, scale=0.08
    )
    attn = clone_attention(base, pe)  # type: ignore[arg-type]

    Xtr, Ytr, Qtr = make_near_neighbor_batch(
        n_train, seq_len, d_model, offset, rng, n_distractors=n_distractors
    )
    Xte, Yte, Qte = make_near_neighbor_batch(
        n_test, seq_len, d_model, offset, rng, n_distractors=n_distractors
    )

    history = []
    for step in range(steps):
        idx = rng.integers(0, n_train, size=64)
        xb, yb, qb = Xtr[idx], Ytr[idx], Qtr[idx]

        x_in = xb + sinusoidal_pe(seq_len, d_model)[None, :, :] if pe == "absolute" else xb
        Q = x_in @ attn.W_q
        K = x_in @ attn.W_k
        # (B,T,D) -> (B,H,T,Dh)
        B = len(qb)
        Dh = d_model // n_heads
        Qh = Q.reshape(B, seq_len, n_heads, Dh).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, seq_len, n_heads, Dh).transpose(0, 2, 1, 3)
        scale = logit_scale / np.sqrt(Dh)
        logits = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale  # (B,H,T,T)
        if pe == "alibi":
            bias = alibi_bias(seq_len, attn.slopes, causal=False)
            logits = logits + bias[None, :, :, :]
        # mean over heads for the ranking loss (same as eval); mask self
        logits_mean = logits.mean(axis=1)
        row_logits = logits_mean[np.arange(B), qb.astype(int), :].copy()
        row_logits[np.arange(B), qb.astype(int)] = -1e9
        row_prob = softmax(row_logits, axis=-1)
        loss = -float(np.mean(np.log(row_prob[np.arange(B), yb] + 1e-12)))
        grad_row = row_prob.copy()
        grad_row[np.arange(B), yb] -= 1.0
        grad_row /= B

        # Backprop through mean-over-heads: each head gets grad/H
        g_logits = np.zeros_like(logits)
        for b in range(B):
            qi = int(qb[b])
            g_logits[b, :, qi, :] = grad_row[b][None, :] / n_heads

        # dL/dQh = scale * g @ K, dL/dKh = scale * g^T @ Q
        gQh = scale * (g_logits @ Kh)
        gKh = scale * (g_logits.transpose(0, 1, 3, 2) @ Qh)

        gQ = gQh.transpose(0, 2, 1, 3).reshape(B, seq_len, d_model)
        gK = gKh.transpose(0, 2, 1, 3).reshape(B, seq_len, d_model)
        gWq = x_in.reshape(-1, d_model).T @ gQ.reshape(-1, d_model)
        gWk = x_in.reshape(-1, d_model).T @ gK.reshape(-1, d_model)
        attn.W_q -= lr * gWq
        attn.W_k -= lr * gWk

        if step % 20 == 0 or step == steps - 1:
            acc = distance_discrimination_accuracy(attn, Xte, Yte, Qte, logit_scale)
            mass = attention_mass_on_target(attn, Xte, Yte, Qte, logit_scale)
            gap = mean_logit_gap(attn, Xte, Yte, Qte)
            history.append(
                {
                    "step": step,
                    "loss": round(loss, 4),
                    "acc": round(acc, 4),
                    "mass": round(mass, 4),
                    "logit_gap": round(gap, 4),
                }
            )

    final_acc = distance_discrimination_accuracy(attn, Xte, Yte, Qte, logit_scale)
    final_mass = attention_mass_on_target(attn, Xte, Yte, Qte, logit_scale)
    final_gap = mean_logit_gap(attn, Xte, Yte, Qte)
    row_logits = _query_logits(attn, Xte[:1], Qte[:1], logit_scale)[0]
    row = softmax(row_logits[None, :], axis=-1)[0]
    return {
        "pe": pe,
        "final_acc": round(float(final_acc), 4),
        "final_mass": round(float(final_mass), 4),
        "final_logit_gap": round(float(final_gap), 4),
        "chance_acc": round(1.0 / seq_len, 4),
        "history": history,
        "attn_row": [round(float(v), 4) for v in row.tolist()],
        "label": int(Yte[0]),
        "query_idx": int(Qte[0]),
        "offset": offset,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "n_distractors": n_distractors,
        "logit_scale": logit_scale,
        "attn": attn,  # keep for extrapolation (stripped before JSON)
    }


def length_extrapolation(
    pe: str,
    *,
    d_model: int = 32,
    L_train: int = 16,
    test_lengths: List[int] | None = None,
    offset: int = 3,
    n_heads: int = 4,
    n_train: int = 512,
    n_test: int = 160,
    steps: int = 140,
    lr: float = 0.08,
    seed: int = 42,
    n_distractors: int = 3,
    logit_scale: float = 8.0,
) -> Dict[str, Any]:
    """Train at L_train, evaluate top-1 acc at longer lengths (same relative offset)."""
    test_lengths = test_lengths or [16, 24, 32, 48]
    trained = train_relative_task(
        pe,
        d_model=d_model,
        seq_len=L_train,
        offset=offset,
        n_heads=n_heads,
        n_train=n_train,
        n_test=n_test,
        steps=steps,
        lr=lr,
        seed=seed,
        n_distractors=n_distractors,
        logit_scale=logit_scale,
    )
    attn: TinyAttention = trained["attn"]
    pe_seed = {"none": 0, "absolute": 1, "alibi": 2}[pe]
    rng = np.random.default_rng(seed + 100 + pe_seed)

    by_len = []
    for L in test_lengths:
        if L <= offset:
            continue
        Xt, Yt, Qt = make_near_neighbor_batch(
            n_test, L, d_model, offset, rng, n_distractors=n_distractors
        )
        acc = distance_discrimination_accuracy(attn, Xt, Yt, Qt, logit_scale)
        mass = attention_mass_on_target(attn, Xt, Yt, Qt, logit_scale)
        by_len.append(
            {
                "seq_len": int(L),
                "acc": round(float(acc), 4),
                "mass": round(float(mass), 4),
                "chance": round(1.0 / L, 4),
            }
        )

    # Drop non-JSON attn object
    trained_clean = {k: v for k, v in trained.items() if k != "attn"}
    return {
        "pe": pe,
        "L_train": L_train,
        "offset": offset,
        "train_acc": trained_clean["final_acc"],
        "by_length": by_len,
        "trained": trained_clean,
    }
