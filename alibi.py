"""ALiBi (Attention with Linear Biases) from scratch in NumPy.

Press et al., "Train Short, Test Long: Attention with Linear Biases Enables
Input Length Extrapolation" (2021).

Instead of learned or sinusoidal positional embeddings, ALiBi adds a static
linear penalty to attention logits based on distance:

    bias[i, j] = -m_h * |i - j|          # bidirectional
    bias[i, j] = -m_h * (i - j)          # causal (j <= i; -inf otherwise)

Head slopes m_h form a geometric sequence so different heads attend at
different distance scales — no parameters, and the bias extends to any length.
"""
from __future__ import annotations

import numpy as np


def get_alibi_slopes(n_heads: int) -> np.ndarray:
    """Geometric ALiBi slopes for ``n_heads`` attention heads (Press et al.).

    For a power-of-two head count the slopes are
    ``2^{-8/n}, 2^{-16/n}, ..., 2^{-8}``.
    For other counts, recursively take the power-of-two sequence of the next
    smaller power of two and interleave with every-other slope of the next
    larger power of two (HuggingFace / fairseq convention matching the paper).

    Returns
    -------
    slopes : (n_heads,) float64
    """
    if n_heads < 1:
        raise ValueError(f"n_heads must be >= 1, got {n_heads}")

    def _pow2_slopes(n: int) -> np.ndarray:
        # ratio r = 2^{-8/n}; slopes = r, r^2, ..., r^n
        start = 2.0 ** (-(2.0 ** -(np.log2(n) - 3)))
        return np.array([start * (start ** i) for i in range(n)], dtype=np.float64)

    if (n_heads & (n_heads - 1)) == 0:
        return _pow2_slopes(n_heads)

    closest = 2 ** int(np.floor(np.log2(n_heads)))
    slopes = _pow2_slopes(closest).tolist()
    extra = _pow2_slopes(2 * closest).tolist()[0::2][: n_heads - closest]
    return np.array(slopes + extra, dtype=np.float64)


def alibi_bias(
    seq_len: int,
    slopes: np.ndarray | float,
    *,
    causal: bool = False,
) -> np.ndarray:
    """Build ALiBi bias tensor.

    Parameters
    ----------
    seq_len : sequence length T
    slopes : scalar or (H,) array of head slopes
    causal : if True, use distance (i - j) for j <= i and -inf above diagonal

    Returns
    -------
    bias : (T, T) if slopes is scalar, else (H, T, T)
    """
    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    slopes_arr = np.atleast_1d(np.asarray(slopes, dtype=np.float64))
    i = np.arange(seq_len, dtype=np.float64)[:, None]
    j = np.arange(seq_len, dtype=np.float64)[None, :]
    if causal:
        dist = i - j  # >= 0 on/below diagonal
        mask = j > i
        base = -dist  # more negative for farther past keys
        base = np.where(mask, -np.inf, base)
    else:
        base = -np.abs(i - j)

    # slopes shape (H,) → bias (H, T, T); scalar → (T, T)
    if slopes_arr.size == 1:
        return slopes_arr.reshape(()) * base
    return slopes_arr[:, None, None] * base[None, :, :]


def apply_alibi(
    logits: np.ndarray,
    slopes: np.ndarray | float,
    *,
    causal: bool = False,
) -> np.ndarray:
    """Add ALiBi bias to attention logits.

    logits : (..., T, T)  — last two dims are query/key positions
    slopes : scalar (broadcast to all leading dims) or (H,) matching a head axis
    """
    logits = np.asarray(logits, dtype=np.float64)
    T = logits.shape[-1]
    if logits.shape[-2] != T:
        raise ValueError(f"logits last two dims must be square, got {logits.shape}")
    bias = alibi_bias(T, slopes, causal=causal)
    # Align bias dims with logits: (T,T) or (H,T,T)
    while bias.ndim < logits.ndim:
        bias = bias[None, ...]
    # If logits is (B, H, T, T) and bias is (1, H, T, T) after one expand from (H,T,T)
    # already handled; if logits is (B, T, T) and bias is (T, T) — also fine.
    return logits + bias


def check_slope_sanity(n_heads: int = 8, atol: float = 1e-12) -> dict:
    """Verify slopes are positive, strictly decreasing, geometric (pow2 case)."""
    slopes = get_alibi_slopes(n_heads)
    positive = bool(np.all(slopes > 0))
    decreasing = bool(np.all(np.diff(slopes) < 0))
    geometric = True
    ratio = None
    if (n_heads & (n_heads - 1)) == 0 and n_heads >= 2:
        ratios = slopes[1:] / slopes[:-1]
        ratio = float(ratios.mean())
        geometric = bool(np.allclose(ratios, ratios[0], atol=atol))
        expected = 2.0 ** (-8.0 / n_heads)
        geometric = geometric and abs(ratio - expected) < 1e-9
    return {
        "n_heads": n_heads,
        "slopes": [round(float(s), 8) for s in slopes.tolist()],
        "positive": positive,
        "strictly_decreasing": decreasing,
        "geometric_pow2": geometric,
        "common_ratio": round(ratio, 8) if ratio is not None else None,
        "pass": bool(positive and decreasing and geometric),
    }


def sinusoidal_pe(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Absolute sinusoidal PE (Vaswani et al.) for comparison — shape (T, D)."""
    pe = np.zeros((seq_len, dim), dtype=np.float64)
    if seq_len == 0:
        return pe
    pos = np.arange(seq_len, dtype=np.float64)[:, None]
    i = np.arange(0, dim, 2, dtype=np.float64)
    div = base ** (i / dim)
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div[: pe[:, 1::2].shape[1]])
    return pe
