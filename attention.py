"""Tiny NumPy attention with no-PE / absolute sinusoidal / ALiBi."""
from __future__ import annotations

from typing import Literal

import numpy as np

from alibi import alibi_bias, get_alibi_slopes, sinusoidal_pe

PEKind = Literal["none", "absolute", "alibi"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Handle -inf rows from causal ALiBi safely
    x = np.where(np.isfinite(x), x, -1e9)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class TinyAttention:
    """Single- or multi-head attention with optional ALiBi / absolute PE.

    Positional encoding modes
    -------------------------
    none     : no position signal
    absolute : add sinusoidal PE to token embeddings before QKV
    alibi    : add static linear distance bias to logits (per-head slopes)
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 1,
        pe: PEKind = "none",
        pe_base: float = 10000.0,
        causal: bool = False,
        rng: np.random.Generator | None = None,
        scale: float = 0.1,
        slopes: np.ndarray | None = None,
    ) -> None:
        if d_model < 2 or d_model % 2 != 0:
            raise ValueError("d_model must be even and >= 2")
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.pe = pe
        self.pe_base = pe_base
        self.causal = causal
        rng = rng or np.random.default_rng()
        self.W_q = rng.normal(0, scale, size=(d_model, d_model))
        self.W_k = rng.normal(0, scale, size=(d_model, d_model))
        self.W_v = rng.normal(0, scale, size=(d_model, d_model))
        self.W_o = rng.normal(0, scale, size=(d_model, d_model))
        self.b_o = np.zeros(d_model)
        if slopes is None:
            self.slopes = get_alibi_slopes(n_heads) if pe == "alibi" else np.ones(n_heads)
        else:
            self.slopes = np.asarray(slopes, dtype=np.float64)
            if self.slopes.shape != (n_heads,):
                raise ValueError(f"slopes must have shape ({n_heads},)")

    def parameters(self) -> list[np.ndarray]:
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_o]

    def _prepare_x(self, x: np.ndarray) -> np.ndarray:
        if self.pe != "absolute":
            return x
        pe = sinusoidal_pe(x.shape[1], self.d_model, base=self.pe_base)
        return x + pe[None, :, :]

    def _split_heads(self, t: np.ndarray) -> np.ndarray:
        # (B, T, D) -> (B, H, T, Dh)
        B, T, _ = t.shape
        return t.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def qkv_logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return Q, K, V (B,H,T,Dh) and attention logits (B, H, T, T)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, ...]
        x = self._prepare_x(x)
        B, T, D = x.shape
        Q = self._split_heads(x @ self.W_q)
        K = self._split_heads(x @ self.W_k)
        V = self._split_heads(x @ self.W_v)
        scale = 1.0 / np.sqrt(self.d_head)
        logits = (Q @ K.transpose(0, 1, 3, 2)) * scale  # (B,H,T,T)
        if self.pe == "alibi":
            bias = alibi_bias(T, self.slopes, causal=self.causal)  # (H,T,T)
            logits = logits + bias[None, :, :, :]
        elif self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            logits = np.where(mask[None, None, :, :], -np.inf, logits)
        return Q, K, V, logits

    def forward(self, x: np.ndarray, *, return_weights: bool = False):
        Q, K, V, logits = self.qkv_logits(x)
        weights = softmax(logits, axis=-1)
        ctx = weights @ V  # (B,H,T,Dh)
        B, H, T, Dh = ctx.shape
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)
        out = ctx @ self.W_o + self.b_o
        if return_weights:
            return out, weights, logits
        return out


def clone_attention(src: TinyAttention, pe: PEKind) -> TinyAttention:
    """Clone weights into a new TinyAttention with a different PE mode."""
    dst = TinyAttention(
        src.d_model,
        n_heads=src.n_heads,
        pe=pe,
        pe_base=src.pe_base,
        causal=src.causal,
        scale=0.0,
        slopes=src.slopes.copy() if pe == "alibi" else None,
    )
    dst.W_q = src.W_q.copy()
    dst.W_k = src.W_k.copy()
    dst.W_v = src.W_v.copy()
    dst.W_o = src.W_o.copy()
    dst.b_o = src.b_o.copy()
    if pe == "alibi" and src.pe != "alibi":
        dst.slopes = get_alibi_slopes(src.n_heads)
    return dst
