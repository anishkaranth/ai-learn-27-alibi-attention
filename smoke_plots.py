"""Matplotlib SVG plots + RESULTS.md writer for ALiBi smoke."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update(
    {
        "svg.hashsalt": "ai-learn-27",
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)
COLS = ["#e76f51", "#e9c46a", "#2a9d8f", "#126782", "#8338ec"]
PE_ORDER = ["none", "absolute", "alibi"]
PE_LABELS = {"none": "no PE", "absolute": "absolute sin", "alibi": "ALiBi"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names: List[str] = []

    # 1) Slopes for n=4 and n=8
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for key, c, marker in (("n4", COLS[2], "o"), ("n8", COLS[3], "s")):
        slopes = m["slope_sanity"][key]["slopes"]
        ax.plot(np.arange(1, len(slopes) + 1), slopes, marker + "-", color=c, label=f"H={len(slopes)}")
    ax.set_xlabel("head index (1..H)")
    ax.set_ylabel("slope $m_h$")
    ax.set_yscale("log")
    ax.set_title("ALiBi geometric head slopes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    names.append(_save(fig, out / "alibi_slopes.svg"))

    # 2) Bias heatmap (head 0) — per-cell rectangles so SVG stays vector (no PNG)
    bias = np.array(m["bias_demo"]["bias_head0"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ny, nx = bias.shape
    cmap = plt.get_cmap("viridis")
    vmin, vmax = float(bias.min()), float(bias.max())
    span = (vmax - vmin) if vmax > vmin else 1.0
    for i in range(ny):
        for j in range(nx):
            c = cmap((bias[i, j] - vmin) / span)
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="none")
            )
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("key j")
    ax.set_ylabel("query i")
    ax.set_title(f"ALiBi bias head0 (m={m['bias_demo']['slopes'][0]})")
    # Vector color legend (avoid colorbar raster PNG embed)
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
    n_grad = 24
    for k in range(n_grad):
        val = vmin + (vmax - vmin) * (k / (n_grad - 1))
        cax.add_patch(
            plt.Rectangle((0, k), 1, 1, facecolor=cmap(k / (n_grad - 1)), edgecolor="none")
        )
    cax.set_xlim(0, 1)
    cax.set_ylim(0, n_grad)
    cax.set_xticks([])
    cax.set_yticks([0, n_grad / 2, n_grad])
    cax.set_yticklabels([f"{vmin:.1f}", f"{0.5*(vmin+vmax):.1f}", f"{vmax:.1f}"])
    names.append(_save(fig, out / "alibi_bias_heatmap.svg"))

    # 3) Locality: mid-query attention rows
    loc = {p["pe"]: p for p in m["locality"]["per_pe"]}
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), sharey=True)
    for ax, pe in zip(axes, PE_ORDER):
        row = np.array(loc[pe]["mid_row"])
        color = COLS[2] if pe == "alibi" else COLS[1] if pe == "absolute" else "#aaa"
        ax.bar(np.arange(len(row)), row, color=color, width=1.0)
        mid = len(row) // 2
        ax.axvline(mid, color=COLS[0], ls="--", lw=1.0, label="query")
        ax.set_title(f"{PE_LABELS[pe]}\nnear={loc[pe]['mean_near_mass']:.3f}")
        ax.set_xlabel("key index")
    axes[0].set_ylabel("attn weight")
    axes[0].legend(fontsize=7)
    fig.suptitle("Locality: attention from mid query (random Q/K)", fontsize=11, y=1.05)
    fig.tight_layout()
    names.append(_save(fig, out / "locality_attention.svg"))

    # 4) Near-mass bars
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    xs = np.arange(len(PE_ORDER))
    near = [loc[p]["mean_near_mass"] for p in PE_ORDER]
    ent = [loc[p]["mean_entropy"] / np.log(m["locality"]["seq_len"]) for p in PE_ORDER]
    w = 0.35
    ax.bar(xs - w / 2, near, w, color=COLS[2], label="near mass (r=2)")
    ax.bar(xs + w / 2, ent, w, color=COLS[0], label="norm. entropy")
    ax.set_xticks(xs)
    ax.set_xticklabels([PE_LABELS[p] for p in PE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Attention locality (higher near mass = prefers nearby keys)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "locality_bars.svg"))

    # 5) Length extrapolation curves
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for pe, c in zip(PE_ORDER, COLS):
        block = next(e for e in m["extrapolation"] if e["pe"] == pe)
        xs = [b["seq_len"] for b in block["by_length"]]
        ys = [b["acc"] for b in block["by_length"]]
        ax.plot(xs, ys, "o-", color=c, label=PE_LABELS[pe])
    ax.axvline(m["config"]["L_train"], color="#555", ls="--", lw=1, label="L_train")
    ax.set_xlabel("test sequence length")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Length extrapolation (train short, test long)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "length_extrapolation.svg"))

    # 6) Acc at L_train vs L_max bars
    L_max = m["headline"]["L_max"]
    L_tr = m["headline"]["L_train"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    xs = np.arange(len(PE_ORDER))
    acc_tr, acc_mx = [], []
    for pe in PE_ORDER:
        block = next(e for e in m["extrapolation"] if e["pe"] == pe)
        acc_tr.append(block["train_acc"])
        acc_mx.append(next(b["acc"] for b in block["by_length"] if b["seq_len"] == L_max))
    w = 0.35
    ax.bar(xs - w / 2, acc_tr, w, color=COLS[3], label=f"acc @ L={L_tr}")
    ax.bar(xs + w / 2, acc_mx, w, color=COLS[2], label=f"acc @ L={L_max}")
    ax.set_xticks(xs)
    ax.set_xticklabels([PE_LABELS[p] for p in PE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-1 acc")
    ax.set_title("Train-length vs extrapolated-length accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "accuracy_comparison.svg"))

    return names


def write_results_md(out: Path, m: Dict[str, Any], plot_names: List[str]) -> None:
    h = m["headline"]
    loc = {p["pe"]: p for p in m["locality"]["per_pe"]}
    lines = [
        "# Results: ai-learn-27-alibi-attention",
        "",
        f"Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).",
        "",
        "## Setup",
        f"- d_model={m['config']['d_model']}, n_heads={m['config']['n_heads']}, "
        f"L_train={m['config']['L_train']}, test lengths={m['config']['test_lengths']}, "
        f"relative offset={m['config']['offset']}.",
        f"- Relative-offset retrieval with {m['config']['n_distractors']} content-matched distractors; "
        "train Q/K only (V/O frozen unused for ranking).",
        f"- Tiny multi-head NumPy attention; {m['config']['steps']} steps, "
        f"logit_scale={m['config']['logit_scale']}.",
        "",
        "## Slope sanity",
        f"- H=4 slopes: `{m['slope_sanity']['n4']['slopes']}` pass={m['slope_sanity']['n4']['pass']}",
        f"- H=8 slopes: `{m['slope_sanity']['n8']['slopes']}` pass={m['slope_sanity']['n8']['pass']}",
        "",
        "## Locality (random Q/K, near radius "
        f"{m['locality']['near_radius']}, T={m['locality']['seq_len']})",
        "| PE | mean near mass | mean entropy |",
        "|---|---|---|",
    ]
    for pe in PE_ORDER:
        lines.append(
            f"| {PE_LABELS[pe]} | {loc[pe]['mean_near_mass']} | {loc[pe]['mean_entropy']} |"
        )
    lines += [
        "",
        "## Length extrapolation (top-1 acc)",
        "| PE | " + " | ".join(f"L={L}" for L in m["config"]["test_lengths"]) + " |",
        "|" + "---|" * (1 + len(m["config"]["test_lengths"])),
    ]
    for pe in PE_ORDER:
        block = next(e for e in m["extrapolation"] if e["pe"] == pe)
        by = {b["seq_len"]: b["acc"] for b in block["by_length"]}
        cells = " | ".join(str(by.get(L, "—")) for L in m["config"]["test_lengths"])
        lines.append(f"| {PE_LABELS[pe]} | {cells} |")
    lines += [
        "",
        "## Headline",
        f"- Slope sanity: **pass={h['slope_sanity_pass']}**.",
        f"- Locality near-mass: ALiBi **{h['alibi_near_mass']}** > no-PE **{h['none_near_mass']}** "
        f"(absolute {h['absolute_near_mass']}).",
        f"- Acc @ L_train={h['L_train']}: ALiBi **{h['alibi_acc_Ltrain']}**, "
        f"absolute **{h['absolute_acc_Ltrain']}**, no-PE **{h['none_acc_Ltrain']}**.",
        f"- Acc @ L_max={h['L_max']}: ALiBi **{h['alibi_acc_Lmax']}** vs absolute "
        f"**{h['absolute_acc_Lmax']}** (ALiBi wins: {h['alibi_beats_absolute_at_Lmax']}).",
        "",
        "## Plots",
    ]
    for name in plot_names:
        lines.append(f"- ![{name}]({name})")
    lines += [
        "",
        "## Notes",
        "- ALiBi bias is parameter-free and defined for any length — absolute PE can fit "
        "short contexts but degrades when query/key slots move past L_train.",
        "- Softmax uses an educational logit_scale so peaks are visible; rankings use the same scores.",
        "- Toy multi-head attention, not a full Transformer — the ALiBi math matches Press et al. 2021.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
