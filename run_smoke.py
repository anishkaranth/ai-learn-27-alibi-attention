#!/usr/bin/env python3
"""ALiBi smoke: slopes, locality, length extrapolation → results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from alibi import check_slope_sanity
from experiments import bias_matrix_demo, length_extrapolation, locality_attention_mass
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "d_model": 32,
    "n_heads": 4,
    "L_train": 16,
    "test_lengths": [16, 24, 32, 48],
    "offset": 1,
    "n_distractors": 3,
    "n_train": 512,
    "n_test": 160,
    "steps": 140,
    "lr": 0.08,
    "logit_scale": 8.0,
    "locality_seq_len": 32,
    "near_radius": 2,
}


def _compact(js: str) -> str:
    js = re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
        js,
    )
    return re.sub(
        r"\{\n([^{}\[\]]*?)\n\s*\}",
        lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}",
        js,
    )


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    slope = check_slope_sanity(CFG["n_heads"])
    slope8 = check_slope_sanity(8)
    bias_demo = bias_matrix_demo(seq_len=12, n_heads=CFG["n_heads"])
    loc = locality_attention_mass(
        d_model=CFG["d_model"],
        seq_len=CFG["locality_seq_len"],
        n_heads=CFG["n_heads"],
        seed=SEED,
        near_radius=CFG["near_radius"],
    )

    extrap = []
    for pe in ("none", "absolute", "alibi"):
        extrap.append(
            length_extrapolation(
                pe,
                d_model=CFG["d_model"],
                L_train=CFG["L_train"],
                test_lengths=CFG["test_lengths"],
                offset=CFG["offset"],
                n_heads=CFG["n_heads"],
                n_train=CFG["n_train"],
                n_test=CFG["n_test"],
                steps=CFG["steps"],
                lr=CFG["lr"],
                seed=SEED,
                n_distractors=CFG["n_distractors"],
                logit_scale=CFG["logit_scale"],
            )
        )

    by_pe = {r["pe"]: r for r in extrap}
    L_max = max(CFG["test_lengths"])
    def acc_at(pe: str, L: int) -> float:
        for row in by_pe[pe]["by_length"]:
            if row["seq_len"] == L:
                return float(row["acc"])
        return float("nan")

    wall = round(time.perf_counter() - t0, 2)
    headline = {
        "slope_sanity_pass": slope["pass"],
        "alibi_near_mass": loc["alibi"]["mean_near_mass"],
        "none_near_mass": loc["none"]["mean_near_mass"],
        "absolute_near_mass": loc["absolute"]["mean_near_mass"],
        "alibi_beats_none_locality": loc["alibi_beats_none_near"],
        "alibi_acc_Ltrain": by_pe["alibi"]["train_acc"],
        "absolute_acc_Ltrain": by_pe["absolute"]["train_acc"],
        "none_acc_Ltrain": by_pe["none"]["train_acc"],
        "alibi_acc_Lmax": acc_at("alibi", L_max),
        "absolute_acc_Lmax": acc_at("absolute", L_max),
        "none_acc_Lmax": acc_at("none", L_max),
        "L_train": CFG["L_train"],
        "L_max": L_max,
        "alibi_beats_absolute_at_Lmax": bool(
            acc_at("alibi", L_max) > acc_at("absolute", L_max)
        ),
    }

    extrap_json = []
    for r in extrap:
        extrap_json.append(
            {
                "pe": r["pe"],
                "L_train": r["L_train"],
                "offset": r["offset"],
                "train_acc": r["train_acc"],
                "by_length": r["by_length"],
                "trained": {
                    "final_acc": r["trained"]["final_acc"],
                    "final_mass": r["trained"]["final_mass"],
                    "final_logit_gap": r["trained"]["final_logit_gap"],
                    "chance_acc": r["trained"]["chance_acc"],
                    "history": r["trained"]["history"],
                    "attn_row": r["trained"]["attn_row"],
                    "label": r["trained"]["label"],
                    "query_idx": r["trained"]["query_idx"],
                },
            }
        )

    metrics: Dict[str, Any] = {
        "project": "ai-learn-27-alibi-attention",
        "seed": SEED,
        "config": CFG,
        "slope_sanity": {"n4": slope, "n8": slope8},
        "bias_demo": {
            "slopes": bias_demo["slopes"],
            "seq_len": bias_demo["seq_len"],
            "n_heads": bias_demo["n_heads"],
            "bias_head0": bias_demo["bias_head0"],
            "bias_head_last": bias_demo["bias_head_last"],
        },
        "locality": {
            "seq_len": loc["seq_len"],
            "near_radius": loc["near_radius"],
            "n_heads": loc["n_heads"],
            "per_pe": [
                {
                    "pe": p["pe"],
                    "mean_near_mass": p["mean_near_mass"],
                    "mean_entropy": p["mean_entropy"],
                    "mid_row": p["mid_row"],
                }
                for p in loc["per_pe"]
            ],
            "alibi_beats_none_near": loc["alibi_beats_none_near"],
        },
        "extrapolation": extrap_json,
        "headline": headline,
        "wall_time_s": wall,
    }

    plot_names = make_plots(RESULTS, metrics)
    write_results_md(RESULTS, metrics, plot_names)

    (RESULTS / "metrics.json").write_text(
        _compact(json.dumps(metrics, indent=1)) + "\n", encoding="utf-8"
    )
    shot = {
        "project": "ai-learn-27-alibi-attention",
        "seed": SEED,
        "config": {
            "d_model": CFG["d_model"],
            "n_heads": CFG["n_heads"],
            "L_train": CFG["L_train"],
            "test_lengths": CFG["test_lengths"],
            "offset": CFG["offset"],
            "steps": CFG["steps"],
            "logit_scale": CFG["logit_scale"],
        },
        "headline": headline,
        "wall_time_s": wall,
    }
    (RESULTS / "JSON.shot").write_text(
        _compact(json.dumps(shot, indent=1)) + "\n", encoding="utf-8"
    )

    print("=== ai-learn-27-alibi-attention smoke ===")
    print(f"seed={SEED} wall_time_s={wall}")
    print(f"slope_sanity_pass={slope['pass']} slopes4={slope['slopes']}")
    print(
        f"locality near-mass: none={loc['none']['mean_near_mass']} "
        f"abs={loc['absolute']['mean_near_mass']} alibi={loc['alibi']['mean_near_mass']}"
    )
    for r in extrap_json:
        lens = ", ".join(f"L{b['seq_len']}={b['acc']}" for b in r["by_length"])
        print(f"  {r['pe']:8s} train_acc={r['train_acc']}  {lens}")
    print("plots:", ", ".join(plot_names))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
