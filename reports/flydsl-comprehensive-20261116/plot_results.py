"""Generate standalone plots from the three-pass paired CSVs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def plot_native():
    source = ROOT / "native-paired.csv"
    if not source.exists():
        return
    with source.open() as f:
        rows = [r for r in csv.DictReader(f) if r["policy"] == "auto"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, regime in zip(axes, ("hot", "rotating")):
        subset = [r for r in rows if r["regime"] == regime]
        for band, color in zip(("M<=16", "17<=M<=128", "129<=M<=1024", "M>1024"),
                               ("#0072B2", "#009E73", "#E69F00", "#CC79A7")):
            group = [r for r in subset if r["m_band"] == band]
            ax.scatter([int(r["n"]) * int(r["k"]) / 2**20 for r in group],
                       [float(r["speedup"]) for r in group], s=13, alpha=.55,
                       label=band, color=color, edgecolors="none")
        ax.axhline(1, color="#333333", linewidth=1)
        ax.axhspan(.95, 1.05, color="#999999", alpha=.12)
        ax.set(xscale="log", yscale="log", xlabel="N × K / 2²⁰ (weight elements)",
               title="Hot weights" if regime == "hot" else "Read-evicted weights")
        ax.grid(True, alpha=.15)
    axes[0].set_ylabel("Native FP8 latency / FlyDSL MXFP4 latency\nAbove 1 favors FlyDSL")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Native FP8 comparison on R9700 (gfx1201)\n{len(rows)//2:,} distinct shapes · median of 3 pass ratios", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "native-speedup.png", dpi=180)
    fig.savefig(ROOT / "native-speedup.svg")
    plt.close(fig)


def plot_vllm():
    source = ROOT / "vllm-paired.csv"
    if not source.exists():
        return
    with source.open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, regime in zip(axes, ("hot", "rotating")):
        subset = [r for r in rows if r["regime"] == regime]
        for band, color in zip(("M<=16", "17<=M<=128", "129<=M<=1024", "M>1024"),
                               ("#0072B2", "#009E73", "#E69F00", "#CC79A7")):
            group = [r for r in subset if r["m_band"] == band]
            ax.scatter([int(r["n"]) * int(r["k"]) / 2**20 for r in group],
                       [float(r["speedup"]) for r in group], s=13, alpha=.55,
                       label=band, color=color, edgecolors="none")
        ax.axhline(1, color="#333333", linewidth=1)
        ax.axhspan(.95, 1.05, color="#999999", alpha=.12)
        ax.set(xscale="log", yscale="log", xlabel="N × K / 2²⁰ (weight elements)",
               title="Hot weights" if regime == "hot" else "Read-evicted weights")
        ax.grid(True, alpha=.15)
    axes[0].set_ylabel("W4A4 emulation latency / FlyDSL W4A8 latency\nAbove 1 favors FlyDSL")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(f"vLLM W4A4 emulation vs FlyDSL W4A8 · R9700 (gfx1201)\n{len(rows)//2:,} supported shapes · median of 3 pass ratios", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "vllm-speedup.png", dpi=180)
    fig.savefig(ROOT / "vllm-speedup.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_native()
    plot_vllm()
