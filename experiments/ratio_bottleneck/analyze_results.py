#!/usr/bin/env python3
"""
Ratio Bottleneck Experiment — Analysis Script

Auto-discovers CFM training logs by `log_cfm_ratio_*` pattern.
Produces:
  1. Summary table (CSV + stdout)
  2. Figure 1: Quantum-Classical MSE gap vs latent_dim (KEY RESULT)
  3. Figure 2: Learning curves (3 panels, one per latent_dim)
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUTPUT_DIR = os.path.dirname(__file__)

LATENT_DIMS = [32, 64, 128]
N_OBS = 28  # pairwise k=2 on 8 qubits: C(8,2) = 28

# Pattern: log_cfm_ratio_{q|c}_lat{dim}_{slurm_id}.csv
QUANTUM_PATTERN = "log_cfm_ratio_q_lat{dim}_*.csv"
CLASSICAL_PATTERN = "log_cfm_ratio_c_lat{dim}_*.csv"

# Number of trailing epochs for mean/std computation
TAIL_EPOCHS = 20


# ── Helpers ────────────────────────────────────────────────────────────────

def find_log(results_dir: str, pattern: str, dim: int) -> str | None:
    """Find the most recent log file matching the pattern for a given dim."""
    glob_pat = os.path.join(results_dir, pattern.format(dim=dim))
    matches = sorted(glob.glob(glob_pat))
    if not matches:
        return None
    return matches[-1]  # latest by name (highest SLURM ID)


def load_log(path: str) -> pd.DataFrame:
    """Load a CFM training log CSV."""
    df = pd.read_csv(path)
    for col in ["train_loss", "val_loss", "eig_min", "eig_max", "time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def tail_stats(df: pd.DataFrame, col: str = "val_loss",
               n: int = TAIL_EPOCHS) -> tuple[float, float]:
    """Return (mean, std) of the last n epochs of a column."""
    vals = df[col].dropna().values[-n:]
    return float(np.mean(vals)), float(np.std(vals))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ratio bottleneck experiment results")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="Directory containing CSV logs")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory for output figures and summary")
    parser.add_argument("--tail-epochs", type=int, default=TAIL_EPOCHS,
                        help="Number of trailing epochs for stats")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Tail epochs: {args.tail_epochs}")
    print()

    # ── 1. Discover and load logs ──────────────────────────────────────────

    rows = []
    q_dfs = {}
    c_dfs = {}
    missing = []

    for dim in LATENT_DIMS:
        ratio = N_OBS / dim

        q_path = find_log(results_dir, QUANTUM_PATTERN, dim)
        c_path = find_log(results_dir, CLASSICAL_PATTERN, dim)

        if q_path is None:
            missing.append(f"quantum lat={dim}")
        if c_path is None:
            missing.append(f"classical lat={dim}")

        if q_path is None or c_path is None:
            continue

        q_df = load_log(q_path)
        c_df = load_log(c_path)
        q_dfs[dim] = q_df
        c_dfs[dim] = c_df

        q_mean, q_std = tail_stats(q_df, "val_loss", args.tail_epochs)
        c_mean, c_std = tail_stats(c_df, "val_loss", args.tail_epochs)
        gap = q_mean - c_mean
        gap_pct = (gap / c_mean * 100) if c_mean > 0 else float("nan")

        rows.append(dict(
            latent_dim=dim,
            ratio=f"{ratio:.4f}",
            quantum_val_mse=f"{q_mean:.6f}",
            quantum_std=f"{q_std:.6f}",
            classical_val_mse=f"{c_mean:.6f}",
            classical_std=f"{c_std:.6f}",
            gap=f"{gap:.6f}",
            gap_pct=f"{gap_pct:.1f}",
            quantum_epochs=len(q_df),
            classical_epochs=len(c_df),
            quantum_log=os.path.basename(q_path),
            classical_log=os.path.basename(c_path),
        ))

    if missing:
        print(f"WARNING: Missing logs for: {', '.join(missing)}")
        print()

    if not rows:
        print("ERROR: No complete (quantum + classical) pairs found.")
        print("Expected log files matching:")
        for dim in LATENT_DIMS:
            print(f"  {QUANTUM_PATTERN.format(dim=dim)}")
            print(f"  {CLASSICAL_PATTERN.format(dim=dim)}")
        sys.exit(1)

    # ── 2. Summary table ──────────────────────────────────────────────────

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "ratio_bottleneck_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=" * 80)
    print("RATIO BOTTLENECK EXPERIMENT — SUMMARY")
    print("=" * 80)
    print()
    print(f"{'lat_dim':>8} {'ratio':>8} {'Q val_mse':>12} {'C val_mse':>12} "
          f"{'gap':>10} {'gap%':>8} {'Q epochs':>9} {'C epochs':>9}")
    print("-" * 80)
    for r in rows:
        print(f"{r['latent_dim']:>8} {r['ratio']:>8} {r['quantum_val_mse']:>12} "
              f"{r['classical_val_mse']:>12} {r['gap']:>10} "
              f"{r['gap_pct']:>7}% {r['quantum_epochs']:>9} "
              f"{r['classical_epochs']:>9}")
    print("-" * 80)
    print()

    # ── Interpretation ─────────────────────────────────────────────────────

    if len(rows) >= 2:
        gaps = [float(r["gap"]) for r in rows]
        dims = [r["latent_dim"] for r in rows]
        if gaps[-1] > gaps[0] * 1.2:
            verdict = "Gap WIDENS -> ratio IS the bottleneck"
        elif gaps[-1] < gaps[0] * 0.8:
            verdict = "Gap NARROWS -> larger latent helps quantum"
        else:
            verdict = "Gap roughly CONSTANT -> ratio is NOT the main bottleneck"
        print(f"Interpretation: {verdict}")
        print(f"  Gap at lat={dims[0]}: {gaps[0]:.6f}")
        print(f"  Gap at lat={dims[-1]}: {gaps[-1]:.6f}")
        print()

    print(f"Summary CSV: {summary_path}")
    print()

    # ── 3. Figure 1: Gap vs Latent Dim (THE KEY RESULT) ──────────────────

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    dims_plot = [r["latent_dim"] for r in rows]
    gaps_plot = [float(r["gap"]) for r in rows]
    q_stds = [float(r["quantum_std"]) for r in rows]
    c_stds = [float(r["classical_std"]) for r in rows]
    # Propagate uncertainty: gap_std = sqrt(q_std^2 + c_std^2)
    gap_stds = [np.sqrt(qs**2 + cs**2) for qs, cs in zip(q_stds, c_stds)]

    ax1.errorbar(dims_plot, gaps_plot, yerr=gap_stds,
                 marker="o", capsize=5, linewidth=2, markersize=8,
                 color="#2196F3", ecolor="#90CAF9")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Latent Dimension", fontsize=12)
    ax1.set_ylabel("Quantum - Classical MSE Gap", fontsize=12)
    ax1.set_title("Ratio Bottleneck: Does the Gap Widen?", fontsize=13)
    ax1.set_xticks(dims_plot)

    # Annotate ratio values
    for d, g in zip(dims_plot, gaps_plot):
        r = N_OBS / d
        ax1.annotate(f"ratio={r:.2f}", (d, g),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="#666666")

    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_gap_vs_latent_dim.pdf")
    fig1.savefig(fig1_path, dpi=150)
    fig1.savefig(fig1_path.replace(".pdf", ".png"), dpi=150)
    print(f"Figure 1: {fig1_path}")
    plt.close(fig1)

    # ── 4. Figure 2: Learning Curves (3 panels) ──────────────────────────

    n_panels = len(q_dfs)
    if n_panels > 0:
        fig2, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4),
                                  sharey=True)
        if n_panels == 1:
            axes = [axes]

        for i, dim in enumerate(sorted(q_dfs.keys())):
            ax = axes[i]
            q_df = q_dfs[dim]
            c_df = c_dfs[dim]
            ratio = N_OBS / dim

            ax.plot(q_df["epoch"], q_df["val_loss"],
                    label="Quantum", color="#E53935", alpha=0.85, linewidth=1.2)
            ax.plot(c_df["epoch"], c_df["val_loss"],
                    label="Classical", color="#1E88E5", alpha=0.85, linewidth=1.2)

            ax.set_xlabel("Epoch", fontsize=11)
            if i == 0:
                ax.set_ylabel("Val MSE Loss", fontsize=11)
            ax.set_title(f"lat_dim={dim}  (ratio={ratio:.2f})", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig2.suptitle("Learning Curves: Quantum vs Classical by Latent Dim",
                      fontsize=13, y=1.02)
        fig2.tight_layout()
        fig2_path = os.path.join(output_dir, "fig2_learning_curves.pdf")
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        fig2.savefig(fig2_path.replace(".pdf", ".png"), dpi=150,
                     bbox_inches="tight")
        print(f"Figure 2: {fig2_path}")
        plt.close(fig2)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
