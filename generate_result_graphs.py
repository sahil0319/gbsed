#!/usr/bin/env python3
"""
generate_result_graphs.py -- reads results_matrix.csv (columns: sequence,
config, frames_total, frames_delivered, delivery_rate, bit_exact_rate,
mean_edge_f1, prediction, prob_class0, prob_class1, ground_truth, match)
and writes a set of comparison/interpretability graphs as PNGs.

This script does not display anything -- it only writes files, so it's
safe to run headless (e.g. over SSH/WSL with no display).

Usage:
    python3 generate_result_graphs.py path/to/results_matrix.csv
    python3 generate_result_graphs.py path/to/results_matrix.csv --out-dir result_graphs

Requires: pandas, matplotlib, numpy
    pip install pandas matplotlib numpy
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless -- write files only, never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ordered from least to most degraded, used as x-axis order on every plot
# so degradation reads left-to-right consistently. Configs not present in
# the data are simply skipped -- e.g. if CAV_Extreme never completed, it
# just won't appear rather than breaking anything.
CONFIG_ORDER = [
    "Baseline",
    "Noise_Low",
    "Noise_Medium",
    "Noise_High",
    "CAV_Good",
    "CAV_Moderate",
    "CAV_Bad",
    "CAV_Extreme",
]

SEQ_COLORS = {
    "seq1": "#3b6fa0",
    "seq2": "#c1666b",
}


def load_results(csv_path):
    if not os.path.isfile(csv_path):
        sys.exit(f"File not found: {csv_path}")

    # engine="python" + on_bad_lines="warn" so a malformed row (wrong number
    # of fields, stray text, etc.) gets printed as a warning and skipped
    # instead of crashing the whole script.
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")

    expected_cols = {
        "sequence", "config", "frames_total", "frames_delivered",
        "delivery_rate", "bit_exact_rate", "mean_edge_f1",
        "prediction", "prob_class0", "prob_class1", "ground_truth", "match",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing expected column(s): {sorted(missing)}\n"
                  f"Found columns: {list(df.columns)}")

    # Coerce numeric columns; anything unparseable becomes NaN rather than
    # raising, so a corrupted single field doesn't take down the whole row.
    numeric_cols = ["frames_total", "frames_delivered", "delivery_rate",
                     "bit_exact_rate", "mean_edge_f1", "prediction",
                     "prob_class0", "prob_class1", "ground_truth"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # "match" may come in as bool, "True"/"False" strings, or blank/NaN for
    # rows where prediction never ran.
    def parse_match(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.strip().lower() == "true":
                return True
            if v.strip().lower() == "false":
                return False
        return np.nan
    df["match"] = df["match"].apply(parse_match)

    bad_rows = df[df[["sequence", "config"]].isna().any(axis=1)]
    if len(bad_rows):
        print(f"WARNING: dropping {len(bad_rows)} row(s) with unparseable sequence/config", file=sys.stderr)
        df = df.dropna(subset=["sequence", "config"])

    return df


def config_sort_key(config):
    if config in CONFIG_ORDER:
        return (0, CONFIG_ORDER.index(config))
    return (1, config)


def ordered_configs(df):
    return sorted(df["config"].unique(), key=config_sort_key)


def ordered_seqs(df):
    return sorted(df["sequence"].unique())


def pivot_metric(df, seqs, configs, metric):
    """Return {seq: [value_per_config_in_order]} with NaN for missing (seq, config)."""
    out = {}
    for seq in seqs:
        sub = df[df["sequence"] == seq].set_index("config")
        out[seq] = [sub[metric].get(c, np.nan) for c in configs]
    return out


# ---------------------------------------------------------------------
# Fidelity plots
# ---------------------------------------------------------------------

def plot_grouped_bar(df, seqs, configs, metric, title, ylabel, out_path):
    values = pivot_metric(df, seqs, configs, metric)
    x = np.arange(len(configs))
    width = 0.8 / max(len(seqs), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.1), 5))
    for i, seq in enumerate(seqs):
        offset = (i - (len(seqs) - 1) / 2) * width
        ax.bar(x + offset, values[seq], width, label=seq, color=SEQ_COLORS.get(seq))

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_metrics_overview(df, seqs, configs, out_path):
    panels = [
        ("delivery_rate", "Delivery rate"),
        ("bit_exact_rate", "Bit-exact rate"),
        ("mean_edge_f1", "Mean edge F1"),
    ]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(configs) * 1.6), 5), sharex=True)

    for ax, (metric, label) in zip(axes, panels):
        values = pivot_metric(df, seqs, configs, metric)
        for seq in seqs:
            ax.plot(x, values[seq], marker="o", label=seq, color=SEQ_COLORS.get(seq))
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.suptitle("Fidelity metrics across channel conditions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------
# Risk-prediction plots
# ---------------------------------------------------------------------

def plot_match_heatmap(df, seqs, configs, out_path):
    """Grid of seq x config, colored green (correct) / red (incorrect) /
    light-gray hatched (no prediction available for that run)."""
    grid = np.full((len(seqs), len(configs)), np.nan)
    for i, seq in enumerate(seqs):
        sub = df[df["sequence"] == seq].set_index("config")
        for j, c in enumerate(configs):
            if c in sub.index:
                m = sub.loc[c, "match"]
                if isinstance(m, pd.Series):  # duplicate rows for same (seq, config)
                    m = m.iloc[0]
                if pd.isna(m):
                    grid[i, j] = np.nan
                else:
                    grid[i, j] = 1.0 if bool(m) else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.1), 2 + len(seqs)))
    cmap = matplotlib.colors.ListedColormap(["#c1443c", "#3f9142"])  # 0=red, 1=green
    cmap.set_bad(color="#d9d9d9")
    masked = np.ma.masked_invalid(grid)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(seqs)):
        for j in range(len(configs)):
            val = grid[i, j]
            if np.isnan(val):
                text, color = "N/A", "#555"
            else:
                text, color = ("correct" if val == 1 else "wrong"), "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(seqs)))
    ax.set_yticklabels(seqs)
    ax.set_title("Risk prediction: correct vs. incorrect vs. no prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_true_class_confidence(df, seqs, configs, out_path):
    """Probability the model assigned to the ACTUAL (ground-truth) class,
    across configs -- shows confidence collapsing as the channel degrades,
    independent of whether the final argmax prediction happened to flip."""
    def true_class_prob(row):
        if pd.isna(row["ground_truth"]) or pd.isna(row["prob_class0"]) or pd.isna(row["prob_class1"]):
            return np.nan
        return row["prob_class1"] if int(row["ground_truth"]) == 1 else row["prob_class0"]

    df = df.copy()
    df["prob_true_class"] = df.apply(true_class_prob, axis=1)
    values = pivot_metric(df, seqs, configs, "prob_true_class")

    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.1), 5))
    for seq in seqs:
        ax.plot(x, values[seq], marker="o", label=seq, color=SEQ_COLORS.get(seq))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="decision boundary (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("P(ground-truth class)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model confidence in the correct class across channel conditions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_fidelity_vs_confidence(df, out_path):
    """Interpretability scatter: does transmission fidelity (mean edge F1)
    track with the model's confidence in the correct class? Colored by
    whether the final prediction was actually correct."""
    def true_class_prob(row):
        if pd.isna(row["ground_truth"]) or pd.isna(row["prob_class0"]) or pd.isna(row["prob_class1"]):
            return np.nan
        return row["prob_class1"] if int(row["ground_truth"]) == 1 else row["prob_class0"]

    d = df.copy()
    d["prob_true_class"] = d.apply(true_class_prob, axis=1)
    d = d.dropna(subset=["mean_edge_f1", "prob_true_class"])
    if d.empty:
        print("Skipping fidelity-vs-confidence scatter: no rows with both fidelity and prediction data")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for is_match, marker, label in [(True, "o", "correct prediction"),
                                     (False, "x", "incorrect prediction")]:
        sub = d[d["match"] == is_match]
        if sub.empty:
            continue
        for seq in sub["sequence"].unique():
            seq_sub = sub[sub["sequence"] == seq]
            ax.scatter(seq_sub["mean_edge_f1"], seq_sub["prob_true_class"],
                       marker=marker, color=SEQ_COLORS.get(seq), s=80,
                       label=f"{seq} ({label})", alpha=0.85,
                       edgecolors=("black" if marker == "o" else None), linewidths=0.5)

    # Points that share the same (fidelity, confidence) coordinate (common
    # here, since e.g. Baseline and CAV_Good often produce identical
    # metrics) get their labels stacked vertically instead of overlapping.
    seen_coords = {}
    for _, row in d.iterrows():
        coord = (round(row["mean_edge_f1"], 4), round(row["prob_true_class"], 4))
        stack_idx = seen_coords.get(coord, 0)
        seen_coords[coord] = stack_idx + 1
        ax.annotate(row["config"], (row["mean_edge_f1"], row["prob_true_class"]),
                    fontsize=7, xytext=(6, 6 + stack_idx * 11), textcoords="offset points")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean edge F1 (transmission fidelity)")
    ax.set_ylabel("P(ground-truth class)")
    ax.set_title("Does transmission fidelity track classifier confidence?")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_accuracy_summary(df, seqs, out_path):
    """Overall accuracy per sequence, over only the runs that have a
    prediction (excludes configs where risk assessment never completed)."""
    rates = []
    counts = []
    for seq in seqs:
        sub = df[(df["sequence"] == seq) & df["match"].notna()]
        counts.append(len(sub))
        rates.append(sub["match"].mean() if len(sub) else np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(seqs, rates, color=[SEQ_COLORS.get(s) for s in seqs])
    for bar, rate, n in zip(bars, rates, counts):
        if not np.isnan(rate):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                    f"{rate:.0%} (n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy (match rate)")
    ax.set_title("Overall risk-prediction accuracy per sequence\n(over configs with a completed prediction)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def report_missing_predictions(df, seqs, configs):
    print("\n--- Coverage check ---")
    for seq in seqs:
        sub = df[df["sequence"] == seq].set_index("config")
        missing = [c for c in configs if c not in sub.index or pd.isna(sub.loc[c, "prediction"])]
        if missing:
            print(f"  {seq}: no risk prediction for: {', '.join(missing)}")
        else:
            print(f"  {seq}: risk prediction present for all {len(configs)} configs")
    all_configs_in_order = [c for c in CONFIG_ORDER if c not in configs]
    if all_configs_in_order:
        print(f"  Configs with NO data at all (missing from CSV entirely): {', '.join(all_configs_in_order)}")
    print("-----------------------\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Path to results_matrix.csv")
    ap.add_argument("--out-dir", default="result_graphs", help="Output folder for PNGs (default: result_graphs)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.csv_path)
    seqs = ordered_seqs(df)
    configs = ordered_configs(df)
    print(f"Loaded {len(df)} row(s). Sequences: {seqs}. Configs found: {configs}")

    report_missing_predictions(df, seqs, configs)

    # Fidelity comparison
    plot_grouped_bar(df, seqs, configs, "delivery_rate",
                      "Frame delivery rate by channel condition", "Delivery rate",
                      os.path.join(args.out_dir, "01_delivery_rate_by_config.png"))
    plot_grouped_bar(df, seqs, configs, "bit_exact_rate",
                      "Bit-exact rate by channel condition", "Bit-exact rate",
                      os.path.join(args.out_dir, "02_bit_exact_rate_by_config.png"))
    plot_metrics_overview(df, seqs, configs,
                           os.path.join(args.out_dir, "03_fidelity_metrics_overview.png"))

    # Risk prediction comparison
    plot_match_heatmap(df, seqs, configs,
                        os.path.join(args.out_dir, "04_prediction_correctness_heatmap.png"))
    plot_true_class_confidence(df, seqs, configs,
                                os.path.join(args.out_dir, "05_true_class_confidence_trend.png"))
    plot_accuracy_summary(df, seqs,
                           os.path.join(args.out_dir, "06_accuracy_summary.png"))

    # Combined interpretability
    plot_fidelity_vs_confidence(df, os.path.join(args.out_dir, "07_fidelity_vs_confidence_scatter.png"))

    print(f"\nDone. All graphs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()