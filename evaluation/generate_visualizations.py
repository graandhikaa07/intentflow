"""
Script to generate publication-quality figures comparing Fine-Tuned DistilBERT
vs. Zero-Shot Gemini 3.1 Flash Lite baseline on the held-out 240-query test set.

Loads strictly from existing completed metrics:
- artifacts/reports/distilbert_test_metrics.json
- artifacts/reports/zero_shot_llm_test_metrics.json

Generates:
1. artifacts/figures/accuracy_comparison.png
2. artifacts/figures/f1_comparison.png
3. artifacts/figures/per_class_f1_comparison.png
4. artifacts/figures/latency_comparison.png
5. artifacts/figures/confusion_matrix_distilbert.png
6. artifacts/figures/confusion_matrix_gemini.png
"""

import os
import sys
import json

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean academic style
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

DISTILBERT_COLOR = "#1f77b4"  # Academic blue
GEMINI_COLOR = "#d95f02"      # Academic warm vermillion

def load_metrics():
    with open("artifacts/reports/distilbert_test_metrics.json", "r", encoding="utf-8") as f:
        distilbert = json.load(f)
    with open("artifacts/reports/zero_shot_llm_test_metrics.json", "r", encoding="utf-8") as f:
        gemini = json.load(f)
    return distilbert, gemini

def plot_accuracy_comparison(d_metrics, g_metrics, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["Fine-Tuned\nDistilBERT", "Zero-Shot\nGemini 3.1 Flash-Lite"]
    accuracies = [d_metrics["test_accuracy"] * 100, g_metrics["test_accuracy"] * 100]
    colors = [DISTILBERT_COLOR, GEMINI_COLOR]

    bars = ax.bar(models, accuracies, color=colors, width=0.48, edgecolor="black", linewidth=1.2, zorder=3)
    ax.set_ylabel("Overall Accuracy (%)", fontweight="bold")
    ax.set_title("Overall Accuracy Comparison (Test N = 240)", pad=14, fontweight="bold")
    ax.set_ylim(80, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f"{acc:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=12)

    diff = accuracies[0] - accuracies[1]
    ax.text(0.5, 82, f"Diff = +{diff:.2f}% (DistilBERT advantage)",
            ha="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")

def plot_f1_comparison(d_metrics, g_metrics, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_names = ["Macro F1-Score", "Weighted F1-Score"]
    
    d_f1 = [d_metrics["macro_avg"]["f1"] * 100, d_metrics["weighted_avg"]["f1"] * 100]
    g_f1 = [g_metrics["macro_avg"]["f1"] * 100, g_metrics["weighted_avg"]["f1"] * 100]

    x = np.arange(len(metrics_names))
    width = 0.32

    rects1 = ax.bar(x - width/2, d_f1, width, label="Fine-Tuned DistilBERT", color=DISTILBERT_COLOR, edgecolor="black", linewidth=1.2, zorder=3)
    rects2 = ax.bar(x + width/2, g_f1, width, label="Zero-Shot Gemini 3.1 Flash-Lite", color=GEMINI_COLOR, edgecolor="black", linewidth=1.2, zorder=3)

    ax.set_ylabel("F1-Score (%)", fontweight="bold")
    ax.set_title("Macro & Weighted F1-Score Comparison", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontweight="bold")
    ax.set_ylim(80, 105)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "f1_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")

def plot_per_class_f1(d_metrics, g_metrics, out_dir):
    classes = ["MATHDS", "TXTSRC", "SEMADPT", "CSTRSET"]
    d_scores = [d_metrics["per_class"][c]["f1"] * 100 for c in classes]
    g_scores = [g_metrics["per_class"][c]["f1"] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width/2, d_scores, width, label="Fine-Tuned DistilBERT", color=DISTILBERT_COLOR, edgecolor="black", linewidth=1.1, zorder=3)
    rects2 = ax.bar(x + width/2, g_scores, width, label="Zero-Shot Gemini 3.1 Flash-Lite", color=GEMINI_COLOR, edgecolor="black", linewidth=1.1, zorder=3)

    ax.set_ylabel("F1-Score (%)", fontweight="bold")
    ax.set_xlabel("Project Intent Class", fontweight="bold", labelpad=8)
    ax.set_title("Per-Class F1-Score Comparison", pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(75, 106)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "per_class_f1_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")

def plot_latency_comparison(d_metrics, g_metrics, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

    metrics = ["Mean", "P50 (Median)", "P95"]
    d_lat = [d_metrics["latency"]["mean_ms"], d_metrics["latency"]["p50_ms"], d_metrics["latency"]["p95_ms"]]
    g_lat = [g_metrics["latency"]["mean_ms"], g_metrics["latency"]["p50_ms"], g_metrics["latency"]["p95_ms"]]

    # Panel 1: Log scale grouped comparison
    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width/2, d_lat, width, label="Fine-Tuned DistilBERT (Local CPU)", color=DISTILBERT_COLOR, edgecolor="black", linewidth=1.1, zorder=3)
    rects2 = ax1.bar(x + width/2, g_lat, width, label="Zero-Shot Gemini (Cloud API)", color=GEMINI_COLOR, edgecolor="black", linewidth=1.1, zorder=3)

    ax1.set_yscale("log")
    ax1.set_ylabel("Inference Latency (ms) [Log Scale]", fontweight="bold")
    ax1.set_title("(A) Latency Distribution (Log Scale)", pad=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontweight="bold")
    ax1.set_ylim(10, 8000)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, which="both", zorder=0)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}ms", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.0f}ms", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: Mean Latency Direct Ratio
    models = ["DistilBERT\n(Local CPU)", "Gemini 3.1\n(Cloud API)"]
    mean_vals = [d_lat[0], g_lat[0]]
    speedup = g_lat[0] / d_lat[0]

    bars = ax2.bar(models, mean_vals, color=[DISTILBERT_COLOR, GEMINI_COLOR], width=0.48, edgecolor="black", linewidth=1.1, zorder=3)
    ax2.set_ylabel("Mean Latency (ms) [Linear Scale]", fontweight="bold")
    ax2.set_title("(B) Mean Query Latency", pad=12, fontweight="bold")
    ax2.set_ylim(0, 2700)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, mean_vals):
        ax2.annotate(f"{val:.1f} ms", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax2.text(0.5, 0.72, f"DistilBERT is {speedup:.1f}x Faster\n(32.9 ms vs 2327.6 ms)",
             transform=ax2.transAxes, ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef7ee", edgecolor="#4caf50", linewidth=1.2))

    plt.tight_layout()
    path = os.path.join(out_dir, "latency_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")

def plot_confusion_matrix(cm_data, labels, title, out_path, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cm = np.array(cm_data)

    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=True,
                xticklabels=labels, yticklabels=labels,
                linewidths=1.2, linecolor="#dddddd",
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)

    ax.set_xlabel("Predicted Intent Class", fontweight="bold", labelpad=8)
    ax.set_ylabel("Ground Truth Class", fontweight="bold", labelpad=8)
    ax.set_title(title, pad=14, fontweight="bold", fontsize=11.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")

def main():
    out_dir = "artifacts/figures"
    os.makedirs(out_dir, exist_ok=True)

    d_metrics, g_metrics = load_metrics()

    print("[*] Generating publication-quality figures under artifacts/figures/...")
    plot_accuracy_comparison(d_metrics, g_metrics, out_dir)
    plot_f1_comparison(d_metrics, g_metrics, out_dir)
    plot_per_class_f1(d_metrics, g_metrics, out_dir)
    plot_latency_comparison(d_metrics, g_metrics, out_dir)

    labels = d_metrics["confusion_matrix"]["labels"]
    plot_confusion_matrix(d_metrics["confusion_matrix"]["matrix"], labels,
                          "Fine-Tuned DistilBERT Confusion Matrix (N = 240)",
                          os.path.join(out_dir, "confusion_matrix_distilbert.png"),
                          cmap="Blues")

    plot_confusion_matrix(g_metrics["confusion_matrix"]["matrix"], labels,
                          "Zero-Shot Gemini 3.1 Flash-Lite Confusion Matrix (N = 240)",
                          os.path.join(out_dir, "confusion_matrix_gemini.png"),
                          cmap="YlOrBr")

    print("[OK] All 6 figures successfully generated!")

if __name__ == "__main__":
    main()
