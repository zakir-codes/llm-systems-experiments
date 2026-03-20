"""
Plot results for LLM systems experiments

Usage:
    python scripts/plot_results.py --results_dir results/

Outputs:
- results/plots/training/
    - throughput_vs_step.png
    - loss_vs_step.png
    - throughput_bar.png
    - memory_vs_throughput.png
- results/plots/inference/
    - kv_cache_scaling.png
    - latency_vs_seq.png
    - speedup.png
- results/plots/finetuning/
    - lora_loss.png
    - lora_performance.png
    - training_time.png
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Utilities ---------------- #

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_summary(results_dir, exp_name):
    path = os.path.join(results_dir, exp_name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_training_csv(results_dir, exp_name):
    path = os.path.join(results_dir, exp_name, "training_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ---------------- Training Plots ---------------- #

def plot_training_time_series(results_dir, plot_dir, experiments):
    """
    Plot:
    - tokens/sec vs step
    - loss vs step
    """

    ensure_dir(plot_dir)

    plt.figure()
    for exp in experiments:
        df = load_training_csv(results_dir, exp)
        if df is None:
            continue
        plt.plot(df["step"], df["tokens_per_sec"], label=exp)

    plt.xlabel("Step")
    plt.ylabel("Tokens/sec")
    plt.title("Training Throughput vs Step")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "throughput_vs_step.png"))
    plt.close()

    # Loss plot
    plt.figure()
    for exp in experiments:
        df = load_training_csv(results_dir, exp)
        if df is None:
            continue
        plt.plot(df["step"], df["loss"], label=exp)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "loss_vs_step.png"))
    plt.close()


def plot_training_bar(results_dir, plot_dir, experiments):
    """
    Aggregate plots:
    - avg tokens/sec
    - training time
    """

    ensure_dir(plot_dir)

    names = []
    throughput = []
    train_time = []

    for exp in experiments:
        summary = load_summary(results_dir, exp)
        if summary is None:
            continue

        names.append(exp)
        throughput.append(summary["avg_tokens_per_sec"])
        train_time.append(summary["total_training_time_sec"])

    # Throughput
    plt.figure()
    plt.bar(names, throughput)
    plt.ylabel("Tokens/sec")
    plt.title("Average Training Throughput")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(plot_dir, "throughput_bar.png"))
    plt.close()

    # Training time
    plt.figure()
    plt.bar(names, train_time)
    plt.ylabel("Time (sec)")
    plt.title("Total Training Time")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(plot_dir, "training_time.png"))
    plt.close()


def plot_memory_vs_throughput(results_dir, plot_dir, experiments):
    """
    Scatter: memory vs throughput
    """

    ensure_dir(plot_dir)

    x_mem = []
    y_thr = []
    labels = []

    for exp in experiments:
        df = load_training_csv(results_dir, exp)
        summary = load_summary(results_dir, exp)

        if df is None or summary is None:
            continue

        # Use max observed memory
        if "gpu_memory_mb" in df.columns:
            mem = df["gpu_memory_mb"].max()
        else:
            mem = -1

        x_mem.append(mem)
        y_thr.append(summary["avg_tokens_per_sec"])
        labels.append(exp)

    plt.figure()
    plt.scatter(x_mem, y_thr)

    for i, label in enumerate(labels):
        plt.annotate(label, (x_mem[i], y_thr[i]))

    plt.xlabel("GPU Memory (MB)")
    plt.ylabel("Tokens/sec")
    plt.title("Memory vs Throughput Tradeoff")
    plt.savefig(os.path.join(plot_dir, "memory_vs_throughput.png"))
    plt.close()


# ---------------- Inference Plots ---------------- #

def plot_kv_cache(results_dir, plot_dir, exp_name="kv_cache_inference"):
    """
    Expect CSV with:
    - sequence_length
    - tokens_per_sec
    - latency_ms
    - kv_cache (True/False)
    """

    ensure_dir(plot_dir)

    df = load_training_csv(results_dir, exp_name)
    if df is None:
        print("No KV cache data found")
        return

    # Tokens/sec vs sequence length
    plt.figure()
    for key, group in df.groupby("kv_cache"):
        plt.plot(group["sequence_length"], group["tokens_per_sec"], label=f"kv_cache={key}")

    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens/sec")
    plt.title("KV Cache Scaling")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "kv_cache_scaling.png"))
    plt.close()

    # Latency
    plt.figure()
    for key, group in df.groupby("kv_cache"):
        plt.plot(group["sequence_length"], group["latency_ms"], label=f"kv_cache={key}")

    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Sequence Length")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "latency_vs_seq.png"))
    plt.close()


# ---------------- LoRA Plots ---------------- #

def plot_lora(results_dir, plot_dir, experiments):
    """
    Compare LoRA vs full FT
    """

    ensure_dir(plot_dir)

    # Loss vs step
    plt.figure()
    for exp in experiments:
        df = load_training_csv(results_dir, exp)
        if df is None:
            continue
        plt.plot(df["step"], df["loss"], label=exp)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("LoRA vs Full FT - Loss")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "lora_loss.png"))
    plt.close()

    # Params vs performance (if stored)
    names = []
    final_loss = []

    for exp in experiments:
        summary = load_summary(results_dir, exp)
        if summary is None:
            continue

        names.append(exp)
        final_loss.append(summary["final_loss"])

    plt.figure()
    plt.bar(names, final_loss)
    plt.ylabel("Final Loss")
    plt.title("Final Performance Comparison")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(plot_dir, "lora_performance.png"))
    plt.close()


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = args.results_dir
    plots_root = os.path.join(results_dir, "plots")

    # ---- Training ---- #
    training_exps = [
        "baseline_training",
        "amp_training",
        "gradient_accumulation",
    ]

    plot_training_time_series(
        results_dir,
        os.path.join(plots_root, "training"),
        training_exps,
    )

    plot_training_bar(
        results_dir,
        os.path.join(plots_root, "training"),
        training_exps,
    )

    plot_memory_vs_throughput(
        results_dir,
        os.path.join(plots_root, "training"),
        training_exps,
    )

    # ---- Inference ---- #
    plot_kv_cache(
        results_dir,
        os.path.join(plots_root, "inference"),
    )

    # ---- LoRA ---- #
    lora_exps = [
        "baseline_training",   # treat as full FT
        "lora_finetuning",
    ]

    plot_lora(
        results_dir,
        os.path.join(plots_root, "finetuning"),
        lora_exps,
    )

    print(f"Plots saved to {plots_root}")


if __name__ == "__main__":
    main()