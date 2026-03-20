"""Benchmark script for training NanoGPT

Compares:
- Different training configurations (e.g., batch size, learning rate, model size)
- Training efficiency and convergence

Outputs:
results/training_benchmark.csv
"""

import os
import time
import json
import csv
import subprocess
import yaml


CONFIG_DIR = "./configs"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "training_benchmark.csv")


def get_config_files():
    return [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".yaml") and f != "common.yaml"
    ]


def run_experiment(config_path):
    print(f"\nRunning: {config_path}")

    start = time.time()

    result = subprocess.run(
        ["python", "-m", "scripts.train", "--config", config_path],
        capture_output=True,
        text=True,
    )

    duration = time.time() - start

    if result.returncode != 0:
        print(f"❌ Failed: {config_path}")
        print(result.stderr)
        return None

    return duration


def load_summary(experiment_name):
    path = os.path.join(RESULTS_DIR, experiment_name, "summary.json")

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def main():
    configs = get_config_files()

    rows = []

    for config_path in configs:

        # extract experiment name from yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        exp_name = cfg["experiment"]["name"]

        duration = run_experiment(config_path)

        if duration is None:
            continue

        summary = load_summary(exp_name)

        if summary is None:
            continue

        row = {
            "experiment": exp_name,
            "tokens_per_sec": summary["avg_tokens_per_sec"],
            "final_loss": summary["final_loss"],
            "total_time_sec": duration,
            "total_tokens": summary["total_tokens_processed"],
        }

        rows.append(row)

    # Save aggregated benchmark
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Benchmark saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
