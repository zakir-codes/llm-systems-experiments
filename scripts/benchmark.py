"""
Unified benchmark runner for LLM system experiments.

Pipeline:
    training → finetuning → inference

Supports:
- training (baseline, AMP, gradient accumulation)
- finetuning (LoRA vs full FT)
- inference (KV cache vs no cache)

Outputs:
- results/training_benchmark.csv
- results/finetuning_benchmark.csv
- results/inference_benchmark.csv
"""

import os
import time
import json
import csv
import subprocess

from utils.config import (
    get_config_files,
    load_yaml,
    sort_configs,
    RESULTS_DIR,
    ARTIFACTS_DIR,
)


# ---------------- Dependency Check ---------------- #

def check_dependency(cfg):
    exp_type = cfg["experiment"]["type"]

    if exp_type == "finetuning":
        base = cfg["finetuning"]["base_model"]
        path = os.path.join(ARTIFACTS_DIR, base, "model.pt")
        if not os.path.exists(path):
            raise ValueError(f"Missing base model for finetuning: {base}")

    elif exp_type == "inference":
        model = cfg["inference"]["model"]
        path = os.path.join(ARTIFACTS_DIR, model, "model.pt")
        if not os.path.exists(path):
            raise ValueError(f"Missing model for inference: {model}")


# ---------------- Execution ---------------- #

def run_experiment(config_path, cfg):
    exp_type = cfg["experiment"]["type"]

    if exp_type == "training":
        cmd = ["python", "-m", "scripts.train", "--config", config_path]

    elif exp_type == "finetuning":
        cmd = ["python", "-m", "scripts.finetune", "--config", config_path]

    elif exp_type == "inference":
        cmd = ["python", "-m", "scripts.inference_benchmark", "--config", config_path]

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    print(f"\n🚀 Running {cfg['experiment']['name']} ({exp_type})")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start

    if result.returncode != 0:
        print(f"❌ Failed: {config_path}")
        print(result.stderr)
        return None

    return duration


# ---------------- Load Results ---------------- #

def load_training_summary(exp_name):
    path = os.path.join(RESULTS_DIR, exp_name, "summary.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def load_inference_results(exp_name):
    path = os.path.join(RESULTS_DIR, f"{exp_name}_inference.csv")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------- Save Utility ---------------- #

def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------- Main ---------------- #

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config_paths = get_config_files()
    sorted_configs = sort_configs(config_paths)

    training_rows = []
    finetuning_rows = []
    inference_rows = []

    for config_path, cfg in sorted_configs:

        exp_name = cfg["experiment"]["name"]
        exp_type = cfg["experiment"]["type"]

        try:
            check_dependency(cfg)
        except ValueError as e:
            print(f"⚠️ Skipping {exp_name}: {e}")
            continue

        duration = run_experiment(config_path, cfg)

        if duration is None:
            continue

        # -------- Training / Finetuning -------- #
        if exp_type in ["training", "finetuning"]:

            summary = load_training_summary(exp_name)
            if summary is None:
                print(f"⚠️ Missing summary for {exp_name}")
                continue

            row = {
                "experiment": exp_name,
                "type": exp_type,
                "tokens_per_sec": summary["avg_tokens_per_sec"],
                "final_loss": summary["final_loss"],
                "total_time_sec": duration,
                "total_tokens": summary["total_tokens_processed"],
                "model_parameters": summary.get("model_parameters"),
                "trainable_parameters": summary.get("trainable_parameters"),
            }

            if exp_type == "training":
                training_rows.append(row)
            else:
                finetuning_rows.append(row)

        # -------- Inference -------- #
        elif exp_type == "inference":

            results = load_inference_results(exp_name)
            if results is None:
                print(f"⚠️ Missing inference results for {exp_name}")
                continue

            for r in results:
                r["experiment"] = exp_name
                inference_rows.append(r)

    # ---------------- Save ---------------- #

    save_csv(os.path.join(RESULTS_DIR, "training_benchmark.csv"), training_rows)
    save_csv(os.path.join(RESULTS_DIR, "finetuning_benchmark.csv"), finetuning_rows)
    save_csv(os.path.join(RESULTS_DIR, "inference_benchmark.csv"), inference_rows)

    print("\n✅ All benchmarks completed successfully")


if __name__ == "__main__":
    main()