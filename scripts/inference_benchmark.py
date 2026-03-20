"""
Benchmark script for inference efficiency in NanoGPT

Compares:
- KV cache vs no KV cache
- Scaling with prompt length

Outputs:
results/inference_benchmark.csv
"""

import argparse
import os
import time
import csv
import torch
import yaml

from model.gpt_model import NanoGPTLanguageModel
from inference.generate import generate
from utils.config import load_config
from utils.logging import setup_experiment_logging


# ---------------- Device ---------------- #


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ---------------- Benchmark Core ---------------- #


def warmup(model, input_ids, max_new_tokens, use_kv_cache, num_warmup=3):
    """Warmup runs to stabilize performance (important for MPS)"""
    for _ in range(num_warmup):
        _ = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )


def benchmark(
    model,
    input_ids,
    max_new_tokens,
    use_kv_cache,
    num_runs=5,
):
    """
    Run multiple inference passes and return averaged metrics
    """

    # Warmup
    warmup(model, input_ids, max_new_tokens, use_kv_cache)

    times = []

    for _ in range(num_runs):

        start = time.time()

        _ = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv_cache,
        )

        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    tokens_per_sec = max_new_tokens / avg_time
    latency_per_token = avg_time / max_new_tokens

    return tokens_per_sec, latency_per_token, avg_time


# ---------------- Main ---------------- #


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference config file",
    )

    args = parser.parse_args()

    # ---------------- Load Config ----------------

    config = load_config(args.config)
    inference_config = config["inference"]
    benchmark_config = config.get("benchmark", {})
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model: {inference_config['model']}")
    print(f"KV Cache: {inference_config['use_kv_cache']}")

    # ---------------- Output Directory ----------------

    results_dir = os.path.join(
        config["logging"]["results_dir"],
        config["experiment"]["name"],
    )

    os.makedirs(results_dir, exist_ok=True)

    # ---------------- Logging ----------------

    setup_experiment_logging(results_dir)

    # Save config for reproducibility
    config_path = os.path.join(results_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # ---------------- Load Model ----------------

    model_name = inference_config["model"]
    model_path = os.path.join("artifacts/models", model_name, "model.pt")
    base_config_path = os.path.join("artifacts/models", model_name, "config.yaml")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Model config not found: {base_config_path}")

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    model = NanoGPTLanguageModel(
        block_size=base_config["model"]["block_size"],
        vocab_size=base_config["model"]["vocab_size"],
        n_layer=base_config["model"]["n_layer"],
        n_head=base_config["model"]["n_head"],
        n_embed=base_config["model"]["n_embed"],
        dropout=base_config["model"]["dropout"],
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # ---------------- Experiment Setup ----------------

    prompt_lengths = inference_config.get("prompt_lengths", [32, 64, 128, 256])
    max_new_tokens = inference_config.get("max_new_tokens", 64)
    num_runs = inference_config.get("num_runs", 5)
    num_warmup = inference_config.get("num_warmup", 3)
    use_kv_cache = inference_config.get("use_kv_cache", False)
    compare_kv_cache = benchmark_config.get("compare_kv_cache", False)
    vocab_size = base_config["model"]["vocab_size"]

    results = []

    # ---------------- Run Experiments ----------------

    for prompt_len in prompt_lengths:

        print(f"\n--- Prompt Length: {prompt_len} ---")

        input_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)

        # Test both KV cache modes if comparison is enabled
        kv_cache_modes = [False, True] if compare_kv_cache else [use_kv_cache]

        for kv_cache_mode in kv_cache_modes:

            tps, latency, total_time = benchmark(
                model,
                input_ids,
                max_new_tokens,
                kv_cache_mode,
                num_runs=num_runs,
            )

            result = {
                "experiment": config["experiment"]["name"],
                "model": model_name,
                "prompt_length": prompt_len,
                "kv_cache": kv_cache_mode,
                "max_new_tokens": max_new_tokens,
                "tokens_per_sec": tps,
                "latency_per_token_ms": latency * 1000,
                "total_time_sec": total_time,
                "num_runs": num_runs,
            }

            results.append(result)

            print(
                f"KV Cache: {kv_cache_mode} | "
                f"{tps:.2f} tok/s | "
                f"{latency * 1000:.2f} ms/token"
            )

    # ---------------- Save Results ----------------

    output_filename = benchmark_config.get("output_file", "inference_benchmark.csv")
    output_path = os.path.join(results_dir, output_filename)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")
    
    # Also save to the main results directory for benchmark aggregation
    main_output_path = os.path.join("results", f"{config['experiment']['name']}.csv")
    with open(main_output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results also saved to {main_output_path} for benchmark aggregation")


if __name__ == "__main__":
    main()
