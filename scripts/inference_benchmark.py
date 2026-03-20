"""
Benchmark script for inference efficiency in NanoGPT

Compares:
- KV cache vs no KV cache
- Scaling with prompt length

Outputs:
results/inference_benchmark.csv
"""

import os
import time
import csv
import torch
import yaml

from model.gpt_model import NanoGPTLanguageModel
from inference.generate import generate


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

    device = get_device()
    print(f"Using device: {device}")

    # ---------------- Load Model ---------------- #

    model_path = "artifacts/models/baseline_training/model.pt"
    config_path = "artifacts/models/baseline_training/config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = NanoGPTLanguageModel(
        block_size=config["model"]["block_size"],
        vocab_size=config["model"]["vocab_size"],
        n_layer=config["model"]["n_layer"],
        n_head=config["model"]["n_head"],
        n_embed=config["model"]["n_embed"],
        dropout=config["model"]["dropout"],
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # ---------------- Experiment Setup ---------------- #

    prompt_lengths = [32, 64, 128, 256]
    max_new_tokens = 64
    vocab_size = config["model"]["vocab_size"]

    results = []

    # ---------------- Run Experiments ---------------- #

    for prompt_len in prompt_lengths:

        print(f"\n--- Prompt Length: {prompt_len} ---")

        input_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)

        for use_kv_cache in [False, True]:

            tps, latency, total_time = benchmark(
                model,
                input_ids,
                max_new_tokens,
                use_kv_cache,
            )

            result = {
                "prompt_length": prompt_len,
                "kv_cache": use_kv_cache,
                "tokens_per_sec": tps,
                "latency_per_token_ms": latency * 1000,
                "total_time_sec": total_time,
            }

            results.append(result)

            print(
                f"KV Cache: {use_kv_cache} | "
                f"{tps:.2f} tok/s | "
                f"{latency * 1000:.2f} ms/token"
            )

    # ---------------- Save Results ---------------- #

    os.makedirs("results", exist_ok=True)
    output_path = "results/inference_benchmark.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Results saved to {output_path}")


if __name__ == "__main__":
    main()
