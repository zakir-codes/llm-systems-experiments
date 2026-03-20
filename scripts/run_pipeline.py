""" Top-level pipeline runner.

Runs:
    1. All experiments (training → finetuning → inference)
    2. Aggregates results
    3. Generates plots

Pipeline:
    run_experiment.py
            ↓
    benchmark.py
                ↓
        (train.py / finetune.py / inference)
                ↓
        results/*.csv + results/*/*.csv + results/*/*.json + results/*/*.log
            ↓
    plot_results.py
                ↓
        results/plots/*
    
    """


import subprocess
import time


def run_step(cmd, name):
    print(f"\n🚀 Running: {name}")
    start = time.time()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed")

    print(f"✅ Finished: {name} ({time.time() - start:.2f}s)")


def main():

    # ---------------- Run Benchmarks ---------------- #
    run_step(
        ["python", "-m", "scripts.benchmark"],
        "Benchmark Pipeline",
    )

    # ---------------- Generate Plots ---------------- #
    run_step(
        ["python", "-m", "scripts.plot_results"],
        "Plot Generation",
    )

    print("\n🎉 Full pipeline completed successfully!")


if __name__ == "__main__":
    main()