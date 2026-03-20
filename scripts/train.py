"""
Training script for NanoGPT language model
"""

import argparse
import csv
import json
import os
import random
import time
import torch
import yaml

import numpy as np

from model.gpt_model import NanoGPTLanguageModel
from training.trainer import Trainer
from utils import config
from utils.dataset import get_dataloader
from utils.config import load_config
from utils.logging import setup_experiment_logging


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )

    args = parser.parse_args()

    # ---------------- Load Config ----------------

    config = load_config(args.config)

    # ---------------- Set Seed ----------------

    set_seed(config["experiment"]["seed"])

    # ---------------- Device ----------------

    device = get_device()
    # Disable AMP if not CUDA
    if device != "cuda":
        config["systems"]["amp"] = False
    # ---------------- Output Directory ----------------

    output_dir = os.path.join(
        config["logging"]["output_dir"],
        config["experiment"]["name"],
    )
    results_dir = os.path.join(
        config["logging"]["results_dir"],
        config["experiment"]["name"],
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Logging ----------------

    setup_experiment_logging(results_dir)

    print(f"Using device: {device}")
    print(f"Experiment: {config['experiment']['name']}")

    # Save config for reproducibility
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # ---------------- Dataset ----------------

    train_loader = get_dataloader(
        file_path=config["data"]["dataset_path"],
        batch_size=config["training"]["batch_size"],
        block_size=config["model"]["block_size"],
        tokenizer_name=config["tokenizer"]["type"],
        config=config,
    )

    # ---------------- Model ----------------

    model = NanoGPTLanguageModel(
        block_size=config["model"]["block_size"],
        vocab_size=config["model"]["vocab_size"],
        n_layer=config["model"]["n_layer"],
        n_head=config["model"]["n_head"],
        n_embed=config["model"]["n_embed"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Model size
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # ---------------- Optimizer ----------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
    )

    # ---------------- Trainer ----------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        accumulation_steps=config["systems"]["gradient_accumulation_steps"],
        use_amp=config["systems"]["amp"],
    )

    # ---------------- Training Loop ----------------
    budget = config["training"]["raw_text_budget"]
    max_iters = config["training"].get("max_iters", None)
    effective_batch_size = (
        config["training"]["batch_size"]
        * config["systems"]["gradient_accumulation_steps"]
    )

    # logging metrics for summary and CSV
    tokens_processed = 0
    log_tokens = 0
    interval_losses = 0.0
    row = {}

    csv_path = os.path.join(results_dir, "training_metrics.csv")

    global_start_time = time.time()
    log_start = time.time()
    for step, (x, y) in enumerate(train_loader):

        step_start = time.time()
        loss = trainer.train_step(x, y)
        step_time = time.time() - step_start

        tokens = x.numel()
        tokens_processed += tokens
        log_tokens += tokens

        # ----- Logging -----

        if step > 0 and step % config["logging"]["log_interval"] == 0:

            elapsed = time.time() - log_start
            tok_per_sec = log_tokens / elapsed if elapsed > 0 else 0
            progress = (tokens_processed / budget * 100) if budget > 0 else 0
            interval_losses += loss

            # GPU memory
            if device == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated() / 1e6
            else:
                gpu_mem = -1

            print(
                f"step {step} | "
                f"loss {loss:.4f} | "
                f"{tok_per_sec:.0f} tokens/sec | "
                f"{progress:.1f}% complete"
            )

            # Log metrics to CSV
            row = {
                "experiment": config["experiment"]["name"],
                "step": step,
                "progress_percent": progress,
                "loss": loss,
                "tokens_per_sec": tok_per_sec,
                "step_time_ms": step_time * 1000,
                "interval_time_sec": elapsed,
                "avg_step_time_ms": (
                    elapsed / max(1, config["logging"]["log_interval"])
                )
                * 1000,
                "gpu_memory_mb": gpu_mem,
                "tokens_processed": tokens_processed,
                "tokens_per_step": tokens,
                "effective_batch_size": effective_batch_size,
                "lr": optimizer.param_groups[0]["lr"],
            }

            # ---- Save metrics to CSV ----
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            # Reset CUDA memory stats
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            log_tokens = 0
            log_start = time.time()

        # ----- Checkpoint -----

        if step % config["logging"]["save_checkpoint_interval"] == 0 and step > 0:
            ckpt_path = os.path.join(output_dir, f"step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # ----- Stop condition -----

        if tokens_processed >= config["training"]["raw_text_budget"]:
            print(f"\nReached raw text budget of  {tokens_processed} / {budget} tokens")
            break

        if max_iters and step >= max_iters:
            print(f"\nReached maximum iterations of {step} / {max_iters}")
            break

    # ---------------- Save Final Model ----------------

    model_path = os.path.join(output_dir, "model.pt")

    torch.save(model.state_dict(), model_path)
    total_time = time.time() - global_start_time

    summary = {
        "experiment": config["experiment"]["name"],
        "model_parameters": num_params,
        "trainable_parameters": trainable_params,
        "device": device,
        "final_loss": loss,
        "avg_tokens_per_sec": tokens_processed / total_time,
        "total_tokens_processed": tokens_processed,
        "avg_interval_loss": interval_losses
        / (step // config["logging"]["log_interval"] + 1),
        "total_training_time_sec": total_time,
        "total_steps": step + 1,
    }

    summary_path = os.path.join(results_dir, "summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nTraining finished")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
