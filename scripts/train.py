"""
Training script for NanoGPT language model
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import yaml

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

    # ---------------- Output Directory ----------------

    output_dir = os.path.join(
        config["logging"]["output_dir"],
        config["experiment"]["name"],
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Logging ----------------

    setup_experiment_logging(output_dir)

    print(f"Using device: {device}")
    print(f"Experiment: {config['experiment']['name']}")

    # Save config for reproducibility
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # ---------------- Dataset ----------------

    train_loader = get_dataloader(
        file_path=config["data"]["train_file"],
        batch_size=config["training"]["batch_size"],
        block_size=config["model"]["block_size"],
        tokenizer_name=config["tokenizer"]["type"],
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
    print(f"Model parameters: {num_params/1e6:.2f}M")

    # ---------------- Optimizer ----------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
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
    tokens_processed = 0
    log_tokens = 0
    log_start = time.time()

    budget = config["training"]["raw_text_budget"]
    max_iters = config["training"].get("max_iters", None)

    for step, (x, y) in enumerate(train_loader):

        loss = trainer.train_step(x, y)

        tokens = x.numel()
        tokens_processed += tokens
        log_tokens += tokens

        # ----- Logging -----

        if step % config["logging"]["log_interval"] == 0:

            elapsed = time.time() - log_start
            tok_per_sec = log_tokens / elapsed
            progress = tokens_processed / budget * 100

            print(
                f"step {step} | "
                f"loss {loss:.4f} | "
                f"{tok_per_sec:.0f} tokens/sec"
                f"{progress:.1f}% complete"
            )

            log_tokens = 0
            log_start = time.time()

        # ----- Checkpoint -----

        if (
            step % config["logging"]["save_checkpoint_interval"] == 0
            and step > 0
        ):
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

    print(f"\nTraining finished")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()