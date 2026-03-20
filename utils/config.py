"""
Configuration utilities

Supports:
- Loading YAML configs
- Merging experiment configs with common configs
- Getting and sorting config files
"""

import yaml
import copy
import os

# Constants
CONFIG_DIR = "./configs"
RESULTS_DIR = "results"
ARTIFACTS_DIR = "artifacts/models"


def load_yaml(path: str):
    """Load YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(base: dict, override: dict):
    """
    Recursively merge two dictionaries.

    Values in `override` replace values in `base`.
    """

    base = copy.deepcopy(base)

    for key, value in override.items():

        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = merge_dicts(base[key], value)

        else:
            base[key] = value

    return base


def load_config(exp_config_path: str):
    """
    Load experiment configuration.

    Steps:
        1. Load common.yaml
        2. Load experiment config
        3. Merge them
    """

    common_config = load_yaml("configs/common.yaml")

    exp_config = load_yaml(exp_config_path)

    config = merge_dicts(common_config, exp_config)

    return config


def get_config_files():
    """Get all config files except common.yaml."""
    return [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".yaml") and f != "common.yaml"
    ]


def sort_configs(config_paths):
    """Sort configs by pipeline order: training -> finetuning -> inference."""
    order = {"training": 0, "finetuning": 1, "inference": 2}

    configs = []
    for path in config_paths:
        cfg = load_yaml(path)  # Use load_yaml instead of load_config to avoid merging
        configs.append((path, cfg))

    return sorted(configs, key=lambda x: order[x[1]["experiment"]["type"]])