"""
Configuration utilities

Supports:
- Loading YAML configs
- Merging experiment configs with common configs
"""

import yaml
import copy


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