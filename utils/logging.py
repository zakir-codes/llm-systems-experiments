"""
Logging utilities for experiments.

Provides:
- Tee: mirror stdout to multiple streams
- setup_experiment_logging: create experiment log file
"""

import os
import sys


class Tee:
    """
    Write output to multiple streams simultaneously.
    Used to mirror stdout to both terminal and log file.
    """

    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


def setup_experiment_logging(output_dir: str):
    """
    Redirect stdout so logs are written to both terminal and file.

    Returns:
        log_file_path
    """

    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, "train.log")

    log_file = open(log_file_path, "a")

    # Mirror prints to terminal + log file
    sys.stdout = Tee(sys.stdout, log_file)

    return log_file_path
