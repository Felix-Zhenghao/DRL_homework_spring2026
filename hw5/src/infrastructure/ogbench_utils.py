"""Load OGBench datasets without racing other workers during downloads."""

import fcntl
import os
from pathlib import Path

import ogbench


def dataset_name_for_env(env_name):
    """Match OGBench's dataset name normalization for task environments."""
    parts = env_name.split("-")
    if "singletask" in parts:
        position = parts.index("singletask")
        return "-".join(parts[:position] + parts[-1:])
    if "oraclerep" in parts:
        return "-".join(parts[:-2] + parts[-1:])
    return env_name


def make_env_and_datasets(env_name, dataset_dir):
    dataset_dir = Path(os.path.expanduser(dataset_dir))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_name_for_env(env_name)
    # OGBench writes to a fixed <dataset>.npz.tmp name. Serialize downloads
    # across processes and nodes sharing this directory, then load in parallel.
    with (dataset_dir / f".{dataset_name}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            ogbench.download_datasets([dataset_name], dataset_dir=str(dataset_dir))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return ogbench.make_env_and_datasets(env_name, dataset_dir=str(dataset_dir))
