"""Inspect the three homework datasets; missing files download to ~/.ogbench/data.

Run all tasks: uv run src/scripts/inspect_datasets.py
Run one task:  uv run src/scripts/inspect_datasets.py --task antmaze-medium
"""

import argparse

import numpy as np
import ogbench


DATASETS = {
    "cube-single": "cube-single-play-singletask-task1-v0",
    "antsoccer-arena": "antsoccer-arena-navigate-singletask-task1-v0",
    "antmaze-medium": "antmaze-medium-navigate-singletask-task1-v0",
}


def inspect_split(name, dataset):
    size = len(dataset["observations"])
    print(f"\n{name}: {size:,} transitions")
    if size == 0:
        return

    print(f"  {'field':<18} {'shape':<18} {'dtype':<8} {'min':>10} {'max':>10} {'mean':>10}")
    for key, values in dataset.items():
        print(
            f"  {key:<18} {str(values.shape):<18} {str(values.dtype):<8} "
            f"{values.min():>10.4g} {values.max():>10.4g} {values.mean(dtype=np.float64):>10.4g}"
        )

    ends = np.flatnonzero(dataset["terminals"])
    if ends.size:
        lengths = np.diff(np.r_[-1, ends])
        print(
            f"  Completed trajectories: {len(lengths):,}; "
            f"length min/mean/max: {lengths.min()}/{lengths.mean():.1f}/{lengths.max()}"
        )
    rewards, counts = np.unique(dataset["rewards"], return_counts=True)
    print("  Reward counts: " + ", ".join(f"{r:g}: {c:,}" for r, c in zip(rewards, counts)))
    # The homework uses masks for Bellman backups; terminals mark trajectory ends.
    print(f"  Training done rate (1 - masks): {1 - dataset['masks'].mean():.2%}")

    print("  First transition:")
    for key, values in dataset.items():
        print(f"    {key}: {values[0]}")


def inspect_task(task):
    dataset_name = DATASETS[task]
    print(f"\n{'=' * 80}\n{task}: {dataset_name}", flush=True)
    env, train, val = ogbench.make_env_and_datasets(dataset_name)
    try:
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        inspect_split("Train", train)
        inspect_split("Validation", val)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DATASETS, help="Inspect one task (default: all three).")
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    for task in [args.task] if args.task else DATASETS:
        inspect_task(task)


if __name__ == "__main__":
    main()
