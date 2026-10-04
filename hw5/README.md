# Homework 5: Offline RL

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Run on the Slurm GPU server

Run the launcher locally with Python 3.9+ and SSH access to `fe.ds`:

```bash
python3 src/scripts/ssh_run.py -- python src/scripts/run.py \
  --run_group=q1 --base_config=sacbc \
  --env_name=cube-single-play-singletask-task1-v0 --seed=0
```

It queries `squeue -u felix020422` on `fe.ds`, selects the first running GPU
allocation (ordered by job ID), expands its node list with `scontrol`, then
uses `srun --jobid` from the frontend to run `uv run` inside that allocation.
This matters when two allocations share a node: a direct SSH login can enter
the other job's GPU cgroup. The default
working directory is `/net/scratch/zhenghao/DRL_homework_spring2026/hw5`.
The remote files and working uv environment are assumed to exist already.
For a single task, everything after `--` is passed to `uv run`; for example, use
`-- python src/scripts/inspect_datasets.py --task antmaze-medium` to inspect data.

To train multiple agents on the same GPU in parallel, put `--njobs` on the SSH
launcher and pass one quoted `JOB` specification per experiment:

```bash
python3 src/scripts/ssh_run.py --push --njobs=2 -- \
  "JOB --run_group=q1 --base_config=iql --seed=0 --alpha=3" \
  "JOB --run_group=q1 --base_config=iql --seed=1 --alpha=3" \
  "JOB --run_group=q1 --base_config=iql --seed=2 --alpha=10"
```

This runs at most two jobs at once, starting the third when a worker finishes.
Each job is an independent `python src/scripts/run.py` process in the same uv
environment. They all use GPU 0 by default; put `--which_gpu=1` inside each `JOB`
to share a different visible GPU. Each process keeps its own model and dataset
in memory, so choose `--njobs` to fit the allocation.

Use distinct seeds or experiment names: the launcher rejects jobs that would
share an output directory. Each job appends stdout and stderr to
`exp/<run_group>/<exp_name>/console.log`, alongside its usual checkpoints and
metrics. The terminal shows job starts and exit codes. Failed jobs do not stop
the remaining queue, but the launcher exits nonzero if any job fails. Ctrl+C
stops the workers and cancels queued jobs. `--pull` downloads results after all
jobs finish. This mode runs entirely on the remote server without Modal; sync
the updated `ssh_run.py` first using `--push` or your usual file transfer.

Use `--job-id JOB_ID` or `--node l001` before `--` to choose an allocation or
node. Only nodes in your running GPU allocations are accepted. `--remote-dir`
can select another project directory under `/net/scratch/zhenghao`.
`--dry-run` performs allocation discovery and prints the SSH command without
running it or copying files. The launcher runs in the foreground and returns
the remote command's exit status; keep the SSH connection and allocation alive.

For the ten QAM chunks, sync the updated source once and start the allocations
with `launch.sh`. Each job ID gets an independent background session and a log
under `.launch_qam/` on the local machine:

```bash
python3 src/scripts/ssh_run.py --sync-only --push
bash launch.sh --job-ids=12345,12346,12347 --training-steps=1000000
bash launch.sh --status
bash launch.sh --stop-job-id=12346
```

The first ID runs chunk 1 (`puzzle-3x3-play`), the second runs chunk 2
(`scene-play`), and so on in the order defined in
`src/scripts/launch_qam.py`. To restart one chunk with a new allocation, give
its original 1-based chunk index, for example
`bash launch.sh --job-ids=23456 --chunk-indices=2`. Stopping one job ID
terminates its running workers and cancels its queued tasks; the other IDs
continue. The start command returns after launching, so closing its terminal
does not stop the jobs. `--stop-job-id` manages launches started through
`launch.sh`. `--status` shows `starting` while the remote Python environment
loads and `workers running` after the training processes start. The per-ID logs
show each worker's exit code. A checkpoint completed at a shorter training
target resumes when `--training-steps` is increased.

Optional file transfers use `scp` through `fe.ds`, which shares `/net/scratch`
with the GPU nodes:

```bash
# Upload src/ and project metadata, run training, then download exp/.
python3 src/scripts/ssh_run.py --push --pull -- python src/scripts/run.py \
  --run_group=q1 --base_config=iql --seed=0

# Copy results later, even when the GPU allocation has ended.
python3 src/scripts/ssh_run.py --sync-only --pull

# Upload code without launching a task.
python3 src/scripts/ssh_run.py --sync-only --push
```

Uploads copy `src/`, `pyproject.toml`, `uv.lock`, `requirements.txt`, and
`README.md`. Downloads merge remote `exp/` into local `exp/`; matching files
are overwritten. `--local-dir` changes the local project directory for transfers.
The remote `.venv` is reused. Runtime caches, uv-managed Python downloads,
temporary files, and W&B files go in the remote project's `.remote-cache/`;
checkpoints and logs go in `exp/`. Existing OGBench datasets are loaded from
`/net/scratch/zhenghao/.ogbench/`. The homework's training configs and dataset
inspector honor `OGBENCH_DATASET_DIR`, which the launcher sets to that path.
Sync these source changes before the first run
(for example, with `--push`). W&B uses the remote account's existing credentials.

## Examples

Here are some example commands. Run them in the `hw5` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```


* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.
  * Re-running the same command will resume from the latest checkpoint in `exp/` if the job was preempted.
  * Use different seeds or explicitly set `exp_name` if you want to force a separate fresh run with the same arguments.


* To run 4 jobs on a single GPU in parallel on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000"
  ```

* To download logs and checkpoints from Modal:
  ```bash
  mkdir -p exp
  uv run modal volume get hw5-offline-rl-volume / exp
  ```
