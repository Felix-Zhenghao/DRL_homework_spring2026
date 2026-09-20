#!/usr/bin/env python3
"""Run a uv command on an existing Slurm GPU allocation, via fe.ds.

Example (run locally, without installing the homework's dependencies):
    python3 src/scripts/ssh_run.py -- python src/scripts/run.py --base_config=iql
"""

import argparse
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRATCH_DIR = PurePosixPath("/net/scratch/zhenghao")
DEFAULT_REMOTE_DIR = str(SCRATCH_DIR / "DRL_homework_spring2026/hw5")
SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
]
SOURCE_FILES = ("src", "pyproject.toml", "uv.lock", "requirements.txt", "README.md")


def ssh_command(host, command):
    return ["ssh", "-T", *SSH_OPTIONS, host, command]


def remote_output(frontend, command):
    return subprocess.run(
        ssh_command(frontend, shlex.join(command)),
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def select_node(args):
    # AllocTRES includes both --gres=gpu and --gpus allocations. Pending jobs
    # have no usable node, and CPU-only allocations must not be selected.
    queue = remote_output(args.frontend, [
        "squeue", "--noheader", f"--user={args.user}", "--states=RUNNING",
        "--sort=i", "--Format=JobID:32,NodeList:256,tres-alloc:1024",
    ])
    for line in queue.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected squeue output: {line!r}")
        job_id, node_list, resources = fields
        if args.job_id and job_id != args.job_id:
            continue
        if not re.search(r"(?:^|,)gres/gpu(?::[^,=]+)?=[1-9][0-9]*(?:,|$)", resources):
            continue
        nodes = remote_output(args.frontend, [
            "scontrol", "show", "hostnames", node_list,
        ]).splitlines()
        if args.node:
            nodes = [node for node in nodes if node == args.node]
        if nodes:
            return job_id, nodes[0]
    requested = f" (job={args.job_id}, node={args.node})" if args.job_id or args.node else ""
    raise RuntimeError(
        f"No running GPU allocation for {args.user} on {args.frontend}{requested}. "
        "Start a GPU allocation first, or check --job-id/--node."
    )


def gpu_command(args, node, command):
    # Authenticate the second hop from the frontend, just as with interactive
    # `ssh fe.ds` followed by `ssh l001`; no local compute-node key is needed.
    inner = ssh_command(f"{args.user}@{node}", shlex.join(["bash", "-lc", command]))
    return ssh_command(args.frontend, shlex.join(inner))


def run_script(remote_dir, command):
    cache = PurePosixPath(remote_dir) / ".remote-cache"
    paths = {
        "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": cache / "python",
        "UV_PYTHON_BIN_DIR": cache / "bin",
        "XDG_CACHE_HOME": cache / "cache",
        "XDG_CONFIG_HOME": cache / "config",
        "XDG_DATA_HOME": cache / "data",
        "TMPDIR": cache / "tmp",
        "OGBENCH_DATASET_DIR": SCRATCH_DIR / ".ogbench",
        "WANDB_DIR": cache / "wandb",
        "WANDB_CACHE_DIR": cache / "wandb-cache",
        "WANDB_DATA_DIR": cache / "wandb-data",
        "WANDB_CONFIG_DIR": cache / "wandb-config",
        "MPLCONFIGDIR": cache / "matplotlib",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TRITON_CACHE_DIR": cache / "triton",
    }
    lines = ["set -eu", "cd -- " + shlex.quote(remote_dir)]
    lines.append(shlex.join(["mkdir", "-p", "--", *map(str, paths.values())]))
    lines.extend(f"export {key}={shlex.quote(str(value))}" for key, value in paths.items())
    lines.extend([
        'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"',
        "export PYTHONUNBUFFERED=1",
        'export MUJOCO_GL="${MUJOCO_GL:-egl}"',
        'hw5_uv=$(command -v uv || true)',
        'if [ -z "$hw5_uv" ]; then hw5_uv="$HOME/.local/bin/uv"; fi',
        'if [ ! -x "$hw5_uv" ]; then printf "uv is not installed on this node.\\n" >&2; exit 127; fi',
        "nvidia-smi -L",
        'exec "$hw5_uv" run ' + shlex.join(command),
    ])
    return "\n".join(lines)


def push_sources(args):
    print(f"Uploading source to {args.frontend}:{args.remote_dir}", flush=True)
    if args.dry_run:
        return
    # /net/scratch is shared by the frontend and compute nodes. scp can use
    # the frontend's existing SSH authentication and write directly there.
    remote_output(args.frontend, ["mkdir", "-p", "--", args.remote_dir])
    with tempfile.TemporaryDirectory(prefix="hw5-upload-") as staging:
        sources = []
        for name in SOURCE_FILES:
            source, target = args.local_dir / name, Path(staging) / name
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(source, target)
            sources.append(str(target))
        subprocess.run([
            "scp", *SSH_OPTIONS, "-r", "--", *sources,
            f"{args.frontend}:{args.remote_dir}/",
        ], check=True)


def pull_results(args):
    print(f"Downloading {args.remote_dir}/exp to {args.local_dir}/exp", flush=True)
    if args.dry_run:
        return
    args.local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "scp", *SSH_OPTIONS, "-r", "--",
        f"{args.frontend}:{args.remote_dir}/exp", str(args.local_dir) + "/",
    ], check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Everything after -- is passed to remote uv run. With no command, run.py is used.",
    )
    parser.add_argument("--frontend", default="fe.ds", help="SSH frontend alias (default: fe.ds)")
    parser.add_argument("--user", default="felix020422", help="Slurm and compute-node user")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Absolute project path on shared scratch")
    parser.add_argument("--local-dir", type=Path, default=PROJECT_DIR, help="Local project directory for scp")
    parser.add_argument("--job-id", help="Use this running GPU allocation instead of the first one")
    parser.add_argument("--node", help="Choose a node belonging to your running GPU allocation")
    parser.add_argument("--push", action="store_true", help="scp source and project metadata before running")
    parser.add_argument("--pull", action="store_true", help="scp exp/ back after running, including on failure")
    parser.add_argument("--sync-only", action="store_true", help="Only perform --push/--pull; no GPU allocation needed")
    parser.add_argument("--dry-run", action="store_true", help="Discover the GPU and print the command without writes")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Arguments for remote uv run, after --")
    args = parser.parse_args()
    args.remote_dir = posixpath.normpath(args.remote_dir)
    if not PurePosixPath(args.remote_dir).is_relative_to(SCRATCH_DIR):
        parser.error(f"--remote-dir must be an absolute directory under {SCRATCH_DIR}")
    args.local_dir = args.local_dir.expanduser().resolve()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.sync_only and (args.command or not (args.push or args.pull)):
        parser.error("--sync-only requires --push or --pull and does not take a command")
    args.command = args.command or ["python", "src/scripts/run.py"]
    return args


def main():
    args = parse_args()
    if args.sync_only:
        if args.push:
            push_sources(args)
        if args.pull:
            pull_results(args)
        return 0

    job_id, node = select_node(args)
    print(f"Using Slurm job {job_id} on {node} via {args.frontend}", flush=True)
    print(f"Working directory: {args.remote_dir}", flush=True)
    print(f"Running: uv run {shlex.join(args.command)}", flush=True)
    if args.push:
        push_sources(args)
    command = gpu_command(args, node, run_script(args.remote_dir, args.command))
    if args.dry_run:
        print(shlex.join(command))
        status = 0
    else:
        try:
            status = subprocess.run(command).returncode
        except KeyboardInterrupt:
            status = 130
    if args.pull:
        try:
            pull_results(args)
        except (OSError, subprocess.CalledProcessError) as exc:
            print("Result download failed; results remain on remote scratch.", file=sys.stderr)
            status = status or getattr(exc, "returncode", 1)
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ssh_run: {exc}", file=sys.stderr)
        sys.exit(exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1)
    except KeyboardInterrupt:
        sys.exit(130)
