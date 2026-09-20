#!/usr/bin/env python3
r"""Run a uv command on an existing Slurm GPU allocation, via fe.ds.

Example (run locally, without installing the homework's dependencies):
    python3 src/scripts/ssh_run.py -- python src/scripts/run.py --base_config=iql

Parallel training on the same GPU:
    python3 src/scripts/ssh_run.py --njobs=2 -- \
        "JOB --base_config=iql --seed=0" "JOB --base_config=iql --seed=1"
"""

import argparse
import os
import posixpath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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
PARALLEL_ENTRYPOINT = (
    "import sys; from scripts.ssh_run import run_jobs; "
    "sys.exit(run_jobs(sys.argv[2:], int(sys.argv[1])))"
)


def ssh_command(host, command, tty=False):
    return ["ssh", "-tt" if tty else "-T", *SSH_OPTIONS, host, command]


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


def gpu_command(args, node, command, tty=False):
    # Authenticate the second hop from the frontend, just as with interactive
    # `ssh fe.ds` followed by `ssh l001`; no local compute-node key is needed.
    inner = ssh_command(f"{args.user}@{node}", shlex.join(["bash", "-lc", command]), tty=tty)
    return ssh_command(args.frontend, shlex.join(inner), tty=tty)


def split_job_specs(job_specs):
    if not job_specs:
        raise ValueError("--njobs requires at least one quoted 'JOB ...' specification")
    jobs = []
    for index, spec in enumerate(job_specs, 1):
        try:
            words = shlex.split(spec)
        except ValueError as exc:
            raise ValueError(f"Job {index}: {exc}") from exc
        if not words or words[0] != "JOB":
            raise ValueError(f"Job {index} must be a quoted 'JOB <run.py arguments>' string")
        if any(word.split("=", 1)[0] == "--njobs" for word in words[1:]):
            raise ValueError(f"Job {index}: set --njobs on ssh_run.py, not inside a JOB")
        jobs.append(words[1:])
    return jobs


def stop_jobs(running):
    """Terminate each worker's process group, including its child processes."""
    for _, process, _ in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for _, process, log in running:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            # The worker may exit before a child that ignored SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            log.close()


def run_jobs(job_specs, njobs):
    """Run on the GPU node, inside one uv environment; no Modal is involved."""
    # Import the training parser only remotely, keeping the local launcher
    # dependency-free. Fresh Python processes give each agent its own CUDA state.
    from scripts.run import get_run_name, setup_arguments

    jobs, directories = [], set()
    try:
        if njobs < 1:
            raise ValueError("--njobs must be a positive integer")
        for index, argv in enumerate(split_job_specs(job_specs), 1):
            args = setup_arguments(argv)
            if args.njobs is not None or args.job_specs:
                raise ValueError(f"Job {index} must describe a single training run")
            directory = Path("exp") / args.run_group / get_run_name(args)
            resolved = directory.resolve()
            if resolved in directories:
                raise ValueError(
                    f"Job {index} shares output directory {directory} with another job; "
                    "use distinct --seed or --exp_name values"
                )
            directories.add(resolved)
            jobs.append((argv, directory))
    except ValueError as exc:
        print(f"ssh_run: {exc}", file=sys.stderr)
        return 2

    interrupted = None

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = signum

    previous_handlers = {
        sig: signal.signal(sig, on_signal)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    running = []
    next_job, status = 0, 0
    print(f"Running {len(jobs)} training jobs, at most {njobs} concurrently.", flush=True)
    try:
        while next_job < len(jobs) or running:
            if interrupted is not None:
                print("Stopping running jobs and cancelling queued jobs.", flush=True)
                return 128 + interrupted
            while next_job < len(jobs) and len(running) < njobs and interrupted is None:
                argv, directory = jobs[next_job]
                index = next_job + 1
                next_job += 1
                directory.mkdir(parents=True, exist_ok=True)
                log_path = directory / "console.log"
                log = log_path.open("ab")
                command = [sys.executable, "-u", "src/scripts/run.py", *argv]
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                log.write(f"\n[{timestamp}] {shlex.join(command)}\n".encode())
                log.flush()
                try:
                    process = subprocess.Popen(
                        command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except OSError:
                    log.close()
                    raise
                running.append((index, process, log))
                print(f"[job {index}/{len(jobs)}] Started PID {process.pid}; log: {log_path}", flush=True)
            for entry in running[:]:
                index, process, log = entry
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                running.remove(entry)
                code = returncode if returncode >= 0 else 128 - returncode
                status = status or code
                print(f"[job {index}/{len(jobs)}] Finished with exit code {code}.", flush=True)
            if running:
                time.sleep(0.1)
        return status
    finally:
        try:
            stop_jobs(running)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


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
        epilog=("After --, pass a uv run command or, with --njobs, quoted 'JOB ...' strings. "
                "With no command, run.py is used."),
    )
    parser.add_argument("--frontend", default="fe.ds", help="SSH frontend alias (default: fe.ds)")
    parser.add_argument("--user", default="felix020422", help="Slurm and compute-node user")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Absolute project path on shared scratch")
    parser.add_argument("--local-dir", type=Path, default=PROJECT_DIR, help="Local project directory for scp")
    parser.add_argument("--job-id", help="Use this running GPU allocation instead of the first one")
    parser.add_argument("--node", help="Choose a node belonging to your running GPU allocation")
    parser.add_argument("--njobs", type=int, help="Run at most N training JOBs concurrently on the selected node")
    parser.add_argument("--push", action="store_true", help="scp source and project metadata before running")
    parser.add_argument("--pull", action="store_true", help="scp exp/ back after running, including on failure")
    parser.add_argument("--sync-only", action="store_true", help="Only perform --push/--pull; no GPU allocation needed")
    parser.add_argument("--dry-run", action="store_true", help="Discover the GPU and print the command without writes")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="uv run arguments, or quoted JOB specs with --njobs")
    args = parser.parse_args()
    args.remote_dir = posixpath.normpath(args.remote_dir)
    if not PurePosixPath(args.remote_dir).is_relative_to(SCRATCH_DIR):
        parser.error(f"--remote-dir must be an absolute directory under {SCRATCH_DIR}")
    args.local_dir = args.local_dir.expanduser().resolve()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.sync_only and (args.command or args.njobs is not None or not (args.push or args.pull)):
        parser.error("--sync-only requires --push or --pull and does not take a command or --njobs")
    if args.njobs is not None:
        if args.njobs < 1:
            parser.error("--njobs must be a positive integer")
        try:
            split_job_specs(args.command)
        except ValueError as exc:
            parser.error(str(exc))
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
    if args.njobs is None:
        remote_command = args.command
        print(f"Running: uv run {shlex.join(remote_command)}", flush=True)
    else:
        remote_command = ["python", "-c", PARALLEL_ENTRYPOINT, str(args.njobs), *args.command]
        print(f"Running {len(args.command)} JOBs with --njobs={args.njobs}", flush=True)
    if args.push:
        push_sources(args)
    # PTYs at both SSH hops propagate interrupts and hangups to the scheduler,
    # allowing it to clean up workers when the connection closes.
    command = gpu_command(
        args, node, run_script(args.remote_dir, remote_command), tty=args.njobs is not None,
    )
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
