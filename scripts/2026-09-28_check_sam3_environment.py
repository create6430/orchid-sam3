"""Read-only PLAN step 4 preflight; does not install packages or load weights.

python scripts/2026-09-28_check_sam3_environment.py --sam3-repo PATH --checkpoint PATH
Run with the intended training Python interpreter. Exit 1 means prerequisites
are missing or unverified. Even a successful preflight is not a training dry run.
"""

import argparse
import csv
from importlib import metadata
import platform
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    rows = []

    def record(check, status, value, detail=""):
        rows.append(dict(check=check, status=status, value=value, detail=detail))

    record("python", "INFO", platform.python_version(), sys.executable)
    record("os", "INFO", platform.platform())
    try:
        output = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader,nounits"])
        devices = list(csv.reader(output.splitlines(), skipinitialspace=True))
        matches = []
        for index, (name, memory, driver) in enumerate(devices):
            matches.append("RTX 5090" in name)
            record(f"gpu_{index}", "INFO", name, f"total_memory_MiB={memory}; driver={driver}")
        record("planned_gpu", "PASS" if any(matches) else "FAIL", "RTX 5090",
               "SPEC requires RTX 5090; no automatic batch-size or resolution changes")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        record("planned_gpu", "FAIL", "unavailable", str(exc))

    for package in ("torch", "torchvision", "sam3"):
        try:
            record(package, "INFO", metadata.version(package), "Installed version; import/runtime tested separately")
        except metadata.PackageNotFoundError:
            record(package, "FAIL", "not installed")
    try:
        import torch
        import torchvision
        record("torchvision_import", "PASS", torchvision.__version__)
        available = torch.cuda.is_available()
        record("torch_cuda", "PASS" if available else "FAIL", str(torch.version.cuda),
               "PyTorch build CUDA version; nvidia-smi driver capability is not an installed toolkit")
        if available:
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    bf16 = torch.cuda.is_bf16_supported()
                    name = torch.cuda.get_device_name(index)
                    record(f"torch_gpu_{index}", "INFO", name,
                           f"capability={torch.cuda.get_device_capability(index)}; bf16={bf16}")
                    if "RTX 5090" in name:
                        record("target_bf16", "PASS" if bf16 else "FAIL", bf16)
    except Exception as exc:
        record("torch_runtime", "FAIL", type(exc).__name__, str(exc))

    if args.sam3_repo:
        try:
            repo = args.sam3_repo.resolve()
            if not (repo / "sam3/model_builder.py").is_file():
                raise ValueError("Expected SAM 3 source tree with sam3/model_builder.py")
            commit = command(["git", "-C", str(repo), "rev-parse", "HEAD"])
            dirty = command(["git", "-C", str(repo), "status", "--porcelain"])
            record("sam3_source", "PENDING" if dirty else "INFO", commit,
                   f"{repo}; dirty={bool(dirty)}; installed package/source alignment still requires dry run")
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            record("sam3_source", "FAIL", "unverified", str(exc))
    else:
        record("sam3_source", "PENDING", "not supplied", "Use --sam3-repo to record the source commit")
    checkpoint = args.checkpoint
    if checkpoint and checkpoint.is_file() and checkpoint.stat().st_size > 0:
        record("checkpoint", "INFO", str(checkpoint.resolve()),
               f"bytes={checkpoint.stat().st_size}; existence only, model compatibility not tested")
    else:
        record("checkpoint", "PENDING", "missing", "Supply an existing SAM 3 base checkpoint with --checkpoint")
    record("training_dry_run", "PENDING", "not executed",
           "Requires real training batch=4, bf16 autocast, forward/backward, checkpoint save/resume")
    record("overall", "FAIL" if any(r["status"] == "FAIL" for r in rows) else "PENDING",
           "step 4 incomplete", "This script inventories prerequisites only")
    report = ROOT / "results/tables/2026-09-28_sam3_environment.csv"
    if not report.resolve().is_relative_to(ROOT / "results/tables"):
        parser.error("Report must resolve within results/tables")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "status", "value", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['status']}: {row['check']}={row['value']} {row['detail']}")
    print(f"Report: {report}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
