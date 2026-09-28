"""Prepare or run pinned SAM 3 full finetuning. See scripts/README.md."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Existing official SAM 3 base weights")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Generate/check configuration; no torch/GPU required")
    parser.add_argument("--smoke-test", action="store_true", help="4 train images, smallest valid prefix covering all classes; default 1 epoch")
    parser.add_argument("--epochs", type=int, help="Default 10 (1 for smoke test)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume this run's checkpoints/checkpoint.pt")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("run-name must use letters, digits, underscore or hyphen")
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke_test else 10)
    if epochs < 1 or args.workers < 0:
        parser.error("epochs must be positive; workers must be nonnegative")
    from omegaconf import OmegaConf
    from orchid_sam3.config import check_source, make_config
    from orchid_sam3.data import CocoSplit, CLASSES, SAM3_COMMIT, artifact_path, file_hash, write_json
    repo = check_source(args.sam3_repo)
    run = artifact_path(ROOT / "results/models" / args.run_name, "results/models")
    last = run / "checkpoints/checkpoint.pt"
    if args.resume and not last.is_file():
        parser.error("--resume requires this run's checkpoints/checkpoint.pt")
    if not args.resume and last.exists():
        parser.error("Run already has checkpoints: choose a new run name or --resume")
    if args.checkpoint and args.checkpoint.resolve().is_relative_to(run):
        parser.error("Base checkpoint must be outside the run output directory")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if not args.prepare_only and (not checkpoint or not checkpoint.is_file()):
        parser.error("Training requires an existing --checkpoint (also when resuming)")
    # Revalidate actual source masks before native training transforms can repair them silently.
    subprocess.run([sys.executable, str(ROOT / "scripts/2026-09-28_validate_orchid_coco.py"), "--quality"], check=True)
    source = ROOT / "data/processed/annotations_area_fixed"
    train, valid = [CocoSplit(source / split / "_annotations.coco.json") for split in ("train", "valid")]
    signature = {"sam3_commit": SAM3_COMMIT, "train": train.fingerprint(), "valid": valid.fingerprint(),
                 "smoke": args.smoke_test, "batch_size": 4, "resolution": 1008,
                 "base_sha256": file_hash(checkpoint) if checkpoint and checkpoint.is_file() else None,
                 "implementation": {p.name: file_hash(p) for p in sorted((ROOT / "src/orchid_sam3").glob("*.py"))}}
    metadata = run / "input_signature.json"
    if args.resume and json.loads(metadata.read_text(encoding="utf-8")) != signature:
        parser.error("Resume input data/base weights/settings differ from saved run")
    run.mkdir(parents=True, exist_ok=True)
    # Remove unused 0=sam3 from DERIVED category table, avoiding an unwanted negative prompt.
    train_copy = deepcopy(train.data)
    train_copy["categories"] = [c for c in train_copy["categories"] if c["id"] in CLASSES]
    train_json = run / "inputs/train.coco.json"
    write_json(train_json, train_copy)
    cfg = make_config(repo, run, checkpoint, train_json, train.path.parent, valid.path,
                      epochs=epochs, workers=args.workers, smoke=args.smoke_test)
    if args.resume:
        cfg.trainer.checkpoint.resume_from = str(last)
    OmegaConf.save(cfg, run / "config_resolved.yaml")
    OmegaConf.save(cfg, run / f"config_target_{epochs:03d}.yaml")
    write_json(metadata, signature)
    write_json(run / "data_limitations.json", {
        "manual_review": "User reports no duplicate plants after manual review (2026-09-29)",
        "capture_sequence_provenance": "unavailable", "pairwise_review_csv": "not supplied",
        "formal_independence_verified": False,
    })
    if args.prepare_only:
        print(f"Configuration prepared: {run}; GPU training/checkpoint runtime NOT validated")
        return 0
    from orchid_sam3.runtime import require_runtime
    environment = require_runtime(repo)
    write_json(run / "environment.json", environment)
    (run / "requirements.actual.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    import torch
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["epoch"] >= epochs:
            parser.error("Checkpoint has already reached --epochs; choose a larger total")
        del saved
    # Single-process NCCL; avoid upstream cluster launcher and changes to its checkout.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    from hydra.utils import instantiate
    try:
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
        write_json(run / "completion.json", {"epochs": epochs, "smoke": args.smoke_test,
                   "training_completed": True, "resume_exercised": args.resume})
    except Exception as exc:
        cause = exc
        while cause is not None:
            if isinstance(cause, torch.cuda.OutOfMemoryError):
                write_json(run / "oom.json", {"allocated_bytes": torch.cuda.memory_allocated(),
                           "reserved_bytes": torch.cuda.memory_reserved(), "batch_size": 4, "resolution": 1008})
                break
            cause = cause.__cause__
        write_json(run / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise  # No automatic change of resolution, freezing, or batch size.
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
