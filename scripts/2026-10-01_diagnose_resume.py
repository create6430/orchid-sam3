"""Bounded, real one-batch SAM3 resume diagnosis; no validation or formal training.

Run in .venv-train, e.g. --case a1 --mode full [--no-ddp].
Only writes new case logs/tables and a uniquely named diagnostic checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/models/smoke"
SOURCE = RUN / "checkpoints/checkpoint.pt"
sys.path.insert(0, str(ROOT / "src"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def smi():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        return {"output": p.stdout.strip(), "error": p.stderr.strip(), "returncode": p.returncode}
    except Exception as exc:
        return {"error": str(exc)}


def worker(args, folder):
    import torch
    import random
    import numpy as np
    from omegaconf import OmegaConf
    from orchid_sam3.runtime import require_runtime
    from orchid_sam3.training import OrchidTrainer
    from sam3.train.trainer import copy_data_to_device
    from sam3.train.utils.checkpoint_utils import load_state_dict_into_model
    repo = ROOT.parent / "sam3-reference"
    start = time.monotonic()
    events = folder / "stages.jsonl"

    def event(stage, **extra):
        row = dict(stage=stage, timestamp=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
                   elapsed=time.monotonic() - start, pid=os.getpid(),
                   allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                   max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved(), **extra)
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()
        print(json.dumps(row, default=str), flush=True)
        return row

    def complete(stage, **extra):
        event(stage + "_cuda_sync_start")
        torch.cuda.synchronize()
        event(stage + "_complete", nvidia_smi=smi(), **extra)

    def state_report(trainer, name):
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        named = dict(model.named_parameters())
        names = {id(p): n for n, p in named.items()}
        opt = trainer.optim.optimizer
        seen, tensors, groups, wrong = [], [], [], []
        for group in opt.param_groups:
            groups.append({k: v for k, v in group.items() if k != "params"})
            for param in group["params"]:
                seen.append(id(param))
                if id(param) not in names:
                    wrong.append("optimizer references non-model parameter")
                for key, value in opt.state.get(param, {}).items():
                    if torch.is_tensor(value):
                        expected = str(param.device) if key != "step" or group.get("capturable") or group.get("fused") else "cpu"
                        tensors.append(dict(parameter=names.get(id(param)), key=key, shape=list(value.shape),
                                            device=str(value.device), expected_device=expected, dtype=str(value.dtype),
                                            bytes=value.numel()*value.element_size(),
                                            value=float(value) if value.numel() == 1 else None))
                        if str(value.device) != expected:
                            wrong.append(f"{names.get(id(param))}:{key}:{value.device}!={expected}")
                        if key != "step" and value.shape != param.shape:
                            wrong.append(f"{names.get(id(param))}:{key}:state shape differs from parameter")
        info = dict(mode=args.mode, ddp=isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel),
                    distributed_initialized=torch.distributed.is_initialized(), optimizer_type=type(opt).__name__,
                    optimizer_defaults=opt.defaults, optimizer_groups=groups, optimizer_state_tensors=tensors,
                    optimizer_state_bytes=sum(t["bytes"] for t in tensors),
                    optimizer_state_cuda_bytes=sum(t["bytes"] for t in tensors if t["device"].startswith("cuda")),
                    parameter_references_valid=not wrong, device_or_reference_errors=wrong,
                    optimizer_duplicate_references=len(seen)-len(set(seen)),
                    model_parameter_count=sum(p.numel() for p in named.values()),
                    trainable_parameter_count=sum(p.numel() for p in named.values() if p.requires_grad),
                    parameters=[dict(name=n, shape=list(p.shape), device=str(p.device), dtype=str(p.dtype),
                                     requires_grad=p.requires_grad, optimizer_member=id(p) in set(seen)) for n,p in named.items()],
                    epoch=trainer.epoch, steps=trainer.steps, amp=vars(trainer.optim_conf.amp),
                    scaler_enabled=trainer.scaler.is_enabled(), scaler_state=trainer.scaler.state_dict(),
                    scheduler_state="Reconstructed stateless schedulers; upstream computes where/step from epoch and loader length",
                    schedulers=[[type(v).__name__ for v in group.values()] for group in trainer.optim.schedulers])
        dump(folder / f"state_{name}.json", info)
        event("state_audit_" + name + "_complete", optimizer_state_bytes=info["optimizer_state_bytes"],
              optimizer_state_cuda_bytes=info["optimizer_state_cuda_bytes"], device_or_reference_errors=wrong)
        return info

    class DiagnosticTrainer(OrchidTrainer):
        def load_checkpoint(self):
            # Bypass upstream's copy-to-checkpoint.pt convention: input stays read-only.
            event("state_audit_before_restore_start")
            state_report(self, "before_restore")
            if args.original_restore:
                from unittest.mock import patch
                original_load = torch.load
                model_load = self.model.load_state_dict
                optimizer_load = self.optim.optimizer.load_state_dict
                load_count = 0

                def tracked_load(*load_args, **load_kwargs):
                    nonlocal load_count
                    load_count += 1
                    label = "checkpoint_load" if load_count == 1 else "rng_checkpoint_reload"
                    event(label + "_start", path=str(SOURCE))
                    result = original_load(*load_args, **load_kwargs)
                    event(label + "_complete", epoch=result.get("epoch"), steps=result.get("steps"))
                    return result

                def tracked_model(*load_args, **load_kwargs):
                    event("model_state_restore_start")
                    result = model_load(*load_args, **load_kwargs)
                    complete("model_state_restore", restored=True)
                    return result

                def tracked_optimizer(*load_args, **load_kwargs):
                    event("optimizer_state_restore_start")
                    result = optimizer_load(*load_args, **load_kwargs) if args.mode == "full" else None
                    complete("optimizer_state_restore", restored=args.mode == "full")
                    return result

                # Full mode observes actual implementations. The explicit weights-only
                # B control skips optimizer restoration; it is never full-resume success.
                with patch.object(torch, "load", tracked_load), patch.object(self.model, "load_state_dict", tracked_model), \
                     patch.object(self.optim.optimizer, "load_state_dict", tracked_optimizer):
                    super()._load_resuming_checkpoint(str(SOURCE))
                complete("checkpoint_restore")
                state_report(self, "after_restore")
                return
            event("checkpoint_load_start", path=str(SOURCE))
            # Same CPU deserialization and restore order as the pinned Trainer.
            checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
            event("checkpoint_load_complete", epoch=checkpoint["epoch"], steps=checkpoint["steps"])
            event("model_state_restore_start")
            if args.mode != "fresh":
                load_state_dict_into_model(model=self.model, state_dict=checkpoint["model"],
                                           ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters)
            complete("model_state_restore", restored=args.mode != "fresh")
            event("optimizer_state_restore_start")
            if args.mode == "full":
                self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
            complete("optimizer_state_restore", restored=args.mode == "full")
            self.loss.load_state_dict(checkpoint["loss"], strict=True)
            self.epoch = checkpoint["epoch"]
            self.steps = dict(checkpoint["steps"])
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
            self.best_meter_values = checkpoint.get("best_meter_values", {})
            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
            if "train_dataset" in checkpoint:
                self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])
            state = checkpoint["orchid_rng"]
            random.setstate(state["python"])
            torch.set_rng_state(state["torch"])
            torch.cuda.set_rng_state_all(state["cuda"])
            ns = state["numpy"]
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
            del checkpoint
            complete("checkpoint_restore")
            state_report(self, "after_restore")

        def _setup_ddp_distributed_training(self, distributed, accelerator):
            event("ddp_setup_start", enabled=not args.no_ddp)
            if not args.no_ddp:
                super()._setup_ddp_distributed_training(distributed, accelerator)
            complete("ddp_setup", enabled=not args.no_ddp)

        def _save_checkpoint(self, checkpoint, checkpoint_path):
            checkpoint["orchid_diagnostic"] = dict(case=args.case, mode=args.mode, ddp=not args.no_ddp,
                                                    original_restore=args.original_restore, completed_batches=1, partial_epoch=True)
            super()._save_checkpoint(checkpoint, checkpoint_path)

    event("diagnostic_settings", case=args.case, mode=args.mode, original_restore=args.original_restore,
          ddp=not args.no_ddp, synchronized_stages=True, validation=False, timeout_seconds=args.timeout)
    event("environment_check_start")
    dump(folder / "environment.json", require_runtime(repo, smoke=True))
    cfg = OmegaConf.load(RUN / "config_target_002.yaml")
    cfg.paths.experiment_log_dir = str(folder)
    cfg.trainer.logging.log_dir = str(folder)
    cfg.trainer.logging.tensorboard_writer.log_dir = str(folder / "tensorboard")
    cfg.trainer.checkpoint.resume_from = None
    # Existing input, source, loader, seed, optimizer, loss, AMP, resolution stay unchanged.
    kwargs = {k: v for k, v in cfg.trainer.items() if k != "_target_"}
    OmegaConf.save(cfg, folder / "config.yaml")
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    event("trainer_initialize_start")
    trainer = DiagnosticTrainer(**kwargs)
    state_report(trainer, "after_ddp")
    trainer.model.train()
    trainer.optim.zero_grad(set_to_none=True)
    event("batch_load_start")
    loader = trainer.train_dataset.get_loader(epoch=int(trainer.epoch))
    batch = next(iter(loader))
    key, data = next(iter(batch.items()))
    event("batch_load_complete", loader_batches=len(loader), image_shape=list(data.img_batch.shape),
          image_sha256=hashlib.sha256(data.img_batch.numpy().tobytes()).hexdigest(),
          metadata=str(data.find_metadatas))
    event("transfer_to_gpu_start")
    data = copy_data_to_device(data, trainer.device, non_blocking=True)
    complete("transfer_to_gpu")
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    probe = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.numel() <= 4096}
    with torch.amp.autocast("cuda", enabled=trainer.optim_conf.amp.enabled, dtype=torch.bfloat16):
        event("forward_start")
        outputs = trainer.model(data)
        complete("forward")
        event("loss_calculation_start")
        targets = [model.back_convert(x) for x in data.find_targets]
        loss = trainer._find_loss(key)(outputs, targets)
        if isinstance(loss, dict):
            loss = trainer._log_loss_detailed_and_return_core_loss(loss, "diagnostic/loss", trainer.steps["train"])
        value = float(loss.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {value}")
        complete("loss_calculation", loss=value)
    event("backward_start")
    trainer.scaler.scale(loss).backward()
    complete("backward")
    trainer.steps["train"] += 1
    trainer.where = trainer.epoch / trainer.max_epochs
    event("scheduler_and_gradient_clip_start")
    trainer.optim.step_schedulers(trainer.where, step=int(trainer.epoch * len(loader)))
    if trainer.gradient_clipper is not None:
        trainer.scaler.unscale_(trainer.optim.optimizer)
        trainer.gradient_clipper(model=trainer.model)
    complete("scheduler_and_gradient_clip")
    event("optimizer_step_start")
    trainer.scaler.step(trainer.optim.optimizer)
    trainer.scaler.update()
    complete("optimizer_step")
    changes = [n for n,p in model.named_parameters() if n in probe and not torch.equal(probe[n],p.detach().cpu())]
    if not changes:
        raise RuntimeError("No sampled trainable parameters updated")
    state_report(trainer, "after_step")
    event("checkpoint_save_start")
    trainer.save_checkpoint(trainer.epoch, [f"diagnostic_{args.case}"])
    path = RUN / f"checkpoints/diagnostic_{args.case}.pt"
    event("checkpoint_save_complete", path=str(path), bytes=path.stat().st_size)
    dump(folder / "result.json", dict(status="PASS", full_resume=args.mode == "full", ddp=not args.no_ddp,
                                       original_restore=args.original_restore,
                                       loss=value, steps=trainer.steps, changed_small_tensors=len(changes),
                                       checkpoint=str(path), mode=args.mode))
    torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=["full", "weights", "fresh"], default="full")
    p.add_argument("--no-ddp", action="store_true")
    p.add_argument("--original-restore", action="store_true", help="Call the unchanged OrchidTrainer restore, including its second load for RNG")
    p.add_argument("--timeout", type=int, default=480)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.original_restore and args.mode == "fresh":
        p.error("--original-restore supports full or weights mode")
    if not re.fullmatch(r"[a-z0-9_-]+", args.case) or not 60 <= args.timeout <= 600:
        p.error("case must be lowercase identifier; timeout must be 60..600 seconds")
    folder = RUN / "logs" / f"resume_diag_{args.case}"
    if args.worker:
        try:
            worker(args, folder)
        except BaseException as exc:
            dump(folder / "result.json", dict(status="ERROR", error_type=type(exc).__name__, error=str(exc)))
            raise
        return 0
    folder.mkdir(exist_ok=False)
    if (RUN / f"checkpoints/diagnostic_{args.case}.pt").exists():
        raise FileExistsError("Diagnostic checkpoint already exists")
    baseline = dict(bytes=SOURCE.stat().st_size, mtime_ns=SOURCE.stat().st_mtime_ns)
    dump(folder / "source_checkpoint.json", dict(path=str(SOURCE), **baseline))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--case", args.case,
               "--mode", args.mode, "--timeout", str(args.timeout)] + (["--no-ddp"] if args.no_ddp else []) + (["--original-restore"] if args.original_restore else [])
    dump(folder / "command.json", command)
    started = time.monotonic()
    timed_out = False
    with (folder / "console.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last = dict(stage="worker_start", monotonic=started)
        next_sample = 0
        while child.poll() is None:
            events = folder / "stages.jsonl"
            records = []
            if events.exists():
                for line in events.read_text().splitlines():
                    try:
                        last = json.loads(line)
                        records.append(last)
                    except json.JSONDecodeError: pass
            anchor = last
            if last["stage"].endswith("_cuda_sync_start"):
                origin = last["stage"].removesuffix("_cuda_sync_start") + "_start"
                anchor = next((r for r in reversed(records) if r["stage"] == origin), last)
            now = time.monotonic()
            if now >= next_sample:
                with (folder / "gpu_monitor.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                                                 stage=last["stage"], gpu=smi())) + "\n")
                print(json.dumps(dict(case=args.case, stage=last["stage"], stage_seconds=round(now-last["monotonic"],1))), flush=True)
                next_sample = now + 30
            if now - anchor["monotonic"] > args.timeout:
                timed_out = True
                dump(folder / "result.json", dict(status="TIMEOUT", stage=last["stage"], last_event=last,
                                                   timeout_seconds=args.timeout, pid=child.pid, mode=args.mode,
                                                   reason="Stage did not return within diagnostic bound; root cause not inferred"))
                # Capture Python stacks with a signal-free external reader if available.
                try:
                    with (folder / "timeout_stack.txt").open("w") as stack:
                        subprocess.run(["/home/benny/.local/bin/uvx", "--offline", "py-spy", "dump", "--pid", str(child.pid)],
                                       stdout=stack, stderr=stack, timeout=10)
                except Exception: pass
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                break
            time.sleep(2)
    unchanged = baseline == dict(bytes=SOURCE.stat().st_size, mtime_ns=SOURCE.stat().st_mtime_ns)
    dump(folder / "source_preserved.json", dict(unchanged=unchanged, **baseline))
    assert unchanged, "Source checkpoint unexpectedly changed"
    import csv
    stages = [json.loads(line) for line in (folder / "stages.jsonl").read_text().splitlines()] if (folder / "stages.jsonl").exists() else []
    table = ROOT / f"results/tables/2026-10-01_resume_diag_{args.case}.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "timestamp", "elapsed", "allocated", "reserved", "max_allocated", "max_reserved"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(stages)
    print((folder / "result.json").read_text() if (folder / "result.json").exists() else f"Worker exit {child.returncode}", flush=True)
    return int(timed_out or child.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
