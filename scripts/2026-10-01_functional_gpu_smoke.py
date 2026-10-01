"""Observe the unchanged training CLI in two separate processes; no tuning or model changes.

Run --phase initial, then --phase resume, then --phase audit with the same run name.
Default batch1 gives four real iterations per phase. Each CLI runs one full smoke epoch.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def smi():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return dict(output=p.stdout.strip(), error=p.stderr.strip(), returncode=p.returncode)
    except Exception as exc:
        return dict(error=str(exc))


def worker(args, run, folder):
    import torch
    import runpy
    from unittest.mock import patch
    from orchid_sam3.training import OrchidTrainer
    from sam3.train.trainer import Trainer
    if args.target_machine:
        from orchid_sam3.runtime import require_runtime
        require_runtime(args.sam3_repo, smoke=False)  # Verify the planned RTX 5090.
    start = time.monotonic()
    current = None
    probes = {}
    updates = []
    losses = []

    def event(stage, **extra):
        row = dict(stage=stage, timestamp=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
                   elapsed=time.monotonic()-start, pid=os.getpid(),
                   step=current.steps["train"] if current is not None else None,
                   allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                   max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved(), **extra)
        with (folder / "stages.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str)+"\n")
        print(json.dumps(row, default=str), flush=True)

    def complete(name, **extra):
        event(name+"_cuda_sync_start")
        torch.cuda.synchronize()
        event(name+"_complete", **extra)

    orig_setup = Trainer._setup_ddp_distributed_training
    orig_step = Trainer._step
    orig_backward = torch.Tensor.backward
    orig_scaler_step = torch.amp.GradScaler.step
    orig_restore = OrchidTrainer._load_resuming_checkpoint
    orig_save = OrchidTrainer._save_checkpoint
    orig_val = OrchidTrainer.run_val
    orig_subprocess = subprocess.run

    def setup(self, *a, **kw):
        nonlocal current
        current = self
        result = orig_setup(self, *a, **kw)
        self.model.register_forward_pre_hook(lambda *unused: event("forward_start"))
        def forward_done(*unused):
            complete("forward")
            event("loss_start")
        self.model.register_forward_hook(forward_done)
        event("ddp_setup_complete", ddp=isinstance(self.model, torch.nn.parallel.DistributedDataParallel))
        return result

    def step(self, batch, model, phase):
        nonlocal probes
        base = self.model.module
        probes = {n:p.detach().cpu().clone() for n,p in base.named_parameters() if p.requires_grad and p.numel()<=4096}
        event("batch_load_complete")
        event("transfer_and_step_start")
        result = orig_step(self, batch, model, phase)
        values = [float(v.detach()) for v in result[0].values()]
        assert all(math.isfinite(v) for v in values), values
        losses.extend(values)
        complete("loss", values=values)
        return result

    def backward(self, *a, **kw):
        event("backward_start")
        result = orig_backward(self, *a, **kw)
        complete("backward")
        return result

    def scaler_step(self, optimizer, *a, **kw):
        event("optimizer_step_start")
        result = orig_scaler_step(self, optimizer, *a, **kw)
        complete("optimizer_step")
        changed = sum(not torch.equal(probes[n], p.detach().cpu()) for n,p in current.model.module.named_parameters() if n in probes)
        assert changed > 0, "No sampled parameters updated"
        updates.append(changed)
        event("parameter_updates_verified", changed_small_tensors=changed)
        return result

    def restore(self, path):
        nonlocal current
        current = self
        event("checkpoint_restore_start", path=str(path))
        original_model = self.model.load_state_dict
        original_optimizer = self.optim.optimizer.load_state_dict
        def model_state(*a, **kw):
            event("model_restore_start")
            value = original_model(*a, **kw)
            complete("model_restore")
            return value
        def optimizer_state(*a, **kw):
            event("optimizer_restore_start")
            value = original_optimizer(*a, **kw)
            complete("optimizer_restore")
            return value
        with patch.object(self.model, "load_state_dict", model_state), patch.object(self.optim.optimizer, "load_state_dict", optimizer_state):
            result = orig_restore(self, path)
        states = self.optim.optimizer.state.values()
        steps = sorted(set(float(v["step"]) for v in states if "step" in v))
        complete("checkpoint_restore", epoch=self.epoch, steps=dict(self.steps), optimizer_steps=steps,
                 scaler=self.scaler.state_dict(), rng_restore="Unchanged OrchidTrainer restores Python/NumPy/torch/CUDA RNG")
        return result

    def save(self, checkpoint, path):
        event("checkpoint_save_start", path=str(path))
        value = orig_save(self, checkpoint, path)
        event("checkpoint_save_complete", path=str(path), bytes=Path(path).stat().st_size)
        return value

    def validation(self):
        event("validation_start")
        value = orig_val(self)
        complete("validation")
        return value

    def subprocess_run(command, *a, **kw):
        # Keep the real quality validator, but retain its report per phase rather
        # than overwriting historical fixed-name evidence. No numerical changes.
        if isinstance(command, list) and any(str(x).endswith("2026-09-28_validate_orchid_coco.py") for x in command):
            command = list(command)+["--report", str(ROOT / f"results/tables/2026-10-01_{args.run_name}_{args.phase}_quality.csv")]
        return orig_subprocess(command, *a, **kw)

    event("cli_start", phase=args.phase, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    sys.argv = [str(ROOT / "scripts/2026-09-28_train_sam3_orchid.py"), "--sam3-repo", str(args.sam3_repo),
                "--checkpoint", str(args.checkpoint), "--run-name", args.run_name, "--smoke-test",
                "--smoke-batch-size", str(args.batch_size), "--workers", "0"]
    if args.phase == "resume":
        sys.argv += ["--epochs", "2", "--resume"]
    with patch.object(Trainer, "_setup_ddp_distributed_training", setup), patch.object(Trainer, "_step", step), \
         patch.object(torch.Tensor, "backward", backward), patch.object(torch.amp.GradScaler, "step", scaler_step), \
         patch.object(OrchidTrainer, "_load_resuming_checkpoint", restore), patch.object(OrchidTrainer, "_save_checkpoint", save), \
         patch.object(OrchidTrainer, "run_val", validation), patch.object(subprocess, "run", subprocess_run):
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    expected = (4 + args.batch_size - 1)//args.batch_size
    assert len(updates)==len(losses)==expected, (updates, losses)
    dump(folder / "worker_result.json", dict(status="PASS", pid=os.getpid(), phase=args.phase, iterations=len(updates),
                                              losses=losses, changed_small_tensors=updates, steps=dict(current.steps)))
    dump(folder / "completion_snapshot.json", json.loads((run / "completion.json").read_text()))
    event("process_complete")


def audit(args, run):
    statuses={}
    for phase in ("initial", "resume"):
        path=run / f"logs/functional_{phase}/result.json"
        statuses[phase]=json.loads(path.read_text()) if path.exists() else dict(status="NOT_RUN")
    if any(v["status"]!="PASS" for v in statuses.values()):
        report=dict(status="FAIL", interpretation="Functional acceptance incomplete; not proof of a training logic defect",
                    name="Target-machine Functional GPU Smoke Test" if args.target_machine else "Development-machine Functional GPU Smoke Test",
                    run=str(run), phases=statuses, formal_training_started=False,
                    reason="Both phases must exit successfully before checkpoint/content acceptance can PASS")
        save_audit(args, run, report)
        print(json.dumps(report,indent=2))
        return 1
    import torch
    initial=statuses["initial"];resumed=statuses["resume"]
    a = json.loads((run / "logs/functional_initial/worker_result.json").read_text())
    b = json.loads((run / "logs/functional_resume/worker_result.json").read_text())
    assert a["pid"] != b["pid"] and a["iterations"]==b["iterations"]==(4+args.batch_size-1)//args.batch_size
    ckpt = torch.load(run / "checkpoints/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    assert ckpt["steps"]["train"] == b["steps"]["train"] == 2*a["steps"]["train"]
    opt_steps = sorted(set(float(v["step"]) for v in ckpt["optimizer"]["state"].values() if "step" in v))
    assert opt_steps == [float(b["steps"]["train"])] and "orchid_rng" in ckpt and "scaler" in ckpt
    events=[json.loads(l) for l in (run / "logs/functional_resume/stages.jsonl").read_text().splitlines()]
    restored=next(x for x in events if x["stage"]=="checkpoint_restore_complete")
    assert restored["steps"]["train"]==a["steps"]["train"] and restored["optimizer_steps"]==[float(a["steps"]["train"])]
    required=["checkpoints/checkpoint.pt","checkpoints/checkpoint_1.pt","checkpoints/checkpoint_2.pt","checkpoints/best.pt",
              "validation/epoch_001.csv","validation/epoch_002.csv","logs/train_stats.json","logs/best_stats.json",
              "environment.json","requirements.actual.txt","input_signature.json","config_target_001.yaml","config_target_002.yaml","completion.json"]
    for name in required: assert (run/name).is_file(), name
    for name in ["validation/epoch_001.csv","validation/epoch_002.csv"]:
        with (run/name).open(encoding="utf-8-sig") as f:
            rows=list(csv.DictReader(f))
        assert rows, name
        for row in rows:
            if row.get("value") not in (None,"","N/A"): assert math.isfinite(float(row["value"]))
    assert list(run.rglob("events.out.tfevents.*")), "TensorBoard output missing"
    completion=json.loads((run/"completion.json").read_text());assert completion["resume_exercised"] and completion["epochs"]==2
    report=dict(status="PASS",name="Target-machine Functional GPU Smoke Test" if args.target_machine else "Development-machine Functional GPU Smoke Test",run=str(run),initial=a,resume=b,
                checkpoint=str(run/"checkpoints/checkpoint.pt"),checkpoint_reload="PASS",optimizer_steps=opt_steps,
                scaler=ckpt["scaler"],rng_present=True,restored=restored,formal_training_started=False,
                limitation="RTX3080/WSL residency and timing variability are separate from this functional result.")
    save_audit(args,run,report)
    print(json.dumps(report,indent=2))
    return 0


def save_audit(args, run, report):
    dump(ROOT/f"results/tables/2026-10-01_{args.run_name}_functional.json",report)
    with (ROOT/f"results/tables/2026-10-01_{args.run_name}_files.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["full_path","bytes","relative_to_plan_root"])
        for path in sorted(run.rglob("*")):
            if path.is_file():w.writerow([str(path),path.stat().st_size,str(path.relative_to(ROOT/"results"))])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["initial","resume","audit"],required=True)
    p.add_argument("--run-name",required=True)
    p.add_argument("--sam3-repo",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--batch-size",type=int,choices=range(1,5),default=1)
    p.add_argument("--target-machine",action="store_true",help="Require the PLAN target RTX 5090; still smoke only")
    p.add_argument("--timeout",type=int,default=900,help="Per-stage bound; accommodates observed slow fresh iterations")
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",args.run_name) or not 60<=args.timeout<=1200:p.error("Invalid run name or timeout (60..1200)")
    run=ROOT/"results/models"/args.run_name
    folder=run/"logs"/f"functional_{args.phase}"
    if args.phase=="audit":return audit(args,run)
    if args.worker:worker(args,run,folder);return 0
    if args.phase=="initial" and run.exists():raise FileExistsError("Choose a new smoke run; existing results are protected")
    if args.phase=="resume":
        assert json.loads((run/"logs/functional_initial/result.json").read_text())["status"]=="PASS"
    folder.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-u",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
    dump(folder/"command.json",command)
    started=time.monotonic();last=dict(stage="worker_start",monotonic=started);timed_out=False;next_sample=0
    with (folder/"console.log").open("w",encoding="utf-8") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            path=folder/"stages.jsonl";events=[]
            if path.exists():
                for line in path.read_text().splitlines():
                    try:events.append(json.loads(line))
                    except json.JSONDecodeError:pass
            if events:last=events[-1]
            anchor=last
            if last["stage"].endswith("_cuda_sync_start"):
                origin=last["stage"].removesuffix("_cuda_sync_start")+"_start"
                anchor=next((v for v in reversed(events) if v["stage"]==origin),last)
            now=time.monotonic()
            if now>=next_sample:
                with (folder/"gpu_monitor.jsonl").open("a") as f:f.write(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),stage=last["stage"],gpu=smi()))+"\n")
                print(json.dumps(dict(phase=args.phase,stage=last["stage"],stage_seconds=round(now-anchor["monotonic"],1))),flush=True);next_sample=now+30
            if now-anchor["monotonic"]>args.timeout:
                timed_out=True;dump(folder/"timeout.json",dict(last_event=last,timeout_seconds=args.timeout,root_cause="Not inferred"))
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                break
            time.sleep(2)
    status="PASS" if child.returncode==0 and not timed_out and (folder/"worker_result.json").exists() else "TIMEOUT" if timed_out else "FAIL"
    dump(folder/"result.json",dict(status=status,exit_code=child.returncode,pid=child.pid,last_stage=last["stage"]))
    if (folder/"stages.jsonl").exists():
        with (ROOT/f"results/tables/2026-10-01_{args.run_name}_{args.phase}_stages.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["stage","timestamp","elapsed","step","allocated","reserved","max_allocated","max_reserved"],extrasaction="ignore");w.writeheader()
            w.writerows(json.loads(l) for l in (folder/"stages.jsonl").read_text().splitlines())
    print((folder/"result.json").read_text(),flush=True)
    return 0 if status=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
