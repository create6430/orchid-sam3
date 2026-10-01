"""Audit existing real GPU smoke artifacts; never starts training or inference.

Run with .venv-train/bin/python scripts/2026-10-01_audit_gpu_smoke.py in WSL.
The established run must have completed one epoch, then resumed to epoch two.
"""
import csv
import json
import math
from pathlib import Path
import re

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/models/smoke"
TABLES = ROOT / "results/tables"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    rows = []

    def check(item, passed, evidence):
        rows.append(dict(item=item, status="PASS" if passed else "FAIL", evidence=str(evidence)))

    completion = read(RUN / "completion.json")
    env = read(RUN / "environment.json")
    cfg = OmegaConf.load(RUN / "config_target_002.yaml")
    signature = read(RUN / "input_signature.json")
    base = Path(cfg.trainer.model.checkpoint_path)
    checkpoints = [RUN / f"checkpoints/checkpoint_{epoch}.pt" for epoch in (1, 2)]
    saved = [torch.load(p, map_location="cpu", weights_only=True, mmap=True) for p in checkpoints if p.is_file()]
    check("resume_checkpoint_exists", len(saved) == 2, checkpoints[1])
    original = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
    original = original.get("model", original)
    original = {k.removeprefix("detector."): v for k, v in original.items() if k.startswith("detector.")}
    comparisons = []
    pairs = [("pretrained_to_epoch1", original, saved[0]["model"])]
    if len(saved) == 2:
        pairs.append(("epoch1_to_epoch2", saved[0]["model"], saved[1]["model"]))
    else:
        check("epoch1_to_epoch2", False, "No saved resumed update available for comparison")
    for label, before, after in pairs:
        changed = []
        examined = 0
        finite = True
        for key, value in after.items():
            if key in before and value.is_floating_point() and value.numel() <= 4096:
                examined += 1
                finite &= bool(torch.isfinite(value).all())
                delta = (value.float() - before[key].float()).abs().max().item()
                if delta > 0:
                    changed.append(dict(name=key, max_abs_delta=delta))
        comparisons.append(dict(comparison=label, examined_small_tensors=examined,
                                changed_small_tensors=len(changed), finite=finite,
                                examples=changed[::max(1, len(changed) // 8)][:10]))
        check(label, len(changed) > 0 and finite, comparisons[-1])
    steps = [sorted({float(s["step"]) for s in c["optimizer"]["state"].values() if "step" in s}) for c in saved]
    check("optimizer_steps", steps == [[4.0], [8.0]], steps)
    check("checkpoint_epochs", [c["epoch"] for c in saved] == [1, 2], [c["epoch"] for c in saved])
    check("training_iterations", [c["steps"]["train"] for c in saved] == [4, 8], [c["steps"] for c in saved])
    check("checkpoint_rng_and_scaler", all("orchid_rng" in c and "scaler" in c for c in saved), "RNG and scaler state saved")
    check("completion", completion == dict(epochs=2, smoke=True, training_completed=True, resume_exercised=True), completion)
    check("gpu", env["gpu"] == "NVIDIA GeForce RTX 3080", env["gpu"])
    check("cuda", torch.cuda.is_available(), torch.version.cuda)
    check("smoke_settings", signature["smoke"] and signature["batch_size"] == 1 and signature["resolution"] == 1008, signature)
    stats = [json.loads(line) for line in (RUN / "logs/train_stats.json").read_text().splitlines() if line.strip()]
    check("finite_initial_epoch_loss", bool(stats) and math.isfinite(stats[0]["Losses/train_all_loss"]),
          stats[0]["Losses/train_all_loss"] if stats else "No initial epoch stats")
    check("finite_epoch_losses", len(stats) == 2 and all(math.isfinite(s["Losses/train_all_loss"]) for s in stats),
          [s["Losses/train_all_loss"] for s in stats])
    resume_path = Path(cfg.trainer.checkpoint.resume_from).resolve()
    check("resume_path", resume_path == (RUN / "checkpoints/checkpoint.pt").resolve(), resume_path)
    latest = torch.load(resume_path, map_location="cpu", weights_only=True, mmap=True)
    check("latest_checkpoint", latest["epoch"] == 2 and latest["steps"]["train"] == 8
          and latest["orchid_run"]["smoke"], dict(epoch=latest["epoch"], steps=latest["steps"]))
    resume_log = (RUN / "logs/smoke_resume_console.log").read_text()
    check("resume_runtime_log", str(resume_path) in resume_log and "Train Epoch: [1]" in resume_log,
          "Restore path and resumed epoch must be in actual console log")
    for epoch, log in ((0, (RUN / "logs/smoke_retry_5_console.log").read_text()), (1, resume_log)):
        iterations = re.findall(r"Train Epoch: \[" + str(epoch) + r"\]\[(\d+)/4\].*?Losses/train_all_loss: ([^ ]+)", log)
        check(f"forward_backward_step_epoch_{epoch}", [int(i) for i, _ in iterations] == [0, 1, 2, 3]
              and all(math.isfinite(float(loss)) for _, loss in iterations), iterations)
    initial_log = (RUN / "logs/smoke_retry_5_console.log").read_text()
    check("pretrained_strict_load", f"SAM3 strict checkpoint load succeeded: {base}; 1134 entries" in initial_log, base)
    for split in ("train", "valid"):
        dataset = ROOT / f"data/processed/annotations_area_fixed/{split}/_annotations.coco.json"
        data = read(dataset)
        names = {c["id"]: c["name"] for c in data["categories"] if c["id"] in (1, 2, 3)}
        ids = {image["id"] for image in data["images"]}
        check(f"{split}_mapping", names == {1: "flower", 2: "leaf", 3: "stem"}, names)
        check(f"{split}_pairing", all((dataset.parent / image["file_name"]).is_file() for image in data["images"])
              and all(a["image_id"] in ids for a in data["annotations"]), dataset)
    for epoch in (1, 2):
        path = RUN / f"validation/epoch_{epoch:03d}.csv"
        if not path.is_file():
            check(f"validation_epoch_{epoch}", False, f"Missing: {path}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        check(f"validation_epoch_{epoch}", len(metrics) == 16 and all(
            math.isfinite(float(row["value"])) and 0 <= float(row["value"]) <= 1 for row in metrics), path)
    required = ["checkpoints/best.pt", "checkpoints/checkpoint.pt", "checkpoints/checkpoint_1.pt",
                "checkpoints/checkpoint_2.pt", "logs/log.txt", "logs/train_stats.json",
                "validation/epoch_001.csv", "validation/epoch_002.csv", "config_target_001.yaml",
                "config_target_002.yaml", "config_resolved.yaml", "environment.json", "requirements.actual.txt",
                "input_signature.json", "inputs/train.coco.json", "smoke_validation_images.json",
                "best.json", "completion.json"]
    for relative in required:
        path = RUN / relative
        check(f"artifact:{relative}", path.is_file() and path.stat().st_size > 0 and path.resolve().is_relative_to(ROOT / "results"), path)
    check("tensorboard", any((RUN / "tensorboard").glob("events.out.tfevents.*")), RUN / "tensorboard")
    check("output_root", Path(cfg.paths.experiment_log_dir).resolve() == RUN.resolve(), RUN)
    check("checkpoint_directory", Path(cfg.trainer.checkpoint.save_dir).resolve() == (RUN / "checkpoints").resolve(), cfg.trainer.checkpoint.save_dir)
    check("validation_subset", read(RUN / "smoke_validation_images.json")["image_ids"] == [0, 1], "2 real validation images covering flower/leaf/stem")
    rows.append(dict(item="formal_test_visualization", status="N/A", evidence="PLAN step 6 formal test only; smoke checkpoint is excluded from formal evaluation"))
    TABLES.mkdir(parents=True, exist_ok=True)
    report = TABLES / "2026-10-01_sam3_gpu_smoke_audit.csv"
    with report.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / "2026-10-01_sam3_gpu_smoke_updates.json").write_text(
        json.dumps(dict(comparisons=comparisons, optimizer_steps=steps), indent=2), encoding="utf-8")
    inventory = TABLES / "2026-10-01_sam3_gpu_smoke_files.csv"
    with inventory.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wsl_path", "windows_path", "bytes"])
        for path in sorted(RUN.rglob("*")):
            if path.is_file():
                writer.writerow([str(path), "D:\\my-project\\orchid-sam3\\" + str(path.relative_to(ROOT)).replace("/", "\\"), path.stat().st_size])
    failed = [row["item"] for row in rows if row["status"] == "FAIL"]
    print(json.dumps(dict(report=str(report), failed=failed, optimizer_steps=steps,
                          changed_small_tensors=[c["changed_small_tensors"] for c in comparisons]), indent=2))
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
