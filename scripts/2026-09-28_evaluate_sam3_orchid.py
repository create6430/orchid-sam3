"""Evaluate a project checkpoint on the complete test split using SPEC metrics."""
import argparse
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, default=ROOT / "data/processed/support_eval/test/manifest.csv")
    parser.add_argument("--preview-count", type=int, default=6)
    args = parser.parse_args()
    if args.preview_count < 1:
        parser.error("preview-count must be positive")
    from orchid_sam3.data import CocoSplit, artifact_path, file_hash, write_json
    from orchid_sam3.evaluation import SupportMasks, evaluate_model, save_preview, write_metrics
    from orchid_sam3.runtime import build_model, require_runtime
    prefix = "2026-09-28_sam3_orchid"
    csv_path = artifact_path(ROOT / f"results/tables/{prefix}_metrics.csv", "results/tables")
    figure_path = artifact_path(ROOT / f"results/figures/{prefix}_examples.png", "results/figures")
    meta_path = artifact_path(ROOT / f"results/tables/{prefix}_evaluation.json", "results/tables")
    if args.support_manifest.resolve() in (csv_path, figure_path, meta_path):
        parser.error("Support input cannot be an output artifact")
    environment = require_runtime(args.sam3_repo)
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not {"model", "optimizer", "epoch", "orchid_run"}.issubset(checkpoint):
        parser.error("Evaluation expects a project training checkpoint, not official base weights")
    if checkpoint["orchid_run"]["smoke"]:
        parser.error("Smoke-test checkpoints cannot be used for final test evaluation")
    del checkpoint
    subprocess.run([sys.executable, str(ROOT / "scripts/2026-09-28_validate_orchid_coco.py"), "--quality"], check=True)
    dataset = CocoSplit(ROOT / "data/processed/annotations_area_fixed/test/_annotations.coco.json")
    support = SupportMasks(args.support_manifest, "test", dataset)
    model = build_model(args.checkpoint, args.sam3_repo.resolve() / "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                        device="cuda", eval_mode=True)
    rows, previews = evaluate_model(model, dataset, support=support, preview_count=args.preview_count)
    rows.append(support.row())
    required = [r for r in rows if r["metric"] == "IoU" and r["category"] != "macro"] + [rows[-1]]
    numerical_pass = all(r["status"] == "PASS" for r in required)
    write_metrics(csv_path, rows)
    save_preview(figure_path, previews)
    write_json(meta_path, {
        "checkpoint_sha256": file_hash(args.checkpoint), "test_fingerprint": dataset.fingerprint(),
        "environment": environment, "numerical_thresholds_pass": numerical_pass,
        "formal_acceptance": False, "reason": "Capture-sequence provenance unavailable; manual preview review also required",
        "support_errors": support.errors,
        "support_manifest_sha256": file_hash(args.support_manifest) if args.support_manifest.is_file() else None,
    })
    print(f"Numerical thresholds: {'PASS' if numerical_pass else 'FAIL / N/A'}; formal acceptance remains unverified")
    return 0 if numerical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
