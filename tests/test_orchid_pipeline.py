"""CPU checks; real COCO data only, never model quality claims or invented labels."""
import ast
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orchid_sam3.config import check_source, make_config
from orchid_sam3.data import CocoSplit, CLASSES, artifact_path
from orchid_sam3.evaluation import Metrics, SupportMasks


@pytest.fixture
def dataset():
    path = ROOT / "data/processed/annotations_area_fixed/test/_annotations.coco.json"
    if not path.is_file():
        pytest.skip("Real project data must be supplied separately")
    return CocoSplit(path)


def test_real_mask_identity_and_area(dataset):
    metrics = Metrics()
    for record in dataset.images:
        truth = dataset.unions(record)
        metrics.update(truth, truth)
    assert all(row["value"] == 1.0 for row in metrics.rows())
    assert sum(len(v) for v in dataset.annotations.values()) == 655


def test_zero_denominators_and_macro():
    rows = Metrics().rows()
    assert all(row["value"] is None and row["status"] == "N/A" for row in rows)
    metrics = Metrics()
    # Arithmetic unit test of count formulas; these are not a dataset/model result.
    metrics.counts["flower"] = [2, 1, 3]
    rows = metrics.rows()
    flower = {r["metric"]: r["value"] for r in rows if r["category"] == "flower"}
    assert flower == {"IoU": 2/6, "Dice": 4/8, "Precision": 2/3, "Recall": 2/5}
    assert all(r["value"] is None for r in rows if r["category"] == "macro")


def test_missing_support_not_zero_false_positive(dataset):
    missing = ROOT / "results/_nonexistent_support_manifest_for_unit_check.csv"
    assert not missing.exists()
    support = SupportMasks(missing, "test", dataset)
    row = support.row()
    assert row["value"] is None and row["status"] == "N/A" and row["numerator"] == ""


def test_original_area_errors_rejected():
    path = ROOT / "data/processed/annotations/train/_annotations.coco.json"
    if not path.is_file():
        pytest.skip("Original export not available")
    dataset = CocoSplit(path)
    with pytest.raises(ValueError, match="incorrect area"):
        dataset.mask(dataset.data["annotations"][0])


def test_output_path_guard():
    with pytest.raises(ValueError):
        artifact_path(ROOT / "data/raw/output.csv", "results/tables")


def test_smoke_validation_covers_all_classes():
    path = ROOT / "data/processed/annotations_area_fixed/valid/_annotations.coco.json"
    if not path.is_file():
        pytest.skip("Real validation data must be supplied separately")
    valid = CocoSplit(path)
    count = valid.class_covering_prefix()
    assert count > 1  # Actual first image has flower/leaf but no stem.
    covered = {ann["category_id"] for r in valid.images[:count] for ann in valid.annotations[r["id"]]}
    previous = {ann["category_id"] for r in valid.images[:count-1] for ann in valid.annotations[r["id"]]}
    assert covered == set(CLASSES)
    assert previous != set(CLASSES)
    assert valid.class_covering_prefix() == count


def test_config_segmentation_and_no_test_selection():
    tmp_path = ROOT / "results/models/_config_unit_check"
    repo = Path(os.environ.get("SAM3_REPO", ROOT.parent / "sam3-reference"))
    if not repo.is_dir():
        pytest.skip("Set SAM3_REPO to the pinned official checkout")
    check_source(repo)
    source = ROOT / "data/processed/annotations_area_fixed"
    cfg = make_config(repo, tmp_path, None, tmp_path / "train.coco.json", source / "train",
                      source / "valid/_annotations.coco.json")
    assert cfg.trainer.data.train.batch_size == 4
    assert cfg.trainer.data.train.drop_last is False
    assert cfg.trainer.data.train.dataset.limit_ids is None
    assert cfg.trainer.data.train.dataset.coco_json_loader.category_chunk_size == 3
    assert cfg.trainer.model._target_ == "orchid_sam3.runtime.build_model"
    assert cfg.trainer.optim.amp.amp_dtype == "bfloat16"
    assert cfg.trainer.checkpoint.save_freq == 1
    assert cfg.trainer.skip_saving_ckpts is False
    assert cfg.trainer.skip_first_val is False
    assert cfg.trainer.max_epochs == 10
    assert cfg.trainer.data.val is None
    assert "/valid/" in cfg.trainer.valid_json.replace("\\", "/")
    assert any(loss._target_.endswith(".Masks") for loss in cfg.trainer.loss.all.loss_fns_find)
    # Verify every referenced upstream callable exists without importing GPU packages.
    from omegaconf import OmegaConf
    def inspect(node):
        if isinstance(node, dict):
            target = node.get("_target_", "")
            if target.startswith("sam3."):
                module, name = target.rsplit(".", 1)
                tree = ast.parse((repo / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
                assert any(isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name for n in tree.body), target
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)
    inspect(OmegaConf.to_container(cfg, resolve=True))


def test_prepare_does_not_change_labels():
    path = ROOT / "results/models/config_check/inputs/train.coco.json"
    original = ROOT / "data/processed/annotations_area_fixed/train/_annotations.coco.json"
    if not path.exists() or not original.exists():
        pytest.skip("Run --prepare-only --run-name config_check first")
    derived = json.loads(path.read_text(encoding="utf-8"))
    source = json.loads(original.read_text(encoding="utf-8"))
    assert derived["images"] == source["images"]
    assert derived["annotations"] == source["annotations"]
    assert {c["id"]: c["name"] for c in derived["categories"]} == CLASSES
