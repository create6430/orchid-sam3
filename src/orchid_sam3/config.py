"""Adapt a pinned official configuration without changing upstream files."""
from pathlib import Path
import subprocess

from .data import SAM3_COMMIT


def check_source(repo):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != SAM3_COMMIT:
        raise ValueError(f"SAM 3 commit must be {SAM3_COMMIT}; got {commit}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("SAM 3 tracked source has local changes")
    return repo


def make_config(repo, run, checkpoint, train_json, train_images, valid_json, epochs=10, workers=4, smoke=False):
    from omegaconf import OmegaConf
    # Resolvers used by the pinned YAML, registered without importing torch.
    if not OmegaConf.has_resolver("times"):
        OmegaConf.register_new_resolver("times", lambda x, y: x * y)
    repo, run = Path(repo), Path(run)
    cfg = OmegaConf.load(repo / "sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml")
    cfg.pop("defaults", None)
    cfg.pop("all_roboflow_supercategories", None)
    cfg.paths.roboflow_vl_100_root = str(train_images.parent)
    cfg.paths.experiment_log_dir = str(run)
    cfg.paths.bpe_path = str(repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    cfg.roboflow_train.supercategory = "orchid"
    cfg.roboflow_train.num_images = 4 if smoke else None
    cfg.scratch.enable_segmentation = True
    cfg.scratch.train_batch_size = 4
    cfg.scratch.gradient_accumulation_steps = 1
    cfg.scratch.num_train_workers = workers
    cfg.scratch.max_data_epochs = epochs
    if smoke:
        # A one-batch smoke test must actually update weights (warmup step 0 otherwise has lr=0).
        cfg.scratch.scheduler_warmup = 0
        cfg.scratch.scheduler_cooldown = 0
    # Deterministic square resize, no hidden random crop/augmentation or query filtering.
    transforms = [
        {"_target_": "sam3.train.transforms.segmentation.DecodeRle"},
        {"_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI", "sizes": 1008,
         "max_size": 1008, "square": True, "consistent_transform": False},
        {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
        {"_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI", "mean": [0.5] * 3, "std": [0.5] * 3},
    ]
    cfg.roboflow_train.train_transforms = [{"_target_": "sam3.train.transforms.basic_for_api.ComposeAPI", "transforms": transforms}]
    cfg.roboflow_train.loss.loss_fns_find.append({
        "_target_": "sam3.train.loss.loss_fns.Masks", "focal_alpha": 0.25, "focal_gamma": 2.0,
        "weight_dict": {"loss_mask": 200.0, "loss_dice": 10.0}, "compute_aux": False,
    })
    cfg.trainer._target_ = "orchid_sam3.training.OrchidTrainer"
    cfg.trainer.valid_json = str(valid_json)
    cfg.trainer.smoke = smoke
    cfg.trainer.skip_saving_ckpts = False
    cfg.trainer.skip_first_val = False
    cfg.trainer.max_epochs = epochs
    cfg.trainer.val_epoch_freq = 1
    cfg.trainer.meters = None
    cfg.trainer.data.val = None  # Project validation uses SPEC pixel unions, not upstream bbox AP.
    cfg.trainer.data.train.dataset.img_folder = str(train_images)
    cfg.trainer.data.train.dataset.ann_file = str(train_json)
    cfg.trainer.data.train.dataset.coco_json_loader = {
        "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON", "_partial_": True,
        "include_negatives": True, "category_chunk_size": 3,
    }
    cfg.trainer.data.train.drop_last = False
    cfg.trainer.model = {
        "_target_": "orchid_sam3.runtime.build_model", "checkpoint_path": str(checkpoint) if checkpoint else None,
        "bpe_path": cfg.paths.bpe_path, "device": "cpu", "eval_mode": False,
    }
    cfg.trainer.checkpoint.save_freq = 1
    cfg.trainer.logging.log_dir = str(run / "logs")
    cfg.trainer.logging.log_freq = 1 if smoke else 10
    cfg.launcher.gpus_per_node = 1
    cfg.submitit.use_cluster = False
    cfg.submitit.pop("job_array", None)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
