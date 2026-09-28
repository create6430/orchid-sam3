"""Official SAM 3 optimization with project validation and resumable checkpoints."""
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
from sam3.train.trainer import Trainer

from .data import CocoSplit, write_json
from .evaluation import evaluate_model, write_metrics


class OrchidTrainer(Trainer):
    def __init__(self, valid_json, smoke=False, **kwargs):
        self.orchid_valid = CocoSplit(valid_json)
        self.orchid_smoke = smoke
        self.orchid_val_limit = self.orchid_valid.class_covering_prefix() if smoke else None
        # Upstream logs every environment variable, which may contain HF credentials.
        # Our entrypoint records explicit runtime metadata instead.
        with patch("sam3.train.trainer.log_env_variables", lambda: None):
            super().__init__(**kwargs)

    def run_val(self):
        # One GPU only: validation runs against the underlying DDP module.
        model = self.model.module if hasattr(self.model, "module") else self.model
        rows, _ = evaluate_model(model, self.orchid_valid, max_images=self.orchid_val_limit)
        score = next(r["value"] for r in rows if r["category"] == "macro" and r["metric"] == "IoU")
        if score is None:
            raise RuntimeError("Validation macro IoU is N/A; cannot select a best checkpoint")
        epoch = int(self.epoch) + 1
        output = Path(self.checkpoint_conf.save_dir).parent
        if self.orchid_smoke:
            write_json(output / "smoke_validation_images.json", {
                "image_ids": [r["id"] for r in self.orchid_valid.images[:self.orchid_val_limit]],
                "selection": "smallest image-ID prefix containing annotations for all three classes",
            })
        write_metrics(output / "validation" / f"epoch_{epoch:03d}.csv", rows)
        key = "orchid/valid_macro_iou"
        if score > self.best_meter_values.get(key, -1.0):
            self.best_meter_values[key] = score
            self.save_checkpoint(epoch, ["best"])
            write_json(output / "best.json", {"epoch": epoch, "valid_macro_iou": score, "smoke": self.orchid_smoke})
        # Update last AND epoch checkpoint after validation, preserving best state.
        self.save_checkpoint(epoch)
        self.logger.log_dict({key: score}, self.epoch)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint["orchid_run"] = {"smoke": self.orchid_smoke}
        np_state = np.random.get_state()
        checkpoint["orchid_rng"] = {
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        }
        super()._save_checkpoint(checkpoint, checkpoint_path)

    def _load_resuming_checkpoint(self, checkpoint_path):
        super()._load_resuming_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("orchid_rng")
        if state:
            random.setstate(state["python"])
            torch.set_rng_state(state["torch"])
            torch.cuda.set_rng_state_all(state["cuda"])
            ns = state["numpy"]
            np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
