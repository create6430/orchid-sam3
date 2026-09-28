"""Dataset-pooled semantic metrics and independently reviewed support masks."""
import csv
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES, ROOT, THRESHOLDS


class Metrics:
    def __init__(self):
        self.counts = {name: [0, 0, 0] for name in CLASSES.values()}

    def update(self, predictions, truth):
        for name in self.counts:
            p, g = predictions[name], truth[name]
            if p.shape != g.shape or p.dtype != bool or g.dtype != bool:
                raise ValueError(f"{name}: masks must be bool arrays in original image coordinates")
            delta = [np.count_nonzero(p & g), np.count_nonzero(p & ~g), np.count_nonzero(~p & g)]
            self.counts[name] = [a + int(b) for a, b in zip(self.counts[name], delta)]

    def rows(self):
        rows = []
        for name, (tp, fp, fn) in self.counts.items():
            for metric, numerator, denominator in (
                ("IoU", tp, tp + fp + fn), ("Dice", 2 * tp, 2 * tp + fp + fn),
                ("Precision", tp, tp + fp), ("Recall", tp, tp + fn),
            ):
                value = numerator / denominator if denominator else None
                threshold = THRESHOLDS[name] if metric == "IoU" else ""
                rows.append(dict(category=name, metric=metric, value=value, numerator=numerator,
                                 denominator=denominator, tp=tp, fp=fp, fn=fn, unit="ratio",
                                 threshold=threshold, status=("N/A" if value is None else
                                 "PASS" if threshold != "" and value >= threshold else
                                 "FAIL" if threshold != "" else "INFO"),
                                 reason="zero denominator" if value is None else ""))
        for metric in ("IoU", "Dice", "Precision", "Recall"):
            values = [r["value"] for r in rows if r["metric"] == metric]
            valid = all(v is not None for v in values)
            rows.append(dict(category="macro", metric=metric, value=sum(values) / 3 if valid else None,
                             numerator="", denominator=3, tp="", fp="", fn="", unit="ratio", threshold="",
                             status="INFO" if valid else "N/A", reason="" if valid else "one or more classes are N/A"))
        return rows


class SupportMasks:
    def __init__(self, manifest, split, dataset):
        self.masks, self.errors = {}, []
        self.numerator = self.denominator = 0
        manifest = Path(manifest)
        if not manifest.is_file():
            self.errors.append(f"Missing support manifest: {manifest}")
            return
        try:
            with manifest.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"split", "image_id", "image_path", "mask_path", "review_status", "reviewer", "reviewed_at"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Support manifest requires {sorted(required)}")
                seen, mask_paths = set(), set()
                for row in reader:
                    try:
                        if None in row or any(not isinstance(v, str) for v in row.values()):
                            raise ValueError("Malformed support manifest row")
                        iid = int(row["image_id"])
                        if row["split"] != split or iid not in dataset.by_id or iid in seen:
                            raise ValueError("Unknown/duplicate image ID or incorrect split")
                        seen.add(iid)
                        if row["review_status"] != "approved" or row["reviewer"].strip().casefold() in ("", "unknown", "pending", "n/a", "待確認"):
                            raise ValueError("Support mask has not been reviewed")
                        if date.fromisoformat(row["reviewed_at"]).isoformat() != row["reviewed_at"]:
                            raise ValueError("Review date must be YYYY-MM-DD")
                        record = dataset.by_id[iid]
                        paths = []
                        for key in ("image_path", "mask_path"):
                            raw = Path(row[key])
                            path = (ROOT / raw).resolve()
                            if raw.is_absolute() or not path.is_relative_to(ROOT):
                                raise ValueError("Manifest paths must be relative to project root")
                            paths.append(path)
                        image_path, mask_path = paths
                        if image_path != dataset.image_path(record) or mask_path in mask_paths:
                            raise ValueError("Ambiguous image/mask mapping")
                        mask_paths.add(mask_path)
                        with Image.open(mask_path) as mask:
                            if mask.format != "PNG" or mask.mode != "L" or mask.size != (record["width"], record["height"]):
                                raise ValueError("Support mask must be original-size 8-bit single-channel PNG")
                            pixels = np.asarray(mask)
                            if not np.all((pixels == 0) | (pixels == 255)):
                                raise ValueError("Support pixels must be 0/255")
                            self.masks[iid] = pixels == 255
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        self.errors.append(f"image_id={row.get('image_id')}: {exc}")
                missing = set(dataset.by_id) - set(self.masks)
                if missing:
                    self.errors.append(f"Missing valid masks for image IDs: {sorted(missing)}")
        except (OSError, ValueError, TypeError) as exc:
            self.errors.append(str(exc))

    def update(self, image_id, stem):
        if self.errors:
            return
        support = self.masks[image_id]
        if support.shape != stem.shape:
            raise ValueError("Support/prediction shape mismatch")
        self.numerator += int(np.count_nonzero(support & stem))
        self.denominator += int(np.count_nonzero(support))

    def row(self):
        valid = not self.errors and self.denominator > 0
        value = 100 * self.numerator / self.denominator if valid else None
        return dict(category="support", metric="stem_false_positive_percent", value=value,
                    numerator="" if self.errors else self.numerator,
                    denominator="" if self.errors else self.denominator, tp="", fp="", fn="",
                    unit="percent", threshold="<5", status="N/A" if not valid else "PASS" if value < 5 else "FAIL",
                    reason="; ".join(self.errors) if self.errors else "zero support pixels" if not valid else "")


def write_metrics(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "N/A" if v is None else v for k, v in row.items()} for row in rows)


def predict_unions(processor, image):
    import torch
    result = {}
    # Upstream filters with >; retain everything here, apply SPEC's >=0.35 below.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for name in CLASSES.values():
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(name, state)
            masks, scores = output["masks"], output["scores"]
            if masks.ndim != 4 or masks.shape[1:] != (1, image.height, image.width):
                raise ValueError(f"SAM 3 masks not restored to source coordinates: {masks.shape}")
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite prediction scores")
            selected = masks[scores >= 0.35, 0]
            result[name] = selected.any(dim=0).cpu().numpy().astype(bool)
    return result


def evaluate_model(model, dataset, support=None, max_images=None, preview_count=0):
    from sam3.model.sam3_image_processor import Sam3Processor
    model.eval()
    processor = Sam3Processor(model, resolution=1008, device="cuda", confidence_threshold=-1.0)
    metrics, previews = Metrics(), []
    records = dataset.images if max_images is None else dataset.images[:max_images]
    for record in records:
        image = dataset.image(record)
        truth = dataset.unions(record)
        predictions = predict_unions(processor, image)
        metrics.update(predictions, truth)
        if support is not None:
            support.update(record["id"], predictions["stem"])
        if len(previews) < preview_count:
            previews.append((record["id"], image, truth, predictions,
                             support.masks.get(record["id"]) if support and not support.errors else None))
    return metrics.rows(), previews


def save_preview(path, previews):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"flower": (1, 0.3, 0.5), "leaf": (0.2, 0.9, 0.3), "stem": (0.2, 0.5, 1)}
    fig, axes = plt.subplots(len(previews), 4, figsize=(16, 4 * len(previews)), squeeze=False)
    for row, (iid, image, truth, predictions, support) in zip(axes, previews):
        for ax in row:
            ax.imshow(image)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        row[0].set_title(f"Image {iid}")
        for ax, masks, title in ((row[1], truth, "Ground truth"), (row[2], predictions, "Prediction >= 0.35")):
            for name, color in colors.items():
                overlay = np.zeros((image.height, image.width, 4))
                overlay[masks[name]] = (*color, 0.5)
                ax.imshow(overlay)
            ax.set_title(title)
        row[3].set_title("Support N/A" if support is None else "Support (yellow); stem overlap (red)")
        if support is not None:
            overlay = np.zeros((image.height, image.width, 4))
            overlay[support] = (1, 1, 0, 0.5)
            overlay[support & predictions["stem"]] = (1, 0, 0, 0.8)
            row[3].imshow(overlay)
    fig.suptitle("flower=pink; leaf=green; stem=blue")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
