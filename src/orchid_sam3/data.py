"""Read-only, fail-closed COCO access shared by training and evaluation."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[2]
CLASSES = {1: "flower", 2: "leaf", 3: "stem"}
THRESHOLDS = {"flower": 0.70, "leaf": 0.80, "stem": 0.75}
SAM3_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def artifact_path(path, folder):
    path = Path(path).resolve()
    base = (ROOT / folder).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError(f"Output must be inside {base}: {path}")
    return path


class CocoSplit:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        cats = self.data["categories"]
        if len({c["id"] for c in cats}) != len(cats):
            raise ValueError("Duplicate category IDs")
        mapping = {c["id"]: c["name"] for c in cats}
        if any(mapping.get(cid) != name for cid, name in CLASSES.items()) or any(
                cid not in CLASSES and (cid != 0 or name != "sam3") for cid, name in mapping.items()):
            raise ValueError("COCO categories must match SPEC")
        self.images = sorted(self.data["images"], key=lambda r: r["id"])
        ids = [r["id"] for r in self.images]
        if not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Image IDs must be nonempty, unique integers within each split")
        self.by_id = {r["id"]: r for r in self.images}
        self.annotations = defaultdict(list)
        ann_ids = set()
        for ann in self.data["annotations"]:
            if ann["id"] in ann_ids or type(ann["category_id"]) is not int or ann["category_id"] not in CLASSES:
                raise ValueError("Duplicate annotation ID or non-SPEC category")
            ann_ids.add(ann["id"])
            if ann["image_id"] not in self.by_id:
                raise ValueError("Annotation references an unknown image")
            self.annotations[ann["image_id"]].append(ann)
        for record in self.images:
            self.image_path(record)

    def image_path(self, record):
        path = (self.path.parent / record["file_name"]).resolve()
        if not path.is_relative_to(self.path.parent) or not path.is_file():
            raise ValueError(f"Missing or unsafe image path: {record['file_name']}")
        return path

    def image(self, record):
        with Image.open(self.image_path(record)) as image:
            image.load()
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"Image/COCO size mismatch: {record['id']}")
            return image.convert("RGB")

    def mask(self, ann):
        record = self.by_id[ann["image_id"]]
        h, w = record["height"], record["width"]
        segmentation = ann["segmentation"]
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
        elif isinstance(segmentation, dict):
            if segmentation.get("size") != [h, w]:
                raise ValueError("RLE dimensions do not match COCO")
            rle = (mask_utils.frPyObjects(segmentation, h, w)
                   if isinstance(segmentation["counts"], list) else segmentation)
        else:
            raise ValueError("Unsupported segmentation format")
        decoded = mask_utils.decode(rle)
        if decoded.shape != (h, w) or not np.all((decoded == 0) | (decoded == 1)):
            raise ValueError("Mask shape or binary values invalid")
        area = int(np.count_nonzero(decoded))
        if area == 0 or area != ann["area"]:
            raise ValueError(f"Empty mask or incorrect area: annotation {ann['id']}")
        return decoded.astype(bool)

    def unions(self, record):
        masks = {name: np.zeros((record["height"], record["width"]), dtype=bool) for name in CLASSES.values()}
        for ann in self.annotations[record["id"]]:
            masks[CLASSES[ann["category_id"]]] |= self.mask(ann)
        return masks

    def fingerprint(self):
        return {"annotation_sha256": file_hash(self.path), "images": [
            {"image_id": r["id"], "file_name": r["file_name"], "sha256": file_hash(self.image_path(r))}
            for r in self.images]}

    def class_covering_prefix(self):
        """Smallest deterministic image-ID prefix with all three annotated classes.

        Used only for smoke validation, never for full validation or test selection.
        """
        present = set()
        for count, record in enumerate(self.images, 1):
            present.update(ann["category_id"] for ann in self.annotations[record["id"]])
            if present == set(CLASSES):
                return count
        raise ValueError("Smoke validation requires annotations for all three classes")
