"""PLAN steps 1-2: read-only COCO inventory and optional quality validation.

Run from the project root:
    python scripts/2026-09-28_validate_orchid_coco.py
    python scripts/2026-09-28_validate_orchid_coco.py --quality

Inventory uses the standard library; --quality needs Pillow, numpy, pycocotools.
Validated data-check environment (not a SAM 3 training environment):
    python -m pip install numpy==2.5.3 pycocotools==2.0.11 Pillow==12.3.0
Exit 0 means requested checks passed, not that leakage or semantics were validated.
Exit 1 indicates data errors; exit 2 indicates a CLI/output error.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import sys
import warnings


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {1: "flower", 2: "leaf", 3: "stem"}
FIELDS = ["split", "source_json", "source_sha256", "record_type", "item",
          "value", "status", "detail"]


def rle_counts(encoded, pixel_count):
    """Validate runs before passing data to the native COCO decoder.

    Compression follows cocoapi/common/maskApi.c:rleFrString (signed 5-bit
    values, with a two-run delta starting at the fourth run).
    """
    if isinstance(encoded, list):
        runs = encoded
    elif isinstance(encoded, str):
        runs, pos = [], 0
        while pos < len(encoded):
            value, shift = 0, 0
            while True:
                if pos >= len(encoded):
                    raise ValueError("Truncated compressed RLE")
                code = ord(encoded[pos]) - 48
                pos += 1
                if not 0 <= code <= 63:
                    raise ValueError("Invalid compressed RLE character")
                value |= (code & 31) << shift
                shift += 5
                if shift > 35:
                    raise ValueError("RLE run exceeds uint32 encoding")
                if not code & 32:
                    if code & 16:
                        value -= 1 << shift
                    break
            if len(runs) > 2:
                value += runs[-2]
            if not 0 <= value <= pixel_count:
                raise ValueError("RLE run is negative or exceeds image size")
            runs.append(value)
    else:
        raise ValueError("RLE counts must be a list or compressed string")
    if not runs or any(type(n) is not int or not 0 <= n <= pixel_count for n in runs):
        raise ValueError("RLE runs must be nonnegative integers within image size")
    if sum(runs) != pixel_count:
        raise ValueError(f"RLE covers {sum(runs)} pixels; expected {pixel_count}")
    return runs


def mask_area(segmentation, height, width, np, mask_utils):
    """Decode polygon/RLE with COCO and return format plus pixel area."""
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(n) is not int for n in size) or size != [height, width]):
            raise ValueError(f"RLE size must equal [{height}, {width}]")
        counts = segmentation.get("counts")
        runs = rle_counts(counts, height * width)
        if isinstance(counts, list):
            kind = "uncompressed_rle"
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            kind = "compressed_rle"
            rle = {"size": size, "counts": counts.encode("ascii")}
    elif isinstance(segmentation, list) and segmentation:
        kind = "polygon"
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("Polygon must contain at least three x/y pairs")
            if any(type(n) not in (int, float) or not math.isfinite(n) for n in polygon):
                raise ValueError("Polygon coordinates must be finite numbers")
            if any(not 0 <= x <= width for x in polygon[::2]) or any(
                    not 0 <= y <= height for y in polygon[1::2]):
                raise ValueError("Polygon coordinates outside image bounds")
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        runs = None
    else:
        raise ValueError("Missing/empty segmentation or unsupported format")
    decoded = mask_utils.decode(rle)
    if decoded.shape != (height, width):
        raise ValueError(f"Decoded mask shape mismatch: {decoded.shape}")
    if not np.all((decoded == 0) | (decoded == 1)):
        raise ValueError("Decoded mask is not binary")
    area = int(np.count_nonzero(decoded))
    if area == 0:
        raise ValueError("Empty mask after COCO decoding")
    if int(mask_utils.area(rle)) != area or (runs is not None and sum(runs[1::2]) != area):
        raise ValueError("COCO area / decoded pixel sum / RLE foreground disagree")
    return kind, area


def inventory(path, split, quality=None):
    """Inspect one split; identifiers are local to this JSON."""
    rows = []
    digest = ""

    def emit(kind, item, value="", status="INFO", detail=""):
        rows.append(dict(zip(FIELDS, [split, str(path), digest, kind, item,
                                      value, status, detail])))

    def error(item, detail):
        emit("error", item, status="FAIL", detail=detail)

    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        error("source", str(exc))
        return rows
    if not isinstance(data, dict):
        error("schema", "COCO root must be an object")
        return rows
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            error(key, "Required field must be a list")
    if rows:
        return rows

    categories = {}
    for index, category in enumerate(data["categories"]):
        if not isinstance(category, dict) or type(category.get("id")) is not int:
            error(f"category[{index}]", "Category must have an integer id")
            continue
        cid = category["id"]
        if cid in categories:
            error(f"category[{index}]", f"Duplicate category id: {cid}")
        categories[cid] = category.get("name")
        expected = {0: "sam3", **CLASSES}.get(cid)
        if expected is None or category.get("name") != expected:
            error(f"category[{index}]", f"Unexpected mapping: {cid}={category.get('name')!r}")
    for cid, name in CLASSES.items():
        if categories.get(cid) != name:
            error("categories", f"Required mapping missing: {cid}={name}")

    images = {}
    existing = set()
    decoded_images = set()
    dimensions = {}
    if quality:
        np, mask_utils, Image = quality
        for package in ("numpy", "pycocotools", "Pillow"):
            emit("environment", package, version(package))
        emit("environment", "python", sys.version.split()[0])
    for index, record in enumerate(data["images"]):
        if not isinstance(record, dict) or type(record.get("id")) is not int:
            error(f"image[{index}]", "Image must have an integer id")
            continue
        iid = record["id"]
        # Ambiguous references cannot satisfy step 1's image correspondence.
        if iid in images:
            error(f"image[{index}]", f"Duplicate image id within this JSON: {iid}")
        images[iid] = record
        name = record.get("file_name")
        if not isinstance(name, str) or not name.strip():
            error(f"image[{index}]", "Missing file_name")
            continue
        relative = Path(name)
        image_path = (path.parent / relative).resolve()
        if relative.is_absolute() or not image_path.is_relative_to(path.parent.resolve()):
            error(f"image[{index}]", f"file_name must stay within JSON directory: {name}")
        elif not image_path.is_file():
            error(f"image[{index}]", f"Image file missing: {name}")
        else:
            existing.add(iid)
            if quality:
                try:
                    height, width = record.get("height"), record.get("width")
                    if any(type(n) is not int or n <= 0 for n in (height, width)):
                        raise ValueError("JSON height/width must be positive integers (px)")
                    # verify checks file structure; reopening + load forces decoding.
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with Image.open(image_path) as image:
                            image.verify()
                        with Image.open(image_path) as image:
                            image.load()
                            if image.size != (width, height):
                                raise ValueError(f"Image size {image.size} != JSON {(width, height)}")
                            if image.format not in ("JPEG", "PNG"):
                                raise ValueError(f"Unsupported image format: {image.format}")
                    dimensions[iid] = (height, width)
                    decoded_images.add(iid)
                    emit("image_quality", str(iid), status="PASS",
                         detail=f"{name}; width={width} px; height={height} px")
                except (OSError, ValueError, SyntaxError, Warning, Image.DecompressionBombError) as exc:
                    error(f"image[{index}] id={iid}", str(exc))

    counts = Counter()
    valid_masks = Counter()
    formats = Counter()
    annotated_images = {cid: set() for cid in CLASSES}
    for index, annotation in enumerate(data["annotations"]):
        label = f"annotation[{index}]"
        if not isinstance(annotation, dict):
            error(label, "Annotation must be an object")
            continue
        label += f" id={annotation.get('id')!r}"
        cid, iid = annotation.get("category_id"), annotation.get("image_id")
        valid_category = type(cid) is int and cid in CLASSES
        if not valid_category:
            error(label, f"category_id must be 1, 2, or 3; got {cid!r}")
        else:
            counts[cid] += 1
        if type(iid) is not int or iid not in images:
            error(label, f"image_id does not resolve in this JSON: {iid!r}")
        elif iid not in existing:
            error(label, f"Referenced image file unavailable: {iid}")
        elif valid_category:
            annotated_images[cid].add(iid)
        if quality:
            if type(iid) is not int or iid not in dimensions:
                error(label, "Mask validation blocked: referenced image did not pass quality checks")
                continue
            try:
                height, width = dimensions[iid]
                kind, area = mask_area(annotation.get("segmentation"), height, width, np, mask_utils)
                stored_area = annotation.get("area")
                if (type(stored_area) not in (int, float) or not math.isfinite(stored_area)
                        or stored_area != area):
                    raise ValueError(f"Stored area {stored_area!r} != decoded area {area} px^2")
                formats[kind] += 1
                if valid_category:
                    valid_masks[cid] += 1
                emit("mask_quality", label, area, "PASS",
                     f"image_id={iid}; category_id={cid}; format={kind}; unit=px^2")
            except (ValueError, TypeError, OverflowError, KeyError) as exc:
                error(label, f"Mask validation: {exc}")

    emit("summary", "images", len(data["images"]))
    emit("summary", "existing_image_ids", len(existing))
    emit("summary", "annotations", len(data["annotations"]))
    if not data["images"]:
        error("images", "Split contains no images")
    for cid, name in CLASSES.items():
        emit("class_annotations", name, counts[cid], detail=f"category_id={cid}")
        emit("class_images", name, len(annotated_images[cid]))
        if not annotated_images[cid]:
            error(name, "No annotations referencing an existing image for this class")
        if quality:
            emit("class_valid_masks", name, valid_masks[cid])
            if not valid_masks[cid]:
                error(name, "No valid nonempty masks for this class")
    if quality:
        emit("summary", "decoded_image_ids", len(decoded_images))
        emit("summary", "valid_masks", sum(valid_masks.values()))
        for kind, count in sorted(formats.items()):
            emit("mask_format", kind, count)
    failed = any(row["status"] == "FAIL" for row in rows)
    emit("check", "quality" if quality else "inventory", status="FAIL" if failed else "PASS",
         detail=("Steps 1-2; sample sufficiency, semantics and leakage not evaluated" if quality
                 else "Step 1 only; mask validity, sample sufficiency and leakage not evaluated"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data/processed/annotations_area_fixed")
    parser.add_argument("--report", type=Path,
                        help="Output CSV within results/tables; defaults depend on --quality")
    parser.add_argument("--quality", action="store_true",
                        help="Also validate image decoding, dimensions, masks and corrected area")
    args = parser.parse_args()
    quality = None
    if args.quality:
        try:
            import numpy as np
            from pycocotools import mask as mask_utils
            from PIL import Image
            quality = (np, mask_utils, Image)
        except ImportError as exc:
            parser.error(f"Quality checks require numpy, pycocotools and Pillow: {exc}")
    report_name = "quality" if args.quality else "inventory"
    report = (args.report or ROOT / f"results/tables/2026-09-28_orchid_coco_{report_name}.csv").resolve()
    # Reports are derived artifacts; never allow a report to overwrite inputs.
    if not report.is_relative_to((ROOT / "results/tables").resolve()) or report.suffix != ".csv":
        parser.error("--report must be a .csv inside this project's results/tables/")
    data_root = args.data_root.resolve()
    if report.is_relative_to(data_root):
        parser.error("Report must not be inside --data-root")
    rows = []
    for split in ("train", "valid", "test"):
        rows.extend(inventory(data_root / split / "_annotations.coco.json", split, quality))
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        parser.error(f"Cannot write report: {exc}")
    for row in rows:
        if row["record_type"] in ("summary", "class_annotations", "error", "check"):
            print(f"[{row['split']}] {row['item']}: {row['value']} "
                  f"{row['status']} {row['detail']}")
    print(f"Report: {report}")
    return int(any(row["status"] == "FAIL" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
