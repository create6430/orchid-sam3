"""PLAN step 3: preserve splits, find leakage, and prepare human review.

Run after validate_orchid_coco.py --quality, from the project root:
    .venv/Scripts/python.exe scripts/2026-09-28_check_orchid_splits.py

Dependencies: numpy==2.5.3, Pillow==12.3.0 (Python 3.10+).
Optional --groups CSV: split,image_id,file_name,plant_id,sequence_id.
IDs must come from capture records, be globally consistent, and cover all images.
Unknown values (including pending/unknown/N/A/待確認) block acceptance.
Optional --reviews CSV: copy the generated pairs CSV to a separate input file;
fill decision (different_source/same_source/unsure), reviewer, reviewed_at
(YYYY-MM-DD), and note. Do not infer plant IDs from exported filenames.

Outputs are deterministic: one manifest, a check report, candidate pairs, and an
HTML review page. Neither inputs nor manually completed reviews are overwritten.
Exit 0: step 3 checks passed; 1: failed or pending; 2: invalid arguments/I/O.
Perceptual hashes are only a heuristic shortlist, not proof of independence.
"""

import argparse
from collections import defaultdict
import csv
from datetime import date
import hashlib
import html
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
import warnings

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 211, "valid": 60, "test": 30}
PREFIX = "2026-09-28_orchid"
UNKNOWN = {"", "unknown", "pending", "n/a", "na", "none", "null", "待確認", "未知"}
MANIFEST_FIELDS = ["split", "image_id", "file_name", "image_path", "source_json",
                   "source_sha256", "file_sha256", "pixel_sha256", "width_px", "height_px",
                   "dhash64", "phash63", "plant_id", "sequence_id"]
PAIR_FIELDS = ["pair_id", "kind", "split_a", "image_id_a", "image_path_a", "file_sha256_a",
               "split_b", "image_id_b", "image_path_b", "file_sha256_b",
               "dhash_distance", "phash_distance", "decision", "reviewer", "reviewed_at", "note"]
REPORT_FIELDS = ["check", "status", "value", "detail"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(values):
    result = 0
    for value in values.ravel():
        result = (result << 1) | int(value)
    return result


def perceptual_hashes(image):
    """64 horizontal differences; 63 low-frequency DCT bits (DC excluded).

    Full exported image is used, including padding, with LANCZOS resizing.
    Thresholds shortlist candidates only; no crop/rotation invariance is claimed.
    """
    gray = image.convert("L")
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    dhash = bits(small[:, 1:] > small[:, :-1])
    grid = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32) + 0.5))
    basis *= np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    coefficients = (basis @ grid @ basis.T).ravel()[1:]
    return dhash, bits(coefficients > np.median(coefficients))


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(required)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_images(data_root, report):
    records = []
    for split, expected in EXPECTED.items():
        source = data_root / split / "_annotations.coco.json"
        try:
            raw = source.read_bytes()
            data = json.loads(raw)
            if not isinstance(data, dict) or not isinstance(data.get("images"), list):
                raise ValueError("Missing COCO images list")
            count = len(data["images"])
            report("split_count", "PASS" if count == expected else "FAIL", count,
                   f"{split}: expected {expected}")
            seen = set()
            for index, entry in enumerate(data["images"]):
                try:
                    if not isinstance(entry, dict) or type(entry.get("id")) is not int:
                        raise ValueError("Image id must be an integer")
                    iid = entry["id"]
                    if iid in seen:
                        raise ValueError(f"Duplicate image id within {split}: {iid}")
                    seen.add(iid)
                    name = entry.get("file_name")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("Missing file_name")
                    path = (source.parent / name).resolve()
                    if Path(name).is_absolute() or not path.is_relative_to(source.parent):
                        raise ValueError(f"Unsafe image path: {name}")
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with Image.open(path) as original:
                            original.verify()
                        with Image.open(path) as original:
                            original.load()
                            if original.size != (entry.get("width"), entry.get("height")):
                                raise ValueError("Image dimensions differ from JSON")
                            image = original.convert("RGB")
                    dhash, phash = perceptual_hashes(image)
                    pixel_hash = sha(f"RGB:{image.width}:{image.height}:".encode() + image.tobytes())
                    records.append(dict(zip(MANIFEST_FIELDS, [
                        split, iid, name, path.relative_to(ROOT).as_posix(),
                        source.relative_to(ROOT).as_posix(), sha(raw), sha(path.read_bytes()),
                        pixel_hash, image.width, image.height, f"{dhash:016x}", f"{phash:016x}", "", ""])))
                except (OSError, ValueError, SyntaxError, Warning, Image.DecompressionBombError) as exc:
                    report("image", "FAIL", index, f"{split}: {exc}")
        except (OSError, ValueError) as exc:
            report("source", "FAIL", split, str(exc))
    order = {split: i for i, split in enumerate(EXPECTED)}
    return sorted(records, key=lambda r: (order[r["split"]], r["image_id"]))


def check_groups(path, records, report):
    if path is None:
        report("group_provenance", "PENDING", len(records),
               "No plant/sequence mapping supplied; cannot confirm split independence")
        return
    rows = read_csv(path, ["split", "image_id", "file_name", "plant_id", "sequence_id"])
    report("groups_source", "INFO", sha(path.read_bytes()), str(path))
    expected = {(r["split"], str(r["image_id"])): r for r in records}
    seen = set()
    for row in rows:
        key = (row["split"].strip(), row["image_id"].strip())
        if key not in expected or key in seen:
            report("group_mapping", "FAIL", str(key), "Unknown or duplicate image key")
            continue
        seen.add(key)
        record = expected[key]
        if row["file_name"] != record["file_name"]:
            report("group_mapping", "FAIL", str(key), "file_name does not match COCO")
            continue
        for field in ("plant_id", "sequence_id"):
            value = row[field].strip()
            if value.casefold() in UNKNOWN:
                report("group_mapping", "PENDING", str(key), f"Unknown {field}")
            else:
                record[field] = value
    for key in sorted(set(expected) - seen):
        report("group_mapping", "PENDING", str(key), "Missing image in mapping")
    for field in ("plant_id", "sequence_id"):
        groups = defaultdict(set)
        for record in records:
            if record[field]:
                groups[record[field]].add(record["split"])
        overlap = {key: sorted(splits) for key, splits in groups.items() if len(splits) > 1}
        for group, splits in sorted(overlap.items()):
            report("group_overlap", "FAIL", group, f"{field}: {','.join(splits)}")
        complete = all(record[field] for record in records)
        report(field, "FAIL" if overlap else "PASS" if complete else "PENDING", len(groups),
               "Global group IDs checked across splits; source authenticity requires provider records")


def compare_pairs(records, dhash_limit, phash_limit):
    pairs, compared = [], 0
    for a, b in combinations(records, 2):
        if a["split"] == b["split"]:
            continue
        compared += 1
        dd = (int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count()
        pd = (int(a["phash63"], 16) ^ int(b["phash63"], 16)).bit_count()
        exact = a["file_sha256"] == b["file_sha256"] or a["pixel_sha256"] == b["pixel_sha256"]
        if not exact and dd > dhash_limit and pd > phash_limit:
            continue
        # Reviews are tied to split identity and actual bytes, not just filenames.
        identity = [[r["split"], r["image_id"], r["image_path"], r["file_sha256"]] for r in (a, b)]
        pair = {"pair_id": sha(json.dumps(identity, ensure_ascii=False).encode()),
                "kind": "exact" if exact else "near", "dhash_distance": dd, "phash_distance": pd,
                "decision": "", "reviewer": "", "reviewed_at": "", "note": ""}
        for suffix, record in (("a", a), ("b", b)):
            for field in ("split", "image_id", "image_path", "file_sha256"):
                pair[f"{field}_{suffix}"] = record[field]
        pairs.append(pair)
    return sorted(pairs, key=lambda r: (r["kind"], r["phash_distance"], r["dhash_distance"], r["pair_id"])), compared


def review_pairs(path, pairs, report):
    reviews = {}
    if path:
        rows = read_csv(path, ["pair_id", "decision", "reviewer", "reviewed_at", "note"])
        report("reviews_source", "INFO", sha(path.read_bytes()), str(path))
        current = {pair["pair_id"] for pair in pairs}
        for row in rows:
            key = row["pair_id"]
            if key in reviews or key not in current:
                report("review_input", "FAIL", key, "Duplicate, unknown or stale pair_id")
                continue
            reviews[key] = row
    pending = 0
    for pair in pairs:
        review = reviews.get(pair["pair_id"])
        if review:
            for field in ("decision", "reviewer", "reviewed_at", "note"):
                pair[field] = review[field].strip()
        if pair["kind"] == "exact":
            report("exact_duplicate", "FAIL", pair["pair_id"], "Cross-split duplicate cannot be waived by review")
            continue
        if pair["decision"] == "same_source":
            report("near_review", "FAIL", pair["pair_id"], "Reviewer identified same source across splits")
            continue
        valid = (pair["decision"] == "different_source" and
                 all(pair[field].casefold() not in UNKNOWN for field in ("reviewer", "note")))
        try:
            parsed = date.fromisoformat(pair["reviewed_at"])
            valid = valid and parsed.isoformat() == pair["reviewed_at"]
        except ValueError:
            valid = False
        if not valid:
            pending += 1
    report("near_review", "PENDING" if pending else "PASS", pending,
           "Candidates awaiting valid human review; absence of candidates is not proof of independence")


def write_review_page(path, pairs):
    sections = []
    for i, pair in enumerate(pairs, 1):
        panels = []
        for side in ("a", "b"):
            target = ROOT / pair[f"image_path_{side}"]
            url = quote(Path(os.path.relpath(target, path.parent)).as_posix(), safe="/.")
            label = f"{pair[f'split_{side}']} / image_id={pair[f'image_id_{side}']}"
            panels.append(f'<figure><img loading="lazy" src="{url}" alt="{html.escape(label)}">'
                          f'<figcaption>{html.escape(label)}<br>{html.escape(pair[f"image_path_{side}"])}</figcaption></figure>')
        sections.append(f'<section><h2>{i}. {pair["kind"]} — dHash {pair["dhash_distance"]}/64, '
                        f'pHash {pair["phash_distance"]}/63</h2><code>{pair["pair_id"]}</code>'
                        f'<div class="pair">{"".join(panels)}</div>'
                        f'<p>Review: {html.escape(pair["decision"] or "PENDING")}</p></section>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
                    '<title>蘭花跨集合近似影像覆核</title><style>'
                    'body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:16px}'
                    '.pair{display:flex;flex-wrap:wrap}figure{margin:12px;flex:1;min-width:240px}'
                    'img{width:100%;max-width:512px}figcaption,code{overflow-wrap:anywhere}'
                    'section{border-top:1px solid #aaa;padding:16px 0}</style>'
                    '<h1>跨集合影像覆核</h1><p>這是候選清單，不代表已確認資料洩漏。'
                    '請搭配來源紀錄確認是否為同一植株／序列；勿只依外觀推定不同來源。'
                    '將 pairs CSV 複製至獨立位置，填入 decision、reviewer、reviewed_at、note，'
                    '再用 --reviews 傳入。原始影像使用相對連結，需在本專案內開啟。</p>'
                    + (''.join(sections) or '<p>目前門檻未篩出候選；仍須來源對照表。</p>')
                    + '</html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", type=Path, help="Provider's plant/sequence mapping CSV")
    parser.add_argument("--reviews", type=Path, help="Human-completed candidate review CSV")
    parser.add_argument("--dhash-distance", type=int, default=8, choices=range(65), metavar="0..64")
    parser.add_argument("--phash-distance", type=int, default=10, choices=range(64), metavar="0..63")
    args = parser.parse_args()
    manifest = ROOT / f"data/processed/splits/{PREFIX}_split_manifest.csv"
    table = ROOT / f"results/tables/{PREFIX}_split_checks.csv"
    pair_csv = ROOT / f"results/tables/{PREFIX}_split_pairs.csv"
    page = ROOT / f"results/figures/{PREFIX}_split_review.html"
    outputs = [manifest, table, pair_csv, page]
    allowed_roots = [ROOT / "data/processed/splits", ROOT / "results/tables", ROOT / "results/figures"]
    for output in outputs:
        if not any(output.resolve().is_relative_to(base) for base in allowed_roots):
            parser.error(f"Output resolves outside permitted artifact directories: {output}")
    inputs = [p.resolve() for p in (args.groups, args.reviews) if p]
    if set(inputs) & {p.resolve() for p in outputs}:
        parser.error("Copy group/review inputs to a separate path; inputs cannot be output files")
    # Never overwrite human review edits in a previously generated CSV.
    if pair_csv.exists():
        try:
            old = read_csv(pair_csv, PAIR_FIELDS)
            if any(row[field].strip() for row in old for field in ("decision", "reviewer", "reviewed_at", "note")):
                parser.error("Pairs output contains review data. Move it to a separate input and use --reviews")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    checks = []

    def report(check, status, value, detail):
        checks.append(dict(zip(REPORT_FIELDS, [check, status, value, detail])))

    try:
        report("algorithm", "INFO", "dhash64/phash63", f"OR thresholds: {args.dhash_distance}/{args.phash_distance}; RGB pixel SHA256 includes dimensions; no randomization")
        report("environment", "INFO", sys.version.split()[0], f"numpy={version('numpy')}; Pillow={version('Pillow')}")
        records = load_images(ROOT / "data/processed/annotations_area_fixed", report)
        if any(r["status"] == "FAIL" for r in checks):
            report("overall", "FAIL", "", "Input checks failed; manifests/pairs not regenerated; any previous artifacts are stale")
            write_csv(table, REPORT_FIELDS, checks)
            print(f"FAIL: input errors; see {table}")
            return 1
        check_groups(args.groups, records, report)
        pairs, compared = compare_pairs(records, args.dhash_distance, args.phash_distance)
        exact = sum(p["kind"] == "exact" for p in pairs)
        report("cross_split_pairs", "INFO", compared, "All cross-split image pairs compared")
        report("exact_duplicates", "FAIL" if exact else "PASS", exact, "File SHA256 or decoded RGB SHA256 equality")
        report("near_candidates", "INFO", len(pairs) - exact, "Heuristic only; review and group provenance required")
        review_pairs(args.reviews, pairs, report)
        status = "FAIL" if any(r["status"] == "FAIL" for r in checks) else "PENDING" if any(r["status"] == "PENDING" for r in checks) else "PASS"
        report("overall", status, len(records), "Step 3 only; original splits unchanged; no training authorized by this result")
        write_csv(manifest, MANIFEST_FIELDS, records)
        write_csv(pair_csv, PAIR_FIELDS, pairs)
        write_review_page(page, pairs)
        write_csv(table, REPORT_FIELDS, checks)
    except (OSError, ValueError) as exc:
        report("input_or_output", "FAIL", "", str(exc))
        report("overall", "FAIL", "", "Run incomplete; previous manifests/pairs may be stale")
        write_csv(table, REPORT_FIELDS, checks)
        print(f"FAIL: {exc}; see {table}", file=sys.stderr)
        return 2
    for row in checks:
        print(f"{row['status']}: {row['check']}={row['value']} {row['detail']}")
    print(f"Manifest: {manifest}\nPairs: {pair_csv}\nReview: {page}\nReport: {table}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
