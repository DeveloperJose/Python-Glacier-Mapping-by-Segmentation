#!/usr/bin/env python3
"""Export selected raw targets and donors from GEE.

Uses Step 1 target metadata and Step 3 narrow slate. Exports are aligned to each
target's native GEE grid and full target domain rectangle. GEE exports to Drive;
final canonical local copies should be placed under:

  dataset/raw_full8/targets/
  dataset/raw_full8/donors/

This script intentionally requires explicit --ids unless --all is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import ee
import rasterio
from rasterio.warp import transform_bounds

PROJECT = "hkh-glacier-mapping"
TARGETS_JSON = Path("dataset/outputs/1_targets.json")
SLATE_CSV = Path("dataset/outputs/3_donor_slate_narrow.csv")
EXPORT_MANIFEST_JSON = Path("dataset/outputs/4_export_manifest.json")
EXPORT_MANIFEST_CSV = Path("dataset/outputs/4_export_manifest.csv")
DRIVE_FOLDER = "HKH_dataset_raw_full8"
PRODUCT_DOMAINS = {
    "c02_toa": {
        "manifest_suffix": "",
        "drive_folder": DRIVE_FOLDER,
        "local_root": "dataset/raw_full8",
    },
    "c02_t1_dn": {
        "manifest_suffix": "_c02t1_dn",
        "drive_folder": "HKH_dataset_raw_full8_c02t1_dn",
        "local_root": "dataset/raw_full8_c02t1_dn",
    },
}

# Landsat 7 ETM+ bands needed to match the legacy fishnet imagery.
# LT05 has one thermal band (B6); for consistent donor stacks, LT05 B6 is
# duplicated into B6_VCID_1 and B6_VCID_2. LE07 keeps both real VCID bands.
IMAGE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT"]
STACK_BANDS = IMAGE_BANDS + QA_BANDS + ["data_present", "clear_valid", "slc_gap"]


def initialize_ee() -> None:
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=PROJECT)


def parse_ids(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def load_targets() -> dict[int, dict[str, Any]]:
    rows = json.loads(TARGETS_JSON.read_text(encoding="utf-8"))
    return {int(row["id"]): row for row in rows}


def load_slate() -> list[dict[str, str]]:
    with SLATE_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def domain_manifest_paths(product_domain: str) -> tuple[Path, Path]:
    suffix = PRODUCT_DOMAINS[product_domain]["manifest_suffix"]
    return (
        Path(f"dataset/outputs/4_export_manifest{suffix}.json"),
        Path(f"dataset/outputs/4_export_manifest{suffix}.csv"),
    )


def domain_collection(collection: str, product_id: str, product_domain: str) -> str:
    if product_domain == "c02_toa":
        return collection
    if product_domain != "c02_t1_dn":
        raise ValueError(f"Unknown product domain: {product_domain}")
    if product_id.startswith("LE07"):
        return "LANDSAT/LE07/C02/T1"
    if product_id.startswith("LT05"):
        return "LANDSAT/LT05/C02/T1"
    raise ValueError(f"Cannot infer raw-DN collection for product_id={product_id}")


def standard_image_bands(img: ee.Image, collection: str) -> ee.Image:
    if "LT05" in collection:
        return (
            img.select(["B1", "B2", "B3", "B4", "B5"])
            .addBands(img.select("B6").rename("B6_VCID_1"))
            .addBands(img.select("B6").rename("B6_VCID_2"))
            .addBands(img.select("B7"))
            .select(IMAGE_BANDS)
        )
    return img.select(IMAGE_BANDS)


def simple_valid_mask(img: ee.Image, collection: str) -> ee.Image:
    return standard_image_bands(img, collection).mask().reduce(ee.Reducer.min()).unmask(0, sameFootprint=False).gt(0)


def qa_clear_mask(img: ee.Image, collection: str) -> ee.Image:
    optical = simple_valid_mask(img, collection)
    qa_pixel = img.select("QA_PIXEL")
    qa_radsat = img.select("QA_RADSAT")
    bad_pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
    # ETM+ QA_RADSAT: band saturation bits plus dropped-pixel bit.
    # Include B6_VCID_1/B6_VCID_2 and B7 for the full 8-band stack.
    bad_radsat_bits = (
        (1 << 0)
        | (1 << 1)
        | (1 << 2)
        | (1 << 3)
        | (1 << 4)
        | (1 << 5)
        | (1 << 6)
        | (1 << 7)
        | (1 << 9)
    )
    qa_clear = qa_pixel.bitwiseAnd(bad_pixel_bits).eq(0)
    unsaturated = qa_radsat.bitwiseAnd(bad_radsat_bits).eq(0)
    return optical.And(qa_clear).And(unsaturated).unmask(0, sameFootprint=False).gt(0)


def raw_stack(img: ee.Image, collection: str) -> ee.Image:
    data_present = simple_valid_mask(img, collection).rename("data_present")
    clear_valid = qa_clear_mask(img, collection).rename("clear_valid")
    slc_gap = data_present.Not().rename("slc_gap")
    return (
        standard_image_bands(img, collection)
        .addBands(img.select(QA_BANDS))
        .addBands(data_present)
        .addBands(clear_valid)
        .addBands(slc_gap)
        .select(STACK_BANDS)
        .toFloat()
        .unmask(0, sameFootprint=False)
    )


def image_by_product(collection: str, product_id: str) -> ee.Image:
    coll = ee.ImageCollection(collection).filter(ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id))
    return ee.Image(coll.first())


def export_region(
    target: dict[str, Any],
    pilot_tile_index: int | None = None,
    template_dir: Path | None = None,
) -> ee.Geometry:
    if pilot_tile_index is None:
        return ee.Geometry(target["export_region_geojson"])
    if template_dir is None:
        raise ValueError("template_dir is required for pilot tile exports")
    tile_path = template_dir / f"image{pilot_tile_index}.tif"
    if not tile_path.exists():
        raise FileNotFoundError(f"Missing pilot tile template: {tile_path}")
    with rasterio.open(tile_path) as src:
        bounds = transform_bounds(
            src.crs,
            target["export_crs"],
            *src.bounds,
            densify_pts=21,
        )
    return ee.Geometry.Rectangle(
        list(bounds),
        proj=target["export_crs"],
        geodesic=False,
    )


def target_export_row(target: dict[str, Any], folder: str, product_domain: str) -> dict[str, Any]:
    date = target["date"].replace("-", "")
    name = f"{int(target['id']):02d}_target_{date}"
    return {
        "target_id": int(target["id"]),
        "role": "target",
        "target_scene": target["scene"],
        "target_date": target["date"],
        "donor_kind": "",
        "donor_date": "",
        "product_id": target["gee_product_id"],
        "collection": domain_collection(
            target["target_collection"], target["gee_product_id"], product_domain
        ),
        "metadata_source_collection": target["target_collection"],
        "product_domain": product_domain,
        "description": name,
        "drive_folder": folder,
        "expected_local_path": f"{PRODUCT_DOMAINS[product_domain]['local_root']}/targets/{name}.tif",
        "thermal_mapping": "LE07 real B6_VCID_1/B6_VCID_2",
    }


def donor_export_row(target: dict[str, Any], donor: dict[str, str], folder: str, product_domain: str) -> dict[str, Any]:
    date = donor["donor_date"].replace("-", "")
    name = f"{int(target['id']):02d}_donor_{donor['donor_kind']}_{date}"
    return {
        "target_id": int(target["id"]),
        "role": "donor",
        "target_scene": target["scene"],
        "target_date": target["date"],
        "donor_kind": donor["donor_kind"],
        "donor_date": donor["donor_date"],
        "product_id": donor["donor_product_id"],
        "collection": domain_collection(
            donor["donor_collection"], donor["donor_product_id"], product_domain
        ),
        "metadata_source_collection": donor["donor_collection"],
        "product_domain": product_domain,
        "description": name,
        "drive_folder": folder,
        "expected_local_path": f"{PRODUCT_DOMAINS[product_domain]['local_root']}/donors/{name}.tif",
        "family_score": donor.get("family_score", ""),
        "caution": donor.get("caution", ""),
        "caution_reasons": donor.get("caution_reasons", ""),
        "thermal_mapping": "LT05 B6 duplicated to B6_VCID_1/B6_VCID_2"
        if "LT05" in donor["donor_collection"]
        else "LE07 real B6_VCID_1/B6_VCID_2",
    }


def queue_export(
    row: dict[str, Any],
    target: dict[str, Any],
    pilot_tile_index: int | None = None,
    template_dir: Path | None = None,
) -> str:
    img = image_by_product(row["collection"], row["product_id"])
    task = ee.batch.Export.image.toDrive(
        image=raw_stack(img, row["collection"]),
        description=row["description"],
        folder=row["drive_folder"],
        fileNamePrefix=row["description"],
        region=export_region(target, pilot_tile_index, template_dir),
        crs=target["export_crs"],
        crsTransform=",".join(str(float(x)) for x in target["export_crs_transform"]),
        maxPixels=1e13,
    )
    task.start()
    return task.id


def write_manifest(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=json_path.parent, delete=False) as tmp:
        json.dump(rows, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(json_path)

    fields = sorted({key for row in rows for key in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=csv_path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    tmp_path.replace(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected raw targets/donors to Drive")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", type=str, help="Comma-separated target IDs, e.g. 04,16,26")
    group.add_argument("--all", action="store_true", help="Export all narrow-slate targets/donors")
    parser.add_argument("--folder", default=None)
    parser.add_argument(
        "--product-domain",
        choices=sorted(PRODUCT_DOMAINS),
        default="c02_toa",
        help="GEE product domain to export. c02_t1_dn maps C02 TOA products to C02/T1 raw DN collections.",
    )
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--donors-only", action="store_true")
    parser.add_argument(
        "--pilot-tile-index",
        type=int,
        default=None,
        help="Restrict export region to one legacy fishnet tile for small GEE tests.",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("/home/devj/local-arch/data/HKH_raw/Landsat7_2005"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.targets_only and args.donors_only:
        raise SystemExit("Use at most one of --targets-only/--donors-only")

    targets = load_targets()
    manifest_json, manifest_csv = domain_manifest_paths(args.product_domain)
    folder = args.folder or PRODUCT_DOMAINS[args.product_domain]["drive_folder"]
    slate = load_slate()
    requested = set(targets) if args.all else parse_ids(args.ids)
    unknown = requested - set(targets)
    if unknown:
        raise SystemExit(f"Unknown target IDs: {sorted(unknown)}")

    rows = []
    for target_id in sorted(requested):
        target = targets[target_id]
        if not args.donors_only:
            rows.append(target_export_row(target, folder, args.product_domain))
        if not args.targets_only:
            for donor in slate:
                if int(donor["target_id"]) == target_id:
                    rows.append(donor_export_row(target, donor, folder, args.product_domain))

    if args.pilot_tile_index is not None:
        suffix = f"_tile{args.pilot_tile_index}"
        for row in rows:
            row["pilot_tile_index"] = args.pilot_tile_index
            row["pilot_template_dir"] = str(args.template_dir)
            row["description"] = f"{row['description']}{suffix}"
            row["expected_local_path"] = row["expected_local_path"].replace(
                ".tif", f"{suffix}.tif"
            )

    print(f"export rows: {len(rows)}")
    for row in rows:
        print(
            f"{row['target_id']:02d} {row['role']:6s} {row['description']} "
            f"{row['product_id']} -> Drive/{row['drive_folder']}"
        )

    if args.dry_run:
        write_manifest(rows, manifest_json, manifest_csv)
        print(f"dry run wrote {manifest_json}")
        return

    initialize_ee()
    queued = []
    for row in rows:
        target = targets[int(row["target_id"])]
        task_id = queue_export(row, target, args.pilot_tile_index, args.template_dir)
        row = {**row, "gee_task_id": task_id}
        queued.append(row)
        write_manifest(queued, manifest_json, manifest_csv)
        print(f"queued {row['description']} task={task_id}")

    write_manifest(queued, manifest_json, manifest_csv)
    print(f"wrote {manifest_json}")
    print(f"wrote {manifest_csv}")


if __name__ == "__main__":
    main()
