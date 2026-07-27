#!/usr/bin/env python3
"""Export the supported HKH Collection 2 Landsat rebuild from one manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import ee


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "dataset/hkh_rebuild_manifest.json"
IMAGE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]
QA_BANDS = ["QA_PIXEL", "QA_RADSAT"]
STACK_BANDS = IMAGE_BANDS + QA_BANDS + ["data_present", "clear_valid", "slc_gap"]


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    return manifest


def resolve_variant(
    manifest: dict[str, Any], variant_name: str
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    variants = manifest["variants"]
    if variant_name not in variants:
        raise ValueError(
            f"Unknown dataset variant '{variant_name}'; choices: {sorted(variants)}"
        )

    targets = {int(row["id"]): row for row in manifest["targets"]}
    donors = {int(target_id): rows for target_id, rows in manifest["donors"].items()}

    chain: list[dict[str, Any]] = []
    current: str | None = variant_name
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"Variant inheritance cycle at '{current}'")
        seen.add(current)
        spec = variants[current]
        chain.append(spec)
        current = spec.get("base")

    for spec in reversed(chain):
        for row in spec.get("target_overrides", []):
            targets[int(row["id"])] = row
        for target_id, rows in spec.get("donor_overrides", {}).items():
            donors[int(target_id)] = rows
    return targets, donors


def initialize_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=project)


def standard_image_bands(image: ee.Image, collection: str) -> ee.Image:
    if "/LT05/" in collection:
        return (
            image.select(["B1", "B2", "B3", "B4", "B5"])
            .addBands(image.select("B6").rename("B6_VCID_1"))
            .addBands(image.select("B6").rename("B6_VCID_2"))
            .addBands(image.select("B7"))
            .select(IMAGE_BANDS)
        )
    return image.select(IMAGE_BANDS)


def simple_valid_mask(image: ee.Image, collection: str) -> ee.Image:
    return (
        standard_image_bands(image, collection)
        .mask()
        .reduce(ee.Reducer.min())
        .unmask(0, sameFootprint=False)
        .gt(0)
    )


def qa_clear_mask(image: ee.Image, collection: str) -> ee.Image:
    qa_pixel = image.select("QA_PIXEL")
    qa_radsat = image.select("QA_RADSAT")
    bad_pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)
    bad_radsat_bits = sum(1 << bit for bit in (*range(8), 9))
    return (
        simple_valid_mask(image, collection)
        .And(qa_pixel.bitwiseAnd(bad_pixel_bits).eq(0))
        .And(qa_radsat.bitwiseAnd(bad_radsat_bits).eq(0))
        .unmask(0, sameFootprint=False)
        .gt(0)
    )


def raw_stack(image: ee.Image, collection: str) -> ee.Image:
    data_present = simple_valid_mask(image, collection).rename("data_present")
    clear_valid = qa_clear_mask(image, collection).rename("clear_valid")
    slc_gap = data_present.Not().rename("slc_gap")
    return (
        standard_image_bands(image, collection)
        .addBands(image.select(QA_BANDS))
        .addBands(data_present)
        .addBands(clear_valid)
        .addBands(slc_gap)
        .select(STACK_BANDS)
        .toFloat()
        .unmask(0, sameFootprint=False)
    )


def image_by_product(collection: str, product_id: str) -> ee.Image:
    images = ee.ImageCollection(collection).filter(
        ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id)
    )
    return ee.Image(images.first())


def export_rows(
    targets: dict[int, dict[str, Any]],
    donors: dict[int, list[dict[str, Any]]],
    target_ids: list[int],
    roles: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        target = targets[target_id]
        if roles in {"all", "targets"}:
            rows.append(
                {
                    "target_id": target_id,
                    "role": "target",
                    "filename": target["filename"],
                    "collection": target["collection"],
                    "product_id": target["product_id"],
                }
            )
        if roles in {"all", "donors"}:
            for donor in donors.get(target_id, []):
                rows.append(
                    {
                        "target_id": target_id,
                        "role": "donor",
                        "filename": donor["filename"],
                        "collection": donor["collection"],
                        "product_id": donor["product_id"],
                    }
                )
    return rows


def queue_export(row: dict[str, Any], target: dict[str, Any], folder: str) -> str:
    description = Path(row["filename"]).stem
    image = image_by_product(row["collection"], row["product_id"])
    task = ee.batch.Export.image.toDrive(
        image=raw_stack(image, row["collection"]),
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=ee.Geometry(target["export_region_geojson"]),
        crs=target["export_crs"],
        crsTransform=",".join(
            str(float(value)) for value in target["export_crs_transform"]
        ),
        maxPixels=1e13,
    )
    task.start()
    return task.id


def load_fishnet(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = REPO_ROOT / manifest["fishnet"]["path"]
    with path.open(encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    return sorted(features, key=lambda row: int(row["properties"]["_export_index"]))


def queue_dem_export(
    feature: dict[str, Any],
    targets: dict[int, dict[str, Any]],
    folder: str,
) -> str:
    index = int(feature["properties"]["_export_index"])
    geometry = ee.Geometry(feature["geometry"])
    target_images = [
        image_by_product(target["collection"], target["product_id"])
        for target in targets.values()
    ]
    reference = ee.ImageCollection.fromImages(target_images).filterBounds(geometry)
    crs = ee.Image(reference.first()).select("B1").projection().crs().getInfo()
    elevation = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope")
    aspect = ee.Terrain.aspect(elevation).rename("aspect")
    curvature = ee.Terrain.slope(slope).rename("curvature")
    dem = elevation.addBands([slope, aspect, curvature]).toFloat()
    description = f"image{index}"
    task = ee.batch.Export.image.toDrive(
        image=dem.clip(geometry),
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=geometry,
        crs=crs,
        scale=30,
        maxPixels=1e13,
    )
    task.start()
    return task.id


def parse_ids(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--variant", default="c02_current")
    parser.add_argument(
        "--asset",
        choices=("landsat", "dem"),
        default="landsat",
        help="Export manifest Landsat scenes or fishnet-aligned NASADEM",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--ids", help="Comma-separated target IDs, or DEM tile indices"
    )
    selection.add_argument("--all", action="store_true", help="Export all items")
    parser.add_argument("--roles", choices=("all", "targets", "donors"), default="all")
    parser.add_argument(
        "--project",
        default=os.environ.get("EE_PROJECT"),
        help="Earth Engine project ID (or set EE_PROJECT)",
    )
    parser.add_argument(
        "--drive-folder",
        default=None,
        help="Google Drive destination folder (default: HKH_<variant>)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print without Earth Engine"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    targets, donors = resolve_variant(manifest, args.variant)
    if args.asset == "dem":
        features = load_fishnet(manifest)
        by_index = {int(row["properties"]["_export_index"]): row for row in features}
        tile_indices = sorted(by_index) if args.all else parse_ids(args.ids)
        unknown = sorted(set(tile_indices) - set(by_index))
        if unknown:
            raise ValueError(f"Unknown DEM tile indices: {unknown}")
        folder = args.drive_folder or "HKH_DEM"
        print(f"asset=dem tiles={len(tile_indices)}")
        for index in tile_indices:
            print(f"{index:03d} DEM -> Drive/{folder}/image{index}.tif")
        if args.dry_run:
            return
        if not args.project:
            raise ValueError("Set --project or EE_PROJECT before starting exports")
        initialize_ee(args.project)
        for index in tile_indices:
            task_id = queue_dem_export(by_index[index], targets, folder)
            print(f"queued image{index}.tif task={task_id}", flush=True)
        return

    target_ids = sorted(targets) if args.all else parse_ids(args.ids)
    unknown = sorted(set(target_ids) - set(targets))
    if unknown:
        raise ValueError(f"Unknown target IDs: {unknown}")
    rows = export_rows(targets, donors, target_ids, args.roles)
    folder = args.drive_folder or f"HKH_{args.variant}"

    print(f"variant={args.variant} targets={len(target_ids)} exports={len(rows)}")
    for row in rows:
        print(
            f"{row['target_id']:02d} {row['role']:6s} {row['product_id']} "
            f"-> Drive/{folder}/{row['filename']}"
        )
    if args.dry_run:
        return
    if not args.project:
        raise ValueError("Set --project or EE_PROJECT before starting exports")

    initialize_ee(args.project)
    for row in rows:
        task_id = queue_export(row, targets[row["target_id"]], folder)
        print(f"queued {row['filename']} task={task_id}", flush=True)


if __name__ == "__main__":
    main()
