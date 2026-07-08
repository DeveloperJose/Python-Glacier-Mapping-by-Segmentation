#!/usr/bin/env python3
"""Audit whether legacy Landsat7_2005 can be exactly rebuilt today.

The original paper script used LANDSAT/LE07/C01/T1_RT + .uint8() + gapfill.js.
This script checks current Earth Engine asset availability and audits local legacy
GeoTIFF evidence. It does not submit exports and does not train.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import ee
import numpy as np
import rasterio
from tqdm import tqdm

PROJECT = "hkh-glacier-mapping"
IDS_JS = Path("google_earth_engine/boundary_aware_unet_paper/ids.js")
LEGACY_JS = Path("google_earth_engine/boundary_aware_unet_paper/Landsat7_2005.js")
GAPFILL_JS = Path("google_earth_engine/boundary_aware_unet_paper/gapfill.js")
LEGACY_RAW = Path("/home/devj/local-arch/data/HKH_raw/Landsat7_2005")
OUT_DIR = Path("dataset/outputs")
COLLECTIONS = [
    "LANDSAT/LE07/C01/T1_RT",
    "LANDSAT/LE07/C01/T1",
    "LANDSAT/LE07/C01/T2",
    "LANDSAT/LE07/C02/T1",
    "LANDSAT/LE07/C02/T2",
]
SAMPLE_TILES = [0, 24, 96, 100, 124, 161, 189, 201]
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6_VCID_1", "B6_VCID_2", "B7"]


def parse_ids() -> list[str]:
    text = IDS_JS.read_text(encoding="utf-8")
    return re.findall(r'"(LE07_\d{6}_\d{8})"', text)


def check_asset(collection: str, image_id: str) -> dict[str, Any]:
    asset = f"{collection}/{image_id}"
    try:
        img = ee.Image(asset)
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        bands = img.bandNames().getInfo()
        props = img.toDictionary(["WRS_PATH", "WRS_ROW", "CLOUD_COVER"]).getInfo()
        return {
            "collection": collection,
            "image_id": image_id,
            "asset": asset,
            "available": True,
            "date": date,
            "bands": bands,
            "wrs_path": props.get("WRS_PATH"),
            "wrs_row": props.get("WRS_ROW"),
            "cloud_cover": props.get("CLOUD_COVER"),
            "error": "",
        }
    except Exception as exc:  # EE errors vary by API layer.
        return {
            "collection": collection,
            "image_id": image_id,
            "asset": asset,
            "available": False,
            "date": None,
            "bands": [],
            "wrs_path": None,
            "wrs_row": None,
            "cloud_cover": None,
            "error": str(exc).split("\n")[0],
        }


def audit_assets(image_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for collection in COLLECTIONS:
        for image_id in tqdm(image_ids, desc=f"Check {collection}"):
            rows.append(check_asset(collection, image_id))
    return rows


def audit_local_tile(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        arr = src.read()
        valid = arr != 0
        band_stats: list[dict[str, Any]] = []
        for band_idx in range(src.count):
            values = arr[band_idx]
            vv = values[valid[band_idx]]
            if vv.size == 0:
                stats = {
                    "band": band_idx + 1,
                    "min": None,
                    "p50": None,
                    "p99": None,
                    "max": None,
                    "frac_255": None,
                }
            else:
                stats = {
                    "band": band_idx + 1,
                    "min": int(vv.min()),
                    "p50": float(np.quantile(vv, 0.50)),
                    "p99": float(np.quantile(vv, 0.99)),
                    "max": int(vv.max()),
                    "frac_255": float((vv >= 255).mean()),
                }
            band_stats.append(stats)
        return {
            "file": str(path),
            "exists": True,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0] if src.dtypes else None,
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "nodata": src.nodata,
            "descriptions": list(src.descriptions),
            "mask_all_valid_fraction": float((src.dataset_mask() > 0).mean()),
            "band_stats": band_stats,
        }


def audit_local_legacy() -> dict[str, Any]:
    files = sorted(LEGACY_RAW.glob("image*.tif"))
    numeric_files = sorted(
        files,
        key=lambda p: int(re.search(r"image(\d+)\.tif$", p.name).group(1))
        if re.search(r"image(\d+)\.tif$", p.name)
        else 10**9,
    )
    sample_rows = []
    for tile in SAMPLE_TILES:
        path = LEGACY_RAW / f"image{tile}.tif"
        if path.exists():
            sample_rows.append(audit_local_tile(path))
        else:
            sample_rows.append({"file": str(path), "exists": False})
    return {
        "directory": str(LEGACY_RAW),
        "file_count": len(numeric_files),
        "min_index": 0 if (LEGACY_RAW / "image0.tif").exists() else None,
        "max_index": max(
            [int(re.search(r"image(\d+)\.tif$", p.name).group(1)) for p in numeric_files]
        )
        if numeric_files
        else None,
        "missing_indices": [
            i for i in range(202) if not (LEGACY_RAW / f"image{i}.tif").exists()
        ],
        "sample_tiles": sample_rows,
    }


def write_markdown(report: dict[str, Any]) -> None:
    rows = report["asset_checks"]
    lines: list[str] = ["# Legacy Landsat7 reproducibility audit", ""]
    lines.append("## Original recipe evidence")
    lines.append("")
    lines.append("Original script uses:")
    lines.append("")
    lines.append("```js")
    lines.append("ee.Image('LANDSAT/LE07/C01/T1_RT/' + image_id)")
    lines.append("gapfill.GapFill(image)")
    lines.append(".select(['B1', ..., 'B7']).uint8()")
    lines.append("all_images.filterBounds(geometry).mosaic().clip(geometry)")
    lines.append("```")
    lines.append("")
    lines.append("## Earth Engine availability")
    lines.append("| collection | available IDs | total IDs | first error |")
    lines.append("|---|---:|---:|---|")
    for collection in COLLECTIONS:
        sub = [r for r in rows if r["collection"] == collection]
        ok = [r for r in sub if r["available"]]
        first_err = next((r["error"] for r in sub if not r["available"]), "")
        lines.append(f"| {collection} | {len(ok)} | {len(sub)} | {first_err} |")
    lines.append("")

    c01_ok = any(
        r["available"]
        for r in rows
        if r["collection"] in {"LANDSAT/LE07/C01/T1_RT", "LANDSAT/LE07/C01/T1", "LANDSAT/LE07/C01/T2"}
    )
    if not c01_ok:
        lines.append(
            "**Conclusion:** exact legacy rebuild from current Earth Engine is blocked: "
            "Collection 1 assets used by the original script are unavailable."
        )
        lines.append("")

    lines.append("## Local legacy raw evidence")
    local = report["local_legacy"]
    lines.append(f"Directory: `{local['directory']}`")
    lines.append("")
    lines.append(f"GeoTIFF count: `{local['file_count']}`")
    lines.append(f"Missing expected indices 0-201: `{local['missing_indices']}`")
    lines.append("")
    lines.append("### Sample tiles")
    lines.append("| tile | dtype | bands | size | CRS | valid mask % | B1 p50/p99/max | B1 frac 255 |")
    lines.append("|---:|---|---:|---|---|---:|---|---:|")
    for row in local["sample_tiles"]:
        tile_match = re.search(r"image(\d+)\.tif", row["file"])
        tile = int(tile_match.group(1)) if tile_match else -1
        if not row.get("exists"):
            lines.append(f"| {tile} | missing | | | | | | |")
            continue
        b1 = row["band_stats"][0]
        lines.append(
            f"| {tile} | {row['dtype']} | {row['count']} | {row['width']}x{row['height']} | "
            f"{row['crs']} | {row['mask_all_valid_fraction']*100:.2f} | "
            f"{b1['p50']:.1f}/{b1['p99']:.1f}/{b1['max']} | {b1['frac_255']:.3f} |"
        )
    lines.append("")

    lines.append("## Implication for next experiments")
    lines.append("")
    lines.append("Do not claim exact legacy reconstruction from current public EE assets. ")
    lines.append("Use existing `/HKH_raw/Landsat7_2005` as canonical legacy baseline, or use C02 fallback only as an approximation.")
    lines.append("")
    lines.append("Recommended next ML dataset, if needed: use canonical local legacy imagery with corrected relaxed-valid/modern label policy, not a C02 approximation.")
    (OUT_DIR / "legacy_reproducibility_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_ids = parse_ids()
    ee.Initialize(project=PROJECT)
    report = {
        "project": PROJECT,
        "ids_js": str(IDS_JS),
        "legacy_js": str(LEGACY_JS),
        "gapfill_js": str(GAPFILL_JS),
        "image_id_count": len(image_ids),
        "image_ids": image_ids,
        "collections_checked": COLLECTIONS,
        "asset_checks": audit_assets(image_ids),
        "local_legacy": audit_local_legacy(),
    }
    (OUT_DIR / "legacy_reproducibility_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report)
    print(f"image_ids={len(image_ids)}")
    print(f"wrote {OUT_DIR / 'legacy_reproducibility_audit.json'}")
    print(f"wrote {OUT_DIR / 'legacy_reproducibility_audit.md'}")


if __name__ == "__main__":
    main()
