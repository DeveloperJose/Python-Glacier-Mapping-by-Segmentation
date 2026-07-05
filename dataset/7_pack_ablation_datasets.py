#!/usr/bin/env python3
"""Pack namespaced Landsat+DEM ablation datasets from processed full datasets.

This avoids collisions with the generic recipe output name used by
scripts/preprocess.py while reusing its audited packing helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.preprocess import PACKED_RECIPES, pack_recipe_dataset

OUTPUT_ROOT = Path("/home/devj/local-arch/data/HKH")
RECIPE_CHANNELS = PACKED_RECIPES["comprehensive_v3_landsat_dem"]

DATASETS = {
    "comprehensive_v3": "comprehensive_v3_legacy_landsat_dem",
    "comprehensive_v3_hkh_full8_raw_target": "comprehensive_v3_hkh_full8_raw_target_landsat_dem",
    "comprehensive_v3_hkh_full8_agreement_quality_step3": "comprehensive_v3_hkh_full8_agreement_quality_step3_landsat_dem",
    "comprehensive_v3_hkh_full8_nspi_timeseries_weighted": "comprehensive_v3_hkh_full8_nspi_timeseries_weighted_landsat_dem",
}


def main() -> None:
    for source_name, target_name in DATASETS.items():
        source_dir = OUTPUT_ROOT / source_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source dataset: {source_dir}")
        pack_recipe_dataset(
            source_dir=source_dir,
            output_root=OUTPUT_ROOT,
            recipe_name=target_name,
            recipe_channels=RECIPE_CHANNELS,
            x_dtype=np.dtype("float32"),
            dry_run=False,
        )


if __name__ == "__main__":
    main()
