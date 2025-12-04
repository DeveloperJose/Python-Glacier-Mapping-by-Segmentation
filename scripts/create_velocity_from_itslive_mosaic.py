#!/usr/bin/env python3
"""
Create velocity products from ITS_LIVE annual mosaics with automatic datacube discovery.

This script uses the official ITS_LIVE catalog (catalog_v02.json) to find all overlapping
datacubes for each image, extracts velocity data for 2002-2008, and generates 4-band
velocity products (v, vx, vy, mask).

Key features:
- Uses official ITS_LIVE catalog for comprehensive datacube coverage
- BBox overlap matching (not just center-point) for better coverage
- Tries multiple overlapping datacubes until one succeeds
- Handles cross-UTM-zone reprojection automatically
- Multiprocessing support for parallel processing

Temporal alignment:
- Labels: 2002-2008 (ICIMOD HKH glacier inventory, 2005±3 years)
- Landsat: 2001-2009 (mosaic with 2000 gap-fill for SLC-off)
- Velocity: 2002-2008 median (7-year robust estimator)

Usage:
    # Test with single image
    uv run python scripts/create_velocity_from_itslive_mosaic.py --max-images 1
    
    # Process first 10 images
    uv run python scripts/create_velocity_from_itslive_mosaic.py --max-images 10
    
    # Process all 202 images (default)
    uv run python scripts/create_velocity_from_itslive_mosaic.py
    
    # Use 8 parallel workers
    uv run python scripts/create_velocity_from_itslive_mosaic.py --workers 8
"""

import argparse
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
import s3fs
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, shape
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ITS_LIVE mosaic configuration
CATALOG_PATH = Path(__file__).parent.parent / "catalog_v02.json"
VELOCITY_YEARS = list(range(2002, 2009))  # 2002-2008 inclusive
VELOCITY_RESOLUTION = 120  # meters
TARGET_RESOLUTION = 30  # meters (Landsat)


def load_catalog(catalog_path: Path) -> List[Dict]:
    """
    Load the official ITS_LIVE datacube catalog.
    
    Args:
        catalog_path: Path to catalog_v02.json
        
    Returns:
        List of datacube entries with geometry, epsg, and URLs
    """
    with open(catalog_path) as f:
        catalog = json.load(f)
    
    datacubes = []
    for feat in catalog.get('features', []):
        props = feat.get('properties', {})
        geom = feat.get('geometry')
        
        if not geom or not props.get('composite_zarr_url'):
            continue
        
        # Parse geometry to shapely
        try:
            geom_shape = shape(geom)
        except Exception:
            continue
        
        # Convert HTTP URL to S3 URL
        composite_url = props.get('composite_zarr_url', '')
        if composite_url.startswith('http://its-live-data.s3.amazonaws.com/'):
            s3_url = composite_url.replace(
                'http://its-live-data.s3.amazonaws.com/',
                's3://its-live-data/'
            )
        elif composite_url.startswith('https://'):
            s3_url = composite_url.replace(
                'https://its-live-data.s3.amazonaws.com/',
                's3://its-live-data/'
            )
        else:
            s3_url = composite_url
        
        datacubes.append({
            'geometry': geom_shape,
            'bbox': geom_shape.bounds,  # (minx, miny, maxx, maxy) in lat/lon
            'epsg': props.get('epsg'),
            'composite_url': s3_url,
            'zarr_url': props.get('zarr_url'),
            'coverage': props.get('roi_percent_coverage', 0),
        })
    
    return datacubes


def find_overlapping_datacubes(
    image_bbox_latlon: Tuple[float, float, float, float],
    catalog: List[Dict]
) -> List[Dict]:
    """
    Find all datacubes that overlap with an image's bounding box.
    
    Args:
        image_bbox_latlon: (min_lon, min_lat, max_lon, max_lat) in WGS84
        catalog: List of datacube entries from load_catalog()
        
    Returns:
        List of overlapping datacubes, sorted by coverage (highest first)
    """
    image_box = box(*image_bbox_latlon)
    
    overlapping = []
    for dc in catalog:
        if dc['geometry'].intersects(image_box):
            # Compute intersection area for sorting
            intersection = dc['geometry'].intersection(image_box)
            overlap_pct = intersection.area / image_box.area * 100
            dc_copy = dc.copy()
            dc_copy['overlap_pct'] = overlap_pct
            overlapping.append(dc_copy)
    
    # Sort by overlap percentage (highest first)
    overlapping.sort(key=lambda x: x['overlap_pct'], reverse=True)
    
    return overlapping


def get_image_bbox_latlon(
    bounds: Tuple[float, float, float, float],
    epsg_code: int
) -> Tuple[float, float, float, float]:
    """
    Convert image bounds from UTM to lat/lon (WGS84).
    
    Args:
        bounds: (minx, miny, maxx, maxy) in UTM coordinates
        epsg_code: EPSG code of the image CRS
        
    Returns:
        (min_lon, min_lat, max_lon, max_lat) in WGS84
    """
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    
    # Transform all corners to handle rotation/distortion
    corners = [
        (bounds[0], bounds[1]),  # SW
        (bounds[0], bounds[3]),  # NW
        (bounds[2], bounds[1]),  # SE
        (bounds[2], bounds[3]),  # NE
    ]
    
    lons, lats = [], []
    for x, y in corners:
        lon, lat = transformer.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    
    return (min(lons), min(lats), max(lons), max(lats))


def load_itslive_mosaic(zarr_url: str) -> xr.Dataset:
    """
    Load ITS_LIVE velocity mosaic from S3.
    
    Args:
        zarr_url: S3 URL to Zarr dataset
        
    Returns:
        xarray Dataset with velocity components
    """
    # Use anonymous S3 access
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=zarr_url, s3=s3, check=False)
    
    ds = xr.open_zarr(store, consolidated=True)
    return ds


def extract_temporal_median(
    ds: xr.Dataset,
    years: List[int],
    bounds: Tuple[float, float, float, float],
    crs: str = "EPSG:32645"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Extract velocity data for specified years and compute temporal median.
    
    Args:
        ds: ITS_LIVE xarray Dataset
        years: List of years to extract
        bounds: (minx, miny, maxx, maxy) in target CRS
        crs: Target CRS (should match ds.crs)
        
    Returns:
        Tuple of (v_median, vx_median, vy_median, transform)
    """
    minx, miny, maxx, maxy = bounds
    
    # Crop to bounding box
    ds_cropped = ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny)  # y decreases
    )
    
    # Filter by years
    year_mask = ds_cropped.time.dt.year.isin(years)
    ds_years = ds_cropped.isel(time=year_mask)
    
    n_years = len(ds_years.time)
    
    if n_years == 0:
        raise ValueError(f"No data found for years {years}")
    
    # Compute temporal median (robust to outliers)
    v_median = ds_years['v'].median(dim='time').values
    vx_median = ds_years['vx'].median(dim='time').values
    vy_median = ds_years['vy'].median(dim='time').values
    
    # Get spatial metadata
    x = ds_years.x.values
    y = ds_years.y.values
    
    # Handle empty arrays
    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"Empty spatial dimensions after cropping")
    
    # Create affine transform
    transform = from_bounds(
        west=float(x.min()),
        south=float(y.min()),
        east=float(x.max()),
        north=float(y.max()),
        width=len(x),
        height=len(y)
    )
    
    return v_median, vx_median, vy_median, transform


def resample_to_target(
    data: np.ndarray,
    src_transform: object,
    src_crs: str,
    dst_shape: Tuple[int, int],
    dst_transform: object,
    dst_crs: str,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Resample velocity data to match target Landsat grid.
    
    Args:
        data: Source velocity array
        src_transform: Source affine transform
        src_crs: Source CRS
        dst_shape: Target (height, width)
        dst_transform: Target affine transform
        dst_crs: Target CRS
        fill_value: Value for pixels outside source extent
        
    Returns:
        Resampled array matching target grid
    """
    dst_array = np.full(dst_shape, fill_value, dtype=np.float32)
    
    reproject(
        source=data.astype(np.float32),
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=fill_value
    )
    
    return dst_array


def try_extract_velocity(
    datacube_url: str,
    datacube_epsg: int,
    landsat_bounds,
    landsat_epsg: int,
    landsat_shape: Tuple[int, int],
    landsat_transform,
    landsat_crs,
    years: List[int]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Try to extract velocity data from a single datacube.
    
    Args:
        datacube_url: S3 URL to the datacube
        datacube_epsg: EPSG code of the datacube
        landsat_bounds: Bounds of the Landsat image
        landsat_epsg: EPSG code of the Landsat image
        landsat_shape: (height, width) of Landsat image
        landsat_transform: Affine transform of Landsat image
        landsat_crs: CRS of Landsat image
        years: List of years to extract
        
    Returns:
        Tuple of (v, vx, vy, mask) arrays or None if extraction fails
    """
    # Load ITS_LIVE mosaic
    ds = load_itslive_mosaic(datacube_url)
    
    # Determine bounds in datacube CRS (handle cross-zone reprojection)
    if datacube_epsg != landsat_epsg:
        # Reproject bounds to datacube CRS
        transformer_bounds = Transformer.from_crs(
            f"EPSG:{landsat_epsg}",
            f"EPSG:{datacube_epsg}",
            always_xy=True
        )
        # Transform all corners
        xs = [landsat_bounds.left, landsat_bounds.right]
        ys = [landsat_bounds.bottom, landsat_bounds.top]
        corners_x, corners_y = [], []
        for x in xs:
            for y in ys:
                tx, ty = transformer_bounds.transform(x, y)
                corners_x.append(tx)
                corners_y.append(ty)
        
        datacube_bounds = (
            min(corners_x), min(corners_y),
            max(corners_x), max(corners_y)
        )
    else:
        datacube_bounds = (
            landsat_bounds.left, landsat_bounds.bottom,
            landsat_bounds.right, landsat_bounds.top
        )
    
    # Extract velocity median
    v_median, vx_median, vy_median, velocity_transform = extract_temporal_median(
        ds=ds,
        years=years,
        bounds=datacube_bounds,
    )
    
    # Resample to Landsat grid (handles CRS conversion automatically)
    datacube_crs_str = f"EPSG:{datacube_epsg}"
    v_resampled = resample_to_target(
        v_median, velocity_transform, datacube_crs_str,
        landsat_shape, landsat_transform, str(landsat_crs)
    )
    vx_resampled = resample_to_target(
        vx_median, velocity_transform, datacube_crs_str,
        landsat_shape, landsat_transform, str(landsat_crs)
    )
    vy_resampled = resample_to_target(
        vy_median, velocity_transform, datacube_crs_str,
        landsat_shape, landsat_transform, str(landsat_crs)
    )
    
    # Create velocity mask (1=valid, 0=no_data)
    velocity_mask = (~np.isnan(v_resampled)).astype(np.float32)
    
    return v_resampled, vx_resampled, vy_resampled, velocity_mask


def process_single_image(
    args: Tuple[int, Path, Path, List[Dict], List[int]]
) -> Dict:
    """
    Process a single image to generate velocity product.
    
    Tries multiple overlapping datacubes until one succeeds.
    
    Args:
        args: Tuple of (image_idx, input_path, output_dir, catalog, years)
        
    Returns:
        Dict with processing statistics
    """
    image_idx, input_path, output_dir, catalog, years = args
    
    output_path = output_dir / input_path.name
    stats_path = output_dir / f"{input_path.stem}_stats.json"
    
    try:
        # Load Landsat metadata
        with rasterio.open(input_path) as src:
            landsat_meta = src.meta.copy()
            landsat_bounds = src.bounds
            landsat_shape = (src.height, src.width)
            landsat_transform = src.transform
            landsat_crs = src.crs
        
        # Extract EPSG code
        epsg_code = int(str(landsat_crs).split(':')[-1])
        
        # Convert image bounds to lat/lon and find overlapping datacubes
        image_bbox_latlon = get_image_bbox_latlon(landsat_bounds, epsg_code)
        overlapping = find_overlapping_datacubes(image_bbox_latlon, catalog)
        
        if not overlapping:
            raise ValueError(f"No overlapping datacubes found for image at {image_bbox_latlon}")
        
        # Try each overlapping datacube until one works
        last_error = None
        datacube_url = None
        datacube_epsg = None
        v_resampled = None
        vx_resampled = None
        vy_resampled = None
        velocity_mask = None
        
        for dc in overlapping:
            datacube_url = dc['composite_url']
            datacube_epsg = dc['epsg']
            
            try:
                result = try_extract_velocity(
                    datacube_url=datacube_url,
                    datacube_epsg=datacube_epsg,
                    landsat_bounds=landsat_bounds,
                    landsat_epsg=epsg_code,
                    landsat_shape=landsat_shape,
                    landsat_transform=landsat_transform,
                    landsat_crs=landsat_crs,
                    years=years
                )
                
                if result is not None:
                    v_resampled, vx_resampled, vy_resampled, velocity_mask = result
                    # Check if we got any valid data
                    if np.sum(velocity_mask) > 0:
                        break  # Success!
                    else:
                        last_error = "No valid velocity data in datacube"
            except Exception as e:
                last_error = str(e)
                continue
        
        if v_resampled is None or velocity_mask is None or np.sum(velocity_mask) == 0:
            raise ValueError(
                f"Failed to extract velocity from {len(overlapping)} datacubes. "
                f"Last error: {last_error}"
            )
        
        # Compute coverage statistics
        total_pixels = velocity_mask.size
        valid_pixels = int(np.sum(velocity_mask))
        coverage_pct = (valid_pixels / total_pixels) * 100
        
        # Velocity statistics (only valid pixels)
        v_valid = v_resampled[~np.isnan(v_resampled)]
        
        # Fill NaN→0 (consistent with slice.py pattern)
        v_filled = np.nan_to_num(v_resampled, nan=0.0)
        vx_filled = np.nan_to_num(vx_resampled, nan=0.0)
        vy_filled = np.nan_to_num(vy_resampled, nan=0.0)
        
        # Stack: [v, vx, vy, mask] = 4 bands
        output_data = np.stack([v_filled, vx_filled, vy_filled, velocity_mask])
        
        # Update metadata
        output_meta = landsat_meta.copy()
        output_meta.update({
            'count': 4,
            'dtype': 'float32',
            'nodata': None  # No NaN after filling
        })
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(output_data.astype(np.float32))
            dst.set_band_description(1, f"velocity_magnitude_median_{years[0]}_{years[-1]}")
            dst.set_band_description(2, f"velocity_vx_median_{years[0]}_{years[-1]}")
            dst.set_band_description(3, f"velocity_vy_median_{years[0]}_{years[-1]}")
            dst.set_band_description(4, "velocity_mask")
        
        # Generate statistics
        stats = {
            'image_name': input_path.stem,
            'image_index': image_idx,
            'datacube_url': datacube_url,
            'epsg_code': epsg_code,
            'datacube_epsg': datacube_epsg,
            'cross_zone_reproj': (datacube_epsg != epsg_code),
            'num_datacubes_tried': len(overlapping),
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'coverage_percent': float(coverage_pct),
            'velocity_stats': {
                'mean': float(np.mean(v_valid)) if len(v_valid) > 0 else None,
                'median': float(np.median(v_valid)) if len(v_valid) > 0 else None,
                'min': float(np.min(v_valid)) if len(v_valid) > 0 else None,
                'max': float(np.max(v_valid)) if len(v_valid) > 0 else None,
                'std': float(np.std(v_valid)) if len(v_valid) > 0 else None,
            },
            'status': 'success'
        }
        
        # Save stats
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to process {input_path.name}: {e}")
        return {
            'image_name': input_path.stem,
            'image_index': image_idx,
            'status': 'failed',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate velocity products from ITS_LIVE mosaics"
    )
    parser.add_argument(
        '--landsat-dir',
        type=Path,
        default=Path('/home/devj/local-debian/datasets/HKH_raw/Landsat7_2005'),
        help='Input Landsat directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/home/devj/local-debian/datasets/HKH_raw/Velocity'),
        help='Output velocity directory'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: 75%% of CPU cores)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2002,
        help='Start year for velocity median'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2008,
        help='End year for velocity median (inclusive)'
    )
    parser.add_argument(
        '--catalog',
        type=Path,
        default=None,
        help='Path to ITS_LIVE catalog JSON (default: catalog_v02.json in project root)'
    )
    
    args = parser.parse_args()
    
    # Load official ITS_LIVE catalog
    catalog_path = args.catalog if hasattr(args, 'catalog') and args.catalog else CATALOG_PATH
    logger.info(f"Loading ITS_LIVE catalog from {catalog_path}...")
    catalog = load_catalog(catalog_path)
    logger.info(f"Loaded {len(catalog)} datacubes from catalog")
    
    # Find all Landsat images
    images = sorted(args.landsat_dir.glob('image*.tif'))
    
    if args.max_images is not None:
        images = images[:args.max_images]
        logger.info(f"Processing first {args.max_images} images")
    else:
        logger.info(f"Processing all {len(images)} images")
    
    # Determine worker count
    cores = multiprocessing.cpu_count()
    workers = args.workers if args.workers is not None else max(1, int(cores * 0.75))
    logger.info(f"Using {workers}/{cores} CPU cores")
    
    # Prepare arguments for multiprocessing
    years = list(range(args.start_year, args.end_year + 1))
    process_args = [
        (i, img_path, args.output_dir, catalog, years)
        for i, img_path in enumerate(images)
    ]
    
    # Process with multiprocessing
    logger.info("="*60)
    logger.info("Starting velocity generation...")
    logger.info("="*60)
    
    results = []
    with multiprocessing.Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_image, process_args),
            total=len(process_args),
            desc="Processing images"
        ):
            results.append(result)
    
    # Generate summary
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    # Create summary DataFrame
    if success:
        df = pd.DataFrame([
            {
                'image_name': r['image_name'],
                'epsg_code': r.get('epsg_code'),
                'datacube_epsg': r.get('datacube_epsg'),
                'cross_zone_reproj': r.get('cross_zone_reproj', False),
                'num_datacubes_tried': r.get('num_datacubes_tried', 1),
                'datacube_url': r.get('datacube_url'),
                'total_pixels': r['total_pixels'],
                'valid_pixels': r['valid_pixels'],
                'coverage_percent': r['coverage_percent'],
                'mean_velocity': r['velocity_stats']['mean'],
                'median_velocity': r['velocity_stats']['median'],
                'min_velocity': r['velocity_stats']['min'],
                'max_velocity': r['velocity_stats']['max'],
                'std_velocity': r['velocity_stats']['std'],
            }
            for r in success
        ])
        
        summary_path = args.output_dir / 'velocity_coverage_summary.csv'
        df.to_csv(summary_path, index=False)
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("VELOCITY COVERAGE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total images processed: {len(results)}")
        logger.info(f"Successful: {len(success)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"\nCoverage statistics:")
        logger.info(f"  Mean coverage:   {df['coverage_percent'].mean():.2f}%")
        logger.info(f"  Median coverage: {df['coverage_percent'].median():.2f}%")
        logger.info(f"  Min coverage:    {df['coverage_percent'].min():.2f}%")
        logger.info(f"  Max coverage:    {df['coverage_percent'].max():.2f}%")
        logger.info(f"\nImages with <10% coverage: {len(df[df['coverage_percent'] < 10])}")
        logger.info(f"Images with <1% coverage:  {len(df[df['coverage_percent'] < 1])}")
        logger.info(f"\nSummary saved to: {summary_path}")
    
    if failed:
        logger.info(f"\nFailed images ({len(failed)}):")
        for fail in failed[:10]:  # Show first 10
            logger.info(f"  {fail['image_name']}: {fail['error']}")
        if len(failed) > 10:
            logger.info(f"  ... and {len(failed) - 10} more")
    
    logger.info("="*60)
    logger.info("Processing complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
