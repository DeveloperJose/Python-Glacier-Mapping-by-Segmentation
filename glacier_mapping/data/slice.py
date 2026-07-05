import os
import pathlib

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from skimage.color import rgb2hsv

from glacier_mapping.data import physics
import glacier_mapping.utils.logging as log

IGNORE_LABEL = 255
DEFAULT_VELOCITY_CLIP_M_PER_YEAR = 1000.0


def _same_raster_grid(src: rasterio.DatasetReader, dst: rasterio.DatasetReader) -> bool:
    if src.crs != dst.crs:
        return False
    if src.width != dst.width or src.height != dst.height:
        return False
    return np.allclose(tuple(src.transform), tuple(dst.transform), atol=1e-6)


def plot_image_and_mask(rgb, mask, title, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mask = mask.astype(np.uint8)

    cmap = mcolors.ListedColormap(["black", "cyan", "yellow", "magenta"])
    bounds = [0, 1, 2, 255, 256]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    rgb = rgb.astype(np.float32)
    rgb /= max(1e-6, np.percentile(rgb, 99))
    rgb = np.clip(rgb, 0, 1)
    axs[0].imshow(rgb)
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    im = axs[1].imshow(mask, cmap=cmap, norm=norm)
    axs[1].set_title(title)
    axs[1].axis("off")

    cbar = fig.colorbar(
        im,
        ax=axs[1],
        ticks=[0.5, 1.5, 2.5, 255.5],
        boundaries=bounds,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(["0 = BG", "1 = CI", "2 = DCI", "255 = IGNORE"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def read_shp(filename):
    """Read shapefile and return geopandas dataframe."""
    shapefile = gpd.read_file(filename)
    return shapefile


def read_tiff(filename):
    """Read GeoTIFF and return rasterio dataset."""
    dataset = rasterio.open(filename)
    return dataset


def check_crs(crs_a, crs_b, verbose=False):
    """Verify that two CRS objects match. Raises ValueError if they don't agree."""
    if verbose:
        log.debug("CRS 1: " + crs_a.to_string() + ", CRS 2: " + crs_b.to_string())
    if rasterio.crs.CRS.from_string(crs_a.to_string()) != rasterio.crs.CRS.from_string(
        crs_b.to_string()
    ):
        raise ValueError("Coordinate reference systems do not agree")


def clip_shapefile(img_bounds, img_meta, shp):
    """Clip shapefile to image bounding box, removing non-overlapping polygons."""
    bbox = box(*img_bounds)
    bbox_poly = gpd.GeoDataFrame(
        {"geometry": bbox}, index=[0], crs=img_meta["crs"].data
    )
    return shp.loc[shp.intersects(bbox_poly["geometry"][0])]


def poly_from_coord(polygon, transform):
    """Transform polygon coordinates using rasterio transform."""
    poly_pts = []
    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        # in case polygonz format
        poly_pts.append(~transform * tuple(i)[:2])
    return Polygon(poly_pts)


def get_mask(tiff, shp, column="Glaciers"):
    """
    Generate multi-class mask from shapefile labels.

    Args:
        tiff: Rasterio dataset or path to GeoTIFF
        shp: Geopandas dataframe with label polygons
        column: Column name for class labels

    Returns:
        Mask array of shape (height, width, num_classes)
    """

    if isinstance(tiff, (str, pathlib.Path)):
        tiff = read_tiff(tiff)

    classes = sorted(list(set(shp[column])))
    # print(f"Classes = {classes}")

    shapefile_crs = rasterio.crs.CRS.from_string(str(shp.crs))

    if shapefile_crs != tiff.meta["crs"]:
        shp = shp.to_crs(tiff.meta["crs"].data)
    check_crs(tiff.crs, shp.crs)
    shapefile = clip_shapefile(tiff.bounds, tiff.meta, shp)
    mask = np.zeros((tiff.height, tiff.width, len(classes)))

    for key, value in enumerate(classes):
        geom = shapefile[shapefile[column] == value]
        poly_shp = []
        im_size = (tiff.meta["height"], tiff.meta["width"])
        for num, row in geom.iterrows():
            if row["geometry"].geom_type == "Polygon":
                poly_shp.append(
                    poly_from_coord(row["geometry"], tiff.meta["transform"])
                )
            else:
                for p in row["geometry"].geoms:
                    poly_shp.append(poly_from_coord(p, tiff.meta["transform"]))
        try:
            channel_mask = rasterize(shapes=poly_shp, out_shape=im_size)
            mask[:, :, key] = channel_mask
        except Exception as e:
            log.warning(f"Rasterization failed for class={key}: {e}")
            continue

    return mask


def add_index(tiff_np, index1, index2):
    numerator = tiff_np[:, :, index1] - tiff_np[:, :, index2]
    denominator = tiff_np[:, :, index1] + tiff_np[:, :, index2]
    rsi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )
    rsi = np.nan_to_num(rsi).clip(-1, 1)
    tiff_np = np.concatenate((tiff_np, np.expand_dims(rsi, axis=2)), axis=2)
    return tiff_np


def compute_dems(dem_np):
    """
    From DEM product bands, build DEM features for the network.

    Assumes:
        dem_np[:, :, 0] = elevation (meters)
        dem_np[:, :, 1] = slope (degrees)

    Returns:
        dem_feat : (H, W, 2) with [elevation_raw, slope_deg_raw]
    """
    elevation = dem_np[:, :, 0][:, :, None]  # raw meters
    slope_deg = dem_np[:, :, 1][:, :, None]  # raw degrees from DEM
    dem_np = np.concatenate((elevation, slope_deg), axis=2)
    return dem_np


def compute_landsat_valid_mask(tiff, landsat_raw: np.ndarray) -> np.ndarray:
    """Recover source-valid Landsat pixels before auxiliary channels are added."""
    dataset_valid = tiff.dataset_mask() > 0
    proxy_valid = np.isfinite(landsat_raw[:, :, :8]).all(axis=2) & np.any(
        landsat_raw[:, :, :8] != 0, axis=2
    )

    if dataset_valid.shape != proxy_valid.shape:
        log.warning(
            "Landsat dataset_mask shape does not match raster shape; using nonzero proxy."
        )
        return proxy_valid

    if np.all(dataset_valid):
        return proxy_valid

    return dataset_valid & proxy_valid


def clip_velocity_channels(
    velocity_np: np.ndarray,
    max_velocity_m_per_year: float | None = DEFAULT_VELOCITY_CLIP_M_PER_YEAR,
) -> np.ndarray:
    """Clip velocity magnitude and vector components while preserving mask semantics."""
    if max_velocity_m_per_year is None:
        return velocity_np

    max_velocity = float(max_velocity_m_per_year)
    if max_velocity <= 0:
        return velocity_np

    clipped = velocity_np.copy()
    velocity_mask = clipped[:, :, 3] > 0.5

    clipped[:, :, 0] = np.where(
        velocity_mask, np.clip(clipped[:, :, 0], 0.0, max_velocity), 0.0
    )

    vx = clipped[:, :, 1]
    vy = clipped[:, :, 2]
    mag_xy = np.sqrt(vx**2 + vy**2)
    scale = np.minimum(1.0, max_velocity / (mag_xy + 1e-8))
    clipped[:, :, 1] = np.where(velocity_mask, vx * scale, 0.0)
    clipped[:, :, 2] = np.where(velocity_mask, vy * scale, 0.0)

    return clipped


def get_tiff_np(
    tiff_fname,
    dem_fname=None,
    velocity_fname=None,
    physics_res=None,
    physics_scale=None,
    add_ndvi=False,
    add_ndwi=False,
    add_ndsi=False,
    add_hsv=False,
    velocity_clip_m_per_year=DEFAULT_VELOCITY_CLIP_M_PER_YEAR,
    return_band_names=False,
    return_valid_mask=False,
    verbose=False,
):
    tiff = read_tiff(tiff_fname)
    landsat_raw = np.transpose(tiff.read(), (1, 2, 0)).astype(np.float32)
    landsat_valid = compute_landsat_valid_mask(tiff, landsat_raw)
    tiff_np = np.nan_to_num(landsat_raw)

    band_names = ["B1", "B2", "B3", "B4", "B5", "B6_VCID1", "B6_VCID2", "B7"]

    use_dem = not (dem_fname is None or not dem_fname.exists())
    dem_np = None

    if use_dem:
        dem = read_tiff(dem_fname)
        dem_np = np.transpose(dem.read(), (1, 2, 0)).astype(np.float32)
        dem_np = np.nan_to_num(dem_np)
        dem_np = compute_dems(dem_np)
        tiff_np = np.concatenate((tiff_np, dem_np), axis=2)
        band_names.extend(["elevation", "slope_deg"])
    tiff_np = np.nan_to_num(tiff_np.astype(np.float32))

    add_velocity_to_output = velocity_fname is not None
    velocity_np = None

    if add_velocity_to_output:
        if velocity_fname.exists():
            velocity = read_tiff(velocity_fname)

            if _same_raster_grid(velocity, tiff):
                velocity_np = np.transpose(velocity.read(), (1, 2, 0)).astype(np.float32)
            else:
                from rasterio.warp import reproject, Resampling

                destination = np.zeros((4, tiff.height, tiff.width), dtype=np.float32)

                for band_idx in range(4):
                    resampling_method = (
                        Resampling.nearest if band_idx == 3 else Resampling.bilinear
                    )
                    reproject(
                        source=rasterio.band(velocity, band_idx + 1),
                        destination=destination[band_idx : band_idx + 1],
                        src_transform=velocity.transform,
                        src_crs=velocity.crs,
                        dst_transform=tiff.transform,
                        dst_crs=tiff.crs,
                        resampling=resampling_method,
                    )

                velocity_np = np.transpose(destination, (1, 2, 0)).astype(np.float32)

            velocity_np = np.nan_to_num(velocity_np)

            velocity_np[:, :, 3] = np.round(velocity_np[:, :, 3]).clip(0, 1)
            invalid = velocity_np[:, :, 3] == 0
            if np.any(invalid):
                velocity_np[invalid, 0] = 0.0
                velocity_np[invalid, 1] = 0.0
                velocity_np[invalid, 2] = 0.0
            velocity_np = clip_velocity_channels(velocity_np, velocity_clip_m_per_year)

            if verbose:
                log.debug(f"Loaded velocity data: shape={velocity_np.shape}")
        else:
            velocity_np = np.zeros(
                (tiff_np.shape[0], tiff_np.shape[1], 4), dtype=np.float32
            )
            if verbose:
                log.warning(
                    f"Velocity file not found: {velocity_fname.name}. Zero-filling with mask=0."
                )

        tiff_np = np.concatenate((tiff_np, velocity_np), axis=2)
        band_names.extend(["velocity", "velocity_x", "velocity_y", "velocity_mask"])

    if add_ndvi:
        tiff_np = add_index(tiff_np, index1=3, index2=2)
        band_names.append("NDVI")
    if add_ndwi:
        tiff_np = add_index(tiff_np, index1=1, index2=3)
        band_names.append("NDWI")
    if add_ndsi:
        tiff_np = add_index(tiff_np, index1=1, index2=4)
        band_names.append("NDSI")
    if add_hsv:
        rgb_img = tiff_np[:, :, [4, 3, 1]].astype(np.float32, copy=False)
        finite_rgb = rgb_img[np.isfinite(rgb_img)]
        rgb_scale = 255.0 if finite_rgb.size and np.nanpercentile(finite_rgb, 99) > 2.0 else 1.0
        rgb_img = np.clip(rgb_img / rgb_scale, 0.0, 1.0)
        hsv_img = rgb2hsv(rgb_img[:, :, [2, 1, 0]])
        tiff_np = np.concatenate((tiff_np, hsv_img), axis=2)
        band_names.extend(["H", "S", "V"])

    use_physics = (
        isinstance(physics_res, (float, str, int))
        and physics_res != "None"
        and isinstance(physics_scale, (float, int))
    )
    if use_physics and use_dem:
        phys_np = physics.compute_phys_v4(
            dem_np[:, :, 0:1],
            physics_res,
            physics_scale,
        )
        tiff_np = np.concatenate((tiff_np, phys_np), axis=2)
        band_names.extend(["flow_accumulation", "tpi", "roughness", "plan_curvature"])

    if verbose:
        log.debug(
            f"use_dem={use_dem}, use_velocity={add_velocity_to_output}, use_physics={use_physics}"
        )
        log.debug(f"Final band order: {band_names}")
        log.debug(f"Final shape: {tiff_np.shape}")

    if return_band_names and return_valid_mask:
        return tiff_np, band_names, landsat_valid
    if return_band_names:
        return tiff_np, band_names
    if return_valid_mask:
        return tiff_np, landsat_valid
    return tiff_np


def save_slices(
    filenum, fname, labels, savepath, save_skipped_visualizations=False, **conf
):
    fname_str = pathlib.Path(fname).name

    tiff_fname = pathlib.Path(conf["image_dir"]) / fname_str
    dem_fname = pathlib.Path(conf["dem_dir"]) / fname_str

    velocity_fname = None
    if conf.get("add_velocity", False) and "velocity_dir" in conf:
        velocity_fname = pathlib.Path(conf["velocity_dir"]) / fname_str

    mask = get_mask(tiff_fname, labels)

    _mask = np.zeros((mask.shape[0], mask.shape[1]))
    for i in range(mask.shape[2]):
        _mask[mask[:, :, i] == 1] = i + 1
    mask = _mask.astype(np.uint8)

    def verify_slice_size(slice, conf):
        if (
            slice.shape[0] != conf["window_size"][0]
            or slice.shape[1] != conf["window_size"][1]
        ):
            if len(slice.shape) == 2:
                temp = np.full(
                    (conf["window_size"][0], conf["window_size"][1]),
                    False if slice.dtype == np.bool_ else 0,
                    dtype=slice.dtype,
                )
                temp[0 : slice.shape[0], 0 : slice.shape[1]] = slice
            else:
                temp = np.full(
                    (conf["window_size"][0], conf["window_size"][1], slice.shape[2]),
                    0,
                    dtype=slice.dtype,
                )
                temp[0 : slice.shape[0], 0 : slice.shape[1], :] = slice

                max_abs = np.max(np.abs(temp))
                if max_abs > 1e6:
                    log.error(
                        f"Detected extreme values in padded slice (max_abs={max_abs:.2e}). "
                        f"This indicates memory corruption. Re-initializing with zeros."
                    )
                    temp.fill(0.0)
                    temp[0 : slice.shape[0], 0 : slice.shape[1], :] = slice
            slice = temp
        return slice

    def filter_percentage(
        slice,
        percentage,
        type="mask",
        name="noname",
        image=None,
        save_visualization=False,
    ):
        if type == "image":
            return True

        valid = slice != IGNORE_LABEL
        num_valid = np.count_nonzero(valid)
        if num_valid == 0:
            return False

        labels = (slice > 0) & valid
        num_labels = np.count_nonzero(labels)
        if num_labels == 0:
            return False

        # Keep all slices with any DCI pixels — they are precious and sparse
        has_dci = np.any(slice == 2)
        if has_dci:
            return True

        frac = num_labels / num_valid

        if frac < percentage:
            if save_visualization:
                ci = np.sum(slice == 1)
                dci = np.sum(slice == 2)
                title = f"CI={ci}, DCI={dci}, labelled={num_labels}, valid={num_valid}, frac={frac:.6f}"

                out_path = pathlib.Path(
                    conf["out_dir"],
                    "skipped_slices",
                    f"{name}_{os.path.basename(savepath)}.png",
                )
                if not out_path.parent.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                plot_image_and_mask(image, slice, title, out_path)

            return False

        return True

    def save_slice(arr, filename):
        np.save(filename, arr)

    def get_pixel_count(mask_slice):
        mask_local = mask_slice.copy()

        ci = np.sum(mask_local == 1)
        deb = np.sum(mask_local == 2)
        mas = np.sum(mask_local == IGNORE_LABEL)
        bg = np.sum(mask_local == 0)

        return bg, ci, deb, mas, mask_local

    os.makedirs(conf["out_dir"], exist_ok=True)

    if filenum == 0:
        result = get_tiff_np(
            tiff_fname,
            dem_fname,
            velocity_fname,
            conf["physics_res"],
            conf["physics_scale"],
            conf["add_ndvi"],
            conf["add_ndwi"],
            conf["add_ndsi"],
            conf["add_hsv"],
            velocity_clip_m_per_year=conf.get(
                "velocity_clip_m_per_year", DEFAULT_VELOCITY_CLIP_M_PER_YEAR
            ),
            return_band_names=True,
            return_valid_mask=True,
            verbose=True,
        )
        tiff_np, band_names, landsat_valid = result
        conf["_band_names"] = band_names
    else:
        tiff_np, landsat_valid = get_tiff_np(
            tiff_fname,
            dem_fname,
            velocity_fname,
            conf["physics_res"],
            conf["physics_scale"],
            conf["add_ndvi"],
            conf["add_ndwi"],
            conf["add_ndsi"],
            conf["add_hsv"],
            velocity_clip_m_per_year=conf.get(
                "velocity_clip_m_per_year", DEFAULT_VELOCITY_CLIP_M_PER_YEAR
            ),
            return_band_names=False,
            return_valid_mask=True,
            verbose=False,
        )

    slicenum = 0
    df_rows = []
    skipped_rows = []
    for row in range(0, tiff_np.shape[0], conf["window_size"][0] - conf["overlap"]):
        for column in range(
            0, tiff_np.shape[1], conf["window_size"][1] - conf["overlap"]
        ):
            mask_slice = mask[
                row : row + conf["window_size"][0],
                column : column + conf["window_size"][1],
            ]
            mask_slice = verify_slice_size(mask_slice, conf)
            mask_slice = mask_slice.copy()

            tiff_slice = tiff_np[
                row : row + conf["window_size"][0],
                column : column + conf["window_size"][1],
                :,
            ]
            tiff_slice = verify_slice_size(tiff_slice, conf)

            landsat_valid_slice = landsat_valid[
                row : row + conf["window_size"][0],
                column : column + conf["window_size"][1],
            ]
            landsat_valid_slice = verify_slice_size(landsat_valid_slice, conf)

            invalid = ~landsat_valid_slice.astype(bool)
            mask_slice[invalid] = IGNORE_LABEL

            rgb_preview = tiff_np[
                row : row + conf["window_size"][0],
                column : column + conf["window_size"][1],
                [2, 1, 0],
            ]

            final_save_slice = np.copy(tiff_slice)
            final_save_slice[invalid] = 0

            bg, ci, deb, mas, modified_mask = get_pixel_count(mask_slice)
            total = bg + ci + deb + mas

            keep = filter_percentage(
                mask_slice,
                conf["filter"],
                type="mask",
                name=f"discarded_{filenum}_slice_{slicenum}",
                image=rgb_preview,
                save_visualization=save_skipped_visualizations,
            )

            if not keep:
                skipped_rows.append(
                    [
                        tiff_fname.name,
                        filenum,
                        slicenum,
                        bg,
                        ci,
                        deb,
                        mas,
                        bg / total if total > 0 else 0.0,
                        ci / total if total > 0 else 0.0,
                        deb / total if total > 0 else 0.0,
                        mas / total if total > 0 else 0.0,
                        os.path.basename(savepath),
                    ]
                )
                slicenum += 1
                continue

            df_rows.append(
                [
                    tiff_fname.name,
                    filenum,
                    slicenum,
                    bg,
                    ci,
                    deb,
                    mas,
                    bg / total if total > 0 else 0.0,
                    ci / total if total > 0 else 0.0,
                    deb / total if total > 0 else 0.0,
                    mas / total if total > 0 else 0.0,
                    os.path.basename(savepath),
                ]
            )

            mask_fname = f"mask_{filenum}_slice_{slicenum}"
            tiff_fname_out = f"tiff_{filenum}_slice_{slicenum}"

            max_abs_val = np.max(np.abs(final_save_slice))
            if max_abs_val > 1e6:
                log.warning(
                    f"CORRUPTION DETECTED in {tiff_fname_out}: max_abs={max_abs_val:.2e}. "
                    f"This slice will be saved but may contain invalid data. "
                    f"Consider regenerating the dataset with single-threaded processing."
                )

            save_slice(modified_mask, savepath / mask_fname)
            save_slice(final_save_slice, savepath / tiff_fname_out)

            slicenum += 1

    band_names = conf.get("_band_names", None)

    return (
        np.mean(tiff_np, axis=(0, 1)),
        np.std(tiff_np, axis=(0, 1)),
        np.min(tiff_np, axis=(0, 1)),
        np.max(tiff_np, axis=(0, 1)),
        df_rows,
        skipped_rows,
        band_names,
    )
