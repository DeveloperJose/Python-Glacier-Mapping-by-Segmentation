#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program takes full TIFF images, feeds the slices to a trained model to produce complete predictions, and saves the prediction and true labels as a TIFF file.

Takes around 10min using 32 CPUs on our server.
"""

from functools import partial
import multiprocessing
import pathlib
from timeit import default_timer as timer

import rasterio
import rasterio.plot
import yaml
from addict import Dict
from tqdm import tqdm

from glacier_mapping.data.slice import get_mask, get_tiff_np, read_shp, read_tiff
from glacier_mapping.model.frame import Framework
import glacier_mapping.utils as utils

# TODO: Update code that gives FutureWarning and DeprecationWarning
import warnings

warnings.filterwarnings("ignore")


def process_data(
    param,
    frame,
    physics_res,
    physics_scale,
    labels,
    window_size,
    threshold,
    tiff_dir,
    dem_dir,
    output_dir,
):
    idx, fname = param
    tiff_fname = tiff_dir / fname
    dem_fname = dem_dir / fname

    split_df = frame.df[frame.df["Landsat ID"] == fname.name]
    if len(split_df) == 0:
        split = "ignored"
    else:
        split = split_df.iloc[0]["split"]

    assert tiff_fname.exists()
    assert dem_fname.exists()

    label_mask = get_mask(tiff_fname, labels)
    x_arr = get_tiff_np(
        tiff_fname,
        dem_fname,
        physics_res=physics_res,
        physics_scale=physics_scale,
        verbose=(idx == 0),
    )
    y_pred, mask = frame.predict_whole(x_arr, window_size, threshold)
    y_true = frame.get_y_true(label_mask, mask)

    with rasterio.Env():
        # Open TIFF file with rasterio
        src = read_tiff(tiff_fname)

        profile = src.profile
        profile["dtype"] = str(y_pred.dtype)
        profile["height"] = y_pred.shape[0]
        profile["width"] = y_pred.shape[1]
        profile["blockxsize"] = 1024
        profile["blockysize"] = 1024
        profile["count"] = 2

        with rasterio.open(
            output_dir / f"{tiff_fname.stem}_{split}.tif", "w", **profile
        ) as dst:
            dst.write(y_pred, 1)
            dst.write(y_true, 2)


if __name__ == "__main__":
    start_time = timer()
    conf = Dict(yaml.safe_load(open("./conf/predict_slices.yaml")))

    # % Prediction-specific config
    runs_dir = pathlib.Path(conf.runs_dir)
    run_name: str = conf.run_name
    gpu_rank: int = conf.gpu_rank
    window_size = conf.window_size
    threshold = conf.threshold
    tiff_dir = pathlib.Path(conf.tiff_dir)
    dem_dir = pathlib.Path(conf.dem_dir)
    labels_path = pathlib.Path(conf.labels_path)

    output_dir: pathlib.Path = (
        pathlib.Path(conf.output_dir) / conf.run_name / f"t={threshold}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    assert tiff_dir.exists()
    assert labels_path.exists()
    if not dem_dir.exists():
        print(
            f"dem_dir provided {dem_dir} does not exist, assuming this model does not use DEM"
        )

    # % Load checkpoint using the training config
    checkpoint_path = runs_dir / conf.run_name / "models" / "model_best.pt"
    frame: Framework = Framework.from_checkpoint(
        checkpoint_path, device=gpu_rank, testing=True
    )
    physics_res, physics_scale = utils.get_physics_from_run_name(run_name)

    # % Load data stuff
    labels = read_shp(labels_path)

    # % Predictions
    data = list(enumerate(tiff_dir.glob("*.tif")))

    # Set multiprocessing start method for CUDA to work
    multiprocessing.set_start_method("spawn")
    process_fn = partial(
        process_data,
        frame=frame,
        physics_res=physics_res,
        physics_scale=physics_scale,
        labels=labels,
        window_size=window_size,
        threshold=threshold,
        tiff_dir=tiff_dir,
        dem_dir=dem_dir,
        output_dir=output_dir,
    )
    with multiprocessing.Pool(10) as pool:
        for _ in tqdm(pool.imap_unordered(process_fn, data), total=len(data)):
            pass

    print(f"Took {timer() - start_time:.2f}s for {conf.run_name}")
