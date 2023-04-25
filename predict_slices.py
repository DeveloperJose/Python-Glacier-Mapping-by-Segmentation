#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program takes full TIFF images and feeds the correct slices to a trained model to produce complete predictions.

@author: Bibek Aryal, Jose G. Perez
"""
import pathlib
from timeit import default_timer as timer

import rasterio
import rasterio.plot
import yaml
from addict import Dict
from tqdm import tqdm

import segmentation.model.functions as fn
from segmentation.data.slice import get_mask, get_tiff_np, read_shp, read_tiff
from segmentation.model.frame import Framework

if __name__ == "__main__":
    start_time = timer()
    conf = Dict(yaml.safe_load(open('./conf/predict_slices.yaml')))

    # % Prediction-specific config
    runs_dir = pathlib.Path(conf.runs_dir)
    run_name: str = conf.run_name
    physics_scale = conf.physics_scale
    physics_res = conf.physics_res
    gpu_rank: int = conf.gpu_rank
    window_size = conf.window_size
    threshold = conf.threshold
    tiff_path = pathlib.Path(conf.tiff_dir)
    dem_path = pathlib.Path(conf.dem_dir) if conf.dem_dir else None
    labels_path = pathlib.Path(conf.labels_path)

    output_dir = pathlib.Path(conf.output_dir) / conf.run_name / f't={threshold}'
    if not output_dir.exists():
        output_dir.mkdir()

    assert tiff_path.exists()
    assert labels_path.exists()
    if dem_path:
        assert dem_path.exists()

    # % Load checkpoint using the training config
    checkpoint_path = runs_dir / conf.run_name / 'models' / 'model_best.pt'
    frame: Framework = Framework.from_checkpoint(checkpoint_path, device=gpu_rank, testing=True)

    # % Load data stuff
    labels = read_shp(labels_path)

    # % Predictions
    for idx, fname in enumerate(tqdm(list(tiff_path.glob('*.tif')))):
        if idx > 0:
            break
        tiff_fname = tiff_path / fname
        dem_fname = dem_path / fname

        split_df = frame.df[frame.df['Landsat ID'] == fname.name]
        if len(split_df) == 0:
            split = 'ignored'
        else:
            split = split_df.iloc[0]['split']

        assert tiff_fname.exists()
        assert dem_fname.exists()

        label_mask = get_mask(tiff_fname, labels)
        x_arr = get_tiff_np(tiff_fname, dem_fname, physics_res=physics_res, physics_scale=physics_scale, verbose=(idx == 0))
        y_pred, mask = frame.predict_whole(x_arr, window_size, threshold)
        y_true = frame.get_y_true(label_mask, mask)

        with rasterio.Env():
            # Open TIFF file with rasterio
            src = read_tiff(tiff_fname)

            profile = src.profile
            profile['dtype'] = str(y_pred.dtype)
            profile['height'] = y_pred.shape[0]
            profile['width'] = y_pred.shape[1]
            profile['count'] = 2

            with rasterio.open(output_dir / f'{tiff_fname.stem}_{split}.tif', 'w', **profile) as dst:
                dst.write(y_pred, 1)
                dst.write(y_true, 2)

    print(f'Took {timer()-start_time:.2f}s for {conf.run_name}')
