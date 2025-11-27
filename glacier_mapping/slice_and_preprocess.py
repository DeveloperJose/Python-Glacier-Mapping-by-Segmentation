#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program can take raw hyperspectral data in TIFF form, a label shapefile, split the data and then create and the arrays needed to train a neural network.
"""

import multiprocessing
import os
import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from addict import Dict
from tqdm import tqdm

import glacier_mapping.data.slice as fn
import glacier_mapping.utils as utils

import matplotlib
matplotlib.use("Agg")   # ensures no GUI backend is used

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    conf = Dict(yaml.safe_load(open("./conf/slice_and_preprocess.yaml")))

    saved_df = pd.DataFrame(
        columns=[
            "Landsat ID",
            "Image",
            "Slice",
            "Background",
            "Clean Ice",
            "Debris",
            "Masked",
            "Background Percentage",
            "Clean Ice Percentage",
            "Debris Percentage",
            "Masked Percentage",
            "split",
        ]
    )

    images = sorted(os.listdir(Path(conf.image_dir)))
    idx = np.random.permutation(len(images))
    splits = {
        "test": sorted([images[i] for i in idx[: int(conf.test * len(images))]]),
        "val": sorted(
            [
                images[i]
                for i in idx[
                    int(conf.test * len(images)) : int(
                        (conf.test + conf.val) * len(images)
                    )
                ]
            ]
        ),
        "train": sorted(
            [images[i] for i in idx[int((conf.test + conf.val) * len(images)) :]]
        ),
    }
    print("Splits:", splits)
    # breakpoint()
    labels = fn.read_shp(Path(conf.labels_dir) / "HKH_CIDC_5basins_all.shp")
    utils.remove_and_create(conf.out_dir)

    with tqdm(total=1, desc="temp") as pbar:
        # def process(i, fname):
        #     global conf, pbar
        #     mean, std, _min, _max, df_rows = fn.save_slices(
        #         i, fname, labels, savepath, pbar, **conf
        #     )
        #     return mean, std, _min, _max, df_rows

        for split, meta in splits.items():
            means, stds, mins, maxs = [], [], [], []
            savepath = Path(conf["out_dir"]) / split
            fn_process = partial(
                fn.save_slices, labels=labels, savepath=savepath, **conf
            )
            utils.remove_and_create(savepath)

            pbar.set_description(f"Processing dataset {split}")
            pbar.reset(len(meta))
            with multiprocessing.Pool(10) as pool:
                for result in utils.istarmap(pool, fn_process, enumerate(meta)):
                    mu, s, mi, ma, df_rows = result
                    means.append(mu)
                    stds.append(s)
                    mins.append(mi)
                    maxs.append(ma)
                    for row in df_rows:
                        saved_df.loc[len(saved_df.index)] = row
                    pbar.update(1)
                    np.save(
                        Path(conf["out_dir"]) / f"normalize_{split}",
                        np.asarray((mu, s, mi, ma)),
                    )

    means = np.mean(np.asarray(means), axis=0)
    stds = np.mean(np.asarray(stds), axis=0)
    mins = np.min(np.asarray(mins), axis=0)
    maxs = np.max(np.asarray(maxs), axis=0)

    saved_df.to_csv(
        Path(conf["out_dir"]) / "slice_meta.csv", encoding="utf-8", index=False
    )

    norm_output_path = Path(conf["out_dir"]) / "normalize_all.npy"
    np.save(norm_output_path, np.asarray((means, stds, mins, maxs)))
    print(f"Saving slices to {conf.out_dir} completed!!!")
