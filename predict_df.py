import multiprocessing
import pathlib
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
import seaborn_image as isns
import torch
import yaml
from addict import Dict
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2hsv
from tqdm import tqdm

import segmentation.model.functions as fn
from segmentation.data.slice import get_mask, get_tiff_np, read_shp, read_tiff
from segmentation.model.frame import Framework
from segmentation.model.metrics import *
from utils import istarmap

# TODO: Update code that gives FutureWarning and DeprecationWarning
import warnings
warnings.filterwarnings("ignore")

def get_tp_fp_fn(pred, true):
    pred, true = torch.from_numpy(pred), torch.from_numpy(true)
    tp, fp, fn = tp_fp_fn(pred, true)
    return tp, fp, fn


def get_precision_recall_iou(tp, fp, fn):
    p, r, i = precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn)
    return p, r, i


if __name__ == '__main__':
    conf = Dict(yaml.safe_load(open('./conf/predict_slices.yaml')))

    # % Prediction-specific config
    runs_dir = pathlib.Path(conf.runs_dir)
    output_dir = pathlib.Path(conf.output_dir)

    # % Load checkpoint using the training config
    checkpoint_path = runs_dir / conf.run_name / 'models' / 'model_best.pt'
    frame: Framework = Framework.from_checkpoint(checkpoint_path, device='cpu', testing=True)

    # % Load data stuff
    tiff_path = pathlib.Path('/home/jperez/data/HKH_raw/Landsat7_2005/')
    dem_path = pathlib.Path('/home/jperez/data/HKH_raw/DEM/')
    labels_fname = pathlib.Path('/home/jperez/data/HKH_raw/labels/HKH_CIDC_5basins_all.shp')
    labels = read_shp(labels_fname)

    # % Extract the useful variables we want from the frame and config
    data_dir = pathlib.Path(frame.loader_opts.processed_dir)

    D_split = {}
    pred_dir = pathlib.Path('pred_runs/images/multi_phys64_s1/t=0.99')
    all_pred_fnames = list(pred_dir.glob('*.tif'))
    for pred_fname in all_pred_fnames:
        s = pred_fname.name.split('_')
        im_fname = s[0]
        split = s[1][:-4]
        D_split[im_fname] = split

    print(D_split)

    columns = ['fname', 'total_n', 
               'bg_n', 'bg_n%', 'bg_prec', 'bg_rec', 'bg_iou',
               'ci_n', 'ci_n%', 'ci_prec', 'ci_rec', 'ci_iou',
               'dci_n', 'dci_n%', 'dci_prec', 'dci_rec', 'dci_iou',
               ]
    df = pd.DataFrame(columns=columns)

    def process(idx, fname):
        global tiff_path, dem_path, labels

        tiff_fname = tiff_path / fname.name
        split = D_split[fname.name[:-4]]
        pred_fname = pred_dir / f'{fname.stem}_{split}.tif'
        dem_fname = dem_path / fname
        assert tiff_fname.exists()
        assert pred_fname.exists()
        assert dem_fname.exists()
        im_tiff = np.transpose(read_tiff(tiff_fname).read(), (1, 2, 0)).astype(np.uint8)
        pred_tiff = np.transpose(read_tiff(pred_fname).read(), (1, 2, 0)).astype(np.uint8)
        mask = np.sum(im_tiff[:, :, :3], axis=2) < 0.01

        label_mask = get_mask(tiff_fname, labels)
        y_true = frame.get_y_true(label_mask, mask)

        total_pixels = pred_tiff.shape[0] * pred_tiff.shape[1]
        row = [fname.stem, total_pixels]
        for i in range(frame.num_classes):
            p = np.zeros((pred_tiff.shape[0], pred_tiff.shape[1]), dtype=np.uint8)
            t = np.zeros_like(p)

            p[pred_tiff[:, :, 0] == i] = 1
            t[y_true == i] = 1

            class_total = np.sum(p)
            class_percent = class_total / total_pixels

            tp, fp, fnn = get_tp_fp_fn(p, t)
            prec, rec, iou = get_precision_recall_iou(tp, fp, fnn)
            row.extend([class_total, class_percent, prec, rec, iou])
        return row

    data = list(enumerate(tiff_path.glob('*.tif')))
    with tqdm(total=len(data), desc='Running Predictions') as pbar:
        with multiprocessing.Pool(32) as pool:
            for result in istarmap(pool, process, data):
                df.loc[len(df.index)] = result
                pbar.update(1)

    df.to_csv('pred.csv')
    print(df)
