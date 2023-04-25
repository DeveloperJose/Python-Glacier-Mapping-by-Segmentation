import os
import pathlib

import numpy as np
import pandas as pd
import torch
import yaml
from addict import Dict
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm

import segmentation.data.slice as sl
from segmentation.model.frame import Framework
from segmentation.model.metrics import *


def get_tp_fp_fn(pred, true):
    pred, true = torch.from_numpy(pred), torch.from_numpy(true)
    tp, fp, fn = tp_fp_fn(pred, true)
    return tp, fp, fn

def get_precision_recall_iou(tp, fp, fn):
    p, r, i = precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn)
    return p, r, i

if __name__ == "__main__":
    print('Loading config and preparing')
    conf = Dict(yaml.safe_load(open('./conf/unet_predict.yaml')))

    #% Prediction-specific config
    runs_dir = pathlib.Path(conf.runs_dir)
    output_dir = pathlib.Path(conf.output_dir) / conf.run_name
    sl.remove_and_create(output_dir)

    #% Load checkpoint using the training config
    checkpoint_path = runs_dir / conf.run_name / 'models' / 'model_best.pt'
    frame: Framework = Framework.from_checkpoint(checkpoint_path, device=int(conf.gpu_rank), testing=True)

    #% Prepare dataframe
    columns = ["tile_name"]
    for class_name in frame.mask_names:
        columns.append(f'{class_name}_precision')
        columns.append(f'{class_name}_recall')
        columns.append(f'{class_name}_IoU')
    df = pd.DataFrame(columns=columns)

    #% Extract the useful variables we want from the frame and config
    data_dir = pathlib.Path(frame.loader_opts.processed_dir)
    normalize:str = frame.loader_opts.normalize
    use_channels = frame.loader_opts.use_channels
    threshold = frame.metrics_opts.threshold
    binary_class_idx = frame.loader_opts.output_classes[0]
    total_classes = len(frame.loader_opts.class_names)

    #% Load normalization arrays
    arr = np.load(data_dir / "normalize_train.npy")
    if normalize == "mean-std":
        _mean, _std = arr[0], arr[1]
    if normalize == "min-max":
        _min, _max = arr[2], arr[3]

    #% Prepare to iterate over test set
    files = os.listdir(data_dir / "test")
    inputs = [x for x in files if "tiff" in x]

    tp_sum = np.zeros(frame.num_classes, dtype=np.float32)
    fp_sum = np.zeros(frame.num_classes, dtype=np.float32)
    fn_sum = np.zeros(frame.num_classes, dtype=np.float32)
    print('Running predictions')
    for x_fname in tqdm(inputs):
        # Load data
        x = np.load(data_dir / "test" / x_fname)
        mask = np.sum(x, axis=2) == 0

        x = frame.normalize(x)
        
        y_fname = x_fname.replace("tiff", "mask")
        save_fname = x_fname.replace("tiff", "pred")
        y_true = np.load(data_dir / "test" / y_fname) + 1
        if frame.is_binary:
            y_true_b = y_true.copy()
            y_true_b[y_true==binary_class_idx+1] = 2
            y_true_b[y_true!=binary_class_idx+1] = 1
            y_true = y_true_b
        y_true[mask] = 0
        y_true = y_true[~mask]

        # Make prediction
        _x = torch.from_numpy(np.expand_dims(x[:,:,use_channels], axis=0)).float()
        pred = frame.infer(_x)
        pred = torch.nn.Softmax(3)(pred)
        pred = np.squeeze(pred.cpu())
        assert pred.shape[2] == frame.num_classes

        # Threshold + fill holes + add mask to prediction
        y_pred = np.zeros((pred.shape[0], pred.shape[1]))
        for i in range(frame.num_classes):
            _class = pred[:, :, i] >= threshold[i]
            _class = binary_fill_holes(_class)
            y_pred[_class] = i
        y_pred += 1
        y_pred[mask] = 0
        y_pred = y_pred
        y_pred = y_pred[~mask]

        # Compute precision, recall, and IoU for all classes in this slice
        # and keep running totals for all of them
        _row = [save_fname]
        for i in range(frame.num_classes):
            if i == 0 and frame.is_binary:
                p = (y_pred != binary_class_idx+1).astype(np.uint8)
                t = (y_true != binary_class_idx).astype(np.uint8)
            elif frame.is_binary:
                p = (y_pred == 2).astype(np.uint8)
                t = (y_true == 2).astype(np.uint8)
            else:
                p = (y_pred == i+1).astype(np.uint8)
                t = (y_true == i).astype(np.uint8)
            tp, fp, fnn = get_tp_fp_fn(p, t)
            prec, rec, iou = get_precision_recall_iou(tp, fp, fnn)
            tp_sum[i] += tp
            fp_sum[i] += fp
            fn_sum[i] += fnn
            _row.extend([prec, rec, iou])
        df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)

    # Compute precision, recall, and IoU for all running totals for all classes
    _row = ['Total']
    for i in range(frame.num_classes):
        class_tp_sum = tp_sum[i]
        class_fp_sum = fp_sum[i]
        class_fn_sum = fn_sum[i]
        prec, rec, iou = get_precision_recall_iou(class_tp_sum, class_fp_sum, class_fn_sum)
        _row.extend([prec, rec, iou])
    
    # Save and print results
    df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
    df.to_csv(output_dir / "metadata.csv")
    print(f"{dict(zip(columns, _row))}")

    
