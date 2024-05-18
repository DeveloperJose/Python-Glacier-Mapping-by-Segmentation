"""
This program is used to generate the metrics (Precision, Recall, IoU) given by predicting all test images of a given trained U-Net model.
"""
import pathlib

import numpy as np
import pandas as pd
import torch
import yaml
from addict import Dict
from tqdm import tqdm

import utils
from model.frame import Framework
from model.metrics import IoU, precision, recall, tp_fp_fn


def get_tp_fp_fn(pred, true):
    pred, true = torch.from_numpy(pred), torch.from_numpy(true)
    tp, fp, fn = tp_fp_fn(pred, true)
    return tp, fp, fn


def get_precision_recall_iou(tp, fp, fn):
    p, r, i = precision(tp, fp, fn), recall(tp, fp, fn), IoU(tp, fp, fn)
    return p, r, i


if __name__ == "__main__":
    print("Loading config and preparing")
    conf = Dict(yaml.safe_load(open("./conf/unet_predict.yaml")))

    # % Prediction-specific config
    runs_dir = pathlib.Path(conf.runs_dir)
    run_name: str = conf.run_name
    threshold = conf.threshold

    output_dir = pathlib.Path(conf.output_dir) / run_name
    utils.remove_and_create(output_dir)

    # % Load checkpoint using the training config
    checkpoint_path = runs_dir / conf.run_name / "models" / "model_best.pt"
    frame: Framework = Framework.from_checkpoint(
        checkpoint_path, device=int(conf.gpu_rank), testing=True
    )

    # % Extract the useful variables we want from the frame
    data_dir = pathlib.Path(frame.loader_opts.processed_dir)

    # % Prepare dataframe and metrics arrays
    columns = ["tile_name"]
    for class_name in frame.mask_names:
        columns.append(f"{class_name}_precision")
        columns.append(f"{class_name}_recall")
        columns.append(f"{class_name}_IoU")
    df = pd.DataFrame(columns=columns)

    tp_sum = np.zeros(frame.num_classes, dtype=np.float32)
    fp_sum = np.zeros(frame.num_classes, dtype=np.float32)
    fn_sum = np.zeros(frame.num_classes, dtype=np.float32)

    print("Running predictions")
    for idx, x_fname in tqdm(list(enumerate((data_dir / "test").glob("tiff*")))):
        # % Predict
        x = np.load(x_fname)
        y_pred, mask = frame.predict_slice(x, threshold)

        # % Load true label
        y_fname = x_fname.parent / x_fname.name.replace("tiff", "mask")
        assert y_fname.exists()
        y_true = np.load(y_fname)
        y_true[mask] = 0

        # % Compute precision, recall, and IoU
        save_fname = x_fname.parent / x_fname.name.replace("tiff", "pred")
        _row = [save_fname]
        for i in range(frame.num_classes):
            p = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
            t = np.zeros_like(p)

            p[y_pred == i] = 1
            t[y_true == i] = 1

            tp, fp, fnn = get_tp_fp_fn(p, t)
            prec, rec, iou = get_precision_recall_iou(tp, fp, fnn)
            tp_sum[i] += tp
            fp_sum[i] += fp
            fn_sum[i] += fnn
            _row.extend([prec, rec, iou])
        df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)

    # Compute precision, recall, and IoU for all running totals for all classes
    _row = ["Total"]
    for i in range(frame.num_classes):
        class_tp_sum = tp_sum[i]
        class_fp_sum = fp_sum[i]
        class_fn_sum = fn_sum[i]
        prec, rec, iou = get_precision_recall_iou(
            class_tp_sum, class_fp_sum, class_fn_sum
        )
        _row.extend([prec, rec, iou])

    # Save and print results
    df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
    df.to_csv(output_dir / "metadata.csv")
    print(f"{dict(zip(columns, _row))} for conf={conf}")
