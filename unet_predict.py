from segmentation.model.frame import Framework
import segmentation.model.functions as fn

import yaml, pdb, os, pathlib, torch
from addict import Dict
import numpy as np
import pandas as pd
from segmentation.model.metrics import *
import segmentation.data.slice as sl
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm

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

    #% Infer some configs
    conf.model_opts_cleanice.args.inchannels = len(conf.use_channels_cleanice)
    conf.model_opts_cleanice.args.outchannels = len(conf.class_names)
    use_physics = 10 in conf.use_channels_cleanice

    data_dir = pathlib.Path(conf.data_dir)
    preds_dir = pathlib.Path(conf.out_processed_dir) / "preds" / conf.run_name
    columns = ["tile_name"]
    for class_name in conf.class_names:
        columns.append(f'{class_name}_precision')
        columns.append(f'{class_name}_recall')
        columns.append(f'{class_name}_IoU')
    df = pd.DataFrame(columns=columns)

    sl.remove_and_create(preds_dir)

    cleanice_model_path = pathlib.Path(conf.folder_name) / conf.run_name / 'models' / 'model_best.pt'
    # debris_model_path = pathlib.Path(conf.folder_name) / conf.run_name / 'models' / 'model_best.pt'

    loss_fn = fn.get_loss(conf.model_opts_cleanice.args.outchannels)
    cleanice_frame = Framework(
        loss_fn=loss_fn,
        model_opts=conf.model_opts_cleanice,
        optimizer_opts=conf.optim_opts,
        device=(int(conf.gpu_rank))
    )
    # debris_frame = Framework(
    #     loss_fn=loss_fn,
    #     model_opts=conf.model_opts_debris,
    #     optimizer_opts=conf.optim_opts,
    #     device=(int(conf.gpu_rank))
    # )
    if torch.cuda.is_available():
        cleanice_state_dict = torch.load(cleanice_model_path)
        # debris_state_dict = torch.load(debris_model_path)
    else:
        cleanice_state_dict = torch.load(cleanice_model_path, map_location="cpu")
        # debris_state_dict = torch.load(debris_model_path, map_location="cpu")
    cleanice_frame.load_state_dict(cleanice_state_dict)
    # debris_frame.load_state_dict(debris_state_dict)

    arr = np.load(data_dir / "normalize_train.npy")
    if conf.normalize == "mean-std":
        _mean, _std = arr[0], arr[1]
    if conf.normalize == "min-max":
        _min, _max = arr[2], arr[3]

    files = os.listdir(data_dir / "test")
    inputs = [x for x in files if "tiff" in x]

    bg_tp_sum, bg_fp_sum, bg_fn_sum = 0, 0, 0
    ci_tp_sum, ci_fp_sum, ci_fn_sum = 0, 0, 0
    debris_tp_sum, debris_fp_sum, debris_fn_sum = 0, 0, 0

    print('Running predictions')
    for x_fname in tqdm(inputs):
        x = np.load(data_dir / "test" / x_fname)
        mask = np.sum(x, axis=2) == 0
        if conf.normalize == "mean-std":
            if use_physics:
                x[:, :, :-1] = (x[:, :, :-1] - _mean[:-1]) / _std[:-1]
            else:
                x = (x - _mean) / _std
        if conf.normalize == "min-max":
            x = (x - _min) / (_max - _min)
        
        y_fname = x_fname.replace("tiff", "mask")
        save_fname = x_fname.replace("tiff", "pred")
        y_cleanice_true = np.load(data_dir / "test" / y_fname) + 1
        # y_debris_true = np.load(data_dir / "test" / y_fname) + 1
        y_cleanice_true = y_cleanice_true[~mask]

        # y_cleanice_true, y_debris_true = y_cleanice_true[~mask], y_debris_true[~mask]

        _x = torch.from_numpy(np.expand_dims(x[:,:,conf.use_channels_cleanice], axis=0)).float()
        pred_cleanice = cleanice_frame.infer(_x)
        pred_cleanice = torch.nn.Softmax(3)(pred_cleanice)
        pred_cleanice = np.squeeze(pred_cleanice.cpu())

        # _x = torch.from_numpy(np.expand_dims(x[:,:,conf.use_channels_debris], axis=0)).float()
        # pred_debris = debris_frame.infer(_x)
        # pred_debris = torch.nn.Softmax(3)(pred_debris)
        # pred_debris = np.squeeze(pred_debris.cpu())

        _pred = np.zeros((pred_cleanice.shape[0], pred_cleanice.shape[1]))

        _bg = pred_cleanice[:, :, 0] >= conf.threshold[0]
        _bg = binary_fill_holes(_bg)
        _ci = pred_cleanice[:, :, 1] >= conf.threshold[1]
        _ci = binary_fill_holes(_ci)
        _debris = pred_cleanice[:, :, 2] >= conf.threshold[2]
        _debris = binary_fill_holes(_debris)

        _pred[_bg] = 1
        _pred[_ci] = 2
        _pred[_debris] = 3
        _pred = _pred+1
        _pred[mask] = 0

        y_pred = _pred
        y_pred = y_pred[~mask]

        y_pred_prob = np.zeros((pred_cleanice.shape[0], pred_cleanice.shape[1], 4))
        y_pred_prob[:,:,1] = pred_cleanice[:, :, 0]
        y_pred_prob[:,:,2] = pred_cleanice[:, :, 1]
        y_pred_prob[:,:,3] = pred_cleanice[:, :, 2]

        # For pixels where DCG(:,:,2) and CIG (:, :, 1) overlap the output is set as DCG?
        y_pred_prob[:,:,0] = np.min(np.concatenate((pred_cleanice[:, :, 1][:,:, None], pred_cleanice[:, :, 2][:,:,None]), axis=2), axis=2)
        y_pred_prob[mask] = 0

        bg_pred, ci_pred, debris_pred = (y_pred == 2).astype(np.int8), (y_pred == 3).astype(np.int8), (y_pred == 4).astype(np.int8)
        bg_true, ci_true, debris_true = (y_cleanice_true == 1).astype(np.int8), (y_cleanice_true == 2).astype(np.int8), (y_cleanice_true == 3).astype(np.int8)

        bg_tp, bg_fp, bg_fn = get_tp_fp_fn(bg_pred, bg_true)
        ci_tp, ci_fp, ci_fn = get_tp_fp_fn(ci_pred, ci_true)
        debris_tp, debris_fp, debris_fn = get_tp_fp_fn(debris_pred, debris_true)

        bg_precision, bg_recall, bg_iou = get_precision_recall_iou(bg_tp, bg_fp, bg_fn)
        ci_precision, ci_recall, ci_iou = get_precision_recall_iou(ci_tp, ci_fp, ci_fn)
        debris_precision, debris_recall, debris_iou = get_precision_recall_iou(debris_tp, debris_fp, debris_fn)

        _row = [save_fname, bg_precision, bg_recall, bg_iou, ci_precision, ci_recall, ci_iou, debris_precision, debris_recall, debris_iou]
        df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
        np.save(preds_dir / save_fname, y_pred_prob)
        bg_tp_sum += bg_tp
        bg_fp_sum += bg_fp
        bg_fn_sum += bg_fn
        ci_tp_sum += ci_tp
        ci_fp_sum += ci_fp
        ci_fn_sum += ci_fn
        debris_tp_sum += debris_tp
        debris_fp_sum += debris_fp
        debris_fn_sum += debris_fn

    bg_precision, bg_recall, bg_iou = get_precision_recall_iou(bg_tp_sum, bg_fp_sum, bg_fn_sum)
    ci_precision, ci_recall, ci_iou = get_precision_recall_iou(ci_tp_sum, ci_fp_sum, ci_fn_sum)
    debris_precision, debris_recall, debris_iou = get_precision_recall_iou(debris_tp_sum, debris_fp_sum, debris_fn_sum)
    _row = ["Total", bg_precision, bg_recall, bg_iou, ci_precision, ci_recall, ci_iou, debris_precision, debris_recall, debris_iou]
    df = df.append(pd.DataFrame([_row], columns=columns), ignore_index=True)
    
    print(f"{dict(zip(columns, _row))}")
    df.to_csv(preds_dir / "metadata.csv")

    