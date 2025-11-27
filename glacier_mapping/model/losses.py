#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import gaussian
from torchvision.ops import sigmoid_focal_loss
from numba import jit, njit
from scipy.ndimage import gaussian_filter

# import skimage.morphology as morph
import cv2


class diceloss(torch.nn.Module):
    def __init__(
        self,
        act=torch.nn.Sigmoid(),
        smooth=1.0,
        outchannels=1,
        label_smoothing=0,
        masked=False,
        boundary=0.5,
        gaussian_blur_sigma=None,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.boundary_weight = boundary

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool
            )

        if self.gaussian_blur_sigma != "None":
            _target = np.zeros_like(target.cpu())
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    _target[i, j, :, :] = gaussian(
                        target[i, j, :, :].cpu(), self.gaussian_blur_sigma
                    )
            target = torch.from_numpy(_target).to(mask.device)

        target = (
            target * (1 - self.label_smoothing)
            + self.label_smoothing / self.outchannels
        )

        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        dice = 1 - (
            (2.0 * (pred * target)[mask].sum(dim=0) + self.smooth)
            / (pred[mask].sum(dim=0) + target[mask].sum(dim=0) + self.smooth)
        )

        dice = dice * torch.tensor([0.0, 1.0]).to(dice.device)

        return dice.sum()


class boundaryloss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, one_hot_gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, C, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2
        )
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class iouloss(torch.nn.Module):
    def __init__(
        self,
        act=torch.nn.Sigmoid(),
        smooth=1.0,
        outchannels=1,
        label_smoothing=0,
        masked=False,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool
            )
        target = (
            target * (1 - self.label_smoothing)
            + self.label_smoothing / self.outchannels
        )

        pred = self.act(pred).permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou = 1 - ((intersection + self.smooth) / (union + self.smooth))

        return iou


class celoss(torch.nn.Module):
    def __init__(
        self,
        act=torch.nn.Sigmoid(),
        smooth=1.0,
        outchannels=1,
        label_smoothing=0,
        masked=False,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(reduction="none")(
            pred, torch.argmax(target, dim=1).long()
        )
        return ce


class nllloss(torch.nn.Module):
    def __init__(
        self,
        act=torch.nn.Sigmoid(),
        smooth=1.0,
        outchannels=1,
        label_smoothing=0,
        masked=False,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        # if self.masked:
        #     mask = torch.sum(target, dim=1) == 1
        # else:
        #     mask = torch.ones(
        #         (target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool
        #     )
        target = (
            target * (1 - self.label_smoothing)
            + self.label_smoothing / self.outchannels
        )

        pred = self.act(pred)
        nll = torch.nn.NLLLoss(weight=self.w.to(device=pred.device))(
            pred, torch.argmax(target, dim=1).long()
        )
        return nll


class focalloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        focal_loss = sigmoid_focal_loss(
            pred, target, alpha=-1, gamma=3, reduction="mean"
        )
        return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        act=torch.nn.Sigmoid(),
        smooth=1.0,
        outchannels=1,
        label_smoothing=0,
        masked=True,
        theta0=3,
        theta=5,
        use_unified=False,
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.theta0 = theta0
        self.theta = theta
        self.use_unified = use_unified
        self.unified_loss_fn = unifiedloss(
            act=act, label_smoothing=label_smoothing, outchannels=outchannels
        )

        if self.outchannels == 2:
            print("customloss using masked dice")

        self.n_sigma = 3 if self.use_unified else 2
        #self.n_sigma = 2

    def forward(self, pred, target):
        if self.use_unified:
            asymmetric_ftl, asymmetric_fl = self.unified_loss_fn(pred.detach().clone(), target.detach().clone())

        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones(
                (target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool
            )

        # target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        # print(pred.shape, target.shape, 'shapes')
        n, c, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = self.act(pred)
        # boundary map
        gt_b = F.max_pool2d(
            1 - target,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        gt_b -= 1 - target
        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2
        )
        pred_b -= 1 - pred
        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )
        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2
        )
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)
        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        # summing BF1 Score for each class and average over mini-batch
        boundaryloss = torch.mean(1 - BF1)

        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        diceloss = 1 - (
            (2.0 * (pred * target)[mask].sum(dim=0) + self.smooth)
            / (pred[mask].sum(dim=0) + target[mask].sum(dim=0) + self.smooth)
        )
        # Only used masked dice loss when doing binary models
        if self.outchannels == 2:
            diceloss = diceloss * torch.tensor([0.0, 1.0]).to(diceloss.device)
        elif self.outchannels == 3:
            diceloss = diceloss * torch.tensor([0.0, 1.0, 1.0]).to(diceloss.device)

        if self.use_unified:
            return [asymmetric_fl, asymmetric_ftl, boundaryloss]
        else:
            return [diceloss.sum(), boundaryloss]


# https://github.com/mlyg/boundary-uncertainty/blob/main/loss_functions.py
# @njit(parallel=True)
# @torch.jit.script
def compute_res(seg, im_erode, im_dilate, alpha, beta):
    # compute inner border and adjust certainty with alpha parameter
    inner = seg - im_erode
    inner = alpha * inner
    # compute outer border and adjust certainty with beta parameter
    outer = im_dilate - seg
    outer = beta * outer
    # combine adjusted borders together with unadjusted image
    res = inner + outer + im_erode
    return res


accumulated_res = 0
accumulated_res_idx = 0


# Function to calculate boundary uncertainty
def border_uncertainty_sigmoid(seg: torch.Tensor, alpha=0.9, beta=0.9):
    global accumulated_res, accumulated_res_idx
    """
    Parameters
    ----------
    alpha : float, optional
        controls certainty of ground truth inner borders, by default 0.9.
        Higher values more appropriate when over-segmentation is a concern
    beta : float, optional
        controls certainty of ground truth outer borders, by default 0.1
        Higher values more appropriate when under-segmentation is a concern
    """
    # start_time = timer()
    # res = np.zeros_like(seg)
    # check_seg = seg.astype(np.bool)
    # seg = np.squeeze(seg)

    res = torch.zeros_like(seg)
    check_seg = seg.to(torch.bool)
    seg = seg.squeeze()

    if check_seg.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        seg_np = seg.cpu().numpy()
        im_erode = torch.from_numpy(cv2.erode(seg_np, kernel, iterations=1)).to(
            seg.device
        )
        im_dilate = torch.from_numpy(cv2.dilate(seg_np, kernel, iterations=1)).to(
            seg.device
        )
        res = compute_res(seg, im_erode, im_dilate, alpha, beta)
        return res
    else:
        return res


# Enables batch processing of boundary uncertainty
# def border_uncertainty_sigmoid_batch(y_true):
#     return torch.from_numpy(np.array([border_uncertainty_sigmoid(y.cpu().numpy()) for y in y_true]).astype(np.float32))


# https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
# For later https://github.com/LIVIAETS/boundary-loss
class unifiedloss(torch.nn.Module):
    def __init__(
        self,
        weight=0.5,
        delta=0.6,
        gamma=0.5,
        act=torch.nn.Sigmoid(),
        label_smoothing=0.1,
        outchannels=2,
        boundary=True,
    ):
        super().__init__()
        """
        weight: float, optional represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
        delta : float, optional controls weight given to each class, by default 0.6
        gamma : float, optional focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
        """
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.act = act
        self.label_smoothing = label_smoothing
        self.outchannels = outchannels
        self.boundary = boundary

    def forward(self, pred, target):
        if self.boundary:
            global accumulated_res_idx, accumulated_res
            device = target.device
            # Convert batch to np
            accumulated_res_idx = 0
            accumulated_res = 0

            # start_time = timer()
            # target_np = target.permute(0, 2, 3, 1)
            target_np = target
            # print(f'\t\t{timer()-start_time:.5f}s for boundary')

            # start_time = timer()
            # target_np = target_np.cpu().numpy()
            # print(f'\t\t{timer()-start_time:.5f}s for cpu numpy')

            # start_time = timer()
            target = [
                border_uncertainty_sigmoid(y_true).cpu().numpy() for y_true in target_np
            ]
            # print(f'\t\t{timer()-start_time:.5f}s for list comprehension')

            # start_time = timer()
            # target = np.array(target)
            # print(f'\t\t{timer()-start_time:.5f}s for np.array(target)')

            # start_time = timer()
            # target = torch.from_numpy(target).to(device).permute(0, 3, 1, 2)
            target = np.asarray(target)
            target = torch.from_numpy(target)
            target = target.to(device)  # .permute(0, 3, 1, 2)
            # print(target.shape)
            # print(f'\t\t{timer()-start_time:.5f}s for last step')
            # print(f'\t\t\t{accumulated_res/accumulated_res_idx:.5f}s for compute_res')

        # Label Smoothing
        # start_time = timer()
        # target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        # print(f'\t\t{timer()-start_time:.5f}s for smoothing')

        # Masking
        # start_time = timer()
        # breakpoint()
        mask = torch.sum(target_np, dim=1) == 1
        # softmax so that predicted map can be distributed in [0, 1]
        pred = self.act(pred)  # .permute(0,2,3,1)
        # target = target.permute(0,2,3,1)
        pred = pred.permute(0, 2, 3, 1)[mask]
        target = target.permute(0, 2, 3, 1)[mask]
        # print(f'\t\t{timer()-start_time:.5f}s for masking')

        # Losses
        # start_time = timer()
        asymmetric_ftl = self.asymmetric_focal_tversky_loss(pred, target)
        # print(f'\t\t{timer()-start_time:.5f}s for tversky')

        # start_time = timer()
        asymmetric_fl = self.asymmetric_focal_loss(pred, target)
        # print(f'\t\t{timer()-start_time:.5f}s for focal')

        # print(asymmetric_ftl, asymmetric_fl)

        # if self.weight is not None:
        #     return (self.weight * asymmetric_ftl) + ((1 - self.weight) * asymmetric_fl)
        # else:
        #     return asymmetric_ftl + asymmetric_fl
        return asymmetric_ftl, asymmetric_fl

    def asymmetric_focal_tversky_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if len(y_pred) == 0 or len(y_true) == 0:
            raise ValueError("Empty tensor")
        epsilon = 1e-07
        y_pred = y_pred.clip(epsilon, 1.0 - epsilon)

        tp = (y_true * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        dice_class = (tp + epsilon) / (
            tp + self.delta * fn + (1 - self.delta) * fp + epsilon
        )
        back_dice = 1 - dice_class[0]
        fore_dice = (1 - dice_class[1]) * torch.pow(1 - dice_class[1], -self.gamma)
        return torch.stack([back_dice, fore_dice], -1).mean()

    def asymmetric_focal_loss(self, y_pred, y_true):
        if len(y_pred) == 0 or len(y_true) == 0:
            raise ValueError("Empty tensor")
        epsilon = 1e-07
        y_pred = y_pred.clip(epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * torch.log(y_pred)
        back_ce = torch.pow(1 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1]
        fore_ce = self.delta * fore_ce
        return torch.stack([back_ce, fore_ce], -1).sum(-1).mean()
