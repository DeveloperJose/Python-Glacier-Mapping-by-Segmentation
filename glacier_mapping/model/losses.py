import torch
import torch.nn as nn
import torch.nn.functional as F


class customloss(nn.Module):
    """
    Multi-class compatible custom loss based on the author's formulation.

    Matches the original paper:
      - Dice loss computed only on valid pixels
      - For binary models: sigmoid + dice only on class 1
      - BF1 boundary loss identical to original
      - Uses target_int==255 as ignore mask (your dataset format)
      - Returns [dice_loss_scalar, boundary_loss_scalar]

    Expected forward signature:
        forward(pred, target, target_int)
    """

    def __init__(
        self,
        act=nn.Softmax(dim=1),
        smooth=1.0,
        label_smoothing=0.0,
        masked=True,
        theta0=3,
        theta=5,
        foreground_classes=None,
        alpha=0.9,  # compatibility with old code
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.theta0 = theta0
        self.theta = theta

        # Default: foreground = class 1 (binary CI or binary DCI)
        self.foreground_classes = (
            foreground_classes if foreground_classes is not None else [1]
        )

        # Framework needs this: number of components for sigma weighting
        self.n_sigma = 2

    def forward(self, pred, target, target_int):
        """
        pred:       (N,C,H,W) raw logits
        target:     (N,C,H,W) one-hot labels
        target_int: (N,H,W)   integer labels, 255 = ignore
        """
        n, c, h, w = pred.shape
        device = pred.device

        target = target.detach()

        # ----------------------------------------------------------
        # IGNORE MASK: True for valid pixels
        # ----------------------------------------------------------
        if target_int is not None:
            ignore_mask = target_int != 255
        else:
            # Fallback: sum of one-hot = 1
            ignore_mask = target.sum(dim=1) == 1

        ignore_mask_exp = ignore_mask.unsqueeze(1).float()  # (N,1,H,W)

        # ----------------------------------------------------------
        # ACTIVATION
        # ----------------------------------------------------------
        if c == 2:
            # Binary debris or binary clean-ice case → match author's Sigmoid
            pred_prob = torch.sigmoid(pred[:, 1:2])  # (N,1,H,W)
            target_prob = target[:, 1:2]
            C_eff = 1
        else:
            # Multiclass case → Softmax
            pred_prob = self.act(pred)
            target_prob = target
            C_eff = c

        # Mask-out invalid pixels
        pred_prob = pred_prob * ignore_mask_exp
        target_prob = target_prob * ignore_mask_exp

        # ----------------------------------------------------------
        # LABEL SMOOTHING
        # ----------------------------------------------------------
        if self.label_smoothing > 0 and C_eff > 1:
            target_prob = (
                target_prob * (1 - self.label_smoothing) + self.label_smoothing / C_eff
            )

        # ----------------------------------------------------------
        # DICE LOSS (author-style)
        # ----------------------------------------------------------
        pred_flat = pred_prob.permute(0, 2, 3, 1)[ignore_mask]  # (M,C_eff)
        targ_flat = target_prob.permute(0, 2, 3, 1)[ignore_mask]  # (M,C_eff)

        if pred_flat.numel() == 0:
            dice_loss_scalar = torch.tensor(0.0, device=device)
        else:
            numerator = 2 * (pred_flat * targ_flat).sum(dim=0) + self.smooth
            denominator = pred_flat.sum(dim=0) + targ_flat.sum(dim=0) + self.smooth
            dice_per_class = 1 - numerator / denominator  # (C_eff,)

            # Match the author: for binary, only class 1 contributes
            if C_eff == 1:
                # Only one channel (class 1), keep as-is
                dice_loss_scalar = dice_per_class.sum()
            else:
                # Multiclass: only include foreground classes
                class_mask = torch.zeros_like(dice_per_class)
                for fg_idx in self.foreground_classes:
                    if 0 <= fg_idx < dice_per_class.shape[0]:
                        class_mask[fg_idx] = 1.0
                dice_loss_scalar = (dice_per_class * class_mask).sum()

        # ----------------------------------------------------------
        # BOUNDARY LOSS (BF1) — identical to original, but masked
        # ----------------------------------------------------------
        # Compute BF1 on *multichannel* representation
        if C_eff == 1:
            pred_b_in = pred_prob
            targ_b_in = target_prob
        else:
            pred_b_in = pred_prob
            targ_b_in = target_prob

        # Boundary maps
        gt_b = F.max_pool2d(1 - targ_b_in, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - targ_b_in
        )
        pred_b = F.max_pool2d(1 - pred_b_in, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - pred_b_in
        )

        # Extended boundaries
        gt_b_ext = F.max_pool2d(gt_b, self.theta, 1, (self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, 1, (self.theta - 1) // 2)

        # Apply mask
        gt_b = gt_b * ignore_mask_exp
        pred_b = pred_b * ignore_mask_exp
        gt_b_ext = gt_b_ext * ignore_mask_exp
        pred_b_ext = pred_b_ext * ignore_mask_exp

        # Flatten for BF1
        gt_b = gt_b.view(n, C_eff, -1)
        pred_b = pred_b.view(n, C_eff, -1)
        gt_b_ext = gt_b_ext.view(n, C_eff, -1)
        pred_b_ext = pred_b_ext.view(n, C_eff, -1)

        P = (pred_b * gt_b_ext).sum(dim=2) / (pred_b.sum(dim=2) + 1e-7)
        R = (pred_b_ext * gt_b).sum(dim=2) / (gt_b.sum(dim=2) + 1e-7)

        BF1 = 2 * P * R / (P + R + 1e-7)
        boundary_loss = torch.mean(1 - BF1)

        return [dice_loss_scalar, boundary_loss]
