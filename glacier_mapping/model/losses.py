import torch
import torch.nn as nn
import torch.nn.functional as F


class customloss(nn.Module):
    """
    Multi-class compatible custom loss that mimics the original paper setup:

      - Dice loss computed per class on valid pixels only
      - Only foreground classes contribute (e.g., CleanIce)
      - Boundary loss (BF1) as in original implementation
      - Returns [dice_loss_scalar, boundary_loss_scalar]
      - n_sigma = 2 so the Framework can apply sigma weighting

    Expected forward signature:
        forward(pred, target, target_int)

        pred       : (N,C,H,W) logits
        target     : (N,C,H,W) one-hot (from dataloader)
        target_int : (N,H,W)   integer labels with 255 = ignore
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
        alpha=0.9,  # compatibility, unused in this refactor
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.theta0 = theta0
        self.theta = theta
        # which class indices count as "foreground" for the dice term
        # e.g., [1] for binary (NOT~CI, CI)
        self.foreground_classes = foreground_classes if foreground_classes is not None else [1]
        # number of separate loss components for sigma weighting
        self.n_sigma = 2

    def forward(self, pred, target, target_int):
        """
        pred:       (N,C,H,W) logits
        target:     (N,C,H,W) one-hot labels
        target_int: (N,H,W)   int labels, where 255=ignore
        """
        n, c, h, w = pred.shape
        device = pred.device

        # Softmax → probabilities
        pred = self.act(pred)
        target = target.detach()

        # ----------------------------------------------------------
        # IGNORE mask (255) or "masked" semantics
        # ----------------------------------------------------------
        if target_int is not None:
            ignore_mask = (target_int != 255)  # True = valid
        else:
            # fallback: consider pixels where sum(target)==1 as valid
            ignore_mask = (target.sum(dim=1) == 1)

        ignore_mask_exp = ignore_mask.unsqueeze(1).float()  # (N,1,H,W)

        # Zero-out ignore regions in both pred + target
        pred = pred * ignore_mask_exp
        target = target * ignore_mask_exp

        # ----------------------------------------------------------
        # Label smoothing (optional)
        # ----------------------------------------------------------
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / c

        # ----------------------------------------------------------
        # DICE LOSS (per class, only foreground classes counted)
        # ----------------------------------------------------------
        # Flatten over valid pixels only
        pred_p = pred.permute(0, 2, 3, 1)   # (N,H,W,C)
        targ_p = target.permute(0, 2, 3, 1)

        valid = ignore_mask  # (N,H,W)
        pred_sel = pred_p[valid]    # (M,C)
        targ_sel = targ_p[valid]    # (M,C)

        if pred_sel.numel() == 0:
            dice_loss_scalar = torch.tensor(0.0, device=device)
        else:
            numerator = 2.0 * (pred_sel * targ_sel).sum(dim=0) + self.smooth
            denominator = pred_sel.sum(dim=0) + targ_sel.sum(dim=0) + self.smooth
            dice_per_class = 1.0 - numerator / denominator  # (C,)

            # Only foreground classes contribute (e.g., CleanIce)
            class_mask = torch.zeros_like(dice_per_class)
            for fg_idx in self.foreground_classes:
                if 0 <= fg_idx < dice_per_class.shape[0]:
                    class_mask[fg_idx] = 1.0

            dice_loss_scalar = (dice_per_class * class_mask).sum()

        # ----------------------------------------------------------
        # BOUNDARY LOSS (BF1), similar to original code
        # ----------------------------------------------------------
        # Original: boundaries from 1 - target / 1 - pred
        gt_b = F.max_pool2d(
            1 - target,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        ) - (1 - target)

        pred_b = F.max_pool2d(
            1 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        ) - (1 - pred)

        gt_b_ext = F.max_pool2d(
            gt_b,
            kernel_size=self.theta,
            stride=1,
            padding=(self.theta - 1) // 2,
        )
        pred_b_ext = F.max_pool2d(
            pred_b,
            kernel_size=self.theta,
            stride=1,
            padding=(self.theta - 1) // 2,
        )

        gt_b = gt_b * ignore_mask_exp
        pred_b = pred_b * ignore_mask_exp
        gt_b_ext = gt_b_ext * ignore_mask_exp
        pred_b_ext = pred_b_ext * ignore_mask_exp

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        BF1 = 2 * P * R / (P + R + 1e-7)
        boundary_loss = torch.mean(1 - BF1)

        return [dice_loss_scalar, boundary_loss]

