import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def make_onehot(
    y_int: torch.Tensor,
    num_classes: int,
    output_classes: List[int],
    device: torch.device,
    binary_non_target_policy: str = "background",
) -> torch.Tensor:
    valid = y_int != 255
    if len(output_classes) == 1:
        binary_class = int(output_classes[0])
        if binary_non_target_policy == "background":
            background = (y_int != binary_class) & valid
        elif binary_non_target_policy == "ignore":
            background = (y_int == 0) & valid
        else:
            raise ValueError(
                "binary_non_target_policy must be 'background' or 'ignore', got "
                f"{binary_non_target_policy!r}"
            )
        target = torch.stack(
            (
                background,
                (y_int == binary_class) & valid,
            ),
            dim=1,
        )
    else:
        target = torch.stack(
            [(y_int == int(class_idx)) & valid for class_idx in output_classes],
            dim=1,
        )
    if target.shape[1] != num_classes:
        raise ValueError(
            f"Target has {target.shape[1]} channels but model has {num_classes} "
            "output channels"
        )
    return target.to(dtype=torch.float32, device=device)


class customloss(nn.Module):
    def __init__(
        self,
        act=nn.Softmax(dim=1),
        smooth=1.0,
        label_smoothing=0.0,
        theta0=3,
        theta=5,
        verbose=False,
        class_weights: Optional[List[float]] = None,
        velocity_high_speed_threshold: float = 3.16,
        velocity_loss_weight: float = 0.05,
        velocity_loss_warmup_epochs: int = 10,
        velocity_loss_ramp_epochs: int = 10,
        output_classes: Optional[List[int]] = None,
        behavior: str = "modern",
        binary_non_target_policy: str = "background",
    ):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.theta0 = theta0
        self.theta = theta
        self.verbose = verbose
        self.class_weights = class_weights
        class_weights_tensor = (
            torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.register_buffer("class_weights_tensor", class_weights_tensor)
        self.velocity_high_speed_threshold = velocity_high_speed_threshold
        self.velocity_loss_weight = velocity_loss_weight
        self.velocity_loss_warmup_epochs = velocity_loss_warmup_epochs
        self.velocity_loss_ramp_epochs = velocity_loss_ramp_epochs
        if output_classes is None:
            output_classes = [0, 1, 2]
        self.output_classes = output_classes
        self.behavior = behavior
        self.binary_non_target_policy = binary_non_target_policy
        if self.behavior not in {"modern", "aryal_2023"}:
            raise ValueError(f"Unsupported loss behavior: {self.behavior!r}")
        if self.behavior == "aryal_2023" and class_weights not in (None, [0, 1]):
            raise ValueError("Aryal loss requires foreground-only class weights [0, 1]")

        self.last_velocity_loss: Optional[torch.Tensor] = None
        self.last_velocity_base: Optional[torch.Tensor] = None
        self.last_velocity_weight: float = 0.0
        self.last_velocity_valid: bool = False

    def forward(
        self,
        pred,
        target_int,
        velocity=None,
        velocity_mask=None,
        current_epoch: Optional[int] = None,
    ):
        n, c, h, w = pred.shape

        zero = pred.new_zeros(())
        self.last_velocity_loss = zero
        self.last_velocity_base = zero
        self.last_velocity_weight = 0.0
        self.last_velocity_valid = False

        if target_int is None:
            raise ValueError("target_int is required")
        y_int = target_int.long()
        ignore_mask = y_int != 255
        ignore_mask_exp = ignore_mask.unsqueeze(1).float().to(pred.device)
        target = make_onehot(
            y_int,
            c,
            self.output_classes,
            pred.device,
            binary_non_target_policy=self.binary_non_target_policy,
        ).detach()

        if c <= 1:
            raise ValueError(
                f"Invalid channel count: {c}. Binary requires 2 channels, multi-class requires 3+ channels."
            )

        if c == 2:
            pred_prob = torch.softmax(pred, dim=1)
            target_prob = target
            C_eff = 2
        else:
            pred_prob = self.act(pred)
            target_prob = target
            C_eff = c

        if self.behavior == "aryal_2023":
            return self._forward_aryal_2023(pred_prob, target)

        pred_prob = pred_prob * ignore_mask_exp
        if self.label_smoothing > 0 and C_eff > 1:
            target_prob = (
                target_prob * (1 - self.label_smoothing) + self.label_smoothing / C_eff
            )
        target_prob = target_prob * ignore_mask_exp

        weights_tensor = self.class_weights_tensor
        if weights_tensor is not None:
            if weights_tensor.numel() != C_eff:
                raise ValueError(
                    f"Length of class_weights ({weights_tensor.numel()}) "
                    f"must match number of effective classes ({C_eff})"
                )
            weights_tensor = weights_tensor.to(dtype=pred_prob.dtype)

        # Broadcast-mask Dice: avoids permute()[ignore_mask] advanced indexing
        # which creates dynamic compacted tensors and unnecessary memory movement.
        # Uses element-wise multiply + sum over (batch, height, width).
        weighted_pred = pred_prob * ignore_mask_exp
        weighted_target = target_prob * ignore_mask_exp
        weighted_prod = pred_prob * target_prob * ignore_mask_exp

        numerator = 2 * weighted_prod.sum(dim=(0, 2, 3)) + self.smooth
        denominator = (
            weighted_pred.sum(dim=(0, 2, 3))
            + weighted_target.sum(dim=(0, 2, 3))
            + self.smooth
        )
        dice_per_class = 1 - numerator / denominator

        if weights_tensor is not None:
            dice_loss_scalar = (dice_per_class * weights_tensor).sum()
        else:
            dice_loss_scalar = dice_per_class.mean()

        pred_b_in = pred_prob
        targ_b_in = target_prob

        gt_b = F.max_pool2d(1 - targ_b_in, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - targ_b_in
        )
        pred_b = F.max_pool2d(1 - pred_b_in, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - pred_b_in
        )

        gt_b_ext = F.max_pool2d(gt_b, self.theta, 1, (self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, 1, (self.theta - 1) // 2)

        gt_b = gt_b * ignore_mask_exp
        pred_b = pred_b * ignore_mask_exp
        gt_b_ext = gt_b_ext * ignore_mask_exp
        pred_b_ext = pred_b_ext * ignore_mask_exp

        gt_b = gt_b.view(n, C_eff, -1)
        pred_b = pred_b.view(n, C_eff, -1)
        gt_b_ext = gt_b_ext.view(n, C_eff, -1)
        pred_b_ext = pred_b_ext.view(n, C_eff, -1)

        P = (pred_b * gt_b_ext).sum(dim=2) / (pred_b.sum(dim=2) + 1e-7)
        R = (pred_b_ext * gt_b).sum(dim=2) / (gt_b.sum(dim=2) + 1e-7)

        BF1 = 2 * P * R / (P + R + 1e-7)
        boundary_loss_unreduced = 1 - BF1

        if weights_tensor is not None:
            boundary_loss = (boundary_loss_unreduced.mean(dim=0) * weights_tensor).sum()
        else:
            boundary_loss = torch.mean(boundary_loss_unreduced)

        velocity_loss = zero
        if velocity is not None and velocity_mask is not None:
            if C_eff == 2:
                bg_prob = pred_prob[:, 0:1]
            else:
                bg_prob = pred_prob[:, :1]

            p_moving_physics = torch.sigmoid(
                (velocity - self.velocity_high_speed_threshold) * 0.5
            )

            per_pixel_penalty = bg_prob * p_moving_physics

            combined_mask = velocity_mask * ignore_mask_exp
            valid_count = combined_mask.sum()
            base_velocity_loss = (per_pixel_penalty * combined_mask).sum()
            base_velocity_loss = base_velocity_loss / valid_count.clamp_min(1.0)
            current_weight = self._compute_velocity_weight(current_epoch)
            velocity_loss = base_velocity_loss * current_weight

            self.last_velocity_base = base_velocity_loss.detach()
            self.last_velocity_loss = velocity_loss.detach()
            self.last_velocity_weight = float(current_weight)
            self.last_velocity_valid = True

        return [dice_loss_scalar, boundary_loss, velocity_loss]

    def _forward_aryal_2023(
        self, pred_prob: torch.Tensor, target: torch.Tensor
    ) -> list[torch.Tensor]:
        """Exact custom-loss arithmetic from Aryal public commit 378c053."""
        n, channels, _, _ = pred_prob.shape
        valid = torch.sum(target, dim=1) == 1

        gt_b = F.max_pool2d(1 - target, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - target
        )
        pred_b = F.max_pool2d(1 - pred_prob, self.theta0, 1, (self.theta0 - 1) // 2) - (
            1 - pred_prob
        )
        gt_b_ext = F.max_pool2d(gt_b, self.theta, 1, (self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, 1, (self.theta - 1) // 2)

        gt_b = gt_b.view(n, channels, -1)
        pred_b = pred_b.view(n, channels, -1)
        gt_b_ext = gt_b_ext.view(n, channels, -1)
        pred_b_ext = pred_b_ext.view(n, channels, -1)
        boundary_precision = (pred_b * gt_b_ext).sum(dim=2) / (pred_b.sum(dim=2) + 1e-7)
        boundary_recall = (pred_b_ext * gt_b).sum(dim=2) / (gt_b.sum(dim=2) + 1e-7)
        boundary_f1 = (
            2
            * boundary_precision
            * boundary_recall
            / (boundary_precision + boundary_recall + 1e-7)
        )
        boundary_loss = torch.mean(1 - boundary_f1)

        pred_hwc = pred_prob.permute(0, 2, 3, 1)
        target_hwc = target.permute(0, 2, 3, 1)
        dice_per_class = 1 - (
            2.0 * (pred_hwc * target_hwc)[valid].sum(dim=0) + self.smooth
        ) / (pred_hwc[valid].sum(dim=0) + target_hwc[valid].sum(dim=0) + self.smooth)
        foreground_weights = pred_prob.new_tensor([0.0, 1.0])
        dice_loss = (dice_per_class * foreground_weights).sum()
        return [dice_loss, boundary_loss, pred_prob.new_zeros(())]

    def _compute_velocity_weight(self, current_epoch: Optional[int]) -> float:
        if current_epoch is None:
            return self.velocity_loss_weight

        if current_epoch < self.velocity_loss_warmup_epochs:
            return 0.0

        ramp_progress = (current_epoch - self.velocity_loss_warmup_epochs) / max(
            1, self.velocity_loss_ramp_epochs
        )

        if ramp_progress >= 1.0:
            return self.velocity_loss_weight

        return self.velocity_loss_weight * ramp_progress
