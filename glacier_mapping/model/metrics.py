import torch


def precision(tp, fp, fn):
    return tp / (tp + fp + 1e-10)


def tp_fp_fn(pred, true, label=1):
    tp = ((pred == label) & (true == label)).sum().item()
    fp = ((pred == label) & (true != label)).sum().item()
    fn = ((pred != label) & (true == label)).sum().item()

    return tp, fp, fn


def recall(tp, fp, fn):
    return tp / (tp + fn + 1e-10)


def dice(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn + 1e-10)


def IoU(tp, fp, fn):
    return tp / (tp + fp + fn + 1e-10)


def l1_reg(params, lambda_reg, device):
    """Compute L1 regularization penalty."""
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.sum(abs(param))
    return penalty


def l2_reg(params, lambda_reg, device):
    """Compute L2 regularization penalty."""
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.norm(param, 2) ** 2
    return penalty
