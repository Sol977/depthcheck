import torch
import torch.nn.functional as F

def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: (B,1,H,W)
    mask: (B,1,H,W) in {0,1}
    """
    diff = (pred - gt).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

def masked_si_log(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Scale-invariant log loss (useful when scale may drift).
    For ARKitScenes/OpenSUN3D depth is metric, so keep it as an auxiliary loss.
    """
    p = torch.log(pred.clamp_min(eps))
    g = torch.log(gt.clamp_min(eps))
    d = (p - g) * mask
    denom = mask.sum().clamp_min(1.0)
    mean = d.sum() / denom
    return (d.pow(2).sum() / denom) - mean.pow(2)
