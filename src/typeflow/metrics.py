"""Image metrics. All of them take float tensors in [0, 1] with ink at 1, shaped ``(N, H, W)``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

INK_THRESHOLD = 0.5


def l1(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).abs().mean(dim=(1, 2))


def _gaussian_window(size: int, sigma: float, like: Tensor) -> Tensor:
    x = torch.arange(size, dtype=like.dtype, device=like.device) - (size - 1) / 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)[None, None]


def ssim(a: Tensor, b: Tensor, size: int = 11, sigma: float = 1.5) -> Tensor:
    """SSIM with the Gaussian window of Wang et al. 2004, averaged over the valid region."""
    c1, c2 = 0.01**2, 0.03**2
    window = _gaussian_window(size, sigma, a)
    x, y = a[:, None], b[:, None]
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    var_x = F.conv2d(x * x, window) - mu_x * mu_x
    var_y = F.conv2d(y * y, window) - mu_y * mu_y
    cov = F.conv2d(x * y, window) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def ink_iou(a: Tensor, b: Tensor) -> Tensor:
    """Overlap of the inked regions, which ignores the blank paper that dominates L1 and SSIM."""
    x, y = a > INK_THRESHOLD, b > INK_THRESHOLD
    intersection = (x & y).sum(dim=(1, 2)).float()
    union = (x | y).sum(dim=(1, 2)).float()
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))


def grey_share(a: Tensor) -> Tensor:
    """Grey pixels as a share of pixels touched by ink. Blur raises this; sharp type does not."""
    touched = (a > 0.1).sum(dim=(1, 2)).float().clamp_min(1)
    return ((a > 0.1) & (a < 0.9)).sum(dim=(1, 2)).float() / touched
