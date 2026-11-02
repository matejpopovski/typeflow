import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from typeflow.evaluate import cluster_bootstrap
from typeflow.metrics import grey_share, ink_iou, l1, ssim


def glyph(seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    image = np.zeros((64, 64), dtype=np.float32)
    for _ in range(5):
        x, y = rng.integers(4, 44, size=2)
        image[y : y + 16, x : x + 5] = 1
    return torch.from_numpy(image)


def test_identical_images_score_perfectly() -> None:
    a = glyph()[None]
    assert l1(a, a).item() == 0
    assert ssim(a, a).item() == pytest.approx(1)
    assert ink_iou(a, a).item() == 1


def test_ssim_matches_scikit_image() -> None:
    a, b = glyph(0), glyph(1)
    reference = structural_similarity(
        a.numpy(),
        b.numpy(),
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert ssim(a[None], b[None]).item() == pytest.approx(reference, abs=1e-3)


def test_blur_raises_grey_share() -> None:
    sharp = glyph()[None]
    kernel = torch.ones(1, 1, 5, 5) / 25
    blurred = torch.conv2d(sharp[None], kernel, padding=2)[0]
    assert grey_share(sharp).item() == 0
    assert grey_share(blurred).item() > 0.3


def test_ink_iou_ignores_blank_paper() -> None:
    a = torch.zeros(1, 64, 64)
    b = torch.zeros(1, 64, 64)
    a[:, 10:20, 10:20] = 1
    b[:, 10:20, 15:25] = 1
    assert ink_iou(a, b).item() == pytest.approx(50 / 150)


def test_two_blank_images_agree() -> None:
    blank = torch.zeros(1, 8, 8)
    assert ink_iou(blank, blank).item() == 1


def test_bootstrap_interval_contains_the_mean_and_tightens_with_data() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=600)
    wide = cluster_bootstrap(values[:30], [str(i) for i in range(30)])
    tight = cluster_bootstrap(values, [str(i) for i in range(600)])
    assert tight[0] < values.mean() < tight[1]
    assert tight[1] - tight[0] < wide[1] - wide[0]


def test_bootstrap_widens_when_values_are_clustered() -> None:
    values = np.repeat(np.random.default_rng(0).normal(size=20), 10)
    independent = cluster_bootstrap(values, [str(i) for i in range(200)])
    clustered = cluster_bootstrap(values, [str(i // 10) for i in range(200)])
    assert clustered[1] - clustered[0] > independent[1] - independent[0]
