import pytest
import torch

from typeflow.model import FiLM, GlyphUNet, ModelConfig, StyleEncoder, TypeFlow, tiny_config

CFG = tiny_config()


def inputs(b: int = 4, cfg: ModelConfig = CFG) -> tuple[torch.Tensor, ...]:
    size = cfg.image_size
    return (
        torch.randn(b, 1, size, size),
        torch.rand(b),
        torch.randint(0, cfg.chars, (b,)),
        torch.randn(b, cfg.style_dim),
    )


def test_velocity_has_the_shape_of_the_image() -> None:
    x, t, chars, style = inputs()
    assert TypeFlow(CFG)(x, t, chars, style).shape == x.shape


def test_new_network_predicts_zero() -> None:
    x, t, chars, style = inputs()
    assert TypeFlow(CFG)(x, t, chars, style).abs().max() == 0


def test_film_starts_as_the_identity() -> None:
    h = torch.randn(2, 8, 4, 4)
    assert torch.equal(FiLM(16, 8)(h, torch.randn(2, 16)), h)


def test_film_scales_and_shifts_each_channel() -> None:
    film = FiLM(4, 2)
    with torch.no_grad():
        film.project.weight.zero_()
        film.project.bias.copy_(torch.tensor([1.0, 0.0, 3.0, 0.0]))
    h = torch.ones(1, 2, 1, 1)
    assert film(h, torch.zeros(1, 4)).flatten().tolist() == [5.0, 1.0]


def test_all_parameters_receive_gradient() -> None:
    torch.manual_seed(0)
    model = TypeFlow(CFG)
    with torch.no_grad():
        model.unet.head.weight.normal_()
        for module in model.modules():
            if isinstance(module, FiLM):
                module.project.weight.normal_(std=0.01)
    x, t, chars, _ = inputs()
    refs = torch.randn(4, 3, CFG.image_size, CFG.image_size)
    ref_chars = torch.randint(0, CFG.chars, (4, 3))
    model(x, t, chars, model.encode_style(refs, ref_chars)).square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None and n != "null_style"]
    assert missing == []


def test_style_encoder_ignores_reference_order() -> None:
    encoder = StyleEncoder(CFG).eval()
    refs = torch.randn(2, 5, CFG.image_size, CFG.image_size)
    chars = torch.randint(0, CFG.chars, (2, 5))
    order = torch.randperm(5)
    a = encoder(refs, chars)
    b = encoder(refs[:, order], chars[:, order])
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 8])
def test_style_encoder_accepts_any_reference_count(k: int) -> None:
    refs = torch.randn(3, k, CFG.image_size, CFG.image_size)
    chars = torch.randint(0, CFG.chars, (3, k))
    assert StyleEncoder(CFG)(refs, chars).shape == (3, CFG.style_dim)


def test_style_encoder_knows_which_letter_it_sees() -> None:
    torch.manual_seed(0)
    encoder = StyleEncoder(CFG)
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, FiLM):
                module.project.weight.normal_(std=0.1)
    refs = torch.randn(1, 1, CFG.image_size, CFG.image_size)
    a = encoder(refs, torch.tensor([[0]]))
    b = encoder(refs, torch.tensor([[1]]))
    assert not torch.allclose(a, b)


def test_pixel_unshuffle_keeps_full_resolution_output() -> None:
    cfg = tiny_config(image_size=48)
    x = torch.randn(1, 1, 48, 48)
    assert GlyphUNet(cfg)(x, torch.zeros(1, cfg.cond_dim)).shape == x.shape


def test_config_round_trip() -> None:
    assert ModelConfig.from_dict(CFG.to_dict()) == CFG


def test_default_model_fits_the_repository_limit_at_half_precision() -> None:
    params = sum(p.numel() for p in TypeFlow(ModelConfig()).parameters())
    assert params * 2 < 19 * 1024 * 1024
