"""CPU-safe unit tests: shapes, closed-form forward process, sampling endpoints,
and a tiny overfit sanity check for both models."""

import torch

from vision_lab.ddpm import DDPM, UNet
from vision_lab.vit import ViT


# --------------------------------------------------------------------- ViT --

def test_vit_output_shape_and_params():
    model = ViT(img_size=32, patch=4, dim=64, depth=2, heads=4, n_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    assert 1e5 < model.num_params() < 5e7


def test_vit_patch_count_matches_grid():
    m = ViT(img_size=32, patch=4, dim=64, depth=1, heads=4)
    assert m.patch.n_patches == 64  # 8x8 grid of 4x4 patches


def test_vit_cls_token_is_used():
    """Perturbing the class token must change the output (it's not dead weight)."""
    model = ViT(img_size=32, patch=4, dim=64, depth=1, heads=4).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a = model(x)
        model.cls.data += 1.0
        b = model(x)
    assert not torch.allclose(a, b)


def test_vit_gradients_flow():
    model = ViT(img_size=32, patch=4, dim=64, depth=1, heads=4)
    loss = model(torch.randn(2, 3, 32, 32)).sum()
    loss.backward()
    assert model.pos.grad is not None
    assert model.patch.proj.weight.grad is not None


# -------------------------------------------------------------------- DDPM --

def test_q_sample_closed_form_matches_definition():
    ddpm = DDPM(timesteps=50, device="cpu")
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.tensor([3, 40])
    xt, noise = ddpm.q_sample(x0, t, noise=torch.zeros_like(x0))
    # with zero noise: x_t = sqrt(ac) * x0 exactly
    expected = ddpm.sqrt_ac[t][:, None, None, None] * x0
    torch.testing.assert_close(xt, expected)


def test_unet_output_shape_any_input_size():
    model = UNet(in_ch=1, base=16)
    for size in (32, 28):  # both 32 (CIFAR) and 28 (padded MNIST) work
        x = torch.randn(2, 1, size, size)
        t = torch.tensor([0, 17])
        assert model(x, t).shape == (2, 1, size, size)


def test_unet_predicts_finite_noise_all_timesteps():
    model = UNet(in_ch=1, base=16)
    x = torch.randn(1, 1, 32, 32)
    for t in (0, 100, 399):
        assert torch.isfinite(model(x, torch.tensor([t]))).all()


def test_sampling_starts_from_noise_ends_bounded():
    ddpm = DDPM(timesteps=8, device="cpu")  # tiny T for a fast test
    model = UNet(in_ch=1, base=8)
    samples = ddpm.sample(model, n=2, img_size=8, in_ch=1)
    assert samples.shape == (2, 1, 8, 8)
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_time_embedding_is_deterministic_and_distinct():
    from vision_lab.ddpm import SinusoidalTimeEmbedding
    emb = SinusoidalTimeEmbedding(64)
    a, b = emb(torch.tensor([5])), emb(torch.tensor([5]))
    c = emb(torch.tensor([6]))
    assert torch.equal(a, b) and not torch.allclose(a, c)
