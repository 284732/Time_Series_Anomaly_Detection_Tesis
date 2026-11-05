import torch
from src.models.deep.conv_ae import ConvAutoencoderNet
from src.models.deep.lstm_ae import LSTMAutoencoderNet
from src.models.deep.vae import VAENet

W, F = 100, 38


def test_conv_ae_has_real_bottleneck():
    net = ConvAutoencoderNet(W, F, latent_dim=16)
    x = torch.randn(8, W, F)
    assert net.encode(x).shape == (8, 16)
    assert net(x).shape == x.shape
    assert 16 < W * F


def test_conv_ae_odd_window():
    net = ConvAutoencoderNet(97, F, latent_dim=16)
    x = torch.randn(4, 97, F)
    assert net(x).shape == x.shape


def test_lstm_ae_shape():
    net = LSTMAutoencoderNet(W, F, latent_dim=16)
    x = torch.randn(8, W, F)
    assert net(x).shape == x.shape


def test_vae_eval_is_deterministic():
    net = VAENet(W, F, latent_dim=16).eval()
    x = torch.randn(8, W, F)
    assert torch.allclose(net(x)[0], net(x)[0])