import pytest
import torch
import numpy as np
import pandas as pd
import sys
import os

# 添加項目根目錄到路徑（確保可相對導入）
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.gans.financial_gan import FinancialGAN, FinancialGenerator, FinancialDiscriminator
from data.gans.stress_test import StressTestGenerator


class TestFinancialGAN:
    """FinancialGAN 單元測試"""

    def test_gan_initialization(self):
        gan = FinancialGAN(sequence_length=10, feature_dim=5, noise_dim=20, hidden_dim=64)
        assert gan.sequence_length == 10
        assert gan.feature_dim == 5
        assert gan.noise_dim == 20
        assert gan.hidden_dim == 64
        if hasattr(gan, 'device') and gan.device is not None:
            assert gan.device.type in ['cuda', 'cpu']
        if hasattr(gan, 'generator') and hasattr(gan, 'discriminator') and torch.__version__:
            assert isinstance(gan.generator, (torch.nn.Module, torch.nn.DataParallel))
            assert isinstance(gan.discriminator, (torch.nn.Module, torch.nn.DataParallel, type(None)))

    def test_generator_forward(self):
        generator = FinancialGenerator(noise_dim=20, hidden_dim=64, output_dim=5)
        batch_size = 8
        sequence_length = 10
        if hasattr(torch, 'randn'):
            noise = torch.randn(batch_size, sequence_length, 20)
            output = generator(noise)
            assert output.shape == (batch_size, sequence_length, 5)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_discriminator_forward(self):
        if not hasattr(torch, 'randn'):
            pytest.skip('Torch not available')
        discriminator = FinancialDiscriminator(input_dim=5, hidden_dim=64)
        batch_size = 8
        sequence_length = 10
        sequences = torch.randn(batch_size, sequence_length, 5)
        out = discriminator(sequences)
        assert out.shape == (batch_size, 1)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert (out >= 0).all() and (out <= 1).all()

    def test_gan_noise_generation(self):
        gan = FinancialGAN(sequence_length=10, feature_dim=5, noise_dim=20)
        batch_size = 4
        noise = gan.generate_noise(batch_size)
        if hasattr(torch, 'is_tensor') and torch.is_tensor(noise):
            assert noise.shape == (batch_size, 10, 20)
            assert not torch.isnan(noise).any()
            assert not torch.isinf(noise).any()
        else:
            assert noise.shape == (batch_size, 10, 20)
            assert not np.isnan(noise).any()
            assert not np.isinf(noise).any()

    def test_gan_sample_generation(self):
        gan = FinancialGAN(sequence_length=10, feature_dim=5, noise_dim=20)
        n_samples = 6
        samples = gan.generate_samples(n_samples)
        assert samples.shape == (n_samples, 10, 5)
        assert not np.isnan(samples).any()
        assert not np.isinf(samples).any()
        assert (samples >= -1).all() and (samples <= 1).all()

    def test_gan_training_step(self):
        gan = FinancialGAN(sequence_length=10, feature_dim=5, noise_dim=20, hidden_dim=64)
        if gan.discriminator is None:
            pytest.skip('Torch not available for training step')
        batch_size = 4
        real_data = torch.randn(batch_size, 10, 5)
        losses = gan.train_epoch(real_data)
        assert 'g_loss' in losses and 'd_loss' in losses
        assert isinstance(losses['g_loss'], float) and isinstance(losses['d_loss'], float)
        assert not np.isnan(losses['g_loss']) and not np.isnan(losses['d_loss'])


class TestStressTestGenerator:
    def test_stress_generator_initialization(self):
        base = pd.DataFrame({
            'open': range(100),
            'high': range(100),
            'low': range(100),
            'close': range(100),
            'volume': range(100)
        })
        stress = StressTestGenerator(base)
        assert stress.gan_model is None

    def test_data_preparation(self):
        base = pd.DataFrame({
            'open': range(50),
            'high': range(50),
            'low': range(50),
            'close': range(50),
            'volume': range(50)
        })
        stress = StressTestGenerator(base)
        feats = stress._prepare_features()
        assert feats.shape == (50, 5)

    def test_sequence_creation(self):
        base = pd.DataFrame({
            'open': range(100),
            'high': range(100),
            'low': range(100),
            'close': range(100),
            'volume': range(100)
        })
        stress = StressTestGenerator(base)
        feats = stress._prepare_features()
        seqs = stress._create_sequences(feats, length=20)
        assert len(seqs) == 80
        assert seqs[0].shape == (20, 5)

    def test_sequence_resizing(self):
        base = pd.DataFrame({
            'open': range(100),
            'high': range(100),
            'low': range(100),
            'close': range(100),
            'volume': range(100)
        })
        stress = StressTestGenerator(base)
        long_seq = np.random.randn(150, 5)
        short_seq = np.random.randn(50, 5)
        same_seq = np.random.randn(100, 5)
        assert len(stress._resize_sequence(long_seq, 100)) == 100
        assert len(stress._resize_sequence(short_seq, 100)) == 100
        assert len(stress._resize_sequence(same_seq, 100)) == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
class TestFinancialGANGPU:
    def test_gpu_device_selection(self):
        gan = FinancialGAN(use_gpu=True)
        assert gan.device.type == 'cuda'
        assert next(gan.generator.parameters()).is_cuda
        assert next(gan.discriminator.parameters()).is_cuda

    def test_gpu_training(self):
        gan = FinancialGAN(use_gpu=True)
        real = torch.randn(4, 10, 5).to(gan.device)
        losses = gan.train_epoch(real)
        assert not np.isnan(losses['g_loss']) and not np.isnan(losses['d_loss'])

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
