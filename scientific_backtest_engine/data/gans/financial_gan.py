import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class FinancialGenerator(nn.Module if TORCH_AVAILABLE else object):
    """簡化版生成器（若無 PyTorch 則退化為 numpy 隨機生成）。"""

    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, dropout: float = 0.2):
        if TORCH_AVAILABLE:
            super().__init__()
            self.noise_dim = noise_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size=noise_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0)
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Tanh(),
            )
        else:
            self.noise_dim = noise_dim
            self.output_dim = output_dim

    def forward(self, noise, conditions=None):
        if TORCH_AVAILABLE:
            lstm_out, _ = self.lstm(noise)
            attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
            return self.output_layer(attn)
        else:
            # numpy 退化版本：直接返回縮放噪聲
            return np.tanh(noise)


class FinancialDiscriminator(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        if TORCH_AVAILABLE:
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
            self.act = nn.LeakyReLU(0.2)
            self.drop = nn.Dropout(0.3)
            self.lstm = nn.LSTM(input_size=hidden_dim // 2, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(self, sequences):
        if not TORCH_AVAILABLE:
            # 無判別器於退化模式
            return None
        x = sequences.transpose(1, 2)
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.fc(last)


class FinancialGAN:
    """金融 GAN 模型（如無 PyTorch 則退化為隨機樣本生成）。"""

    def __init__(self, sequence_length: int = 60, feature_dim: int = 5, noise_dim: int = 100, hidden_dim: int = 128, learning_rate: float = 0.001, use_gpu: bool | None = None):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.device = None
        if TORCH_AVAILABLE:
            if use_gpu is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
            self.generator = FinancialGenerator(noise_dim, hidden_dim, feature_dim).to(self.device)
            self.discriminator = FinancialDiscriminator(feature_dim, hidden_dim).to(self.device)
            # 多卡支援
            if torch.cuda.device_count() > 1:
                self.generator = nn.DataParallel(self.generator)
                self.discriminator = nn.DataParallel(self.discriminator)
            self.g_optimizer = optim.Adam(self._params(self.generator), lr=learning_rate)
            self.d_optimizer = optim.Adam(self._params(self.discriminator), lr=learning_rate)
            self.criterion = nn.BCELoss()
        else:
            self.generator = FinancialGenerator(noise_dim, hidden_dim, feature_dim)
            self.discriminator = None

    def _params(self, module):
        return module.parameters() if not isinstance(module, nn.DataParallel) else module.module.parameters()

    def generate_noise(self, batch_size: int):
        if TORCH_AVAILABLE:
            return torch.randn(batch_size, self.sequence_length, self.noise_dim, device=self.device)
        else:
            return np.random.randn(batch_size, self.sequence_length, self.noise_dim)

    def train_epoch(self, real_data) -> Dict[str, float]:
        if not TORCH_AVAILABLE:
            return {"g_loss": 0.0, "d_loss": 0.0}
        bs = real_data.size(0)
        self.d_optimizer.zero_grad()
        # Label smoothing
        real_labels = torch.full((bs, 1), 0.9, device=self.device)
        fake_labels = torch.full((bs, 1), 0.1, device=self.device)
        real_out = self.discriminator(real_data)
        real_out = torch.nan_to_num(real_out, nan=0.5, posinf=0.999, neginf=0.001).clamp(1e-6, 1 - 1e-6)
        d_loss_real = self.criterion(real_out, real_labels)
        noise = self.generate_noise(bs)
        fake = self.generator(noise)
        # NaN/Inf 檢查
        if torch.isnan(fake).any() or torch.isinf(fake).any():
            return {"g_loss": float("nan"), "d_loss": float("nan")}
        fake_out = self.discriminator(fake.detach())
        fake_out = torch.nan_to_num(fake_out, nan=0.5, posinf=0.999, neginf=0.001).clamp(1e-6, 1 - 1e-6)
        d_loss_fake = self.criterion(fake_out, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params(self.discriminator), max_norm=1.0)
        self.d_optimizer.step()
        self.g_optimizer.zero_grad()
        noise = self.generate_noise(bs)
        fake = self.generator(noise)
        fake_out = self.discriminator(fake)
        fake_out = torch.nan_to_num(fake_out, nan=0.5, posinf=0.999, neginf=0.001).clamp(1e-6, 1 - 1e-6)
        g_loss = self.criterion(fake_out, real_labels)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params(self.generator), max_norm=1.0)
        self.g_optimizer.step()
        return {"g_loss": float(g_loss.item()), "d_loss": float(d_loss.item())}

    def generate_samples(self, n_samples: int) -> np.ndarray:
        if TORCH_AVAILABLE:
            self.generator.eval()
            with torch.no_grad():
                noise = self.generate_noise(n_samples)
                out = self.generator(noise)
                return out.cpu().numpy()
        else:
            noise = self.generate_noise(n_samples)
            out = self.generator.forward(noise)
            return out

    def fit(self, data: np.ndarray, epochs: int = 20, batch_size: int = 16, print_interval: int = 50):
        if not TORCH_AVAILABLE:
            return
        data_tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
        self.generator.train(); self.discriminator.train()
        for epoch in range(epochs):
            for (batch,) in loader:
                self.train_epoch(batch)
