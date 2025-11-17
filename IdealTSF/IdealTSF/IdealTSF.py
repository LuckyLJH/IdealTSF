import torch
import random
import numpy as np
import math
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from .utils.attention import scaled_dot_product_attention
from .utils.dataset import LabeledDataset
from .utils.revin import RevIN
from .utils.ECOS import ECOS
from scipy.special import gamma
import torch.nn.functional as F
class IdealTSFArchitecture(nn.Module):
    def __init__(self, num_channels, seq_len, hid_dim, pred_horizon, use_revin=True):
        super().__init__()
        self.revin = RevIN(num_features=num_channels, condition_dim=20)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)
        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)
        self.use_revin = use_revin

    def forward(self, x, flatten_output=True):
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2)  # (n, D, L)
        else:
            x_norm = x

        queries = self.compute_queries(x_norm)  # (n, D, hid_dim)
        keys = self.compute_keys(x_norm)  # (n, D, hid_dim)
        values = self.compute_values(x_norm)  # (n, D, L)

        if hasattr(nn.functional, 'scaled_dot_product_attention'):
            att_score = nn.functional.scaled_dot_product_attention(queries, keys, values)  # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values)  # (n, D, L)

        attention_weights = att_score

        out = x_norm + att_score  # (n, D, L)

        out = self.linear_forecaster(out)  # (n, D, H)

        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2)  # (n, D, H)

        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]]), attention_weights
        else:
            return out, attention_weights




class TimeSeriesAugmentations:
    def __init__(self, noise_std=0.01, max_erase_length=300, erase_prob=0.1, noise_scales=[0.1, 0.05, 0.02],
                 min_erase_length=4):
        self.noise_std = noise_std
        self.erase_prob = erase_prob
        self.max_erase_length = max_erase_length
        self.noise_scales = noise_scales
        self.min_erase_length = min_erase_length

    def stable_jump_noise(self, x, jump_prob=0.3, max_jump_scale=5.0):
        if random.random() > jump_prob:
            return x

        device = x.device
        n_samples, n_channels, n_timesteps = x.size()

        jump_time = torch.randint(0, n_timesteps, (n_samples,), device=device)

        jumps = torch.zeros_like(x)
        for i in range(n_samples):
            scale = torch.empty(1, device=device).uniform_(1.0, max_jump_scale)
            jumps[i, :, jump_time[i]] = torch.randn(n_channels, device=device) * scale

        return x + jumps

    def add_multiscale_noise(self, x):
        device = x.device
        n_samples, n_channels, n_timesteps = x.size()

        noise = torch.zeros_like(x, device=device)

        for i, scale in enumerate(self.noise_scales):
            noise_level = scale * self.noise_std
            noise_component = torch.randn_like(x) * noise_level
            window_size = 2 ** (i + 1)
            noise_component = noise_component.unfold(2, window_size, 1).mean(dim=3)

            if noise_component.size(2) < n_timesteps:
                padding_size = n_timesteps - noise_component.size(2)
                noise_component = torch.cat(
                    [noise_component, torch.zeros(n_samples, n_channels, padding_size, device=device)], dim=2)
            elif noise_component.size(2) > n_timesteps:
                noise_component = noise_component[:, :, :n_timesteps]

            noise += noise_component

        return x + noise

    def structured_erase(self, x):
        if np.random.rand() < self.erase_prob:
            erase_length = np.random.randint(self.min_erase_length, min(self.max_erase_length, x.size(2) // 4) + 1)
            start_idx = np.random.randint(0, x.size(2) - erase_length)
            x[:, :, start_idx:start_idx + erase_length] = 0

        return x

    def augment(self, x, use_stable_jump=None):
        if x is not None and use_stable_jump:
            x = self.stable_jump_noise(x)

        x = self.add_multiscale_noise(x)
        x = self.structured_erase(x)
        return x

    @staticmethod
    def preprocess_series(x, z_thresh=2.5, use_iqr=True, window_size=5):
        assert x.ndim == 3, "Input tensor must be 3D: (batch, channels, timesteps)"
        device = x.device

        nan_mask = torch.isnan(x)
        def linear_interpolate(seq):
            # seq: (B, C, T)
            left = seq.clone()
            right = seq.clone()

            for t in range(1, seq.size(-1)):
                left[:, :, t] = torch.where(
                    torch.isnan(left[:, :, t]),
                    left[:, :, t - 1],
                    left[:, :, t]
                )

            for t in reversed(range(seq.size(-1) - 1)):
                right[:, :, t] = torch.where(
                    torch.isnan(right[:, :, t]),
                    right[:, :, t + 1],
                    right[:, :, t]
                )

            interp = 0.5 * (left + right)
            return interp

        x = torch.where(nan_mask, torch.tensor(0.0, device=device), x)  # 临时置 0
        x = linear_interpolate(x)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        z_score = (x - mean) / std
        outlier_mask = (z_score.abs() > z_thresh)

        if use_iqr:
            q1 = x.quantile(0.25, dim=-1, keepdim=True)
            q3 = x.quantile(0.75, dim=-1, keepdim=True)
            iqr = q3 - q1
            iqr_mask = (x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))
            outlier_mask |= iqr_mask

        kernel = torch.ones(1, 1, 3, device=device) / 3.0
        x_padded = F.pad(x, (1, 1), mode='replicate')
        local_mean = F.conv1d(x_padded.view(-1, 1, x.size(-1) + 2), kernel).view(x.shape)

        x = torch.where(outlier_mask, local_mean, x)

        avg_kernel = torch.ones(1, 1, window_size, device=device) / window_size
        pad = window_size // 2
        x_padded = F.pad(x, (pad, pad), mode='reflect')
        x_smooth = F.conv1d(x_padded.view(-1, 1, x.size(-1) + 2 * pad), avg_kernel).view(x.shape)

        return x_smooth



class IdealTSF:

    def __init__(self, device='cuda:0', num_epochs=100, batch_size=256, base_optimizer=torch.optim.AdamW,
                 learning_rate=1e-3, weight_decay=1e-5, rho=0.5, use_revin=True, random_state=None,use_preprocessing=False,
                 augmentations=None):
        self.network = None
        self.criterion = nn.L1Loss()
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.random_state = 123
        self.augmentations = augmentations
        self.use_preprocessing = use_preprocessing

    def fit(self, x, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        self.network = IdealTSFArchitecture(num_channels=x.shape[1], seq_len=x.shape[2], hid_dim=16,
                                             pred_horizon=y.shape[1] // x.shape[1], use_revin=self.use_revin)
        self.criterion = self.criterion.to(self.device)
        self.network = self.network.to(self.device)
        self.network.train()

        optimizer = ECOS(self.network.parameters(), base_optimizer=self.base_optimizer, rho=self.rho,
                        lr=self.learning_rate, weight_decay=self.weight_decay)

        train_dataset = LabeledDataset(x, y)
        data_loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            loss_list = []
            for (x_batch, y_batch) in data_loader_train:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if self.augmentations:
                    x_batch = self.augmentations.augment(x_batch)

                out_batch, attention_weights = self.network(x_batch)
                loss = self.criterion(out_batch, y_batch)

                if optimizer.__class__.__name__ == 'ECOS':
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out_batch, attention_weights = self.network(x_batch)
                    loss = self.criterion(out_batch, y_batch)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_list.append(loss.item())

            train_loss = np.mean(loss_list)
            self.network.train()
            progress_bar.set_description(f"Epoch {epoch}: Train Loss {train_loss:.4f}", refresh=True)

            if epoch % 100 == 0:
                self.plot_attention_heatmap(attention_weights, epoch)

        return

    def plot_attention_heatmap(self, attention_weights, epoch):

        att_weights_np = attention_weights.cpu().detach().numpy()
        n_samples, n_heads, seq_len = att_weights_np.shape

        sample_idx = 0

        plt.figure(figsize=(10, 8))
        plt.imshow(att_weights_np[sample_idx], cmap='inferno', aspect='auto')  # 'hot'、'viridis'、'plasma'、'inferno'
        plt.colorbar()
        plt.title(f"Attention Weights (Epoch {epoch})")
        plt.xlabel("Time Step")
        plt.ylabel("Attention Head")
        plt.tight_layout()
        plt.show()

    def forecast(self, x, batch_size=256):
        self.network.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outs = []
        all_attention_weights = []

        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)

            if self.augmentations:
                x = self.augmentations.augment(x)

            with torch.no_grad():
                out, attention_weights = self.network(x)

            outs.append(out.cpu())
            all_attention_weights.append(attention_weights.cpu())

        outs = torch.cat(outs)
        all_attention_weights = torch.cat(all_attention_weights)

        self.plot_attention_heatmap(all_attention_weights, "Forecast")

        return outs.cpu().numpy()

    def predict(self, x, batch_size=256):
        return self.forecast(x, batch_size=batch_size)

    # def predict(self, x, batch_size=256):
    #     return self.forecast(x, batch_size=batch_size)
    #
    # def print_model_summary(self):
    #     """Prints a summary of the model parameters."""
    #     print("Model Summary:")
    #     torchsummary.summary(self.network, input_size=(x.shape[1], x.shape[2]))  # Adjust input size
    #
    # def compute_macs(self):
    #     """Computes MACs of the model."""
    #     print("Model MACs and FLOPs:")
    #     macs, params = get_model_complexity_info(self.network, (x.shape[1], x.shape[2]), as_strings=True,
    #                                               print_per_layer_stat=True, verbose=True)
    #     print(f"MACs: {macs}")
    #     print(f"Params: {params}")

class Pretrainer:
    def __init__(self, device='cuda:0', num_epochs=50, batch_size=256, learning_rate=1e-3,
                 weight_decay=1e-5, rho=0.5, augmentations=None, use_revin=True):
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.augmentations = augmentations
        self.network = None

    def pretrain(self, x, y):
        self.network = IdealTSFArchitecture(
            num_channels=x.shape[1],
            seq_len=x.shape[2],
            hid_dim=16,
            pred_horizon=y.shape[1] // x.shape[1],
            use_revin=self.use_revin
        ).to(self.device)

        optimizer = ECOS(self.network.parameters(), base_optimizer=torch.optim.AdamW, rho=self.rho,
                        lr=self.learning_rate, weight_decay=self.weight_decay)

        criterion = nn.L1Loss().to(self.device)
        dataset = LabeledDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.network.train()
        for epoch in tqdm(range(self.num_epochs), desc='[Pretraining]'):
            epoch_losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                if self.augmentations:
                    xb = self.augmentations.augment(xb)

                out, _ = self.network(xb)
                loss = criterion(out, yb)

                if isinstance(optimizer, ECOS):
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out, _ = self.network(xb)
                    loss = criterion(out, yb)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
            print(f"[Pretrain Epoch {epoch}] Loss: {np.mean(epoch_losses):.4f}")
        return self.network

class Trainer:
    def __init__(self, pretrained_model, device='cuda:0', num_epochs=50, batch_size=256, learning_rate=1e-4,
                 weight_decay=1e-5, rho=0.5, use_revin=True):
        self.network = pretrained_model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin

    def fine_tune(self, x, y):
        optimizer = ECOS(self.network.parameters(), base_optimizer=torch.optim.AdamW, rho=self.rho,
                        lr=self.learning_rate, weight_decay=self.weight_decay)

        criterion = nn.L1Loss().to(self.device)
        dataset = LabeledDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.network.to(self.device)
        self.network.train()

        for epoch in tqdm(range(self.num_epochs), desc='[Fine-tuning]'):
            epoch_losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)


                out, _ = self.network(xb)
                loss = criterion(out, yb)

                if isinstance(optimizer, ECOS):
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out, _ = self.network(xb)
                    loss = criterion(out, yb)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
            print(f"[Fine-tune Epoch {epoch}] Loss: {np.mean(epoch_losses):.4f}")

    def forecast(self, x):
        self.network.eval()
        preds = []
        attn_weights = []

        loader = DataLoader(torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32)),
                            batch_size=self.batch_size)
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                out, attn = self.network(xb)
                preds.append(out.cpu())
                attn_weights.append(attn.cpu())

        return torch.cat(preds).numpy(), torch.cat(attn_weights).numpy()
