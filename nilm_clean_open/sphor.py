from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LinearSpHORNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)
        self.label_embeddings = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.label_embeddings, a=math.sqrt(5.0))

    def encode_h(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.encoder(x))

    def project_z(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode_h(x), p=2, dim=1)

    def label_directions(self) -> torch.Tensor:
        return F.normalize(self.label_embeddings, p=2, dim=1)

    def vmf_logits(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        return torch.matmul(self.project_z(x), self.label_directions().t()) / float(temperature)


def make_balanced_sampler(y_train: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = np.bincount(y_train)
    weights = np.asarray([1.0 / max(1, counts[int(y)]) for y in y_train], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    smoothing = float(smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    target = torch.full(
        (labels.numel(), num_classes),
        fill_value=smoothing / max(1, num_classes),
        device=labels.device,
        dtype=torch.float32,
    )
    target.scatter_(1, labels[:, None], 1.0 - smoothing + smoothing / max(1, num_classes))
    return target


def apply_feature_mixup(x: torch.Tensor, soft_y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or x.shape[0] <= 1:
        return x, soft_y
    lam = np.random.beta(float(alpha), float(alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    x_mix = float(lam) * x + (1.0 - float(lam)) * x[perm]
    y_mix = float(lam) * soft_y + (1.0 - float(lam)) * soft_y[perm]
    return torch.cat([x, x_mix], dim=0), torch.cat([soft_y, y_mix], dim=0)


def orthogonality_regularizer(mu: torch.Tensor, temperature: float) -> torch.Tensor:
    if mu.shape[0] <= 1:
        return torch.zeros((), device=mu.device)
    sim2 = torch.matmul(mu, mu.t()).pow(2)
    off_diag = sim2[~torch.eye(mu.shape[0], dtype=torch.bool, device=mu.device)]
    return torch.log(torch.mean(torch.exp(off_diag / float(temperature))) + 1e-12)


@dataclass
class SpHORConfig:
    embedding_dim: int = 128
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    eval_batch_size: int = 1024
    temperature: float = 0.10
    ortho_weight: float = 0.1
    label_smoothing: float = 0.1
    mixup_alpha: float = 1.0
    grad_clip: float = 5.0


def train_sphor_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    config: SpHORConfig,
    device: torch.device,
    seed: int,
) -> tuple[LinearSpHORNet, pd.DataFrame]:
    model = LinearSpHORNet(
        input_dim=x_train.shape[1],
        embedding_dim=int(config.embedding_dim),
        num_classes=int(num_classes),
    ).to(device)
    dataset = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(
        dataset,
        batch_size=min(int(config.batch_size), len(dataset)),
        sampler=make_balanced_sampler(y_train, seed=int(seed) + 31337),
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.epochs)))
    history: List[Dict[str, object]] = []

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        sums = {"loss": 0.0, "vmf": 0.0, "ortho": 0.0, "hard_acc": 0.0}
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            soft_y = smooth_one_hot(batch_y, num_classes=num_classes, smoothing=float(config.label_smoothing))
            train_x, train_soft_y = apply_feature_mixup(batch_x, soft_y, alpha=float(config.mixup_alpha))

            optimizer.zero_grad(set_to_none=True)
            logits = model.vmf_logits(train_x, temperature=float(config.temperature))
            vmf = -(train_soft_y * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            ortho = orthogonality_regularizer(model.label_directions(), temperature=float(config.temperature))
            loss = vmf + float(config.ortho_weight) * ortho
            loss.backward()
            if float(config.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.grad_clip))
            optimizer.step()

            batch_n = int(batch_y.numel())
            seen += batch_n
            with torch.no_grad():
                hard_logits = model.vmf_logits(batch_x, temperature=float(config.temperature))
                hard_acc = torch.mean((torch.argmax(hard_logits, dim=1) == batch_y).float())
            sums["loss"] += float(loss.detach().cpu()) * batch_n
            sums["vmf"] += float(vmf.detach().cpu()) * batch_n
            sums["ortho"] += float(ortho.detach().cpu()) * batch_n
            sums["hard_acc"] += float(hard_acc.detach().cpu()) * batch_n
        scheduler.step()
        history.append(
            {
                "epoch": int(epoch),
                "loss": sums["loss"] / max(1, seen),
                "vmf": sums["vmf"] / max(1, seen),
                "ortho": sums["ortho"] / max(1, seen),
                "hard_acc": sums["hard_acc"] / max(1, seen),
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
    return model, pd.DataFrame(history)


@torch.no_grad()
def encode_raw_h(model: LinearSpHORNet, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=min(int(batch_size), max(1, len(tensor))), shuffle=False)
    parts: List[np.ndarray] = []
    for (batch_x,) in loader:
        h = model.encode_h(batch_x.to(device, non_blocking=True))
        parts.append(h.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)


@torch.no_grad()
def encode_normalized_z(model: LinearSpHORNet, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=min(int(batch_size), max(1, len(tensor))), shuffle=False)
    parts: List[np.ndarray] = []
    for (batch_x,) in loader:
        z = model.project_z(batch_x.to(device, non_blocking=True))
        parts.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)
