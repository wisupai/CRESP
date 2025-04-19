#!/usr/bin/env python
# examples/mnist/mnist_cresp.py

"""
Minimal MNIST demo using CRESP workflow
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import contextlib
import io
import argparse

CRESP_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(CRESP_ROOT))

from cresp.core import Workflow

# Device detection (CUDA/MPS/CPU)
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

# Minimal CNN for MNIST
default_batch_size = 64
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(7 * 7 * 32, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7 * 7 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Workflow config
parser = argparse.ArgumentParser(description="Minimal MNIST demo using CRESP workflow")
parser.add_argument(
    "--mode",
    choices=["experiment", "reproduction"],
    default="experiment",
    help="Workflow mode: experiment or reproduction (default: experiment)",
)
args = parser.parse_args()
mode = args.mode
workflow = Workflow(
    title="MNIST Demo with CRESP",
    authors=[{"name": "Wisup AI Team", "affiliation": "Wisup AI", "email": "team@wisup.ai"}],
    description="Minimal reproducible MNIST classification demo using CRESP",
    config_path="cresp.yaml",
    seed=42,
    mode=mode,
    skip_unchanged=True,
)

@workflow.stage(
    id="prepare_data",
    description="Download and prepare MNIST dataloaders",
    outputs=[{"path": "data", "shared": True}],
    reproduction_mode="strict",
)
def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_root = workflow.get_shared_data_path("data")
    # Silence stdout during download
    with contextlib.redirect_stdout(io.StringIO()):
        train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=default_batch_size, shuffle=True, **workflow.get_dataloader_kwargs())
    test_loader = DataLoader(test, batch_size=1000, shuffle=False, **workflow.get_dataloader_kwargs())
    return {"train_loader": train_loader, "test_loader": test_loader}

@workflow.stage(
    id="train",
    description="Train CNN on MNIST",
    dependencies=["prepare_data"],
    outputs=[{"path": "outputs/mnist_model.pt", "shared": False}],
    skip_if_unchanged=False,
    reproduction_mode="strict",
)
def train():
    torch.use_deterministic_algorithms(True)
    device = get_device()
    data = prepare_data()
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = data["train_loader"]
    losses = []
    for _ in range(3):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    torch.save({"model": model.state_dict(), "losses": losses}, workflow.get_output_path("outputs/mnist_model.pt"))
    return {"model": model, "losses": losses, "device": device}

@workflow.stage(
    id="evaluate",
    description="Evaluate model accuracy",
    dependencies=["train"],
    outputs=[{"path": "outputs/accuracy.txt", "shared": False}],
    reproduction_mode="strict",
)
def evaluate():
    data = prepare_data()
    result = train()
    model = result["model"]
    device = result["device"]
    test_loader = data["test_loader"]
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    acc = 100.0 * correct / len(test_loader.dataset)
    with open(workflow.get_output_path("outputs/accuracy.txt"), "w") as f:
        f.write(f"Test accuracy: {acc:.2f}%\n")
    return {"accuracy": acc}

if __name__ == "__main__":
    workflow.run()
