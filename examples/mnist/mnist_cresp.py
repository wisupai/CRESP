#!/usr/bin/env python
# examples/mnist/mnist_cresp.py

"""
MNIST example using CRESP workflow

This example demonstrates how to use CRESP to create a reproducible
MNIST classification experiment using PyTorch.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CRESP_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(CRESP_ROOT))

# Import CRESP Workflow
from cresp.core.config import Workflow

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create a CRESP workflow
workflow = Workflow(
    title="MNIST Classification with PyTorch",
    authors=[
        {"name": "John Doe", "affiliation": "Example University", "email": "john.doe@example.edu"}
    ],
    description="A simple example of reproducible MNIST classification using CRESP",
    config_path="mnist_cresp.yaml",
    seed=42
)

# Define workflow stages using decorators
@workflow.stage(
    id="download_data",
    description="Download MNIST dataset",
    outputs=["data/MNIST"]
)
def download_mnist_data():
    """Download MNIST dataset using torchvision"""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download training data
    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test data
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Downloaded MNIST dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    return {"train_samples": len(train_dataset), "test_samples": len(test_dataset)}

@workflow.stage(
    id="prepare_data",
    description="Prepare data loaders",
    dependencies=["download_data"]
)
def prepare_data_loaders():
    """Create data loaders for training and testing"""
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data
    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=False,
        transform=transform
    )
    
    # Load test data
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=False,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Prepared data loaders with batch sizes: train={64}, test={1000}")
    return {"train_loader": train_loader, "test_loader": test_loader}

@workflow.stage(
    id="train_model",
    description="Train MNIST model",
    dependencies=["prepare_data"],
    outputs=["outputs/mnist_model.pt"]
)
def train_model():
    """Train a simple CNN model on MNIST"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get data from previous stage
    data = prepare_data_loaders()
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    
    # Create model, optimizer, and loss function
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 5
    train_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")
    
    # Save the model
    model_path = OUTPUT_DIR / "mnist_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Return the trained model and losses
    return {"model": model, "train_losses": train_losses}

@workflow.stage(
    id="evaluate_model",
    description="Evaluate trained model",
    dependencies=["train_model"],
    outputs=["outputs/accuracy.txt", "outputs/loss_curve.png"]
)
def evaluate_model():
    """Evaluate the trained model on test data"""
    # Get data and model from previous stages
    data = prepare_data_loaders()
    train_result = train_model()
    
    test_loader = data["test_loader"]
    model = train_result["model"]
    train_losses = train_result["train_losses"]
    
    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    
    # Save accuracy to file
    accuracy_path = OUTPUT_DIR / "accuracy.txt"
    with open(accuracy_path, 'w') as f:
        f.write(f"Test accuracy: {accuracy:.2f}%\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
    
    # Plot and save learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('MNIST Training Loss Curve')
    plt.grid(True)
    
    loss_curve_path = OUTPUT_DIR / "loss_curve.png"
    plt.savefig(loss_curve_path)
    plt.close()
    
    print(f"Evaluation results saved to {accuracy_path} and {loss_curve_path}")
    
    return {"accuracy": accuracy, "test_loss": test_loss}

@workflow.stage(
    id="generate_report",
    description="Generate experiment report",
    dependencies=["evaluate_model"],
    outputs=["outputs/report.md"]
)
def generate_report():
    """Generate a simple markdown report of the experiment"""
    # Get results from previous stages
    eval_results = evaluate_model()
    
    # Create report
    report = [
        "# MNIST Classification Experiment Report",
        "",
        "## Experiment Summary",
        "",
        f"- Accuracy: {eval_results['accuracy']:.2f}%",
        f"- Test Loss: {eval_results['test_loss']:.4f}",
        "",
        "## Model Architecture",
        "",
        "```python",
        "class SimpleCNN(nn.Module):",
        "    def __init__(self):",
        "        super(SimpleCNN, self).__init__()",
        "        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)",
        "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)",
        "        self.fc1 = nn.Linear(7 * 7 * 64, 128)",
        "        self.fc2 = nn.Linear(128, 10)",
        "```",
        "",
        "## Results",
        "",
        "![Training Loss Curve](loss_curve.png)",
        "",
        "## Reproducibility",
        "",
        "This experiment was conducted using CRESP for reproducibility. Random seed was set to 42."
    ]
    
    # Save report
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated and saved to {report_path}")
    
    return {"report_path": str(report_path)}

def main():
    """Run the MNIST workflow"""
    print("Starting MNIST workflow with CRESP...")
    
    # Run all workflow stages
    results = workflow.run()
    
    # Save workflow configuration
    workflow.save_config()
    
    print("\nWorkflow completed successfully!")
    print(f"Final accuracy: {results['evaluate_model']['accuracy']:.2f}%")
    print(f"Report generated at: {results['generate_report']['report_path']}")

if __name__ == "__main__":
    main()
