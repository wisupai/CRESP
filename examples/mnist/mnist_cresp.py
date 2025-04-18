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
import time

CRESP_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(CRESP_ROOT))

from cresp.core.config import Workflow, ReproductionError

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Global cache for stage results to avoid recomputation
_stage_results_cache = {}

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

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

def create_experiment_workflow():
    """Create a workflow in experiment mode to record outputs"""
    return Workflow(
        title="MNIST Classification with PyTorch",
        authors=[
            {"name": "John Doe", "affiliation": "Example University", "email": "john.doe@example.edu"}
        ],
        description="A simple example of reproducible MNIST classification using CRESP",
        config_path="cresp.yaml",
        seed=42,
        mode="experiment",
        skip_unchanged=True,
        verbose_seed_setting=True  # 只在创建时显示一次种子信息
    )

def create_reproduction_workflow():
    """Create a workflow in reproduction mode to validate outputs"""
    return Workflow(
        title="MNIST Classification with PyTorch",
        authors=[
            {"name": "John Doe", "affiliation": "Example University", "email": "john.doe@example.edu"}
        ],
        description="A simple example of reproducible MNIST classification using CRESP",
        config_path="cresp.yaml",
        seed=42,
        mode="reproduction",
        skip_unchanged=True,
        reproduction_failure_mode="continue",  # Default: "stop" or "continue"
        save_reproduction_report=True,     # Default: True
        reproduction_report_path="reproduction_report.md", # Default path
        verbose_seed_setting=True  # 只在创建时显示一次种子信息
    )

# Create the workflow based on mode
if len(sys.argv) > 1 and sys.argv[1] == "reproduction":
    workflow = create_reproduction_workflow()
else:
    workflow = create_experiment_workflow()

@workflow.stage(
    id="download_data",
    description="Download MNIST dataset",
    outputs=["data"], 
    reproduction_mode="strict"
    # skip_if_unchanged uses workflow default (True)
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
    dependencies=["download_data"],
    reproduction_mode="strict"
    # skip_if_unchanged uses workflow default (True)
)
def prepare_data_loaders():
    """Create data loaders for training and testing"""
    # Check if we already have results cached
    if "prepare_data" in _stage_results_cache:
        return _stage_results_cache["prepare_data"]
        
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
    
    # Get reproducible dataloader settings from workflow
    dataloader_kwargs = workflow.get_dataloader_kwargs()
    
    # Create data loaders with reproducibility settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False,
        **dataloader_kwargs
    )
    
    print(f"Prepared data loaders with batch sizes: train={64}, test={1000}")
    
    # Cache and return results
    result = {"train_loader": train_loader, "test_loader": test_loader}
    _stage_results_cache["prepare_data"] = result
    return result

@workflow.stage(
    id="train_model",
    description="Train MNIST model",
    dependencies=["prepare_data"],
    outputs=["outputs/mnist_model.pt"],
    reproduction_mode="standard",
    tolerance_relative=1e-5,
    skip_if_unchanged=False # Override: Always rerun training
)
def train_model():
    """Train a simple CNN model on MNIST"""
    # Check if we already have results cached
    if "train_model" in _stage_results_cache:
        return _stage_results_cache["train_model"]
    
    # No need to set seeds manually, the workflow will do it before stage execution
    
    # Get device
    device = get_device()
    
    # Get data from previous stage
    data = prepare_data_loaders()
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    
    # Create model, optimizer, and loss function
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 1
    train_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'train_losses': train_losses,
        'random_seed': workflow.seed
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Cache and return results
    result = {"model": model, "train_losses": train_losses, "device": device}
    _stage_results_cache["train_model"] = result
    return result

@workflow.stage(
    id="evaluate_model",
    description="Evaluate trained model",
    dependencies=["train_model"],
    outputs=[
        {
            "path": "outputs/accuracy.txt",
            "reproduction": { "mode": "standard", "tolerance_absolute": 0.5 }
        },
        "outputs/loss_curve.png"
    ]
    # skip_if_unchanged uses workflow default (True)
)
def evaluate_model():
    """Evaluate the trained model on test data"""
    # No need to set seeds manually, the workflow will do it before stage execution
    
    # Check if we already have results cached
    if "evaluate_model" in _stage_results_cache:
        return _stage_results_cache["evaluate_model"]
        
    # Get data and model from previous stages
    data = prepare_data_loaders()
    train_result = _stage_results_cache.get("train_model") or train_model()
    
    test_loader = data["test_loader"]
    model = train_result["model"]
    device = train_result["device"]
    train_losses = train_result["train_losses"]
    
    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
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
        f.write(f"Random seed: {workflow.seed}\n")
    
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
    
    # Cache and return results
    result = {"accuracy": accuracy, "test_loss": test_loss}
    _stage_results_cache["evaluate_model"] = result
    return result

@workflow.stage(
    id="generate_report",
    description="Generate experiment report",
    dependencies=["evaluate_model"],
    outputs=["outputs/report.md"],
    reproduction_mode="tolerant",
    similarity_threshold=0.9
    # skip_if_unchanged uses workflow default (True)
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
    print(f"Mode: {workflow.mode}")

    try:
        # Run all workflow stages
        results = workflow.run()

        # Save workflow configuration (only strictly necessary in experiment mode, but safe to do always)
        workflow.save_config()

        # Success messages depend on the mode and if failures occurred (if mode=continue)
        print("\nWorkflow run finished.")
        if workflow.mode == "experiment":
            print("Experiment results and hashes have been recorded.")
        elif results: # Check if results is not None (it might be if error occurred early)
            print("Reproduction validation completed.")
            # Check if results exist for key stages before trying to print them
            if 'evaluate_model' in results and results['evaluate_model']:
                # Ensure accuracy is treated as float before formatting
                accuracy = results['evaluate_model'].get('accuracy')
                if accuracy is not None:
                    print(f"Final accuracy (from run): {float(accuracy):.2f}%")
                else:
                     print("Final accuracy (from run): N/A")
            if 'generate_report' in results and results['generate_report']:
                print(f"Report generated at (from run): {results['generate_report'].get('report_path', 'N/A')}")
        # If failure mode was 'continue', a warning is printed inside workflow.run()

    except ReproductionError as e:
        print(f"\n[bold red]Workflow halted due to reproduction failure:[/bold red]")
        print(f"[red]{e}[/red]")
        # Optionally save config even on failure? Maybe not.
        sys.exit(1) # Exit with error code

    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred during workflow execution:[/bold red]")
        print(f"[red]{e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    main()
