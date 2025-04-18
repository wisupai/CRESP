# CRESP Reproduction Report
Workflow: MNIST Classification with PyTorch
Config: cresp.yaml
Timestamp: 2025-04-18 23:27:06
**Overall Status:** ❌ Failed


## ❌ Stage: train_model
| File | Status | Mode | Details |
|------|--------|------|---------|
| `outputs/mnist_model.pt` | **Failed** | `standard` | Standard validation failed: content mismatch |

## ❌ Stage: evaluate_model
| File | Status | Mode | Details |
|------|--------|------|---------|
| `outputs/accuracy.txt` | **Failed** | `standard` | Standard validation failed: content mismatch |
| `outputs/loss_curve.png` | **Passed** | `strict` | Strict validation passed: exact hash match |
