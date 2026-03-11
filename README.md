# BKNO (Binary Kernel Neural Operator)

This repository contains a BKNO implementation for:
- Pure Python training (with binary kernels + STE).
- Optional C++ kernel acceleration for inference on CPU.
- Export artifacts for LibTorch/C++ deployment.

## Files

- `BKNO.py`
  - Core model: `BKNO`, `BKNOBlock`.
  - Jacobian utilities:
    - `jacobian_u_to_m_vjp(...)`
    - `jacobian_u_to_m_full(...)`
  - Training utility:
    - `train_bkno(...)` (L1+L2 mixed loss + sparsity regularization)
  - Checkpoint I/O:
    - `save_bkno_checkpoint(...)`
    - `load_bkno_checkpoint(...)`
  - Deployment export:
    - `export_bkno_for_libtorch(...)`

- `cpp/bkno_binary_kernel.cpp`
  - Custom CPU C++ op used by BKNO when `use_cpp_kernel=True`.
  - Implements binary conv logic using XNOR/POPCNT-style computation.

- `scripts/test_bkno_cpp.py`
  - Builds/loads the C++ extension and runs a small output-shape smoke test.

- `scripts/check_cpp_env.sh`
  - Linux environment checker for C++ extension prerequisites.

- `scripts/run_cpp_test_linux.sh`
  - Linux one-command runner for the C++ kernel smoke test.

- `scripts/check_cpp_env.ps1`, `scripts/run_cpp_test_with_msvc.ps1`
  - Windows equivalents (if needed).

## Input/Output Convention

BKNO expects:
- `M`: medium distribution, shape `[H, W, T]` or `[B, H, W, T]`
- `S`: source field, same shape as `M`

Output:
- `U`: predicted field, same shape as `M` / `S`

## Python Training

Use pure Python path for training:

```python
import BKNO

model = BKNO.build_bkno(
    hidden_channels=16,
    num_layers=4,
    rho=2,
    n_basis=4,
    use_cpp_kernel=False,  # training path
)

history = BKNO.train_bkno(
    model,
    train_loader,               # yields (M, S, U) or {"M","S","U"}
    val_loader=val_loader,
    epochs=50,
    lr=1e-3,
    l1_weight=0.5,              # L1/L2 mixed loss
    l2_weight=0.5,
    sparsity_weight=1e-4,
)
```

## Save / Load Checkpoints

```python
BKNO.save_bkno_checkpoint(model, "ckpt/bkno.pt", history=history)

model2, ckpt = BKNO.load_bkno_checkpoint(
    "ckpt/bkno.pt",
    use_cpp_kernel=False,
)
```

## Export for LibTorch / C++

```python
paths = BKNO.export_bkno_for_libtorch(
    model,
    export_dir="export/bkno",
    example_hwv=(32, 32, 128),
)
print(paths)
```

Export outputs:
- `train_state.pt`: full floating-point training state.
- `deploy_binary_state.pt`: pre-binarized kernels + scaling terms for custom C++ runtime.
- `model_config.json`: model metadata.
- `model.ts`: TorchScript fallback model (for LibTorch path without custom op).

## Run C++ Kernel on Linux (Ubuntu 22 / Jetson AGX)

### 1) Install build requirements

```bash
sudo apt update
sudo apt install -y build-essential ninja-build cmake python3-dev
```

Use your PyTorch environment (venv/conda) and confirm `torch` is installed there.

### 2) Environment check

```bash
chmod +x scripts/check_cpp_env.sh scripts/run_cpp_test_linux.sh
./scripts/check_cpp_env.sh python3
```

If your Python is not `python3`, pass full path:

```bash
./scripts/check_cpp_env.sh /path/to/venv/bin/python
```

### 3) Build + test C++ extension

```bash
./scripts/run_cpp_test_linux.sh python3
```

Expected output includes:
- Torch version
- C++ extension build logs
- `cpp extension loaded: True`
- output tensor shape/dtype

## Notes

- Training should generally use `use_cpp_kernel=False`.
- Deployment/inference can enable `use_cpp_kernel=True` on CPU when C++ extension is available.
- For large Jacobians, prefer `jacobian_u_to_m_vjp(...)` over full Jacobian construction.
