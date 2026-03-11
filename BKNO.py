from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gelu_derivative(x: torch.Tensor) -> torch.Tensor:
    # Exact derivative of GELU(x) = x * Phi(x).
    sqrt_2 = 1.4142135623730951
    inv_sqrt_2pi = 0.3989422804014327
    cdf = 0.5 * (1.0 + torch.erf(x / sqrt_2))
    pdf = torch.exp(-0.5 * x * x) * inv_sqrt_2pi
    return cdf + x * pdf


def _ste_binarize01(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for binarization into {0, 1}.
    Forward: hard threshold.
    Backward: identity surrogate.
    """
    hard = (x >= 0).to(x.dtype)
    return hard.detach() - x.detach() + x


class _BKNOCppOp:
    def __init__(self) -> None:
        self._mod = None
        self._tried = False

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from torch.utils.cpp_extension import load
        except Exception:
            return

        src = Path(__file__).resolve().parent / "cpp" / "bkno_binary_kernel.cpp"
        if not src.exists():
            return

        try:
            is_windows = platform.system().lower().startswith("win")
            cflags = ["/O2"] if is_windows else ["-O3"]
            self._mod = load(
                name="bkno_binary_kernel_ext",
                sources=[str(src)],
                extra_cflags=cflags,
                build_directory=str(Path(__file__).resolve().parent / ".torch_extensions"),
                verbose=False,
            )
        except Exception:
            self._mod = None

    def available(self) -> bool:
        self._load()
        return self._mod is not None

    def conv3d(
        self,
        input_bits: torch.Tensor,
        weight_bits: torch.Tensor,
        padding: Tuple[int, int, int],
    ) -> torch.Tensor:
        self._load()
        if self._mod is None:
            raise RuntimeError("BKNO C++ operator is unavailable.")
        pd, ph, pw = padding
        return self._mod.binary_conv3d_forward(input_bits, weight_bits, pd, ph, pw)


_bkno_cpp = _BKNOCppOp()


class BKNOBlock(nn.Module):
    """
    BKNO layer:
    A^(l) ≈ sum_i beta_i * B_i,  K^(l) ≈ sum_j lambda_j * Lambda_j
    Z = sum_i sum_j beta_i lambda_j * (Lambda_j (*) B_i) + omega * A
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        rho: int = 2,
        n_basis: int = 4,
        use_cpp_kernel: bool = True,
    ) -> None:
        super().__init__()
        kd, kh, kw = kernel_size
        if kd % 2 == 0 or kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("kernel_size should be odd to keep same output shape.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = (kd // 2, kh // 2, kw // 2)
        self.rho = rho
        self.n_basis = n_basis
        self.use_cpp_kernel = use_cpp_kernel

        self.input_threshold = nn.Parameter(torch.zeros(rho))
        self.beta_raw = nn.Parameter(torch.ones(rho))

        self.kernel_logits = nn.Parameter(
            torch.randn(n_basis, channels, channels, kd, kh, kw) * 0.02
        )
        self.lambda_raw = nn.Parameter(torch.ones(n_basis))

        self.omega = nn.Parameter(torch.tensor(0.0))

    def _binary_conv(self, x_bits: torch.Tensor, w_bits: torch.Tensor) -> torch.Tensor:
        # x_bits: [B, C, D, H, W], values in {0,1}
        # w_bits: [O, C, kD, kH, kW], values in {0,1}
        if self.use_cpp_kernel and x_bits.device.type == "cpu" and _bkno_cpp.available():
            return _bkno_cpp.conv3d(
                x_bits.to(torch.uint8).contiguous(),
                w_bits.to(torch.uint8).contiguous(),
                self.padding,
            ).to(x_bits.dtype)

        return F.conv3d(
            x_bits,
            w_bits,
            bias=None,
            stride=1,
            padding=self.padding,
        )

    def _forward_with_cache(self, a: torch.Tensor):
        # a: [B, C, D, H, W]
        beta = F.softplus(self.beta_raw)
        lamb = F.softplus(self.lambda_raw)
        beta_sum = beta.sum()

        z = torch.zeros_like(a)
        kernel_bits = []
        for i in range(self.rho):
            b_i = _ste_binarize01(a - self.input_threshold[i])
            for j in range(self.n_basis):
                lam_j = _ste_binarize01(self.kernel_logits[j])
                if i == 0:
                    kernel_bits.append(lam_j)
                conv_ij = self._binary_conv(b_i, lam_j)
                z = z + beta[i] * lamb[j] * conv_ij

        z = z + self.omega * a
        out = F.gelu(z)
        cache = {
            "z": z,
            "lamb": lamb,
            "beta_sum": beta_sum,
            "kernel_bits": kernel_bits,
        }
        return out, cache

    def backward_input_closed_form(self, grad_out: torch.Tensor, cache) -> torch.Tensor:
        """
        Closed-form Jacobian-vector product for this block:
        grad_in = (d A^(l+1) / d A^(l))^T @ grad_out
        """
        z = cache["z"]
        lamb = cache["lamb"]
        beta_sum = cache["beta_sum"]
        kernel_bits = cache["kernel_bits"]

        theta_z = grad_out * _gelu_derivative(z)
        grad_in = self.omega * theta_z

        for j in range(self.n_basis):
            # For conv3d forward with weight [C_out, C_in, ...],
            # gradient wrt input uses conv_transpose3d with same weight.
            g = F.conv_transpose3d(
                theta_z,
                kernel_bits[j],
                bias=None,
                stride=1,
                padding=self.padding,
            )
            grad_in = grad_in + (beta_sum * lamb[j]) * g
        return grad_in

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward_with_cache(a)
        return out


class BKNO(nn.Module):
    """
    Binary Kernel Neural Operator model.
    Inputs:
      M: [H, W, T] or [B, H, W, T]
      S: [H, W, T] or [B, H, W, T]
    Output:
      U: same shape as M/S
    """

    def __init__(
        self,
        hidden_channels: int = 16,
        num_layers: int = 4,
        rho: int = 2,
        n_basis: int = 4,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        use_cpp_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.lift = nn.Conv3d(2, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                BKNOBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    rho=rho,
                    n_basis=n_basis,
                    use_cpp_kernel=use_cpp_kernel,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Conv3d(hidden_channels, 1, kernel_size=1)

    @staticmethod
    def _to_bdhwt(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        # Input [H,W,T] or [B,H,W,T] -> [B,H,W,T], return whether original had batch.
        if x.dim() == 3:
            return x.unsqueeze(0), False
        if x.dim() == 4:
            return x, True
        raise ValueError("Expected input shape [H,W,T] or [B,H,W,T].")

    def forward_batched_no_check(self, m_bhwt: torch.Tensor, s_bhwt: torch.Tensor) -> torch.Tensor:
        # Input: [B,H,W,T] and [B,H,W,T], output: [B,H,W,T]
        x = torch.stack([m_bhwt, s_bhwt], dim=1).permute(0, 1, 4, 2, 3).contiguous()
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        u = self.proj(x)
        return u.permute(0, 3, 4, 2, 1).squeeze(-1).contiguous()

    def forward(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        m_bhwt, had_batch_m = self._to_bdhwt(m)
        s_bhwt, had_batch_s = self._to_bdhwt(s)
        if had_batch_m != had_batch_s:
            raise ValueError("M and S must either both have batch dimension or both not.")
        if m_bhwt.shape != s_bhwt.shape:
            raise ValueError("M and S must have the same shape.")

        u = self.forward_batched_no_check(m_bhwt, s_bhwt)
        if not had_batch_m:
            u = u.squeeze(0)
        return u

    def _forward_with_caches(self, m: torch.Tensor, s: torch.Tensor):
        m_bhwt, had_batch_m = self._to_bdhwt(m)
        s_bhwt, had_batch_s = self._to_bdhwt(s)
        if had_batch_m != had_batch_s:
            raise ValueError("M and S must either both have batch dimension or both not.")
        if m_bhwt.shape != s_bhwt.shape:
            raise ValueError("M and S must have the same shape.")

        x = torch.stack([m_bhwt, s_bhwt], dim=1).permute(0, 1, 4, 2, 3).contiguous()
        x = self.lift(x)

        block_caches = []
        for blk in self.blocks:
            x, cache = blk._forward_with_cache(x)
            block_caches.append(cache)
        u = self.proj(x)
        u_bhwt = u.permute(0, 3, 4, 2, 1).squeeze(-1).contiguous()
        if not had_batch_m:
            u_bhwt = u_bhwt.squeeze(0)
        return u_bhwt, block_caches, had_batch_m

    def jacobian_u_to_m_vjp(
        self,
        m: torch.Tensor,
        s: torch.Tensor,
        grad_u: Optional[torch.Tensor] = None,
        method: str = "auto",
    ) -> torch.Tensor:
        """
        Compute vector-Jacobian product from U to M:
            vjp = (dU/dM)^T @ grad_u

        Args:
          m, s: [H,W,T] or [B,H,W,T]
          grad_u: same shape as U (if None, uses ones)
          method:
            - "auto" / "closed_form": use paper-style closed-form chain rule.
            - "autograd": use torch autograd fallback.
        Returns:
          Tensor with same shape as M.
        """
        if method not in {"auto", "closed_form", "autograd"}:
            raise ValueError("method must be one of: auto, closed_form, autograd")

        if method == "autograd":
            return self._jacobian_u_to_m_vjp_autograd(m, s, grad_u)

        u, block_caches, had_batch = self._forward_with_caches(m, s)
        if grad_u is None:
            grad_u = torch.ones_like(u)
        if grad_u.shape != u.shape:
            raise ValueError("grad_u must have same shape as U.")

        if not had_batch:
            grad_u = grad_u.unsqueeze(0)

        # [B,H,W,T] -> [B,1,D,H,W]
        grad = grad_u.permute(0, 3, 1, 2).unsqueeze(1).contiguous()

        # Through projection layer (1x1x1 conv).
        grad = F.conv_transpose3d(
            grad,
            self.proj.weight,
            bias=None,
            stride=1,
            padding=0,
        )

        # Through BKNO blocks in reverse.
        for blk, cache in zip(reversed(self.blocks), reversed(block_caches)):
            grad = blk.backward_input_closed_form(grad, cache)

        # Through lift layer back to 2 input channels [M,S].
        grad_in = F.conv_transpose3d(
            grad,
            self.lift.weight,
            bias=None,
            stride=1,
            padding=0,
        )
        grad_m = grad_in[:, 0]  # [B,D,H,W]
        grad_m = grad_m.permute(0, 2, 3, 1).contiguous()  # [B,H,W,T]

        if not had_batch:
            grad_m = grad_m.squeeze(0)
        return grad_m

    def _jacobian_u_to_m_vjp_autograd(
        self,
        m: torch.Tensor,
        s: torch.Tensor,
        grad_u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Autograd path requires float conv kernels (no C++ uint8 custom op).
        old_flags = [blk.use_cpp_kernel for blk in self.blocks]
        for blk in self.blocks:
            blk.use_cpp_kernel = False

        try:
            m_var = m.detach().clone().requires_grad_(True)
            u = self.forward(m_var, s)
            if grad_u is None:
                grad_u = torch.ones_like(u)
            if grad_u.shape != u.shape:
                raise ValueError("grad_u must have same shape as U.")
            grad_m = torch.autograd.grad(
                outputs=u,
                inputs=m_var,
                grad_outputs=grad_u,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            return grad_m
        finally:
            for blk, f in zip(self.blocks, old_flags):
                blk.use_cpp_kernel = f

    def jacobian_u_to_m_full(
        self,
        m: torch.Tensor,
        s: torch.Tensor,
        method: str = "autograd",
        max_output_elements: int = 512,
    ) -> torch.Tensor:
        """
        Build full Jacobian dU/dM with shape U_shape + M_shape.
        This is expensive; intended for debugging or very small grids.
        """
        u = self.forward(m, s)
        num_out = u.numel()
        if num_out > max_output_elements:
            raise ValueError(
                f"U has {num_out} elements; full Jacobian disabled above {max_output_elements}. "
                "Use jacobian_u_to_m_vjp instead."
            )

        rows = []
        flat_basis = torch.eye(num_out, dtype=u.dtype, device=u.device)
        for k in range(num_out):
            grad_u = flat_basis[k].reshape_as(u)
            row = self.jacobian_u_to_m_vjp(m, s, grad_u=grad_u, method=method).reshape(-1)
            rows.append(row)
        jac = torch.stack(rows, dim=0)
        return jac.reshape(*u.shape, *m.shape)


def build_bkno(
    hidden_channels: int = 16,
    num_layers: int = 4,
    rho: int = 2,
    n_basis: int = 4,
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    use_cpp_kernel: bool = True,
) -> BKNO:
    return BKNO(
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        rho=rho,
        n_basis=n_basis,
        kernel_size=kernel_size,
        use_cpp_kernel=use_cpp_kernel,
    )


def _unpack_batch(batch: Any, device: torch.device):
    """
    Supported batch formats:
      1) (M, S, U)
      2) {"M": M, "S": S, "U": U}
    """
    if isinstance(batch, dict):
        m = batch["M"].to(device)
        s = batch["S"].to(device)
        u = batch["U"].to(device)
        return m, s, u
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        m, s, u = batch
        return m.to(device), s.to(device), u.to(device)
    raise ValueError("Batch must be (M,S,U) or dict with keys M,S,U.")


def _kernel_sparsity_penalty(model: BKNO, target_on_ratio: Optional[float] = None) -> torch.Tensor:
    """
    Sparsity regularizer over kernel logits.
    Uses sigmoid(logits) as a smooth proxy for binary activation probability.
    """
    penalties = []
    for blk in model.blocks:
        prob = torch.sigmoid(blk.kernel_logits)
        if target_on_ratio is None:
            penalties.append(prob.mean())
        else:
            penalties.append((prob.mean() - target_on_ratio).abs())
    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


def train_bkno(
    model: BKNO,
    train_loader: Iterable[Any],
    *,
    val_loader: Optional[Iterable[Any]] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    l1_weight: float = 0.5,
    l2_weight: float = 0.5,
    sparsity_weight: float = 1e-4,
    target_kernel_on_ratio: Optional[float] = None,
    grad_clip_norm: Optional[float] = 1.0,
    device: Optional[torch.device] = None,
    log_every: int = 20,
) -> Dict[str, Any]:
    """
    Full-Python training loop with STE binarization in forward pass.
    Reconstruction loss is a weighted mixture of L1 and L2 losses.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training is pure Python path.
    for blk in model.blocks:
        blk.use_cpp_kernel = False

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    denom = l1_weight + l2_weight
    if denom <= 0:
        raise ValueError("l1_weight + l2_weight must be > 0.")
    l1_ratio = l1_weight / denom
    l2_ratio = l2_weight / denom

    history: Dict[str, Any] = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_main = 0.0
        running_sparse = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            m, s, u_gt = _unpack_batch(batch, device)
            opt.zero_grad(set_to_none=True)

            u_pred = model(m, s)
            loss_l1 = F.l1_loss(u_pred, u_gt)
            loss_l2 = F.mse_loss(u_pred, u_gt)
            loss_main = l1_ratio * loss_l1 + l2_ratio * loss_l2
            loss_sparse = _kernel_sparsity_penalty(model, target_kernel_on_ratio)
            loss = loss_main + sparsity_weight * loss_sparse

            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

            running_loss += float(loss.detach())
            running_main += float(loss_main.detach())
            running_sparse += float(loss_sparse.detach())
            num_batches += 1

            if log_every > 0 and step % log_every == 0:
                print(
                    f"[train] epoch={ep}/{epochs} step={step} "
                    f"loss={running_loss / num_batches:.6f} "
                    f"mse={running_main / num_batches:.6f} "
                    f"sparse={running_sparse / num_batches:.6f}"
                )

        train_epoch_loss = running_loss / max(1, num_batches)
        history["train_loss"].append(train_epoch_loss)

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    m, s, u_gt = _unpack_batch(batch, device)
                    u_pred = model(m, s)
                    val_l1 = F.l1_loss(u_pred, u_gt)
                    val_l2 = F.mse_loss(u_pred, u_gt)
                    val_running += float(l1_ratio * val_l1 + l2_ratio * val_l2)
                    val_batches += 1
            val_loss = val_running / max(1, val_batches)
            history["val_loss"].append(val_loss)
            print(
                f"[epoch] {ep}/{epochs} train_loss={train_epoch_loss:.6f} val_loss={val_loss:.6f}"
            )
        else:
            print(f"[epoch] {ep}/{epochs} train_loss={train_epoch_loss:.6f}")

    return history


def _model_config(model: BKNO) -> Dict[str, Any]:
    if len(model.blocks) == 0:
        raise ValueError("Model has no BKNO blocks.")
    blk0 = model.blocks[0]
    return {
        "hidden_channels": model.hidden_channels,
        "num_layers": len(model.blocks),
        "rho": blk0.rho,
        "n_basis": blk0.n_basis,
        "kernel_size": list(blk0.kernel_size),
    }


def save_bkno_checkpoint(
    model: BKNO,
    checkpoint_path: str,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    history: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_config": _model_config(model),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if history is not None:
        payload["history"] = history
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, str(p))


def load_bkno_checkpoint(
    checkpoint_path: str,
    *,
    map_location: Optional[str] = None,
    use_cpp_kernel: bool = False,
) -> Tuple[BKNO, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    cfg = ckpt["model_config"]
    model = build_bkno(
        hidden_channels=int(cfg["hidden_channels"]),
        num_layers=int(cfg["num_layers"]),
        rho=int(cfg["rho"]),
        n_basis=int(cfg["n_basis"]),
        kernel_size=tuple(cfg["kernel_size"]),
        use_cpp_kernel=use_cpp_kernel,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def export_bkno_for_libtorch(
    model: BKNO,
    export_dir: str,
    *,
    example_hwv: Tuple[int, int, int] = (32, 32, 128),
    dtype: torch.dtype = torch.float32,
    save_scripted_model: bool = True,
) -> Dict[str, str]:
    """
    Export files for C++/LibTorch deployment.

    Outputs:
      - train_state.pt: full floating-point state_dict (for continued training or fallback inference)
      - deploy_binary_state.pt: pre-binarized kernels/scales for C++ custom BKNO kernel
      - model_config.json: architecture and tensor metadata
      - model.ts (optional): TorchScript module for fallback C++ inference (non-custom path)
    """
    model = model.eval().cpu()
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _model_config(model)
    cfg["example_hwv"] = list(example_hwv)
    cfg["dtype"] = str(dtype).replace("torch.", "")

    full_state_path = out_dir / "train_state.pt"
    torch.save({"model_config": cfg, "model_state_dict": model.state_dict()}, str(full_state_path))

    deploy: Dict[str, Any] = {
        "model_config": cfg,
        "lift_weight": model.lift.weight.detach().cpu(),
        "lift_bias": model.lift.bias.detach().cpu() if model.lift.bias is not None else None,
        "proj_weight": model.proj.weight.detach().cpu(),
        "proj_bias": model.proj.bias.detach().cpu() if model.proj.bias is not None else None,
        "blocks": [],
    }

    for blk in model.blocks:
        beta = F.softplus(blk.beta_raw.detach()).cpu()
        lamb = F.softplus(blk.lambda_raw.detach()).cpu()
        kernel_bits = (blk.kernel_logits.detach().cpu() >= 0).to(torch.uint8)
        block_info = {
            "input_threshold": blk.input_threshold.detach().cpu(),
            "beta": beta,
            "lambda": lamb,
            "omega": blk.omega.detach().cpu(),
            "kernel_bits": kernel_bits,
            "kernel_size": list(blk.kernel_size),
            "padding": list(blk.padding),
            "rho": int(blk.rho),
            "n_basis": int(blk.n_basis),
            "channels": int(blk.channels),
        }
        deploy["blocks"].append(block_info)

    deploy_path = out_dir / "deploy_binary_state.pt"
    torch.save(deploy, str(deploy_path))

    config_path = out_dir / "model_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    paths: Dict[str, str] = {
        "train_state": str(full_state_path),
        "deploy_binary_state": str(deploy_path),
        "model_config": str(config_path),
    }

    if save_scripted_model:
        old_flags = [b.use_cpp_kernel for b in model.blocks]
        for b in model.blocks:
            b.use_cpp_kernel = False
        try:
            ts_path = out_dir / "model.ts"
            try:
                scripted = torch.jit.script(model)
            except Exception:
                H, W, V = example_hwv
                m_ex = torch.rand(1, H, W, V, dtype=dtype)
                s_ex = torch.rand(1, H, W, V, dtype=dtype)
                class _DeployWrapper(nn.Module):
                    def __init__(self, core: BKNO) -> None:
                        super().__init__()
                        self.core = core

                    def forward(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                        return self.core.forward_batched_no_check(m, s)

                scripted = torch.jit.trace(_DeployWrapper(model), (m_ex, s_ex), strict=False)
            scripted.save(str(ts_path))
            paths["torchscript"] = str(ts_path)
        finally:
            for b, flg in zip(model.blocks, old_flags):
                b.use_cpp_kernel = flg

    return paths


if __name__ == "__main__":
    # run a quick shape check
    model = build_bkno(hidden_channels=8, num_layers=2, rho=2, n_basis=2, use_cpp_kernel=False)
    H, W, T = 32, 32, 64
    M = torch.rand(H, W, T)
    S = torch.rand(H, W, T)
    U = model(M, S)
    print("Output shape:", tuple(U.shape))
    Jv = model.jacobian_u_to_m_vjp(M, S)
    print("VJP shape (dU/dM)^T * 1:", tuple(Jv.shape))
