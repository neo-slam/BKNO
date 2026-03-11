from __future__ import annotations

import platform
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "cpp" / "bkno_binary_kernel.cpp"
    build_dir = root / ".torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch={torch.__version__}")
    print(f"source={src}")
    print(f"build_dir={build_dir}")

    is_windows = platform.system().lower().startswith("win")
    extra_cflags = ["/O2"] if is_windows else ["-O3"]

    ext = load(
        name="bkno_binary_kernel_ext_test",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        build_directory=str(build_dir),
        verbose=True,
    )

    x = torch.randint(0, 2, (1, 2, 8, 8, 16), dtype=torch.uint8)
    w = torch.randint(0, 2, (3, 2, 3, 3, 3), dtype=torch.uint8)
    y = ext.binary_conv3d_forward(x, w, 1, 1, 1)
    print("cpp extension loaded:", ext is not None)
    print("output shape:", tuple(y.shape))
    print("output dtype:", y.dtype)
    print("sample value:", float(y.flatten()[0]))


if __name__ == "__main__":
    main()
