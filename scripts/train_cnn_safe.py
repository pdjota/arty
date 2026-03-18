"""
Run `scripts/train_cnn.py` with a simple batch-size fallback strategy.

On some GPU setups, `--batch-size` can cause OOM and crash early.
This wrapper retries once with a smaller batch size if the first run
exits non-zero.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_train(
    *,
    arch: str,
    epochs: int,
    batch_size: int,
    cpu: bool,
    extra_args: list[str] | None = None,
) -> int:
    ROOT = Path(__file__).resolve().parent.parent
    train_script = ROOT / "scripts" / "train_cnn.py"

    cmd = [sys.executable, str(train_script), "--arch", arch, "--epochs", str(epochs), "--batch-size", str(batch_size)]
    if cpu:
        cmd.append("--cpu")
    if extra_args:
        cmd += extra_args

    r = subprocess.run(cmd, cwd=str(ROOT))
    return int(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Train with a batch-size fallback strategy.")
    p.add_argument("--arch", choices=["cnn", "cnnrnn"], required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size-primary", type=int, default=32)
    p.add_argument("--batch-size-fallback", type=int, default=16)
    p.add_argument("--cpu", action="store_true", help="Force CPU (mainly for local debugging)")
    p.add_argument("--extra-args", nargs="*", default=None, help="Extra args passed through to train_cnn.py")
    args = p.parse_args()

    rc = run_train(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size_primary,
        cpu=args.cpu,
        extra_args=args.extra_args,
    )
    if rc == 0:
        return

    if args.batch_size_fallback and args.batch_size_fallback != args.batch_size_primary:
        rc2 = run_train(
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size_fallback,
            cpu=args.cpu,
            extra_args=args.extra_args,
        )
        if rc2 != 0:
            sys.exit(rc2)
        return

    sys.exit(rc)


if __name__ == "__main__":
    main()

