#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"[RUN] {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("[CWD] " + (str(cwd) if cwd else "") + "\n")
        f.write("[CMD] " + " ".join(cmd) + "\n")
        f.write("=" * 90 + "\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
            bufsize=1,
        )

        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)

        return p.wait()

def try_write_repro_info(wd: Path, out_dir: Path):
    # Best-effort: don't fail training if these commands fail
    info = []
    def sh(cmd):
        try:
            r = subprocess.check_output(cmd, cwd=str(wd), text=True, stderr=subprocess.STDOUT)
            return r.strip()
        except Exception as e:
            return f"<failed: {e}>"

    info.append(("time", datetime.now().isoformat(timespec="seconds")))
    info.append(("git_commit", sh(["git", "rev-parse", "HEAD"])))
    info.append(("git_status", sh(["git", "status", "--porcelain"])))
    info.append(("python", sh([sys.executable, "--version"])))
    info.append(("pip_freeze", sh([sys.executable, "-m", "pip", "freeze"])))

    p = out_dir / "repro.txt"
    with p.open("w", encoding="utf-8") as f:
        for k, v in info:
            f.write(f"## {k}\n{v}\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", default="~/jkim46_research/diffusers/examples/unconditional_image_generation")
    ap.add_argument("--train_script", default="train_unconditional.py")

    # CIFAR-10 (prepared as image folder)
    ap.add_argument("--data", default="/yunity/jkim46/data/cifar10_64/train")

    # You can pass either --out_dir directly, or --out_root to auto-name by seed/time
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--out_root", default="/yunity/jkim46/outputs")
    ap.add_argument("--run_name", default="cifar10_ddpm")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--num_processes", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--train_batch_size", type=int, default=128)
    ap.add_argument("--num_epochs", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--ddpm_beta_schedule", default="squaredcos_cap_v2")
    ap.add_argument("--prediction_type", default="epsilon", choices=["epsilon", "sample"])
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
    ap.add_argument("--checkpointing_steps", type=int, default=2000)
    ap.add_argument("--checkpoints_total_limit", type=int, default=10)

    args = ap.parse_args()

    wd = Path(os.path.expanduser(args.wd)).resolve()

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_root).resolve() / f"{args.run_name}_seed{args.seed}_{stamp}"
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    # Record reproducibility metadata
    try_write_repro_info(wd=wd, out_dir=out_dir)

    cmd = [
        "accelerate", "launch",
        f"--num_processes={args.num_processes}",
        f"--mixed_precision={args.mixed_precision}",
        args.train_script,
        f"--train_data_dir={args.data}",
        f"--resolution={args.resolution}",
        f"--train_batch_size={args.train_batch_size}",
        f"--num_epochs={args.num_epochs}",
        f"--learning_rate={args.learning_rate}",
        f"--ddpm_beta_schedule={args.ddpm_beta_schedule}",
        f"--prediction_type={args.prediction_type}",
        "--use_ema" if args.use_ema else "",
        f"--logger={args.logger}",
        f"--checkpointing_steps={args.checkpointing_steps}",
        f"--checkpoints_total_limit={args.checkpoints_total_limit}",
        f"--output_dir={str(out_dir)}",
        f"--seed={args.seed}",  # ✅ 중요: seed 전달
    ]
    cmd = [c for c in cmd if c != ""]  # remove empty

    env = os.environ.copy()
    rc = run(cmd, log_path=log_path, cwd=wd, env=env)

    if rc != 0:
        print(f"\n[FAIL] return code={rc}. Check log: {log_path}", file=sys.stderr)
    else:
        print(f"\n[OK] done. Log: {log_path}")
        print(f"[OK] outputs: {out_dir}")
    sys.exit(rc)

if __name__ == "__main__":
    main()
