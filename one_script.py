#!/usr/bin/env python3
import os, json, argparse, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_tensor

from diffusers import UNet2DModel, DDPMScheduler

# Optional metrics
try:
    import lpips  # pip install lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Helpers
# -------------------------
def sha256_tensor(x: torch.Tensor) -> str:
    arr = x.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def pick_unet_dir(ckpt_dir: str, prefer_ema: bool) -> str:
    ema = os.path.join(ckpt_dir, "unet_ema")
    unet = os.path.join(ckpt_dir, "unet")
    if prefer_ema and os.path.isdir(ema):
        return ema
    if os.path.isdir(unet):
        return unet
    if os.path.isdir(ema):
        return ema
    return ckpt_dir


def load_unet(ckpt_dir: str, prefer_ema: bool):
    p = pick_unet_dir(ckpt_dir, prefer_ema)
    m = UNet2DModel.from_pretrained(p).to(DEVICE).eval()
    return m, p


def load_scheduler_from_ckpt(ckpt_dir: str) -> Tuple[DDPMScheduler, str]:
    cfg_path = os.path.join(ckpt_dir, "scheduler", "scheduler_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    sch = DDPMScheduler.from_config(cfg)
    return sch, cfg_path


def cfg_to_plain_dict(cfg_obj) -> dict:
    # diffusers config might be FrozenDict or plain dict
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    if hasattr(cfg_obj, "to_dict"):
        return cfg_obj.to_dict()
    # FrozenDict in diffusers often behaves like dict
    try:
        return dict(cfg_obj)
    except Exception:
        return {k: getattr(cfg_obj, k) for k in dir(cfg_obj) if not k.startswith("_")}


def assert_same_scheduler(a: DDPMScheduler, b: DDPMScheduler, nameA="A", nameB="B"):
    da = cfg_to_plain_dict(a.config)
    db = cfg_to_plain_dict(b.config)
    # ignore name/path-like keys if present
    for k in list(da.keys()):
        if k in ["_name_or_path"]:
            da.pop(k, None)
    for k in list(db.keys()):
        if k in ["_name_or_path"]:
            db.pop(k, None)
    if da != db:
        # Print compact hint
        bad = []
        for k in sorted(set(da.keys()).union(db.keys())):
            if da.get(k) != db.get(k):
                bad.append((k, da.get(k), db.get(k)))
        msg = "\n".join([f"{k}: {va} != {vb}" for k, va, vb in bad[:50]])
        raise AssertionError(f"[MISMATCH] Scheduler config differs ({nameA} vs {nameB})\n{msg}")


def load_grid_png_as_batch(
    png_path: str,
    image_size: int,
    channels: int,
    grid_nrow: int,
    grid_padding: int,
) -> torch.Tensor:
    """
    Load a grid PNG (made by torchvision.make_grid with nrow and padding)
    and recover a batch tensor (N,C,H,W) in [-1,1].

    Assumes each tile is exactly (image_size x image_size).
    """
    img = Image.open(png_path).convert("RGB" if channels == 3 else "L")
    x = to_tensor(img)  # (C,H,W) in [0,1]

    C, Htot, Wtot = x.shape
    tile = image_size
    pad = grid_padding

    # infer rows/cols
    ncol = (Wtot - pad) // (tile + pad)
    nrow = (Htot - pad) // (tile + pad)

    if grid_nrow is not None and grid_nrow > 0:
        # expected columns in make_grid
        ncol = grid_nrow

    # sanity
    if (Wtot - pad) % (tile + pad) != 0 or (Htot - pad) % (tile + pad) != 0:
        raise ValueError(
            f"Grid geometry mismatch. PNG={png_path} size={Wtot}x{Htot}, "
            f"tile={tile}, pad={pad}. "
            f"Try adjusting --grid_padding or confirm the PNG was made with make_grid."
        )

    tiles = []
    for r in range(nrow):
        for c in range(ncol):
            top = pad + r * (tile + pad)
            left = pad + c * (tile + pad)
            crop = x[:, top : top + tile, left : left + tile]
            if crop.shape[1] != tile or crop.shape[2] != tile:
                continue
            tiles.append(crop)

    if len(tiles) == 0:
        raise RuntimeError(f"Failed to recover tiles from grid PNG: {png_path}")

    batch = torch.stack(tiles, dim=0)  # (N,C,H,W) in [0,1]
    batch = batch * 2.0 - 1.0          # [-1,1]
    return batch


def to_01(x_m11: torch.Tensor) -> torch.Tensor:
    return (x_m11.clamp(-1, 1) + 1) / 2.0


# -------------------------
# Manual DDPM step (epsilon prediction)
# -------------------------
@torch.no_grad()
def q_sample_x_t(scheduler: DDPMScheduler, x0: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
    """
    Forward noising (closed form):
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
    """
    a_bar = scheduler.alphas_cumprod[t].to(x0.device, dtype=torch.float32)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


@torch.no_grad()
def predict_x0_from_eps(scheduler: DDPMScheduler, x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
    a_bar = scheduler.alphas_cumprod[t].to(x_t.device, dtype=torch.float32)
    return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)


@torch.no_grad()
def ddpm_step_manual_eps(
    scheduler: DDPMScheduler,
    x_t: torch.Tensor,
    eps: torch.Tensor,
    t: int,
    z_t: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    One reverse step: x_t -> x_{t-1} for epsilon-prediction DDPM.

    mean = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
    x_{t-1} = mean + sigma_t * z_t  (if t>0)
    """
    betas = scheduler.betas.to(x_t.device, dtype=torch.float32)
    alphas = scheduler.alphas.to(x_t.device, dtype=torch.float32)
    alpha_t = alphas[t]
    beta_t = betas[t]
    a_bar = scheduler.alphas_cumprod[t].to(x_t.device, dtype=torch.float32)

    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - a_bar)) * eps)

    if t == 0:
        return mean

    # variance according to scheduler config
    # use internal helper if available
    if hasattr(scheduler, "_get_variance"):
        var = scheduler._get_variance(t, predicted_variance=None).to(x_t.device, dtype=torch.float32)
    else:
        # fallback to "fixed_small" as common default
        # var = beta_t * (1 - alpha_bar_{t-1})/(1 - alpha_bar_t)
        a_bar_prev = scheduler.alphas_cumprod[t - 1].to(x_t.device, dtype=torch.float32)
        var = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar)

    vtype = getattr(scheduler.config, "variance_type", "fixed_small")
    if isinstance(vtype, str) and vtype.endswith("_log"):
        var = torch.exp(var)

    sigma = torch.sqrt(var).view(1, 1, 1, 1)

    if z_t is None:
        raise ValueError("z_t must be provided for t>0 to enforce shared stochasticity.")
    return mean + sigma * z_t


# -------------------------
# Metrics
# -------------------------
def mse_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a, b, reduction="mean").item()


def ssim_mean(a01: torch.Tensor, b01: torch.Tensor) -> float:
    """
    SSIM on [0,1]. Computes mean over batch. Uses skimage if available.
    """
    if not HAS_SKIMAGE:
        # fallback: a crude proxy using (1 - normalized MSE) (not real SSIM)
        # Better to install skimage.
        return float("nan")

    a = a01.detach().cpu().numpy()
    b = b01.detach().cpu().numpy()
    # a,b: (N,C,H,W)
    vals = []
    for i in range(a.shape[0]):
        # skimage expects HxWxC in [0,1]
        ai = np.transpose(a[i], (1, 2, 0))
        bi = np.transpose(b[i], (1, 2, 0))
        v = sk_ssim(ai, bi, data_range=1.0, channel_axis=2)
        vals.append(v)
    return float(np.mean(vals))


class LPIPSMetric:
    def __init__(self, net="alex"):
        if not HAS_LPIPS:
            raise RuntimeError("lpips not installed. pip install lpips")
        self.fn = lpips.LPIPS(net=net).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, a01: torch.Tensor, b01: torch.Tensor) -> float:
        # LPIPS expects [-1,1]
        a = a01 * 2.0 - 1.0
        b = b01 * 2.0 - 1.0
        d = self.fn(a, b)  # (N,1,1,1)
        return float(d.mean().item())


# -------------------------
# Core experiment
# -------------------------
@dataclass
class TrajResult:
    # snapshot -> x0pred in [0,1] on CPU
    x0pred_snapshots: Dict[int, torch.Tensor]


@torch.no_grad()
def run_one_model_from_input_xt(
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    x_t0: torch.Tensor,                 # (N,C,H,W) in [-1,1] on DEVICE
    t0: int,
    snapshots: Set[int],
    step_noises: Dict[int, torch.Tensor],  # t -> (N,C,H,W) on DEVICE (only for t>0)
) -> TrajResult:
    x = x_t0.clone()

    x0pred_snaps = {}

    for t in range(t0, -1, -1):
        t_tensor = torch.full((x.shape[0],), t, device=DEVICE, dtype=torch.long)
        eps = unet(x, t_tensor).sample  # epsilon prediction

        if t in snapshots:
            x0pred = predict_x0_from_eps(scheduler, x, eps, t)
            x0pred_snaps[t] = to_01(x0pred).detach().cpu()

        z = step_noises.get(t, None) if t > 0 else None
        x = ddpm_step_manual_eps(scheduler, x, eps, t, z)

    return TrajResult(x0pred_snapshots=x0pred_snaps)


def save_grid_batch(batch01: torch.Tensor, out_path: str, nrow: int = 8, padding: int = 2):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid = make_grid(batch01, nrow=nrow, padding=padding)
    save_image(grid, out_path)


def parse_snapshots(s: str) -> List[int]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        items.append(int(part))
    return sorted(set(items), reverse=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--ckpt_c", required=True)

    ap.add_argument("--png_a", required=True)
    ap.add_argument("--png_b", required=True)
    ap.add_argument("--png_c", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--channels", type=int, default=3)

    ap.add_argument("--grid_nrow", type=int, default=8)
    ap.add_argument("--grid_padding", type=int, default=2)

    ap.add_argument("--t0", type=int, default=800, help="restart timestep")
    ap.add_argument("--num_samples", type=int, default=64, help="how many tiles to use from the grid")
    ap.add_argument("--forward_noise_seed", type=int, default=1234)
    ap.add_argument("--step_seed", type=int, default=777)

    ap.add_argument("--snapshots", type=str, default="800,600,400,200,100,50,10,0")

    ap.add_argument("--prefer_ema", action="store_true")
    ap.add_argument("--lpips_net", type=str, default="alex")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Device:", DEVICE)

    # Load models
    unetA, pA = load_unet(args.ckpt_a, args.prefer_ema)
    unetB, pB = load_unet(args.ckpt_b, args.prefer_ema)
    unetC, pC = load_unet(args.ckpt_c, args.prefer_ema)
    print("UNet A:", pA)
    print("UNet B:", pB)
    print("UNet C:", pC)

    # Load schedulers (and assert same)
    schA, sA = load_scheduler_from_ckpt(args.ckpt_a)
    schB, sB = load_scheduler_from_ckpt(args.ckpt_b)
    schC, sC = load_scheduler_from_ckpt(args.ckpt_c)
    print("Scheduler A:", sA)
    print("Scheduler B:", sB)
    print("Scheduler C:", sC)

    assert_same_scheduler(schA, schB, "A", "B")
    assert_same_scheduler(schA, schC, "A", "C")

    scheduler = schA.to(DEVICE)

    # Load input PNG grids -> batch x0pred in [-1,1]
    x0A = load_grid_png_as_batch(args.png_a, args.image_size, args.channels, args.grid_nrow, args.grid_padding)
    x0B = load_grid_png_as_batch(args.png_b, args.image_size, args.channels, args.grid_nrow, args.grid_padding)
    x0C = load_grid_png_as_batch(args.png_c, args.image_size, args.channels, args.grid_nrow, args.grid_padding)

    # truncate to num_samples
    x0A = x0A[: args.num_samples]
    x0B = x0B[: args.num_samples]
    x0C = x0C[: args.num_samples]

    print("Recovered tiles:", x0A.shape, x0B.shape, x0C.shape)
    print("Input A sha256:", sha256_tensor(x0A))
    print("Input B sha256:", sha256_tensor(x0B))
    print("Input C sha256:", sha256_tensor(x0C))

    # Prepare shared forward noise (same noise tensor for ALL three inputs for fairness)
    g_fwd = torch.Generator(device="cpu").manual_seed(args.forward_noise_seed)
    noise_shared = torch.randn(x0A.shape, generator=g_fwd, device="cpu", dtype=torch.float32)  # (N,C,H,W)

    # Create x_t0 for each input via forward diffusion
    t0 = args.t0
    x0A_d = x0A.to(DEVICE)
    x0B_d = x0B.to(DEVICE)
    x0C_d = x0C.to(DEVICE)
    noise_d = noise_shared.to(DEVICE)

    xA_t0 = q_sample_x_t(scheduler, x0A_d, t0, noise_d)
    xB_t0 = q_sample_x_t(scheduler, x0B_d, t0, noise_d)
    xC_t0 = q_sample_x_t(scheduler, x0C_d, t0, noise_d)

    print(f"t0={t0} x_t0 sha256(A/B/C):",
          sha256_tensor(xA_t0), sha256_tensor(xB_t0), sha256_tensor(xC_t0))

    # Prepare shared step noises z_t for t in [t0..1]
    g_step = torch.Generator(device="cpu").manual_seed(args.step_seed)
    step_noises = {}
    for t in range(t0, 0, -1):
        z = torch.randn(xA_t0.shape, generator=g_step, device="cpu", dtype=torch.float32)
        step_noises[t] = z.to(DEVICE)

    snapshots = set(parse_snapshots(args.snapshots))
    print("Snapshots:", sorted(snapshots, reverse=True))

    # Metrics
    lpips_fn = LPIPSMetric(net=args.lpips_net) if HAS_LPIPS else None
    if not HAS_LPIPS:
        print("[WARN] lpips not installed -> LPIPS will be NaN")
    if not HAS_SKIMAGE:
        print("[WARN] scikit-image not installed -> SSIM will be NaN (install scikit-image)")

    # Run experiment for each input (A/B/C) and each model (A/B/C)
    inputs = [("inputA", xA_t0, x0A), ("inputB", xB_t0, x0B), ("inputC", xC_t0, x0C)]
    models = [("A", unetA), ("B", unetB), ("C", unetC)]

    import csv

    for in_name, x_t0, x0_in in inputs:
        subdir = os.path.join(args.out_dir, in_name)
        os.makedirs(subdir, exist_ok=True)

        # Save the input grid (what we are conditioning on)
        save_grid_batch(to_01(x0_in), os.path.join(subdir, f"{in_name}_x0_input_grid.png"), nrow=args.grid_nrow, padding=args.grid_padding)

        # Run all models
        results = {}
        for mname, unet in models:
            res = run_one_model_from_input_xt(
                unet=unet,
                scheduler=scheduler,
                x_t0=x_t0,
                t0=t0,
                snapshots=snapshots,
                step_noises=step_noises,
            )
            results[mname] = res

            # Save snapshot grids per model
            for t in sorted(snapshots, reverse=True):
                if t not in res.x0pred_snapshots:
                    continue
                outp = os.path.join(subdir, f"t{t:04d}_{in_name}_model{mname}_x0pred_grid.png")
                save_grid_batch(res.x0pred_snapshots[t], outp, nrow=args.grid_nrow, padding=args.grid_padding)

        # Compute pairwise metrics per snapshot
        out_csv = os.path.join(subdir, "xt_summary.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t",
                "mse_AB_mean", "mse_AC_mean", "mse_BC_mean",
                "ssim_AB_mean", "ssim_AC_mean", "ssim_BC_mean",
                "lpips_AB_mean", "lpips_AC_mean", "lpips_BC_mean",
            ])

            for t in sorted(snapshots, reverse=True):
                A = results["A"].x0pred_snapshots[t]
                B = results["B"].x0pred_snapshots[t]
                C = results["C"].x0pred_snapshots[t]

                mseAB = mse_mean(A, B)
                mseAC = mse_mean(A, C)
                mseBC = mse_mean(B, C)

                ssimAB = ssim_mean(A, B)
                ssimAC = ssim_mean(A, C)
                ssimBC = ssim_mean(B, C)

                if lpips_fn is None:
                    lpAB = lpAC = lpBC = float("nan")
                else:
                    lpAB = lpips_fn(A.to(DEVICE), B.to(DEVICE))
                    lpAC = lpips_fn(A.to(DEVICE), C.to(DEVICE))
                    lpBC = lpips_fn(B.to(DEVICE), C.to(DEVICE))

                writer.writerow([t, mseAB, mseAC, mseBC, ssimAB, ssimAC, ssimBC, lpAB, lpAC, lpBC])

        print("Saved summary:", out_csv)

        # Plot curves (matplotlib, one figure each)
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            df = pd.read_csv(out_csv)
            df = df.sort_values("t", ascending=False)

            def plot_one(ycols, title, ylabel, fname):
                plt.figure()
                x = df["t"].to_numpy()
                for c in ycols:
                    plt.plot(x, df[c].to_numpy(), label=c.replace("_mean", ""))
                plt.title(title)
                plt.xlabel("timestep t (higher = noisier, lower = closer to x0)")
                plt.ylabel(ylabel)
                plt.gca().invert_xaxis()
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(subdir, fname), dpi=200)
                plt.close()

            plot_one(["mse_AB_mean", "mse_AC_mean", "mse_BC_mean"],
                     "Cross-input: x0_pred divergence (MSE)", "MSE(x0_pred)", "curve_mse_x0pred.png")
            plot_one(["ssim_AB_mean", "ssim_AC_mean", "ssim_BC_mean"],
                     "Cross-input: x0_pred similarity (SSIM)", "SSIM(x0_pred)", "curve_ssim_x0pred.png")
            plot_one(["lpips_AB_mean", "lpips_AC_mean", "lpips_BC_mean"],
                     "Cross-input: x0_pred perceptual distance (LPIPS)", "LPIPS(x0_pred) (lower=more similar)", "curve_lpips_x0pred.png")

            print("Saved plots under:", subdir)

        except Exception as e:
            print("[WARN] plotting failed:", repr(e))

    print("\nDone. Output dir:", args.out_dir)


if __name__ == "__main__":
    main()
