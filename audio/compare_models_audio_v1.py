"""
compare_models_audio_v1.py

두 독립적으로 학습된 오디오 DDPM 모델(예: seed0 vs seed42)에서
동일한 초기 가우시안 노이즈(starting seed)로 DDIM 샘플링을 수행하고,
최종 출력(x_0)을 LPIPS(spectrogram 공간)와 LPAPS(waveform 공간)로 비교한다.

핵심 설계:
  - 생성된 x_0는 .pt로 캐싱 → 재실행 시 생성 단계를 건너뜀 (resume 기능)
  - 두 모델 모두 캐시된 경우에만 해당 seed를 skip
  - 모델 경로는 HF repo ID 또는 로컬 경로 모두 지원

현재 단계 (Step 1-A/B/C):
  - tau=0 (완전 디노이즈된 최종 출력 x_0)만 비교
  - tau별 중간 상태 비교는 아래 주석처리된 섹션 참고 (Step 2를 진행시 저 주석만 해제하면 된다.)
 - 스탭 1에서 샘플 1개끼리 비교 리뷰가 온전하게 끝났을 경우 추가 샘플링은 아래 커맨드에서 n_samples의 숫자를 바꾸면 된다.

Usage:
    python compare_models_audio_v1.py \\
        --model_a researchdiffusion/sc09-baseline-seed0 \\
        --model_b researchdiffusion/sc09-baseline-seed42 \\
        --n_samples 1 \\
        --cache_dir ./cache \\
        --out_dir ./results \\
        --hop_length 250 --sample_rate 16000 --n_fft 2048 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import csv
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DModel
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm.auto import tqdm

# LPAPS용 CLAP import
# 아래 패키지 미설치 시 실행 불가, 먼저 설치 필요:
#   pip install laion-clap
import laion_clap

# Mel → waveform 역변환용
from diffusers.pipelines.audio_diffusion import Mel


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------

def load_unet_and_scheduler(model_id: str, device: str):
    """HF repo ID 또는 로컬 경로에서 UNet2DModel을 로드하고
    DDIMScheduler(eta=0, 확정적)를 생성한다.
    """
    # 로컬 경로이면 unet/ 서브폴더에서, HF repo이면 바로 from_pretrained
    if os.path.isdir(model_id):
        unet_path = os.path.join(model_id, "unet")
    else:
        unet_path = model_id  # HF repo의 경우 from_pretrained가 unet 서브폴더를 처리

    # 로컬이면 unet/ 서브폴더 직접 지정, HF repo이면 subfolder 인자 사용
    if os.path.isdir(model_id):
        model = UNet2DModel.from_pretrained(unet_path)
    else:
        model = UNet2DModel.from_pretrained(model_id, subfolder="unet")

    model.to(device)
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
    )
    return model, scheduler


# ---------------------------------------------------------------------------
# 캐시 유틸
# ---------------------------------------------------------------------------

def cache_path(cache_dir: str, model_label: str, seed: int) -> str:
    return os.path.join(cache_dir, model_label, f"seed_{seed:05d}.pt")


def is_cached(cache_dir: str, model_label: str, seed: int) -> bool:
    return os.path.exists(cache_path(cache_dir, model_label, seed))


def save_to_cache(tensor: torch.Tensor, cache_dir: str, model_label: str, seed: int):
    path = cache_path(cache_dir, model_label, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)


def load_from_cache(cache_dir: str, model_label: str, seed: int) -> torch.Tensor:
    return torch.load(cache_path(cache_dir, model_label, seed), map_location="cpu")


# ---------------------------------------------------------------------------
# 샘플 생성 (DDIM, x_0까지 완전 디노이즈, 모든 생성 코드는 @torch.no_grad()로 쓴다.)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_x0(
    model,
    scheduler,
    seed: int,
    device: str,
    sample_shape: tuple,
    # -----------------------------------------------------------------------
    # [Step 2용 — 현재 주석처리]
    # tau_list: List[int] = None,  # 중간 tau에서도 저장하려면 활성화
    # -----------------------------------------------------------------------
) -> torch.Tensor:
    """단일 seed에서 DDIM reverse diffusion을 x_0까지 돌려 최종 출력을 반환.

    Returns:
        x_0: tensor of shape [C, H, W] (CPU)
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(sample_shape, generator=generator, device=device)
    x = x.unsqueeze(0)  # [1, C, H, W]

    # -----------------------------------------------------------------------
    # [Step 2용 — 현재 주석처리]
    # tau_captures = {}  # {tau_int: tensor[C, H, W]}
    # taus_remaining = set(tau_list) if tau_list else set()
    # -----------------------------------------------------------------------

    for t in scheduler.timesteps:
        t_int = int(t)

        # -------------------------------------------------------------------
        # [Step 2용 — 현재 주석처리]
        # if t_int in taus_remaining:
        #     tau_captures[t_int] = x.squeeze(0).detach().cpu().clone()
        #     taus_remaining.discard(t_int)
        # -------------------------------------------------------------------

        t_batch = torch.tensor(t_int, device=device)
        noise_pred = model(x, t_batch).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    x_0 = x.squeeze(0).detach().cpu()

    # -----------------------------------------------------------------------
    # [Step 2용 — 현재 주석처리]
    # return x_0, tau_captures
    # -----------------------------------------------------------------------

    return x_0


# ---------------------------------------------------------------------------
# LPIPS 계산
# ---------------------------------------------------------------------------

def compute_lpips(
    tensors_a: List[torch.Tensor],
    tensors_b: List[torch.Tensor],
    device: str,
) -> Tuple[float, float]:
    """Spectrogram 공간에서 pairwise LPIPS를 계산한다.

    입력 텐서 shape: [1, H, W], 값 범위: 약 [-1.5, 1.5]
    LPIPS는 3채널 입력을 기대하므로 채널을 1→3으로 복제.
    normalize=True로 [0,1] 범위로 자동 변환.
    """
    lpips_fn = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)

    scores = []
    for ta, tb in zip(tensors_a, tensors_b):
        # [1, H, W] → [1, 3, H, W], 값을 [0,1]로 clamp
        ta_img = ta.unsqueeze(0).repeat(1, 3, 1, 1).clamp(-1, 1)
        tb_img = tb.unsqueeze(0).repeat(1, 3, 1, 1).clamp(-1, 1)
        # normalize=True는 [-1,1] 입력을 기대 — clamp 후 그대로 전달
        score = lpips_fn(ta_img.to(device), tb_img.to(device))
        scores.append(score.item())

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Mel 역정규화 + waveform 변환 (LPAPS용)
# ---------------------------------------------------------------------------

def tensor_to_waveform(
    tensor: torch.Tensor,
    mel: Mel,
) -> np.ndarray:
    """[1, H, W] tensor → waveform (numpy array).

    학습 시 Normalize([0.5],[0.5])로 정규화됐으므로
    역정규화: x_orig = x * 0.5 + 0.5 → [0,1] → [0,255] uint8 → PIL → audio
    """
    from PIL import Image

    arr = tensor.squeeze(0).numpy()       # [H, W]
    arr = arr * 0.5 + 0.5                  # 역정규화 → [0,1] 근사
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255).astype(np.uint8)
    image = Image.fromarray(arr_uint8, mode="L")
    audio = mel.image_to_audio(image)
    return audio


# ---------------------------------------------------------------------------
# LPAPS 계산
# ---------------------------------------------------------------------------

def compute_lpaps(
    tensors_a: List[torch.Tensor],
    tensors_b: List[torch.Tensor],
    mel: Mel,
    sample_rate: int,
    clap_model,
) -> Tuple[float, float]:
    """Waveform 공간에서 CLAP 임베딩 기반 코사인 거리(LPAPS)를 계산한다.
    LPAPS = 1 - cosine_similarity(CLAP(w_a), CLAP(w_b))
    값이 0이면 완전히 같음, 1이면 완전히 다름.
    """
    scores = []
    for ta, tb in zip(tensors_a, tensors_b):
        wav_a = tensor_to_waveform(ta, mel).astype(np.float32)
        wav_b = tensor_to_waveform(tb, mel).astype(np.float32)

        emb_a = clap_model.get_audio_embedding_from_data(
            x=[wav_a], use_tensor=False
        )
        emb_b = clap_model.get_audio_embedding_from_data(
            x=[wav_b], use_tensor=False
        )

        emb_a = torch.tensor(emb_a)
        emb_b = torch.tensor(emb_b)
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1).item()
        scores.append(1.0 - cos_sim)

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="두 오디오 DDPM 모델의 최종 출력을 LPIPS/LPAPS로 비교. "
        "생성된 샘플은 .pt로 캐싱되며 resume 지원."
    )
    ap.add_argument("--model_a", type=str, required=True,
                    help="모델 A: 로컬 경로")
    ap.add_argument("--model_b", type=str, required=True,
                    help="모델 B: 로컬 경로")
    ap.add_argument("--n_samples", type=int, required=True,
                    help="비교할 샘플 수 (starting seed 0부터 n_samples-1까지)")
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="생성된 x_0 텐서를 저장할 캐시 디렉토리")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="결과 CSV를 저장할 디렉토리")
    ap.add_argument("--hop_length", type=int, required=True)
    ap.add_argument("--sample_rate", type=int, required=True)
    ap.add_argument("--n_fft", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip_lpaps", action="store_true",
                    help="LPAPS 계산을 건너뜀 (CLAP 미설치 시 자동 활성화)")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    skip_lpaps = args.skip_lpaps

    # 캐시 레이블 자동 추론 (경로 또는 HF repo ID의 마지막 부분)
    label_a = args.model_a.rstrip("/").split("/")[-1]
    label_b = args.model_b.rstrip("/").split("/")[-1]

    # 모델 로드
    print(f"[모델 A 로딩] {args.model_a}")
    model_a, scheduler_a = load_unet_and_scheduler(args.model_a, args.device)
    scheduler_a.set_timesteps(scheduler_a.config.num_train_timesteps)

    print(f"[모델 B 로딩] {args.model_b}")
    model_b, scheduler_b = load_unet_and_scheduler(args.model_b, args.device)
    scheduler_b.set_timesteps(scheduler_b.config.num_train_timesteps)

    # 샘플 shape 추론
    inferred_size = model_a.config.sample_size
    if isinstance(inferred_size, (tuple, list)):
        height, width = inferred_size
    else:
        height, width = inferred_size, inferred_size
    channels = model_a.config.in_channels
    sample_shape = (channels, height, width)
    print(f"[샘플 shape] channels={channels}, height={height}, width={width}")

    # Mel 초기화 (LPAPS용 waveform 변환)
    mel = Mel(
        x_res=width,
        y_res=height,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

    # CLAP 로드 (LPAPS용)
    clap_model = None
    if not skip_lpaps:
        print("[CLAP 로딩]")
        import laion_clap
        clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        clap_model.load_ckpt()

    # 샘플 생성 + 캐싱 (resume 지원)
    seeds = list(range(args.n_samples))
    tensors_a, tensors_b = [], []

    print(f"\n[샘플 생성 / 캐시 로드] n_samples={args.n_samples}")
    for seed in tqdm(seeds, desc="seeds", unit="seed"):
        # 모델 A
        if is_cached(args.cache_dir, label_a, seed):
            ta = load_from_cache(args.cache_dir, label_a, seed)
        else:
            ta = generate_x0(model_a, scheduler_a, seed, args.device, sample_shape)
            save_to_cache(ta, args.cache_dir, label_a, seed)

        # 모델 B
        if is_cached(args.cache_dir, label_b, seed):
            tb = load_from_cache(args.cache_dir, label_b, seed)
        else:
            tb = generate_x0(model_b, scheduler_b, seed, args.device, sample_shape)
            save_to_cache(tb, args.cache_dir, label_b, seed)

        tensors_a.append(ta)
        tensors_b.append(tb)

    # LPIPS 계산
    print(f"\n[LPIPS 계산] n={args.n_samples}")
    mean_lpips, std_lpips = compute_lpips(tensors_a, tensors_b, args.device)
    print(f"  LPIPS mean={mean_lpips:.4f}, std={std_lpips:.4f}")

    # LPAPS 계산
    mean_lpaps, std_lpaps = float("nan"), float("nan")
    if not skip_lpaps:
        print(f"[LPAPS 계산] n={args.n_samples}")
        mean_lpaps, std_lpaps = compute_lpaps(
            tensors_a, tensors_b, mel, args.sample_rate, clap_model
        )
        print(f"  LPAPS mean={mean_lpaps:.4f}, std={std_lpaps:.4f}")
    else:
        print("[LPAPS 건너뜀]")

    # CSV 저장
    # -----------------------------------------------------------------------
    # [Step 2용 — tau별 결과를 CSV에 추가하려면 아래 구조를 확장]
    # 현재는 tau=0 (x_0, 완전 디노이즈) 단일 행만 저장
    # -----------------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, f"results_n{args.n_samples}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_samples", "tau", "mean_lpips", "std_lpips",
                        "mean_lpaps", "std_lpaps"]
        )
        writer.writeheader()
        writer.writerow({
            "n_samples": args.n_samples,
            "tau": 0,   # x_0 (완전 디노이즈)
            "mean_lpips": f"{mean_lpips:.6f}",
            "std_lpips": f"{std_lpips:.6f}",
            "mean_lpaps": f"{mean_lpaps:.6f}",
            "std_lpaps": f"{std_lpaps:.6f}",
        })
    print(f"\n[결과 저장] {csv_path}")


if __name__ == "__main__":
    main()
