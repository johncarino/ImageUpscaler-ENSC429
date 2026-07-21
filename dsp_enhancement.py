# -*- coding: utf-8 -*-
"""DSP post-processing enhancement stage for the Super-Resolution project.

This module implements the classical Digital Signal Processing (DSP) stage that
runs *after* an image has been upscaled (by an interpolation baseline, SRCNN, or
ESRGAN). The goal is to recover / emphasize high-frequency detail that upscaling
tends to smooth out, using the filtering theory covered in ENSC 429.

Implemented techniques
----------------------
1. Unsharp masking          -> detail = image - blur(image);  out = image + amount*detail
2. Laplacian high-pass boost -> out = image + alpha * Laplacian(image)
3. Frequency-domain high-pass (FFT) -> attenuate low frequencies, keep edges
4. FFT magnitude-spectrum analysis  -> quantitative "how much high-freq energy"

Everything operates on 8-bit BGR images (OpenCV convention) so it slots directly
into the existing pipeline, which already uses cv2 to load DIV2K images.

Author: John Patrick Carino (DSP enhancement + evaluation module)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core DSP filters
# ---------------------------------------------------------------------------
def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    """Sharpen an image using unsharp masking.

    Unsharp masking builds a "detail" (high-pass) signal by subtracting a
    low-pass (Gaussian-blurred) version of the image from the original, then
    adds a scaled copy of that detail back:

        detail = image - gaussian_blur(image)
        output = image + amount * detail

    This is the spatial-domain dual of boosting high frequencies (see
    lecture 07 - DT Filter Design and lecture 05 - LTI System Analysis).

    Parameters
    ----------
    image : np.ndarray
        Input image, uint8, shape (H, W, C) or (H, W).
    sigma : float
        Standard deviation of the Gaussian low-pass kernel. Larger sigma
        sharpens coarser structure; smaller sigma sharpens fine texture.
    amount : float
        Strength of the sharpening. 0 = no change, 1 = standard, >1 = strong.
    threshold : int
        Only pixels whose local contrast exceeds this value (0-255) are
        sharpened. Helps avoid amplifying noise in flat regions.

    Returns
    -------
    np.ndarray
        Sharpened image, uint8, same shape as input.
    """
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = img - blurred
    sharpened = img + amount * detail

    if threshold > 0:
        low_contrast = np.abs(detail) < threshold
        sharpened = np.where(low_contrast, img, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Sharpen an image by adding a scaled Laplacian (2nd-derivative) high-pass.

    The discrete Laplacian is a high-pass filter; adding it back to the image
    emphasizes edges:

        output = image + alpha * Laplacian(image)

    Parameters
    ----------
    image : np.ndarray
        Input image, uint8.
    alpha : float
        Edge-boost strength. Typical range 0.2 - 1.0.

    Returns
    -------
    np.ndarray
        Sharpened image, uint8.
    """
    img = image.astype(np.float32)
    laplacian = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=3)
    sharpened = img - alpha * laplacian  # minus because cv2 Laplacian sign
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def fft_highpass(image: np.ndarray, cutoff_ratio: float = 0.10, boost: float = 1.5) -> np.ndarray:
    """Boost high spatial frequencies using a frequency-domain (FFT) filter.

    Each color channel is transformed with a 2D FFT. Frequencies inside a
    circular radius around DC (the low frequencies) are left alone, while
    frequencies outside that radius (the high frequencies / detail) are
    multiplied by ``boost``. This is a direct application of the frequency
    response ideas from lecture 05 and the filter-design ideas from lecture 07.

    Parameters
    ----------
    image : np.ndarray
        Input image, uint8, shape (H, W, C) or (H, W).
    cutoff_ratio : float
        Radius of the low-frequency "keep" region as a fraction of the
        smaller image dimension. Smaller = more frequencies get boosted.
    boost : float
        Multiplier applied to the high-frequency region ( >1 sharpens ).

    Returns
    -------
    np.ndarray
        Enhanced image, uint8.
    """
    if image.ndim == 2:
        channels = [image]
    else:
        channels = cv2.split(image)

    h, w = channels[0].shape
    cy, cx = h // 2, w // 2
    radius = int(cutoff_ratio * min(h, w))

    # Build a high-frequency boost mask (1.0 inside low-pass region, `boost` outside).
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = np.where(dist <= radius, 1.0, boost).astype(np.float32)

    out_channels = []
    for ch in channels:
        f = np.fft.fftshift(np.fft.fft2(ch.astype(np.float32)))
        f_filtered = f * mask
        rec = np.fft.ifft2(np.fft.ifftshift(f_filtered))
        out_channels.append(np.clip(np.abs(rec), 0, 255))

    if len(out_channels) == 1:
        return out_channels[0].astype(np.uint8)
    return cv2.merge(out_channels).astype(np.uint8)


def denoise_bilateral(image: np.ndarray, d: int = 5, sigma_color: float = 50, sigma_space: float = 5) -> np.ndarray:
    """Edge-preserving denoise (optional pre-sharpen step).

    A bilateral filter smooths flat regions while preserving edges, which is
    useful before sharpening so that noise is not amplified.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ---------------------------------------------------------------------------
# Frequency-domain analysis (for the report / evaluation)
# ---------------------------------------------------------------------------
def fft_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Return the log magnitude spectrum of a (grayscale) image for plotting.

    Useful to *visually* compare how much high-frequency content a method
    produces (blurry baselines have energy concentrated near the center).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    magnitude = 20 * np.log(np.abs(f) + 1.0)
    return magnitude


def high_frequency_energy(image: np.ndarray, cutoff_ratio: float = 0.10) -> float:
    """Fraction of total spectral energy that lives in the high-frequency band.

    A single scalar summarizing image "sharpness" in the frequency domain.
    Higher = more fine detail. This gives a DSP-grounded, model-agnostic
    sharpness metric to complement PSNR/SSIM/LPIPS.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = int(cutoff_ratio * min(h, w))
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    high_mask = dist > radius
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[high_mask].sum() / total)


# ---------------------------------------------------------------------------
# Enhancement pipeline (compose the filters above)
# ---------------------------------------------------------------------------
# Named presets keep experiments fair & reproducible (small, fixed parameter grid).
PRESETS: Dict[str, dict] = {
    "none": {},
    "unsharp": {"method": "unsharp", "sigma": 1.0, "amount": 1.0},
    "unsharp_strong": {"method": "unsharp", "sigma": 1.5, "amount": 1.8},
    "laplacian": {"method": "laplacian", "alpha": 0.5},
    "fft_highpass": {"method": "fft", "cutoff_ratio": 0.10, "boost": 1.5},
    "denoise_unsharp": {"method": "denoise_unsharp", "sigma": 1.0, "amount": 1.0},
}


def enhance(image: np.ndarray, preset: str = "unsharp") -> np.ndarray:
    """Apply a named DSP enhancement preset to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image, uint8 BGR.
    preset : str
        One of the keys in :data:`PRESETS`.

    Returns
    -------
    np.ndarray
        Enhanced image, uint8 BGR.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESETS)}.")

    cfg = PRESETS[preset]
    method = cfg.get("method")

    if method is None:  # "none"
        return image.copy()
    if method == "unsharp":
        return unsharp_mask(image, sigma=cfg["sigma"], amount=cfg["amount"])
    if method == "laplacian":
        return laplacian_sharpen(image, alpha=cfg["alpha"])
    if method == "fft":
        return fft_highpass(image, cutoff_ratio=cfg["cutoff_ratio"], boost=cfg["boost"])
    if method == "denoise_unsharp":
        denoised = denoise_bilateral(image)
        return unsharp_mask(denoised, sigma=cfg["sigma"], amount=cfg["amount"])

    raise ValueError(f"Preset '{preset}' has unknown method '{method}'.")


# ---------------------------------------------------------------------------
# CLI: enhance a whole directory of super-resolved images
# ---------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def enhance_directory(in_dir: str, out_dir: str, preset: str = "unsharp") -> int:
    """Apply a DSP preset to every image in ``in_dir`` and write to ``out_dir``.

    Returns the number of images processed.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        print(f"[dsp] No images found in {in_dir}")
        return 0

    for name in files:
        src = os.path.join(in_dir, name)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[dsp] Skipping unreadable file: {src}")
            continue
        out = enhance(img, preset=preset)
        cv2.imwrite(os.path.join(out_dir, name), out)

    print(f"[dsp] Enhanced {len(files)} images with preset '{preset}' -> {out_dir}")
    return len(files)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply DSP enhancement to super-resolved images.")
    p.add_argument("--in_dir", required=True, help="Directory of input (super-resolved) images.")
    p.add_argument("--out_dir", required=True, help="Directory to write enhanced images.")
    p.add_argument(
        "--preset",
        default="unsharp",
        choices=list(PRESETS.keys()),
        help="DSP enhancement preset to apply.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    enhance_directory(args.in_dir, args.out_dir, preset=args.preset)


if __name__ == "__main__":
    main()
