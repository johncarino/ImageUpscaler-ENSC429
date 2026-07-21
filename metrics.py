# -*- coding: utf-8 -*-
"""Image-quality and efficiency metrics for the Super-Resolution project.

This module provides a single, consistent place to compute every metric used in
the evaluation stage, so that baselines, SRCNN, ESRGAN and the DSP-enhanced
variants are all scored the same way (same color space, same alignment, same
data ranges). That fairness is the whole point of the evaluation section.

Quality metrics
---------------
* PSNR  (Peak Signal-to-Noise Ratio)      -> higher is better, in dB
* SSIM  (Structural Similarity)           -> higher is better, in [0, 1]
* LPIPS (Learned Perceptual Image Patch Similarity) -> lower is better

Efficiency metrics
------------------
* Inference / processing time per image (seconds)
* Peak memory used during processing (MB)

Author: John Patrick Carino (DSP enhancement + evaluation module)
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# LPIPS + torch are optional heavy deps; import lazily so the rest of the module
# still works (e.g. on a machine without a GPU / without lpips installed).
_LPIPS_MODEL = None
_LPIPS_DEVICE = None


# ---------------------------------------------------------------------------
# Image loading / alignment helpers
# ---------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """Load an image as uint8 BGR (OpenCV convention)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def match_size(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize ``target`` to match ``reference`` if their shapes differ.

    SR outputs should already match the HR size, but rounding in the LR
    degradation step can produce a 1-2 pixel mismatch that would otherwise
    crash the metric functions. We resize with bicubic to be safe.
    """
    if target.shape[:2] != reference.shape[:2]:
        target = cv2.resize(
            target,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return target


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def compute_psnr(hr: np.ndarray, sr: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher is better)."""
    sr = match_size(hr, sr)
    return float(sk_psnr(hr, sr, data_range=255))


def compute_ssim(hr: np.ndarray, sr: np.ndarray) -> float:
    """Structural Similarity in [0, 1] (higher is better)."""
    sr = match_size(hr, sr)
    return float(
        sk_ssim(hr, sr, data_range=255, channel_axis=2)
    )


def _get_lpips_model(device: Optional[str] = None):
    """Lazily construct (and cache) the LPIPS model."""
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if _LPIPS_MODEL is None:
        import lpips  # local import: heavy dependency
        import torch

        _LPIPS_DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(_LPIPS_DEVICE)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL, _LPIPS_DEVICE


def compute_lpips(hr: np.ndarray, sr: np.ndarray) -> float:
    """Learned Perceptual Image Patch Similarity (lower is better).

    Returns ``nan`` if lpips / torch are unavailable, so evaluation can still
    proceed with PSNR/SSIM only.
    """
    try:
        import torch

        model, device = _get_lpips_model()
        sr = match_size(hr, sr)

        def to_tensor(bgr: np.ndarray) -> "torch.Tensor":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # LPIPS expects values in [-1, 1], shape (1, 3, H, W)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            return (t * 2 - 1).to(device)

        with torch.no_grad():
            d = model(to_tensor(hr), to_tensor(sr))
        return float(d.item())
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[metrics] LPIPS unavailable ({exc}); returning nan.")
        return float("nan")


def compute_quality(hr: np.ndarray, sr: np.ndarray, with_lpips: bool = True) -> dict:
    """Compute all quality metrics for a single HR/SR pair."""
    result = {
        "psnr": compute_psnr(hr, sr),
        "ssim": compute_ssim(hr, sr),
    }
    result["lpips"] = compute_lpips(hr, sr) if with_lpips else float("nan")
    return result


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------
@contextmanager
def measure(profile: dict):
    """Context manager that records wall-clock time and peak memory.

    Usage
    -----
    >>> stats = {}
    >>> with measure(stats):
    ...     do_work()
    >>> stats["time_s"], stats["peak_mem_mb"]
    """
    tracemalloc.start()
    start = time.perf_counter()
    try:
        yield profile
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile["time_s"] = elapsed
        profile["peak_mem_mb"] = peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Simple accumulator for averaging over a dataset
# ---------------------------------------------------------------------------
class MetricAccumulator:
    """Collects per-image metric dicts and reports the mean / std."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, row: dict) -> None:
        self._rows.append(row)

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def summary(self) -> dict:
        """Return {metric: {"mean": .., "std": ..}} across all added rows."""
        if not self._rows:
            return {}
        keys = [k for k in self._rows[0] if isinstance(self._rows[0][k], (int, float))]
        out = {}
        for k in keys:
            vals = np.array([r[k] for r in self._rows], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                out[k] = {"mean": float("nan"), "std": float("nan")}
            else:
                out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        return out
