# -*- coding: utf-8 -*-
"""Evaluation + analysis runner for the Super-Resolution project.

This is the "glue" script for the evaluation stage. It takes the ground-truth
high-resolution (HR) images plus one or more folders of reconstructed images
(baselines, SRCNN, ESRGAN, and their DSP-enhanced variants), scores every image
with the metrics in :mod:`metrics`, and produces report-ready outputs:

    results/
      metrics_per_image.csv   <- one row per (method, image)
      metrics_summary.csv     <- mean / std per method
      plot_quality_vs_time.png
      plot_psnr_vs_lpips.png
      plot_fft_spectrum.png
      qualitative_<img>.png

It can also *apply* a DSP enhancement preset on the fly (``method=dir:preset``),
so you can compare e.g. "SRCNN" vs "SRCNN + unsharp" without pre-generating a
whole extra folder.

Examples
--------
Score three method folders against HR:

    python evaluate.py --hr_dir ./data/DIV2K/DIV2K_valid_HR --out_dir ./results \
        --method bicubic=./results/bicubic \
        --method srcnn=./results/srcnn \
        --method esrgan=./results/esrgan

Add DSP-enhanced variants on the fly (dir:preset):

    python evaluate.py --hr_dir ./data/DIV2K/DIV2K_valid_HR --out_dir ./results \
        --method srcnn=./results/srcnn \
        --method "srcnn+unsharp=./results/srcnn:unsharp" \
        --method "srcnn+fft=./results/srcnn:fft_highpass"

Author: John Patrick Carino (DSP enhancement + evaluation module)
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import dsp_enhancement as dsp
import metrics as M

# matplotlib is only needed for the plots; keep import local-friendly.
import matplotlib

matplotlib.use("Agg")  # headless-safe (works in Colab / servers)
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Method specification parsing
# ---------------------------------------------------------------------------
def parse_method_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    """Parse a ``name=dir`` or ``name=dir:preset`` CLI method spec.

    Returns ``(name, directory, preset_or_None)``.
    """
    if "=" not in spec:
        raise ValueError(
            f"Method spec '{spec}' must look like name=dir or name=dir:preset"
        )
    name, rhs = spec.split("=", 1)
    preset = None
    directory = rhs
    # Allow an optional ':preset' suffix, but don't confuse it with Windows
    # drive letters like C:\...  -> only split on the LAST colon if what
    # follows is a known preset name.
    if ":" in rhs:
        head, tail = rhs.rsplit(":", 1)
        if tail in dsp.PRESETS:
            directory, preset = head, tail
    return name.strip(), directory.strip(), preset


def list_images(directory: str) -> List[str]:
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTS))


def match_filename(hr_name: str, sr_files: List[str]) -> Optional[str]:
    """Match an HR filename to an SR filename by stem (ignores suffixes).

    DIV2K HR is e.g. ``0801.png`` while an SR output might be ``0801.png`` or
    ``0801_SRF_4.png``; we match on the leading numeric/stem token.
    """
    hr_stem = os.path.splitext(hr_name)[0]
    # exact stem
    for f in sr_files:
        if os.path.splitext(f)[0] == hr_stem:
            return f
    # stem-prefix (e.g. 0801_x4.png)
    for f in sr_files:
        if os.path.splitext(f)[0].startswith(hr_stem):
            return f
    return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate_method(
    name: str,
    directory: str,
    hr_dir: str,
    preset: Optional[str],
    with_lpips: bool,
    max_images: Optional[int],
) -> List[dict]:
    """Score every image of one method (optionally DSP-enhanced) against HR."""
    hr_files = list_images(hr_dir)
    sr_files = list_images(directory)
    if max_images is not None:
        hr_files = hr_files[:max_images]

    rows: List[dict] = []
    for hr_name in hr_files:
        sr_name = match_filename(hr_name, sr_files)
        if sr_name is None:
            print(f"[eval] {name}: no match for {hr_name}, skipping")
            continue

        hr = M.load_image(os.path.join(hr_dir, hr_name))
        sr = M.load_image(os.path.join(directory, sr_name))

        # Optional on-the-fly DSP enhancement, timed so we capture its cost.
        profile: dict = {}
        with M.measure(profile):
            if preset is not None and preset != "none":
                sr = dsp.enhance(sr, preset=preset)
        # If no DSP was applied, the timing reflects a no-op (near zero).

        quality = M.compute_quality(hr, sr, with_lpips=with_lpips)
        hf_energy = dsp.high_frequency_energy(sr)

        rows.append(
            {
                "method": name,
                "image": hr_name,
                "psnr": quality["psnr"],
                "ssim": quality["ssim"],
                "lpips": quality["lpips"],
                "hf_energy": hf_energy,
                "dsp_time_s": profile.get("time_s", 0.0),
                "dsp_peak_mem_mb": profile.get("peak_mem_mb", 0.0),
            }
        )

    print(f"[eval] {name}: scored {len(rows)} images"
          + (f" (DSP preset '{preset}')" if preset else ""))
    return rows


def summarize(all_rows: List[dict]) -> Dict[str, dict]:
    """Aggregate per-image rows into per-method mean/std."""
    summary: Dict[str, dict] = {}
    methods = sorted({r["method"] for r in all_rows})
    for m in methods:
        acc = M.MetricAccumulator()
        for r in all_rows:
            if r["method"] == m:
                acc.add({k: v for k, v in r.items() if k not in ("method", "image")})
        summary[m] = acc.summary()
    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_per_image_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {path}")


def write_summary_csv(summary: Dict[str, dict], path: str) -> None:
    metric_keys = ["psnr", "ssim", "lpips", "hf_energy", "dsp_time_s", "dsp_peak_mem_mb"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method"]
        for k in metric_keys:
            header += [f"{k}_mean", f"{k}_std"]
        writer.writerow(header)
        for method, stats in summary.items():
            row = [method]
            for k in metric_keys:
                s = stats.get(k, {"mean": float("nan"), "std": float("nan")})
                row += [f"{s['mean']:.4f}", f"{s['std']:.4f}"]
            writer.writerow(row)
    print(f"[eval] wrote {path}")


def print_summary_table(summary: Dict[str, dict]) -> None:
    print("\n=== Summary (mean over dataset) ===")
    header = f"{'method':<18}{'PSNR(dB)':>10}{'SSIM':>8}{'LPIPS':>8}{'HF-energy':>11}"
    print(header)
    print("-" * len(header))
    for method, stats in summary.items():
        psnr = stats.get("psnr", {}).get("mean", float("nan"))
        ssim = stats.get("ssim", {}).get("mean", float("nan"))
        lpips = stats.get("lpips", {}).get("mean", float("nan"))
        hf = stats.get("hf_energy", {}).get("mean", float("nan"))
        print(f"{method:<18}{psnr:>10.3f}{ssim:>8.4f}{lpips:>8.4f}{hf:>11.4f}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_psnr_vs_lpips(summary: Dict[str, dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, stats in summary.items():
        psnr = stats.get("psnr", {}).get("mean", float("nan"))
        lpips = stats.get("lpips", {}).get("mean", float("nan"))
        ax.scatter(psnr, lpips, s=80)
        ax.annotate(method, (psnr, lpips), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("PSNR (dB)  ->  higher is better")
    ax.set_ylabel("LPIPS  ->  lower is better")
    ax.set_title("Perceptual (LPIPS) vs Pixel (PSNR) quality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] wrote {path}")


def plot_quality_vs_hf(summary: Dict[str, dict], path: str) -> None:
    """Bar chart: SSIM and high-frequency energy per method (sharpness view)."""
    methods = list(summary.keys())
    ssim = [summary[m].get("ssim", {}).get("mean", float("nan")) for m in methods]
    hf = [summary[m].get("hf_energy", {}).get("mean", float("nan")) for m in methods]

    x = np.arange(len(methods))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(max(7, len(methods) * 1.3), 5))
    ax1.bar(x - width / 2, ssim, width, label="SSIM", color="#4c72b0")
    ax1.set_ylabel("SSIM", color="#4c72b0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, hf, width, label="HF energy", color="#dd8452")
    ax2.set_ylabel("High-frequency energy", color="#dd8452")

    ax1.set_title("Structural similarity vs high-frequency detail")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] wrote {path}")


def plot_fft_spectrum(
    method_dirs: List[Tuple[str, str, Optional[str]]],
    sample_image: Optional[str],
    path: str,
) -> None:
    """Show the FFT log-magnitude spectrum of one sample image per method."""
    n = len(method_dirs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, directory, preset) in zip(axes, method_dirs):
        files = list_images(directory)
        if not files:
            continue
        target = sample_image if sample_image in files else files[0]
        img = M.load_image(os.path.join(directory, target))
        if preset:
            img = dsp.enhance(img, preset=preset)
        spec = dsp.fft_magnitude_spectrum(img)
        ax.imshow(spec, cmap="viridis")
        ax.set_title(f"{name}\n(FFT spectrum)")
        ax.axis("off")
    fig.suptitle("Frequency content: brighter edges = more high-frequency detail")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate & analyze SR methods (with optional DSP enhancement).",
    )
    p.add_argument("--hr_dir", required=True, help="Directory of ground-truth HR images.")
    p.add_argument("--out_dir", default="./results", help="Where to write metrics + plots.")
    p.add_argument(
        "--method",
        action="append",
        required=True,
        dest="methods",
        help="Method spec 'name=dir' or 'name=dir:preset'. Repeatable.",
    )
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS (faster, no torch).")
    p.add_argument("--max_images", type=int, default=None, help="Limit #images (quick test).")
    p.add_argument("--fft_sample", default=None, help="Filename to use for the FFT panel.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    specs = [parse_method_spec(s) for s in args.methods]

    all_rows: List[dict] = []
    for name, directory, preset in specs:
        if not os.path.isdir(directory):
            print(f"[eval] WARNING: directory not found for '{name}': {directory}")
            continue
        all_rows.extend(
            evaluate_method(
                name=name,
                directory=directory,
                hr_dir=args.hr_dir,
                preset=preset,
                with_lpips=not args.no_lpips,
                max_images=args.max_images,
            )
        )

    if not all_rows:
        print("[eval] No results produced. Check your --method paths.")
        return

    summary = summarize(all_rows)

    # Tables
    write_per_image_csv(all_rows, os.path.join(args.out_dir, "metrics_per_image.csv"))
    write_summary_csv(summary, os.path.join(args.out_dir, "metrics_summary.csv"))
    print_summary_table(summary)

    # Plots
    plot_psnr_vs_lpips(summary, os.path.join(args.out_dir, "plot_psnr_vs_lpips.png"))
    plot_quality_vs_hf(summary, os.path.join(args.out_dir, "plot_quality_vs_hf.png"))
    valid_specs = [(n, d, p) for (n, d, p) in specs if os.path.isdir(d)]
    if valid_specs:
        plot_fft_spectrum(
            valid_specs, args.fft_sample, os.path.join(args.out_dir, "plot_fft_spectrum.png")
        )


if __name__ == "__main__":
    main()
