# DSP Enhancement & Evaluation Module

**Author:** John Patrick Carino
**Role in project:** DSP enhancement module + evaluation metrics and analysis
**Course:** ENSC 429 — Single Image Super-Resolution Using Deep Learning and DSP-Based Enhancement

---

## 1. What this part of the project does

Our group builds a 4× super-resolution (SR) pipeline and compares three kinds of
methods: classical interpolation **baselines** (nearest / bilinear / bicubic), a
**CNN model** (SRCNN), and a **pretrained GAN** (Real-ESRGAN).

My job is the **last two stages** of the pipeline:

1. **DSP enhancement** — a classical signal-processing step that runs *after*
   upscaling to sharpen detail that the upscaling smooths away.
2. **Evaluation & analysis** — score every method the same way (quality +
   speed + memory) and turn the numbers into report-ready tables and plots.

These stages are model-agnostic: they work on any folder of output images, so
they plug into whatever the baseline / SRCNN / ESRGAN stages produce.

```
LR image ──► [ upscale: bicubic / SRCNN / ESRGAN ] ──► SR image
                                                        │
                                     (my part) ─────────┤
                                                        ▼
                                          [ DSP enhancement ]  dsp_enhancement.py
                                                        │
                                                        ▼
                                          [ metrics + analysis ]  metrics.py + evaluate.py
                                                        │
                                                        ▼
                                     CSV tables + PNG plots in ./results
```

---

## 2. Files I added

| File | What it contains |
|------|------------------|
| `dsp_enhancement.py` | The DSP post-processing filters: unsharp masking, Laplacian sharpening, FFT high-pass boost, edge-preserving denoise, plus FFT analysis tools. Runs standalone on a folder of images. |
| `metrics.py` | One consistent place to compute **PSNR**, **SSIM**, **LPIPS**, plus **runtime** and **peak memory**. Handles image loading, size alignment and averaging. |
| `evaluate.py` | The runner that ties it together: scores every method against the HR ground truth, can apply a DSP preset on the fly, and writes CSV tables + comparison plots. |
| `requirements.txt` | The Python packages these scripts need. |

I did **not** change my teammates' files (dataset, training, baseline generation) —
my code consumes their output folders.

---

## 3. The DSP enhancement techniques (and the theory behind them)

All of these come straight from ENSC 429 lecture material. Upscaling acts like a
**low-pass filter** (it blurs), so the DSP stage is about carefully adding back
**high frequencies** without amplifying noise.

| Technique (`preset`) | Idea | ENSC 429 link |
|----------------------|------|----------------|
| `unsharp` / `unsharp_strong` | `out = image + amount·(image − blur(image))`. The `(image − blur)` term is a high-pass (detail) signal. | Filter design (Lec 07), LTI/frequency response (Lec 05) |
| `laplacian` | Add a scaled **Laplacian** (2nd-derivative high-pass) to boost edges. | Convolution / LTI systems (Lec 02), Filter design (Lec 07) |
| `fft_highpass` | Take the **2D FFT**, multiply high-frequency region by a boost factor, inverse FFT. Directly manipulates the spectrum. | Frequency response (Lec 05), complex exponentials / FFT (Lec 02a) |
| `denoise_unsharp` | Edge-preserving **bilateral denoise** first, then sharpen — so sharpening doesn't amplify noise. | Sampling/reconstruction intuition (Lec 04), Filter design (Lec 07) |

I also added a **frequency-domain sharpness metric** (`high_frequency_energy`):
the fraction of spectral energy in the high-frequency band. Blurry baselines have
energy bunched near DC; good SR spreads energy outward. This gives a DSP-grounded,
model-independent sharpness number to back up PSNR/SSIM/LPIPS.

---

## 4. The evaluation metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| **PSNR** (dB) | Pixel-level error vs ground truth | higher |
| **SSIM** | Structural similarity (contrast/structure) | higher (max 1) |
| **LPIPS** | Learned *perceptual* similarity (matches human judgement) | lower |
| **HF energy** | High-frequency content = sharpness (my FFT metric) | higher (up to a point) |
| **Time (s)** | Processing time per image | lower |
| **Peak mem (MB)** | Memory used | lower |

Reporting all of them together is the point of the project: it exposes the
**quality-vs-cost tradeoff** (e.g. ESRGAN looks best perceptually but is slow;
bicubic is instant but blurry; DSP can cheaply push SRCNN closer to ESRGAN).

---

## 5. How to run it

Install dependencies (already available on Colab, but for a fresh machine):

```bash
pip install -r requirements.txt
```

### A) Apply DSP enhancement to a folder of SR images

```bash
python dsp_enhancement.py --in_dir ./results/srcnn --out_dir ./results/srcnn_unsharp --preset unsharp
```

Available presets: `none`, `unsharp`, `unsharp_strong`, `laplacian`,
`fft_highpass`, `denoise_unsharp`.

### B) Evaluate and compare methods

```bash
python evaluate.py --hr_dir ./data/DIV2K/DIV2K_valid_HR --out_dir ./results \
    --method bicubic=./results/bicubic \
    --method srcnn=./results/srcnn \
    --method esrgan=./results/esrgan
```

### C) Compare a method against its DSP-enhanced version (no extra folders needed)

Use the `dir:preset` syntax to enhance on the fly:

```bash
python evaluate.py --hr_dir ./data/DIV2K/DIV2K_valid_HR --out_dir ./results \
    --method srcnn=./results/srcnn \
    --method "srcnn+unsharp=./results/srcnn:unsharp" \
    --method "srcnn+fft=./results/srcnn:fft_highpass"
```

Useful flags: `--no_lpips` (skip LPIPS for a fast run), `--max_images N`
(quick test on N images), `--fft_sample 0801.png` (choose the FFT panel image).

---

## 6. What it produces (in `./results`)

| Output | Description |
|--------|-------------|
| `metrics_per_image.csv` | One row per (method, image) — for detailed / worst-case analysis. |
| `metrics_summary.csv` | Mean ± std of every metric per method — the main results table. |
| `plot_psnr_vs_lpips.png` | Scatter of pixel vs perceptual quality (shows the tradeoff). |
| `plot_quality_vs_hf.png` | SSIM vs high-frequency energy per method (sharpness view). |
| `plot_fft_spectrum.png` | FFT spectra side-by-side — visual proof of who has more detail. |

A summary table is also printed to the console at the end of each run.

---

## 7. How this connects to the lectures (`/lec`)

| Lecture | Used for |
|---------|----------|
| `02-DT_Signals_and_Systems` | Images as 2D discrete signals; **convolution** = filtering (unsharp, Laplacian). |
| `02a-Complex_Numbers_Refresher` / `03-z-Transform` | Foundation for the **FFT** used in `fft_highpass` and the spectrum analysis. |
| `04-Sampling` | Why upscaling loses detail (sampling/reconstruction); motivates the whole DSP stage. |
| `05-LTI_System_Analysis` | **Frequency response** — interpreting what high-pass boosting does. |
| `07-DT_Filter_Design` (+ `07a` examples) | Designing/tuning the **high-pass and sharpening filters** used here. |

**Note:** PSNR/SSIM/LPIPS and the CNN/GAN models themselves are outside the pure
DSP course scope; those come from the SR literature cited in our proposal.
