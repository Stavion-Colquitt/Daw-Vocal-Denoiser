# DAW Vocal Denoiser v1c

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

🤗 [Model on Hugging Face](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c) ·
v1: [Daw-Vocal-Denoiser](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser)

Version 1c of my [DAW Vocal Denoiser](https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser): the same causal,
real-time model (48 kHz, **10 ms algorithmic latency**, 1.25M parameters) retrained with about **four times more of
my own singing**. The question v1c asks: does more in-domain vocal data (my voice, my rooms, my mics) help?

---

## Results (dev set)

| | v1 | v1c |
|---|---|---|
| Own vocal training data | 38.8 min, 8 songs | about 202 min, 38 songs |
| Training steps | 108,773 | 108,779 (47,079 main run + 61,700 extension) |
| Dev SI-SDR gain, published checkpoint | +2.39 dB | **+2.57 dB** |
| Peak dev SI-SDR gain during training | +2.39 dB | +2.59 dB (step 95,000) |

At the same number of training steps, v1c is **+0.18 dB** above v1 on the dev set. The published checkpoint is the
one with the lowest dev loss (step 108,779). The held-out test set (`Say Dat Name` plus held-out VocalSet singers) has
not been run yet; those numbers will be added here.

| Real-time property | Value |
|--------------------|-------|
| Algorithmic latency | 10 ms (32 ms analysis / 10 ms synthesis windows, 5 ms hop) |
| CPU per 5 ms block (ONNX Runtime, desktop CPU) | 0.24 ms (real-time factor 0.048) |
| ONNX vs PyTorch parity | 113.6 dB |
| Parameters | 1,250,563 |

---

## What changed from v1

- **Data:** 1,246 more of my own vocal takes (163.7 min from 30 songs), pulled from the vocal tracks across my
  Ableton sessions, deduplicated, takes under 1 s skipped. The dev song (`Mad`) and test song (`Say Dat Name`) stay
  out of training. Everything else (VocalSet, DNS Challenge speech and noise, headphone bleed from my instrumentals,
  synthetic rooms, hiss and hum) is the same as v1.
- **Training:** a 47,079-step main run on an RTX 4060, then a 61,700-step extension on an RTX 2060 (learning rate
  re-warmed to 3e-4, cosine to 1e-5) so the total matches v1's step count.
- **Model:** unchanged. Asymmetric analysis/synthesis windows (Wang et al. 2021), compressed magnitude ->
  Linear -> LayerNorm -> 2-layer GRU (256) -> sigmoid magnitude mask, noisy phase kept.

---

## Files

| File | What it does |
|------|--------------|
| `model.py` | Model definition and asymmetric-window STFT |
| `stream.py` | Real-time block engine (240-sample blocks in and out) and `process_file` for offline use |
| `dataset.py` | On-the-fly noisy/clean mixing (SNR 0-40 dB, rooms, bleed, hiss, hum) |
| `prep_data.py` | Builds the v1 training set (resampling, splits) |
| `prep_own_extra.py` | Collects the extra v1c vocal takes from my Ableton sessions |
| `train.py` | Training loop (compressed spectral + multi-resolution STFT loss, AdamW, EMA 0.999, MLflow) |
| `run_v1c_ext.ps1` | The extension run (`--resume --extend_steps 61700`), then ONNX export and parity check |
| `export_onnx.py` | Exports the one-frame streaming ONNX model |
| `check_parity.py` | Checks ONNX vs PyTorch and measures CPU time per block |
| `eval.py` | Held-out test evaluation (SI-SDR, PESQ, pass-through) |

The weights (`daw_denoiser.onnx`, `daw_denoiser.pth`) are on
[Hugging Face](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c) and attached to this repo's release.
My own vocal recordings are not included.

## Usage

```python
# pip install onnxruntime numpy
from stream import StreamEngine          # expects daw_denoiser.onnx next to it
eng = StreamEngine()
y_block = eng.process(x_block)           # float32[240] at 48 kHz -> 240 denoised samples, 10 ms later

from stream import process_file          # offline: whole mono 48 kHz array in, denoised array out
y = process_file(x)
```

## Training data and licenses

| Source | License |
|--------|---------|
| My own vocal takes and instrumentals | Mine (not distributed) |
| [VocalSet](https://zenodo.org/records/1442513) (Wilkins et al., ISMIR 2018) | CC BY 4.0 |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband speech and noise (subsets) | Per source, see the DNS repo |

No commercial sample packs. Code is MIT licensed (see `LICENSE`).

## Limits

- For monitoring and cleaning vocal takes (singing or speech) recorded in untreated rooms.
- Removes noise, not reverb. Mono, 48 kHz only (resample other rates first).
- Not tested on heavy distortion or SNR below 0 dB; very loud headphone bleed can survive.
