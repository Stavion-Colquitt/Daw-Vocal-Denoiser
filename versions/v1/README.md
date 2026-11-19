# DAW Vocal Denoiser v1: the baseline

**Question:** Can a 1.25M-parameter causal GRU clean vocal takes in real time at 48 kHz with 10 ms latency, on a CPU?

## What it is

- 2-layer GRU (256), asymmetric 32 ms analysis / 10 ms synthesis windows (Wang et al. 2021), compressed magnitude in, sigmoid magnitude mask out, noisy phase kept.
- Data: 38.8 minutes of my own vocal takes (8 songs), VocalSet (CC BY 4.0), DNS Challenge speech and noise, simulated rooms, hiss, hum and headphone bleed from my instrumentals.
- 108,773 steps (a 9-hour budget) on an RTX 4060 and RTX 2060.

## Result

Held-out test (180 clips, identical for every version, see the [main results table](../../README.md#results-held-out-test)):
SI-SDR 22.95 dB | 3.05 | 45.6 dB  (SI-SDR | PESQ-WB | clean pass-through).

The first published version. Still has the best FAD-MERT (outputs closest to clean vocals as a distribution) and remains a good general-purpose choice for other people's voices.

## Use it

- Weights: [Hugging Face: DAW_Vocal_Denoiser](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser) (`daw_denoiser.onnx`, `daw_denoiser.pth`)
- Settings for the shared engine: `DAW_HIDDEN=256`, `DAW_LOOKAHEAD=0`
- Docker: no published image for v1; build one with `docker/Dockerfile` (default settings)
- Retrain: `.\run.ps1` in this folder

This version used to be its own repository; its full commit history is part of this repo now.
