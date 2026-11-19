# DAW Vocal Denoiser v1d: 5 ms of lookahead

**Question:** How much quality does 5 ms of lookahead buy?

## What it is

- v1c's exact recipe (GRU 256, same data, 108,779 steps) with one change: each frame's mask is computed after the model hears one more 5 ms hop of input, for 15 ms total latency.
- Training pads the magnitude input by one frame and shifts the mask back one frame; the streaming engine holds each spectrum one hop until its mask arrives. `denoiser/test_lookahead.py` confirms the output depends on input exactly one hop ahead and no further.
- 9.3 hours on an RTX 4060.

## Result

Held-out test (180 clips, identical for every version, see the [main results table](../../README.md#results-held-out-test)):
SI-SDR 23.43 dB | 3.11 | 50.5 dB  (SI-SDR | PESQ-WB | clean pass-through).

5 ms of lookahead buys about as much as doubling the model: tied with v1b on every metric at half its CPU cost and file size, with clean pass-through close to v1c (50.5 dB). The price is 15 ms latency instead of 10.

## Use it

- Weights: [Hugging Face: DAW_Vocal_Denoiser_v1d](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1d) (`daw_denoiser.onnx`, `daw_denoiser.pth`)
- Settings for the shared engine: `DAW_HIDDEN=256`, `DAW_LOOKAHEAD=1`
- Docker: `stavion/daw-denoiser:v1d`
- Retrain: `.\run.ps1` in this folder

This version used to be its own repository; its full commit history is part of this repo now.
