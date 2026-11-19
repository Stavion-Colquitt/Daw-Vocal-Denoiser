# DAW Vocal Denoiser v1b: a bigger model

**Question:** Does a bigger model help more than more data did?

## What it is

- v1c's data and step count with the GRU widened from 256 to 384 units (2.51M parameters).
- 108,779 steps from scratch, 7.7 hours on an RTX 5070.

## Result

Held-out test (180 clips, identical for every version, see the [main results table](../../README.md#results-held-out-test)):
SI-SDR 23.46 dB | 3.10 | 47.6 dB  (SI-SDR | PESQ-WB | clean pass-through).

The most accurate version so far (best SI-SDR and SDR). Clearly better than v1c on PESQ for all voices, and wins back other singers (+0.38 dB SI-SDR on VocalSet vs v1c, interval above zero). Costs about 2x the CPU per block and touches clean takes a little more (47.6 dB pass-through).

## Use it

- Weights: [Hugging Face: DAW_Vocal_Denoiser_v1b](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1b) (`daw_denoiser.onnx`, `daw_denoiser.pth`)
- Settings for the shared engine: `DAW_HIDDEN=384`, `DAW_LOOKAHEAD=0`
- Docker: `stavion/daw-denoiser:v1b`
- Retrain: `.\run.ps1` in this folder

This version used to be its own repository; its full commit history is part of this repo now.
