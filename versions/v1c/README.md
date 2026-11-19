# DAW Vocal Denoiser v1c: more of my own singing

**Question:** Does more in-domain vocal data (my voice, my rooms, my mics) help?

## What it is

- Same model as v1.
- Data: v1 plus 1,246 more of my own vocal takes (163.7 minutes from 30 songs), pulled from my Ableton sessions. Dev song *Mad* and test song *Say Dat Name* stay out of training.
- 47,079-step main run on an RTX 4060, then a 61,700-step extension on an RTX 2060 (108,779 total).

## Result

Held-out test (180 clips, identical for every version, see the [main results table](../../README.md#results-held-out-test)):
SI-SDR 23.27 dB | 3.03 | 51.1 dB  (SI-SDR | PESQ-WB | clean pass-through).

Specializes to my voice: +0.95 dB SI-SDR over v1 on my vocals, slightly worse on other singers, and the most untouched clean takes (51.1 dB pass-through). Full v1 vs v1c comparison with the by-voice split: `results/compare_v1_v1c.md`.

## Use it

- Weights: [Hugging Face: DAW_Vocal_Denoiser_v1c](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c) (`daw_denoiser.onnx`, `daw_denoiser.pth`)
- Settings for the shared engine: `DAW_HIDDEN=256`, `DAW_LOOKAHEAD=0`
- Docker: `stavion/daw-denoiser:v1c`
- Retrain: `.\run.ps1` in this folder

This version used to be its own repository; its full commit history is part of this repo now.
