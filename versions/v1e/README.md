# DAW Vocal Denoiser v1e: bigger model + 5 ms of lookahead

**Question:** Do a bigger model (v1b) and lookahead (v1d) stack?

## What it is

- v1b's GRU 384 (2.51M parameters) plus v1d's one frame of lookahead (15 ms total latency), on v1c's data (v1 data plus 1,246 extra takes of my own singing), 108,779 steps from scratch.
- Same training code and settings as v1b/v1d; only the data loader got faster (4 workers instead of 2).
- About 9 h on an RTX 4060 across five sessions, Nov 22-23, 2026 (every pause resumed from a saved checkpoint).
- Published checkpoint: step 103,000 (lowest dev loss, dev SI-SDR gain +2.82 dB).

## Result

Held-out test (180 clips, identical for every version, see the [main results table](../../README.md#results-held-out-test)):
SI-SDR 23.45 dB | PESQ-WB 3.12 | ViSQOL 4.00 | clean pass-through 53.1 dB.

**They don't stack on SI-SDR:** v1e ties v1b (-0.01 ± 0.21 dB) and v1d (+0.01 ± 0.19 dB). It does edge ahead on the
perceptual metrics (best PESQ, ViSQOL and MERT-MSE of any version; PESQ +0.02 ± 0.02 over v1b, interval excludes zero)
and has the best clean pass-through of any version (53.1 dB vs 47.6 for v1b, 50.5 for v1d): it changes clean vocals
the least. The cost is v1b's CPU and file size plus v1d's 15 ms latency. Read together with v1b and v1d, this suggests
the GRU-mask design has plateaued around 23.4-23.5 dB SI-SDR on this test; the next gain needs a new idea, not a
bigger or later-looking version of this model.

## Use it

- Weights: [Hugging Face: DAW_Vocal_Denoiser_v1e](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1e) (`daw_denoiser.onnx`, `daw_denoiser.pth`)
- Settings for the shared engine: `DAW_HIDDEN=384`, `DAW_LOOKAHEAD=1`
- Docker: `stavion/daw-denoiser:v1e`
- Retrain: `.\run.ps1` in this folder

The shared engine reproduces this model exactly (streaming vs. PyTorch parity 107.5 dB), and `denoiser/test_lookahead.py`
passes with these settings.
