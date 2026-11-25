# DAW Vocal Denoiser v1_clean: v1 without the VocalSet leak

**Question:** how much of v1's held-out score came from VocalSet files that were in its training pool?

## What differs from v1

Only the data fix. `denoiser\dataset.py` now filters VocalSet-derived files out of the DNS speech branch (the DNS
clean pool carries a second, unsplit copy of VocalSet: folder `VocalSet_48kHz_mono`, 3,613 files named
`vocalset_<singer>_<take>_48kHz.wav`, all 20 singers including the held-out female9 and male11). The files were not
moved or deleted, they are simply never drawn as a clean target. See
`denoiser-report\LEAKAGE_AUDIT_2026-11-25.md`.

Everything else is v1: 2-layer GRU (256), 32 ms / 10 ms asymmetric windows, no lookahead, batch 32, lr 1e-3, AdamW,
same loss, same fixed dev split (seed 1234) and eval mixes (seed 777), `--workers 4`, `--steps 108773` (v1's step
count: it ran a 9-hour budget that ended at 108,773 steps). v1's setup has no training seed to copy, so there is no
seed difference to worry about.

Speech pool after the filter: train 13,026 -> 9,602 files (3,424 dropped), dev 707 -> 518 (189 dropped).

## How it ran

`.\run.ps1`, pinned to GPU 1 (RTX 5070) with `CUDA_VISIBLE_DEVICES=1` so GPU 0 (the 4060, running the TTS training)
was never used. Started 2026-11-25 07:06. Checkpoints, MLflow db and the ONNX export land in this folder.

## Result

Trained to step 108,773 (6.17 hours); the last four dev evaluations read +2.15, +2.14, +2.16 and +2.17 dB SI-SDR gain, so training plateaued. The ONNX export is 5.0 MB and runs at 0.218 ms per 5 ms block on CPU. Weights are not published; this is a control run.

Scored once against v1 with `compare.py --n 180 --split test` (GPU 1, one call, nothing tuned). Note that `--n 180` draws 163 noisy plus 17 clean clips, while the main results used `--n 200`, so this covers 163 of the 180 noisy held-out clips (86 of 94 VocalSet, 77 of 86 own voice). v1's per-clip scores on them are identical to the main comparison. Paired, v1_clean minus v1, mean ± 95% CI over clips:

| Held-out group | Clips | SI-SDR (dB) v1 to v1_clean | Change | SDR change | PESQ-WB change | ViSQOL change |
|---|---|---|---|---|---|---|
| VocalSet | 86 | 23.75 to 23.59 | -0.15 ± 0.24 | -0.22 ± 0.20 | -0.01 ± 0.02 | -0.02 ± 0.03 |
| Own voice (control) | 77 | 21.59 to 21.47 | -0.12 ± 0.45 | -0.07 ± 0.38 | -0.00 ± 0.02 | +0.01 ± 0.02 |
| All noisy | 163 | 22.73 to 22.59 | -0.14 ± 0.25 | -0.15 ± 0.21 | -0.00 ± 0.01 | -0.01 ± 0.02 |

**Reading:** VocalSet fell 0.03 dB more than own voice on SI-SDR (± 0.51) and 0.15 dB more on SDR (± 0.43); every interval includes zero. The leak did not measurably inflate v1's held-out VocalSet scores. One retrain, so no seed variance; v1c, v1b, v1d and v1e were not retrained. Clean pass-through SI-SDR is 44.5 dB for v1_clean against 47.6 dB for v1, not explained. Raw output: `compare_v1_clean.json` and `compare_v1_clean.md`.
