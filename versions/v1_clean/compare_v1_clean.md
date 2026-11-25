Same 163 test clips for every version (fixed seed 777), streaming engine.

| Version | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL | MERT-MSE ↓ | FAD-MERT ↓ | Pass-through SI-SDR (dB) |
|---|---|---|---|---|---|---|---|
| noisy | 18.80 | 18.78 | 2.35 | 3.52 | 0.0234 | 1.743 | — |
| v1 | 22.73 | 23.51 | 3.02 | 3.96 | 0.0134 | 0.427 | 47.6 |
| v1_clean | 22.59 | 23.36 | 3.02 | 3.95 | 0.0135 | 0.422 | 44.5 |

Paired differences between consecutive versions (newer - older, mean ± 95% CI; for MERT-MSE negative = better):

- **v1_clean vs v1**: SI-SDR (dB) -0.14 ± 0.25, SDR (dB) -0.15 ± 0.21, PESQ-WB -0.00 ± 0.01, ViSQOL -0.01 ± 0.02, MERT-MSE ↓ +0.0001 ± 0.0002

By voice (mean score; paired change from the first to the last version, ± 95% CI):

- **own** (77 clips), v1_clean vs v1: SI-SDR (dB) -0.12 ± 0.45, SDR (dB) -0.07 ± 0.38, PESQ-WB -0.00 ± 0.02, ViSQOL +0.01 ± 0.02, MERT-MSE ↓ -0.0000 ± 0.0002
- **vocalset** (86 clips), v1_clean vs v1: SI-SDR (dB) -0.15 ± 0.24, SDR (dB) -0.22 ± 0.20, PESQ-WB -0.01 ± 0.02, ViSQOL -0.02 ± 0.03, MERT-MSE ↓ +0.0002 ± 0.0002
