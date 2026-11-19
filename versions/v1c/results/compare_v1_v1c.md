Same 180 test clips for every version (fixed seed 777), streaming engine.

| Version | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL | MERT-MSE ↓ | FAD-MERT ↓ | Pass-through SI-SDR (dB) |
|---|---|---|---|---|---|---|---|
| noisy | 19.04 | 19.02 | 2.37 | 3.53 | 0.0231 | 1.732 | — |
| v1 | 22.95 | 23.70 | 3.05 | 3.97 | 0.0131 | 0.419 | 45.6 |
| v1c-main(47k) | 23.05 | 23.65 | 2.97 | 3.96 | 0.0138 | 0.513 | 50.3 |
| v1c-final(108k) | 23.27 | 23.85 | 3.03 | 3.98 | 0.0132 | 0.485 | 51.1 |

Paired differences between consecutive versions (newer - older, mean ± 95% CI; for MERT-MSE negative = better):

- **v1c-main(47k) vs v1**: SI-SDR (dB) +0.10 ± 0.37, SDR (dB) -0.05 ± 0.31, PESQ-WB -0.08 ± 0.02, ViSQOL -0.01 ± 0.02, MERT-MSE ↓ +0.0007 ± 0.0002
- **v1c-final(108k) vs v1c-main(47k)**: SI-SDR (dB) +0.21 ± 0.14, SDR (dB) +0.20 ± 0.12, PESQ-WB +0.06 ± 0.01, ViSQOL +0.02 ± 0.01, MERT-MSE ↓ -0.0006 ± 0.0001

By voice (mean score; paired change from the first to the last version, ± 95% CI):

- **own** (86 clips), v1c-final(108k) vs v1: SI-SDR (dB) +0.95 ± 0.67, SDR (dB) +0.62 ± 0.57, PESQ-WB +0.02 ± 0.02, ViSQOL +0.03 ± 0.02, MERT-MSE ↓ -0.0001 ± 0.0002
- **vocalset** (94 clips), v1c-final(108k) vs v1: SI-SDR (dB) -0.27 ± 0.27, SDR (dB) -0.29 ± 0.24, PESQ-WB -0.05 ± 0.02, ViSQOL -0.02 ± 0.04, MERT-MSE ↓ +0.0004 ± 0.0002
