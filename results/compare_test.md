Same 180 test clips for every version (fixed seed 777), streaming engine.

| Version | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL | MERT-MSE ↓ | FAD-MERT ↓ | Pass-through SI-SDR (dB) |
|---|---|---|---|---|---|---|---|
| noisy | 19.04 | 19.02 | 2.37 | 3.53 | 0.0231 | 1.732 | — |
| v1 | 22.95 | 23.70 | 3.05 | 3.97 | 0.0131 | 0.419 | 45.6 |
| v1c | 23.27 | 23.85 | 3.03 | 3.98 | 0.0132 | 0.485 | 51.1 |
| v1b | 23.46 | 24.07 | 3.10 | 3.99 | 0.0127 | 0.448 | 47.6 |
| v1d | 23.43 | 24.03 | 3.11 | 3.99 | 0.0127 | 0.446 | 50.5 |
| v1e | 23.45 | 24.08 | 3.12 | 4.00 | 0.0126 | 0.445 | 53.1 |

Paired differences between consecutive versions (newer - older, mean ± 95% CI; for MERT-MSE negative = better):

- **v1c vs v1**: SI-SDR (dB) +0.32 ± 0.36, SDR (dB) +0.14 ± 0.30, PESQ-WB -0.02 ± 0.02, ViSQOL +0.01 ± 0.02, MERT-MSE ↓ +0.0001 ± 0.0002
- **v1b vs v1c**: SI-SDR (dB) +0.19 ± 0.25, SDR (dB) +0.22 ± 0.22, PESQ-WB +0.07 ± 0.02, ViSQOL +0.01 ± 0.02, MERT-MSE ↓ -0.0005 ± 0.0001
- **v1d vs v1b**: SI-SDR (dB) -0.02 ± 0.21, SDR (dB) -0.04 ± 0.19, PESQ-WB +0.01 ± 0.02, ViSQOL -0.00 ± 0.02, MERT-MSE ↓ +0.0000 ± 0.0001
- **v1e vs v1d**: SI-SDR (dB) +0.01 ± 0.19, SDR (dB) +0.05 ± 0.17, PESQ-WB +0.02 ± 0.02, ViSQOL +0.01 ± 0.02, MERT-MSE ↓ -0.0001 ± 0.0001

By voice (mean score; paired change from the first to the last version, ± 95% CI):

- **own** (86 clips), v1e vs v1: SI-SDR (dB) +1.03 ± 0.50, SDR (dB) +0.83 ± 0.41, PESQ-WB +0.12 ± 0.03, ViSQOL +0.06 ± 0.03, MERT-MSE ↓ -0.0010 ± 0.0003
- **vocalset** (94 clips), v1e vs v1: SI-SDR (dB) +0.00 ± 0.31, SDR (dB) -0.05 ± 0.27, PESQ-WB +0.03 ± 0.02, ViSQOL -0.01 ± 0.04, MERT-MSE ↓ -0.0001 ± 0.0002
